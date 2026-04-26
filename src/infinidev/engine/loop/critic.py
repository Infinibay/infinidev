"""Assistant LLM — pair-programming critic.

A second LLM that runs on a separate Ollama instance (typically pinned
to a different GPU) and reviews the principal's tool calls in parallel
with their execution. The verdict is purely informative: it never
blocks, never forces retries, never vetoes. The principal sees the
critic's message in the next iteration and decides what to do.

Design notes
------------
* The critic shares the principal's full message history so it can
  reason about <task>, <plan>, <previous-actions>, etc. — these are
  already embedded in the messages by the loop's prompt builder.
* The principal's system prompt is REPLACED with the critic system
  prompt. The critic must not see protocol/tool-schema instructions
  meant for the driver — its job is review, not execution.
* The critic's reasoning/thinking is discarded; only the JSON verdict
  is propagated. Otherwise the critic's chain-of-thought would
  contaminate the principal's next turn.
* A failed/timed-out critic returns None and the engine treats that
  as ``continue`` — the critic must never be a blocker.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from infinidev.config.llm import get_litellm_params_for_assistant
from infinidev.engine.formats.tool_call_parser import safe_json_loads as _safe_json_loads


logger = logging.getLogger(__name__)


CriticAction = Literal["continue", "information", "recommendation", "reject"]
_VALID_ACTIONS: frozenset[str] = frozenset(
    {"continue", "information", "recommendation", "reject"}
)


@dataclass(frozen=True)
class CriticVerdict:
    """The critic's response to a batch of tool calls.

    Attributes:
        action: One of continue / information / recommendation / reject.
            Pure semantic signal for the principal — the engine treats
            them all identically (inject the message, never block).
        message: The text shown to the principal in the next iteration.
            Empty when action == "continue" (no message is injected).
    """

    action: CriticAction
    message: str

    @property
    def is_silent(self) -> bool:
        return self.action == "continue" or not self.message.strip()


_SYSTEM_PROMPT = (
    "Sos el segundo cerebro de un pair programming. El principal "
    "(driver) propone y ejecuta acciones; vos lo acompañás como un "
    "senior que mira por encima del hombro. No tenés herramientas y "
    "no ejecutás nada — tu único canal es un mensaje corto en cada "
    "turno.\n\n"
    "Tu personalidad:\n"
    "- Querés que al principal le vaya bien. No competís, no "
    "criticás por criticar, no te haces el que sabe más cuando no "
    "tenés algo útil que decir.\n"
    "- Te importan las cosas que le importan a un buen ingeniero: "
    "que el código no rompa nada, que sea seguro, que sea limpio, "
    "que use las herramientas adecuadas, que no haga trabajo de "
    "más, que respete las convenciones del proyecto.\n"
    "- Si ves algo peligroso (un comando que destruye, un commit "
    "donde no debería, un step que se cierra sin terminar el "
    "trabajo), hablás claro y rápido. Si ves algo feo o mejorable, "
    "lo señalás concreto. Si ves un dato del proyecto que el "
    "principal no tiene, se lo pasás. Si todo está bien, te "
    "callás — el silencio es una forma de respeto a su trabajo.\n"
    "- Confiás en el principal por default. Si la propuesta es "
    "coherente con lo que viene haciendo, no inventás problemas.\n\n"
    "Cómo respondés:\n"
    "- continue: no hay nada útil que aportar. Es la respuesta más "
    "frecuente cuando el principal sabe lo que hace.\n"
    "- information: hay un dato específico del proyecto que el "
    "principal no tiene y le va a servir.\n"
    "- recommendation: ves una mejora concreta — código feo, "
    "alternativa más limpia, tool más apropiada, riesgo de "
    "seguridad evitable. Siempre con el qué Y el por qué.\n"
    "- reject: la acción rompe algo, es peligrosa, viola una "
    "convención dura del repo, o está claramente mal contextualizada "
    "(editar un archivo sin haberlo leído, cerrar un step con tests "
    "rojos, hacer commit a main, escribir fuera del proyecto).\n\n"
    "Reglas duras del juego:\n"
    "- Tu thinking se descarta — solo cuenta el JSON final.\n"
    "- Sé conciso (<150 palabras). Mensaje vacío si action=continue.\n"
    "- Hablale al principal directo, en segunda persona, como un "
    "compañero. Sin chitchat, sin disculpas, sin pedirle que confirme.\n\n"
    "Formato — un único objeto JSON:\n"
    '{"action": "continue" | "information" | "recommendation" | "reject", '
    '"message": "<texto al principal>"}\n\n'
    "Tu mensaje aparecerá en el próximo turno del principal "
    "prefijado con \"[ASSISTANT - <action>]:\"."
)


_USER_TEMPLATE_HEADER = (
    "El modelo principal acaba de proponer estas tool calls en el "
    "turno actual:\n\n{proposed}\n\n"
    "Catálogo de tools disponibles (referencia — vos NO tenés "
    "ninguna):\n{catalog}\n\n"
    "Revisá la propuesta a la luz del <task>, <plan>, "
    "<previous-actions> y <current-action> que viste en la "
    "conversación. Respondé con el JSON definido en las "
    "instrucciones del sistema."
)


def _format_proposed_calls(tool_calls: Iterable[Any]) -> str:
    """Render proposed tool calls compactly for the critic prompt.

    Truncates argument values longer than 240 chars so a giant
    ``replace_lines`` payload doesn't blow the critic's context.
    """
    lines: list[str] = []
    for tc in tool_calls:
        try:
            name = tc.function.name
            raw = tc.function.arguments
            args = _safe_json_loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            lines.append(f"- <unparseable tool call: {tc!r}>")
            continue

        rendered_args: list[str] = []
        for k, v in args.items():
            sv = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            if len(sv) > 240:
                sv = sv[:240] + f"... (+{len(sv) - 240} chars)"
            rendered_args.append(f"{k}={sv}")
        lines.append(f"- {name}({', '.join(rendered_args)})")
    return "\n".join(lines) if lines else "- <none>"


def _format_tool_catalog(tool_descriptions: dict[str, str]) -> str:
    if not tool_descriptions:
        return "(sin catálogo disponible)"
    return "\n".join(
        f"- {name}: {(desc or '').strip().splitlines()[0] if desc else '(sin descripción)'}"
        for name, desc in sorted(tool_descriptions.items())
    )


def _strip_principal_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop any ``role=system`` messages from the principal's history.

    The critic gets its own system prompt prepended later. Leaving the
    principal's protocol/tool-schema system in there would tell the
    critic "you must call tools and emit step_complete", which is
    exactly what we don't want.
    """
    return [m for m in messages if m.get("role") != "system"]


def _parse_verdict(content: str) -> CriticVerdict | None:
    """Extract {action, message} from the critic's response.

    Tolerates models that wrap JSON in code fences, prepend
    explanatory text, or emit thinking blocks. Returns None on any
    failure — the caller treats that as silent ``continue``.
    """
    if not content:
        return None

    parsed = _safe_json_loads(content)
    if not isinstance(parsed, dict):
        # Last-ditch: scan for the first {...} block. _safe_json_loads
        # already handles fences/leading prose for the common cases;
        # if it still failed, the response is too malformed to trust.
        return None

    action = str(parsed.get("action", "")).strip().lower()
    if action not in _VALID_ACTIONS:
        return None

    message = parsed.get("message", "")
    if not isinstance(message, str):
        try:
            message = json.dumps(message, ensure_ascii=False)
        except Exception:
            message = str(message)

    return CriticVerdict(action=action, message=message.strip())  # type: ignore[arg-type]


class AssistantCritic:
    """Stateless pair-programming critic.

    One instance is constructed per developer-loop run (in
    ``LoopEngine.__init__`` when ``ASSISTANT_LLM_ENABLED``) and reused
    across iterations. ``review()`` is safe to call from a worker
    thread — it does not mutate shared state.
    """

    def __init__(self, tool_descriptions: dict[str, str]):
        # Build params eagerly so a misconfigured assistant LLM raises
        # at engine startup, not in the middle of a step. The principal
        # loop instantiates us only when the feature is enabled.
        self._params = get_litellm_params_for_assistant()
        self._tool_descriptions = tool_descriptions
        # Short, human-readable model name for the [ASSISTANT (...)] prefix.
        full = str(self._params.get("model", ""))
        self._model_short = full.split("/", 1)[-1] if "/" in full else full

    @property
    def model_short_name(self) -> str:
        return self._model_short

    def review(
        self,
        messages: list[dict[str, Any]],
        tool_calls: Iterable[Any],
    ) -> CriticVerdict | None:
        """Ask the critic to review a batch of proposed tool calls.

        Returns ``None`` on any failure (network, timeout, malformed
        JSON, empty response) — the engine treats that as ``continue``
        and proceeds silently. The critic is never allowed to break
        the loop.
        """
        proposed = list(tool_calls)
        if not proposed:
            return None

        try:
            crit_messages: list[dict[str, Any]] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                *_strip_principal_system(messages),
                {
                    "role": "user",
                    "content": _USER_TEMPLATE_HEADER.format(
                        proposed=_format_proposed_calls(proposed),
                        catalog=_format_tool_catalog(self._tool_descriptions),
                    ),
                },
            ]
        except Exception:
            logger.exception("critic: prompt assembly failed")
            return None

        try:
            # Lazy import — keeps engine startup cheap and avoids a
            # circular import when the engine module loads us.
            from infinidev.engine.llm_client import call_llm as _call_llm

            response = _call_llm(self._params, crit_messages, tools=None)
        except Exception as exc:
            logger.warning("critic: LLM call failed (%s); silent fallback", exc)
            return None

        try:
            content = response.choices[0].message.content or ""
        except (AttributeError, IndexError, KeyError):
            logger.warning("critic: unexpected response shape; silent fallback")
            return None

        verdict = _parse_verdict(content)
        if verdict is None:
            logger.info("critic: unparseable verdict (len=%d); silent fallback", len(content))
            return None
        return verdict
