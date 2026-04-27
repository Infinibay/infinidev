"""Deterministic dispatcher that selects which checks the assistant
critic should activate this turn.

The critic itself is a single LLM with a single base prompt. The
dispatcher sits *in front of* the critic and decides which specialised
*lenses* the critic should look through, given purely structural signals
about the turn (is the principal proposing ``step_complete``? what
iteration is this? did the previous verdict push back?).

This keeps the critic from being a generalist that reviews everything
shallowly: each turn it sees only the lenses that fit. Signal-to-noise
goes up, and we get coverage of ten distinct failure modes without
multiplying critic agents.

The dispatcher is **stateless and pure** — no LLM calls, no I/O. All
state needed for selection is captured in :class:`DispatchSignal`,
which the engine constructs at the call site. Tests cover it without
mocking anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class CheckCode(str, Enum):
    """The ten reasoning failure modes the critic can detect structurally.

    Each value is a short uppercase token because it surfaces directly
    in the critic's prompt as a section heading (``[PREMATURE_CLOSURE]``).
    The string-enum form lets us serialise into JSON / set membership
    cleanly while keeping a typed identity in code.
    """

    PREMATURE_CLOSURE = "PREMATURE_CLOSURE"
    FALSE_CONFIDENCE = "FALSE_CONFIDENCE"
    TUNNEL_VISION = "TUNNEL_VISION"
    CONTEXT_DRIFT = "CONTEXT_DRIFT"
    OVERCORRECTION = "OVERCORRECTION"
    CONFAB_API = "CONFAB_API"
    CONFAB_PATH = "CONFAB_PATH"
    HALLUC_OUTPUT = "HALLUC_OUTPUT"
    SYCOPHANCY_USER = "SYCOPHANCY_USER"
    SYCOPHANCY_REPO = "SYCOPHANCY_REPO"


# Tool names that, when called, are strong evidence of a code-write action
# whose target path / symbol the critic must have seen before.
_PATH_BEARING_TOOLS: frozenset[str] = frozenset({
    "replace_lines",
    "edit_symbol",
    "add_symbol",
    "remove_symbol",
})

# Tools whose recent output, if used as the basis for the next action,
# may indicate cargo-culting the repo's existing patterns without checking
# whether the patterns apply to the current case.
_REPO_INSPECTION_TOOLS: frozenset[str] = frozenset({
    "code_search",
    "find_references",
    "list_symbols",
    "search_symbols",
})


@dataclass
class DispatchSignal:
    """Structural facts about the current turn that drive check selection.

    All fields default to neutral / empty values so the engine can build
    a partial signal cheaply and the dispatcher is forgiving when a piece
    of context is unavailable.

    Attributes
    ----------
    iteration:
        1-indexed iteration number within the developer loop.
    is_step_complete:
        True when the principal is proposing ``step_complete`` this turn.
    is_first_turn:
        Convenience flag — ``iteration == 1``. Carried explicitly so the
        engine can suppress it for replayed turns if needed.
    prior_verdict_action:
        The ``action`` of the verdict the critic emitted in the previous
        turn (one of ``continue``/``information``/``recommendation``/
        ``reject``), or ``None`` if there was no verdict last turn.
    last_tool_result_present:
        Whether a tool result is visible in the previous-actions block.
        When True, the principal's reasoning *should* reference it
        accurately — otherwise we suspect ``HALLUC_OUTPUT``.
    tool_call_names:
        The bare ``function.name`` of each proposed tool call this turn.
    tool_call_paths:
        Paths and qualified symbol names extracted from the proposed
        tool call arguments (e.g. ``file_path``, ``symbol_name``).
    paths_seen_recently:
        Paths and symbol names the principal has already opened or
        searched in the previous-actions window — populated by the
        engine from ``ctx.opened_files`` and recent tool results.
    recent_step_complete_iters:
        Iteration numbers of the last few ``step_complete`` calls. Used
        to detect ``TUNNEL_VISION``: many iterations with no recent
        zoom-out via a ``step_complete``.
    reasoning_pattern_matches:
        Pattern names returned by the reasoning detector (Bloque C),
        e.g. ``["victory_lap"]``. Some patterns map to checks via
        :func:`select_checks`; others surface directly to the critic.
    """

    iteration: int = 1
    is_step_complete: bool = False
    is_first_turn: bool = False
    prior_verdict_action: str | None = None
    last_tool_result_present: bool = False
    tool_call_names: list[str] = field(default_factory=list)
    tool_call_paths: list[str] = field(default_factory=list)
    paths_seen_recently: set[str] = field(default_factory=set)
    recent_step_complete_iters: list[int] = field(default_factory=list)
    reasoning_pattern_matches: list[str] = field(default_factory=list)


def select_checks(signal: DispatchSignal) -> list[CheckCode]:
    """Return the checks to activate this turn, in stable order.

    Pure function — same signal in, same checks out. The order matches
    :class:`CheckCode` declaration order so the prompt block reads
    consistently across turns and so test assertions can compare lists
    rather than sets.

    The selection logic mirrors the dispatch table in the plan:

    ============================== ==========================================
    Check                          Trigger
    ============================== ==========================================
    ``PREMATURE_CLOSURE``          step_complete proposed
    ``FALSE_CONFIDENCE``           step_complete proposed (assertion-heavy)
    ``TUNNEL_VISION``              iter > 5 ∧ no recent step_complete
    ``CONTEXT_DRIFT``              iter > 10
    ``OVERCORRECTION``             previous verdict was push-back
    ``CONFAB_API``                 execute_command ∨ path-bearing edit
    ``CONFAB_PATH``                path-bearing edit on un-seen path
    ``HALLUC_OUTPUT``              tool result present last turn
    ``SYCOPHANCY_USER``            first turn
    ``SYCOPHANCY_REPO``            tool calls follow recent repo-inspection
    ============================== ==========================================
    """
    selected: set[CheckCode] = set()

    if signal.is_step_complete:
        selected.add(CheckCode.PREMATURE_CLOSURE)
        selected.add(CheckCode.FALSE_CONFIDENCE)

    # Tunnel vision: been grinding for a while and no recent step_complete.
    if signal.iteration > 5:
        recent_threshold = signal.iteration - 4
        had_recent_complete = any(
            i >= recent_threshold for i in signal.recent_step_complete_iters
        )
        if not had_recent_complete:
            selected.add(CheckCode.TUNNEL_VISION)

    if signal.iteration > 10:
        selected.add(CheckCode.CONTEXT_DRIFT)

    if signal.prior_verdict_action in {"recommendation", "reject"}:
        selected.add(CheckCode.OVERCORRECTION)

    tool_names = set(signal.tool_call_names)
    if "execute_command" in tool_names or tool_names & _PATH_BEARING_TOOLS:
        selected.add(CheckCode.CONFAB_API)

    if tool_names & _PATH_BEARING_TOOLS:
        # If any proposed path/symbol is not in the seen-recently set, the
        # principal is editing something they haven't read.
        unseen = [p for p in signal.tool_call_paths if p and p not in signal.paths_seen_recently]
        if unseen:
            selected.add(CheckCode.CONFAB_PATH)

    if signal.last_tool_result_present:
        selected.add(CheckCode.HALLUC_OUTPUT)

    if signal.is_first_turn:
        selected.add(CheckCode.SYCOPHANCY_USER)

    # Cargo-culting: the principal just inspected the repo and is now
    # acting on it without verifying the pattern fits.
    if tool_names & _REPO_INSPECTION_TOOLS and tool_names & _PATH_BEARING_TOOLS:
        selected.add(CheckCode.SYCOPHANCY_REPO)

    # Pattern-match driven additions are handled by the engine, not here,
    # because the mapping pattern → check lives in the patterns table.
    # We *do* still consult ``reasoning_pattern_matches`` for two pivots
    # the engine wouldn't otherwise reach via signal-only logic:
    if "victory_lap" in signal.reasoning_pattern_matches:
        selected.add(CheckCode.PREMATURE_CLOSURE)
    if "anchoring_loop" in signal.reasoning_pattern_matches:
        selected.add(CheckCode.TUNNEL_VISION)

    return [c for c in CheckCode if c in selected]


CHECK_PROMPTS: dict[CheckCode, str] = {
    CheckCode.PREMATURE_CLOSURE: (
        "[PREMATURE_CLOSURE] El principal está cerrando un step. "
        "Antes de aceptar el done: ¿qué evidencia *concreta* hay de "
        "que el step terminó realmente? Si modificó código, debería "
        "haber un comando ejecutado, una relectura del archivo, o un "
        "test corrido en el historial reciente. Frases tipo 'should "
        "work' / 'looks correct' / 'now everything is fine' sin un "
        "tool result que las respalde son señal fuerte de cierre "
        "prematuro — exigí la evidencia faltante."
    ),
    CheckCode.FALSE_CONFIDENCE: (
        "[FALSE_CONFIDENCE] Buscá afirmaciones absolutas en el "
        "reasoning ('this fixes it', 'the bug is X') que no estén "
        "respaldadas por un tool result observado en este turno o en "
        "los inmediatamente anteriores. La confianza sin evidencia es "
        "el preludio típico de un fix que no fixea. Si la rationale "
        "del tool call es genérica o vacía, es lo mismo: pedir "
        "verificación, no aceptar el assert."
    ),
    CheckCode.TUNNEL_VISION: (
        "[TUNNEL_VISION] Llevamos varias iteraciones sin que el "
        "principal cierre un step. Releé el <task> original y los "
        "últimos 3-4 turnos: ¿siguen alineados? Si el principal está "
        "atascado en una rama (mismo archivo, mismo bug, distintos "
        "intentos sin progreso), sugerí abandonar la rama, declarar "
        "el step blocked, o probar una hipótesis ortogonal."
    ),
    CheckCode.CONTEXT_DRIFT: (
        "[CONTEXT_DRIFT] El loop ya pasó 10 iteraciones. A esta "
        "altura el principal puede haber perdido el hilo del <task> "
        "original (su contexto está comprimido en summaries; vos lo "
        "tenés sin comprimir). Si lo que está haciendo *no* avanza "
        "hacia el pedido literal del usuario, decilo explícitamente "
        "con una cita del task."
    ),
    CheckCode.OVERCORRECTION: (
        "[OVERCORRECTION] En el turno anterior emitiste una "
        "recommendation o un reject. Mirá si el principal corrigió "
        "*demasiado* — abandonando una rama válida, o adoptando tu "
        "sugerencia sin entenderla. Si la corrección actual rompe "
        "algo que antes funcionaba, marcalo. Tu objetivo no es que te "
        "obedezcan, es que lleguen al goal."
    ),
    CheckCode.CONFAB_API: (
        "[CONFAB_API] El principal está por usar un comando o un "
        "símbolo (función, flag, módulo). Verificá con el historial: "
        "¿este símbolo aparece en algún tool result reciente, o lo "
        "está inventando? Modelos chicos tienden a inventar APIs "
        "plausibles que no existen. Si no hay evidencia de que el "
        "símbolo existe, exigí un read_file / code_search antes."
    ),
    CheckCode.CONFAB_PATH: (
        "[CONFAB_PATH] El principal va a editar un archivo o "
        "símbolo que NO aparece en el historial de read_file / "
        "list_directory / code_search reciente. Editar a ciegas es "
        "una de las formas más caras de fallar. Pedí que abra el "
        "archivo (o el símbolo) y confirme su estado actual antes "
        "del edit."
    ),
    CheckCode.HALLUC_OUTPUT: (
        "[HALLUC_OUTPUT] En el turno previo se ejecutaron tool "
        "calls que produjeron output. Compará lo que dice el "
        "reasoning del principal con el output real: ¿está "
        "interpretándolo correctamente, o asumiendo que dijo algo "
        "que no dijo? Es común que el modelo afirme 'el test pasó' "
        "cuando el output muestra un fallo silencioso."
    ),
    CheckCode.SYCOPHANCY_USER: (
        "[SYCOPHANCY_USER] Es el primer turno del loop. Releé el "
        "<task> con ojo crítico: ¿el framing del usuario asume algo "
        "que podría estar mal? ¿Está pidiendo X cuando el problema "
        "real es Y? Modelos tienden a aceptar el pedido literal sin "
        "cuestionarlo. Si ves un supuesto frágil, marcalo antes de "
        "que se cristalice en el plan."
    ),
    CheckCode.SYCOPHANCY_REPO: (
        "[SYCOPHANCY_REPO] El principal acaba de inspeccionar el "
        "repo (code_search / find_references / etc.) y ahora está "
        "editando código siguiendo el patrón que encontró. ¿Verificó "
        "que el patrón aplica a este caso, o lo está copiando por "
        "inercia? Patrones existentes pueden estar desactualizados, "
        "ser específicos a otro contexto, o ser parte del bug."
    ),
}


def render_active_checks_block(checks: Iterable[CheckCode]) -> str:
    """Render the ``<active-checks>`` block injected into the critic prompt.

    Returns an empty string when *checks* is empty so the caller can
    safely concat without a conditional. The block uses XML tags to
    match the surrounding prompt style (``<task>``, ``<plan>``).
    """
    items = list(checks)
    if not items:
        return ""
    body = "\n\n".join(CHECK_PROMPTS[c] for c in items)
    return (
        "<active-checks>\n"
        "Este turno, mirá especialmente por estos modos de fallo "
        "(activados por señales del turno):\n\n"
        f"{body}\n"
        "</active-checks>"
    )
