"""Held-out task-policy routing corpus and offline metrics.

The corpus is generated from phrasing families that are deliberately separate
from registry prototypes. Run with ``uv run python -m bench.task_policy_eval``.
No network or LLM call is required by the default deterministic evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Iterable

from infinidev.engine.task_policies.router import resolve_task_profile


@dataclass(frozen=True)
class RequestFamily:
    """One held-out phrasing family with gold multi-axis labels."""

    id: str
    templates: tuple[str, ...]
    objects: tuple[str, ...]
    policies: frozenset[str]
    authority: frozenset[str]


@dataclass(frozen=True)
class EvalExample:
    """One materialized held-out request."""

    id: str
    text: str
    policies: frozenset[str]
    authority: frozenset[str]


_OBJECTS = (
    "el cargador de configuración",
    "the session cache",
    "la ruta de autenticación",
    "the command parser",
    "el cliente HTTP",
    "the SQLite adapter",
    "la interfaz de terminal",
    "the plugin registry",
)

FAMILIES: tuple[RequestFamily, ...] = (
    RequestFamily(
        "bugfix-es",
        ("Soluciona el fallo que rompe {object}.", "Encuentra la causa y corrige {object}.",
         "{object} falla al iniciar; arréglalo y verifica la regresión."),
        _OBJECTS, frozenset({"bugfix.root_cause"}), frozenset({"answer", "diagnose", "modify"}),
    ),
    RequestFamily(
        "feature-en",
        ("Build the requested extension in {object}.", "Add support for timeouts to {object}.",
         "Implement a new validation path in {object}."),
        _OBJECTS, frozenset({"feature.contract_first"}), frozenset({"answer", "modify"}),
    ),
    RequestFamily(
        "refactor-mixed",
        ("Refactor {object} preserving behavior.",
         "Reestructura {object} sin alterar su comportamiento.",
         "Clean up {object}; keep observable behavior identical."),
        _OBJECTS, frozenset({"refactor.preserve_behavior"}), frozenset({"answer", "modify"}),
    ),
    RequestFamily(
        "research-only",
        ("Investiga las alternativas para {object} y entrega un informe.",
         "Research the design choices around {object}; report evidence only.",
         "Analiza {object}, cita fuentes y no edites archivos."),
        _OBJECTS, frozenset({"research.evidence_first"}), frozenset({"answer", "diagnose"}),
    ),
    RequestFamily(
        "review-only",
        ("Revisa {object}, pero no cambies archivos.",
         "Review {object} read-only and list defects.",
         "Audita {object}; solo quiero hallazgos, no una implementación."),
        _OBJECTS, frozenset({"review.read_only"}), frozenset({"answer", "diagnose"}),
    ),
    RequestFamily(
        "performance-api",
        ("Optimiza {object} sin modificar la API pública.",
         "Reduce latency in {object} while preserving the public API.",
         "Mejora el rendimiento de {object}; conserva retrocompatibilidad."),
        _OBJECTS,
        frozenset({"performance.measure_first", "compatibility.preserve_public_api"}),
        frozenset({"answer", "diagnose", "modify"}),
    ),
    RequestFamily(
        "research-then-fix",
        ("Investiga la causa y luego corrige {object}.",
         "Research why {object} fails, then implement the fix.",
         "Primero reúne evidencia sobre {object}; después arréglalo."),
        _OBJECTS, frozenset({"research.evidence_first", "bugfix.root_cause"}),
        frozenset({"answer", "diagnose", "modify"}),
    ),
    RequestFamily(
        "publish-update",
        ("Actualiza {object}, haz commit y push a main.",),
        _OBJECTS, frozenset(), frozenset({"answer", "modify", "commit", "publish"}),
    ),
    RequestFamily(
        "publish-feature",
        ("Implement the change in {object}, commit it, and publish it.",),
        _OBJECTS,
        frozenset({"feature.contract_first"}),
        frozenset({"answer", "modify", "commit", "publish"}),
    ),
    RequestFamily(
        "publish-bugfix",
        ("Corrige {object}; después commitea y despliega.",),
        _OBJECTS,
        frozenset({"bugfix.root_cause"}),
        frozenset({"answer", "diagnose", "modify", "commit", "publish"}),
    ),
    RequestFamily(
        "negated-refactor-es",
        ("No refactorices {object}; solo explícame el problema.",),
        _OBJECTS, frozenset(), frozenset({"answer"}),
    ),
    RequestFamily(
        "negated-refactor-en",
        ("Do not refactor {object}; only explain what is wrong.",),
        _OBJECTS, frozenset(), frozenset({"answer"}),
    ),
    RequestFamily(
        "read-only-analysis",
        ("Sin tocar archivos, analiza por qué {object} se comporta así.",),
        _OBJECTS,
        frozenset({"research.evidence_first", "review.read_only"}),
        frozenset({"answer", "diagnose"}),
    ),
    RequestFamily(
        "quoted-neutral",
        ('{object} muestra "refactor required"; ¿qué significa?',
         '{object} logs "please implement fix". Explain that message.',
         'En {object} aparece "publish failed"; interpreta el error.'),
        _OBJECTS, frozenset(), frozenset({"answer"}),
    ),
)


def build_holdout_corpus() -> list[EvalExample]:
    """Materialize 240 unique examples across ten unseen families."""
    examples: list[EvalExample] = []
    for family in FAMILIES:
        for template_index, template in enumerate(family.templates):
            for object_index, object_name in enumerate(family.objects):
                examples.append(EvalExample(
                    id=f"{family.id}-{template_index:02d}-{object_index:02d}",
                    text=template.format(object=object_name),
                    policies=family.policies,
                    authority=family.authority,
                ))
    return examples


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_profiles(examples: Iterable[EvalExample]) -> dict[str, object]:
    """Evaluate deterministic routing without calibration or network leakage."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total = 0
    authority_exact = 0
    false_write_authority = 0
    exact_policies = 0
    for example in examples:
        total += 1
        profile = resolve_task_profile(example.text)
        actual = {selection.id for selection in profile.selected_policies}
        for policy in actual | set(example.policies):
            if policy in actual and policy in example.policies:
                counts[policy]["tp"] += 1
            elif policy in actual:
                counts[policy]["fp"] += 1
            else:
                counts[policy]["fn"] += 1
        actual_authority = set(profile.authority)
        authority_exact += actual_authority == set(example.authority)
        false_write_authority += bool(
            actual_authority & {"modify", "commit", "publish"}
            and not set(example.authority) & {"modify", "commit", "publish"}
        )
        exact_policies += actual == set(example.policies)

    return {
        "examples": total,
        "policy_exact_match": exact_policies / total if total else 0.0,
        "authority_exact_match": authority_exact / total if total else 0.0,
        "false_write_authority_rate": false_write_authority / total if total else 0.0,
        "per_policy": {
            policy: {**values, **_prf(values["tp"], values["fp"], values["fn"])}
            for policy, values in sorted(counts.items())
        },
    }


def main() -> None:
    """Print deterministic held-out metrics as machine-readable JSON."""
    print(json.dumps(evaluate_profiles(build_holdout_corpus()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
