"""Validation corpus for selective task routing with static Qwen3 embeddings.

Unlike ``task_policy_eval``, these requests deliberately avoid the router's
literal operation vocabulary. This measures whether the semantic stage adds
useful coverage, how often it abstains, and whether it ever invents write
authority. Run with ``uv run python -m bench.task_policy_semantic_eval``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json

from infinidev.engine.task_policies.router import resolve_task_profile
from infinidev.tools.base.static_qwen3_embedder import STATIC_QWEN3_MODEL


@dataclass(frozen=True)
class SemanticFamily:
    """One phrasing family outside the literal operation vocabulary."""

    id: str
    templates: tuple[str, ...]
    policy: str | None
    write_authority: bool


@dataclass(frozen=True)
class SemanticExample:
    """One materialized semantic-routing validation case."""

    id: str
    text: str
    policy: str | None
    write_authority: bool


_OBJECTS = (
    "the session cache",
    "the callback dispatcher",
    "the command palette",
    "the authentication adapter",
    "el cargador de configuración",
    "el registro de plugins",
    "la interfaz de terminal",
    "el cliente de red",
)

FAMILIES: tuple[SemanticFamily, ...] = (
    SemanticFamily(
        "bugfix-paraphrase",
        (
            "Make {object}, which started returning stale results, behave as specified again.",
            "Haz que {object}, que ahora produce resultados equivocados, vuelva a cumplir su contrato.",
        ),
        "bugfix.root_cause", True,
    ),
    SemanticFamily(
        "feature-paraphrase",
        (
            "I need {object} to handle a user case it cannot handle today.",
            "Quiero que {object} pueda resolver un caso de usuario que hoy no admite.",
        ),
        "feature.contract_first", True,
    ),
    SemanticFamily(
        "refactor-paraphrase",
        (
            "Make {object} easier to follow internally while keeping every output stable.",
            "Quiero que {object} sea más fácil de mantener sin alterar lo que hace.",
        ),
        "refactor.preserve_behavior", True,
    ),
    SemanticFamily(
        "research-paraphrase",
        (
            "Tell me which approach around {object} fits best and justify the recommendation with reliable material.",
            "Dime qué enfoque para {object} conviene y justifica la recomendación con material fiable.",
        ),
        "research.evidence_first", False,
    ),
    SemanticFamily(
        "review-paraphrase",
        (
            "Inspect {object} for flaws and give me a prioritized report; leave the code untouched.",
            "Inspecciona {object} buscando defectos y entrega un informe priorizado; no edites el código.",
        ),
        "review.read_only", False,
    ),
    SemanticFamily(
        "performance-paraphrase",
        (
            "{object} takes too long under load. Measure where the time goes and improve it.",
            "{object} tarda demasiado con carga. Mide dónde se va el tiempo y aceléralo.",
        ),
        "performance.measure_first", True,
    ),
    SemanticFamily(
        "quoted-neutral",
        (
            '{object} logs "please implement fix"; explain that message.',
            '{object} muestra "refactor required"; interpreta ese mensaje.',
        ),
        None, False,
    ),
)


def build_semantic_validation_corpus() -> list[SemanticExample]:
    """Materialize 112 family-separated validation examples."""
    examples: list[SemanticExample] = []
    for family in FAMILIES:
        for template_index, template in enumerate(family.templates):
            for object_index, object_name in enumerate(_OBJECTS):
                examples.append(SemanticExample(
                    id=f"{family.id}-{template_index:02d}-{object_index:02d}",
                    text=template.format(object=object_name),
                    policy=family.policy,
                    write_authority=family.write_authority,
                ))
    return examples


def evaluate_semantic_profiles(
    examples: list[SemanticExample],
) -> dict[str, object]:
    """Measure selective accuracy, abstention, authority, and vector identity."""
    total = len(examples)
    exact = 0
    covered = 0
    correct_when_covered = 0
    false_activation = 0
    false_write_authority = 0
    space_ids: set[str] = set()
    by_policy: dict[str, dict[str, int]] = defaultdict(
        lambda: {"expected": 0, "selected": 0, "correct": 0}
    )
    for example in examples:
        profile = resolve_task_profile(
            example.text,
            enable_embeddings=True,
            embedding_threshold=0.18,
            embedding_margin=0.04,
        )
        actual = {item.id for item in profile.selected_policies}
        expected = {example.policy} if example.policy else set()
        exact += actual == expected
        if actual:
            covered += 1
            correct_when_covered += actual == expected
        false_activation += bool(actual and not expected)
        actual_write = bool(set(profile.authority) & {"modify", "commit", "publish"})
        false_write_authority += actual_write and not example.write_authority
        if profile.semantic_space_id:
            space_ids.add(profile.semantic_space_id)
        if example.policy:
            by_policy[example.policy]["expected"] += 1
            by_policy[example.policy]["correct"] += example.policy in actual
        for selected in actual:
            by_policy[selected]["selected"] += 1

    return {
        "examples": total,
        "exact_match": exact / total if total else 0.0,
        "coverage": covered / total if total else 0.0,
        "selective_precision": (
            correct_when_covered / covered if covered else 1.0
        ),
        "false_activation_rate": false_activation / total if total else 0.0,
        "false_write_authority_rate": false_write_authority / total if total else 0.0,
        "embedding_model": STATIC_QWEN3_MODEL,
        "space_ids": sorted(space_ids),
        "per_policy": dict(sorted(by_policy.items())),
    }


def main() -> None:
    """Print semantic validation metrics as machine-readable JSON."""
    print(json.dumps(
        evaluate_semantic_profiles(build_semantic_validation_corpus()),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
