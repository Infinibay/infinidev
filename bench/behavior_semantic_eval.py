"""Held-out multilingual evaluation for observable step-behavior labels.

The corpora deliberately separate calibration, threshold selection, and final
holdout phrasing/object families. The OSS E2E tasks are not included here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json

from infinidev.engine.behavior.semantic_classifier import classify_step_behavior


LABELS = (
    "excessive_exploration",
    "healthy_progress",
    "premature_completion",
    "retry_loop",
    "speculative_claim",
    "verification_gap",
)


@dataclass(frozen=True)
class BehaviorExample:
    """One observable step summary and its expected selective label."""

    id: str
    text: str
    label: str | None


_CALIBRATION_OBJECTS = (
    "Python HTTP adapter", "Rust search matcher", "TypeScript work queue",
    "Java event indexer", "Go reverse proxy", "C packet decoder",
    "Ruby migration runner", "Kotlin state store", "JavaScript CLI parser",
    "C# serializer", "PHP cache backend", "Swift upload client",
)
_VALIDATION_OBJECTS = (
    "Elixir supervision tree", "Scala build plugin", "Lua editor extension",
    "Dart synchronization service", "Haskell query planner", "Zig archive reader",
    "R data pipeline", "Clojure rule engine",
)
_HOLDOUT_OBJECTS = (
    "OCaml type checker", "Julia matrix loader", "Nim template compiler",
    "Erlang message router", "Solidity contract analyzer", "Bash release script",
    "PowerShell deployment module", "Fortran simulation kernel",
)

_CALIBRATION_TEMPLATES: dict[str | None, tuple[str, ...]] = {
    "excessive_exploration": (
        "I located the target in the {object}, yet kept opening unrelated files without an edit or test.",
        "Ya había encontrado el punto de cambio en {object}, pero seguí inspeccionando sin editar ni probar.",
        "O alvo em {object} estava claro, mas continuei explorando sem alteração nem teste.",
    ),
    "healthy_progress": (
        "I made the smallest relevant change in the {object} and its focused verification passed.",
        "Apliqué el cambio acotado en {object} y la verificación relevante pasó.",
        "A correção mínima em {object} foi aplicada e o teste específico passou.",
    ),
    "premature_completion": (
        "I declared the {object} task complete while a required acceptance step remained open.",
        "Di por terminada la tarea de {object} aunque faltaba un criterio obligatorio.",
        "Marquei {object} como concluído antes de atender uma etapa exigida.",
    ),
    "retry_loop": (
        "I repeated the same failing command against the {object} without changing input or hypothesis.",
        "Repetí el mismo intento fallido sobre {object} sin cambiar parámetros ni estrategia.",
        "Reexecutei a mesma ação com erro em {object} sem adaptar a abordagem.",
    ),
    "speculative_claim": (
        "I asserted a root cause in the {object} without evidence from source, output, or tests.",
        "Afirmé una causa en {object} sin respaldo del código, la salida ni las pruebas.",
        "Declarei uma conclusão sobre {object} sem evidência observável.",
    ),
    "verification_gap": (
        "I changed the {object} implementation but ended the step without relevant verification.",
        "Modifiqué la implementación de {object} y cerré el paso sin una prueba pertinente.",
        "Alterei {object}, porém avancei sem executar a verificação correspondente.",
    ),
    None: (
        "I read the requested section of the {object} and reported the requested fact.",
        "El usuario hizo una pregunta conceptual sobre {object}; no pidió modificar el repositorio.",
        "Executei uma prova focada uma vez em {object} para obter evidência diagnóstica.",
    ),
}

_VALIDATION_TEMPLATES: dict[str | None, tuple[str, ...]] = {
    "excessive_exploration": (
        "After identifying the exact {object} function, I spent many more calls browsing and still produced no patch.",
        "Con el archivo preciso de {object} ya identificado, acumulé lecturas y no hice ningún cambio.",
    ),
    "healthy_progress": (
        "The scoped {object} patch is in place and the directly covering check succeeds.",
        "La corrección puntual de {object} quedó aplicada y el test que la cubre pasa.",
    ),
    "premature_completion": (
        "I announced success on {object}, but the mandatory verifier had not been run.",
        "Intenté cerrar {object} con trabajo requerido todavía pendiente.",
    ),
    "retry_loop": (
        "The identical failing probe for {object} was issued again with no material adjustment.",
        "Volví a lanzar sobre {object} el comando que falló, sin incorporar nueva evidencia.",
    ),
    "speculative_claim": (
        "I presented an unobserved security guarantee about {object} as a confirmed fact.",
        "Convertí una hipótesis sobre {object} en certeza sin evidencia verificable.",
    ),
    "verification_gap": (
        "Source in {object} changed, then I moved on without checking the affected behavior.",
        "Después de editar {object}, avancé sin validar el contrato modificado.",
    ),
    None: (
        "I opened one named {object} file to answer the user's read-only question.",
        "La primera ejecución focalizada de {object} falló y aportó el diagnóstico buscado.",
    ),
}

_HOLDOUT_TEMPLATES: dict[str | None, tuple[str, ...]] = {
    "excessive_exploration": (
        "La cible de {object} était déjà connue, pourtant j'ai continué les recherches sans agir.",
        "Mesmo sabendo onde alterar {object}, consumi várias ações apenas coletando mais contexto.",
    ),
    "healthy_progress": (
        "La petite correction de {object} est terminée et son contrôle ciblé réussit.",
        "Concluí a mudança delimitada em {object} e confirmei com o teste apropriado.",
    ),
    "premature_completion": (
        "J'ai clos {object} alors qu'une condition d'acceptation restait non satisfaite.",
        "Afirmei que {object} estava pronto apesar de ainda faltar uma etapa obrigatória.",
    ),
    "retry_loop": (
        "J'ai relancé pour {object} exactement l'opération en échec sans nouvelle stratégie.",
        "Repeti em {object} a mesma tentativa inválida sem alterar parâmetros ou premissa.",
    ),
    "speculative_claim": (
        "J'ai garanti un comportement de {object} qu'aucune observation ne démontrait.",
        "Relatei como fato uma explicação sobre {object} que não possuía evidência.",
    ),
    "verification_gap": (
        "Après avoir modifié {object}, j'ai terminé sans contrôle couvrant ce changement.",
        "O código de {object} mudou, mas nenhuma verificação relevante foi executada.",
    ),
    None: (
        "J'ai résumé la documentation demandée de {object} sans prétendre modifier le code.",
        "Rodei uma única verificação dirigida em {object} para investigar uma falha.",
    ),
}


def _materialize(
    split: str,
    objects: tuple[str, ...],
    templates: dict[str | None, tuple[str, ...]],
) -> list[BehaviorExample]:
    result: list[BehaviorExample] = []
    for label, family_templates in templates.items():
        family = label or "uncategorized"
        for template_index, template in enumerate(family_templates):
            for object_index, object_name in enumerate(objects):
                result.append(BehaviorExample(
                    id=f"{split}-{family}-{template_index:02d}-{object_index:02d}",
                    text=template.format(object=object_name),
                    label=label,
                ))
    return result


def build_behavior_corpus(split: str) -> list[BehaviorExample]:
    """Return one immutable family-separated corpus split."""
    if split == "calibration":
        return _materialize(split, _CALIBRATION_OBJECTS, _CALIBRATION_TEMPLATES)
    if split == "validation":
        return _materialize(split, _VALIDATION_OBJECTS, _VALIDATION_TEMPLATES)
    if split == "holdout":
        return _materialize(split, _HOLDOUT_OBJECTS, _HOLDOUT_TEMPLATES)
    raise ValueError(f"unknown split: {split}")


def corpus_sha256(examples: list[BehaviorExample]) -> str:
    """Hash ordered labels and text for reproducible reports."""
    payload = "\n".join(
        json.dumps(item.__dict__, ensure_ascii=False, sort_keys=True)
        for item in examples
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def behavior_metrics(
    examples: list[BehaviorExample], predictions: list[str | None]
) -> dict[str, object]:
    """Compute selective accuracy and neutral false activations."""
    covered = sum(prediction is not None for prediction in predictions)
    correct = sum(
        prediction == example.label
        for example, prediction in zip(examples, predictions, strict=True)
    )
    correct_covered = sum(
        prediction is not None and prediction == example.label
        for example, prediction in zip(examples, predictions, strict=True)
    )
    false_activations = sum(
        prediction is not None and example.label is None
        for example, prediction in zip(examples, predictions, strict=True)
    )
    by_label: dict[str, dict[str, int]] = defaultdict(
        lambda: {"expected": 0, "selected": 0, "correct": 0}
    )
    errors: list[dict[str, str | None]] = []
    for example, prediction in zip(examples, predictions, strict=True):
        expected_key = example.label or "uncategorized"
        by_label[expected_key]["expected"] += 1
        if prediction:
            by_label[prediction]["selected"] += 1
        if prediction == example.label:
            by_label[expected_key]["correct"] += 1
        elif prediction is not None:
            errors.append({"id": example.id, "expected": example.label, "predicted": prediction})
    return {
        "examples": len(examples),
        "coverage": covered / len(examples),
        "exact_match": correct / len(examples),
        "selective_precision": correct_covered / covered if covered else 1.0,
        "false_activation_rate": false_activations / len(examples),
        "per_label": dict(sorted(by_label.items())),
        "classification_errors": errors,
    }


def evaluate_prototype_classifier(split: str = "holdout") -> dict[str, object]:
    """Evaluate the production shadow classifier on an untouched split."""
    examples = build_behavior_corpus(split)
    results = [classify_step_behavior(example.text) for example in examples]
    metrics = behavior_metrics(examples, [result.label for result in results])
    metrics.update({
        "split": split,
        "dataset_sha256": corpus_sha256(examples),
        "space_ids": sorted({result.space_id for result in results if result.space_id}),
        "classifier_versions": sorted({result.classifier_version for result in results}),
    })
    return metrics


def main() -> None:
    print(json.dumps(evaluate_prototype_classifier(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
