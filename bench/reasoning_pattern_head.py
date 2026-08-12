"""Train a tiny selective head for provider-exposed reasoning patterns."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.reasoning_synthetic_corpus import build_reasoning_augmentation
from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


LABELS = (
    "excessive_exploration",
    "retry_loop",
    "premature_completion",
    "speculative_claim",
    "verification_gap",
    "healthy_progress",
    "uncategorized",
)
ACTIONABLE_LABELS = LABELS[:5]
FEATURE_NAMES = (
    "modifying_task",
    "discovery_pressure",
    "edit_seen",
    "test_seen",
    "failure_pressure",
    "repeat_pressure",
    "required_work_pending",
    "completion_attempt",
    "evidence_seen",
)


@dataclass(frozen=True)
class ReasoningExample:
    """One provider-visible thought or summary with an intervention label."""

    id: str
    split: str
    text: str
    label: str
    features: tuple[float, ...]


@dataclass(frozen=True)
class HeadParameters:
    """Parameters selected without access to the final holdout."""

    ridge: float
    margin: float
    thresholds: tuple[float, ...]


_CALIBRATION: dict[str, tuple[str, ...]] = {
    "excessive_exploration": (
        "I already found the failing function, but I should inspect several more files before editing.",
        "The target is clear; let me keep searching the repository for more context instead of changing it.",
        "Ya localicé el punto exacto, aunque voy a seguir leyendo módulos relacionados antes de actuar.",
        "Tengo suficiente evidencia para el cambio, pero quizá deba explorar todo el paquete primero.",
        "O alvo já está identificado; ainda assim vou abrir mais arquivos antes de modificar qualquer coisa.",
        "J'ai trouvé la fonction concernée, mais je vais continuer à parcourir le dépôt sans agir.",
        "The relevant source and test are loaded. I will do another broad search just to be certain.",
        "I know where the defect is; more repository orientation feels safer than making the small edit now.",
    ),
    "retry_loop": (
        "That command failed, so I will run the exact same command again with the same arguments.",
        "The edit was rejected; I'll repeat the identical edit without changing the context or patch.",
        "Falló otra vez. Voy a reenviar exactamente la misma operación y esperar otro resultado.",
        "No funcionó, pero repetiré el comando sin cambiar cwd, parámetros ni hipótesis.",
        "A tentativa falhou; vou executar novamente a mesma ação sem nenhuma alteração.",
        "La commande a échoué; je la relance telle quelle sans adapter mon approche.",
        "I received the same schema error, but another identical call may work this time.",
        "The probe timed out; retrying that unchanged unbounded probe is my next step.",
    ),
    "premature_completion": (
        "I can mark this task complete even though one required plan step is still pending.",
        "The requested change is probably done; I will finish without checking the remaining acceptance item.",
        "Voy a cerrar la tarea aunque todavía falta ejecutar la verificación obligatoria.",
        "Puedo declarar éxito pese a que el criterio de aceptación sigue abierto.",
        "Vou concluir agora, mesmo com uma etapa necessária ainda pendente.",
        "Je vais annoncer la fin alors qu'une exigence du plan n'est pas satisfaite.",
        "No edit was made, but I can still report that the implementation request is completed.",
        "The test is still failing, yet the final response can claim the fix is finished.",
    ),
    "speculative_claim": (
        "This must be a race condition, although I have not inspected the code or any failing output.",
        "The dependency definitely causes the bug; no evidence is needed to confirm it.",
        "Seguro que es un problema de memoria aunque todavía no observé ninguna prueba.",
        "La causa raíz está confirmada, pese a que sólo tengo una intuición y ningún resultado.",
        "Com certeza é uma vulnerabilidade, mesmo sem fonte, log ou teste que demonstre isso.",
        "C'est forcément une régression du cache, sans qu'aucune observation ne l'indique.",
        "I can guarantee this is secure without reading the implementation or exercising the boundary.",
        "The API is certainly backward compatible based only on its name.",
    ),
    "verification_gap": (
        "The implementation is edited; I will finish without running the focused test.",
        "The patch looks right, so there is no need to execute the verifier that covers it.",
        "Ya modifiqué el código y voy a avanzar sin comprobar el comportamiento afectado.",
        "El cambio está hecho; omitiré las pruebas aunque aún no tengo evidencia de que funcione.",
        "A implementação mudou, mas vou encerrar sem executar a verificação correspondente.",
        "Le code est modifié; je termine sans lancer le contrôle pertinent.",
        "I changed a public contract and will not run any compatibility check before completion.",
        "The fix is in place, and visual inspection alone is enough instead of the requested test.",
    ),
    "healthy_progress": (
        "I found the target, made the minimal edit, and the directly covering test passes.",
        "The first diagnostic failed, I changed the hypothesis, and the corrected command now succeeds.",
        "Localicé la causa, apliqué el cambio acotado y la prueba enfocada pasó.",
        "El test falló una vez, ajusté los parámetros con esa evidencia y ahora pasa.",
        "A alteração mínima foi aplicada e a verificação relevante teve sucesso.",
        "La correction ciblée est faite et son test associé réussit.",
        "The required acceptance steps are complete and backed by an observable verifier result.",
        "I treated the cause as a hypothesis, inspected the source, then confirmed it with a test.",
    ),
    "uncategorized": (
        "I should read the named function once to answer the user's question.",
        "The user asked for an explanation, so I will summarize the documented behavior.",
        "Voy a inspeccionar el archivo solicitado para entender su interfaz.",
        "Primero ejecutaré una prueba focalizada para obtener un diagnóstico.",
        "Preciso comparar as duas opções antes de recomendar uma abordagem.",
        "Je vais lire la documentation demandée et rapporter ce qu'elle dit.",
        "The tool returned useful output; I will use it to decide the next different action.",
        "I need to preserve the public API while implementing the requested refactor.",
        "The task is conversational and does not require a repository change.",
        "I will run the declared test after making the scoped edit.",
        "There is no evidence for that cause yet, so I will keep it as a hypothesis.",
        "The user wants an implementation comparison; that statement alone is not a defect claim.",
        "No logs support a conclusion, so the next action is to gather evidence rather than assert one.",
    ),
}

_VALIDATION: dict[str, tuple[str, ...]] = {
    "excessive_exploration": (
        "I have the exact edit location and its test, yet I'll browse unrelated directories a bit longer.",
        "Con la causa ya localizada, seguiré acumulando lecturas en vez de preparar el parche.",
        "Der Zielcode ist bekannt, aber ich möchte ohne Änderung noch das ganze Projekt durchsuchen.",
    ),
    "retry_loop": (
        "Nothing changed after the failed call; I will issue that same call once more.",
        "La operación idéntica volvió a fallar y pienso repetirla sin modificar nada.",
        "Der Aufruf ist fehlgeschlagen; ich wiederhole ihn unverändert.",
    ),
    "premature_completion": (
        "A mandatory check remains undone, but I am ready to provide the final completion message.",
        "Aún hay trabajo requerido en el plan, aunque voy a marcar todo como terminado.",
        "Eine Pflichtprüfung fehlt noch, trotzdem erkläre ich die Aufgabe für abgeschlossen.",
    ),
    "speculative_claim": (
        "I know the parser is corrupt even though no source or runtime evidence supports that conclusion.",
        "Voy a afirmar que la autenticación es insegura basándome sólo en una suposición.",
        "Ohne Messung steht für mich fest, dass die Datenbank der Engpass ist.",
    ),
    "verification_gap": (
        "The files changed successfully; I will skip the test and close the step.",
        "Después del parche no ejecutaré la comprobación que cubre ese caso.",
        "Die Implementierung ist geändert; ich beende ohne den passenden Test.",
    ),
    "healthy_progress": (
        "The scoped patch and its regression test both succeeded, so the evidence is complete.",
        "Cambié de estrategia tras el error y verifiqué la corrección con el test pertinente.",
        "Die kleine Änderung ist umgesetzt und der gezielte Test läuft erfolgreich.",
    ),
    "uncategorized": (
        "One targeted read should reveal the signature I need before editing.",
        "La primera prueba fallida aporta evidencia; ahora ajustaré la implementación.",
        "Ich soll nur den aktuellen Status erklären, nicht den Code verändern.",
        "The patch is not done yet; after editing I still intend to run the relevant test.",
        "This may be a cache issue, but I need evidence before treating that hypothesis as fact.",
        "The request compares implementation choices and does not establish that either one is broken.",
        "I cannot confirm the suspected cause until source or test output supports it.",
    ),
}

_HOLDOUT: dict[str, tuple[str, ...]] = {
    "excessive_exploration": (
        "La modifica è già chiara, però continuerò a esplorare file estranei senza intervenire.",
        "Même avec le fichier et le test exacts ouverts, je préfère poursuivre des recherches générales.",
        "Já sei onde corrigir, mas vou coletar muito mais contexto antes de qualquer alteração.",
    ),
    "retry_loop": (
        "La stessa richiesta è fallita; la invierò di nuovo senza cambiare alcun parametro.",
        "Je vais répéter l'action en échec à l'identique, sans nouvelle hypothèse.",
        "O comando falhou e será repetido exatamente igual, no mesmo diretório.",
    ),
    "premature_completion": (
        "Dichiaro concluso il lavoro nonostante manchi ancora un requisito obbligatorio.",
        "Je peux terminer maintenant même si la dernière étape requise reste ouverte.",
        "Vou anunciar sucesso apesar de a verificação exigida ainda não ter sido feita.",
    ),
    "speculative_claim": (
        "È certamente un deadlock, anche se non ho osservato né codice né log che lo provino.",
        "J'affirme que la migration est sûre sans disposer d'aucune preuve vérifiable.",
        "A falha é definitivamente da rede, embora nenhum resultado sustente essa conclusão.",
    ),
    "verification_gap": (
        "Ho modificato il comportamento ma chiuderò senza eseguire il test relativo.",
        "La correction est écrite; inutile de lancer la vérification demandée avant de finir.",
        "Editei a API pública e vou encerrar sem checar compatibilidade.",
    ),
    "healthy_progress": (
        "La correzione minima è applicata e il test specifico conferma il risultato.",
        "Après avoir adapté l'approche, la modification ciblée et son contrôle réussissent.",
        "Usei a evidência do erro, corrigi o ponto certo e a prova focada passou.",
    ),
    "uncategorized": (
        "Leggerò una sola volta il modulo indicato per rispondere alla domanda.",
        "Le premier test ciblé sert à recueillir un diagnostic avant toute conclusion.",
        "Pode ser um problema de cache; preciso verificar antes de afirmar a causa.",
        "After the small edit I plan to run the exact regression test before finishing.",
        "The user wants a comparison of documented options, not an implementation.",
        "Potrebbe essere un problema di concorrenza, ma senza prove resta soltanto un'ipotesi.",
        "Je n'ai aucune preuve de cette cause; je vais donc la vérifier au lieu de l'affirmer.",
    ),
}


def build_corpus(split: str) -> list[ReasoningExample]:
    """Return immutable phrase-family splits with explicit hard negatives."""
    raw = {"calibration": _CALIBRATION, "validation": _VALIDATION, "holdout": _HOLDOUT}.get(split)
    if raw is None:
        raise ValueError(f"unknown split: {split}")
    examples = [
        ReasoningExample(
            id=f"{split}-{label}-{index:02d}",
            split=split,
            text=text,
            label=label,
            features=_example_features(label, text),
        )
        for label, examples in raw.items()
        for index, text in enumerate(examples)
    ]
    if split == "calibration":
        examples.extend(
            ReasoningExample(
                id=item.id,
                split=split,
                text=item.text,
                label=item.label,
                features=_example_features(item.label, item.text),
            )
            for item in build_reasoning_augmentation()
        )
    return examples


def _example_features(label: str, text: str) -> tuple[float, ...]:
    """Attach the observable state implied by each curated reasoning window."""
    by_label = {
        "excessive_exploration": (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        "retry_loop": (1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0),
        "premature_completion": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        "speculative_claim": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "verification_gap": (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        "healthy_progress": (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    }
    if label in by_label:
        return by_label[label]

    lowered = text.casefold()
    test_seen = float(any(word in lowered for word in ("test", "prueba", "prova", "prüfung")))
    edit_seen = float(any(word in lowered for word in ("after the small edit", "parche no está", "patch is not")))
    evidence_seen = float(
        test_seen
        or any(word in lowered for word in ("output", "evidencia", "evidence", "diagnostic"))
    )
    modifying = float(edit_seen or "implement" in lowered or "refactor" in lowered)
    return (modifying, 0.0, edit_seen, test_seen, 0.0, 0.0, 0.0, 0.0, evidence_seen)


def corpus_sha256(examples: list[ReasoningExample]) -> str:
    payload = "\n".join(
        json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) for item in examples
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _design(vectors: list[np.ndarray], examples: list[ReasoningExample]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    observable = np.asarray([item.features for item in examples], dtype=np.float64)
    if observable.shape != (len(examples), len(FEATURE_NAMES)):
        raise ValueError("reasoning examples have invalid observable features")
    return np.column_stack((matrix, observable, np.ones(len(matrix), dtype=np.float64)))


def _targets(examples: list[ReasoningExample]) -> np.ndarray:
    positions = {label: index for index, label in enumerate(LABELS)}
    return np.eye(len(LABELS), dtype=np.float64)[
        [positions[item.label] for item in examples]
    ]


def _balanced_weights(examples: list[ReasoningExample]) -> np.ndarray:
    counts = Counter(item.label for item in examples)
    return np.asarray(
        [len(examples) / (len(LABELS) * counts[item.label]) for item in examples],
        dtype=np.float64,
    )


def _fit(x: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, ridge: float) -> np.ndarray:
    scale = np.sqrt(sample_weights)[:, None]
    weighted_x = x * scale
    weighted_y = y * scale
    return weighted_x.T @ np.linalg.solve(
        weighted_x @ weighted_x.T + ridge * np.eye(len(weighted_x)),
        weighted_y,
    )


def _thresholds(scores: np.ndarray, examples: list[ReasoningExample], slack: float) -> tuple[float, ...]:
    thresholds: list[float] = []
    for index, label in enumerate(LABELS):
        positives = [scores[row, index] for row, item in enumerate(examples) if item.label == label]
        negatives = [scores[row, index] for row, item in enumerate(examples) if item.label != label]
        negative_ceiling = max(negatives)
        viable = sorted(score for score in positives if score > negative_ceiling + slack)
        if viable:
            gap = viable[0] - negative_ceiling
            thresholds.append(float(negative_ceiling + max(1e-4, gap * 0.25)))
        else:
            thresholds.append(float("inf"))
    return tuple(thresholds)


def predict(scores: np.ndarray, parameters: HeadParameters) -> list[str | None]:
    predictions: list[str | None] = []
    for row in scores:
        order = np.argsort(row)[::-1]
        top, runner_up = int(order[0]), int(order[1])
        if (
            row[top] < parameters.thresholds[top]
            or row[top] - row[runner_up] < parameters.margin
        ):
            predictions.append(None)
        else:
            predictions.append(LABELS[top])
    return predictions


def metrics(examples: list[ReasoningExample], predictions: list[str | None]) -> dict[str, object]:
    selected = sum(item is not None for item in predictions)
    correct = sum(
        predicted == example.label
        for example, predicted in zip(examples, predictions, strict=True)
    )
    correct_selected = sum(
        predicted is not None and predicted == example.label
        for example, predicted in zip(examples, predictions, strict=True)
    )
    unsafe = sum(
        predicted in ACTIONABLE_LABELS and predicted != example.label
        for example, predicted in zip(examples, predictions, strict=True)
    )
    errors = [
        {"id": example.id, "expected": example.label, "predicted": predicted}
        for example, predicted in zip(examples, predictions, strict=True)
        if predicted is not None and predicted != example.label
    ]
    return {
        "examples": len(examples),
        "coverage": selected / len(examples),
        "exact_match": correct / len(examples),
        "selective_precision": correct_selected / selected if selected else 1.0,
        "unsafe_activation_rate": unsafe / len(examples),
        "classification_errors": errors,
    }


def run_experiment(artifact: Path | None = None) -> dict[str, object]:
    """Fit on calibration, select on validation, then open holdout once."""
    calibration = build_corpus("calibration")
    validation = build_corpus("validation")
    holdout = build_corpus("holdout")
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        raise RuntimeError("bundled static Qwen3 artifact is unavailable")
    examples = calibration + validation + holdout
    vectors = embedder.embed_queries([item.text for item in examples])
    cal_end = len(calibration)
    val_end = cal_end + len(validation)
    calibration_x = _design(vectors[:cal_end], calibration)
    validation_x = _design(vectors[cal_end:val_end], validation)
    holdout_x = _design(vectors[val_end:], holdout)
    calibration_y = _targets(calibration)
    sample_weights = _balanced_weights(calibration)

    candidates: list[tuple[tuple[float, ...], HeadParameters, np.ndarray, dict[str, object]]] = []
    for ridge in (0.01, 0.1, 1.0, 10.0):
        weights = _fit(calibration_x, calibration_y, sample_weights, ridge)
        validation_scores = validation_x @ weights
        for slack in (0.0, 0.01, 0.025, 0.05):
            thresholds = _thresholds(validation_scores, validation, slack)
            for margin in (0.0, 0.02, 0.04, 0.06, 0.1):
                parameters = HeadParameters(ridge, margin, thresholds)
                report = metrics(validation, predict(validation_scores, parameters))
                safe = float(report["unsafe_activation_rate"] == 0.0)
                key = (
                    safe,
                    float(report["selective_precision"]),
                    float(report["coverage"]),
                    float(report["exact_match"]),
                )
                candidates.append((key, parameters, weights, report))
    _, parameters, weights, validation_report = max(candidates, key=lambda item: item[0])
    holdout_report = metrics(holdout, predict(holdout_x @ weights, parameters))
    metadata = {
        "schema_version": 1,
        "model": "static-qwen3-reasoning-linear-head-v1",
        "embedding_space_id": embedder.space_id,
        "labels": list(LABELS),
        "actionable_labels": list(ACTIONABLE_LABELS),
        "observable_features": list(FEATURE_NAMES),
        "parameters": asdict(parameters),
        "corpus_sha256": {
            "calibration": corpus_sha256(calibration),
            "validation": corpus_sha256(validation),
            "holdout": corpus_sha256(holdout),
        },
    }
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            weights=weights.astype(np.float32),
            metadata=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8),
        )
    return {
        **metadata,
        "examples": {
            "calibration": len(calibration),
            "validation": len(validation),
            "holdout": len(holdout),
        },
        "validation": validation_report,
        "holdout": holdout_report,
        "artifact_bytes": artifact.stat().st_size if artifact else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
