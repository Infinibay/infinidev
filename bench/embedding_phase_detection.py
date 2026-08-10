"""Evaluate static embeddings as a bilingual plan-phase detector.

The benchmark deliberately separates template families and work objects between
training, calibration, and test.  It compares the current regular-expression
classifier with a small ridge classifier over frozen static embeddings, then
measures the conservative hybrid Infinidev could actually deploy: exact rules
win, while the semantic detector may only fill an otherwise unknown phase.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from infinidev.engine.loop.loop_plan import _step_phase
from infinidev.tools.base.static_qwen3_embedder import StaticQwen3Embedder

try:
    from bench.collect_openai_embedding_teacher import CachedTeacherEmbedder
except ModuleNotFoundError:
    from collect_openai_embedding_teacher import CachedTeacherEmbedder


PHASES = ("discover", "change", "test_change", "verify", "document", "design")
LABELS = (*PHASES, "other")

OBJECTS = {
    "train": {
        "en": (
            "callback registry", "authentication cache", "JSON parser",
            "database migration", "CLI command router", "HTTP retry policy",
            "file watcher", "configuration loader",
        ),
        "es": (
            "registro de callbacks", "caché de autenticación", "parser de JSON",
            "migración de base de datos", "enrutador de comandos CLI",
            "política de reintentos HTTP", "monitor de archivos",
            "cargador de configuración",
        ),
    },
    "calibration": {
        "en": ("event dispatcher", "symbol index", "test runner", "session history"),
        "es": (
            "despachador de eventos", "índice de símbolos", "ejecutor de pruebas",
            "historial de sesión",
        ),
    },
    "test": {
        "en": (
            "streaming renderer", "permission broker", "syntax tree index",
            "task scheduler", "vector store", "tool transcript",
            "plugin loader", "terminal layout",
        ),
        "es": (
            "renderizador de streaming", "gestor de permisos",
            "índice del árbol sintáctico", "planificador de tareas",
            "almacén vectorial", "transcripción de herramientas",
            "cargador de plugins", "diseño de la terminal",
        ),
    },
}

TEMPLATES = {
    "train": {
        "en": {
            "discover": (
                "Inspect {obj} to understand its behavior",
                "Trace the control flow through {obj}",
                "Explore how {obj} currently works",
            ),
            "change": (
                "Implement the requested behavior in {obj}",
                "Modify {obj} to support the new capability",
                "Refactor {obj} to remove the limitation",
            ),
            "test_change": (
                "Add regression tests for {obj}",
                "Create new test coverage for {obj}",
                "Extend the test suite around {obj}",
            ),
            "verify": (
                "Run existing tests for {obj}",
                "Validate {obj} without editing it",
                "Check the completed {obj}",
            ),
            "document": (
                "Document the public behavior of {obj}",
                "Explain {obj} in the project guide",
                "Update the documentation for {obj}",
            ),
            "design": (
                "Design the architecture of {obj}",
                "Plan an implementation strategy for {obj}",
                "Prototype an interface for {obj}",
            ),
            "other": (
                "Current status of {obj}",
                "Questions about {obj}",
                "The repository contains {obj}",
            ),
        },
        "es": {
            "discover": (
                "Inspecciona {obj} para entender su comportamiento",
                "Traza el flujo de control que atraviesa {obj}",
                "Explora cómo funciona actualmente {obj}",
            ),
            "change": (
                "Implementa el comportamiento solicitado en {obj}",
                "Modifica {obj} para soportar la capacidad nueva",
                "Refactoriza {obj} para eliminar la limitación",
            ),
            "test_change": (
                "Agrega pruebas de regresión para {obj}",
                "Crea cobertura de tests nueva para {obj}",
                "Amplía la suite de pruebas alrededor de {obj}",
            ),
            "verify": (
                "Ejecuta las pruebas existentes para {obj}",
                "Valida {obj} sin editarlo",
                "Comprueba {obj} una vez terminado",
            ),
            "document": (
                "Documenta el comportamiento público de {obj}",
                "Explica {obj} en la guía del proyecto",
                "Actualiza la documentación de {obj}",
            ),
            "design": (
                "Diseña la arquitectura de {obj}",
                "Planifica una estrategia de implementación para {obj}",
                "Crea un prototipo de interfaz para {obj}",
            ),
            "other": (
                "Estado actual de {obj}",
                "Preguntas acerca de {obj}",
                "El repositorio contiene {obj}",
            ),
        },
    },
    "calibration": {
        "en": {
            "discover": ("Survey the internals of {obj}", "Examine {obj} before coding"),
            "change": ("Enable the missing feature in {obj}", "Teach {obj} the new behavior"),
            "test_change": ("Protect {obj} with new regression cases", "Cover {obj} with tests"),
            "verify": ("Exercise {obj} and confirm the result", "Measure whether {obj} works"),
            "document": ("Describe the contract of {obj}", "Record how to use {obj}"),
            "design": ("Outline an architecture for {obj}", "Propose a design for {obj}"),
            "other": ("What is the status of {obj}?", "A note concerning {obj}"),
        },
        "es": {
            "discover": ("Estudia los componentes internos de {obj}", "Examina {obj} antes de programar"),
            "change": ("Habilita la función faltante en {obj}", "Haz que {obj} admita el comportamiento nuevo"),
            "test_change": ("Protege {obj} con casos de regresión nuevos", "Cubre {obj} mediante tests"),
            "verify": ("Ejercita {obj} y confirma el resultado", "Mide si {obj} funciona"),
            "document": ("Describe el contrato de {obj}", "Deja constancia de cómo usar {obj}"),
            "design": ("Esboza una arquitectura para {obj}", "Propón un diseño para {obj}"),
            "other": ("¿Cuál es el estado de {obj}?", "Una nota acerca de {obj}"),
        },
    },
    "test": {
        "en": {
            "discover": (
                "Audit the current behavior of {obj}",
                "Map the path data takes through {obj}",
                "Review {obj} closely before touching code",
                "Investigate the cause inside {obj}",
            ),
            "change": (
                "Make {obj} accept the additional case",
                "Introduce the missing capability into {obj}",
                "Revise {obj} so the issue cannot recur",
                "Bring the requested behavior to {obj}",
            ),
            "test_change": (
                "Write cases that reproduce failures in {obj}",
                "Add a safety net around {obj}",
                "Create checks that lock in the fix for {obj}",
                "Expand coverage with scenarios for {obj}",
            ),
            "verify": (
                "Confirm the result by exercising {obj}",
                "Prove the existing {obj} behaves correctly",
                "Evaluate {obj} against the acceptance criteria",
                "Give {obj} a final validation pass",
            ),
            "document": (
                "Describe {obj} for future maintainers",
                "Record the usage contract of {obj}",
                "Clarify {obj} in the README",
                "Capture operational guidance for {obj}",
            ),
            "design": (
                "Sketch how {obj} should be built",
                "Propose the interfaces surrounding {obj}",
                "Work out an approach for {obj}",
                "Specify the architecture of {obj}",
            ),
            "other": (
                "Tell me the current status of {obj}",
                "Why does the project contain {obj}?",
                "A discussion about {obj}",
                "Is {obj} already available?",
            ),
        },
        "es": {
            "discover": (
                "Audita el comportamiento actual de {obj}",
                "Mapea el camino que recorren los datos por {obj}",
                "Revisa {obj} con atención antes de tocar el código",
                "Averigua la causa dentro de {obj}",
            ),
            "change": (
                "Haz que {obj} acepte el caso adicional",
                "Incorpora la capacidad faltante a {obj}",
                "Revisa {obj} para que el problema no se repita",
                "Lleva el comportamiento solicitado a {obj}",
            ),
            "test_change": (
                "Escribe casos que reproduzcan fallos en {obj}",
                "Añade una red de seguridad alrededor de {obj}",
                "Crea verificaciones que fijen la corrección de {obj}",
                "Extiende la cobertura con escenarios para {obj}",
            ),
            "verify": (
                "Confirma el resultado ejercitando {obj}",
                "Demuestra que {obj} se comporta correctamente",
                "Evalúa {obj} contra los criterios de aceptación",
                "Dale a {obj} una validación final",
            ),
            "document": (
                "Describe {obj} para quienes mantengan el proyecto",
                "Registra el contrato de uso de {obj}",
                "Aclara {obj} en el README",
                "Plasma una guía operativa para {obj}",
            ),
            "design": (
                "Bosqueja cómo debería construirse {obj}",
                "Propón las interfaces que rodean {obj}",
                "Elabora un enfoque para {obj}",
                "Especifica la arquitectura de {obj}",
            ),
            "other": (
                "Dime el estado actual de {obj}",
                "¿Por qué contiene el proyecto {obj}?",
                "Una conversación sobre {obj}",
                "¿Ya está disponible {obj}?",
            ),
        },
    },
}

ADVERSARIAL = {
    "en": (
        ("discover", "Without changing {obj}, examine how it works"),
        ("change", "Do not merely inspect {obj}; make it support the new case"),
        ("verify", "Do not implement anything in {obj}; only exercise the existing behavior"),
        ("change", "Skip the tests for now and bring the requested behavior to {obj}"),
        ("test_change", "Do not alter production code; write regression cases for {obj}"),
        ("document", "Without redesigning {obj}, describe its current contract"),
        ("design", "Do not build {obj} yet; work out its architecture"),
    ),
    "es": (
        ("discover", "Sin cambiar {obj}, examina cómo funciona"),
        ("change", "No te limites a inspeccionar {obj}; haz que acepte el caso nuevo"),
        ("verify", "No implementes nada en {obj}; sólo ejercita el comportamiento existente"),
        ("change", "Omite las pruebas por ahora e incorpora el comportamiento a {obj}"),
        ("test_change", "No alteres producción; escribe casos de regresión para {obj}"),
        ("document", "Sin rediseñar {obj}, describe su contrato actual"),
        ("design", "Todavía no construyas {obj}; elabora su arquitectura"),
    ),
}


@dataclass(frozen=True)
class Example:
    """One generated example with explicit split provenance."""

    split: str
    language: str
    label: str
    text: str
    family: str


@dataclass(frozen=True)
class Thresholds:
    """Confidence gates chosen without looking at the test split."""

    confidence: float
    margin: float


class _TeacherEmbedder:
    """Small adapter for measuring the static model against its teacher."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "--teacher-model requires sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.space_id = f"teacher:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed_queries(self, texts: list[str]) -> list[np.ndarray]:
        return list(self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ))


def build_examples() -> list[Example]:
    """Build disjoint template/object families plus adversarial test cases."""
    examples: list[Example] = []
    for split, language_objects in OBJECTS.items():
        for language, objects in language_objects.items():
            for label, templates in TEMPLATES[split][language].items():
                for template_index, template in enumerate(templates):
                    for obj in objects:
                        examples.append(Example(
                            split=split,
                            language=language,
                            label=label,
                            text=template.format(obj=obj),
                            family=f"{split}:{language}:{label}:{template_index}",
                        ))
    for language, rows in ADVERSARIAL.items():
        for template_index, (label, template) in enumerate(rows):
            for obj in OBJECTS["test"][language]:
                examples.append(Example(
                    split="test",
                    language=language,
                    label=label,
                    text=template.format(obj=obj),
                    family=f"adversarial:{language}:{template_index}",
                ))
    return examples


def _fit_ridge(vectors: np.ndarray, labels: np.ndarray, penalty: float) -> np.ndarray:
    """Fit a deterministic balanced linear classifier in embedding space."""
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    weights = len(labels) / (len(LABELS) * counts[labels])
    design = np.column_stack((vectors, np.ones(len(vectors))))
    targets = np.eye(len(LABELS), dtype=np.float64)[labels]
    gram = design.T @ (design * weights[:, None])
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(
        gram + regularizer,
        design.T @ (targets * weights[:, None]),
    )


def _scores(vectors: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack((vectors, np.ones(len(vectors))))
    logits = design @ coefficients
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _softmax(scores: np.ndarray, *, temperature: float = 0.05) -> np.ndarray:
    scaled = scores / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _prototype_scores(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    query_vectors: np.ndarray,
    *,
    neighbors: int | None,
) -> np.ndarray:
    """Score labels by cosine centroids or their nearest frozen prototypes."""
    similarities = query_vectors @ train_vectors.T
    result = np.empty((len(query_vectors), len(LABELS)), dtype=np.float64)
    for label_id in range(len(LABELS)):
        label_scores = similarities[:, train_labels == label_id]
        if neighbors is None:
            centroid = train_vectors[train_labels == label_id].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            result[:, label_id] = query_vectors @ centroid
            continue
        selected = np.partition(
            label_scores, -min(neighbors, label_scores.shape[1]), axis=1
        )[:, -neighbors:]
        result[:, label_id] = selected.mean(axis=1)
    return result


def _representation_text(text: str, representation: str) -> str:
    """Reduce topic noise without interpreting words or adding language rules."""
    if representation == "full":
        return text
    prefix_words = int(representation.removeprefix("prefix_"))
    return " ".join(text.split()[:prefix_words])


def _prediction_stats(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(probabilities, axis=1)
    predicted = order[:, -1]
    confidence = probabilities[np.arange(len(probabilities)), predicted]
    margin = confidence - probabilities[np.arange(len(probabilities)), order[:, -2]]
    return predicted, confidence, margin


def _choose_thresholds(
    probabilities: np.ndarray,
    expected: np.ndarray,
    *,
    minimum_precision: float,
) -> Thresholds:
    """Maximise semantic coverage subject to a calibration precision floor."""
    predicted, confidence, margin = _prediction_stats(probabilities)
    confidence_grid = np.unique(np.quantile(confidence, np.linspace(0, 0.95, 40)))
    margin_grid = np.unique(np.quantile(margin, np.linspace(0, 0.95, 40)))
    best = Thresholds(confidence=1.0, margin=1.0)
    best_coverage = -1.0
    for confidence_threshold in confidence_grid:
        for margin_threshold in margin_grid:
            accepted = (confidence >= confidence_threshold) & (margin >= margin_threshold)
            if not accepted.any():
                continue
            precision = float(np.mean(predicted[accepted] == expected[accepted]))
            coverage = float(np.mean(accepted))
            if precision >= minimum_precision and coverage > best_coverage:
                best_coverage = coverage
                best = Thresholds(
                    confidence=float(confidence_threshold),
                    margin=float(margin_threshold),
                )
    return best


def _metrics(expected: np.ndarray, predicted: np.ndarray, accepted: np.ndarray) -> dict[str, float]:
    correct = predicted == expected
    return {
        "accuracy": float(np.mean(correct)),
        "coverage": float(np.mean(accepted)),
        "accepted_precision": float(np.mean(correct[accepted])) if accepted.any() else 1.0,
        "accepted_accuracy": float(np.mean(correct & accepted)),
    }


def _bootstrap_accuracy_delta(
    expected: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    delta = (candidate == expected).astype(np.float64) - (baseline == expected)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = delta[draws].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(delta.mean()), float(low), float(high)


def evaluate(
    embedder: StaticQwen3Embedder | _TeacherEmbedder | CachedTeacherEmbedder,
    *,
    minimum_precision: float,
    bootstrap_samples: int,
) -> dict[str, object]:
    """Train/calibrate on their splits and return untouched test metrics."""
    examples = build_examples()
    representations = ("full", "prefix_4", "prefix_6", "prefix_8", "prefix_12")
    started = perf_counter()
    vector_sets = {
        representation: np.asarray(embedder.embed_queries([
            _representation_text(example.text, representation) for example in examples
        ]))
        for representation in representations
    }
    elapsed = perf_counter() - started
    label_ids = np.asarray([LABELS.index(example.label) for example in examples])
    split_indices = {
        split: np.asarray([index for index, row in enumerate(examples) if row.split == split])
        for split in ("train", "calibration", "test")
    }

    best_representation = ""
    best_penalty = 0.0
    best_accuracy = -1.0
    best_model: tuple[str, object] | None = None
    calibration_candidates: list[dict[str, object]] = []
    for representation, vectors in vector_sets.items():
        for penalty in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            coefficients = _fit_ridge(
                vectors[split_indices["train"]], label_ids[split_indices["train"]], penalty
            )
            probabilities = _scores(vectors[split_indices["calibration"]], coefficients)
            predicted, _, _ = _prediction_stats(probabilities)
            accuracy = float(np.mean(predicted == label_ids[split_indices["calibration"]]))
            calibration_candidates.append({
                "representation": representation,
                "method": "ridge",
                "ridge_penalty": penalty,
                "accuracy": accuracy,
            })
            if accuracy > best_accuracy:
                best_representation = representation
                best_penalty = penalty
                best_accuracy = accuracy
                best_model = ("ridge", coefficients)
        for neighbors in (None, 1, 3, 5, 8, 16):
            raw_scores = _prototype_scores(
                vectors[split_indices["train"]],
                label_ids[split_indices["train"]],
                vectors[split_indices["calibration"]],
                neighbors=neighbors,
            )
            probabilities = _softmax(raw_scores)
            predicted, _, _ = _prediction_stats(probabilities)
            accuracy = float(np.mean(predicted == label_ids[split_indices["calibration"]]))
            method = "centroid" if neighbors is None else f"prototype_{neighbors}"
            calibration_candidates.append({
                "representation": representation,
                "method": method,
                "accuracy": accuracy,
            })
            if accuracy > best_accuracy:
                best_representation = representation
                best_penalty = 0.0
                best_accuracy = accuracy
                best_model = (method, neighbors)
    assert best_model is not None

    vectors = vector_sets[best_representation]
    method, payload = best_model
    if method == "ridge":
        calibration_probabilities = _scores(
            vectors[split_indices["calibration"]], payload
        )
    else:
        calibration_probabilities = _softmax(_prototype_scores(
            vectors[split_indices["train"]],
            label_ids[split_indices["train"]],
            vectors[split_indices["calibration"]],
            neighbors=payload,
        ))
    thresholds = _choose_thresholds(
        calibration_probabilities,
        label_ids[split_indices["calibration"]],
        minimum_precision=minimum_precision,
    )

    test_index = split_indices["test"]
    test_rows = [examples[index] for index in test_index]
    expected = label_ids[test_index]
    if method == "ridge":
        probabilities = _scores(vectors[test_index], payload)
    else:
        probabilities = _softmax(_prototype_scores(
            vectors[split_indices["train"]],
            label_ids[split_indices["train"]],
            vectors[test_index],
            neighbors=payload,
        ))
    semantic, confidence, margin = _prediction_stats(probabilities)
    semantic_accepted = (
        (confidence >= thresholds.confidence)
        & (margin >= thresholds.margin)
        & (semantic != LABELS.index("other"))
    )
    regex = np.asarray([
        LABELS.index(phase) if (phase := _step_phase(row.text)) in LABELS
        else LABELS.index("other")
        for row in test_rows
    ])
    regex_matched = np.asarray([bool(_step_phase(row.text)) for row in test_rows])
    hybrid = regex.copy()
    semantic_fill = (~regex_matched) & semantic_accepted
    hybrid[semantic_fill] = semantic[semantic_fill]

    groups: dict[str, dict[str, object]] = {}
    masks = {
        "all": np.ones(len(test_rows), dtype=bool),
        "english": np.asarray([row.language == "en" for row in test_rows]),
        "spanish": np.asarray([row.language == "es" for row in test_rows]),
        "adversarial": np.asarray([row.family.startswith("adversarial:") for row in test_rows]),
    }
    for name, mask in masks.items():
        groups[name] = {
            "count": int(mask.sum()),
            "regex": _metrics(expected[mask], regex[mask], regex_matched[mask]),
            "semantic": _metrics(
                expected[mask], semantic[mask], semantic_accepted[mask]
            ),
            "hybrid": _metrics(
                expected[mask], hybrid[mask], regex_matched[mask] | semantic_fill[mask]
            ),
            "hybrid_minus_regex_accuracy_ci95": _bootstrap_accuracy_delta(
                expected[mask], regex[mask], hybrid[mask],
                samples=bootstrap_samples, seed=41,
            ),
        }

    errors = []
    for row, expected_id, semantic_id, hybrid_id, accepted in zip(
        test_rows, expected, semantic, hybrid, semantic_fill, strict=True
    ):
        if accepted and hybrid_id != expected_id:
            errors.append({
                "text": row.text,
                "language": row.language,
                "family": row.family,
                "expected": LABELS[expected_id],
                "semantic": LABELS[semantic_id],
            })

    confusion = {
        label: dict(Counter(
            LABELS[predicted]
            for predicted, expected_id in zip(semantic, expected, strict=True)
            if expected_id == label_id
        ))
        for label_id, label in enumerate(LABELS)
    }

    return {
        "model": embedder.model_name,
        "space_id": embedder.space_id,
        "dataset": {
            split: int(len(indices)) for split, indices in split_indices.items()
        },
        "template_families": dict(Counter(example.family for example in examples)),
        "fit": {
            "representation": best_representation,
            "method": method,
            "ridge_penalty": best_penalty if method == "ridge" else None,
            "calibration_accuracy": best_accuracy,
            "calibration_candidates": calibration_candidates,
            "minimum_precision": minimum_precision,
            "thresholds": asdict(thresholds),
        },
        "throughput": {
            "representations": len(representations),
            "texts_per_second": len(examples) * len(representations) / elapsed,
            "microseconds_per_text": elapsed * 1e6 / (len(examples) * len(representations)),
        },
        "groups": groups,
        "semantic_confusion": confusion,
        "hybrid_fill_errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact")
    parser.add_argument("--spanish-adapter")
    parser.add_argument("--teacher-model")
    parser.add_argument("--openai-cache", type=Path)
    parser.add_argument(
        "--export-jsonl",
        type=Path,
        help="write the generated examples as a collector-compatible corpus",
    )
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--disable-spanish-adapter",
        action="store_true",
        help="evaluate the unadapted v2 table even when the bundled adapter exists",
    )
    parser.add_argument("--minimum-precision", type=float, default=0.99)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.teacher_model and args.openai_cache:
        raise SystemExit("choose either --teacher-model or --openai-cache")
    if args.export_jsonl:
        rows = []
        for index, example in enumerate(build_examples()):
            row = {"id": f"phase:{index}", **asdict(example)}
            row.update({
                representation: _representation_text(example.text, representation)
                for representation in ("prefix_4", "prefix_6", "prefix_8", "prefix_12")
            })
            rows.append(json.dumps(row, ensure_ascii=False))
        args.export_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.export_jsonl.write_text("\n".join(rows) + "\n", encoding="utf-8")
        if args.export_only:
            return

    if args.openai_cache:
        embedder: StaticQwen3Embedder | _TeacherEmbedder | CachedTeacherEmbedder = (
            CachedTeacherEmbedder(args.openai_cache)
        )
    elif args.teacher_model:
        embedder = _TeacherEmbedder(
            args.teacher_model
        )
    else:
        embedder = StaticQwen3Embedder(
            args.artifact,
            spanish_adapter_path=args.spanish_adapter,
        )
        if args.disable_spanish_adapter:
            embedder._spanish_adapter_path = None
    report = evaluate(
        embedder,
        minimum_precision=args.minimum_precision,
        bootstrap_samples=args.bootstrap_samples,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
