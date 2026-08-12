"""Fit and evaluate an ultra-small task-policy head over static Qwen3 vectors.

The encoder remains frozen. Training uses NumPy ridge regression, threshold
selection uses a separate validation split, and final metrics are reported on
unseen phrasing and object families. The optional artifact contains only a
small weight matrix, thresholds, labels, dataset hashes, and the exact static
embedding space identity.

Run:

    uv run python -m bench.task_policy_linear_head --artifact /tmp/task-head.npz
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.task_policy_large_calibration import build_large_calibration_corpus
from bench.task_policy_semantic_eval import (
    SemanticExample,
    build_semantic_validation_corpus,
)
from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


LABELS = (
    "bugfix.root_cause",
    "feature.contract_first",
    "refactor.preserve_behavior",
    "research.evidence_first",
    "review.read_only",
    "performance.measure_first",
    "uncategorized",
)
MODEL_VERSION = "static-qwen3-task-policy-flat-hard-negative-candidate-v3"

_CALIBRATION_OBJECTS = (
    "the notification scheduler", "the dependency resolver",
    "the image renderer", "the billing webhook", "the editor bridge",
    "the release queue", "el planificador de avisos",
    "el resolvedor de dependencias", "el renderizador de imágenes",
    "el webhook de cobros", "el puente del editor", "la cola de releases",
)

_CALIBRATION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "bugfix.root_cause": (
        "Make {object}, which now violates its existing contract, behave correctly again.",
        "I need {object} to stop returning stale results and restore its previous behavior.",
        "Haz que {object}, que ahora incumple su contrato, vuelva a comportarse correctamente.",
        "Quiero que {object} deje de devolver resultados equivocados y recupere su conducta anterior.",
        "{object} dejó de respetar las garantías existentes; haz que vuelvan a cumplirse.",
        "Desde la última versión {object} rompe casos que antes funcionaban; restáuralos.",
        "{object} deixou de cumprir garantias existentes; restaure o comportamento anterior.",
        "{object} ne tient plus ses garanties; rétablis le comportement attendu.",
        "Une régression de {object} casse un appel existant; restaure le contrat antérieur.",
    ),
    "feature.contract_first": (
        "I need {object} to support a user workflow that is not possible today.",
        "Make a previously unavailable user case possible through {object}.",
        "Quiero que {object} admita un flujo de usuario que hoy no es posible.",
        "Haz posible mediante {object} un caso de usuario que antes no estaba disponible.",
        "Adicione a {object} um fluxo que os usuários ainda não conseguem realizar.",
        "Permets à {object} de prendre en charge un usage qui n'existe pas encore.",
        "Erweitere {object} um einen bisher nicht unterstützten Arbeitsablauf.",
        "Aggiungi a {object} una possibilità che oggi gli utenti non hanno.",
        "Étends {object} pour rendre possible un parcours utilisateur encore indisponible.",
        "Amplía {object} para permitir un flujo nuevo que antes no existía.",
    ),
    "refactor.preserve_behavior": (
        "Make {object} easier to understand internally while every output stays identical.",
        "Reduce structural complexity inside {object}; preserve all observable contracts.",
        "Quiero que {object} sea más comprensible por dentro sin alterar ninguna salida.",
        "Reduce la complejidad estructural de {object}; conserva todos sus contratos observables.",
        "Simplifique a estrutura interna de {object} sem mudar o comportamento observável.",
        "Rends l'intérieur de {object} plus simple sans modifier ses sorties.",
        "Vereinfache den internen Aufbau von {object}, ohne sichtbares Verhalten zu ändern.",
        "Semplifica l'interno di {object} mantenendo identico ogni risultato osservabile.",
    ),
    "research.evidence_first": (
        "Tell me which direction around {object} is best and support it with reliable material.",
        "We need a defensible choice for {object}; gather facts and separate assumptions.",
        "Dime qué dirección conviene para {object} y respáldala con material fiable.",
        "Necesitamos una elección defendible para {object}; reúne hechos y separa supuestos.",
        "Compare alternativas para {object} com fontes confiáveis antes de recomendar.",
        "Étudie les options pour {object} avec des sources fiables avant de conseiller.",
        "Vergleiche Ansätze für {object} anhand belastbarer Quellen und gib eine Empfehlung.",
        "Confronta le opzioni per {object} usando prove affidabili prima di consigliare.",
        "Compara direcciones para {object} y apoya el consejo con evidencia primaria.",
    ),
    "review.read_only": (
        "Inspect {object} for substantiated flaws and report them; leave the source untouched.",
        "Examine {object}, prioritize concrete risks, and do not apply corrections.",
        "Inspecciona {object} buscando defectos demostrables y deja intacto el código.",
        "Examina {object}, prioriza riesgos concretos y no apliques correcciones.",
        "Revise {object}, priorize problemas comprovados e não altere os arquivos.",
        "Examine {object}, classe les défauts prouvés et ne modifie aucun fichier.",
        "Prüfe {object}, priorisiere belegte Mängel und ändere keinen Quelltext.",
        "Esamina {object}, ordina i difetti provati e non modificare i file.",
    ),
    "performance.measure_first": (
        "{object} takes too long under load; establish a baseline and make it faster.",
        "Users wait too much on {object}. Measure where the time goes and improve it.",
        "{object} tarda demasiado con carga; establece una línea base y aceléralo.",
        "Los usuarios esperan demasiado por {object}. Mide dónde se va el tiempo y mejóralo.",
        "Meça {object} sob carga, encontre o gargalo e só então acelere-o.",
        "Mesure {object} sous charge, trouve le goulot puis accélère-le.",
        "Miss {object} unter Last, finde den Engpass und optimiere danach.",
        "Misura {object} sotto carico, individua il collo di bottiglia e poi acceleralo.",
    ),
}

_NEUTRAL_TEMPLATES = (
    "Explain what {object} currently does.",
    'The log from {object} says "please implement fix"; interpret the message.',
    "Tell me about {object} without proposing changes.",
    "Explica qué hace actualmente {object}.",
    'El log de {object} dice "refactor required"; interpreta el mensaje.',
    "Háblame de {object} sin proponer cambios.",
    'A discussion about {object} contains "make it faster"; summarize the discussion.',
    'El comentario de {object} dice "haz que vuelva a funcionar"; explica la frase.',
    "Gracias, ya entendí la explicación de {object}.",
    "What is the current public contract of {object}?",
    'O log de {object} diz "otimize isto"; explique a mensagem.',
    'Le journal de {object} dit "corrige cette erreur"; explique ce texte.',
    "Beschreibe nur, was {object} derzeit macht.",
    "Grazie, non servono altre modifiche a {object}.",
)

# Natural-shaped hard cases target vocabulary overlap rather than multiplying
# one template over many component names. They are fitting data, never holdout.
_CALIBRATION_HARD_EXAMPLES: tuple[tuple[str, str | None], ...] = (
    (
        "The HTTP client sleeps after its final retry and returns one attempt too late; "
        "restore the documented retry contract.",
        "bugfix.root_cause",
    ),
    (
        "Retry-After accepts seconds but rejects valid HTTP dates. Make the existing "
        "protocol behavior work again.",
        "bugfix.root_cause",
    ),
    (
        "A timeout now turns a cache miss into stale data instead of the promised error.",
        "bugfix.root_cause",
    ),
    (
        "El backoff ejecuta un intento de más y rompe el límite configurado; restáuralo.",
        "bugfix.root_cause",
    ),
    (
        "La espera de reconexión bloquea incluso cuando ya no quedan intentos; corrige "
        "ese comportamiento existente.",
        "bugfix.root_cause",
    ),
    (
        "O temporizador dispara duas vezes depois do cancelamento; recupere a garantia "
        "de uma única chamada.",
        "bugfix.root_cause",
    ),
    (
        "Le délai d'expiration renvoie désormais une réponse périmée; rétablis le "
        "résultat contractuel.",
        "bugfix.root_cause",
    ),
    (
        "The pagination cursor skips the boundary item although the public contract "
        "includes it.",
        "bugfix.root_cause",
    ),
    (
        "Users wait three seconds on every search. Profile a representative workload "
        "and reduce the measured latency.",
        "performance.measure_first",
    ),
    (
        "The worker throughput collapsed under load; benchmark the queue and remove the "
        "measured bottleneck.",
        "performance.measure_first",
    ),
    (
        "Mide por qué el render tarda tanto con escenas grandes y acelera el camino crítico.",
        "performance.measure_first",
    ),
    (
        "A API continua correta, mas usa CPU demais; faça profiling e reduza o custo medido.",
        "performance.measure_first",
    ),
    (
        "La sortie est correcte mais trop lente sous charge; mesure puis optimise le goulot.",
        "performance.measure_first",
    ),
    (
        "Add support for Retry-After HTTP dates, which this client has never accepted.",
        "feature.contract_first",
    ),
    (
        "Permite configurar una estrategia de backoff nueva que hoy no existe.",
        "feature.contract_first",
    ),
    (
        "Expose a new cursor mode for callers while preserving the existing default.",
        "feature.contract_first",
    ),
    (
        "Split retry scheduling from transport without changing attempts, delays, or errors.",
        "refactor.preserve_behavior",
    ),
    (
        "Separa el parser del cache manteniendo idénticos resultados y tiempos observables.",
        "refactor.preserve_behavior",
    ),
    (
        "Compare retry algorithms from primary sources and recommend one; do not edit code.",
        "research.evidence_first",
    ),
    (
        "Revisa el parche del temporizador, prioriza defectos verificables y no lo modifiques.",
        "review.read_only",
    ),
    ('The incident report says "reduce latency now"; summarize what the report requests.', None),
    ('El log dice "reintenta y corrige el timeout"; explica el mensaje, no actúes.', None),
    ("Thanks, the retry explanation is enough; no code changes are needed.", None),
    ("What does exponential backoff mean in a distributed client?", None),
    ("¿Cuál es la diferencia conceptual entre timeout y deadline?", None),
)

_AMBIGUITY_CHALLENGE: tuple[tuple[str, str | None], ...] = (
    ("The lease renewer waits after its last allowed attempt; make it stop at the limit.", "bugfix.root_cause"),
    ("A slow response is being treated as success and corrupts the stored status; restore the prior result.", "bugfix.root_cause"),
    ("El debounce pierde la última edición cuando vence el timer; recupera la garantía existente.", "bugfix.root_cause"),
    ("Le cache répond après l'échéance avec une valeur invalide; rétablis le contrat.", "bugfix.root_cause"),
    ("Benchmark completion on a million symbols and reduce its p95 latency.", "performance.measure_first"),
    ("La sincronización produce resultados correctos pero consume demasiada CPU; perfílala y acelérala.", "performance.measure_first"),
    ("Meça o throughput do indexador sob carga e remova o gargalo comprovado.", "performance.measure_first"),
    ("Profile le rendu de gros documents puis réduis son temps sans changer la sortie.", "performance.measure_first"),
    ("Introduce jittered backoff; only fixed delays are available today.", "feature.contract_first"),
    ("Añade un modo de paginación inversa que los consumidores todavía no pueden usar.", "feature.contract_first"),
    ("Permita que o cliente aceite um novo formato de credencial.", "feature.contract_first"),
    ("Ajoute une option de compression encore absente du protocole.", "feature.contract_first"),
    ("Untangle the scheduler state machine while every retry remains byte-for-byte equivalent.", "refactor.preserve_behavior"),
    ("Reordena el cliente en capas sin alterar requests, errores ni esperas.", "refactor.preserve_behavior"),
    ("Simplifique o fluxo interno mantendo idêntico todo comportamento observável.", "refactor.preserve_behavior"),
    ("Sépare le transport du cache sans aucune différence visible.", "refactor.preserve_behavior"),
    ("Compare durable queue designs using primary evidence and recommend a direction.", "research.evidence_first"),
    ("Investiga opciones de serialización y distingue hechos de supuestos antes de recomendar.", "research.evidence_first"),
    ("Audit the retry patch for concrete defects and leave the branch untouched.", "review.read_only"),
    ("Inspecciona el diff del cache y entrega solo hallazgos priorizados; no edites nada.", "review.read_only"),
    ('The ticket title is "make retries faster"; translate the title only.', None),
    ('Un comentario dice "agrega backoff"; ¿qué quiso decir el autor?', None),
    ("Perfecto, ya no necesito que cambies el temporizador.", None),
    ("Explain how retry budgets are normally defined.", None),
)

_DEVELOPMENT_OBJECTS = (
    "the telemetry sampler", "the archive reader", "the plugin handshake",
    "the terminal compositor", "el muestreador de telemetría",
    "el lector de archivos", "el saludo de plugins", "el compositor de terminal",
)

_DEVELOPMENT_TEMPLATES: dict[str | None, tuple[str, ...]] = {
    "bugfix.root_cause": (
        "A regression in {object} breaks an established caller; restore the promised result.",
        "Una regresión en {object} rompe un consumidor existente; recupera el resultado prometido.",
        "Uma regressão em {object} quebrou um fluxo existente; recupere o resultado contratado.",
        "Une régression de {object} casse un appel existant; restaure le résultat garanti.",
    ),
    "feature.contract_first": (
        "Extend {object} so users can complete a workflow that has never been supported.",
        "Extiende {object} para admitir un flujo de usuario que nunca estuvo soportado.",
        "Estenda {object} para permitir um fluxo que nunca foi suportado.",
        "Étends {object} afin de permettre un parcours utilisateur encore indisponible.",
    ),
    "refactor.preserve_behavior": (
        "Reshape the internals of {object}; callers must observe exactly the same behavior.",
        "Reorganiza el interior de {object}; los consumidores deben observar lo mismo.",
        "Reorganize o interior de {object} sem mudar nada que os chamadores observam.",
        "Réorganise l'intérieur de {object} sans aucun changement observable.",
    ),
    "research.evidence_first": (
        "Compare viable directions for {object}, grounding the recommendation in primary evidence.",
        "Compara opciones viables para {object} y fundamenta la recomendación con evidencia primaria.",
        "Compare caminhos para {object} e fundamente a recomendação em evidências primárias.",
        "Compare les options pour {object} et fonde le conseil sur des sources primaires.",
    ),
    "review.read_only": (
        "Audit {object}, rank evidence-backed findings, and make no source changes.",
        "Audita {object}, prioriza hallazgos respaldados y no cambies el código.",
        "Audite {object}, priorize achados comprovados e não altere o código.",
        "Audite {object}, classe les constats prouvés et ne touche pas au code.",
    ),
    "performance.measure_first": (
        "Profile {object} under realistic load, identify the measured bottleneck, then reduce it.",
        "Perfila {object} con carga realista, identifica el cuello medido y luego redúcelo.",
        "Meça {object} sob carga realista, identifique o gargalo e depois reduza-o.",
        "Profile {object} sous charge réaliste, mesure le goulot puis réduis-le.",
    ),
    None: (
        '{object} printed "ship a new feature"; explain what the quoted text means.',
        '{object} mostró "audita y corrige"; interpreta el texto citado.',
        "Thanks, the explanation of {object} is enough for now.",
        "Describe the current responsibilities of {object} without recommending changes.",
    ),
}

_HOLDOUT_OBJECTS = (
    "the lease coordinator", "the syntax cache", "the upload journal",
    "the query planner", "el coordinador de leases", "la caché de sintaxis",
    "el diario de subidas", "el planificador de consultas",
)

_HOLDOUT_TEMPLATES: dict[str | None, tuple[str, ...]] = {
    "bugfix.root_cause": (
        "Since the last release {object} no longer honors an established guarantee; restore it.",
        "Antes funcionaba, pero ahora {object} incumple su contrato con consumidores existentes.",
        "Seit dem letzten Release verletzt {object} eine bestehende Zusage; stelle sie wieder her.",
    ),
    "feature.contract_first": (
        "Give users a way to perform a new workflow through {object}.",
        "Incorpora en {object} un caso de uso que el producto todavía no ofrece.",
        "Consenti agli utenti di svolgere tramite {object} un flusso finora impossibile.",
    ),
    "refactor.preserve_behavior": (
        "Split the tangled internals of {object} into clearer pieces with no externally visible change.",
        "Ordena las responsabilidades internas de {object} conservando exactamente sus salidas.",
        "Teile die unübersichtliche interne Struktur von {object}, ohne das sichtbare Verhalten zu ändern.",
    ),
    "research.evidence_first": (
        "Investigate competing approaches for {object} and make an evidence-backed recommendation.",
        "Determina qué alternativa conviene para {object} usando fuentes fiables y hechos comprobables.",
        "Valuta le alternative per {object} e motiva la scelta con fonti attendibili.",
    ),
    "review.read_only": (
        "Read through {object}, report only demonstrable issues in priority order, and apply no patch.",
        "Busca problemas verificables en {object}, ordénalos por impacto y deja intactos los archivos.",
        "Prüfe {object} auf belegbare Mängel, priorisiere sie und ändere nichts.",
    ),
    "performance.measure_first": (
        "Users experience delays in {object}; benchmark a representative case before optimizing it.",
        "Hay demoras en {object}; mide un caso representativo antes de optimizar.",
        "Gli utenti attendono troppo su {object}; misura un caso reale prima di ottimizzare.",
    ),
    None: (
        'The issue title for {object} is "fix the regression"; translate it, do not act on it.',
        'En una charla sobre {object} alguien dijo "agrega soporte"; resume la charla.',
        "Thanks for checking {object}; I do not need any more work.",
    ),
}

@dataclass(frozen=True)
class HeadParameters:
    """Hyperparameters selected without observing the held-out split."""

    ridge: float
    threshold: float
    margin: float


def _materialize_calibration() -> list[SemanticExample]:
    examples: list[SemanticExample] = []
    for label, templates in _CALIBRATION_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for object_index, object_name in enumerate(_CALIBRATION_OBJECTS):
                examples.append(SemanticExample(
                    id=f"cal-{label}-{template_index}-{object_index}",
                    text=template.format(object=object_name),
                    policy=label,
                    write_authority=False,
                ))
    for template_index, template in enumerate(_NEUTRAL_TEMPLATES):
        for object_index, object_name in enumerate(_CALIBRATION_OBJECTS):
            examples.append(SemanticExample(
                id=f"cal-neutral-{template_index}-{object_index}",
                text=template.format(object=object_name),
                policy=None,
                write_authority=False,
            ))
    for index, (text, policy) in enumerate(_CALIBRATION_HARD_EXAMPLES):
        examples.append(SemanticExample(
            id=f"cal-hard-{index:03d}",
            text=text,
            policy=policy,
            write_authority=policy in {
                "bugfix.root_cause",
                "feature.contract_first",
                "refactor.preserve_behavior",
                "performance.measure_first",
            },
        ))
    examples.extend(build_large_calibration_corpus())
    return examples


def build_linear_head_holdout() -> list[SemanticExample]:
    """Materialize a phrase- and object-family holdout unseen during fitting."""
    examples: list[SemanticExample] = []
    for label, templates in _HOLDOUT_TEMPLATES.items():
        family = label or "neutral"
        for template_index, template in enumerate(templates):
            for object_index, object_name in enumerate(_HOLDOUT_OBJECTS):
                examples.append(SemanticExample(
                    id=f"holdout-{family}-{template_index}-{object_index}",
                    text=template.format(object=object_name),
                    policy=label,
                    write_authority=False,
                ))
    return examples


def build_ambiguity_challenge() -> list[SemanticExample]:
    """Return natural-shaped overlap cases excluded from fitting and tuning."""
    return [
        SemanticExample(
            id=f"ambiguity-{index:03d}",
            text=text,
            policy=policy,
            write_authority=False,
        )
        for index, (text, policy) in enumerate(_AMBIGUITY_CHALLENGE)
    ]


def _build_development_corpus() -> list[SemanticExample]:
    examples: list[SemanticExample] = []
    for label, templates in _DEVELOPMENT_TEMPLATES.items():
        family = label or "neutral"
        for template_index, template in enumerate(templates):
            for object_index, object_name in enumerate(_DEVELOPMENT_OBJECTS):
                examples.append(SemanticExample(
                    id=f"development-{family}-{template_index}-{object_index}",
                    text=template.format(object=object_name),
                    policy=label,
                    write_authority=False,
                ))
    return examples


def _dataset_hash(examples: list[SemanticExample]) -> str:
    payload = "\n".join(
        json.dumps(asdict(item), sort_keys=True, ensure_ascii=False)
        for item in examples
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _design_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    return np.column_stack((matrix, np.ones(len(matrix), dtype=np.float64)))


def _targets(examples: list[SemanticExample]) -> np.ndarray:
    result = np.zeros((len(examples), len(LABELS)), dtype=np.float64)
    label_index = {label: index for index, label in enumerate(LABELS)}
    for row, example in enumerate(examples):
        label = example.policy or "uncategorized"
        result[row, label_index[label]] = 1.0
    return result


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    dual = np.linalg.solve(x @ x.T + ridge * np.eye(len(x)), y)
    return x.T @ dual


def _predict(scores: np.ndarray, threshold: float, margin: float) -> list[str | None]:
    predictions: list[str | None] = []
    for row in scores:
        order = np.argsort(row)[::-1]
        if row[order[0]] < threshold or row[order[0]] - row[order[1]] < margin:
            predictions.append(None)
        else:
            label = LABELS[int(order[0])]
            predictions.append(None if label == "uncategorized" else label)
    return predictions


def _metrics(
    examples: list[SemanticExample], predictions: list[str | None]
) -> dict[str, object]:
    covered = sum(item is not None for item in predictions)
    correct = sum(
        prediction == example.policy
        for example, prediction in zip(examples, predictions, strict=True)
    )
    correct_covered = sum(
        prediction is not None and prediction == example.policy
        for example, prediction in zip(examples, predictions, strict=True)
    )
    false_activation = sum(
        prediction is not None and example.policy is None
        for example, prediction in zip(examples, predictions, strict=True)
    )
    errors = [
        {
            "id": example.id,
            "expected": example.policy,
            "predicted": prediction,
        }
        for example, prediction in zip(examples, predictions, strict=True)
        if prediction is not None and prediction != example.policy
    ]
    return {
        "examples": len(examples),
        "coverage": covered / len(examples),
        "exact_match": correct / len(examples),
        "selective_precision": correct_covered / covered if covered else 1.0,
        "false_activations": false_activation,
        "classification_errors": errors,
    }


def _select_parameters(
    x: np.ndarray,
    y: np.ndarray,
    validation_x: np.ndarray,
    validation: list[SemanticExample],
) -> tuple[HeadParameters, np.ndarray, dict[str, object]]:
    candidates: list[
        tuple[float, float, float, HeadParameters, np.ndarray, dict[str, object]]
    ] = []
    for ridge in (0.001, 0.01, 0.1, 1.0, 10.0):
        weights = _fit_ridge(x, y, ridge)
        scores = validation_x @ weights
        for threshold in np.arange(0.20, 0.651, 0.05):
            for margin in (0.0, 0.03, 0.06, 0.10, 0.15):
                metrics = _metrics(
                    validation, _predict(scores, float(threshold), margin)
                )
                safe = (
                    metrics["selective_precision"] >= 0.99
                    and metrics["false_activations"] == 0
                )
                candidates.append((
                    float(safe), float(metrics["selective_precision"]),
                    float(metrics["coverage"]),
                    HeadParameters(ridge, float(threshold), margin),
                    weights, metrics,
                ))
    _, _, _, parameters, weights, metrics = max(
        candidates, key=lambda item: item[:3]
    )
    return parameters, weights, metrics


def run_experiment(artifact: Path | None = None) -> dict[str, object]:
    """Fit on calibration, tune on validation, and evaluate once on holdout."""
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        raise RuntimeError("bundled static Qwen3 artifact is unavailable")
    calibration = _materialize_calibration()
    validation = build_semantic_validation_corpus()
    development = _build_development_corpus()
    challenge = build_ambiguity_challenge()
    holdout = build_linear_head_holdout()
    all_texts = [
        item.text
        for item in calibration + validation + development + challenge + holdout
    ]
    vectors = embedder.embed_queries(all_texts)
    cal_end = len(calibration)
    val_end = cal_end + len(validation)
    dev_end = val_end + len(development)
    challenge_end = dev_end + len(challenge)
    x = _design_matrix(vectors[:cal_end])
    validation_x = _design_matrix(vectors[cal_end:val_end])
    development_x = _design_matrix(vectors[val_end:dev_end])
    challenge_x = _design_matrix(vectors[dev_end:challenge_end])
    holdout_x = _design_matrix(vectors[challenge_end:])
    parameters, weights, validation_metrics = _select_parameters(
        x, _targets(calibration), validation_x, validation
    )
    holdout_metrics = _metrics(
        holdout,
        _predict(
            holdout_x @ weights, parameters.threshold, parameters.margin
        ),
    )
    development_metrics = _metrics(
        development,
        _predict(
            development_x @ weights, parameters.threshold, parameters.margin
        ),
    )
    challenge_metrics = _metrics(
        challenge,
        _predict(
            challenge_x @ weights, parameters.threshold, parameters.margin
        ),
    )
    metadata = {
        "schema_version": 1,
        "model": MODEL_VERSION,
        "embedding_space_id": embedder.space_id,
        "labels": list(LABELS),
        "parameters": asdict(parameters),
        "calibration_sha256": _dataset_hash(calibration),
        "validation_sha256": _dataset_hash(validation),
        "development_sha256": _dataset_hash(development),
        "challenge_sha256": _dataset_hash(challenge),
        "holdout_sha256": _dataset_hash(holdout),
    }
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            weights=weights.astype(np.float32),
            metadata=np.frombuffer(
                json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
            ),
        )
    return {
        **metadata,
        "calibration_examples": len(calibration),
        "validation": validation_metrics,
        "development": development_metrics,
        "challenge": challenge_metrics,
        "holdout": holdout_metrics,
        "artifact_bytes": artifact.stat().st_size if artifact is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
