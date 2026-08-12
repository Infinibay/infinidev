"""Large causal-family calibration augmentation for the task-method mini-head.

The rows are synthetic fitting data. Templates describe distinct failure or
work families and are materialized over fictional project components; they are
never reported as natural traffic or used as a final holdout.
"""

from __future__ import annotations

from collections import Counter

from bench.task_policy_semantic_eval import SemanticExample


AUGMENTATION_VERSION = "task-policy-large-calibration-v1"

_COMPONENTS = (
    "AtlasParser",
    "NimbusQueue",
    "KestrelClient",
    "OrchidCache",
    "QuartzWorker",
    "HarborGateway",
    "CobaltIndexer",
    "LumenTransport",
)

_METHOD_TEMPLATES: dict[str, tuple[str, ...]] = {
    "bugfix.root_cause": (
        "{object} accepted this valid input before the last release and now rejects it; restore the contract.",
        "A cache invalidation in {object} returns stale state instead of the documented result.",
        "{object} performs one retry beyond the configured limit; make the established bound hold again.",
        "Timezone conversion in {object} shifts an existing timestamp by one day; correct the regression.",
        "{object} crashes on an empty but valid payload; recover the previous behavior.",
        "Duplicate delivery in {object} violates the exactly-once acknowledgement guarantee.",
        "Una comprobación de permisos en {object} invirtió allow y deny; restaura la semántica existente.",
        "El cursor de {object} omite el elemento del límite; recupera la paginación prometida.",
        "Una tarea cancelada en {object} todavía persiste resultados; corrige esa violación del contrato.",
        "A serialização de {object} deixou de ler dados da versão anterior; restaure a compatibilidade.",
        "Une course dans {object} perd une mise à jour déjà confirmée; rétablis la garantie.",
        "{object} normalisiert Unicode jetzt falsch; stelle das bisherige Ergebnis wieder her.",
    ),
    "feature.contract_first": (
        "Add a streaming export workflow to {object}; no export stream exists today.",
        "Let {object} accept a new credential type while preserving current authentication.",
        "Introduce reverse pagination in {object}, a caller workflow not currently supported.",
        "Give {object} an offline mode that users cannot enable today.",
        "Add a webhook event to {object} without changing existing event payloads.",
        "Allow {object} to restore archived sessions through a new command.",
        "Permite que {object} configure jitter en backoff; hoy solo existen esperas fijas.",
        "Añade a {object} un formato de salida que el producto todavía no ofrece.",
        "Cria em {object} uma opção para pausar e retomar trabalhos ativos.",
        "Ajoute à {object} la compression d'un protocole qui ne la prend pas encore en charge.",
        "Erweitere {object} um einen neuen schreibgeschützten Diagnosemodus.",
        "Aggiungi a {object} un flusso batch che gli utenti non possono ancora eseguire.",
    ),
    "refactor.preserve_behavior": (
        "Split the state machine in {object} into focused components with identical transitions.",
        "Remove internal duplication from {object} without changing bytes on the wire.",
        "Extract private scheduling logic from {object}; retries, delays, and errors must stay identical.",
        "Replace tangled helpers in {object} with clearer boundaries while preserving every caller contract.",
        "Reorganize dependency injection inside {object} with no externally visible change.",
        "Simplify the parser pipeline in {object} while keeping all accepted and rejected inputs stable.",
        "Separa responsabilidades internas de {object} sin modificar ninguna salida observable.",
        "Reduce la complejidad ciclomática de {object} conservando API y comportamiento.",
        "Extraia módulos privados de {object} mantendo os mesmos efeitos e resultados.",
        "Découpe {object} en couches plus claires sans changer son comportement.",
        "Vereinfache die interne Struktur von {object}, ohne Aufrufer zu beeinflussen.",
        "Riorganizza {object} internamente mantenendo identici output ed errori.",
    ),
    "research.evidence_first": (
        "Compare durable queue designs for {object} using primary benchmarks and recommend one.",
        "Investigate why the ecosystem around {object} adopted its current protocol; cite authoritative sources.",
        "Evaluate maintained libraries for {object}, including licensing and compatibility evidence.",
        "Gather facts about deployment strategies for {object} and mark unresolved assumptions.",
        "Study competing storage approaches for {object} before recommending a direction.",
        "Research relevant standards for {object} and distinguish requirements from interpretation.",
        "Investiga alternativas para {object} y respalda la recomendación con fuentes primarias.",
        "Compara implementaciones de {object}, separando hechos comprobados de inferencias.",
        "Pesquise opções para {object} e registre riscos ainda incertos antes de recomendar.",
        "Étudie les architectures possibles de {object} avec des documents officiels.",
        "Untersuche die Alternativen für {object} anhand belastbarer Quellen.",
        "Confronta gli approcci per {object} e motiva la scelta con prove verificabili.",
    ),
    "review.read_only": (
        "Review the patch to {object} for concrete regressions and make no source changes.",
        "Audit {object}, rank evidence-backed risks, and return findings only.",
        "Inspect the permission boundary in {object}; do not implement corrections.",
        "Check {object} for compatibility defects and leave every file untouched.",
        "Perform a read-only security review of {object} and separate facts from speculation.",
        "Examine the tests and implementation of {object}, reporting only demonstrable gaps.",
        "Revisa {object} y prioriza defectos reproducibles sin editar el repositorio.",
        "Audita el diff de {object}; entrega hallazgos, no un parche.",
        "Revise {object} e relate regressões comprovadas sem aplicar correções.",
        "Examine {object} et classe les défauts sans toucher au code.",
        "Prüfe {object} auf belegbare Risiken und ändere nichts.",
        "Esamina {object} e riporta problemi concreti senza correggerli.",
    ),
    "performance.measure_first": (
        "Benchmark cold startup in {object} and reduce the measured initialization time.",
        "Profile allocations in the correct but memory-heavy path through {object}.",
        "Measure throughput in {object} under realistic load and remove the observed bottleneck.",
        "Establish query p95 for {object}, then reduce it on the same workload.",
        "Profile CPU use in {object}; outputs are correct but processing is too expensive.",
        "Measure tail latency in {object} before optimizing its critical path.",
        "Mide el pico de memoria de {object} y reduce el consumo comprobado.",
        "Perfila {object}, que funciona correctamente pero tarda demasiado bajo carga.",
        "Meça as alocações de {object} e reduza o custo observado.",
        "Mesure le débit de {object} puis élimine le goulot démontré.",
        "Miss die p99-Latenz von {object} und optimiere den gemessenen Engpass.",
        "Misura il tempo di avvio di {object} e migliora il percorso critico.",
    ),
}

_UNCATEGORIZED_TEMPLATES = (
    "Explain what {object} currently does; I am not asking for changes.",
    "What is the public contract of {object}? Answer conceptually.",
    "Thanks, the explanation of {object} is enough for now.",
    "Give me a status summary for {object}; do not continue implementation.",
    'The log from {object} says "fix the retry bug"; translate that text only.',
    'A ticket about {object} is titled "add streaming export"; summarize the title.',
    'The documentation for {object} says "optimize latency"; explain what it means.',
    "If we ever rewrote {object}, what tradeoffs would matter? Do nothing now.",
    "No modifiques {object}; solo explícame el flujo actual.",
    "Gracias, ya no hace falta cambiar {object}.",
    "Resume el estado de {object} sin ejecutar acciones.",
    'El log de {object} dice "corrige el timeout"; interpreta el mensaje.',
    "¿Qué significa backpressure en un componente como {object}?",
    "Tal vez migremos {object} el año próximo; no prepares cambios todavía.",
    "Explique apenas o comportamento atual de {object}.",
    'O comentário em {object} diz "adicione suporte"; traduza a frase.',
    "Décris le rôle actuel de {object} sans proposer de modification.",
    "Merci, aucun autre travail sur {object} n'est nécessaire.",
    "Beschreibe nur die aktuelle Verantwortung von {object}.",
    "Spiega il messaggio di errore di {object} senza modificare file.",
)


def build_large_calibration_corpus() -> list[SemanticExample]:
    """Materialize 736 fitting rows from distinct method and discourse families."""
    examples: list[SemanticExample] = []
    modifying_policies = {
        "bugfix.root_cause",
        "feature.contract_first",
        "refactor.preserve_behavior",
        "performance.measure_first",
    }
    for policy, templates in _METHOD_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for component_index, component in enumerate(_COMPONENTS):
                examples.append(SemanticExample(
                    id=(
                        f"large-cal-{policy}-{template_index:02d}-"
                        f"{component_index:02d}"
                    ),
                    text=template.format(object=component),
                    policy=policy,
                    write_authority=policy in modifying_policies,
                ))
    for template_index, template in enumerate(_UNCATEGORIZED_TEMPLATES):
        for component_index, component in enumerate(_COMPONENTS):
            examples.append(SemanticExample(
                id=f"large-cal-neutral-{template_index:02d}-{component_index:02d}",
                text=template.format(object=component),
                policy=None,
                write_authority=False,
            ))
    return examples


def audit_large_calibration_corpus() -> dict[str, object]:
    """Report size, balance, and exact duplicate safety for the augmentation."""
    examples = build_large_calibration_corpus()
    return {
        "version": AUGMENTATION_VERSION,
        "examples": len(examples),
        "by_policy": dict(sorted(Counter(
            example.policy or "uncategorized" for example in examples
        ).items())),
        "families_by_policy": {
            policy: len(templates)
            for policy, templates in sorted(_METHOD_TEMPLATES.items())
        },
        "neutral_families": len(_UNCATEGORIZED_TEMPLATES),
        "duplicate_ids": len(examples) - len({example.id for example in examples}),
        "duplicate_texts": len(examples) - len({example.text for example in examples}),
    }


__all__ = [
    "AUGMENTATION_VERSION",
    "audit_large_calibration_corpus",
    "build_large_calibration_corpus",
]
