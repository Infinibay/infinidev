"""Synthetic compound-task corpus for multi-label task-method detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


COMPOUND_CORPUS_VERSION = "task-policy-compound-v1"


@dataclass(frozen=True)
class CompoundExample:
    """One request requiring two compatible task methods."""

    id: str
    text: str
    policies: tuple[str, ...]
    split: str
    write_authority: bool


_COMPONENTS = {
    "calibration": (
        "AsterRouter", "BirchScheduler", "CinderParser", "DeltaStore",
        "EmberClient", "FjordWorker", "GroveIndexer", "HelixGateway",
    ),
    "validation": ("IonBroker", "JuniperCache", "KeplerRuntime", "LotusQueue"),
    "holdout": ("MosaicTransport", "NovaCompiler", "OpalLedger", "PrairieServer"),
}


_TEMPLATES: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("bugfix.root_cause", "refactor.preserve_behavior"): {
        "calibration": (
            "Repair the state leak in {object} and untangle the responsible module without changing unrelated behavior.",
            "Restore the broken contract in {object}; restructure the duplicated path that caused it while preserving other outputs.",
            "Corrige la regresión de {object} y reorganiza la lógica implicada sin alterar los demás contratos.",
            "Répare le défaut de {object} puis simplifie la structure fautive sans autre changement visible.",
            "Fix the existing boundary error in {object} and extract the responsible logic with all other results unchanged.",
            "Restaure o comportamento quebrado de {object} e separe o código acoplado sem novas mudanças visíveis.",
        ),
        "validation": (
            "Fix the incorrect retry result in {object} and extract the tangled scheduler while all other behavior stays stable.",
            "Corrige el fallo reproducible de {object} y separa responsabilidades internas conservando el resto.",
        ),
        "holdout": (
            "Restore the existing guarantee in {object} and cleanly split the coupled code that broke it.",
            "Repara o contrato quebrado de {object} e reorganiza apenas a estrutura responsável.",
        ),
    },
    ("feature.contract_first", "refactor.preserve_behavior"): {
        "calibration": (
            "Add streaming support to {object} and reshape its transport boundary only as needed, preserving existing callers.",
            "Introduce the new workflow in {object}; separate the coupled internals while old outputs remain identical.",
            "Añade el modo solicitado a {object} y refactoriza el límite interno necesario sin romper usos actuales.",
            "Ajoute la nouvelle capacité à {object} et réorganise uniquement la frontière requise.",
            "Create the missing user workflow in {object} and split the necessary internals while preserving existing paths.",
            "Erweitere {object} um den neuen Ablauf und ordne nur die betroffene interne Grenze neu.",
        ),
        "validation": (
            "Give {object} a new credential flow and extract the authentication layer without changing current credentials.",
            "Implementa una capacidad nueva en {object} y ordena su estructura preservando el comportamiento existente.",
        ),
        "holdout": (
            "Extend {object} with reverse pagination and refactor the cursor boundary while keeping forward pagination stable.",
            "Adicione um novo fluxo a {object} e simplifique a implementação sem alterar os fluxos antigos.",
        ),
    },
    ("bugfix.root_cause", "performance.measure_first"): {
        "calibration": (
            "Correct the stale result in {object} and recover the latency baseline that regressed with it.",
            "Fix the timeout semantics in {object}, then measure and remove the newly introduced bottleneck.",
            "Corrige el resultado incorrecto de {object} y recupera el rendimiento medido anterior.",
            "Répare la réponse erronée de {object} puis mesure et corrige la régression de débit.",
            "Restore the correct output in {object}, benchmark the affected path, and recover its prior p95.",
            "Corrija a resposta inválida de {object} e elimine o gargalo medido que surgiu com a regressão.",
        ),
        "validation": (
            "Restore correct pagination in {object} and eliminate its measured p95 regression on the same workload.",
            "Corrige el reintento extra de {object} y reduce la latencia comprobada sin cambiar el contrato.",
        ),
        "holdout": (
            "Repair the corrupted cache response in {object} and bring measured throughput back to baseline.",
            "Restaure o resultado correto de {object} e elimine a regressão de desempenho verificada.",
        ),
    },
    ("feature.contract_first", "research.evidence_first"): {
        "calibration": (
            "Compare supported protocol options for {object}, choose with evidence, then implement the requested new mode.",
            "Research maintained libraries for {object} and add the new workflow using the best-supported choice.",
            "Investiga alternativas para {object}, recomienda una con fuentes y después implementa la capacidad nueva.",
            "Étudie les options de {object}, justifie le choix puis ajoute la fonctionnalité demandée.",
            "Evaluate authoritative design options for {object}, document the choice, and build the new capability.",
            "Untersuche belastbare Optionen für {object}, begründe die Auswahl und implementiere danach den neuen Ablauf.",
        ),
        "validation": (
            "Determine the best documented storage approach for {object} and then build the new archival path.",
            "Compara opciones fiables para {object} y crea después el nuevo flujo solicitado.",
        ),
        "holdout": (
            "Evaluate primary evidence for authentication designs in {object}, select one, and add the new credential type.",
            "Pesquise alternativas para {object} e implemente a nova capacidade com base na evidência.",
        ),
    },
    ("bugfix.root_cause", "research.evidence_first"): {
        "calibration": (
            "Investigate the documented protocol behavior around {object}, establish the cause, and repair the regression.",
            "Gather evidence about the compatibility break in {object}, then restore the correct contract.",
            "Investiga con evidencia por qué falla {object} y después corrige la causa demostrada.",
            "Recherche la cause documentée du défaut de {object} puis répare le comportement.",
            "Consult reliable protocol evidence for the {object} regression, confirm the cause, and implement its repair.",
            "Pesquise a garantia violada por {object}, confirme a causa com evidências e corrija-a.",
        ),
        "validation": (
            "Use authoritative sources to determine why {object} rejects valid input, then implement the correction.",
            "Averigua la causa verificable del fallo de {object} y corrígela después.",
        ),
        "holdout": (
            "Establish from reliable evidence which guarantee {object} violates and then restore it.",
            "Reúna evidências sobre a regressão de {object} e corrija a causa confirmada.",
        ),
    },
    ("feature.contract_first", "performance.measure_first"): {
        "calibration": (
            "Add the requested batch API to {object} with a measurable throughput budget and benchmark the new path.",
            "Implement streaming in {object}; define its latency target and optimize against a representative baseline.",
            "Añade el nuevo flujo a {object} con un presupuesto de rendimiento y verifica la carga real.",
            "Ajoute le mode demandé à {object} avec un objectif de débit mesurable.",
            "Build the new bulk operation in {object} and prove that its representative latency stays within budget.",
            "Implementiere die neue Fähigkeit in {object} und optimiere sie gegen einen reproduzierbaren Benchmark.",
        ),
        "validation": (
            "Create the new indexing mode in {object} and ensure its measured p95 meets the acceptance criterion.",
            "Implementa la capacidad nueva de {object} y optimiza su benchmark representativo.",
        ),
        "holdout": (
            "Extend {object} with compressed export and validate both its contract and measured throughput.",
            "Adicione o novo endpoint a {object} com uma meta de latência comprovável.",
        ),
    },
    ("research.evidence_first", "review.read_only"): {
        "calibration": (
            "Review {object} without edits and compare its design against authoritative guidance before reporting findings.",
            "Audit the patch in {object}, research the relevant standard, and deliver an evidence-backed read-only report.",
            "Revisa {object} sin modificarlo y contrasta los hallazgos con fuentes primarias.",
            "Examine {object} sans changement et compare le code aux recommandations officielles.",
            "Inspect {object} without edits and validate every reported risk against the governing specification.",
            "Revise {object} sem aplicar correções e sustente os achados com documentação autoritativa.",
        ),
        "validation": (
            "Perform a read-only review of {object} and research the protocol rules needed to validate each finding.",
            "Audita {object} sin editar y sustenta cada hallazgo con documentación fiable.",
        ),
        "holdout": (
            "Inspect {object} for defects, consult primary specifications, and return findings without a patch.",
            "Revise {object} sem alterações e valide os riscos usando fontes oficiais.",
        ),
    },
}


def build_compound_corpus(split: str) -> list[CompoundExample]:
    """Materialize one leakage-separated compound split."""
    if split not in _COMPONENTS:
        raise ValueError(f"unknown compound split: {split}")
    examples: list[CompoundExample] = []
    for policies, by_split in _TEMPLATES.items():
        for template_index, template in enumerate(by_split[split]):
            for component_index, component in enumerate(_COMPONENTS[split]):
                examples.append(CompoundExample(
                    id=(
                        f"compound-{split}-{policies[0]}-{policies[1]}-"
                        f"{template_index:02d}-{component_index:02d}"
                    ),
                    text=template.format(object=component),
                    policies=policies,
                    split=split,
                    write_authority="review.read_only" not in policies,
                ))
    return examples


def audit_compound_corpus() -> dict[str, object]:
    """Report balance, pair coverage, duplicates, and cross-split leakage."""
    by_split = {split: build_compound_corpus(split) for split in _COMPONENTS}
    pair_counts = Counter(
        (example.split, example.policies)
        for examples in by_split.values()
        for example in examples
    )
    texts_by_split = {
        split: {example.text.casefold() for example in examples}
        for split, examples in by_split.items()
    }
    overlaps = {
        f"{left}:{right}": len(texts_by_split[left] & texts_by_split[right])
        for left_index, left in enumerate(_COMPONENTS)
        for right in tuple(_COMPONENTS)[left_index + 1:]
    }
    all_examples = [example for examples in by_split.values() for example in examples]
    return {
        "version": COMPOUND_CORPUS_VERSION,
        "splits": {split: len(examples) for split, examples in by_split.items()},
        "pairs": len(_TEMPLATES),
        "pair_counts": {
            f"{split}:{'+'.join(pair)}": count
            for (split, pair), count in sorted(pair_counts.items())
        },
        "duplicate_ids": len(all_examples) - len({item.id for item in all_examples}),
        "duplicate_texts": len(all_examples) - len({item.text for item in all_examples}),
        "cross_split_text_overlap": overlaps,
    }


__all__ = [
    "COMPOUND_CORPUS_VERSION",
    "CompoundExample",
    "audit_compound_corpus",
    "build_compound_corpus",
]
