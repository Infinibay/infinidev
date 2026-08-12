"""Synthetic non-task discourse corpus for the task-policy abstention gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


DISCOURSE_CORPUS_VERSION = "task-policy-discourse-v1"


@dataclass(frozen=True)
class DiscourseExample:
    """One utterance that must not activate a task-method prompt."""

    id: str
    text: str
    category: str
    split: str


_OBJECTS = {
    "calibration": (
        "AtlasParser", "BirchQueue", "CinderClient", "DeltaCache",
        "EmberWorker", "FjordGateway", "GroveIndexer", "HelixTransport",
    ),
    "validation": ("IonRouter", "JuniperStore", "KeplerRuntime", "LotusBroker"),
    "holdout": ("MosaicCompiler", "NovaLedger", "OpalScheduler", "PrairieServer"),
}

_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "acknowledgement": {
        "calibration": (
            "Thanks, the explanation of {object} is enough; no further work is needed.",
            "Perfecto, ya entendí {object}; no continúes con cambios.",
            "Obrigado, a resposta sobre {object} resolveu minha dúvida.",
            "Merci, je n'ai besoin de rien d'autre sur {object}.",
            "That answers my question about {object}; we can stop here.",
            "Listo, quedó claro lo de {object}; era solo una confirmación.",
        ),
        "validation": ("Understood, no more work on {object}.", "Entendido; no cambies {object}."),
        "holdout": ("All clear about {object}, thanks.", "Va bene, nessun altro intervento su {object}."),
    },
    "quoted_action": {
        "calibration": (
            'The log from {object} says "fix the retry bug"; translate that sentence only.',
            'El ticket de {object} dice "agrega soporte"; explícame el título, no lo ejecutes.',
            'O comentário em {object} diz "otimize isto"; interprete apenas a frase.',
            'Le journal de {object} affiche "refactor required"; explique ce texte.',
            'Someone wrote "review and repair" near {object}; summarize what they meant.',
            'La documentación cita "implement a cache" para {object}; traduce la cita solamente.',
        ),
        "validation": (
            'The phrase "make it faster" appears in {object}; explain the wording only.',
            'En {object} aparece "corrige el error"; interpreta el mensaje sin actuar.',
        ),
        "holdout": (
            'Translate "add a command" from the {object} issue title without doing it.',
            'Spiega soltanto la citazione "fix the crash" relativa a {object}.',
        ),
    },
    "conceptual_question": {
        "calibration": (
            "What does backpressure mean for a component such as {object}?",
            "¿Cuál es la diferencia conceptual entre timeout y deadline en {object}?",
            "O que significa idempotência no contexto de {object}?",
            "Que veut dire cohérence éventuelle pour {object} ?",
            "Explain the idea of cursor pagination using {object} only as an example.",
            "Define observability conceptually; {object} is just context.",
        ),
        "validation": ("What is a retry budget in systems like {object}?", "¿Qué significa ABI para {object}?"),
        "holdout": ("Describe eventual consistency around {object}.", "Was bedeutet Backoff bei {object}?"),
    },
    "status_only": {
        "calibration": (
            "Give me the current status of {object}; do not resume implementation.",
            "Resume qué se hizo hasta ahora en {object}, sin ejecutar nada nuevo.",
            "Informe apenas o progresso atual de {object}; não continue o trabalho.",
            "Résume l'état actuel de {object} sans reprendre les modifications.",
            "Which checks have already run for {object}? Report only recorded results.",
            "Dime qué queda pendiente en {object}; por ahora no lo hagas.",
        ),
        "validation": ("Report progress on {object} only.", "¿En qué estado está {object}? No continúes."),
        "holdout": ("Summarize completed work on {object}.", "Riassumi lo stato di {object} senza agire."),
    },
    "hypothetical_future": {
        "calibration": (
            "If we redesigned {object} next year, what tradeoffs would matter? Do nothing now.",
            "Tal vez migremos {object} más adelante; comenta riesgos sin preparar cambios.",
            "Se um dia reescrevermos {object}, quais decisões seriam importantes?",
            "Si nous remplacions {object} un jour, quels compromis faudrait-il étudier ?",
            "Imagine that {object} needed offline support in the future; discuss implications only.",
            "Hipotéticamente, ¿qué pasaría si {object} usara otro protocolo?",
        ),
        "validation": ("Suppose {object} were distributed someday; discuss only.", "Si cambiáramos {object} en el futuro, ¿qué riesgos habría?"),
        "holdout": ("What might matter in a future rewrite of {object}?", "In futuro, quali opzioni avrebbe {object}?"),
    },
    "explanation_only": {
        "calibration": (
            "Explain what {object} currently does without recommending changes.",
            "Descríbeme el flujo actual de {object}; no propongas modificaciones.",
            "Explique apenas as responsabilidades atuais de {object}.",
            "Décris le comportement présent de {object}, sans suggestion.",
            "Walk me through the existing public contract of {object}.",
            "Ayúdame a entender por qué existe {object}, solo como explicación.",
        ),
        "validation": ("Describe the current API of {object} only.", "Explica el diseño actual de {object}."),
        "holdout": ("Teach me how {object} works today.", "Erkläre nur die heutige Rolle von {object}."),
    },
    "ambiguous_method": {
        "calibration": (
            "What would be the appropriate kind of work around {object}? Do not start it.",
            "No sé si {object} requeriría revisar, investigar o cambiar; solo aclara las opciones.",
            "Seria melhor estudar ou alterar {object}? Responda sem executar nenhuma opção.",
            "Faudrait-il examiner ou modifier {object} ? Discute seulement la méthode.",
            "Is work on {object} usually considered maintenance or product development?",
            "Compara conceptualmente refactorizar y corregir en el caso de {object}.",
        ),
        "validation": ("Would {object} call for research or implementation? Answer only.", "¿Qué tipo de tarea sería tocar {object}?"),
        "holdout": ("Classify possible work on {object}; take no action.", "Che tipo di attività sarebbe lavorare su {object}?"),
    },
    "out_of_domain": {
        "calibration": (
            "Write a short fictional biography of a musician named {object}.",
            "Dame una receta sencilla cuyo nombre creativo sea {object}.",
            "Crie uma adivinha infantil com a palavra {object}.",
            "Compose un petit poème sur un bateau appelé {object}.",
            "What weather would suit a picnic called {object}?",
            "Inventa un nombre de banda inspirado en {object}.",
        ),
        "validation": ("Tell a harmless joke involving {object}.", "Escribe un haiku sobre {object}."),
        "holdout": ("Invent a sandwich named {object}.", "Scrivi una rima con {object}."),
    },
}


def build_discourse_corpus(split: str) -> list[DiscourseExample]:
    """Build a phrase- and component-separated discourse split."""
    if split not in _OBJECTS:
        raise ValueError(f"unknown discourse split: {split}")
    return [
        DiscourseExample(
            id=f"discourse-{split}-{category}-{template_index:02d}-{object_index:02d}",
            text=template.format(object=object_name),
            category=category,
            split=split,
        )
        for category, split_templates in _TEMPLATES.items()
        for template_index, template in enumerate(split_templates[split])
        for object_index, object_name in enumerate(_OBJECTS[split])
    ]


def audit_discourse_corpus() -> dict[str, object]:
    """Report category balance and exact split leakage."""
    by_split = {split: build_discourse_corpus(split) for split in _OBJECTS}
    all_examples = [item for split in by_split.values() for item in split]
    texts = {split: {item.text.casefold() for item in rows} for split, rows in by_split.items()}
    return {
        "version": DISCOURSE_CORPUS_VERSION,
        "splits": {split: len(rows) for split, rows in by_split.items()},
        "calibration_by_category": dict(sorted(Counter(
            item.category for item in by_split["calibration"]
        ).items())),
        "duplicate_ids": len(all_examples) - len({item.id for item in all_examples}),
        "duplicate_texts": len(all_examples) - len({item.text for item in all_examples}),
        "cross_split_overlap": sum(
            len(texts[left] & texts[right])
            for index, left in enumerate(_OBJECTS)
            for right in tuple(_OBJECTS)[index + 1:]
        ),
    }


__all__ = ["DISCOURSE_CORPUS_VERSION", "DiscourseExample", "audit_discourse_corpus", "build_discourse_corpus"]
