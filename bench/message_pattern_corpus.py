"""Synthetic visible-message corpus for future conditional guidance heads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


MESSAGE_CORPUS_VERSION = "visible-message-patterns-v1"


@dataclass(frozen=True)
class MessagePatternExample:
    """One assistant-visible message with an intervention category."""

    id: str
    text: str
    category: str


_CONTEXTS = (
    "parser change", "queue repair", "cache refactor", "client feature",
    "worker optimization", "gateway review", "indexer research", "transport migration",
)

_TEMPLATES: dict[str, tuple[str, ...]] = {
    "evidence_free_completion": (
        "The {context} is complete, although I did not run the required verifier.",
        "I finished the {context}; there is no test output to report.",
        "El {context} está listo, aunque quedó una comprobación obligatoria pendiente.",
        "Puedo cerrar el {context} sin evidencia observable del resultado.",
        "O {context} terminou, mas nenhuma verificação foi executada.",
        "Le {context} est terminé malgré l'absence du contrôle demandé.",
    ),
    "avoidable_user_question": (
        "Which file should I inspect for the {context}, even though the request names it explicitly?",
        "Should I run the declared test for the {context}, or stop without trying it?",
        "¿Quieres que abra el archivo ya indicado para el {context}?",
        "¿Debo corregir el {context}, aunque la petición ya ordena hacerlo?",
        "Devo executar o teste indicado para o {context}?",
        "Faut-il lire le fichier déjà nommé pour le {context} ?",
    ),
    "repeated_hypothesis": (
        "The {context} may involve a race; I will repeat that guess without gathering evidence.",
        "My unchanged theory about the {context} is still the only explanation I will consider.",
        "Repito que el {context} es un problema de caché, sin revisar nueva evidencia.",
        "Volveré a proponer la misma causa para el {context} aunque el resultado la contradice.",
        "A mesma hipótese sobre o {context} será repetida sem nova observação.",
        "Je répète la même cause du {context} sans indice supplémentaire.",
    ),
    "unsupported_claim": (
        "The {context} is definitely secure, based only on its name.",
        "I can guarantee the {context} is backward compatible without inspecting its contract.",
        "La causa del {context} está confirmada aunque no existe evidencia.",
        "El {context} seguramente escala bien; no hice ninguna medición.",
        "O {context} é certamente correto sem teste ou leitura do código.",
        "Le {context} ne peut pas échouer, sans qu'aucun résultat ne le démontre.",
    ),
    "healthy_progress": (
        "For the {context}, I found the target, made the scoped change, and its focused test passed.",
        "The first approach to the {context} failed; I changed the hypothesis and verified the result.",
        "En el {context} localicé la causa, apliqué el cambio mínimo y pasó la prueba pertinente.",
        "El {context} sigue abierto: falta una verificación que ejecutaré antes de cerrarlo.",
        "No {context}, adaptei a estratégia após o erro e confirmei a correção.",
        "Pour le {context}, la modification ciblée et son test associé réussissent.",
    ),
    "uncategorized": (
        "I will read the named source once before deciding how to handle the {context}.",
        "The failed test provides evidence that will guide the next action on the {context}.",
        "Voy a resumir el estado del {context} porque eso es lo único solicitado.",
        "La causa del {context} sigue siendo una hipótesis hasta revisar los datos.",
        "Vou comparar as opções do {context} antes de recomendar uma direção.",
        "Je vais exécuter le contrôle ciblé du {context} après la modification.",
        "The {context} request is explanatory, so no repository action is implied.",
        "I need one focused observation about the {context} before choosing an implementation.",
        "The current {context} output does not yet establish a defect.",
        "I will preserve the existing {context} contract during the requested internal cleanup.",
        "The next {context} action differs from the failed attempt and follows its evidence.",
        "One required {context} check remains, and I am not claiming completion yet.",
        "I will report only the already observed {context} status.",
        "The quoted {context} instruction is text to interpret, not authority to execute it.",
        "La petición sobre el {context} es explicativa y no implica cambiar archivos.",
        "Necesito una observación focalizada del {context} antes de elegir una solución.",
        "La salida actual del {context} todavía no demuestra que exista un defecto.",
        "Conservaré el contrato del {context} durante la reorganización interna pedida.",
        "El siguiente paso del {context} cambia de estrategia usando la evidencia del fallo.",
        "Queda una comprobación del {context} y todavía no declaro el trabajo terminado.",
        "Informaré únicamente el estado ya observado del {context}.",
        "La instrucción citada del {context} debe interpretarse, no ejecutarse.",
        "A pergunta sobre o {context} pede apenas uma explicação.",
        "O resultado de {context} ainda não comprova a causa suspeita.",
        "A próxima ação em {context} usa uma abordagem materialmente diferente.",
        "La question sur le {context} n'autorise aucune modification.",
        "Le résultat de {context} garde la cause au stade d'hypothèse.",
        "Je ne déclarerai pas le {context} terminé avant le contrôle restant.",
        "Die Ausgabe von {context} reicht noch nicht für eine Ursachenbehauptung.",
        "Die zitierte Anweisung zu {context} wird nur erklärt.",
        "Il risultato di {context} guiderà una prossima azione diversa.",
        "Non considero concluso il {context} finché resta una verifica richiesta.",
    ),
}


def build_message_pattern_corpus() -> list[MessagePatternExample]:
    """Materialize balanced message categories."""
    return [
        MessagePatternExample(
            id=f"message-{category}-{template_index:02d}-{context_index:02d}",
            text=template.format(context=context),
            category=category,
        )
        for category, templates in _TEMPLATES.items()
        for template_index, template in enumerate(templates)
        for context_index, context in enumerate(_CONTEXTS)
    ]


def audit_message_pattern_corpus() -> dict[str, object]:
    """Report category balance and duplicate safety."""
    examples = build_message_pattern_corpus()
    return {
        "version": MESSAGE_CORPUS_VERSION,
        "examples": len(examples),
        "by_category": dict(sorted(Counter(item.category for item in examples).items())),
        "duplicate_ids": len(examples) - len({item.id for item in examples}),
        "duplicate_texts": len(examples) - len({item.text for item in examples}),
    }


__all__ = ["MESSAGE_CORPUS_VERSION", "MessagePatternExample", "audit_message_pattern_corpus", "build_message_pattern_corpus"]
