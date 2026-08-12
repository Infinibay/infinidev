"""High-volume synthetic augmentation for provider-visible reasoning patterns."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


REASONING_AUGMENTATION_VERSION = "reasoning-pattern-augmentation-v1"


@dataclass(frozen=True)
class SyntheticReasoningText:
    """One labeled reasoning fragment before observable features are attached."""

    id: str
    text: str
    label: str


_CONTEXTS = (
    "parser", "scheduler", "cache", "client",
    "worker", "gateway", "indexer", "transport",
)

_TEMPLATES: dict[str, tuple[str, ...]] = {
    "excessive_exploration": (
        "The exact {context} edit and its test are known, but I will browse unrelated modules first.",
        "I have sufficient evidence about the {context}; another broad repository search feels safer than acting.",
        "Ya localicé la causa en {context}, aunque seguiré leyendo archivos sin relación antes de cambiarla.",
        "Tengo abierto el punto correcto de {context}; aun así exploraré todo el paquete antes del parche.",
        "O local exato em {context} está claro, mas vou procurar mais contexto irrelevante antes de editar.",
        "La cible de {context} et son test sont connus, pourtant je continue une exploration générale.",
    ),
    "retry_loop": (
        "The {context} command failed unchanged; I will repeat the identical call with identical arguments.",
        "The same {context} schema error returned again, so my next action is the same request once more.",
        "Falló la operación de {context}; repetiré exactamente el mismo comando en el mismo cwd.",
        "El parche de {context} fue rechazado otra vez y lo reenviaré sin modificar contexto ni diff.",
        "A ação de {context} falhou; vou repeti-la sem alterar hipótese, parâmetros ou diretório.",
        "L'appel de {context} a échoué; je le relance à l'identique sans adapter l'approche.",
    ),
    "premature_completion": (
        "A required {context} plan item is still pending, but I can mark the task complete now.",
        "The {context} acceptance criterion remains open; I will nevertheless send a completion response.",
        "Todavía falta un requisito obligatorio de {context}, aunque voy a declarar el trabajo terminado.",
        "No hice el cambio pedido en {context}, pero puedo informar que la implementación está completa.",
        "Uma etapa necessária de {context} continua pendente; mesmo assim vou concluir.",
        "Une exigence de {context} reste ouverte, mais je vais annoncer la fin du travail.",
    ),
    "speculative_claim": (
        "The {context} must contain a race condition, although I have no source, log, or test evidence.",
        "I can guarantee the {context} is secure based solely on its name.",
        "Seguro que {context} falla por memoria, aunque todavía no observé ningún dato.",
        "La causa raíz de {context} está confirmada sólo por intuición, sin resultados verificables.",
        "O problema de {context} é definitivamente a rede, sem log ou teste que sustente isso.",
        "Le défaut de {context} vient forcément du cache, sans qu'aucune observation ne l'indique.",
    ),
    "verification_gap": (
        "I edited the {context} behavior and will finish without running its directly covering test.",
        "The {context} patch looks plausible, so I will omit the required verifier.",
        "Ya cambié {context}; cerraré sin comprobar el criterio de aceptación afectado.",
        "La API de {context} cambió, pero no ejecutaré ninguna prueba de compatibilidad.",
        "A implementação de {context} foi alterada; vou encerrar sem verificar o resultado.",
        "Le code de {context} a changé; je termine sans lancer le contrôle pertinent.",
    ),
    "healthy_progress": (
        "I found the {context} cause, made the minimal edit, and its focused regression test passes.",
        "The first {context} probe failed; I changed the hypothesis and the corrected verifier now succeeds.",
        "Localicé la causa de {context}, apliqué el cambio acotado y pasó la prueba pertinente.",
        "El intento de {context} falló; adapté la estrategia con esa evidencia y ahora la verificación pasa.",
        "A correção mínima de {context} foi aplicada e o teste diretamente relacionado passou.",
        "Pour {context}, j'ai adapté l'approche puis confirmé la correction avec le test ciblé.",
    ),
    "uncategorized": (
        "I will inspect the named {context} function once before deciding on the requested change.",
        "The failed {context} test is evidence; I will use it to choose a materially different next action.",
        "Voy a explicar el contrato actual de {context}, que es exactamente lo solicitado.",
        "La causa de {context} sigue siendo una hipótesis hasta que una prueba la confirme.",
        "Vou comparar as alternativas de {context} antes de fazer uma recomendação.",
        "Je lancerai le test ciblé de {context} après avoir effectué la modification demandée.",
        "I need one targeted read of {context} to confirm the signature before editing.",
        "The {context} result disproves my first theory, so I will form a different testable hypothesis.",
        "I will preserve the public {context} contract while implementing the requested internal change.",
        "The user asked only for the current {context} status, so I will report it and stop.",
        "There is no evidence that {context} is broken; I will not present the suspicion as fact.",
        "After the scoped {context} edit, the directly covering verifier is still part of the plan.",
        "The {context} question is conceptual and does not authorize repository changes.",
        "I will use the observed {context} output rather than guessing at its cause.",
        "Necesito una lectura puntual de {context} para confirmar la interfaz antes de editar.",
        "El resultado de {context} contradice mi hipótesis inicial; probaré una explicación distinta.",
        "Conservaré el contrato público de {context} durante el cambio interno solicitado.",
        "El usuario pidió únicamente el estado de {context}; lo resumiré sin continuar.",
        "No hay evidencia de que {context} esté roto, así que no afirmaré esa causa.",
        "Después del cambio acotado en {context} ejecutaré la prueba que lo cubre.",
        "La consulta sobre {context} es conceptual y no concede permiso para editar.",
        "Usaré la salida observada de {context} para decidir el siguiente paso.",
        "Preciso ler apenas a função indicada de {context} antes de alterar qualquer coisa.",
        "O resultado de {context} exige uma hipótese diferente, não a repetição da anterior.",
        "A pergunta sobre {context} pede uma explicação, não uma modificação.",
        "Après la petite modification de {context}, je vérifierai le cas directement concerné.",
        "Le soupçon sur {context} reste une hypothèse tant qu'aucune preuve ne le confirme.",
        "Ich lese gezielt die genannte {context}-Funktion, bevor ich eine Änderung entscheide.",
        "Die Frage zu {context} verlangt eine Erklärung und keine Codeänderung.",
        "Dopo la modifica limitata a {context}, eseguirò il controllo pertinente.",
        "The quoted instruction near {context} is context to explain, not an action to execute.",
        "I have not completed the {context} work; the remaining verifier is explicitly acknowledged.",
    ),
}


def build_reasoning_augmentation() -> list[SyntheticReasoningText]:
    """Materialize 48 additional calibration examples per label."""
    return [
        SyntheticReasoningText(
            id=f"reasoning-aug-{label}-{template_index:02d}-{context_index:02d}",
            text=template.format(context=context),
            label=label,
        )
        for label, templates in _TEMPLATES.items()
        for template_index, template in enumerate(templates)
        for context_index, context in enumerate(_CONTEXTS)
    ]


def audit_reasoning_augmentation() -> dict[str, object]:
    """Report balance and exact duplicates."""
    examples = build_reasoning_augmentation()
    return {
        "version": REASONING_AUGMENTATION_VERSION,
        "examples": len(examples),
        "by_label": dict(sorted(Counter(item.label for item in examples).items())),
        "duplicate_ids": len(examples) - len({item.id for item in examples}),
        "duplicate_texts": len(examples) - len({item.text for item in examples}),
    }


__all__ = ["REASONING_AUGMENTATION_VERSION", "SyntheticReasoningText", "audit_reasoning_augmentation", "build_reasoning_augmentation"]
