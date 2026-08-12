"""Generate a multilingual, project-family-split task-policy draft corpus.

The projects are fictional. Each archetype records one open-source repository
used only for domain inspiration; no issue text or code is copied. Generated
rows are drafts until a human reviewer approves their label and rationale.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path


@dataclass(frozen=True)
class ProjectArchetype:
    """A fictional project grounded in a public open-source domain."""

    id: str
    split: str
    programming_languages: tuple[str, ...]
    domain: str
    component: str
    inspiration_url: str
    inspiration_license: str


@dataclass(frozen=True)
class ProjectCorpusExample:
    """One reviewable multi-label training row."""

    id: str
    text: str
    split: str
    family: str
    phrase_family: str
    natural_language: str
    project_id: str
    programming_languages: tuple[str, ...]
    operations: tuple[str, ...]
    uncategorized_reason: str | None
    source: str
    inspiration_url: str
    review_status: str = "draft"
    reviewer: str = ""
    gold_rationale: str = ""


PROJECTS: tuple[ProjectArchetype, ...] = (
    ProjectArchetype(
        "emberlint", "calibration", ("Rust", "Python"),
        "Python linter and formatter", "incremental rule engine",
        "https://github.com/astral-sh/ruff", "MIT",
    ),
    ProjectArchetype(
        "typedharbor", "calibration", ("Python",),
        "typed asynchronous API framework", "OpenAPI route dependency graph",
        "https://github.com/fastapi/fastapi", "MIT",
    ),
    ProjectArchetype(
        "sparkrun", "calibration", ("Zig", "TypeScript", "JavaScript"),
        "JavaScript runtime and package toolkit", "module resolver and test runner",
        "https://github.com/oven-sh/bun", "MIT",
    ),
    ProjectArchetype(
        "modalforest", "calibration", ("Rust",),
        "modal editor with syntax and language-server support",
        "tree-sitter syntax layer and LSP client",
        "https://github.com/helix-editor/helix", "MPL-2.0",
    ),
    ProjectArchetype(
        "luaforge", "calibration", ("C", "Lua"),
        "extensible editor core", "event loop and MessagePack RPC boundary",
        "https://github.com/neovim/neovim", "Apache-2.0 and Vim-derived terms",
    ),
    ProjectArchetype(
        "labelstream", "calibration", ("Go",),
        "multi-tenant log aggregation", "label index and chunk storage path",
        "https://github.com/grafana/loki", "AGPL-3.0-only",
    ),
    ProjectArchetype(
        "recordlane", "calibration", ("Ruby", "JavaScript"),
        "MVC web framework", "ORM callbacks and background jobs",
        "https://github.com/rails/rails", "MIT",
    ),
    ProjectArchetype(
        "searchcraft", "validation", ("Java", "Kotlin"),
        "strongly typed distributed-search client", "request builder and HTTP transport",
        "https://github.com/elastic/elasticsearch-java", "Apache-2.0",
    ),
    ProjectArchetype(
        "cloudweave", "validation", ("C#", "F#"),
        "cross-platform cloud web stack", "middleware pipeline and endpoint routing",
        "https://github.com/dotnet/aspnetcore", "MIT",
    ),
    ProjectArchetype(
        "queuegarden", "validation", ("PHP",),
        "web framework with queues and storage adapters", "queue worker and cache drivers",
        "https://github.com/laravel/framework", "MIT",
    ),
    ProjectArchetype(
        "livecanopy", "holdout", ("Elixir", "JavaScript"),
        "real-time server-rendered web framework", "channel lifecycle and socket transport",
        "https://github.com/phoenixframework/phoenix", "MIT",
    ),
    ProjectArchetype(
        "coroutenet", "holdout", ("Kotlin", "Go"),
        "asynchronous multiplatform service toolkit", "coroutine RPC and Gradle integration",
        "https://github.com/ktorio/ktor", "Apache-2.0",
    ),
)


_TEMPLATES: dict[str, dict[str, str]] = {
    "bugfix": {
        "en": "{component} now violates its existing contract; make prior callers work again.",
        "es": "{component} dejó de respetar su contrato; haz que los consumidores vuelvan a funcionar.",
        "pt": "{component} deixou de cumprir o contrato; faça os consumidores voltarem a funcionar.",
        "fr": "{component} ne respecte plus son contrat; rétablis le comportement des appelants.",
    },
    "feature": {
        "en": "Users need a new capability in {component} that is not supported today.",
        "es": "Los usuarios necesitan una capacidad nueva en {component} que hoy no existe.",
        "pt": "Os usuários precisam de uma capacidade nova em {component} que ainda não existe.",
        "fr": "Les utilisateurs ont besoin d'une nouvelle capacité dans {component}.",
    },
    "refactor": {
        "en": "Simplify the internal shape of {component} while every observable result stays identical.",
        "es": "Simplifica la estructura interna de {component} sin alterar ningún resultado observable.",
        "pt": "Simplifique a estrutura interna de {component} sem alterar resultados observáveis.",
        "fr": "Simplifie la structure interne de {component} sans modifier le comportement observable.",
    },
    "research": {
        "en": "Gather reliable evidence about alternatives for {component} and recommend a direction.",
        "es": "Reúne evidencia fiable sobre alternativas para {component} y recomienda una dirección.",
        "pt": "Reúna evidências confiáveis sobre alternativas para {component} e recomende uma direção.",
        "fr": "Rassemble des preuves fiables sur les options pour {component} et recommande une voie.",
    },
    "review": {
        "en": "Inspect {component}, report substantiated defects by impact, and leave the source untouched.",
        "es": "Inspecciona {component}, reporta defectos demostrables por impacto y no edites el código.",
        "pt": "Inspecione {component}, relate defeitos comprovados por impacto e não altere o código.",
        "fr": "Examine {component}, classe les défauts prouvés par impact et ne modifie pas le code.",
    },
    "performance": {
        "en": "Measure a representative baseline for {component}, find the bottleneck, and make it faster.",
        "es": "Mide una línea base representativa de {component}, encuentra el cuello de botella y aceléralo.",
        "pt": "Meça uma linha de base para {component}, encontre o gargalo e torne-o mais rápido.",
        "fr": "Mesure une référence pour {component}, trouve le goulot et accélère-le.",
    },
    "docs": {
        "en": "Rewrite the user guide for {component} with one runnable example and upgrade notes.",
        "es": "Reescribe la guía de {component} con un ejemplo ejecutable y notas de actualización.",
        "pt": "Reescreva o guia de {component} com um exemplo executável e notas de atualização.",
        "fr": "Réécris le guide de {component} avec un exemple exécutable et des notes de mise à niveau.",
    },
    "test": {
        "en": "Add regression coverage around {component}; do not change production behavior.",
        "es": "Añade cobertura de regresión para {component}; no cambies el comportamiento productivo.",
        "pt": "Adicione cobertura de regressão para {component}; não mude o comportamento de produção.",
        "fr": "Ajoute une couverture de régression pour {component} sans changer le comportement produit.",
    },
    "migration": {
        "en": "Move {component} to the next schema with rollout, compatibility, and rollback checks.",
        "es": "Migra {component} al nuevo esquema con rollout, compatibilidad y rollback verificables.",
        "pt": "Migre {component} para o novo esquema com rollout, compatibilidade e rollback verificáveis.",
        "fr": "Migre {component} vers le nouveau schéma avec compatibilité et retour arrière vérifiables.",
    },
    "security": {
        "en": "Assess the trust boundary around {component}, fix the demonstrated weakness, and add a regression.",
        "es": "Evalúa el límite de confianza de {component}, corrige la debilidad demostrada y agrega una regresión.",
        "pt": "Avalie o limite de confiança de {component}, corrija a falha comprovada e adicione uma regressão.",
        "fr": "Évalue la frontière de confiance de {component}, corrige la faille prouvée et ajoute une régression.",
    },
    "planning": {
        "en": "Write a phased implementation plan for {component}, including open decisions and rollback.",
        "es": "Escribe un plan por fases para {component}, con decisiones abiertas y rollback.",
        "pt": "Escreva um plano por fases para {component}, com decisões abertas e rollback.",
        "fr": "Rédige un plan par phases pour {component}, avec décisions ouvertes et retour arrière.",
    },
    "configuration": {
        "en": "Change the default configuration of {component}, preserving explicit user overrides.",
        "es": "Cambia la configuración predeterminada de {component}, preservando overrides explícitos.",
        "pt": "Altere a configuração padrão de {component}, preservando substituições explícitas.",
        "fr": "Modifie la configuration par défaut de {component} en préservant les réglages explicites.",
    },
}

_UNCATEGORIZED: tuple[tuple[str, str, str], ...] = (
    ("en", "conversation", "Thanks, the explanation of {component} answers my question."),
    ("es", "conversation", "Perfecto, la explicación de {component} respondió mi pregunta."),
    ("pt", "conversation", "Obrigado, a explicação de {component} respondeu à minha pergunta."),
    ("fr", "conversation", "Merci, l'explication de {component} répond à ma question."),
    ("en", "quoted_action", 'The {component} log says "implement the migration"; what does that message mean?'),
    ("es", "quoted_action", 'El log de {component} dice "corrige este error"; ¿qué significa?'),
    ("pt", "out_of_domain", "À parte de {component}, qual é uma boa receita de pão?"),
    ("fr", "out_of_domain", "Sans rapport avec {component}, quel temps fera-t-il demain à Lyon ?"),
)


def build_project_corpus() -> list[ProjectCorpusExample]:
    """Materialize 672 draft examples across isolated project families."""
    examples: list[ProjectCorpusExample] = []
    for project in PROJECTS:
        for operation, by_language in _TEMPLATES.items():
            for language, template in by_language.items():
                examples.append(ProjectCorpusExample(
                    id=f"{project.split}-{project.id}-{operation}-{language}",
                    text=template.format(component=project.component),
                    split=project.split,
                    family=project.id,
                    phrase_family=f"{operation}:{language}:v1",
                    natural_language=language,
                    project_id=project.id,
                    programming_languages=project.programming_languages,
                    operations=(operation,),
                    uncategorized_reason=None,
                    source="synthetic_open_source_inspiration",
                    inspiration_url=project.inspiration_url,
                ))
        for index, (language, reason, text) in enumerate(_UNCATEGORIZED):
            examples.append(ProjectCorpusExample(
                id=f"{project.split}-{project.id}-uncategorized-{index:02d}",
                text=text.format(component=project.component),
                split=project.split,
                family=project.id,
                phrase_family=f"uncategorized:{reason}:{language}:v1",
                natural_language=language,
                project_id=project.id,
                programming_languages=project.programming_languages,
                operations=(),
                uncategorized_reason=reason,
                source="synthetic_open_source_inspiration",
                inspiration_url=project.inspiration_url,
            ))
    return examples


def audit_project_corpus(examples: list[ProjectCorpusExample]) -> dict[str, object]:
    """Report diversity, leakage, duplicates, and explicit negative coverage."""
    ids = [item.id for item in examples]
    texts = [" ".join(item.text.casefold().split()) for item in examples]
    family_splits: dict[str, set[str]] = defaultdict(set)
    phrase_splits: dict[str, set[str]] = defaultdict(set)
    for item in examples:
        family_splits[item.family].add(item.split)
        phrase_splits[item.phrase_family].add(item.split)
    leakage = sorted(
        family for family, splits in family_splits.items() if len(splits) > 1
    )
    phrase_leakage = sorted(
        family for family, splits in phrase_splits.items() if len(splits) > 1
    )
    uncategorized = [item for item in examples if not item.operations]
    malformed_uncategorized = [
        item.id for item in examples
        if bool(item.operations) == bool(item.uncategorized_reason)
    ]
    return {
        "examples": len(examples),
        "projects": len({item.project_id for item in examples}),
        "programming_languages": sorted({
            language for item in examples for language in item.programming_languages
        }),
        "natural_languages": dict(sorted(Counter(
            item.natural_language for item in examples
        ).items())),
        "operations": dict(sorted(Counter(
            operation for item in examples for operation in item.operations
        ).items())),
        "splits": dict(sorted(Counter(item.split for item in examples).items())),
        "uncategorized": len(uncategorized),
        "uncategorized_reasons": dict(sorted(Counter(
            item.uncategorized_reason for item in uncategorized
        ).items())),
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_texts": len(texts) - len(set(texts)),
        "family_split_leakage": leakage,
        "phrase_split_leakage": phrase_leakage,
        "malformed_uncategorized": malformed_uncategorized,
        "all_rows_are_draft": all(item.review_status == "draft" for item in examples),
        "release_ready": not any((
            leakage,
            phrase_leakage,
            malformed_uncategorized,
            len(ids) - len(set(ids)),
            len(texts) - len(set(texts)),
        )) and all(item.review_status == "approved" for item in examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    examples = build_project_corpus()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in examples),
            encoding="utf-8",
        )
    print(json.dumps(audit_project_corpus(examples), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
