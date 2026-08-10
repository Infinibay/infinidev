"""Inspect the bundled static Qwen3 embedding space in English and Spanish.

This benchmark uses controlled contrasts instead of assigning meanings to
individual coordinates. Optionally, it compares the static table with its Qwen3
teacher and measures how much Spanish signal survives the rank-512 projection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from infinidev.tools.base.static_qwen3_embedder import StaticQwen3Embedder

try:
    from bench.collect_openai_embedding_teacher import CachedTeacherEmbedder
except ModuleNotFoundError:
    from collect_openai_embedding_teacher import CachedTeacherEmbedder


OBJECTS = {
    "en": (
        "callback registry",
        "authentication cache",
        "JSON parser",
        "database migration",
        "CLI command router",
        "HTTP retry policy",
        "file watcher",
        "configuration loader",
        "event dispatcher",
        "symbol index",
        "test runner",
        "session history",
    ),
    "es": (
        "registro de callbacks",
        "cache de autenticación",
        "parser de JSON",
        "migración de base de datos",
        "enrutador de comandos CLI",
        "política de reintentos HTTP",
        "monitor de archivos",
        "cargador de configuración",
        "despachador de eventos",
        "índice de símbolos",
        "ejecutor de pruebas",
        "historial de sesión",
    ),
}

NEUTRAL = {"en": "Work with the {obj}", "es": "Trabaja con {obj}"}

TRAIN = {
    "en": {
        "discover": (
            "Inspect the {obj} to understand its current behavior",
            "Trace how the {obj} currently works",
        ),
        "change": (
            "Modify the {obj} to implement the requested behavior",
            "Extend the {obj} with a new capability",
        ),
        "test_change": (
            "Add automated regression tests for the {obj}",
            "Expand the test suite to cover the {obj}",
        ),
        "verify": (
            "Run existing checks to verify the {obj}",
            "Validate the completed {obj} without editing it",
        ),
        "document": (
            "Document the behavior and usage of the {obj}",
            "Update the project documentation for the {obj}",
        ),
        "design": (
            "Design an implementation strategy for the {obj}",
            "Plan the architecture and interfaces of the {obj}",
        ),
    },
    "es": {
        "discover": (
            "Inspecciona {obj} para entender su comportamiento actual",
            "Traza cómo funciona actualmente {obj}",
        ),
        "change": (
            "Modifica {obj} para implementar el comportamiento solicitado",
            "Extiende {obj} con una capacidad nueva",
        ),
        "test_change": (
            "Agrega pruebas de regresión automatizadas para {obj}",
            "Amplía la suite de tests para cubrir {obj}",
        ),
        "verify": (
            "Ejecuta verificaciones existentes para validar {obj}",
            "Comprueba que {obj} funcione sin editarlo",
        ),
        "document": (
            "Documenta el comportamiento y uso de {obj}",
            "Actualiza la documentación del proyecto para {obj}",
        ),
        "design": (
            "Diseña una estrategia de implementación para {obj}",
            "Planifica la arquitectura y las interfaces de {obj}",
        ),
    },
}

HOLDOUT = {
    "en": {
        "discover": "Survey the internals of the {obj} before making changes",
        "change": "Enable the requested behavior in the {obj}",
        "test_change": "Cover the {obj} with new regression cases",
        "verify": "Confirm the {obj} works correctly by exercising it",
        "document": "Describe the public contract of the {obj} in the README",
        "design": "Sketch an approach for building the {obj}",
    },
    "es": {
        "discover": "Estudia los componentes internos de {obj} antes de cambiarlo",
        "change": "Habilita el comportamiento solicitado en {obj}",
        "test_change": "Cubre {obj} con casos de regresión nuevos",
        "verify": "Confirma el funcionamiento de {obj} ejercitándolo",
        "document": "Describe el contrato público de {obj} en el README",
        "design": "Esboza un enfoque para construir {obj}",
    },
}

SIMILARITY_PAIRS = {
    "duplicate": (
        ("Implement one-shot callbacks", "Add callbacks that fire once"),
        ("Fix malformed JSON tool arguments", "Repair invalid tool-call JSON"),
        ("Prevent duplicate plan steps", "Deduplicate equivalent plan steps"),
        ("Agregar callbacks de una sola ejecución", "Soportar listeners de un uso"),
        ("Corregir argumentos JSON inválidos", "Reparar el JSON malformado de tools"),
        ("Ejecutar las pruebas de regresión", "Correr los tests que cubren el bug"),
    ),
    "related_distinct": (
        ("Implement one-shot callbacks", "Test one-shot callbacks"),
        ("Fix malformed JSON tool arguments", "Document the tool-call JSON format"),
        ("Inspect authentication routing", "Change authentication routing"),
        ("Agregar callbacks de una sola ejecución", "Probar callbacks de un uso"),
        ("Diseñar el despachador de eventos", "Implementar el despachador de eventos"),
        ("Crear pruebas de prioridad", "Ejecutar las pruebas de prioridad existentes"),
    ),
    "unrelated": (
        ("Implement one-shot callbacks", "Document database migrations"),
        ("Fix malformed JSON tool arguments", "Render images in the terminal"),
        ("Inspect authentication routing", "Generate Minecraft textures"),
        ("Agregar callbacks de una sola ejecución", "Documentar migraciones SQL"),
        ("Corregir argumentos JSON inválidos", "Renderizar imágenes en la terminal"),
        ("Inspeccionar autenticación", "Generar texturas de Minecraft"),
    ),
}


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _all_texts() -> list[str]:
    texts: set[str] = set()
    for language in ("en", "es"):
        for obj in OBJECTS[language]:
            texts.add(NEUTRAL[language].format(obj=obj))
            for templates in TRAIN[language].values():
                texts.update(template.format(obj=obj) for template in templates)
            texts.update(
                template.format(obj=obj) for template in HOLDOUT[language].values()
            )
    for pairs in SIMILARITY_PAIRS.values():
        for left, right in pairs:
            texts.update((left, right))
    texts.update({
        "callback registry",
        "Do not modify the callback registry; only inspect it",
        "Modify the callback registry; do not merely inspect it",
    })
    return sorted(texts)


def _language_texts(language: str) -> list[str]:
    """Return controlled texts whose language is known by construction."""
    texts: set[str] = set()
    for obj in OBJECTS[language]:
        texts.add(NEUTRAL[language].format(obj=obj))
        for templates in TRAIN[language].values():
            texts.update(template.format(obj=obj) for template in templates)
        texts.update(
            template.format(obj=obj) for template in HOLDOUT[language].values()
        )
    return sorted(texts)


def _directions(
    language: str,
    vectors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for phase, templates in TRAIN[language].items():
        deltas: list[np.ndarray] = []
        for obj in OBJECTS[language]:
            neutral = vectors[NEUTRAL[language].format(obj=obj)]
            deltas.extend(
                vectors[template.format(obj=obj)] - neutral for template in templates
            )
        result[phase] = _normalize(np.mean(deltas, axis=0))
    return result


def _action_accuracy(
    language: str,
    vectors: dict[str, np.ndarray],
    directions: dict[str, np.ndarray],
) -> tuple[float, float]:
    correct = 0
    margins: list[float] = []
    for expected, template in HOLDOUT[language].items():
        for obj in OBJECTS[language]:
            delta = _normalize(
                vectors[template.format(obj=obj)]
                - vectors[NEUTRAL[language].format(obj=obj)]
            )
            scores = sorted(
                (float(delta @ direction), phase)
                for phase, direction in directions.items()
            )
            correct += scores[-1][1] == expected
            margins.append(scores[-1][0] - scores[-2][0])
    total = len(HOLDOUT[language]) * len(OBJECTS[language])
    return correct / total, float(np.median(margins))


def _pair_distributions(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        label: np.asarray([float(vectors[left] @ vectors[right]) for left, right in pairs])
        for label, pairs in SIMILARITY_PAIRS.items()
    }


def _teacher_vectors(model: str, texts: list[str]) -> dict[str, np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "--teacher requires sentence-transformers; install the finetune extra"
        ) from exc
    teacher = SentenceTransformer(model)
    encoded = teacher.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return {
        text: np.asarray(vector, dtype=np.float64)
        for text, vector in zip(texts, encoded, strict=True)
    }


def _mconala_rows(path: str) -> list[dict[str, str]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        row for row in raw
        if row.get("rewritten_intent") and row.get("snippet")
    ]


def _codesearchnet_rows(path: str, maximum: int) -> list[dict[str, str]]:
    """Load a stable English docstring-to-code holdout from test parquet."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("CodeSearchNet evaluation requires pandas and pyarrow") from exc
    frame = pd.read_parquet(
        path,
        columns=["func_documentation_string", "func_code_string", "func_code_url"],
    )
    rows: list[dict[str, str]] = []
    seen_queries: set[str] = set()
    for item in frame.to_dict(orient="records"):
        query = " ".join(str(item["func_documentation_string"]).split())[:700].strip()
        code = str(item["func_code_string"])[:700].strip()
        if len(query) < 16 or len(code) < 24 or query.casefold() in seen_queries:
            continue
        seen_queries.add(query.casefold())
        rows.append({"query": query, "passage": code, "id": str(item["func_code_url"])})
    return sorted(
        rows, key=lambda row: hashlib.sha256(row["id"].encode()).digest()
    )[:maximum]


def _m2crb_rows(
    path: str,
    natural_languages: Sequence[str],
    maximum_per_programming_language: int,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Load stable natural-language-to-code groups from the M2CRB holdout."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("M2CRB evaluation requires pandas and pyarrow") from exc
    frame = pd.read_parquet(
        path,
        columns=[
            "identifier",
            "docstring",
            "docstring_summary",
            "function",
            "language",
            "docstring_language",
        ],
    )
    requested = {language.casefold() for language in natural_languages}
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in frame.to_dict(orient="records"):
        natural = str(item["docstring_language"]).casefold()
        programming = str(item["language"]).casefold()
        if natural not in requested:
            continue
        summary = item.get("docstring_summary")
        raw_query = summary if isinstance(summary, str) and summary.strip() else item["docstring"]
        query = " ".join(str(raw_query).split())[:700].strip()
        code = str(item["function"])[:700].strip()
        group = (natural, programming)
        canonical = query.casefold()
        if len(query) < 16 or len(code) < 24 or canonical in seen[group]:
            continue
        seen[group].add(canonical)
        identity = hashlib.sha256(
            f"{natural}\0{programming}\0{item['identifier']}\0{code}".encode()
        ).hexdigest()
        family = hashlib.sha256(code.encode()).hexdigest()[:24]
        groups[group].append({
            "query": query,
            "passage": code,
            "id": identity,
            "family": family,
        })
    return {
        group: sorted(
            rows, key=lambda row: hashlib.sha256(row["id"].encode()).digest()
        )[:maximum_per_programming_language]
        for group, rows in sorted(groups.items())
    }


def _m2crb_split(
    natural: str, programming: str, family: str, seed: int = 17
) -> str:
    payload = f"{seed}\0m2crb_{natural}_{programming}\0{family}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _retrieval_metrics(
    queries: np.ndarray,
    passages: np.ndarray,
) -> dict[str, float]:
    return _metrics_from_ranks(_retrieval_ranks(queries, passages))


def _retrieval_ranks(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    """Return one-based paired-target ranks for a retrieval matrix."""
    order = np.argsort(-(queries @ passages.T), axis=1)
    return np.asarray([
        int(np.flatnonzero(order[index] == index)[0]) + 1
        for index in range(len(order))
    ])


def _metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    """Compute retrieval metrics from one-based ranks."""
    metrics = {
        f"recall@{cutoff}": float(np.mean(ranks <= cutoff))
        for cutoff in (1, 5, 10, 20)
    }
    metrics["mrr"] = float(np.mean(1.0 / ranks))
    metrics["median_rank"] = float(np.median(ranks))
    return metrics


def _paired_bootstrap_deltas(
    baseline_ranks: np.ndarray,
    candidate_ranks: np.ndarray,
    *,
    samples: int,
    seed: int = 17,
) -> dict[str, tuple[float, float, float]]:
    """Return observed delta and paired 95% bootstrap interval per metric."""
    if baseline_ranks.shape != candidate_ranks.shape:
        raise ValueError("paired bootstrap requires equally shaped rank arrays")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(baseline_ranks), size=(samples, len(baseline_ranks)))
    result: dict[str, tuple[float, float, float]] = {}
    for name, baseline_values, candidate_values in (
        *(
            (
                f"recall@{cutoff}",
                (baseline_ranks <= cutoff).astype(np.float64),
                (candidate_ranks <= cutoff).astype(np.float64),
            )
            for cutoff in (1, 5, 10, 20)
        ),
        ("mrr", 1.0 / baseline_ranks, 1.0 / candidate_ranks),
    ):
        observed = float(np.mean(candidate_values - baseline_values))
        bootstrap = np.mean(candidate_values[draws] - baseline_values[draws], axis=1)
        low, high = np.quantile(bootstrap, (0.025, 0.975))
        result[name] = observed, float(low), float(high)
    observed_median = float(np.median(candidate_ranks) - np.median(baseline_ranks))
    median_bootstrap = np.median(candidate_ranks[draws], axis=1) - np.median(
        baseline_ranks[draws], axis=1
    )
    low, high = np.quantile(median_bootstrap, (0.025, 0.975))
    result["median_rank"] = observed_median, float(low), float(high)
    return result


def _print_retrieval(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label}: "
        + ", ".join(
            f"{name}={value:.3f}" if name != "median_rank" else f"{name}={value:.0f}"
            for name, value in metrics.items()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", help="alternative static .npz artifact")
    parser.add_argument(
        "--compare-artifact",
        help="second artifact for paired MCoNaLa bootstrap comparison",
    )
    parser.add_argument(
        "--compare-spanish-adapter",
        help="query-only Spanish adapter used with the comparison/base artifact",
    )
    parser.add_argument(
        "--compare-openai-cache",
        type=Path,
        help="complete local OpenAI teacher cache for paired retrieval comparison",
    )
    parser.add_argument(
        "--export-mconala-jsonl",
        type=Path,
        help="export MCoNaLa rows for the resumable teacher collector",
    )
    parser.add_argument(
        "--export-codesearchnet-jsonl",
        type=Path,
        help="export the stable CodeSearchNet sample for teacher collection",
    )
    parser.add_argument(
        "--export-m2crb-jsonl",
        type=Path,
        help="export stable M2CRB groups for teacher collection",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument(
        "--comparison-query-only",
        action="store_true",
        help="apply the comparison artifact only to queries, retaining primary passages",
    )
    parser.add_argument("--teacher", help="optional SentenceTransformer model name")
    parser.add_argument(
        "--mconala",
        help="path to MCoNaLa es_test.json for external Spanish retrieval evaluation",
    )
    parser.add_argument(
        "--codesearchnet",
        help="path to CodeSearchNet test parquet for English regression evaluation",
    )
    parser.add_argument("--codesearchnet-records", type=int, default=1_000)
    parser.add_argument("--m2crb", help="path to the external M2CRB test parquet")
    parser.add_argument(
        "--m2crb-language",
        action="append",
        default=[],
        help="natural language to evaluate; repeatable (default: es)",
    )
    parser.add_argument("--m2crb-records", type=int, default=500)
    parser.add_argument(
        "--m2crb-split",
        choices=("all", "train", "validation", "test"),
        default="all",
        help="evaluate a deterministic code-family split",
    )
    parser.add_argument("--m2crb-seed", type=int, default=17)
    args = parser.parse_args()

    if args.export_mconala_jsonl:
        if not args.mconala:
            raise SystemExit("--export-mconala-jsonl requires --mconala")
        export_rows = _mconala_rows(args.mconala)
        args.export_mconala_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.export_mconala_jsonl.write_text("\n".join(
            json.dumps({
                "id": f"mconala:{index}",
                "text": row["rewritten_intent"],
                "parallel_text": row["snippet"],
            }, ensure_ascii=False)
            for index, row in enumerate(export_rows)
        ) + "\n", encoding="utf-8")
    if args.export_codesearchnet_jsonl:
        if not args.codesearchnet:
            raise SystemExit(
                "--export-codesearchnet-jsonl requires --codesearchnet"
            )
        export_rows = _codesearchnet_rows(
            args.codesearchnet, args.codesearchnet_records
        )
        args.export_codesearchnet_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.export_codesearchnet_jsonl.write_text("\n".join(
            json.dumps({
                "id": f"codesearchnet:{row['id']}",
                "text": row["query"],
                "parallel_text": row["passage"],
            }, ensure_ascii=False)
            for row in export_rows
        ) + "\n", encoding="utf-8")
    m2crb_groups = (
        _m2crb_rows(
            args.m2crb,
            args.m2crb_language or ["es"],
            args.m2crb_records,
        )
        if args.m2crb else {}
    )
    if args.m2crb_split != "all":
        m2crb_groups = {
            (natural, programming): [
                row for row in rows
                if _m2crb_split(
                    natural, programming, row["family"], args.m2crb_seed
                ) == args.m2crb_split
            ]
            for (natural, programming), rows in m2crb_groups.items()
        }
    if args.export_m2crb_jsonl:
        if not args.m2crb:
            raise SystemExit("--export-m2crb-jsonl requires --m2crb")
        args.export_m2crb_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.export_m2crb_jsonl.write_text("\n".join(
            json.dumps({
                "id": f"m2crb:{natural}:{programming}:{row['id']}",
                "source": f"m2crb_{natural}_{programming}",
                "path": row["family"],
                "language": natural,
                "programming_language": programming,
                "text": row["query"],
                "parallel_text": row["passage"],
            }, ensure_ascii=False)
            for (natural, programming), rows in m2crb_groups.items()
            for row in rows
        ) + "\n", encoding="utf-8")

    embedder = StaticQwen3Embedder(args.artifact)
    texts = _all_texts()
    vectors = {
        text: np.asarray(vector, dtype=np.float64)
        for text, vector in zip(texts, embedder(texts), strict=True)
    }

    directions = {language: _directions(language, vectors) for language in ("en", "es")}
    for language in ("en", "es"):
        accuracy, margin = _action_accuracy(language, vectors, directions[language])
        print(
            f"{language} held-out action accuracy={accuracy:.1%} "
            f"median_margin={margin:.3f}"
        )

    print("action direction alignment, English versus Spanish")
    for phase in TRAIN["en"]:
        score = float(directions["en"][phase] @ directions["es"][phase])
        print(f"  {phase:11s} cosine={score:.3f}")

    print("similarity distributions")
    for label, scores in _pair_distributions(vectors).items():
        print(
            f"  {label:16s} min={scores.min():.3f} "
            f"median={np.median(scores):.3f} max={scores.max():.3f}"
        )

    neutral = vectors["callback registry"]
    print("negation probes")
    for text in (
        "Do not modify the callback registry; only inspect it",
        "Modify the callback registry; do not merely inspect it",
    ):
        delta = _normalize(vectors[text] - neutral)
        ranked = sorted(
            (float(delta @ direction), phase)
            for phase, direction in directions["en"].items()
        )
        print(f"  {text}")
        print("    " + ", ".join(f"{phase}={score:.3f}" for score, phase in ranked[::-1]))

    speed_batch = [
        f"Inspect symbol {index} in src/module_{index % 31}.py"
        for index in range(1_000)
    ]
    embedder(speed_batch[:10])
    started = perf_counter()
    embedder(speed_batch)
    elapsed = perf_counter() - started
    print(
        f"batch throughput={len(speed_batch) / elapsed:.0f} texts/s "
        f"({elapsed * 1e6 / len(speed_batch):.1f} us/text, n={len(speed_batch)})"
    )

    mconala_rows = _mconala_rows(args.mconala) if args.mconala else []
    if mconala_rows:
        query_texts = [row["rewritten_intent"] for row in mconala_rows]
        passage_texts = [row["snippet"] for row in mconala_rows]
        static_queries = np.asarray(embedder.embed_queries(query_texts))
        static_passages = np.asarray(embedder.embed_passages(passage_texts))
        _print_retrieval(
            "MCoNaLa Spanish static",
            _retrieval_metrics(static_queries, static_passages),
        )
        if args.compare_artifact or args.compare_spanish_adapter or args.compare_openai_cache:
            comparison = (
                CachedTeacherEmbedder(args.compare_openai_cache)
                if args.compare_openai_cache
                else StaticQwen3Embedder(
                    args.compare_artifact or args.artifact,
                    spanish_adapter_path=args.compare_spanish_adapter,
                )
            )
            comparison_queries = np.asarray(comparison.embed_queries(query_texts))
            comparison_passages = (
                static_passages
                if args.comparison_query_only
                else np.asarray(comparison.embed_passages(passage_texts))
            )
            _print_retrieval(
                "MCoNaLa Spanish comparison",
                _retrieval_metrics(comparison_queries, comparison_passages),
            )
            deltas = _paired_bootstrap_deltas(
                _retrieval_ranks(static_queries, static_passages),
                _retrieval_ranks(comparison_queries, comparison_passages),
                samples=args.bootstrap_samples,
            )
            print("comparison minus primary, paired bootstrap 95% intervals")
            for name, (observed, low, high) in deltas.items():
                print(f"  {name:11s} delta={observed:+.3f} CI=[{low:+.3f}, {high:+.3f}]")

    codesearchnet_rows = (
        _codesearchnet_rows(args.codesearchnet, args.codesearchnet_records)
        if args.codesearchnet else []
    )
    if codesearchnet_rows:
        query_texts = [row["query"] for row in codesearchnet_rows]
        passage_texts = [row["passage"] for row in codesearchnet_rows]
        primary_queries = np.asarray(embedder.embed_queries(query_texts))
        primary_passages = np.asarray(embedder.embed_passages(passage_texts))
        _print_retrieval(
            "CodeSearchNet English static",
            _retrieval_metrics(primary_queries, primary_passages),
        )
        if args.compare_artifact or args.compare_spanish_adapter or args.compare_openai_cache:
            comparison = (
                CachedTeacherEmbedder(args.compare_openai_cache)
                if args.compare_openai_cache
                else StaticQwen3Embedder(
                    args.compare_artifact or args.artifact,
                    spanish_adapter_path=args.compare_spanish_adapter,
                )
            )
            comparison_queries = np.asarray(comparison.embed_queries(query_texts))
            comparison_passages = (
                primary_passages
                if args.comparison_query_only
                else np.asarray(comparison.embed_passages(passage_texts))
            )
            _print_retrieval(
                "CodeSearchNet English comparison",
                _retrieval_metrics(comparison_queries, comparison_passages),
            )
            deltas = _paired_bootstrap_deltas(
                _retrieval_ranks(primary_queries, primary_passages),
                _retrieval_ranks(comparison_queries, comparison_passages),
                samples=args.bootstrap_samples,
            )
            print("comparison minus primary, English paired bootstrap 95% intervals")
            for name, (observed, low, high) in deltas.items():
                print(f"  {name:11s} delta={observed:+.3f} CI=[{low:+.3f}, {high:+.3f}]")

    for (natural, programming), rows in m2crb_groups.items():
        query_texts = [row["query"] for row in rows]
        passage_texts = [row["passage"] for row in rows]
        primary_queries = np.asarray(embedder.embed_queries(query_texts))
        primary_passages = np.asarray(embedder.embed_passages(passage_texts))
        label = f"M2CRB {natural}->{programming} static"
        _print_retrieval(label, _retrieval_metrics(primary_queries, primary_passages))
        if args.compare_artifact or args.compare_spanish_adapter or args.compare_openai_cache:
            comparison = (
                CachedTeacherEmbedder(args.compare_openai_cache)
                if args.compare_openai_cache
                else StaticQwen3Embedder(
                    args.compare_artifact or args.artifact,
                    spanish_adapter_path=args.compare_spanish_adapter,
                )
            )
            comparison_queries = np.asarray(comparison.embed_queries(query_texts))
            comparison_passages = (
                primary_passages
                if args.comparison_query_only
                else np.asarray(comparison.embed_passages(passage_texts))
            )
            _print_retrieval(
                f"M2CRB {natural}->{programming} comparison",
                _retrieval_metrics(comparison_queries, comparison_passages),
            )
            deltas = _paired_bootstrap_deltas(
                _retrieval_ranks(primary_queries, primary_passages),
                _retrieval_ranks(comparison_queries, comparison_passages),
                samples=args.bootstrap_samples,
            )
            print(
                f"comparison minus primary, M2CRB {natural}->{programming} "
                "paired bootstrap 95% intervals"
            )
            for name, (observed, low, high) in deltas.items():
                print(f"  {name:11s} delta={observed:+.3f} CI=[{low:+.3f}, {high:+.3f}]")

    if not args.teacher:
        return
    teacher_texts = texts + [
        text
        for row in mconala_rows
        for text in (row["rewritten_intent"], row["snippet"])
    ]
    teacher = _teacher_vectors(args.teacher, teacher_texts)
    spanish_texts = _language_texts("es")
    alignment = np.asarray([vectors[text] @ teacher[text] for text in spanish_texts])
    teacher_es = _directions("es", teacher)
    teacher_accuracy, _ = _action_accuracy("es", teacher, teacher_es)
    print(
        f"static/teacher Spanish cosine mean={alignment.mean():.3f} "
        f"p10={np.quantile(alignment, 0.1):.3f}"
    )
    print(f"teacher Spanish held-out action accuracy={teacher_accuracy:.1%}")

    embedder._load()
    assert embedder._projection is not None
    projection = embedder._projection.astype(np.float64)
    gram_inverse = np.linalg.inv(projection @ projection.T)
    projected: dict[str, np.ndarray] = {}
    retained: list[float] = []
    for text, target in teacher.items():
        latent = target @ projection.T @ gram_inverse
        vector = _normalize(latent @ projection)
        projected[text] = vector
        retained.append(float(target @ vector))
    projected_directions = _directions("es", projected)
    projected_accuracy, _ = _action_accuracy("es", projected, projected_directions)
    print(f"teacher projection retained cosine mean={np.mean(retained):.3f}")
    print(f"projected-teacher Spanish action accuracy={projected_accuracy:.1%}")

    if mconala_rows:
        teacher_queries = np.asarray([
            teacher[row["rewritten_intent"]] for row in mconala_rows
        ])
        teacher_passages = np.asarray([teacher[row["snippet"]] for row in mconala_rows])
        _print_retrieval(
            "MCoNaLa Spanish teacher",
            _retrieval_metrics(teacher_queries, teacher_passages),
        )


if __name__ == "__main__":
    main()
