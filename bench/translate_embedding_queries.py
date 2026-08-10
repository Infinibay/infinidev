"""Translate balanced embedding queries without exposing code tokens to MT.

The translator is intentionally narrow: it derives Spanish retrieval queries
from English CommitPack instructions while preserving identifiers, paths,
command-line flags, issue references, and quoted code byte-for-byte.  Records
are selected with a deterministic per-programming-language reservoir, and the
output is append-only/resumable so a model download or long CPU run is safe to
restart.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Iterator

try:
    from bench.language_id import detect_target_language
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    from language_id import detect_target_language


MODEL = "Helsinki-NLP/opus-mt-en-es"
REVISION = "5bc4493d463cf000c1f0b50f8d56886a392ed4ab"
TARGET_LANGUAGE = "es"
EXCLUDED_PROGRAMMING_LANGUAGES = {"zig"}

# Ordered from broad quoted spans to individual technical tokens. Plain words
# remain visible to the translator; anything matched here is spliced back in
# after translation rather than trusting SentencePiece to preserve it.
_PROTECTED = re.compile(
    r"(`[^`\n]+`|'[^'\n]+'|\"[^\"\n]+\"|"
    r"https?://\S+|(?:[A-Za-z]:)?(?:[./~][\w.@+-]+)+(?:/[-\w.@+]*)?|"
    r"(?:[\w.@+-]+/)+[-\w.@+]+|"
    r"--?[A-Za-z][\w-]*|#[0-9]+|"
    r"\b(?:[A-Z][A-Z0-9_]{1,}|[a-z]+_[A-Za-z0-9_]+|"
    r"[A-Za-z]+[A-Z][A-Za-z0-9]*|[A-Za-z_$][\w$]*\.[A-Za-z_$][\w.$]*)\b)"
)


@dataclass(frozen=True)
class ProtectedText:
    """Natural-language chunks plus immutable separators between them."""

    chunks: tuple[str, ...]
    protected: tuple[str, ...]

    def rebuild(self, translated_chunks: Iterable[str]) -> str:
        values = list(translated_chunks)
        if len(values) != len(self.chunks):
            raise ValueError("translated chunk count does not match source")
        output: list[str] = []
        for index, value in enumerate(values):
            output.append(value)
            if index < len(self.protected):
                output.append(self.protected[index])
        return "".join(output).strip()


def protect_technical_text(text: str) -> ProtectedText:
    """Split text so matched technical spans never pass through the MT model."""
    chunks: list[str] = []
    protected: list[str] = []
    cursor = 0
    for match in _PROTECTED.finditer(text):
        chunks.append(text[cursor:match.start()])
        protected.append(match.group(0))
        cursor = match.end()
    chunks.append(text[cursor:])
    return ProtectedText(tuple(chunks), tuple(protected))


def _priority(seed: int, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{identity}".encode()).digest(), "big")


def select_records(
    rows: Iterable[dict[str, Any]],
    *,
    per_language: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Hash-sample eligible training records independently by code language."""
    if per_language <= 0:
        raise ValueError("per_language must be positive")
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        programming_language = str(
            row.get("programming_language", row.get("parallel_language", ""))
        ).casefold()
        if not programming_language or programming_language in EXCLUDED_PROGRAMMING_LANGUAGES:
            continue
        if row.get("split", "train") != "train" or row.get("language") != "en":
            continue
        if float(row.get("language_confidence", 1.0)) < 0.75:
            continue
        text = row.get("text")
        if not isinstance(text, str) or len(text.strip()) < 10:
            continue
        identity = str(row.get("id", ""))
        if not identity:
            continue
        priority = _priority(seed, identity)
        row = dict(row)
        row.setdefault("programming_language", programming_language)
        entry = (-priority, identity, row)
        heap = heaps[programming_language]
        if len(heap) < per_language:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected = [entry[2] for heap in heaps.values() for entry in heap]
    return sorted(selected, key=lambda row: (str(row["programming_language"]), str(row["id"])))


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def translated_record(row: dict[str, Any], translated: str) -> dict[str, Any] | None:
    """Create one provenance-rich Spanish derivative after strict validation."""
    translated = " ".join(translated.split()).strip()
    language, confidence = detect_target_language(translated)
    if language != TARGET_LANGUAGE or confidence < 0.70:
        return None
    source_id = str(row["id"])
    result = dict(row)
    result.update({
        "id": hashlib.sha256(f"opus-mt-en-es\0{source_id}\0{translated}".encode()).hexdigest()[:24],
        "source": f"{row.get('source', 'unknown')}_opus_mt_es",
        "source_id": source_id,
        "source_text": row["text"],
        "text": translated,
        "language": TARGET_LANGUAGE,
        "language_confidence": confidence,
        "translation_model": MODEL,
        "translation_revision": REVISION,
        "translation_method": "protected_span_local_mt",
        "split": "train",
    })
    return result


def _load_translator(
    batch_size: int,
    *,
    local_files_only: bool = False,
    ct2_model: Path | None = None,
) -> Callable[[list[str]], list[str]]:
    try:
        from transformers import MarianTokenizer
    except ImportError as exc:
        raise SystemExit("translation requires the finetune optional dependencies") from exc

    tokenizer = MarianTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=local_files_only
    )
    if ct2_model is not None:
        try:
            import ctranslate2
        except ImportError as exc:
            raise SystemExit("--ct2-model requires ctranslate2") from exc
        import os

        translator = ctranslate2.Translator(
            str(ct2_model),
            device="cpu",
            compute_type="int8",
            intra_threads=max(1, min(8, os.cpu_count() or 1)),
        )

        def translate_ct2(texts: list[str]) -> list[str]:
            results: list[str] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                tokens = [
                    tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
                    for text in batch
                ]
                generated = translator.translate_batch(
                    tokens,
                    beam_size=1,
                    max_decoding_length=256,
                )
                for result in generated:
                    token_ids = tokenizer.convert_tokens_to_ids(result.hypotheses[0])
                    results.append(
                        tokenizer.decode(token_ids, skip_special_tokens=True)
                    )
            return results

        return translate_ct2

    try:
        import torch
        from transformers import MarianMTModel
    except ImportError as exc:
        raise SystemExit("translation requires torch and transformers") from exc
    model = MarianMTModel.from_pretrained(
        MODEL, revision=REVISION, local_files_only=local_files_only
    )
    model.eval()

    def translate(texts: list[str]) -> list[str]:
        results: list[str] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=256, do_sample=False)
            results.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return results

    return translate


def translate_rows(
    rows: list[dict[str, Any]],
    translate: Callable[[list[str]], list[str]],
) -> Iterator[dict[str, Any]]:
    """Translate natural chunks in bulk and reconstruct each source record."""
    protected_rows = [protect_technical_text(str(row["text"])) for row in rows]
    translatable: list[tuple[str, str, str]] = []
    for item in protected_rows:
        for chunk in item.chunks:
            match = re.fullmatch(r"(\s*)(.*?)(\s*)", chunk, re.DOTALL)
            assert match is not None
            if match.group(2):
                translatable.append((match.group(1), match.group(2), match.group(3)))
    translated_chunks = iter(translate([core for _, core, _ in translatable]))
    translated_with_spacing = iter(
        leading + translated + trailing
        for (leading, _, trailing), translated in zip(
            translatable, translated_chunks, strict=True
        )
    )
    for row, protected in zip(rows, protected_rows, strict=True):
        rebuilt: list[str] = []
        for chunk in protected.chunks:
            rebuilt.append(next(translated_with_spacing) if chunk.strip() else chunk)
        result = translated_record(row, protected.rebuild(rebuilt))
        if result is not None:
            yield result
    try:
        next(translated_with_spacing)
    except StopIteration:
        return
    raise ValueError("translator returned too many chunks")


def build(args: argparse.Namespace) -> dict[str, Any]:
    selected = select_records(_rows(args.input), per_language=args.per_language, seed=args.seed)
    selected_counts = Counter(str(row["programming_language"]) for row in selected)
    if args.dry_run:
        return {
            "selected": len(selected),
            "by_programming_language": dict(sorted(selected_counts.items())),
        }

    existing_source_ids: set[str] = set()
    if args.output.exists():
        existing_source_ids = {str(row.get("source_id")) for row in _rows(args.output)}
    pending = [row for row in selected if str(row["id"]) not in existing_source_ids]
    translate = _load_translator(
        args.batch_size,
        local_files_only=args.local_files_only,
        ct2_model=args.ct2_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rejected = 0
    with args.output.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), args.write_batch):
            batch = pending[start:start + args.write_batch]
            results = list(translate_rows(batch, translate))
            for result in results:
                result["translation_backend"] = (
                    "ctranslate2-int8" if args.ct2_model else "transformers-float32"
                )
            rejected += len(batch) - len(results)
            for result in results:
                output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
            output.flush()
            print(json.dumps({
                "processed": min(start + len(batch), len(pending)),
                "pending": len(pending),
                "written": written,
                "rejected": rejected,
            }), flush=True)
    return {
        "selected": len(selected),
        "already_present": len(selected) - len(pending),
        "written": written,
        "rejected": rejected,
        "by_programming_language": dict(sorted(selected_counts.items())),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-language", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--write-batch", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--ct2-model", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.write_batch <= 0:
        parser.error("batch sizes must be positive")
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
