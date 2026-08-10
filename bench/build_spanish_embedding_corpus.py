"""Build an auditable Spanish programming corpus for embedding calibration.

The output is metadata-rich JSONL, not a trained artifact.  Permissive PSF-
licensed Python documentation is enabled by default.  ShareAlike sources need
an explicit flag so an experiment cannot silently change the distribution or
the licensing obligations of a future model artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Iterator


_SPACE_RE = re.compile(r"[ \t\f\v]+")
_RST_ROLE_RE = re.compile(r":(?:[a-zA-Z0-9_-]+:)?[a-zA-Z0-9_-]+:`([^`]+)`")
_MARKDOWN_LINK_RE = re.compile(r"!?\[([^]]*)]\([^)]+\)")
_MARKDOWN_REF_RE = re.compile(r"!?\[([^]]*)]\[[^]]*]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ¿¡`])")

_SOURCES = {
    "python_docs_es": {
        "license": "PSF-2.0",
        "license_class": "permissive",
        "url": "https://github.com/python/python-docs-es",
        "format": "gettext-po",
    },
    "django_girls_es": {
        "license": "CC-BY-SA-4.0",
        "license_class": "sharealike",
        "url": "https://github.com/DjangoGirls/tutorial",
        "format": "markdown",
    },
}


def _clean_markup(text: str) -> str:
    """Remove presentation markup while retaining identifiers and prose."""
    text = _RST_ROLE_RE.sub(r"\1", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_REF_RE.sub(r"\1", text)
    text = text.replace("``", "`")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith((".. _", ".. index::")):
            continue
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        stripped = re.sub(r"^[-*+]\s+", "", stripped)
        stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
        lines.append(_SPACE_RE.sub(" ", stripped))
    return "\n".join(lines).strip()


def _split_long(text: str, max_chars: int) -> Iterator[str]:
    sentences = _SENTENCE_RE.split(text)
    chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if chunk:
                yield chunk
                chunk = ""
            for start in range(0, len(sentence), max_chars):
                yield sentence[start:start + max_chars].strip()
            continue
        candidate = f"{chunk} {sentence}".strip()
        if chunk and len(candidate) > max_chars:
            yield chunk
            chunk = sentence
        else:
            chunk = candidate
    if chunk:
        yield chunk


def _chunks(text: str, *, max_chars: int, min_chars: int = 32) -> Iterator[str]:
    """Create short semantic units without merging unrelated sections."""
    cleaned = _clean_markup(text)
    for paragraph in re.split(r"\n\s*\n", cleaned):
        paragraph = " ".join(part.strip() for part in paragraph.splitlines()).strip()
        if len(paragraph) < min_chars:
            continue
        yield from (
            chunk for chunk in _split_long(paragraph, max_chars)
            if len(chunk) >= min_chars
        )


def _code_chunks(text: str, max_chars: int) -> Iterator[str]:
    """Split code only at line boundaries and retain every non-empty line."""
    chunk: list[str] = []
    size = 0
    for line in text.splitlines():
        if chunk and size + len(line) + 1 > max_chars:
            yield "\n".join(chunk).strip()
            chunk = []
            size = 0
        if len(line) > max_chars:
            if chunk:
                yield "\n".join(chunk).strip()
                chunk = []
                size = 0
            yield line[:max_chars].strip()
            continue
        chunk.append(line)
        size += len(line) + 1
    if chunk:
        yield "\n".join(chunk).strip()


def _record(
    *,
    source: str,
    relative_path: str,
    ordinal: int,
    kind: str,
    text: str,
    parallel_text: str | None = None,
) -> dict[str, object]:
    metadata = _SOURCES[source]
    identity = f"{source}\0{relative_path}\0{ordinal}\0{text}".encode()
    record: dict[str, object] = {
        "id": hashlib.sha256(identity).hexdigest()[:24],
        "source": source,
        "source_url": metadata["url"],
        "license": metadata["license"],
        "license_class": metadata["license_class"],
        "path": relative_path,
        "kind": kind,
        "language": "es",
        "text": text,
        "characters": len(text),
        "words": len(text.split()),
    }
    if parallel_text:
        record["parallel_language"] = "en"
        record["parallel_text"] = parallel_text
    return record


def iter_python_docs(root: Path, max_chars: int) -> Iterator[dict[str, object]]:
    """Yield translated messages from a python-docs-es checkout."""
    try:
        from babel.messages.pofile import read_po
    except ImportError as exc:
        raise SystemExit("python-docs-es extraction requires Babel") from exc

    for path in sorted(root.rglob("*.po")):
        relative = path.relative_to(root).as_posix()
        with path.open("r", encoding="utf-8") as source_file:
            catalog = read_po(source_file, locale="es")
        ordinal = 0
        for message in catalog:
            if not message.id or "fuzzy" in message.flags:
                continue
            translated = message.string
            values = translated if isinstance(translated, tuple) else (translated,)
            originals = message.id if isinstance(message.id, tuple) else (message.id,)
            for value_index, value in enumerate(values):
                if not value:
                    continue
                spanish_chunks = list(_chunks(value, max_chars=max_chars))
                original = originals[min(value_index, len(originals) - 1)]
                english_chunks = list(_chunks(str(original), max_chars=max_chars))
                aligned = (
                    english_chunks
                    if len(spanish_chunks) == len(english_chunks)
                    else [None] * len(spanish_chunks)
                )
                for text, parallel_text in zip(spanish_chunks, aligned, strict=True):
                    yield _record(
                        source="python_docs_es",
                        relative_path=relative,
                        ordinal=ordinal,
                        kind="technical_prose",
                        text=text,
                        parallel_text=parallel_text,
                    )
                    ordinal += 1


def _markdown_sections(text: str) -> Iterator[tuple[str, str, str]]:
    """Yield (heading, kind, content) while keeping fenced code separate."""
    heading = ""
    prose: list[str] = []
    code: list[str] = []
    in_fence = False

    def flush_prose() -> Iterator[tuple[str, str, str]]:
        nonlocal prose
        if prose:
            yield heading, "technical_prose", "\n".join(prose)
            prose = []

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if in_fence:
                if code:
                    yield heading, "code", "\n".join(code)
                code = []
            else:
                yield from flush_prose()
            in_fence = not in_fence
            continue
        if in_fence:
            code.append(line)
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            yield from flush_prose()
            heading = _clean_markup(match.group(1))
            continue
        prose.append(line)
    if code:
        yield heading, "code", "\n".join(code)
    yield from flush_prose()


def iter_django_girls(root: Path, max_chars: int) -> Iterator[dict[str, object]]:
    """Yield Spanish tutorial prose and contextualized code blocks."""
    spanish_root = root / "es" if (root / "es").is_dir() else root
    for path in sorted(spanish_root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        raw = path.read_text(encoding="utf-8")
        ordinal = 0
        for heading, kind, content in _markdown_sections(raw):
            if kind == "code":
                clean_code = content.strip()
                if len(clean_code) < 8:
                    continue
                candidates = (
                    f"{heading}\n{chunk}".strip() if heading else chunk
                    for chunk in _code_chunks(
                        clean_code, max(64, max_chars - len(heading) - 1)
                    )
                )
            else:
                candidates = (
                    f"{heading}: {chunk}" if heading else chunk
                    for chunk in _chunks(
                        content, max_chars=max(64, max_chars - len(heading) - 2)
                    )
                )
            for text in candidates:
                yield _record(
                    source="django_girls_es",
                    relative_path=relative,
                    ordinal=ordinal,
                    kind=kind,
                    text=text,
                )
                ordinal += 1


def _deduplicate(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    result = []
    for record in records:
        canonical = _SPACE_RE.sub(" ", str(record["text"]).casefold()).strip()
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        result.append(record)
    return result


def _select(records: list[dict[str, object]], maximum: int | None) -> list[dict[str, object]]:
    """Select a reproducible hash-uniform subset without order bias."""
    if maximum is None or maximum >= len(records):
        return records
    selected = sorted(records, key=lambda record: str(record["id"]))[:maximum]
    return sorted(selected, key=lambda record: (str(record["source"]), str(record["id"])))


def build(args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if args.python_docs:
        records.extend(iter_python_docs(args.python_docs, args.max_chars))
    if args.django_girls:
        if not args.include_sharealike:
            raise SystemExit("--django-girls requires --include-sharealike")
        records.extend(iter_django_girls(args.django_girls, args.max_chars))
    if not records:
        raise SystemExit("provide --python-docs and/or --django-girls")
    return _select(_deduplicate(records), args.max_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-docs", type=Path, help="python-docs-es checkout")
    parser.add_argument("--django-girls", type=Path, help="Django Girls tutorial checkout")
    parser.add_argument(
        "--include-sharealike",
        action="store_true",
        help="explicitly admit CC BY-SA sources for research output",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=700)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    if args.max_chars < 128:
        parser.error("--max-chars must be at least 128")

    records = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    sources = Counter(str(record["source"]) for record in records)
    kinds = Counter(str(record["kind"]) for record in records)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "output": str(args.output),
        "records": len(records),
        "sources": dict(sorted(sources.items())),
        "kinds": dict(sorted(kinds.items())),
        "sha256": digest,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
