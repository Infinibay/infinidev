"""Build an auditable multilingual code-retrieval corpus from The Stack Smol XL.

The builder streams only requested programming-language shards, retains exact
repository/license provenance, and balances output by language.  Tree-sitter
parsers already shipped by Infinidev identify symbols; small structural
extractors cover assembly, Perl, and PowerShell where no grammar is bundled.
Generated files stay outside Git and remain subject to their upstream licenses.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

try:
    from bench.language_id import detect_target_language
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    from language_id import detect_target_language


DATASET = "bigcode/the-stack-smol-xl"
REVISION = "e782ebf35c7e4cafccb08ca680b0a76706533067"
DEFAULT_LANGUAGES = (
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c-sharp",
    "go",
    "rust",
    "assembly",
    "python",
    "ruby",
    "perl",
    "shell",
    "powershell",
    "php",
    "kotlin",
    "dart",
    "lua",
    "sql",
    "zig",
)
CANONICAL_LANGUAGE = {
    "c++": "cpp",
    "c-sharp": "csharp",
    "shell": "bash",
}
DEFAULT_EXTENSION = {
    "assembly": ".s",
    "bash": ".sh",
    "c": ".c",
    "cpp": ".cpp",
    "csharp": ".cs",
    "dart": ".dart",
    "go": ".go",
    "java": ".java",
    "javascript": ".js",
    "kotlin": ".kt",
    "lua": ".lua",
    "perl": ".pl",
    "php": ".php",
    "powershell": ".ps1",
    "python": ".py",
    "ruby": ".rb",
    "rust": ".rs",
    "sql": ".sql",
    "typescript": ".ts",
    "zig": ".zig",
}
PERMISSIVE_LICENSES = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSL-1.0",
    "CC0-1.0",
    "ISC",
    "MIT",
    "MIT-0",
    "Python-2.0",
    "Unlicense",
    "Zlib",
}
_DECLARATION = re.compile(
    r"^\s*(?:pub\s+|public\s+|private\s+|protected\s+|static\s+|async\s+)*"
    r"(?:function\s+|fn\s+|func\s+|def\s+|sub\s+|class\s+|interface\s+)?"
    r"([A-Za-z_$][\w.$-]*)\s*(?:\([^;]*\)|:)",
)
_ASSEMBLY_LABEL = re.compile(r"^\s*([A-Za-z_.$?][\w.$?@]*):(?:\s|$)")
_PERL_SUB = re.compile(r"^\s*sub\s+([A-Za-z_]\w*)\b")
_POWERSHELL_FUNCTION = re.compile(
    r"^\s*(?:function|filter)\s+(?:global:|script:|local:|private:)?"
    r"([A-Za-z_][\w-]*)\b",
    re.IGNORECASE,
)
_COMMENT_PREFIXES = {
    "assembly": (";", "#", "@", "//", "/*", "*", "*/"),
    "bash": ("#",),
    "lua": ("--",),
    "perl": ("#",),
    "powershell": ("#", "<#", "#>"),
    "python": ("#",),
    "ruby": ("#",),
    "sql": ("--", "/*", "*", "*/"),
}
_C_STYLE_COMMENTS = ("//", "/*", "*", "*/")


def _canonical(dataset_language: str) -> str:
    return CANONICAL_LANGUAGE.get(dataset_language, dataset_language)


def _clean_comment(text: str, maximum: int) -> str:
    lines = []
    for line in text.splitlines():
        clean = re.sub(r"^\s*(?://+|#+|;+|--+|/\*+|\*+|\*/|@+)\s?", "", line)
        clean = re.sub(r"\s*\*/\s*$", "", clean)
        clean = re.sub(r"</?(?:pre|p|br)\s*/?>", " ", clean, flags=re.IGNORECASE)
        clean = " ".join(clean.split())
        if clean and not clean.startswith(("SPDX-", "Copyright ")):
            lines.append(clean)
    return " ".join(lines)[:maximum].strip()


def _preceding_comment(
    lines: list[str], line_start: int, maximum: int, language: str
) -> str:
    index = max(0, line_start - 2)
    collected: list[str] = []
    blank_budget = 1
    while index >= 0 and len(collected) < 24:
        stripped = lines[index].strip()
        if not stripped and blank_budget:
            blank_budget -= 1
            index -= 1
            continue
        prefixes = _COMMENT_PREFIXES.get(language, _C_STYLE_COMMENTS)
        if stripped.startswith(prefixes):
            collected.append(lines[index])
            index -= 1
            continue
        break
    return _clean_comment("\n".join(reversed(collected)), maximum)


def _assembly_dialect(content: str) -> str:
    lowered = f" {content.casefold()} "
    x86 = sum(lowered.count(token) for token in (
        " eax", " ebx", " rax", " rsp", " push ", " pop ", " xmm", " ymm",
    ))
    arm = sum(lowered.count(token) for token in (
        " x0", " w0", " r0", " ldr ", " str ", " bl ", " bx ", " vldr ",
    ))
    if x86 > arm * 1.5:
        return "x86"
    if arm > x86 * 1.5:
        return "arm"
    return "unknown"


def _fallback_symbols(
    language: str, lines: list[str]
) -> Iterator[tuple[str, str, int, int, str]]:
    """Yield (kind, name, start, end, doc) without guessing semantics."""
    if language == "assembly":
        matcher = _ASSEMBLY_LABEL
        kind = "label"
    elif language == "perl":
        matcher = _PERL_SUB
        kind = "function"
    elif language == "powershell":
        matcher = _POWERSHELL_FUNCTION
        kind = "function"
    else:
        matcher = _DECLARATION
        kind = "symbol"
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines, 1):
        match = matcher.match(line)
        if match:
            name = match.group(1)
            if language == "assembly" and name.startswith(".L"):
                continue
            starts.append((index, name))
    for position, (start, name) in enumerate(starts):
        next_start = starts[position + 1][0] if position + 1 < len(starts) else len(lines) + 1
        end = min(len(lines), max(start, next_start - 1), start + 119)
        yield kind, name, start, end, ""


def _tree_sitter_symbols(
    language: str, path: str, content: str
) -> Iterator[tuple[str, str, int, int, str]]:
    try:
        from infinidev.code_intel.parsers import get_parser
        from infinidev.code_intel.syntax_check import _load_parser

        parser = _load_parser(language)
        extractor = get_parser(language)
        if parser is None or extractor is None:
            return
        source = content.encode("utf-8")
        tree = parser.parse(source)
        for symbol in extractor.extract_symbols(tree, source, path):
            end = symbol.line_end or symbol.line_start
            yield (
                symbol.kind.value,
                symbol.qualified_name or symbol.name,
                symbol.line_start,
                end,
                symbol.docstring,
            )
    except (ImportError, RuntimeError, ValueError):
        return


def _detect_natural_language(text: str, generated: bool) -> tuple[str, float]:
    if generated:
        return "en", 1.0
    return detect_target_language(text)


def _provenance(row: dict[str, Any]) -> dict[str, Any] | None:
    for prefix in ("max_stars", "max_forks", "max_issues"):
        repository = row.get(f"{prefix}_repo_name")
        path = row.get(f"{prefix}_repo_path")
        revision = row.get(f"{prefix}_repo_head_hexsha")
        licenses = row.get(f"{prefix}_repo_licenses") or []
        licenses = [str(value) for value in licenses]
        if repository and path and revision and set(licenses) & PERMISSIVE_LICENSES:
            return {
                "repository": str(repository),
                "path": str(path),
                "revision": str(revision),
                "licenses": sorted(set(licenses) & PERMISSIVE_LICENSES),
            }
    return None


def records_from_file(
    row: dict[str, Any],
    dataset_language: str,
    *,
    max_query_chars: int,
    max_code_chars: int,
) -> Iterator[dict[str, Any]]:
    """Extract auditable text/code pairs from one source file."""
    provenance = _provenance(row)
    content = row.get("content")
    if provenance is None or not isinstance(content, str) or not content.strip():
        return
    language = _canonical(dataset_language)
    path = provenance["path"]
    if not Path(path).suffix:
        path += DEFAULT_EXTENSION.get(language, "")
    lines = content.splitlines()
    symbols = list(_tree_sitter_symbols(language, path, content))
    if not symbols:
        symbols = list(_fallback_symbols(language, lines))
    dialect = _assembly_dialect(content) if language == "assembly" else None
    for kind, name, start, end, parser_doc in symbols:
        if start <= 0 or start > len(lines) or not name.strip():
            continue
        code = "\n".join(lines[start - 1:min(end, len(lines))]).strip()
        if len(code) < 24:
            continue
        code = code[:max_code_chars].rstrip()
        document = _clean_comment(parser_doc, max_query_chars)
        if not document:
            document = _preceding_comment(lines, start, max_query_chars, language)
        if document.casefold().startswith(("jadx warning:", "generated by ")):
            document = ""
        generated = not document
        query = (
            f"{kind} {name} in {language} source file {Path(path).stem}"
            if generated else document
        )
        natural_language, confidence = _detect_natural_language(query, generated)
        code_digest = hashlib.sha256(code.encode()).hexdigest()
        identity = hashlib.sha256(
            f"{provenance['repository']}\0{provenance['revision']}\0{path}"
            f"\0{name}\0{code_digest}".encode()
        ).hexdigest()[:24]
        source_url = (
            f"https://github.com/{provenance['repository']}/blob/"
            f"{provenance['revision']}/{path}"
        )
        record = {
            "id": identity,
            "source": f"the_stack_smol_xl_{language}",
            "source_dataset": DATASET,
            "source_revision": REVISION,
            "source_url": source_url,
            "repository": provenance["repository"],
            "revision": provenance["revision"],
            "path": path,
            "licenses": provenance["licenses"],
            "kind": "text_to_code_retrieval",
            "symbol_kind": kind,
            "symbol": name,
            "language": natural_language,
            "language_confidence": confidence,
            "programming_language": language,
            "text": query,
            "parallel_language": language,
            "parallel_text": code,
            "query_origin": "symbol_identity" if generated else "documentation",
            "code_sha256": code_digest,
        }
        if dialect:
            record["assembly_dialect"] = dialect
        yield record


def _stream_language(language: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("streaming The Stack requires the finetune extra") from exc
    return load_dataset(
        DATASET,
        data_dir=f"data/{language}",
        split="train",
        revision=REVISION,
        streaming=True,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    seen_code: set[str] = set()
    counts: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    natural: Counter[str] = Counter()
    files_seen: Counter[str] = Counter()
    bytes_seen: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for dataset_language in args.language:
            canonical = _canonical(dataset_language)
            for row in _stream_language(dataset_language):
                if files_seen[canonical] >= args.files_per_language:
                    break
                content = row.get("content")
                if not isinstance(content, str) or len(content.encode()) > args.max_file_bytes:
                    continue
                files_seen[canonical] += 1
                bytes_seen[canonical] += len(content.encode())
                for record in records_from_file(
                    row,
                    dataset_language,
                    max_query_chars=args.max_query_chars,
                    max_code_chars=args.max_code_chars,
                ):
                    digest = str(record["code_sha256"])
                    if digest in seen_code:
                        continue
                    seen_code.add(digest)
                    output.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    counts[canonical] += 1
                    origins[str(record["query_origin"])] += 1
                    natural[str(record["language"])] += 1
                    if counts[canonical] >= args.pairs_per_language:
                        break
                if counts[canonical] >= args.pairs_per_language:
                    break
            print(json.dumps({
                "language": canonical,
                "files": files_seen[canonical],
                "input_bytes": bytes_seen[canonical],
                "pairs": counts[canonical],
            }, sort_keys=True), flush=True)
    return {
        "output": str(args.output),
        "records": sum(counts.values()),
        "records_by_programming_language": dict(sorted(counts.items())),
        "records_by_natural_language": dict(sorted(natural.items())),
        "query_origins": dict(sorted(origins.items())),
        "files_by_programming_language": dict(sorted(files_seen.items())),
        "input_bytes_by_programming_language": dict(sorted(bytes_seen.items())),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", action="append", choices=DEFAULT_LANGUAGES)
    parser.add_argument("--files-per-language", type=int, default=2_500)
    parser.add_argument("--pairs-per-language", type=int, default=5_000)
    parser.add_argument("--max-file-bytes", type=int, default=512_000)
    parser.add_argument("--max-query-chars", type=int, default=500)
    parser.add_argument("--max-code-chars", type=int, default=1_400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.language = args.language or list(DEFAULT_LANGUAGES)
    if args.files_per_language <= 0 or args.pairs_per_language <= 0:
        parser.error("file and pair limits must be positive")
    report = build(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
