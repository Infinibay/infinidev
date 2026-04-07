"""Tool: find methods that look like a given method.

Backed by ``code_intel.method_index``. The tool fetches the target
method's fingerprint, runs the size-bounded Jaccard search across
``ci_method_bodies``, and renders the results as a compact table the
model can consume directly.

Use cases the tool is designed for:

  * **Refactoring**: before extracting a method to a new file, the
    model checks whether the same logic already lives elsewhere.
  * **Duplication detection**: explicit "find duplicates of X" with a
    tighter threshold.
  * **Test discovery**: find tests that look like a known-good test,
    as starting points for new test cases.
"""

from __future__ import annotations

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class FindSimilarMethodsInput(BaseModel):
    qualified_name: str = Field(
        ...,
        description=(
            "Qualified name of the method to compare against, e.g. "
            "'VirtioSocketWatcherService.connectToVm' or just 'parseTimestamp' "
            "for a top-level function. Must already be indexed — call read_file "
            "or list_symbols on its file first if unsure."
        ),
        min_length=1,
    )
    file_path: str = Field(
        default="",
        description=(
            "Optional file path to disambiguate when two classes in different "
            "files share a method name. Leave empty to take the first match."
        ),
    )
    threshold: float = Field(
        default=0.7,
        description=(
            "Minimum Jaccard similarity in [0, 1]. Lower values surface more "
            "candidates but with weaker resemblance. 0.7 is the default; raise "
            "to 0.85 for near-duplicates only, lower to 0.5 for loose matches."
        ),
        ge=0.0, le=1.0,
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return (1-50).",
        ge=1, le=50,
    )


class FindSimilarMethodsTool(InfinibayBaseTool):
    name: str = "find_similar_methods"
    description: str = (
        "Find methods elsewhere in the project whose body looks like a given "
        "method. Uses normalized-token Jaccard similarity over indexed method "
        "bodies. Returns a ranked list with file paths, line ranges, and a "
        "similarity score; methods with identical normalized bodies are flagged "
        "as exact duplicates. Useful for refactoring (spot duplicates before "
        "extracting), code review (find copy-paste), and onboarding."
    )
    args_schema: Type[BaseModel] = FindSimilarMethodsInput

    def _run(
        self,
        qualified_name: str,
        file_path: str = "",
        threshold: float = 0.7,
        limit: int = 10,
    ) -> str:
        from infinidev.code_intel.method_index import find_similar, fetch_fingerprint

        project_id = self.project_id
        if not project_id:
            return self._error("No project context — cannot query the method index.")

        # Coerce numeric params (LLMs sometimes send strings)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.7
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10

        target_path = file_path or None
        target = fetch_fingerprint(project_id, qualified_name, target_path)
        if target is None and "." not in qualified_name:
            # Bare-name fallback: resolve via find_definition (which now
            # understands both bare and qualified names) to recover the
            # canonical qualified_name, then re-query the fingerprint
            # store. Lets the model pass `connectToVm` and still get
            # results without having to remember the class prefix.
            from infinidev.code_intel.query import find_definition
            defs = find_definition(project_id, qualified_name, kind="method")
            if not defs:
                defs = find_definition(project_id, qualified_name, kind="function")
            if defs:
                # Try each candidate's qualified_name in priority order
                # (find_definition already sorts function/method first).
                for d in defs[:5]:
                    target = fetch_fingerprint(project_id, d.qualified_name, d.file_path)
                    if target is not None:
                        qualified_name = d.qualified_name
                        break
        if target is None:
            return self._error(
                f"No fingerprint for '{qualified_name}'"
                + (f" in {file_path}" if file_path else "")
                + ". The method may not be indexed yet (try read_file on its "
                "containing file first), it may be too small to fingerprint "
                "(skipped under 6 lines), or you may have a typo in the name."
            )

        hits = find_similar(
            project_id,
            qualified_name=target.qualified_name,
            file_path=target.file_path,
            threshold=threshold,
            limit=limit,
        )

        # Header always includes the target so the model can sanity-check
        # we found the right thing.
        header_lines = [
            f"Target: {target.qualified_name}  "
            f"({target.file_path}:{target.line_start}-{target.line_end}, "
            f"{target.body_size} lines, {target.language})",
        ]

        if not hits:
            header_lines.append(
                f"No similar methods found (threshold={threshold}, "
                f"size window ±40%, project-wide)."
            )
            header_lines.append(
                "Try lowering the threshold (e.g. 0.5) or check whether other "
                "files in the project have been read at least once — only "
                "indexed files contribute to the similarity index."
            )
            return "\n".join(header_lines)

        header_lines.append(
            f"Found {len(hits)} similar method(s) "
            f"(threshold={threshold}, sorted by similarity):"
        )

        rows = []
        for h in hits:
            tag = "EXACT-DUP" if h.is_exact_dup else f"{h.similarity:.0%}"
            rows.append(
                f"  [{tag:>9}] {h.qualified_name}  "
                f"— {h.file_path}:{h.line_start}-{h.line_end} "
                f"({h.body_size} lines)"
            )

        return "\n".join(header_lines + rows)
