"""Backward-compatible wrapper for semantic ``search_knowledge`` calls."""

from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.knowledge.search_findings_input import SearchFindingsInput


class SearchFindingsTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "search_findings"
    description: str = (
        "Search findings by semantic similarity. Returns findings whose "
        "topic and content are similar to the query, ranked by similarity score. "
        "Use this to check if a finding already exists before recording "
        "a new one, or to find related findings across tasks."
    )
    args_schema: Type[BaseModel] = SearchFindingsInput

    def _run(
        self,
        query: str,
        threshold: float = 0.65,
        session_id: str | None = None,
        include_content: bool = False,
        limit: int = 20,
    ) -> str:
        from infinidev.tools.knowledge.search_knowledge_tool import SearchKnowledgeTool

        return SearchKnowledgeTool._run(
            self,
            query=query,
            sources=["findings"],
            session_id=session_id,
            mode="semantic",
            threshold=threshold,
            include_content=include_content,
            limit=limit,
        )
