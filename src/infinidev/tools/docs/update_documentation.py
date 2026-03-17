"""Tool to fetch and generate library documentation from the web."""

import logging
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


class UpdateDocumentationInput(BaseModel):
    library_name: str = Field(..., description="Name of the library to document")
    language: str = Field(default="unknown", description="Programming language")
    version: str = Field(default="latest", description="Library version")


class UpdateDocumentationTool(InfinibayBaseTool):
    name: str = "update_documentation"
    description: str = (
        "Fetch library documentation from the web, organize it into sections, "
        "and store it locally for future reference. Uses web search + LLM to "
        "produce structured documentation."
    )
    args_schema: Type[BaseModel] = UpdateDocumentationInput

    def _run(
        self,
        library_name: str,
        language: str = "unknown",
        version: str = "latest",
    ) -> str:
        from infinidev.tools.docs.doc_flow import DocFlow

        flow = DocFlow()
        try:
            result = flow.execute(library_name, language, version)
            return self._success(result)
        except Exception as e:
            logger.error("update_documentation failed: %s", e, exc_info=True)
            return self._error(f"Documentation generation failed: {e}")
