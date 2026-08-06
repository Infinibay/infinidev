"""Build fine-tuning tool schemas from the live developer tool surface."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from infinidev.config.settings import settings
from infinidev.engine.schema_sanitizer import (
    ADD_NOTE_SCHEMA,
    ADD_SESSION_NOTE_SCHEMA,
    STEP_COMPLETE_SCHEMA,
    tool_to_openai_schema,
)
from infinidev.tools import (
    CHAT_TOOLS,
    CODE_INTEL_TOOLS,
    DOCS_TOOLS,
    FILE_TOOLS,
    GIT_TOOLS,
    HISTORY_TOOLS,
    KNOWLEDGE_TOOLS,
    META_TOOLS,
    SHELL_TOOLS,
    WEB_TOOLS,
)
from infinidev.tools.file import ViewImageTool
from infinidev.tools.knowledge import ReadCommandOutputTool


_LOCAL_DEVELOPER_TOOL_CLASSES = (
    FILE_TOOLS
    + GIT_TOOLS
    + SHELL_TOOLS
    + WEB_TOOLS
    + KNOWLEDGE_TOOLS
    + CHAT_TOOLS
    + DOCS_TOOLS
    + CODE_INTEL_TOOLS
    + META_TOOLS
    + HISTORY_TOOLS
)
_ENGINE_SCHEMAS = (STEP_COMPLETE_SCHEMA, ADD_NOTE_SCHEMA, ADD_SESSION_NOTE_SCHEMA)


def get_training_tool_schemas(
    *,
    supports_vision: bool = False,
    command_output_capture: bool | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic local schemas matching the configured developer role.

    MCP tools are environment-specific and are intentionally excluded from a
    portable training dataset. Vision and command-output tools follow the same
    feature gates as the runtime.
    """
    if command_output_capture is None:
        command_output_capture = settings.COMMAND_OUTPUT_CAPTURE_ENABLED

    classes = [
        tool_class
        for tool_class in _LOCAL_DEVELOPER_TOOL_CLASSES
        if (supports_vision or tool_class is not ViewImageTool)
        and (command_output_capture or tool_class is not ReadCommandOutputTool)
    ]
    schemas = [tool_to_openai_schema(tool_class()) for tool_class in classes]
    schemas.extend(deepcopy(schema) for schema in _ENGINE_SCHEMAS)
    return schemas


def get_training_schema_map(**kwargs: Any) -> dict[str, dict[str, Any]]:
    """Return live function schemas keyed by their public tool names."""
    return {
        schema["function"]["name"]: schema["function"]
        for schema in get_training_tool_schemas(**kwargs)
    }
