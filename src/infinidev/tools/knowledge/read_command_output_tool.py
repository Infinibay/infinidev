"""Read bounded ranges from private command-output artifacts."""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.knowledge.read_command_output_input import (
    ReadCommandOutputInput,
)


class ReadCommandOutputTool(InfinibayBaseTool):
    """Resolve an opaque handle only inside the bound project and session."""

    name: str = "read_command_output"
    is_read_only: bool = True
    description: str = (
        "Read a bounded UTF-8 byte range from a command_output handle. "
        "Pass the handle's artifact_id, type, stream, char_count, and byte_count "
        "unchanged. Access is restricted to this tool's bound project and session; "
        "storage paths are never returned."
    )
    args_schema: Type[BaseModel] = ReadCommandOutputInput

    def _run(
        self,
        artifact_id: int,
        stream: str,
        char_count: int,
        byte_count: int,
        type: str,
        offset: int = 0,
        limit: int = 16_384,
    ) -> str:
        project_id = self.project_id
        session_id = self.session_id
        if project_id.__class__ is not int or project_id <= 0 or not session_id:
            return self._error(
                "read_command_output requires a verified bound project and session"
            )

        # Lazy to keep package import order acyclic: engine.analysis imports the
        # tool registry while the engine package itself is still initialising.
        from infinidev.engine.command_output_store import (
            CommandOutputHandle,
            CommandOutputStore,
            CommandOutputStoreError,
        )

        handle = CommandOutputHandle(
            artifact_id=artifact_id,
            artifact_type=type,
            stream=stream,
            char_count=char_count,
            byte_count=byte_count,
        )
        try:
            content, returned_start, returned_end, has_more = (
                CommandOutputStore().read_range(
                    handle,
                    project_id=project_id,
                    session_id=session_id,
                    offset=offset,
                    limit=limit,
                )
            )
        except CommandOutputStoreError as exc:
            return self._error(str(exc))

        return self._success({
            "artifact_id": artifact_id,
            "type": "command_output",
            "stream": stream,
            "content": content,
            "offset": returned_start,
            "returned_end": returned_end,
            "returned_bytes": returned_end - returned_start,
            "total_bytes": byte_count,
            "has_more": has_more,
            "next_offset": returned_end if has_more else None,
        })
