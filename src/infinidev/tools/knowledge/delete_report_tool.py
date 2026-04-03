"""Tool for deleting reports/artifacts."""

import os
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, get_db_path
from infinidev.tools.knowledge.delete_report_input import DeleteReportInput


class DeleteReportTool(InfinibayBaseTool):
    name: str = "delete_report"
    description: str = (
        "Delete a report by its artifact ID. "
        "Optionally removes the file from disk as well."
    )
    args_schema: Type[BaseModel] = DeleteReportInput

    def _run(self, artifact_id: int, delete_file: bool = True) -> str:
        def _delete(conn: sqlite3.Connection) -> dict:
            row = conn.execute(
                "SELECT id, file_path, description FROM artifacts WHERE id = ?",
                (artifact_id,),
            ).fetchone()

            if not row:
                raise ValueError(f"Artifact {artifact_id} not found")

            conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
            conn.commit()
            return {
                "file_path": row["file_path"],
                "description": row["description"],
            }

        try:
            result = execute_with_retry(_delete)
        except ValueError as e:
            return self._error(str(e))
        except Exception as e:
            return self._error(f"Failed to delete report: {e}")

        # Try to delete file from disk
        file_deleted = False
        if delete_file and result["file_path"]:
            # Resolve host path from pod path
            db_path = get_db_path()
            host_base = os.path.dirname(os.path.abspath(db_path))
            # Pod paths look like /artifacts/project_X/reports/foo.md
            # Host paths are relative to the DB directory
            pod_path = result["file_path"]
            if pod_path.startswith("/artifacts/"):
                host_path = os.path.join(host_base, pod_path.lstrip("/"))
            else:
                host_path = pod_path

            try:
                if os.path.exists(host_path):
                    os.remove(host_path)
                    file_deleted = True
            except OSError:
                pass

        self._log_tool_usage(f"Deleted report #{artifact_id}")
        return self._success({
            "artifact_id": artifact_id,
            "deleted": True,
            "file_deleted": file_deleted,
            "description": result["description"],
        })

