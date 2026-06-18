"""Tool to delete locally cached library documentation."""

import logging
import sqlite3
from typing import Optional, Type

from pydantic import BaseModel, Field

from infinidev.db.service import execute_with_retry
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)
from infinidev.tools.docs.delete_documentation_input import DeleteDocumentationInput


class DeleteDocumentationTool(InfinibayBaseTool):
    name: str = "delete_documentation"
    description: str = (
        "Delete locally cached library documentation. "
        "Can delete a single section by title, or all sections for a library/version."
    )
    args_schema: Type[BaseModel] = DeleteDocumentationInput

    def _run(
        self,
        library_name: str,
        language: str = "unknown",
        version: str = "latest",
        section: Optional[str] = None,
    ) -> str:
        def _delete(conn: sqlite3.Connection):
            # Resolve version='latest' to the newest stored version, mirroring
            # find_documentation's _resolve_version — otherwise docs fetched
            # under a concrete version (e.g. '3.0') are never deleted because
            # the literal 'latest' row doesn't exist.
            resolved = version
            if version == "latest":
                row = conn.execute(
                    "SELECT version FROM library_docs "
                    "WHERE library_name = ? AND language = ? "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (library_name, language),
                ).fetchone()
                if row is None:
                    return 0
                resolved = row["version"]
            if section:
                cursor = conn.execute(
                    """\
                    DELETE FROM library_docs
                    WHERE library_name = ? AND language = ? AND version = ?
                      AND section_title LIKE ?
                    """,
                    (library_name, language, resolved, f"%{section}%"),
                )
            else:
                cursor = conn.execute(
                    """\
                    DELETE FROM library_docs
                    WHERE library_name = ? AND language = ? AND version = ?
                    """,
                    (library_name, language, resolved),
                )
            conn.commit()
            return cursor.rowcount

        deleted = execute_with_retry(_delete)

        if not deleted:
            return self._error(f"No documentation found matching the criteria")

        if section:
            return self._success({"deleted": deleted, "message": f"Deleted {deleted} section(s) matching '{section}' from {library_name}"})
        return self._success({"deleted": deleted, "message": f"Deleted all {deleted} section(s) for {library_name} ({language}) v{version}"})

