"""SQLite database layer — re-export shim.

The real implementation lives in :mod:`infinidev.code_intel._db`.
This shim exists so the rest of the codebase can keep importing
``from infinidev.tools.base.db import execute_with_retry`` without
breaking, but new callers (especially anything in ``code_intel/``)
should import from the canonical location to avoid loading the
``infinidev.tools`` package init — which transitively pulls litellm
and adds ~4 seconds of cold-start to any code-intelligence query
running in the ``code_interpreter`` subprocess.
"""

from infinidev.code_intel._db import (
    DBConnection,
    execute_with_retry,
    get_connection,
    get_db_path,
    get_pooled_connection,
    parse_query_or_terms,
    sanitize_fts5_query,
)

__all__ = [
    "DBConnection",
    "execute_with_retry",
    "get_connection",
    "get_db_path",
    "get_pooled_connection",
    "parse_query_or_terms",
    "sanitize_fts5_query",
]
