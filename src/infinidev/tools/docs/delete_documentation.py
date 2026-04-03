"""Tool to delete locally cached library documentation."""

import logging
import sqlite3
from typing import Optional, Type

from pydantic import BaseModel, Field

from infinidev.db.service import execute_with_retry
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


from infinidev.tools.docs.delete_documentation_input import DeleteDocumentationInput
from infinidev.tools.docs.delete_documentation_tool import DeleteDocumentationTool
