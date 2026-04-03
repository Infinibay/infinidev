"""Tool to find and read locally cached library documentation."""

import json
import logging
import sqlite3
from typing import Optional, Type

import numpy as np
from pydantic import BaseModel, Field

from infinidev.db.service import execute_with_retry
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.embeddings import compute_embedding, embedding_from_blob

logger = logging.getLogger(__name__)


from infinidev.tools.docs.find_documentation_input import FindDocumentationInput
from infinidev.tools.docs.find_documentation_tool import FindDocumentationTool
