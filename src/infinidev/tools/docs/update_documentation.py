"""Tool to fetch and generate library documentation from the web."""

import logging
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


from infinidev.tools.docs.update_documentation_input import UpdateDocumentationInput
from infinidev.tools.docs.update_documentation_tool import UpdateDocumentationTool
