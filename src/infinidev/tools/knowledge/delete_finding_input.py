"""Input schema for deleting a research finding."""

from pydantic import BaseModel, Field


class DeleteFindingInput(BaseModel):
    """Arguments accepted by :class:`DeleteFindingTool`."""

    finding_id: int = Field(..., description="ID of the finding to delete")
