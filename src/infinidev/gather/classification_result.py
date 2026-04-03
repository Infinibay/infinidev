"""Data models for the information gathering phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from infinidev.gather.depth_level import DepthLevel
from infinidev.gather.ticket_type import TicketType


class ClassificationResult(BaseModel):
    """Result of ticket type + depth classification."""

    ticket_type: TicketType = TicketType.other
    reasoning: str = ""
    keywords: list[str] = Field(default_factory=list)
    depth: DepthLevel = DepthLevel.standard
    depth_reasoning: str = ""


