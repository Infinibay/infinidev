"""Data models for the information gathering phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class Question:
    """A single question to investigate about the codebase."""

    id: str
    question: str
    context_prompt: str  # Expanded prompt with {ticket_description} placeholder
    max_tool_calls: int = 15
    timeout_seconds: int = 120


