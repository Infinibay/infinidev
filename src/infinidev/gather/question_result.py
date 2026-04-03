"""Data models for the information gathering phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QuestionResult(BaseModel):
    """Result of answering one question."""

    question_id: str
    question_text: str
    answer: str
    tool_calls_used: int = 0
    phase: str = "fixed"  # "fixed" or "dynamic"


