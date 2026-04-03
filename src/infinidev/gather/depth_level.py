"""Data models for the information gathering phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DepthLevel(str, Enum):
    """How much analysis and control the engine applies."""
    minimal = "minimal"   # "Do it yourself" — single free run
    light = "light"       # "Think first" — force-read then free execute
    standard = "standard" # "Follow the process" — full phase pipeline
    deep = "deep"         # "I'm watching every move" — strict guardrails


