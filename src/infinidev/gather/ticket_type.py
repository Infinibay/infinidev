"""Data models for the information gathering phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TicketType(str, Enum):
    bug = "bug"
    feature = "feature"
    refactor = "refactor"
    sysadmin = "sysadmin"
    other = "other"


