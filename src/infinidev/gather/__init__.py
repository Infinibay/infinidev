"""Information Gathering Phase for Infinidev.

Structured pipeline that classifies tickets and gathers relevant codebase
context before implementation begins.
"""

from infinidev.gather.models import GatherBrief, TicketType
from infinidev.gather.runner import run_gather

__all__ = ["run_gather", "GatherBrief", "TicketType"]
