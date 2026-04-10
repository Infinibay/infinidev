"""Data models for ContextRank results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RankedItem:
    """A single scored resource (file, symbol, or finding)."""

    target: str          # path or qualified name or finding id
    target_type: str     # 'file' | 'symbol' | 'finding'
    score: float
    reason: str = ""     # human-readable explanation


@dataclass
class ContextRankResult:
    """Top-k ranked resources ready for prompt injection."""

    files: list[RankedItem] = field(default_factory=list)
    symbols: list[RankedItem] = field(default_factory=list)
    findings: list[RankedItem] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.files and not self.symbols and not self.findings
