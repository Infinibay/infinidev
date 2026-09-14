"""A small shopping cart."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cart:
    """A cart that keeps a quantity per SKU."""

    items: dict[str, int] = field(default_factory=dict)

    def add(self, sku: str, quantity: int = 1) -> None:
        """Add ``quantity`` of ``sku``."""
        self.items[sku] = self.items.get(sku, 0) + quantity

    def remove(self, sku: str, quantity: int = 1) -> None:
        """Remove ``quantity`` of ``sku``, dropping the line at zero."""
        remaining = self.items.get(sku, 0) - quantity
        if remaining > 0:
            self.items[sku] = remaining
        else:
            self.items.pop(sku, None)

    def total_units(self) -> int:
        """Total quantity across every line."""
        return sum(self.items.values())
