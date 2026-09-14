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

    def discounted(self, percent: int) -> "Cart":
        """Return a new cart with every quantity reduced by ``percent``.

        Each line is scaled by ``(100 - percent) / 100`` and rounded half up, so
        no quantity can drift on a binary float. A line that reaches zero is
        dropped. The cart this is called on is left untouched.
        """
        remaining: dict[str, int] = {}
        for sku, quantity in self.items.items():
            kept = (2 * quantity * (100 - percent) + 100) // 200
            if kept > 0:
                remaining[sku] = kept
        return Cart(items=remaining)
