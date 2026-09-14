"""Hidden regression contract for Cart.discounted.

Withheld from the agent's workspace and restored only to be run. It checks the
parts of the request a visible test cannot: that the new method returns a new
object instead of mutating in place, that the round-half-up rule holds, that a
line reaching zero disappears, and that every existing method still behaves as
it did.
"""

import sys

from src.cart import Cart

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        failures.append(f"{label}{': ' + detail if detail else ''}")


# ── the new method ────────────────────────────────────────────────────

cart = Cart()
cart.add("a", 3)
cart.add("b", 1)
cart.add("c", 10)

discounted = cart.discounted(50)

check(
    "discounted returns a Cart",
    isinstance(discounted, Cart),
    f"got {type(discounted).__name__}",
)
check(
    "the original cart is not modified",
    cart.items == {"a": 3, "b": 1, "c": 10},
    f"original became {cart.items}",
)
check("discounted is a different object", discounted is not cart)
check(
    "quantities round half up",
    discounted.items == {"a": 2, "b": 1, "c": 5},
    f"got {discounted.items}, expected {{'a': 2, 'b': 1, 'c': 5}}",
)

# A line that reaches zero leaves the cart entirely.
empty_line = Cart()
empty_line.add("only", 1)
check(
    "a line reduced to zero is dropped",
    empty_line.discounted(100).items == {},
    f"got {empty_line.discounted(100).items}",
)

# Zero percent keeps every quantity, still in a new object.
untouched = Cart()
untouched.add("a", 7)
zero = untouched.discounted(0)
check("percent 0 keeps the quantities", zero.items == {"a": 7}, f"got {zero.items}")
check("percent 0 still copies", zero is not untouched)

# An empty cart stays empty and does not raise.
check("an empty cart survives", Cart().discounted(25).items == {})

# 9 * 50% = 4.5 -> 5: a half unit rounds up, not to the nearest even number.
boundary = Cart()
boundary.add("a", 9)
check(
    "half a unit rounds up",
    boundary.discounted(50).items == {"a": 5},
    f"got {boundary.discounted(50).items}",
)
# 1 * 100% = 0 -> the line leaves the cart.
check(
    "a full discount empties the cart",
    empty_line.discounted(100).items == {},
    f"got {empty_line.discounted(100).items}",
)

# ── the existing surface still works ──────────────────────────────────

legacy = Cart()
legacy.add("a", 2)
legacy.add("a", 3)
check("add still accumulates", legacy.items == {"a": 5}, f"got {legacy.items}")
legacy.remove("a", 2)
check("remove still decrements", legacy.items == {"a": 3}, f"got {legacy.items}")
legacy.remove("a", 3)
check("remove still drops the line", legacy.items == {}, f"got {legacy.items}")
legacy.add("a", 1)
legacy.add("b", 4)
check("total_units still sums", legacy.total_units() == 5, f"got {legacy.total_units()}")

# The dataclass constructor keeps working for callers that pass items directly.
direct = Cart(items={"x": 3})
check("the constructor still accepts items", direct.items == {"x": 3})

if failures:
    for failure in failures:
        print(failure)
    sys.exit(1)

print("cart contract passed")
