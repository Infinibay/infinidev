"""Order pricing helpers."""


def apply_discount(cents: int, percent: int) -> int:
    """Return the price in cents after a percentage discount.

    ``percent`` is a whole number between 0 and 100.
    """
    return cents - (cents * percent // 100)
