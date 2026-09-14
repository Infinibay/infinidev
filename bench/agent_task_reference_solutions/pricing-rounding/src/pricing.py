"""Order pricing helpers."""


def apply_discount(cents: int, percent: int) -> int:
    """Return the price in cents after a percentage discount.

    ``percent`` is a whole number between 0 and 100. The discount is rounded
    half up to the nearest whole cent, computed in integers so no float
    rounding can drift.
    """
    discount = (2 * cents * percent + 100) // 200
    return cents - discount
