from src.cart import Cart


def test_add_accumulates_quantities() -> None:
    cart = Cart()
    cart.add("widget", 2)
    cart.add("widget", 3)
    assert cart.items == {"widget": 5}


def test_total_units_sums_every_line() -> None:
    cart = Cart()
    cart.add("a", 2)
    cart.add("b", 4)
    assert cart.total_units() == 6


def test_remove_drops_a_line_at_zero() -> None:
    cart = Cart()
    cart.add("a", 2)
    cart.remove("a", 2)
    assert cart.items == {}
