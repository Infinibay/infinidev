from src.pricing import apply_discount


def test_plain_discount_is_subtracted() -> None:
    assert apply_discount(1000, 10) == 900


def test_full_discount_is_free() -> None:
    assert apply_discount(100, 100) == 0


def test_no_discount_is_the_same_price() -> None:
    assert apply_discount(250, 0) == 250
