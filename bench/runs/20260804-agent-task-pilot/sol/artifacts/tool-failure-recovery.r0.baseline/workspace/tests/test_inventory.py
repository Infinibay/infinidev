from src.inventory import Inventory


def test_available_never_becomes_negative() -> None:
    assert Inventory(stock=2, reserved=5).available() == 0


def test_available_subtracts_valid_reservations() -> None:
    assert Inventory(stock=7, reserved=3).available() == 4
