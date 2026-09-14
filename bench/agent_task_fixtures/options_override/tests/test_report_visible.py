from src.report import render


def test_defaults_render_two_decimals() -> None:
    assert render([1.2345]) == "1.23"


def test_values_are_joined_by_the_default_separator() -> None:
    assert render([1.0, 2.0]) == "1.00, 2.00"


def test_an_empty_report_is_empty() -> None:
    assert render([]) == ""
