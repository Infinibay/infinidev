from src.tags import normalize_tags


def test_equivalent_case_uses_first_spelling() -> None:
    assert normalize_tags(["Python", " python ", "PYTHON", "Rust"]) == ["Python", "Rust"]


def test_preserves_order_and_does_not_mutate_input() -> None:
    source = [" beta ", "Alpha", "BETA", "gamma"]
    assert normalize_tags(source) == ["beta", "Alpha", "gamma"]
    assert source == [" beta ", "Alpha", "BETA", "gamma"]


def test_omits_blank_tags() -> None:
    assert normalize_tags(["", "  ", "valid"]) == ["valid"]
