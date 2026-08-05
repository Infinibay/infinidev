from src.tags import normalize_tags


def test_result_remains_a_plain_list_of_original_spellings() -> None:
    result = normalize_tags(["API", "api", "Web"])
    assert type(result) is list
    assert result == ["API", "Web"]
