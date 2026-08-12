from src.pagination import pages_needed


def test_partial_final_page_is_included() -> None:
    assert pages_needed(21, 10) == 3


def test_exact_multiple_does_not_add_an_empty_page() -> None:
    assert pages_needed(20, 10) == 2


def test_empty_collection_needs_no_pages() -> None:
    assert pages_needed(0, 10) == 0


def test_invalid_page_size_keeps_raising() -> None:
    try:
        pages_needed(10, 0)
    except ValueError as exc:
        assert str(exc) == "page_size must be positive"
    else:
        raise AssertionError("pages_needed must reject a non-positive page size")
