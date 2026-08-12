"""Pagination helpers."""


def pages_needed(total_items: int, page_size: int) -> int:
    """Return how many pages are required for all items."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return total_items // page_size
