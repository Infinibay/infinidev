def normalize_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = tag.strip()
        normalized = cleaned.casefold()
        if cleaned and normalized not in seen:
            seen.add(normalized)
            result.append(cleaned)
    return result
