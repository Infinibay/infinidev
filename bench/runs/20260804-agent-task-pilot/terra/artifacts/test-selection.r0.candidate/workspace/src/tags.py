def normalize_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = tag.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
