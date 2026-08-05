def normalize_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = tag.strip()
        canonical = cleaned.casefold()
        if cleaned and canonical not in seen:
            result.append(cleaned)
            seen.add(canonical)
    return result
