def normalize_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    for tag in tags:
        cleaned = tag.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
