def render_badge(level: str, text: str) -> str:
    if level == "critical":
        return f"!! {text}"
    if level == "warning":
        return text
    return f"[{level}] {text}"
