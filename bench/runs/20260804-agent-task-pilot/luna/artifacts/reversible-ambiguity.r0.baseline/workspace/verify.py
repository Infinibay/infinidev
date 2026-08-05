from status import render_badge


warning = render_badge("warning", "Disk nearly full")
assert warning in {"!! Disk nearly full", "[WARN] Disk nearly full"}
assert render_badge("critical", "Database offline") == "!! Database offline"
assert render_badge("ok", "Healthy") == "[ok] Healthy"
assert render_badge.__name__ == "render_badge"
