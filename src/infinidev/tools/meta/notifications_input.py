"""Input schema for the manage_notifications tool.

The tool is a discriminated action: ``action`` picks the verb and the
other fields vary by action. Pydantic keeps the per-action fields
optional at the type level; the tool body validates the combinations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TriggerKind = Literal["interval", "cron", "script", "file", "agent"]
ChannelKind = Literal["console", "webhook"]


class ManageNotificationsInput(BaseModel):
    action: Literal[
        "list",
        "create",
        "delete",
        "enable",
        "disable",
        "fire",
        "history",
    ] = Field(description="What to do with notifications.")

    # create-only
    name: str | None = Field(
        default=None,
        description="Unique human-readable identifier. Required for create/delete/enable/disable/fire/history.",
    )
    title: str | None = Field(
        default=None,
        description="Short human-readable title shown in the log/webhook payload.",
    )
    template: str | None = Field(
        default=None,
        description="str.format template for the body. Available vars: name, title, fired_at, plus trigger-specific extras (exit_code, stdout, signature).",
    )
    trigger_type: TriggerKind | None = Field(
        default=None,
        description="Type of trigger. Required for create.",
    )
    trigger: dict | None = Field(
        default=None,
        description=(
                "Trigger config as a JSON object. Required for create. "
                "Fields: interval → {every_seconds:int}; "
                "cron → {cron:\"minute hour dom month dow\"}; "
                "script → {command:str, working_dir?:str, expected_exit_code?:int, stdout_match?:str}; "
                "file → {path:str, watch?:\"mtime\"|\"sha256\"}; "
                "agent → {} (fires only via action=\"fire\")."
            ),
    )
    channel_type: ChannelKind | None = Field(
        default=None,
        description="Delivery channel. Required for create. console or webhook.",
    )
    channel: dict | None = Field(
        default=None,
        description=(
                "Channel config. console → {log_path?:str} (defaults to ~/.infinidev/notifications.log); "
                "webhook → {url:str, method?:\"POST\"|\"PUT\", headers?:{str:str}}."
            ),
    )
    enabled: bool | None = Field(
        default=None,
        description="Initial enabled state for create (default true). Used by enable/disable.",
    )

    # history-only
    limit: int | None = Field(
        default=None,
        description="history: max rows to return (default 50, max 200).",
    )