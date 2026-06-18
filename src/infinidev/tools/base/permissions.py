"""Permission checking utilities for Infinidev tools."""

import os

from infinidev.config.settings import settings


def check_command_permission(command: str) -> bool:
    """Check if a command is allowed based on current permission settings.

    Note: This is the legacy sandbox-based check. The active permission
    system for execute_command lives in ExecuteCommandTool._check_permission()
    and uses EXECUTE_COMMANDS_PERMISSION + request_permission().
    """
    if not settings.SANDBOX_ENABLED:
        return True

    if hasattr(settings, 'ALLOWED_COMMANDS') and settings.ALLOWED_COMMANDS:
        command_name = command.split()[0] if command else ""
        return command_name in settings.ALLOWED_COMMANDS

    return True


def _workspace_root() -> str | None:
    """Best-effort root of the active workspace for ``auto`` file scoping.

    Falls back to the process CWD when no workspace is bound to the context
    (e.g. classic mode), so edits under the directory the CLI was launched in
    still auto-approve.
    """
    try:
        from infinidev.tools.base.context import get_current_workspace_path
        ws = get_current_workspace_path()
    except Exception:
        ws = None
    if not ws:
        try:
            ws = os.getcwd()
        except OSError:
            ws = None
    return ws


def _is_overbroad_root(ws: str) -> bool:
    """Whether *ws* is too broad to treat as a trusted auto-approve workspace.

    Launching from ``/`` or ``$HOME`` would otherwise auto-approve edits to
    ``~/.ssh/authorized_keys``, shell rc files, ``/etc/*`` etc. with no prompt.
    Those roots fall through to the confirmation path instead.
    """
    ws_real = os.path.realpath(ws)
    if ws_real == os.sep or os.path.dirname(ws_real) == ws_real:
        return True  # filesystem root
    try:
        home = os.path.realpath(os.path.expanduser("~"))
    except OSError:
        home = None
    return bool(home and ws_real == home)


def check_file_permission(action: str, path: str) -> str | None:
    """Check if a file write/edit operation is allowed.

    Args:
        action: "write_file" or "edit_file"
        path: The file path being modified

    Returns:
        None if allowed, error string if denied.
    """
    mode = settings.FILE_OPERATIONS_PERMISSION

    if mode == "auto_approve":
        return None

    if mode == "auto":
        # Auto-approve edits inside the active workspace; prompt for anything
        # outside it (home dotfiles, /etc, sibling repos). A coding agent's job
        # is to edit project files freely, but silently touching files outside
        # the project should always require confirmation.
        real = os.path.realpath(path)
        ws = _workspace_root()
        if ws and not _is_overbroad_root(ws):
            ws_real = os.path.realpath(ws)
            if real == ws_real or real.startswith(ws_real + os.sep):
                return None
        # Outside the workspace (or no workspace context) — fall through to ask.
        from infinidev.tools.permission import (
            is_permission_handler_registered,
            request_permission,
        )
        if not is_permission_handler_registered():
            # Non-interactive: fail closed rather than silently writing outside
            # the workspace (e.g. ~/.ssh, /etc) with no human to approve.
            return (
                f"File operation outside workspace requires confirmation but no "
                f"approval UI is available. Set FILE_OPERATIONS_PERMISSION="
                f"auto_approve to allow non-interactively: {path}"
            )
        approved = request_permission(
            tool_name=action,
            description=f"{'Write' if action == 'write_file' else 'Edit'} file outside workspace",
            details=path,
        )
        if not approved:
            return f"File operation denied by user: {path}"
        return None

    if mode == "allowed_paths":
        allowed = settings.ALLOWED_FILE_PATHS
        if not allowed:
            return "File operation denied: no paths in allowed list"
        # Resolve symlinks/.. and require an exact match or an os.sep-bounded
        # subpath (mirrors _validate_sandbox_path). A raw startswith would
        # authorize '/home/user/proj-secrets' under an allowed '/home/user/proj'
        # and let a '..'-containing path escape the allowed directory.
        real = os.path.realpath(path)
        for allowed_path in allowed:
            allowed_real = os.path.realpath(allowed_path)
            if real == allowed_real or real.startswith(allowed_real + os.sep):
                return None
        return f"File operation denied: '{path}' not in allowed paths"

    if mode == "ask":
        from infinidev.tools.permission import request_permission
        approved = request_permission(
            tool_name=action,
            description=f"{'Write' if action == 'write_file' else 'Edit'} file",
            details=path,
        )
        if not approved:
            return f"File operation denied by user: {path}"
        return None

    return None  # Unknown mode — allow
