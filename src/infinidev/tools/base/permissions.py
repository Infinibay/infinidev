"""Permission checking utilities for Infinidev tools."""

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

    if mode == "allowed_paths":
        allowed = settings.ALLOWED_FILE_PATHS
        if not allowed:
            return f"File operation denied: no paths in allowed list"
        for allowed_path in allowed:
            if path.startswith(allowed_path):
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
