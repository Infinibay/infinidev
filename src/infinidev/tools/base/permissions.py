"""Permission checking utilities for Infinidev tools."""

from typing import List, Optional
from infinidev.config.settings import settings


def check_command_permission(command: str) -> bool:
    """
    Check if a command is allowed based on current permission settings.
    
    Args:
        command: The command to check
        
    Returns:
        True if command is allowed, False if blocked
    """
    # If sandbox is disabled, we allow all commands (unless specific restrictions)
    if not settings.SANDBOX_ENABLED:
        return True
    
    # Check if we have a specific command permission setting
    if hasattr(settings, 'COMMAND_PERMISSION') and settings.COMMAND_PERMISSION == 'ask':
        # If permission is set to 'ask', check if command is in allowed list
        if hasattr(settings, 'ALLOWED_COMMANDS') and settings.ALLOWED_COMMANDS:
            # Check if command is in allowed list
            command_name = command.split()[0] if command else ""
            return command_name in settings.ALLOWED_COMMANDS
        # If no allowed commands list, ask for permission (allow by default for now)
        return True
    elif hasattr(settings, 'COMMAND_PERMISSION') and settings.COMMAND_PERMISSION == 'auto_approve':
        # Auto approve - always allow
        return True
    else:
        # Default behavior - allow all commands
        return True


def check_file_permission(action: str, path: str) -> bool:
    """
    Check if a file operation is allowed based on current permission settings.
    
    Args:
        action: The action being performed (e.g., 'read', 'write', 'delete', 'create')
        path: The path being accessed
        
    Returns:
        True if file operation is allowed, False if blocked
    """
    # If sandbox is disabled, we allow all file operations
    if not settings.SANDBOX_ENABLED:
        return True
    
    # Check file permission settings
    if hasattr(settings, 'FILE_PERMISSION') and settings.FILE_PERMISSION == 'ask':
        # If permission is set to 'ask', check if path is in allowed paths
        if hasattr(settings, 'ALLOWED_FILE_PATHS') and settings.ALLOWED_FILE_PATHS:
            # Check if path is in allowed list
            for allowed_path in settings.ALLOWED_FILE_PATHS:
                if path.startswith(allowed_path):
                    return True
        # If no allowed paths, ask for permission (allow by default for now)
        return True
    elif hasattr(settings, 'FILE_PERMISSION') and settings.FILE_PERMISSION == 'auto_approve':
        # Auto approve - always allow
        return True
    else:
        # Default behavior - allow all file operations
        return True