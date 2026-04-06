"""Punish using execute_command for things with dedicated tools."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class ShellWhenToolExistsChecker(PromptBehaviorChecker):
    name = "shell_when_tool_exists"
    description = "Punish execute_command when a dedicated tool exists"
    default_enabled = True
    delta_range = (-2, 0)
    settings_message = "Shell-when-tool-exists — punishes shell hacks for cat/grep/find/git status (-2..0)"

    criteria = (
        "Punish shell hacks when a dedicated tool exists. Check the "
        "tool_calls in latest_message. If the agent invoked execute_command "
        "with a shell command that has a dedicated equivalent, penalize:\n"
        "- 'cat'/'head'/'tail'/'less' for reading files       → use read_file / partial_read\n"
        "- 'grep' / 'rg' for searching code                    → use code_search\n"
        "- 'find' / 'ls' for listing files                     → use glob / list_directory\n"
        "- 'git status'/'git diff'/'git log'/'git branch'      → use git_status / git_diff / git_branch\n"
        "- 'sed' / 'awk' for editing files                     → use replace_lines / edit_symbol\n"
        "Scoring:\n"
        "- -2 if the command is a clear-cut substitute for a dedicated tool.\n"
        "- -1 if it's a borderline case (e.g., piped command with one of the above).\n"
        "- 0 if execute_command was used for something legitimate (running a "
        "  test, building, installing, running a script, etc.).\n"
        "Never return positive deltas for this criterion."
    )
