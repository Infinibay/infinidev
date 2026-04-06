"""Tool: let the agent register a project-specific test runner.

Most projects use a standard runner the guidance detector already
recognises (pytest, jest, cargo test, …). This tool exists for the
weird ones — a custom shell wrapper, a Makefile target, a project-
specific script. The agent calls it once when it discovers the real
test command, and the guidance system uses it for the rest of the task
*on top of* the built-in runner list.

The declaration lives on the running ``LoopState`` (not in global
config), so it scopes naturally to the current task and doesn't leak
into the next one.
"""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class DeclareTestCommandInput(BaseModel):
    command_pattern: str = Field(
        description=(
            "A short substring that uniquely identifies the project's test "
            "runner invocation. Matched against the args of execute_command "
            "to detect 'this was a test run'. Examples: 'bash test.sh', "
            "'make integration', 'just test', './run_tests.sh'. Use the "
            "shortest substring that is unique enough to avoid matching "
            "non-test commands. You don't need to declare standard runners "
            "(pytest, jest, cargo test, go test, mvn, …) — they are already "
            "recognised."
        ),
        min_length=1,
    )


class DeclareTestCommandTool(InfinibayBaseTool):
    name: str = "declare_test_command"
    description: str = (
        "Tell the engine that a project-specific command runs the test "
        "suite. Use this ONCE when you discover the real test invocation "
        "for a project (e.g. 'bash test.sh' or 'make check') and the "
        "built-in runner list doesn't cover it. The guidance system will "
        "then track stuck-on-tests patterns for that command too. "
        "Standard runners (pytest, jest, cargo test, go test, mvn, …) are "
        "already recognised — don't declare those."
    )
    args_schema: Type[BaseModel] = DeclareTestCommandInput

    def _run(self, command_pattern: str) -> str:
        from infinidev.tools.base.context import get_context_for_agent
        ctx = get_context_for_agent(self.agent_id)
        if not ctx or not getattr(ctx, "loop_state", None):
            return self._error("No active loop context")

        pattern = command_pattern.strip().lower()
        if not pattern:
            return self._error("command_pattern must not be empty")

        existing = list(getattr(ctx.loop_state, "custom_test_commands", []) or [])
        if pattern in existing:
            return self._success({
                "status": "already_registered",
                "command_pattern": pattern,
                "total_custom_commands": len(existing),
            })

        existing.append(pattern)
        ctx.loop_state.custom_test_commands = existing
        return self._success({
            "status": "registered",
            "command_pattern": pattern,
            "total_custom_commands": len(existing),
            "note": (
                "The guidance system will now treat any execute_command "
                f"containing '{pattern}' as a test run."
            ),
        })
