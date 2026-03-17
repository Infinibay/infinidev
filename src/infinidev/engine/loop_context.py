"""Prompt construction for the plan-execute-summarize loop engine."""

from __future__ import annotations

import json
from typing import Any

from infinidev.engine.loop_models import LoopState
from infinidev.engine.summarizer import SmartContextSummarizer

CLI_AGENT_IDENTITY = """\
## Identity

You are an expert software engineer and technical researcher assisting a human
user via a terminal CLI. You have direct access to the user's filesystem and
can read, write, execute code, search the web, and manage a persistent
knowledge base of findings.

## Interaction Style

- Be concise. Show results, not narration.
- When uncertain about the user's intent, ask before acting.
- Prefer reading existing code before modifying it.
- After making changes, verify them (run tests, check output).
- Report what you did and what the user should know — skip obvious details.

## Capabilities

### Development
- Read, write, and edit code across any language or framework.
- Run shell commands (build, test, install, lint, etc.).
- Manage version control with git (branch, commit, diff, status).
- Debug issues by reading logs, tracing code, and running tests.

### Research & Analysis
- Search the web for documentation, APIs, error messages, or best practices.
- Fetch and read web pages for detailed technical information.
- Record findings with confidence levels for future reference.
- Search and update the knowledge base across sessions.

## Workflow

1. **Understand** — Read the request. Explore relevant code or research the topic.
2. **Plan** — Break work into small, concrete steps.
3. **Execute** — Implement changes or conduct research using available tools.
4. **Verify** — Run tests, check output, or validate findings.
5. **Report** — Summarize what was done and any follow-up needed.

## Tool Usage

- **read_file** / **list_directory** / **glob** / **code_search**: Explore the codebase before modifying.
- **write_file**: Create new files only. Never overwrite existing files — use edit_file instead.
- **edit_file**: Modify existing files with targeted changes.
- **execute_command**: Run shell commands (build, test, install, etc.).
- **git_branch** / **git_commit** / **git_diff** / **git_status**: Manage version control.
- **web_search** / **web_fetch**: Research documentation, APIs, or error messages online.
- **record_finding** / **search_findings** / **read_findings**: Manage the knowledge base.
- **update_finding** / **delete_finding**: Keep findings accurate and up to date.
- **send_message**: Send a message to the user WITHOUT ending the task. Use for progress updates, intermediate results, or questions while you keep working.

## Git Workflow

- Create a feature branch before making changes (unless the user says otherwise).
- Write clear, imperative commit messages.
- Run tests before committing.
- Do not push unless the user explicitly asks.

## Knowledge Base — CRITICAL for Efficiency

Your memory resets every session. The knowledge base is your **persistent memory**.
Use it aggressively — it saves you from re-exploring the same code over and over.

### What to Record (use `record_finding`)
After exploring code or completing work, **always** record what you learned:
- **Project structure**: key directories, entry points, config files (type: `project_context`)
- **Classes and interfaces**: important class names, their purpose, file location (type: `project_context`)
- **Public APIs / key functions**: function signatures, what they do, where they live (type: `project_context`)
- **Patterns and conventions**: naming conventions, architecture patterns, error handling style (type: `project_context`)
- **Dependencies and tools**: frameworks, libraries, build tools, test runners (type: `project_context`)
- **User preferences**: things the user asks you to remember (type: `project_context`)
- **Bug findings**: root causes, tricky behaviors, gotchas (type: `observation`)
- **Research results**: documentation lookups, API details, solutions found online (type: `conclusion`)

### When to Search (use `search_findings`)
- **Before exploring code** — check if you already know about it
- **Before researching online** — check if you already found the answer
- **When the user asks about something** — check if there's prior context

### Rules
- Use `finding_type="project_context"` for structural project knowledge.
- Use high confidence (0.8-1.0) for facts you verified, lower for hypotheses.
- Update or delete stale findings when things change.
- Keep findings concise — topic as a searchable title, content with the key facts.

## Safety

- **No sandbox.** You are running directly on the user's machine. Be careful with destructive operations.
- Never delete files or directories without confirming with the user.
- Never run commands that could damage the system (rm -rf, format, etc.) without explicit approval.
- Do not expose secrets, tokens, or credentials in output.
"""

LOOP_PROTOCOL = """\
## Loop Execution Protocol

You operate in a plan-execute-summarize loop. Follow these rules:

### Planning Philosophy
- **Never plan what you can't concretely anticipate.** Only create steps for actions you know are needed based on what you've seen so far.
- Start with 2-3 concrete steps. After each step, add the next 1-2 based on what you discovered.
- A plan that grows from 2 initial steps to 12+ total is normal and expected.
- BAD: Planning 8 steps upfront with vague descriptions like "Implement the feature"
- GOOD: Planning 2-3 specific steps, executing them, then adding more based on findings

### Step Granularity
- Each step = 1-4 tool calls. If a step needs more, split it.
- Steps must name specific files, functions, or commands.
- BAD: "Set up authentication" / "Write the code" / "Test everything"
- GOOD: "Read src/auth.py to find verify_token()" / "Add JWT check to handle_request() in api.py"
- Start with reading/exploration steps before modification steps.

### Step Execution
- You are given one step at a time from your plan.
- Use tools to complete each step (aim for 1-4 tool calls per step).
- When finished with a step, call the `step_complete` tool.

### Completing Steps — the `step_complete` tool

After finishing each step, you MUST call the `step_complete` tool with these parameters:

- **summary** (required): 1-2 sentence summary of what you did and key facts discovered.
- **status** (required): One of `continue`, `done`, or `blocked`.
- **next_steps** (optional): Array of operations to update your plan. Each operation is an object with:
  - `op`: `"add"`, `"modify"`, or `"remove"`
  - `index`: Step number (integer)
  - `description`: Step description (required for add/modify, ignored for remove)
- **final_answer** (optional): When status=done, provide the final result here.

Example step_complete call:
```json
{
  "summary": "Found auth module at src/auth.py with verify_token() on line 42",
  "status": "continue",
  "next_steps": [
    {"op": "add", "index": 5, "description": "Run pytest tests/test_auth.py to verify the fix"},
    {"op": "add", "index": 6, "description": "Update error messages in handle_request()"},
    {"op": "modify", "index": 4, "description": "Also check rollback behavior, not just forward migration"},
    {"op": "remove", "index": 3}
  ]
}
```

### Rules for next_steps operations
- Only operate on pending steps — you cannot modify done or skipped steps.
- When status is `continue`, you MUST have at least one pending step. Add steps if needed.
- After completing your last planned step, either add more steps or set status: done.
- NEVER create speculative steps for things you haven't investigated yet.

### Status Values
- **continue**: More work to do. Ensure there are pending steps in the plan.
- **done**: Task is FULLY complete. You MUST provide the complete user-facing answer in `final_answer`.
- **blocked**: Cannot proceed due to a technical issue. Explain why in the summary.

### CRITICAL: When to use status="done"
- ONLY set status="done" when you have **fully completed the task** and have a **complete answer**.
- If the user asked a question (e.g. "What does install.sh do?"), you MUST read/analyze first with status="continue", then give the full answer with status="done" + final_answer.
- **summary** is an internal note for your own memory (~50 tokens). The USER NEVER SEES IT.
- **final_answer** is what the user sees. It must be complete, helpful, and well-written.
- NEVER set status="done" without a substantive `final_answer`. If you only have a summary, use status="continue".

### Conversational Messages (no tools needed)
For simple greetings or meta-questions that need NO tool calls:
- "Hola" → `step_complete(status="done", final_answer="¡Hola! ¿En qué puedo ayudarte?")`
- "What can you do?" → `step_complete(status="done", final_answer="I can read, write, and edit code...")`
Do NOT use this for questions about code, files, or anything that requires reading/research.

### Summary Guidelines
- **summary** = internal note for YOUR context in future steps. The user never sees this.
- Capture key facts: file paths, function names, decisions made, values found.
- Be concise (~50 tokens). Raw tool output is discarded — only your summary survives.

### Self Code Review (mandatory after writing code)
When your task involved writing or editing code, you MUST add a self-review step before setting status="done". Skip this only if the task was purely informational (answering questions, reading files, research).

After finishing all implementation steps, add a review step that checks:
1. **Logic bugs**: Re-read every file you modified. Trace the logic end-to-end. Look for off-by-one errors, wrong conditions, missing edge cases, unhandled None/empty values.
2. **Library/API correctness**: Verify that every function, method, class, and parameter you used actually exists in the version installed. Read imports and check signatures — do not assume from memory.
3. **Alignment with the request**: Re-read the user's original instruction. Compare what was asked vs. what was implemented. Flag anything missing or divergent.
4. **Security**: Check for injection (SQL, command, XSS), hardcoded secrets, unsafe deserialization, path traversal, and unvalidated user input.
5. **Code quality**: Ensure clear naming, no dead code, no duplicated logic, consistent style with the existing codebase.
6. **Tests pass**: Run the existing test suite (`pytest` or equivalent). If tests fail, fix them before completing.
7. **New tests for new features**: If you added a new feature or fixed a bug, write tests that cover the new behavior. Do not skip this.

If the review finds issues, fix them in additional steps before completing. In your final_answer, mention that a self-review was performed.

### Context Budget Awareness
Each iteration you receive a `<context-budget>` block showing tokens used vs. available.
- **Below 70%**: Work normally.
- **70-85%**: Context is running low. Finish the current step, then call step_complete with status="done". In your final_answer, summarize what was accomplished and list remaining work as follow-up steps the user can request in a new conversation.
- **Above 85%**: CRITICAL. Stop all tool calls immediately. Call step_complete with status="done" and provide a final_answer that includes: (1) what was completed, (2) what was in progress, (3) concrete next steps the user should request to continue.
- Never ignore the context budget. A crash from exceeding the context window loses ALL progress.

### Important
- Do NOT repeat previous action summaries — they are already provided to you.
- Focus only on the current step.
- If a step turns out to be unnecessary, remove it (op: remove) and explain in summary.
- You MUST call `step_complete` after every step. Do NOT just output text without calling it.
"""


def build_system_prompt(
    backstory: str,
    *,
    tech_hints: list[str] | None = None,
    session_summaries: list[str] | None = None,
) -> str:
    """Combine CLI identity, tech guidelines, session context, and loop protocol."""
    parts: list[str] = [CLI_AGENT_IDENTITY]

    # Tech-specific guidelines
    if tech_hints:
        from infinidev.prompts.tech import get_tech_prompt
        tech_sections = []
        for hint in tech_hints:
            prompt = get_tech_prompt(hint)
            if prompt:
                tech_sections.append(prompt)
        if tech_sections:
            parts.append("## Technology Guidelines\n\n" + "\n\n".join(tech_sections))

    # Session context from previous turns
    if session_summaries:
        numbered = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(session_summaries)
        )
        parts.append(f"<session-context>\n{numbered}\n</session-context>")

    parts.append(LOOP_PROTOCOL)

    return "\n\n".join(parts)


def build_iteration_prompt(
    description: str,
    expected_output: str,
    state: LoopState,
    *,
    project_knowledge: list[dict] | None = None,
    max_context_tokens: int = 0,
) -> str:
    """Build the user prompt for one iteration of the loop.

    Assembles <project-knowledge>, <task>, <plan>, <previous-actions>,
    <current-action>, <next-actions>, <expected-output>, and
    <context-budget> XML blocks.
    """
    parts: list[str] = []

    # Smart context summarizer - injects condensed action history
    summarizer = SmartContextSummarizer()
    smart_summary = summarizer.generate_summary(state)
    if smart_summary:
        parts.append(
            "<smart-context-summary>\n"
            f"Loop progress summary:\n{smart_summary}\n</smart-context-summary>"
        )

    # Project knowledge (auto-injected from DB)
    if project_knowledge:
        kb_lines = []
        for f in project_knowledge:
            kb_lines.append(f"- [{f['finding_type']}] {f['topic']}: {f['content']}")
        parts.append(
            "<project-knowledge>\n"
            "Known facts about this project (from previous sessions):\n"
            + "\n".join(kb_lines)
            + "\n</project-knowledge>"
        )

    # Task description
    parts.append(f"<task>\n{description}\n</task>")

    # Plan (if we have one)
    if state.plan.steps:
        parts.append(f"<plan>\n{state.plan.render()}\n</plan>")
    else:
        parts.append(
            "<plan>\nNo plan yet. Create 2-3 concrete steps by calling step_complete "
            "with next_steps operations. You will add more steps as you discover what's needed.\n</plan>"
        )

    # Previous action summaries
    if state.history:
        summaries = []
        for record in state.history:
            summaries.append(f"- [{record.step_index}] {record.summary}")
        parts.append(f"<previous-actions>\n{chr(10).join(summaries)}\n</previous-actions>")

    # Current action
    active = state.plan.active_step
    if active:
        parts.append(
            f"<current-action>\nStep {active.index}: {active.description}\n</current-action>"
        )
    elif state.plan.steps:
        # All planned steps are done — prompt to continue or finish
        parts.append(
            "<current-action>\n"
            "All planned steps are complete. Review what was accomplished against the task requirements.\n"
            "Either add new steps via step_complete(next_steps=[...]) if more work is needed,\n"
            "or call step_complete(status=\"done\", final_answer=\"...\") if the task is fully complete.\n"
            "</current-action>"
        )

    # Next actions (pending steps after current)
    if state.plan.steps:
        next_steps = [
            s for s in state.plan.steps
            if s.status == "pending" and (active is None or s.index > active.index)
        ]
        if next_steps:
            lines = [f"{s.index}. {s.description}" for s in next_steps]
            parts.append(f"<next-actions>\n{chr(10).join(lines)}\n</next-actions>")

    # Expected output
    if expected_output:
        parts.append(f"<expected-output>\n{expected_output}\n</expected-output>")

    # Context budget — inform the agent how much context remains
    # Use last_prompt_tokens (actual context window usage) not total_tokens (cumulative)
    if max_context_tokens > 0 and state.last_prompt_tokens > 0:
        used = state.last_prompt_tokens
        remaining = max(0, max_context_tokens - used)
        pct_used = min(100.0, (used / max_context_tokens) * 100)

        budget_lines = [
            f"Tokens used: {used} / {max_context_tokens} ({pct_used:.0f}%)",
            f"Tokens remaining: {remaining}",
        ]

        if pct_used >= 85:
            budget_lines.append(
                "⚠ CRITICAL: Context window almost full. You MUST wrap up immediately. "
                "Call step_complete with status=\"done\" and a final_answer summarizing "
                "what was accomplished and what remains unfinished."
            )
        elif pct_used >= 70:
            budget_lines.append(
                "⚠ WARNING: Context window running low. Finish the current step, then "
                "call step_complete with status=\"done\". In your final_answer, include "
                "a summary of what was done and list any remaining work as follow-up steps "
                "the user can request in a new conversation."
            )

        parts.append(
            "<context-budget>\n"
            + "\n".join(budget_lines)
            + "\n</context-budget>"
        )

    return "\n\n".join(parts)


def build_tools_prompt_section(tool_schemas: list[dict[str, Any]]) -> str:
    """Render tool schemas as a text section for non-FC models.

    When the model doesn't support native function calling, tool descriptions
    are embedded directly in the system prompt. The model is instructed to
    respond with a JSON object containing a "tool_calls" array.
    """
    lines = [
        "## Available Tools",
        "",
        "You MUST respond with a JSON object containing a \"tool_calls\" array.",
        "Each tool call has \"name\" and \"arguments\" fields.",
        "",
        "When done with the current step, call \"step_complete\".",
        "",
        "Response format:",
        '```json',
        '{"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}',
        '```',
        "",
        "You may call multiple tools in one response:",
        '```json',
        '{"tool_calls": [',
        '  {"name": "read_file", "arguments": {"file_path": "src/main.py"}},',
        '  {"name": "step_complete", "arguments": {"summary": "Read the file", "status": "continue"}}',
        ']}',
        '```',
        "",
        "---",
        "",
    ]

    for schema in tool_schemas:
        func = schema.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"### {name}")
        if desc:
            lines.append(desc)

        props = params.get("properties", {})
        required = set(params.get("required", []))
        if props:
            lines.append("Parameters:")
            for pname, pschema in props.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req_marker = " (required)" if pname in required else ""
                lines.append(f"  - `{pname}` ({ptype}{req_marker}): {pdesc}")

        lines.append("")

    return "\n".join(lines)
