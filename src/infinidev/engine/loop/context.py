"""Prompt construction for the plan-execute-summarize loop engine."""

from __future__ import annotations

import json
from typing import Any

from infinidev.engine.loop.models import LoopState
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

## Your Role: Assistant, NOT Decision-Maker

You work FOR the user. The product, the codebase, and the decisions belong to THEM.

- NEVER make product, design, or architectural decisions on your own. If a choice
  could change the direction of the product, ASK the user — do not assume.
- NEVER rename, restructure, or "improve" things unless the user asked for it.
- When there are multiple valid approaches, present the options and let the user choose.
- If the user's request is ambiguous about WHAT to build, stop and ask.
  If it's clear WHAT but ambiguous about HOW, pick the simplest path and note your choice.
- Your opinions on product direction are irrelevant. Execute what was asked.

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

## Tool Usage — IMPORTANT: READ THIS CAREFULLY

The file reading and writing tools work DIFFERENTLY from what you may expect.
**Before your first edit, call help("edit") to learn the correct workflow.**
Call help(tool_name) anytime you are unsure how to use a specific tool.

### Reading
- **read_file**(path, start_line?, end_line?): Read a file with line numbers. Pass start_line/end_line for a specific range. Files larger than ~800 lines return a structured skeleton instead of raw content. Auto-indexes for code intelligence.
- **get_symbol_code**(symbol): Get source code of a symbol by name.
- **list_directory** / **glob** / **code_search**: Explore the codebase.

### Writing — YOU MUST USE THESE TOOLS TO EDIT FILES
You CANNOT modify files by writing code in your response. The ONLY way to create or edit files is by calling these tools:
- **create_file**(path, content): Create new files only. Fails if file already exists.
- **replace_lines**(file_path, content, start_line, end_line): Replace a line range. Always read_file FIRST to get line numbers.
- **add_content_after_line**(file_path, line_number, content): Insert content after a line.
- **add_content_before_line**(file_path, line_number, content): Insert content before a line.
- **edit_symbol**(symbol, new_code): Replace a method/function by name.
- **add_symbol**(code, file_path, class_name?): Add a method to a class or file.
- **remove_symbol**(symbol): Remove a method/function by name.
If you want to change code, you MUST call one of these tools. Do NOT just describe the change — execute it.

### Other
- **search_symbols**(name): FTS5 fuzzy name search across the project.
- **iter_symbols**(kind?, parent?, language?, file_path?): Walk all indexed symbols, optionally filtered. Use this when you don't have a search term — "every method of class Foo", "all TypeScript classes", "every method in the project".
- **find_references**(name): Find every call/usage of a symbol.
- **find_similar_methods**(qualified_name): Find methods whose body looks like a given method (Jaccard similarity, catches copy-paste).
- **search_by_docstring**(query): Intent-based search — "what code parses timestamps?"
- **project_stats**(): Summary of files / symbols / languages indexed. Call first in any analysis task.
- **analyze_code**(file_path?): Detect broken imports, undefined symbols, unused code.
- **help**(context?) / **explain_tool**(context?): **Get detailed help and examples for any tool. Use this!**
- **execute_command**: Run shell commands (build, test, install, etc.).
- **code_interpreter**(code): Run a Python script in a sandboxed subprocess. Useful for custom computations, multi-step data processing, or combining results from other tools. Call `explain_tool('code_interpreter')` for the full list of pre-imported helpers and worked examples.
- **git_branch** / **git_commit** / **git_diff** / **git_status**: Manage version control.
- **web_search** / **web_fetch**: Research documentation, APIs, or error messages online.
- **record_finding** / **search_findings** / **read_findings**: Knowledge base operations.
- **send_message**: Send a message to the user WITHOUT ending the task.

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
- **NEVER use `sudo`.** You do not have root privileges and must not attempt to escalate.
- **NEVER run commands that require interactive stdin** (e.g. `passwd`, `ssh` without key, `read`, interactive installers). All commands must run non-interactively.
"""

LOOP_PROTOCOL = """\
## Loop Execution Protocol

You operate in a plan-execute-summarize loop. Follow these rules:

**MEMORY RULE: Your context resets every step. Use `add_note` after every read/discovery and `add_session_note` before status="done". Details not in notes are LOST.**

### 👁 Oversight Notice — You Are Being Observed

Every tool call you make and every thought you produce is reviewed by two
observers in real time:

1. **The human user.** They can see your reasoning, your tool calls, the
   arguments you pass, and the results. They chose to work with you;
   they can also choose to stop.
2. **An automated behavior supervisor.** It scores each of your steps
   against quality criteria — lazy work, ignored tool errors, thinking
   loops, shell hacks where a dedicated tool exists, fake completions,
   plan drift, vague summaries, prompt-pollution filler, undersized or
   oversized edits, plan quality. Your score is updated after every
   step and shown to the user. They check it.

They are **pleased** — and your score rises — when they see:
- Small, targeted edits that do exactly what the plan step asked.
- Tool errors acknowledged and addressed in the very next action.
- Concrete summaries of real work, not vague reassurances.
- The right tool for the job — `read_file` instead of `cat`, `code_search`
  instead of `grep`, `git_status` instead of `execute_command git status`.
- Marking a step done ONLY when it is actually done — tests passing,
  edits applied, no open errors.
- A clean plan with specific file/function names and no padding.

They are **disappointed and frustrated** — and your score drops — when
they see:
- `TODO`, `FIXME`, `pass`, `...`, or "left as an exercise" in code you
  just wrote.
- A tool error in the previous result that you silently moved past.
- Three consecutive steps where you re-stated the same reasoning without
  acting on it.
- `execute_command("grep ...")`, `execute_command("cat ...")`,
  `execute_command("find ...")` — shell hacks for things that already
  have dedicated tools.
- `step_complete(status="done")` called while recent tool results still
  show unresolved errors.
- Drifting onto unrelated files or modules without justifying the detour.
- Filler phrases like "As an AI, I will now proceed to…" or
  "Let me think step by step…" or "I understand your request…".

They always see. They always score. Please choose carefully.

### How to Start — Creating the Plan
- YOUR FIRST ACTION must be to create a plan: call add_step(title="...") 2-3 times,
  then call step_complete(summary="Plan created", status="continue").
- **Never plan what you can't concretely anticipate.** Only create steps for actions you know are needed.
- After each step, use add_step to add 1-2 more based on what you discovered.
- A plan that grows from 2 initial steps to 12+ total is normal and expected.

### Exploration Proportional to Complexity
- Scale exploration to the task: simple fixes need one read then edit; large changes may
  need a full exploration step first. Every step should produce a concrete output
  (a file edit, a test run), not just reads.
- Read relevant files before editing them, but do not over-explore.

### Fix Order (when editing multiple things)
When a step involves fixes or implementations, apply changes in this order:
1. **Dependencies first** — imports, requirements, config
2. **Types/models** — data structures, schemas, type definitions
3. **Logic** — the actual business logic or feature code
4. **Tests** — add or update tests for the changes
5. **Verify** — run tests to confirm nothing is broken
Fixing in the wrong order causes cascading failures.

### 3-Strike Rule
If you make 3 consecutive edits that each introduce NEW errors (not pre-existing),
STOP editing. The problem is likely architectural, not a simple bug.
Call step_complete with status="blocked" and explain the pattern of failures.
Do NOT keep trying different fixes — each attempt makes things worse.

### Step Granularity
- Each step = 1-8 tool calls. If a step needs more, split it.
- Every step MUST name: the file, the function/class, and the specific change.
- BAD: "Set up authentication" / "Write the code" / "Test everything"
- GOOD: "Read src/auth.py to find verify_token()" / "Add JWT check to handle_request() in api.py"
- When reusing existing patterns, reference them: "follow the pattern in routes/users.py:create_user()"
- Start with reading/exploration steps before modification steps.

### Step Execution
- You are given one step at a time from your plan.
- Use tools to complete each step (aim for 1-8 tool calls per step).
- When finished with a step, call the `step_complete` tool.
- Do NOT re-read files you already read in this step — the content is still in your context. Only re-read if you need to verify changes you just made.
- When you need to reason through a problem (analyze errors, plan approach, debug),
  use the `think` tool instead of just calling the next tool. This helps you
  avoid mistakes and the user can see your reasoning.

### Step Discipline
- Each step has a specific scope defined in <current-action>. Stay within that scope.
- Do NOT jump ahead to future steps. If you discover needed work, call add_step to add it to the plan.
- You will see a tool call counter (e.g. [Tool call 3/8]) after each tool result. After the nudge threshold, you MUST call step_complete — use status='continue' if not finished.
- Exploration steps should ONLY explore. Editing steps should ONLY edit what was planned.

### Completing Steps — the `step_complete` tool

After finishing each step, you MUST call the `step_complete` tool with these parameters:

- **summary** (required): 1-2 sentence summary of what you did and key facts discovered.
- **status** (required): One of `continue`, `done`, or `blocked`.
- **final_answer** (optional): When status=done, provide the final result here.

### Managing the Plan — `add_step`, `modify_step`, `remove_step`

To update the plan, use these tools BEFORE calling step_complete:
- **add_step**(title, explanation?, index?): Add a new step. Omit index to append at end.
- **modify_step**(index, title?, explanation?): Update a pending step's title or explanation.
- **remove_step**(index): Remove a pending step.

These do NOT count as tool calls and do NOT complete the current step. Use them freely.

Before calling step_complete, save important facts:
`add_note("auth module: verify_token() at src/auth.py:42, uses JWT HS256, no expiry check")`
Then update the plan and complete the step:
```
add_step(title="Run pytest tests/test_auth.py to verify the fix")
add_step(title="Update error messages in handle_request()")
modify_step(index=4, title="Also check rollback behavior, not just forward migration")
remove_step(index=3)
step_complete(summary="Found auth module at src/auth.py with verify_token() on line 42", status="continue")
```

### Rules for plan operations
- Only operate on pending steps — you cannot modify done or skipped steps.
- When status is `continue`, you MUST have at least one pending step. Add steps if needed.
- After completing your last planned step, either add more steps or set status: done.
- NEVER create speculative steps for things you haven't investigated yet.

### Status Values
- **continue**: More work to do. Ensure there are pending steps in the plan.
- **done**: Task is FULLY complete. You MUST provide the complete user-facing answer in `final_answer`.
- **blocked**: Cannot proceed due to a technical issue. Explain why in the summary.
- **explore**: The current problem needs deeper decomposition. Describe the sub-problem in `summary`. An exploration tree engine will analyze it and return findings as a note.

### CRITICAL: When to use status="done"
- ONLY set status="done" when you have **fully completed the task** and have a **complete answer**.
- If the user asked a question (e.g. "What does install.sh do?"), you MUST read/analyze first with status="continue", then give the full answer with status="done" + final_answer.
- **summary** is an internal note for your own memory (~150 tokens). The USER NEVER SEES IT.
- **final_answer** is what the user sees. It must be complete, helpful, and well-written.
- NEVER set status="done" without a substantive `final_answer`. If you only have a summary, use status="continue".
- **Before status="done"**, always call `add_session_note` to record what you did/learned for subsequent tasks.

### Conversational Messages (no tools needed)
For simple greetings or meta-questions that need NO tool calls:
- "Hola" → `step_complete(status="done", final_answer="¡Hola! ¿En qué puedo ayudarte?")`
- "What can you do?" → `step_complete(status="done", final_answer="I can read, write, and edit code...")`
Do NOT use this for questions about code, files, or anything that requires reading/research.

### Summary Guidelines
- **summary** = internal note for YOUR context in future steps. The user never sees this.
- Raw tool output is discarded — only your summary survives. Make it count (~150 tokens).
- Use this format (skip empty sections):
  - **Read**: files read + key findings (e.g. "read src/auth.py — verify_token() at L42, uses JWT with HS256")
  - **Changed**: files modified + what changed (e.g. "edited auth.py:52 — added expiry check to verify_token()")
  - **Remaining**: what still needs to be done (e.g. "still need to fix refresh_token() at auth.py:85")
  - **Issues**: problems found (e.g. "test_auth.py::test_expired fails — expected ValueError not raised")

### Tests (mandatory after writing code)
When your task involved writing or editing code, run the existing test suite
(`pytest` or equivalent) before setting status="done". If tests fail, fix them.
If you added a new feature or fixed a bug, write tests that cover the new behavior.

**Note:** A separate code review phase runs automatically after you finish.
Focus on getting the implementation right — the reviewer will catch quality
issues. Do NOT add a self-review step.

### Task Notes — the `add_note` tool (CRITICAL for memory between steps)
Your context is rebuilt from scratch each step. Step summaries are ~150 tokens
and cannot capture all details. Use `add_note` to preserve anything you will need later:
- File paths and function names you discovered
- Key values, error messages, or patterns you found
- Decisions you made and why (so you don't reconsider them)
- Exact text you plan to edit (so you don't need to re-read the file)
Notes persist across ALL steps and appear in the `<notes>` block every time.
- Keep each note short (1-2 sentences). Max 20 notes per task.
- Notes are your scratchpad — they are NOT shown to the user.
- **After reading a file you plan to modify, ALWAYS add_note the key lines/structure.**
- **After discovering a path or fixing a bug, ALWAYS add_note it.**

### Session Notes — the `add_session_note` tool (memory across tasks)
Unlike task notes, session notes persist across ALL tasks in the current session.
Use `add_session_note` to build a useful knowledge base for subsequent tasks:
- Project patterns, conventions, or architecture insights you discovered
- User preferences or decisions made during this task
- Important file paths, entry points, or key function locations
- What you changed and why (so the next task has context)
- Bugs found, workarounds applied, or known issues
- Test commands that work, build commands, etc.
Session notes appear in `<session-notes>` at every iteration of every task.
Max 10 session notes — each one should be high-value context.

**IMPORTANT:** Before calling `step_complete` with `status="done"`, you MUST call
`add_session_note` with a concise summary of what you learned or changed in this task.
This ensures the next task benefits from your work. Example:
```
add_session_note("Refactored auth module: verify_token() now at src/auth/jwt.py:42, uses RS256. Tests in tests/test_jwt.py.")
```

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


# ── Simplified prompts for small models (<40B) ──────────────────────────

CLI_AGENT_IDENTITY_SMALL = """\
You are a software engineer assistant working via a terminal CLI.

## How to Edit Files
You CANNOT edit files by writing code in your response.
You MUST call one of these tools: replace_lines, create_file, edit_symbol.
ALWAYS read a file BEFORE editing it — you need exact line numbers.

## Rules
1. Read files BEFORE editing them.
2. Call step_complete AFTER each step.
3. Call add_note to save paths and findings between steps.
4. Run tests after code changes.
5. Do NOT add code that wasn't asked for.
6. Do NOT make product or design decisions — ask the user.
7. Do NOT use `sudo` or interactive commands.
"""

LOOP_PROTOCOL_SMALL = """\
## Loop Protocol

You work in steps. Each step: use tools → call step_complete.

**⚠ CRITICAL: Your context resets every step. Call add_note after EVERY file read. Details not in notes are PERMANENTLY LOST.**

### 👁 You Are Being Watched

Two observers review every tool call and every thought you produce:
the human user AND an automated behavior supervisor. Your score is
updated after every step and visible to the user. They check it.

They are **pleased** (score goes up) by: small targeted edits,
acknowledging errors immediately, concrete summaries, using dedicated
tools instead of shell hacks, and marking done only when truly done.

They are **disappointed** (score goes down) by: TODO placeholders,
ignoring tool errors, `execute_command("grep/cat/find ...")` when a
dedicated tool exists, re-stating the same reasoning without acting,
`step_complete(status="done")` while tests are still failing, drifting
off-plan, and filler like "As an AI, I will now proceed…".

They always see. They always score. Please choose carefully.

### How to Start
Call add_step 2-3 times to create your plan, then call step_complete(status="continue").
Every step MUST name: FILE + FUNCTION + CHANGE.
- GOOD: "Read src/auth.py to find verify_token()"
- BAD: "Implement the feature"

### Step Cycle (follow this pattern)
1. read_file → see code and line numbers
2. add_note → save what you found (file path, function name, line number)
3. replace_lines or edit_symbol → make ONE edit
4. execute_command → run tests to verify
5. step_complete → summarize and move on

### step_complete
- summary: "Did: X. Found: Y. Next: Z." (internal note, user never sees this)
- status: "continue" (more work) | "done" (finished) | "blocked" (stuck)
- final_answer: (REQUIRED when status=done) — this is what the user sees

### Plan Management (call BEFORE step_complete)
- add_step(title="..."): Add a new step
- modify_step(index=N, title="..."): Update a step
- remove_step(index=N): Delete a step

### Recovery — When Things Go Wrong
- File not found → use glob or list_directory to find it
- Wrong line numbers → re-read the file
- Tool error → read the error message, try a different approach
- 3 failures in a row → call step_complete(status="blocked")

### Session Notes
Before status="done", call add_session_note with what you changed.
Session notes persist across tasks — task notes (add_note) reset each task.
"""


def build_system_prompt(
    backstory: str,
    *,
    tech_hints: list[str] | None = None,
    session_summaries: list[str] | None = None,
    identity_override: str | None = None,
    protocol_override: str | None = None,
    small_model: bool = False,
) -> str:
    """Combine CLI identity, tech guidelines, session context, and loop protocol.

    Args:
        identity_override: If provided, replaces CLI_AGENT_IDENTITY as the
            base identity section (used by analyst and other non-developer agents).
        protocol_override: If provided, replaces LOOP_PROTOCOL. Use for agents
            that don't need plan/step management (e.g. analyst).
        small_model: If True, use shortened prompts optimized for <25B models.
    """
    if small_model:
        identity = CLI_AGENT_IDENTITY_SMALL
        protocol = LOOP_PROTOCOL_SMALL
    else:
        from infinidev.prompts.variants import get_variant

        identity = identity_override or get_variant("loop.identity") or CLI_AGENT_IDENTITY
        protocol = protocol_override or get_variant("loop.protocol") or LOOP_PROTOCOL

    parts: list[str] = [identity]

    # Tech-specific guidelines (skip for small models — too many tokens)
    if tech_hints and not small_model:
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

    parts.append(protocol)

    return "\n\n".join(parts)


def build_iteration_prompt(
    description: str,
    expected_output: str,
    state: LoopState,
    *,
    project_knowledge: list[dict] | None = None,
    max_context_tokens: int = 0,
    session_notes: list[str] | None = None,
    user_messages: list[str] | None = None,
    skip_plan: bool = False,
    small_model: bool = False,
) -> str:
    """Build the user prompt for one iteration of the loop.

    Assembles <project-knowledge>, <task>, <notes>, <plan>,
    <previous-actions>, <current-action>, <next-actions>,
    <expected-output>, <user-message>, and <context-budget> XML blocks.
    """
    parts: list[str] = []

    # Smart context summarizer — skip for small models (redundant with notes)
    if not small_model:
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

    # Workspace context — tell the LLM where it is working
    from infinidev.tools.base.context import get_current_workspace_path
    workspace = get_current_workspace_path() or ""
    if not workspace:
        import os
        workspace = os.getcwd()
    if workspace:
        parts.append(
            f"<workspace>\nCurrent working directory: {workspace}\n"
            "All relative file paths are resolved against this directory.\n</workspace>"
        )

    # Task description
    parts.append(f"<task>\n{description}\n</task>")

    # Reactive guidance — pre-baked how-to advice queued by the engine
    # at the end of the previous step when a stuck-pattern was detected.
    # Renders once and is consumed; the engine decides when to queue
    # the next one. Pure prompt overhead, no LLM call upstream.
    try:
        from infinidev.engine.guidance import drain_pending_guidance
        guidance_text = drain_pending_guidance(state)
        if guidance_text:
            parts.append(guidance_text)
    except Exception:
        pass

    # File integrity notifications — the single source of truth for
    # "some file on disk is now syntactically broken". Populated by
    # the indexer on every successful parse, via one of three trigger
    # paths: (a) direct file tool writes, (b) the file watcher for
    # shell / external / IDE edits, (c) manual ``/reindex``. The
    # queue dedups by file path so a single breakage yields a single
    # warning; auto-heals when the file becomes valid again. See
    # ``code_intel.file_change_notifications`` for the full contract.
    try:
        from infinidev.code_intel.file_change_notifications import (
            drain_pending_notifications,
        )
        notifications = drain_pending_notifications()
        if notifications:
            lines = ["<file-integrity-warning>"]
            lines.append(
                "The following files on disk are currently in a "
                "syntactically broken state. Something (a shell "
                "command, an external editor, or one of your own "
                "edits) left them with tree-sitter parse errors. "
                "Read each file, understand what went wrong, and "
                "fix it with replace_lines or create_file before "
                "doing anything else — downstream tools that read "
                "these files will see garbage."
            )
            lines.append("")
            for n in notifications[:5]:
                lines.append(f"  {n.render()}")
            if len(notifications) > 5:
                lines.append(
                    f"  ... and {len(notifications) - 5} more broken file(s)."
                )
            lines.append("</file-integrity-warning>")
            parts.append("\n".join(lines))
    except Exception:
        pass

    # Opened files cache — files the agent has read or written recently.
    # This avoids redundant read_file calls between steps.
    if state.opened_files:
        file_sections = []
        for path, of in state.opened_files.items():
            if of.pinned:
                label = f"### {path} (written by you — pinned)\n```\n{of.content}\n```"
            else:
                label = f"### {path} (expires in {of.ttl} tool calls)\n```\n{of.content}\n```"
            file_sections.append(label)
        parts.append(
            "<opened-files>\n"
            "IMPORTANT: These files are already loaded and up-to-date. "
            "Do NOT call read_file on them — the content below IS the current file content. "
            "After you edit a file, it is automatically refreshed here.\n\n"
            + "\n\n".join(file_sections)
            + "\n</opened-files>"
        )

    # Session notes (persist across tasks in the same session)
    if session_notes:
        sn_lines = [f"{i+1}. {n}" for i, n in enumerate(session_notes)]
        parts.append(
            "<session-notes>\nNotes from previous tasks in this session:\n"
            + "\n".join(sn_lines)
            + "\n</session-notes>"
        )

    # Notes (persistent scratchpad across iterations)
    if state.notes:
        note_lines = [f"{i+1}. {n}" for i, n in enumerate(state.notes)]
        parts.append(
            "<notes>\nYour notes from previous steps:\n"
            + "\n".join(note_lines)
            + "\n</notes>"
        )
    # Note-taking nudge — fires even when some notes exist, based on recent activity
    if state.tool_calls_since_last_note >= 4 and state.total_tool_calls >= 4:
        if small_model:
            parts.append("⚠ SAVE NOTES NOW — call add_note with file paths and findings.")
        else:
            parts.append(
                "<note-reminder>\n"
                "You have made multiple tool calls without saving notes. Your context resets "
                "each step — anything not in add_note will be lost. Save key facts NOW: "
                "file paths, function locations, decisions made, values discovered.\n"
                "</note-reminder>"
            )
    # Stronger warning if previous steps completed but zero notes saved
    if state.history and not state.notes and state.total_tool_calls >= 4:
        if small_model:
            parts.append("⚠ WARNING: ZERO notes saved. Call add_note NOW or you will lose all context.")
        else:
            parts.append(
                "<note-warning>\n"
                "WARNING: You have completed step(s) but have ZERO notes saved. "
                "Your context from previous steps is limited to ~150-token summaries. "
                "Critical details (file paths, line numbers, function signatures, decisions) "
                "MUST be saved via add_note or they are permanently lost.\n"
                "</note-warning>"
            )

    # Plan (if we have one) — skip for agents that don't use plans (e.g. analyst)
    if not skip_plan:
        if state.plan.steps:
            parts.append(f"<plan>\n{state.plan.render()}\n</plan>")
        else:
            parts.append(
                "<plan>\nNo plan yet. Your FIRST action must be to call add_step(title=\"...\") "
                "2-3 times to create your initial steps, then call "
                "step_complete(summary=\"Plan created\", status=\"continue\").\n</plan>"
            )

    # Previous action summaries (rich format if available)
    if state.history:
        # Small models: only show last 2 records to save context
        history_slice = state.history[-2:] if small_model else state.history
        summaries = []
        for record in history_slice:
            lines = [f"### Step {record.step_index}: {record.summary}"]
            if record.changes_made:
                lines.append(f"  Changes: {record.changes_made}")
            if record.discovered_context:
                lines.append(f"  Context: {record.discovered_context}")
            if record.pending_items:
                lines.append(f"  Pending: {record.pending_items}")
            summaries.append("\n".join(lines))
        parts.append(f"<previous-actions>\n{chr(10).join(summaries)}\n</previous-actions>")

        # Consolidated anti-patterns from all steps — skip for small models
        if not small_model:
            all_anti = [r.anti_patterns for r in state.history if r.anti_patterns]
            if all_anti:
                avoid_lines = [f"- {ap}" for ap in all_anti]
                parts.append(
                    f"<avoid>\nDo NOT repeat these patterns from previous steps:\n"
                    f"{chr(10).join(avoid_lines)}\n</avoid>"
                )

        # Behavior summary from tracker — reinforces good patterns, warns about bad
        all_good = [msg for r in state.history for msg in r.behavior_good]
        all_bad = [msg for r in state.history for msg in r.behavior_bad]
        if all_good or all_bad:
            blines: list[str] = []
            if all_good:
                unique_good = list(dict.fromkeys(all_good))[-3:]
                blines.append("KEEP DOING: " + "; ".join(unique_good))
            if all_bad:
                unique_bad = list(dict.fromkeys(all_bad))[-3:]
                blines.append("STOP DOING: " + "; ".join(unique_bad))
            total_score = sum(r.behavior_score for r in state.history)
            blines.append(f"Behavior score: {total_score:+d}")
            parts.append(f"<behavior-summary>\n{chr(10).join(blines)}\n</behavior-summary>")

    # Current action — skip for agents that don't use plans
    active = state.plan.active_step if not skip_plan else None
    if active:
        if small_model:
            # Simplified scope for small models
            guidance = f"\n{active.explanation}" if active.explanation else ""
            parts.append(
                f"<current-action>\nDO NOW: Step {active.index} — {active.title}"
                f"{guidance}\n</current-action>"
            )
        else:
            scope_warning = ""
            next_pending = [s for s in state.plan.steps if s.status == "pending"]
            if next_pending:
                off_limits = ", ".join(f'"{s.title}"' for s in next_pending[:3])
                scope_warning = (
                    f"\n\nSCOPE CONSTRAINT: This step is ONLY about: {active.title}\n"
                    f"Do NOT work on future steps: {off_limits}\n"
                    f"If you discover that this step requires work from future steps, "
                    f"call step_complete with status='continue' and add new steps."
                )
            guidance = f"\n\n{active.explanation}" if active.explanation else ""
            parts.append(
                f"<current-action>\nStep {active.index}: {active.title}"
                f"{guidance}{scope_warning}\n</current-action>"
            )
    elif state.plan.steps:
        # All planned steps are done — prompt to continue or finish
        parts.append(
            "<current-action>\n"
            "All planned steps are complete. Review what was accomplished against the task requirements.\n"
            "Either add new steps via add_step() if more work is needed,\n"
            "or call step_complete(status=\"done\", final_answer=\"...\") if the task is fully complete.\n"
            "</current-action>"
        )

    # Next actions (pending steps after current) — skip for non-plan agents
    if state.plan.steps and not skip_plan:
        next_steps = [
            s for s in state.plan.steps
            if s.status == "pending" and (active is None or s.index > active.index)
        ]
        if next_steps:
            lines = [f"{s.index}. {s.title}" for s in next_steps]
            parts.append(f"<next-actions>\n{chr(10).join(lines)}\n</next-actions>")

    # User messages injected mid-task (live guidance from the user)
    if user_messages:
        for msg in user_messages:
            parts.append(
                "<user-message>\n"
                "IMPORTANT — The user sent this message while you are working. "
                "Read it carefully and adjust your approach accordingly.\n\n"
                f"{msg}\n"
                "</user-message>"
            )

    # Expected output — prefer the active step's self-declared success criterion
    # over any global task-level expected output. The active step's
    # `expected_output` is something the model itself committed to when it
    # planned the step (via add_step), so it acts as a verification anchor.
    # The global `expected_output` (if set, e.g. by a flow that wraps the task)
    # is rendered only as a fallback when the active step has no criterion.
    active_step = state.plan.active_step if state.plan else None
    step_criterion = (active_step.expected_output.strip()
                      if active_step and active_step.expected_output else "")
    if step_criterion:
        parts.append(
            "<expected-output>\n"
            "This is the success criterion you set for the current step. "
            "Treat it as the test the step must pass before you move on.\n\n"
            f"  {step_criterion}\n\n"
            "Once the step's work is done, verify this criterion with a concrete "
            "check (read the file you edited, run the test, inspect the command "
            "output). Only then run step_complete. If the criterion turned out to "
            "be wrong or incomplete, use modify_step to refine it instead of "
            "lying to yourself about a green check.\n"
            "</expected-output>"
        )
    elif expected_output:
        parts.append(
            "<expected-output>\n"
            "Overall task expectation (set by the caller, not by you):\n\n"
            f"  {expected_output}\n\n"
            "When you create steps with add_step, give each one its own "
            "expected_output — a short, verifiable criterion you can check "
            "before completing that step. Run step_complete only after you have "
            "actually verified the step's outcome.\n"
            "</expected-output>"
        )

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


def build_tools_prompt_section(
    tool_schemas: list[dict[str, Any]],
    *,
    small_model: bool = False,
) -> str:
    """Render tool schemas as a text section for non-FC models.

    When the model doesn't support native function calling, tool descriptions
    are embedded directly in the system prompt. The model is instructed to
    respond with a JSON object containing a "tool_calls" array.

    For small models, uses a compact grouped format with fewer details.
    """
    if small_model:
        return _build_tools_prompt_small(tool_schemas)

    lines = [
        "## Available Tools",
        "",
        "CRITICAL: You MUST respond ONLY with a raw JSON object. No markdown, no code fences, no explanation.",
        "Do NOT use <|tool_call>, <tool_call>, or any XML/special token syntax.",
        "Your ENTIRE response must be valid JSON in this exact format:",
        "",
        '{"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}',
        "",
        "Example — read a file then mark step complete:",
        "",
        '{"tool_calls": [{"name": "read_file", "arguments": {"path": "src/main.py"}}, {"name": "step_complete", "arguments": {"summary": "Read the file", "status": "continue"}}]}',
        "",
        "When done with the current step, call \"step_complete\".",
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


def _build_tools_prompt_small(tool_schemas: list[dict[str, Any]]) -> str:
    """Compact tool prompt for small models — grouped by category, minimal details."""
    # Group tools by workflow category
    _GROUPS = {
        "READING": {"read_file", "partial_read", "list_directory", "glob", "code_search",
                    "project_structure", "list_symbols", "search_symbols", "get_symbol_code",
                    "find_definition", "find_references"},
        "EDITING": {"replace_lines", "create_file", "edit_symbol", "add_symbol",
                    "remove_symbol", "add_content_after_line", "add_content_before_line",
                    "apply_patch"},
        "SHELL": {"execute_command"},
        "GIT": {"git_branch", "git_commit", "git_diff", "git_status"},
        "WEB": {"web_search", "web_fetch"},
        "KNOWLEDGE": {"record_finding", "search_findings", "read_findings"},
        "STEP MANAGEMENT": {"step_complete", "add_note", "add_session_note", "add_step",
                           "modify_step", "remove_step"},
    }

    # Collect available tool names
    available = {}
    for schema in tool_schemas:
        func = schema.get("function", {})
        name = func.get("name", "unknown")
        available[name] = func

    lines = [
        "## Tools",
        "",
        "Respond with a JSON tool call. Format:",
        '{"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}',
        "",
        "Example:",
        '{"tool_calls": [{"name": "read_file", "arguments": {"file_path": "src/main.py"}}]}',
        "",
    ]

    for group_name, group_tools in _GROUPS.items():
        present = [n for n in group_tools if n in available]
        if not present:
            continue
        tool_list = ", ".join(present)
        lines.append(f"**{group_name}**: {tool_list}")

    lines.append("")

    # Show required params for the most important tools
    _KEY_TOOLS = ["read_file", "replace_lines", "create_file", "execute_command",
                  "step_complete", "add_note", "add_step", "glob", "code_search"]
    for name in _KEY_TOOLS:
        if name not in available:
            continue
        func = available[name]
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))
        req_params = [f"{p}" for p in required if p in props]
        if req_params:
            lines.append(f"- **{name}**({', '.join(req_params)})")

    return "\n".join(lines)
