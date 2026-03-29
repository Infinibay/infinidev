# Fine-Tuning Dataset: Methodology

How to generate training examples for each repo, step by step.
Designed to be interruptible — you can stop at any point and resume later.

---

## Overview

```
For each repo:
  1. Study   → read structure, key files, understand the project
  2. Plan    → decide which scenarios to write (5-8 per repo)
  3. Craft   → write each scenario with full tool call sequences
  4. Validate → review for correctness
  5. Save    → write to finetune/scenarios/{repo}.jsonl
```

Progress is tracked in `finetune/PROGRESS.md`.

---

## File Format

Each repo gets a file: `finetune/scenarios/{repo}.jsonl`

Each line is one complete training example in JSON:

```json
{
  "repo": "flask",
  "lang": "python",
  "type": "bug_fix",
  "prompt": "The user's task description — what they want done",
  "steps": [
    {
      "step_num": 1,
      "description": "Exploration step",
      "tool_calls": [
        {
          "think": "I need to fix a bug in make_response. Let me first learn the editing tools.",
          "tool": "help",
          "args": {"context": "edit"}
        },
        {
          "think": "Now let me read the file to find the bug.",
          "tool": "read_file",
          "args": {"path": "src/flask/sansio/app.py"}
        },
        {
          "tool": "add_note",
          "args": {"note": "Bug in make_response() line 320-355. Missing generator check."}
        },
        {
          "tool": "step_complete",
          "args": {
            "summary": "Read app.py. Found make_response() at lines 320-355. Missing isinstance check for generators before the tuple check.",
            "status": "continue",
            "next_steps": [{"op": "add", "index": 1, "description": "Fix generator handling in make_response and test"}]
          }
        }
      ]
    },
    {
      "step_num": 2,
      "description": "Fix and verify",
      "tool_calls": [
        {
          "think": "The file is in <opened-files> so I don't need to re-read. I'll fix make_response by adding a generator check.",
          "tool": "edit_symbol",
          "args": {
            "symbol": "Flask.make_response",
            "new_code": "    def make_response(self, rv):\n        if isinstance(rv, Generator):\n            rv = Response(rv, content_type='application/octet-stream')\n        ...",
            "file_path": "src/flask/sansio/app.py"
          }
        },
        {
          "tool": "execute_command",
          "args": {"command": "cd /path/to/flask && python -m pytest tests/test_basic.py -x -q 2>&1 | tail -20"}
        },
        {
          "tool": "step_complete",
          "args": {
            "summary": "Fixed make_response() — added generator isinstance check. Tests pass.",
            "status": "done",
            "final_answer": "Fixed the bug in `Flask.make_response()`. Added a check for generator objects before the tuple check. Generators are now wrapped in a Response with `application/octet-stream` content type."
          }
        }
      ]
    }
  ]
}
```

---

## Rules for Each Tool Call

### think (before most tool calls)
- Write as if talking to the user
- Say WHAT you're doing and WHY
- 1-3 sentences
- MUST use before first edit, after reading a file, after test failure
- Can omit for trivial follow-ups (add_note after read, step_complete)

### help("edit")
- MUST appear in step 1 of every example that involves editing
- Always first tool call of the step

### read_file
- MUST appear before any edit to the same file
- Use real file paths from the repo
- NEVER read a file that's already in <opened-files>

### edit_symbol / replace_lines / add_content_after_line / etc.
- **The code MUST be real, correct, and complete**
- Use actual function signatures, class names, imports from the repo
- Indentation must be correct
- Newlines as `\n` in JSON strings

### execute_command
- Command must be a string, not a list
- Use `cd /path && command` pattern
- Always `2>&1 | tail -20` for test commands
- Use the repo's actual test command (pytest, cargo test, npm test, go test, etc.)

### add_note
- Use after discovering key info (line numbers, function names, decisions)
- 1-2 sentences

### step_complete
- MUST end every step
- Summary: ~150 tokens, structured (Read/Changed/Remaining/Decisions)
- Status: continue (more work) or done (task complete)
- final_answer MUST be present when status=done

---

## Scenario Types to Include

For each repo, aim for 5-8 scenarios covering a MIX of these:

| Type | Description | Min per repo |
|------|-------------|-------------|
| `bug_fix` | Fix a realistic bug | 1 |
| `add_method` | Add method to existing class | 1 |
| `add_function` | Add standalone function to file | 1 |
| `modify_method` | Change behavior of existing method | 1 |
| `refactor_rename` | Rename symbol project-wide | 0-1 |
| `refactor_move` | Move symbol to another file | 0-1 |
| `add_test` | Write tests for existing code | 1 |
| `config_change` | Modify config/setup file | 0-1 |

---

## Step-by-Step Process for Each Repo

### Step 1: Study the repo

Read these files in order:
1. `README.md` or top-level docs — what the project does
2. Directory structure — `ls` or `tree`
3. Main source directory — find the entry point
4. 2-3 key source files — understand patterns, naming, style
5. Test directory — understand test patterns

**Write a brief summary** of what you learned (used to inform scenario creation).

### Step 2: Pick scenarios

Based on what you read, decide 5-8 specific scenarios:
- Each must reference REAL files, REAL symbols, REAL line numbers
- Each must be doable in 2-3 steps (5-15 tool calls total)
- Vary the types (don't do 5 bug fixes)
- Make them realistic for the project's domain

### Step 3: Craft each scenario

For each scenario, write the complete JSON entry:
1. Write the user prompt (clear, specific, like a real user would ask)
2. Write step 1 (exploration): help → read_file → add_note → step_complete
3. Write step 2+ (action): think → edit → test → step_complete
4. Verify:
   - All tool calls have correct params
   - Code in edits is syntactically valid
   - File paths and line numbers are real
   - JSON strings have proper escaping (\n, \", etc.)
   - think() messages are informative
   - step_complete summaries follow the format

### Step 4: Save

Append each scenario as a line to `finetune/scenarios/{repo}.jsonl`

---

## Progress Tracking

`finetune/PROGRESS.md` tracks which repos are done:

```markdown
## Progress

| Repo | Lang | Status | Scenarios | Notes |
|------|------|--------|-----------|-------|
| flask | python | done | 6 | |
| httpx | python | in_progress | 3/6 | stopped at add_test |
| click | python | pending | - | |
| ... | ... | ... | ... | ... |
```

Status: `pending` → `in_progress` → `done`

When resuming, check PROGRESS.md to know where you left off.

---

## Language-Specific Notes

### Python
- Test command: `python -m pytest tests/ -x -q 2>&1 | tail -20`
- Import style: `from module import Class`
- Indent: 4 spaces

### TypeScript
- Test command: `npm test 2>&1 | tail -30` or `npx vitest run 2>&1 | tail -30`
- Import style: `import { Thing } from './module'`
- Indent: 2 spaces

### Rust
- Test command: `cargo test 2>&1 | tail -30`
- No classes — use `impl` blocks, structs, traits
- Use `pub fn` for public, `fn` for private
- Indent: 4 spaces

### C
- Test command: `make test 2>&1 | tail -30` or project-specific
- No classes — use structs + functions
- Header files (.h) for declarations, source (.c) for implementations
- Indent: varies (check project style)

### Go
- Test command: `go test ./... 2>&1 | tail -30`
- No classes — use structs + methods with receivers
- Public: uppercase first letter. Private: lowercase.
- Indent: tabs

### Java
- Test command: `mvn test 2>&1 | tail -30` or `gradle test 2>&1 | tail -30`
- Always classes
- Indent: 4 spaces

---

## Quality Checklist (Per Scenario)

Before saving, verify:

- [ ] User prompt is clear and specific (like a real user)
- [ ] Step 1 starts with `help("edit")` if editing will happen
- [ ] Files are read before they're edited
- [ ] No re-reading of files in <opened-files>
- [ ] think() messages are informative and concise
- [ ] Code in tool calls is syntactically valid
- [ ] File paths are real paths in the repo
- [ ] Line numbers are real (checked against actual file)
- [ ] step_complete has correct status + summary
- [ ] final_answer is present when status=done
- [ ] JSON is properly escaped (no raw newlines, quotes escaped)
- [ ] Total tool calls per step is 1-8
- [ ] Scenario is 2-3 steps total
