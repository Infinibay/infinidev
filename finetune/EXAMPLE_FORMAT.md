# Training Example Format v2

Each example is a complete multi-turn conversation in JSONL.

## Structure

```json
{
  "repo": "flask",
  "lang": "python",
  "type": "bug_fix",
  "prompt": "User's task description",
  "turns": [
    {"role": "user", "content": "<task>\n...\n</task>"},

    {"role": "assistant", "content": "<tool_call>...</tool_call>"},
    {"role": "tool", "content": "[tool result — real content]"},

    {"role": "assistant", "content": "<tool_call>...</tool_call>\n<tool_call>...</tool_call>"},
    {"role": "tool", "content": "[result 1]\n---\n[result 2]"},

    ...repeat...
  ]
}
```

## Rules

1. Every assistant turn contains one or more `<tool_call>` blocks
2. Every assistant turn is followed by a `tool` turn with real results
3. `think()` calls appear as tool_calls but their result is empty (user just sees the reasoning)
4. `step_complete` results are empty (engine processes internally)
5. `add_note` results are "Note saved."
6. `read_file` results contain the ACTUAL file content with line numbers
7. `execute_command` results contain realistic test output (pass/fail)
8. `edit_symbol`/`replace_lines`/etc results contain the JSON success response
9. File content in results should be truncated to relevant portions (200-300 lines max)
10. Each example should have 8-20 turns (4-10 assistant turns)

## Turn Types by Phase

### Investigation Phase (steps 1-3)
- help("edit") → help text
- read_file → real file content
- code_search → real search results
- find_references → real reference list
- list_directory → real directory listing
- think → reasoning about what was found
- add_note → save key findings
- step_complete(status="continue") → plan next steps

### Execution Phase (steps 4-8)
- think → explain what you'll change and why
- edit_symbol/replace_lines/add_content_* → success JSON
- execute_command("pytest...") → test output (may fail!)
- If tests fail: think → analyze error → fix → retest
- step_complete(status="continue") → next step

### Verification Phase (steps 9-10)
- execute_command("pytest...") → all tests pass
- git_diff → show changes made
- step_complete(status="done", final_answer="...") → complete summary
