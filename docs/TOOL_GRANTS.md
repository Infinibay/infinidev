# Tools for one principal request

Interactive classic and TUI accept one atomic command:

```
/tools run read_file,create_file,edit_file -- Create the private candidate described in brief.md.
```

`/tools` lists actual available names. The command grants only that selection for
this request, including its hook continuation. The next ordinary request uses
defaults. There is no persistent session setting or pending next-selection slot.
An empty selection (`/tools run -- question`) leaves only runtime tools.

`finish_turn` and `update_plan` are always fresh runtime-owned tools; the example
therefore offers five tools. Names are exact and case sensitive: `write_file` is
not an alias for `create_file` here. Unknown/duplicate/ambiguous names, legacy
pseudo-tools and unsupported principal modes are rejected. No tool is enabled
outside the currently available effective agent catalog.

The body after the first ` -- ` delimiter is preserved, including spaces,
Unicode, newlines and further delimiters. It is never re-dispatched as a local
`/command` or `!shell` command. TUI uses staged `/attach` items without detecting
or removing attachment paths from this body. During context admission the task,
attachments and selection stay together; a second submission cannot overwrite
it. Cancel task discards the pending request. Old modal callbacks cannot launch
a newer pending task. An active task cannot change its selection through this
command or injected guidance.

SDK callers use the existing runtime vocabulary:

```python
from infinidev.engine.turns.catalog import select_tool_names

selected = select_tool_names(agent.tools, ("read_file", "create_file", "edit_file"))
run_task(..., task_tools=selected)
```

`task_tools=None` preserves defaults; an empty sequence grants no workspace tools.
Non-runtime entries must be the canonical instances (`is`) in the effective agent
catalog, not replacements with equal names. Membership/names are snapshotted at
parent admission and checked before each child. Mutating the original list or
agent catalog cannot widen that explicit grant. Renaming selected tools fails.
The tool internals are not deeply frozen. Runtime-reserved replacements are
ignored and rebuilt using the current child state.

Explicit grants currently require the ordinary principal (task mode, no phase
engine, autonomous mode or gather). `--prompt` does not accept this control
command and rejects it before bootstrap; use an interactive frontend or SDK.
Compatibility/default paths retain their catalogs when no override is supplied.
With an explicit grant, eligibility and names are resolved at admission and
cannot be widened by hooks. Without an override, existing hook-sensitive mode
selection and gather behavior remain unchanged. TUI/classic carry the frozen
names from accepted input, including across an admission dialog and worker start.

The same selected catalog supplies native/manual schemas and dispatch validation.
An ungranted tool request rejects its whole batch before execution, including
otherwise valid calls. Permission checks still apply to granted tools. Rejected
violations do not mean a programming task succeeded, and this feature does not
certify model claims or semantic progress.

A rejected exact name reports `tool_not_offered` at `tool_name`, separately from
argument/schema errors. Feedback includes `executed=false`, `requested_name` and
`offered_names` from that request's dispatch snapshot. Long diagnostics carry
explicit completeness flags and the total offered count; the full schemas remain
the authority. No prefix, alias or spelling is automatically corrected. A retry
uses the existing budgets and three consecutive rejected-batch limit. Structural
archive diagnostics retain names/types, never argument values or reasoning.

Generated conversation-history guidance recommends `chat_history_search` only
when the explicit grant includes it. Archive envelopes retain their exact
reference or title but recommend `recall_context` only when it is offered. Prior
host envelopes are projected for the current catalog without rewriting saved
references, user messages, notes or source text. With no explicit override the
history renderer keeps its existing behavior; archive guidance follows the actual
runtime catalog. Evidence views remain bounded to 500 characters each; oversized
or previously clipped host envelopes are explicitly incomplete neutral excerpts,
not executable complete references. Loading a view does not shorten the saved
original reference. Project instructions, human history, custom agent identities and
tool-authored output are not scanned or rewritten to remove tool names.

A tool selection is not a path or OS sandbox. A granted tool can still reach its
ordinary permitted paths and effects; user-installed preparation hooks and
separate agents are not sandboxed by this model-tool catalog. No automatic
approval, permission setting, intent classifier, regex authorization, new DB
schema or live configuration change is introduced.
