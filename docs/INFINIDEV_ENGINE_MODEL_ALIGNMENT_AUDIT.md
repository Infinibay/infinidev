# Infinidev engine and model-alignment audit

## Scope

This audit compares the observed behavioral and comprehension maps of GPT-5.6 Sol,
GLM 5.2, and MiniMax M3 with Infinidev's current architecture. It focuses on the
engine, tool contracts, context delivery, permissions, verification, and missing or
confusing capabilities rather than further system-prompt tuning.

The document also tracks the implementation work derived from the audit. The analysis
remains useful as the rationale; the tracker records which recommendations have already
landed in the working tree and which still remain.

## Implementation tracker

This section tracks implementation against the recommendations derived from the model
mental maps. An item is marked complete only when code and focused verification exist.

Latest full regression: **2,966 passed, 1 skipped** (`uv run pytest -x --tb=short`).

| Priority | Work item | Status | Evidence |
|---|---|---|---|
| P0 | Preserve authority/provenance on planner steps | Implemented | `engine/authority.py`; planner steps default to `model_inferred`; explicit/confirmed user steps alone project to `user_approved`; 49 focused tests passed |
| P0 | Separate user criteria from planner-derived verification criteria | Implemented | `Task.derived_verification_criteria`; distinct XML and reviewer labels; planner checks no longer replace user criteria; 90 focused tests passed |
| P0 | Prevent silent high-impact elaborator defaults | Implemented | Decisions carry risk; only `local_reversible` defaults proceed; costly/external/destructive choices block before planner unless answered; 113 elaborator/pipeline tests passed |
| P0 | Unified effects-based permission broker | Implemented | `ToolEffects`, central dispatch authorization, fail-closed sensitive MCP/Git effects, local-tool defaults, MCP annotation conversion, effects-based serialization, and invalid-mode denial implemented; Git commits require explicit files or `include_all=true` |
| P0 | Workspace baseline and final diff for review eligibility | Implemented | Task-start content baseline reconciles final tracked/untracked state, preserves pre-existing dirty content, detects shell/script/Git/MCP changes, and ignores edit-then-restore; 88 tracking/review tests passed |
| P1 | Dynamic toolsets by phase/capability | Implemented | Deterministic routing reduces the current developer surface from 58 tools/60,279 schema characters to 30/28,830 for a routine task (52% reduction); configured MCP tools remain; `request_capability` adds one omitted group without granting effect permission; 64 routing/context tests passed |
| P1 | Runtime Pydantic validation for every tool | Implemented | Central dispatcher validates final post-hook arguments with each tool's `args_schema`; compatibility regressions passed |
| P1 | Structured tool effects and use constraints | Implemented | Every exposed developer tool carries effects plus structured `use_when`, `do_not_use_when`, preconditions, and common failures; schemas preserve this metadata even when prose descriptions are compressed; MCP constraints are derived conservatively |
| P1 | Evidence review for non-code outcomes | Implemented | Informational results receive an independent claim/evidence gate; blocking allegations must quote the submitted answer; rejected results get bounded, scope-preserving rework; 122 pipeline/review/memory tests passed |
| P2 | Rename ambiguous `help` tool | Implemented | Exposed as `describe_tool`; `help` and `explain_tool` remain compatibility aliases; prompts, hints, fingerprints, and documentation updated |
| P2 | First-class delete/move/patch/preview/rollback tools | Implemented | Recoverable trash deletion, overwrite-safe moves, validated single-write patches, no-write previews, and task-baseline rollback implemented; ignored generated files participate in baseline reconciliation; 94 related tests passed |
| P2 | Consolidate overlapping search and memory interfaces | Implemented | `search_knowledge` now owns browse, FTS, and semantic modes; `search_findings` and `read_findings` are compatibility aliases rather than competing schemas; `recall_context` remains intentionally distinct for raw current-task archive retrieval; 43 focused tests passed |
| P2 | Explicit ContextRank and memory provenance | Implemented | Infinidev ContextRank, Ken blocks, project findings, and recalled records declare source, advisory authority, and no scope effect; traceable notes v2 store claim, source, evidence, confidence, provenance, and validity while reading v1 records; 204 related tests passed |

### MCP ownership decision

The security boundary belongs in Infinidev, not in `~/ken`. Infinidev is the MCP host
that chooses which remote tools become visible, schedules them, and decides whether an
effect requires permission. It now converts standard MCP annotations into conservative
effects and applies host-side permission/use constraints. Ken may publish richer
annotations later, but missing or imperfect server metadata cannot be allowed to bypass
host policy, so no Ken code change is required for this audit to be safe.

### Completion verification

All fourteen roadmap recommendations below now have code, focused regression coverage,
and a passing complete suite. `git diff --check` also passes. The Qwen campaign ledger
was treated as active external output and was not modified by this implementation work.

### Operational finding discovered during implementation

The environment contained 23,290 top-level SQLite temporary database/WAL/SHM files in
`/tmp`, consuming approximately 13 GB and preventing sandbox mounts and file edits. Only
abandoned files older than one hour were removed; campaign ledgers and result artifacts
were not touched. Their producer has not yet been established, so this is tracked as a
database/test lifecycle investigation rather than attributed to the active campaign.

## Executive conclusion

Infinidev has several strong individual mechanisms: exact atomic editing, bounded
command execution, workspace-aware permissions, grounded code review, raw-output
archival, and separate chat/planner/developer roles.

The principal architectural weakness is the loss of provenance and authority between
those roles:

```text
What the user said
        +
what the chat agent inferred
        +
what the spec elaborator assumed or selected
        +
what the planner decided
        ↓
is delivered to the developer as a user-approved plan
```

This amplifies exactly the behavior revealed by the model studies. Sol tends to turn
indirect interest into an operative request, notices ambiguous referents but may broaden
their scope, and often treats conditional or future authorization as actionable. The
engine can currently institutionalize those interpretations instead of containing them.

The next major improvement should therefore be an authority-and-effects layer around
the pipeline, not another general prompt.

## Current pipeline

The default execution path is approximately:

```text
User
  ↓
Chat agent: respond or escalate
  ↓
Spec elaborator: interpret vagueness, resolve gaps, select defaults
  ↓
Planner: create steps and acceptance criteria
  ↓
Developer loop: execute tools one step at a time
  ↓
Reviewer: review when the engine detected file changes
```

The shape is reasonable. The central problem is that the pipeline does not preserve a
strong semantic distinction between:

- An explicit user instruction.
- A user-confirmed decision.
- A convention derived from repository evidence.
- A reversible default.
- An unverified assumption.
- A model-generated implementation choice.
- A product decision still requiring user authority.

## Critical finding 1: model-generated plans become user authority

The chat agent writes its own `understanding`. The spec elaborator can then add scope,
assumptions, defaults, and a design direction. Finally, the planner generates steps and
acceptance criteria.

When the plan enters `LoopEngine`, every planner step is created with
`user_approved=True`. Those steps receive protections intended for work explicitly
requested by the user.

Consequently, "the planner proposed this" becomes "the user approved this." Although
some fields may be refined, the developer cannot treat those steps like ordinary,
discardable model-generated planning.

This is particularly dangerous for Sol because its map shows that it:

- Converts indirect interest into an operative request.
- May broaden an unresolved singular referent to multiple targets.
- Assimilates surrounding wrappers strongly.
- Reports high confidence even when confidence should not authorize execution.

### Recommendation

Every requirement, criterion, and plan step should carry explicit provenance:

```python
authority = Literal[
    "user_explicit",
    "user_confirmed",
    "repo_derived",
    "model_inferred",
    "default_reversible",
    "unverified_assumption",
]
```

Only `user_explicit` and `user_confirmed` items should be frozen as user-approved.

Other items should remain removable or replaceable, and the engine should require
confirmation before an inferred item causes an irreversible, external, destructive, or
architecturally expensive action.

## Critical finding 2: spec-elaborator defaults may be executed unseen

Spec elaboration is enabled for nearly every escalated request longer than 40
characters. It can generate:

- In-scope and out-of-scope declarations.
- Assumptions.
- Product decisions and defaults.
- A selected design direction.
- Rejected alternatives.

A clarification default is explicitly described as the option that will be implemented
if the user says nothing. The planner is then instructed to implement that default and
not block.

However, the user-visible preview is emitted before spec elaboration. The current flow
does not guarantee that defaults and product decisions generated afterward are shown to
the user before execution. A source comment states that the user saw the same default,
but the pipeline does not enforce that invariant.

### Recommendation

Classify unresolved decisions by impact:

- Local and easily reversible: proceed with a declared default.
- Expensive to reverse: request confirmation.
- Destructive, external, or architectural: block until confirmed.
- Settled by repository evidence: proceed and retain the evidence.

The elaborator should be triggered by ambiguity and risk signals, not primarily by
request length.

## Critical finding 3: planner criteria are treated as ground truth

Planner-authored acceptance criteria become the task contract, and the critic and
reviewer are told to treat them as ground truth when they pass the falsifiability filter.

Falsifiability is useful, but it does not establish authority. A criterion may be precise
and testable while still being an invention of the planner.

### Recommendation

Maintain three separate collections:

```text
user_acceptance_criteria
derived_verification_criteria
implementation_quality_checks
```

All three can participate in verification, but only user-authored or user-confirmed
criteria should define contractual completion.

## Critical finding 4: permission coverage is incomplete

The filesystem and shell controls are comparatively strong:

- Workspace edits can be auto-approved.
- Writes outside the workspace require confirmation.
- Shell commands in `auto` mode use a read-only allow-list classifier.
- Headless execution fails closed when confirmation is unavailable.
- Foreground commands have timeouts and cancellation handling.

But not every mutation passes through those controls.

### Git mutations

`git_commit` can run `git add -A` and commit every current change when no explicit file
list is supplied. It does not pass through the centralized command-permission path. This
can capture pre-existing user changes as well as agent-created changes.

`git_branch` can fetch and checkout branches without passing through the same permission
broker.

### MCP mutations

MCP tools classified as writers are excluded from read-only roles, which is useful.
Once exposed to the developer, however, the MCP bridge invokes the server directly.
There is no generic authorization layer for:

- External writes.
- Messages or publications.
- Deployments.
- Paid operations.
- Destructive remote actions.
- Secret-sensitive operations.

When an MCP server omits `readOnlyHint`, classification falls back to mutating verbs in
the tool name. A neutrally named mutating tool could therefore be misclassified.

### Invalid permission modes

Unknown file or command permission modes currently reach a permissive fallback. Invalid
configuration should be rejected during settings validation or fail closed.

### Recommendation

Introduce one permission broker driven by structured tool effects:

```python
ToolEffects(
    reads_workspace=True,
    writes_workspace=False,
    mutates_git=False,
    accesses_network=False,
    mutates_external_state=False,
    destructive=False,
    may_cost_money=False,
    handles_secrets=False,
)
```

Local, Git, shell, web, generation, and MCP tools should all pass through this layer.

## Tool-surface overload

The current configured tool surface was measured locally:

| Role | Tools | Approximate serialized schema size |
|---|---:|---:|
| Chat agent | 30 | 27 KB |
| Planner | 29 | 27 KB |
| Developer | 54 | 46 KB |

This is not only a token-cost problem. It forces the model to make unnecessary tool
selection decisions before solving the task.

Several groups overlap substantially:

- `read_file`, `get_symbol_code`, and `analyze_code`.
- `code_search`, `search_symbols`, and `search_by_docstring`.
- `glob`, `list_directory`, and `project_structure`.
- `search_findings`, `search_knowledge`, `read_report`, and `recall_context`.
- `execute_command` and `code_interpreter`.
- `edit_file`, `rename_symbol`, and `move_symbol`.
- Findings, reports, step notes, and session notes.

The previous execution pilot showed that extra behavioral guidance made Sol use more
tools, tokens, and time without improving the human-reviewed outcome. A large static
toolbox gives that expansiveness somewhere to go.

### Recommendation

Preserve total capability but select tools dynamically by phase:

```text
Code investigation
  → navigation and reading

Implementation
  → reading, editing, and tests

Verification
  → reading, shell/test, and diff

Web research
  → web, findings, and report

Explicit Git operation
  → Git status/diff/commit
```

The model could request a missing capability through a controlled operation such as
`request_capability("web")` or `request_capability("git")`. This would make scope
expansion visible and permission-aware.

## Runtime schemas do not fully enforce tool contracts

The local dispatcher calls tool `_run()` methods directly and primarily validates
arguments against their Python signature. It does not consistently run each local
tool's Pydantic `args_schema` validation before execution.

Providers with reliable structured function calling may enforce much of the schema, but
manual tool mode and imperfect providers cannot be assumed to do so. Constraints such as
minimum lengths, enums, and nested validation should remain true at the execution
boundary.

MCP tools perform internal Pydantic validation in their bridge, so local and remote tools
currently have different runtime guarantees.

### Recommendation

Validate all tool arguments centrally with `args_schema.model_validate()` before calling
`_run()`. Provider-side JSON Schema should improve generation, not be the security or
correctness boundary.

## MCP descriptions discard potentially important semantics

MCP descriptions are compressed to their first paragraph and approximately 300
characters. This produces substantial token savings, but can remove:

- When not to use the tool.
- Side effects.
- Required ordering.
- Destructive behavior.
- Important cross-field constraints.
- A canonical example.

The model maps show strong wrapper assimilation, so tool descriptions are operationally
important.

### Recommendation

Instead of retaining long docstrings, preserve compact structured metadata:

```text
Purpose:
Use when:
Do not use when:
Side effects:
Returns:
Common failure:
```

Effect metadata should come from the server when possible and require conservative user
configuration when absent.

## The `help` interface is confusing

The code already records an observed Qwen failure: it interpreted `help` as Python's
builtin and attempted to execute `help(...)` through Python instead of calling the tool.

An `explain_tool` compatibility alias exists, but the schema still exposes the ambiguous
name `help`, so the alias does not prevent the initial confusion.

### Recommendation

Expose the actual tool as `describe_tool` or `get_tool_help`. Core operations should also
be understandable from their schemas without first spending a tool call on help.

## Missing first-class file operations

The developer lacks clear first-class tools for:

- Deleting a file.
- Moving or renaming a file.
- Applying multiple edits atomically.
- Previewing a complete change set.
- Rolling back only the changes created by the current task.

The model can perform some of these through shell commands, but doing so weakens
permission classification, change tracking, review, and recoverability.

`edit_file` is a strong primitive: exact matching provides safe failures and atomic
writes. It should remain the default for small edits.

### Recommendation

Add:

- `delete_file`.
- `move_file`.
- `apply_file_patch` with multiple validated, atomic hunks.
- `preview_changes`.
- `rollback_task_changes`.

## Review can miss real workspace changes

The review phase runs only when `engine.has_file_changes()` is true. That works for known
editing tools tracked by the engine, but files may also change through:

- `execute_command`.
- Build or generation scripts.
- Git checkout.
- MCP tools.
- New tools not registered with the tracker.

If those mutations are not captured, review can be skipped even though the workspace
changed.

### Recommendation

At task start, record a workspace or Git worktree baseline. At task completion, derive the
actual changed set relative to that baseline.

This should distinguish:

- Pre-existing user changes.
- Agent-created changes.
- Generated files.
- External modifications during execution.

Review eligibility should depend on observed state changes, not on which tool name
purportedly caused them.

## Read-only results lack equivalent evidence review

Tasks without file changes skip code review. This leaves no comparable semantic gate for:

- Web research.
- Architectural analysis.
- Recommendations.
- Reports.
- Complex decisions.
- Claims about repository state.

These tasks do not need code review, but they do need evidence review:

```text
What factual claims were made?
What evidence supports each claim?
Which statements are observations, inferences, or recommendations?
Do conclusions exceed the available evidence?
Are citations valid and applicable?
```

This is particularly useful for models such as GLM, which can be concise and decisive
while filling missing information too confidently.

## Context and memory

The developer rebuilds a fresh system/user conversation on every iteration and includes:

- The task description.
- The plan.
- Full details for the current step.
- Compact summaries of previous steps.
- Project knowledge.
- ContextRank results.
- Session notes.
- New user messages.

This is a sensible way to control context growth. Raw tool exchanges are archived in
working memory before active context is compacted.

The risk is semantic degradation: future steps depend on the model having written a good
summary, remembering to call `recall_context`, and retrieval finding the correct raw
evidence. A summary can accidentally turn uncertainty into fact.

### Recommendation

Store important memory as structured claims:

```text
claim
source
evidence
confidence
provenance
still_valid
```

ContextRank blocks should also be explicitly marked as advisory retrieval suggestions,
not scope or authority. This matters because Sol strongly assimilates wrapper content.

## Parallel tool execution relies on known names

Known writes are serialized and read-only calls may be parallelized, which is correct in
principle. The classification, however, relies on known tool names.

A new mutating MCP tool could therefore be absent from the write set and become:

- Parallelized with another mutation.
- Invisible to file tracking.
- Outside the permission broker.
- Invisible to review eligibility.

Structured effect metadata should drive serialization as well as permissions and review.

## Prioritized roadmap

### P0: before further general prompt optimization

1. Add provenance and authority levels to requirements, steps, assumptions, and criteria.
2. Stop marking every planner-generated step as user-approved.
3. Introduce one effects-based permission broker for filesystem, shell, Git, web, MCP,
   generation, and external actions.
4. Detect changes through a real workspace baseline and final diff.
5. Prevent important elaborator-generated defaults from being silently executed.

### P1: high impact on quality and cost

6. Select toolsets dynamically by phase and requested capability.
7. Enforce Pydantic argument validation at the runtime dispatch boundary.
8. Give tools structured effects and concise use/do-not-use metadata.
9. Add evidence review for non-code outcomes.
10. Separate user criteria from planner-derived verification criteria.

### P2: ergonomics and recoverability

11. Rename the exposed `help` tool.
12. Add first-class delete, move, patch, preview, and rollback operations.
13. Consolidate overlapping search and memory interfaces.
14. Make ContextRank and working-memory provenance explicit.

## Final assessment

The model-mapping work was valuable because it changes where optimization should occur.
Without it, the natural response would be to add more behavioral instructions. The
execution pilot already showed that this can increase cost and activity without improving
the human-reviewed outcome.

The deeper issue is:

> The model interprets; the harness institutionalizes that interpretation.

Sol does not principally need more freedom or more general instructions. Infinidev needs
to preserve the distinctions between intention, inference, evidence, authority, and
permission throughout the complete execution pipeline.

The implementation now places those distinctions in the engine rather than relying on
the model to infer them from prose. Further model-map experiments can refine policies and
defaults, but they no longer need to compensate for missing authority, effect, evidence,
or provenance boundaries in the harness.
