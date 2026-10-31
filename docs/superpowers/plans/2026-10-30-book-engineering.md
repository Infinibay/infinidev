# Book-Derived Engineering Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Turn the useful book-skill ideas into scoped engineering guidance and
prevent partial instructions from reaching the model.

**Architecture:** Reuse stable engineering judgment, existing role/phase task
methods and opt-in capability resources. Conditional composition treats instruction
bodies as indivisible; runtime signatures and policy routing stay unchanged.

**Tech Stack:** Existing Python, standard library, packaged JSON and pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-book-engineering-design.md`

## Global Constraints

- Preserve previous local changes and the existing checkout/branch; no commits.
- All commands use `uv run --frozen --no-sync`; no dependencies or remote models.
- No new settings/capabilities, no user's profile changes, no classifier changes.
- Keep `bugfix.developer@3` and `refactor.developer@1` text and approval unchanged.
- Increment versions of modified conditional method fragments.
- Public signatures, role/phase filters, authority and budget meaning stay unchanged.

## Review Focus

- A block with its safety clause at the end must be omitted rather than clipped.
- Multibyte text fitting an exact cap must arrive intact; one byte less omits it.
- Read-only authority and optional-capability defaults survive richer guidance.
- Existing model-specific approvals never authorize changed method content.
- More guidance does not silently omit a built-in role method at default budget.

### Task 1: Atomic conditional instruction bodies

**Files:** `src/infinidev/engine/prompt_composition.py`,
`src/infinidev/engine/task_policies/rendering.py`;
create `tests/test_atomic_prompt_fragments.py`.

**Interfaces:** Existing `select_conditional_fragments(...)` and
`select_conditional_catalog(...) -> ConditionalPromptSelection`; existing
`render_task_policy_layer(...)` and `render_all_conditional_task_policy_layer(...)`.

- [x] RED: synthetic body ending in an essential final condition must be omitted
      with `fragment-budget` when too long, in selected and catalog modes.
- [x] Cover exact UTF-8 boundary, empty text and a later short block still selected.
- [x] Cover both renderers with complete literal body, escaped conditions and
      ordinary budget omission; no partially rendered body may appear.
- [x] Remove `_clip_utf8` and renderer slicing; budget full text, preserve body bytes.
- [x] GREEN: run new atomic tests plus prompt-composition and task-policy suites.

### Task 2: Scoped book-derived engineering judgment

**Files:** `src/infinidev/prompts/shared.py`, `flows/develop.py`,
`variants/lean.py`, `variants/generalized.py`,
`src/infinidev/engine/task_policies/fragments.py`, and existing capability JSONs
08, 10, 12, 13, 14, 15 and 36. Tests use existing prompt composition/profile/style
and routing suites plus catalog integration coverage in Task 1's test file.

**Interfaces:** Existing `ENGINEERING_JUDGMENT`, `get_develop_identity(...)`,
`TASK_METHOD_FRAGMENTS`, `OPTIONAL_CAPABILITIES`; no new runtime APIs.

- [x] Add concise shared ownership/vocabulary/invariant guidance while preserving
      authority, outcome, UI, communication and continuity rules.
- [x] Replace blanket mocking and mechanical splits with cohesive responsibilities,
      observable contracts and real boundary checks where the boundary is the risk.
      Preserve original security/scope requirements and lean compactness.
- [x] Refine planner/reviewer bugfix/refactor and feature/performance methods using
      the spec's role responsibilities. Version changed fragments, leave approved
      developer fragments unchanged, and keep methods under individual caps.
- [x] Refine seven existing opt-in resources with failure, recovery, source-of-truth,
      stale projection, invariant, compatibility and characterization guidance.
- [x] Verify actual assembled prompts for every style and role, capability opt-in,
      complete catalog inclusion, read-only routing, version/approval gates and
      cache prefix. Avoid tests that merely freeze instructional wording.
- [x] Run related tests, independent review and `uv run --frozen --no-sync pytest`.

## Execution ledger

- Baseline: 173 tests passed in 6.68s.
- Workspace: existing `improve/agent-judgment-and-verification` checkout, continuing
  the user's authorized in-place work and preserving all earlier uncommitted edits.
- Pre-flight: Task 2 method bodies consume Task 1's unchanged fragment interface;
  full-text selection requires all edited bodies to fit their existing 900-byte cap.
- Ruling: prompt prose changes use existing composition/profile/style coverage and
  a manual contract audit, not substring tests pretending to prove model behavior.
  No empirical task-quality gain is asserted without a model evaluation.

- Builder check: nothing important left.
- Task 1 complete: seven regressions failed for clipped selection/rendering and
  whitespace accounting; all 16 atomic composition cases now pass. Combined
  composition/task-policy/atomic tests: 71 passed in 3.60s.
- Ruling: the layer is XML-like prose (its protocol literally names <if>);
  tests parse only actual prompt-fragment blocks, not the explanatory protocol.
- Task 2 complete: ten method bodies versioned, seven opt-in resources refined,
  full/lean/generalized construction guidance aligned. Approved developer bodies
  and all routing/filter/authority fields match the saved baseline exactly.
- Related suite: 868 passed in 17.00s; static method audit found all bodies below
  their 900-byte cap and every role catalog complete within default budget.
- Independent review: no Critical or Important findings; reviewer independently
  ran the 71 atomic/composition/task-policy checks successfully.
- Final: minor (deferred): add exact raw rendered-body/hash assertions with
  surrounding whitespace and multibyte content; current renderer is correct,
  but the XML semantic check strips body whitespace and does not pin the hash.
- Final: Ruling: outer XML wrapper budget accounting is unchanged and outside
  this round — existing behavior preserved deliberately; no exact wire-budget
  claim is made. Changing it would require consumer/budget compatibility work.
- Final: Ruling: all-conditional mode bypasses evidence gates as before; selected
  mode retains approved model/version checks. Do not silently change that mode.
- Final: Ruling: model-quality effects cannot be inferred from unit tests. Report
  changed guidance and verified runtime contracts, not a measured task success gain.
- Final full suite: 5473 passed, 13 skipped, 11 pre-existing Pydantic serializer
  warnings in 182.92s; process exit 0. Sixteen new composition regressions.
- `git diff --check` passed on all tracked files touched in this round.
- No commits, pushes, dependency changes or modifications to personal skills.


## Follow-up: confirm active application and close review coverage

- User explicitly requested implementation/application again. Confirmed actual
  composed developer system prompt under the current configuration: lean style,
  shadow disabled, all seven conditional methods present, updated common judgment
  and cohesive core present. Optional capabilities remain at their configured opt-in
  defaults; no user profile changed.
- The previously deferred minor is resolved: renderer tests now compare exact raw
  payload and fixed expected SHA-256 for plain and whitespace/multibyte bodies in
  both selected and catalog modes.
- Mutation checks run only in separate Python processes (no production edits):
  stripping rendered whitespace caused two expected failures; replacing emitted
  hashes with zeros caused four expected failures. Real source is unchanged.
- Related source/composition checks: 75 passed. Final full suite: 5477 passed,
  13 skipped, 11 pre-existing warnings in 189.38s; exit 0.
