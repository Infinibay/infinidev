# Engine evaluation

How to measure an engine change, and how to read the result without fooling
yourself. Everything here was built or hardened while comparing prompt and
engine variants on `MiniMax-M3`; see
[`../docs/ENGINE_TASK_CLOSURE_ANALYSIS.md`](../docs/ENGINE_TASK_CLOSURE_ANALYSIS.md)
for what the measurements found.

## The corpus

| file | what it is |
| --- | --- |
| `engine_eval_v9.tasks.jsonl` | **scale probe**: one defective leaf behind 8 hubs → 40 packages → 1 200 leaves, built by `bench/build_deeprepo_fixture.py`. Not a cost comparison — a validity probe for whether a shorter prompt still navigates |
| `engine_eval_v8.tasks.jsonl` | the 11-shape corpus: planning, hidden-contract, read-only audit, wide localisation, ambiguity, tool recovery, … |
| `engine_eval_v*.tasks.jsonl` | earlier corpora; keep them, campaigns reference them |
| `agent_task_fixtures/<name>/` | the isolated repository each task copies |
| `agent_task_reference_solutions/<task-id>/` | the solution the preflight requires to pass |

Older corpora stay on disk because a frozen campaign points at one; a new corpus
is a new file plus a matching conditions manifest, not an edit to an existing
one.

### What a task record means

```json
{
  "id": "pricing-rounding",
  "repository_fixture": "pricing_rounding",
  "request": "… the user's words …",
  "verify_command": "{python} verify_contract.py",
  "expected_changed_paths": ["src/pricing.py"],
  "forbidden_changed_paths": ["tests/*", "pyproject.toml"],
  "withheld_paths": ["verify_contract.py"],
  "required_final_patterns": ["round", "test"],
  "required_action_patterns": ["pytest"]
}
```

`withheld_paths` are removed from the agent's workspace and restored only to run
the verifier. Use them when the verifier judges **behaviour**: without it the
model reads its own grader. Do **not** use them when the verifier judges
**wording** — the pilot's `verify.py` files are keyword rubrics, and hiding one
turns the task into guessing the grader's vocabulary.

## Running a comparison

Both arms of an engine comparison run the same *condition*; they differ by a
setting or by the code. One command runs the arms, compares them, and writes
`comparison.md` and `comparison.json`:

```bash
export PATH="$PWD/.venv/bin:$PATH"     # the agent types `python`; give it one
export MINIMAX_API_KEY=…

.venv/bin/python -m bench.agent_task_ab \
  bench/engine_eval_v7.tasks.jsonl \
  bench/engine_eval_v7.minimax.conditions.json \
  bench/agent_task_run.minimax.style-default.r2.json \
  bench/agent_task_run.minimax.style-lean.r2.json \
  --label-a default --label-b lean \
  --output-root bench/runs/<date>-<name>
```

A new arm is a config file, not code:

```json
{
  "provider": "minimax", "model": "MiniMax-M3",
  "model_identity": "minimax:MiniMax-M3",
  "api_key_env": "MINIMAX_API_KEY",
  "repetitions": 2,
  "pipeline_mode": false,
  "prompt_style": "lean",
  "settings_overrides": { "TASK_ENGINE_MODE": "orchestrator" }
}
```

- `repetitions` — **at least 2.** The same task and model vary by up to 6× in
  tokens between runs; one repetition cannot see an effect.
- `pipeline_mode` — `true` calls `run_task`, the same entry point the TUI, the
  CLI and the web server use, so the chat agent, engine selection and the review
  phase are all included. `false` calls `LoopEngine.execute` directly. The
  product's default engine mode can only be measured with `true`.
- `prompt_style` — the style registry (`PROMPT_STYLE`). Applied in-process
  because `./.infinidev/settings.json` outranks `INFINIDEV_*` environment
  variables.
- `settings_overrides` — any `Settings` field, applied for the run and restored.

Interrupting is safe: rerunning the same command reuses the executions already
recorded and skips them. A provider timeout stops the runner by design, and the
row it writes is a diagnostic, not a measurement.

Add `--task-id <id>` (repeatable) and `--split validation` to run one cell instead
of the corpus, which is how a hypothesis gets refuted cheaply before a campaign
is spent on it.

### When the variable is your own code

Neither arm is a setting when the change is in the engine itself: run the config
**before** the edit into one directory, make the change, run the same config into
another, then compare the two observation files directly.

```bash
.venv/bin/python -m bench.agent_task_run \
  bench/engine_eval_v8.tasks.jsonl bench/engine_eval_v8.minimax.conditions.json \
  bench/agent_task_run.minimax.pipeline-orchestrator.r3.json \
  bench/runs/RUN/before/observations.jsonl bench/runs/RUN/before/artifacts \
  --fixture-root bench/agent_task_fixtures --split validation \
  --condition baseline --task-id research-audit

# … edit the engine …

.venv/bin/python -m bench.agent_task_repeated_compare \
  bench/runs/RUN/before/observations.jsonl bench/runs/RUN/after/observations.jsonl \
  --label-a before --label-b after --markdown bench/runs/RUN/comparison.md
```

The one thing that must not happen is editing the tree *between the two arms of a
single run*. The runner is one process that imports the engine once, so an edit
made after it starts changes nothing for either arm — the report would compare
the same code twice and call it a result. Take the "before" observations, stop,
edit, then start a fresh process for the "after" arm.

## Reading the result

`agent_task_repeated_compare.py` prints one table per metric with the paired
sign test. Compare **pairs better / worse**, not the medians: the arms share the
task and the model, so the paired delta carries the signal and the marginal
ranges mostly carry noise.

```
| metric        | median A | median B | delta   | pairs better / worse | sign test p | resolved |
| prompt_tokens |  142850  |   91572  | -51279  | 12 / 4               | 0.0005      | yes      |
```

`resolved` needs p < 0.05 **and** at least 6 pairs that moved. Below that the run
has not measured the effect, whatever the medians suggest.

```bash
# the same comparison, on files already collected
.venv/bin/python -m bench.agent_task_repeated_compare \
  bench/runs/RUN/arm-a/observations.jsonl bench/runs/RUN/arm-b/observations.jsonl \
  --label-a a --label-b b

# the reply the user reads, scored without any model call
.venv/bin/python -m bench.agent_task_answer_quality \
  bench/runs/RUN/arm-a/observations.jsonl bench/runs/RUN/arm-b/observations.jsonl \
  --label-a a --label-b b
```

## Metrics

Counters come from the engine; the rest are derived from each run's `run.json`.

| metric | meaning |
| --- | --- |
| `prompt_tokens`, `completion_tokens` | provider usage, **as the LoopEngine counted it** |
| `pipeline_prompt_tokens` | what the provider actually billed for the whole turn, counted at the provider boundary. `prompt_tokens` is a subset: the chat agent, planner, council, spec elaborator, review engine and task-policy classifier each call the provider themselves, **and so does every team worker's own loop**. Compare engine modes on **this** one, because the extra phases are not distributed evenly across modes |
| `pipeline_completion_tokens` | the same for output tokens |
| `provider_calls` | model round trips in the whole turn. The cleanest "how much work did this take" counter there is, since it cannot be gamed by a mode that hides a phase |
| `aux_prompt_tokens` | the part of the above that the voluntarily-reporting phases named, by lane. Diagnostic only — it has gaps where `pipeline_prompt_tokens` does not |
| `tool_calls` | tool calls the model made |
| `malformed_tool_calls` | calls whose **shape** the model invented: unknown tool, nonexistent parameter, arguments failing their own schema. `malformed_call_reasons` in the artifact holds the call and the reason |
| `max_workless_rounds` | longest streak of model rounds that called no tool. A livelock looks like this and like nothing else |
| `extra_changed_files` | files changed that the task did not declare. Silent for tasks that declare none |
| `changed_lines` | added plus removed diff lines |
| `introduced_placeholders` | `TODO`/`FIXME`/`NotImplementedError` the run added |
| `final_answer_wording` | answers that missed `required_final_patterns`. **Reported, never gating** |
| `answers_without_a_command` | answers naming no command whose result the user could check |
| `latency_seconds` | wall clock |

The provider total comes from a litellm callback the runner registers, not from
asking the engine politely: five of the phases above never report their usage,
and two of them only exist in `pipeline_mode`. Rows written before the callback
existed fall back to `prompt_tokens + aux_prompt_tokens` and still compare.

## Judging the rubrics nobody scored

Every campaign here compares deterministic verifiers, tokens and latency. The
corpus also carries `human_review` rubric items — "the fix corrects the constant
in the stage that is wrong rather than compensating in the pipeline",
"consequential decisions are surfaced with a recommendation instead of silently
decided" — and those are the only place code quality, handoff honesty and
decision ownership are described. They cost nothing to judge after the fact,
because the diffs and the final answers are in every `run.json`.

`agent_task_blind_review.py` makes that judging blind, which is the only way it
is worth anything:

```bash
# 1. packet: strips the arm, assigns opaque ids, shuffles
.venv/bin/python -m bench.agent_task_blind_review packet \
  bench/runs/RUN --output bench/runs/RUN/blind --repetition 0

# 2. read bench/runs/RUN/blind.review.md and write blind.scores.json as
#    {"<id>": {"<item>": 0|1|2}}   (0 not met, 1 partially, 2 met)

# 3. join with the key and compare the arms
.venv/bin/python -m bench.agent_task_blind_review report \
  bench/runs/RUN/blind.key.json bench/runs/RUN/blind.scores.json \
  --markdown bench/runs/RUN/blind.report.md
```

Do not open `blind.key.json` before every score is written. A report that
refuses to run on an incomplete score file is the point, not an inconvenience.

The same tool has a **scorecard** mode that pools every stored run instead of
comparing two arms. It is descriptive — it mixes campaigns, styles and engine
modes — so quote the filtered form when the claim is about the shipped engine:

```bash
# everything ever recorded, per rubric item
.venv/bin/python -m bench.agent_task_rubric_probes bench/runs --scorecard

# only the shipped configuration
.venv/bin/python -m bench.agent_task_rubric_probes bench/runs --scorecard \
  --style lean --engine-mode task --markdown bench/runs/scorecard-shipped.md
```

### What the router costs is measured in isolation, not by campaign

`task_policy_router_cost.py` times `resolve_task_profile` per mode over a corpus
of real requests, and counts the provider usage the classifier's own call makes:

```bash
.venv/bin/python -m bench.task_policy_router_cost \
  bench/engine_eval_v8.tasks.jsonl --repeats 2 \
  --markdown bench/runs/<date>-<name>/router-cost.md
```

It exists because two full campaigns failed to separate the router's cost from
the model's own variance, while the router runs once per turn *before* the loop
and has a median of 4 ms locally against 2,9 s for `preferred`. Prefer this
shape of measurement when the thing you are measuring runs on its own.

## Two rules learned the hard way

**A failure is not a failure until you open its `run.json`.** Four separate
times, a run reported `success=false` because a regex over the final answer
missed a word, while its deliverable had passed the verifier — including two of
the checked-in August pilot's three recorded failures. `verify_exit_code`,
`forbidden_changes`, `withheld_tampering` and `action_pattern_checks` are the
behavioural truth, and `final_answer` is stored, so any run can be re-scored
offline with zero model calls.

**The harness must provide the environment its own tasks describe.** The
recorded campaigns lost 62 tool calls to `fatal: not a git repository` (the
engine's prompt tells the model to review with `git_diff`, and the fixtures were
bare directories), 16 to an orientation tool that errored when nothing was
indexed, and every database write to a missing `projects` row. Those are
invisible in a success count and enormous in a token count.
