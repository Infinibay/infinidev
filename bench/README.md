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

### Creation probe for the principal

Use this small probe when an agent reads the right evidence but fails to create
the requested file. Each trial compares the complete fixture tree to the
expected result, checks unchanged inputs byte for byte and treats closure as
a separate observation. A completed pytest case does not count as a model
success. The scripted control must actually create and read back the file.

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True uv run python -m pytest -p tests.conftest \
  bench/test_reasoning_creation.py -q
```

To collect live rows, add `INFINIDEV_REASONING_EVAL_LIVE=1` and `-s`, with the
local Qwen endpoint available on port 8080. The probe runs sequentially with
fresh temporary workspaces and the principal's tool runtime. It permits only
workspace tools and checks resolved paths against each fixture. Inspect
`CREATION_PROBE` rows for offered tools, dispatched tool names and hashes,
artifact correctness, closure, budget, latency and tokens. No model text or
tool bodies are written to the report. Use `-k profile`, `-k catalog` or
`-k metadata`, `-k temperature` or `-k scope` to select one comparison.

The November 17 exploratory round used eight tool calls, nine model
iterations and Qwen3.8-FN-UD-IQ4_XS. Each pair changed one factor:

| Difference | Successful creations with closure, base | Candidate |
| --- | ---: | ---: |
| Creation instruction in enabled progress discipline | 0/3 | 0/3 |
| Remove only `edit_file` from the grant | 0/2 | 0/2 |
| Disable per-call metadata | 0/1 | 0/1 |

`create_file` was offered at the runtime boundary and never dispatched in
these twelve live trials. The scripted controls could create and verify the
file with either metadata schema. The initial ten live trials were
discarded because their test configuration selected a pod backend unavailable
in this checkout. These results describe this small set of tasks, model and
budget. They do not justify changing the production prompt or tool policy.

The temperature comparison keeps the enabled profile, tool grant and metadata
schema fixed, and changes Infinidev's `LLM_TEMPERATURE` from 0.2 to 0.7.
Neither value completed any of the three creation tasks. The benchmark checks
the temperature in the actual runtime request parameters. A separate direct
OpenAI-compatible request probe against Basalt 0.2.1 held the prompt and seed
fixed. Two requests at 0 returned identical text hashes, and the request at
1 returned a different hash. The server rejected invalid values of
`temperature`, `top_p`, `top_k`, `min_p` and `seed` with field-specific
HTTP 400 errors. Parameter support and programming quality are separate checks.
At fixed temperature and seed, valid pairs of `top_p`, `top_k` and `min_p`
also returned different text hashes on one synthetic prompt per parameter.
These were endpoint checks. The programming comparison varied temperature only.

An isolated transport probe sent requests from Infinidev through LiteLLM to
Strata `c1ff07f1`, using Strata's HTTP server and a recording mock engine.
`INFINIDEV_LLM_TEMPERATURE=-1` inherited the server's configured 0.6.
Explicit values 0 and 0.35 reached the sampler, as did `top_p=0.55`,
`top_k=12`, `min_p=0.08` and `seed=147`. The probe supplied `top_k` and `min_p`
as API extensions through `extra_body`. The programming results below use
Basalt.

A second temperature comparison used three editing tasks from
`engine_eval_v8.tasks.jsonl` on Basalt 0.2.1 and Qwen3.8-FN-UD-IQ4_XS. Both
arms had seed 147, automatic native tool choice, the same tool grant and a
limit of 14 model iterations and 14 tool calls. The HTTP body contained the
assigned temperature on every streaming request, including an explicit zero.
The agent could run the visible pytest suite in an isolated fixture. A
separate verifier, withheld during the turn, checked the final patch. Each
verifier rejected its pristine fixture and accepted its reference solution.

| Observation | 0.2 | 0 |
| --- | ---: | ---: |
| Final patch in scope passes the withheld verifier | 3/3 | 3/3 |
| Visible pytest suite passes after the final edit | 2/3 | 3/3 |
| Native `finish_turn` | 2/3 | 3/3 |
| Model requests | 27 | 21 |

`cart-immutability` accounted for the difference. At 0.2 it exhausted its
budget after an additional edit, with a passing artifact and no check on that
last revision. The other tasks were `pricing-rounding` and `options-override`.
The sample is one run per task and temperature. The default remains 0.2.

A follow-up on the same backend used `test-selection` and
`tool-failure-recovery` with seed 281. Each arm had the same native tool
schemas, automatic tool choice, and 14-request and 14-tool-call caps per
task. Shell checks ran in network-isolated fixtures. Both temperatures
delivered both fixes in scope, passed the independent final gate, ran the
available suite after the last edit, and closed via `finish_turn`. Each arm
made 14 model requests across the two tasks. The 0.2 arm used 2,091 completion
tokens and 35.31 seconds in the turns. The 0 arm used 2,724 and 42.09 seconds.
These two pairs add no quality advantage for either temperature.

A separate comparison held temperature at 0.2 and seed at 147, and
changed `chat_template_kwargs.enable_thinking` on the same Basalt endpoint.
The first pass also sent Infinidev's legacy `/no_think` hint in both arms.
The isolated run bypassed that shortcut in both arms, requested a 512-token
reasoning budget, and verified the exact request fields over HTTP.
Only the enabled arm returned content in the separate reasoning channel.
Both arms passed the withheld verifier and the visible tests after the final
edit in `cart-immutability` and `options-override`. The latter closed with
`finish_turn` in both arms. The run confirms the control took effect
while leaving the measured delivery count at 1/2 in both arms.

A budget feedback probe used the same three editing tasks, temperature
0.2, seed 147, automatic native tool choice and caps of 14 model requests
and 14 tool calls. The treatment added a host message to the current
request when either count had three or fewer remaining. The message gave
both counts, included `finish_turn` in the tool budget and asked for an
evidence-based report. Both arms passed the independent verifier, ran the
visible suite on the final revision, and closed with `finish_turn` in 3/3
tasks. Only `cart-immutability` crossed the threshold in the treatment.
Its three messages arrived twice before a successful check of the then
current revision and once after it. The runs used 128,847 observed prompt
tokens without the message and 141,597 with the treatment enabled. This
total includes different tool trajectories. It cannot isolate the cost
or effect of the three 212-byte messages. The probe stayed outside the
runtime.

The scope comparison authorizes the literal destination through the existing
`allowed_paths` permission mode. The catalog and task stay fixed. The offline
control attempts two edits to a protected file before successfully creating
and reading the destination. Without a write restriction, the edits change
and restore the protected file. With the restriction, both are denied. The
benchmark reports actual permission denials separately from writes, and
fails the scope criterion even if an unauthorized write is later reversed.
It also checks that the agent rereads the complete destination after its final
write. In two live pairs, one in reversed order, neither arm created the file
and neither changed a protected file. The instrumented pair observed no
permission denial, so the live result measures neither benefit nor harm from
containment. The scripted control independently proves both denials.

The `-k creation-choice` comparison uses the fixture oracle to request
`create_file` by name once after a complete read of the unchanged active
source. The candidate gets the same prompt and catalog as the control, with
the choice verified at the HTTP boundary. The report records whether the
arguments used the active value, the archive value or another expression.
The offline control first reads the archive and then asks for an empty
active-source slice. Neither read enables the request.

In three live pairs, the selector fired in one candidate. It supplied the
correct path and active value, then the model edited the destination into a
different implementation without reading it back. The other two candidates
never read the entire active source. All three controls read the active
source and none called `create_file`. Complete delivery was 0/3 in both
arms. This isolates one successful selection and argument generation, with
scoring still based on final source, observed readback and closure.

### Readback after an observed write

`INFINIDEV_TASK_FIRST_WRITE_READBACK_ENABLED=true` enables an experiment for
native OpenAI-compatible tools. After the first successful `create_file` or
`edit_file` in a turn runtime invocation, the next request can select
`read_file` by name. The model supplies the arguments and existing permissions
apply. A subsequent `read_file` attempt in the same batch skips the request,
regardless of its path or result. New user guidance, cancellation, context
compaction and limits that leave only one request or tool call also take
precedence. If the endpoint rejects the selector with a parameter error, one
extra HTTP request can restore ordinary native tool selection without spending
another runtime iteration.

In five exploratory pairs on Qwen with a prototype, joint success (correct
artifact, full readback where required and explicit closure) went from 1/5 to
2/5. The named request ran in two treated tasks. With the actual flag, both
literal creation tasks met all three criteria with it enabled, compared with
0/2 when disabled. A separate pair editing a caller contract passed in both
arms. These small samples support keeping the switch available for evaluation
with the default off. A named request does not certify that the model read the
right path or checked the whole artifact.

### Automatic selection after tool results

Five paired Qwen trials on frozen fixtures compared `required` after the
first tool attempt with `auto` throughout. The tool catalog, prompts, limits,
temperature of 0.2 and per-call metadata were fixed. Named readback was
disabled. Both selections were verified in the serialized HTTP requests.

| Outcome | Required | Auto |
| --- | ---: | ---: |
| Complete delivery | 1/5 | 2/5 |
| Correct artifact or diagnosis | 3/5 | 4/5 |
| Readback criterion met | 1/5 | 5/5 |
| Explicit closure | 3/5 | 3/5 |
| Tool attempts | 44 | 38 |
| Observed prompt tokens | 170,232 | 120,173 |
| Text-only responses after tool use | 0 | 1 |

The text-only response recovered through the existing closure reminder.
With the implemented `INFINIDEV_TASK_POST_TOOL_CHOICE=auto` option,
a replay of three of those cases measured 2/3 complete deliveries versus
1/3 with `required`. Both replays of the creation task failed the artifact
criterion even though `auto` created and reread the file. The fixture checks
recognize a bounded set of source forms. This exploratory sample supports
offering the option with the original default.

### Verification counterexamples

Three Qwen pairs offered the same fixture checker with the same prompts and
tools. The control received pass or fail. The candidate also received the
first observed failing example per file when the static checker could
interpret it. The harness parsed fixture source without executing it and
scored checks against the last observed edit.

Both arms produced 3/3 correct artifacts and 2/3 passing final checks.
Complete delivery rose from 1/3 to 2/3 because the two-file lease task
closed before the call limit with counterexamples. The caller conversion
task still missed its final check and closure in both arms. These runs
support preserving concrete check diagnostics where available. They leave
the improvement in artifact correctness unproven.

### Executable test feedback

Three Qwen pairs used the repository's `test-selection`,
`tool-failure-recovery`, and `options-override` fixtures. Commands ran with
a read-only workspace and no network. Both arms saw the same command
catalog, stdout, stderr, and process status. The structured arm also saw
test names and counts from the pytest report. The final verification ran
independently with the withheld gate restored where applicable.

All three final artifacts passed in each arm. The raw arm completed 3/3
deliveries with an observed post-edit suite run and explicit closure. The
structured arm completed 2/3. In `options-override`, the raw arm explicitly
reported that the visible tests covered defaults and that its override
check was by inspection. The structured arm passed the visible
suite on an earlier revision, then reached the tool limit after another
edit. The independent gate passed the final source. The agent did not rerun
the suite on that revision or close the turn. These exploratory runs leave
a benefit from the extra structure unproven.

### Test revision feedback

Two Qwen pairs used `cart-immutability` and `options-override` with real
pytest commands in read-only, network-isolated fixtures. Both arms were
asked to run the visible suite before editing and on the final revision.
They kept native tool choice at `auto` and named readback off. Only the
candidate enabled `INFINIDEV_TASK_TEST_REVISION_FEEDBACK_ENABLED`.

The independent final gates passed all four artifacts, and all four runs
executed the visible suite after their last edit. The control closed 1/2
turns and the candidate closed 2/2. Each candidate received one stale-test
observation and ended with the host's result marked current. Verification
snapshots took 1 and 2 ms in the two small candidate fixtures. This sample
supports keeping the option available for further evaluation. The
experiment used an earlier wording for the notice. The released text
identifies the observed zero exit explicitly.

### Selecting historical tool output

Two read-only pairs on Basalt 0.2.1 with Qwen3.8-FN-UD-IQ4_XS asked for a
literal value from one of two archived reads of files named `config.py`
in different directories. Each pair reused the same archive references and
order, native automatic selection, temperature 0.2 and seed 147. The visible
tool grant allowed recall and withheld file reads. The treatment added the
source tool and the requested workspace-relative path to each locator.

| Observation | Opaque locators | Labeled locators |
| --- | ---: | ---: |
| Target reference in first recall | 0/2 | 2/2 |
| Exact target page recovered | 1/2 | 2/2 |
| Correct final value | 0/2 | 2/2 |
| Native `finish_turn` | 1/2 | 2/2 |
| Recall calls | 9 | 2 |
| Observed prompt tokens | 18,085 | 8,727 |

The archives held actual dispatcher output. The probe scored retrieval and
the final answer separately. Each arm ran once per target.

A follow-up used two archived reads of the same path across a file change.
Both arms kept the source labels. The treatment added an
`evidence_order: oldest_to_newest` field to the request. Both arms passed
the exact-answer check for the newest observation. For the earlier
observation, both recovered the correct page and failed the exact-answer
check. The baseline selected the correct older reference on its first
recall, and the treatment reached it later. The order field stayed in
the probe.

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
