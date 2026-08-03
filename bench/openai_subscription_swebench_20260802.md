# SWE-bench Lite Benchmark with `openai_subscription`

- **Report date (UTC):** 2026-08-02
- **Run identifier:** `20260802T200907Z`
- **Final status:** 2/2 generation runs completed; 1/2 instances officially resolved.
- **Artifact directory:** [`runs/20260802T200907Z/`](runs/20260802T200907Z/)
- **Sanitized evidence index:** [`evidence.json`](runs/20260802T200907Z/evidence.json)

> Results come from the official SWE-bench evaluator, not from an inference based on the patches. [`evidence.json`](runs/20260802T200907Z/evidence.json) retains prevalidation, complete commands, exit codes, timings, sanitized log summaries, and the mapping between each prediction and its official verdict. It contains no credentials or account data.

## Objective

Run Infinidev on two identified SWE-bench Lite instances using only the ChatGPT session previously created by `codex login`, through the `openai_subscription` provider and the `gpt-5.6-sol` model. Then evaluate every non-empty prediction with the existing [`evaluate.py`](evaluate.py) wrapper, which delegates to the official SWE-bench evaluator.

The provider obtains OAuth authorization at runtime from the Codex CLI session. The reproducible configuration contains only the provider and model.

## Fixed configuration

| Parameter | Value |
|---|---|
| Harness | [`run_swebench.py`](run_swebench.py) |
| Dataset | `princeton-nlp/SWE-bench_Lite` |
| Split | `test` |
| Provider | `openai_subscription` |
| Requested model | `gpt-5.6-sol` |
| Model normalized by Infinidev | `openai/responses/gpt-5.6-sol` |
| Agent timeout per instance | `1800 s` |
| Resume | disabled (`--no-resume`) |
| Instances | `astropy__astropy-12907`, `astropy__astropy-14182` |
| Evaluation workers | `1` |
| Evaluation timeout per instance | `1800 s` |

Selection is explicit by `instance_id`; it does not depend on the dataset's current order. The normalized prefix identifies the Responses API protocol, but the request is sent to the subscription's Codex backend rather than the usage-billed API endpoint.

## Prevalidation and environment

Prevalidation is retained in the `prevalidation` section of [`evidence.json`](runs/20260802T200907Z/evidence.json). It explicitly identifies:

- the generation CLI [`bench/run_swebench.py`](run_swebench.py), invoked as `python -m bench.run_swebench`;
- the [`bench/evaluate.py`](evaluate.py) wrapper, which delegates to the official `python -m swebench.harness.run_evaluation` CLI;
- provider selection through `LLM_PROVIDER=openai_subscription` and `LLM_MODEL=gpt-5.6-sol` in `.infinidev/settings.json` inside each case's isolated `HOME`;
- the exact dependencies declared in [`bench/requirements.txt`](requirements.txt): `datasets>=2.14.0` and `swebench>=1.0.0`;
- the reviewed configuration files under [`src/infinidev/config/`](../src/infinidev/config/) and the focused [`tests/test_openai_subscription.py`](../tests/test_openai_subscription.py) test.

| Component | Observed version/status | Retained evidence |
|---|---:|---|
| Infinidev | `0.14.0` | [`evidence.json`](runs/20260802T200907Z/evidence.json), `prevalidation.dependencies` |
| Python | `3.13.3` | [`evidence.json`](runs/20260802T200907Z/evidence.json), `prevalidation.dependencies` |
| `datasets` | `4.8.4` (declared `>=2.14.0`) | [`requirements.txt`](requirements.txt) and [`evidence.json`](runs/20260802T200907Z/evidence.json) |
| `swebench` | `4.1.0` (declared `>=1.0.0`) | [`requirements.txt`](requirements.txt) and [`evidence.json`](runs/20260802T200907Z/evidence.json) |
| LiteLLM | `1.82.2` | [`evidence.json`](runs/20260802T200907Z/evidence.json), `prevalidation.dependencies` |
| Docker | SDK `7.1.0`; server `5.4.1` | [`evaluation/metadata.json`](runs/20260802T200907Z/evaluation/metadata.json) |
| Focused provider tests | `57/57` passed | Output retained in [`validation-pytest.txt`](runs/20260802T200907Z/validation-pytest.txt) and summarized in [`evidence.json`](runs/20260802T200907Z/evidence.json) |
| Dataset/split | available; both IDs found | Per-attempt summaries in [`evidence.json`](runs/20260802T200907Z/evidence.json) |
| Codex session | configured, unexpired, and usable without an API key during prevalidation | Sanitized Boolean status in [`evidence.json`](runs/20260802T200907Z/evidence.json) |

The authorization check stored only sanitized Boolean status. No OAuth tokens, credential paths, account data, or environment dumps were retained.

### Retained validations

- [`validation-pytest.txt`](runs/20260802T200907Z/validation-pytest.txt) retains the command, runner versions, exit code, and complete result of the 57 focused tests: 57 passed. It neither retains nor claims a full-suite run.
- [`validation-links.json`](runs/20260802T200907Z/validation-links.json) retains the algorithm, report hash, every checked local target, counts, and exit code from Markdown link validation: 44 occurrences checked, 21 unique local targets, and 0 failures.
- [`validation-sensitive-data.json`](runs/20260802T200907Z/validation-sensitive-data.json) retains the versionable scope, file selection and hashes, eight patterns, scanner version, and exit code. The result was zero matches without emitting potentially sensitive content.

### Versioned inventory for this delivery

The retained set contains exactly 20 paths: this report and 19 artifacts. They are:

```text
bench/openai_subscription_swebench_20260802.md
bench/runs/20260802T200907Z/astropy__astropy-12907/attempt-1/command.txt
bench/runs/20260802T200907Z/astropy__astropy-12907/attempt-1/metadata.json
bench/runs/20260802T200907Z/astropy__astropy-12907/command.txt
bench/runs/20260802T200907Z/astropy__astropy-12907/metadata.json
bench/runs/20260802T200907Z/astropy__astropy-12907/predictions.jsonl
bench/runs/20260802T200907Z/astropy__astropy-14182/attempt-1/command.txt
bench/runs/20260802T200907Z/astropy__astropy-14182/attempt-1/metadata.json
bench/runs/20260802T200907Z/astropy__astropy-14182/command.txt
bench/runs/20260802T200907Z/astropy__astropy-14182/metadata.json
bench/runs/20260802T200907Z/astropy__astropy-14182/predictions.jsonl
bench/runs/20260802T200907Z/evaluation/command.txt
bench/runs/20260802T200907Z/evaluation/gpt-5.6-sol.infinidev-openai-subscription-20260802T200907Z.json
bench/runs/20260802T200907Z/evaluation/metadata.json
bench/runs/20260802T200907Z/evaluation/prediction_results.json
bench/runs/20260802T200907Z/evidence.json
bench/runs/20260802T200907Z/predictions.combined.jsonl
bench/runs/20260802T200907Z/validation-links.json
bench/runs/20260802T200907Z/validation-pytest.txt
bench/runs/20260802T200907Z/validation-sensitive-data.json
```

The two `home/.infinidev/settings.json` files created under the per-case directories remain as ignored local entries for reproducing the isolated environment, but they are not part of those 20 paths or the versioned delivery. They contain only `LLM_PROVIDER` and `LLM_MODEL`; the same non-sensitive configuration is transcribed above and in `evidence.json`. Other generated content under those `home/` directories is not retained either.

## Exact commands

Each case's [`command.txt`](runs/20260802T200907Z/astropy__astropy-12907/command.txt) file and `generation_invocation_template` in [`evidence.json`](runs/20260802T200907Z/evidence.json) retain the complete commands. From the repository root, the reproducible form was:

```bash
REPO="$PWD"
RUN_ID="20260802T200907Z"
RUN_ROOT="$REPO/bench/runs/$RUN_ID"
CODEX_SESSION_HOME="${CODEX_HOME:-$HOME/.codex}"
mkdir -p "$RUN_ROOT"
```

Each ID used an isolated `HOME` under the run directory. The minimal configuration selected only `openai_subscription`; `CODEX_HOME` continued to point to the existing Codex CLI login:

```bash
INSTANCE_ID="astropy__astropy-12907"  # repeated with astropy__astropy-14182
CASE_DIR="$RUN_ROOT/$INSTANCE_ID"
mkdir -p "$CASE_DIR/bench" "$CASE_DIR/home/.infinidev" "$CASE_DIR/work"
cat >"$CASE_DIR/home/.infinidev/settings.json" <<'JSON'
{
  "LLM_PROVIDER": "openai_subscription",
  "LLM_MODEL": "gpt-5.6-sol"
}
JSON

cd "$CASE_DIR"
env \
  HOME="$CASE_DIR/home" \
  CODEX_HOME="$CODEX_SESSION_HOME" \
  PYTHONPATH="$REPO:$REPO/src" \
  uv --project "$REPO" run python -m bench.run_swebench \
    --dataset princeton-nlp/SWE-bench_Lite \
    --split test \
    --instance-id "$INSTANCE_ID" \
    --model gpt-5.6-sol \
    --timeout 1800 \
    --output "$CASE_DIR/predictions.jsonl" \
    --workdir "$CASE_DIR/work" \
    --no-resume
```

Each case retains `command.txt`, `metadata.json`, and `predictions.jsonl`; raw logs remain ignored local artifacts. To keep the reviewable evidence independent of those ignored files, `attempts` in [`evidence.json`](runs/20260802T200907Z/evidence.json) retains their relevant sanitized facts: exit code and duration, loading of the requested ID, timeout, patch recovery, and completion. The first launch for each ID stopped before loading the dataset because the relative `bench/` subdirectory required by `FileHandler` was missing; each attempt's metadata was preserved without overwriting it under `attempt-1/`. After that directory was created, the second attempt processed both cases and produced the reported result.

The environment was not dumped; versionable artifacts retain only the configuration and sanitized status needed to reproduce and audit the run.

## Per-instance results

| Instance | Infinidev run | Exit code | Duration | Prediction | Patch | Official evaluation | Evidence |
|---|---|---:|---:|---|---|---|---|
| `astropy__astropy-12907` | Complete with patch recovered after timeout (attempt 2) | `0` | `1824.271 s` | Valid JSONL | Non-empty (44 reported lines) | **Resolved** | [`evidence`](runs/20260802T200907Z/evidence.json), [`metadata`](runs/20260802T200907Z/astropy__astropy-12907/metadata.json), [`prediction`](runs/20260802T200907Z/astropy__astropy-12907/predictions.jsonl), [`command`](runs/20260802T200907Z/astropy__astropy-12907/command.txt) |
| `astropy__astropy-14182` | Complete with patch recovered after timeout (attempt 2) | `0` | `1824.297 s` | Valid JSONL | Non-empty (60 reported lines) | **Unresolved** | [`evidence`](runs/20260802T200907Z/evidence.json), [`metadata`](runs/20260802T200907Z/astropy__astropy-14182/metadata.json), [`prediction`](runs/20260802T200907Z/astropy__astropy-14182/predictions.jsonl), [`command`](runs/20260802T200907Z/astropy__astropy-14182/command.txt) |

Both runners started at `2026-08-02T20:14:45Z` and ended at `20:45:09Z` on commit `7fbd80b9f726a76c57ddb71e37caa402ba8088c3`. A zero runner exit code confirms only that the harness finished; the table's verdicts come exclusively from the official evaluator.

## Aggregates

| Measure | Result | Source |
|---|---:|---|
| Selected cases | `2` | Explicit IDs in the commands |
| Total launches | `4` | Two retained initial attempts and two relaunches |
| Complete evaluable runs | `2 / 2` | Per-case metadata and JSONL |
| Predictions with non-empty patches | `2 / 2` | [`predictions.combined.jsonl`](runs/20260802T200907Z/predictions.combined.jsonl) |
| Cases resolved by SWE-bench | `1 / 2` | Official result |
| Resolution rate | **50%** | `1 / 2` evaluated cases |
| Evaluation errors | `0` | Official result |
| Cumulative agent time | `3648.568 s` (`60 min 48.568 s`) | Sum of per-case metadata |
| Official evaluation | `380.3 s` (`6 min 20.3 s`) | Evaluation metadata |

The two agents ran in parallel, so the sum above represents cumulative per-instance time rather than the series' wall-clock time. Initial aborted attempts do not count as evaluated cases and remain documented only for operational traceability.

## Evaluation

The two predictions were combined without changing the original JSONL files and validated with the wrapper's summary mode:

```bash
uv run python -m bench.evaluate \
  --predictions "bench/runs/20260802T200907Z/predictions.combined.jsonl" \
  --summary-only
```

The official evaluation command was:

```bash
uv run python -m bench.evaluate \
  --predictions "bench/runs/20260802T200907Z/predictions.combined.jsonl" \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --run-id infinidev-openai-subscription-20260802T200907Z \
  --max-workers 1 \
  --timeout 1800
```

It exited with code `0`: 2 instances submitted and completed, 1 resolved (`astropy__astropy-12907`), 1 unresolved (`astropy__astropy-14182`), 0 empty patches, and 0 errors. Per-instance traceability is retained in [`evaluation/prediction_results.json`](runs/20260802T200907Z/evaluation/prediction_results.json): each entry links the original prediction to concrete fields in the [official JSON result](runs/20260802T200907Z/evaluation/gpt-5.6-sol.infinidev-openai-subscription-20260802T200907Z.json). The command, environment status, and timings are retained in [`evaluation/command.txt`](runs/20260802T200907Z/evaluation/command.txt) and [`evaluation/metadata.json`](runs/20260802T200907Z/evaluation/metadata.json).

## Classification criteria

- **Complete run:** the runner process finished and produced a valid JSONL line for the expected ID. It can contain an empty patch.
- **Failed run:** the process or harness reported a failure after starting the case; any partial patch and the error are retained.
- **Blocked run:** a verifiable external precondition prevented the case from being processed, such as unavailable authentication, an inaccessible dataset, or an impossible checkout.
- **Evaluable prediction:** valid JSONL with a non-empty `instance_id`, `model_name_or_path`, and `model_patch`.
- **Resolved:** only the official SWE-bench evaluator reports success.

## Minimal reproduction

1. Install and sign in to the Codex CLI with `codex login`, selecting ChatGPT sign-in.
2. Confirm in sanitized form that Infinidev recognizes the session and that `gpt-5.6-sol` appears in the Codex catalog.
3. Run the two commands above with a new run directory; do not reuse `20260802T200907Z` if it already exists.
4. Validate and evaluate the combined JSONL through [`evaluate.py`](evaluate.py).
5. Compare `metadata.json`, logs, predictions, [`prediction_results.json`](runs/20260802T200907Z/evaluation/prediction_results.json), and the official result; do not use the runner exit code as a substitute for the SWE-bench result.

If the session does not exist or has expired, repeat `codex login`.

## Known limitations

- The two-case sample is a reproducible test of the harness and provider, not a statistically representative estimate of the complete SWE-bench Lite dataset.
- The timeout limits agent wall-clock time, so a partial edit can be retained without representing a final solution.
- The runner records per-instance preparation failures as empty predictions; the report must also inspect logs and must not interpret an empty JSONL line as a successful run.
- The result depends on the hosted model's state and can vary between runs even when local parameters are identical.
- The figures apply only to this identified run and its recorded versions; they must not be extrapolated to another Infinidev revision, model, or environment.

## Integrity and privacy

The artifacts were saved under a new run ID and did not overwrite the historical `bench/predictions*.jsonl` or `bench/run.log` files that existed at the time. Final validation confirmed that versionable artifacts contained only technical benchmark data. Complete logs remain ignored local evidence; no account or session data is referenced.
