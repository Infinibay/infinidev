# Continuation handoff: conditional semantic prompting

This file is the operational handoff for continuing the active mini-model work on the GPU
server. The repository snapshot intentionally includes all current source, benchmark, test, and
documentation changes. Downloaded candidate text, generated splits, model caches, and checkpoints
remain outside Git for licensing and size reasons. The minimized human review ledgers are committed
under `data/task-policy-reviews/` with separate data licensing and attribution.

## Objective

Train and validate a small multilingual multi-label classifier for conditional prompting. It must
detect `bugfix`, `feature`, `performance`, `refactor`, `research`, and `review`, including compound
requests and zero-label requests, with at least 95% binary accuracy for every individual label on
natural family-disjoint data. Accuracy is not sufficient when a label has no positive support:
always report precision, recall, F1, and positive support as well.

After the classifier passes, use its task, message, and reasoning-pattern detections to add or
remove generic and model-family-specific prompt fragments at the appropriate runtime moments.
Then validate the resulting agent behavior end to end with MiniMax M3 and GPT subscription models,
measuring quality, tokens, tool calls, and errors. Before any Infinidev E2E, run
`uv run ken install . --embed`.

## Main-model classifier decision (2026-08-13)

The user chose to pay for one small call to the same selected main model before
the main task prompt rather than depend exclusively on the local classifier.
The runtime now defaults `TASK_POLICIES_LLM_CLASSIFIER_MODE` to `preferred`.
It obtains the normal selected provider/model dynamically, requests only the
six method labels plus confidence, and then injects the matching conditional
prompt fragments. The response replaces local method categories in preferred
mode, but it cannot grant modify, commit, publish, or any other authority.

For Z.AI GLM models, the helper call sends the official per-request
`thinking: {type: disabled}` switch through `extra_body`. This matters because
the installed LiteLLM 1.82.2 adapter otherwise rejects or misroutes the field.
The normal task call is unchanged and retains the user's reasoning setting.

The final GLM-5.2 pilot used 26 deterministic, length-stratified rows from the
already-open natural evaluation split: three single-label examples per method,
four zero-label examples, and four compound examples. Results were:

- exact match: 21/26, **80.77%**;
- no request or JSON failures at 256 output tokens;
- F1: bugfix 100%, feature 75%, performance 100%, refactor 75%, research 100%,
  review 75%;
- latency: 1,788 ms p50, 2,498 ms p95, 6,237 ms maximum/cold first call.

On those same 26 rows, the warmed local Qwen encoder reached 61.54% exact
match, with 46 ms p50 and 68 ms p95 after a 6,264 ms cold model load. On its
much larger 586-row evaluation, its selected epoch 3 reached 83.94% exact
match, so do not compare that aggregate directly with the deliberately small,
rare-label-heavy GLM pilot. The GLM prompt was tuned after inspecting this
development split; these numbers are development evidence, not sealed holdout
performance.

The outcome is mixed but actionable: using the main model improved the
identical hard pilot by 19.23 exact-match points and substantially improved rare
`research`/`review` recognition, while its 1.8-second median does not meet the
earlier 300 ms target. The default follows the user's later preference for
quality over that latency target. Set the mode to `off` to avoid the extra call,
or `fallback` to keep it only for unresolved local routing.

Reproduce the live pilot without storing prompt text or credentials:

```bash
uv run python -m bench.task_policy_main_model_classifier_eval \
  --data-root /home/andres/tmp/task-policy-natural-split-v1 \
  --partition evaluation --per-label 3 --zero-label 4 --compound 4 \
  --max-tokens 256 --output /tmp/task-policy-main-model-pilot.json
```

## Current state

- All natural annotation queues are complete: **2,901 manually reviewed requests**.
  - Open-SWE: 1,021.
  - WildChat: 1,880.
- Scripts only validate, join, group, and split data. Every label was written manually.
- The family splitter groups conversation/repository identity and conservative lexical
  near-duplicates before assignment.
- The reviewed corpus contains 2,336 independent families.
- The current development split is fixed with seed 2027 and 128 deterministic balancing trials:

| Split | Rows | Families | bugfix | feature | performance | refactor | research | review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 1,728 | 1,517 | 292 | 645 | 50 | 61 | 21 | 21 |
| Calibration | 587 | 407 | 104 | 225 | 17 | 21 | 7 | 7 |
| Evaluation | 586 | 412 | 105 | 230 | 17 | 21 | 7 | 7 |

The evaluation split is natural and family-disjoint, but it is development evaluation once its
errors influence another model decision. It is much stronger than the older 60-row pilot and has
positive support for all six labels. Research and review still have only seven evaluation
positives each, so state that uncertainty honestly.

The original WildChat sealed reserve has **120 rows** and has not been opened. Its SHA-256 remains:

```text
2cb4818be6221a51ecc4a2b16e5151e80f7851b20cac5d5f57e9b1ba89d790ae
```

Do not inspect its text, labels, predictions, or errors until architecture, thresholds, and
predictions are frozen. It has no labels yet; reviewing it is the final one-time evaluation step.

## Last attempted measurement

The first full E5-small run was started locally with:

- `intfloat/multilingual-e5-small`;
- LoRA rank 8, alpha 16;
- label-attention plus bounded lexical features;
- natural accuracy calibration;
- maximum length 192;
- 16 epochs maximum, patience 4;
- one development fold plus the 586-row natural evaluation split.

It was manually interrupted during training to move work to the GPU server. It produced **no
checkpoint and no new accuracy result**. Do not describe the enlarged dataset or the running
process as measured improvement. The older natural 600/60 pilot had only about 81.7% bugfix and
80.0% feature accuracy and is not evidence for the current corpus.

At the time of this handoff, `bench/task_policy_manual_finetune_cv.py` still ran only on CPU: it
did not move the model, encoded batches, or targets to CUDA, and converted tensors directly with
`.numpy()`. The GPU server progress below records the completed fix and its validation. Do not
assume that `CUDA_VISIBLE_DEVICES=0` alone enables GPU execution.

## GPU server progress (2026-08-12)

CUDA plumbing is now implemented in `bench/task_policy_manual_finetune_cv.py`. The CLI accepts
`--device`, defaults to `auto`, rejects unavailable explicit CUDA devices, moves every encoded
batch and target batch to the resolved device, places model and loss weights on that device, and
moves inference outputs back through `.detach().cpu().numpy()`. The selected device is recorded in
partial, fold, and final reports.

The exact environment command used on this server was:

```bash
uv sync --extra finetune
```

It installed PyTorch 2.10.0+cu128. Runtime validation reports CUDA available on one NVIDIA RTX
A5000 (24,564 MiB, compute capability 8.6). The focused suite passes 20/20, including a real
CPU-to-CUDA batch transfer and CUDA-to-NumPy output test:

```bash
uv run pytest tests/test_task_policy_manual_finetune_cv.py -q
```

A one-epoch smoke run completed the full E5-small + LoRA +
`label_attention_lexical` training/inference path on CUDA in 5.32 seconds and wrote
`/home/andres/tmp/infinidev-e5small-cuda-smoke.json`. Its quality metrics are not evidence for
the requested classifier gate: it intentionally used only one epoch and the repository corpus,
not the 2,901-row natural split.

The five candidate queues are present under
`/home/andres/infinidev/.infinidev/external-data/`; acquisition verified all pinned SHA-256
digests and exact row counts (240/240/240/600 Open-SWE and 2,000 WildChat). After pulling commit
`c35219a`, the bootstrap consumed the 37 tracked review ledgers and rebuilt
`/home/andres/tmp/task-policy-natural-split-v1`. The manifest reports exactly 2,901 rows and 2,336
families, and all six split hashes match the fixed values below.

Threshold and checkpoint selection also had a degenerate rare-label path: when the accuracy gate
was unreachable, predicting no positives could win on accuracy and count as meeting the checkpoint
target despite zero recall. Selection now requires positive recall for a label to count toward the
checkpoint target and prefers supported recall before fallback accuracy. Focused regression tests
cover both the individual and natural-domain selectors.

### GPU development measurements

Three controlled fold-0 runs used the fixed natural split, CUDA, the same
`label_attention_lexical` head, 192-token maximum, seed 41, 16-epoch maximum, and patience 4. The
development evaluation split has now influenced model selection and remains development evidence;
the sealed 120-row reserve was not opened.

| Encoder/tuning | Epochs run | Best epoch | Time | Exact match | bugfix | feature | refactor | research | review | performance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E5-base, LoRA 8 | 10 | 6 | 224.1 s | 67.92% | 91.30% | 81.23% | 95.05% | 98.29% | 94.71% | 96.76% |
| E5-base, last 4 layers | 6 | 2 | 111.5 s | **71.33%** | 90.78% | **83.11%** | **96.08%** | **99.15%** | **98.12%** | **96.93%** |
| BGE-M3, LoRA 8 | 10 | 6 | 629.8 s | 63.65% | 90.61% | 82.76% | 94.88% | 98.12% | 88.57% | 97.95% |

All six policies have positive predictions and non-zero recall in these corrected runs. None passes
the required 95% binary accuracy for every label: bugfix and feature remain the stable failures.
Oracle threshold scans on E5-base LoRA reach only 91.98% bugfix and 82.94% feature accuracy, so
recalibration cannot close the gap. A separate E5-small 512-token run was worse than its 192-token
counterpart, so instruction truncation is not the current explanation either. Do not spend more
runs on thresholds, length, or epochs without new evidence; the next iteration needs a deliberate
family-generalization/data-boundary change or a materially different classifier design.

The full repository suite also passes after the CUDA and evaluation-corpus fixes:

```text
3788 passed, 1 skipped, 1 warning in 181.28s
```

## External state to copy to the server

Git deliberately excludes the natural source text, but the minimized human review ledgers now ship
under `data/task-policy-reviews/`. After cloning or pulling, the server only needs the candidates;
the bootstrap can download them reproducibly. Direct transfer remains an optional faster path:

```bash
rsync -a --info=progress2 \
  /home/andres/infinidev/.infinidev/external-data/ \
  USER@SERVER:/PATH/TO/infinidev/.infinidev/external-data/

uv run python -m bench.task_policy_data_bootstrap
```

If the server uses a different home, update all `/home/andres/tmp` arguments below. Do not place
these artifacts under `/tmp`; use the persistent user-owned temporary directory.

Validate the copied split hashes against
`/home/andres/tmp/task-policy-natural-split-v1/manifest.json`. Current hashes are:

```text
training candidates   56db89617c770be35890201567ffce41a38d93d3545811fe208ab5838ecb4f85
training reviews      46e51a247b19b8d02ea7b7e57193176fd1622bb57680fc55819994505db5a1dd
calibration candidates ab0362c9cf052fbef08155c546d75fa5d039d3150f42ef163f93252e7851119d
calibration reviews   13d8674180a720d9ad89774865a101a43be924cc3d4ca421b870af72f5b12e9d
evaluation candidates 7226c603eb564320883ea52aeed14a0a8fdcbdf1c4057c9621e540c2f89c1561
evaluation reviews    39fb81b3b6616ebcfe210766651c82fe742bc032104593adef6bbe5c5da4e562
```

## Server bootstrap

```bash
cd /PATH/TO/infinidev
uv sync
uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
uv run pytest tests/test_task_policy_natural_split.py -q
```

The candidate acquisition and split rebuild are now available as one guarded command:

```bash
uv run python -m bench.task_policy_data_bootstrap
```

It downloads or reuses all five pinned candidate queues, verifies their exact SHA-256 digests,
loads the 37 separately licensed manual review ledgers committed under
`data/task-policy-reviews/`, and recreates the fixed 2,901-row split. It will stop instead of
inventing labels when those ledgers are absent. If reviews live separately:

```bash
uv run python -m bench.task_policy_data_bootstrap \
  --review-root /PATH/TO/TRANSFERRED/external-data
```

Use `--mode acquire` to download and verify candidates without building splits, or `--mode build`
to rebuild from existing candidates and ledgers without network access. `--force-download`
deliberately overwrites reproducible candidate queues but never review ledgers.

If the locked PyTorch build is CPU-only, install the matching CUDA wheel in the project environment
without changing unrelated dependencies, confirm `torch.cuda.is_available()`, and record the exact
environment command. The warning about the optional `kernels-community` 4-bit GEMM package is not
a failure for this BF16/FP32 E5 run and does not justify adding it unless the selected path actually
uses 4-bit inference.

## Reproduce the fixed natural split

The already copied split is preferred because its manifest fixes the exact artifacts. If it must
be regenerated, use `bench.task_policy_natural_split` with all four Open-SWE candidate sources,
the WildChat candidate source, every completed Open-SWE review ledger, the three original WildChat
review ledgers, `family_round1_development_reviews.jsonl`, and all sixteen
`family_round1_queue_N_reviews.jsonl` ledgers. Use:

```text
--seed 2027 --trials 128 --minimum-positive-support 5
```

The regenerated manifest must again report 2,901 rows, 2,336 families, and the table above. A
difference means the input set or algorithm changed and must be investigated before training.

## GPU training command

After implementing and testing CUDA device plumbing, run the same configuration that was
interrupted locally:

```bash
uv run python -m bench.task_policy_manual_finetune_cv \
  --only-fold 0 \
  --model intfloat/multilingual-e5-small \
  --training-candidates /home/andres/tmp/task-policy-natural-split-v1/training_candidates.jsonl \
  --training-reviews /home/andres/tmp/task-policy-natural-split-v1/training_reviews.jsonl \
  --calibration-candidates /home/andres/tmp/task-policy-natural-split-v1/calibration_candidates.jsonl \
  --calibration-reviews /home/andres/tmp/task-policy-natural-split-v1/calibration_reviews.jsonl \
  --external-candidates /home/andres/tmp/task-policy-natural-split-v1/evaluation_candidates.jsonl \
  --external-reviews /home/andres/tmp/task-policy-natural-split-v1/evaluation_reviews.jsonl \
  --architecture label_attention_lexical \
  --threshold-calibration accuracy \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-length 192 \
  --batch-size 12 \
  --epochs 16 \
  --patience 4 \
  --minimum-method-accuracy 0.95 \
  --device cuda \
  --output /home/andres/tmp/infinidev-natural2901-e5small-lora8-lexical-fold0.json
```

Choose a larger batch only after checking actual A500 memory. An out-of-memory error should be
handled by reducing the batch size first; do not reduce sequence length or change architecture in
the same retry because that would confound the comparison.

## How to judge the result

Read `external_evaluation.consensus_metrics` when multiple folds exist, otherwise read the single
fold's `external_evaluation.metrics`. For every canonical policy report:

- binary accuracy;
- precision;
- recall;
- F1;
- positive support;
- false positives and false negatives by source and family.

The requested gate is binary accuracy greater than or equal to 95% for each label. Also reject a
degenerate result with zero recall or no positive predictions. Exact match is informative but is
not the user's primary gate.

If a class misses the gate, inspect its score distribution and errors before changing anything:

1. Recalibrate thresholds only when separability is good but the operating point is wrong.
2. Increase maximum length only when operative instructions are demonstrably truncated.
3. Change model capacity/embedding only when errors remain semantically inseparable.
4. Add data only for a clearly underrepresented natural boundary; never relabel for balance.
5. Add epochs only when training/validation curves show underfitting, not merely because accuracy
   is low.

Do not open the sealed reserve while making these decisions.

## Validation already completed

- All sixteen WildChat family queues were reviewed and audited; the final queue was 83/83.
- The last apparent within-family conflict was legitimate: near-identical scraper code was asked
  once for read-only review and once for explanation only.
- `uv run pytest tests/test_task_policy_natural_split.py -q` passed: **3 passed**.
- Bootstrap plus natural-split focused tests passed: **7 passed**.
- All five acquired candidate queues match their pinned hashes and expected row counts.
- The natural split manifest successfully enforces at least five positives for every label in all
  three splits.
- `git diff --check` was clean before this handoff.

The full suite was rerun after the CUDA plumbing and passed as recorded in the GPU server progress
section. Rerun the focused training tests after further classifier changes and `uv run pytest`
again before release or push.

## Remaining work after the classifier gate

1. Freeze the chosen classifier and thresholds as a versioned runtime artifact small enough for
   the package; do not commit downloaded source text or large training checkpoints.
2. Confirm runtime latency, memory, and cold-start behavior on CPU and GPU. Infinidev's normal
   product path must not require a training environment.
3. Complete the task/message/reasoning policy integration and verify generic plus model-family
   prompt composition. Existing work is under `src/infinidev/engine/task_policies/`,
   `src/infinidev/engine/behavior/`, prompt composition, loop guidance, and orchestration paths.
4. Run `uv run ken install . --embed` before every Infinidev-vs-Codex E2E comparison.
5. Request MiniMax credentials from the user only when the live MiniMax E2E is ready. Never place
   API keys in Git, reports, commands recorded in documents, or committed fixtures.
6. Compare MiniMax M3 and GPT subscription tasks only after classifier quality and runtime prompts
   are frozen. Measure outcome quality, tokens, tool calls, tool errors, retries, and elapsed time.
7. Update the conditional prompting and dataset documents with the final honest metrics and
   limitations, run the full suite, and only then decide whether the active goal is complete.
