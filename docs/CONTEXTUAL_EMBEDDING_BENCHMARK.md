# Contextual embedding benchmark for task-policy detection

This experiment asks whether a small contextual encoder can replace the
bundled static Qwen3 representation for conditional system prompting. It uses
the same leakage-separated calibration, validation, and holdout examples for
both backends, including zero-label discourse and compatible multi-label
tasks.

The candidate is `intfloat/multilingual-e5-small`. It is an optional research
dependency: the model cache stays outside the repository, no checkpoint is
packaged, and the runtime still uses the existing hybrid router.

## Acceptance gate

A replacement must exceed all of these values on holdout:

- 95% exact multi-label match;
- 95% macro F1;
- 95% micro precision;
- 95% micro recall;
- zero activations on zero-label requests.

The gate is intentionally stricter than ordinary intent classification because
a wrong result changes the instructions given to the coding model.

## Results

| Encoder and head | Exact | Macro F1 | Precision | Recall | False activations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static Qwen3, frozen MLP | 59.38% | 66.14% | 68.35% | 63.28% | 29 |
| E5-small, frozen MLP | 82.29% | 87.38% | 94.09% | 80.86% | 4 |
| E5-small, final two layers tuned for 3 epochs | 83.33% | 86.66% | 91.27% | 81.64% | 2 |

E5 is a substantial semantic improvement, but neither version passes. Tuning
3.0% of the encoder for three epochs took 539 seconds on CPU and improved
exact match by only 1.04 points over the frozen E5 head. It also reduced macro
F1 and precision. The best validation epoch reached 91.04% exact match and
zero false activations, which did not transfer to holdout.

The main failure is family generalization, not simple class imbalance. For
example, the unseen English bugfix family was consistently classified as a
feature. Research recall and feature precision also remained below the gate.
Adding epochs without changing this evidence would optimize the development
distribution rather than repair that gap.

## Runtime cost

The contextual encoder has 384 dimensions versus 1,024 for the static
representation, but its transformer inference is much heavier:

| Backend | Corpus throughput | Warm request p50 | Warm request p95 |
| --- | ---: | ---: | ---: |
| Static Qwen3 | 6,914 examples/s | 1.48 ms | 3.33 ms |
| E5-small | 183 examples/s | 15.08 ms | 18.34 ms |

Peak RSS observations are process-level upper bounds rather than clean model
allocations. In the measured processes, static loading added roughly 400 MiB,
E5 inference roughly 595 MiB, and partial fine-tuning reached roughly 1.34
GiB resident memory.

## Decision

Do not replace the runtime encoder yet. Keep E5 as an offline teacher and
benchmark candidate. The next useful iteration is a higher-diversity,
family-balanced corpus plus a fresh sealed holdout, followed by one controlled
comparison of:

1. frozen E5 with a calibrated abstention/cardinality head;
2. contrastive or supervised E5 tuning on calibration families only;
3. the existing deterministic-plus-semantic hybrid router.

The current holdout has now been inspected repeatedly, so it must not be used
to choose further hyperparameters. A new holdout should contain natural user
requests and entirely unseen project and phrasing families. Runtime integration
is justified only after that sealed evaluation passes.

## Reproduction

```bash
uv run python -m bench.contextual_embedding_benchmark --backend static
uv run python -m bench.contextual_embedding_benchmark --backend contextual
uv run python -m bench.contextual_task_policy_finetune --epochs 3 --unfrozen-layers 2
```

The contextual commands download the model through the configured Hugging Face
cache and therefore require network access on the first run.
