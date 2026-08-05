# Agent-task pilot completion audit

The original falsification pilot is complete. Machine-readable proof is in
`bench/runs/20260804-agent-task-pilot/completion-audit.json` and can be regenerated with:

```bash
uv run python -m bench.agent_task_completion_audit \
  bench/agent_task_pilot.approved.jsonl \
  bench/runs/20260804-agent-task-pilot \
  bench/runs/20260804-agent-task-pilot/completion-audit.json
```

## Requirement-by-requirement result

| Original requirement | Authoritative evidence | Result |
| --- | --- | --- |
| Use real repository tasks instead of more MCQs | Six approved fixture-backed tasks in `bench/agent_task_pilot.approved.jsonl` | Complete |
| Evaluate Sol, Terra, and Luna candidates | Frozen three-route `campaign-plan.json` and route observations | Complete |
| Compare current prompt with model-specific guidance | Every route contains six `baseline` and six `candidate` observations bound to condition hashes | Complete |
| Cover the six named task types | Planning, reversible implementation, test selection, code review, user-owned trade-off, and tool recovery are distinct approved tasks | Complete |
| Run exactly 36 executions | Completion audit finds 36 unique observations and 36 run artifacts | Complete |
| One run per condition | Every task/model/condition key has repetition zero and appears exactly once | Complete |
| Deterministic verifier per task | Preflight negative and positive controls passed for all six fixtures; every artifact retains verifier output | Complete |
| Review preference rubrics before execution | Hash-bound task review approved all six tasks and rubrics before the approved dataset/manifests were produced | Complete, author-side review only |
| Preserve functional results, tests, requirements, authorization, verification, interaction, tools, recovery, cost, and final answer | Observation schema plus full run artifacts and the campaign dossier retain those fields and concrete traces | Complete |
| Sequential and isolated execution | Frozen contract requires globally single-flight execution plus a fresh workspace and agent session per run; the unified campaign runner owns the global lock | Complete by enforced runner contract |
| Minimum two seconds between internal LLM requests | All route configs and manifests require at least two seconds and the runtime uses the shared host-wide request pacer | Complete by enforced runtime contract |
| Stop on first 429/provider/runtime error, without retry | Frozen contract and runner are fail-fast; all 36 completed with zero such errors | Complete |
| Blind outcome comparison and decision criteria | Each route has a hash-bound candidate-blind packet, condition key, complete rubric reviews, and preregistered outcome decision | Complete, author-side blind review |
| Do not treat one run as deployment evidence | Every outcome has `deployment_authorized: false`; the strongest allowed positive result only advances to a larger campaign | Complete |
| Report model behavior by task/category | `MODEL_DECISION_MAPS.md`, `CAMPAIGN_DOSSIER.json`, and `PILOT_RESULTS_ANALYSIS.md` | Complete |

The user explicitly authorized execution in the conversation after manifest review. That human
authorization is not represented as a synthetic repository approval record.

## Results

- **Sol:** `inconclusive_rewrite_or_repeat`. Its single deterministic improvement was a lexical
  evaluator artifact, not a human-reviewed outcome improvement; candidate cost increased.
- **Terra:** `discard_no_effect`. The candidate did not change the observed behavior and increased
  latency.
- **Luna:** `discard_no_effect`. The candidate did not improve human-reviewed outcomes and increased
  cost.

No candidate is authorized for runtime deployment. The evidence supports retaining the baseline and
using the observed product problems to design the next comprehension and execution experiments.

## Evidence boundaries

The completion audit proves that this particular 36-run pilot was performed and reviewed according
to its frozen contract. It does not prove repeatability, independent reviewer agreement, general
model traits, or access to hidden reasoning. All outcome reviews were candidate-blind while scoring,
but author-side and performed by a reviewer with prior exposure to campaign progress.
