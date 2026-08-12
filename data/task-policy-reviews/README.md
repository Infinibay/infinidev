# Task-policy review annotations

This directory contains the human-authored decisions used to train and evaluate Infinidev's
conditional task-policy classifier. The 37 JSONL ledgers contain 2,901 decisions across Open-SWE
and WildChat candidate queues.

Each row contains only:

- the upstream candidate identifier;
- whether the row is included;
- zero or more Infinidev policy labels;
- an optional zero-label reason;
- a short reviewer-authored decision note.

Original user requests, conversations, issue bodies, model responses, credentials, and generated
splits are not included. The candidate text is downloaded separately from pinned upstream
revisions and remains governed by its source terms. Run the guarded bootstrap from the repository
root:

```bash
uv run python -m bench.task_policy_data_bootstrap
```

The bootstrap verifies all candidate hashes, joins these ledgers, preserves source and
near-duplicate families, and recreates the fixed train/calibration/evaluation split. Scripts may
validate or partition annotations but must never infer replacements for missing human labels.

See [LICENSE.md](LICENSE.md) for the data license, upstream attribution, and scope boundaries.
The surrounding Infinidev software remains MIT licensed.

## Provenance summary

| Ledger directory | Candidate source | Pinned revision | Source terms |
| --- | --- | --- | --- |
| `open-swe/` | NVIDIA Open-SWE-Traces, with issue statements sourced from SWE-rebench-V2 | `ad4805a5aa7de70d99cab0bb8f99b15304c76de0` | CC BY 4.0; source repositories identify MIT, Apache-2.0, BSD-2-Clause, or BSD-3-Clause |
| `wildchat/` | AllenAI WildChat | `f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4` | ODC-By 1.0 for the source database; individual content may carry separate rights |

The acquisition manifests generated under `.infinidev/external-data/` retain row-level source
metadata and exact artifact hashes. That directory is deliberately ignored because it contains
the downloaded candidate text.

## Privacy and publication audit

At publication time, the committed ledgers contained exactly the five documented fields, no
email addresses, URLs, IP addresses, API-key-shaped strings, or source prompt bodies. The longest
review note was 181 characters. Future edits must preserve this minimized schema and rerun the
same privacy and secret audit before release.
