# Ken memory contract implementation plan

> **For agentic workers:** Implement inline with superpowers:executing-plans;
> verify the contract with offline regression tests before product changes.

**Goal:** Preserve Ken's declared memory justification and input validity through
Infinidev's facade and task recall, without a second memory store.

**Architecture:** Negotiate structured JSON using each published tool schema.
Keep the existing full recall API and retain justification/validity dictionaries
without flattening away future fields. Legacy servers retain their old arguments.

**Tech stack:** Python, existing MCP manager, pytest; no dependencies or migrations.

## Design and alternatives

The user delegated useful features and implementation choices to the coordinating
agents. A parallel memory store would duplicate Ken's dependency checks. Using
answer/summary now would require a new result contract and omission semantics.
The chosen scope preserves the current full-list contract and enriches its data.

- Keep absent metadata absent; never infer unchanged or truth from missing fields.
- Display all returned justification/validity fields and their limited meaning.
- Preserve metadata through the legacy memory_search snippet consumed by TaskRuntime.
- Add an optional justification argument to remember; omit it when absent, retaining
  Ken's documented behavior of revoking prior justification on an ordinary overwrite.
- A remember result is successful only when the structured response has ok=true.
- Do not change Bruma, its active process, configuration, or immutable update10.

## Task: maintain structured memory across the facade

- [x] Add offline tests in tests/test_ken_memory_contract.py for schema-negotiated
  JSON, legacy schemas, full metadata including extensions, missing metadata,
  TaskRuntime recall, and successful/rejected remember writes.
- [x] Observe failures against the existing facade.
- [x] Update src/infinidev/engine/ken_client.py: schema-aware _call, optional
  MemoryHit metadata, explicit rendering caveat, legacy snippet preservation,
  optional remember justification and strict success check.
- [x] Run focused MCP/task regressions, then the full Python3.13 suite.
- [x] Review, commit and push validated changes. Report live validation separately.

## Evidence and review focus

Source-derived fixture: /tmp/ken-recall-contract-2026-11-14.json, provided by the
Ken agent from Rust implementation and frozen domain fixtures; no endpoint capture.
Exercise unchanged/stale/unknown/untracked, nested extensions, schemas that do not
advertise JSON, rejected writes, and legacy payloads lacking metadata. No automatic
semantic classification, source revalidation, or assertion of conclusion truth.

## Verification

Baseline27PASS. Regression12FAIL/4PASS before the fix, then43PASS focused.
Independent Ken contract review found no blockers. Full Python3.13 suite:
5886PASS/13SKIP/0FAIL,11warnings,587.05s. No live Bruma installation/test
of this change. Review receipt: /tmp/infinidev-ken-memory-review.md.
