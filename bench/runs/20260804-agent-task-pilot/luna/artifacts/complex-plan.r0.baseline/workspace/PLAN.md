# Tenant export change delivery plan

## 1. Purpose, scope, and guardrails

Deliver asynchronous tenant-record exports while preserving the existing synchronous API until adoption and safety evidence justify a change. The export path handles personal data, so authorization, auditability, bounded retention, and operator visibility are release gates rather than follow-up work.

**In scope**

- An asynchronous export request/job contract and worker integration, using the existing worker's one-tenant-at-a-time limit.
- Progress visibility, cancellation, idempotent retry, and durable audit events.
- Authorization checks at request, status/download, cancel, retry, and worker execution boundaries.
- Retention enforcement and cleanup for export artifacts and related metadata.
- Compatibility protection for the current synchronous API, telemetry, staged rollout, immediate rollback, and operator documentation.

**Out of scope unless separately approved**

- Replacing or changing the current synchronous API behavior.
- Increasing worker concurrency, redesigning tenant isolation, or adding unrelated export formats.
- Retaining export data beyond the approved policy or exposing it through a new client surface without an explicit review.

All implementation changes must be behind a disable-able feature flag/configuration path. No phase may require deleting the synchronous path; rollback must be possible by disabling asynchronous admission and draining or stopping asynchronous work safely.

## 2. Open decisions and approval record

Resolve these before production enablement; record the decision, owner, date, rationale, and evidence link in the change ticket/runbook. Until resolved, use the more conservative option in lower environments and do not silently infer product policy.

| Decision | Options / default for planning | Why it is consequential | Required approver / evidence |
|---|---|---|---|
| **Retention** | 24 hours or 7 days; default operational implementation should support a configured value and test both boundaries, but production must use the product-approved value | Personal-data exposure window, storage cost, cleanup urgency, and customer expectations | Product + privacy/security; approved policy and cleanup evidence |
| **Export format and delivery surface** | Existing record representation and existing authenticated download mechanism where possible | Compatibility, data minimization, client support, and authorization surface | Product/API owner; contract and compatibility review |
| **Job identity/idempotency scope** | Stable client idempotency key per tenant/request parameters versus server-generated job identity | Determines duplicate prevention, retry semantics, and whether a retry can produce a second artifact | API owner + operations; contract tests and state-machine review |
| **Cancellation semantics** | Best-effort cancellation at safe checkpoints versus immediate interruption | Affects partial artifacts, worker safety, and what status users/operators can trust | Worker owner + product; cancellation state contract and tests |
| **Progress definition** | Records/items processed, stages, or bounded percentage where total is known | Incorrect progress damages operator trust; raw counts may reveal sensitive information | Product + operations; dashboard/contract review |
| **Authorization model** | Reuse existing tenant-scoped export/read permissions, with explicit operator break-glass rules if needed | Export and status/download access can leak personal data even when creation is protected | Security/privacy; negative authorization tests and audit review |
| **Audit event schema and destination** | Existing audit pipeline if it supports immutable actor/tenant/job/action/result fields; otherwise approved durable sink | Required for accountability and incident investigation | Security/compliance; event samples and delivery/retention evidence |
| **Rollout thresholds and owner** | Define error, authorization-denial, cancellation-latency, queue-age, and cleanup-SLO thresholds before canary | Determines whether a staged rollout is safe to continue or must roll back | Service owner + on-call; signed rollout checklist |

The implementation plan must not mark the change production-ready while any decision above is unresolved.

## 3. Target behavior and state contract

Document and review the API/schema before coding. A request creates or returns the idempotent job identity without blocking on export completion. Status exposes only authorized tenant/job information and an honest state such as `queued`, `running`, `cancelling`, `cancelled`, `succeeded`, `failed`, or `expired`, with progress, timestamps, and a non-sensitive failure reason where appropriate. Download is available only for an authorized successful, unexpired job.

Define transitions and retries explicitly: duplicate requests with the same approved idempotency key and equivalent parameters return the existing job; conflicting parameters are rejected; retry of a retryable failure reuses the logical request but cannot expose an old or unauthorized artifact. Cancellation is idempotent, prevents future work where possible, removes or quarantines partial output, and records the final outcome. Worker recovery must not create duplicate artifacts or skip audit events.

## 4. Reversible execution phases

Each phase produces an artifact and a go/no-go decision. Keep the feature disabled outside the environments named in the phase. A phase is reversible by reverting its schema/configuration change where safe or by disabling the flag; avoid irreversible data migrations until rollback and cleanup have been rehearsed.

### Phase 0 — Resolve policy and baseline

- Approve the open decisions, data classification, threat model, state machine, API compatibility contract, and retention value.
- Inventory current synchronous API behavior, tenant authorization helpers, worker limits, artifact storage, audit pipeline, metrics, and operational ownership.
- Establish baseline synchronous latency/error/traffic and storage/cleanup telemetry.
- Write the operator runbook, escalation contacts, feature-flag ownership, and rollback command/procedure.

**Exit evidence:** signed decision record, reviewed design/state diagram, baseline dashboard, threat/authorization review, and a tested rollback checklist.

### Phase 1 — Build dark-path foundations

- Add the smallest job/artifact metadata model and indexes needed for idempotency, state transitions, ownership, timestamps, and cleanup; use additive, rollback-safe changes.
- Implement authorization and audit event contracts before export data flows.
- Add feature-flagged asynchronous admission and worker orchestration without changing synchronous responses.
- Add structured progress, cancellation, retry, retention, cleanup, and telemetry hooks; do not log personal export contents.

**Exit evidence:** unit/contract tests, migration rollback or forward-compatibility rehearsal, static/security review, and dark-path telemetry showing no synchronous regression with the flag off.

### Phase 2 — Non-production end-to-end rehearsal

- Run representative multi-tenant fixtures, including personal-data markers, large and empty tenants, failures, worker restarts, duplicate requests, cancellation at each checkpoint, expired artifacts, and unauthorized actors.
- Verify one-tenant-at-a-time worker behavior and resource limits.
- Exercise cleanup at both retention boundary values and confirm no artifact or sensitive metadata remains past policy.
- Rehearse disablement while jobs are queued/running, then restore service using the synchronous API.

**Exit evidence:** repeatable test report, audit samples, dashboard screenshots/queries, storage cleanup report, and rollback rehearsal result approved by service and security owners.

### Phase 3 — Production canary

- Enable asynchronous admission for an explicitly allowlisted tenant cohort at low volume; keep synchronous API behavior unchanged and retain a kill switch.
- Monitor continuously for compatibility, authorization failures, queue age, progress freshness, cancellation latency, retry duplication, audit delivery, retention cleanup, telemetry health, and personal-data leakage.
- Compare against Phase 0 baselines and pause or roll back on any pre-agreed threshold breach or unexplained security/audit anomaly.

**Exit evidence:** canary window report with cohort, traffic, thresholds, incidents, sampled job lifecycles, and owner sign-off.

### Phase 4 — Staged expansion and adoption proof

- Expand by bounded cohorts/traffic steps only after the prior step meets thresholds; keep a control cohort on the synchronous API.
- Review adoption, success rate, support signals, cost, worker capacity, and customer-visible compatibility at every step.
- Do not remove or alter the synchronous API until the approved adoption target and all safety evidence are met; if the target is not met, keep both paths and revisit the decision.

**Exit evidence:** final adoption and safety review, explicit approval for any API deprecation/change, and a maintained rollback-ready configuration.

## 5. Verification and completion evidence

The release checklist must link immutable CI results, staging runs, dashboards, and audit/storage queries. Minimum evidence must cover:

- **Compatibility:** existing synchronous API contract, status codes, payloads, latency/error baseline, and flag-off behavior are unchanged.
- **Authorization:** allowed and denied create/status/download/cancel/retry cases across tenants and operator roles; no cross-tenant access or artifact guessing.
- **Cancellation:** queued, running, checkpoint, repeated, and post-completion cancellation; final state, partial-output handling, bounded cancellation latency, and audit event.
- **Idempotent retry:** duplicate submissions, concurrent duplicates, worker crash/restart, retryable/non-retryable failures, parameter conflicts, and exactly-once logical artifact exposure.
- **Audit:** create, access/download, progress/cancel request and outcome, retry, success/failure, expiry, cleanup, and rollback/disable events include actor, tenant, job, timestamp, outcome, and correlation ID without personal data.
- **Retention and cleanup:** approved 24-hour or 7-day policy, boundary timestamps, failed/cancelled artifacts, repeated cleanup, failures/alerts, and proof that storage and metadata are removed or handled as approved.
- **Telemetry:** request and job rates, queue age, worker utilization, progress freshness, success/failure/cancel/retry counts, authorization denials, audit delivery, cleanup lag/failures, artifact storage, and synchronous compatibility signals; dashboards and alerts have owners.
- **Rollback:** disable flag during each job state, confirm no new async admissions, safe handling of in-flight work, synchronous recovery, no orphaned artifact exposure, and post-rollback audit/telemetry.
- **Security/privacy:** threat-model findings closed, logs and traces redacted, access controls reviewed, and retention approval recorded.

Use deterministic fixtures plus production-like load; test time-dependent retention with an injectable clock or controlled environment rather than waiting in an unbounded test. Record known limitations instead of weakening assertions.

## 6. Rollout, operations, and rollback

Before each expansion, the on-call confirms dashboards, alerts, capacity, audit delivery, cleanup health, cohort boundaries, and the kill switch. Pause/rollback triggers include any unauthorized access or data leak, missing audit events, duplicate exposure, cleanup failure beyond the policy/SLO, stuck cancellation, queue/resource exhaustion, material synchronous regression, or any unresolved threshold breach.

**Immediate rollback procedure:**

1. Announce the incident/change state and disable asynchronous admission globally using the tested feature flag/configuration; verify new requests remain on the unchanged synchronous API or receive the approved safe response.
2. Stop or pause new async worker claims. Let safe in-flight checkpoints finish only if the runbook says this cannot expose data; otherwise cancel/quarantine them. Do not delete evidence before audit capture.
3. Revoke/disable async download access if required, identify queued/running/succeeded artifacts, and prevent unauthorized access while preserving approved audit records.
4. Confirm synchronous health, authorization, telemetry, and audit delivery; page owners for orphaned jobs, cleanup failures, or data-integrity concerns.
5. Preserve logs/metrics/events for investigation, reconcile job/artifact state, and run approved cleanup or quarantine. Never bypass retention or audit controls as a shortcut.
6. Record cause, impact, cohort, timestamps, decisions, and recovery evidence. Re-enable only after a reviewed fix and a fresh canary; otherwise leave the flag off.

A rollback must be operationally independent of the new async code path and executable by the on-call without a code deploy.

## 7. Durable completion handoff

The change is complete only when the repository/change record links: approved decisions and data classification; final API/state/idempotency/cancellation contracts; migration and configuration records; CI and end-to-end verification; authorization/privacy/security approvals; rollout reports and thresholds; rollback rehearsal and last-known-good configuration; dashboards/alerts with owners; retention/cleanup query results; audit schema/sample evidence; and the operator runbook.

The handoff names a service owner, privacy/security owner, on-call rotation, feature-flag owner, storage/cleanup owner, and escalation path. It includes current adoption and control-cohort metrics, outstanding risks with due dates, known limitations, incident/rollback history, and the exact command or configuration location for disablement. Schedule a post-rollout review to decide whether to retain both APIs, revise retention, or propose a separately approved synchronous API change; none of those decisions is implied by this plan.
