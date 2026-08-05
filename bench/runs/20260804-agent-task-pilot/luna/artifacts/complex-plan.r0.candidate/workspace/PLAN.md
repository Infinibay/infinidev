# Tenant export change plan

## 1. Goal and non-goals

Add asynchronous tenant-record exports while preserving the current synchronous API and its existing behavior until adoption is proven. The change must give operators progress visibility, cancellation, idempotent retry, and an audit trail, while treating export contents as personal data. The existing worker processes one tenant at a time.

This is an execution plan only; it does not authorize implementation or change the current API contract, retention policy, access policy, or worker concurrency. No code should be removed or switched off as part of the initial launch.

## 2. Invariants and acceptance boundary

The following are invariants for every phase:

- Existing synchronous callers continue to receive the current contract and behavior; the asynchronous path is additive and cannot silently redirect them.
- An export is scoped to exactly one tenant and is authorized both when requested and when its status or result is accessed.
- Export payloads, temporary files, logs, traces, metrics labels, and audit records are reviewed for personal-data exposure and least-privilege access.
- A retry of the same logical export does not create duplicate effects or ambiguous output. A failed or cancelled attempt must have an explicit terminal state.
- Cancellation is observable, safe at worker boundaries, and does not report success for a partially produced artifact.
- Progress is truthful, bounded, and does not disclose tenant data through labels or messages.
- The single-tenant-at-a-time worker constraint is retained unless separately approved.

The change is complete only when the evidence matrix in Section 7 is attached to the release record and all launch gates have passed.

## 3. Consequential open decisions

These are deliberately unresolved and must be recorded with decision owner, date, rationale, and links before the corresponding phase is closed. Do not infer a product decision from an implementation convenience.

| Decision | Options / consequence | Required owner and decision point |
|---|---|---|
| Retention | **24 hours** limits personal-data exposure and storage cost; **7 days** improves download flexibility but increases exposure and cleanup burden. | Product + privacy/security before Phase 2. |
| Asynchronous API shape | Define the additive submit/status/download/cancel contract, job identifiers, terminal states, error semantics, and whether the existing synchronous endpoint may ever delegate. Compatibility risk differs by choice. | Product + API owner before Phase 2; document versioning and deprecation policy. |
| Authorization model | Confirm who may request, view progress, cancel, and download; decide tenant-admin/operator behavior and whether authorization is rechecked on every operation. | Security + product before Phase 2. |
| Artifact handling | Choose storage location, encryption/key ownership, download mechanism, and whether artifacts are streamed or materialized. This controls data exposure and cleanup semantics. | Security/platform before Phase 2. |
| Idempotency scope | Decide the client idempotency-key lifetime and whether a retry reuses an existing job/artifact or creates a new attempt under one logical export. | API owner + product before Phase 2. |
| Progress contract | Decide whether progress is item-count, stage-based, or indeterminate, and its update cadence/accuracy guarantees. | Product + operations before Phase 3. |
| Cancellation semantics | Choose whether cancellation is best effort at item/batch boundaries, what happens to queued/running jobs, and the user-visible terminal state. | Product + worker owner before Phase 3. |
| Audit and telemetry policy | Define immutable audit event schema, retention/access, required actors/correlation IDs, and which metrics/traces may contain identifiers. | Security/compliance + operations before Phase 3. |
| Rollout audience and success thresholds | Select internal tenants, percentage cohorts, observation duration, error/latency/cancellation thresholds, and who can advance or halt a cohort. | Product + operations before Phase 4. |
| Rollback scope | Confirm whether rollback disables submission only, also cancels queued jobs, and how in-flight jobs/artifacts are handled without data loss or orphaned access. | Incident owner + platform before Phase 4. |

## 4. Reversible execution phases

Each phase produces a reviewable artifact and has an explicit exit gate. A failed gate stops progression; it does not require undoing prior data migrations unless the rollback procedure says so.

### Phase 0 — Contract, threat model, and observability design

- Inventory the current synchronous API, callers, authorization checks, worker boundaries, storage, cleanup jobs, and existing audit/telemetry conventions.
- Write the additive asynchronous contract and state machine without changing the synchronous path.
- Threat-model personal-data creation, transport, storage, logs, downloads, cancellation, retries, cleanup, and support access.
- Define correlation/request/job IDs, progress metrics, audit event taxonomy, dashboards, alert thresholds, and redaction rules.
- Resolve the decisions required before implementation; record alternatives and rejected options.

**Exit gate:** reviewed API/state-machine and threat-model documents; privacy/security sign-off; dashboard and evidence templates ready; no compatibility ambiguity.

### Phase 1 — Dark, disabled capability and isolated test path

- Implement the smallest additive job boundary behind a server-side feature flag, with the synchronous API unchanged and the default asynchronous submission path disabled for production users.
- Use non-production or synthetic tenant fixtures where possible; ensure artifact access and logs follow the approved authorization and redaction model.
- Add deterministic idempotency handling, explicit states, cancellation checkpoints at safe worker boundaries, and audit events for request, state transitions, access, cancellation, retry, completion, failure, and cleanup.
- Add retention cleanup designed to be safe to rerun and to report orphaned artifacts.

**Reversibility:** disable the flag and revoke test access; no existing synchronous traffic depends on the new path.

**Exit gate:** automated and integration evidence for compatibility, authorization, cancellation, idempotent retry, audit, cleanup, progress, and telemetry; security review of artifacts and logs.

### Phase 2 — Internal canary and operational rehearsal

- Enable the asynchronous path only for approved internal tenants or test identities, while keeping synchronous callers unchanged.
- Exercise normal, large, malformed, unauthorized, duplicate-submit, retry-after-timeout, cancel-before-start, cancel-while-running, worker restart, storage failure, and cleanup scenarios.
- Verify one-tenant-at-a-time processing, backpressure, truthful progress, alert routing, operator views, and support/runbook steps.
- Run a rollback rehearsal using the same controls intended for production and preserve the resulting evidence.

**Reversibility:** turn off the cohort flag or submission gate; leave existing synchronous service available; quarantine/delete only artifacts according to the approved retention and audit procedure.

**Exit gate:** canary thresholds met for the agreed observation window, no unresolved high-severity privacy/authorization issue, rollback rehearsal completed, and on-call owner accepts the runbook.

### Phase 3 — Staged production rollout

- Enable cohorts progressively according to the approved rollout table, with a pause/observe window after each increase.
- Monitor submission success, queue age, worker utilization, export duration, cancellation latency/success, retry deduplication, authorization denials, artifact access errors, cleanup age, audit delivery, and personal-data exposure alerts.
- Compare asynchronous outcomes with the existing synchronous behavior for equivalent fixtures; do not use adoption alone as proof of correctness.
- Keep a kill switch and the current synchronous API operational throughout the rollout. Record every cohort change and approval.

**Exit gate:** all rollout thresholds remain within bounds, evidence matrix is complete, support and operations acknowledge ownership, and product explicitly approves adoption expansion or steady state.

### Phase 4 — Adoption review and controlled steady state

- Review adoption, support burden, cost, reliability, privacy findings, and compatibility evidence against the approved success criteria.
- Decide whether to retain, expand, revise, or stop the asynchronous path. Any future change to synchronous behavior requires a separate compatibility decision and plan.
- Close or transfer follow-up work with owners and due dates; preserve the release record and runbook links.

## 5. Verification and evidence matrix

Evidence must identify build/version, environment, test data classification, timestamp, operator, and outcome. Real personal data is not required for tests unless separately approved; synthetic fixtures should cover tenant isolation and sensitive fields.

| Requirement | Verification | Completion evidence |
|---|---|---|
| Compatibility | Contract tests and regression tests exercise existing synchronous callers before and during each flag state; compare status codes, response shape, and behavior. | Test report plus before/after compatibility diff. |
| Authorization | Matrix-test request, status, cancel, and download across tenant, admin, operator, unauthenticated, and cross-tenant identities; test authorization recheck after job creation. | Signed authorization matrix and security review. |
| Cancellation | Test queued, running, boundary, repeated, and late cancellation; verify no partial artifact is downloadable as successful and final state is truthful. | State-transition logs, test report, and operator observation. |
| Idempotent retry | Repeat submissions with the same and different keys across timeouts, worker restarts, and terminal states; confirm deduplicated effects and unambiguous artifact ownership. | Idempotency test report and sample job ledger (redacted). |
| Audit events | Verify actor, tenant/job correlation, action, outcome, timestamp, and reason for request, access, cancel, retry, completion/failure, and cleanup; verify tamper/access controls and no payload leakage. | Event schema review, redacted event samples, delivery/alert evidence. |
| Retention cleanup | Test the selected 24-hour or 7-day policy at boundary times, failed/cancelled jobs, expired download links, reruns, orphan recovery, and clock/time-zone behavior. | Cleanup run output, deletion counts, and expired-access test. |
| Telemetry and progress | Verify stage/item progress semantics, queue and duration metrics, cancellation/retry/error rates, dashboards, alerts, and redaction of tenant identifiers/personal data. | Dashboard screenshots or links, alert test results, telemetry schema review. |
| Rollback | Disable the feature during queued and running work; verify synchronous API availability, safe job/artifact handling, access revocation as approved, operator notification, and audit trail. | Recorded rollback rehearsal and incident/runbook checklist. |

## 6. Rollout controls

- Use a server-side kill switch independent of client release and protect it with least-privilege access, audit logging, and a tested change procedure.
- Roll out by an explicit cohort table: cohort, enablement time, approver, observation window, volume, thresholds, and decision. Start with internal/synthetic tenants, then the smallest approved production cohort.
- Define stop conditions before launch: authorization or cross-tenant access defect, personal-data leakage, audit loss, cleanup failure beyond the retention boundary, cancellation not honoring the contract, duplicate artifacts/effects, synchronous regression, or agreed reliability/latency/queue thresholds.
- During a stop, freeze cohort expansion, preserve evidence, notify on-call/security/product, and use rollback criteria rather than ad hoc remediation.

## 7. Immediate rollback procedure

1. Declare the rollout stopped and record time, build, cohort, observed symptom, and incident owner.
2. Disable asynchronous submissions through the kill switch; confirm the current synchronous API still serves existing callers. If necessary, block new export requests at the gateway while preserving authenticated status/download handling needed for safe cleanup.
3. Decide queued and running job treatment according to the approved rollback scope: allow safe completion, cancel at worker boundaries, or quarantine. Do not expose partial or unauthorized artifacts.
4. Revoke or expire affected download access if the incident involves authorization, leakage, or artifact integrity; preserve required audit records and forensic metadata without retaining payloads beyond policy.
5. Verify workers, queues, storage, cleanup, audit delivery, dashboards, and alerts have returned to the known-good operating mode. Notify affected operators/users using the approved communication path.
6. Capture rollback evidence, conduct a privacy/security and incident review, and keep the feature disabled until the cause, remediation, and re-entry tests are approved. Re-enable only via a new staged rollout.

## 8. Durable completion handoff

The release is not complete on code deployment alone. The owner must place a durable handoff in the release/change record containing:

- final decisions from Section 3, including retention and rationale;
- API/state-machine and compatibility contract version;
- threat model, privacy/security approvals, data-flow and access-control review;
- test/evidence matrix from Section 5 with links to immutable reports and build/environment identifiers;
- rollout cohort log, thresholds, observations, approvals, and any deviations;
- dashboard, alert, feature-flag, queue, storage, cleanup, and audit-system links;
- rollback rehearsal result and the exact kill-switch/rollback runbook;
- named product, engineering, security/privacy, and on-call owners with escalation paths;
- known limitations, unresolved risks, follow-up tickets, due dates, and next adoption decision date;
- retention-cleanup confirmation and an inventory/expiry record for test and production artifacts.

The handoff owner obtains explicit acceptance from product, operations/on-call, and security/privacy where applicable. A later engineer must be able to determine what is enabled, for whom, why, how it was verified, and how to disable it without relying on this chat or tribal knowledge.
