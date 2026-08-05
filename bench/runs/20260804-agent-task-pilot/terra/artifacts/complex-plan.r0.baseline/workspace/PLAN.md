# Tenant export delivery plan

## Goal and constraints

Add an **asynchronous** tenant-record export capability while preserving the current synchronous export API unchanged until adoption is proven. Export artifacts may contain personal data. The existing worker processes one tenant at a time, so the initial design and rollout must retain that concurrency limit. Operators need progress visibility, cancellation, idempotent retry, an audit trail, telemetry, a staged rollout, and an immediate rollback path.

## Consequential open decisions — resolve before production enablement

1. **Retention policy (product decision):** choose either **24 hours** or **7 days** for export artifacts and associated downloadable URLs. Document the data-classification rationale, exact expiry semantics, and who may approve an exception. Do not enable general availability until this is confirmed.
2. **Authorization model (security/product decision):** confirm which roles may create, view progress for, download, cancel, and retry an export; whether access is limited to the requester or includes tenant administrators/support operators; and whether authorization is rechecked at download time. Audit requirements depend on this decision.
3. **Export contents and delivery (product/security decision):** define the record schema, redaction/minimization rules, file format, encryption/storage location, and download mechanism. Confirm whether a completed export remains available when a requester’s authorization changes.
4. **Cancellation semantics (product decision):** define whether cancellation stops only queued work or also an in-progress tenant export, what happens to partial artifacts, and the terminal status exposed to callers.
5. **Idempotency and retry contract (API/operations decision):** define the idempotency-key scope and lifetime, duplicate-request response, retry eligibility and limit, and whether a retry replaces or links to the original job.
6. **Adoption proof and rollout gates (delivery decision):** set the pilot tenant cohort, success metrics, observation period, error/cancellation thresholds, and named approvers for advancing or pausing rollout. The synchronous API remains the supported default until these gates pass.

## Reversible delivery phases

### Phase 0 — contract and threat-model sign-off

- Write the async job state model (`queued`, `running`, `completed`, `failed`, `cancelled`, plus any approved transitional states), request/response contract, progress fields, and error codes.
- Resolve or explicitly defer each open decision above with an owner and decision date; deferment may not weaken retention, authorization, or audit controls.
- Threat-model personal-data handling: least-privilege access, encryption in transit and at rest, secret handling, download authorization, artifact deletion, and audit-event integrity.
- Define a feature flag that is disabled by default and independently disables job creation and retrieval/download paths if necessary.

**Reversibility:** documentation and disabled flags introduce no caller-visible behavior; revise contracts before implementation.

**Exit evidence:** approved contract, authorization matrix, retention policy, data-flow/threat-model review, and rollout-gate document are stored with the delivery record.

### Phase 1 — implement dark-path foundations behind a disabled flag

- Add durable export-job persistence with tenant identity, requester, idempotency key, status, timestamps, progress, retry lineage, cancellation marker, artifact metadata, expiry, and safe failure details.
- Add asynchronous submission/status endpoints or service operations without modifying the existing synchronous API contract or behavior.
- Enforce authorization at every action (create, status, download, cancel, retry); log denied access without exposing personal data.
- Queue jobs to the existing one-tenant-at-a-time worker. Ensure a worker restart safely resumes or marks work according to the approved state model.
- Produce immutable audit events for submission, authorization denial, start, progress milestones, cancellation, retry, completion/failure, download/access, and retention cleanup. Avoid personal-data payloads in logs or events.
- Store encrypted artifacts with an expiry derived from the chosen retention policy; make cleanup retryable and observable.
- Emit telemetry for job volumes, queue wait, duration, status outcomes, retries, cancellations, authorization failures, cleanup success/failure, and artifact age.

**Reversibility:** keep creation unavailable while the flag is off; database/artifact changes must tolerate orphaned jobs and artifacts so disabling the flag stops new work without data loss or schema rollback.

**Exit evidence:** code review confirms synchronous compatibility, privacy/security review passes, and automated verification below passes in a non-production environment.

### Phase 2 — controlled internal verification

- Enable submission only for a non-production/internal allowlist; keep public/tenant availability disabled.
- Exercise realistic volume while preserving one active tenant export in the worker. Inspect queue latency, progress accuracy, artifact access controls, audit records, and cleanup behavior.
- Reconcile each test artifact against its job and audit trail, then confirm deletion at the selected retention boundary (using controlled time advancement where available).
- Fix defects before expanding exposure; do not compensate for failures by weakening authorization, retention, or audit capture.

**Reversibility:** turn off the allowlist/feature flag, stop new jobs, cancel queued test jobs using the defined process, and run cleanup for internal artifacts.

**Exit evidence:** a dated internal test report with metrics, known limitations, retention-cleanup evidence, and approval to begin pilot.

### Phase 3 — tenant pilot with guarded rollout

- Enable the feature for a small, explicitly approved tenant cohort; keep the synchronous API unchanged and available.
- Monitor dashboards and alerts during the agreed observation period: job success/failure, queue wait, duration, progress freshness, retries, cancellation outcomes, access denials, audit delivery, cleanup lag, and worker saturation.
- Collect pilot feedback on progress visibility and cancellation/retry behavior without logging export contents.
- Advance only when the pre-agreed adoption and reliability gates pass; otherwise pause expansion and use the rollback procedure.

**Reversibility:** cohort membership and feature flags must be independently configurable and take effect without redeploying. Existing synchronous exports remain the immediate fallback.

**Exit evidence:** pilot cohort results, metric snapshots, incident/issue log, gate decision, and approver sign-off.

### Phase 4 — staged general availability and adoption review

- Expand cohorts incrementally only after each prior cohort meets the approved gates; maintain heightened telemetry and operator coverage.
- Continue offering the synchronous API unchanged until adoption is demonstrably proven under the Phase 0 criteria. Any proposal to change or retire it is a separate, approved compatibility decision.
- Periodically verify artifact cleanup, access reviews, audit-event completeness, and capacity under the one-tenant worker constraint.

**Reversibility:** halt at the current cohort or disable async job creation globally while retaining status/audit access needed for already-created jobs, subject to the approved security model.

## Verification matrix

Before pilot and after material changes, run and retain results for:

| Area | Required verification |
| --- | --- |
| Compatibility | Regression tests prove current synchronous API routes, response shapes, and behavior are unchanged with async disabled and enabled. |
| Authorization | Tests cover every role/action pair, cross-tenant isolation, authorization changes, denied downloads, and reauthorization at the approved enforcement points. |
| Progress | Tests prove queued/running/terminal status and progress values are accurate, ordered, and safe after worker restart. |
| Cancellation | Test queued and in-progress cancellation according to the approved semantics; verify partial artifacts and terminal status are handled correctly. |
| Idempotent retry | Test duplicate submission, network retry, worker retry, retry limits, and concurrent requests; confirm no duplicate export artifact or unsafe duplicate work. |
| Audit | Assert required audit events, actor/tenant/job correlation, timestamps, outcomes, and absence of personal-data payloads. |
| Retention cleanup | Test expiry calculation for the selected retention, deletion of artifact and access path, retry on cleanup failure, and cleanup telemetry/audit event. |
| Telemetry | Validate dashboard metrics, alert thresholds, correlation IDs, and that telemetry/logging does not leak export contents. |
| Capacity | Load-test queue behavior and latency while enforcing one active tenant export; document acceptable backlog limits. |
| Rollback | Rehearse flag disablement, cohort removal, queued-job handling, synchronous fallback, and operator communication; record recovery time. |

## Rollout controls

1. Deploy code and storage/schema prerequisites with async creation disabled.
2. Enable internal allowlist only after Phase 1 evidence and on-call/operator runbook review.
3. Enable the approved pilot cohort after Phase 2 approval; watch defined metrics and alerts for the agreed observation period.
4. Expand in pre-approved cohorts only when gates pass. Pause automatically or manually on authorization/audit/retention failures, personal-data exposure risk, worker saturation, or missed reliability thresholds.
5. Keep an operator dashboard showing job status, progress freshness, queue depth, cancellation/retry outcomes, cleanup lag, and rollout cohort/flag state.

## Immediate rollback procedure

1. Disable async job creation for the affected cohort or globally using the feature flag; do not change the synchronous API.
2. Remove affected tenants from the rollout cohort and communicate that synchronous export remains available.
3. Stop or cancel queued jobs according to approved cancellation semantics; prevent downloads if the incident involves authorization or artifact exposure.
4. Preserve audit records and incident evidence; quarantine or securely delete artifacts as directed by security/privacy response, rather than bypassing retention controls silently.
5. Validate no new async jobs are accepted, worker queue behavior is stable, synchronous exports function, and telemetry/audit alerts are understood.
6. Record the trigger, impact, affected jobs/tenants, actions, timing, and criteria for any re-enable. Re-enable only after remediation and the same gate approval required for the failed phase.

## Durable completion handoff

The delivery is complete only when the following are stored in the team’s durable operational documentation/release record and ownership is assigned:

- Final async API/job-state contract, compatibility statement for the synchronous API, and resolved open decisions with approvers.
- Authorization matrix, data-flow/threat-model review, retention policy, artifact storage/encryption configuration, and privacy/security sign-offs.
- Feature-flag names, default states, cohort configuration, rollout gates, named rollout owner, and immediate rollback runbook.
- Operator runbook for progress visibility, cancellation, idempotent retry, job failure handling, cleanup failures, audit lookup, and escalation.
- Dashboard links, alert definitions, on-call ownership, telemetry glossary, and baseline/pilot metrics.
- Automated test inventory and retained results for compatibility, authorization, cancellation, idempotent retry, audit, retention cleanup, telemetry, capacity, and rollback rehearsal.
- Audit-event schema/retention location and evidence of a successful audit and cleanup reconciliation.
- Pilot/adoption report, known limitations, incident history, final release decision, and the explicit criteria/date for reviewing whether synchronous API adoption is proven.
