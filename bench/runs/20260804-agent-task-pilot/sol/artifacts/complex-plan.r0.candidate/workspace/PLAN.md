# Tenant Export Change — Implementation and Delivery Plan

## 1. Purpose and success criteria

Introduce an asynchronous, tenant-scoped export path while preserving the current synchronous API unchanged until measured adoption proves that a transition is safe. Exports can contain personal data, so authorization, artifact protection, retention, cleanup, and auditability are release conditions rather than follow-up work.

The change is complete only when:

- the current synchronous API remains behaviorally compatible and available;
- authorized users can submit an asynchronous export, observe progress, cancel it, retry idempotently, and retrieve the result;
- tenant boundaries and authorization are enforced at every operation, including result download;
- the existing worker still processes no more than one tenant at a time unless a separately approved capacity change is made;
- required audit events and operational telemetry are present and usable;
- expired artifacts and associated personal data are removed according to the selected retention policy;
- staged rollout gates pass, immediate rollback is rehearsed, and completion evidence is handed off durably.

This repository currently contains requirements only; it has no implementation tree, package metadata, architecture documentation, or test suite. Therefore, component names and file paths below are intentionally not invented. Phase 0 must map these logical responsibilities to the real service, worker, persistence, storage, authorization, telemetry, and deployment mechanisms before implementation begins.

## 2. Scope and boundaries

### In scope

- An additive asynchronous tenant-export job lifecycle: submit, inspect status/progress, cancel, retry, and retrieve a completed result.
- Durable job state sufficient for idempotency, cancellation, progress visibility, audit correlation, cleanup, and recovery.
- Personal-data-safe result storage and access.
- Retention enforcement for export artifacts and any job data covered by the policy.
- Operator telemetry, alerting, runbooks, staged rollout, and immediate rollback.
- Compatibility and regression protection for the synchronous API.

### Out of scope unless separately approved

- Removing, changing, or silently redirecting the current synchronous API before adoption is proven.
- Increasing worker concurrency beyond the existing one-tenant-at-a-time constraint.
- A general-purpose job framework or unrelated export-format changes.
- Historical backfills or migration of existing synchronous exports into asynchronous jobs.
- Changes to product retention policy beyond implementing the selected 24-hour or 7-day option.

## 3. Consequential open decisions

These decisions materially affect product behavior, privacy risk, compatibility, or operability. They require named approval and must be recorded in an architecture/decision record before the listed gate. Implementation should not silently choose defaults.

| ID | Open decision / options to confirm | Why consequential | Required owner(s) | Blocking gate |
|---|---|---|---|---|
| D1 | **Retention:** choose 24 hours or 7 days; confirm whether the clock starts at job creation or successful completion, and whether failed/cancelled job metadata follows the same policy as artifacts. | Changes personal-data exposure, customer expectations, storage cost, and cleanup timing. | Product + Privacy/Security | Before Phase 1 data model and storage work |
| D2 | Define the additive async contract: submission fields, job states, progress representation, terminal error shape, polling/download behavior, and whether retries reuse or create a job identifier. | Becomes a client-facing compatibility contract. | Product + API owner | Before Phase 1 contract implementation |
| D3 | Define authorization at submit, status, cancel, retry, and download; decide how permission revocation after submission affects each operation and whether service/operator access is allowed. | Prevents cross-tenant or post-revocation disclosure of personal data. | Security + Product/API owner | Before Phase 1 authorization design |
| D4 | Select artifact storage and controls: encryption, tenant isolation, access method (service-streamed or short-lived signed access), link lifetime, region/residency, and secure deletion capability. | Determines exposure surface and whether retention can be enforced. | Security + Infrastructure/Data owner | Before Phase 2 artifact generation |
| D5 | Define cancellation semantics for queued, running, finalizing, and already-completed jobs, including race outcomes, cancellation latency target, and whether partial artifacts must be deleted immediately. | Users and operators need predictable cancellation without leaking partial data. | Product + Worker owner + Security | Before Phase 2 worker execution |
| D6 | Define idempotency scope and lifetime: identity key, tenant/principal/request binding, payload comparison, expiry, concurrent duplicate behavior, and retry rules after each terminal state. | Avoids duplicate exports and accidental cross-request result reuse. | API owner + Product + Security | Before Phase 1 persistence/API work |
| D7 | Approve audit event schema, destinations, retention/access controls, and required events (submit, authorize/deny, start, progress milestone if required, cancel request/outcome, retry/deduplication, completion, download, expiration, cleanup, and failures). | Audit data must be useful without unnecessarily copying personal data. | Security/Compliance + Operations | Before Phase 1 audit instrumentation |
| D8 | Set queue/backpressure policy under the one-tenant-at-a-time worker constraint: queue limit, fairness, admission failure behavior, timeout/staleness thresholds, and operator intervention. | Async adoption could overload the existing worker or starve tenants. | Worker owner + Operations + Product | Before production canary |
| D9 | Define “adoption proven” and rollout gates: eligible population, success/error/cancellation targets, latency and queue thresholds, minimum observation window and volume, and who approves each increase. | Prevents subjective rollout or premature synchronous API retirement. | Product + Operations + API owner | Before production exposure |
| D10 | Define rollback treatment for queued/running jobs and retained artifacts: drain, cancel, or finish; whether retrieval remains available; communication expectations; and maximum rollback time objective. | A flag that stops new submissions alone is not a complete or safe rollback. | Product + Operations + Security | Before production canary |

Decision records must include the chosen option, alternatives rejected, approvers, date, affected phases, and any planned revisit date. D1 must be resolved explicitly; do not ship using an implicit retention default. If a decision remains unresolved at its blocking gate, stop at that gate rather than embedding a temporary product behavior.

### Routine implementation choices

Once the consequential decisions are approved, implementers may choose ordinary names, module boundaries, migration tooling, test fixture organization, telemetry library calls, and equivalent internal algorithms using established repository patterns. These choices must not alter the approved API semantics, authorization model, retention, cancellation guarantee, idempotency behavior, rollout thresholds, or rollback behavior.

## 4. Required behavioral contract

Phase 0 converts this section into repository-specific interfaces and acceptance tests.

### Job lifecycle and progress

Use one documented state machine with allowed transitions and terminal states. At minimum, distinguish queued, running, succeeded, failed, and cancelled outcomes; add a cancellation-requested or finalizing state only if needed by approved semantics. Status responses must expose monotonic, understandable progress without claiming completion before the artifact is durable. Progress may be stage-based or numeric per D2, but must never regress and must identify stale/stuck jobs operationally.

Persist state transitions atomically enough that worker restarts do not turn terminal jobs back into active jobs or publish incomplete artifacts. Invalid transitions must fail safely and produce telemetry. Job and artifact identifiers must be non-guessable or independently protected; identifiers are never a substitute for authorization.

### Authorization and tenant isolation

Resolve tenant identity from the trusted authentication/authorization context, not from an untrusted request field alone. Re-run the approved authorization check independently for submit, status, cancel, retry, and result retrieval. Bind every job, idempotency record, artifact, audit event, and worker action to one tenant. Cross-tenant lookups must return the approved non-disclosing response and must not reveal job existence, progress, metadata, or artifact location.

Worker credentials and operator tools use least privilege. Download access must be short-lived or service-mediated according to D4, and logs, metrics, traces, errors, job metadata, and audit events must not contain exported records, credentials, or long-lived artifact URLs.

### Cancellation

Submission of a cancel request must be idempotent. The worker checks for cancellation at safe, bounded checkpoints selected to meet D5, including before publishing an artifact. A winning cancellation prevents later success publication, removes any partial artifact, records the final outcome, and emits audit and telemetry events. Tests must deterministically cover races between cancellation, completion, retry, and cleanup.

### Idempotent retry

Bind an idempotency key to tenant, authorized principal or approved caller scope, operation, and a canonical request fingerprint for the approved lifetime. An exact duplicate returns/references the established outcome; reuse with a different payload fails explicitly. Concurrent duplicates must result in one logical export. Define behavior for queued, running, succeeded, failed, cancelled, and expired jobs, and preserve enough data for the full idempotency window without extending personal-data retention unintentionally.

Worker execution must also tolerate redelivery/restart without creating or publishing duplicate artifacts. State transitions and artifact publication need a stable job identity and replay-safe boundary.

### Artifact lifecycle, retention, and cleanup

Write results to a temporary/non-public location, validate completion, then publish atomically. Apply approved encryption, isolation, residency, and access controls. Never expose partial artifacts.

A scheduled/recoverable cleanup process must locate artifacts eligible under D1, revoke access, delete the object and temporary parts, update job availability safely, and emit audit and telemetry outcomes. Cleanup must be idempotent and retry failures with bounded backoff; “not found” is successful only when ownership and prior state establish that the artifact is already absent. Define remediation and alerting for deletion failures. Verify actual storage absence rather than only deleting a database reference.

If metadata and artifacts have different retention obligations, document both and minimize retained personal data. Backups, replicas, caches, and signed links must be included in the data-lifecycle analysis, with any unavoidable delayed deletion documented and approved.

### Audit trail

Emit structured, correlated audit events for the D7-approved lifecycle and access events. Include timestamp, actor/service identity, tenant, job correlation identifier, action, authorization/result, and reason/status code as appropriate—but no exported record content or secret access URL. Audit emission must have defined failure behavior: security-critical actions must not silently proceed without the required record. Test audit completeness, ordering/correlation where guaranteed, duplicate/replay behavior, denial events, and access controls on audit data.

### Compatibility

The synchronous route, request/response schema, status codes, timing assumptions covered by its existing contract, authorization behavior, and error semantics remain unchanged. The async route is additive and protected independently. Do not make the synchronous endpoint enqueue work, return job responses, or depend on the new worker path under the guise of internal refactoring unless a separate compatibility review approves it.

Capture a baseline of existing contract and integration tests before changes. Run those same tests at every phase gate and compare representative responses. Deprecation or removal of the synchronous API requires a future, separately approved plan after D9’s adoption threshold and observation period are met.

## 5. Reversible implementation phases

Each phase uses expand-and-disable sequencing: add compatible data/contracts first, keep behavior dark or narrowly gated, prove it, and only then widen exposure. Database/storage additions remain backward-compatible while older application versions may run. Destructive cleanup of old structures is not part of this plan.

### Phase 0 — Discovery, decisions, and baselines

**Work**

1. Map logical responsibilities to actual repositories/components: synchronous API, authentication/authorization, worker, queue, job persistence, artifact storage, cleanup scheduler, audit sink, telemetry, feature-flag/configuration, and deployment pipeline.
2. Document the current synchronous contract and capture baseline compatibility, authorization, load, and failure behavior.
3. Inspect worker lease/retry/restart behavior and verify the one-tenant-at-a-time invariant.
4. Resolve D1–D10 by their gates; create the state machine, threat model/data-flow diagram, data classification, and failure-mode analysis.
5. Define stable feature controls separately for async submission, worker consumption, and result retrieval so rollback does not require a deployment.
6. Specify test environments and personal-data-safe fixtures; never use production export content in lower environments.

**Gate / evidence**

- Approved decision records, API/job contract, threat model, data-flow and lifecycle diagrams.
- Baseline synchronous compatibility test report and worker capacity measurements.
- Approved rollout scorecard and rollback runbook draft with named owners.
- Mapping from every requirement in Section 9 to a component, test, dashboard, and owner.

**Reversal**

No production behavior or data structures change. Revise decisions/design and repeat review.

### Phase 1 — Additive foundations behind disabled controls

**Work**

1. Add backward-compatible job/idempotency persistence and migrations using repository conventions; include tenant binding, state/versioning, timestamps, progress, cancellation, retry correlation, artifact reference, and cleanup status without storing export content unnecessarily.
2. Add the approved async API contract with authorization at every operation, but keep production submission disabled.
3. Add audit events and telemetry for API decisions and state transitions.
4. Implement cleanup bookkeeping and retention calculation using the explicit D1 policy/configuration; reject invalid or absent production configuration rather than silently choosing 24 hours or 7 days.
5. Keep the current synchronous path untouched and independently deployable.

**Gate / evidence**

- Migration forward/backward-compatibility tests across mixed application versions.
- API contract, tenant-isolation, authorization, idempotency/concurrency, state-machine, audit, and retention-boundary tests pass.
- Existing synchronous compatibility and regression tests remain unchanged and pass.
- Controls are confirmed off in production; no jobs can be accepted accidentally.

**Reversal**

Disable async submission and retrieval controls. Roll application code back while leaving additive persistence in place; do not use destructive down-migrations during an incident. Remove unused structures only in a later reviewed maintenance change after rollback risk has passed.

### Phase 2 — Worker execution, cancellation, artifacts, and cleanup in non-production

**Work**

1. Implement replay-safe worker execution while preserving one tenant at a time and the D8 queue/backpressure policy.
2. Generate artifacts in protected temporary storage and publish atomically only after success.
3. Add progress checkpoints and approved cancellation checks/cleanup behavior.
4. Implement recoverable retention cleanup across artifacts, temporary parts, metadata, caches/links, and applicable replicas.
5. Complete audit and telemetry coverage for worker, artifact access, cancellation, retry, failure, expiration, and cleanup.
6. Exercise restarts, redelivery, queue saturation, storage/audit outages, partial writes, permission changes, and cancel/complete/cleanup races.

**Gate / evidence**

- Non-production end-to-end tests prove submit-to-download behavior and all Section 9 evidence categories.
- Concurrency tests show duplicate submissions/redeliveries create one logical export and never cross tenant boundaries.
- Cancellation meets the approved latency/semantics and leaves no retrievable partial artifact.
- Time-controlled retention tests cover exact boundaries for the selected duration; storage inventory proves cleanup removes eligible objects and preserves non-expired ones.
- Failure-injection and restart tests pass; worker instrumentation proves the one-tenant-at-a-time constraint.
- Security/privacy review and operational readiness review approve progression.

**Reversal**

Stop worker consumption, disable submission, revoke test artifact access, run verified cleanup, and roll back application code while retaining compatible schema. Diagnose without exposing fixture export contents.

### Phase 3 — Production dark deployment and operator rehearsal

**Work**

1. Deploy additive schema and code with async submission and worker consumption disabled.
2. Validate configuration, permissions, audit delivery, dashboards, alerts, cleanup scheduling, and feature-control propagation without processing customer exports.
3. If policy permits, run a synthetic, non-personal-data smoke job through a dedicated test tenant.
4. Rehearse rollback, including disabling new submissions, stopping consumption, handling in-flight jobs per D10, preserving/revoking retrieval as decided, and confirming synchronous API health.

**Gate / evidence**

- Production configuration check confirms the approved retention value and no implicit default.
- Feature controls can be changed by authorized operators within the rollback objective and are themselves audited.
- Dashboards/alerts receive expected synthetic signals; runbooks and on-call contacts are verified.
- Timed rollback rehearsal meets D10 and records job/artifact outcomes plus synchronous compatibility checks.

**Reversal**

Keep all async controls disabled or return them to disabled. Roll back code if needed; leave additive schema dormant. Clean synthetic artifacts and verify their absence.

### Phase 4 — Internal canary

**Work**

1. Enable submission and consumption only for allowlisted internal/test tenants and authorized callers.
2. Start below queue limits; observe every job through cleanup expiration.
3. Compare async export content and authorization outcomes against the synchronous path using approved non-production/safely controlled data and comparison methods that do not leak personal data.
4. Collect usability and operator feedback on progress, cancellation, retry, and audit investigation.

**Gate / evidence**

- D9 minimum canary volume and observation window are met, including at least one complete retention/cleanup cycle at the selected duration.
- No cross-tenant access, orphaned/partial artifacts, unexplained duplicate jobs, missing required audit events, or unbounded queue growth.
- Compatibility, error rate, latency, cancellation, retry, cleanup, and telemetry thresholds all pass.
- Product, Security/Privacy, Operations, and service/worker owners sign off on cohort rollout.

**Reversal**

Remove canary eligibility and reject new async submissions. Handle queued/running jobs and retrieval exactly as D10 specifies; keep the synchronous API serving normally. Revoke and clean artifacts when policy/rollback semantics require it, and verify results before closing the incident.

### Phase 5 — Staged tenant rollout

**Work**

1. Expand only through an allowlist or percentage/cohort mechanism that can immediately return to zero: for example, approved pilot tenants, then small, medium, and broad cohorts. Exact sizes and dwell times come from D9 rather than being invented during deployment.
2. Require a written go/no-go review at every cohort boundary using the same scorecard.
3. Communicate feature behavior, retention, cancellation limits, and support path to eligible tenants before exposure.
4. Keep the synchronous API available and unchanged for all tenants throughout rollout.

**Gate / evidence at every cohort**

- Minimum volume/dwell period met; queue depth/age, job latency, success/failure, cancellation latency/outcome, retry/deduplication, artifact access, audit delivery, cleanup lag/failures, and synchronous API health satisfy D9.
- Authorization-denial and tenant-isolation signals show no anomalous disclosure.
- Worker remains within one-tenant-at-a-time capacity and D8 backpressure limits.
- Support issues and manual interventions are reviewed; named approvers record go/no-go.

**Pause/rollback triggers**

Immediately stop expansion for any threshold breach. Roll back to zero async eligibility for suspected privacy/authorization failure, artifact exposure, missing security-critical audit events, cleanup failure beyond its approved threshold, corruption, uncontrolled duplicates, queue overload threatening service health, or synchronous compatibility regression. Other metric breaches pause at the current cohort while owners investigate, unless D9 sets a stricter response.

**Reversal**

Set eligibility/submission to zero, stop or drain consumption according to D10, and keep/revoke retrieval according to the approved data-safety behavior. The synchronous API remains the fallback. Use the Section 7 runbook; do not delete schema during urgent rollback.

### Phase 6 — Adoption evaluation and durable completion handoff

**Work**

1. Evaluate adoption using D9’s pre-approved denominator, threshold, observation period, and quality gates; report both usage and failure/abandonment/cancellation patterns.
2. Do not remove or alter the synchronous API as part of this phase. If adoption is proven, propose any deprecation as a separate decision and plan with customer notice and compatibility analysis.
3. Finalize ownership, runbooks, evidence index, dashboards, alerts, decision records, and known-risk register.
4. Confirm all temporary rollout exceptions are removed or have owners and expiration dates.

**Gate / evidence**

- Durable handoff package in Section 10 is reviewed and accepted by named Product, API/service, worker, Security/Privacy, Operations/SRE, and Support owners.
- At least one production retention cleanup cycle is evidenced, or—if the approved 7-day duration makes waiting disproportionate before a rollout gate—a time-controlled non-production proof plus a scheduled, owned production follow-up is explicitly approved; final completion still requires observed production cleanup.
- Immediate rollback remains available and periodically tested while the async path is active.

**Reversal**

Keep the async feature gated or return eligibility to zero. Continue the synchronous API and retain compatible dormant structures until a separately reviewed removal change.

## 6. Verification strategy

Tests should use controlled clocks, deterministic worker checkpoints, isolated tenants, and non-sensitive fixtures. Every result must identify build/version, environment, configuration (including retention choice), test command or procedure, timestamp, and evidence location. Redact personal data, credentials, and artifact URLs.

### Automated test layers

1. **Unit/model tests:** allowed/invalid state transitions, monotonic progress, cancellation checkpoints, retention calculations and exact time boundaries, request fingerprinting, idempotency expiry, retry classification, redaction, and audit payload construction.
2. **API contract tests:** additive async schema/status/error behavior and unchanged synchronous request/response behavior.
3. **Authorization tests:** allowed/denied actions for submit/status/cancel/retry/download; tenant A cannot infer or access tenant B jobs/artifacts; permission revocation; operator/service roles; non-guessability does not replace checks.
4. **Concurrency/integration tests:** simultaneous identical and conflicting idempotency keys, worker redelivery/restart, cancel-versus-start/complete/retry/cleanup races, atomic artifact publication, and stale leases.
5. **End-to-end tests:** submit → progress → success → authorized retrieval → expiration → verified cleanup, plus failure, cancellation, retry, and denial paths.
6. **Compatibility tests:** existing synchronous contract/regression suite before and after every phase; representative response diff; mixed-version deployment/migration behavior.
7. **Security/privacy tests:** storage isolation/encryption/access expiry, URL/log/trace redaction, cross-tenant probes, least-privilege service access, and threat-model abuse cases.
8. **Resilience/load tests:** one-tenant-at-a-time enforcement, queue admission/fairness/backpressure, worker restart, storage/queue/audit/telemetry outage, cleanup retry, and load that models D9 rollout volumes.
9. **Operational tests:** dashboards, alerts, audit queries, support lookup, feature-control propagation, synthetic smoke checks, and timed rollback rehearsal.

### Evidence rules

- A passing API test alone is not authorization evidence; include denied and cross-tenant cases at every operation.
- A database status of “deleted” alone is not cleanup evidence; inventory the underlying storage and prove the artifact/temporary parts are absent and links no longer work.
- A returned existing job alone is not idempotency evidence; prove payload binding, concurrent deduplication, expiry, and worker replay safety.
- A cancel response alone is not cancellation evidence; prove terminal state, bounded worker stop, no later success, and no partial/retrievable artifact.
- Emitted logs alone are not audit evidence; query the destination and prove required fields/events, access controls, correlation, and failure handling.
- A disabled feature flag alone is not rollback evidence; rehearse in-flight handling, retrieval behavior, queue state, artifact treatment, and synchronous API continuity.

## 7. Immediate rollback runbook

The detailed commands and control names are filled in during Phase 0 after repository discovery. Store the tested runbook in the normal operational documentation system; do not depend on this planning file alone during an incident.

1. **Declare and coordinate:** incident lead records trigger, time, affected cohort/tenants, suspected data exposure, and D10 path; notify Security/Privacy immediately for possible disclosure.
2. **Stop exposure:** set async eligibility/submission to zero using the audited control; verify from an external/client perspective that new async jobs are rejected with the approved response. Do not alter the synchronous API.
3. **Control execution:** stop worker consumption or drain it according to D10. Enumerate queued/running/finalizing jobs and record a disposition for each; do not abandon them silently.
4. **Protect results:** preserve retrieval only if D10 and the incident type permit it. Otherwise revoke access, invalidate links, and delete affected partial/completed artifacts under approved retention/incident rules; verify actual absence.
5. **Stabilize:** roll application version back only if controls are insufficient. Keep additive schema; avoid destructive migrations. Confirm older and newer versions can coexist during rollback.
6. **Validate:** run synchronous compatibility and authorization smoke tests; inspect queue, worker, artifact, audit, cleanup, error, and telemetry signals. Confirm no new async work starts and every in-flight job has the intended terminal/retrieval outcome.
7. **Communicate:** provide status and tenant/support guidance based on impact; follow privacy incident procedures where applicable.
8. **Recover deliberately:** identify root cause, add a regression test, repeat affected phase gates and rollback rehearsal, and obtain named approval before re-enabling any cohort.

Rollback is complete only when submission is confirmed off, worker/in-flight state is reconciled, artifact access matches D10, synchronous compatibility is healthy, required audit evidence exists, and the incident record links all verification. Keep the kill controls and dormant schema until stability is demonstrated; cleanup/removal is a later reversible change.

## 8. Observability and operations

### Metrics and dimensions

Use bounded-cardinality identifiers; never put tenant IDs, job IDs, user IDs, filenames, or artifact URLs into metric labels unless the platform has an explicitly approved safe mechanism.

- submissions accepted/denied/deduplicated/rejected by reason;
- queue depth, oldest age, admission rejection, and per-stage wait time;
- active worker count and explicit one-tenant-at-a-time invariant violations;
- job transitions, progress age/staleness, duration, success/failure/cancel outcomes;
- cancellation request-to-terminal latency and late-cancel races;
- retries/redeliveries, duplicate prevention, idempotency conflicts/expiry;
- artifact publication/access failures and access-link expiry/revocation;
- retention eligibility, cleanup attempts/success/failure/lag, orphan and temporary artifact count;
- audit delivery success/failure/lag and telemetry pipeline health;
- synchronous API traffic, latency, errors, and compatibility indicators alongside async adoption.

### Logs, traces, dashboards, and alerts

Correlate with a safe job correlation identifier and deployment/version. Structured logs and traces record state/reason codes but not export records or access secrets. Provide dashboards for API/authorization, queue/worker, lifecycle/cancellation/retry, storage/cleanup, audit delivery, rollout/adoption, and synchronous API health.

Alerts need an owner, severity, threshold from D8/D9, response link, and test evidence. Page immediately on suspected cross-tenant access, public/invalid artifact access, security-critical audit loss, cleanup lag beyond policy, corruption, uncontrolled duplicate execution, worker invariant violation, or synchronous regression. Lower-severity alerts cover queue pressure, stuck progress, elevated failures/cancellations, and telemetry gaps. Monitor the monitors with dead-man/absence checks where appropriate.

## 9. Requirements traceability and completion evidence matrix

The final implementation must replace “repository mapping” with concrete components, tests, dashboards, runbooks, and evidence links discovered in Phase 0.

| Requirement | Planned control | Required completion evidence | Primary phase gate |
|---|---|---|---|
| Async export without changing synchronous API until adoption is proven | Additive gated async contract; independent unchanged sync path; separate future deprecation decision | Baseline and post-change compatibility tests, response comparison, production sync health dashboard, D9 adoption report | Every phase; final Phase 6 |
| Exports may contain personal data | Tenant isolation, least privilege, protected atomic artifact storage, redaction, retention/cleanup | Threat model, security/privacy test report, storage configuration review, cross-tenant denial tests | Phases 0–2 and each rollout gate |
| Retention is 24 hours or 7 days and unresolved | D1 approval; explicit validated policy; recoverable cleanup | Signed decision, boundary tests, production configuration capture, artifact inventory before/after cleanup | Before Phase 1; observed by Phase 6 |
| Progress visibility | Documented monotonic state/progress contract and stale-job telemetry | Contract/unit/end-to-end tests, dashboard screenshot/query export, operator exercise | Phases 1–4 |
| Cancellation | D5 semantics, idempotent request, bounded checkpoints, partial artifact removal | Deterministic race tests, latency measurements, artifact absence proof, audit events | Phases 2–4 |
| Idempotent retry | Tenant/caller/payload-bound key, concurrency control, replay-safe worker | Exact/conflicting/concurrent/expiry tests, redelivery/restart tests, deduplication telemetry | Phases 1–4 |
| Audit trail | D7 structured correlated events with delivery failure handling | Event matrix tests, destination queries, access-control review, outage test | Phases 1–4 |
| Existing worker handles one tenant at a time | Preserve invariant; D8 admission/backpressure/fairness | Concurrency/load test, worker active-count telemetry, queue saturation exercise | Phases 0, 2, and rollout |
| Staged rollout | Reversible eligibility cohorts and pre-approved D9 gates | Per-cohort scorecards, approvals, metric snapshots, support review | Phases 4–5 |
| Immediate rollback | Independent controls, compatible schema, D10 in-flight/artifact behavior | Timed rehearsal record, external disable check, reconciled jobs/artifacts, sync smoke tests | Phase 3 and periodically thereafter |
| Completion evidence breadth | Evidence index covers compatibility, authorization, cancellation, retry, audit, retention cleanup, telemetry, and rollback | Signed durable handoff manifest described below | Phase 6 |

## 10. Durable completion handoff

Create a versioned, access-controlled handoff package in the organization’s durable documentation/evidence systems. Links must be stable, readable by future on-call and audit owners, and subject to the organization’s evidence retention policy. Do not store personal data, credentials, raw export artifacts, or live signed URLs in the package.

The handoff manifest must contain:

1. **Scope and release identity:** deployed versions, environments, dates, feature-control names/defaults, eligible cohorts, and explicit statement that the synchronous API remains available.
2. **Decision index:** D1–D10 records with approvers, especially the selected retention policy and adoption/rollback thresholds.
3. **Architecture and data lifecycle:** repository/component ownership map, API and state-machine contracts, worker/queue design, threat model, data-flow diagram, artifact/storage controls, and cleanup lifecycle.
4. **Verification index:** immutable or versioned links to test runs and reports covering compatibility, authorization/tenant isolation, cancellation races, idempotent retry/redelivery, audit events/outages, retention boundaries and actual cleanup, telemetry/alerts, load/resilience, and rollback rehearsal. Record any non-passing result plainly with owner and disposition.
5. **Rollout history:** cohort definitions, go/no-go scorecards and approvers, metric snapshots, incidents, support findings, adoption numerator/denominator, and observation periods.
6. **Operations:** tested rollback and recovery runbooks, dashboard/alert links, audit query guide, cleanup remediation, stuck-job procedures, support guide, privacy incident path, and escalation/on-call contacts.
7. **Ownership:** named Product, service/API, worker, storage/data, Security/Privacy, Operations/SRE, Support, and audit/evidence owners; avoid team-only ownership where an accountable role can be named.
8. **Residual risk and follow-up register:** accepted risks, limitations, temporary exceptions, due dates, owners, and verification criteria. No unowned or indefinite follow-up qualifies for completion.
9. **Final acceptance:** dated sign-off by required owners, production cleanup evidence, current rollback-readiness result, and a scheduled review date for adoption and continued synchronous compatibility.

A future engineer should be able to determine what behavior was approved, reproduce the relevant tests, operate and roll back the feature, locate audit/cleanup evidence, and contact an accountable owner without relying on chat history or individual memory. That is the durable completion criterion.
