# Tenant export change plan

## Goal and constraints

Add an asynchronous tenant-record export path while preserving the existing synchronous API unchanged until adoption is proven. Exports can contain personal data, so authorization, auditable access, secure storage, and bounded retention are release gates. The existing worker processes only one tenant at a time; the design must retain that constraint rather than introduce parallel tenant processing implicitly.

This repository contains only the requirement and its plan verifier, not the service implementation. The implementation owner should map the named integration points below to the actual API, job queue, worker, authorization policy, object store, audit pipeline, configuration system, and deployment controls before phase 1 begins.

## Consequential open decisions — resolve before implementation

1. **Retention period (product decision): 24 hours or 7 days.** This changes customer access expectations, storage cost, exposure window for personal data, cleanup schedules, and the audit policy. Record the selected duration as a centrally configured value; do not ship with an undocumented default.
2. **Export delivery and access model.** Confirm whether a completed export is downloaded through an authenticated endpoint, delivered to a controlled destination, or both. Define whether URLs are short-lived, single-use, tenant-scoped, and revocable. Never embed export contents or durable credentials in job status, logs, or audit metadata.
3. **Authorization policy.** Confirm which roles may request, view progress for, cancel, retry, and download an export; whether permissions are checked only at request time or again at download time; and whether an administrator may act across tenants. Re-check authorization for every externally callable operation.
4. **Export snapshot semantics.** Decide whether the export represents request-time data, a best-effort read during processing, or a consistent database snapshot. Document pagination, ordering, time zone/format, schema versioning, and behavior for records that change or are deleted while the job runs.
5. **Cancellation semantics.** Define the point at which cancellation is guaranteed, whether a completed artifact is deleted immediately on cancellation, and the response for a race between completion and cancellation. Cancellation must be cooperative and observable, not merely a UI state change.
6. **Idempotency and retry contract.** Define the client idempotency-key scope and lifetime, duplicate-request response, retryable versus terminal failures, maximum retry policy, and whether retry reuses an existing artifact or creates a new revision. Preserve a stable job identity and audit each attempt.
7. **Audit and privacy policy.** Confirm required event fields, retention, access controls, and who can query the audit trail. At minimum capture actor, tenant, job ID, action, outcome, timestamp, correlation/request ID, and failure/cancellation reason without including personal-data payloads.
8. **Capacity and service-level limits.** With one tenant processed at a time, set queue limits, per-tenant export size/time limits, backpressure behavior, and user-facing status for queued work. Confirm whether the worker’s global serial behavior is acceptable during staged adoption.

## Reversible implementation phases

### Phase 0 — design, threat model, and migration preparation

- Inventory the synchronous API contract, callers, authorization middleware, worker/job framework, storage lifecycle support, audit event schema, telemetry conventions, and operational rollback controls.
- Write the asynchronous API contract separately from the synchronous API. Use versioning or a new endpoint/feature capability; do not alter synchronous request/response behavior.
- Define a durable job model with tenant ID, requester identity, authorization context or policy reference, request parameters, idempotency key/hash, status, progress counters, attempt/retry data, artifact reference, expiry timestamp, timestamps, cancellation state, and schema/version fields.
- Define explicit state transitions such as `queued`, `running`, `cancellation_requested`, `cancelled`, `succeeded`, `failed`, and `expired`; enforce legal transitions transactionally.
- Conduct a privacy/security review covering tenant isolation, least-privilege worker credentials, encryption in transit/at rest, download authorization, secret-safe logging, artifact lifecycle, and audit integrity.
- Prepare additive, backward-compatible schema and configuration migrations. Make the new path inactive by default behind a server-side feature flag and ensure migrations can remain safely deployed when the flag is off.

**Phase exit:** Contract, state machine, decision log, data classification, and rollback runbook are approved; no existing API behavior changes.

### Phase 1 — durable job creation and safe status visibility

- Add an asynchronous export request endpoint or capability that validates tenant scope and authorization before creating a job. Persist the job and queue message atomically or use a transactional outbox so accepted jobs are not lost.
- Enforce idempotency using a tenant-scoped key plus a canonical request fingerprint. Repeating the same request returns the existing job; conflicting reuse returns a clear client error without starting a second export.
- Add tenant-scoped job status/progress retrieval. Return only metadata appropriate for the requesting role, with no export data or storage credential exposure.
- Emit audit events for request acceptance, duplicate acceptance, authorization denial, status access, and validation failure. Add correlation IDs spanning API, queue, worker, storage, and audit events.
- Keep execution disabled by feature flag. This phase is reversible by disabling the asynchronous endpoint/capability while retaining additive records for investigation.

**Phase exit:** Authorized users can create and inspect inert test jobs; duplicate submission and denied cross-tenant access are proven by tests.

### Phase 2 — single-tenant worker execution, artifact security, and cancellation

- Extend the existing one-tenant-at-a-time worker to claim jobs safely, update progress at bounded intervals, and use leases/heartbeats so abandoned work can be recovered without concurrent processing of the same job.
- Generate an export according to the approved snapshot and format contract. Store artifacts under tenant-isolated paths with least-privilege credentials, encryption, integrity metadata, and an explicit expiry derived from the selected retention policy.
- Implement cooperative cancellation checks before start, between read/write batches, and before marking success. On cancellation, stop work, remove any partial artifact, transition to `cancelled`, and emit an audit event; define and test the completion race outcome.
- Implement classified, bounded retry with exponential/backoff policy for transient failures. Make worker attempts idempotent through job ownership/lease and artifact-finalization guards. Terminal failures retain diagnosable non-sensitive reasons and audit events.
- Add cleanup work that deletes expired artifacts and transitions/records jobs as expired according to the approved policy. Make cleanup idempotent, observable, and safe to rerun after partial failure.

**Phase exit:** The worker completes one authorized job at a time; cancellation, retry, artifact expiry, and cleanup behavior pass integration tests under failure injection.

### Phase 3 — observability and operator controls

- Provide operator-visible progress: queued/running state, phase/counters where safe, elapsed time, attempt count, cancellation state, and terminal reason. Apply authorization to operational views and avoid exposing personal data.
- Add telemetry for request rate, queue depth/age, time queued, execution duration, rows/bytes exported, success/failure/cancellation rates, retry count, authorization denials, cleanup lag/failures, artifact deletion count, and worker lease/recovery events. Tag safely by environment and outcome; avoid tenant identifiers or personal data in high-cardinality metrics.
- Create dashboards and alerts for stuck jobs, growing queue age, error/cancellation anomalies, cleanup failures, storage growth beyond expected retention, and audit-pipeline delivery failures.
- Document operator actions: inspect progress, request/confirm cancellation, safely retry, investigate a stuck lease, validate cleanup, and disable the feature. Every privileged operator action must be audited.

**Phase exit:** On-call staff can diagnose a representative job end-to-end from correlation ID, telemetry, progress state, and audit trail without reading export content.

### Phase 4 — staged rollout and adoption evidence

- Start with the feature flag off in production after migrations and observability are deployed.
- Enable only for internal/test tenants, then a small allowlist of consenting tenants, then progressively wider cohorts. Increase only after the agreed observation window shows stable authorization, worker capacity, completion latency, retry/cancellation behavior, audit delivery, and retention cleanup.
- Preserve the synchronous API and route all existing callers to it throughout the experiment. Do not deprecate or change it based solely on feature availability.
- Establish explicit promotion gates: no unresolved security/privacy defects; no cross-tenant access; accepted error/queue-age thresholds; successful cleanup within retention SLA; audit completeness; and documented customer/support feedback.
- Maintain a deployment log recording cohort, flag state, release version, metrics snapshot, incidents, and approver for each expansion.

**Phase exit:** Adoption is proven against pre-agreed gates over a representative cohort; only then propose a separate decision to change default behavior or deprecate the synchronous API.

## Verification matrix

Run unit, integration, and end-to-end tests in an environment that exercises the actual queue, worker, storage lifecycle, authorization policy, audit sink, and feature flag.

| Area | Required evidence |
| --- | --- |
| Compatibility | Existing synchronous API contract tests remain unchanged and pass with async feature disabled and enabled. |
| Authorization | Request, status, cancel, retry, download, and operator actions reject unauthenticated, unauthorized, and cross-tenant actors; permitted roles succeed. |
| Idempotency and retry | Repeated identical submissions create one logical job; conflicting keys fail safely; worker crash/lease expiry and transient storage/queue failures do not duplicate output or corrupt state. |
| Progress | Queued, running, terminal, and recovered jobs expose truthful bounded progress and never leak export contents. |
| Cancellation | Cancellation before claim, during processing, and racing with completion meets the approved contract; partial artifacts are inaccessible/deleted. |
| Audit | Required events occur once per relevant transition with required identifiers, actor/outcome, and no personal-data payloads; audit-sink failure behavior is tested. |
| Retention and cleanup | Expiry reflects the selected 24-hour or 7-day policy; expiry/deletion jobs are idempotent; failed cleanup alerts and recovery work. |
| Telemetry | Metrics, logs, traces, dashboards, and alerts expose the defined operational signals without sensitive data. |
| Capacity | Load tests demonstrate queue/backpressure behavior with the one-tenant worker constraint and enforce agreed limits. |
| Rollback | A production-like exercise disables new requests immediately, drains or cancels work per policy, prevents downloads if required, and confirms synchronous API remains healthy. |

Archive test reports, migration identifiers, configuration/flag values, dashboard links, alert test results, and the rollout decision record with the release.

## Rollout controls and immediate rollback

**Immediate rollback mechanism:** a server-side kill switch must stop acceptance/dispatch of new asynchronous exports without deploying code. Keep the synchronous API independent of that switch.

**Rollback procedure:**

1. Set the async export feature flag off and verify new async requests are rejected or hidden according to the documented contract.
2. Pause worker consumption/dispatch for export jobs; do not delete job records or audit evidence.
3. Decide active-job treatment using the approved cancellation policy: allow a safely finalizing job to finish only if risk is acceptable, otherwise request cancellation and verify partial artifacts are removed.
4. Revoke artifact delivery access where the incident requires it; retain only the minimum metadata/audit evidence required by policy.
5. Confirm synchronous API health, queue stabilization, worker state, audit delivery, storage access, and absence of new artifact creation using telemetry.
6. Preserve logs, correlation IDs, job IDs, flag timestamps, and incident decisions for investigation. Re-enable only through a new staged cohort after root cause and verification.

Database and queue changes must be additive and backward compatible so application rollback is possible without data loss. Do not make destructive schema changes or remove the synchronous path during this rollout.

## Durable completion handoff

Before declaring the change complete, provide a handoff package to engineering, operations, security/privacy, support, and product containing:

- the final async API/state-machine contract and compatibility statement for the synchronous API;
- resolved open decisions, especially the retention selection and effective date;
- architecture/data-flow and threat-model review records;
- migration, configuration, feature-flag, queue, storage-lifecycle, and least-privilege permission details;
- authorization matrix; audit event catalog and audit-query instructions;
- idempotency, retry, cancellation, cleanup, and recovery runbooks;
- dashboards, alerts, telemetry definitions, alert ownership, and tested escalation paths;
- verification results for every matrix row, including load, failure-injection, privacy/security, and rollback exercises;
- rollout cohort history, release versions, promotion-gate evidence, incidents, and final approval;
- a rollback procedure tested against the deployed version, plus artifact revocation/cleanup instructions;
- a named operational owner and review date for retention, access policy, capacity limits, and eventual synchronous-API deprecation decision.

The handoff is complete only when the owners can execute a cancellation, retry, expiry cleanup check, and immediate rollback from the runbooks in a production-like environment, and when the product decision on retention is recorded.
