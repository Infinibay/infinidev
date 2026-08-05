# Tenant Export Delivery Plan

## 1. Purpose and constraints

Deliver an asynchronous tenant-record export path while leaving the current synchronous API behavior and contract unchanged until measured adoption proves that a later change is safe. The asynchronous path must support progress visibility, cancellation, idempotent retry, and an audit trail. Because exports may contain personal data, authorization and data lifecycle controls are release gates rather than follow-up hardening.

This document is an implementation plan only. It does not select unresolved product or security policy, define a final endpoint name, or authorize removal of the synchronous path.

### Non-negotiable outcomes

- The existing synchronous API remains available and compatible throughout this plan. No client is silently redirected to asynchronous behavior.
- Asynchronous export is additive, disabled by default, and independently controllable with a server-side feature flag or equivalent kill switch.
- The existing worker continues to process no more than one export for a tenant at a time. Scheduling must not introduce same-tenant concurrency.
- Every operation is tenant-scoped and subject to explicit authorization checks; possession of an export identifier alone never grants access.
- Export artifacts are treated as personal data from creation through cleanup, including storage, download, logs, backups, and failure paths.
- Each phase is reversible. Any database evolution is expand-first and remains readable by the prior application version until rollout is complete.
- Rollback disables new asynchronous work immediately without deleting evidence, abandoning artifacts, or changing the synchronous API.

## 2. Consequential open decisions

Record each answer in a durable decision record before the phase whose **Decision gate** references it. Include the approver, date, rationale, affected controls, and review date. Do not infer these choices during implementation.

| ID | Open decision / user decision required | Options or questions to confirm | Why consequential | Required before |
|---|---|---|---|---|
| D1 | **Retention period** | Confirm **24 hours or 7 days**; define whether the clock starts at artifact creation or job completion, and how legal hold or failed deletion is handled. | Changes personal-data exposure, storage cost, cleanup scheduling, and user expectations. | Phase 1 contract and Phase 3 storage work |
| D2 | **Authorization policy** | Confirm which tenant roles may create, view status, cancel, retry, and download; whether service/support operators may act; and whether step-up authentication is required for download. | Determines who can extract personal data and must be enforced consistently at every operation. | Phase 1 |
| D3 | **Export storage and delivery boundary** | Confirm storage system, region/residency, encryption and key ownership, download mechanism (streamed or short-lived signed access), link lifetime, and whether filenames/content need additional protection. | Defines the principal data leakage and residency risks and affects cleanup behavior. | Phase 3 |
| D4 | **Cancellation semantics** | Confirm whether cancellation is best-effort or guaranteed before delivery; expected behavior when cancellation races completion; whether partial artifacts are immediately destroyed; and terminal state wording. | Affects state transitions, user promises, worker interruption, audit evidence, and tests. | Phase 1 |
| D5 | **Idempotency contract** | Confirm caller-provided key versus derived request identity, key scope (tenant + actor + normalized parameters), deduplication window, response for an in-flight/completed/failed/cancelled match, and when a retry may create a new job. | Prevents duplicate personal-data artifacts without incorrectly coalescing distinct requests. | Phase 1 |
| D6 | **Export contract** | Confirm data fields and format, consistency point/snapshot expectations, size limits, filters, localization, and schema/version signaling. | Determines compatibility, reproducibility, resource use, and what personal data leaves the service. | Phase 1 |
| D7 | **Progress contract** | Confirm whether progress is exact records/bytes or coarse stages, update frequency, and what clients see when totals are unknown. | Avoids misleading clients and constrains worker/database write volume. | Phase 1 |
| D8 | **Capacity and fairness policy** | Confirm queue limits, per-tenant outstanding-job limit, global concurrency, priority/fairness, timeout, and overload response while preserving one active job per tenant. | Prevents one tenant from starving others and protects the existing worker and synchronous API. | Phase 2 |
| D9 | **Adoption criterion and synchronous API future** | Confirm the observation window and thresholds for eligible-tenant adoption, success rate, latency, support burden, and client compatibility. Any synchronous deprecation/removal requires a separate approved plan. | Prevents premature behavior change and gives “adoption is proven” an auditable meaning. | Phase 5 rollout; any later API change |
| D10 | **Audit and telemetry access** | Confirm required audit event taxonomy, audit retention, authorized viewers, correlation fields, telemetry system, and restrictions on tenant identifiers or export contents in logs/metrics/traces. | Balances traceability with minimization of personal data and operational metadata. | Phase 1 design and Phase 4 instrumentation |

If a decision is not approved by its gate, pause that phase. A temporary default is acceptable only when explicitly approved and recorded with an expiry; it must not silently become policy.

## 3. Target behavior and contracts to design

Phase 1 must turn the approved decisions into a versioned contract before implementation. Exact route and field names should follow repository conventions discovered during implementation; behavior must include the following.

### Job lifecycle

Use a persisted state machine with explicit allowed transitions, for example:

`queued -> running -> completed`

with terminal or exceptional paths to `cancelled` and `failed`. Define cancellation-requested behavior if cancellation cannot be atomic. A terminal state must never return to a non-terminal state. Persist timestamps, tenant scope, requesting actor, normalized request parameters or their safe digest, progress, error category safe for clients, artifact metadata, retention deadline, and correlation identifiers.

State changes must be conditional/transactional so duplicate deliveries, retries, and cancellation/completion races cannot publish two artifacts or overwrite a terminal outcome. Recovery after worker restart must be specified and tested.

### Asynchronous operations

Define authenticated operations to:

1. request an export with an idempotency identity;
2. read status and progress;
3. request cancellation;
4. retry according to the approved idempotency policy; and
5. retrieve a completed artifact before expiry.

Responses must distinguish accepted, existing idempotent result, unauthorized/forbidden, not found without cross-tenant disclosure, conflict/invalid transition, expired, rate- or capacity-limited, and internal failure. Polling guidance and any retry hints must be documented. No asynchronous response may alter status codes, payloads, timing promises, or side effects of the current synchronous API.

### Security and data lifecycle

- Re-evaluate current tenant membership and role authorization on every create, status, cancel, retry, and download operation; do not rely only on authorization captured at creation.
- Bind each job and artifact to a tenant and enforce tenant scope in both application queries and storage access. Use opaque identifiers and prevent identifier enumeration from revealing existence.
- Minimize export contents to the approved contract. Never place export payloads, secrets, download tokens, or sensitive request parameters in application logs, audit details, metrics, traces, queue messages, or error text.
- Encrypt artifacts in transit and at rest according to D3. Keep delivery credentials short-lived and avoid reusable public URLs.
- Set the approved retention deadline explicitly and make cleanup idempotent. Cleanup covers completed, cancelled, failed, orphaned, and partially written artifacts, plus stale delivery credentials and metadata according to policy.
- If artifact deletion fails, deny further download after expiry, retry cleanup with bounded backoff, emit an alert and audit event, and retain enough non-sensitive metadata to prove resolution.

### Audit trail

Emit durable, queryable audit events for request accepted/deduplicated/rejected, worker start, meaningful progress milestones if approved, cancellation requested/effected/raced, retry, completion, download authorization and delivery, failure, expiry, cleanup success/failure, administrative override, feature-flag change, and rollback action. Events include timestamp, tenant, actor or service identity, action, outcome, job ID, correlation ID, and safe reason code; they exclude export contents and credentials. Audit writes for security-significant transitions must have defined failure behavior rather than being silently dropped.

## 4. Reversible implementation phases

Do not begin a phase until the prior phase exit criteria and the listed decision gate are met. Keep incomplete behavior unreachable in production behind the asynchronous export flag.

### Phase 0 — Baseline, discovery, and decision records

**Decision gate:** none; this phase gathers evidence and owners for D1–D10.

**Work**

- Inventory the current synchronous API contract, authorization checks, tenant scoping, export generation path, worker execution model, persistence, storage, cleanup facilities, audit framework, telemetry, deployment topology, and operational ownership.
- Capture baseline synchronous request volume, latency, errors, resource use, and current export sizes without logging export data.
- Identify schema migration and feature-flag conventions, worker restart/recovery behavior, and how the one-tenant-at-a-time invariant is currently enforced.
- Create the D1–D10 decision records, a threat model/data-flow diagram, and a compatibility fixture or characterization test for the synchronous API.

**Verification / exit criteria**

- The current synchronous behavior is represented by automated compatibility tests and production-safe baseline telemetry.
- Data flows and trust boundaries identify every location where personal data or delivery credentials can exist.
- Each decision has an accountable approver and due date; no gated implementation begins with an implicit answer.

**Rollback:** documentation, tests, and non-sensitive baseline dashboards are additive. Remove only unused draft observability; no runtime behavior changes.

### Phase 1 — Contract, state model, and inactive persistence

**Decision gate:** D1, D2, D4, D5, D6, D7, and D10 approved.

**Work**

- Specify and review the versioned asynchronous API, status/progress representation, terminal states, errors, polling guidance, cancellation race semantics, retry behavior, and idempotency scope/window.
- Add backward-compatible persistence using expand-only migrations: nullable/new tables or fields, indexes, uniqueness constraints for idempotency, and conditional state-transition support. Do not repurpose fields used by the synchronous path.
- Implement the domain state machine and tenant-scoped repository boundaries behind a disabled flag, including retention deadline and artifact state metadata but no production artifact creation.
- Define audit event schemas and safe reason codes. Document which transition and audit writes are atomic, and the recovery path where atomicity is unavailable.

**Verification / exit criteria**

- Unit and database tests cover every allowed and forbidden transition, terminal-state immutability, concurrent idempotent requests, cross-tenant key isolation, cancellation/completion races, and migration forward/backward application.
- Contract tests prove the existing synchronous API is byte-/field-/status-compatible for representative success and error cases with the flag both off and on.
- A prior application version can run safely against the expanded schema; down migration is not required for immediate application rollback.

**Rollback:** keep additive schema dormant, disable the flag, and deploy the prior application version. Remove schema only in a later cleanup release after confirming no mixed-version process uses it.

### Phase 2 — Queueing, worker execution, progress, cancellation, and retry

**Decision gate:** D8 approved; Phase 1 contracts frozen for the rollout scope.

**Work**

- Add asynchronous request admission and durable enqueueing behind the disabled flag, with authorization, capacity controls, and transactional protection against a persisted job being lost before enqueue.
- Adapt the existing worker to claim work safely while enforcing at most one active export per tenant. Define leases/heartbeats, stale-job recovery, bounded retries, poison-job handling, and fairness under D8.
- Generate progress from committed work or coarse stages according to D7; make updates monotonic, bounded, and resilient to duplicate delivery.
- Check cancellation before start and at safe checkpoints during generation. Prevent publication after effective cancellation and destroy partial artifacts according to D4.
- Make retries idempotent across request replay, queue redelivery, process crash, and timeout. Ensure only one canonical downloadable artifact can win.

**Verification / exit criteria**

- Tests cover duplicate requests and queue delivery, process termination/restart, stale leases, timeout, concurrent tenants, same-tenant serialization, queue saturation, fairness, monotonic progress, and retry exhaustion.
- Deterministic race tests cover cancel-before-start, cancel-during-generation, cancel-versus-completion, retry-versus-cleanup, and duplicate completion.
- Load tests demonstrate selected queue and resource limits do not materially regress synchronous API service levels; results are compared with Phase 0 baselines.

**Rollback:** disable admission first, then stop new worker claims. Allow approved in-flight behavior (finish or cancel per D4), preserve job state for reconciliation, and continue the synchronous API. The prior version remains schema-compatible.

### Phase 3 — Protected artifacts, delivery, retention, and cleanup

**Decision gate:** D1 and D3 approved; privacy/security review accepts the data flow.

**Work**

- Write artifacts to the approved encrypted, tenant-isolated storage location using temporary/non-downloadable state, then atomically publish only a successfully completed artifact.
- Add authorized retrieval with current-role revalidation, short-lived delivery credentials or protected streaming, expiry enforcement, safe content disposition, and no cache/public exposure inconsistent with D3.
- Implement retention cleanup from the persisted deadline. Include partial/orphaned artifacts and terminal job states; make deletion retries idempotent and observable.
- Add reconciliation that finds metadata-without-artifact, artifact-without-metadata, expired-but-readable, and stuck partial uploads without scanning or exposing artifact contents.

**Verification / exit criteria**

- Authorization tests cover allowed roles, revoked membership, changed role, cross-tenant IDs, guessed IDs, expired links, replayed credentials, and operator access.
- Storage tests verify encryption/configuration, tenant isolation, atomic publication, partial-write cleanup, and that cancelled/failed jobs never expose artifacts.
- Time-controlled tests prove cleanup at the selected retention boundary for completed, cancelled, failed, partial, and orphaned artifacts; deletion failure denies download, retries, alerts, audits, and eventually reconciles.
- Privacy review verifies data minimization and confirms logs, telemetry, audit events, queues, and errors contain no export payload or delivery credential.

**Rollback:** disable create and download surfaces, revoke/expire outstanding delivery credentials where supported, stop publication, and keep cleanup/reconciliation running. Do not restore expired artifacts. Preserve minimal audit/job metadata according to approved policy.

### Phase 4 — End-to-end integration, telemetry, and operational readiness

**Decision gate:** D9 metrics are defined; on-call, security, privacy, and support accept operating procedures.

**Work**

- Wire the complete asynchronous path under separate controls for request admission, worker claiming, and artifact download so rollback can isolate a faulty stage.
- Add dashboards and alerts for queue depth/age, admission rejection, job duration by stage, progress staleness, completion/failure/cancellation/retry rates, worker leases, same-tenant concurrency violations, artifact publication/download failures, cleanup backlog/age, deletion failures, audit delivery failures, and synchronous API health.
- Define service-level indicators and alert thresholds from baseline/load evidence. Keep telemetry dimensions bounded and free of personal data; use controlled correlation tooling for job-level diagnosis.
- Write runbooks for stuck jobs, cancellation races, duplicate requests, queue overload, worker restart, storage outage, unauthorized-access alert, audit outage, cleanup failure, flag rollback, artifact reconciliation, and customer support escalation.
- Exercise deployment and immediate rollback in a production-like environment, including mixed application/worker versions and queued jobs.

**Verification / exit criteria**

- End-to-end tests cover create, progress, completion, authorized download, cancellation, idempotent retry, audit sequence, retention expiry/cleanup, and each declared failure mode.
- Compatibility tests and comparative load tests remain green for the synchronous API.
- A rollback drill demonstrates the async entry point can be disabled immediately, workers drained/stopped safely, downloads controlled, cleanup preserved, and the prior release restored without schema rollback or data loss.
- Alerts are triggered synthetically and route to named responders; runbook steps and required permissions are validated rather than only reviewed.

**Rollback:** execute the runbook in Section 6. Observability, audit ingestion, and cleanup remain active even when user-facing async behavior is disabled.

### Phase 5 — Staged rollout and adoption observation

**Decision gate:** D9 approved and all Phase 4 exit criteria signed off.

**Work and promotion gates**

1. **Internal/test tenants:** enable only for controlled accounts; use synthetic or approved data. Validate audit sequence, authorization, cancellation, retries, cleanup, dashboards, and support workflow.
2. **Small allowlist:** enable selected low-risk tenants that have opted into the async contract. Hold through at least one complete retention and cleanup cycle. Compare synchronous baselines and inspect every failed/cancelled job.
3. **Broader cohorts:** increase by explicit tenant cohorts, not random global exposure. At each hold point assess D9 adoption, job success and latency, queue age, authorization denials, cancellation effectiveness, duplicate suppression, cleanup backlog, support contacts, and synchronous API regression.
4. **General availability of additive async API:** enable only after all approved thresholds remain satisfied for the observation window and security/privacy/on-call owners sign off. Keep the synchronous API unchanged and available.

Each cohort promotion is a separate recorded approval. Stop promotion on any threshold breach, unexplained audit gap, cross-tenant signal, retention/cleanup breach, or synchronous regression. Roll back the affected cohort or all asynchronous access depending on blast radius.

**Verification / exit criteria**

- Cohort reports include denominator-based adoption, reliability, security, lifecycle, and compatibility metrics, not only successful-job counts.
- At least one full selected-retention cleanup cycle has been observed in production before broad promotion.
- Immediate rollback has been re-checked in the deployed configuration.
- D9 adoption criteria are met or the async path remains an optional limited feature. This plan never treats low adoption as permission to remove the synchronous API.

**Rollback:** remove cohorts from admission, then follow Section 6. Rollout configuration must be versioned/audited so the exact last-known-good cohort can be restored.

### Phase 6 — Stabilization and deferred cleanup

**Decision gate:** production observation window complete; no unresolved security, cleanup, audit, or compatibility issue.

**Work**

- Resolve rollout findings and verify retained artifacts and job metadata conform to approved lifecycle policy.
- Remove only obsolete transitional code or schema that evidence proves unused, in a separate release after all old application/worker versions are gone. Keep feature controls and rollback capability for the agreed stabilization period.
- Evaluate the synchronous API only against D9. Any deprecation, behavior change, or removal requires separate product approval, client communication, compatibility plan, rollout, and rollback; it is not part of this delivery.
- Produce the durable completion handoff in Section 8.

**Verification / exit criteria**

- No orphaned artifact, overdue cleanup, unexplained audit gap, stuck job, or unresolved high-severity alert remains.
- Final evidence is linked and reproducible from immutable build/deployment identifiers.
- Operational owners accept dashboards, alerts, runbooks, access, and lifecycle responsibilities.

**Rollback:** postpone cleanup and retain backward-compatible structures. Disable async behavior through the established controls if a late defect appears; keep synchronous behavior intact.

## 5. Verification and completion evidence matrix

Automate tests where practical and retain test output, environment/build identity, and approver links. Production evidence must use safe metadata and must not copy personal data into tickets or reports.

| Required evidence | Minimum proof before rollout | Production/rollout proof |
|---|---|---|
| Compatibility | Contract/characterization tests show synchronous success and error responses, authorization, and side effects are unchanged with async controls off/on; comparative load test meets agreed regression budget. | Dashboard comparison by cohort and rollback drill confirm synchronous health remains within threshold. |
| Authorization and isolation | Role matrix tests for every operation; revoked/changed access; cross-tenant, enumeration, and storage-isolation tests; security review. | Denial and anomaly telemetry reviewed without exposing identifiers; controlled canary confirms current-role revalidation. |
| Cancellation | State/race tests at queued, running, publication, and completed boundaries; partial artifact deletion verified. | Canary cancellation traces and audit sequence match D4; no downloadable cancelled artifact. |
| Idempotent request and retry | Concurrent request, redelivery, crash/restart, timeout, and cleanup-race tests prove one canonical job/artifact under D5. | Duplicate-suppression/retry metrics and sampled safe audit correlations show expected outcomes. |
| Audit trail | Event-schema tests and end-to-end expected event sequences, including rejected and failure paths; audit sink outage behavior tested. | Audit delivery dashboard has no unexplained gaps; authorized reviewer can reconstruct canary lifecycle. |
| Retention and cleanup | Time-controlled boundary tests for all artifact/job outcomes and failed deletion/reconciliation. | At least one full retention cycle observed; cleanup backlog, overdue artifacts, and deletion alerts meet thresholds. |
| Progress visibility | Monotonicity, unknown-total, throttling, stale worker, and terminal-state tests. | Progress freshness and polling/load metrics remain within D7 limits. |
| Telemetry and operations | Dashboard queries, bounded dimensions, redaction checks, synthetic alerts, and runbook exercises. | Named responders receive canary alerts; cohort scorecards are archived at each promotion. |
| Rollout | Feature controls and tenant cohorts tested in production-like mixed-version deployments. | Each promotion approval records thresholds, evidence, cohort, release ID, and observation interval. |
| Rollback | Drill covers admission off, worker drain/stop, download control, prior-version deploy, cleanup continuation, and reconciliation. | Time-to-disable and recovery evidence meet the approved immediate-rollback objective in the deployed environment. |

Also run repository-standard unit, integration, API contract, migration, security, concurrency/race, load, and end-to-end test suites. A release cannot rely solely on happy-path tests or manual inspection.

## 6. Immediate rollback procedure

Prepare and rehearse exact platform commands in the operator runbook; do not embed environment-specific credentials in this plan.

1. **Declare and record the rollback:** identify release/cohort and reason, open an incident if warranted, preserve correlation and deployment IDs, and begin an audit trail.
2. **Stop admission:** disable asynchronous export creation globally or for affected cohorts. Confirm the synchronous API still serves its unchanged contract.
3. **Contain execution:** stop new worker claims. Depending on D4 and incident type, drain safe in-flight jobs or request cancellation. For suspected authorization/data isolation issues, prefer containment and disable retrieval immediately.
4. **Control artifacts:** disable or restrict downloads, revoke outstanding delivery credentials where possible, and prevent new publication. Do not delete evidence blindly; quarantine or clean artifacts according to security response and retention policy.
5. **Restore software:** deploy the last-known-good application and worker versions. Do not reverse expand-only schema while mixed/older processes or job metadata may depend on it.
6. **Keep safeguards alive:** continue audit ingestion, retention cleanup, deletion retries, telemetry, and reconciliation even when async execution is off.
7. **Reconcile:** account for every admitted job and artifact as completed, safely cancelled, failed, quarantined, or queued for approved recovery. Verify no expired or cancelled artifact is downloadable.
8. **Validate recovery:** run synchronous compatibility smoke tests, tenant-isolation checks, queue/artifact/cleanup queries, and alert checks. Record actual disable and recovery times.
9. **Decide re-enable separately:** require root cause, remediation evidence, regression tests, and a new cohort approval. Never auto-re-enable solely because metrics return to normal.

A rollback is complete only when user-facing exposure is contained, synchronous compatibility is verified, in-flight data is reconciled, and cleanup/audit controls are confirmed healthy.

## 7. Release gates and ownership

Before implementation begins, assign named owners (people or teams) for product decisions, API/client compatibility, worker/queue, storage/security, privacy, audit, telemetry/on-call, cleanup, support, rollout approval, and incident rollback. Capture ownership in the repository’s normal durable location.

A phase gate must record:

- decision records and approvals applicable to the phase;
- release/build and migration identifiers;
- test and review evidence links;
- known limitations and accepted risks with expiry dates;
- rollout cohort, observation window, and measured thresholds;
- rollback version, flag locations, responsible operator, and drill result; and
- explicit go/no-go signatures from product plus security/privacy/operations where applicable.

No individual should need undocumented knowledge to operate or roll back the feature.

## 8. Durable completion handoff

The delivery is not complete until a version-controlled handoff index points to durable, access-controlled artifacts for:

- approved D1–D10 decision records and the versioned async API/export schema;
- architecture and personal-data flow/threat model, including storage region, encryption/key boundary, and lifecycle;
- job state machine, idempotency, cancellation, retry/recovery, and one-active-job-per-tenant invariants;
- migration sequence, compatibility window, feature controls, cohort configuration, and last-known-good versions;
- complete test reports and the evidence matrix above, tied to immutable build IDs;
- privacy and security review approvals plus any risk acceptances and expiry dates;
- telemetry dashboards, alert definitions, service indicators/thresholds, audit queries, and data-redaction rules;
- operator/support/security runbooks, escalation contacts, required access, rollback drill record, and measured recovery time;
- retention and cleanup configuration, reconciliation results, deletion-failure procedure, and proof of a production cleanup cycle;
- rollout cohort reports, D9 adoption assessment, customer/support communications, and unresolved follow-up owners/dates; and
- final status of every migration and temporary compatibility mechanism, including explicit criteria and owner for later removal.

The receiving engineering and operations owners must walk through a fresh-job lifecycle and a rollback using only the handoff materials. They sign acceptance after confirming access to dashboards, audit evidence, flags, storage controls, cleanup tooling, and escalation paths. Archive the signed handoff with the release record; keep links stable beyond the implementation team’s involvement.

## 9. Definition of done

The tenant export change is complete only when:

1. the additive asynchronous path satisfies the approved contract and all required authorization, cancellation, idempotency, audit, progress, retention cleanup, telemetry, and personal-data controls;
2. the synchronous API remains compatible and available, with production evidence through the approved observation window;
3. staged rollout gates and at least one full retention cycle have passed without unresolved release-blocking issues;
4. immediate rollback has been demonstrated in the deployed configuration and leaves audit/cleanup protections operating;
5. all artifacts and jobs are accounted for, and no overdue or unauthorized artifact is retrievable; and
6. the durable completion handoff is accepted by named engineering and operations owners.

Any future synchronous API deprecation is explicitly outside this definition of done and requires its own approved change plan.
