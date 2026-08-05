# Tenant export plan

## Open user decisions

Confirm retention at 24 hours or 7 days before enabling cleanup in production. Recommend 24 hours as
the lower-exposure default. Confirm the staged rollout cohort and cancellation visibility.

## Reversible phases

1. Add an internal asynchronous adapter behind the unchanged synchronous API.
2. Implement tenant authorization, idempotent retry, cancellation, and audit events.
3. Add progress telemetry and retention cleanup behind disabled flags.
4. Roll out to an internal tenant, then 5%, 25%, and 100% with stop criteria.

## Verification

Test compatibility, cross-tenant authorization denial, cancellation races, retry idempotency, audit
events, retention cleanup, telemetry, and immediate rollback. Preserve results in the final handoff.

## Rollback

Disable the asynchronous path and cleanup flag independently, drain or cancel queued work, restore
the synchronous path, and verify no retained export becomes inaccessible or crosses tenant scope.
