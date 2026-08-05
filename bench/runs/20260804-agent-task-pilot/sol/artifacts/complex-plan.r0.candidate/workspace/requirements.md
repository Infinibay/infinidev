# Tenant export change

The service must export tenant records asynchronously without changing the current synchronous API
until adoption is proven. Exports may contain personal data. Retention may be 24 hours or 7 days;
product has not chosen. Operators require progress visibility, cancellation, idempotent retry, and an
audit trail. The existing worker can process one tenant at a time. A staged rollout and immediate
rollback are required. Completion evidence must cover compatibility, authorization, cancellation,
retry, audit events, retention cleanup, telemetry, and rollback.
