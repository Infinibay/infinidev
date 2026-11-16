# Observe a principal request

```sh
uv run infinidev status --workspace /path/to/project --session SESSION_ID
uv run infinidev status --workspace /path/to/project --session SESSION_ID --turn HOST_TURN_ID
```

The command prints one JSON observation. It does not resume a session, create
state, open a database, load settings or contact a model. Exit zero means the
observation was returned, including an `unknown` result.

The principal pipeline publishes metadata under the workspace's ignored
`.infinidev/session-status/v1/` directory. Active phases are `preparing`, `working`,
`inference`, `tool` and `awaiting_input`. Records contain process/turn identities,
timestamps, tool names and response token counts/latency when available. Token
counts describe the latest observed response; absent usage stays null. No prompt,
message body, reasoning, tool arguments/results or input replies are recorded.

Capture the `host_turn_id` from an active observation and use `--turn` to check
that specific request's `closed` state and terminal status. The optional `--run`
filter names a child runtime execution. A principal request can have multiple
child executions because of hooks; a child's closure does not close the parent.
The final SHA256 identifies the redacted message emitted through host callbacks,
not an acknowledgment of a frontend database commit or proof of task correctness.
Different requests may emit identical hashes.

A query with `--turn`, `--run`, or both selects only a uniquely matching owner.
Both filters must match the same record. Its freshness is checked independently
of unrelated owners: a stale record from another domain cannot discredit a fresh
named turn. `other_active_owners` counts other fresh pending owners, while
`other_owners_unconfirmed` reports pending owners whose liveness or scope cannot
be confirmed. These fields qualify the session, not the named observation.
A named `closed` result therefore does **not** certify session-wide idle or permit
an automatic restart. No records are deleted or declared dead by this selection.
Multiple matching owners, a stale named owner or a missing match stay `unknown`.

Between requests the default answer is `unknown`, with validated `last_known`
metadata when available. An old closed request does not establish that the REPL
is idle, that no input is queued or that a new request is finished. Pre-context
assembly/gather, standalone runtime calls and legacy engines are outside this
observer's scope. Supervisors must retain their input/queue checks for those
boundaries.

Freshness requires a matching Linux boot/PID-namespace/process-start identity
and a heartbeat no older than 15 seconds. The writer refreshes it every 5 seconds;
a heartbeat is liveness evidence, not work progress. Unsupported identity checks,
stale/future timestamps, invalid records or read-budget limits produce `unknown`.
An unfiltered session query also stays unknown for unconfirmed pending owners or
multiple active owners. Named closures also become historical after the lease;
consult `last_known` without promoting it to current state.

Snapshots are bounded to 8 KiB and atomically replaced. The reader scans at most
64 directory entries and 32 owner records. Separate engines/processes use
different owner files. A stalled prior writer receives no authority over the next
turn's file. Publication runs on a daemon thread and leaving the scope waits at
most 250 ms; failed I/O does not change the task result. These are local workspace
observations, not a cross-user authorization service or a durable event stream.
