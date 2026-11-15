# Host session observer — concrete storage and lifecycle proposal

Status: implemented and validated offline; installation/live validation deferred. No live
process, settings, inference or private-history access. Scope is supervision, not
code quality or task certification. Source inspection at e3db0f9.

## Important lifecycle distinction

turns/runtime.py::outcome emits turn_closed for ONE run_turn invocation. In
pipeline.py::_run_principal_turn, TASK_END_INSTRUCTION may trigger a second run_turn
with a new run_id; first reply is then an intermediate public message. A safe
observer must not promote that child-run close or repeated final-message hash to
closure of the overall principal request.

Introduce a host_turn_id for _run_principal_turn, retain existing child run_id,
and close the parent only after post-hooks/cancellation and final publication.
Record delivery as host-emitted, not proof the frontend committed its final DB row.
A reader may expose the public final hash, not the body. Same hash in two requests
never establishes identity. Preserve request start/closure times and event sequence.

## Storage decision

A standalone standard-library module owns metadata snapshots under workspace
.infinidev/session-status/v1/<sha256(session_id)>/. A filename is the random
owner_instance_id, not the session's arbitrary text. One owner overwrites its own
snapshot across turns; separate processes never share a file. No database/schema
migration or transcript/history reads. Reader never creates directories or files.

Publish bounded JSON via temporary sibling created0600/O_EXCL, complete write and
atomic os.replace; private directories0700. Reject symlink targets/components and
oversized/invalid snapshots rather than read arbitrary files. These are local
owner-controlled workspace artifacts, not a new untrusted remote service. Reuse
atomic-write pattern from existing stores; those class-private asset/credential
methods are not suitable imports for a lightweight observer module.

Maximum8KiB per snapshot,32owner records examined per session. If enumeration or
validation budget is exceeded, return unknown, never pick an apparently idle record
while an unexamined owner could be active. Retention: writer may prune only owners
positively known dead, keeping a small bounded historical tail; unknown owners are
not silently removed. Saturation remains explicitly unknown. No unbounded event log.

## Owner identity and freshness

Owner UUID plus PID, boot identity and process-start identity; PID alone is unsafe.
On Linux, read boot_id and /proc/<pid>/stat starttime with bounded I/O, without
spawning a command. On unsupported platforms or insufficient access, liveness is
unknown. Missing/reused PID or changed boot must not make an old idle/closed state
current. Age uses same-boot monotonic time; future/invalid clocks are unknown.

A principal-scope heartbeat thread updates only this metadata every5s; lease15s.
Retain separate last_event and last_progress timestamps: heartbeat is not progress.
Stop/join boundedly when leaving scope. Status I/O failure must not change the agent
result; cease positive liveness publication and let the lease become unknown. Never
store raw exception text. No background model calls or recovery/restart behavior.

## Schema and reader semantics

Strict allowlist: schema_version, session_id, owner_instance_id, pid/process/boot
identity, host_turn_id, child_run_id, event_sequence, phase, observed/started/closed
and heartbeat timestamps, last event type, last tool name/completion timestamp,
host terminal status and public final hash, response usage/latency when observed.
Never arguments/results/prompts/messages, reasoning/signatures, tool rationale,
stdin/approval text, headers, environment or arbitrary nested payloads.

Use fixed host phases: preparing/working/inference/tool/awaiting_input/closed;
unknown is the reader's fallback. A tool-completed event means invocation returned,
not that a shell command succeeded. Token source is the response usage object,
not a claim of provider-measured tokens when an adapter may estimate them. Missing
usage stays missing, not synthetic zero. A heartbeat does not update progress.

Session availability and named-turn terminality are different. Default query returns
active verified observation or unknown plus last-known metadata. It does NOT infer
session idle merely from a prior closed turn, nor infer it from absent records.
Closed is a fact about a named host_turn_id, not task success or absence of new input.
An optional expected host-turn/child-run filter rejects historical closure mismatch.
If more than one live owner claims the same session, return ambiguous/unknown instead
of arbitrarily picking a terminal snapshot. Same final hash cannot bypass the filter.

The principal scope begins after pre-turn context assembly. Queues, earlier gather/
Ken preparation, and the interactive REPL between requests are outside this first
scope: expose unknown, not invented idle. A new request not yet in that scope cannot
be certified by an older closure. Root can retain its latest-user timestamp check
until frontend acceptance events are explicitly added in a separate scope.

## CLI and integration boundaries

infinidev.cli.entry already routes web before importing the heavy CLI. Add a parallel
status route there to a new lightweight argparse module. Proposed invocation:
infinidev status --workspace PATH --session ID [--turn HOST_TURN_ID] [--run CHILD_RUN_ID].
Output JSON only. Exit0 means observation returned, including unknown; not task success.
No bootstrap, settings, DB migrations, agent, provider or web-server startup.
Existing web endpoints remain untouched and retain their current semantics.

Pipeline owns parent scope and final close. Runtime emits allowlisted phase/tool/
usage signals into that scope; child turn_closed only closes child identity. Host
permission and stdin wait callbacks can expose awaiting_input through a no-op-when-
unbound context manager around the blocking handler. No text is passed to observer.
Direct standalone run_turn and legacy engines remain unsupported/unknown unless
explicitly integrated later; do not silently treat their absence as idle.

## Cost and boundaries

Expected implementation: one stdlib status module, one CLI reader module, small
entry routing plus principal pipeline/runtime and permission/stdin integration
(approximately6-8source files), focused storage/lifecycle/CLI tests. No large engine
rewrite, storage migration, new dependency or UI refactor. Parent scope must cover
exceptional exit and hook continuation; if it cannot be added without changing
public behavior, stop and report the specific refactor before widening scope.

Before coding, coordinating agents should confirm the conservative unknown-between-
turns semantics and host-turn vs child-run identity. Fully authoritative queued/idle
states require frontend acceptance/REPL events and are intentionally NOT promised.

## Offline acceptance

1. Reader has zero file/DB/bootstrap/provider side effects on missing/invalid data.
2. Atomic replacement/concurrent reads never expose partial state or mixed turns;
   bounds/collisions/symlinks fail to unknown without leaking raw payloads.
3. Script real principal integration through preparation, model boundary, tool,
   blocking user callback, hook second pass and final close. No inference.
4. Old closure vs newer intermediate, identical hash in two requests, mismatched
   expected IDs, multiple live instances, PID reuse, reboot, stale/future heartbeat
   all avoid false idle/current closure.
5. Private strings in model/tool/user wait data never enter status artifacts/output;
   fields are produced from named primitives, never arbitrary traces.
6. Cancellation, exceptions and publication failure preserve task behavior and cannot
   manufacture final-emitted status; I/O failure remains observational only.
7. Long blocked inference/tool keeps heartbeat but not fake progress; no usage becomes0.
8. CLI routing imports no legacy CLI, settings, DB, bootstrap or provider client.

After agreement: TDD, read-only Ken review when available, full suite, frozen candidate
and deferred Root live validation. Do not install/restart Bruma as part of this work.
