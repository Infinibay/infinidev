# Host session status implementation plan

Goal: read-only metadata-only observer for the principal request, without bootstrap/resume or inference. Spec: docs/superpowers/specs/2026-11-15-session-status-design.md, scope adopted by Root.

Architecture: stdlib snapshot store per workspace/session-hash/engine-owner UUID, atomic replace; parent host_turn_id independent of runtime child run IDs. Bounded reader validates owner/process/boot/lease, returns unknown on ambiguity, stale records, limits or unsupported phases. No session idle inference from historical closure. Lightweight entry route before heavy CLI import.

Source: new engine/session_status.py + cli/status.py, small edits cli/entry.py, pipeline.py, turns/runtime.py, tools/permission.py, tools/stdin_prompt.py. No DB schema, new dependencies, UI rewrite or live changes.

- [x] Core: bounded records, exact ownership, stale/PID reuse/multiple owners, old closure/hash collision and privacy; implement reader/publisher.
- [x] Lifecycle: RED principal integration for hook second pass and real final publication, errors/cancellation/user waits; attach parent scope and runtime allowlisted signals.
- [x] CLI: RED no-bootstrap subprocess/import tests; add JSON-only status route.
- [x] Read-only Ken and independent review, focused/full Python3.13 validation and source hash check.

Publication: the frozen candidate receipt records the exact commit, push verification,
source/manifest hashes, test logs and installation instructions. This plan does not
authorize installation into an active process.

Constraints:8KiB record,32owners/read; heartbeat5s/lease15s. Reader never creates state; writer failures never alter task results. Background publisher isolates slow I/O; bounded join on exit. Linux process identity verified from /proc, other platforms/insufficient evidence=unknown. No raw text, arguments, reasoning, environment or exception message. Current observation vs last-known history remain distinct.

Validation ledger: initial core RED was a missing-module collection failure, not a
functional regression. Integration RED:6failed/2passed on missing lifecycle binding;
CLI RED:2failed on absent route/module. Bounds tests added after implementation
were green, not claimed as pre-fix reproductions. Focal current:136passed, including
real principal/runtime continuation with scripted provider, publication/post-hook
failure, cancellation, waits, stale ownership, blocked writer and subprocess reader.
Additional RED isolated a Unicode encoding failure in the new final-message hash;
that observation now remains unknown without changing the emitted response/result.
Current focal:138passed, including an automatically heartbeating blocked provider.
Initial full:5981passed/14skipped/0failed; final full after the encoding fix:
5983passed/14skipped/0failed/11warnings,312.20s. Python3.13, locked retained worktree;
1138source/test files match the implementation tree by SHA256.
Independent source review (session_status_review):no Critical/Important findings;
it ran no tests or live processes. Ken peer reviewed the7source/3test files,
including the encoding fix, with no blockers within the declared scope. It ran
no suite or Strata. Both reviews retain the limits: no frontend queue/idle
certification, hostile concurrent directory mutation protection or absolute
filesystem-latency guarantee. No live model/install used.
