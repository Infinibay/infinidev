# Classic permission wait repair

Root authorized offline implementation and a tested candidate; live recovery and
installation remain Root-owned. No model inference or process input/restart here.

Defect: main drains PermissionQueue after run_with_live_status, but its worker
waits on a permission before it can finish. Cancel flags do not set req.done.
Offline characterization demonstrates cancel remains blocked and denial releases.

Contract: service the existing queue inside the active app. Show full description
and details, require Enter to confirm y/yes; all other input/default/Ctrl-D denies.
First Ctrl-C cancels and denies; existing second-press behavior stays. Cancellation
is observed by the queue, settled requests cannot be approved later, and cancelled
requests are skipped. UI construction precedes worker startup; UI errors release
queued and already-claimed permissions. Cancellation is not rollback after an
operation has already been authorized/started. No permission-policy or classifier
changes, autoapproval, external IPC, settings or provider changes.

TDD: original cancel characterization; ten new RED before implementation; a
constructor-failure RED from review; actual prompt_toolkit pipe-input gates and
queue/claimed/EOF/error/cancellation tests. Then focused/full, exact source review,
commit/push and immutable source/wheel candidate. Do not claim live recovery from
new code that is not loaded by the existing blocked process.

Review added a before-enqueue/cancel/reset/late-approval race gate: RED before
tracking all registered requests, GREEN after. Denial snapshots under the registry
lock and resolves outside it; enqueue holds no registry lock. This protects
registered requests, not every possible old worker or requests begun after reset.
All three classic callers (ordinary, init, explore/brainstorm) pass the same queue.
