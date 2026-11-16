# Principal Human Guidance Continuity Plan

> Execution: inline using executing-plans; independent source review by Ken.

**Goal:** Preserve original human steering between hook children of one parent.
**Spec:** Root explicitly authorized this bounded transport fix on 2026-11-16.
**Architecture:** Parent-local list of original InjectedUserMessage objects, passed
through private `_parent_guidance` keyword to LoopEngine/runtime. Reproject the
existing snapshot once when a new child starts; append only newly drained ingress.
**Tech Stack:** Python, existing injector/renderer, scripted pytest providers.

## Constraints

No live inference/install. No reenqueue, system promotion, model/tool/notes-derived
ingress, durable storage, cross-parent/session retention or general contract API.
Keep identities, FIFO order, literal Unicode and ephemeral attachments. Pending
new ingress remains later than replayed guidance. Cancellation wins as before.
Default direct run_turn without a parent ledger is unchanged. No new context cap.

## Review focus

Late input in hook/finish races; duplicate text with distinct identities; image
and non-vision rendering; stale finish/cancel preemption; leaks into later parents.

## Task 1

Files: engine/turns/runtime.py; engine/loop/engine.py;
engine/orchestration/pipeline.py; tests/test_principal_guidance_continuity.py.

- [x] Port original two native/manual RED gates into durable tests.
- [x] Add order/identity/late/cancel/Unicode/attachments/isolation/default gates.
- [x] Confirm RED without live provider.
- [x] Create parent-local list; forward private optional parameter.
- [x] Record original drained objects; seed the next child after original input.
- [x] Focused tests and Ken review; resolve substantive findings.
- [x] Full suite and source hashes; commit/push and publish candidate, no install.

Decision: do not extend TurnState/TurnResult, change injector queue semantics or
recover steering by interpreting stored text. Original user_input already crosses
children; only subsequently drained original human ingress needs transport.

Validation:10RED/6PASS before source;175focusedPASS;6098fullPASS/14SKIP/0FAIL,
11existing warnings142.63s. Source/test hashes equal across main/worktree1145files.
Independent Ken source review:no blockers in4exact hashes; no tests/model by reviewer.
Source checkpoint/candidate receipt stays outside repo; no live installation.
