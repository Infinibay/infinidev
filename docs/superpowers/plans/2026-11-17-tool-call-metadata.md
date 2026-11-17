# Opt-in model-declared call metadata

Root chose transport B and authorized implementation. SDK `tool_call_metadata=False` by default; principal-only, per invocation. Active catalog derives native/manual transport from each existing schema: `{metadata:{reason,expected_result},input:<original args>}`. Two nonblank strict strings, max500 characters each, public operational purpose/evidence only. No synthesis, inference of intent or permission effect. Existing input/rationale remain unchanged.

Routes: turns/action_metadata.py owns schema/envelope; turns/runtime.py validates every batch before dispatch, unwraps only input, preserves original transport history and delivers declared metadata publicly; loop/engine.py and orchestration/pipeline.py forward explicit flag. Legacy routes reject opt-in early; default schemas/history remain unchanged.

RED: native/manual schema equality; literal input and legacy rationale unchanged; bad metadata in a later call executes none; missing/blank/nonstring/extra/oversize rejected; metadata preserved publicly/history; optional flag reaches real principal and hook children; default and compatibility unchanged. Then focused suite, independent review and full suite on exact hashes. No live reload/model or installation. Result-provenance extension paused.

CLI pilot control: explicit --tool-call-metadata forwarded through main/classic/one-shot without changing default kwargs. Reject unsupported TUI/phase/autonomous/flow routes, preserve durable resume and dependencies. No live enable/reload.
