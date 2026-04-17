"""TUI-specific :class:`OrchestrationHooks` implementation.

Lives in the ``ui/`` package on purpose: it has to know about the
prompt_toolkit ``InfinidevApp`` to update its panels and to coordinate
the question-answer protocol via :class:`threading.Event`. The
orchestration pipeline never imports this — it only sees an
``OrchestrationHooks`` Protocol.

Why a separate file instead of methods on :class:`InfinidevApp`:

  Keeping the hooks in their own module enforces the single direction
  of dependency we want — ``ui/`` can import ``engine/orchestration``,
  but ``engine/orchestration`` never imports anything from ``ui/``.
  If we glued the hook methods straight onto the app, every refactor
  of the pipeline would tempt us to reach back into UI internals.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp


class TUIHooks:
    """Implements :class:`engine.orchestration.OrchestrationHooks` for the TUI.

    Every method is callable from a worker thread — the
    ``concurrent.futures`` thread pool that runs the pipeline never
    touches the prompt_toolkit event loop directly. ``app.invalidate()``
    is the standard way to schedule a redraw from any thread.
    """

    def __init__(self, app: InfinidevApp) -> None:
        self._app = app

    # ── Phase / status ───────────────────────────────────────────────────

    def on_phase(self, phase: str) -> None:
        # Drives the "Actions" indicator and the chat-history "context flow"
        # tag. Idle clears both so the panel returns to its resting state.
        labels = {
            "chat":     "Thinking...",
            "analysis": "Planning...",
            "gather":   "Gathering context...",
            "execute":  "Working...",
            "review":   "Code review...",
            "idle":     "Idle",
        }
        self._app._actions_text = labels.get(phase, phase or "Idle")
        if phase == "idle":
            self._app._context_flow = ""
        elif phase == "execute":
            # Will be overwritten by run_task with the real flow name when
            # known; this gives a sensible default if execute() is reached
            # before notify() has set anything.
            self._app._context_flow = self._app._context_flow or "develop"
        self._app.invalidate()

    def on_status(self, level: str, msg: str) -> None:
        # Status updates that aren't conversational go into the actions
        # line for transient feedback (gather summary, review verdict).
        # Errors get promoted to a System message so they're persistent.
        if level == "error":
            self._app.add_message("Error", msg, "system")
            return
        # Soft updates: surface in the actions text but don't pollute
        # the chat history.
        if msg:
            self._app._actions_text = msg
            self._app.invalidate()

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        self._app._chat_history_control.show_thinking = False
        self._app.add_message(speaker, msg, kind)
        # Re-enable the thinking indicator so the next phase can show it
        # if it produces a long-running call. The pipeline owns the
        # high-level flow; this just keeps the spinner consistent.
        self._app._chat_history_control.show_thinking = True
        self._app.invalidate()

    def notify_error(
        self, speaker: str, msg: str, traceback_text: str,
    ) -> None:
        # Direct append so we can attach the custom fields that
        # ErrorWidget consumes — add_message() only accepts
        # sender/text/type.
        self._app._chat_history_control.show_thinking = False
        self._app.chat_messages.append({
            "sender": speaker,
            "text": msg,
            "type": "error",
            "error_traceback": traceback_text or "",
            "collapsed": True,
        })
        self._app._chat_history_control.invalidate_cache()
        self._app._chat_history_control.show_thinking = True
        try:
            self._app.invalidate()
        except Exception:
            pass

    def notify_stream_chunk(
        self, speaker: str, chunk: str, kind: str = "agent",
    ) -> None:
        """Append *chunk* to the in-progress streaming message.

        The first chunk for a given ``(speaker, kind)`` creates a new
        chat entry (same shape as :meth:`notify`) with ``streaming=True``;
        subsequent chunks extend it in place. Markdown rendering is
        deferred until :meth:`notify_stream_end` so unclosed ``**`` /
        backticks don't render as literal text mid-stream.
        """
        self._app._chat_history_control.show_thinking = False
        self._app.append_to_last_message(speaker, chunk, kind)

    def notify_stream_end(
        self, speaker: str, kind: str = "agent",
    ) -> None:
        """Flip the streaming flag on the last message and re-render.

        After this call the message widget re-parses the full text with
        markdown / syntax highlighting applied. The user sees plain text
        appear chunk-by-chunk and then "snap" into styled form once the
        LLM finishes producing it.
        """
        self._app.finalize_streaming_message(speaker, kind)

    # ── User interaction ─────────────────────────────────────────────────

    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        """Block waiting for the user to type an answer.

        Reuses the existing ``_analysis_event`` protocol so message
        routing in :meth:`InfinidevApp.handle_user_input` (which already
        knows how to route input to ``_analysis_answer`` when
        ``_analysis_waiting`` is set) keeps working unchanged.

        Returns ``None`` only on EOF / interruption — for the TUI we
        always have an interactive user, so returning a string (possibly
        empty) is the normal case.

        UX note: previous versions silently relied on ``notify()`` having
        rendered the question right before, which is true for the
        ``clarification`` kind (analyst Q&A) but NOT for ``confirm``
        (the develop spec confirmation). The result was a 10-15 second
        "phantom hang" between analysis and develop where the user had
        no idea the system was waiting for them. We now render the
        prompt + actions hint for ``confirm`` so the user can see they
        need to type ``y`` to proceed.
        """
        app = self._app
        if kind == "confirm" and prompt:
            app.add_message("Infinidev", prompt, "system")
            app._actions_text = "Waiting for your confirmation (y / n / feedback)..."
        app._chat_history_control.show_thinking = False
        app._analysis_event = threading.Event()
        app._analysis_waiting = True
        app._analysis_answer = ""
        app.invalidate()

        try:
            app._analysis_event.wait()
        except (KeyboardInterrupt, SystemExit):
            return None
        finally:
            app._analysis_event = None
            app._analysis_waiting = False

        answer = app._analysis_answer
        app._chat_history_control.show_thinking = True
        app.invalidate()
        return answer

    # ── Progress / structured updates ────────────────────────────────────

    def on_step_start(
        self,
        step_num: int,
        total: int,
        all_steps: list[dict],
        completed: list[int],
    ) -> None:
        # Update the STEPS panel with check marks, an arrow on the active
        # step, and o for upcoming steps. Same shape as the bespoke
        # _on_step_start callback that lived inside run_plan_task.
        completed_set = set(completed)
        lines: list[str] = []
        for s in all_steps:
            s_num = s.get("step", 0)
            s_title = s.get("title", "")
            if s_num in completed_set or s_num < step_num:
                lines.append(f"v {s_title}")
            elif s_num == step_num:
                lines.append(f"> {s_title}")
            else:
                lines.append(f"o {s_title}")
        self._app._steps_text = "\n".join(lines)
        self._app._actions_text = f"Executing step {step_num}/{total}..."
        self._app.invalidate()

    def on_file_change(self, path: str) -> None:
        # Diff tracking is owned by InfinidevApp._file_diffs and updated
        # by the engine hooks (engine/hooks/ui_hooks.py). Nothing to do
        # here yet — included so the Protocol contract is satisfied and
        # so future "highlight changed file" UX has a place to land.
        return None
