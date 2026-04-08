"""Interactive responsiveness tests for the Infinidev TUI.

These three tests establish the BASELINE for the responsiveness
work the user requested on 2026-04-08:

  A. ``Hola`` should respond in <5 seconds  (currently FAILS)
  B. ``fix bug X`` should produce visible task-start indicators
     within 10 seconds  (currently PASSES — sanity check)
  C. A second message sent ~2s after a long task should be
     acknowledged within 5 seconds  (currently FAILS — silence)

After implementing the planned fixes (conversational fast-path,
ack injection, mid-step polling), all three should pass.

Run with:

    uv run pytest tests/interactive/test_responsiveness.py -v -s

The ``-s`` is important — these tests print elapsed timings that
are useful even when they fail.

NOTE: each test takes 5-30 seconds because it spawns a real TUI
and waits on real LLM output. They're slow on purpose. Mark them
``@pytest.mark.slow`` so a fast ``pytest -m 'not slow'`` skips them.
"""

from __future__ import annotations

import time

import pexpect
import pytest

from tests.interactive._session import Session

pytestmark = pytest.mark.slow


# ─────────────────────────────────────────────────────────────────────
# Test A — Conversational fast-path
# ─────────────────────────────────────────────────────────────────────


def test_a_hola_responds_quickly():
    """``Hola`` is conversational, should NOT trigger the planning loop.

    Target: response visible in <5 seconds. Current behaviour:
    ~30+ seconds because it goes through analysis → plan → execute.
    """
    with Session() as s:
        s.wait_for_ready(timeout=30)

        t0 = time.perf_counter()
        s.send("Hola")
        # Look for the exact text of the hardcoded fast-path reply.
        # Using a unique substring guarantees we're matching the
        # assistant reply, not the TTY echo of the user typing "Hola".
        try:
            s.wait_for(r"En qu[ée] te puedo ayudar", timeout=15)
        except pexpect.TIMEOUT:
            elapsed = time.perf_counter() - t0
            pytest.fail(
                f"Hola did not get a conversational reply within 15s "
                f"(elapsed: {elapsed:.1f}s). Likely the engine is "
                f"running the full plan/execute pipeline for a "
                f"trivial greeting — implement the conversational "
                f"fast-path."
            )

        elapsed = time.perf_counter() - t0
        print(f"\n[A] hola took {elapsed:.2f}s")
        assert elapsed < 5.0, (
            f"Hola took {elapsed:.1f}s — target is <5s. The engine "
            f"is going through full pipeline. Implement the "
            f"conversational fast-path."
        )


# ─────────────────────────────────────────────────────────────────────
# Test B — Real task starts visibly (sanity check)
# ─────────────────────────────────────────────────────────────────────


def test_b_real_task_visible_start():
    """A real task instruction should produce visible activity within 10s.

    Sanity check: this should already pass with the current TUI.
    If it doesn't, the spawn / boot path is broken and the other
    tests will be unreliable.
    """
    with Session() as s:
        s.wait_for_ready(timeout=30)

        t0 = time.perf_counter()
        s.send("List the files in the current directory")
        try:
            # Either: the model called list_directory (visible as
            # tool call in the chat), or it printed any planning
            # marker. Both are acceptable signs of "loop started".
            s.wait_for(
                r"(list_directory|Step \d|Planning|Analyzing)",
                timeout=15,
            )
        except pexpect.TIMEOUT:
            elapsed = time.perf_counter() - t0
            pytest.fail(
                f"Real task showed no visible activity within 15s "
                f"(elapsed: {elapsed:.1f}s). The TUI may be frozen "
                f"or rendering output below the visible region."
            )

        elapsed = time.perf_counter() - t0
        print(f"\n[B] task became visible in {elapsed:.2f}s")
        assert elapsed < 10.0


# ─────────────────────────────────────────────────────────────────────
# Test C — Mid-task acknowledgment
# ─────────────────────────────────────────────────────────────────────


def test_c_midtask_message_gets_acked():
    """User message arriving mid-task should be acknowledged in <5s.

    Sequence:
      1. Kick off a long task ("explore this codebase deeply")
      2. Wait 3s for it to actually start
      3. Send a second message
      4. Verify an ack appears within 5s of step (3)

    Current behaviour: silence until the current step ends, often
    minutes. Target after fix: ack visible almost immediately.
    """
    with Session() as s:
        s.wait_for_ready(timeout=30)

        # Kick off a real task. Phrasing chosen to keep the model
        # busy for at least 30s — gives us a wide window to inject.
        s.send(
            "Explore the project structure carefully. Read every "
            "directory you find, then summarise."
        )

        # Wait for the loop to actually start working. Without this
        # the second send may race the first.
        try:
            s.wait_for(
                r"(list_directory|read_file|Step \d|Planning)",
                timeout=20,
            )
        except pexpect.TIMEOUT:
            pytest.fail("Initial long task never started — TUI broken?")

        # Give it a moment to settle into a step.
        time.sleep(2)

        # Inject the second message.
        t0 = time.perf_counter()
        s.send("espera, otra cosa rapida")

        # Look for ANY ack-style response. Patterns we'd expect from
        # a well-behaved ack injection:
        #   "Recibí tu mensaje"
        #   "Got your message"
        #   "Espera" / "Hold on" / "One moment"
        try:
            s.wait_for(
                r"(Recib[ií]|Got your message|Espera|hold on|"
                r"One moment|Acknowledged|Entendido)",
                timeout=15,
            )
        except pexpect.TIMEOUT:
            elapsed = time.perf_counter() - t0
            pytest.fail(
                f"Mid-task message was not acknowledged within 15s "
                f"(elapsed: {elapsed:.1f}s). The user message is "
                f"likely sitting in a queue waiting for the current "
                f"step to finish. Implement ack injection in the "
                f"prompt builder."
            )

        elapsed = time.perf_counter() - t0
        print(f"\n[C] mid-task ack took {elapsed:.2f}s")
        assert elapsed < 5.0
