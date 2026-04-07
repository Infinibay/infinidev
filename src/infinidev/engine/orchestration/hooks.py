"""Reusable :class:`OrchestrationHooks` implementations.

These are the default adapters that ship with the orchestration
package. Each entry point picks the one that matches its execution
environment, or implements its own (the TUI does, because it has to
marshal everything back to a prompt_toolkit event loop).

  * :class:`NoOpHooks`         — drops every call. Used by tests.
  * :class:`ClickHooks`        — terminal output via ``click.echo``.
                                 Asks questions through a
                                 ``prompt_toolkit.PromptSession`` if
                                 one is supplied, otherwise via
                                 ``input()``.
  * :class:`NonInteractiveHooks` — never blocks. ``ask_user`` always
                                   returns ``None`` so the pipeline
                                   takes the non-interactive branch.
                                   Used by ``--prompt`` mode.
"""

from __future__ import annotations

from typing import Any


class NoOpHooks:
    """Silent default. Useful in tests and as a base for partial overrides.

    Every method is a no-op except :meth:`ask_user`, which returns
    ``None`` to signal that no UI is attached. The pipeline must always
    treat ``None`` as "skip this question, use defaults".
    """

    def on_phase(self, phase: str) -> None:
        return None

    def on_status(self, level: str, msg: str) -> None:
        return None

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        return None

    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        return None

    def on_step_start(
        self,
        step_num: int,
        total: int,
        all_steps: list[dict],
        completed: list[int],
    ) -> None:
        return None

    def on_file_change(self, path: str) -> None:
        return None


class ClickHooks(NoOpHooks):
    """Terminal-friendly hooks built on ``click.echo``.

    Used by the interactive classic CLI. Pass a
    ``prompt_toolkit.PromptSession`` for nicer line editing during
    :meth:`ask_user`; without one, falls back to ``input()``.

    The level → colour mapping in :meth:`on_status` mirrors the legacy
    ``cli/phases.py`` formatting so users see no behaviour change.
    """

    _LEVEL_COLOURS: dict[str, tuple[str, bool]] = {
        # level: (fg, dim)
        "info": ("cyan", True),
        "warn": ("yellow", True),
        "error": ("red", False),
        "verification_pass": ("green", True),
        "verification_fail": ("red", False),
        "approved": ("green", True),
        "rejected": ("red", False),
        "max_reviews": ("yellow", True),
    }

    _SPEAKER_COLOURS: dict[str, tuple[str, bool]] = {
        "Analyst": ("cyan", False),
        "Planner": ("cyan", False),
        "Reviewer": ("magenta", False),
        "Verifier": ("magenta", False),
        "Infinidev": ("green", False),
        "System": ("white", True),
        "Error": ("red", False),
    }

    def __init__(self, session: Any | None = None) -> None:
        self._session = session

    def on_phase(self, phase: str) -> None:
        # Phase transitions don't print anything in classic CLI; the
        # individual on_status calls inside each phase already give the
        # user enough feedback. Override in a subclass if you want a
        # banner.
        return None

    def on_status(self, level: str, msg: str) -> None:
        import click
        fg, dim = self._LEVEL_COLOURS.get(level, ("white", False))
        click.echo(click.style(msg, fg=fg, dim=dim))

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        import click
        fg, dim = self._SPEAKER_COLOURS.get(speaker, ("white", False))
        click.echo(click.style(f"\n[{speaker}]", fg=fg, bold=True, dim=dim))
        click.echo(msg)

    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        import click
        click.echo(click.style(prompt, fg="cyan"))
        if self._session is not None:
            try:
                return self._session.prompt("> ")
            except (EOFError, KeyboardInterrupt):
                return None
        try:
            return input("> ")
        except (EOFError, KeyboardInterrupt):
            return None


class NonInteractiveHooks(ClickHooks):
    """Hooks for one-shot ``--prompt`` mode.

    Inherits the colourised output from :class:`ClickHooks` but refuses
    to ask questions: every call to :meth:`ask_user` returns ``None``,
    which the pipeline must interpret as "skip the question, use
    sensible defaults". The analyst question loop short-circuits on the
    first ``None``; the spec confirmation step proceeds without asking.
    """

    def __init__(self) -> None:
        super().__init__(session=None)

    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        # Print the prompt so the user (looking at logs) can see what
        # was being asked, but never block waiting for an answer.
        import click
        click.echo(
            click.style(
                f"[non-interactive: skipping question] {prompt}",
                fg="yellow",
                dim=True,
            )
        )
        return None
