"""Tests for the compact critic group (ui/controls/critic_widget.py).

The critic used to render as full system messages — name header, model
line, whole body, in amber — on most steps. These pin the three
properties that fix: it collapses by default, a lone verdict collapses
too, and severity survives the collapse.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.ui.controls.critic_widget import build_critic_group
from infinidev.ui.controls.message_groups import (
    COMPACT_GROUP_TYPES,
    NEVER_GROUP_TYPES,
    identify_groups,
)


def _verdict(action="recommendation", text="body text", source="tools"):
    return {
        "sender": f"Assistant · {action.upper()}",
        "text": text,
        "type": "critic",
        "critic_action": action,
        "critic_model": "responses/gpt-5.6-sol",
        "critic_source": source,
    }


def _render(messages, *, collapsed=True, expanded=None, width=100):
    return build_critic_group(
        messages,
        collapsed=collapsed,
        expanded_set=expanded or set(),
        width=width,
        on_toggle_group=lambda: None,
        on_toggle_item=lambda i: None,
    )


def _text(result) -> str:
    return "\n".join("".join(t for _, t in line) for line in result.lines)


class TestCollapsedByDefault:
    """One line is the whole point."""

    def test_collapsed_group_is_a_single_line(self):
        result = _render([_verdict(), _verdict(), _verdict()])
        # summary + trailing blank separator
        assert len(result.lines) == 2
        assert "3 notes" in _text(result)

    def test_body_text_is_hidden_when_collapsed(self):
        result = _render([_verdict(text="a very specific recommendation")])
        assert "a very specific recommendation" not in _text(result)

    def test_a_lone_verdict_also_collapses(self):
        """Unlike tool groups: one critic paragraph is as interruptive as three."""
        result = _render([_verdict()])
        assert len(result.lines) == 2
        assert "1 note" in _text(result)

    def test_singular_and_plural_units(self):
        assert "1 note" in _text(_render([_verdict()]))
        assert "2 notes" in _text(_render([_verdict(), _verdict()]))


class TestSeveritySurvivesCollapse:
    """A reject is worth opening; a recommendation usually is not."""

    def test_reject_count_shows_in_the_summary(self):
        result = _render([_verdict(), _verdict("reject"), _verdict("reject")])
        assert "2 reject" in _text(result)

    def test_no_reject_note_when_all_advisory(self):
        assert "reject" not in _text(_render([_verdict(), _verdict()]))

    def test_reject_changes_the_summary_icon(self):
        plain = _text(_render([_verdict()]))
        rejected = _text(_render([_verdict("reject")]))
        assert plain.split()[0] != rejected.split()[0]

    def test_unknown_action_falls_back_to_info(self):
        """A new critic action must not crash the transcript."""
        result = _render([_verdict("brand-new-verdict-kind")], collapsed=False)
        assert "INFO" in _text(result)


class TestExpansion:
    """Two levels: group open, then a verdict open."""

    def test_expanded_group_lists_one_line_per_verdict(self):
        result = _render([_verdict(), _verdict("reject")], collapsed=False)
        body = _text(result)
        assert "RECOMMEND" in body
        assert "REJECT" in body

    def test_index_line_previews_but_does_not_dump_the_body(self):
        long_text = "T" * 400
        result = _render([_verdict(text=long_text)], collapsed=False)
        assert long_text not in _text(result)
        assert "…" in _text(result)

    def test_expanding_a_verdict_shows_its_full_body(self):
        result = _render(
            [_verdict(text="the entire verdict body")],
            collapsed=False, expanded={0},
        )
        assert "the entire verdict body" in _text(result)

    def test_model_appears_only_once_expanded(self):
        collapsed = _text(_render([_verdict()], collapsed=False))
        opened = _text(_render([_verdict()], collapsed=False, expanded={0}))
        assert "gpt-5.6-sol" not in collapsed
        assert "gpt-5.6-sol" in opened

    def test_source_is_shown_when_it_is_not_the_default(self):
        result = _render(
            [_verdict(source="step_complete")], collapsed=False,
        )
        assert "re: step_complete" in _text(result)

    def test_default_source_adds_no_noise(self):
        result = _render([_verdict(source="tools")], collapsed=False)
        assert "re:" not in _text(result)


class TestClickTargets:
    """The affordance users already learned from tool groups."""

    def test_summary_line_toggles_the_group(self):
        toggled = []
        result = build_critic_group(
            [_verdict()], collapsed=True, expanded_set=set(), width=80,
            on_toggle_group=lambda: toggled.append("group"),
            on_toggle_item=lambda i: toggled.append(i),
        )
        result.clickable_offsets[0]()
        assert toggled == ["group"]

    def test_each_index_line_toggles_its_own_verdict(self):
        toggled = []
        result = build_critic_group(
            [_verdict(), _verdict()], collapsed=False, expanded_set=set(),
            width=80, on_toggle_group=lambda: None,
            on_toggle_item=lambda i: toggled.append(i),
        )
        for offset in sorted(result.clickable_offsets)[1:]:
            result.clickable_offsets[offset]()
        assert toggled == [0, 1]


class TestGrouping:
    """Consecutive verdicts must reach the compact renderer as one group."""

    def test_critic_takes_the_compact_path(self):
        assert "critic" in COMPACT_GROUP_TYPES
        assert "critic" not in NEVER_GROUP_TYPES

    def test_consecutive_verdicts_form_one_group(self):
        groups = identify_groups([_verdict(), _verdict(), _verdict()])
        assert len(groups) == 1
        assert len(groups[0].messages) == 3

    def test_an_interleaved_message_splits_the_run(self):
        groups = identify_groups([
            _verdict(),
            {"type": "agent", "sender": "Infinidev", "text": "reply"},
            _verdict(),
        ])
        assert [g.msg_type for g in groups] == ["critic", "agent", "critic"]


class TestEventHandlerEmitsStructuredFields:
    """Severity/model/source travel as fields, not baked into the body."""

    def _dispatch(self, data):
        from infinidev.ui.event_handler import _dispatch as dispatch_event

        captured = []

        class _App:
            def add_message(self, sender, text, msg_type="agent", **fields):
                captured.append(
                    {"sender": sender, "text": text, "type": msg_type, **fields}
                )

            def __getattr__(self, name):
                # The handler touches several unrelated app attributes;
                # this keeps the stub to the one method under test.
                return lambda *a, **k: None

        dispatch_event(_App(), "loop_assistant_message", data)
        return captured

    def test_verdict_becomes_a_critic_message(self):
        got = self._dispatch({
            "action": "reject", "message": "no", "model": "m", "source": "tools",
        })
        assert len(got) == 1
        assert got[0]["type"] == "critic"
        assert got[0]["critic_action"] == "reject"
        assert got[0]["critic_model"] == "m"

    def test_body_is_the_verdict_alone(self):
        """A body prefixed with "(model)" would poison the preview line."""
        got = self._dispatch({
            "action": "recommendation", "message": "do the thing",
            "model": "m", "source": "step_complete",
        })
        assert got[0]["text"] == "do the thing"
        assert got[0]["critic_source"] == "step_complete"

    def test_empty_verdict_is_dropped(self):
        assert self._dispatch({"action": "reject", "message": "   "}) == []


class TestAddMessageCarriesFields:
    """The transport the structured fields ride on."""

    def test_extra_fields_land_on_the_message_dict(self):
        from infinidev.ui.app import InfinidevApp

        app = SimpleNamespace(
            chat_messages=[],
            _seal_open_stream=lambda: None,
            _chat_history_control=SimpleNamespace(invalidate_cache=lambda: None),
            invalidate=lambda: None,
        )
        InfinidevApp.add_message(
            app, "Assistant · REJECT", "body", "critic", critic_action="reject",
        )
        assert app.chat_messages[0]["critic_action"] == "reject"
        assert app.chat_messages[0]["type"] == "critic"
