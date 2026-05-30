"""Tests for the multi-agent council (engine/council/).

The LLM-driven pieces (moderator + members) are monkeypatched so these
tests exercise the orchestration logic — channel mechanics, the
round barrier, convergence/conclude stopping, and artifact parsing —
without any model calls.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from infinidev.engine.council.brief import (
    Alternative,
    CouncilRoster,
    DesignBrief,
    MemberAssignment,
    OpeningThread,
)
from infinidev.engine.council.channel import Channel
from infinidev.engine.council.member import MemberTurn
from infinidev.engine.council import moderator as MOD
from infinidev.engine.council import runner as RUN


# ── Channel ──────────────────────────────────────────────────────────────


class TestChannel:
    def test_open_thread_and_post(self):
        c = Channel("How to cache?")
        t = c.open_thread(
            author="moderator", title="Approach", opening_text="debate", round=0,
        )
        msg = c.post(author="mvp", thread_id=t.id, text="keep it simple", round=1)
        assert msg is not None
        assert msg.author == "mvp"
        assert len(c.all_messages()) == 2

    def test_post_to_missing_thread_returns_none(self):
        c = Channel("Q")
        assert c.post(author="x", thread_id="t-999", text="hi", round=1) is None

    def test_digest_marks_own_messages(self):
        c = Channel("Q")
        t = c.open_thread(author="moderator", title="T", opening_text="seed", round=0)
        c.post(author="mvp", thread_id=t.id, text="mine", round=1)
        c.post(author="skeptic", thread_id=t.id, text="theirs", round=1)
        digest = c.render_digest(for_author="mvp", current_round=1)
        assert "YOU said: mine" in digest
        assert "skeptic: theirs" in digest

    def test_digest_summarises_old_rounds(self):
        c = Channel("Q")
        t = c.open_thread(author="moderator", title="T", opening_text="seed", round=0)
        c.post(author="mvp", thread_id=t.id, text="old point", round=1)
        c.post(author="mvp", thread_id=t.id, text="fresh point", round=5)
        # current_round=5, recent_rounds=2 → cutoff=3, round-1 msg summarised
        digest = c.render_digest(current_round=5, recent_rounds=2)
        assert "fresh point" in digest
        assert "old point" not in digest
        assert "omitted for brevity" in digest

    def test_snapshot_is_isolated(self):
        c = Channel("Q")
        t = c.open_thread(author="moderator", title="T", opening_text="seed", round=0)
        snap = c.snapshot()
        # Mutating the live channel must not change the snapshot.
        c.post(author="mvp", thread_id=t.id, text="after snapshot", round=1)
        assert len(snap.all_messages()) == 1
        assert len(c.all_messages()) == 2

    def test_refs_render(self):
        c = Channel("Q")
        t = c.open_thread(author="moderator", title="T", opening_text="seed", round=0)
        c.post(author="mvp", thread_id=t.id, text="grounded", round=1, refs=["a.py"])
        assert "refs: a.py" in c.render_digest(current_round=1)


# ── Roster parsing ───────────────────────────────────────────────────────


class TestRosterParsing:
    def test_valid_roster(self):
        args = {
            "question": "LRU or LFU?",
            "members": [
                {"member_id": "a", "persona": "mvp", "objective": "simplest"},
                {"member_id": "b", "persona": "skeptic", "objective": "refute"},
            ],
            "opening_threads": [{"title": "T", "prompt": "go"}],
        }
        roster = MOD._parse_roster(args, "handoff")
        assert roster.question == "LRU or LFU?"
        assert len(roster.members) == 2
        assert roster.members[0].member_id == "a"

    def test_duplicate_member_ids_deduped(self):
        args = {
            "question": "Q",
            "members": [
                {"member_id": "x", "persona": "p1", "objective": "o1"},
                {"member_id": "x", "persona": "p2", "objective": "o2"},
            ],
            "opening_threads": [{"title": "T", "prompt": "go"}],
        }
        roster = MOD._parse_roster(args, "handoff")
        ids = [m.member_id for m in roster.members]
        assert len(set(ids)) == len(ids), "member ids must be unique"

    def test_members_without_persona_or_objective_dropped(self):
        args = {
            "question": "Q",
            "members": [
                {"member_id": "a", "persona": "", "objective": "o"},
                {"member_id": "b", "persona": "p", "objective": ""},
                {"member_id": "c", "persona": "p", "objective": "o"},
            ],
            "opening_threads": [{"title": "T", "prompt": "go"}],
        }
        # Only c is well-formed → underspecified → fallback roster.
        roster = MOD._parse_roster(args, "handoff")
        assert len(roster.members) >= 2  # fallback supplies a full council

    def test_underspecified_falls_back(self):
        roster = MOD._parse_roster({"question": "", "members": []}, "handoff text")
        assert isinstance(roster, CouncilRoster)
        assert len(roster.members) >= 2
        assert roster.opening_threads

    def test_member_count_clamped(self, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "COUNCIL_MAX_MEMBERS", 3)
        args = {
            "question": "Q",
            "members": [
                {"member_id": f"m{i}", "persona": "p", "objective": "o"}
                for i in range(6)
            ],
            "opening_threads": [{"title": "T", "prompt": "go"}],
        }
        roster = MOD._parse_roster(args, "handoff")
        assert len(roster.members) == 3


# ── Brief parsing ────────────────────────────────────────────────────────


class TestBriefParsing:
    def test_basic_brief(self):
        args = {
            "chosen_approach": "LRU+TTL",
            "rationale": "bounded",
            "alternatives_considered": [
                {"approach": "global dict", "why_rejected": "unbounded"},
            ],
            "open_risks": ["invalidation"],
            "research_findings": ["caches in llm.py"],
            "affected_files": ["x.py"],
            "dissent": ["skeptic disagrees"],
        }
        brief = MOD._parse_brief(args, "Q")
        assert brief.chosen_approach == "LRU+TTL"
        assert brief.alternatives_considered[0].approach == "global dict"
        assert brief.user_decision_required is False

    def test_user_decision_requires_questions(self):
        # Flag set but no questions → coerced to False (no useless interrupt).
        args = {
            "chosen_approach": "x",
            "user_decision_required": True,
            "open_questions_for_user": [],
        }
        brief = MOD._parse_brief(args, "Q")
        assert brief.user_decision_required is False

    def test_user_decision_with_questions(self):
        args = {
            "chosen_approach": "x",
            "user_decision_required": True,
            "open_questions_for_user": ["latency or cost?"],
        }
        brief = MOD._parse_brief(args, "Q")
        assert brief.user_decision_required is True
        assert "latency or cost?" in brief.render_questions_for_user()

    def test_render_for_planner_includes_sections(self):
        brief = DesignBrief(
            question="Q", chosen_approach="A", rationale="R",
            alternatives_considered=[Alternative("B", "no")],
            research_findings=["f"], affected_files=["x.py"],
            open_risks=["r"], dissent=["d"],
        )
        out = brief.render_for_planner()
        for needle in ("Chosen approach", "A", "B", "f", "x.py", "r", "d"):
            assert needle in out


# ── Runner orchestration (LLM monkeypatched) ─────────────────────────────


def _roster(n=2):
    return CouncilRoster(
        question="How to build it?",
        members=[
            MemberAssignment(member_id=f"m{i}", persona="p", objective="o")
            for i in range(n)
        ],
        opening_threads=[OpeningThread(title="Approach", prompt="debate")],
    )


class TestRunner:
    def test_disabled_returns_none(self, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "COUNCIL_ENABLED", False)
        assert RUN.run_council("handoff") is None

    def test_full_run_returns_brief(self, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "COUNCIL_ENABLED", True)
        monkeypatch.setattr(settings_mod.settings, "COUNCIL_MAX_ROUNDS", 2)

        monkeypatch.setattr(MOD, "seed_council", lambda *a, **k: _roster(2))
        # Each member posts once per round.
        posts = []

        def fake_member(assignment, **kw):
            posts.append((assignment.member_id, kw["round_num"]))
            return MemberTurn(
                member_id=assignment.member_id, action="post",
                message=f"{assignment.member_id} says hi r{kw['round_num']}",
            )

        monkeypatch.setattr(RUN, "run_member_round", fake_member)
        # Never converge early → runs all rounds.
        monkeypatch.setattr(
            MOD, "judge_convergence", lambda *a, **k: (False, "keep going"),
        )
        captured = {}

        def fake_synth(digest, question, **kw):
            captured["digest"] = digest
            return DesignBrief(question=question, chosen_approach="synth")

        monkeypatch.setattr(MOD, "synthesize", fake_synth)

        brief = RUN.run_council("handoff")
        assert brief is not None
        assert brief.chosen_approach == "synth"
        # 2 members × 2 rounds = 4 member invocations
        assert len(posts) == 4
        # Both members' posts reached the digest handed to synth.
        assert "m0 says hi" in captured["digest"]
        assert "m1 says hi" in captured["digest"]

    def test_convergence_stops_early(self, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "COUNCIL_ENABLED", True)
        monkeypatch.setattr(settings_mod.settings, "COUNCIL_MAX_ROUNDS", 5)
        monkeypatch.setattr(MOD, "seed_council", lambda *a, **k: _roster(2))

        rounds_seen = []

        def fake_member(assignment, **kw):
            rounds_seen.append(kw["round_num"])
            return MemberTurn(
                member_id=assignment.member_id, action="post", message="x",
            )

        monkeypatch.setattr(RUN, "run_member_round", fake_member)
        # Converge after round 1.
        monkeypatch.setattr(
            MOD, "judge_convergence", lambda *a, **k: (True, "done"),
        )
        monkeypatch.setattr(
            MOD, "synthesize",
            lambda d, q, **k: DesignBrief(question=q, chosen_approach="ok"),
        )

        RUN.run_council("handoff")
        # Only round 1 ran before convergence stopped it.
        assert set(rounds_seen) == {1}

    def test_all_conclude_stops_early(self, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "COUNCIL_ENABLED", True)
        monkeypatch.setattr(settings_mod.settings, "COUNCIL_MAX_ROUNDS", 5)
        monkeypatch.setattr(MOD, "seed_council", lambda *a, **k: _roster(2))

        calls = {"judge": 0}

        def fake_member(assignment, **kw):
            return MemberTurn(
                member_id=assignment.member_id, action="conclude",
                final_position="done",
            )

        def fake_judge(*a, **k):
            calls["judge"] += 1
            return (False, "n/a")

        monkeypatch.setattr(RUN, "run_member_round", fake_member)
        monkeypatch.setattr(MOD, "judge_convergence", fake_judge)
        monkeypatch.setattr(
            MOD, "synthesize",
            lambda d, q, **k: DesignBrief(question=q, chosen_approach="ok"),
        )

        RUN.run_council("handoff")
        # All members concluded in round 1 → judge never consulted.
        assert calls["judge"] == 0

    def test_seed_failure_returns_none(self, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "COUNCIL_ENABLED", True)
        empty = CouncilRoster(question="Q", members=[], opening_threads=[])
        monkeypatch.setattr(MOD, "seed_council", lambda *a, **k: empty)
        assert RUN.run_council("handoff") is None


# ── Escalation packet wiring ─────────────────────────────────────────────


class TestEscalationWiring:
    def test_escalate_parses_council_fields(self):
        from infinidev.engine.orchestration.chat_agent import _build_escalate
        import types

        tc = types.SimpleNamespace(
            function=types.SimpleNamespace(
                arguments=(
                    '{"understanding": "wants a debate", '
                    '"council_requested": true, "council_focus": "both"}'
                ),
            ),
        )
        result = _build_escalate(tc, "usá varios subagentes para diseñar esto")
        assert result.kind == "escalate"
        assert result.escalation.council_requested is True
        assert result.escalation.council_focus == "both"

    def test_invalid_focus_defaults_to_design(self):
        from infinidev.engine.orchestration.chat_agent import _build_escalate
        import types

        tc = types.SimpleNamespace(
            function=types.SimpleNamespace(
                arguments=(
                    '{"understanding": "x", "council_requested": true, '
                    '"council_focus": "nonsense"}'
                ),
            ),
        )
        result = _build_escalate(tc, "do it")
        assert result.escalation.council_focus == "design"
