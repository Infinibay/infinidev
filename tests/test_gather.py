"""Tests for the information gathering phase."""

import json
from unittest.mock import patch, MagicMock

import pytest

from infinidev.gather.models import (
    TicketType, Question, QuestionResult, ClassificationResult, GatherBrief,
)
from infinidev.gather.questions import get_questions_for_type
from infinidev.gather.compiler import compile_brief


# ── Models ───────────────────────────────────────────────────────────────────


class TestTicketType:
    def test_all_types_exist(self):
        assert TicketType.bug == "bug"
        assert TicketType.feature == "feature"
        assert TicketType.refactor == "refactor"
        assert TicketType.sysadmin == "sysadmin"
        assert TicketType.other == "other"

    def test_from_string(self):
        assert TicketType("bug") == TicketType.bug
        assert TicketType("feature") == TicketType.feature


class TestQuestion:
    def test_create_question(self):
        q = Question(id="test", question="What?", context_prompt="Find {ticket_description}")
        assert q.id == "test"
        assert q.max_tool_calls == 30
        assert q.timeout_seconds == 120

    def test_context_prompt_formatting(self):
        q = Question(id="t", question="Q", context_prompt="About: {ticket_description}")
        result = q.context_prompt.format(ticket_description="fix the bug")
        assert "fix the bug" in result


class TestQuestionResult:
    def test_create(self):
        r = QuestionResult(
            question_id="test", question_text="What?", answer="Found it.",
            tool_calls_used=5, phase="fixed",
        )
        assert r.tool_calls_used == 5
        assert r.phase == "fixed"


class TestClassificationResult:
    def test_defaults(self):
        r = ClassificationResult()
        assert r.ticket_type == TicketType.other
        assert r.reasoning == ""
        assert r.keywords == []


class TestGatherBrief:
    def test_render_empty(self):
        brief = GatherBrief()
        rendered = brief.render()
        assert "<gathered-context>" in rendered
        assert "<classification" in rendered
        assert "</gathered-context>" in rendered

    def test_render_with_answers(self):
        brief = GatherBrief(
            classification=ClassificationResult(
                ticket_type=TicketType.bug,
                reasoning="It's a bug",
                keywords=["crash", "error"],
            ),
            fixed_answers=[
                QuestionResult(
                    question_id="files",
                    question_text="What files?",
                    answer="src/main.py has the bug at line 42.",
                    phase="fixed",
                ),
            ],
            dynamic_answers=[
                QuestionResult(
                    question_id="dyn_0",
                    question_text="Any related issues?",
                    answer="No related issues found.",
                    phase="dynamic",
                ),
            ],
        )
        rendered = brief.render()
        assert 'type="bug"' in rendered
        assert "crash" in rendered
        assert "src/main.py" in rendered
        assert "line 42" in rendered
        assert 'phase="fixed"' in rendered
        assert 'phase="dynamic"' in rendered

    def test_render_truncates_long_answers(self):
        brief = GatherBrief(
            fixed_answers=[
                QuestionResult(
                    question_id="long",
                    question_text="What?",
                    answer="x" * 5000,
                    phase="fixed",
                ),
            ],
        )
        rendered = brief.render()
        assert "[truncated]" in rendered
        assert len(rendered) < 5000

    def test_summary(self):
        brief = GatherBrief(
            classification=ClassificationResult(ticket_type=TicketType.feature),
            fixed_answers=[
                QuestionResult(question_id="a", question_text="Q", answer="A", phase="fixed"),
                QuestionResult(question_id="b", question_text="Q", answer="A", phase="fixed"),
            ],
            dynamic_answers=[
                QuestionResult(question_id="c", question_text="Q", answer="A", phase="dynamic"),
            ],
        )
        s = brief.summary()
        assert "3 context items" in s
        assert "feature" in s


# ── Questions Registry ───────────────────────────────────────────────────────


class TestQuestionsRegistry:
    def test_bug_questions_exist(self):
        qs = get_questions_for_type(TicketType.bug)
        assert len(qs) >= 3
        ids = [q.id for q in qs]
        assert "related_files" in ids
        assert "root_cause_candidates" in ids

    def test_feature_questions_exist(self):
        qs = get_questions_for_type(TicketType.feature)
        assert len(qs) >= 3
        ids = [q.id for q in qs]
        assert "existing_patterns" in ids
        assert "integration_points" in ids

    def test_refactor_questions_exist(self):
        qs = get_questions_for_type(TicketType.refactor)
        assert len(qs) >= 3

    def test_sysadmin_questions_exist(self):
        qs = get_questions_for_type(TicketType.sysadmin)
        assert len(qs) >= 3

    def test_other_questions_exist(self):
        qs = get_questions_for_type(TicketType.other)
        assert len(qs) >= 2

    def test_all_questions_have_context_prompt(self):
        for tt in TicketType:
            for q in get_questions_for_type(tt):
                assert "{ticket_description}" in q.context_prompt, (
                    f"Question {q.id} for {tt} missing {{ticket_description}} placeholder"
                )

    def test_returns_copy(self):
        """Modifying returned list shouldn't affect the registry."""
        qs1 = get_questions_for_type(TicketType.bug)
        qs1.pop()
        qs2 = get_questions_for_type(TicketType.bug)
        assert len(qs2) > len(qs1)


# ── Compiler ─────────────────────────────────────────────────────────────────


class TestCompiler:
    def test_compile_brief(self):
        classification = ClassificationResult(
            ticket_type=TicketType.bug, reasoning="It crashes",
        )
        fixed = [
            QuestionResult(question_id="f1", question_text="Q1", answer="A1", phase="fixed"),
        ]
        dynamic = [
            QuestionResult(question_id="d1", question_text="Q2", answer="A2", phase="dynamic"),
        ]
        brief = compile_brief("fix the bug", classification, fixed, dynamic)
        assert brief.ticket_description == "fix the bug"
        assert brief.classification.ticket_type == TicketType.bug
        assert len(brief.fixed_answers) == 1
        assert len(brief.dynamic_answers) == 1


# ── Classifier ───────────────────────────────────────────────────────────────


class TestClassifier:
    def test_extract_json(self):
        from infinidev.gather.classifier import _extract_json

        # Clean JSON
        assert _extract_json('{"ticket_type": "bug"}') == {"ticket_type": "bug"}

        # JSON in markdown fences
        assert _extract_json('```json\n{"ticket_type": "feature"}\n```') == {"ticket_type": "feature"}

        # JSON with surrounding text
        result = _extract_json('Here is the result: {"ticket_type": "bug", "reasoning": "crashes"} done.')
        assert result is not None
        assert result["ticket_type"] == "bug"

        # No JSON
        assert _extract_json("just plain text") is None

        # Empty
        assert _extract_json("") is None

    def test_fallback_on_no_agent(self):
        from infinidev.gather.classifier import classify_ticket
        result = classify_ticket("some ticket", agent=None)
        assert result.ticket_type == TicketType.other


# ── Dynamic Questions Parser ─────────────────────────────────────────────────


class TestDynamicQuestionsParser:
    def test_parse_json_array(self):
        from infinidev.gather.runner import _parse_dynamic_questions_result
        text = '[{"id": "q1", "question": "Is there a related bug?", "context_prompt": "search {ticket_description}"}]'
        result = _parse_dynamic_questions_result(text, 10)
        assert len(result) == 1
        assert result[0].id == "q1"
        assert result[0].question == "Is there a related bug?"

    def test_parse_json_object_with_questions_key(self):
        from infinidev.gather.runner import _parse_dynamic_questions_result
        text = '{"questions": [{"id": "q1", "question": "Why does it crash?"}]}'
        result = _parse_dynamic_questions_result(text, 10)
        assert len(result) == 1

    def test_parse_text_list(self):
        from infinidev.gather.runner import _parse_dynamic_questions_result
        text = """Here are additional questions:
1. Are there other case-sensitive comparisons in the module?
2. Does the fix need to handle unicode characters?
3. What about backward compatibility?"""
        result = _parse_dynamic_questions_result(text, 10)
        assert len(result) >= 2
        assert any("case-sensitive" in q.question for q in result)

    def test_parse_bullet_list(self):
        from infinidev.gather.runner import _parse_dynamic_questions_result
        text = """- Are there similar patterns in other parsers?
- Should we add regression tests for this?"""
        result = _parse_dynamic_questions_result(text, 10)
        assert len(result) >= 1

    def test_parse_empty_array(self):
        from infinidev.gather.runner import _parse_dynamic_questions_result
        assert _parse_dynamic_questions_result("[]", 10) == []

    def test_max_limit(self):
        from infinidev.gather.runner import _parse_dynamic_questions_result
        text = json.dumps([{"id": f"q{i}", "question": f"Question {i}?"} for i in range(20)])
        result = _parse_dynamic_questions_result(text, 3)
        assert len(result) == 3

    def test_empty_text(self):
        from infinidev.gather.runner import _parse_dynamic_questions_result
        assert _parse_dynamic_questions_result("", 10) == []


# ── Tool Aliases ─────────────────────────────────────────────────────────────


class TestToolAliases:
    def test_query_alias_for_pattern(self):
        """query should be auto-corrected to pattern in tool dispatch."""
        from infinidev.engine.loop_tools import execute_tool_call
        # We can't fully test without a real tool, but we can verify the alias map
        # exists in the code
        import infinidev.engine.loop_tools as lt
        source = open(lt.__file__).read()
        assert '"query": "pattern"' in source

    def test_metadata_params_stripped(self):
        """description/reason should be silently stripped, not cause errors."""
        import infinidev.engine.loop_tools as lt
        source = open(lt.__file__).read()
        assert "_METADATA_PARAMS" in source
        assert '"description"' in source


# ── Read File Aliases ────────────────────────────────────────────────────────


class TestReadFileAliases:
    def test_start_line_end_line(self, bound_tool, workspace_dir):
        """start_line and end_line should work as aliases for offset/limit."""
        from infinidev.tools.file.read_file import ReadFileTool
        tool = bound_tool(ReadFileTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"),
            start_line=2,
            end_line=3,
        )
        assert "line two" in result
        assert "line three" in result
        assert "line one" not in result
        assert "line four" not in result

    def test_start_line_only(self, bound_tool, workspace_dir):
        """start_line without end_line should read from that line to end."""
        from infinidev.tools.file.read_file import ReadFileTool
        tool = bound_tool(ReadFileTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"),
            start_line=4,
        )
        assert "line four" in result
        assert "line five" in result
        assert "line one" not in result

    def test_line_range_dash(self, bound_tool, workspace_dir):
        """line_range='2-3' should read lines 2 and 3."""
        from infinidev.tools.file.read_file import ReadFileTool
        tool = bound_tool(ReadFileTool)
        result = tool._run(path=str(workspace_dir / "sample.txt"), line_range="2-3")
        assert "line two" in result
        assert "line three" in result
        assert "line one" not in result

    def test_line_range_colon(self, bound_tool, workspace_dir):
        """line_range='4:5' should read lines 4 and 5."""
        from infinidev.tools.file.read_file import ReadFileTool
        tool = bound_tool(ReadFileTool)
        result = tool._run(path=str(workspace_dir / "sample.txt"), line_range="4:5")
        assert "line four" in result
        assert "line five" in result

    def test_offset_takes_precedence(self, bound_tool, workspace_dir):
        """If both offset and start_line provided, offset wins."""
        from infinidev.tools.file.read_file import ReadFileTool
        tool = bound_tool(ReadFileTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"),
            offset=1,
            limit=1,
            start_line=5,  # Should be ignored since offset is provided
        )
        assert "line one" in result


# ── Step Result Coercion ─────────────────────────────────────────────────────


class TestStepResultCoercion:
    def test_final_answer_dict_coerced_to_string(self):
        """final_answer passed as dict should be JSON-serialized."""
        from infinidev.engine.loop_engine import _parse_step_complete_args
        result = _parse_step_complete_args({
            "summary": "classified",
            "status": "done",
            "final_answer": {"ticket_type": "bug", "reasoning": "it crashes"},
        })
        assert isinstance(result.final_answer, str)
        parsed = json.loads(result.final_answer)
        assert parsed["ticket_type"] == "bug"

    def test_final_answer_list_coerced_to_string(self):
        from infinidev.engine.loop_engine import _parse_step_complete_args
        result = _parse_step_complete_args({
            "summary": "generated questions",
            "status": "done",
            "final_answer": [{"id": "q1", "question": "Why?"}],
        })
        assert isinstance(result.final_answer, str)
        assert "q1" in result.final_answer

    def test_final_answer_string_unchanged(self):
        from infinidev.engine.loop_engine import _parse_step_complete_args
        result = _parse_step_complete_args({
            "summary": "done",
            "status": "done",
            "final_answer": "The answer is 42.",
        })
        assert result.final_answer == "The answer is 42."

    def test_final_answer_none_unchanged(self):
        from infinidev.engine.loop_engine import _parse_step_complete_args
        result = _parse_step_complete_args({
            "summary": "continuing",
            "status": "continue",
        })
        assert result.final_answer is None
