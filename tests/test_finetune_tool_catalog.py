"""Regression tests for the production fine-tuning dataset surface."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from finetune import build_dataset
from finetune.tool_catalog import get_training_schema_map, get_training_tool_schemas
from finetune.validate_quality import validate_scenario
from infinidev.engine.schema_sanitizer import (
    ADD_NOTE_SCHEMA,
    ADD_SESSION_NOTE_SCHEMA,
    STEP_COMPLETE_SCHEMA,
)
from infinidev.engine.tool_dispatch import _RETIRED_TOOLS
from infinidev.tools import get_tools_for_role


def _scenario(extra_call: dict | None = None) -> dict:
    turns = [
        {"role": "user", "content": "<task>Change the greeting.</task>"},
        {
            "role": "assistant",
            "tool_calls": [
                {"name": "read_file", "arguments": {"file_path": "src/app.py"}}
            ],
        },
        {"role": "tool", "content": "1: greeting = 'hello'"},
    ]
    if extra_call is not None:
        turns.extend(
            [
                {"role": "assistant", "tool_calls": [extra_call]},
                {"role": "tool", "content": "Updated src/app.py"},
            ]
        )
    turns.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "execute_command",
                        "arguments": {"command": "pytest tests/test_app.py"},
                    }
                ],
            },
            {"role": "tool", "content": "1 passed"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "step_complete",
                        "arguments": {
                            "summary": "Updated and verified the greeting.",
                            "evidence_summary": (
                                "pytest tests/test_app.py completed with one passing test."
                            ),
                            "status": "done",
                            "final_answer": "The greeting is updated and its test passes.",
                        },
                    }
                ],
            },
        ]
    )
    return {
        "repo": "example",
        "lang": "python",
        "type": "bug_fix",
        "prompt": "Change the greeting.",
        "turns": turns,
        "format_version": "v3_structured",
    }


def test_training_catalog_matches_local_runtime_tools() -> None:
    schemas = get_training_tool_schemas(
        supports_vision=False,
        command_output_capture=False,
    )
    actual_names = [schema["function"]["name"] for schema in schemas]

    with patch("infinidev.tools.discover_mcp_tool_classes", return_value=[]):
        runtime_tools = get_tools_for_role("developer", supports_vision=False)
    expected_names = [tool.name for tool in runtime_tools if tool.name != "read_command_output"]
    expected_names.extend(
        schema["function"]["name"]
        for schema in (STEP_COMPLETE_SCHEMA, ADD_NOTE_SCHEMA, ADD_SESSION_NOTE_SCHEMA)
    )

    assert actual_names == expected_names


def test_training_catalog_uses_current_edit_and_read_contracts() -> None:
    schemas = get_training_schema_map(command_output_capture=False)

    assert "edit_file" in schemas
    assert not _RETIRED_TOOLS.keys() & schemas.keys()
    assert schemas["read_file"]["parameters"]["required"] == ["file_path"]
    assert set(schemas["edit_file"]["parameters"]["required"]) == {
        "file_path",
        "old_string",
        "new_string",
    }


def test_training_prompt_contains_no_retired_tool_names() -> None:
    prompt = build_dataset.build_system_prompt_with_tools()

    for retired in _RETIRED_TOOLS:
        assert retired not in prompt
    assert '"name": "read_file"' in prompt
    assert '"file_path": "src/main.py"' in prompt


def test_scenario_validation_rejects_retired_tool_call() -> None:
    scenario = _scenario(
        {
            "name": "replace_lines",
            "arguments": {
                "file_path": "src/app.py",
                "start_line": 1,
                "end_line": 1,
                "content": "greeting = 'hi'",
            },
        }
    )

    result = validate_scenario(scenario)

    assert any("not in the live developer schema" in error for error in result["errors"])


def test_dataset_build_refuses_invalid_batch_without_writing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenarios_dir = tmp_path / "scenarios"
    output_dir = tmp_path / "dataset"
    scenarios_dir.mkdir()
    scenario = _scenario(
        {
            "name": "replace_lines",
            "arguments": {"file_path": "src/app.py", "content": "greeting = 'hi'"},
        }
    )
    (scenarios_dir / "obsolete.jsonl").write_text(json.dumps(scenario) + "\n")
    monkeypatch.setattr(build_dataset, "SCENARIOS_DIR", scenarios_dir)
    monkeypatch.setattr(build_dataset, "DATASET_DIR", output_dir)

    with pytest.raises(build_dataset.DatasetValidationError):
        build_dataset.build_dataset("qwen_native")

    assert not output_dir.exists()
