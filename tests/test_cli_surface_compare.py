from types import SimpleNamespace

import bench.cli_surface_compare as cli_surface_compare
from bench.cli_surface_compare import (
    _prepare_ken,
    _remove_runtime_artifacts,
    parse_codex_jsonl,
    parse_infinidev_output,
)


def test_prepare_ken_embeds_workspace_before_infinidev(
    tmp_path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    artifact_dir = tmp_path / "artifacts"
    workspace.mkdir()
    artifact_dir.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="indexed\n", stderr="")

    monkeypatch.setattr(cli_surface_compare.subprocess, "run", fake_run)

    elapsed = _prepare_ken(workspace, artifact_dir, "/opt/bin/ken")

    assert elapsed >= 0
    assert calls == [
        (
            ["/opt/bin/ken", "install", ".", "--embed"],
            {
                "cwd": workspace,
                "text": True,
                "capture_output": True,
                "timeout": 300,
                "check": False,
            },
        )
    ]
    assert (artifact_dir / "ken-install.log").read_text() == "indexed\n"


def test_parse_codex_jsonl_extracts_usage_and_unique_completed_tools() -> None:
    text = "\n".join(
        (
            '{"type":"item.completed","item":{"id":"a","type":"command_execution",'
            '"exit_code":0,"status":"completed"}}',
            '{"type":"item.completed","item":{"id":"b","type":"file_change",'
            '"status":"completed"}}',
            '{"type":"item.completed","item":{"id":"c","type":"command_execution",'
            '"exit_code":2,"status":"failed"}}',
            '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":40,'
            '"output_tokens":20,"reasoning_output_tokens":5}}',
        )
    )

    parsed = parse_codex_jsonl(text)

    assert parsed["tool_calls"] == 3
    assert parsed["failed_tool_calls"] == 1
    assert parsed["input_tokens"] == 100
    assert parsed["cached_input_tokens"] == 40
    assert parsed["output_tokens"] == 20
    assert parsed["reasoning_output_tokens"] == 5
    assert parsed["token_metric"] == "provider-reported-exact"


def test_parse_infinidev_output_marks_token_count_as_lower_bound() -> None:
    parsed = parse_infinidev_output(
        "Task policies: bugfix.root_cause\n"
        "▸ read_file src/a.py  9.7ktk\n"
        "▸ execute_command python -m pytest  20.1ktk  ✗ exit 1\n"
    )

    assert parsed["tool_calls"] == 2
    assert parsed["failed_tool_calls"] == 1
    assert parsed["input_tokens"] == 20_100
    assert parsed["output_tokens"] is None
    assert parsed["token_metric"] == "observable-developer-loop-lower-bound"
    assert parsed["selected_policies"] == ("bugfix.root_cause",)


def test_runtime_artifacts_are_removed_before_fixture_commit(tmp_path) -> None:
    generated = (
        tmp_path / ".infinidev" / "infinidev.db",
        tmp_path / "src" / "__pycache__" / "module.pyc",
        tmp_path / "pytest-cache-files-example" / "README.md",
    )
    for path in generated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime")
    source = tmp_path / "src" / "module.py"
    source.write_text("value = 1\n")

    _remove_runtime_artifacts(tmp_path)

    assert source.exists()
    assert all(not path.exists() for path in generated)
