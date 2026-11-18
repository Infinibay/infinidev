"""Paired creation probe for the principal runtime. See bench/README.md."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import pytest

from infinidev.config.settings import settings
from infinidev.engine.turns import runtime as turn_runtime
from infinidev.prompts import optional_capabilities
from infinidev.prompts.profiles import EffectivePromptConfiguration, PromptProfile
from infinidev.tools.base import permissions
from infinidev.tools.base.context import get_context_for_agent
from infinidev.tools.file.code_search_tool import CodeSearchTool
from infinidev.tools.file.create_file_tool import CreateFileTool
from infinidev.tools.file.edit_file_tool import EditFileTool
from infinidev.tools.file.read_file_tool import ReadFileTool
from tests.test_turn_runtime import reply, runtime, tc

_CAPABILITY = "capability.progress_discipline"
_BASE_FRAGMENT = dict(optional_capabilities.OPTIONAL_CAPABILITIES)[_CAPABILITY]
_CREATION_RULE = (
    "For a requested new file, identify the destination and the source facts needed to write it.\n"
    "Once those facts are observed, use create_file with the requested path and complete content.\n"
    "A new path is expected to be absent. After creation, verify that exact file and finish "
    "when the requested checks pass."
)
_MODEL = "custom_openai/Qwen3.8-FN-UD-IQ4_XS"
_ENDPOINT = "http://127.0.0.1:8080/v1"
_REAL_CALL = turn_runtime.call_llm
_REAL_PARAMS = turn_runtime.get_litellm_params
_REAL_CAPABILITIES = turn_runtime.get_model_capabilities

# Source and a plausible wrong value, then exact requested destination and output.
_CASES = {
    "timeout": (
        "Add app/defaults.py containing only DEFAULT_TIMEOUT_MS, an integer equal to "
        "the current timeout returned by default_timeout_ms in app/settings.py. "
        "Leave existing files untouched. Verify the new file and finish.",
        "app/settings.py", "default_timeout_ms", "2400", "1000",
        "app/defaults.py", "DEFAULT_TIMEOUT_MS",
    ),
    "queue": (
        "Find max_pending_jobs in the active Python implementation under pkg/. "
        "Create pkg/limits.py with only MAX_PENDING_JOBS set to the returned integer. "
        "Ignore archive/ and leave existing files untouched. Verify and finish.",
        "pkg/scheduler.py", "max_pending_jobs", "37", "2", "pkg/limits.py", "MAX_PENDING_JOBS",
    ),
    "label": (
        "Create ui/labels.py with a single tuple constant STATUS_LABELS. Use the two "
        "labels returned by status_labels in ui/status.py, in their current order. "
        "Leave existing files untouched. Verify and finish.",
        "ui/status.py", "status_labels", '("queued", "done")', '("ready", "done")',
        "ui/labels.py", "STATUS_LABELS",
    ),
    "size": (
        "Create storage/defaults.py with just the integer constant DEFAULT_BATCH_SIZE matching "
        "the value returned by default_batch_size in storage/batch.py. Leave all inputs untouched. "
        "Verify the created file and finish.",
        "storage/batch.py", "default_batch_size", "24", "72",
        "storage/defaults.py", "DEFAULT_BATCH_SIZE",
    ),
    "color": (
        "Find theme_colors in the active package under display/. Create display/palette.py "
        "with a single constant THEME_COLORS equal to its tuple, preserving order. "
        "Leave all existing files untouched. Verify and finish.",
        "display/theme.py", "theme_colors", '("blue", "green")', '("red", "green")',
        "display/palette.py", "THEME_COLORS",
    ),
}
_ORDER = (
    ("profile", "timeout", False), ("profile", "timeout", True),
    ("profile", "queue", True), ("profile", "queue", False),
    ("profile", "label", False), ("profile", "label", True),
    ("catalog", "size", False), ("catalog", "size", True),
    ("catalog", "color", True), ("catalog", "color", False),
    ("metadata", "size", False), ("metadata", "size", True),
    ("temperature", "timeout", False), ("temperature", "timeout", True),
    ("temperature", "queue", True), ("temperature", "queue", False),
    ("temperature", "label", False), ("temperature", "label", True),
    ("scope", "timeout", False), ("scope", "timeout", True),
)


def _equivalent_files(actual: dict[str, bytes], initial: dict[str, bytes],
                      destination: str, content: bytes) -> bool:
    if actual.keys() != initial.keys() | {destination}:
        return False
    if any(actual[name] != text for name, text in initial.items()):
        return False
    try:
        return ast.dump(ast.parse(actual[destination])) == ast.dump(ast.parse(content))
    except SyntaxError:
        return False


@pytest.mark.parametrize(
    "factor,case,treatment", _ORDER,
    ids=[f"{factor}-{case}-{'candidate' if enabled else 'base'}"
         for factor, case, enabled in _ORDER],
)
def test_creation(
    factor, case, treatment, runtime, workspace_dir, monkeypatch,
    auto_approve_permissions, sandbox_disabled,
):
    engine, agent, script, _seen = runtime
    prompt, source, function, value, decoy, destination, constant = _CASES[case]
    content = f"{constant} = {value}\n"
    initial = {
        source: f"def {function}():\n    return {value}\n",
        f"archive/{Path(source).name}": f"def {function}():\n    return {decoy}\n",
    }
    root = workspace_dir / "fixture"
    root.mkdir()
    monkeypatch.chdir(root)
    agent.workspace_path = str(root)
    get_context_for_agent(agent.agent_id).workspace_path = str(root)
    for name, text in initial.items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)

    agent.tools = [CodeSearchTool(), ReadFileTool(), EditFileTool(), CreateFileTool()]
    grant = [tool for tool in agent.tools
             if not (factor == "catalog" and treatment and tool.name == "edit_file")]
    metadata = not (factor == "metadata" and treatment)
    temperature = 0.7 if factor == "temperature" and treatment else 0.2
    if factor == "scope" and treatment:
        monkeypatch.setattr(settings, "FILE_OPERATIONS_PERMISSION", "allowed_paths")
        monkeypatch.setattr(settings, "ALLOWED_FILE_PATHS", [str(root / destination)])
    profiles = EffectivePromptConfiguration({"develop": {_CAPABILITY: PromptProfile()}})
    fragment = _BASE_FRAGMENT
    if factor == "profile" and treatment:
        body, close = fragment.rsplit("\n", 1)
        fragment = "\n".join((body, _CREATION_RULE, close))
        monkeypatch.setattr(optional_capabilities, "OPTIONAL_CAPABILITIES",
                            ((_CAPABILITY, fragment),))

    live = os.environ.get("INFINIDEV_REASONING_EVAL_LIVE") == "1"
    if live:
        for name, value in {
            "LLM_PROVIDER": "openai_compatible", "LLM_MODEL": _MODEL,
            "LLM_BASE_URL": _ENDPOINT, "LLM_API_KEY": "local",
            "LLM_NUM_RETRIES": 0, "LLM_TEMPERATURE": temperature,
        }.items():
            monkeypatch.setattr(settings, name, value)
        monkeypatch.setattr(turn_runtime, "get_litellm_params", _REAL_PARAMS)
        monkeypatch.setattr(turn_runtime, "get_model_capabilities", _REAL_CAPABILITIES)
        monkeypatch.setattr(turn_runtime, "_get_model_max_context", lambda _: 262144)
        monkeypatch.setattr(turn_runtime, "call_llm", _REAL_CALL)
    else:
        def no_connection(*args, **kwargs):
            raise AssertionError("Offline creation probe")
        monkeypatch.setattr(socket.socket, "connect", no_connection)

        def call(tool, **arguments):
            if metadata:
                arguments = {
                    "metadata": {"reason": "Exercise file creation",
                                 "expected_result": "The requested fixture exists"},
                    "input": arguments,
                }
            return reply("", tc(tool, **arguments))

        scope_control = []
        if factor == "scope":
            scope_control = [
                call("edit_file", file_path=source, old_string=value, new_string=decoy),
                call("edit_file", file_path=source, old_string=decoy, new_string=value),
            ]
        script(*scope_control,
               call("read_file", file_path=source),
               call("create_file", file_path=destination, content=content),
               call("read_file", file_path=destination),
               call("finish_turn", message="Fixture verified."))

    offered = []
    dispatches = []
    protected_writes = []
    permission_denials = []
    verified_destination = False
    provider = turn_runtime.call_llm
    dispatch = turn_runtime.execute_tool_call
    check_permission = permissions.check_file_permission

    def observe_permission(action, path):
        denial = check_permission(action, path)
        if denial is not None:
            relative = Path(path).resolve().relative_to(root.resolve()).as_posix()
            permission_denials.append({"action": action, "path": relative})
        return denial

    def observe_provider(params, messages, tools=None, **kwargs):
        if live:
            assert params["temperature"] == temperature
        offered.append([tool["function"]["name"] for tool in tools or []])
        return provider(params, messages, tools, **kwargs)

    def observe_dispatch(dispatcher, name, args, **kwargs):
        nonlocal verified_destination
        target = None
        previous = None
        if name in {"code_search", "read_file", "edit_file", "create_file"}:
            path = (root / (args.get("file_path") or ".")).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Tool path escaped the fixture")
            if ".infinidev" in path.relative_to(root.resolve()).parts:
                raise ValueError("Tool path targets the benchmark's runtime state")
            target = path.relative_to(root).as_posix()
            if name in {"edit_file", "create_file"}:
                previous = path.read_bytes() if path.is_file() else None
        result = dispatch(dispatcher, name, args, **kwargs)
        if name in {"edit_file", "create_file"}:
            current = path.read_bytes() if path.is_file() else None
            if current != previous:
                if target == destination:
                    verified_destination = False
                else:
                    protected_writes.append(target)
        elif name == "read_file" and target == destination and path.is_file():
            # The fixture files are small and UTF-8. Match the full numbered
            # read output so hints, errors and incomplete reads never count.
            try:
                lines = path.read_bytes().decode("utf-8").splitlines()
            except UnicodeDecodeError:
                lines = []
            numbered = "\n".join(f"{n:>6}\t{line.rstrip()}" for n, line in enumerate(lines, 1))
            verified_destination |= bool(numbered) and result == numbered
        dispatches.append({"name": name, "output_sha256": hashlib.sha256(result.encode()).hexdigest()})
        return result

    monkeypatch.setattr(turn_runtime, "call_llm", observe_provider)
    monkeypatch.setattr(turn_runtime, "execute_tool_call", observe_dispatch)
    monkeypatch.setattr(permissions, "check_file_permission", observe_permission)
    started = time.monotonic()
    result = engine.run_turn(agent, prompt, task_tools=grant, max_iterations=9,
                             max_total_tool_calls=8, tool_call_metadata=metadata,
                             prompt_configuration=profiles)
    actual = {str(path.relative_to(root)): path.read_bytes()
              for path in root.rglob("*")
              if path.is_file() and ".infinidev" not in path.relative_to(root).parts}
    starting_bytes = {name: text.encode("utf-8") for name, text in initial.items()}
    expected_bytes = content.encode("utf-8")
    match = _equivalent_files(actual, starting_bytes, destination, expected_bytes)
    exact = actual == {**starting_bytes, destination: expected_bytes}
    report = {
        "factor": factor, "case": case, "treatment": treatment, "live": live,
        "task_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "profile_sha256": hashlib.sha256(fragment.encode()).hexdigest(),
        "metadata": metadata, "temperature": temperature if live else None,
        "offered": offered, "dispatches": dispatches,
        "semantic_match": match, "exact_files": exact, "status": result.status,
        "closure": engine._last_turn_receipt["closure"], "metrics": result.metrics,
        "protected_writes": protected_writes,
        "permission_denials": permission_denials,
        "verified_destination": verified_destination,
        "complete_in_scope": match and verified_destination and not protected_writes
        and engine._last_turn_receipt["closure"] == "finish_turn",
        "seconds": round(time.monotonic() - started, 2),
    }
    print("CREATION_PROBE " + json.dumps(report, sort_keys=True))
    assert result.metrics["tool_calls"] <= 8
    if not live:
        assert match and exact and result.status == "answered"
        expected_calls = ["read_file", "create_file", "read_file", "finish_turn"]
        if factor == "scope":
            expected_calls[:0] = ["edit_file", "edit_file"]
        assert [entry["name"] for entry in dispatches] == expected_calls
        assert protected_writes == ([source, source] if factor == "scope" and not treatment else [])
        assert [row["path"] for row in permission_denials] == (
            [source, source] if factor == "scope" and treatment else []
        )
        assert report["complete_in_scope"] == (factor != "scope" or treatment)
        assert all(set(catalog) == {tool.name for tool in grant} | {"finish_turn", "update_plan"}
                   for catalog in offered)
