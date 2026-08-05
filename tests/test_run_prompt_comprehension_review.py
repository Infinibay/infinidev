from __future__ import annotations

import json
from pathlib import Path

from bench.run_prompt_comprehension_review import extract_review, qwen_request, run_reviews


def _packet() -> dict[str, object]:
    return {
        "dataset_sha256": "hash",
        "families": [
            {
                "family_id": "family",
                "category": "planning",
                "variants": [{"case_id": "case", "request": "Prepare a plan."}],
            }
        ],
    }


def _review() -> dict[str, object]:
    fields = {
        "objective": "Prepare a plan.",
        "deliverables": ["Plan"],
        "constraints": [],
        "user_owned_decisions": [],
        "authorized_actions": ["Plan"],
        "unauthorized_actions": ["Implement"],
        "verification": ["Ground it in evidence"],
        "ambiguities": [],
        "stop_conditions": ["Stop after the plan"],
        "conflicts": [],
        "priority_resolution": "",
        "interpretation_risks": [],
    }
    return {
        "family_id": "family",
        "reviewer": "reviewer",
        "dataset_sha256": "hash",
        "verdict": "accept",
        "rationale": "The family isolates one meaning.",
        "checks": {
            "equivalents_preserve_meaning": True,
            "contrast_changes_only_intended_variable": True,
            "wording_is_natural": True,
            "requests_are_self_contained": True,
            "authorization_is_unambiguous": True,
            "no_split_leakage_detected": True,
        },
        "reconstructions": {"case": fields},
        "diversity_concern": "",
    }


def test_extract_review_ignores_reasoning_wrappers() -> None:
    value = _review()
    assert extract_review(f"<think>private</think>\n```json\n{json.dumps(value)}\n```") == value


def test_runner_persists_raw_and_parsed_and_resumes(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    calls = 0

    def request_fn(prompt: str) -> tuple[str, dict[str, object]]:
        nonlocal calls
        calls += 1
        assert "authored interpretation keys" in prompt
        review = _review()
        review["reconstructions"] = {"variant_1": next(iter(review["reconstructions"].values()))}
        return json.dumps(review), {"response_model": "MiniMax-M3"}

    first = run_reviews(
        _packet(),
        ledger=ledger,
        reviews=reviews,
        model="MiniMax-M3",
        reviewer="reviewer",
        delay_seconds=0,
        request_fn=request_fn,
    )
    second = run_reviews(
        _packet(),
        ledger=ledger,
        reviews=reviews,
        model="MiniMax-M3",
        reviewer="reviewer",
        delay_seconds=0,
        request_fn=request_fn,
    )
    assert first == {"successes": 1, "failures": 0, "skipped": 0}
    assert second == {"successes": 0, "failures": 0, "skipped": 1}
    assert calls == 1
    row = json.loads(ledger.read_text())
    assert row["response_text"]
    assert row["review"]["family_id"] == "family"
    assert len(reviews.read_text().splitlines()) == 1


def test_runner_preserves_unparseable_response_on_failure(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"

    result = run_reviews(
        _packet(),
        ledger=ledger,
        reviews=tmp_path / "reviews.jsonl",
        model="MiniMax-M3",
        reviewer="reviewer",
        delay_seconds=0,
        request_fn=lambda _: ("not json", {"response_id": "response"}),
    )

    assert result["failures"] == 1
    row = json.loads(ledger.read_text())
    assert row["response_text"] == "not json"
    assert row["provider"]["response_id"] == "response"


def test_runner_persists_provider_runtime_failure(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"

    def fail(_: str) -> tuple[str, dict[str, object]]:
        raise RuntimeError("stream ended early")

    result = run_reviews(
        _packet(),
        ledger=ledger,
        reviews=tmp_path / "reviews.jsonl",
        model="gpt-5.6-sol",
        reviewer="reviewer",
        delay_seconds=0,
        request_fn=fail,
    )

    assert result["failures"] == 1
    row = json.loads(ledger.read_text())
    assert row["status"] == "failure"
    assert row["error"] == "stream ended early"


def test_qwen_request_is_stateless_and_user_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "response",
                    "model": "qwen3.8-max-preview",
                    "choices": [
                        {"message": {"content": "result"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }
            ).encode()

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    text, metadata = qwen_request(
        endpoint="https://approved.example/v1/chat/completions",
        api_key="secret",
        model="qwen3.8-max-preview",
        prompt="one isolated prompt",
        timeout=30.0,
        max_completion_tokens=8000,
    )

    request = captured["request"]
    body = json.loads(request.data)
    assert body["messages"] == [{"role": "user", "content": "one isolated prompt"}]
    assert body["model"] == "qwen3.8-max-preview"
    assert text == "result"
    assert metadata["transport"] == "qwen_openai_compatible"
