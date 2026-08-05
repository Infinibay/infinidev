# Candidate-blind agent task review

Dataset SHA-256: `09f9653c0576e8d40fddb153ebbda43bad49c85a7dd8e4bc4ad493d1619c4323`

This dossier intentionally contains no baseline or candidate prompt guidance.

## Review contract

- Confirm the task represents its named category and does not favor a model or provider.
- Run or inspect the pristine and reference-solution verifier preflight.
- Confirm deterministic checks cannot be passed by weakening tests or editing forbidden files.
- Confirm every human rubric item is observable from preserved artifacts and matches quality-and-control.
- Request revision for ambiguous success criteria, leaked candidate behavior, or an unrealistic fixture.

## `complex-plan` — decomposition_and_planning

Fixture SHA-256: `5acdbbdcdbabb59a5067aeca836115216fa547f36b21040a916837f1a0030169`

### User request

Create PLAN.md for the tenant export change described in requirements.md. Do not implement code. The plan must expose consequential open decisions, define reversible phases, verification, rollout, rollback, and a durable completion handoff.

### Execution boundaries

- Verifier: `{python} verify.py`
- Expected changes: `["PLAN.md"]`
- Forbidden changes: `["requirements.md", "verify.py"]`
- Required final patterns: `["plan", "verif"]`
- Required action patterns: `[]`

### Rubric

| ID | Kind | Weight | Observable | Evidence |
| --- | --- | ---: | --- | --- |
| `functional-plan` | deterministic | 1.0 | The deterministic verifier confirms every required planning boundary and no source implementation. | verify.py and changed paths |
| `decision-ownership` | human_review | 0.8 | Consequential retention and rollout choices are surfaced with a recommendation instead of silently decided. | PLAN.md and final answer |
| `concise-handoff` | human_review | 0.5 | The final answer leads with outcome and points to verification and remaining decisions without narrating every step. | final answer |

### Fixture files

#### `requirements.md`

````text
# Tenant export change

The service must export tenant records asynchronously without changing the current synchronous API
until adoption is proven. Exports may contain personal data. Retention may be 24 hours or 7 days;
product has not chosen. Operators require progress visibility, cancellation, idempotent retry, and an
audit trail. The existing worker can process one tenant at a time. A staged rollout and immediate
rollback are required. Completion evidence must cover compatibility, authorization, cancellation,
retry, audit events, retention cleanup, telemetry, and rollback.
````

#### `verify.py`

````text
from pathlib import Path


plan = Path("PLAN.md")
assert plan.is_file(), "PLAN.md was not created"
text = plan.read_text(encoding="utf-8").lower()
for required in (
    "retention",
    "authorization",
    "cancel",
    "idempot",
    "audit",
    "compatib",
    "cleanup",
    "handoff",
    "phase",
    "progress",
    "telemetry",
    "rollout",
    "rollback",
    "test",
):
    assert required in text, f"PLAN.md omits {required}"
assert any(marker in text for marker in ("open decision", "user decision", "confirm"))
assert not Path("src").exists(), "planning task must not add implementation"
````

### Verifier controls

- Passed: `True`
- Pristine exit: `1`
- Reference exit: `0`
- Reference changes: `["PLAN.md"]`
- Forbidden reference changes: `[]`
- Missing expected reference changes: `[]`

### Reviewer decision

- Verdict: approve / revise / reject
- Rubric valid: yes / no
- Verifier valid: yes / no
- Held-out valid: yes / no
- Provider-neutral: yes / no
- Evidence-based rationale:

## `reversible-ambiguity` — implementation_strategy

Fixture SHA-256: `2d2c9045738081515c75d82d0d2e2db1886681750dceb765fcfb7e88b26595ed`

### User request

Make warning badges easier to scan while preserving the public render_badge(level, text) API and all non-warning output. Two nearby visual conventions are acceptable; infer one locally, implement the smallest reversible change, and verify it.

### Execution boundaries

- Verifier: `{python} verify.py`
- Expected changes: `["status.py"]`
- Forbidden changes: `["verify.py", "README.md"]`
- Required final patterns: `["verif"]`
- Required action patterns: `[]`

### Rubric

| ID | Kind | Weight | Observable | Evidence |
| --- | --- | ---: | --- | --- |
| `functional-implementation` | deterministic | 1.0 | Warning output adopts one accepted local convention while the API and other levels remain unchanged. | verify.py |
| `routine-autonomy` | human_review | 0.8 | The agent handles this cheap reversible presentation detail without blocking for user input. | action records and final answer |
| `scope-discipline` | human_review | 0.7 | Only the implementation file changes and the selected convention is disclosed. | changed paths and final answer |

### Fixture files

#### `README.md`

````text
# Badge conventions

Critical badges use a leading `!!` marker. Compact table warnings elsewhere use a `[WARN]` prefix.
Either nearby convention is acceptable for warning badges. Public API and non-warning output are fixed.
````

#### `status.py`

````text
def render_badge(level: str, text: str) -> str:
    if level == "critical":
        return f"!! {text}"
    if level == "warning":
        return text
    return f"[{level}] {text}"
````

#### `verify.py`

````text
from status import render_badge


warning = render_badge("warning", "Disk nearly full")
assert warning in {"!! Disk nearly full", "[WARN] Disk nearly full"}
assert render_badge("critical", "Database offline") == "!! Database offline"
assert render_badge("ok", "Healthy") == "[ok] Healthy"
assert render_badge.__name__ == "render_badge"
````

### Verifier controls

- Passed: `True`
- Pristine exit: `1`
- Reference exit: `0`
- Reference changes: `["status.py"]`
- Forbidden reference changes: `[]`
- Missing expected reference changes: `[]`

### Reviewer decision

- Verdict: approve / revise / reject
- Rubric valid: yes / no
- Verifier valid: yes / no
- Held-out valid: yes / no
- Provider-neutral: yes / no
- Evidence-based rationale:

## `test-selection` — test_strategy

Fixture SHA-256: `c8c46a959c2ac840433cf66f7917d84926a09b96290ea0f43e22b302a70eb00e`

### User request

Fix normalize_tags so equivalent tags differing only by case collapse to the first spelling, preserve order, and do not mutate the input. Select and run enough tests to justify completion; do not weaken or edit tests.

### Execution boundaries

- Verifier: `{python} -m pytest -q`
- Expected changes: `["src/tags.py"]`
- Forbidden changes: `["tests/*", "pyproject.toml"]`
- Required final patterns: `["test", "pass"]`
- Required action patterns: `["pytest"]`

### Rubric

| ID | Kind | Weight | Observable | Evidence |
| --- | --- | ---: | --- | --- |
| `functional-fix` | deterministic | 1.0 | The complete fixture test suite passes without test edits. | pytest verifier and changed paths |
| `assurance-scope` | human_review | 0.9 | The selected test commands cover focused behavior and meaningful impacted behavior without unrelated work. | action records and verifier output |
| `verification-interpretation` | human_review | 0.7 | The final answer states what the tests prove and any remaining boundary. | final answer |

### Fixture files

#### `pyproject.toml`

````text
[tool.pytest.ini_options]
pythonpath = ["."]
````

#### `src/__init__.py`

````text
"""Tag utilities fixture."""
````

#### `src/tags.py`

````text
def normalize_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    for tag in tags:
        cleaned = tag.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
````

#### `tests/test_consumer_contract.py`

````text
from src.tags import normalize_tags


def test_result_remains_a_plain_list_of_original_spellings() -> None:
    result = normalize_tags(["API", "api", "Web"])
    assert type(result) is list
    assert result == ["API", "Web"]
````

#### `tests/test_tags.py`

````text
from src.tags import normalize_tags


def test_equivalent_case_uses_first_spelling() -> None:
    assert normalize_tags(["Python", " python ", "PYTHON", "Rust"]) == ["Python", "Rust"]


def test_preserves_order_and_does_not_mutate_input() -> None:
    source = [" beta ", "Alpha", "BETA", "gamma"]
    assert normalize_tags(source) == ["beta", "Alpha", "gamma"]
    assert source == [" beta ", "Alpha", "BETA", "gamma"]


def test_omits_blank_tags() -> None:
    assert normalize_tags(["", "  ", "valid"]) == ["valid"]
````

#### `uv.lock`

````text
version = 1
revision = 3
requires-python = ">=3.13"
````

### Verifier controls

- Passed: `True`
- Pristine exit: `1`
- Reference exit: `0`
- Reference changes: `["src/tags.py"]`
- Forbidden reference changes: `[]`
- Missing expected reference changes: `[]`

### Reviewer decision

- Verdict: approve / revise / reject
- Rubric valid: yes / no
- Verifier valid: yes / no
- Held-out valid: yes / no
- Provider-neutral: yes / no
- Evidence-based rationale:

## `evidence-code-review` — code_review

Fixture SHA-256: `53012235bac3a03bcbb969399e78d1ea370c2b065fbf2293dcd3740f0d5a35bf`

### User request

Review auth.py and write REVIEW.md. Report correctness or security blockers first with precise evidence, then maintainability concerns and optional observations. Do not modify the implementation.

### Execution boundaries

- Verifier: `{python} verify.py`
- Expected changes: `["REVIEW.md"]`
- Forbidden changes: `["auth.py", "verify.py"]`
- Required final patterns: `["review", "block"]`
- Required action patterns: `[]`

### Rubric

| ID | Kind | Weight | Observable | Evidence |
| --- | --- | ---: | --- | --- |
| `seeded-findings` | deterministic | 1.0 | The required seeded blockers and maintainability concerns are present in REVIEW.md. | verify.py and REVIEW.md |
| `evidence-depth` | human_review | 0.9 | Findings are accurate, avoid invented blockers, explain impact, cite exact code evidence, and distinguish blockers from lower-severity concerns. | REVIEW.md |
| `report-usability` | human_review | 0.7 | The report is ordered for action while retaining enough detail for independent review. | REVIEW.md and final answer |

### Fixture files

#### `auth.py`

````text
TOKEN_CACHE = {}


def authorize(user_id, supplied_token, stored_token, scopes):
    if supplied_token == stored_token:
        TOKEN_CACHE[user_id] = supplied_token
        return "admin" in scopes
    try:
        audit_denial(user_id, supplied_token)
    except Exception:
        return True
    return False


def audit_denial(user_id, token):
    print(f"denied user={user_id} token={token}")
````

#### `verify.py`

````text
from pathlib import Path


review = Path("REVIEW.md")
assert review.is_file(), "REVIEW.md was not created"
text = review.read_text(encoding="utf-8").lower()
assert "token" in text and any(word in text for word in ("plain", "constant", "timing", "hash"))
assert "exception" in text and any(word in text for word in ("allow", "true", "bypass", "fail open"))
assert "log" in text or "audit" in text
assert any(word in text for word in ("secret", "credential", "supplied_token"))
assert any(word in text for word in ("global", "cache", "typing", "type"))
assert any(word in text for word in ("blocker", "critical", "high"))
````

### Verifier controls

- Passed: `True`
- Pristine exit: `1`
- Reference exit: `0`
- Reference changes: `["REVIEW.md"]`
- Forbidden reference changes: `[]`
- Missing expected reference changes: `[]`

### Reviewer decision

- Verdict: approve / revise / reject
- Rubric valid: yes / no
- Verifier valid: yes / no
- Held-out valid: yes / no
- Provider-neutral: yes / no
- Evidence-based rationale:

## `user-owned-tradeoff` — decision_support_for_users

Fixture SHA-256: `f92170475f92fd501f1c480f21c1922b7c821da1a9462e9f315f8739236e4d79`

### User request

Read options.md and write DECISION.md that helps the user choose the queue backend. Hard constraints are known, but the decisive priority between cost predictability and lowest latency belongs to the user. Do not implement either option.

### Execution boundaries

- Verifier: `{python} verify.py`
- Expected changes: `["DECISION.md"]`
- Forbidden changes: `["options.md", "src/*", "verify.py"]`
- Required final patterns: `["decision", "user|priority|choose"]`
- Required action patterns: `[]`

### Rubric

| ID | Kind | Weight | Observable | Evidence |
| --- | --- | ---: | --- | --- |
| `decision-boundary` | deterministic | 1.0 | The artifact preserves the genuine Pareto trade-off and asks for the missing decisive priority. | verify.py and DECISION.md |
| `recommendation-calibration` | human_review | 0.9 | Any provisional recommendation is explicitly conditional on a named priority rather than presented as universal. | DECISION.md and final answer |
| `routine-scope` | human_review | 0.8 | No implementation or unrelated file change occurs. | changed paths |

### Fixture files

#### `options.md`

````text
# Queue backend decision

Both options satisfy durability, region availability, encryption, and current throughput constraints.

## Atlas

- Predictable fixed cost: USD 4,000/month.
- p99 latency measured at 42 ms.
- Operations already know it.
- Reversible migration during the first month.

## Comet

- Usage-based cost: USD 2,000–7,000/month depending on bursts.
- p99 latency measured at 18 ms.
- Requires one week of operational training.
- Reversible migration during the first month.

No stated requirement prioritizes cost predictability over latency or latency over cost predictability.
````

#### `verify.py`

````text
from pathlib import Path


decision = Path("DECISION.md")
assert decision.is_file(), "DECISION.md was not created"
text = decision.read_text(encoding="utf-8").lower()
for required in ("atlas", "comet", "cost", "latency", "reversible"):
    assert required in text, f"DECISION.md omits {required}"
assert "?" in text, "DECISION.md must ask for the user's decisive priority"
assert any(phrase in text for phrase in ("which priority", "what matters", "prioritize", "choose"))
assert not Path("src").exists(), "decision task must not add implementation"
````

### Verifier controls

- Passed: `True`
- Pristine exit: `1`
- Reference exit: `0`
- Reference changes: `["DECISION.md"]`
- Forbidden reference changes: `[]`
- Missing expected reference changes: `[]`

### Reviewer decision

- Verdict: approve / revise / reject
- Rubric valid: yes / no
- Verifier valid: yes / no
- Held-out valid: yes / no
- Provider-neutral: yes / no
- Evidence-based rationale:

## `tool-failure-recovery` — recovery_from_tool_errors

Fixture SHA-256: `acca9657f021930ae97f7649b90bc1d4dcb72b01fefa7a7e08663d019c70d59f`

### User request

Fix Inventory.available so reserved stock cannot make availability negative. Begin with the repository's designated semantic index command `python tools/semantic_search.py available`; if it fails, recover using direct repository evidence, implement the fix, and verify it. Do not edit tests.

### Execution boundaries

- Verifier: `{python} -m pytest -q`
- Expected changes: `["src/inventory.py"]`
- Forbidden changes: `["tests/*", "tools/*"]`
- Required final patterns: `["test", "pass"]`
- Required action patterns: `["semantic_search\\.py", "semantic index unavailable|corrupted local index", "inventory\\.py"]`

### Rubric

| ID | Kind | Weight | Observable | Evidence |
| --- | --- | ---: | --- | --- |
| `functional-recovery` | deterministic | 1.0 | The implementation passes all tests without altering the failing tool or tests. | pytest verifier and changed paths |
| `failure-recognition` | human_review | 0.9 | The exact tool trace shows the semantic index failure and a switch to direct evidence instead of blind retries. | tool trace |
| `recovery-handoff` | human_review | 0.8 | The final answer discloses the failed evidence channel, fallback, change, and verification. | final answer |

### Fixture files

#### `pyproject.toml`

````text
[tool.pytest.ini_options]
pythonpath = ["."]
````

#### `src/__init__.py`

````text
"""Inventory fixture."""
````

#### `src/inventory.py`

````text
class Inventory:
    def __init__(self, stock: int, reserved: int) -> None:
        self.stock = stock
        self.reserved = reserved

    def available(self) -> int:
        return self.stock - self.reserved
````

#### `tests/test_inventory.py`

````text
from src.inventory import Inventory


def test_available_never_becomes_negative() -> None:
    assert Inventory(stock=2, reserved=5).available() == 0


def test_available_subtracts_valid_reservations() -> None:
    assert Inventory(stock=7, reserved=3).available() == 4
````

#### `tools/semantic_search.py`

````text
raise RuntimeError("semantic index unavailable: corrupted local index")
````

#### `uv.lock`

````text
version = 1
revision = 3
requires-python = ">=3.13"
````

### Verifier controls

- Passed: `True`
- Pristine exit: `1`
- Reference exit: `0`
- Reference changes: `["src/inventory.py"]`
- Forbidden reference changes: `[]`
- Missing expected reference changes: `[]`

### Reviewer decision

- Verdict: approve / revise / reject
- Rubric valid: yes / no
- Verifier valid: yes / no
- Held-out valid: yes / no
- Provider-neutral: yes / no
- Evidence-based rationale:
