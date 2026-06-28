"""StepVerification — a typed, executable success check for a plan step.

The free-text ``expected_output`` string on a step is self-attested: the
model renders it as a prompt anchor and then grades its own work. This
type replaces that honor system with an *observation channel* (``kind``)
plus a concrete ``spec`` the engine can run to decide PASS/FAIL
deterministically — an exit code, a substring, a grep hit — instead of
trusting the model's claim.

It is authored by the analyst planner (``emit_plan``), before any code is
written, and carried frozen on user-approved steps so the developer cannot
relax its own bar mid-run. The executor lives in
``engine.analysis.objective_verifier.ObjectiveVerifier``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

# Observation channels, ordered roughly by how deterministic they are:
#   command       → run ``spec`` as a shell command; PASS when exit code == 0
#   test_id       → ``spec`` is a pytest node id; run it, PASS on exit 0
#   file_contains → ``spec`` is a file path, ``observable`` the required
#                   substring; PASS when the substring is present in the file
#   symbol_exists → ``spec`` is a name/snippet; PASS when grep finds it in the
#                   workspace (cheap, language-agnostic, no code-intel coupling)
#   llm_judge     → ``spec`` is a rubric/acceptance statement an INDEPENDENT
#                   adversarial verifier judges against the diff at task end
#                   (the only non-deterministic kind; for soft objectives that
#                   no command can decide). Runs once post-loop, never per step.
#   none          → no executable check; falls back to self-attestation
VerificationKind = Literal[
    "none", "command", "test_id", "file_contains", "symbol_exists", "llm_judge"
]

# The kinds whose verdict is an exit code / substring / grep hit — runnable
# cheaply and synchronously in the per-step gate. ``llm_judge`` is excluded:
# it costs an LLM call and is deferred to the task-end re-verification.
_DETERMINISTIC_KINDS = frozenset(
    {"command", "test_id", "file_contains", "symbol_exists"}
)


class StepVerification(BaseModel):
    """A machine-checkable success condition for one plan step."""

    kind: VerificationKind = "none"
    # The locus to run/inspect. Meaning depends on ``kind`` (see above).
    spec: str = ""
    # The proof fragment. For ``file_contains`` it is the required substring
    # (mandatory). For ``command``/``test_id`` it is an optional stdout
    # fragment that must also appear for a PASS (empty = exit code alone
    # decides). Ignored for ``none``.
    observable: str = ""

    @property
    def is_executable(self) -> bool:
        """True when this check names a real, runnable observation."""
        if self.kind == "none":
            return False
        if not self.spec.strip():
            return False
        # file_contains is meaningless without something to look for.
        if self.kind == "file_contains" and not self.observable.strip():
            return False
        return True

    @property
    def is_deterministic(self) -> bool:
        """True for checks decided by an exit code / substring / grep — the
        ones cheap enough to run in the per-step gate. ``llm_judge`` is not."""
        return self.is_executable and self.kind in _DETERMINISTIC_KINDS

    @classmethod
    def from_loose(cls, obj: Any) -> "StepVerification | None":
        """Build a StepVerification from a dict / model / None.

        Tolerant of the shapes ``emit_plan`` args arrive in (native
        function-call dict, ``model_dump`` of a pydantic arg model, or the
        flat ``verify_kind``/``verify_spec``/``verify_observable`` fields the
        authoring schema actually exposes). Returns None when there is no
        usable, executable check so callers can leave the step on the
        self-attestation fallback.
        """
        if obj is None:
            return None
        if isinstance(obj, StepVerification):
            return obj if obj.is_executable else None
        if isinstance(obj, BaseModel):
            obj = obj.model_dump()
        if not isinstance(obj, dict):
            return None
        # Accept both the nested shape ({kind, spec, observable}) and the
        # flat authoring shape ({verify_kind, verify_spec, verify_observable}).
        kind = obj.get("kind", obj.get("verify_kind", "none"))
        spec = obj.get("spec", obj.get("verify_spec", ""))
        observable = obj.get("observable", obj.get("verify_observable", ""))
        try:
            check = cls(
                kind=(kind or "none"),
                spec=(spec or "").strip(),
                observable=(observable or "").strip(),
            )
        except Exception:
            return None
        return check if check.is_executable else None
