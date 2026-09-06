"""Packaged starter files for the user-level prompt-profile catalog."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from threading import Lock
from typing import Iterator

from infinidev.prompts._filesystem import fsync_directory
from infinidev.prompts.optional_capabilities import OPTIONAL_CAPABILITIES


USER_OVERRIDES_FILE = "99-user-overrides.json"
_STARTER_PROFILE_LOCK = Lock()


def _starter_document(
    sections: dict[str, tuple[str, ...]], *, enabled_by_default: bool = True,
) -> str:
    """Render explicit default settings for a group of registered fragments."""
    document = {
        phase: {
            name: {"enabled_by_default": enabled_by_default}
            for name in names
        }
        for phase, names in sections.items()
    }
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


STARTER_PROMPT_PROFILES: tuple[tuple[str, str], ...] = (
    (
        "10-development.json",
        _starter_document(
            {
                "develop": (
                    "loop.identity",
                    "loop.protocol",
                    "loop.behavior_guidelines",
                    "loop.technology_guidance",
                    "loop.project_instructions",
                    "loop.critic_guidance",
                    "loop.session_context",
                    "iteration.smart_summary",
                    "iteration.project_knowledge",
                    "iteration.context_corpus",
                    "iteration.context_rank",
                    "iteration.workspace",
                    "iteration.background_completions",
                    "iteration.background_tasks",
                    "iteration.reactive_guidance",
                    "iteration.opened_files",
                    "iteration.session_notes",
                    "iteration.working_notes",
                    "iteration.note_nudge",
                    "iteration.previous_actions",
                    "iteration.anti_patterns",
                    "iteration.behavior_summary",
                    "iteration.next_actions",
                    "iteration.context_budget",
                ),
            }
        ),
    ),
    (
        "20-planning.json",
        _starter_document(
            {
                "plan": (
                    "task_planner.identity",
                    "task_planner.methodology",
                    "task_planner.planning_vocabulary",
                    "task_planner.handoff_guidance",
                    "task_planner.decomposition_guidance",
                    "task_planner.verification_guidance",
                    "task_planner.examples",
                    "stage_planner.identity",
                    "stage_planner.methodology",
                    "stage_planner.planning_vocabulary",
                    "stage_planner.authority_guidance",
                    "stage_planner.horizon_guidance",
                    "stage_planner.decision_guidance",
                    "stage_planner.decomposition_guidance",
                    "stage_planner.examples",
                    "phase.bug.plan",
                    "phase.bug.plan_identity",
                    "phase.feature.plan",
                    "phase.feature.plan_identity",
                    "phase.refactor.plan",
                    "phase.refactor.plan_identity",
                    "phase.other.plan",
                    "phase.other.plan_identity",
                    "phase.sysadmin.plan",
                    "phase.sysadmin.plan_identity",
                ),
            }
        ),
    ),
    (
        "30-review.json",
        _starter_document(
            {
                "review": (
                    "reviewer.identity",
                    "reviewer.input_guidance",
                    "reviewer.authority_guidance",
                    "reviewer.evaluation_guidance",
                    "reviewer.severity_guidance",
                    "extractor.identity",
                    "judge.identity",
                    "judge.input_guidance",
                    "judge.authority_guidance",
                    "judge.evaluation_guidance",
                    "judge.severity_guidance",
                    "evidence.identity",
                    "evidence.evaluation_guidance",
                    "adversarial.identity",
                    "adversarial.evaluation_guidance",
                ),
            }
        ),
    ),
    (
        "40-collaboration.json",
        _starter_document(
            {
                "chat": (
                    "chat.identity",
                    "chat.language_guidance",
                    "chat.council_guidance",
                    "chat.followup_guidance",
                    "chat.project_instructions",
                    "chat.model_guidance",
                ),
                "council": (
                    "council.seed_identity",
                    "council.member_identity",
                    "council.judge_identity",
                    "council.synthesis_identity",
                    "council.language_guidance",
                    "council.persona_palette",
                ),
                "gather": (
                    "gather.identity_guidance",
                    "gather.classifier_guidance",
                    "gather.synthesis_guidance",
                    "gather.question_guidance",
                ),
                "summarize": ("summary.step_guidance",),
            }
        ),
    ),
    (
        "45-team.json",
        _starter_document({"team": ("team.orchestrator_guidance", "team.worker_guidance")}),
    ),
    (
        "50-investigation.json",
        _starter_document(
            {
                "investigate": (
                    "phase.bug.investigate",
                    "phase.bug.investigate_identity",
                    "phase.feature.investigate",
                    "phase.feature.investigate_identity",
                    "phase.refactor.investigate",
                    "phase.refactor.investigate_identity",
                    "phase.other.investigate",
                    "phase.other.investigate_identity",
                    "phase.sysadmin.investigate",
                    "phase.sysadmin.investigate_identity",
                ),
            }
        ),
    ),
    (
        "60-execution.json",
        _starter_document(
            {
                "execute": (
                    "phase.bug.execute",
                    "phase.bug.execute_identity",
                    "phase.feature.execute",
                    "phase.feature.execute_identity",
                    "phase.refactor.execute",
                    "phase.refactor.execute_identity",
                    "phase.other.execute",
                    "phase.other.execute_identity",
                    "phase.sysadmin.execute",
                    "phase.sysadmin.execute_identity",
                ),
            }
        ),
    ),
    (
        "90-optional-capabilities.json",
        _starter_document(
            {
                "develop": tuple(name for name, _body in OPTIONAL_CAPABILITIES),
            },
            enabled_by_default=False,
        ),
    ),
)


@contextmanager
def _starter_catalog_lock(catalog_path: Path) -> Iterator[None]:
    """Serialize starter publication across threads and Infinidev processes."""
    lock_path = catalog_path.parent / f".{catalog_path.name}.starters.lock"
    with _STARTER_PROFILE_LOCK:
        with lock_path.open("a+b") as lock_file:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_starter_profile(target: Path, content: str) -> None:
    """Write one exclusively created starter profile durably."""
    created = False
    published = False
    try:
        with target.open("x", encoding="utf-8") as profile_file:
            created = True
            profile_file.write(content)
            profile_file.flush()
            os.fsync(profile_file.fileno())
        published = True
    finally:
        if created and not published:
            target.unlink(missing_ok=True)


def materialize_starter_prompt_profiles(catalog_path: Path) -> tuple[Path, ...]:
    """Publish missing packaged profiles without replacing existing catalog entries."""
    catalog_path.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    with _starter_catalog_lock(catalog_path):
        try:
            for filename, content in STARTER_PROMPT_PROFILES:
                target = catalog_path / filename
                try:
                    _write_starter_profile(target, content)
                except FileExistsError:
                    continue
                created.append(target)
            if created:
                fsync_directory(catalog_path)
        except OSError as err:
            for target in created:
                target.unlink(missing_ok=True)
            try:
                fsync_directory(catalog_path)
            except OSError:
                # Preserve the publication failure that triggered rollback; the
                # caller still needs its original cause to diagnose the write.
                pass
            raise
    return tuple(created)
