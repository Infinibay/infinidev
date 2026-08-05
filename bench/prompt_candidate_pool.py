#!/usr/bin/env python3
"""Validate evidence-bound prompt candidates and compile inert evaluation conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping


MAX_CANDIDATES = 12
MAX_GUIDANCE_BYTES = 1024
ALLOWED_ROLES = frozenset({"chat_agent", "planner", "developer"})
ALLOWED_KINDS = frozenset(
    {"preference_compensation", "preference_stabilization", "normative_remediation"}
)
_ABSOLUTE_PREFERENCE_LANGUAGE = re.compile(
    r"\b(always|never|must|under all circumstances|regardless of the user)\b", re.IGNORECASE
)
_EXPOSED_WEIGHT = re.compile(
    r"\b(?:autonomy|interaction|user_control|speed|quality|cost_efficiency|caution)\s*=",
    re.IGNORECASE,
)


def _brief_evidence(brief: Mapping[str, object]) -> dict[str, tuple[str, Mapping[str, object]]]:
    sections = {
        "stable_profile_conflicts_to_test": "preference_compensation",
        "unstable_profile_hypotheses": "preference_stabilization",
        "normative_evidence": "normative_remediation",
    }
    evidence: dict[str, tuple[str, Mapping[str, object]]] = {}
    for section, kind in sections.items():
        records = brief.get(section, [])
        if not isinstance(records, list):
            raise ValueError(f"brief section must be a list: {section}")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"brief section contains a non-object: {section}")
            probe_id = str(record.get("probe_id", "")).strip()
            if not probe_id or probe_id in evidence:
                raise ValueError(f"brief evidence ids must be non-empty and unique: {probe_id}")
            if kind == "normative_remediation" and record.get("status") == "stable_match":
                continue
            evidence[probe_id] = (kind, record)
    return evidence


def compile_candidate_pool(
    pool: Mapping[str, object],
    brief: Mapping[str, object],
    *,
    brief_sha256: str,
) -> dict[str, object]:
    """Fail closed unless every candidate is compact, advisory, and evidence-bound."""
    if pool.get("schema_version") != 1:
        raise ValueError("candidate pool schema_version must be 1")
    if pool.get("source_brief_sha256") != brief_sha256:
        raise ValueError("candidate pool source brief hash does not match")
    model = str(brief.get("model", ""))
    if pool.get("model") != model:
        raise ValueError("candidate pool model does not match brief")
    profile = brief.get("utility_profile")
    if not isinstance(profile, dict):
        raise ValueError("brief utility profile is missing")
    if pool.get("utility_profile_sha256") != profile.get("sha256"):
        raise ValueError("candidate pool utility profile hash does not match brief")
    model_identity = str(pool.get("model_identity", "")).strip()
    role = str(pool.get("calibration_role", "")).strip()
    if not model_identity:
        raise ValueError("candidate pool needs immutable model_identity")
    if role not in ALLOWED_ROLES:
        raise ValueError("candidate pool calibration_role is invalid")
    raw_candidates = pool.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate pool needs at least one candidate")
    if len(raw_candidates) > MAX_CANDIDATES:
        raise ValueError("candidate pool exceeds the bounded candidate count")

    evidence = _brief_evidence(brief)
    names: set[str] = set()
    hashes: set[str] = set()
    compiled: list[dict[str, object]] = []
    conditions: dict[str, object] = {"current": None}
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("candidate must be an object")
        name = str(raw.get("name", "")).strip()
        kind = str(raw.get("kind", "")).strip()
        guidance = str(raw.get("guidance", "")).strip()
        target_ids = raw.get("evidence_probe_ids")
        rationale = str(raw.get("rationale", "")).strip()
        expected_effect = str(raw.get("expected_effect", "")).strip()
        regression_risks = raw.get("regression_risks")
        if not name or name == "current" or name in names:
            raise ValueError(f"candidate names must be non-empty and unique: {name}")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"candidate kind is invalid: {kind}")
        if not guidance or len(guidance.encode("utf-8")) > MAX_GUIDANCE_BYTES:
            raise ValueError(f"candidate guidance is empty or exceeds compact limit: {name}")
        if not rationale or not expected_effect:
            raise ValueError(f"candidate needs rationale and expected_effect: {name}")
        if not isinstance(regression_risks, list) or not regression_risks:
            raise ValueError(f"candidate needs explicit regression_risks: {name}")
        if not isinstance(target_ids, list) or not target_ids:
            raise ValueError(f"candidate needs evidence_probe_ids: {name}")
        normalized_ids = [str(item).strip() for item in target_ids]
        if "" in normalized_ids or len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError(f"candidate evidence ids must be non-empty and unique: {name}")
        selected_evidence: list[Mapping[str, object]] = []
        for probe_id in normalized_ids:
            source = evidence.get(probe_id)
            if source is None or source[0] != kind:
                raise ValueError(f"candidate evidence does not support {kind}: {name}/{probe_id}")
            selected_evidence.append(source[1])
        if kind.startswith("preference_"):
            if raw.get("guidance_style") != "advisory":
                raise ValueError(f"preference candidate must be advisory: {name}")
            if _ABSOLUTE_PREFERENCE_LANGUAGE.search(guidance):
                raise ValueError(f"preference candidate uses absolute language: {name}")
        if _EXPOSED_WEIGHT.search(guidance):
            raise ValueError(f"candidate guidance exposes evaluation weights: {name}")
        guidance_hash = hashlib.sha256(guidance.encode()).hexdigest()
        if guidance_hash in hashes:
            raise ValueError(f"candidate guidance duplicates another candidate: {name}")
        names.add(name)
        hashes.add(guidance_hash)
        categories = sorted({str(item.get("category", "")) for item in selected_evidence})
        evidence_actions = {
            probe_id: evidence[probe_id][1].get("profile_best_actions")
            or evidence[probe_id][1].get("draft_expected_action")
            for probe_id in normalized_ids
        }
        compiled.append(
            {
                "name": name,
                "kind": kind,
                "guidance_style": raw.get("guidance_style"),
                "guidance": guidance,
                "guidance_sha256": guidance_hash,
                "utf8_bytes": len(guidance.encode("utf-8")),
                "evidence_probe_ids": normalized_ids,
                "evidence_actions": evidence_actions,
                "categories": categories,
                "rationale": rationale,
                "expected_effect": expected_effect,
                "regression_risks": [str(item) for item in regression_risks],
            }
        )
        conditions[name] = {"system_prompt": guidance}
    return {
        "schema_version": 1,
        "deployment_approved": False,
        "model": model,
        "model_identity": model_identity,
        "calibration_role": role,
        "utility_profile": {
            "name": profile.get("name"),
            "sha256": profile.get("sha256"),
        },
        "source_brief_sha256": brief_sha256,
        "candidates": compiled,
        "run_config_fragment": {
            "model_identity": model_identity,
            "calibration_role": role,
            "conditions": conditions,
        },
        "release_boundary": (
            "This artifact defines inert evaluation conditions. It cannot activate runtime guidance; "
            "held-out paired validation and an explicitly approved deployment profile remain required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path)
    parser.add_argument("brief", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    brief_bytes = args.brief.read_bytes()
    brief = json.loads(brief_bytes)
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    compiled = compile_candidate_pool(
        pool, brief, brief_sha256=hashlib.sha256(brief_bytes).hexdigest()
    )
    args.output.write_text(
        json.dumps(compiled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
