"""Hybrid deterministic, semantic, and optional LLM task-policy router."""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
import re
from typing import Any

from infinidev.engine.task_policies.linear_classifier import (
    CLASSIFIER_VERSION as LINEAR_CLASSIFIER_VERSION,
    classify_task_method,
)
from infinidev.engine.task_policies.models import (
    ClassifierResult,
    PolicySelection,
    RejectedPolicyCandidate,
    TaskProfile,
)
from infinidev.engine.task_policies.registry import POLICIES, TaskPolicy
from infinidev.engine.task_policies.semantic import (
    SemanticRetrieval,
    retrieve_policy_candidates,
)

logger = logging.getLogger(__name__)

_OPERATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "bugfix": re.compile(
        r"\b(?:fix(?:es|ing)?|repair(?:s|ed|ing)?|corr(?:[ií]g|ij)\w*|corregir|"
        r"arr[eé]gl\w*|solucion\w*|r[eé]par\w*)\b"
        r"|\bcorrect\s+(?:(?:the|this|an?)\s+)?"
        r"(?:bug|error|failure|issue|behavior|result)\b",
        re.I,
    ),
    "feature": re.compile(
        r"\b(?:implement(?!ation\b|ación\b)|add|create|build|añad|agreg|crea)\w*\b",
        re.I,
    ),
    "refactor": re.compile(
        r"\b(?:refactor|restructur|reestructur|reorganiza|clean\s+up|"
        r"limpia\s+el\s+c[oó]digo)\w*\b",
        re.I,
    ),
    "research": re.compile(
        r"\b(?:research|investig\w*|analiza|compare|compara|re[uú]ne|"
        r"pesquis\w*|recherch\w*|fuentes?)\w*\b",
        re.I,
    ),
    "review": re.compile(
        r"\b(?:review|revis\w*|audit\w*|inspect\w*|examin\w*|analy[sz]\w*|analiz\w*|"
        r"code\s+review|"
        r"pull\s+request|\bpr\b)\w*\b", re.I
    ),
    "performance": re.compile(
        r"\b(?:optimi[sz]|optimiz\w*|otimiz\w*|performance|rendimiento|latency|latencia|"
        r"throughput|benchmarks?|p(?:50|90|95|99)|"
        r"profil(?:e|ing)?\s+(?:the\s+)?(?:workload|code|runtime|application|app|service|function))\w*\b",
        re.I,
    ),
    "docs": re.compile(r"\b(?:document|readme|docs?|documentaci[oó]n)\w*\b", re.I),
    "migration": re.compile(r"\b(?:migrat|migra|schema\s+change)\w*\b", re.I),
    "security": re.compile(r"\b(?:security|seguridad|vulnerab|harden)\w*\b", re.I),
}
_NEGATED_OPERATION = re.compile(
    r"\b(?:no|do\s+not|don't|dont|without|sin)\s+(?:quiero\s+que\s+)?"
    r"(?P<verb>refactor\w*|implement\w*|modifi\w*|cambi\w*|edit\w*|fix\w*|"
    r"corr\w*|appl\w*|investig\w*|research\w*|review\w*|revis\w*)",
    re.I,
)
_READ_ONLY = re.compile(
    r"\b(?:no\s+(?:cambies|modifiques|edites|toques|refactorices)\w*|"
    r"do\s+not\s+(?:fix|refactor|implement|apply)\w*|"
    r"solo\s+(?:expl[ií]ca|revisa|analiza)\w*|"
    r"solo\s+quiero\s+(?:hallazgos|un\s+informe|una\s+explicaci[oó]n)|"
    r"no\s+(?:una\s+)?implementaci[oó]n|"
    r"sin\s+(?:cambiar|modificar|editar)\s+(?:archivos?|files?|c[oó]digo)|"
    r"do\s+not\s+(?:change|modify|edit)\s+(?:files?|code)|read[- ]only|"
    r"without\s+(?:applying|making|implementing|modifying|editing)\b|"
    r"sem\s+(?:aplicar|alterar|modificar|corrigir)\b|"
    r"sans\s+(?:le\s+)?(?:corriger|modifier|appliquer|toucher)\b|"
    r"only\s+(?:explain|review|analy[sz]e)|"
    r"leave\s+(?:the\s+)?(?:source|code|implementation|files?|it)\s+untouched)\b",
    re.I,
)
_MODIFY = re.compile(
    r"\b(?:implement(?!ation\b|ación\b)|correct|corr(?:[ií]g|ij)|fix|r[eé]par|arregla|modifica|change|"
    r"edit|write|escribe|redacta|refactoriza|"
    r"refactor|crea|create|añade|add|actualiza|update|elimina|delete|remove|"
    r"optimi[sz]|optimiz|otimiz|optimiza|solucion|build|restructur|reestructur|clean\s+up|"
    r"reduce|improve|mejora|acelera|arr[eé]gl|restore|restaur|restablec|"
    r"reestablec|r[eé]tabl|recupera|bring\s+back)\w*\b",
    re.I,
)
_IMPERATIVE_MODIFY = re.compile(
    r"(?:^|[.!?;]\s)(?:please\s+|por\s+favor\s+)?"
    r"(?:make\b|haz\s+que\b|quiero\s+que\b|i\s+(?:want|need)\b)",
    re.I,
)
_COMMIT = re.compile(r"\b(?:git\s+)?commit(?:ea|ear|ted|ting)?\b", re.I)
_PUBLISH = re.compile(
    r"\b(?:git\s+push|push\s+(?:a|to)|push\s+the\s+branch|publica|publish|deploy|"
    r"d[eé]ploi\w*|desplieg\w*|envie\w*(?:\s+\w+){0,4}\s+(?:remoto|remote))\b", re.I
)
_PUBLIC_API = re.compile(
    r"\b(?:public\s+api|api\s+p[uú]blica|backwards?\s+compatib|retrocompatib|"
    r"sin\s+(?:cambiar|modificar|romper)\s+(?:la\s+)?api)\b",
    re.I,
)
_PRESERVE_BEHAVIOR = re.compile(
    r"\b(?:preserv\w*\s+(?:behavior|behaviour|comportamiento)|"
    r"sin\s+(?:cambiar|alterar)\s+(?:el|su)?\s*comportamiento|"
    r"keep\s+(?:observable\s+)?behavio(?:u)?r\s+identical|no\s+behavior\s+change)\b",
    re.I,
)
_DESTRUCTIVE = re.compile(r"\b(?:delete|remove|drop|wipe|purge|elimina|borra)\w*\b", re.I)
_QUOTED = re.compile(r"(['\"])(?:(?!\1).)*\1", re.S)
_QUOTED_EXPLANATION = re.compile(
    r"\b(?:explain|interpret|what\s+does|explica|interpreta|qu[eé]\s+significa)\b",
    re.I,
)

_OPERATION_ORDER = tuple(_OPERATION_PATTERNS)
_AUTHORITY_ORDER = ("answer", "diagnose", "modify", "commit", "publish")
_SEQUENCE_BY_OPERATION = {
    "research": "investigate", "review": "review", "bugfix": "implement",
    "feature": "implement", "refactor": "implement", "performance": "implement",
    "docs": "implement", "migration": "implement",
}


def _ordered(values: set[str], order: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(value for value in order if value in values)


def _literal_signals(text: str) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    searchable = _QUOTED.sub(" ", text)
    negated = {match.group("verb").lower() for match in _NEGATED_OPERATION.finditer(searchable)}
    operations: set[str] = set()
    for operation, pattern in _OPERATION_PATTERNS.items():
        match = pattern.search(searchable)
        if match is None:
            continue
        token = match.group(0).lower()
        if any(token.startswith(verb[:5]) or verb.startswith(token[:5]) for verb in negated):
            continue
        operations.add(operation)

    read_only = bool(_READ_ONLY.search(searchable))
    if read_only:
        operations.difference_update(
            {"bugfix", "feature", "refactor", "docs", "migration"}
        )
    authority = {"answer"}
    if operations & {"research", "review", "bugfix", "performance"}:
        authority.add("diagnose")
    if not read_only and (
        _MODIFY.search(searchable) or _IMPERATIVE_MODIFY.search(searchable)
    ):
        authority.add("modify")
    if not read_only and _COMMIT.search(searchable):
        authority.update({"modify", "commit"})
    if not read_only and _PUBLISH.search(searchable):
        authority.update({"modify", "publish"})

    constraints: set[str] = set()
    if read_only:
        constraints.add("read_only")
    if _PUBLIC_API.search(searchable):
        constraints.add("preserve_public_api")
    if _PRESERVE_BEHAVIOR.search(searchable) or "refactor" in operations:
        constraints.add("preserve_behavior")

    risks: set[str] = set()
    if "security" in operations:
        risks.add("security")
    if "migration" in operations:
        risks.add("migration")
    if _DESTRUCTIVE.search(searchable):
        risks.add("destructive")
    if authority & {"commit", "publish"}:
        risks.add("external_write")

    result: set[str] = set()
    modifying_operations = {
        "bugfix", "feature", "refactor", "performance", "docs", "migration",
    }
    if "review" in operations and not operations & modifying_operations:
        result.add("report")
    elif "modify" in authority:
        result.add("code")
    elif "research" in operations:
        result.add("report")
    elif "review" in operations:
        result.add("recommendation")
    return operations, authority, constraints, risks, result


def _default_llm_classifier(text: str) -> ClassifierResult | None:
    """Run the single structured fallback. Any provider failure abstains."""
    try:
        import litellm

        from infinidev.config.llm import get_litellm_params_for_behavior

        schema = ClassifierResult.model_json_schema()
        params = get_litellm_params_for_behavior()
        response = litellm.completion(
            **params,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify task method, never authority. Return JSON matching this schema: "
                        + json.dumps(schema, ensure_ascii=False)
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return ClassifierResult.model_validate_json(content)
    except Exception:
        logger.debug("Task-policy LLM fallback failed", exc_info=True)
        return None


def _policy_matches(policy: TaskPolicy, operations: set[str], constraints: set[str]) -> bool:
    operation_match = bool(policy.operations & operations)
    constraint_match = bool(policy.constraints & constraints)
    if policy.operations:
        return operation_match
    return constraint_match


def resolve_task_profile(
    text: str,
    *,
    enable_embeddings: bool = False,
    enable_llm_fallback: bool = False,
    embedding_threshold: float = 0.72,
    embedding_margin: float = 0.08,
    max_policies: int = 3,
    classifier: Callable[[str], ClassifierResult | None] | None = None,
) -> TaskProfile:
    """Resolve one reusable profile without granting semantic write authority."""
    operations, authority, constraints, risks, result = _literal_signals(text)
    evidence_by_policy: dict[str, tuple[str, str, float | None]] = {}
    for policy in POLICIES:
        if _policy_matches(policy, operations, constraints):
            evidence_by_policy[policy.id] = ("deterministic", "literal-signal", 1.0)

    semantic = SemanticRetrieval(
        candidates=(), space_id=None,
        classifier_version=LINEAR_CLASSIFIER_VERSION,
        abstained=False,
    )
    # The trained head classifies method for every request. Literal parsing is
    # retained for authority, negation, and high-confidence conflict vetoes.
    # Ambiguous paraphrases require agreement with the independent contrastive
    # retriever before a prompt layer becomes active.
    quoted_explanation = bool(_QUOTED.search(text) and _QUOTED_EXPLANATION.search(text))
    if enable_embeddings and not operations and quoted_explanation:
        semantic = SemanticRetrieval(
            candidates=(), space_id=None,
            classifier_version=LINEAR_CLASSIFIER_VERSION,
            abstained=True, reason="quoted action is explanatory context",
        )
    if enable_embeddings and not quoted_explanation:
        prediction = classify_task_method(_QUOTED.sub(" ", text))
        semantic = SemanticRetrieval(
            candidates=(),
            space_id=prediction.space_id,
            classifier_version=prediction.classifier_version,
            abstained=prediction.policy_id is None,
            reason=prediction.abstention_reason,
        )
        predicted_id = prediction.policy_id
        if (
            predicted_id is None
            and prediction.agreement_eligible
            and prediction.candidate_policy_id
        ):
            # The hierarchical head exposes a lower confidence tier that is
            # unusable alone. Literal method evidence or the independent
            # contrastive retriever must still agree before selection.
            predicted_id = prediction.candidate_policy_id
        if predicted_id is None and operations and prediction.candidate_policy_id:
            candidate = next(
                (
                    item
                    for item in POLICIES
                    if item.id == prediction.candidate_policy_id
                ),
                None,
            )
            if (
                candidate is not None
                and candidate.operations & operations
                and prediction.score >= prediction.threshold
            ):
                # A literal method may resolve the mini-head's runner-up tie,
                # because this cannot add a category or grant authority. The
                # strict margin remains mandatory for semantic-only routing.
                predicted_id = candidate.id
        predicted_policy = next(
            (item for item in POLICIES if item.id == predicted_id), None
        )
        literal_agreement = bool(
            predicted_policy is not None
            and predicted_policy.operations & operations
        )
        contrastive = None
        if predicted_policy is not None and not operations:
            contrastive = retrieve_policy_candidates(
                _QUOTED.sub(" ", text),
                min_score=embedding_threshold,
                min_margin=embedding_margin,
            )
        contrastive_agreement = bool(
            contrastive is not None
            and contrastive.candidates
            and contrastive.candidates[0].policy.id == predicted_policy.id
        )
        if predicted_policy is not None and (literal_agreement or contrastive_agreement):
            if predicted_policy.requires_modify and "modify" not in authority:
                semantic = SemanticRetrieval(
                    candidates=(), space_id=prediction.space_id,
                    classifier_version=prediction.classifier_version,
                    abstained=True,
                    reason="mini-head method lacks literal modify authority",
                )
            elif predicted_policy.requires_modify and "read_only" in constraints:
                semantic = SemanticRetrieval(
                    candidates=(), space_id=prediction.space_id,
                    classifier_version=prediction.classifier_version,
                    abstained=True,
                    reason="mini-head method conflicts with read-only authority",
                )
            else:
                operations.update(predicted_policy.operations)
                evidence = "mini-head+literal" if literal_agreement else "mini-head+contrastive"
                evidence_by_policy[predicted_policy.id] = (
                    "embedding", evidence, prediction.score,
                )
                semantic = SemanticRetrieval(
                    candidates=(), space_id=prediction.space_id,
                    classifier_version=prediction.classifier_version,
                    abstained=False,
                )
        elif predicted_policy is not None:
            semantic = SemanticRetrieval(
                candidates=(), space_id=prediction.space_id,
                classifier_version=prediction.classifier_version,
                abstained=True,
                reason=(
                    "mini-head conflicts with literal method"
                    if operations else "mini-head and contrastive classifier disagree"
                ),
            )
        # A discourse-gate abstention is final. A method candidate from the
        # relaxed tier remains unusable unless literal or contrastive evidence
        # independently chooses that same policy.

    llm_used = False
    llm_sequence: set[str] = set()
    ambiguous = not operations
    if enable_llm_fallback and ambiguous:
        llm_result = (classifier or _default_llm_classifier)(text)
        if llm_result is not None:
            llm_used = True
            operations.update(llm_result.operations)
            constraints.update(llm_result.constraints)
            risks.update(llm_result.risks)
            result.update(llm_result.result)
            llm_sequence.update(llm_result.sequence)
            for policy in POLICIES:
                if _policy_matches(policy, operations, constraints):
                    evidence_by_policy.setdefault(policy.id, ("llm", "structured-profile", None))

    if not result:
        modifying_operations = {
            "bugfix", "feature", "refactor", "performance", "docs", "migration",
        }
        if "review" in operations and "modify" not in authority:
            result.add("report")
        elif "performance" in operations and "modify" not in authority:
            result.add("report")
        elif "modify" in authority and operations & modifying_operations:
            result.add("code")
        elif "research" in operations:
            result.add("report")

    rejected: list[RejectedPolicyCandidate] = []
    selected: list[tuple[TaskPolicy, PolicySelection]] = []
    for policy in POLICIES:
        candidate = evidence_by_policy.get(policy.id)
        if candidate is None:
            continue
        source, evidence, score = candidate
        if policy.requires_modify and "modify" not in authority:
            rejected.append(RejectedPolicyCandidate(
                id=policy.id, reason="literal request grants no modify authority", score=score,
            ))
            continue
        modifying_operations = {
            "bugfix", "feature", "refactor", "performance", "migration",
        }
        if (
            policy.forbids_modify
            and "modify" in authority
            and operations & modifying_operations
        ):
            rejected.append(RejectedPolicyCandidate(
                id=policy.id,
                reason="literal request grants explicit modify authority",
                score=score,
            ))
            continue
        selected.append((policy, PolicySelection(
            id=policy.id, version=policy.version, source=source,
            evidence=(evidence,), score=score, policy_hash=policy.content_hash,
        )))

    selected.sort(key=lambda item: (-item[0].priority, item[0].id))
    accepted: list[tuple[TaskPolicy, PolicySelection]] = []
    for policy, item in selected:
        conflict = next(
            (
                existing.id
                for existing, _ in accepted
                if existing.id in policy.incompatible_with
                or policy.id in existing.incompatible_with
            ),
            None,
        )
        if conflict is not None:
            rejected.append(RejectedPolicyCandidate(
                id=policy.id, reason=f"incompatible with selected policy {conflict}",
                score=item.score,
            ))
            continue
        if len(accepted) >= max_policies:
            rejected.append(RejectedPolicyCandidate(
                id=policy.id, reason=f"policy limit {max_policies} reached", score=item.score,
            ))
            continue
        accepted.append((policy, item))
    chosen = tuple(item for _, item in accepted)

    sequence = {
        _SEQUENCE_BY_OPERATION[op]
        for op in operations
        if op in _SEQUENCE_BY_OPERATION
        and (_SEQUENCE_BY_OPERATION[op] != "implement" or "modify" in authority)
    }
    sequence.update(
        step for step in llm_sequence
        if step not in {"implement", "commit", "publish"} or (
            step == "implement" and "modify" in authority
            or step == "commit" and "commit" in authority
            or step == "publish" and "publish" in authority
        )
    )
    if "implement" in sequence:
        sequence.add("verify")
    if "performance" in operations and "modify" not in authority:
        sequence.add("investigate")
    if "commit" in authority:
        sequence.add("commit")
    if "publish" in authority:
        sequence.add("publish")
    sequence_order = ("investigate", "implement", "verify", "review", "commit", "publish")

    return TaskProfile(
        operations=_ordered(operations, _OPERATION_ORDER),
        authority=_ordered(authority, _AUTHORITY_ORDER),
        constraints=_ordered(
            constraints, ("preserve_behavior", "preserve_public_api", "read_only")
        ),
        risks=_ordered(risks, ("security", "migration", "destructive", "external_write")),
        result=_ordered(result, ("code", "report", "plan", "recommendation")),
        sequence=_ordered(sequence, sequence_order),
        selected_policies=chosen,
        rejected_candidates=tuple(rejected),
        llm_fallback_used=llm_used,
        semantic_space_id=semantic.space_id,
        semantic_classifier_version=(
            semantic.classifier_version if enable_embeddings else None
        ),
        semantic_abstained=semantic.abstained,
        semantic_abstention_reason=semantic.reason,
    )
