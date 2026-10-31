"""Conditional guidance must reach the model as whole instruction blocks."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from infinidev.engine.prompt_composition import (
    ConditionalPromptFragment,
    select_conditional_catalog,
    select_conditional_fragments,
)
from infinidev.engine.task_policies import rendering
from infinidev.engine.task_policies.router import resolve_task_profile


def _fragment(content, *, limit=900, name="method", priority=0):
    return ConditionalPromptFragment(
        id=name, policy_id="refactor.preserve_behavior", content=content,
        roles=frozenset({"developer"}), phases=frozenset({"execute"}),
        condition_reason='the task requests "refactoring" & permits it',
        max_utf8_bytes=limit, priority=priority,
    )


def _select(fragments, *, catalog, budget=3600):
    kwargs = dict(role="developer", phase="execute", max_utf8_bytes=budget)
    if catalog:
        return select_conditional_catalog(fragments, **kwargs)
    profile = resolve_task_profile("Refactoriza este módulo sin cambiar su comportamiento.")
    return select_conditional_fragments(profile, fragments, **kwargs)


@pytest.mark.parametrize("catalog", [False, True])
def test_overlong_instruction_is_omitted_instead_of_losing_its_final_condition(catalog):
    result = _select((
        _fragment("Inspect the file. Do not commit.", limit=19, priority=10),
        _fragment("Preserve behavior.", name="short"),
    ), catalog=catalog)

    assert [fragment.id for fragment in result.fragments] == ["short"]
    assert result.omitted == (("method", "fragment-budget"),)


@pytest.mark.parametrize("catalog", [False, True])
@pytest.mark.parametrize("limit,selected", [(6, True), (5, False)])
def test_individual_cap_counts_complete_utf8_characters(catalog, limit, selected):
    result = _select((_fragment("á🧪", limit=limit),), catalog=catalog)

    assert bool(result.fragments) is selected
    assert result.omitted == (() if selected else (("method", "fragment-budget"),))


@pytest.mark.parametrize("catalog", [False, True])
def test_blank_guidance_is_empty_rather_than_a_budget_failure(catalog):
    result = _select((_fragment(" \n ", limit=1),), catalog=catalog)

    assert result.fragments == ()
    assert result.omitted == (("method", "empty"),)


def test_selected_body_accounting_preserves_whitespace_bytes():
    result = _select((_fragment("  Keep the API.\n"),), catalog=False)

    assert result.used_utf8_bytes == 16


@pytest.mark.parametrize("catalog", [False, True])
@pytest.mark.parametrize("limit,expected", [(900, True), (19, False)])
@pytest.mark.parametrize("body,selected_hash,catalog_hash", [
    (
        "Inspect the file. Do not commit.",
        "69e493c8d8ffaf90eb8c295c710730a24febb65237d963a502e4fc24528d0321",
        "e2d54c1e5d2f6d789bb08a5e799b48c07cf619dc2ae6533cb00439dd42f44248",
    ),
    (
        " \tRevisa 🧪.\nNo publiques.\n ",
        "9fde14167b578dc88c2ffc80a6a6c164121f6bed973e16ee2505e62c1e881e32",
        "8976263f99ca8c9bdbe7c56ba6c4fe7de033d93239f5ad1e9907231f39e0691b",
    ),
])
def test_renderer_emits_the_complete_instruction_or_no_layer(
    monkeypatch, catalog, limit, expected, body, selected_hash, catalog_hash,
):
    monkeypatch.setattr(rendering, "TASK_METHOD_FRAGMENTS", (_fragment(body, limit=limit),))
    kwargs = dict(role="developer", phase="execute", force=True)
    if catalog:
        layer = rendering.render_all_conditional_task_policy_layer(**kwargs)
    else:
        profile = resolve_task_profile("Refactoriza este módulo sin cambiar su comportamiento.")
        layer = rendering.render_task_policy_layer(profile, **kwargs)

    if not expected:
        assert layer == ""
        return
    blocks = re.findall(r"<prompt-fragment\b.*?</prompt-fragment>", layer, flags=re.S)
    assert len(blocks) == 1
    fragment = ET.fromstring(blocks[0])
    payload = blocks[0].partition(">\n")[2].removesuffix("\n</prompt-fragment>")
    if catalog:
        condition = fragment.find("if")
        assert condition.attrib["reason"] == 'the task requests "refactoring" & permits it'
        payload = payload.partition(">\n")[2].removesuffix("\n</if>")
    assert payload == body
    assert fragment.attrib["sha256"] == (catalog_hash if catalog else selected_hash)


@pytest.mark.parametrize("role,phase", [
    ("planner", "plan"), ("developer", "execute"), ("reviewer", "review"),
])
def test_default_budget_delivers_every_builtin_method_in_each_role(role, phase):
    layer = rendering.render_all_conditional_task_policy_layer(
        role=role, phase=phase, force=True,
    )
    blocks = re.findall(r"<prompt-fragment\b.*?</prompt-fragment>", layer, flags=re.S)

    assert {ET.fromstring(block).attrib["id"] for block in blocks} == {
        f"compatibility.{role}", f"review.{role}", f"bugfix.{role}", f"refactor.{role}",
        f"feature.{role}", f"performance.{role}", f"research.{role}",
    }
