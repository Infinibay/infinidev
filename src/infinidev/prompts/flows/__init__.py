"""Flow identity prompts — registers all flows at import time."""

from __future__ import annotations

from infinidev.engine.flows import FlowConfig, register_flow, get_flow_config


def get_flow_identity(flow_name: str, available_tools: set[str] | None = None) -> str:
    """Return the identity prompt for *flow_name*, respecting ``PROMPT_STYLE``.

    For the ``develop`` flow with *available_tools* provided and style ``full``,
    delegates to ``get_develop_identity(available_tools)`` which generates a
    conditional tool section.
    """
    from infinidev.prompts.variants import resolve_style, get_variant

    style = resolve_style()

    # Special case: develop + available_tools + full → dynamic tool section
    if flow_name == "develop" and available_tools is not None and style == "full":
        from infinidev.prompts.flows.develop import get_develop_identity
        return get_develop_identity(available_tools)

    return get_variant(f"flow.{flow_name}.identity", style) or get_flow_config(flow_name).identity_prompt

from infinidev.prompts.flows.develop import (
    DEVELOP_BACKSTORY,
    DEVELOP_EXPECTED_OUTPUT,
    DEVELOP_IDENTITY,
)
from infinidev.prompts.flows.research import (
    RESEARCH_BACKSTORY,
    RESEARCH_EXPECTED_OUTPUT,
    RESEARCH_IDENTITY,
)
from infinidev.prompts.flows.document import (
    DOCUMENT_BACKSTORY,
    DOCUMENT_EXPECTED_OUTPUT,
    DOCUMENT_IDENTITY,
)
from infinidev.prompts.flows.sysadmin import (
    SYSADMIN_BACKSTORY,
    SYSADMIN_EXPECTED_OUTPUT,
    SYSADMIN_IDENTITY,
)
from infinidev.prompts.flows.explore import (
    EXPLORE_BACKSTORY,
    EXPLORE_EXPECTED_OUTPUT,
    EXPLORE_IDENTITY,
)
from infinidev.prompts.flows.brainstorm import (
    BRAINSTORM_BACKSTORY,
    BRAINSTORM_EXPECTED_OUTPUT,
    BRAINSTORM_IDENTITY,
)

register_flow(FlowConfig(
    name="develop",
    identity_prompt=DEVELOP_IDENTITY,
    expected_output=DEVELOP_EXPECTED_OUTPUT,
    backstory=DEVELOP_BACKSTORY,
    run_review=True,
))

register_flow(FlowConfig(
    name="research",
    identity_prompt=RESEARCH_IDENTITY,
    expected_output=RESEARCH_EXPECTED_OUTPUT,
    backstory=RESEARCH_BACKSTORY,
))

register_flow(FlowConfig(
    name="document",
    identity_prompt=DOCUMENT_IDENTITY,
    expected_output=DOCUMENT_EXPECTED_OUTPUT,
    backstory=DOCUMENT_BACKSTORY,
))

register_flow(FlowConfig(
    name="sysadmin",
    identity_prompt=SYSADMIN_IDENTITY,
    expected_output=SYSADMIN_EXPECTED_OUTPUT,
    backstory=SYSADMIN_BACKSTORY,
))

register_flow(FlowConfig(
    name="explore",
    identity_prompt=EXPLORE_IDENTITY,
    expected_output=EXPLORE_EXPECTED_OUTPUT,
    backstory=EXPLORE_BACKSTORY,
))

register_flow(FlowConfig(
    name="brainstorm",
    identity_prompt=BRAINSTORM_IDENTITY,
    expected_output=BRAINSTORM_EXPECTED_OUTPUT,
    backstory=BRAINSTORM_BACKSTORY,
))
