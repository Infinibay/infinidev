"""Flow identity prompts — registers all flows at import time."""

from infinidev.engine.flows import FlowConfig, register_flow

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
