"""Flow configuration registry for multi-flow task routing.

Each flow defines a specialized agent persona (identity prompt, backstory,
expected output template) while reusing the same LoopEngine and tools.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FlowConfig", "FLOW_REGISTRY", "register_flow", "get_flow_config"]


@dataclass(frozen=True)
class FlowConfig:
    """Configuration for an execution flow."""

    name: str  # "develop", "research", "document", "sysadmin"
    identity_prompt: str  # System prompt identity (replaces CLI_AGENT_IDENTITY)
    expected_output: str  # Template for expected_output of the task prompt
    backstory: str  # Short backstory for the agent
    run_review: bool = False  # Whether to run code review after execution


FLOW_REGISTRY: dict[str, FlowConfig] = {}


def register_flow(config: FlowConfig) -> None:
    """Register a flow configuration."""
    FLOW_REGISTRY[config.name] = config


def get_flow_config(flow_name: str) -> FlowConfig:
    """Get a flow configuration by name.

    Raises ValueError if the flow is not registered.
    """
    if flow_name not in FLOW_REGISTRY:
        raise ValueError(
            f"Unknown flow: {flow_name!r}. "
            f"Available flows: {', '.join(FLOW_REGISTRY)}"
        )
    return FLOW_REGISTRY[flow_name]
