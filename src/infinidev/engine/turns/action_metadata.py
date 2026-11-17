"""Explicit model declarations outside literal executable tool input."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class ActionMetadata(BaseModel):
    """Public purpose and expected evidence; never host-certified reasoning."""

    model_config = ConfigDict(extra='forbid')
    reason: StrictStr = Field(max_length=500, description='Brief public operational purpose, not private reasoning.')
    expected_result: StrictStr = Field(max_length=500, description='Observable evidence expected, not a claim of success.')

    @field_validator('reason', 'expected_result')
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('A public declaration must not be blank')
        return value


class ActionEnvelope(BaseModel):
    """A transport boundary; only input reaches normal tool validation."""

    model_config = ConfigDict(extra='forbid')
    metadata: ActionMetadata
    input: dict[str, Any]


def transport_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Derive both transport modes from the same existing catalog schema."""
    wrapped = deepcopy(schema)
    original = wrapped['function']['parameters']
    definitions = original.pop('$defs', None)
    parameters = {'type': 'object', 'properties': {
        'metadata': ActionMetadata.model_json_schema(), 'input': original},
        'required': ['metadata', 'input'], 'additionalProperties': False}
    if definitions:
        parameters['$defs'] = definitions
    wrapped['function']['parameters'] = parameters
    return wrapped
