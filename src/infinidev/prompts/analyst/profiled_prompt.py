"""Composition helpers for planner prompts with optional guidance blocks."""

from __future__ import annotations

from collections.abc import Mapping

from infinidev.prompts.profiles import (
    EffectivePromptConfiguration,
    resolve_prompt_fragment,
)


def compose_profiled_planner_prompt(
    prompt: str,
    *,
    configuration: EffectivePromptConfiguration | None,
    identity_name: str,
    methodology_name: str,
    section_names: Mapping[str, str],
) -> str:
    """Apply profiles to guidance while retaining unregistered contract sections.

    Planner prompt modules keep their built-in strings as the compatibility
    baseline. The first paragraph is the role identity, the rest of the preamble
    is methodology, and registered Markdown sections are independently optional.
    Sections absent from ``section_names`` are structural engine contracts and
    are always retained.
    """
    if configuration is None:
        return prompt

    preamble, *raw_sections = prompt.split("\n\n## ")
    identity, separator, methodology = preamble.partition("\n\n")
    chunks = [
        resolve_prompt_fragment(
            identity_name,
            "plan",
            identity,
            configuration=configuration,
        )
    ]
    if separator and methodology:
        chunks.append(
            resolve_prompt_fragment(
                methodology_name,
                "plan",
                methodology,
                configuration=configuration,
            )
        )

    for raw_section in raw_sections:
        title, section_separator, body = raw_section.partition("\n\n")
        section = f"## {title}"
        if section_separator:
            section = f"{section}\n\n{body}"
        profile_name = section_names.get(title)
        if profile_name is None:
            chunks.append(section)
        else:
            chunks.append(
                resolve_prompt_fragment(
                    profile_name,
                    "plan",
                    section,
                    configuration=configuration,
                )
            )

    return "\n\n".join(chunk for chunk in chunks if chunk)


__all__ = ["compose_profiled_planner_prompt"]
