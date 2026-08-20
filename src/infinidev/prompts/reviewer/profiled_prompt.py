"""Composition helpers for reviewer prompts with optional guidance blocks."""

from __future__ import annotations

from collections.abc import Mapping
import re

from infinidev.prompts.profiles import (
    EffectivePromptConfiguration,
    resolve_prompt_fragment,
)


def compose_profiled_reviewer_prompt(
    prompt: str,
    *,
    configuration: EffectivePromptConfiguration,
    section_names: Mapping[str, str],
) -> str:
    """Profile reviewer guidance while retaining output contracts verbatim."""
    if prompt.startswith("## "):
        preamble = ""
        marker = "## "
        sections = prompt[3:]
    else:
        preamble, marker, sections = prompt.partition("\n## ")
    chunks: list[str | None] = [preamble] if preamble else []
    if marker:
        for raw_section in re.split(r"(?m)^## ", sections):
            if not raw_section.strip():
                continue
            title, separator, body = raw_section.partition("\n\n")
            section = f"## {title}"
            if separator:
                section = f"{section}\n\n{body}"
            profile_name = section_names.get(title.strip())
            if profile_name is None:
                chunks.append(section)
            else:
                chunks.append(
                    resolve_prompt_fragment(
                        profile_name,
                        "review",
                        section,
                        configuration=configuration,
                    )
                )
    return "\n\n".join(chunk for chunk in chunks if chunk)


__all__ = ["compose_profiled_reviewer_prompt"]
