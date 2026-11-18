"""Reduce a rejected principal request using its exact archived tool evidence."""

from __future__ import annotations

import json
from typing import Any

import httpx

_RECALL_LOCATOR = '[Archived evidence; recall_context reference: '
_EXCERPT_CHARACTERS = 256


def is_context_window_rejection(error: Exception) -> bool:
    """Recognize a typed limit or the compatible endpoint's context error."""
    from litellm.exceptions import ContextWindowExceededError

    if isinstance(error, ContextWindowExceededError):
        return True
    response = getattr(error, 'response', None)
    if getattr(error, 'status_code', None) not in (400, 413):
        return False
    detail = getattr(error, 'body', None)
    if not isinstance(detail, dict) and isinstance(response, httpx.Response):
        try:
            detail = response.json()
        except (ValueError, httpx.ResponseNotRead):
            return False
    if isinstance(detail, dict):
        detail = detail.get('error', detail)
    if not isinstance(detail, dict):
        return False
    if detail.get('code') == 'context_length_exceeded':
        return True
    message = detail.get('message')
    # Basalt Chat Completions returns this fixed prefix without a code.
    return (
        detail.get('type') == 'invalid_request_error'
        and isinstance(message, str)
        and message.startswith('request exceeds the context window:')
    )


def shorten_archived_results(messages: list[dict[str, Any]]) -> int:
    """Replace long results with excerpts only when their exact archive is usable."""
    shortened = 0
    for message in messages:
        locator = message.get('_archive_reference', '')
        original_size = message.get('_archive_characters')
        content = message.get('content')
        if (
            not isinstance(locator, str)
            or not locator.startswith(_RECALL_LOCATOR + 'wm-archive:')
            or not locator.endswith(']')
            or type(original_size) is not int
            or not isinstance(content, str)
        ):
            continue
        heading = ''
        excerpt = content
        if message.get('_tool_result'):
            header, separator, excerpt = content.partition('\n')
            heading = header + separator
        excerpt = excerpt.removesuffix('\n' + locator)[:_EXCERPT_CHARACTERS]
        reference = locator[len(_RECALL_LOCATOR):-1]
        preview = heading + json.dumps({
            'status': 'archived_for_context',
            'authority': 'historical_data',
            'reference': reference,
            'original_characters': original_size,
            'excerpt': excerpt,
            'recovery': 'Use recall_context with this reference and bounded start and length.',
        }, ensure_ascii=False)
        if len(preview.encode()) < len(content.encode()):
            message['content'] = preview
            message['_archive_reference'] = reference
            shortened += 1
    return shortened
