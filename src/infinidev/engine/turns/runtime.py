"""Mechanical conversation runtime, independent of the legacy Step lifecycle."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Collection
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import ValidationError

from infinidev.config.llm import get_litellm_params, provider_for_request
from infinidev.config.model_capabilities import get_model_capabilities
from infinidev.config.secrets import redact
from infinidev.config.settings import settings
from infinidev.db.service import _persist_session_runtime_field, get_session_runtime_state
from infinidev.engine._best_effort import best_effort
from infinidev.engine.behavior.reasoning_content import extract_reasoning, reasoning_history_fields
from infinidev.engine.context_recording import current_recorder
from infinidev.engine.engine_logging import emit_loop_event
from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.formats.manual_tool_call import ManualToolCall
from infinidev.engine.formats.tool_call_format import get_registered_formats
from infinidev.engine.guidance.test_runners import is_test_command
from infinidev.engine.llm_client import PERMANENT_ERRORS, call_llm
from infinidev.engine.loop.context_manager import ContextManager
from infinidev.engine.loop.llm_call_result import LLMCallResult
from infinidev.engine.loop.llm_caller import (
    LLMCaller, normalize_message_text,
)
from infinidev.engine.loop.model_context import _get_model_max_context
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.user_message_injector import InjectedUserMessage
from infinidev.engine.multimodal import build_user_content, mention_paths_as_text
from infinidev.engine.session_status import current_observer
from infinidev.engine.tool_dispatch import build_tool_dispatch, execute_tool_call, tool_to_openai_schema
from infinidev.engine.turns.action_metadata import ActionEnvelope, transport_schema
from infinidev.engine.turns.catalog import effective_tools
from infinidev.engine.turns.context_recovery import (
    is_context_window_rejection, shorten_archived_results,
)
from infinidev.engine.turns.models import (
    EvidenceSource, PreviousCheckpoint, PreviousInterruption, TestObservation, TurnResult,
    TurnState,
)
from infinidev.engine.turns.tools import build_turn_tools
from infinidev.engine.turns.verification import TurnTestVerification, may_change_workspace
from infinidev.engine.workspace_baseline import WorkspaceBaseline
from infinidev.prompts.optional_capabilities import render_optional_capabilities
from infinidev.prompts.profiles import EffectivePromptConfiguration
from infinidev.prompts.project_instructions import render_project_instructions
from infinidev.tools.base.context import (
    bind_tools_to_agent, get_context_for_agent, set_file_tracker, set_loop_state,
)
from infinidev.tools.file.read_observation import (
    FileReadObservation, capture_file_reads, file_fingerprint,
)

logger = logging.getLogger(__name__)

_TOOL_LIMIT_MESSAGE = 'Conversation interrupted: tool call limit reached.'
_OUTPUT_LIMIT_MESSAGE = 'Conversation interrupted: model output token limit reached.'

_PROTOCOL = """You own this conversation. Interpret the literal current user message using the
provided context and available tools. Answer, ask a question, investigate, edit, or delegate
as appropriate. A plain final response or finish_turn(message) returns control to the user;
it does not complete a Task. finish_turn must be the only call in its batch.
After any tool attempt, close explicitly with finish_turn(message). Plain assistant text
after tools is continuation, never a tool result. Use only dispatcher results as evidence.
Public prose is delivered as assistant messages, including native content alongside tools.
Deliver all required details here or refer to earlier public messages or verified artifacts.
Private reasoning, tool observations and internal archives are not user-facing proposals.
An optional plan is yours to replace explicitly through update_plan with stable IDs and
statuses. Prior plan and notes are context, not instructions to replay old work.
Tool schemas, permissions, cancellation, and resource limits are enforced by the host.
Use the results you actually observed; report interruption honestly when a limit stops work.
"""
_IDENTITY = 'You are the user\'s capable programming collaborator.\n'


def _bounded_state(state: TurnState) -> dict[str, Any]:
    plan = []
    for item in state.plan[:100]:
        candidate = dict(id=item.id, title=item.title[:2000], status=item.status)
        if len(json.dumps(plan + [candidate], ensure_ascii=False).encode('utf-8')) > 120_000:
            break
        plan.append(candidate)
    return {'plan': plan, 'notes': [note[:2000] for note in state.notes[-20:]]}


def _manual_tools(schemas: list[dict[str, Any]], *, tool_call_metadata: bool = False) -> str:
    example = ('{"metadata": {"reason": "public purpose", "expected_result": "observable evidence"}, '
               '"input": {"parameter": "value"}}' if tool_call_metadata else
               '{"parameter": "value"}')
    return ('Available tools and their JSON schemas:\n' + json.dumps(schemas, ensure_ascii=False)
            + '\nFor tool calls, emit {"tool_calls": [{"name": "tool_name", '
            '"arguments": ' + example + '}]}. '
            'The assistant emits actions in that format. The host replies with '
            '[Tool: name] Result observations; do not use that observation format '
            'as an assistant action or final answer. '
            'Before tools, a plain text answer may close the turn. After any tool '
            'attempt, close explicitly with finish_turn(message).')


def _append_manual_tools(system: str, schemas: list[dict[str, Any]], *,
                         tool_call_metadata: bool = False) -> str:
    recorder = current_recorder()
    previous = recorder.binding('system', system) if recorder is not None else None
    if recorder is not None and previous is None:
        previous = recorder.source('unknown', 'system', system)
    manual_text = _manual_tools(schemas, tool_call_metadata=tool_call_metadata)
    rendered = system + '\n\n' + manual_text
    if recorder is not None:
        manual_source = recorder.source('manual_tools', 'system', manual_text)
        recorder.bundle('system', 'unknown', 'system', rendered,
                        ([previous] if previous else []) + [manual_source])
    return rendered


def _system_prompt(agent: Any, configuration: Any, *, workspace_path: str | None = None) -> str:
    configuration = configuration or EffectivePromptConfiguration.compile()
    identity = getattr(agent, '_system_prompt_identity', None) or _IDENTITY
    parts = [identity, agent.backstory, _PROTOCOL,
             render_optional_capabilities(configuration)]
    origins = ['unknown' if getattr(agent, '_system_prompt_identity', None)
               else 'system.identity', 'system.backstory', 'system.protocol',
               'system.optional_capability']
    for hint in getattr(agent, '_tech_hints', None) or []:
        from infinidev.prompts.tech import get_tech_prompt
        parts.append(get_tech_prompt(hint))
        origins.append('system.tech_hint')
    parts.append(render_project_instructions(
        workspace_path or getattr(agent, 'workspace_path', None)))
    origins.append('system.project_instructions')
    summaries = getattr(agent, '_session_summaries', None) or []
    if summaries:
        parts.append('<session_context>\n' + '\n'.join(summaries) + '\n</session_context>')
        origins.append('system.session_summary')
    rendered = '\n\n'.join(part for part in parts if part)
    recorder = current_recorder()
    if recorder is not None:
        sources = [recorder.source(origin, 'system', part)
                   for origin, part in zip(origins, parts) if part]
        recorder.bundle('system', 'unknown', 'system', rendered, sources)
    return rendered


def _compact(messages: list[dict[str, Any]]) -> None:
    references = {id(message): message.get('_archive_reference') for message in messages}
    action_contents = {id(message): message.get('content') for message in messages
                       if message.get('_action_metadata')}
    latest = next((message for message in reversed(messages)
                   if message.get('role') == 'assistant'), {})
    manual_content = latest['content'] if latest.get('_manual_action') else None
    manual_results = [message for message in messages if message.get('_tool_result')]
    for message in manual_results:
        message['role'] = 'tool'
    ContextManager.compact_for_pressure(messages)
    if manual_content is not None:
        # The current observation needs the full action that produced it.
        latest['content'] = manual_content
    for message in manual_results:
        message['role'] = 'user'
    for message in messages:
        reference = references.get(id(message))
        if reference and reference not in message.get('content', ''):
            message['content'] += '\n' + reference
        declaration = message.get('_action_metadata')
        if (declaration and message.get('role') == 'assistant'
                and message.get('content') != action_contents.get(id(message))
                and not message['content'].endswith('\n' + declaration)):
            # Manual actions live in content; restore only their bounded declarations
            # after every compactor, without reviving large executable inputs.
            message['content'] += '\n' + declaration


def _stored_receipt(conversation: Any) -> dict[str, Any]:
    receipt = conversation.get('receipt') if isinstance(conversation, dict) else None
    return receipt if isinstance(receipt, dict) else {}


def _load_state(session_id: str) -> tuple[TurnState, list[str]]:
    previous_run_id = None
    try:
        data = get_session_runtime_state(session_id).get('ui_state', {}).get('conversation', {})
        receipt = _stored_receipt(data)
        if isinstance(run_id := receipt.get('run_id'), str):
            previous_run_id = run_id
        references = data.get('evidence', [])
        references = [value for value in references[-40:]
                      if isinstance(value, str) and ' turn_provenance_' not in value]
        state = TurnState.model_validate({
            **data, 'evidence_sources': {}, 'previous_interruption': None,
            'previous_checkpoint': None, 'previous_run_id': previous_run_id,
            'previous_test_observation': None,
        })
        try:
            if receipt.get('closure') == 'interrupted':
                state.previous_interruption = PreviousInterruption.model_validate(receipt)
            elif receipt.get('closure') == 'unclosed':
                state.previous_checkpoint = PreviousCheckpoint.model_validate(receipt)
        except ValidationError:
            pass  # Older receipts do not record a terminal status.
        if isinstance(observation := data.get('test_observation'), dict):
            try:
                state.previous_test_observation = TestObservation.model_validate(observation)
            except ValidationError:
                pass
        sources = data.get('evidence_sources')
        if isinstance(sources, dict):
            for ref in references:
                if ref not in sources:
                    continue
                try:
                    state.evidence_sources[ref] = EvidenceSource.model_validate(sources[ref])
                except ValidationError:
                    continue
        return state, references
    except (ValidationError, TypeError, ValueError):
        logger.warning('Ignoring invalid persisted conversation state', exc_info=True)
        return TurnState(previous_run_id=previous_run_id), []


def _reference_for_tools(reference: str, tool_names: Collection[str]) -> str:
    """Project only host-owned archive envelopes; their identifiers stay opaque."""
    for field in ('reference', 'query'):
        for instruction in ('recall_context ', ''):
            prefix = f'[Archived evidence; {instruction}{field}: '
            if reference.startswith(prefix):
                complete = reference.endswith(']')
                identifier = reference[len(prefix):-1] if complete else reference[len(prefix):]
                available = 'recall_context ' if 'recall_context' in tool_names else ''
                projected = f'[Archived evidence; {available}{field}: {identifier}]'
                if complete and len(projected) <= 500:
                    return projected
                excerpt = f'[Archived evidence; {field} excerpt (incomplete): '
                return excerpt + identifier[:500 - len(excerpt) - 1] + ']'
    return reference[:500]


def _evidence_source(
    name: str, args: dict[str, Any], workspace: str | None,
    file_read: FileReadObservation | None = None,
) -> EvidenceSource:
    """Keep a bounded workspace-relative label, without recording command arguments."""
    target = args.get('file_path')
    path = None
    fingerprint = None
    if workspace and isinstance(target, str) and target:
        try:
            root = Path(os.path.abspath(workspace))
            requested = Path(target).expanduser()
            if not requested.is_absolute():
                requested = Path(os.path.normpath(root / requested))
            relative = requested.relative_to(root)
            if requested.resolve().is_relative_to(root.resolve()):
                path = redact(relative.as_posix())
                if (name == 'read_file' and file_read is not None
                        and len(path) <= 200 and path == relative.as_posix()
                        and Path(file_read.path) == requested):
                    fingerprint = file_read.fingerprint
        except (OSError, RuntimeError, ValueError):
            pass
    if path and len(path) > 200:
        path = path[:199] + '…'
    return EvidenceSource(tool=name[:100], requested_path=path, file_fingerprint=fingerprint)


def _workspace_file_revision(workspace: str | None, path: str) -> tuple[str, str] | None:
    if not workspace:
        return None
    try:
        root = Path(workspace).resolve(strict=True)
        target = (root / path).resolve(strict=True)
        relative = target.relative_to(root).as_posix()
        fingerprint = file_fingerprint(target.stat())
        return (relative, fingerprint) if fingerprint is not None else None
    except (OSError, RuntimeError, ValueError):
        return None


def _file_revision_status(
    source: EvidenceSource, workspace: str | None,
) -> Literal['changed', 'unavailable'] | None:
    if not source.requested_path or not source.file_fingerprint or not workspace:
        return None
    revision = _workspace_file_revision(workspace, source.requested_path)
    if revision is None:
        return 'unavailable'
    return 'changed' if revision[1] != source.file_fingerprint else None


def _output_recovery_revision(tracker: FileChangeTracker) -> str | None:
    with best_effort('output recovery workspace snapshot unavailable'):
        return tracker.workspace_revision()
    return None


def _recalled_file_source(
    sources: dict[str, EvidenceSource], reference: str, body: str,
) -> EvidenceSource | None:
    source = (sources.get(f'[Archived evidence; recall_context reference: {reference}]')
              or sources.get(f'[Archived evidence; reference: {reference}]'))
    if source is None or not source.file_fingerprint:
        return None
    try:
        page = json.loads(body)
    except ValueError:
        return None
    if isinstance(page, dict) and page.get('status') == 'ok' and page.get('reference') == reference:
        return source
    return None


def _user_content(text: str, attachments: list[Any] | None, vision: bool) -> Any:
    if not attachments:
        return text
    return (build_user_content if vision else mention_paths_as_text)(text, attachments)


def _drain_guidance(engine: Any, messages: list[dict[str, Any]], vision: bool,
                    retained: list[InjectedUserMessage] | None = None) -> bool:
    items = engine._user_message_injector._drain_items()
    if retained is not None:
        retained.extend(items)
    for item in items:
        messages.append({'role': 'user', 'content': _user_content(
            item.text, list(item.attachments), vision)})
    return bool(items)


def _protocol_payload(text: str) -> bool:
    """Recognize failed transport envelopes without interpreting user intent."""
    stripped = text.lstrip()
    if not stripped.startswith(('<tool_call>', '<function_call>', '<|tool_call',
                                '<function=', '{')):
        return False
    if any(format_type().detect(text) for format_type in get_registered_formats()
           if format_type.name != 'manual_json'):
        return True
    if any(marker in text for marker in (
        '<tool_call>', '<function_call>', '<|tool_call', '<function=',
    )):
        return True
    stripped = text.lstrip()
    if stripped.startswith(('{', '```')):
        return ('"tool_calls"' in text or ('"name"' in text and any(
            key in text for key in ('"arguments"', '"parameters"'))))
    return False


def _budget_report_text(result: LLMCallResult) -> str:
    if result.finish_reason not in {None, 'stop'} or result.tool_calls:
        return ''
    report = result.raw_content.strip()
    candidate = report
    fence, _, body = report.partition('\n')
    if fence.casefold() in {'```', '```json'} and body.endswith('\n```'):
        inner = body.removesuffix('\n```')
        if '\n```' not in inner:
            candidate = inner.strip()
    if _protocol_payload(candidate) or candidate.startswith('[Tool:'):
        return ''
    return report


def _manual_actions(content: str) -> list[ManualToolCall] | None:
    """Decode only a complete, visible manual action envelope."""
    text = content.strip()
    payloads = []
    try:
        if text.startswith('<tool_call>'):
            decoder = json.JSONDecoder()
            while text:
                if not text.startswith('<tool_call>'):
                    return None
                value, end = decoder.raw_decode(text[len('<tool_call>'):].lstrip())
                remainder = text[len('<tool_call>'):].lstrip()[end:].lstrip()
                if not remainder.startswith('</tool_call>'):
                    return None
                payloads.append(value)
                text = remainder[len('</tool_call>'):].strip()
        else:
            value = json.loads(text)
            if not isinstance(value, dict) or not isinstance(value.get('tool_calls'), list):
                return None
            payloads = value['tool_calls']
    except (ValueError, TypeError):
        return None
    calls = []
    for value in payloads:
        if not isinstance(value, dict):
            return None
        function = value.get('function', value)
        if not isinstance(function, dict) or not isinstance(function.get('name'), str):
            return None
        arguments = function.get('arguments', {})
        calls.append(ManualToolCall(id='turn_' + uuid.uuid4().hex, name=function['name'],
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments)))
    return calls or None


def _visible_message(message: Any) -> tuple[str, Any]:
    """Separate only leading reasoning frames; quoted body data stays opaque."""
    raw = normalize_message_text(message.content)
    remaining = raw.lstrip()
    reasoning = []
    pairs = (('<think>', '</think>'), ('<thinking>', '</thinking>'),
             ('<thoughts>', '</thoughts>'))
    while True:
        pair = next((pair for pair in pairs if remaining.startswith(pair[0])), None)
        if pair is None:
            break
        opening, closing = pair
        end = remaining.find(closing, len(opening))
        if end < 0:
            return '', message
        reasoning.append(remaining[len(opening):end].strip())
        remaining = remaining[end + len(closing):].lstrip()
    if not reasoning:
        return raw, message
    fields = reasoning_history_fields(message)
    fields['reasoning_content'] = '\n\n'.join(part for part in
        [extract_reasoning(message).text, *reasoning] if part)
    return remaining, SimpleNamespace(content=remaining, **fields)


def _call(
    ctx: Any, caller: LLMCaller, messages: list[dict[str, Any]],
    *, tool_choice: str | dict[str, Any],
    named_choice_fallback: Literal['required', 'auto'],
) -> LLMCallResult:
    reporting_only = tool_choice == 'none'
    schemas = None if ctx.manual_tc or reporting_only else ctx.tool_schemas
    recorder = current_recorder()
    if recorder is not None:
        recorder.request(ctx.recording_run_id, ctx.recording_iteration, messages,
                         [] if reporting_only else ctx.tool_schemas,
                         mode='manual' if ctx.manual_tc else 'function_calling')
    measurement = [{k: v for k, v in message.items()
                    if k not in {'_recording_source', '_action_metadata'}}
                   for message in messages]
    caller._record_request_payload(ctx, measurement, schemas, mode=(
        'manual' if ctx.manual_tc else 'function_calling'))
    provider_messages = [{key: value for key, value in message.items()
                          if not key.startswith('_')} for message in messages]
    response = call_llm(ctx.llm_params, provider_messages, schemas, tool_choice=tool_choice,
        named_choice_fallback=named_choice_fallback,
        retry_attempts=1 if reporting_only else None,
        use_json_mode=False,
        on_request_metadata=ctx.on_request_metadata,
        on_thinking_chunk=caller._on_thinking_chunk,
        on_stream_status=caller._on_stream_status)
    caller._track_usage(ctx, response)
    usage = getattr(response, 'usage', None)
    def provided(*names: str) -> bool:
        for name in names:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if type(value) is int and value >= 0:
                return True
        return False
    ctx.status_usage_fields = (provided('prompt_tokens', 'input_tokens'),
                               provided('completion_tokens', 'output_tokens'))
    choice = response.choices[0]
    message = choice.message
    calls = getattr(message, 'tool_calls', None)
    origin = 'native' if calls else 'none'
    finish_reason = getattr(choice, 'finish_reason', None)
    if ctx.manual_tc and not calls and finish_reason != 'length':
        calls = _manual_actions(normalize_message_text(message.content))
        if calls:
            origin = 'parsed_text'
    if origin == 'parsed_text':
        # Arguments in the action channel are opaque, including literal think tags.
        content = normalize_message_text(message.content)
    else:
        content, message = _visible_message(message)
        if ctx.manual_tc and not calls and finish_reason != 'length':
            calls = _manual_actions(content)
            if calls:
                origin = 'parsed_text'
    return LLMCallResult(message=message, tool_calls=calls,
        raw_content=content,
        reasoning_content=extract_reasoning(message).text, finish_reason=finish_reason,
        tool_call_origin=origin)


class _ToolNotOffered(ValueError):
    """An exact name is absent from this request's dispatch snapshot."""

    def __init__(self, name: str, offered_names: tuple[str, ...]) -> None:
        super().__init__('Requested tool is not offered for this turn.')
        self.name = name
        self.offered_names = offered_names


def _tool_name_details(name: str, offered_names: tuple[str, ...]) -> dict[str, Any]:
    """Bound diagnostics without presenting a partial catalog as complete."""
    names: list[str] = []
    size = 2  # JSON array brackets.
    for offered in offered_names:
        added = len(json.dumps(offered, ensure_ascii=False).encode('utf-8'))
        added += 2 if names else 0
        if size + added > 8192:
            break
        names.append(offered)
        size += added
    return {'requested_name': name[:100], 'requested_name_complete': len(name) <= 100,
            'offered_names': names, 'offered_names_complete': len(names) == len(offered_names),
            'offered_name_count': len(offered_names)}


def _validated_arguments(ctx: Any, call: Any) -> dict[str, Any]:
    name = call.function.name
    if not isinstance(name, str):
        raise ValueError('Tool name must be a string')
    tool = ctx.tool_dispatch.get(name)
    if tool is None:
        raise _ToolNotOffered(name, tuple(ctx.tool_dispatch))
    raw = call.function.arguments
    args = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(args, dict):
        raise ValueError('Tool arguments must be a JSON object')
    if getattr(ctx, 'tool_call_metadata', False):
        args = ActionEnvelope.model_validate(args).input
    schema = getattr(tool, 'args_schema', None)
    if schema is not None:
        unknown = set(args) - set(schema.model_fields)
        if unknown:
            raise ValidationError.from_exception_data(schema.__name__, [
                {'type': 'extra_forbidden', 'loc': (key,), 'input': args[key]}
                for key in sorted(unknown)])
        args = schema.model_validate(args).model_dump()
    return args


def _confirmed_write_target(name: str, body: str, workspace: str | None) -> str | None:
    expected = {'create_file': 'created', 'edit_file': 'modified'}.get(name)
    if expected is None:
        return None
    try:
        result = json.loads(body)
    except ValueError:
        return None
    if not (
        isinstance(result, dict) and 'error' not in result
        and result.get('action') == expected
        and isinstance(result.get('file_path'), str) and bool(result['file_path'])
    ):
        return None
    revision = _workspace_file_revision(workspace, result['file_path'])
    if revision is None:
        return None
    path = revision[0]
    return path if redact(path) == path else None


def _action_message(ctx: Any, result: LLMCallResult, calls: list[Any]) -> dict[str, Any]:
    """Retain the action channel in the transcript for either transport."""
    message = {'role': 'assistant', **reasoning_history_fields(result.message)}
    if ctx.manual_tc:
        actions = []
        for call in calls:
            arguments = call.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            actions.append({'name': call.function.name, 'arguments': arguments})
        message['content'] = json.dumps({'tool_calls': actions}, ensure_ascii=False)
        message['_manual_action'] = True
    else:
        message['content'] = result.raw_content or None
        message['tool_calls'] = [dict(id=call.id, type='function', function=dict(
            name=call.function.name, arguments=call.function.arguments
            if isinstance(call.function.arguments, str) else json.dumps(call.function.arguments)))
            for call in calls]
    return message


def run_turn(engine: Any, agent: Any, user_input: str, *, turn_context: str = '',
             initial_attachments: list[Any] | None = None, task_tools: list[Any] | None = None,
             max_iterations: int | None = None, max_total_tool_calls: int | None = None,
             prompt_configuration: Any = None, preserve_file_tracker: bool = False,
             on_public_message: Callable[[str], None] | None = None,
             _parent_guidance: list[InjectedUserMessage] | None = None,
             tool_call_metadata: bool = False) -> TurnResult:
    """Execute one bounded conversation without creating or advancing a work Task.

    Cancellation belongs to the host turn: this method never clears it. Zero limits
    mean unlimited. Each invocation accounts for its own usage, including retries.
    Only host hook continuations preserve the existing file tracker and baseline.
    Internal _parent_guidance retains original human ingress only within that
    parent; it is neither model state nor a queue of new input.
    """
    turn_started = time.monotonic()
    legacy_state = LoopState()  # Usage and existing ToolContext compatibility only.
    state = TurnState()
    references: list[str] = []
    public_messages: list[str] = []
    ctx = None
    iterations = 0
    invalid_batches = 0
    validation_diagnostics: list[str] = []
    unclosed_responses = 0
    receipt = {'run_id': uuid.uuid4().hex, 'closure': 'interrupted', 'dispatches': 0}
    verification: TurnTestVerification | None = None
    verification_dirty = False
    observer = current_observer()
    if observer is not None:
        observer.record('preparing', 'child_started', child_run_id=receipt['run_id'])
    tool_context = get_context_for_agent(agent.agent_id)
    session_id = tool_context.session_id or agent.agent_id
    workspace_path = tool_context.workspace_path or getattr(agent, 'workspace_path', None)

    def trace(event: str, **data: Any) -> None:
        recorder = current_recorder()
        if recorder is not None:
            recorder.event(receipt['run_id'], iterations, event, data)
        if observer is not None:
            if event in {'runtime_request', 'provider_request'}:
                observer.record('inference', 'model_request' if event == 'runtime_request'
                                else 'provider_request')
            elif event == 'model_response':
                prompt, completion = getattr(ctx, 'status_usage_fields', (False, False))
                observer.record('working', 'model_response', progress=True,
                    model_elapsed_ms=data.get('elapsed_ms'),
                    prompt_tokens=data.get('observed_prompt_tokens') if prompt else None,
                    completion_tokens=data.get('observed_completion_tokens') if completion else None)
            elif event == 'turn_closed':
                observer.record('working', 'child_run_closed')
        if ctx is None:
            return
        body = json.dumps({'trace': 'turn_provenance', 'run_id': receipt['run_id'],
                           'iteration': iterations, 'event': event, **data})
        _archive(ctx, iterations, 'turn_provenance_' + receipt['run_id'] + '_' + event,
                 {}, body, kind='runtime_trace')
    max_iterations = settings.LOOP_MAX_ITERATIONS if max_iterations is None else max_iterations
    max_calls = (settings.LOOP_MAX_TOTAL_TOOL_CALLS if max_total_tool_calls is None
                 else max_total_tool_calls)

    def retain_evidence(reference: str, source: EvidenceSource | None = None) -> None:
        references.append(reference)
        if source is not None:
            state.evidence_sources[reference] = source
        if len(references) > 40:
            state.evidence_sources.pop(references.pop(0), None)

    def owns_work_snapshot(current: Any) -> bool:
        # Another run may have saved newer progress while this run was in flight.
        stored = _stored_receipt(current)
        stored_run_id = stored.get('run_id')
        if stored_run_id == receipt['run_id']:
            return stored.get('closure') == 'unclosed'
        return stored_run_id == state.previous_run_id

    def persist_work(*, checkpoint: bool = False) -> None:
        saved_receipt = {**receipt, 'closure': 'unclosed'} if checkpoint else receipt
        try:
            saved = {
                **_bounded_state(state), 'evidence': references[-40:], 'receipt': saved_receipt,
                'evidence_sources': {ref: source.model_dump(exclude_none=True)
                                     for ref, source in state.evidence_sources.items()},
            }
            if (
                verification is not None and verification.observation is not None
                and not engine.is_cancelled
            ):
                saved['test_observation'] = verification.observation.model_dump(exclude_none=True)
            _persist_session_runtime_field(session_id, 'conversation',
                saved, replace_if=owns_work_snapshot)
        except Exception:
            logger.warning('Conversation state persistence failed', exc_info=True)

    def outcome(status: str, message: str) -> TurnResult:
        if status != 'answered' and validation_diagnostics:
            message += '\nRejected tool batches (no calls executed in these batches):\n'
            message += '\n'.join(validation_diagnostics)
        if engine._last_file_tracker is not None:
            engine._last_file_tracker.reconcile_workspace()
        if verification is not None and verification.status is not None:
            if status == 'cancelled':
                receipt['test_verification'] = 'unconfirmed'
            else:
                verification.refresh()
                receipt['test_verification'] = verification.status
            if verification.from_prior_turn:
                receipt['test_verification_origin'] = 'prior_turn'
            receipt['test_revision_snapshot_ms'] = verification.snapshot_ns // 1_000_000
        engine._last_state = legacy_state
        engine._last_status = status
        engine._last_total_tool_calls = legacy_state.total_tool_calls
        receipt['status'] = status
        engine._last_turn_receipt = dict(receipt)
        trace('turn_closed',
              message_sha256=hashlib.sha256(message.encode()).hexdigest(), **receipt)
        # Preflight failure or cancellation must leave the work checkpoint available.
        if iterations:
            persist_work()
        return TurnResult(message=message, status=status, state=state, engine=engine,
                          public_messages=list(public_messages), metrics={
            'observed_iterations': iterations,
            'observed_tool_calls': legacy_state.total_tool_calls,
            'observed_prompt_tokens': legacy_state.total_prompt_tokens,
            'observed_completion_tokens': legacy_state.total_completion_tokens,
            'tool_calls': legacy_state.total_tool_calls,
        })

    def publish(content: str) -> None:
        if not content.strip():
            return
        if on_public_message is not None:
            on_public_message(content)
        public_messages.append(content)
        trace('public_message', channel='assistant_content', ordinal=len(public_messages),
              callback_delivered=on_public_message is not None,
              content_bytes=len(content.encode()),
              content_sha256=hashlib.sha256(content.encode()).hexdigest())

    try:
        if engine.is_cancelled:
            return outcome('cancelled', 'Conversation cancelled.')
        state, references = _load_state(session_id)
        params = get_litellm_params()
        if params is None:
            raise RuntimeError('No language model is configured')
        post_tool_choice = settings.TASK_POST_TOOL_CHOICE
        if post_tool_choice is None:
            post_tool_choice = ('auto' if settings.LLM_PROVIDER == 'openai_compatible'
                                else 'required')
        capabilities = get_model_capabilities()
        manual = not capabilities.supports_function_calling
        vision = bool(getattr(capabilities, 'supports_vision', False))
        tools = list(task_tools if task_tools is not None else getattr(agent, 'tools', []) or [])
        tools = effective_tools(tools)
        tools.extend(build_turn_tools(state))
        bind_tools_to_agent(tools, agent.agent_id)
        schemas = [tool_to_openai_schema(tool) for tool in tools]
        if tool_call_metadata:
            schemas = [transport_schema(schema) for schema in schemas]
        if getattr(capabilities, 'needs_schema_sanitization', False):
            from infinidev.engine.schema_sanitizer import _sanitize_tool_schema
            schemas = [_sanitize_tool_schema(schema) for schema in schemas]
        system = _system_prompt(agent, prompt_configuration, workspace_path=workspace_path)
        if manual:
            system = _append_manual_tools(system, schemas, tool_call_metadata=tool_call_metadata)
        baseline_started = time.monotonic()
        tracker = engine._last_file_tracker if preserve_file_tracker else None
        if tracker is None:
            tracker = FileChangeTracker(baseline=WorkspaceBaseline.capture(workspace_path))
        baseline_elapsed_ms = int((time.monotonic() - baseline_started) * 1000)
        engine._last_file_tracker = tracker
        if settings.TASK_TEST_REVISION_FEEDBACK_ENABLED:
            verification = TurnTestVerification(tracker)
            if state.previous_test_observation is not None:
                verification.restore(state.previous_test_observation)
        set_loop_state(agent.agent_id, legacy_state)
        set_file_tracker(agent.agent_id, tracker)
        ctx = SimpleNamespace(agent=agent, agent_id=agent.agent_id, project_id=agent.project_id,
            session_id=session_id, llm_params=params, manual_tc=manual, is_small=False,
            state=legacy_state, tools=tools, tool_schemas=schemas,
            tool_dispatch=build_tool_dispatch(tools), system_prompt=system,
            tool_call_metadata=tool_call_metadata)
        readback_available = (
            settings.TASK_FIRST_WRITE_READBACK_ENABLED
            and not settings.SANDBOX_ENABLED
            and not manual
            and not getattr(capabilities, 'restricts_tool_choice_to_auto', False)
            and provider_for_request(params, settings.LLM_PROVIDER) == 'openai_compatible'
            and 'read_file' in ctx.tool_dispatch
        )
        readback_pending: str | None = None
        recorder = current_recorder()
        ctx.recording_run_id = receipt['run_id']
        ctx.recording_iteration = 0
        caller = LLMCaller(
            on_thinking_chunk=lambda text: emit_loop_event('loop_thinking_chunk',
                agent.project_id, agent.agent_id, {'text': text}),
            on_stream_status=lambda phase, count, name: emit_loop_event('loop_stream_status',
                agent.project_id, agent.agent_id,
                {'phase': phase, 'token_count': count, 'tool_name': name}))
        messages = [{'role': 'system', 'content': system}]
        if recorder is not None:
            recorder.attach(messages[0], recorder.binding('system', system))
        if turn_context:
            messages.append({'role': 'user', 'content':
                '<conversation_context>\n' + turn_context + '\n</conversation_context>'})
            if recorder is not None:
                parent = recorder.binding('turn_context', turn_context)
                node = recorder.bundle('conversation_message', 'conversation_context', 'user',
                                       messages[-1]['content'], [parent] if parent else [])
                recorder.attach(messages[-1], node)
        prior = _bounded_state(state)
        if state.previous_interruption is not None:
            prior['previous_interruption'] = state.previous_interruption.model_dump()
        if state.previous_checkpoint is not None:
            prior['previous_checkpoint'] = state.previous_checkpoint.model_dump()
        if verification is not None and verification.from_prior_turn:
            prior_observation = {
                'revision_status': verification.status,
                'recognized_test_command_exit_code': 0,
                'launch_cwd': state.previous_test_observation.launch_cwd,
            }
            locator = state.previous_test_observation.locator
            if locator and 'recall_context' in ctx.tool_dispatch:
                prior_observation['locator'] = _reference_for_tools(locator, ctx.tool_dispatch)
            prior['previous_test_observation'] = prior_observation
        if references or any(prior.values()):
            evidence = []
            for ref in references:
                locator = _reference_for_tools(ref, ctx.tool_dispatch)
                source = state.evidence_sources.get(ref)
                if source is not None and 'recall_context' in ctx.tool_dispatch:
                    label = source.model_dump(exclude={'file_fingerprint'}, exclude_none=True)
                    if status := _file_revision_status(source, workspace_path):
                        label['file_revision_status'] = status
                    evidence.append({'locator': locator, **label})
                else:
                    evidence.append(locator)
            prior['evidence'] = evidence
            prior_json = json.dumps(prior, ensure_ascii=False).replace('<', r'\u003c')
            messages.append({'role': 'user', 'content':
                '<prior_state_context>\n' + prior_json + '\n</prior_state_context>'})
            if recorder is not None:
                recorder.attach(messages[-1], recorder.source(
                    'prior_state', 'user', messages[-1]['content']))
        messages.append({'role': 'user', 'content': _user_content(
            user_input, initial_attachments, vision)})
        if recorder is not None:
            recorder.attach(messages[-1], recorder.source(
                'literal_input', 'user', messages[-1]['content']))
        # Replayed ingress stays after the original request; pending input is newer.
        for item in tuple(_parent_guidance or ()):
            messages.append({'role': 'user', 'content': _user_content(
                item.text, list(item.attachments), vision)})
        engine._last_turn_messages = messages
        window = _get_model_max_context(params)
        trace('turn_setup', scope='run_turn',
              elapsed_ms=int((time.monotonic() - turn_started) * 1000),
              baseline_elapsed_ms=baseline_elapsed_ms,
              baseline_available=tracker.baseline is not None,
              baseline_files=len(tracker.baseline.files) if tracker.baseline else 0)
        context_repaired = False
        output_retries = 0
        first_cutoff_revision: str | None = None
        workspace_effect_since_cutoff = False
        reported_archive_revisions: dict[str, str] = {}
        while True:
            if engine.is_cancelled:
                return outcome('cancelled', 'Conversation cancelled.')
            if _drain_guidance(engine, messages, vision, _parent_guidance):
                readback_pending = None
                verification_dirty = True
            if max_iterations and iterations >= max_iterations:
                return outcome('exhausted', 'Conversation interrupted: model call limit reached.')
            reporting_only = bool(max_calls and legacy_state.total_tool_calls >= max_calls)
            if reporting_only and (invalid_batches or not receipt['dispatches']):
                return outcome('exhausted', _TOOL_LIMIT_MESSAGE)
            if not context_repaired and ContextManager.under_context_pressure(
                legacy_state.last_prompt_tokens, window,
            ):
                _compact(messages)
                readback_pending = None
            if verification is not None and verification_dirty:
                verification_dirty = False
                if verification.refresh():
                    notice = {'role': 'user', 'content':
                        'Host verification: the workspace changed after the last recognized '
                        'test command returned exit 0. '
                        'Rerun the relevant check if authorized and needed for your report.'}
                    messages.append(notice)
                    if recorder is not None:
                        recorder.attach(notice, recorder.source(
                            'workspace_verification', 'user', notice['content']))
                    trace('test_verification_notice', status='stale')
            iterations += 1
            ctx.recording_iteration = iterations
            tool_choice: str | dict[str, Any] = (
                post_tool_choice if legacy_state.total_tool_calls else 'auto'
            )
            if reporting_only:
                tool_choice = 'none'
                messages[0]['content'] += ('\n\nWhen the host requests a budget report, answer '
                                          'in plain text without calling tools.')
                notice = {'role': 'user', 'content':
                          'Host: the tool call limit has been reached. Summarize the work and '
                          'results already observed, including unfinished work. No tools are '
                          'available for this report.'}
                messages.append(notice)
                if recorder is not None:
                    recorder.attach(messages[0], recorder.source(
                        'budget_report_system', 'system', messages[0]['content']))
                    recorder.attach(notice, recorder.source(
                        'tool_budget_report', 'user', notice['content']))
            elif readback_pending is not None:
                if (
                    not ctx.manual_tc
                    and (not max_iterations or iterations < max_iterations)
                    and (not max_calls or legacy_state.total_tool_calls + 1 < max_calls)
                ):
                    tool_choice = {'type': 'function', 'function': {'name': 'read_file'}}
                    target = json.dumps({'file_path': readback_pending}, ensure_ascii=False)
                    target = target.replace('<', r'\u003c')
                    notice = {'role': 'user', 'content': '<write_readback_context>\n' + target
                              + '\n</write_readback_context>'}
                    messages.append(notice)
                    if recorder is not None:
                        recorder.attach(notice, recorder.source(
                            'write_readback_context', 'user', notice['content']))
                readback_pending = None
            trace('runtime_request', mode='manual' if ctx.manual_tc else 'function_calling',
                  requested_tool_choice=tool_choice,
                  finish_turn_offered=not reporting_only and any(
                      s['function']['name'] == 'finish_turn' for s in ctx.tool_schemas),
                  message_roles=[m['role'] for m in messages], reminder=unclosed_responses)
            ctx.on_request_metadata = lambda metadata: trace('provider_request', **metadata)
            usage_before = (legacy_state.total_prompt_tokens, legacy_state.total_completion_tokens)
            call_started = time.monotonic()
            try:
                result = _call(ctx, caller, messages, tool_choice=tool_choice,
                               named_choice_fallback=post_tool_choice)
            except Exception as exc:
                if engine.is_cancelled:
                    return outcome('cancelled', 'Conversation cancelled.')
                if reporting_only:
                    trace('tool_budget_report_failed', error_type=type(exc).__name__)
                    return outcome('exhausted', _TOOL_LIMIT_MESSAGE)
                if is_context_window_rejection(exc):
                    if (
                        not context_repaired
                        and 'recall_context' in ctx.tool_dispatch
                        and (not max_iterations or iterations < max_iterations)
                    ):
                        shortened = shorten_archived_results(messages)
                        if shortened:
                            context_repaired = True
                            trace('context_window_recovery', archived_results=shortened)
                            continue
                    raise
                if not ctx.manual_tc and any(p in str(exc).lower() for p in PERMANENT_ERRORS):
                    ctx.manual_tc = True
                    messages[0]['content'] = _append_manual_tools(messages[0]['content'], schemas,
                        tool_call_metadata=tool_call_metadata)
                    recorder = current_recorder()
                    if recorder is not None:
                        recorder.attach(messages[0], recorder.binding('system', messages[0]['content']))
                    continue
                raise
            context_repaired = False
            if engine.is_cancelled:
                return outcome('cancelled', 'Conversation cancelled.')
            output_truncated = result.finish_reason == 'length'
            reported_calls = [] if output_truncated else (result.tool_calls or [])
            trace('model_response', source='model', origin=result.tool_call_origin,
                  elapsed_ms=int((time.monotonic() - call_started) * 1000),
                  observed_prompt_tokens=legacy_state.total_prompt_tokens - usage_before[0],
                  observed_completion_tokens=legacy_state.total_completion_tokens - usage_before[1],
                  finish_reason=result.finish_reason,
                  content_sha256=hashlib.sha256(result.raw_content.encode()).hexdigest(),
                  content_bytes=len(result.raw_content.encode()),
                  calls=[{'id_sha256': hashlib.sha256(call.id.encode()).hexdigest(),
                          'tool': call.function.name}
                         for call in reported_calls])
            if reporting_only:
                report = ('' if engine._user_message_injector.has_pending
                          else _budget_report_text(result))
                return outcome('exhausted', _TOOL_LIMIT_MESSAGE + ('\n\n' + report if report else ''))
            if output_truncated:
                if output_retries >= 2 or (max_iterations and iterations >= max_iterations):
                    return outcome('exhausted', _OUTPUT_LIMIT_MESSAGE)
                if output_retries == 0:
                    first_cutoff_revision = _output_recovery_revision(tracker)
                else:
                    if not workspace_effect_since_cutoff or first_cutoff_revision is None:
                        return outcome('exhausted', _OUTPUT_LIMIT_MESSAGE)
                    revision = _output_recovery_revision(tracker)
                    if revision is None or revision == first_cutoff_revision:
                        return outcome('exhausted', _OUTPUT_LIMIT_MESSAGE)
                output_retries += 1
                guidance = ('Host: the previous response hit the model output limit and was '
                            'discarded. No action from that response ran. Keep the next response '
                            'short. If a tool is needed, send one complete call with fewer '
                            'arguments or a smaller edit.')
                if output_retries == 2:
                    guidance += (' If the work is ready to report, call finish_turn by itself '
                                 'with a brief account of the results already observed.')
                notice = {'role': 'user', 'content': guidance}
                messages.append(notice)
                if recorder is not None:
                    recorder.attach(notice, recorder.source(
                        'output_limit_recovery', 'user', notice['content']))
                trace('output_limit_recovery', reason='length', retry=output_retries)
                continue
            from infinidev.engine.hooks.hooks import HookContext, HookEvent, hook_manager
            hook_manager.dispatch(HookContext(
                event=HookEvent.POST_MODEL_MESSAGE,
                project_id=agent.project_id, agent_id=agent.agent_id,
                metadata={'raw_content': result.raw_content,
                          'reasoning_content': result.reasoning_content,
                          'tool_calls': result.tool_calls, 'messages': messages,
                          'agent_name': getattr(agent, 'role', agent.agent_id),
                          'task': user_input, 'plan_snapshot': _bounded_state(state)},
            ))
            if engine.is_cancelled:
                return outcome('cancelled', 'Conversation cancelled.')
            calls = result.tool_calls or []
            if not calls:
                if _drain_guidance(engine, messages, vision, _parent_guidance):
                    continue
                if legacy_state.total_tool_calls:
                    if not ctx.manual_tc or not _protocol_payload(result.raw_content):
                        publish(result.raw_content)
                    unclosed_responses += 1
                    if unclosed_responses >= 3:
                        return outcome('failed', 'Conversation interrupted: explicit '
                                       'finish_turn missing after tool attempts.')
                    messages.append({'role': 'assistant', 'content': result.raw_content,
                                     **reasoning_history_fields(result.message)})
                    messages.append({'role': 'user', 'content':
                        'Host protocol: your last message was model-authored text, not a tool '
                        'result, and did not close this turn. Continue the pending work or call '
                        'finish_turn with an honest summary of observed results and blockers.'})
                    trace('protocol_reminder', number=unclosed_responses,
                          reason='explicit_close_required')
                    continue
                if not result.raw_content.strip() or (
                    ctx.manual_tc and _protocol_payload(result.raw_content)):
                    return outcome('failed', 'Conversation interrupted: invalid model response.')
                receipt['closure'] = 'plain_text'
                return outcome('answered', result.raw_content)
            unclosed_responses = 0
            mixed_close = len(calls) > 1 and any(call.function.name == 'finish_turn' for call in calls)
            close_deferred = (not mixed_close and calls[0].function.name == 'finish_turn'
                              and engine._user_message_injector.has_pending)
            if result.tool_call_origin == 'native' and not (close_deferred or mixed_close):
                publish(result.raw_content)
            # Validate the entire transport batch before executing any of it.
            validated = []
            errors = {}
            unoffered = {}
            for call in calls:
                if mixed_close and call.function.name == 'finish_turn':
                    errors[call.id] = [{'type': 'finish_turn_requires_sole_call', 'loc': ('batch',),
                                        'msg': 'finish_turn must be the only call in its batch.'}]
                    continue
                try:
                    validated.append((call, _validated_arguments(ctx, call)))
                except _ToolNotOffered as exc:
                    unoffered[call.id] = exc
                    errors[call.id] = [{'type': 'tool_not_offered', 'loc': ('tool_name',),
                                        'msg': str(exc)}]
                except (ValueError, TypeError) as exc:
                    errors[call.id] = (exc.errors(include_input=False, include_context=False,
                                                include_url=False)
                                       if isinstance(exc, ValidationError) else
                                       [{'type': 'invalid_arguments',
                                         'msg': str(exc)[:500]}])
            if errors:
                invalid_batches += 1
                legacy_state.total_tool_calls += len(calls)
                # Persist only structural diagnostics, never arguments or model prose.
                rejected = [{'tool': call.function.name[:100], 'errors': [
                    {'field': str(problem.get('loc', ('<arguments>',))[0])[:100]
                              if problem.get('loc') else '<arguments>',
                     'type': problem['type']}
                    for problem in errors.get(call.id, [])]}
                    for call in calls]
                for call, item in zip(calls, rejected):
                    if call.id in unoffered:
                        failure = unoffered[call.id]
                        item.update(_tool_name_details(failure.name, failure.offered_names))
                report = json.dumps({'event': 'tool_validation_rejected',
                    'batch': iterations, 'executed': False, 'calls': rejected}, ensure_ascii=False)
                recorder = current_recorder()
                if recorder is not None:
                    recorder.event(receipt['run_id'], iterations, 'tool_validation_rejected',
                                   {'calls': rejected, 'executed': False})
                reference = _archive(ctx, iterations, 'tool_validation_rejected', {}, report)
                if reference:
                    retain_evidence(reference)
                details = [f"{item['tool']}.{problem['field']} ({problem['type']})"
                           for item in rejected for problem in item['errors']]
                validation_diagnostics.append(
                    f"batch {iterations}: " + '; '.join(details[:20]) +
                    (f'\n{reference}' if reference else '\nDiagnostic archival unavailable.'))
                validation_diagnostics[:] = validation_diagnostics[-3:]
                messages.append(_action_message(ctx, result, calls))
                for call in calls:
                    recovery = 'Correct the invalid arguments using the tool schema and '
                    recovery += 'resubmit the pending calls. No call in this batch executed.'
                    details = {}
                    if unoffered:
                        offered = next(iter(unoffered.values())).offered_names
                        details = _tool_name_details(call.function.name, offered)
                        recovery = ('Use an exact offered tool name and its schema. '
                                    'If offered_names_complete is false, consult the request '
                                    'tool schemas for the full catalog. Correct any argument '
                                    'errors and resubmit the pending calls. '
                                    'No call in this batch executed.')
                    if mixed_close:
                        recovery = ('No call in this batch executed. Correct any argument or tool '
                                    'name errors and retry the pending work without finish_turn. '
                                    'After completing the work, call finish_turn by itself.')
                    body = json.dumps({'error': 'Tool batch rejected before execution.',
                        'executed': False, 'errors': errors.get(call.id, []),
                        **details, 'recovery': recovery}, ensure_ascii=False)
                    messages.append({'role': 'user', 'content':
                        f'[Tool: {call.function.name}] Result:\n{body}', '_tool_result': True}
                        if ctx.manual_tc else {'role': 'tool', 'tool_call_id': call.id,
                                               'content': body})
                if invalid_batches >= 3:
                    return outcome('failed',
                        'Conversation interrupted: tool validation failed in 3 batches.')
                continue
            invalid_batches = 0
            action = _action_message(ctx, result, calls)
            if tool_call_metadata and not close_deferred:
                declarations = [{'tool': call.function.name, 'metadata':
                    ActionEnvelope.model_validate_json(call.function.arguments).metadata.model_dump()
                    if isinstance(call.function.arguments, str) else
                    ActionEnvelope.model_validate(call.function.arguments).metadata.model_dump()}
                    for call in calls]
                declared = 'Model-declared tool call metadata (not host evidence):\n' + json.dumps(
                    declarations, ensure_ascii=False)
                publish(declared)
                if ctx.manual_tc:
                    action['_action_metadata'] = declared
            messages.append(action)
            pending_images = []
            recalled_files: dict[str, EvidenceSource] = {}
            readback_revisions: set[str] = set()
            for call, args in validated:
                if engine.is_cancelled:
                    return outcome('cancelled', 'Conversation cancelled.')
                reference = ''
                attachments: list[Any] = []
                if max_calls and legacy_state.total_tool_calls >= max_calls:
                    body = json.dumps({'status': 'tool_budget_exhausted', 'executed': False})
                elif call.function.name == 'finish_turn' and engine._user_message_injector.has_pending:
                    body = 'Turn close deferred because a new user message arrived. '
                    if close_deferred:
                        body += 'Accompanying assistant text was not published. '
                    body += 'Respond to the update.'
                else:
                    previous_plan = state.plan
                    tool = ctx.tool_dispatch[call.function.name]
                    workspace_effect = may_change_workspace(
                        call.function.name, getattr(tool, 'effects', None))
                    engine._begin_tool_batch()
                    try:
                        if engine.is_cancelled:
                            engine._tool_cancel_event.set()
                        # Count attempts, including host-denied calls, against the global fuse.
                        legacy_state.total_tool_calls += 1
                        command = args.get('command')
                        test_attempt = (
                            verification is not None and call.function.name == 'execute_command'
                            and isinstance(command, str) and is_test_command(command, legacy_state)
                        )
                        revision_before = verification.snapshot() if test_attempt else None
                        if observer is not None:
                            observer.record('tool', 'tool_started', tool_name=call.function.name)
                        with capture_file_reads() as file_reads:
                            body = execute_tool_call(ctx.tool_dispatch, call.function.name, args,
                                hook_metadata={'project_id': agent.project_id,
                                    'agent_id': agent.agent_id,
                                    'cancel_event': engine._tool_cancel_event},
                                attachments_out=attachments, file_tracker=tracker,
                                file_change_hooks=engine._hooks)
                        if readback_available:
                            readback_pending = _confirmed_write_target(
                                call.function.name, body, workspace_path)
                            readback_available = readback_pending is None
                        elif (readback_pending is not None and call.function.name == 'read_file'
                              and len(file_reads) == 1):
                            readback_revisions.add(file_reads[0].fingerprint)
                        if observer is not None:
                            observer.record('working', 'tool_completed', tool_name=call.function.name,
                                            progress=True)
                        if call.function.name not in {'finish_turn', 'update_plan'}:
                            receipt['dispatches'] += 1
                            trace('tool_result', source='dispatcher',
                                  call_id_sha256=hashlib.sha256(call.id.encode()).hexdigest(),
                                  tool=call.function.name,
                                  output_sha256=hashlib.sha256(body.encode()).hexdigest(),
                                  output_bytes=len(body.encode()))
                        if verification is not None:
                            if test_attempt:
                                verification.observe_test(
                                    None if engine.is_cancelled else revision_before, body)
                                verification_dirty = False
                            elif verification.status == 'current':
                                verification_dirty |= workspace_effect
                    finally:
                        engine._finish_tool_batch()
                    # Exact pages already identify their durable source. Re-archiving and
                    # appending a footer would invalidate the complete JSON response/budget.
                    recalled_ref = args.get('reference') if call.function.name == 'recall_context' else None
                    if isinstance(recalled_ref, str):
                        source = _recalled_file_source(state.evidence_sources, recalled_ref, body)
                        if source is not None:
                            recalled_files[recalled_ref] = source
                    reference = '' if recalled_ref is not None else _archive(
                        ctx, iterations, call.function.name, args, body)
                    if reference:
                        if call.function.name not in {'finish_turn', 'update_plan'}:
                            file_read = file_reads[0] if len(file_reads) == 1 and not engine.is_cancelled else None
                            retain_evidence(reference, _evidence_source(
                                call.function.name, args, workspace_path, file_read))
                    if test_attempt and verification.observation is not None:
                        verification.observation.locator = reference or None
                    if workspace_effect or state.plan is not previous_plan:
                        persist_work(checkpoint=True)
                    if output_retries == 1:
                        workspace_effect_since_cutoff |= workspace_effect
                archive_metadata = {'_archive_reference': reference, '_archive_characters': len(body)}
                if reference:
                    body += '\n' + reference
                if ctx.manual_tc:
                    messages.append({'role': 'user', 'content':
                        f'[Tool: {call.function.name}] Result:\n{body}',
                        **archive_metadata, '_tool_result': True})
                else:
                    messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': body,
                        **archive_metadata})
                if attachments:
                    pending_images.append({'role': 'user', 'content': _user_content(
                        'Tool image attachments', attachments, vision)})
                if engine.is_cancelled:
                    state.finished, state.reply = False, None
                    return outcome('cancelled', 'Conversation cancelled.')
                if state.finished:
                    if _drain_guidance(engine, messages, vision, _parent_guidance):
                        state.finished, state.reply = False, None
                    else:
                        receipt['closure'] = 'finish_turn'
                        return outcome('answered', state.reply or '')
            messages.extend(pending_images)
            if readback_pending is not None:
                revision = _workspace_file_revision(workspace_path, readback_pending)
                if revision is None or revision[1] in readback_revisions:
                    readback_pending = None
            revision_notices = []
            for recalled_ref, source in recalled_files.items():
                status = _file_revision_status(source, workspace_path)
                if status is not None and reported_archive_revisions.get(recalled_ref) != status:
                    reported_archive_revisions[recalled_ref] = status
                    revision_notices.append({'reference': recalled_ref,
                                             'requested_path': source.requested_path,
                                             'file_revision_status': status})
            if revision_notices:
                revision_json = json.dumps(revision_notices, ensure_ascii=False).replace('<', r'\u003c')
                notice = {'role': 'user', 'content':
                    '<archive_revision_context>\n' + revision_json + '\n</archive_revision_context>'}
                messages.append(notice)
                if recorder is not None:
                    recorder.attach(notice, recorder.source(
                        'archive_revision_context', 'user', notice['content']))
    except Exception as exc:
        logger.exception('Conversation runtime failed')
        return outcome('cancelled' if engine.is_cancelled else 'failed',
            'Conversation cancelled.' if engine.is_cancelled else f'Conversation interrupted: {exc}')


def _archive(ctx: Any, iteration: int, name: str, args: dict[str, Any], body: str,
             *, kind: str = 'tool_output') -> str:
    """Retain full evidence in bounded archive chunks before prompt compaction."""
    try:
        from infinidev.engine.working_memory import MAX_ARCHIVE_CHARS, get_working_memory
        memory = get_working_memory(ctx.session_id)
        titles = []
        archive_id = uuid.uuid4().hex
        if kind == 'tool_output' and name in getattr(ctx, 'tool_dispatch', {}):
            reference = memory.archive_output(
                f'turn {archive_id} result {iteration}/{ctx.state.total_tool_calls} {name}',
                body, step_index=iteration)
            if reference:
                return _reference_for_tools('[Archived evidence; reference: ' + reference + ']',
                                            ctx.tool_dispatch)
        for offset in range(0, len(body), MAX_ARCHIVE_CHARS):
            title = (f'turn {archive_id} result {iteration}/{ctx.state.total_tool_calls} '
                     f'{name} part {offset // MAX_ARCHIVE_CHARS + 1}')
            if memory.remember(title, body[offset:offset + MAX_ARCHIVE_CHARS],
                               kind=kind, step_index=iteration):
                titles.append(title)
        if titles:
            return _reference_for_tools('[Archived evidence; query: ' + titles[0] + ']',
                                        getattr(ctx, 'tool_dispatch', {}))
    except Exception:
        logger.warning('Conversation evidence archival failed', exc_info=True)
    return ''
