"""Opt-in, bounded metadata for one principal invocation; never prompt bodies."""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any
import uuid

from infinidev.config.secrets import redact
from infinidev.engine.history.redaction import redact_text

logger = logging.getLogger(__name__)
_CURRENT: contextvars.ContextVar[ContextRecorder | None] = contextvars.ContextVar(
    'passive_context_recording', default=None)
_ORIGINS = {
    'history', 'literal_input', 'conversation_context', 'prior_state', 'ken',
    'task_start', 'workspace', 'task_end_instruction', 'injected_guidance', 'delivery_verification',
    'manual_tools', 'system.identity', 'system.backstory', 'system.protocol',
    'system.optional_capability', 'system.tech_hint', 'system.project_instructions',
    'system.session_summary', 'unknown',
}
_ROLES = {'system', 'user', 'assistant', 'tool'}
_KNOWN_ROLES = _ROLES | {'developer', 'work_summary'}
_PRIVATE_ORIGINS = {'prior_state', 'system.session_summary', 'system.project_instructions',
                    'system.backstory', 'ken', 'task_start', 'task_end_instruction',
                    'delivery_verification'}


def _digest(value: str) -> str:
    return hashlib.sha256(redact_text(redact(value)).encode('utf-8')).hexdigest()


def _open_directory(path: Path, workspace: Path) -> int:
    path = path.absolute()
    if '..' in path.parts or path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError('Recording directory must be outside the workspace')
    if not hasattr(os, 'O_NOFOLLOW') or os.open not in os.supports_dir_fd:
        raise ValueError('Recording requires no-follow directory descriptors')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, flags, dir_fd=parent)
            os.close(parent)
            parent = child
        os.mkdir(path.name, mode=0o700, dir_fd=parent)
        return os.open(path.name, flags, dir_fd=parent)
    finally:
        os.close(parent)


class ContextRecorder:
    """Write fixed metadata projections; write failure never retries task work."""

    def __init__(self, directory_fd: int, *, max_bytes: int = 4 * 1024 * 1024) -> None:
        self._directory_fd = directory_fd
        self._fd = os.open('metadata.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL
                           | os.O_NOFOLLOW, mode=0o600, dir_fd=directory_fd)
        self._max_bytes = max_bytes
        self._bytes = 0
        self._sequence = 0
        self._source_sequence = 0
        self._complete = True
        self._omitted = 0
        self._closed = False
        self._owner_thread = threading.get_ident()
        self._nodes: dict[str, dict[str, Any]] = {}
        self._bindings: dict[str, str] = {}
        self._children: list[str] = []
        self.invocation_id = uuid.uuid4().hex
        self.session_sha256: str | None = None

    def _write(self, record: dict[str, Any]) -> None:
        if self._closed or threading.get_ident() != self._owner_thread:
            return
        self._sequence += 1
        data = json.dumps({'version': 1, 'sequence': self._sequence, **record},
                          ensure_ascii=False, separators=(',', ':')).encode() + b'\n'
        if (not self._complete or len(data) > 65536
                or self._bytes + len(data) > self._max_bytes):
            self._complete = False
            self._omitted += 1
            return
        try:
            count = os.write(self._fd, data)
            if count != len(data):
                raise OSError('Short metadata write')
            self._bytes += count
        except OSError:
            self._complete = False
            self._omitted += 1
            try:
                os.ftruncate(self._fd, self._bytes)
            except OSError:
                pass
            logger.warning('Context recording incomplete; task execution is unchanged')

    def source(
        self, origin: str, role: str, content: object, *, source_id: int | None = None,
        loader_clipped: bool | None = None, renderer_clipped: bool | None = None,
        public_digest: bool = True, original_bytes: int | None = None,
        clipping: dict[str, int] | None = None,
    ) -> str:
        """Describe one actual assembly input without retaining its content."""
        if self._closed or threading.get_ident() != self._owner_thread:
            return ''
        self._source_sequence += 1
        identity = f'source-{self._source_sequence}'
        origin = origin if origin in _ORIGINS else 'unknown'
        allowed_roles = {'user', 'assistant'} if origin == 'history' else _ROLES - {'tool'}
        public = (isinstance(role, str) and role in allowed_roles
                  and origin not in _PRIVATE_ORIGINS and origin != 'unknown')
        known = public and isinstance(content, str) and public_digest
        record = {'kind': 'source', 'id': identity, 'origin': origin,
                     'phase': 'history_selected_view' if origin == 'history' else 'assembly_fragment',
                     'role': role if isinstance(role, str) and role in _KNOWN_ROLES else 'unknown',
                     'source_id': source_id if type(source_id) is int else None,
                     'loader_clipped': loader_clipped, 'renderer_clipped': renderer_clipped,
                     'clipping': {k: v for k, v in (clipping or {}).items()
                                  if k in {'loader_limit_chars', 'renderer_head_bytes',
                                           'renderer_tail_bytes'} and type(v) is int and v >= 0},
                     'content_bytes': len(content.encode()) if known else None,
                     'original_bytes': original_bytes if known else None,
                     'original_content_sha256': None,
                     'content_sha256': _digest(content) if known else None,
                     'hash_scope': 'redacted_public' if known else 'omitted_nonpublic_or_unknown'}
        self._write(record)
        self._nodes[identity] = {**record, '_fingerprint': self._fingerprint(content),
                                 '_lineage_known': origin != 'unknown'}
        return identity

    @staticmethod
    def _fingerprint(content: object) -> str | None:
        # Internal equality check only; never serialize a private-content fingerprint.
        return hashlib.sha256(content.encode()).hexdigest() if isinstance(content, str) else None

    def bundle(self, label: str, origin: str, role: str, content: str,
               parts: list[str]) -> str:
        """Link producer-declared inputs in actual assembly order, without text matching."""
        if self._closed or threading.get_ident() != self._owner_thread:
            return ''
        known = bool(parts) and all(self._nodes.get(p, {}).get('_lineage_known') for p in parts)
        public = known and all(self._nodes[p].get('content_sha256') for p in parts)
        self._source_sequence += 1
        identity = f'source-{self._source_sequence}'
        record = {'kind': 'bundle', 'id': identity, 'parts': list(parts),
                  'phase': 'assembly_composite',
                  'origin': origin if origin in _ORIGINS else 'unknown',
                  'role': role if isinstance(role, str) and role in _KNOWN_ROLES else 'unknown',
                  'mapping_scope': 'producer_declared_logical_components',
                  'parts_known': known,
                  'content_sha256': _digest(content) if public else None,
                  'hash_scope': 'redacted_public' if public else 'omitted_composite'}
        self._write(record)
        self._nodes[identity] = {**record, '_fingerprint': self._fingerprint(content),
                                 '_lineage_known': known}
        self._bindings[label] = identity
        return identity

    def binding(self, label: str, content: object) -> str | None:
        identity = self._bindings.get(label)
        node = self._nodes.get(identity, {})
        return identity if node.get('_fingerprint') == self._fingerprint(content) else None

    def attach(self, message: dict[str, Any], identity: str | None) -> None:
        if identity is not None:
            message['_recording_source'] = identity

    def request(self, child_run_id: str, iteration: int, messages: list[dict[str, Any]],
                schemas: list[dict[str, Any]], *, mode: str) -> None:
        """Snapshot canonical assembly metadata, not adapted HTTP/wire input."""
        if self._closed or threading.get_ident() != self._owner_thread:
            return
        if child_run_id not in self._children:
            self._children.append(child_run_id)
        described = []
        for index, message in enumerate(messages):
            identity = message.get('_recording_source')
            node = self._nodes.get(identity, {})
            fingerprint = self._fingerprint(message.get('content'))
            valid = (bool(node) and fingerprint is not None
                     and node.get('_fingerprint') == fingerprint
                     and node.get('role') == message.get('role'))
            role = message.get('role')
            role = role if isinstance(role, str) and role in _KNOWN_ROLES else 'unknown'
            described.append({'index': index, 'role': role,
                              'source_id': identity if valid else None,
                              'assembly_ancestor': identity, 'mapping_valid': valid,
                              'content_sha256': node.get('content_sha256') if valid else None})
        names = [redact_text(redact(s['function']['name']))[:100] for s in schemas]
        self._write({'kind': 'request', 'scope': 'principal_canonical_before_adapter',
                     'child_run_id': child_run_id, 'iteration': iteration, 'mode': mode,
                     'tool_names': names, 'schemas_sha256': _digest(json.dumps(
                         schemas, ensure_ascii=False, sort_keys=True, separators=(',', ':'))),
                     'schemas_hash_scope': 'redacted_canonical_schema_json',
                     'messages': described})

    def event(self, child_run_id: str, iteration: int, event: str,
              facts: dict[str, Any]) -> None:
        """Project existing host events; never serialize arbitrary event payloads."""
        if self._closed or threading.get_ident() != self._owner_thread:
            return
        if event not in {'provider_request', 'model_response', 'tool_result',
                         'tool_validation_rejected', 'turn_closed', 'public_message'}:
            return
        record: dict[str, Any] = {'kind': 'event', 'event': event,
                                  'child_run_id': child_run_id, 'iteration': iteration,
                                  'evidence_layer': 'existing_host_trace_projection'}
        if event == 'provider_request':
            record['scope'] = 'litellm_input_metadata_not_wire'
            names = facts.get('tools') or []
            record['tool_names'] = [redact_text(redact(n))[:100] for n in names
                                    if isinstance(n, str)]
            choice = facts.get('tool_choice')
            record['tool_choice'] = choice if choice in ('auto', 'none', 'required') else 'other'
        if event in {'model_response', 'tool_validation_rejected'}:
            calls = facts.get('calls') or []
            described = []
            for call in calls[:100]:
                name = call.get('tool')
                if not isinstance(name, str):
                    continue
                projected = redact_text(redact(name))[:100]
                item = {'tool': projected, 'tool_name_complete': projected == name}
                if event == 'tool_validation_rejected':
                    item['errors'] = [{'type': redact_text(redact(p['type']))[:100]}
                                      for p in call.get('errors', []) if isinstance(p.get('type'), str)]
                described.append(item)
            record['calls'] = described
            record['calls_complete'] = len(calls) <= 100
        if event == 'model_response':
            origin = facts.get('origin')
            record['origin'] = origin if origin in ('native', 'parsed_text', 'none') else 'unknown'
            reason = facts.get('finish_reason')
            record['finish_reason'] = (reason if reason in ('stop', 'length', 'tool_calls',
                                                           'function_call', 'content_filter')
                                       else 'unknown')
        if event == 'tool_validation_rejected':
            record['executed'] = False
        if event == 'tool_result' and isinstance(facts.get('tool'), str):
            record['tool'] = redact_text(redact(facts['tool']))[:100]
            record['tool_name_complete'] = record['tool'] == facts['tool']
            record['result_origin'] = 'dispatcher'
            record['result_outcome'] = 'unknown_opaque_result_not_parsed'
        if event == 'turn_closed':
            for name, allowed in (('status', {'answered', 'failed', 'cancelled', 'exhausted'}),
                                  ('closure', {'plain_text', 'finish_turn', 'interrupted'})):
                value = facts.get(name)
                record[name] = value if value in allowed else 'unknown'
        for name in ('attempt', 'dispatches', 'ordinal'):
            if type(facts.get(name)) is int and facts[name] >= 0:
                record[name] = facts[name]
        self._write(record)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._fd)
        except OSError:
            self._complete = False
            logger.warning('Context recording close unavailable')
        manifest = json.dumps({'schema_version': 1, 'complete': self._complete,
                               'scope': 'recording_invocation_not_session_state',
                               'recording_invocation_id': self.invocation_id,
                               'session_sha256': self.session_sha256,
                               'child_run_ids': self._children,
                               'source_checkpoint': None,
                               'records_attempted': self._sequence,
                               'omitted_records': self._omitted,
                               'metadata_bytes': self._bytes}).encode()
        try:
            fd = os.open('manifest.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | os.O_NOFOLLOW, mode=0o600, dir_fd=self._directory_fd)
            try:
                if os.write(fd, manifest) != len(manifest):
                    raise OSError('Short manifest write')
            finally:
                os.close(fd)
        except OSError:
            logger.warning('Context recording manifest unavailable')
        finally:
            try:
                os.close(self._directory_fd)
            except OSError:
                logger.warning('Context recording directory close unavailable')


def current_recorder() -> ContextRecorder | None:
    """Return this invocation's recorder, never a session-global switch."""
    recorder = _CURRENT.get()
    return (recorder if recorder is not None and not recorder._closed
            and recorder._owner_thread == threading.get_ident() else None)


@contextmanager
def recording(
    directory: Path | str | None, *, workspace: Path | str, max_bytes: int = 4 * 1024 * 1024,
):
    """Preflight one exclusive private directory; disabled means no I/O."""
    recorder = None
    if directory is not None:
        fd = _open_directory(Path(directory), Path(workspace))
        try:
            recorder = ContextRecorder(fd, max_bytes=max_bytes)
        except BaseException:
            os.close(fd)
            raise
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT.reset(token)
        if recorder is not None:
            recorder.close()


def recorded_context(function):
    """Bind recording to an explicitly opted-in SDK invocation."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        directory = kwargs.get('record_context_dir')
        if directory is None and current_recorder() is None:
            return function(*args, **kwargs)
        # Match activate_context: a previous tool context is replaced before the turn.
        workspace = os.environ.get('INFINIDEV_WORKSPACE') or Path.cwd()
        with recording(directory, workspace=workspace) as recorder:
            if recorder is not None and isinstance(kwargs.get('session_id'), str):
                recorder.session_sha256 = _digest(kwargs['session_id'])
            return function(*args, **kwargs)
    return wrapped
