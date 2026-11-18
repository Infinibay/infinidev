"""Synthetic native calls reach a local HTTP endpoint through real LiteLLM."""
from __future__ import annotations

import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading

import pytest

from tests.test_turn_runtime import runtime


@pytest.fixture
def local_completion_endpoint(monkeypatch, tmp_path):
    requests = []
    reply_name = ['finish_turn']

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            payload = json.loads(raw)
            index = len(requests)
            requests.append((self.path, payload, raw))
            (tmp_path / f'request-{index}.json').write_bytes(raw)
            if reply_name[0] == 'reject_context':
                failure = json.dumps({'error': {
                    'type': 'invalid_request_error',
                    'message': ('request exceeds the context window: prompt (32000 tokens) '
                                'leaves no room to answer in the context (32000)'),
                }}).encode()
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(failure)))
                self.end_headers()
                self.wfile.write(failure)
                return
            if isinstance(payload.get('tool_choice'), dict) and reply_name[0] == 'reject_named':
                failure = b'{"error":{"message":"Named tool_choice is not supported"}}'
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(failure)))
                self.end_headers()
                self.wfile.write(failure)
                return
            truncating = reply_name[0] == 'truncate_once' and index == 0
            if reply_name[0] == 'truncate_once':
                arguments = ('{"message":"partial body to discard' if truncating
                             else '{"message":"Recovered."}')
                function = {'name': 'finish_turn', 'arguments': arguments}
            else:
                function = {
                    'name': 'finish_turn' if reply_name[0] == 'reject_named' else reply_name[0],
                    'arguments': '{}',
                }
            reason = 'length' if truncating else 'tool_calls'
            call = {'id': 'synthetic-call', 'type': 'function', 'function': function}
            common = {'id': 'synthetic-completion', 'created': 1,
                      'model': 'Qwen3.8-FN-UD-IQ4_XS'}
            if payload.get('stream'):
                chunks = [
                    {**common, 'object': 'chat.completion.chunk', 'choices': [{
                        'index': 0, 'delta': {'role': 'assistant',
                                             'tool_calls': [{'index': 0, **call}]},
                        'finish_reason': None}]},
                    {**common, 'object': 'chat.completion.chunk', 'choices': [{
                        'index': 0, 'delta': {}, 'finish_reason': reason}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 2,
                                  'total_tokens': 12}},
                ]
                body = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks)
                raw_body = (body + 'data: [DONE]\n\n').encode()
                content_type = 'text/event-stream'
            else:
                raw_body = json.dumps({**common, 'object': 'chat.completion', 'choices': [{
                    'index': 0, 'message': {'role': 'assistant', 'content': None,
                                             'tool_calls': [call]},
                    'finish_reason': reason}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 2,
                              'total_tokens': 12}}).encode()
                content_type = 'application/json'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw_body)))
            self.end_headers()
            self.wfile.write(raw_body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    original_connect = socket.socket.connect
    def only_endpoint(sock, address):
        if address != server.server_address:
            raise AssertionError('Offline HTTP gate attempted another destination')
        return original_connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', only_endpoint)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests, reply_name, tmp_path
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        assert not worker.is_alive()


def synthetic_request(monkeypatch, endpoint, required_supported):
    import litellm
    from infinidev.config.model_capabilities import ModelCapabilities
    from infinidev.config.settings import settings
    from infinidev.engine.tool_dispatch import tool_to_openai_schema
    from infinidev.engine.turns.models import TurnState
    from infinidev.engine.turns.tools import build_turn_tools
    from infinidev.tools.file.create_file_tool import CreateFileTool
    from infinidev.tools.file.read_file_tool import ReadFileTool

    for name, value in {
        'LLM_PROVIDER': 'openai_compatible',
        'LLM_MODEL': 'custom_openai/Qwen3.8-FN-UD-IQ4_XS',
        'LLM_API_KEY': 'synthetic-not-a-secret', 'LLM_BASE_URL': endpoint,
        'LLM_NUM_RETRIES': 0, 'LLM_TIMEOUT': 3, 'LLM_TEMPERATURE': 0.2,
        'THINKING_ENABLED': False, 'PROMPT_CACHE_ENABLED': False,
    }.items():
        monkeypatch.setattr(settings, name, value)
    caps = ModelCapabilities(supports_function_calling=True,
                             supports_tool_choice_required=required_supported,
                             supports_json_mode=False)
    monkeypatch.setattr('infinidev.config.model_capabilities.get_model_capabilities',
                        lambda: caps)
    monkeypatch.setattr(litellm, 'drop_params', False)
    if hasattr(litellm, 'telemetry'):
        monkeypatch.setattr(litellm, 'telemetry', False)
    schemas = [tool_to_openai_schema(t) for t in
               [ReadFileTool(), CreateFileTool(), *build_turn_tools(TurnState())]]
    messages = [{'role': 'system', 'content': 'Synthetic host protocol.'},
                {'role': 'user', 'content': 'Synthetic context.'},
                {'role': 'user', 'content': 'Synthetic prior state.'},
                {'role': 'user', 'content': 'Synthetic request.'}]
    return messages, schemas


def test_principal_retries_a_truncated_native_stream_with_the_same_request_limit(
    runtime, monkeypatch, local_completion_endpoint,
):
    from infinidev.config.llm import get_litellm_params
    from infinidev.engine.turns import runtime as turn_runtime

    endpoint, requests, reply_name, _ = local_completion_endpoint
    synthetic_request(monkeypatch, endpoint, True)
    reply_name[0] = 'truncate_once'
    monkeypatch.setattr(turn_runtime, 'get_litellm_params',
                        lambda: {**get_litellm_params(), 'max_tokens': 64})
    engine, agent, _, _ = runtime

    result = engine.run_turn(agent, 'Report briefly.', task_tools=[], max_iterations=2)

    assert result.status == 'answered'
    assert result.message == 'Recovered.'
    assert result.metrics['tool_calls'] == 1
    assert len(requests) == 2
    first, second = [payload for _, payload, _ in requests]
    assert first['max_tokens'] == second['max_tokens'] == 64
    assert first['tools'] == second['tools']
    assert second['stream'] is True
    assert all(message['role'] != 'tool' for message in second['messages'])
    assert 'partial body to discard' not in json.dumps(second['messages'])
    assert 'output limit' in second['messages'][-1]['content']


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('requested,supported,expected', [
    ('auto', False, 'auto'), ('required', False, 'auto'), ('required', True, 'required'),
    ({'type': 'function', 'function': {'name': 'read_file'}}, True,
     {'type': 'function', 'function': {'name': 'read_file'}}),
])
def test_exact_catalog_roles_and_choice_survive_real_http(
        monkeypatch, local_completion_endpoint, stream, requested, supported, expected):
    from infinidev.config.llm import get_litellm_params
    from infinidev.engine.llm_client import call_llm

    endpoint, requests, _, output = local_completion_endpoint
    messages, schemas = synthetic_request(monkeypatch, endpoint, supported)
    original_messages, original_schemas = copy.deepcopy((messages, schemas))
    metadata = []
    response = call_llm(get_litellm_params(), messages, schemas, tool_choice=requested,
                        retry_attempts=1, use_json_mode=False,
                        on_thinking_chunk=(lambda text: pytest.fail('Unexpected reasoning'))
                        if stream else None, on_request_metadata=metadata.append)
    assert len(requests) == 1
    path, payload, raw = requests[0]
    assert path == '/v1/chat/completions'
    assert payload['tools'] == original_schemas
    assert [s['function']['name'] for s in payload['tools']] == [
        'read_file', 'create_file', 'finish_turn', 'update_plan']
    adapted_messages = copy.deepcopy(original_messages)
    adapted_messages[-1]['content'] = '/no_think\n' + adapted_messages[-1]['content']
    assert payload['messages'] == adapted_messages
    assert payload['tool_choice'] == expected
    assert payload.get('stream', False) is stream
    if stream:
        assert payload['stream_options'] == {'include_usage': True}
    assert payload['model'] == 'Qwen3.8-FN-UD-IQ4_XS'
    assert payload['chat_template_kwargs']['enable_thinking'] is False
    assert 'functions' not in payload and 'function_call' not in payload
    assert 'response_format' not in payload
    assert schemas == original_schemas
    assert response.choices[0].message.tool_calls[0].function.name == 'finish_turn'
    assert len(metadata) == 1 and metadata[0]['scope'] == 'litellm_input'
    receipt = {'synthetic_only': True, 'request_path': path,
               'request_sha256': hashlib.sha256(raw).hexdigest(),
               'serialized_bytes': len(raw), 'tools_equal': True,
               'roles': [m['role'] for m in payload['messages']],
               'tool_names': [s['function']['name'] for s in payload['tools']],
               'requested_choice': requested, 'supports_required_fixture': supported,
               'serialized_choice': payload['tool_choice'], 'stream': stream,
               'thinking_fixture': False, 'last_user_prefix': '/no_think',
               'template_enable_thinking': payload['chat_template_kwargs']['enable_thinking'],
               'returned_name': response.choices[0].message.tool_calls[0].function.name}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


@pytest.mark.parametrize('content', [
    'Synthetic request.',
    'Explain the Python string literal "/no_think".',
    [
        {'type': 'text', 'text': 'Inspect this fixture.'},
        {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/fixture.png'}},
    ],
])
def test_thinking_controls_preserve_user_content_across_requests(
        monkeypatch, local_completion_endpoint, content):
    from infinidev.config.llm import get_litellm_params
    from infinidev.config.settings import settings
    from infinidev.engine.llm_client import call_llm

    endpoint, requests, _, _ = local_completion_endpoint
    messages, schemas = synthetic_request(monkeypatch, endpoint, True)
    monkeypatch.setattr(settings, 'LLM_MODEL', 'custom_openai/synthetic-large')
    monkeypatch.setattr(settings, 'LLM_TEMPERATURE', 0.0)
    monkeypatch.setattr(settings, 'THINKING_BUDGET', 'medium')
    messages[-1]['content'] = content
    original = copy.deepcopy(messages)
    params = get_litellm_params()

    call_llm(params, messages, schemas, thinking_enabled=False, retry_attempts=1)
    after_suppression = copy.deepcopy(messages)
    call_llm(params, messages, schemas, thinking_enabled=True, retry_attempts=1)

    suppressed, normal = [payload for _, payload, _ in requests]
    if isinstance(content, list):
        expected = [{'type': 'text', 'text': '/no_think'}, *content]
    else:
        expected = '/no_think\n' + content
    assert suppressed['messages'][-1]['content'] == expected
    assert normal['messages'] == original
    assert after_suppression == original
    assert messages == original
    assert suppressed['temperature'] == normal['temperature'] == 0.0


def test_thinking_override_does_not_change_shared_provider_parameters(
        monkeypatch, local_completion_endpoint):
    from infinidev.config.llm import get_litellm_params
    from infinidev.config.settings import settings
    from infinidev.engine.llm_client import call_llm

    endpoint, requests, _, _ = local_completion_endpoint
    messages, schemas = synthetic_request(monkeypatch, endpoint, True)
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'llama_cpp')
    params = get_litellm_params()
    original = copy.deepcopy(params)
    assert original['extra_body']['chat_template_kwargs']['enable_thinking'] is False

    call_llm(params, messages, schemas, thinking_enabled=True, retry_attempts=1)

    (_, payload, _), = requests
    assert payload['chat_template_kwargs']['enable_thinking'] is True
    assert params == original


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('fallback,required_supported,expected', [
    ('required', True, 'required'),
    ('auto', True, 'auto'),
    ('required', False, 'auto'),
])
def test_rejected_named_choice_returns_to_supported_native_choice(
        monkeypatch, local_completion_endpoint, stream, fallback, required_supported, expected):
    from infinidev.config import model_capabilities
    from infinidev.config.llm import get_litellm_params
    from infinidev.engine.llm_client import call_llm

    endpoint, requests, reply_name, _ = local_completion_endpoint
    messages, schemas = synthetic_request(monkeypatch, endpoint, required_supported)
    reply_name[0] = 'reject_named'
    named = {'type': 'function', 'function': {'name': 'read_file'}}
    response = call_llm(get_litellm_params(), messages, schemas, tool_choice=named,
                        retry_attempts=1, on_thinking_chunk=(lambda _: None) if stream else None,
                        named_choice_fallback=fallback)
    assert [payload['tool_choice'] for _, payload, _ in requests] == [named, expected]
    assert all(payload['tools'] == schemas for _, payload, _ in requests)
    assert model_capabilities.get_model_capabilities().supports_tool_choice_required is required_supported
    assert response.choices[0].message.tool_calls[0].function.name == 'finish_turn'


def test_post_llm_hook_failure_does_not_repeat_a_received_response(
        monkeypatch, local_completion_endpoint):
    from infinidev.config.llm import get_litellm_params
    from infinidev.engine.hooks.hooks import HookEvent, hook_manager
    from infinidev.engine.llm_client import call_llm

    endpoint, requests, reply_name, _ = local_completion_endpoint
    messages, schemas = synthetic_request(monkeypatch, endpoint, True)
    reply_name[0] = 'read_file'

    def fail_after_response(ctx):
        if ctx.event is HookEvent.POST_LLM_CALL:
            raise RuntimeError('Hook failed while inspecting tool_choice')

    monkeypatch.setattr(hook_manager, 'dispatch', fail_after_response)
    named = {'type': 'function', 'function': {'name': 'read_file'}}
    with pytest.raises(RuntimeError, match='Hook failed'):
        call_llm(get_litellm_params(), messages, schemas, tool_choice=named, retry_attempts=1)
    assert [payload['tool_choice'] for _, payload, _ in requests] == [named]


def test_compatible_context_rejection_is_visible_through_litellm(
        monkeypatch, local_completion_endpoint):
    from litellm.exceptions import BadRequestError

    from infinidev.config.llm import get_litellm_params
    from infinidev.engine.llm_client import call_llm
    from infinidev.engine.turns.context_recovery import is_context_window_rejection

    endpoint, requests, reply_name, _ = local_completion_endpoint
    messages, schemas = synthetic_request(monkeypatch, endpoint, True)
    reply_name[0] = 'reject_context'
    with pytest.raises(BadRequestError) as exc:
        call_llm(get_litellm_params(), messages, schemas, retry_attempts=1,
                 on_thinking_chunk=lambda _: None)
    assert is_context_window_rejection(exc.value)
    assert len(requests) == 1


@pytest.mark.parametrize('returned_name', ['read_file', 'Read', 'write_file', 'devkit:read_file'])
def test_native_response_name_is_not_normalized_by_client(
        monkeypatch, local_completion_endpoint, returned_name):
    from infinidev.config.llm import get_litellm_params
    from infinidev.engine.llm_client import call_llm

    endpoint, requests, reply_name, output = local_completion_endpoint
    reply_name[0] = returned_name
    messages, schemas = synthetic_request(monkeypatch, endpoint, False)
    response = call_llm(get_litellm_params(), messages, schemas, retry_attempts=1,
                        use_json_mode=False, on_thinking_chunk=lambda _: None)
    assert len(requests) == 1
    assert response.choices[0].message.tool_calls[0].function.name == returned_name
    (output / 'response-name.json').write_text(json.dumps({
        'synthetic_only': True, 'server_name': returned_name,
        'decoded_name': response.choices[0].message.tool_calls[0].function.name,
        'stream': True,
    }, indent=2) + '\n')
