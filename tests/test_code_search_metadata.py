"""I1a adds factual search metadata while retaining the legacy search contract."""
from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from infinidev.engine.subprocess_runner import CapturedProcess
from infinidev.tools.file import code_search_tool as module
from infinidev.tools.file.code_search_input import CodeSearchInput
from infinidev.tools.file.code_search_tool import CodeSearchTool
from tests.test_turn_runtime import runtime


@pytest.fixture
def search(bound_tool, monkeypatch):
    monkeypatch.setattr(CodeSearchTool, '_has_git', staticmethod(lambda _: False))
    return bound_tool(CodeSearchTool)


def legacy(result):
    return {key: value for key, value in result.items() if key != 'search_metadata'}


def metadata(result):
    observed = result['search_metadata']
    assert observed['schema_version'] == 1
    return observed


def test_input_schema_and_public_arguments_stay_unchanged():
    schema = json.dumps(CodeSearchInput.model_json_schema(), sort_keys=True).encode()
    assert hashlib.sha256(schema).hexdigest() == (
        'f2e9891c6eac9f1f110efdb6134981af938f07d75e6d945fe8343ecff3476101')
    assert list(inspect.signature(CodeSearchTool._run).parameters) == [
        'self','pattern','file_path','file_extensions','case_sensitive',
        'max_results','context_lines','max_depth']


@pytest.mark.parametrize('context', [0, 1])
def test_file_hits_keep_legacy_payload_and_declare_python(search, workspace_dir, context):
    source = workspace_dir/'source.rs'
    source.write_text('before\nlet value = items[0];\nafter\n')
    result = json.loads(search._run(pattern=r'items\[0\]', file_path=str(source),
                                    context_lines=context))
    hit = dict(file=str(source), line=2, content='let value = items[0];')
    if context:
        hit['context'] = [dict(line=1,content='before'),dict(line=2,content=hit['content']),
                          dict(line=3,content='after')]
    assert legacy(result) == dict(pattern=r'items\[0\]',file_path=str(source),
                                  match_count=1,truncated=False,matches=[hit])
    info = metadata(result)
    assert (info['mode'],info['backend'],info['dialect'],info['scope']) == (
        'regex','python_re','python','file')
    assert info['outcome'] == 'matches'
    assert info['coverage'] == 'completed_scope'
    assert info['root'] == str(source)


def test_file_empty_is_only_in_declared_scope(search, workspace_dir):
    source = workspace_dir/'source.rs'; source.write_text('needle\n')
    result = json.loads(search._run(pattern='absent', file_path=str(source)))
    assert legacy(result) == dict(pattern='absent',file_path=str(source),match_count=0,
                                  truncated=False,matches=[])
    assert metadata(result)['outcome'] == 'empty'
    assert metadata(result)['scope'] == 'file'


@pytest.mark.parametrize('cap,expected', [(1,False),(2,False)])
def test_file_exact_cap_is_not_invented_partial(search, workspace_dir, cap, expected):
    source = workspace_dir/'source.rs'; source.write_text('needle\n')
    result = json.loads(search._run(pattern='needle',file_path=str(source),max_results=cap))
    assert result['truncated'] is expected
    assert metadata(result)['coverage'] == 'completed_scope'


def test_file_more_than_cap_preserves_hits_and_partial(search, workspace_dir):
    source = workspace_dir/'source.rs'; source.write_text('needle\nneedle\n')
    result = json.loads(search._run(pattern='needle',file_path=str(source),max_results=1))
    assert result['match_count'] == 1 and result['truncated'] is True
    assert metadata(result)['outcome'] == 'partial'
    assert metadata(result)['coverage'] == 'limited'


def test_invalid_regex_stays_error_without_zero_count(search, workspace_dir):
    source = workspace_dir/'source.rs'; source.write_text('needle\n')
    result = json.loads(search._run(pattern='[',file_path=str(source)))
    assert result['error'].startswith('Invalid search regex:')
    assert 'match_count' not in result
    assert metadata(result)['outcome'] == 'error'
    assert metadata(result)['coverage'] == 'unknown'


def test_missing_path_stays_error(search, workspace_dir):
    missing = workspace_dir/'missing'
    result = json.loads(search._run(pattern='needle',file_path=str(missing)))
    assert legacy(result) == {'error':f'Directory not found: {missing}'}
    assert metadata(result)['outcome'] == 'error'


def test_sandbox_error_keeps_legacy_message(search, monkeypatch, workspace_dir):
    monkeypatch.setattr(CodeSearchTool, '_validate_sandbox_path', lambda self, path: 'DENIED')
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir)))
    assert legacy(result) == {'error':'DENIED'}
    assert metadata(result)['outcome'] == 'error'


@pytest.mark.parametrize('code,stdout,stderr,outcome', [
    (0,'source.rs:1:needle\n','','matches'), (1,'','','empty'),
    (0,'source.rs:1:needle\n','backend warning','partial'),
    (1,'','backend warning','partial'), (0,'unparsed output\n','','partial'),
    (1,'omitted output\n','','partial'), (2,'','backend error','error')])
def test_directory_result_states_preserve_legacy(search, monkeypatch, workspace_dir,
                                                 code, stdout, stderr, outcome):
    monkeypatch.setattr(module,'run_captured',lambda *a,**kw:
                        CapturedProcess(code,stdout,stderr))
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir)))
    info = metadata(result)
    assert (info['backend'],info['dialect'],info['scope']) == ('grep','ere','directory')
    assert info['outcome'] == outcome
    if code == 2:
        assert legacy(result) == {'error':'Search failed: backend error'}
        assert 'match_count' not in result
    elif code == 0 and stdout.startswith('source.rs:'):
        assert result['match_count'] == 1
        assert result['matches'][0]['content'] == 'needle'
    else:
        assert legacy(result) == dict(pattern='needle',file_path=str(workspace_dir),
                                      match_count=0,truncated=False,matches=[])
    if stderr and code != 2:
        assert 'backend_stderr' in info['reasons']
        assert info['backend_stderr'] == stderr


@pytest.mark.parametrize('stdout', ['', 'source.rs:1:needle\n'])
@pytest.mark.parametrize('stderr', ['', 'aggregate warning'])
def test_xargs_123_never_claims_clean_empty(search, monkeypatch, workspace_dir, stdout, stderr):
    monkeypatch.setattr(module,'run_captured',lambda *a,**kw:
                        CapturedProcess(123,stdout,stderr))
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir),max_depth=1))
    assert legacy(result) == dict(pattern='needle',file_path=str(workspace_dir),
                                  match_count=0,truncated=False,matches=[])
    info = metadata(result)
    assert info['outcome'] == 'partial'
    assert info['coverage'] == 'unknown'
    assert 'aggregate_exit' in info['reasons']


def test_git_backend_does_not_certify_depth_with_extension_pathspec(
        search, monkeypatch, workspace_dir):
    monkeypatch.setattr(CodeSearchTool,'_has_git',staticmethod(lambda _: True))
    commands = []
    def captured(command, **kwargs):
        commands.append(command)
        return CapturedProcess(1,'','')
    monkeypatch.setattr(module,'run_captured',captured)
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir),
        file_extensions=['rs'],max_depth=2,case_sensitive=False))
    info = metadata(result)
    assert info['backend'] == 'git_grep'
    assert info['filters_applied']['file_extensions'] == ['.rs']
    assert info['filters_applied']['max_depth'] is None
    assert info['coverage'] == 'unknown'
    assert 'git_depth_with_pathspec_uncertain' in info['reasons']
    assert '-i' in commands[0] and '--max-depth' in commands[0]


def test_real_git_depth_with_extension_pathspec_is_not_certified(
        search, monkeypatch, workspace_dir):
    assert module.run_captured(['git', 'init', '-q'], cwd=str(workspace_dir),
                               timeout=5).exit_code == 0
    nested = workspace_dir / 'nested'
    nested.mkdir()
    (nested / 'source.rs').write_text('needle\n')
    monkeypatch.setattr(CodeSearchTool, '_has_git', staticmethod(lambda _: True))
    bounded = json.loads(search._run(pattern='needle', file_path=str(workspace_dir),
                                     max_depth=0))
    wildcard = json.loads(search._run(pattern='needle', file_path=str(workspace_dir),
                                      max_depth=0, file_extensions=['rs']))
    assert bounded['match_count'] == 0
    assert metadata(bounded)['filters_applied']['max_depth'] == 0
    assert metadata(bounded)['coverage'] == 'completed_scope'
    # Git 2.56.1 here applies the depth even with this wildcard; the manual
    # documents an exception. Do not assert ignored/effective across versions.
    assert wildcard['match_count'] in (0, 1)
    assert metadata(wildcard)['filters_applied']['max_depth'] is None
    assert metadata(wildcard)['coverage'] == 'unknown'
    assert 'git_depth_with_pathspec_uncertain' in metadata(wildcard)['reasons']


def test_real_find_grep_does_not_claim_recursive_directory_exclusions(search, workspace_dir):
    build = workspace_dir / 'build'
    build.mkdir()
    (build / 'source.rs').write_text('needle\n')
    recursive = json.loads(search._run(pattern='needle', file_path=str(workspace_dir)))
    explicit = json.loads(search._run(pattern='needle', file_path=str(workspace_dir),
                                      max_depth=2))
    assert recursive['match_count'] == 0
    assert metadata(recursive)['filters_applied']['excludes'] == 'grep_defaults'
    assert explicit['match_count'] == 1
    assert metadata(explicit)['filters_applied']['excludes'] is None
    assert metadata(explicit)['coverage'] == 'unknown'
    assert 'directory_exclusions_not_applied' in metadata(explicit)['reasons']


def test_file_does_not_claim_directory_filters(search, workspace_dir):
    source = workspace_dir/'source.rs'; source.write_text('needle\n')
    result = json.loads(search._run(pattern='needle',file_path=str(source),
                                    file_extensions=['py'],max_depth=1))
    assert result['match_count'] == 1
    filters = metadata(result)['filters_applied']
    assert filters.get('file_extensions') is None and filters.get('max_depth') is None


def test_timeout_and_unavailable_are_not_empty(search, monkeypatch, workspace_dir):
    monkeypatch.setattr(module,'run_captured',lambda *a,**kw:
                        CapturedProcess(-1,'partial','',timed_out=True))
    timed = json.loads(search._run(pattern='needle',file_path=str(workspace_dir)))
    assert legacy(timed) == {'error':'Search timed out (pattern may be too broad)'}
    assert metadata(timed)['outcome'] == 'error'
    def unavailable(*args, **kwargs):
        raise OSError('synthetic unavailable')
    monkeypatch.setattr(module,'run_captured',unavailable)
    failed = json.loads(search._run(pattern='needle',file_path=str(workspace_dir)))
    assert metadata(failed)['outcome'] == 'unavailable'
    assert 'match_count' not in failed


@pytest.mark.parametrize('count', [0,1])
def test_pod_without_coverage_echo_stays_unknown(search, monkeypatch, workspace_dir, count):
    monkeypatch.setattr(CodeSearchTool,'_is_pod_mode',lambda self: True)
    data = {'match_count':count,'matches':[],'truncated':True}
    monkeypatch.setattr(CodeSearchTool,'_exec_in_pod',lambda *a,**kw:
                        CapturedProcess(0,json.dumps({'ok':True,'data':data}),''))
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir)))
    assert legacy(result) == dict(pattern='needle',file_path='.',
                                  match_count=count,truncated=False,matches=[])
    info = metadata(result)
    assert (info['backend'],info['mode'],info['coverage']) == ('pod_helper','unknown','unknown')
    assert info['outcome'] == 'partial'
    assert info['filters_applied'] is None
    assert 'remote_metadata_unknown' in info['reasons']


def test_same_results_keep_stable_metadata(search, workspace_dir):
    source = workspace_dir/'source.rs'; source.write_text('needle\n')
    first = search._run(pattern='absent',file_path=str(source))
    second = search._run(pattern='absent',file_path=str(source))
    assert 'search_metadata' in json.loads(first)
    assert first == second


def test_real_dialect_difference_is_visible_without_changing_search(search, workspace_dir):
    source = workspace_dir/'source.txt'; source.write_text('needle\n')
    single = json.loads(search._run(pattern='(?:needle)',file_path=str(source)))
    directory = json.loads(search._run(pattern='(?:needle)',file_path=str(workspace_dir)))
    assert single['match_count'] == 1 and directory['match_count'] == 0
    assert metadata(single)['dialect'] == 'python'
    assert metadata(directory)['dialect'] == 'ere'
    assert metadata(directory)['outcome'] == 'partial'
    assert 'backend_stderr' in metadata(directory)['reasons']


@pytest.mark.parametrize('count', [1,2])
def test_directory_cap_keeps_legacy_truncation_and_hits(search, monkeypatch, workspace_dir, count):
    stdout = ''.join(f'source.rs:{line}:needle\n' for line in range(1,count+1))
    monkeypatch.setattr(module,'run_captured',lambda *a,**kw:CapturedProcess(0,stdout,''))
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir),max_results=1))
    assert result['match_count'] == 1 and result['truncated'] is True
    assert metadata(result)['outcome'] == 'partial'
    assert 'result_limit' in metadata(result)['reasons']


def test_context_parse_uncertainty_does_not_remove_actual_hits(search, monkeypatch, workspace_dir):
    stdout = 'source.rs:2:needle\nsource.rs-1-before\n--\nsource.rs-3-after\n'
    monkeypatch.setattr(module,'run_captured',lambda *a,**kw:CapturedProcess(0,stdout,''))
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir),context_lines=1))
    assert result['match_count'] == 1 and result['truncated'] is False
    assert result['matches'] == [dict(file='source.rs',line=2,content='needle')]
    assert metadata(result)['coverage'] == 'unknown'
    assert metadata(result)['unparsed_output_lines'] == 2


def test_unicode_case_handling_keeps_file_semantics(search, workspace_dir):
    source = workspace_dir/'source.rs'; source.write_text('Ñeedle\n')
    result = json.loads(search._run(pattern='ñEEDLE',file_path=str(source),case_sensitive=False))
    assert result['match_count'] == 1
    assert result['matches'][0]['content'] == 'Ñeedle'
    assert metadata(result)['filters_applied']['case_sensitive'] is False


def test_file_read_error_stays_error(search, monkeypatch, workspace_dir):
    source = workspace_dir/'source.rs'; source.write_text('needle\n')
    def denied(*args, **kwargs):
        raise PermissionError('synthetic denied')
    monkeypatch.setattr(module,'open',denied,raising=False)
    result = json.loads(search._run(pattern='needle',file_path=str(source)))
    assert legacy(result) == {'error':f'Could not read {source}: synthetic denied'}
    assert metadata(result)['outcome'] == 'error'


def test_new_stderr_preview_is_bounded_without_trimming_legacy_error(search, monkeypatch, workspace_dir):
    stderr = 'backend warning\n' * 300
    monkeypatch.setattr(module,'run_captured',lambda *a,**kw:CapturedProcess(2,'',stderr))
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir)))
    assert legacy(result) == {'error':'Search failed: '+stderr.strip()}
    assert metadata(result)['backend_stderr'] == stderr[:2000]
    assert metadata(result)['backend_stderr_truncated'] is True


@pytest.mark.parametrize('stdout,code', [('not-json',0),('{"ok":false,"error":"denied"}',0),('',2)])
def test_pod_errors_have_unknown_coverage(search, monkeypatch, workspace_dir, stdout, code):
    monkeypatch.setattr(CodeSearchTool,'_is_pod_mode',lambda self:True)
    monkeypatch.setattr(CodeSearchTool,'_exec_in_pod',lambda *a,**kw:CapturedProcess(code,stdout,''))
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir)))
    assert 'error' in result and 'match_count' not in result
    assert metadata(result)['outcome'] == 'error'
    assert metadata(result)['coverage'] == 'unknown'


@pytest.mark.parametrize('backend', ['git', 'pod'])
def test_exit_123_is_only_called_aggregate_for_xargs(search, monkeypatch, workspace_dir, backend):
    if backend == 'git':
        monkeypatch.setattr(CodeSearchTool,'_has_git',staticmethod(lambda _:True))
        monkeypatch.setattr(module,'run_captured',lambda *a,**kw:CapturedProcess(123,'',''))
    else:
        monkeypatch.setattr(CodeSearchTool,'_is_pod_mode',lambda self:True)
        monkeypatch.setattr(CodeSearchTool,'_exec_in_pod',lambda *a,**kw:CapturedProcess(123,'',''))
    result = json.loads(search._run(pattern='needle',file_path=str(workspace_dir),max_depth=1))
    info = metadata(result)
    assert 'aggregate_exit' not in info['reasons']
    assert info['coverage'] == 'unknown'


@pytest.mark.parametrize('manual', [False, True], ids=['native','manual'])
def test_metadata_reaches_the_provider_as_tool_observation(runtime, monkeypatch,
                                                          workspace_dir, manual):
    from types import SimpleNamespace
    from tests.test_turn_runtime import reply, tc
    engine, agent, script, seen = runtime
    agent.tools = [CodeSearchTool()]
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=not manual, supports_vision=False))
    source = workspace_dir/'source.rs'; source.write_text('needle\n')
    calls = [('code_search',dict(pattern='absent',file_path=str(source))),
             ('finish_turn',dict(message='Observed.'))]
    if manual:
        script(*(reply(json.dumps({'tool_calls':[dict(name=name,arguments=args)]}))
                 for name,args in calls))
    else:
        script(*(reply('',tc(name,**args),finish_reason='tool_calls') for name,args in calls))
    assert engine.run_turn(agent,'Read the source.').status == 'answered'
    observations = [message['content'] for message in seen[1][0]
                    if message['role']=='tool' or (manual and message['role']=='user'
                        and isinstance(message['content'],str)
                        and message['content'].startswith('[Tool: code_search] Result:'))]
    assert len(observations) == 1
    body = observations[0].split('\n',1)[1] if manual else observations[0]
    payload,_ = json.JSONDecoder().raw_decode(body)
    assert payload['match_count'] == 0
    assert payload['search_metadata']['outcome'] == 'empty'
    assert payload['search_metadata']['backend'] == 'python_re'
