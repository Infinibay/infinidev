"""Deterministic browser-test server. Never used by the production CLI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace

workspace = Path(tempfile.mkdtemp(prefix='infinidev-browser-')).resolve()
os.chdir(workspace)
(workspace / 'model.py').write_text('def forward(x):\n    return x\n')
subprocess.run(['git', 'init', '-q'], check=True)
subprocess.run(['git', 'add', 'model.py'], check=True)
subprocess.run(['git', '-c', 'user.name=Browser Test', '-c', 'user.email=test@example.invalid',
                'commit', '-qm', 'Initial test fixture'], check=True)
(workspace / 'model.py').write_text('def forward(x):\n    return x * 2\n')

from infinidev.config.settings import settings
from infinidev.db.service import init_db, register_session, rename_session, store_session_message
from infinidev.engine.team.store import TeamStore
from infinidev.server.app import create_app
from infinidev.tools.base.db import execute_with_retry

settings.LLM_PROVIDER = 'openai'
settings.LLM_MODEL = 'openai/gpt-6-astra'
settings.THINKING_BUDGET = 'high'
settings.LLM_BASE_URL = 'https://api.openai.com/v1'
settings.save_user_settings({
    'LLM_PROVIDER': settings.LLM_PROVIDER, 'LLM_MODEL': settings.LLM_MODEL,
    'THINKING_BUDGET': settings.THINKING_BUDGET, 'LLM_BASE_URL': settings.LLM_BASE_URL,
})
init_db()
session_id = 'browser-test-session'
register_session(session_id, str(workspace))
rename_session(session_id, 'Investigate memory routing')
store_session_message(session_id, {
    'id': 'user-1', 'speaker': 'You', 'kind': 'user', 'text':
    'Review the memory routing experiment. Understand the baseline, delegate independent '
    'questions, and propose the next experiment with clear evidence.', 'created_at': time.time(),
})
store_session_message(session_id, {
    'id': 'agent-1', 'speaker': 'Infinidev', 'kind': 'agent', 'text':
    'I’ve organized the investigation into three focused questions. **Mara** is reviewing '
    'the research notes, **Leo** is tracing the implementation, and **Nora** is checking '
    'the evaluation.\n\nThe immediate goal is to establish a reproducible baseline before '
    'changing the routing policy. I’ll review their evidence and bring the next experiment '
    'back here.', 'created_at': time.time(),
})
store_session_message(session_id, {
    'id': 'tool-1', 'speaker': 'Mara', 'kind': 'tool', 'text': 'ken_recall', 'state': 'completed',
    'data': {'tool_arguments': {'topic': 'memory routing baseline'},
             'tool_result_full': 'Baseline evaluation: 3 seeds, 1,024 held-out examples.'},
})
key = json.dumps([1, str(workspace), session_id])
store = TeamStore(hashlib.sha256(key.encode()).hexdigest())

def seed(state, emit):
    state.update(agents={
        'orchestrator': {'id': 'orchestrator', 'name': 'Orchestrator', 'role': 'Coordination', 'status': 'idle'},
        'w_mara': {'id': 'w_mara', 'name': 'Mara', 'role': 'Researcher', 'status': 'idle', 'tools': ['ken_recall', 'read_file'], 'system_prompt': 'Review the evidence and cite sources.'},
        'w_leo': {'id': 'w_leo', 'name': 'Leo', 'role': 'Developer', 'status': 'idle', 'tools': ['read_file', 'edit_file', 'execute_command']},
        'w_nora': {'id': 'w_nora', 'name': 'Nora', 'role': 'Evaluation', 'status': 'idle', 'tools': ['read_file', 'execute_command']},
    }, tickets={
        't_baseline': {'id': 't_baseline', 'title': 'Establish the baseline', 'objective': 'Review existing findings and identify reproducible results.', 'status': 'accepted', 'assignee': 'w_mara', 'acceptance': ['Cite the original run configuration'], 'result': 'Baseline identified across three seeds.'},
        't_routing': {'id': 't_routing', 'title': 'Trace the routing policy', 'objective': 'Follow how retrieval scores affect the memory read path.', 'status': 'review', 'assignee': 'w_leo'},
        't_eval': {'id': 't_eval', 'title': 'Check evaluation coverage', 'objective': 'Verify held-out examples and leakage checks.', 'status': 'pending', 'assignee': 'w_nora'},
    })
    emit('note', 'w_mara', json.dumps({'kind': 'finding', 'text': '**Baseline located.** Three seeds use the same evaluation split. Keep the seed set fixed for the next comparison.'}))
    question = emit('message', 'w_leo', 'Can you confirm whether the evaluation includes unseen memory keys?', recipient='w_nora', ticket_id='t_eval', message_type='request')
    emit('message', 'w_nora', 'Yes. The held-out set includes unseen keys; I’m checking that the retrieval cache is cleared between runs.', recipient='w_leo', ticket_id='t_eval', reply_to=question, thread_id=question, message_type='reply')
store.update(seed)

def finding(conn):
    conn.execute("INSERT INTO findings (project_id, topic, content, finding_type, confidence, status) VALUES (1, ?, ?, ?, ?, ?)",
                 ('Memory routing baseline', 'Three seeded runs establish the initial memory retrieval baseline. Compare against the same held-out split.', 'observation', 0.9, 'active'))
    conn.commit()
execute_with_retry(finding)

# Keep all browser scenarios offline: only the execution boundary is substituted.
import infinidev.agents.base
import infinidev.engine.orchestration
import infinidev.cli.session_resume
infinidev.cli.session_resume.begin_resumed_session = lambda _: None
infinidev.agents.base.InfinidevAgent = lambda **_: SimpleNamespace(
    activate_context=lambda **_: None, deactivate=lambda: None)

def run_task(**kwargs):
    hooks = kwargs['hooks']
    hooks.on_phase('execute')
    if kwargs['user_input'] == 'idle fixture':
        from infinidev.engine.team.runtime import ROOT, TeamRuntime

        runtime = TeamRuntime(session_id=hooks.session.session_id, project_id=1,
                              workspace_path=str(workspace), root_agent_id='browser-root',
                              catalog=[], on_status=hooks.on_status)
        engine = kwargs['engine']
        engine._team_runtime, engine._team_actor = runtime, ROOT
        try:
            runtime.poll(ROOT)
            result = runtime.idle(ROOT, events=['note'], reason='Waiting for your research note')
            return 'Woke after shared note.' if result['reason'] == 'event' else 'Wait cancelled.'
        finally:
            runtime.close()
            engine._team_runtime = None
    if 'permission' in kwargs['user_input'].lower():
        answer = hooks.session.ask('Run the baseline test?', 'permission', details='pytest tests/test_memory.py')
        result = 'Permission granted.' if answer == 'allow' else 'Permission denied.'
    else:
        result = 'The research task completed with evidence and validation.'
        for chunk in ['The research task ', 'completed with evidence ', 'and validation.']:
            if kwargs['engine']._cancel_event.is_set():
                return 'Task cancelled.'
            hooks.notify_stream_chunk('Infinidev', chunk)
            time.sleep(0.12)
        hooks.notify_stream_end('Infinidev')
        hooks.mark_reply_shown()
    return result
infinidev.engine.orchestration.run_task = run_task

from infinidev.tools.shell.background_manager import get_background_manager
get_background_manager().start(
    "python3 -u -c 'import time; print(\"Starting baseline evaluation\", flush=True); "
    "[(print(\"epoch\", i, \"loss\", round(2.4/(i+1), 4), flush=True), time.sleep(1)) for i in range(120)]'",
    'Baseline evaluation · seed 42', str(workspace),
)

import uvicorn
uvicorn.run(create_app(token='browser-test-token', initialize=False), host='127.0.0.1',
            port=int(os.environ.get('INFINIDEV_E2E_PORT', '18765')))
