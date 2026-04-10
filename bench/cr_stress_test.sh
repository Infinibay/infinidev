#!/usr/bin/env bash
# ContextRank stress test: 10 diverse tasks against opencode
# Measures ranking performance after each task.
set -euo pipefail

PROJECT_DIR="/home/andres/opencode"
INFINIDEV="infinidev"
TRACE_DIR="${PROJECT_DIR}/.infinidev/traces"
mkdir -p "$TRACE_DIR"

# Array of diverse prompts touching different areas of the codebase
PROMPTS=(
  "Read the CLI entry point in packages/opencode/src/cli/. How does the CLI boot up?"
  "What is the agent system? Read packages/opencode/src/agent/ and explain the agent loop."
  "How does the session management work? Read packages/opencode/src/session/."
  "Read the provider system in packages/opencode/src/provider/. How are LLM providers configured?"
  "What MCP servers does this project support? Read packages/opencode/src/mcp/."
  "How does the message/conversation system work? Read packages/opencode/src/message/."
  "Read the LSP server code in packages/opencode/src/lsp/. What capabilities does it expose?"
  "What is the permission system? Read packages/opencode/src/permission/."
  "How does file editing work? Read the edit tool in packages/opencode/src/tool/edit.ts."
  "Read the bash tool implementation in packages/opencode/src/tool/bash.ts. How does it sandbox commands?"
  "How does the tool registration system work? Show me define() and fromPlugin()."
  "Read the config system in packages/opencode/src/config/. What settings are available?"
)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║      ContextRank Stress Test — 12 diverse tasks         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_DIR"

for i in "${!PROMPTS[@]}"; do
  TASK_NUM=$((i + 1))
  PROMPT="${PROMPTS[$i]}"
  TRACE_FILE="${TRACE_DIR}/task_${TASK_NUM}.log"

  echo "▶ Task ${TASK_NUM}/12: ${PROMPT:0:60}..."

  # Run the task with trace
  INFINIDEV_TRACE_FILE="$TRACE_FILE" $INFINIDEV --prompt "$PROMPT" 2>&1 | tail -2

  # Measure ranking performance
  python3 -c "
import time, sys
sys.path.insert(0, '/home/andres/infinidev/src')
from infinidev.config.settings import settings
settings.DB_PATH = '${PROJECT_DIR}/.infinidev/infinidev.db'
from infinidev.tools.base.context import set_context
set_context(project_id=1, workspace_path='${PROJECT_DIR}')
from infinidev.code_intel._db import execute_with_retry

# Count data
def counts(conn):
    ctx = conn.execute('SELECT COUNT(*) FROM cr_contexts').fetchone()[0]
    ints = conn.execute('SELECT COUNT(*) FROM cr_interactions').fetchone()[0]
    scores = conn.execute('SELECT COUNT(*) FROM cr_session_scores').fetchone()[0]
    return ctx, ints, scores
ctx_count, int_count, score_count = execute_with_retry(counts)

# Benchmark ranking
from infinidev.engine.context_rank.ranker import rank
t0 = time.perf_counter()
result = rank('test query about tools and files', 'bench-session', 'bench-task', 0)
t1 = time.perf_counter()
rank_ms = (t1 - t0) * 1000

print(f'   📊 contexts={ctx_count}  interactions={int_count}  scores={score_count}  rank_time={rank_ms:.1f}ms  files={len(result.files)}  symbols={len(result.symbols)}')
" 2>&1
  echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "Final benchmark..."
python3 -c "
import time, sys
sys.path.insert(0, '/home/andres/infinidev/src')
from infinidev.config.settings import settings
settings.DB_PATH = '${PROJECT_DIR}/.infinidev/infinidev.db'
settings.CONTEXT_RANK_MIN_SIMILARITY = 0.4
from infinidev.tools.base.context import set_context
set_context(project_id=1, workspace_path='${PROJECT_DIR}')
from infinidev.code_intel._db import execute_with_retry
from infinidev.engine.context_rank.ranker import rank
from infinidev.engine.loop.context import _render_context_rank
import time

# Benchmark 10 different queries
queries = [
    'How does the tool system work?',
    'Show me the CLI entry point and boot sequence',
    'What providers are supported for LLM?',
    'How does file editing work in this project?',
    'Read the bash tool and explain sandboxing',
    'What is the agent loop?',
    'How are sessions managed?',
    'What MCP capabilities does this have?',
    'Show me the permission system',
    'How does the LSP server work?',
]

print('╔══════════════════════════════════════════════════════════╗')
print('║          Final Benchmark — 10 diverse queries           ║')
print('╚══════════════════════════════════════════════════════════╝')
print()

times = []
for q in queries:
    t0 = time.perf_counter()
    result = rank(q, 'bench-final', 'bench-final', 0)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    times.append(ms)

    rendered = _render_context_rank(result)
    cr_lines = len(rendered.split(chr(10))) if rendered else 0

    print(f'  {ms:6.1f}ms  files={len(result.files)}  syms={len(result.symbols)}  lines={cr_lines:2d}  {q[:50]}')

print()
print(f'  Avg: {sum(times)/len(times):.1f}ms  Min: {min(times):.1f}ms  Max: {max(times):.1f}ms')

# Show data volume
def counts(conn):
    ctx = conn.execute('SELECT COUNT(*) FROM cr_contexts').fetchone()[0]
    ints = conn.execute('SELECT COUNT(*) FROM cr_interactions').fetchone()[0]
    scores = conn.execute('SELECT COUNT(*) FROM cr_session_scores').fetchone()[0]
    sessions = conn.execute('SELECT COUNT(DISTINCT session_id) FROM cr_session_scores').fetchone()[0]
    embs = conn.execute('SELECT COUNT(*) FROM cr_contexts WHERE embedding IS NOT NULL').fetchone()[0]
    return ctx, ints, scores, sessions, embs
ctx, ints, scores, sessions, embs = execute_with_retry(counts)
print()
print(f'  Data: {ctx} contexts ({embs} embedded), {ints} interactions, {scores} score entries, {sessions} sessions')
" 2>&1
