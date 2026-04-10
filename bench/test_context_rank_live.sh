#!/usr/bin/env bash
# Live test of ContextRank against /home/andres/opencode
# Uses qwen3.5:9b via Ollama, runs 3 sequential tasks to build up
# interaction data, then queries the CR tables.

set -euo pipefail

PROJECT_DIR="/home/andres/opencode"
DB_PATH="${PROJECT_DIR}/.infinidev/infinidev.db"
INFINIDEV="uv run --project /home/andres/infinidev infinidev"

# Ensure clean state
rm -f "$DB_PATH"
mkdir -p "${PROJECT_DIR}/.infinidev"

# Write temporary settings for this test
cat > "${PROJECT_DIR}/.infinidev/settings.json" <<'EOF'
{
  "LLM_PROVIDER": "ollama",
  "LLM_MODEL": "ollama_chat/qwen3.5:9b",
  "LLM_BASE_URL": "http://localhost:11434",
  "LLM_API_KEY": "ollama",
  "EXECUTE_COMMANDS_PERMISSION": "auto_approve",
  "FILE_OPERATIONS_PERMISSION": "auto_approve",
  "CONTEXT_RANK_LOGGING_ENABLED": true,
  "CONTEXT_RANK_ENABLED": true,
  "LOOP_MAX_ITERATIONS": 8,
  "LOOP_MAX_TOTAL_TOOL_CALLS": 30,
  "ANALYSIS_ENABLED": false,
  "REVIEW_ENABLED": false,
  "THINKING_ENABLED": false,
  "CODE_INTEL_ENABLED": true
}
EOF

echo "=== ContextRank Live Test ==="
echo "Project: $PROJECT_DIR"
echo "Model: qwen3.5:9b"
echo ""

# ── Task 1 ────────────────────────────────────────────────────────────
echo "▶ Task 1: Explore project structure"
cd "$PROJECT_DIR"
$INFINIDEV --prompt "List the main directories in packages/opencode/src and tell me what each one does. Read the main entry point file." 2>&1 | tail -5
echo ""
echo "── CR tables after Task 1 ──"
sqlite3 "$DB_PATH" "SELECT COUNT(*) as contexts FROM cr_contexts;"
sqlite3 "$DB_PATH" "SELECT COUNT(*) as interactions FROM cr_interactions;"
sqlite3 "$DB_PATH" "SELECT event_type, COUNT(*) as cnt FROM cr_interactions GROUP BY event_type ORDER BY cnt DESC;"
echo ""

# ── Task 2 ────────────────────────────────────────────────────────────
echo "▶ Task 2: Understand the tool system"
$INFINIDEV --prompt "Read the tool system in packages/opencode/src/tool/ directory. List all available tools and explain how they are registered." 2>&1 | tail -5
echo ""
echo "── CR tables after Task 2 ──"
sqlite3 "$DB_PATH" "SELECT COUNT(*) as contexts FROM cr_contexts;"
sqlite3 "$DB_PATH" "SELECT COUNT(*) as interactions FROM cr_interactions;"
sqlite3 "$DB_PATH" "SELECT target, SUM(weight) as total_weight FROM cr_interactions GROUP BY target ORDER BY total_weight DESC LIMIT 10;"
echo ""

# ── Task 3 ────────────────────────────────────────────────────────────
echo "▶ Task 3: Similar task (should trigger predictive ranking)"
$INFINIDEV --prompt "How does the tool registration work? Read the tool index file and the base tool class." 2>&1 | tail -5
echo ""
echo "── CR tables after Task 3 ──"
sqlite3 "$DB_PATH" "SELECT COUNT(*) as contexts FROM cr_contexts;"
sqlite3 "$DB_PATH" "SELECT COUNT(*) as interactions FROM cr_interactions;"
echo ""
echo "── Session scores (snapshots) ──"
sqlite3 "$DB_PATH" "SELECT target, target_type, score, access_count FROM cr_session_scores ORDER BY score DESC LIMIT 15;"
echo ""

# ── Final analysis ────────────────────────────────────────────────────
echo "=== Context Escalera ==="
sqlite3 "$DB_PATH" "SELECT context_type, COUNT(*) as cnt, SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding FROM cr_contexts GROUP BY context_type;"
echo ""
echo "=== Top targets across all sessions ==="
sqlite3 "$DB_PATH" "SELECT i.target, i.target_type, COUNT(*) as accesses, SUM(i.weight) as total_weight FROM cr_interactions i GROUP BY i.target, i.target_type ORDER BY total_weight DESC LIMIT 15;"
echo ""
echo "=== Done ==="
