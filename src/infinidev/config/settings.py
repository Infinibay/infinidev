"""Centralized configuration for Infinidev CLI."""

import json
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Base directory: .infinidev in the current working directory.
# Using cwd ensures the DB is relative to wherever the engine runs.
# NOTE: get_base_dir() and get_settings_file() are called lazily at runtime
# (not at module import time), so they always reflect the actual cwd.
SETTINGS_FILE_NAME = "settings.json"
DB_FILE_NAME = "infinidev.db"


def _get_base_dir() -> Path:
    """Return the .infinidev directory path relative to the current working directory.

    Recomputed on every call so it always tracks the real cwd, even if the
    process has changed directory since import time. Creates the directory if
    it does not exist.
    """
    base = Path.cwd() / ".infinidev"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_base_dir() -> Path:
    """Return the .infinidev base directory path.

    This is the root directory for all engine data (DB, settings, logs) and
    is always relative to the current working directory at call time.
    """
    return _get_base_dir()


def get_settings_file() -> Path:
    """Return the path to the settings.json file.

    The file lives inside .infinidev in the current working directory.
    """
    return _get_base_dir() / SETTINGS_FILE_NAME


def get_db_path() -> Path:
    """Return the path to the SQLite database file (infinidev.db).

    The DB lives inside .infinidev in the current working directory.
    """
    return _get_base_dir() / DB_FILE_NAME

# Backward-compatibility alias: module-level Path so tests can patch it.
# Runtime code should prefer get_settings_file() / get_db_path() / get_base_dir()
# which always track the real cwd.
SETTINGS_FILE = get_settings_file()

class Settings(BaseSettings):
    # Database
    DB_PATH: str = Field(default_factory=lambda: str(get_db_path()))
    MAX_RETRIES: int = 5
    RETRY_BASE_DELAY: float = 0.1

    # Timeouts
    COMMAND_TIMEOUT: int = 120
    WEB_TIMEOUT: int = 30
    GIT_PUSH_TIMEOUT: int = 120

    # Sandbox (Disabled for local CLI by default)
    SANDBOX_ENABLED: bool = False
    ALLOWED_BASE_DIRS: list[str] = ["/"]  # Allow all for local CLI
    ALLOWED_COMMANDS: list[str] = []      # Not used if SANDBOX_ENABLED=False

    # Permissions
    EXECUTE_COMMANDS_PERMISSION: str = "ask"  # "auto_approve", "ask", "allowed_list"
    ALLOWED_COMMANDS_LIST: list[str] = []  # List of allowed commands when permission is "allowed_list"
    FILE_OPERATIONS_PERMISSION: str = "ask"  # "ask", "auto_approve", "allowed_paths"
    ALLOWED_FILE_PATHS: list[str] = []  # List of allowed paths when permission is "allowed_paths"

    # File limits
    MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5MB
    MAX_DIR_LISTING: int = 1000

    # LLM (via LiteLLM)
    LLM_PROVIDER: str = "ollama"  # Provider ID: ollama, llama_cpp, vllm, openai, anthropic, gemini, zai, zai_coding, kimi, minimax, openrouter, qwen, openai_compatible
    LLM_MODEL: str = "ollama_chat/qwen2.5-coder:7b"
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_API_KEY: str = "ollama"
    LLM_TIMEOUT: int = 1800  # Request timeout in seconds (default 30 min for large local models)
    LLM_NUM_RETRIES: int = 3  # Retry transient provider errors (OpenRouter mid-stream drops, 5xx, timeouts)
    LLM_TEMPERATURE: float = 0.2  # Default temp for the developer loop. Low values favour reliable tool-calling and deterministic edits. Set < 0 to defer to the model/provider default.
    OLLAMA_NUM_CTX: int = 16384  # Context window for Ollama models (0 = use model default)

    # Thinking / Reasoning
    # NOTE: Anthropic, OpenAI, and Gemini enforce thinking budgets server-side.
    # Local providers (Ollama, llama.cpp, vLLM) use prompt tags (/no_think)
    # which the model may ignore — disabling thinking is best-effort only.
    THINKING_ENABLED: bool = True  # Master toggle �� disables all reasoning when False
    # Budget presets: "low", "medium", "high", "ultra", "custom"
    THINKING_BUDGET: str = "medium"
    THINKING_BUDGET_TOKENS: int = 4096  # Used when THINKING_BUDGET="custom"

    # Embedding / Knowledge
    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_BASE_URL: str = "http://localhost:11434"

    # Loop Engine (plan-execute-summarize)
    LOOP_MAX_ITERATIONS: int = 50
    LOOP_MAX_TOOL_CALLS_PER_ACTION: int = 0  # 0 = unlimited (only global limit applies)
    LOOP_MAX_TOTAL_TOOL_CALLS: int = 1000
    LOOP_HISTORY_WINDOW: int = 0  # 0 = keep all
    LOOP_STEP_NUDGE_THRESHOLD: int = 6  # Nudge agent to call step_complete after N tool calls
    LOOP_SUMMARIZER_ENABLED: bool = True  # Use dedicated LLM call for step summaries
    LOOP_SUMMARIZER_MAX_INPUT_TOKENS: int = 4000  # Max tokens from step messages to feed summarizer
    LOOP_SUMMARIZER_TIMEOUT: int = 30  # Seconds; falls back to raw summary on timeout
    LOOP_REQUIRE_NOTE_BEFORE_COMPLETE: bool = True  # Gate step_complete on add_note for small models
    LOOP_VALIDATE_SYNTAX_BEFORE_WRITE: bool = True  # tree-sitter syntax check before writing files
    LOOP_GUIDANCE_ENABLED: bool = True  # Inject pre-baked how-to advice when small models get stuck
    LOOP_GUIDANCE_MAX_PER_TASK: int = 3  # Hard cap on guidance entries per task
    # NB: static-analysis latency reporting is opt-in via the
    # INFINIDEV_ENABLE_SA_TIMER env var, not a settings field — see
    # ``engine.static_analysis_timer.is_enabled``.
    # Comma-separated substrings that mark project-specific test runners
    # (e.g. "bash test.sh,make integration"). Added on top of the built-in
    # runner list (pytest/jest/cargo/etc.) used by the guidance detector.
    LOOP_CUSTOM_TEST_COMMANDS: str = ""

    # Gather phase (pre-implementation info collection)
    GATHER_ENABLED: bool = False
    GATHER_MAX_TOOL_CALLS_PER_QUESTION: int = 30
    GATHER_QUESTION_TIMEOUT: int = 120
    GATHER_MAX_DYNAMIC_QUESTIONS: int = 10

    # Code Intelligence (tree-sitter indexing)
    CODE_INTEL_ENABLED: bool = True
    CODE_INTEL_MAX_FILE_SIZE: int = 1_000_000
    CODE_INTEL_AUTO_INDEX: bool = True

    # Web tools
    WEB_CACHE_TTL_SECONDS: int = 3600
    WEB_RPM_LIMIT: int = 20
    WEB_ROBOTS_CACHE_TTL: int = 3600

    # Semantic dedup
    DEDUP_SIMILARITY_THRESHOLD: float = 0.82

    # Workspace
    WORKSPACE_BASE_DIR: str = str(Path.cwd())

    # Code Interpreter
    CODE_INTERPRETER_TIMEOUT: int = 120
    CODE_INTERPRETER_MAX_OUTPUT: int = 50000

    # Tree Exploration Engine
    TREE_MAX_NODES: int = 20
    TREE_MAX_DEPTH: int = 4
    TREE_MAX_CHILDREN: int = 4
    TREE_MAX_LLM_CALLS: int = 200
    TREE_MAX_TOOL_CALLS: int = 200
    TREE_MAX_TOOL_CALLS_PER_NODE: int = 20
    TREE_INNER_LOOP_MAX: int = 8

    # Brainstorm-specific limits (wide & shallow exploration)
    TREE_BRAINSTORM_MAX_DEPTH: int = 2          # ideas + 1 level max
    TREE_BRAINSTORM_INNER_LOOP_MAX: int = 4     # quick validation per idea
    TREE_BRAINSTORM_TOOL_CALLS_PER_NODE: int = 3  # few lookups, not exhaustive

    # Phases
    ANALYSIS_ENABLED: bool = True
    REVIEW_ENABLED: bool = True

    # Multi-pass code review: split extraction from judgment for complex diffs.
    # "off" = always single-pass | "auto" = split when complexity > threshold
    # | "always" = always two passes.
    REVIEW_MULTI_PASS_MODE: str = "auto"
    # Complexity score = changed_lines + 50 * changed_files. 400 ≈ 150 lines
    # across 5 files, or 400 lines in a single file.
    REVIEW_MULTI_PASS_COMPLEXITY_THRESHOLD: int = 400
    # Optional override for the extractor pass. Each is "" by default and
    # falls back to the main LLM_* setting. Point this at a cheap/fast model
    # (e.g. ollama/qwen2.5:3b) while keeping a heavy model for the judge.
    REVIEW_EXTRACTOR_LLM_PROVIDER: str = ""
    REVIEW_EXTRACTOR_LLM_MODEL: str = ""
    REVIEW_EXTRACTOR_LLM_BASE_URL: str = ""
    REVIEW_EXTRACTOR_LLM_API_KEY: str = ""

    # Prompt Caching
    PROMPT_CACHE_ENABLED: bool = True  # Enable provider-specific prompt caching

    # Prompt Style
    PROMPT_STYLE: str = "auto"  # "auto", "full", "generalized", "coding", "extra_simple"

    # UI
    MARKDOWN_MESSAGES: bool = False  # Render LLM responses with markdown styling
    DIFF_DISPLAY_MODE: str = "unified"  # "unified" (git diff) | "side_by_side"
    UI_SIDEBAR_VISIBLE: bool = True  # Right sidebar panel visibility (toggled with Alt+.)
    # Behavior Checkers (modular punish/promote scoring after each model message)
    BEHAVIOR_CHECKERS_ENABLED: bool = False  # Master toggle
    BEHAVIOR_HISTORY_WINDOW: int = 4         # Recent messages fed to each checker
    # "stochastic" (default, zero LLM calls) | "llm" (legacy batched judge)
    # | "hybrid" (stochastic first, escalate low-confidence to LLM)
    BEHAVIOR_JUDGE_MODE: str = "stochastic"
    # "per_step" (one evaluation per completed step, default) | "per_message"
    # (legacy: evaluate after every model message inside the inner loop)
    BEHAVIOR_CHECK_MODE: str = "per_step"
    # Below this confidence, hybrid mode escalates a stochastic verdict to LLM.
    BEHAVIOR_HYBRID_CONFIDENCE_THRESHOLD: float = 0.6
    # Cosine similarity at/above which RepetitiveThinkingChecker fires.
    BEHAVIOR_REPETITION_COSINE_THRESHOLD: float = 0.88
    # ChattyThinkingChecker triggers above this many reasoning characters.
    BEHAVIOR_CHATTY_CHAR_THRESHOLD: int = 2000
    # Independent LLM endpoint for the behavior judge.
    # Each field is "" by default → falls back to the main LLM_* setting.
    # Use this to point checkers at a small/fast model (e.g. ollama/qwen2.5:3b)
    # while the main agent runs on a heavier model.
    BEHAVIOR_LLM_PROVIDER: str = ""
    BEHAVIOR_LLM_MODEL: str = ""
    BEHAVIOR_LLM_BASE_URL: str = ""
    BEHAVIOR_LLM_API_KEY: str = ""
    BEHAVIOR_CHECKER_LAZY_WORK: bool = True
    BEHAVIOR_CHECKER_GOOD_FOCUS: bool = False
    BEHAVIOR_CHECKER_REPETITIVE_THINKING: bool = True
    BEHAVIOR_CHECKER_GRACEFUL_RECOVERY: bool = True
    BEHAVIOR_CHECKER_SMALL_SAFE_EDITS: bool = True
    BEHAVIOR_CHECKER_IGNORES_TOOL_ERROR: bool = True
    BEHAVIOR_CHECKER_SHELL_WHEN_TOOL_EXISTS: bool = True
    BEHAVIOR_CHECKER_PLAN_DRIFT: bool = True
    BEHAVIOR_CHECKER_CHATTY_THINKING: bool = False
    BEHAVIOR_CHECKER_FAKE_COMPLETION: bool = True
    BEHAVIOR_CHECKER_PROMPT_POLLUTION: bool = False
    BEHAVIOR_CHECKER_PLAN_QUALITY: bool = True

    # ContextRank (cross-session context prioritization)
    CONTEXT_RANK_ENABLED: bool = False
    CONTEXT_RANK_TOP_K_FILES: int = 5
    CONTEXT_RANK_TOP_K_SYMBOLS: int = 5
    CONTEXT_RANK_TOP_K_FINDINGS: int = 3
    # Exponential decay λ applied per iteration to reactive (in-session)
    # interactions.  At Δ=10 iterations a score drops to exp(-0.35*10)≈3%,
    # so actions from 10+ iterations ago effectively vanish.  Bumped from
    # 0.15 in v3 because 0.15 kept ~22% of the weight at Δ=10, which made
    # long tasks feel like every past action still mattered equally.
    CONTEXT_RANK_REACTIVE_DECAY: float = 0.35
    # Threshold for penalising "confusion" read patterns.  If the model
    # re-reads a file this many times without editing it, the reactive
    # score is damped by a multiplier < 1.0 (see _compute_reactive_scores).
    CONTEXT_RANK_REACTIVE_MANY_READS: int = 3
    # Per-week session decay applied to past contexts in the predictive
    # channel.  0.95^(days_ago/7): 1 week = 0.95, 4 weeks = 0.81,
    # 12 weeks = 0.57, 24 weeks = 0.32.  Lower = forget older sessions
    # faster.  Phase 2 v3 switched from decay^order_in_result to this
    # real-time formula — the old one penalised result position, not
    # actual age, and gave inconsistent decays depending on the fetch
    # LIMIT.
    CONTEXT_RANK_SESSION_DECAY: float = 0.95
    # Max age (in days) of historical contexts considered by the
    # predictive channel.  Contexts older than this are excluded at
    # SQL level to keep the fetch tight.  180 days ≈ 6 months, enough
    # for a long-running project to build meaningful cross-session
    # memory while dropping truly ancient noise.
    CONTEXT_RANK_CONTEXT_MAX_AGE_DAYS: int = 180
    # Hard upper bound on how many historical contexts the predictive
    # channel fetches per rank call.  Up from 500 (v2) because the
    # old cap was a temporal sample (recent ≠ relevant) that silently
    # dropped old-but-relevant contexts.  2000 × 384-dim cosine ≈ 4ms,
    # still well inside the per-pivot budget.
    CONTEXT_RANK_CONTEXT_FETCH_LIMIT: int = 2000
    # Max age (in days) for co-occurrence signal.  Co-occurrence pairs
    # older than this are excluded — stale "A always with B" edges
    # from refactored-away modules shouldn't keep boosting files.
    CONTEXT_RANK_COOC_MAX_AGE_DAYS: int = 90
    CONTEXT_RANK_MIN_SIMILARITY: float = 0.4
    CONTEXT_RANK_MIN_CONFIDENCE: float = 0.5
    CONTEXT_RANK_LOGGING_ENABLED: bool = True

    # Outlier detection — when a few suggestions score dramatically
    # higher than the rest, show only those (the rest are noise).
    #
    # CONTEXT_RANK_OUTLIER_PERCENTILE: what percentile of the noise
    # distribution a score must exceed to count as an outlier.
    # Accepts a number (95) or a percentage string ("95%" / "99.5%").
    # Higher = stricter (fewer outliers, higher confidence).
    #   90   → top 10% — aggressive (shows more suggestions)
    #   95   → top 5%  — default (balanced)
    #   99   → top 1%  — conservative
    #   99.7 → top 0.3% (very strict, classic 3-sigma)
    #
    # Rationale for 95% default: the cost of showing too many items
    # (wasted prompt tokens every iteration) is higher than the cost
    # of hiding a marginal item (the model can still read_file it).
    CONTEXT_RANK_OUTLIER_PERCENTILE: float | str = 95
    # Max number of outliers to show.  Above this, the "cluster" is
    # too large to be a clean signal → fall back to showing all items.
    CONTEXT_RANK_OUTLIER_MAX_COUNT: int = 3
    # Minimum top score required to attempt outlier filtering.
    # Below this, scores are too close to the confidence floor.
    CONTEXT_RANK_OUTLIER_MIN_TOP_SCORE: float = 1.0

    model_config = {"env_prefix": "INFINIDEV_"}

    @classmethod
    def load_user_settings(cls):
        file_settings = {}
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r") as f:
                    file_settings = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load settings from {SETTINGS_FILE}: {e}")

        # Env vars take precedence over file settings
        return cls(**file_settings)

    def save_user_settings(self, updates: dict):
        """Save specific updates to the settings file."""
        current_data = {}
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r") as f:
                    current_data = json.load(f)
            except Exception:
                pass

        current_data.update(updates)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(current_data, f, indent=2)


def reload_all():
    """Reload settings from file and update the global instance."""
    new_s = Settings.load_user_settings()
    for key, value in new_s.model_dump().items():
        setattr(settings, key, value)

settings = Settings.load_user_settings()
