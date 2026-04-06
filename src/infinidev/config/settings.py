"""Centralized configuration for Infinidev CLI."""

import os
import json
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directory: .infinidev inside the current working directory.
# This keeps settings, DB, and logs project-local.
DEFAULT_BASE_DIR = Path.cwd() / ".infinidev"
DEFAULT_BASE_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = DEFAULT_BASE_DIR / "settings.json"

class Settings(BaseSettings):
    # Database
    DB_PATH: str = str(DEFAULT_BASE_DIR / "infinidev.db")
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
    LLM_PROVIDER: str = "ollama"  # Provider ID: ollama, llama_cpp, vllm, openai, anthropic, gemini, zai, kimi, minimax, openrouter, qwen, openai_compatible
    LLM_MODEL: str = "ollama_chat/qwen2.5-coder:7b"
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_API_KEY: str = "ollama"
    LLM_TIMEOUT: int = 1800  # Request timeout in seconds (default 30 min for large local models)
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

    # Prompt Caching
    PROMPT_CACHE_ENABLED: bool = True  # Enable provider-specific prompt caching

    # Prompt Style
    PROMPT_STYLE: str = "auto"  # "auto", "full", "generalized", "coding", "extra_simple"

    # UI
    MARKDOWN_MESSAGES: bool = False  # Render LLM responses with markdown styling

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



    model_config = {"env_prefix": "INFINIDEV_"}

    @classmethod
    def load_user_settings(cls):
        """Load settings from JSON file and env vars."""
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
