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
    LLM_MODEL: str = "ollama_chat/qwen2.5-coder:7b"
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_API_KEY: str = "ollama"
    LLM_TIMEOUT: int = 1800  # Request timeout in seconds (default 30 min for large local models)

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
