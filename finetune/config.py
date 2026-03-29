"""Configuration for the fine-tuning dataset pipeline."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
REPOS_DIR = BASE_DIR / "repos"
OUTPUT_DIR = BASE_DIR / "output"
SCENARIOS_DIR = OUTPUT_DIR / "scenarios"
DATASET_DIR = OUTPUT_DIR / "dataset"

# ── Repos to download ────────────────────────────────────────────────────────

# Small-to-medium projects with clear structure, tests, and variety
REPOS = [
    # ── Python ────────────────────────────────────────────────────────────
    {"url": "https://github.com/pallets/flask", "branch": "main", "name": "flask", "lang": "python"},
    {"url": "https://github.com/encode/httpx", "branch": "master", "name": "httpx", "lang": "python"},
    {"url": "https://github.com/pallets/click", "branch": "main", "name": "click", "lang": "python"},
    {"url": "https://github.com/Textualize/rich", "branch": "master", "name": "rich", "lang": "python"},
    {"url": "https://github.com/tqdm/tqdm", "branch": "master", "name": "tqdm", "lang": "python"},
    {"url": "https://github.com/pydantic/pydantic-settings", "branch": "main", "name": "pydantic-settings", "lang": "python"},
    {"url": "https://github.com/marshmallow-code/marshmallow", "branch": "dev", "name": "marshmallow", "lang": "python"},
    {"url": "https://github.com/pytest-dev/pluggy", "branch": "main", "name": "pluggy", "lang": "python"},
    {"url": "https://github.com/Delgan/loguru", "branch": "master", "name": "loguru", "lang": "python"},
    {"url": "https://github.com/ijl/orjson", "branch": "master", "name": "orjson", "lang": "python"},

    # ── TypeScript ────────────────────────────────────────────────────────
    {"url": "https://github.com/colinhacks/zod", "branch": "main", "name": "zod", "lang": "typescript"},
    {"url": "https://github.com/trpc/trpc", "branch": "main", "name": "trpc", "lang": "typescript"},
    {"url": "https://github.com/Effect-TS/effect", "branch": "main", "name": "effect", "lang": "typescript"},
    {"url": "https://github.com/pmndrs/zustand", "branch": "main", "name": "zustand", "lang": "typescript"},
    {"url": "https://github.com/sindresorhus/got", "branch": "main", "name": "got", "lang": "typescript"},
    {"url": "https://github.com/date-fns/date-fns", "branch": "main", "name": "date-fns", "lang": "typescript"},
    {"url": "https://github.com/unjs/nitro", "branch": "main", "name": "nitro", "lang": "typescript"},
    {"url": "https://github.com/honojs/hono", "branch": "main", "name": "hono", "lang": "typescript"},

    # ── Rust ──────────────────────────────────────────────────────────────
    {"url": "https://github.com/serde-rs/serde", "branch": "master", "name": "serde", "lang": "rust"},
    {"url": "https://github.com/tokio-rs/tokio", "branch": "master", "name": "tokio", "lang": "rust"},
    {"url": "https://github.com/clap-rs/clap", "branch": "master", "name": "clap", "lang": "rust"},
    {"url": "https://github.com/BurntSushi/ripgrep", "branch": "master", "name": "ripgrep", "lang": "rust"},
    {"url": "https://github.com/sharkdp/bat", "branch": "master", "name": "bat", "lang": "rust"},
    {"url": "https://github.com/sharkdp/fd", "branch": "master", "name": "fd", "lang": "rust"},
    {"url": "https://github.com/BurntSushi/regex", "branch": "master", "name": "regex-rs", "lang": "rust"},
    {"url": "https://github.com/rayon-rs/rayon", "branch": "main", "name": "rayon", "lang": "rust"},

    # ── C ─────────────────────────────────────────────────────────────────
    {"url": "https://github.com/redis/redis", "branch": "unstable", "name": "redis", "lang": "c"},
    {"url": "https://github.com/jqlang/jq", "branch": "master", "name": "jq", "lang": "c"},
    {"url": "https://github.com/DaveGamble/cJSON", "branch": "master", "name": "cjson", "lang": "c"},
    {"url": "https://github.com/antirez/sds", "branch": "master", "name": "sds", "lang": "c"},
    {"url": "https://github.com/sqlite/sqlite", "branch": "master", "name": "sqlite", "lang": "c"},
    {"url": "https://github.com/madler/zlib", "branch": "develop", "name": "zlib", "lang": "c"},

    # ── Go ────────────────────────────────────────────────────────────────
    {"url": "https://github.com/junegunn/fzf", "branch": "master", "name": "fzf", "lang": "go"},
    {"url": "https://github.com/charmbracelet/bubbletea", "branch": "master", "name": "bubbletea", "lang": "go"},
    {"url": "https://github.com/charmbracelet/lipgloss", "branch": "master", "name": "lipgloss", "lang": "go"},
    {"url": "https://github.com/spf13/cobra", "branch": "main", "name": "cobra", "lang": "go"},

    # ── Java ──────────────────────────────────────────────────────────────
    {"url": "https://github.com/google/gson", "branch": "main", "name": "gson", "lang": "java"},
    {"url": "https://github.com/google/guava", "branch": "master", "name": "guava", "lang": "java"},
]

# ── Scenario types ───────────────────────────────────────────────────────────

SCENARIO_TYPES = [
    "bug_fix",           # Fix a bug in existing code
    "add_method",        # Add a new method to a class
    "add_function",      # Add a standalone function
    "modify_method",     # Change behavior of existing method
    "add_import",        # Add an import and use it
    "refactor_rename",   # Rename a symbol across project
    "refactor_extract",  # Extract a method from a large function
    "refactor_move",     # Move a function/method to another file
    "add_test",          # Write a test for existing code
    "fix_import",        # Fix a broken import
    "config_change",     # Modify a config file
    "documentation",     # Add/update docstrings
]

# ── Dataset parameters ───────────────────────────────────────────────────────

# Target examples per scenario type per repo
EXAMPLES_PER_TYPE_PER_REPO = 3

# Max file size to include in examples (chars)
MAX_FILE_CONTENT = 30000

# Qwen ChatML tokens
CHATML_SYSTEM = "<|im_start|>system\n"
CHATML_USER = "<|im_start|>user\n"
CHATML_ASSISTANT = "<|im_start|>assistant\n"
CHATML_END = "<|im_end|>\n"
