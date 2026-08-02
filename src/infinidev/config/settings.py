"""Centralized configuration for Infinidev CLI."""

import json
import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

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
    # wait_for_background_task: default block when the caller omits a timeout,
    # and a hard ceiling so a single wait can never freeze the CLI indefinitely.
    BACKGROUND_WAIT_TIMEOUT: int = 120
    BACKGROUND_WAIT_MAX_TIMEOUT: int = 600

    # Sandbox (Disabled for local CLI by default)
    SANDBOX_ENABLED: bool = False
    ALLOWED_BASE_DIRS: list[str] = ["/"]  # Allow all for local CLI
    ALLOWED_COMMANDS: list[str] = []  # Not used if SANDBOX_ENABLED=False

    # Permissions
    # "auto" (default): auto-approve provably read-only commands / in-workspace
    # edits, and prompt for anything risky. Other modes: "auto_approve" (allow
    # everything), "ask" (prompt for everything), "allowed_list"/"allowed_paths"
    # (allow only an explicit list).
    EXECUTE_COMMANDS_PERMISSION: str = "auto"  # "auto", "auto_approve", "ask", "allowed_list"
    ALLOWED_COMMANDS_LIST: list[str] = []  # Allowed commands when permission is "allowed_list"
    FILE_OPERATIONS_PERMISSION: str = "auto"  # "auto", "ask", "auto_approve", "allowed_paths"
    ALLOWED_FILE_PATHS: list[str] = []  # Allowed paths when permission is "allowed_paths"

    # File limits
    MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5MB
    MAX_DIR_LISTING: int = 1000

    # Private command-output capture. Disabled by default so execute_command's
    # return shape and filesystem/database side effects stay backward compatible.
    # Enabling is fail-closed unless every bound below is a positive integer.
    COMMAND_OUTPUT_CAPTURE_ENABLED: bool = False
    COMMAND_OUTPUT_MAX_ARTIFACT_BYTES: int = 10 * 1024 * 1024
    COMMAND_OUTPUT_MAX_SESSION_BYTES: int = 100 * 1024 * 1024
    COMMAND_OUTPUT_MAX_PROJECT_BYTES: int = 500 * 1024 * 1024
    COMMAND_OUTPUT_STORE_TIMEOUT_SECONDS: int = 5
    COMMAND_OUTPUT_RETENTION_SECONDS: int = 7 * 24 * 60 * 60
    COMMAND_OUTPUT_SWEEP_GRACE_SECONDS: int = 60 * 60
    # Independent opt-ins: capture may expose a handle without creating durable
    # notes, and note compaction can be evaluated separately from note creation.
    COMMAND_OUTPUT_AUTO_NOTES_ENABLED: bool = False
    COMMAND_OUTPUT_NOTE_COMPACTION_ENABLED: bool = False

    # LLM (via LiteLLM)
    # Provider ID — the authoritative list is PROVIDERS in config/providers.py.
    # `openai_subscription` bills against a ChatGPT plan via `codex login`
    # instead of an API key; it ignores LLM_API_KEY and LLM_BASE_URL.
    LLM_PROVIDER: str = "ollama"
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
    # Deterministic per-step objective verification: when a planner-authored
    # step carries an executable ``verify`` check, run it on step_complete and
    # block closure (with the failure output) until it passes.
    LOOP_OBJECTIVE_VERIFY_ENABLED: bool = True
    # Max correction turns forced per step before the engine stops blocking
    # (to avoid starving the global budget on one stuck objective). On
    # exhaustion the step is allowed to close but the unmet objective is
    # logged and noted so it surfaces rather than silently passing.
    LOOP_OBJECTIVE_VERIFY_MAX_ATTEMPTS: int = 3
    LOOP_VALIDATE_SYNTAX_BEFORE_WRITE: bool = True  # tree-sitter syntax check before writing files
    LOOP_GUIDANCE_ENABLED: bool = True  # Inject pre-baked how-to advice when small models get stuck
    LOOP_GUIDANCE_MAX_PER_TASK: int = 3  # Hard cap on guidance entries per task
    # End-of-task work summary: a hidden conversation turn recording what
    # the developer loop did (files + why, per-file changes, challenges) so
    # the NEXT turn's chat agent has continuity instead of starting cold.
    LOOP_WORK_SUMMARY_ENABLED: bool = True  # Generate the hidden end-of-task summary
    LOOP_WORK_SUMMARY_USE_LLM: bool = True  # Synthesize via the model (vs. deterministic assembly only)

    # ── Spec elaboration ─────────────────────────────────────────────
    # Turns a vague requirement into a grounded spec BEFORE planning.
    # Runs once per task between the chat agent's escalation and the
    # planner, on the single configured model. See engine/analysis/
    # spec_elaborator.py and docs_spec_elaboration_loop.md.
    SPEC_ELABORATION_ENABLED: bool = True
    SPEC_ELABORATION_MIN_CHARS: int = 40  # Skip elaboration for requests shorter than this (trivial)
    SPEC_ELABORATION_MAX_EVIDENCE_CALLS: int = 4  # Read-only tool calls budget in the grounding pass
    SPEC_ELABORATION_CANDIDATES: int = 3  # N candidate design directions generated in the critique pass
    # Hard ceiling on the product decisions surfaced to the user per task.
    # A senior engineer raises at most one or two before starting; anything
    # past this is demoted to a stated assumption, never dropped. 0 disables
    # asking entirely (every decision becomes an assumption).
    SPEC_ELABORATION_MAX_CLARIFICATIONS: int = 2
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
    TREE_BRAINSTORM_MAX_DEPTH: int = 2  # ideas + 1 level max
    TREE_BRAINSTORM_INNER_LOOP_MAX: int = 4  # quick validation per idea
    TREE_BRAINSTORM_TOOL_CALLS_PER_NODE: int = 3  # few lookups, not exhaustive

    # Phases
    ANALYSIS_ENABLED: bool = True
    REVIEW_ENABLED: bool = True
    # Post-loop objective re-verification: at task end, re-run every
    # planner-authored step verification together (a backstop for the
    # cross-objective regression the per-step gate cannot see) and feed any
    # failures back to the developer. Bounded by MAX_ROUNDS re-execution cycles.
    REVIEW_OBJECTIVE_REVERIFY_ENABLED: bool = True
    REVIEW_OBJECTIVE_REVERIFY_MAX_ROUNDS: int = 2
    # Adversarial LLM verifier for soft objectives (verify_kind='llm_judge'):
    # an independent, skeptical judge with a cited-evidence verdict that is
    # substring-grounded against the diff. Runs only at task end (one LLM call
    # per soft objective per round). Uses the assistant model when configured.
    REVIEW_ADVERSARIAL_VERIFY_ENABLED: bool = True

    # Council (multi-agent deliberation, opt-in design/research phase).
    # The feature is available by default but only FIRES when the chat
    # agent flags council_requested (user asked for it, or judged the
    # task complex). It is expensive, so it never runs implicitly.
    COUNCIL_ENABLED: bool = True
    # Optional model override for ALL council agents (members + moderator).
    # Empty → reuse the behavior-judge model (same as chat agent/planner).
    # Point this at a cloud provider for real parallelism, since local
    # Ollama serialises concurrent members at the GPU.
    COUNCIL_MODEL: str = ""
    COUNCIL_MAX_MEMBERS: int = 5  # hard upper bound on roster size
    COUNCIL_MAX_ROUNDS: int = 3  # runaway guard; moderator may stop earlier
    # Iterations per member per round (each = one LLM call). This is a
    # runaway guard, NOT a quality gate: a member should be able to
    # investigate deeply — chain many web_search/web_fetch calls, read
    # several files, follow references — before it channel_posts or
    # concludes. Keep it generous so research is never cut off
    # mid-investigation; the loop terminates as soon as the member calls
    # a terminator, so most turns end well under the cap.
    COUNCIL_MEMBER_MAX_ITERS: int = 60
    # Iterations for the moderator's exploration-heavy turns (seeding the
    # roster, and synthesising the final brief). Generous like the member
    # cap: framing good personas and grounding the brief both benefit
    # from real codebase/web exploration. The convergence-judge turn is
    # separately capped low (it just needs to read the digest and vote).
    COUNCIL_MODERATOR_MAX_ITERS: int = 60
    COUNCIL_MAX_CONCURRENCY: int = 4  # members run in parallel up to this many
    # Off by default: the chat agent may PROPOSE a council on complexity,
    # but auto-triggering without a user signal risks the model's own
    # over-estimation. Reserved for a future opt-in.
    COUNCIL_AUTO_TRIGGER: bool = False

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
    # Transcript-first by default: the sidebar's panels (context, thinking,
    # steps, actions, logs) are all still there, one Alt+. away, but they
    # no longer take a third of the terminal before the user asks for them.
    UI_SIDEBAR_VISIBLE: bool = False
    # Behavior Checkers (modular punish/promote scoring after each model message)
    BEHAVIOR_CHECKERS_ENABLED: bool = False  # Master toggle
    BEHAVIOR_HISTORY_WINDOW: int = 4  # Recent messages fed to each checker
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

    # Assistant LLM — pair-programming critic running on a second GPU.
    # When enabled, a second model runs in parallel with the principal's
    # tool execution and emits a short observation that is injected into
    # the next iteration's context. Purely informative: never blocks,
    # never forces retries, never vetoes. The principal reads the
    # observation and decides what to do.
    # Each ASSISTANT_LLM_* field is "" by default → falls back to the
    # matching LLM_* main setting, so a typical setup only needs to
    # toggle ENABLED + override MODEL/BASE_URL.
    ASSISTANT_LLM_ENABLED: bool = False
    ASSISTANT_LLM_PROVIDER: str = ""
    ASSISTANT_LLM_MODEL: str = ""
    ASSISTANT_LLM_BASE_URL: str = ""
    ASSISTANT_LLM_API_KEY: str = ""
    ASSISTANT_LLM_TIMEOUT: int = 600
    ASSISTANT_LLM_INCLUDE_STEP_COMPLETE: bool = True

    # Runtime task scheduler and unified conversation memory
    RUNTIME_ENABLED: bool = True
    RUNTIME_MEMORY_KEEP: int = 12
    RUNTIME_PERSIST_EVENTS: bool = True
    # Working memory: raw tool output evicted from the prompt at each step
    # close is archived and indexed so the model can pull it back with
    # ``recall_context`` instead of re-running the read. See
    # engine/working_memory.py.
    WORKING_MEMORY_ENABLED: bool = True
    # How many full step summaries stay in the prompt. Older ones collapse
    # to one line each and remain retrievable via recall_context.
    # 0 = never collapse (old behaviour: every summary stays verbatim).
    WORKING_MEMORY_VERBATIM_STEPS: int = 4

    # User hooks: shell commands bound to the six lifecycle events and
    # declared in .infinidev/hooks.json. Nothing ships enabled — with no
    # config file the whole subsystem is inert — so this switch exists to
    # turn off hooks a user already wrote, not to opt into the feature.
    # See engine/user_hooks/.
    HOOKS_ENABLED: bool = True
    # Fallback deadline per hook, overridable per hook in the config. A
    # hook runs between two steps with the loop waiting on it, so an
    # unbounded one would look like a hang with no explanation.
    HOOKS_TIMEOUT: int = 60

    # MCP — generic Model Context Protocol client. Ken is the default server
    # but any number of MCP servers can be registered via .mcp.json.
    MCP_ENABLED: bool = True
    MCP_AUTOLOAD_CONFIG: bool = True
    MCP_REQUEST_TIMEOUT: int = 30
    # Handshake budget: a cold Ken loads its embedding model before it can
    # answer `initialize`, which is far slower than a steady-state call.
    MCP_STARTUP_TIMEOUT: int = 60
    # Which discovered MCP tools reach the model, as a comma-separated glob
    # list ("*" = every tool the servers advertise). Every tool a server
    # publishes is exposed by default under its own name; narrowing this is
    # for small models, where a hundred-tool schema hurts selection more
    # than the extra capability helps. Example: "ken_search_*,ken_rank,ken_recall".
    MCP_TOOL_FILTER: str = "*"
    # Report the session to Ken's daemon (start/prompt/tool/turn/end), so its
    # ranker gets the event stream its reactive, predictive and
    # explicit-mention channels are computed from. Without it Ken answers
    # with only the name/text channels — measured: reactive 0, explicit 0,
    # findings 0 — because a query string cannot say what this session has
    # been doing. Off by default: it changes what the model sees, so it
    # wants measuring before it becomes the default.
    KEN_SESSION_ENABLED: bool = False

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
            except (json.JSONDecodeError, OSError) as e:
                # Narrow + log via the logger (not print, which is invisible in
                # the TUI and absent from log files). Falling back to defaults
                # silently runs on the wrong model / missing API key, so the
                # failure must be observable.
                logger.error(
                    "Could not load settings from %s (%s); falling back to "
                    "defaults — model/base_url/API keys may be wrong.",
                    SETTINGS_FILE,
                    e,
                )

        # Env vars take precedence over file settings
        return cls(**file_settings)

    def save_user_settings(self, updates: dict):
        """Save specific updates to the settings file."""
        current_data = {}
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r") as f:
                    current_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # Do NOT fall through to overwrite. Defaulting to {} here and
                # then dumping would clobber every previously-saved setting
                # (model, base_url, API keys) on a transient or corrupt read.
                # Abort and leave the existing file untouched instead.
                logger.error(
                    "Refusing to save settings: cannot read existing %s (%s). "
                    "Existing configuration left untouched.",
                    SETTINGS_FILE,
                    e,
                )
                return

        current_data.update(updates)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(current_data, f, indent=2)


def reload_all():
    """Reload settings from file and update the global instance."""
    new_s = Settings.load_user_settings()
    for key, value in new_s.model_dump().items():
        setattr(settings, key, value)


settings = Settings.load_user_settings()
