"""Harness configuration for SWE-bench runs."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchConfig:
    # Model (LiteLLM format). Empty = use whatever is in
    # ~/.infinidev/settings.json. Setting an explicit value forwards
    # it via ``--model`` to the CLI, which only overrides
    # ``LLM_MODEL`` — not ``LLM_BASE_URL`` / ``LLM_PROVIDER`` /
    # ``LLM_API_KEY``. Combining a foreign model name with the
    # current settings' endpoint usually produces a mismatch, so the
    # safe default is to inherit from settings.
    model: str = ""
    # HuggingFace dataset name
    dataset: str = "princeton-nlp/SWE-bench_Lite"
    split: str = "test"
    # Working directory for repo checkouts
    workdir: Path = Path("/tmp/infinidev-bench")
    # Output predictions file
    output: Path = Path("bench/predictions.jsonl")
    # Seconds per instance (0 = no timeout)
    timeout: int = 0
    # 0 = run all instances
    max_instances: int = 0
    # Filter to specific instance IDs
    instance_ids: list[str] = field(default_factory=list)
    # Repo clone cache
    cache_dir: Path = Path("/tmp/infinidev-bench/.cache")
    # Resume from existing predictions (skip already-done instances)
    resume: bool = True
    # Settings.json to mirror into each instance's ``.infinidev/``
    # before running. Defaults to the user's home settings (the same
    # one ``infinidev`` reads when launched from $HOME). Empty string
    # disables the copy and lets infinidev fall back to schema
    # defaults — almost certainly wrong unless you know why.
    settings_source: Path = Path.home() / ".infinidev" / "settings.json"

    def __post_init__(self):
        self.workdir = Path(self.workdir)
        self.output = Path(self.output)
        self.cache_dir = Path(self.cache_dir)
