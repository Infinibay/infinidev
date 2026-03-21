"""Harness configuration for SWE-bench runs."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchConfig:
    # Model (LiteLLM format)
    model: str = "ollama_chat/qwen3.5:27b"
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

    def __post_init__(self):
        self.workdir = Path(self.workdir)
        self.output = Path(self.output)
        self.cache_dir = Path(self.cache_dir)
