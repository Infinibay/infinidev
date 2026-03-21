"""Repository setup for SWE-bench instances.

Handles cloning, caching, and checking out the correct base commit.
"""

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a subprocess, raise on failure."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True)


def clone_or_cache(repo: str, cache_dir: Path) -> Path:
    """Clone a repo into cache_dir if not already cached. Returns cached repo path.

    Args:
        repo: GitHub repo in "owner/name" format (e.g. "django/django")
        cache_dir: Directory to store bare clones
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = repo.replace("/", "__")
    cached = cache_dir / safe_name

    if cached.exists():
        log.info("Using cached repo: %s", cached)
        # Fetch latest
        try:
            _run(["git", "fetch", "--all"], cwd=cached)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            log.warning("Failed to fetch updates for %s, using stale cache", repo)
        return cached

    log.info("Cloning %s into cache...", repo)
    url = f"https://github.com/{repo}.git"
    _run(["git", "clone", "--bare", url, str(cached)], timeout=600)
    return cached


def setup_instance(
    repo: str,
    base_commit: str,
    workdir: Path,
    cache_dir: Path,
    instance_id: str,
) -> Path:
    """Prepare a working copy for a SWE-bench instance.

    1. Clone from cache (local clone, fast)
    2. Checkout the base_commit
    3. Return the working directory path

    Args:
        repo: "owner/name" format
        base_commit: The commit SHA to checkout
        workdir: Parent directory for instance workdirs
        cache_dir: Cache directory for bare clones
        instance_id: SWE-bench instance ID (used as dir name)

    Returns:
        Path to the instance working directory
    """
    cached = clone_or_cache(repo, cache_dir)

    instance_dir = workdir / instance_id.replace("/", "__")
    if instance_dir.exists():
        shutil.rmtree(instance_dir)

    instance_dir.mkdir(parents=True, exist_ok=True)

    # Local clone from bare cache (fast, uses hardlinks)
    _run(["git", "clone", "--no-checkout", str(cached), str(instance_dir)], timeout=120)
    _run(["git", "checkout", base_commit], cwd=instance_dir)

    # Clean state
    _run(["git", "checkout", "-b", "infinidev-bench"], cwd=instance_dir)

    log.info("Instance %s ready at %s", instance_id, instance_dir)
    return instance_dir


def get_patch(instance_dir: Path) -> str:
    """Get the git diff (patch) from the instance directory."""
    result = _run(["git", "diff", "HEAD"], cwd=instance_dir)
    return result.stdout


def cleanup_instance(instance_dir: Path) -> None:
    """Remove instance working directory."""
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
