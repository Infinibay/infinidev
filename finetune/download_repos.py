"""Download and prepare open source repos for dataset generation."""

import subprocess
import sys
from pathlib import Path

from finetune.config import REPOS, REPOS_DIR


def download_repo(repo: dict) -> Path:
    """Shallow-clone a repo. Returns the local path."""
    name = repo["name"]
    url = repo["url"]
    branch = repo.get("branch", "main")
    dest = REPOS_DIR / name

    if dest.exists():
        print(f"  [skip] {name} already exists")
        return dest

    print(f"  [clone] {name} from {url} ({branch})")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)],
        check=True,
        capture_output=True,
    )
    return dest


def download_all():
    """Download all configured repos."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(REPOS)} repos to {REPOS_DIR}")

    for repo in REPOS:
        try:
            download_repo(repo)
        except subprocess.CalledProcessError as e:
            print(f"  [error] {repo['name']}: {e.stderr.decode()[:200]}")
        except Exception as e:
            print(f"  [error] {repo['name']}: {e}")

    print("Done.")


if __name__ == "__main__":
    download_all()
