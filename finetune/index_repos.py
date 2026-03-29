"""Index downloaded repos using infinidev's code_intel system."""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from finetune.config import REPOS, REPOS_DIR


def index_repo(name: str, repo_path: str, project_id: int) -> dict:
    """Index a single repo and return stats."""
    from infinidev.code_intel.indexer import index_directory

    print(f"  [index] {name}...")
    start = time.time()
    stats = index_directory(project_id, repo_path)
    elapsed = time.time() - start
    print(
        f"    {stats.get('files_indexed', 0)} files, "
        f"{stats.get('symbols_total', 0)} symbols, "
        f"{elapsed:.1f}s"
    )
    return stats


def index_all():
    """Index all downloaded repos. Each repo gets a unique project_id."""
    from infinidev.config.settings import settings
    from infinidev.db.service import init_db

    # Use a dedicated DB for fine-tuning
    ft_db = str(REPOS_DIR.parent / "output" / "finetune.db")
    os.makedirs(os.path.dirname(ft_db), exist_ok=True)
    settings.DB_PATH = ft_db
    init_db()

    print(f"Indexing repos (DB: {ft_db})")

    for i, repo in enumerate(REPOS):
        name = repo["name"]
        repo_path = REPOS_DIR / name
        if not repo_path.exists():
            print(f"  [skip] {name} not downloaded")
            continue

        project_id = i + 1  # 1-based
        try:
            index_repo(name, str(repo_path), project_id)
        except Exception as e:
            print(f"  [error] {name}: {e}")

    print("Done.")


if __name__ == "__main__":
    index_all()
