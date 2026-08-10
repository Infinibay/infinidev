"""Build verified software-issue-to-patch pairs from the pinned SWE-Gym set."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DATASET = "SWE-Gym/SWE-Gym"
REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
SOURCE_FILE = "data/train-00000-of-00001.parquet"
REPOSITORY_LICENSES = {
    "Project-MONAI/MONAI": "Apache-2.0",
    "bokeh/bokeh": "BSD-3-Clause",
    "conan-io/conan": "MIT",
    "dask/dask": "BSD-3-Clause",
    "facebookresearch/hydra": "MIT",
    "getmoto/moto": "Apache-2.0",
    "iterative/dvc": "Apache-2.0",
    "modin-project/modin": "Apache-2.0",
    "pandas-dev/pandas": "BSD-3-Clause",
    "pydantic/pydantic": "MIT",
    "python/mypy": "MIT AND Python-2.0",
}


def _split(repository: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(repository.encode()).digest()[:4], "big") % 10
    if bucket < 8:
        return "train"
    return "validation" if bucket == 8 else "test"


def record_from_row(
    row: dict[str, Any], *, max_problem_chars: int, max_patch_chars: int
) -> dict[str, Any] | None:
    repository = str(row.get("repo") or "")
    license_name = REPOSITORY_LICENSES.get(repository)
    if license_name is None:
        return None
    problem = str(row.get("problem_statement") or "").strip()[:max_problem_chars]
    patch = str(row.get("patch") or "").strip()[:max_patch_chars]
    instance_id = str(row.get("instance_id") or "")
    base_commit = str(row.get("base_commit") or "")
    if len(problem) < 30 or "@@" not in patch or not instance_id or not base_commit:
        return None
    identity = hashlib.sha256(
        f"{instance_id}\0{base_commit}\0{problem}\0{patch}".encode()
    ).hexdigest()[:24]
    return {
        "id": identity,
        "source": "swe_gym_verified",
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "source_url": f"https://github.com/{repository}/tree/{base_commit}",
        "source_instance": instance_id,
        "repository": repository,
        "revision": base_commit,
        "licenses": [license_name],
        "kind": "issue_to_patch",
        "language": "en",
        "programming_language": "python",
        "text": " ".join(problem.split()),
        "parallel_language": "python",
        "parallel_text": patch,
        "split": _split(repository),
    }


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=512):
        yield from batch.to_pylist()


def _download(cache_dir: Path | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=DATASET,
        filename=SOURCE_FILE,
        repo_type="dataset",
        revision=REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    ))


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = _download(args.cache_dir)
    records = [
        record
        for row in _rows(source)
        if (record := record_from_row(
            row,
            max_problem_chars=args.max_problem_chars,
            max_patch_chars=args.max_patch_chars,
        )) is not None
    ]
    records.sort(key=lambda record: (str(record["repository"]), str(record["id"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "output": str(args.output),
        "source_bytes": source.stat().st_size,
        "output_bytes": args.output.stat().st_size,
        "records": len(records),
        "records_by_repository": dict(sorted(Counter(
            str(record["repository"]) for record in records
        ).items())),
        "records_by_split": dict(sorted(Counter(
            str(record["split"]) for record in records
        ).items())),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-problem-chars", type=int, default=3_000)
    parser.add_argument("--max-patch-chars", type=int, default=6_000)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
