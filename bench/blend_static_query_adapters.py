"""Blend two query-only static adapters tied to the same base artifact.

Residual interpolation is performed in float32 and requantized once.  The
selector can be inherited from either parent, allowing semantic adaptation and
language routing to be calibrated independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from bench.fit_static_qwen3_spanish import _quantize_rows
except ModuleNotFoundError:
    from fit_static_qwen3_spanish import _quantize_rows


def _load(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name] for name in artifact.files}
    arrays["meta_json"] = json.loads(bytes(arrays["meta"]).decode("utf-8"))
    arrays["delta_float"] = (
        arrays["delta"].astype(np.float32)
        * arrays["delta_scale"].astype(np.float32)[:, None]
    )
    return arrays


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blend(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    second_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted row ids and interpolated float residuals."""
    if not 0.0 <= second_weight <= 1.0:
        raise ValueError("second_weight must be between zero and one")
    first_parent = first["meta_json"].get("parent_sha256")
    second_parent = second["meta_json"].get("parent_sha256")
    if first_parent != second_parent:
        raise ValueError("adapters do not share the same exact parent artifact")
    first_by_row = {
        int(row): vector
        for row, vector in zip(first["rows"], first["delta_float"], strict=True)
    }
    second_by_row = {
        int(row): vector
        for row, vector in zip(second["rows"], second["delta_float"], strict=True)
    }
    rows = np.asarray(sorted(first_by_row.keys() | second_by_row.keys()), dtype=np.int32)
    dimension = int(first["delta_float"].shape[1])
    result = np.zeros((len(rows), dimension), dtype=np.float32)
    for index, row in enumerate(rows):
        if int(row) in first_by_row:
            result[index] += (1.0 - second_weight) * first_by_row[int(row)]
        if int(row) in second_by_row:
            result[index] += second_weight * second_by_row[int(row)]
    keep = np.linalg.norm(result, axis=1) > 1e-8
    return rows[keep], result[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--second-weight", type=float, required=True)
    parser.add_argument(
        "--selector-from", choices=("first", "second"), default="first"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = _load(args.first)
    second = _load(args.second)
    rows, residual = blend(first, second, second_weight=args.second_weight)
    quantized, scales = _quantize_rows(residual)
    selector = first if args.selector_from == "first" else second
    meta = dict(selector["meta_json"])
    meta["name"] = "ken/static-qwen3-r512-v2-es-query-adapter-blended-experimental"
    meta["blend"] = {
        "first_sha256": _sha256(args.first),
        "second_sha256": _sha256(args.second),
        "second_weight": args.second_weight,
        "selector_from": args.selector_from,
    }
    meta["residual_rows"] = int(len(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rows=rows,
        delta=quantized,
        delta_scale=scales,
        language_log_odds=selector["language_log_odds"],
        language_threshold=selector["language_threshold"],
        meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8),
    )
    print(json.dumps({
        "output": str(args.output),
        "residual_rows": len(rows),
        "second_weight": args.second_weight,
        "selector_from": args.selector_from,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
