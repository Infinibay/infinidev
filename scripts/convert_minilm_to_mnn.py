"""Convert ChromaDB's cached ONNX MiniLM-L6-v2 to MNN format.

One-shot script: produces the `.mnn` file needed by the MNN embedder. Run
it once per machine after installing infinidev + MNN. The output path is
printed at the end; set INFINIDEV_MNN_MODEL_PATH to that path to activate
the accelerated embedder.

Usage:
    uv run python scripts/convert_minilm_to_mnn.py
    # then: export INFINIDEV_MNN_MODEL_PATH=~/.infinidev/models/minilm.mnn
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ensure_onnx_cached() -> Path:
    """Return the local path to ChromaDB's all-MiniLM-L6-v2 ONNX model.

    ChromaDB lazily downloads the model on first DefaultEmbeddingFunction
    call. We trigger that path if the cache is missing.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    cached = Path.home() / ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/model.onnx"
    if not cached.is_file():
        print("ONNX model not in cache; triggering ChromaDB download ...")
        DefaultEmbeddingFunction()(["warmup"])
    if not cached.is_file():
        raise FileNotFoundError(f"ChromaDB did not produce {cached}")
    return cached


def _patch_mnn_execstack() -> None:
    """Clear PT_GNU_STACK executable bit on MNN .so files (hardened kernels).

    Mirrors the runtime patch in tools.base.mnn_embedder so mnnconvert
    can import cleanly on CachyOS / Arch-hardened / Ubuntu-hardened.
    """
    from infinidev.tools.base.mnn_embedder import _patch_mnn_execstack as _p
    if _p():
        print("Patched MNN .so files for hardened-kernel compatibility")


def main() -> None:
    out_dir = Path.home() / ".infinidev" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mnn = out_dir / "minilm.mnn"

    onnx = _ensure_onnx_cached()
    print(f"Source ONNX: {onnx}")
    print(f"Target MNN:  {out_mnn}")

    _patch_mnn_execstack()

    cmd = [
        sys.executable, "-m", "MNN.tools.mnnconvert",
        "-f", "ONNX",
        "--modelFile", str(onnx),
        "--MNNModel", str(out_mnn),
        "--bizCode", "minilm",
        "--fp16",
        "--transformerFuse",
    ]
    # mnnconvert's entry module differs by wheel; try the console script
    # first which is known to work.
    console = shutil.which("mnnconvert")
    if console:
        cmd = [
            console, "-f", "ONNX",
            "--modelFile", str(onnx),
            "--MNNModel", str(out_mnn),
            "--bizCode", "minilm",
            "--fp16",
            "--transformerFuse",
        ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print("STDERR:", r.stderr, file=sys.stderr)
        sys.exit(r.returncode)
    # mnnconvert is chatty on success; only print the last useful lines.
    for line in r.stdout.splitlines()[-10:]:
        print(line)

    print(f"\nDone. Set this to activate the MNN embedder:\n")
    print(f"    export INFINIDEV_MNN_MODEL_PATH={out_mnn}")


if __name__ == "__main__":
    main()
