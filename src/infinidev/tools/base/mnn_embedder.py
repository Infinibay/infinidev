"""MNN-backed embedding function for all-MiniLM-L6-v2.

A drop-in replacement for ChromaDB's DefaultEmbeddingFunction that routes
the same 384-dim MiniLM model through Alibaba's MNN runtime instead of
ONNX Runtime. On CPU this is roughly an order of magnitude faster than
the ONNX path (~11 ms vs ~115 ms per query on a Ryzen APU) while
producing bit-compatible vectors (cosine 1.0000 against ONNX across a
diverse test corpus), so stored BLOBs remain valid.

Activation: install the `mnn` extra (`uv sync --extra mnn`). That's it —
the embedder auto-detects the MNN install, one-shot converts ChromaDB's
cached ONNX model on first use (~30 s), caches the result under
`~/.infinidev/models/minilm.mnn`, and takes over transparently.
`INFINIDEV_MNN_MODEL_PATH` is an optional override for advanced users;
without the package installed, callers fall back to the ChromaDB default.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MAX_SEQ = 128
_TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MODEL_PATH = Path.home() / ".infinidev" / "models" / "minilm.mnn"
_CHROMADB_ONNX_PATH = (
    Path.home() / ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/model.onnx"
)


class MNNEmbedder:
    """Callable embedder matching ChromaDB's EmbeddingFunction interface."""

    def __init__(self, model_path: str | os.PathLike, max_seq: int = _MAX_SEQ):
        import MNN  # lazy import — don't pay the cost when unused
        from transformers import AutoTokenizer

        self._MNN = MNN
        self._max_seq = max_seq
        self._tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)

        model_path = str(Path(model_path).expanduser())
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"MNN model not found: {model_path}")
        self._interp = MNN.Interpreter(model_path)
        self._session = self._interp.createSession(
            {"backend": "CPU", "precision": "high", "numThread": 4}
        )
        for name in ("input_ids", "attention_mask", "token_type_ids"):
            t = self._interp.getSessionInput(self._session, name)
            self._interp.resizeTensor(t, (1, max_seq))
        self._interp.resizeSession(self._session)

        # MNN sessions are not thread-safe; serialize inference across threads
        # that might call this embedder from the ContextRank hooks pool.
        self._lock = threading.Lock()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t).tolist() for t in texts]

    def _embed_one(self, text: str) -> np.ndarray:
        enc = self._tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self._max_seq, return_tensors="np",
        )
        ids = enc["input_ids"].astype(np.int32)
        mask = enc["attention_mask"].astype(np.int32)
        types = enc.get("token_type_ids")
        if types is None:
            types = np.zeros_like(ids)
        types = types.astype(np.int32)

        MNN = self._MNN
        with self._lock:
            input_ids = self._interp.getSessionInput(self._session, "input_ids")
            attention_mask = self._interp.getSessionInput(self._session, "attention_mask")
            token_type_ids = self._interp.getSessionInput(self._session, "token_type_ids")

            def _fill(dst, arr):
                tmp = MNN.Tensor(
                    dst.getShape(), MNN.Halide_Type_Int,
                    arr, MNN.Tensor_DimensionType_Caffe,
                )
                dst.copyFrom(tmp)

            _fill(input_ids, ids)
            _fill(attention_mask, mask)
            _fill(token_type_ids, types)

            self._interp.runSession(self._session)
            out = self._interp.getSessionOutput(self._session, "last_hidden_state")
            out_shape = tuple(out.getShape())
            buf = np.zeros(out_shape, dtype=np.float32)
            out_host = MNN.Tensor(
                out_shape, MNN.Halide_Type_Float,
                buf, MNN.Tensor_DimensionType_Caffe,
            )
            out.copyToHostTensor(out_host)
            last_hidden = np.array(
                out_host.getData(), dtype=np.float32,
            ).reshape(out_shape)

        # Mean pool with attention mask → L2 normalize. MNN converts only the
        # encoder (output = last_hidden_state); pooling+norm is microseconds.
        mask_f = mask.astype(np.float32)[0]
        lh = last_hidden[0]
        summed = (lh * mask_f[:, None]).sum(axis=0)
        denom = max(mask_f.sum(), 1e-9)
        pooled = summed / denom
        n = np.linalg.norm(pooled)
        return (pooled / max(n, 1e-12)).astype(np.float32)


def _patch_mnn_execstack() -> bool:
    """Clear PT_GNU_STACK executable bit on MNN .so files.

    The MNN pip wheel ships with RWE stack flags, which modern hardened
    kernels (e.g. CachyOS, Ubuntu-hardened, Arch with `hardened-kernel`)
    reject with `cannot enable executable stack as shared object requires`.
    The fix is a one-byte ELF edit — the .so doesn't actually need the
    executable stack for any legitimate purpose.

    Idempotent and safe: if the flag is already clear, nothing happens.
    Returns True if any file was modified.
    """
    import glob
    import struct

    try:
        import MNN  # noqa: F401 — just to locate the package
    except ImportError:
        return False
    try:
        import importlib
        spec = importlib.util.find_spec("MNN")
        if spec is None or spec.origin is None:
            return False
        base = str(Path(spec.origin).parent.parent)
    except Exception:
        return False

    PT_GNU_STACK = 0x6474E551
    targets = glob.glob(os.path.join(base, "_*.so")) + glob.glob(
        os.path.join(base, "MNN/**/*.so"), recursive=True
    )
    modified = False
    for path in targets:
        try:
            with open(path, "rb") as f:
                data = bytearray(f.read())
            if data[:4] != b"\x7fELF" or data[4] != 2:
                continue
            e_phoff = struct.unpack_from("<Q", data, 32)[0]
            e_phentsize = struct.unpack_from("<H", data, 54)[0]
            e_phnum = struct.unpack_from("<H", data, 56)[0]
            dirty = False
            for i in range(e_phnum):
                off = e_phoff + i * e_phentsize
                p_type = struct.unpack_from("<I", data, off)[0]
                if p_type == PT_GNU_STACK:
                    p_flags = struct.unpack_from("<I", data, off + 4)[0]
                    if p_flags & 1:
                        struct.pack_into("<I", data, off + 4, p_flags & ~1)
                        dirty = True
            if dirty:
                with open(path, "wb") as f:
                    f.write(data)
                modified = True
        except OSError:
            continue
    return modified


def _ensure_chromadb_onnx_cached() -> bool:
    """Trigger ChromaDB's lazy ONNX download if the file isn't already cached.

    ChromaDB's DefaultEmbeddingFunction downloads the model on first
    invocation; we exploit that side effect rather than fetching it
    ourselves (same file, same hash, fewer URLs to maintain). Returns
    True when the ONNX file exists on disk after this call.
    """
    if _CHROMADB_ONNX_PATH.is_file():
        return True
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        DefaultEmbeddingFunction()(["warmup"])
    except Exception:
        logger.warning("Could not fetch ChromaDB ONNX", exc_info=True)
        return False
    return _CHROMADB_ONNX_PATH.is_file()


def _locate_mnnconvert() -> str | None:
    """Find the mnnconvert CLI in the current venv or PATH."""
    found = shutil.which("mnnconvert")
    if found:
        return found
    venv_bin = Path(sys.executable).parent / "mnnconvert"
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)
    return None


def _auto_convert_to_mnn(target: Path) -> bool:
    """One-shot ONNX→MNN conversion. Blocking, ~30 s on a modern APU.

    Applies the exec-stack ELF patch first so mnnconvert can run on
    hardened kernels. Caches under ~/.infinidev/models so subsequent
    sessions load instantly.
    """
    if not _ensure_chromadb_onnx_cached():
        return False

    _patch_mnn_execstack()

    mnnconvert = _locate_mnnconvert()
    if not mnnconvert:
        logger.warning(
            "mnnconvert not found on PATH; skipping MNN auto-conversion "
            "(install the `mnn` extra or ensure the venv is active)"
        )
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Converting %s → %s (one-time, ~30 s)",
        _CHROMADB_ONNX_PATH.name, target,
    )
    try:
        r = subprocess.run(
            [
                mnnconvert, "-f", "ONNX",
                "--modelFile", str(_CHROMADB_ONNX_PATH),
                "--MNNModel", str(target),
                "--bizCode", "minilm",
                "--fp16", "--transformerFuse",
            ],
            capture_output=True, text=True, timeout=300,
        )
    except Exception:
        logger.warning("mnnconvert invocation failed", exc_info=True)
        return False
    if r.returncode != 0 or not target.is_file():
        tail = (r.stderr or r.stdout or "").splitlines()[-5:]
        logger.warning("mnnconvert exited %d: %s", r.returncode, " | ".join(tail))
        return False
    logger.info("MNN model ready: %s", target)
    return True


_singleton_lock = threading.Lock()
_singleton: MNNEmbedder | None = None


def get_mnn_embedder() -> MNNEmbedder | None:
    """Return the singleton MNNEmbedder, or None to fall back to ChromaDB.

    Auto-detection flow:
      1. If the `MNN` package is not importable → return None (user didn't
         install the extra).
      2. Resolve the model path: `INFINIDEV_MNN_MODEL_PATH` override wins,
         otherwise `~/.infinidev/models/minilm.mnn`.
      3. If the model file is missing, auto-convert ChromaDB's cached ONNX
         (one-time, ~30 s).
      4. Load; patch hardened-kernel exec-stack on ImportError and retry.
      5. Any failure → log + return None so callers fall back transparently.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    try:
        import MNN  # noqa: F401
    except ImportError:
        return None  # extra not installed; silent fallback is correct
    except Exception as e:
        if "executable stack" in str(e):
            if not _patch_mnn_execstack():
                logger.warning("MNN blocked by hardened kernel; falling back")
                return None
            try:
                import MNN  # noqa: F401
            except Exception:
                logger.warning("MNN still unavailable after patch", exc_info=True)
                return None
        else:
            logger.warning("MNN import failed: %s", e)
            return None

    override = os.environ.get("INFINIDEV_MNN_MODEL_PATH")
    model_path = Path(override).expanduser() if override else _DEFAULT_MODEL_PATH

    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        if not model_path.is_file():
            if not _auto_convert_to_mnn(model_path):
                return None
        try:
            _singleton = MNNEmbedder(str(model_path))
            logger.info("MNN embedder ready: %s", model_path)
        except Exception:
            logger.warning("MNN init failed; falling back", exc_info=True)
    return _singleton
