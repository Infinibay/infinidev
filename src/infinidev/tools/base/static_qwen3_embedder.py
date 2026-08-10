"""Fast static embedding table runtime used by Ken-compatible artifacts.

The bundled ``ken/static-qwen3-r512-v2`` artifact was developed for Ken and
is vendored here under the repository's MIT license.  At inference time there
is no transformer: the teacher tokenizer maps text to rows in a learned table,
those rows are summed, projected from rank 512 to the teacher's 1024-dimensional
space, and L2-normalized.

This module intentionally owns the runtime rather than invoking the ``ken``
CLI or importing a separately installed Ken package.  Infinidev therefore gets
one lazy, in-process embedder with a stable artifact and no network dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

STATIC_QWEN3_MODEL = "ken/static-qwen3-r512-v2"
STATIC_QWEN3_DIM = 1024
STATIC_QWEN3_FAMILY = "ken/static-qwen3-r512-"
STATIC_OPENAI_FAMILY = "ken/static-openai-te3-large-r512-"
_SUPPORTED_FAMILIES = (STATIC_QWEN3_FAMILY, STATIC_OPENAI_FAMILY)
_BUNDLED_ARTIFACT = (
    Path(__file__).resolve().parent
    / "data"
    / "ken__static-qwen3-r512-v2.npz"
)
_BUNDLED_SPANISH_ADAPTER = (
    Path(__file__).resolve().parent
    / "data"
    / "ken__static-qwen3-r512-v2-es-query-adapter.npz"
)
_REQUIRED_ARRAYS = ("lut", "A", "B", "tokenizer", "meta")


def _artifact_path() -> Path:
    """Return the configured table path, defaulting to the bundled artifact."""
    override = os.environ.get("INFINIDEV_STATIC_EMBEDDING_PATH")
    return Path(override).expanduser() if override else _BUNDLED_ARTIFACT


class StaticQwen3Embedder:
    """Embedding-function-compatible wrapper around a supported static table."""

    model_name = STATIC_QWEN3_MODEL

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        spanish_adapter_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else _artifact_path()
        adapter_override = os.environ.get("INFINIDEV_STATIC_SPANISH_ADAPTER_PATH")
        if spanish_adapter_path is not None:
            self._spanish_adapter_path: Path | None = Path(spanish_adapter_path)
        elif adapter_override:
            self._spanish_adapter_path = Path(adapter_override).expanduser()
        elif (
            self._path.resolve() == _BUNDLED_ARTIFACT.resolve()
            and _BUNDLED_SPANISH_ADAPTER.is_file()
        ):
            self._spanish_adapter_path = _BUNDLED_SPANISH_ADAPTER
        else:
            self._spanish_adapter_path = None
        self._lock = threading.Lock()
        self._lut: np.ndarray | None = None
        self._table: np.ndarray | None = None
        self._projection: np.ndarray | None = None
        self._tokenizer: Any = None
        self._dim = 0
        self._space_id = ""
        self._base_sha256 = ""
        self._adapter_row_index: np.ndarray | None = None
        self._adapter_delta: np.ndarray | None = None
        self._language_log_odds: np.ndarray | None = None
        self._language_threshold = float("inf")
        self.meta: dict[str, Any] = {}
        self.spanish_adapter_meta: dict[str, Any] = {}

    def _load(self) -> None:
        if self._lut is not None:
            return
        with self._lock:
            if self._lut is not None:
                return
            if not self._path.is_file():
                raise FileNotFoundError(
                    f"Static embedding table not found: {self._path}"
                )

            from tokenizers import Tokenizer

            with np.load(self._path, allow_pickle=False) as artifact:
                missing = [name for name in _REQUIRED_ARRAYS if name not in artifact.files]
                if missing:
                    raise ValueError(
                        f"Invalid static embedding table {self._path}: missing {missing}"
                    )
                table = artifact["A"]
                if table.dtype == np.int8:
                    if "A_scale" not in artifact.files:
                        raise ValueError("Quantized static table is missing A_scale")
                    table = (
                        table.astype(np.float32)
                        * artifact["A_scale"].astype(np.float32)[:, None]
                    )
                projection = np.ascontiguousarray(artifact["B"], dtype=np.float32)
                metadata = json.loads(bytes(artifact["meta"]).decode("utf-8"))
                tokenizer = Tokenizer.from_str(
                    bytes(artifact["tokenizer"]).decode("utf-8")
                )
                lut = np.asarray(artifact["lut"], dtype=np.intp)

            dim = int(projection.shape[1])
            artifact_name = metadata.get("name")
            if (
                not isinstance(artifact_name, str)
                or not artifact_name.startswith(_SUPPORTED_FAMILIES)
                or dim != STATIC_QWEN3_DIM
            ):
                raise ValueError(
                    "Static embedding artifact identity does not match the "
                    f"a supported static family ({STATIC_QWEN3_DIM} dimensions)"
                )
            if table.ndim != 2 or projection.ndim != 2:
                raise ValueError("Static embedding table arrays must be matrices")
            if table.shape[1] != projection.shape[0]:
                raise ValueError("Static embedding rank does not match its projection")
            if lut.ndim != 1 or lut.size == 0 or int(lut.max()) >= table.shape[0]:
                raise ValueError("Static embedding lookup table contains invalid rows")

            self._table = np.ascontiguousarray(table, dtype=np.float32)
            self._projection = projection
            self._lut = lut
            self._tokenizer = tokenizer
            self._dim = dim
            self.model_name = artifact_name
            self.meta = metadata
            digest = hashlib.sha256()
            with self._path.open("rb") as artifact_file:
                for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._base_sha256 = digest.hexdigest()
            self._space_id = f"{self.model_name}:{self._dim}:{digest.hexdigest()[:16]}"
            self._load_spanish_adapter()
            logger.info(
                "Loaded %s (%d rows, rank %d, dim %d) from %s",
                self.model_name,
                self._table.shape[0],
                self._table.shape[1],
                self._dim,
                self._path,
            )

    def _load_spanish_adapter(self) -> None:
        """Load an optional query-only residual tied to this exact base table."""
        if self._spanish_adapter_path is None:
            return
        if not self._spanish_adapter_path.is_file():
            raise FileNotFoundError(
                f"Static Spanish query adapter not found: {self._spanish_adapter_path}"
            )
        assert self._table is not None
        assert self._lut is not None
        with np.load(self._spanish_adapter_path, allow_pickle=False) as adapter:
            required = {
                "rows",
                "delta",
                "delta_scale",
                "language_log_odds",
                "language_threshold",
                "meta",
            }
            missing = sorted(required - set(adapter.files))
            if missing:
                raise ValueError(f"Spanish query adapter is missing {missing}")
            rows = np.asarray(adapter["rows"], dtype=np.int64)
            delta = np.asarray(adapter["delta"])
            scales = np.asarray(adapter["delta_scale"], dtype=np.float32)
            log_odds = np.asarray(adapter["language_log_odds"], dtype=np.float32)
            threshold = float(np.asarray(adapter["language_threshold"]).item())
            metadata = json.loads(bytes(adapter["meta"]).decode("utf-8"))

        if metadata.get("parent_sha256") != self._base_sha256:
            raise ValueError("Spanish query adapter does not match the base artifact")
        if (
            rows.ndim != 1
            or delta.shape != (len(rows), self._table.shape[1])
            or scales.shape != (len(rows),)
            or np.any(rows < 0)
            or np.any(rows >= self._table.shape[0])
            or len(np.unique(rows)) != len(rows)
        ):
            raise ValueError("Spanish query adapter residual arrays are invalid")
        if delta.dtype != np.int8 or log_odds.shape != (len(self._lut),):
            raise ValueError("Spanish query adapter detector arrays are invalid")

        row_index = np.full(self._table.shape[0], -1, dtype=np.int32)
        row_index[rows] = np.arange(len(rows), dtype=np.int32)
        self._adapter_row_index = row_index
        self._adapter_delta = np.ascontiguousarray(
            delta.astype(np.float32) * scales[:, None]
        )
        self._language_log_odds = log_odds
        self._language_threshold = threshold
        self.spanish_adapter_meta = metadata
        logger.info(
            "Loaded Spanish query adapter (%d residual rows) from %s",
            len(rows),
            self._spanish_adapter_path,
        )

    @property
    def dim(self) -> int:
        """Output dimension of the loaded embedding table."""
        self._load()
        return self._dim

    @property
    def space_id(self) -> str:
        """Exact vector-space identity, including a digest of the artifact."""
        self._load()
        return self._space_id

    def _encode(
        self, texts: list[str], *, adapt_spanish_queries: bool = False
    ) -> list[np.ndarray]:
        self._load()
        assert self._tokenizer is not None
        assert self._lut is not None
        assert self._table is not None
        assert self._projection is not None

        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        lengths = np.fromiter(
            (len(encoding.ids) for encoding in encodings),
            dtype=np.int64,
            count=len(encodings),
        )
        flat_ids = np.fromiter(
            (token_id for encoding in encodings for token_id in encoding.ids),
            dtype=np.int64,
            count=int(lengths.sum()),
        )
        output = np.zeros((len(texts), self._dim), dtype=np.float32)
        if flat_ids.size:
            rows = self._lut[flat_ids]
            gathered = self._table[rows]
            starts = np.cumsum(lengths) - lengths
            nonempty = lengths > 0
            if bool(np.all(nonempty)):
                pooled = np.add.reduceat(gathered, starts, axis=0)
            else:
                pooled = np.zeros(
                    (len(texts), self._table.shape[1]), dtype=np.float32
                )
                pooled[nonempty] = np.add.reduceat(
                    gathered, starts[nonempty], axis=0
                )
            if adapt_spanish_queries and self._adapter_delta is not None:
                assert self._adapter_row_index is not None
                assert self._language_log_odds is not None
                for index, encoding in enumerate(encodings):
                    if not encoding.ids:
                        continue
                    token_ids = np.asarray(encoding.ids, dtype=np.int64)
                    unique_ids = np.unique(token_ids)
                    language_score = float(
                        self._language_log_odds[unique_ids].sum()
                        / np.sqrt(len(unique_ids))
                    )
                    if language_score < self._language_threshold:
                        continue
                    adapter_rows = self._adapter_row_index[self._lut[token_ids]]
                    adapter_rows = adapter_rows[adapter_rows >= 0]
                    if len(adapter_rows):
                        pooled[index] += self._adapter_delta[adapter_rows].sum(axis=0)
            output = pooled @ self._projection
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        return list(output / np.maximum(norms, 1e-12))

    def __call__(self, texts: list[str]) -> list[np.ndarray]:
        """Embed documents using the Chroma-compatible callable contract."""
        return self._encode(texts) if texts else []

    def embed_passages(self, texts: list[str]) -> list[np.ndarray]:
        """Embed stored passages; this symmetric table uses the same head."""
        return self._encode(texts) if texts else []

    def embed_queries(self, texts: list[str]) -> list[np.ndarray]:
        """Embed queries, applying the calibrated residual only to Spanish."""
        return self._encode(texts, adapt_spanish_queries=True) if texts else []

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query, applying the calibrated residual only to Spanish."""
        return self._encode([text], adapt_spanish_queries=True)[0]


_singleton_lock = threading.Lock()
_singleton: StaticQwen3Embedder | None = None


def get_static_qwen3_embedder() -> StaticQwen3Embedder | None:
    """Return the lazy singleton, or ``None`` when the artifact is unavailable."""
    global _singleton
    if _singleton is not None:
        return _singleton
    path = _artifact_path()
    if not path.is_file():
        return None
    with _singleton_lock:
        if _singleton is None:
            _singleton = StaticQwen3Embedder(path)
    return _singleton


__all__ = [
    "STATIC_QWEN3_DIM",
    "STATIC_QWEN3_FAMILY",
    "STATIC_QWEN3_MODEL",
    "STATIC_OPENAI_FAMILY",
    "StaticQwen3Embedder",
    "get_static_qwen3_embedder",
]
