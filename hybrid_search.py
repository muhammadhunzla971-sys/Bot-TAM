"""Optional local multilingual ONNX embeddings for hybrid corpus retrieval."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "models" / "multilingual-e5-small-quantized.onnx"
TOKENIZER_PATH = ROOT / "data" / "models" / "multilingual-e5-small-tokenizer.json"
MODEL_REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
MODEL_URL = (
    "https://huggingface.co/Xenova/multilingual-e5-small/resolve/"
    f"{MODEL_REVISION}/onnx/model_quantized.onnx"
)
TOKENIZER_URL = (
    "https://huggingface.co/Xenova/multilingual-e5-small/resolve/"
    f"{MODEL_REVISION}/tokenizer.json"
)
MODEL_SHA256 = "f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193"
TOKENIZER_SHA256 = "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
MODEL_MAX_BYTES = 125 * 1024 * 1024
TOKENIZER_MAX_BYTES = 20 * 1024 * 1024
MAX_SEQUENCE_LENGTH = 256
_RUNTIME_LOCK = RLock()
_RUNTIME = None


class HybridModelUnavailable(RuntimeError):
    """Raised when the optional local model cannot be used."""


def _ensure_asset(
    path: Path, url: str, expected_hash: str, max_bytes: int
) -> Path:
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == expected_hash:
            return path
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    request = Request(url, headers={"User-Agent": "PunjabTourismAssistant/1.0"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(request, timeout=60) as response:
            with temporary_path.open("wb") as asset_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise HybridModelUnavailable(
                            "A multilingual retrieval asset exceeded its size limit."
                        )
                    digest.update(chunk)
                    asset_file.write(chunk)
        if digest.hexdigest() != expected_hash:
            raise HybridModelUnavailable(
                "A downloaded multilingual retrieval asset failed its SHA-256 check."
            )
        temporary_path.replace(path)
        return path
    except (OSError, TimeoutError, URLError, HybridModelUnavailable):
        temporary_path.unlink(missing_ok=True)
        raise


def _load_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME
        try:
            import numpy as np
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as error:
            raise HybridModelUnavailable(
                "Install the project's ONNX Runtime, NumPy, and tokenizer dependencies."
            ) from error

        model_path = _ensure_asset(
            MODEL_PATH, MODEL_URL, MODEL_SHA256, MODEL_MAX_BYTES
        )
        tokenizer_path = _ensure_asset(
            TOKENIZER_PATH, TOKENIZER_URL, TOKENIZER_SHA256, TOKENIZER_MAX_BYTES
        )
        try:
            session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            input_names = {item.name for item in session.get_inputs()}
            if not {"input_ids", "attention_mask"}.issubset(input_names):
                raise HybridModelUnavailable(
                    "The multilingual retrieval model has unexpected inputs."
                )
            if input_names - {"input_ids", "attention_mask", "token_type_ids"}:
                raise HybridModelUnavailable(
                    "The multilingual retrieval model has unsupported inputs."
                )
            outputs = session.get_outputs()
            if not outputs:
                raise HybridModelUnavailable(
                    "The multilingual retrieval model has no outputs."
                )
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            pad_token = "<pad>"
            pad_id = tokenizer.token_to_id(pad_token)
            if pad_id is None:
                pad_token = "[PAD]"
                pad_id = tokenizer.token_to_id(pad_token)
            if pad_id is None:
                pad_id = 0
            tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
            tokenizer.enable_padding(
                length=MAX_SEQUENCE_LENGTH,
                pad_id=pad_id,
                pad_token=pad_token,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise HybridModelUnavailable(
                "The multilingual retrieval model could not be loaded."
            ) from error
        _RUNTIME = (np, session, tokenizer, input_names)
        return _RUNTIME


def embed_texts(texts: list[str], role: str) -> list[list[float]]:
    """Encode query or passage texts locally, returning normalized vectors."""
    if role not in {"query", "passage"}:
        raise ValueError("Embedding role must be query or passage.")
    if not texts:
        return []
    try:
        np, session, tokenizer, input_names = _load_runtime()
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), 8):
            batch = texts[offset : offset + 8]
            encodings = tokenizer.encode_batch(
                [f"{role}: {text[:12_000]}" for text in batch]
            )
            input_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
            attention_mask = np.asarray(
                [item.attention_mask for item in encodings], dtype=np.int64
            )
            feeds = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)
            output = np.asarray(session.run(None, feeds)[0], dtype=np.float32)
            if output.ndim == 3:
                mask = attention_mask[:, :, None].astype(np.float32)
                output = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            if output.ndim != 2 or output.shape[0] != len(batch):
                raise HybridModelUnavailable(
                    "The multilingual retrieval model returned an invalid embedding."
                )
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            if not np.isfinite(output).all() or (norms == 0).any():
                raise HybridModelUnavailable(
                    "The multilingual retrieval model returned an invalid embedding."
                )
            vectors.extend((output / norms).tolist())
        return vectors
    except HybridModelUnavailable:
        raise
    except (OSError, RuntimeError, ValueError, URLError) as error:
        raise HybridModelUnavailable(
            "The multilingual retrieval model is unavailable."
        ) from error
