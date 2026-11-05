"""Small, local perceptual-hash helpers for tourism-image matching."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from io import BytesIO
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import urllib.request

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "models" / "clip-vit-base-patch32-vision-quantized.onnx"
MODEL_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
    "6b7c523703ae26e39be4b03774966f8571c7f523/onnx/vision_model_quantized.onnx"
)
MODEL_SHA256 = "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299"
MODEL_MAX_BYTES = 100 * 1024 * 1024
TEXT_MODEL_PATH = ROOT / "data" / "models" / "clip-vit-base-patch32-text-quantized.onnx"
TOKENIZER_PATH = ROOT / "data" / "models" / "clip-vit-base-patch32-tokenizer.json"
TEXT_MODEL_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
    "6b7c523703ae26e39be4b03774966f8571c7f523/onnx/text_model_quantized.onnx"
)
TOKENIZER_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
    "6b7c523703ae26e39be4b03774966f8571c7f523/tokenizer.json"
)
TEXT_MODEL_SHA256 = "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a"
TOKENIZER_SHA256 = "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MIN_VISUAL_SIMILARITY = 0.78
TEXT_RERANK_WEIGHT = 0.5
PREPROCESS_SIZE = 224

KNOWN_IMAGE_PLACE_NAMES = {
    "lahore-1": "Minar-e-Pakistan, Lahore",
}


def _read_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is empty or larger than 10 MiB.")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP", "GIF"}:
                raise ValueError("Use a JPEG, PNG, WebP, or GIF image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large.")
            image.seek(0)
            oriented = ImageOps.exif_transpose(image)
            if "transparency" in oriented.info or "A" in oriented.getbands():
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                return background.convert("RGB")
            return oriented.convert("RGB")
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a valid supported image.") from error


def _model_image_tensor(image: Image.Image) -> np.ndarray:
    width, height = image.size
    scale = PREPROCESS_SIZE / min(width, height)
    resized = image.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - PREPROCESS_SIZE) // 2
    top = (resized.height - PREPROCESS_SIZE) // 2
    pixels = np.asarray(
        resized.crop(
            (left, top, left + PREPROCESS_SIZE, top + PREPROCESS_SIZE)
        ),
        dtype=np.float32,
    ) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    return np.expand_dims(((pixels - mean) / std).transpose(2, 0, 1), axis=0)


def difference_hash(image_bytes: bytes) -> str:
    """Return a 64-bit difference hash as a legacy/index diagnostic."""
    image = _read_image(image_bytes).convert("L").resize(
        (9, 8), Image.Resampling.LANCZOS
    )
    pixels = list(image.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return f"{value:016x}"


def _ensure_asset(path: Path, url: str, expected_hash: str) -> Path:
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == expected_hash:
            return path
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PunjabTourismAssistant/1.0"},
    )
    try:
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary_path.open("wb") as model_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MODEL_MAX_BYTES:
                        raise RuntimeError("A CLIP model asset exceeded 100 MiB.")
                    digest.update(chunk)
                    model_file.write(chunk)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError("A downloaded CLIP asset failed its SHA-256 check.")
        temporary_path.replace(path)
        return path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def ensure_image_model() -> Path:
    """Download the pinned quantized CLIP image encoder if it is not present."""
    return _ensure_asset(MODEL_PATH, MODEL_URL, MODEL_SHA256)


@lru_cache(maxsize=1)
def _load_model():
    try:
        import onnxruntime
    except ImportError as error:
        raise RuntimeError(
            "Image matching requires dependencies from requirements.txt."
        ) from error
    model_path = ensure_image_model()
    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    if (
        len(session.get_inputs()) != 1
        or session.get_inputs()[0].name != "pixel_values"
        or len(session.get_outputs()) != 1
        or session.get_outputs()[0].name != "image_embeds"
    ):
        raise RuntimeError("The configured image model has an unexpected format.")
    return session


@lru_cache(maxsize=1)
def _load_text_model():
    try:
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "Image label matching requires dependencies from requirements.txt."
        ) from error
    model_path = _ensure_asset(
        TEXT_MODEL_PATH, TEXT_MODEL_URL, TEXT_MODEL_SHA256
    )
    tokenizer_path = _ensure_asset(
        TOKENIZER_PATH, TOKENIZER_URL, TOKENIZER_SHA256
    )
    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    if (
        len(session.get_inputs()) != 1
        or session.get_inputs()[0].name != "input_ids"
        or len(session.get_outputs()) != 1
        or session.get_outputs()[0].name != "text_embeds"
    ):
        raise RuntimeError("The configured CLIP text model has an unexpected format.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(
        length=77, pad_id=49407, pad_token="<|endoftext|>"
    )
    return session, tokenizer


def text_embedding(text: str) -> list[int]:
    """Encode a tourism image label in the shared CLIP image/text space."""
    session, tokenizer = _load_text_model()
    tokens = tokenizer.encode(text[:1_000])
    input_ids = np.asarray([tokens.ids], dtype=np.int64)
    output = session.run(["text_embeds"], {"input_ids": input_ids})[0][0]
    norm = float(np.linalg.norm(output))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("The text model returned an invalid embedding.")
    normalized = output / norm
    return np.rint(np.clip(normalized, -1, 1) * 127).astype(np.int8).astype(int).tolist()


def image_label_text(title: str, alt: str, image_url: str) -> str:
    """Create a short CLIP prompt from an image caption, filename, or page title."""
    if any(character.isalnum() for character in alt):
        return f"a photograph of {alt}"

    filename = Path(unquote(urlsplit(image_url).path)).stem
    words = [word for word in re.findall(r"[A-Za-z]+", filename) if len(word) > 2]
    generic_words = {"image", "photo", "picture", "banner", "thumb", "slider"}
    if words and not all(word.casefold() in generic_words for word in words):
        return "a photograph of " + " ".join(words)
    return f"a photograph of {title}"


def image_place_name(title: str, alt: str, image_url: str) -> str:
    """Return the most specific place name available for an indexed image."""
    filename = Path(unquote(urlsplit(image_url).path)).stem
    known_name = KNOWN_IMAGE_PLACE_NAMES.get(filename.casefold())
    if known_name:
        return known_name

    generic_alts = {
        "cultural activities",
        "eco tourism",
        "family activities",
        "heritage",
        "outdoor adventures",
        "punjab tourism department",
        "shopping experience",
        "spiritual journey",
        "tourism, archaeology and museums department",
    }
    if any(character.isalnum() for character in alt) and alt.casefold() not in generic_alts:
        return alt.strip()

    page_title = re.sub(
        r"\s*\|\s*Tourism, Archaeology and Museums Department.*$",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    if page_title and page_title.casefold() not in {
        "welcome to tourism",
        "welcome to tourism, archaeology and museums department",
        "punjab tourism department",
    }:
        return page_title

    words = re.sub(r"[_+]+", " ", filename)
    words = re.sub(r"(?i)\b(?:banner|image|photo|picture|thumbnail|thumb)\b", " ", words)
    words = re.sub(r"\s+\d+$", "", words)
    words = re.sub(r"\s+", " ", words).strip(" -_")
    if words and not words.casefold().replace(" ", "").isdigit():
        return words.title()
    return page_title or title


def visual_embedding(image_bytes: bytes) -> list[int]:
    """Return a normalized, compact CLIP image embedding for visual search."""
    image = _read_image(image_bytes)
    session = _load_model()
    tensor = _model_image_tensor(image)
    output = session.run(["image_embeds"], {"pixel_values": tensor})[0][0]
    norm = float(np.linalg.norm(output))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("The image model returned an invalid embedding.")
    normalized = output / norm
    return np.rint(np.clip(normalized, -1, 1) * 127).astype(np.int8).astype(int).tolist()


def match_image(image_bytes: bytes, documents: list[dict[str, object]]) -> list[dict[str, object]]:
    """Find the most visually similar indexed tourism website images."""
    uploaded_embedding = np.asarray(visual_embedding(image_bytes), dtype=np.float32)
    text_reranking_available = any(
        isinstance(document.get("text_embedding"), list)
        and len(document["text_embedding"]) == len(uploaded_embedding)
        for document in documents
    )
    ranked_pages: dict[str, tuple[float, dict[str, object]]] = {}

    for document in documents:
        images = document.get("images", [])
        if not isinstance(images, list):
            continue
        page_url = str(document.get("url", ""))
        text_similarity = 0.0
        page_text_embedding = document.get("text_embedding")
        if (
            text_reranking_available
            and isinstance(page_text_embedding, list)
            and len(page_text_embedding) == len(uploaded_embedding)
        ):
            try:
                page_text = np.asarray(page_text_embedding, dtype=np.float32)
                text_norm = float(np.linalg.norm(page_text))
                if math.isfinite(text_norm) and text_norm:
                    text_similarity = float(
                        np.dot(uploaded_embedding, page_text)
                        / (np.linalg.norm(uploaded_embedding) * text_norm)
                    )
            except (TypeError, ValueError):
                text_similarity = 0.0
        for image in images:
            if not isinstance(image, dict):
                continue
            image_url = image.get("url")
            embedding = image.get("embedding")
            if (
                not isinstance(image_url, str)
                or not image_url.startswith("https://tourism.punjab.gov.pk/")
                or not isinstance(embedding, list)
                or len(embedding) != len(uploaded_embedding)
            ):
                continue
            try:
                reference = np.asarray(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(reference))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(norm) or norm == 0:
                continue
            similarity = float(np.dot(uploaded_embedding, reference) / (
                np.linalg.norm(uploaded_embedding) * norm
            ))
            text_score = text_similarity
            label_embedding = image.get("label_embedding")
            if (
                text_reranking_available
                and isinstance(label_embedding, list)
                and len(label_embedding) == len(uploaded_embedding)
            ):
                try:
                    label = np.asarray(label_embedding, dtype=np.float32)
                    label_norm = float(np.linalg.norm(label))
                    if math.isfinite(label_norm) and label_norm:
                        text_score = max(
                            text_score,
                            float(
                                np.dot(uploaded_embedding, label)
                                / (np.linalg.norm(uploaded_embedding) * label_norm)
                            ),
                        )
                except (TypeError, ValueError):
                    pass
            combined_score = (
                similarity + TEXT_RERANK_WEIGHT * (text_score - 0.3)
                if text_reranking_available
                else similarity
            )
            if combined_score >= MIN_VISUAL_SIMILARITY:
                alt = str(image.get("alt", ""))
                if not any(character.isalnum() for character in alt):
                    alt = ""
                candidate = {
                    "title": str(document.get("title") or image_url),
                    "place_name": image_place_name(
                        str(document.get("title") or ""),
                        alt,
                        image_url,
                    ),
                    "page_url": page_url,
                    "image_url": image_url,
                    "alt": alt,
                    "confidence": round(max(0, min(100, similarity * 100))),
                }
                previous = ranked_pages.get(page_url)
                if previous is None or combined_score > previous[0]:
                    ranked_pages[page_url] = (combined_score, candidate)

    ranked = list(ranked_pages.values())
    ranked.sort(key=lambda result: result[0], reverse=True)
    return [result for _, result in ranked[:3]]
