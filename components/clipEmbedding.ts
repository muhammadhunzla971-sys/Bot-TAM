/**
 * Compute a CLIP image embedding in the visitor's browser.
 *
 * Uses the same pinned, quantized Xenova/clip-vit-base-patch32 vision model
 * that the Python indexer used, so embeddings are comparable with the ones
 * stored in data/tourism_pages.jsonl. The image never leaves the device; only
 * the 512 int8 values are sent to /api/image. The model (~89 MB) is
 * downloaded from Hugging Face on first use and cached by the browser.
 */

import { PREPROCESS_SIZE, clipPixelValues } from "./clipPreprocess";

const TRANSFORMERS_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";
const MODEL_ID = "Xenova/clip-vit-base-patch32";
const MODEL_REVISION = "6b7c523703ae26e39be4b03774966f8571c7f523";
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 40_000_000;
const SUPPORTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);

/* eslint-disable @typescript-eslint/no-explicit-any */
type Runtime = { model: any; Tensor: any };
let runtime: Promise<Runtime> | null = null;

function loadRuntime(): Promise<Runtime> {
  if (!runtime) {
    runtime = (async () => {
      const transformers: any = await import(
        /* webpackIgnore: true */ /* turbopackIgnore: true */ TRANSFORMERS_URL
      );
      transformers.env.allowLocalModels = false;
      const model = await transformers.CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, {
        revision: MODEL_REVISION,
        dtype: "q8", // onnx/vision_model_quantized.onnx, as used by the Python indexer
      });
      return { model, Tensor: transformers.Tensor };
    })();
    runtime.catch(() => {
      runtime = null; // allow a retry after a network failure
    });
  }
  return runtime;
}

/** Decode the file with EXIF orientation applied and return raw, unpremultiplied RGBA. */
async function decodeImage(file: File): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, {
      imageOrientation: "from-image",
      premultiplyAlpha: "none",
      colorSpaceConversion: "none", // Pillow ignores embedded colour profiles too
    });
  } catch {
    throw new Error("The uploaded file is not a valid supported image.");
  }
  try {
    const { width, height } = bitmap;
    if (width * height > MAX_IMAGE_PIXELS) throw new Error("Image dimensions are too large.");
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("This browser cannot process images.");
    context.drawImage(bitmap, 0, 0);
    return { data: context.getImageData(0, 0, width, height).data, width, height };
  } finally {
    bitmap.close();
  }
}

/** numpy.rint: round half to even. */
function rint(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

export async function clipImageEmbedding(
  file: File,
  onStatus?: (message: string) => void,
): Promise<number[]> {
  if (!file.size || file.size > MAX_IMAGE_BYTES) throw new Error("Image is empty or larger than 10 MiB.");
  if (!SUPPORTED_TYPES.has(file.type)) throw new Error("Use a JPEG, PNG, WebP, or GIF image.");
  const image = await decodeImage(file);
  const pixelValues = clipPixelValues(image.data, image.width, image.height);
  onStatus?.("Loading the image-matching model in your browser (the first use downloads about 90 MB)...");
  let loaded: Runtime;
  try {
    loaded = await loadRuntime();
  } catch {
    throw new Error("Visual image matching is unavailable: the image model could not be loaded.");
  }
  onStatus?.("Comparing with Punjab Tourism website images...");
  const { image_embeds: embeds } = await loaded.model({
    pixel_values: new loaded.Tensor("float32", pixelValues, [1, 3, PREPROCESS_SIZE, PREPROCESS_SIZE]),
  });
  const values = Array.from(embeds.data as Float32Array);
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  if (!Number.isFinite(norm) || norm === 0) throw new Error("The image model returned an invalid embedding.");
  // Same int8 quantization as image_matching.visual_embedding.
  return values.map((value) => rint(Math.max(-1, Math.min(1, value / norm)) * 127));
}
