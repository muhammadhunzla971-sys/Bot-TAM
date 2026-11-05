/**
 * Pixel-exact port of image_matching.py preprocessing so browser embeddings
 * match the ones stored in the index:
 *   1. PIL alpha_composite onto a white background (only matters for alpha),
 *   2. PIL Image.resize(..., BICUBIC) so the shortest edge is 224 px,
 *   3. centre crop to 224 x 224,
 *   4. CLIP mean/std normalisation, laid out as a [1, 3, 224, 224] tensor.
 * The resize mirrors Pillow's Resample.c (antialiased bicubic, a = -0.5,
 * 22-bit fixed-point coefficients, 8-bit intermediate between passes).
 */

export const PREPROCESS_SIZE = 224;
const MEAN = [0.48145466, 0.4578275, 0.40821073];
const STD = [0.26862954, 0.26130258, 0.27577711];
const PRECISION_BITS = 32 - 8 - 2;

/** Python round(): ties go to the even integer. */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** PIL ImageMode RGBA -> RGB via alpha_composite over opaque white. */
export function flattenOnWhite(rgba: Uint8Array | Uint8ClampedArray, pixelCount: number): Uint8Array {
  const rgb = new Uint8Array(pixelCount * 3);
  const shiftForDiv255 = (value: number) => ((value >>> 8) + value) >>> 8;
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const alpha = rgba[pixel * 4 + 3];
    const coef1 = alpha * 128; // src.a * 255 * 255 * 2^7 / (255 * 255)
    const coef2 = 255 * 128 - coef1;
    for (let channel = 0; channel < 3; channel += 1) {
      const value = rgba[pixel * 4 + channel] * coef1 + 255 * coef2;
      rgb[pixel * 3 + channel] = shiftForDiv255(value + (0x80 << 7)) >>> 7;
    }
  }
  return rgb;
}

function bicubic(x: number): number {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

function precomputeCoefficients(inSize: number, outSize: number) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = 2 * filterScale;
  const kernelSize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const coefficients = new Int32Array(outSize * kernelSize);
  const weights = new Float64Array(kernelSize);
  for (let out = 0; out < outSize; out += 1) {
    const center = (out + 0.5) * scale;
    let min = Math.trunc(center - support + 0.5);
    if (min < 0) min = 0;
    let max = Math.trunc(center + support + 0.5);
    if (max > inSize) max = inSize;
    max -= min;
    let total = 0;
    for (let x = 0; x < max; x += 1) {
      const weight = bicubic((x + min - center + 0.5) / filterScale);
      weights[x] = weight;
      total += weight;
    }
    for (let x = 0; x < max; x += 1) {
      const normalized = total !== 0 ? weights[x] / total : weights[x];
      coefficients[out * kernelSize + x] = Math.trunc(
        normalized * (1 << PRECISION_BITS) + (normalized < 0 ? -0.5 : 0.5),
      );
    }
    bounds[out * 2] = min;
    bounds[out * 2 + 1] = max;
  }
  return { bounds, coefficients, kernelSize };
}

function clip8(value: number): number {
  const shifted = Math.floor(value / (1 << PRECISION_BITS));
  return shifted < 0 ? 0 : shifted > 255 ? 255 : shifted;
}

/** Pillow-compatible BICUBIC resize of a packed RGB image. */
export function resizeBicubic(
  rgb: Uint8Array,
  width: number,
  height: number,
  outWidth: number,
  outHeight: number,
): Uint8Array {
  let current = rgb;
  let currentWidth = width;
  if (outWidth !== width) {
    const { bounds, coefficients, kernelSize } = precomputeCoefficients(width, outWidth);
    const next = new Uint8Array(outWidth * height * 3);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < outWidth; x += 1) {
        const min = bounds[x * 2];
        const count = bounds[x * 2 + 1];
        for (let channel = 0; channel < 3; channel += 1) {
          let sum = 1 << (PRECISION_BITS - 1);
          for (let k = 0; k < count; k += 1) {
            sum += current[(y * width + min + k) * 3 + channel] * coefficients[x * kernelSize + k];
          }
          next[(y * outWidth + x) * 3 + channel] = clip8(sum);
        }
      }
    }
    current = next;
    currentWidth = outWidth;
  }
  if (outHeight !== height) {
    const { bounds, coefficients, kernelSize } = precomputeCoefficients(height, outHeight);
    const next = new Uint8Array(currentWidth * outHeight * 3);
    for (let y = 0; y < outHeight; y += 1) {
      const min = bounds[y * 2];
      const count = bounds[y * 2 + 1];
      for (let x = 0; x < currentWidth; x += 1) {
        for (let channel = 0; channel < 3; channel += 1) {
          let sum = 1 << (PRECISION_BITS - 1);
          for (let k = 0; k < count; k += 1) {
            sum += current[((min + k) * currentWidth + x) * 3 + channel] * coefficients[y * kernelSize + k];
          }
          next[(y * currentWidth + x) * 3 + channel] = clip8(sum);
        }
      }
    }
    current = next;
  }
  return current;
}

/** Return CLIP pixel_values (CHW float32) for an RGBA image. */
export function clipPixelValues(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): Float32Array {
  const rgb = flattenOnWhite(rgba, width * height);
  const scale = PREPROCESS_SIZE / Math.min(width, height);
  const resizedWidth = roundHalfEven(width * scale);
  const resizedHeight = roundHalfEven(height * scale);
  const resized = resizeBicubic(rgb, width, height, resizedWidth, resizedHeight);
  const left = Math.floor((resizedWidth - PREPROCESS_SIZE) / 2);
  const top = Math.floor((resizedHeight - PREPROCESS_SIZE) / 2);
  const area = PREPROCESS_SIZE * PREPROCESS_SIZE;
  const output = new Float32Array(3 * area);
  for (let y = 0; y < PREPROCESS_SIZE; y += 1) {
    for (let x = 0; x < PREPROCESS_SIZE; x += 1) {
      const source = ((top + y) * resizedWidth + left + x) * 3;
      for (let channel = 0; channel < 3; channel += 1) {
        output[channel * area + y * PREPROCESS_SIZE + x] =
          (Math.fround(resized[source + channel] / 255) - MEAN[channel]) / STD[channel];
      }
    }
  }
  return output;
}
