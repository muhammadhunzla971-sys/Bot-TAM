import { normalizePlaceName } from "./text";
import { placeNames } from "./places";
import type { ImageResult, PageDocument, Place } from "./types";

export const EMBEDDING_SIZE = 512;
const MIN_VISUAL_SIMILARITY = 0.78;
const TEXT_RERANK_WEIGHT = 0.5;
const TOURISM_PREFIX = "https://tourism.punjab.gov.pk/";

const KNOWN_IMAGE_PLACE_NAMES: Record<string, string> = {
  "lahore-1": "Minar-e-Pakistan, Lahore",
};

/** Python Path(unquote(urlsplit(url).path)).stem */
export function urlFileStem(imageUrl: string): string {
  let pathname: string;
  try {
    pathname = new URL(imageUrl).pathname;
  } catch {
    pathname = imageUrl.split(/[?#]/)[0];
  }
  try {
    pathname = decodeURIComponent(pathname);
  } catch {
    // keep the raw path
  }
  const name = pathname.split("/").filter(Boolean).pop() ?? "";
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(0, dot) : name;
}

function hasAlphanumeric(value: string): boolean {
  return /[\p{L}\p{N}]/u.test(value);
}

function titleCase(value: string): string {
  // Python str.title(): every run of letters starts with an uppercase letter.
  return value.replace(/\p{L}+/gu, (word) => word[0].toUpperCase() + word.slice(1).toLowerCase());
}

/** Return the most specific place name available for an indexed image. */
export function imagePlaceName(title: string, alt: string, imageUrl: string): string {
  const filename = urlFileStem(imageUrl);
  const known = KNOWN_IMAGE_PLACE_NAMES[filename.toLowerCase()];
  if (known) return known;

  const genericAlts = new Set([
    "cultural activities",
    "eco tourism",
    "family activities",
    "heritage",
    "outdoor adventures",
    "punjab tourism department",
    "shopping experience",
    "spiritual journey",
    "tourism, archaeology and museums department",
  ]);
  if (hasAlphanumeric(alt) && !genericAlts.has(alt.toLowerCase())) return alt.trim();

  const pageTitle = title
    .replace(/\s*\|\s*Tourism, Archaeology and Museums Department.*$/is, "")
    .trim();
  if (
    pageTitle &&
    ![
      "welcome to tourism",
      "welcome to tourism, archaeology and museums department",
      "punjab tourism department",
    ].includes(pageTitle.toLowerCase())
  ) {
    return pageTitle;
  }

  let words = filename.replace(/[_+]+/g, " ");
  words = words.replace(/\b(?:banner|image|photo|picture|thumbnail|thumb)\b/gi, " ");
  words = words.replace(/\s+\d+$/, "");
  words = words.replace(/\s+/g, " ").replace(/^[ \-_]+|[ \-_]+$/g, "");
  if (words && !/^\d+$/.test(words.replace(/ /g, ""))) return titleCase(words);
  return pageTitle || title;
}

function imageMatchesPlace(
  title: string,
  alt: string,
  imageUrl: string,
  place: Place,
  places: Place[],
): boolean {
  const filename = urlFileStem(imageUrl);
  const metadata = normalizePlaceName(`${alt} ${filename.replace(/_/g, " ")}`);
  const namesFor = (record: Place) =>
    placeNames(record).map(normalizePlaceName).filter(Boolean);
  const inMetadata = (name: string) => ` ${metadata} `.includes(` ${name} `);

  const requested = namesFor(place);
  if (requested.some(inMetadata)) return true;
  for (const other of places) {
    if (other.id === place.id) continue;
    if (namesFor(other).some(inMetadata)) return false;
  }
  const pageTitle = normalizePlaceName(title);
  return requested.some((name) => ` ${pageTitle} `.includes(` ${name} `));
}

export function tourismPageImages(
  documents: PageDocument[],
  limit = 3,
  place: Place | null = null,
  places: Place[] = [],
): ImageResult[] {
  const images: ImageResult[] = [];
  const seen = new Set<string>();
  for (const document of documents) {
    const pageUrl = String(document.url ?? "");
    const title = String(document.title || pageUrl);
    if (!Array.isArray(document.images)) continue;
    for (const image of document.images) {
      if (!image || typeof image !== "object") continue;
      const imageUrl = image.url;
      if (typeof imageUrl !== "string" || !imageUrl.startsWith(TOURISM_PREFIX) || seen.has(imageUrl)) {
        continue;
      }
      const alt = String(image.alt ?? "");
      if (place && !imageMatchesPlace(title, alt, imageUrl, place, places)) continue;
      images.push({
        title,
        place_name: place
          ? String(place.name_en || place.name_ur || "")
          : imagePlaceName(title, alt, imageUrl),
        page_url: pageUrl,
        image_url: imageUrl,
        alt,
      });
      seen.add(imageUrl);
      if (images.length >= limit) return images;
    }
  }
  return images;
}

/** Python's round(): ties go to the even integer. */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

function isEmbedding(value: unknown, size = EMBEDDING_SIZE): value is number[] {
  return Array.isArray(value) && value.length === size && value.every((item) => typeof item === "number");
}

function norm(vector: number[]): number {
  let sum = 0;
  for (const value of vector) sum += value * value;
  return Math.sqrt(sum);
}

function cosine(first: number[], firstNorm: number, second: number[]): number | null {
  const secondNorm = norm(second);
  if (!Number.isFinite(secondNorm) || secondNorm === 0) return null;
  let dot = 0;
  for (let index = 0; index < first.length; index += 1) dot += first[index] * second[index];
  return dot / (firstNorm * secondNorm);
}

export function countVisualImages(documents: PageDocument[]): number {
  let count = 0;
  for (const document of documents) {
    if (!Array.isArray(document.images)) continue;
    for (const image of document.images) {
      if (image && typeof image === "object" && isEmbedding(image.embedding)) count += 1;
    }
  }
  return count;
}

/**
 * Rank indexed tourism images against an uploaded image's CLIP embedding.
 * Port of image_matching.match_image; the embedding itself is computed in
 * the visitor's browser with the same pinned quantized CLIP vision model.
 */
export function matchImageEmbedding(uploaded: number[], documents: PageDocument[]) {
  const uploadedNorm = norm(uploaded);
  if (!Number.isFinite(uploadedNorm) || uploadedNorm === 0) {
    throw new Error("The image embedding is invalid.");
  }
  const textRerankingAvailable = documents.some((document) =>
    isEmbedding(document.text_embedding, uploaded.length),
  );
  const rankedPages = new Map<string, [number, ImageResult]>();

  for (const document of documents) {
    if (!Array.isArray(document.images)) continue;
    const pageUrl = String(document.url ?? "");
    let textSimilarity = 0;
    if (textRerankingAvailable && isEmbedding(document.text_embedding, uploaded.length)) {
      textSimilarity = cosine(uploaded, uploadedNorm, document.text_embedding) ?? 0;
    }
    for (const image of document.images) {
      if (!image || typeof image !== "object") continue;
      const imageUrl = image.url;
      if (
        typeof imageUrl !== "string" ||
        !imageUrl.startsWith(TOURISM_PREFIX) ||
        !isEmbedding(image.embedding, uploaded.length)
      ) {
        continue;
      }
      const similarity = cosine(uploaded, uploadedNorm, image.embedding);
      if (similarity === null) continue;
      let textScore = textSimilarity;
      if (textRerankingAvailable && isEmbedding(image.label_embedding, uploaded.length)) {
        const labelScore = cosine(uploaded, uploadedNorm, image.label_embedding);
        if (labelScore !== null) textScore = Math.max(textScore, labelScore);
      }
      const combined = textRerankingAvailable
        ? similarity + TEXT_RERANK_WEIGHT * (textScore - 0.3)
        : similarity;
      if (combined < MIN_VISUAL_SIMILARITY) continue;
      let alt = String(image.alt ?? "");
      if (!hasAlphanumeric(alt)) alt = "";
      const candidate: ImageResult = {
        title: String(document.title || imageUrl),
        place_name: imagePlaceName(String(document.title || ""), alt, imageUrl),
        page_url: pageUrl,
        image_url: imageUrl,
        alt,
        confidence: roundHalfEven(Math.max(0, Math.min(100, similarity * 100))),
      };
      const previous = rankedPages.get(pageUrl);
      if (!previous || combined > previous[0]) rankedPages.set(pageUrl, [combined, candidate]);
    }
  }

  return [...rankedPages.values()]
    .sort((first, second) => second[0] - first[0])
    .slice(0, 3)
    .map(([, result]) => result);
}
