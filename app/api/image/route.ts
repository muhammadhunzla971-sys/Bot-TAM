import { getCorpus } from "@/lib/data";
import { json, readJsonBody } from "@/lib/http";
import { EMBEDDING_SIZE, countVisualImages, matchImageEmbedding } from "@/lib/images";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Accepts a CLIP image embedding computed in the visitor's browser (the image
 * itself never leaves the device) and ranks indexed tourism images.
 */
export async function POST(request: Request) {
  const parsed = await readJsonBody(request);
  if ("error" in parsed) return parsed.error;
  const body = parsed.body as { embedding?: unknown } | null;
  const embedding = body && typeof body === "object" ? body.embedding : undefined;
  if (
    !Array.isArray(embedding) ||
    embedding.length !== EMBEDDING_SIZE ||
    !embedding.every((value) => Number.isInteger(value) && value >= -127 && value <= 127) ||
    embedding.every((value) => value === 0)
  ) {
    return json({ error: "The image embedding is invalid. Please try the image again." }, 400);
  }

  const corpus = getCorpus();
  if (!countVisualImages(corpus.documents)) {
    return json(
      {
        error:
          "The visual image index is not built. Run index_tourism_images.py to prepare reference images.",
      },
      503,
    );
  }
  const matches = matchImageEmbedding(embedding as number[], corpus.documents);
  const answer = matches.length
    ? `This image appears to show ${matches[0].place_name}. It most closely matches an image on ` +
      `${matches[0].title} on the Punjab Tourism website (${matches[0].confidence}% visual similarity).`
    : "I couldn't find a close visual match in the indexed Punjab Tourism website images.";
  return json({ answer, matches });
}
