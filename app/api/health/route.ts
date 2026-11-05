import { getCorpus } from "@/lib/data";
import { json } from "@/lib/http";
import { countVisualImages } from "@/lib/images";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export function GET() {
  const corpus = getCorpus();
  const images = corpus.documents.reduce(
    (total, document) => total + (Array.isArray(document.images) ? document.images.length : 0),
    0,
  );
  return json({
    ok: true,
    pages: corpus.documents.length,
    images,
    visual_images: countVisualImages(corpus.documents),
  });
}
