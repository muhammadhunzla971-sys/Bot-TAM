import { getCorpus, getPlaces } from "@/lib/data";
import { json, readJsonBody } from "@/lib/http";
import { ItineraryError, createItinerary } from "@/lib/itinerary";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const parsed = await readJsonBody(request);
  if ("error" in parsed) return parsed.error;
  const body = parsed.body;
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return json({ error: "Request must be a JSON object." }, 400);
  }
  const { place_ids: placeIds, days } = body as { place_ids?: unknown; days?: unknown };
  try {
    return json(createItinerary(getCorpus(), getPlaces(), placeIds, days));
  } catch (error) {
    if (error instanceof ItineraryError) return json({ error: error.message }, 400);
    throw error;
  }
}
