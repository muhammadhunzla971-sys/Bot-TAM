import { NO_ANSWER, localCorpusAnswer } from "./answers";
import type { Corpus } from "./corpus";
import { pageMentionsPlace } from "./places";
import { googleMapsSearchUrl } from "./text";
import type { PageDocument, Place } from "./types";

export class ItineraryError extends Error {}

export function createItinerary(corpus: Corpus, places: Place[], placeIds: unknown, dayCount: unknown) {
  if (typeof dayCount !== "number" || !Number.isInteger(dayCount) || dayCount < 1 || dayCount > 7) {
    throw new ItineraryError("Choose between 1 and 7 days.");
  }
  if (!Array.isArray(placeIds) || placeIds.length < 1 || placeIds.length > 10) {
    throw new ItineraryError("Choose between 1 and 10 tourism places.");
  }
  if (placeIds.some((placeId) => typeof placeId !== "string")) {
    throw new ItineraryError("The selected tourism places are invalid.");
  }
  if (new Set(placeIds).size !== placeIds.length) {
    throw new ItineraryError("Choose each tourism place only once.");
  }
  const placeById = new Map(places.filter((place) => typeof place.id === "string").map((place) => [place.id, place]));

  const selected: Array<{ place: Place; page: PageDocument | null; description: string }> = [];
  for (const placeId of placeIds as string[]) {
    const place = placeById.get(placeId);
    if (!place) throw new ItineraryError("One or more selected tourism places are unavailable.");
    let page =
      corpus.documents.find(
        (document) => document.url === place.source_url && pageMentionsPlace(document, place),
      ) ?? null;
    const excerpt = page ? localCorpusAnswer(String(place.name_en ?? ""), [page]) : { answer: NO_ANSWER };
    let description: string;
    if (excerpt.answer === NO_ANSWER) {
      page = null;
      description =
        "No dedicated Punjab Tourism information is indexed for this place. " +
        "This is a planning placeholder; use the map link for location only.";
    } else {
      description = excerpt.answer;
    }
    selected.push({ place, page, description });
  }

  const timeSlots = ["Morning (suggested)", "Afternoon (suggested)", "Evening (suggested)"];
  const days = Array.from({ length: dayCount }, (_, dayIndex) => {
    const items: Array<Record<string, unknown>> = [];
    selected.forEach(({ place, page, description }, index) => {
      const assignedDay = Math.min(dayCount - 1, Math.floor((index * dayCount) / selected.length));
      if (assignedDay !== dayIndex) return;
      const placeName = String(place.name_en || place.name_ur || "");
      const sourceUrl = page
        ? String(page.url)
        : googleMapsSearchUrl(
            [placeName, String(place.district || ""), "Punjab, Pakistan"].filter(Boolean).join(", "),
          );
      items.push({
        place_id: String(place.id),
        place_name: placeName,
        district: String(place.district || ""),
        timing: items.length < timeSlots.length ? timeSlots[items.length] : "Flexible (suggested)",
        description,
        source: {
          title: page
            ? String(page.title || sourceUrl)
            : "Google Maps place search (location only; outside Punjab Tourism)",
          url: sourceUrl,
        },
      });
    });
    return { day: dayIndex + 1, items };
  });

  return {
    days,
    note:
      "Suggested time slots only; the tourism pages do not establish " +
      "opening hours, prices, travel times, or route order. Days without " +
      "selected places are left open.",
  };
}
