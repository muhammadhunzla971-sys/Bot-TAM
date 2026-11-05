import { verifiedPlaceCoordinates } from "./data";
import { matchPlaces } from "./places";
import { isUrdu, normalizePlaceName, urlencode } from "./text";
import type { LocationCandidate, LocationResult, Place } from "./types";

export const LOCATION_LOOKUP_URL = "https://photon.komoot.io/api/";
const PUNJAB_BBOX = "69.3,27.7,75.5,34.0";
const [SOUTH, NORTH, WEST, EAST] = [27.7, 34.0, 69.3, 75.5];
const KNOWN_NON_PUNJAB_REGIONS = new Set([
  "sindh", "balochistan", "khyber pakhtunkhwa", "kpk",
  "islamabad capital territory", "gilgit-baltistan", "azad kashmir",
]);

type Feature = { geometry?: { coordinates?: unknown }; properties?: Record<string, unknown> };

function toFloat(value: unknown): number | null {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return null;
}

function text(value: unknown): string {
  return value === undefined || value === null || value === "" ? "" : String(value);
}

function locationCandidate(feature: Feature, query: string): LocationCandidate | null {
  const coordinates = feature?.geometry && typeof feature.geometry === "object"
    ? feature.geometry.coordinates
    : undefined;
  const properties = feature?.properties;
  if (
    !Array.isArray(coordinates) ||
    coordinates.length < 2 ||
    !properties ||
    typeof properties !== "object" ||
    text(properties.countrycode).toLowerCase() !== "pk"
  ) {
    return null;
  }
  const longitude = toFloat(coordinates[0]);
  const latitude = toFloat(coordinates[1]);
  if (
    longitude === null ||
    latitude === null ||
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude) ||
    latitude < SOUTH || latitude > NORTH ||
    longitude < WEST || longitude > EAST
  ) {
    return null;
  }

  const state =
    ["state", "province", "region"]
      .map((key) => properties[key])
      .find((value): value is string => typeof value === "string" && Boolean(value.trim()))
      ?.trim() ?? "";
  if (KNOWN_NON_PUNJAB_REGIONS.has(state.toLowerCase())) return null;

  const name = text(properties.name_en || properties.name || properties.city || query).trim();
  const city = text(properties.city || properties.locality).trim();
  const district = text(properties.district || properties.county).trim();
  const country = text(properties.country || "Pakistan").trim();
  const address = [name, city, district, state, country].filter(Boolean).join(" ");
  const queryTokens = new Set(normalizePlaceName(query).split(" ").filter(Boolean));
  const candidateTokens = new Set(normalizePlaceName(address).split(" ").filter(Boolean));
  let overlap = 0;
  for (const token of queryTokens) if (candidateTokens.has(token)) overlap += 1;
  let confidence = overlap / Math.max(1, queryTokens.size);
  confidence = Math.min(1, confidence + (query.toLowerCase().includes(name.toLowerCase()) ? 0.25 : 0));
  return { latitude, longitude, name, city, district, country, state, confidence, precision: "approximate" };
}

/** Prefer verified local coordinates, then use a Punjab-filtered Photon search. */
export async function lookupLocation(
  place: string,
  places: Place[],
  timeoutMs = 5000,
): Promise<LocationResult> {
  const requestedPlace = place;
  const registryMatches = matchPlaces(place, places);
  if (registryMatches.length) {
    const [bestScore, bestPlace] = registryMatches[0];
    if (
      registryMatches.length > 1 &&
      registryMatches[1][0] >= bestScore - 0.03 &&
      registryMatches[1][1].id !== bestPlace.id
    ) {
      const candidates = registryMatches.slice(0, 5).map(([, candidate]) => {
        const details: LocationCandidate = {
          name: candidate.name_en,
          district: candidate.district,
          precision: candidate.verified ? "verified" : "approximate",
          source_url: candidate.source_url,
        };
        const coordinates = verifiedPlaceCoordinates(candidate);
        if (coordinates) [details.latitude, details.longitude] = coordinates;
        return details;
      });
      return {
        status: "ambiguous",
        precision: "search only",
        message: "More than one place matched. Please specify the place name or district.",
        candidates,
      };
    }
    if (bestPlace.verified) {
      return {
        status: "found",
        precision: "verified",
        latitude: bestPlace.lat as number,
        longitude: bestPlace.lng as number,
        label: isUrdu(place) ? bestPlace.name_ur : bestPlace.name_en,
        city: "",
        district: bestPlace.district,
        country: "Pakistan",
        source_url: bestPlace.source_url,
      };
    }
    place = `${bestPlace.name_en}, ${bestPlace.district}, Punjab, Pakistan`;
  }

  let result: unknown;
  try {
    const response = await fetch(
      LOCATION_LOOKUP_URL +
        "?" +
        urlencode({ q: place, limit: 5, bbox: PUNJAB_BBOX, lang: isUrdu(requestedPlace) ? "ur" : "en" }),
      {
        headers: { "User-Agent": "PunjabTourismAssistant/1.0", Accept: "application/json" },
        signal: AbortSignal.timeout(timeoutMs),
        cache: "no-store",
      },
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const body = await response.text();
    if (body.length > 128 * 1024) throw new Error("Response too large");
    result = JSON.parse(body);
  } catch {
    console.error("Location lookup failed (service or response error).");
    return {
      status: "error",
      precision: "search only",
      message: "Location service is temporarily unavailable. The map search link is still available.",
    };
  }

  const features =
    result && typeof result === "object" && !Array.isArray(result)
      ? (result as { features?: unknown }).features
      : undefined;
  if (!Array.isArray(features)) {
    return {
      status: "error",
      precision: "search only",
      message: "Location service returned an invalid response. The map search link is still available.",
    };
  }
  const firstFeatures = features.slice(0, 5).filter(
    (feature): feature is Feature => Boolean(feature) && typeof feature === "object" && !Array.isArray(feature),
  );
  const candidates = firstFeatures
    .map((feature) => locationCandidate(feature, place))
    .filter((candidate): candidate is LocationCandidate => candidate !== null && (candidate.confidence ?? 0) >= 0.34)
    .sort((first, second) => (second.confidence ?? 0) - (first.confidence ?? 0));
  if (!candidates.length) {
    return {
      status: "not_found",
      precision: "search only",
      message: "No matching place was found in Punjab, Pakistan. Please add a city or district.",
      candidates: [],
    };
  }
  if (
    candidates.length > 1 &&
    (candidates[1].confidence ?? 0) >= (candidates[0].confidence ?? 0) - 0.12 &&
    (candidates[1].latitude !== candidates[0].latitude || candidates[1].longitude !== candidates[0].longitude)
  ) {
    return {
      status: "ambiguous",
      precision: "search only",
      message: "Multiple places match. Please specify the city or district.",
      candidates: candidates.slice(0, 5),
    };
  }
  const selected = candidates[0];
  let label = selected.name;
  if (isUrdu(requestedPlace)) {
    for (const feature of firstFeatures) {
      const candidate = locationCandidate(feature, place);
      if (
        candidate &&
        candidate.latitude === selected.latitude &&
        candidate.longitude === selected.longitude &&
        typeof feature.properties?.name === "string"
      ) {
        label = feature.properties.name as string;
        break;
      }
    }
  }
  return {
    status: "found",
    precision: "approximate",
    latitude: selected.latitude,
    longitude: selected.longitude,
    label,
    city: selected.city,
    district: selected.district,
    country: selected.country,
  };
}
