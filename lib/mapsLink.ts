import { lookup } from "node:dns/promises";
import { isIP } from "node:net";

import { lookupLocation } from "./location";
import { collapseWhitespace, openStreetMapUrl } from "./text";
import type { ChatResponse, Place, Source } from "./types";

const MAP_LINK_TIMEOUT_MS = 3000;
const MAP_LINK_MAX_REDIRECTS = 3;
const MAP_LINK_HOSTS = new Set([
  "google.com",
  "www.google.com",
  "maps.google.com",
  "maps.app.goo.gl",
  "goo.gl",
]);

type ErrorCode = "invalid" | "unsupported" | "too_many_redirects" | "unavailable";

class MapLinkError extends Error {
  constructor(readonly code: ErrorCode) {
    super(code);
  }
}

export type MapLink =
  | { error: string }
  | {
      url: string;
      resolved_url: string;
      coordinates: [number, number] | null;
      place_query: string | null;
    };

function mapLinkCandidate(url: string): boolean {
  const lowered = url.toLowerCase();
  if (["maps.google.", "maps.app.goo.gl", "goo.gl"].some((host) => lowered.includes(host))) return true;
  try {
    return new URL(url).pathname.toLowerCase().includes("/maps");
  } catch {
    return lowered.includes("maps");
  }
}

function isMapsPath(path: string): boolean {
  return path === "/maps" || path.startsWith("/maps/");
}

function validatedMapUrl(url: string): { parsed: URL; host: string } {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new MapLinkError("invalid");
  }
  const host = parsed.hostname.toLowerCase().replace(/\.$/, "");
  const authority = url.replace(/^[a-z][a-z0-9+.-]*:\/\//i, "").split(/[/?#]/)[0];
  if (
    parsed.protocol !== "https:" ||
    !host ||
    authority.includes("@") ||
    !(parsed.port === "" || parsed.port === "443") ||
    !MAP_LINK_HOSTS.has(host)
  ) {
    throw new MapLinkError("unsupported");
  }
  const path = parsed.pathname || "/";
  if (
    ((host === "google.com" || host === "www.google.com") && !isMapsPath(path)) ||
    (host === "goo.gl" && !isMapsPath(path)) ||
    (host === "maps.app.goo.gl" && path === "/") ||
    (host === "maps.google.com" && path !== "/" && !isMapsPath(path))
  ) {
    throw new MapLinkError("unsupported");
  }
  return { parsed, host };
}

function parseCoordinatePair(value: string): [number, number] | null {
  const match = /(?<![\d.])(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)(?![\d.])/.exec(value);
  if (!match) return null;
  const latitude = Number(match[1]);
  const longitude = Number(match[2]);
  if (
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude) ||
    latitude < -90 || latitude > 90 ||
    longitude < -180 || longitude > 180
  ) {
    return null;
  }
  return [latitude, longitude];
}

/** urllib.parse.parse_qs(keep_blank_values=False) */
function parseQuery(parsed: URL): Map<string, string[]> {
  const query = new Map<string, string[]>();
  for (const [key, value] of parsed.searchParams) {
    if (!value) continue;
    const values = query.get(key) ?? [];
    values.push(value);
    query.set(key, values);
  }
  return query;
}

const COORDINATE_KEYS = ["query", "q", "ll", "center", "destination"];

function mapUrlCoordinates(url: string, query: Map<string, string[]>): [number, number] | null {
  for (const key of COORDINATE_KEYS) {
    for (const value of query.get(key) ?? []) {
      const coordinates = parseCoordinatePair(value);
      if (coordinates) return coordinates;
    }
  }
  for (const pattern of [
    /@(-?\d{1,2}(?:\.\d+)?),(-?\d{1,3}(?:\.\d+)?)/,
    /!3d(-?\d{1,2}(?:\.\d+)?)!4d(-?\d{1,3}(?:\.\d+)?)/,
  ]) {
    const match = pattern.exec(url);
    if (match) {
      const coordinates = parseCoordinatePair(`${match[1]},${match[2]}`);
      if (coordinates) return coordinates;
    }
  }
  const swapped = /!2d(-?\d{1,3}(?:\.\d+)?)!3d(-?\d{1,2}(?:\.\d+)?)/.exec(url);
  if (swapped) {
    const coordinates = parseCoordinatePair(`${swapped[2]},${swapped[1]}`);
    if (coordinates) return coordinates;
  }
  return null;
}

function mapUrlPlaceQuery(parsed: URL, query: Map<string, string[]>): string | null {
  for (const key of ["query", "q", "destination"]) {
    for (const value of query.get(key) ?? []) {
      const cleaned = collapseWhitespace(value).trim();
      if (cleaned) return cleaned;
    }
  }
  const parts = parsed.pathname.split("/").filter(Boolean);
  for (let index = 0; index < parts.length - 1; index += 1) {
    if (["place", "search"].includes(parts[index].toLowerCase())) {
      let decoded = parts[index + 1];
      try {
        decoded = decodeURIComponent(decoded);
      } catch {
        // keep raw segment
      }
      const place = collapseWhitespace(decoded.replace(/\+/g, " "));
      if (place && !place.startsWith("@")) return place;
    }
  }
  return null;
}

function isGlobalAddress(address: string): boolean {
  const version = isIP(address);
  if (version === 4) {
    const [a, b, c] = address.split(".").map(Number);
    return !(
      a === 0 || a === 10 || a === 127 || a >= 224 ||
      (a === 100 && b >= 64 && b <= 127) ||
      (a === 169 && b === 254) ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168) ||
      (a === 192 && b === 0 && (c === 0 || c === 2)) ||
      (a === 198 && (b === 18 || b === 19)) ||
      (a === 198 && b === 51 && c === 100) ||
      (a === 203 && b === 0 && c === 113)
    );
  }
  if (version === 6) {
    const lowered = address.toLowerCase();
    const mapped = /^::ffff:(\d+\.\d+\.\d+\.\d+)$/.exec(lowered);
    if (mapped) return isGlobalAddress(mapped[1]);
    return !(
      lowered === "::" ||
      lowered === "::1" ||
      /^f[cd]/.test(lowered) ||
      /^fe[89ab]/.test(lowered) ||
      lowered.startsWith("ff") ||
      lowered.startsWith("2001:db8") ||
      lowered.startsWith("64:ff9b")
    );
  }
  return false;
}

async function assertPublicHost(host: string): Promise<void> {
  let addresses: { address: string }[];
  try {
    addresses = await lookup(host, { all: true });
  } catch {
    throw new MapLinkError("unavailable");
  }
  if (!addresses.length || addresses.some(({ address }) => !isGlobalAddress(address))) {
    throw new MapLinkError("unsupported");
  }
}

/**
 * Follow an allowlisted short Google Maps link manually, re-validating each
 * redirect target against the host allowlist and public-IP checks.
 */
async function followShortMapLink(url: string): Promise<string> {
  let currentUrl = url;
  for (let redirectCount = 0; redirectCount <= MAP_LINK_MAX_REDIRECTS; redirectCount += 1) {
    const { host } = validatedMapUrl(currentUrl);
    await assertPublicHost(host);
    let response: Response;
    try {
      response = await fetch(currentUrl, {
        method: "GET",
        redirect: "manual",
        headers: { Range: "bytes=0-0" },
        signal: AbortSignal.timeout(MAP_LINK_TIMEOUT_MS),
        cache: "no-store",
      });
    } catch {
      throw new MapLinkError("unavailable");
    }
    await response.body?.cancel().catch(() => undefined);
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      const destination = response.headers.get("location");
      if (!destination) throw new MapLinkError("invalid");
      if (redirectCount >= MAP_LINK_MAX_REDIRECTS) throw new MapLinkError("too_many_redirects");
      try {
        currentUrl = new URL(destination, currentUrl).toString();
      } catch {
        throw new MapLinkError("invalid");
      }
      validatedMapUrl(currentUrl);
      continue;
    }
    if (response.status < 200 || response.status >= 300) throw new MapLinkError("unavailable");
    return currentUrl;
  }
  throw new MapLinkError("too_many_redirects");
}

const MAP_LINK_MESSAGES: Record<ErrorCode, string> = {
  invalid:
    "I couldn't parse a place or coordinates from that Google Maps link. " +
    "اس Google Maps لنک سے جگہ یا coordinates نہیں مل سکے۔",
  unsupported:
    "Unsupported map link. Use an HTTPS Google Maps link from an " +
    "allowlisted Maps host. غیر معاون لنک؛ HTTPS Google Maps لنک استعمال کریں۔",
  too_many_redirects:
    "That short Google Maps link redirected too many times. " +
    "اس مختصر Google Maps لنک نے بہت زیادہ redirects کیے۔",
  unavailable:
    "I couldn't safely resolve that short Google Maps link. " +
    "اس مختصر Google Maps لنک کو محفوظ طریقے سے resolve نہیں کیا جا سکا۔",
};

/** Parse an allowlisted Google Maps URL without fetching arbitrary URLs. */
export async function resolveGoogleMapsLink(text: string): Promise<MapLink | null> {
  const links = text.match(/https?:\/\/[^\s<>]+/gi);
  if (!links) return null;
  const url = links
    .map((link) => link.replace(/[.,!?;:)\]}]+$/, ""))
    .find(mapLinkCandidate);
  if (!url) return null;
  try {
    let { parsed, host } = validatedMapUrl(url);
    let query = parseQuery(parsed);
    let coordinates = mapUrlCoordinates(url, query);
    const coordinateTexts = COORDINATE_KEYS.flatMap((key) => query.get(key) ?? []);
    if (
      coordinates === null &&
      coordinateTexts.some((value) => /^\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*$/.test(value))
    ) {
      throw new MapLinkError("invalid");
    }
    let placeQuery = mapUrlPlaceQuery(parsed, query);
    const isShortLink = host === "maps.app.goo.gl" || host === "goo.gl";
    let resolvedUrl = url;
    if (coordinates === null && !placeQuery && isShortLink) {
      resolvedUrl = await followShortMapLink(url);
      ({ parsed } = validatedMapUrl(resolvedUrl));
      query = parseQuery(parsed);
      coordinates = mapUrlCoordinates(resolvedUrl, query);
      placeQuery = mapUrlPlaceQuery(parsed, query);
    }
    if (coordinates === null && !placeQuery) throw new MapLinkError("invalid");
    if (coordinates !== null) placeQuery = null;
    return { url, resolved_url: resolvedUrl, coordinates, place_query: placeQuery };
  } catch (error) {
    if (error instanceof MapLinkError) return { error: MAP_LINK_MESSAGES[error.code] };
    throw error;
  }
}

export async function googleMapsLinkResponse(mapLink: MapLink, places: Place[]): Promise<ChatResponse> {
  if ("error" in mapLink) return { answer: mapLink.error, sources: [], maps: [] };
  const originalUrl = mapLink.url;
  const googleSource: Source = {
    title: "User-provided Google Maps link (not independently verified)",
    url: originalUrl,
  };
  if (mapLink.coordinates) {
    const [latitude, longitude] = mapLink.coordinates;
    const osmUrl = openStreetMapUrl(latitude, longitude);
    return {
      answer:
        "Coordinates parsed from your Google Maps link are not independently verified. " +
        "These are not a verified location pin. آپ کے Google Maps لنک سے " +
        "coordinates لیے گئے ہیں؛ ان کی الگ سے تصدیق نہیں ہوئی۔",
      sources: [
        googleSource,
        {
          title: "OpenStreetMap coordinates (from user-provided link; outside Punjab Tourism website)",
          url: osmUrl,
        },
      ],
      maps: [{ title: "Google Maps link", url: originalUrl }],
      location: {
        status: "found",
        precision: "coordinates parsed from link",
        latitude,
        longitude,
        label: "Coordinates from Google Maps link (unverified)",
        message: "Coordinates parsed from the Google Maps link; not independently verified.",
        sources: [googleSource, { title: "OpenStreetMap location", url: osmUrl }],
      },
      answer_origin: "user-provided Google Maps link",
    };
  }

  const placeQuery = mapLink.place_query ?? "";
  const location = await lookupLocation(placeQuery, places);
  let answer: string;
  if (location.status === "found") {
    const precision = location.precision || "search only";
    const lookupSource = precision === "verified" ? "the verified places registry" : "Photon";
    answer =
      `I extracted ${placeQuery} from the Google Maps link and resolved ` +
      `it using ${lookupSource}. Location precision: ${precision}. ` +
      "لنک سے جگہ حاصل کی گئی؛ مقام کی درستگی اوپر درج ہے۔";
  } else {
    answer =
      `I extracted ${placeQuery} from the Google Maps link, but couldn't ` +
      "geocode it to coordinates. میں نے لنک سے جگہ حاصل کی، مگر اس کے " +
      "coordinates نہیں ملے۔";
  }
  if (Array.isArray(location.sources)) location.sources.unshift(googleSource);
  return {
    answer,
    sources: [googleSource],
    maps: [{ title: "Google Maps link", url: originalUrl }],
    location,
    answer_origin: "Google Maps place link; location geocoded with Photon",
  };
}
