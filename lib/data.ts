import "server-only";

import { readFileSync } from "node:fs";
import path from "node:path";

import { Corpus } from "./corpus";
import type { PageDocument, Place, PublicPlace } from "./types";

const DATA_DIR = path.join(process.cwd(), "data");

let corpusInstance: Corpus | null = null;
let placesInstance: Place[] | null = null;

/** Load and validate the JSONL corpus once per server instance. */
export function getCorpus(): Corpus {
  if (corpusInstance) return corpusInstance;
  const filePath = path.join(DATA_DIR, "tourism_pages.jsonl");
  const lines = readFileSync(filePath, "utf-8").split(/\r?\n/);
  const documents: PageDocument[] = [];
  lines.forEach((line, index) => {
    if (!line.trim()) return;
    let document: unknown;
    try {
      document = JSON.parse(line);
    } catch (error) {
      throw new Error(`Invalid JSONL at ${filePath}:${index + 1}: ${(error as Error).message}`);
    }
    if (!document || typeof document !== "object" || Array.isArray(document)) {
      throw new Error(`Expected a JSON object at ${filePath}:${index + 1}.`);
    }
    const record = document as Record<string, unknown>;
    if (typeof record.url !== "string" || typeof record.text !== "string") {
      throw new Error(
        `Each JSONL record needs string 'url' and 'text' fields (${filePath}:${index + 1}).`,
      );
    }
    documents.push(record as unknown as PageDocument);
  });
  if (!documents.length) {
    throw new Error(`No usable page records found in ${filePath}.`);
  }
  corpusInstance = new Corpus(documents);
  return corpusInstance;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Load and validate data/places.json once per server instance. */
export function getPlaces(): Place[] {
  if (placesInstance) return placesInstance;
  const filePath = path.join(DATA_DIR, "places.json");
  const places: unknown = JSON.parse(readFileSync(filePath, "utf-8"));
  if (!Array.isArray(places)) {
    throw new Error(`Places database must be a JSON array: ${filePath}`);
  }
  const required = [
    "id", "name_en", "name_ur", "aliases", "lat", "lng", "district",
    "category", "source_url", "verified", "note",
  ];
  places.forEach((place, index) => {
    if (
      !place ||
      typeof place !== "object" ||
      !required.every((key) => key in place) ||
      !Array.isArray(place.aliases) ||
      !place.aliases.every((alias: unknown) => typeof alias === "string") ||
      typeof place.verified !== "boolean"
    ) {
      throw new Error(`Invalid places database entry at index ${index}.`);
    }
    const { lat, lng } = place;
    if ((lat === null) !== (lng === null)) {
      throw new Error(`Place coordinates must both be set or both be null: ${place.id}.`);
    }
    if (place.verified && (lat === null || lng === null)) {
      throw new Error(`Verified place needs coordinates: ${place.id}.`);
    }
    if (
      lat !== null &&
      (!isFiniteNumber(lat) || !isFiniteNumber(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180)
    ) {
      throw new Error(`Invalid coordinates in places database: ${place.id}.`);
    }
  });
  placesInstance = places as Place[];
  return placesInstance;
}

export function verifiedPlaceCoordinates(place: Place): [number, number] | null {
  const { lat, lng } = place;
  if (
    place.verified !== true ||
    !isFiniteNumber(lat) ||
    !isFiniteNumber(lng) ||
    lat < -90 || lat > 90 || lng < -180 || lng > 180
  ) {
    return null;
  }
  return [lat, lng];
}

/** Public projection used by GET /api/places and the page's server render. */
export function publicPlaces(): PublicPlace[] {
  return getPlaces().map((place) => {
    const coordinates = verifiedPlaceCoordinates(place);
    return {
      id: place.id,
      name_en: place.name_en,
      name_ur: place.name_ur,
      district: place.district,
      category: place.category,
      source_url: place.source_url,
      verified: place.verified === true && coordinates !== null,
      latitude: coordinates ? coordinates[0] : null,
      longitude: coordinates ? coordinates[1] : null,
    };
  });
}
