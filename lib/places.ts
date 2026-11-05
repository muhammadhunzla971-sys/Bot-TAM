import { normalizePlaceName, sequenceRatio } from "./text";
import type { PageDocument, Place } from "./types";

export function placeNames(place: Place): string[] {
  const names = [String(place.name_en ?? ""), String(place.name_ur ?? "")];
  if (Array.isArray(place.aliases)) {
    names.push(...place.aliases.filter((alias) => typeof alias === "string"));
  }
  return names;
}

function contains(haystack: string, needle: string): boolean {
  return ` ${haystack} `.includes(` ${needle} `);
}

export function matchPlaces(query: string, places: Place[]): Array<[number, Place]> {
  const normalizedQuery = normalizePlaceName(query);
  if (!normalizedQuery) return [];
  const exactMatches: Array<[number, Place]> = [];
  const fuzzyMatches: Array<[number, Place]> = [];
  const queryTokens = new Set(normalizedQuery.split(" "));
  for (const place of places) {
    let bestScore = 0;
    for (const name of placeNames(place)) {
      const normalizedName = normalizePlaceName(name);
      if (!normalizedName) continue;
      if (normalizedName === normalizedQuery || contains(normalizedQuery, normalizedName)) {
        bestScore = 1;
        break;
      }
      const nameTokens = normalizedName.split(" ");
      const overlap =
        new Set(nameTokens.filter((token) => queryTokens.has(token))).size /
        Math.max(1, nameTokens.length);
      bestScore = Math.max(bestScore, sequenceRatio(normalizedQuery, normalizedName), overlap * 0.9);
    }
    let score = bestScore;
    const district = normalizePlaceName(String(place.district ?? ""));
    if (district && contains(normalizedQuery, district)) score += 0.25;
    if (bestScore === 1) exactMatches.push([score, place]);
    else if (bestScore >= 0.86) fuzzyMatches.push([score, place]);
  }
  const matches = exactMatches.length ? exactMatches : fuzzyMatches;
  return [...matches].sort((first, second) => second[0] - first[0]);
}

export function findPlaceInText(text: string, places: Place[]): Place | null {
  const normalizedText = normalizePlaceName(text);
  if (!normalizedText) return null;
  const exactMatches: Place[] = [];
  const fuzzyMatches: Array<[number, Place]> = [];
  const textTokens = normalizedText.split(" ");
  for (const place of places) {
    let bestScore = 0;
    for (const name of placeNames(place)) {
      const normalizedName = normalizePlaceName(name);
      if (!normalizedName) continue;
      if (contains(normalizedText, normalizedName)) {
        bestScore = 1;
        break;
      }
      const nameTokens = normalizedName.split(" ");
      if (nameTokens.length > textTokens.length) continue;
      for (let start = 0; start + nameTokens.length <= textTokens.length; start += 1) {
        const window = textTokens.slice(start, start + nameTokens.length).join(" ");
        bestScore = Math.max(bestScore, sequenceRatio(window, normalizedName));
      }
    }
    if (bestScore === 1) exactMatches.push(place);
    else if (bestScore >= 0.88) fuzzyMatches.push([bestScore, place]);
  }
  if (exactMatches.length === 1) return exactMatches[0];
  if (exactMatches.length > 1) return null;
  fuzzyMatches.sort((first, second) => second[0] - first[0]);
  if (
    fuzzyMatches.length &&
    (fuzzyMatches.length === 1 || fuzzyMatches[0][0] - fuzzyMatches[1][0] >= 0.04)
  ) {
    return fuzzyMatches[0][1];
  }
  return null;
}

export function pageMentionsPlace(document: PageDocument, place: Place): boolean {
  const pageText = normalizePlaceName(`${document.title ?? ""} ${document.text ?? ""}`);
  return placeNames(place)
    .map(normalizePlaceName)
    .some((name) => name && contains(pageText, name));
}
