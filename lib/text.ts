/**
 * Text helpers ported from chatbot.py.
 *
 * Python's `\w` (with re.UNICODE) matches characters for which str.isalnum()
 * is true plus "_". The closest JavaScript equivalent is [\p{L}\p{N}_] with
 * the `u` flag. JavaScript's `\b` is ASCII-only, so code that must respect
 * Urdu word boundaries uses explicit lookarounds built from WORD_CHAR.
 */

export const WORD_CHAR = "[\\p{L}\\p{N}_]";
const TOKEN_PATTERN = /[\p{L}\p{N}_]+/gu;

export const STOP_WORDS = new Set([
  "a", "about", "an", "and", "are", "as", "at", "be", "by", "can",
  "do", "for", "from", "how", "i", "in", "is", "it", "me", "of", "on",
  "or", "please", "tell", "the", "this", "to", "what", "where", "which",
  "who", "with", "you", "your", "ka", "ki", "ko", "kya", "mein", "hai",
  "hain", "kaise", "ke", "kiya", "koi", "kahan", "mujhe", "par", "se",
  "page", "pages", "website", "site", "visitors", "visitor", "listed",
  "mention", "mentions", "bary", "baray", "bare", "k", "batao", "bataein",
  "bataen", "maloomat", "tafseel", "kis", "liye",
]);

/** Approximation of Python's str.casefold(). */
export function casefold(text: string): string {
  return text.toLowerCase();
}

/** All `\w+` runs in the case-folded text (no stop-word filtering). */
export function wordTokens(text: string): string[] {
  return casefold(text).match(TOKEN_PATTERN) ?? [];
}

export function tokenize(text: string): string[] {
  return wordTokens(text).filter(
    (token) => !STOP_WORDS.has(token) && Array.from(token).length > 1,
  );
}

/** Python `" ".join(value.split())`. */
export function collapseWhitespace(value: string): string {
  return value.split(/\s+/).filter(Boolean).join(" ");
}

export function normalizePlaceName(value: string): string {
  return wordTokens(value).join(" ");
}

export function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function isUrdu(text: string): boolean {
  return /[؀-ۿ]/.test(text);
}

/** Strip the given characters from both ends (Python str.strip(chars)). */
export function stripChars(value: string, chars: string): string {
  let start = 0;
  let end = value.length;
  while (start < end && chars.includes(value[start])) start += 1;
  while (end > start && chars.includes(value[end - 1])) end -= 1;
  return value.slice(start, end);
}

export function containsTokenPhrase(tokens: string[], phrase: string[]): boolean {
  if (!phrase.length) return false;
  const width = phrase.length;
  for (let index = 0; index + width <= tokens.length; index += 1) {
    let matched = true;
    for (let offset = 0; offset < width; offset += 1) {
      if (tokens[index + offset] !== phrase[offset]) {
        matched = false;
        break;
      }
    }
    if (matched) return true;
  }
  return false;
}

/**
 * Port of difflib.SequenceMatcher(None, a, b).ratio(), including the
 * "autojunk" popular-element heuristic for sequences of 200+ items.
 */
export function sequenceRatio(first: string, second: string): number {
  const a = Array.from(first);
  const b = Array.from(second);
  const total = a.length + b.length;
  if (!total) return 1;

  const b2j = new Map<string, number[]>();
  b.forEach((element, index) => {
    const indices = b2j.get(element);
    if (indices) indices.push(index);
    else b2j.set(element, [index]);
  });
  if (b.length >= 200) {
    const limit = Math.floor(b.length / 100) + 1;
    for (const [element, indices] of b2j) {
      if (indices.length > limit) b2j.delete(element);
    }
  }

  const findLongestMatch = (alo: number, ahi: number, blo: number, bhi: number) => {
    let bestI = alo;
    let bestJ = blo;
    let bestSize = 0;
    let j2len = new Map<number, number>();
    for (let i = alo; i < ahi; i += 1) {
      const next = new Map<number, number>();
      for (const j of b2j.get(a[i]) ?? []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) ?? 0) + 1;
        next.set(j, k);
        if (k > bestSize) {
          bestI = i - k + 1;
          bestJ = j - k + 1;
          bestSize = k;
        }
      }
      j2len = next;
    }
    while (bestI > alo && bestJ > blo && a[bestI - 1] === b[bestJ - 1]) {
      bestI -= 1;
      bestJ -= 1;
      bestSize += 1;
    }
    while (
      bestI + bestSize < ahi &&
      bestJ + bestSize < bhi &&
      a[bestI + bestSize] === b[bestJ + bestSize]
    ) {
      bestSize += 1;
    }
    return [bestI, bestJ, bestSize] as const;
  };

  let matches = 0;
  const queue: Array<[number, number, number, number]> = [[0, a.length, 0, b.length]];
  while (queue.length) {
    const [alo, ahi, blo, bhi] = queue.pop()!;
    const [i, j, k] = findLongestMatch(alo, ahi, blo, bhi);
    if (!k) continue;
    matches += k;
    if (alo < i && blo < j) queue.push([alo, i, blo, j]);
    if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
  }
  return (2 * matches) / total;
}

/** Python urllib.parse.urlencode for a flat object. */
export function urlencode(params: Record<string, string | number>): string {
  return Object.entries(params)
    .map(
      ([key, value]) =>
        `${quotePlus(key)}=${quotePlus(String(value))}`,
    )
    .join("&");
}

function quotePlus(value: string): string {
  return encodeURIComponent(value)
    .replace(/[!'()*~]/g, (character) =>
      character === "~" ? "~" : `%${character.charCodeAt(0).toString(16).toUpperCase()}`,
    )
    .replace(/%20/g, "+");
}

export function googleMapsSearchUrl(query: string): string {
  return "https://www.google.com/maps/search/?" + urlencode({ api: "1", query });
}

/** Python's repr-style float formatting for coordinates (e.g. 31.5 -> "31.5"). */
export function formatNumber(value: number): string {
  return Number.isInteger(value) ? value.toFixed(1) : String(value);
}

export function openStreetMapUrl(latitude: number, longitude: number): string {
  const lat = formatNumber(latitude);
  const lng = formatNumber(longitude);
  return (
    "https://www.openstreetmap.org/?" +
    urlencode({ mlat: lat, mlon: lng }) +
    `#map=17/${lat}/${lng}`
  );
}
