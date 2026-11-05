import { findPlaceInText, matchPlaces } from "./places";
import { evidenceTokens } from "./corpus";
import {
  WORD_CHAR,
  collapseWhitespace,
  escapeRegExp,
  googleMapsSearchUrl,
  stripChars,
  tokenize,
  wordTokens,
} from "./text";
import type { ChatMessage, PageDocument, Place, Source } from "./types";

export const NO_ANSWER =
  "I couldn't find that information in the Punjab Tourism pages available " +
  "to me. Please check the official website or contact the department.";
export const BOT_IDENTITY_ANSWER =
  "I'm the Punjab Tourism Assistant. I answer questions using the tourism website pages.";
export const TOURISM_DEPARTMENT_ANSWER =
  "The Punjab Tourism Department works with affiliated departments to " +
  "promote, preserve, enhance, and harness tourism potential across Punjab.";
const GREETING_ANSWER =
  "Hi! I'm the Punjab Tourism Assistant. Ask me about places, heritage, " +
  "culture, or tourism activities in Punjab.";
const WELLBEING_ANSWER =
  "I'm doing well, thanks! I'm here to help with Punjab tourism. What would you like to know?";
const BOT_CONTEXT_ANSWER =
  "I'm a virtual Punjab Tourism Assistant, not a person with a physical " +
  "location. I answer questions using the Punjab Tourism website pages.";
const INVITATION_ANSWER =
  "Of course! Ask away. I can help with Punjab destinations, heritage, " +
  "culture, and information from the tourism website.";
export const UNVERIFIED_DETAILS_ANSWER =
  "I can't provide specific opening hours, prices, or distances without a source.";

const UNSOURCED_DETAIL_PATTERNS = [
  /\b(?:opening hours?|business hours?|entry fee|ticket price|price|cost|distance)\b|(?:کھلنے کے اوقات|اوقاتِ کار|قیمت|فیس|فاصلہ|کلومیٹر|روپے|بجے)|\b(?:fasla|qeemat|rupay|ticket price|opening time)\b/i,
  /\b(?:opening hours?|business hours?|open(?:s|ing)?|close(?:s|d)?)\b[^.!?\n]{0,80}\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\b/i,
  /\b(?:entry fee|ticket price|price|cost|fee)\b[^.!?\n]{0,50}\b\d[\d,]*(?:\.\d+)?\b/i,
  /(?:\b(?:PKR|Rs\.?|USD|EUR|GBP|CAD|AUD)\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:rupees?|PKR|USD|EUR|GBP)\b|[$€£]\s*\d[\d,]*(?:\.\d+)?)/i,
  /\b\d[\d,]*(?:\.\d+)?\s*(?:km|kilomet(?:er|re)s?|mi(?:le)?s?|meters?|metres?)\b/i,
  /\b(?:distance|away)\b[^.!?\n]{0,50}\b\d[\d,]*(?:\.\d+)?\b/i,
];

export function hasUnsourcedSpecificDetails(answer: string): boolean {
  return UNSOURCED_DETAIL_PATTERNS.some((pattern) => pattern.test(answer));
}

export function localAnswer(question: string): string | null {
  const normalized = question.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");
  if (
    [
      "whatareudoingrightnowandfromwhere",
      "whatareyoudoingrightnowandfromwhere",
      "whatareudoingrightnow",
      "whatareyoudoingrightnow",
      "whereareyoufrom",
      "whereareufrom",
      "whereareyou",
      "whereareu",
    ].includes(normalized)
  ) {
    return BOT_CONTEXT_ANSWER;
  }
  if (["whoareyou", "whoareu", "whoru", "whatsyourname", "whatcanyoudo"].includes(normalized)) {
    return BOT_IDENTITY_ANSWER;
  }
  if (["tumkyahu", "tumkyaho", "aapkyaho", "aapkonho", "apkonho"].includes(normalized)) {
    return (
      "Main Punjab Tourism Assistant hoon. Main Punjab Tourism website ki " +
      "maloomat ke mutabiq sawalon ke jawab deta hoon."
    );
  }
  if (["hi", "hello", "hey", "salam", "assalamualaikum"].includes(normalized)) return GREETING_ANSWER;
  if (["howareyou", "howareu", "howru"].includes(normalized)) return WELLBEING_ANSWER;
  if (
    ["caniaskyouanything", "caniaskyousomething", "caniaskasomething", "caniasomething"].includes(
      normalized,
    )
  ) {
    return INVITATION_ANSWER;
  }
  return null;
}

export function localCorpusAnswer(
  question: string,
  documents: PageDocument[],
): { answer: string; sources: Source[] } {
  const queryTerms = new Set(tokenize(question));
  const asksLocation =
    /\b(where|location|located|situated|kahan|kidhar|famous|known)\b/.test(question.toLowerCase());
  let best: { score: number; index: number; sentence: string } | null = null;

  documents.forEach((document, index) => {
    const text = String(document.text ?? "");
    const expanded: string[] = [];
    for (const raw of text.split(/(?<=[.!?])\s+|\n+/)) {
      const sentence = collapseWhitespace(raw);
      if (expanded.length && /^(?:this|it|they|these)\b/i.test(sentence)) {
        expanded[expanded.length - 1] += " " + sentence;
      } else {
        expanded.push(sentence);
      }
    }
    for (const sentence of expanded) {
      if (sentence.length < 30 || sentence.length > 700) continue;
      const sentenceTerms = new Set(tokenize(sentence));
      let matching = 0;
      for (const term of queryTerms) if (sentenceTerms.has(term)) matching += 1;
      if (!matching) continue;
      let score = matching / Math.max(1, queryTerms.size);
      if (asksLocation) {
        const folded = sentence.toLowerCase();
        if (/\b(near|south|north|east|west|km|kilomet(?:er|re)s?)\b/.test(folded)) score += 0.5;
        if (/\b(located|situated)\b/.test(folded)) score += 2;
      }
      if (!best || score > best.score) best = { score, index, sentence };
    }
  });

  if (!best) return { answer: NO_ANSWER, sources: [] };
  const { index, sentence } = best as { score: number; index: number; sentence: string };
  const document = documents[index];
  return {
    answer: sentence,
    sources: [{ title: String(document.title || document.url), url: String(document.url) }],
  };
}

export function isImageRequest(question: string): boolean {
  return /\b(?:image|images|photo|photos|picture|pictures|pic|pics|tasveer|tasveerein|tasveerain)\b|(?:تصویر|تصاویر|فوٹو|تصویریں)/.test(
    question.toLowerCase(),
  );
}

export function isPlaceOrMapQuestion(question: string): boolean {
  return /\b(?:where|location|located|situated|kahan|kidhar|map|maps|direction|directions|famous|known|why|kyun|kiun|address|coordinates?|latitude|longitude|google|pin|exact|naqsha|raasta|rasta|rastha|route|reach|pohanch)\b|(?:کہاں|کدھر|مقام|لوکیشن|نقشہ|راستہ|پتہ|ہدایات|عرض البلد|طول البلد)/.test(
    question.toLowerCase(),
  );
}

export function isDirectionsRequest(question: string): boolean {
  return /\b(?:direction|directions|route|how\s+to\s+reach|how\s+(?:do|can)\s+i\s+reach|get\s+to|raasta|rasta|rastha|pohanch(?:na|un)|kaise\s+ja\w+|raah)\b|(?:راستہ|ہدایات|کیسے\s+جائیں|کیسے\s+پہنچیں)/.test(
    question.toLowerCase(),
  );
}

export function routeIntent(question: string): string {
  if (isImageRequest(question)) return "image";
  if (isDirectionsRequest(question)) return "directions";
  if (isPlaceOrMapQuestion(question)) return "location";
  if (localAnswer(question) !== null) return "smalltalk";
  return "info";
}

export function contextualizePlaceQuestion(question: string, history: ChatMessage[]): string {
  if (
    !/\b(?:it|this|that|there|ye|yeh|is|us|wahan|yahan|isko|isjaga|iski|iske|uski|uske|iska|unka|unke|unki)\b|(?:اس|یہ|وہ|اسکا|اسکی|اسکے|اسے|وہاں|یہاں|اس کا|اس کی|ان کا|ان کی)/.test(
      question.toLowerCase(),
    )
  ) {
    return question;
  }
  const previous = [...history].reverse().find((message) => message.role === "user")?.content ?? "";
  return previous ? `${previous} ${question}`.trim() : question;
}

const PLACE_QUERY_PHRASES = [
  "could you please", "can you please", "google maps link",
  "exact location", "map location", "show me", "tell me",
  "where is", "where are", "located at", "located in", "directions to",
  "how to reach", "how do i reach", "how can i reach", "get directions",
  "find location", "find the location", "share location", "open map",
  "google maps", "location of", "map of", "map for", "address of",
  "give me", "please", "location", "located", "situated", "directions",
  "direction", "coordinates", "latitude", "longitude", "exact", "place",
  "map", "maps", "link", "url", "where", "kahan hai", "kidhar hai",
  "kahan", "kidhar", "jagah", "jaga", "jagha", "ka maqam", "ka pata",
  "ka naqsha", "ki location", "ka location", "naqsha dikhao",
  "naqsha dikhaye", "raasta batao", "raasta dikhao", "mujhe batao",
  "mujhe dikhao", "batao", "btao", "dikhao", "dikhaye", "jaga dikhao",
  "کہاں ہے", "کہاں", "کدھر ہے", "مقام", "لوکیشن", "نقشہ دکھائیں",
  "نقشہ دکھاؤ", "نقشہ", "راستہ بتائیں", "راستہ دکھائیں",
  "پتہ بتائیں", "مجھے بتائیں", "مجھے دکھائیں", "کی جگہ",
  "کا مقام", "کی لوکیشن", "کے بارے میں",
];
// Python's sorted(key=len, reverse=True) is stable, so equal-length phrases keep their order.
const SORTED_PHRASES = PLACE_QUERY_PHRASES.map((phrase, index) => ({ phrase, index }))
  .sort((first, second) => second.phrase.length - first.phrase.length || first.index - second.index)
  .map(({ phrase }) => new RegExp(`(?<!${WORD_CHAR})${escapeRegExp(phrase)}(?!${WORD_CHAR})`, "giu"));

const W = "[\\p{L}\\p{N}_'-]";
const REGION_AFTER = new RegExp(
  `(?:\\b(?:in|near|district|tehsil|city|of)\\s+${W}+(?:[\\s,]+(?:Punjab|Pakistan|[A-Z]${W}+)){0,3}|,\\s*${W}+(?:,\\s*${W}+){0,2})`,
  "iu",
);
const REGION_BEFORE = new RegExp(
  `(?:\\b(?:in|near|district|city|of)\\s+${W}+(?:[\\s,]+(?:Punjab|Pakistan)){0,2})\\s*$`,
  "iu",
);

/** Extract a place name while retaining its spelling and explicit region. */
export function extractPlaceQuery(question: string, places: Place[]): string | null {
  const text = question.replace(/https?:\/\/\S+/gi, " ");
  if (!text.trim()) return null;
  const matched = findPlaceInText(text, places);
  if (matched) {
    const aliases = [String(matched.name_en), String(matched.name_ur), ...matched.aliases.map(String)];
    aliases.sort((first, second) => second.length - first.length);
    for (const alias of aliases) {
      const match = new RegExp(`(?<!${WORD_CHAR})${escapeRegExp(alias)}(?!${WORD_CHAR})`, "iu").exec(text);
      if (!match) continue;
      const start = match.index;
      const end = start + match[0].length;
      const before = text.slice(Math.max(0, start - 48), start);
      const after = text.slice(end, end + 64);
      const regionAfter = REGION_AFTER.exec(after);
      const region = regionAfter ? regionAfter[0] : REGION_BEFORE.exec(before)?.[0] ?? "";
      return stripChars(collapseWhitespace(`${match[0]} ${region}`), " ,");
    }
  }

  let cleaned = text;
  for (const pattern of SORTED_PHRASES) cleaned = cleaned.replace(pattern, " ");
  const place = stripChars(collapseWhitespace(cleaned.replace(/[^\p{L}\p{N}_\s,'-]/gu, " ")), " ,");
  if (!place || !/[A-Za-z\u0080-￿]/.test(place)) return null;
  return place;
}

export function googleMapsSource(
  question: string,
  placeQuery: string | null,
  intent: string | null,
  places: Place[],
): Source | null {
  if (
    !(
      isPlaceOrMapQuestion(question) ||
      isDirectionsRequest(question) ||
      intent === "location" ||
      intent === "directions"
    )
  ) {
    return null;
  }
  const place = placeQuery || extractPlaceQuery(question, places);
  if (!place) return null;
  return { title: `Google Maps: ${place}`, url: googleMapsSearchUrl(place) };
}

export function isTourismDepartmentQuestion(question: string): boolean {
  const terms = new Set(wordTokens(question));
  const asksAboutRole = ["what", "whats", "who", "about", "does", "mean", "purpose", "role"].some((term) =>
    terms.has(term),
  );
  const asksAboutDepartment =
    terms.has("tourism") && ["department", "depart", "deprt", "dept"].some((term) => terms.has(term));
  return asksAboutRole && asksAboutDepartment;
}

export function answerIsGrounded(answer: string, sources: Source[], documents: PageDocument[]): boolean {
  const sourceUrls = new Set(
    sources.filter((source) => source && typeof source.url === "string").map((source) => source.url),
  );
  const cited = documents.filter((document) => sourceUrls.has(String(document.url ?? "")));
  if (!cited.length) return false;

  const evidence = new Set<string>();
  for (const document of cited) for (const token of evidenceTokens(document)) evidence.add(token);

  const claims = answer
    .split(/(?<=[.!?؟])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  for (const claim of claims) {
    const terms = new Set(tokenize(claim));
    if (!terms.size) continue;
    let overlap = 0;
    for (const term of terms) if (evidence.has(term)) overlap += 1;
    if (overlap < Math.min(2, terms.size) || overlap / terms.size < 0.5) return false;
  }
  return claims.length > 0;
}

export { matchPlaces };
