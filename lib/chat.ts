import "server-only";

import {
  NO_ANSWER,
  TOURISM_DEPARTMENT_ANSWER,
  UNVERIFIED_DETAILS_ANSWER,
  answerIsGrounded,
  contextualizePlaceQuestion,
  extractPlaceQuery,
  googleMapsSource,
  hasUnsourcedSpecificDetails,
  isTourismDepartmentQuestion,
  localAnswer,
  localCorpusAnswer,
  routeIntent,
} from "./answers";
import { responseCache, responseCacheKey } from "./cache";
import { getCorpus, getPlaces } from "./data";
import { tourismPageImages } from "./images";
import { lookupLocation } from "./location";
import { googleMapsLinkResponse, resolveGoogleMapsLink } from "./mapsLink";
import { findPlaceInText } from "./places";
import {
  ProviderError,
  callGemini,
  callGroq,
  configuredProviderOrder,
  providerConfig,
  type ProviderName,
} from "./providers";
import { googleMapsSearchUrl, openStreetMapUrl, urlencode } from "./text";
import type { ChatMessage, ChatResponse, PageDocument } from "./types";

export const MAX_QUESTION_LENGTH = 2_000;
export const MAX_HISTORY_MESSAGES = 10;

export class RequestError extends Error {
  constructor(message: string, readonly status = 400) {
    super(message);
  }
}

function titleCase(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function validateHistory(history: unknown): ChatMessage[] {
  if (history === undefined) return [];
  if (!Array.isArray(history) || history.length > MAX_HISTORY_MESSAGES) {
    throw new RequestError("Conversation history is invalid.");
  }
  return history.map((message) => {
    if (
      !message ||
      typeof message !== "object" ||
      !["user", "assistant"].includes(message.role) ||
      typeof message.content !== "string"
    ) {
      throw new RequestError("Conversation history is invalid.");
    }
    return { role: message.role, content: message.content.slice(0, MAX_QUESTION_LENGTH) };
  });
}

export async function answerChat(body: unknown): Promise<ChatResponse> {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new RequestError("Request must be a JSON object.");
  }
  const request = body as { question?: unknown; history?: unknown };
  if (typeof request.question !== "string" || !request.question.trim()) {
    throw new RequestError("Enter a question first.");
  }
  const question = request.question.trim();
  if (question.length > MAX_QUESTION_LENGTH) throw new RequestError("Question is too long.");

  const canned = localAnswer(question);
  if (canned !== null) return { answer: canned, sources: [] };

  const corpus = getCorpus();
  const places = getPlaces();

  if (isTourismDepartmentQuestion(question)) {
    const departmentPage = corpus
      .search("Tourism Department", 5)
      .find((document) => String(document.title ?? "").toLowerCase().includes("department"));
    if (departmentPage) {
      return {
        answer: TOURISM_DEPARTMENT_ANSWER,
        sources: [{ title: String(departmentPage.title || departmentPage.url), url: String(departmentPage.url) }],
      };
    }
  }

  const history = validateHistory(request.history);
  const contextualQuestion = contextualizePlaceQuestion(question, history);

  const mapLink = await resolveGoogleMapsLink(question);
  if (mapLink !== null) return googleMapsLinkResponse(mapLink, places);

  const intent = routeIntent(contextualQuestion);
  const wantsImages = intent === "image";
  const identifiedPlace = findPlaceInText(contextualQuestion, places);
  let searchQuery = contextualQuestion;
  if (wantsImages) {
    const previousUserQuestion =
      [...history].reverse().find((message) => message.role === "user")?.content ?? "";
    const refersToPrevious =
      /\b(?:it|that|this|iska|iski|iske|uska|uski|uske)\b/.test(question.toLowerCase()) ||
      /(?:اس|یہ|وہ|اسکا|اسکی|اسکے|اسے|وہاں|یہاں)/.test(question);
    if (refersToPrevious && previousUserQuestion) searchQuery = previousUserQuestion;
  }

  let matchedDocuments: PageDocument[] = corpus.search(searchQuery, 5);
  if (identifiedPlace) {
    const placePage = corpus.documents.find((document) => document.url === identifiedPlace.source_url);
    matchedDocuments = placePage ? [placePage] : [];
  } else if (!corpus.hasEvidence(searchQuery, matchedDocuments)) {
    matchedDocuments = [];
  }

  let mapSource = googleMapsSource(
    contextualQuestion,
    extractPlaceQuery(contextualQuestion, places),
    intent,
    places,
  );
  if (mapSource && identifiedPlace) {
    mapSource = {
      title: `Google Maps: ${identifiedPlace.name_en}`,
      url: googleMapsSearchUrl(identifiedPlace.name_en),
    };
  }

  let locationPromise: ReturnType<typeof lookupLocation> | null = null;
  if (mapSource && !wantsImages) {
    const place = identifiedPlace ? identifiedPlace.name_en : mapSource.title.replace(/^Google Maps: /, "");
    locationPromise = lookupLocation(place, places);
  }

  const config = providerConfig();
  const providerOrder = configuredProviderOrder(config);

  let cacheKey: string | null = null;
  if (!wantsImages && !mapSource) {
    cacheKey = responseCacheKey(
      contextualQuestion,
      history,
      matchedDocuments,
      Object.fromEntries(providerOrder.map((name) => [name, config.models[name]])),
    );
    const cached = responseCache.get(cacheKey);
    if (cached) return cached;
  }

  const pageImages = wantsImages ? tourismPageImages(matchedDocuments, 3, identifiedPlace, places) : [];
  if (wantsImages && pageImages.length) {
    return {
      answer: `Here are pictures from the official Punjab Tourism website for ${pageImages[0].place_name}.`,
      sources: [
        {
          title: `${String(matchedDocuments[0].title)} (Punjab Tourism website)`,
          url: String(matchedDocuments[0].url),
        },
      ],
      images: pageImages,
    };
  }
  if (wantsImages && identifiedPlace) {
    const placeSourceUrl = String(identifiedPlace.source_url || "");
    return {
      answer:
        `No indexed image for ${identifiedPlace.name_en} was found on the Punjab Tourism ` +
        "website. I did not substitute images from another place.",
      sources: placeSourceUrl ? [{ title: "Punjab Tourism place page", url: placeSourceUrl }] : [],
      images: [],
    };
  }
  if (wantsImages) {
    return {
      answer:
        "I couldn't identify a matching place image in the Punjab Tourism index. " +
        "No unrelated images were substituted.",
      sources: [],
      images: [],
      answer_origin: "Punjab Tourism image index",
    };
  }

  let result: ChatResponse | null = null;
  let providerUsed: ProviderName | null = null;
  const failedProviders: ProviderName[] = [];
  let quotaLimited = false;
  for (const providerName of providerOrder) {
    const call = providerName === "gemini" ? callGemini : callGroq;
    try {
      result = await call(question, history, matchedDocuments, config.keys[providerName], config.models[providerName]);
    } catch (error) {
      const rateLimited = error instanceof ProviderError && error.rateLimited;
      console.error(
        `Provider ${providerName} failed (${rateLimited ? "rate limited" : "request failed"}); ` +
          "trying the next configured provider.",
      );
      failedProviders.push(providerName);
      quotaLimited = quotaLimited || rateLimited;
      continue;
    }
    providerUsed = providerName;
    break;
  }

  let usedLocalFallback = result === null;
  if (result === null) {
    result = matchedDocuments.length
      ? localCorpusAnswer(contextualQuestion, matchedDocuments)
      : { answer: NO_ANSWER, sources: [] };
  }
  const groundingRejected = Boolean(
    providerUsed &&
      matchedDocuments.length &&
      result.sources?.length &&
      !answerIsGrounded(String(result.answer ?? ""), result.sources, matchedDocuments),
  );
  if (groundingRejected) {
    result = localCorpusAnswer(contextualQuestion, matchedDocuments);
    usedLocalFallback = true;
  }

  const location = locationPromise ? await locationPromise : undefined;
  if (location?.status === "found" && location.latitude !== undefined && location.longitude !== undefined) {
    if (mapSource) mapSource.url = googleMapsSearchUrl(`${location.latitude},${location.longitude}`);
    location.sources = [
      {
        title: "OpenStreetMap location (via Photon; outside Punjab Tourism website)",
        url: openStreetMapUrl(location.latitude, location.longitude),
      },
    ];
  }

  if (pageImages.length) result.images = pageImages;
  if (mapSource) result.maps = [mapSource];
  if (location) result.location = location;

  let unverifiedGeneralAnswer = false;
  if (groundingRejected) {
    result.answer_origin = "grounded local excerpt";
    result.source_note =
      "Generated details could not all be verified in the cited pages, so this " +
      "answer was replaced with a source excerpt.";
  } else if (usedLocalFallback) {
    result.answer_origin = "local fallback";
    let reason: string;
    if (providerOrder.length) {
      reason = quotaLimited
        ? "An AI provider quota or rate limit was reached."
        : "The configured AI providers were unavailable.";
    } else {
      reason = "No AI provider key is configured.";
    }
    result.source_note = matchedDocuments.length
      ? `${reason} This answer uses local tourism pages only.`
      : `${reason} No matching local tourism evidence was found.`;
  } else if (result.sources?.length) {
    const hasTourismSource = result.sources.some((source) =>
      String(source?.url ?? "").startsWith("https://tourism.punjab.gov.pk/"),
    );
    result.answer_origin = hasTourismSource ? "Punjab Tourism website" : "OpenStreetMap / Photon";
  } else if (matchedDocuments.length) {
    result.answer_origin = "local fallback";
  } else if (result.answer === NO_ANSWER) {
    result.answer_origin = "no matching Tourism website evidence";
    result.source_note = "I could not find supporting information in the Punjab Tourism website.";
    result.external_search_url = "https://www.google.com/search?" + urlencode({ q: question });
  } else {
    result.answer_origin = "external general knowledge";
    unverifiedGeneralAnswer = true;
    if (hasUnsourcedSpecificDetails(String(result.answer ?? ""))) result.answer = UNVERIFIED_DETAILS_ANSWER;
    result.source_note =
      "Unverified: This answer is not supported by the Punjab Tourism website, and no " +
      "independent source was checked. Please verify it before relying on it; opening " +
      "hours, prices, and distances will not be stated unless a source is cited.";
    result.external_search_url = "https://www.google.com/search?" + urlencode({ q: question });
  }
  if (providerUsed && failedProviders.length && !groundingRejected && !unverifiedGeneralAnswer) {
    const failedNames = failedProviders.map(titleCase).join(" and ");
    const reason = quotaLimited ? "hit a quota or rate limit" : "was unavailable";
    result.source_note = `${failedNames} ${reason}; this answer was generated by ${titleCase(providerUsed)}.`;
  }
  if (location?.sources?.length) {
    result.sources = [...(result.sources ?? []), ...location.sources];
  }
  if (cacheKey !== null) responseCache.set(cacheKey, result);
  return result;
}
