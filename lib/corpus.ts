import { containsTokenPhrase, tokenize } from "./text";
import type { PageDocument } from "./types";

export const RETRIEVAL_CONFIDENCE_THRESHOLD = 0.4;

export function headingText(document: PageDocument): string {
  const headings = document.headings;
  if (!Array.isArray(headings)) return "";
  return headings
    .filter((item) => item && typeof item === "object" && typeof item.text === "string")
    .map((item) => item.text as string)
    .join(" ");
}

function countTokens(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);
  return counts;
}

/** Cache of the full evidence token list per document (title/desc/headings/body). */
const evidenceTokenCache = new WeakMap<PageDocument, string[]>();

export function evidenceTokens(document: PageDocument): string[] {
  let tokens = evidenceTokenCache.get(document);
  if (!tokens) {
    tokens = tokenize(
      `${document.title ?? ""} ${document.description ?? ""} ${headingText(document)} ${document.text ?? ""}`,
    );
    evidenceTokenCache.set(document, tokens);
  }
  return tokens;
}

const phraseTokenCache = new WeakMap<PageDocument, { titleDesc: string[]; heading: string[]; body: string[] }>();

function phraseTokens(document: PageDocument) {
  let cached = phraseTokenCache.get(document);
  if (!cached) {
    cached = {
      titleDesc: tokenize(`${document.title ?? ""} ${document.description ?? ""}`),
      heading: tokenize(headingText(document)),
      body: tokenize(document.text ?? ""),
    };
    phraseTokenCache.set(document, cached);
  }
  return cached;
}

/** Port of has_corpus_evidence (BM25-only; hybrid retrieval is not deployed). */
export function hasCorpusEvidence(question: string, documents: PageDocument[]): boolean {
  const queryTokens = tokenize(question);
  if (!queryTokens.length) return false;
  const querySet = new Set(queryTokens);
  for (const document of documents) {
    const tokens = new Set(evidenceTokens(document));
    let matched = 0;
    for (const token of querySet) if (tokens.has(token)) matched += 1;
    let confidence = matched / querySet.size;
    const phrases = phraseTokens(document);
    if (containsTokenPhrase(phrases.titleDesc, queryTokens)) {
      confidence = Math.max(confidence, 1.0);
    } else if (containsTokenPhrase(phrases.heading, queryTokens)) {
      confidence = Math.max(confidence, 0.85);
    } else if (containsTokenPhrase(phrases.body, queryTokens)) {
      confidence = Math.max(confidence, 0.7);
    }
    if (confidence >= RETRIEVAL_CONFIDENCE_THRESHOLD) return true;
  }
  return false;
}

/** Read-only corpus with title-aware BM25-style ranking (port of chatbot.Corpus). */
export class Corpus {
  readonly documents: PageDocument[];
  private readonly tokens: Map<string, number>[];
  private readonly titleTokens: Map<string, number>[];
  private readonly documentFrequency = new Map<string, number>();
  private readonly titleTextTokens: string[][];
  private readonly headingTextTokens: string[][];
  private readonly bodyTextTokens: string[][];

  constructor(documents: PageDocument[]) {
    this.documents = documents;
    this.tokens = [];
    this.titleTokens = [];
    this.titleTextTokens = [];
    this.headingTextTokens = [];
    this.bodyTextTokens = [];
    for (const document of documents) {
      const heading = headingText(document);
      const titleText = [document.title ?? "", document.description ?? ""]
        .map((value) => String(value))
        .join(" ");
      const allTokens = countTokens(tokenize(`${titleText} ${document.text}`));
      this.tokens.push(allTokens);
      this.titleTokens.push(countTokens(tokenize(`${titleText} ${heading}`)));
      for (const term of allTokens.keys()) {
        this.documentFrequency.set(term, (this.documentFrequency.get(term) ?? 0) + 1);
      }
      this.titleTextTokens.push(tokenize(titleText));
      this.headingTextTokens.push(tokenize(heading));
      this.bodyTextTokens.push(tokenize(document.text));
    }
  }

  get retrievalMode(): string {
    return "bm25";
  }

  hasEvidence(query: string, documents: PageDocument[]): boolean {
    return hasCorpusEvidence(query, documents);
  }

  search(query: string, limit = 5): PageDocument[] {
    const queryTokens = tokenize(query);
    const terms = new Set(queryTokens);
    if (!terms.size) return [];

    const total = this.documents.length;
    const ranked: Array<[number, number]> = [];
    this.tokens.forEach((counts, index) => {
      const document = this.documents[index];
      let score = 0;
      for (const term of terms) {
        const frequency = counts.get(term) ?? 0;
        if (!frequency) continue;
        const df = this.documentFrequency.get(term) ?? 0;
        const inverseFrequency = Math.log(1 + (total - df + 0.5) / (df + 0.5));
        const titleFrequency = this.titleTokens[index].get(term) ?? 0;
        score += inverseFrequency * (1 + Math.log(frequency) + 2 * titleFrequency);
      }
      if (containsTokenPhrase(this.titleTextTokens[index], queryTokens)) score += 10;
      else if (containsTokenPhrase(this.headingTextTokens[index], queryTokens)) score += 6;
      else if (containsTokenPhrase(this.bodyTextTokens[index], queryTokens)) score += 4;
      const priority = Number(document.source_priority ?? 1);
      if (document.source_group !== "secondary" && Number.isFinite(priority)) {
        score *= 1 + Math.max(0, 5 - priority) * 0.05;
      }
      if (score) ranked.push([score, index]);
    });
    // Stable sort, highest score first (matches Python's list.sort(reverse=True)).
    ranked.sort((first, second) => second[0] - first[0]);
    return ranked.slice(0, Math.max(1, limit)).map(([, index]) => this.documents[index]);
  }
}
