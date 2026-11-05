import { createHash } from "node:crypto";

import { wordTokens } from "./text";
import type { ChatMessage, ChatResponse, PageDocument } from "./types";

const MAX_ENTRIES = 128;
const TTL_MS = 60_000;

/**
 * Small in-memory LRU response cache. On Vercel each warm function instance
 * keeps its own copy, so this only deduplicates bursts of identical questions.
 */
class ResponseCache {
  private entries = new Map<string, { expiresAt: number; value: string }>();

  get(key: string): ChatResponse | null {
    const entry = this.entries.get(key);
    if (!entry) return null;
    if (entry.expiresAt <= Date.now()) {
      this.entries.delete(key);
      return null;
    }
    this.entries.delete(key);
    this.entries.set(key, entry);
    return JSON.parse(entry.value) as ChatResponse;
  }

  set(key: string, value: ChatResponse): void {
    const now = Date.now();
    for (const [existingKey, entry] of this.entries) {
      if (entry.expiresAt <= now) this.entries.delete(existingKey);
    }
    this.entries.delete(key);
    this.entries.set(key, { expiresAt: now + TTL_MS, value: JSON.stringify(value) });
    while (this.entries.size > MAX_ENTRIES) {
      const oldest = this.entries.keys().next().value as string;
      this.entries.delete(oldest);
    }
  }
}

export const responseCache = new ResponseCache();

export function responseCacheKey(
  question: string,
  history: ChatMessage[],
  documents: PageDocument[],
  providerModels: Record<string, string>,
): string {
  const material = {
    history: history.map((message) => [message.role, wordTokens(message.content).join(" ")]),
    providers: Object.entries(providerModels).sort(([first], [second]) => first.localeCompare(second)),
    question: wordTokens(question).join(" "),
    source_ids: documents.map((document) => String(document.url ?? "")),
  };
  return createHash("sha256").update(JSON.stringify(material)).digest("hex");
}
