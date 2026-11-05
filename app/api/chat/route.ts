import { RequestError, answerChat } from "@/lib/chat";
import { json, readJsonBody } from "@/lib/http";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
// Provider calls can retry and fall back between models/providers.
export const maxDuration = 60;

export async function POST(request: Request) {
  const parsed = await readJsonBody(request);
  if ("error" in parsed) return parsed.error;
  try {
    return json(await answerChat(parsed.body));
  } catch (error) {
    if (error instanceof RequestError) return json({ error: error.message }, error.status);
    console.error("Chat request failed:", error instanceof Error ? error.name : "unknown error");
    return json({ error: "The chatbot could not answer right now." }, 500);
  }
}
