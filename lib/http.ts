import "server-only";

export const MAX_BODY_BYTES = 64 * 1024;

export function json(payload: unknown, status = 200): Response {
  return Response.json(payload, {
    status,
    headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" },
  });
}

/** Read a JSON request body with the same size limits as the Python server. */
export async function readJsonBody(
  request: Request,
  maxBytes = MAX_BODY_BYTES,
): Promise<{ body: unknown } | { error: Response }> {
  const declared = Number(request.headers.get("content-length") ?? "NaN");
  if (Number.isFinite(declared) && (declared < 1 || declared > maxBytes)) {
    return { error: json({ error: "Request is empty or too large." }, 413) };
  }
  const bytes = new Uint8Array(await request.arrayBuffer());
  if (bytes.byteLength < 1 || bytes.byteLength > maxBytes) {
    return { error: json({ error: "Request is empty or too large." }, 413) };
  }
  try {
    return { body: JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)) };
  } catch {
    return { error: json({ error: "Request must contain valid JSON." }, 400) };
  }
}
