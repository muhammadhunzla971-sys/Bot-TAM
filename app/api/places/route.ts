import { publicPlaces } from "@/lib/data";
import { json } from "@/lib/http";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export function GET() {
  return json({ places: publicPlaces() });
}
