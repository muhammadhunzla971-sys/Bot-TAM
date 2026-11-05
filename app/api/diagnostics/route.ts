import { json } from "@/lib/http";
import { PROVIDER_PROBE_URLS, probeEndpoint, providerConfig, type ProviderName } from "@/lib/providers";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const config = providerConfig();
  const names: ProviderName[] = ["gemini", "groq"];
  const [providerResults, geocoder] = await Promise.all([
    Promise.all(
      names.map(async (name) =>
        config.keys[name]
          ? { configured: true, ...(await probeEndpoint(PROVIDER_PROBE_URLS[name])) }
          : { configured: false, reachable: false, latency_ms: null },
      ),
    ),
    probeEndpoint("https://photon.komoot.io/"),
  ]);
  return json({
    providers: Object.fromEntries(names.map((name, index) => [name, providerResults[index]])),
    geocoder: { service: "Photon", ...geocoder },
  });
}
