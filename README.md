# Punjab Tourism Assistant (Next.js)

The chatbot web app is a Next.js (App Router, TypeScript) project that deploys
to Vercel. It answers questions from `data/tourism_pages.jsonl` using
title/phrase-weighted BM25 retrieval, Gemini (with Groq and local-excerpt
fallbacks), Photon geocoding, Google Maps link parsing, a trip planner, and
image matching against the indexed tourism-site images.

## Run locally

Requires Node.js 20 or later.

```powershell
npm install
Copy-Item .\.env.example .\.env.local   # then add your GEMINI_API_KEY
npm run dev                             # http://localhost:3000
```

Production check: `npm run build` then `npm start`.

## Deploy to Vercel

1. Push this folder to a Git repository and import it in Vercel (framework
   preset: **Next.js**; no build settings need changing). Or run `npx vercel`.
2. In **Project -> Settings -> Environment Variables**, add `GEMINI_API_KEY`
   (and optionally `AI_PROVIDER`, `GEMINI_MODEL`, `GROQ_API_KEY`,
   `GROQ_MODEL`). Redeploy after changing them.
3. Keep `.env`/`.env.local` out of Git; `.gitignore` and `.vercelignore`
   already exclude them.

The corpus (`data/tourism_pages.jsonl`) and place registry (`data/places.json`)
are bundled with the API functions. To update content, regenerate them with
the Python tools below and redeploy.

## Project layout

| Path | Responsibility |
|---|---|
| `app/page.tsx`, `components/` | Chat UI, Leaflet maps, nearby places, trip planner, image upload |
| `app/api/*/route.ts` | `chat`, `image`, `itinerary`, `places`, `health`, `diagnostics` endpoints |
| `lib/` | Retrieval, answer grounding, providers, geocoding, map links, itinerary, image ranking |
| `components/clipEmbedding.ts`, `components/clipPreprocess.ts` | In-browser CLIP image embedding |
| `data/` | Corpus and place registry used at runtime |
| `*.py`, `tests/` | Offline Python tools: scraper, image indexer, retrieval evaluation, original server |

### Differences from the original Python server

- **Image matching runs in the browser.** The 89 MB CLIP model is too large
  for a Vercel function, so the browser downloads the same pinned, quantized
  model from Hugging Face on first use (cached afterwards), reproduces the
  Pillow preprocessing exactly, and sends only the 512-number embedding to
  `POST /api/image`. The uploaded image never leaves the device.
  `/api/image` now accepts `{"embedding": [...]}` JSON instead of a
  multipart upload.
- **Optional hybrid (embedding) retrieval is not deployed**; BM25 (the
  original default) is used.
- The 60-second response cache is per serverless instance.

---

# Python tooling and original server

Project overview, audit findings, operating instructions, data/privacy notes,
and proposed enhancements are documented in [`docs/`](docs/README.md).

## Run the chatbot

The local Python chatbot reads `data\tourism_pages.jsonl` directly and searches
those page records with title/phrase-weighted BM25 and a confidence threshold.
Optional local multilingual ONNX embeddings are fused with BM25 using
reciprocal rank fusion. Tourism answers are grounded in retrieved pages with
links to their sources; generated claims that fail grounding are replaced by a
source excerpt. Unrelated general-knowledge answers are labeled as outside the
official government source pages and are not independently verified.
If the AI provider is unavailable or no API key is configured, the chatbot
tries Groq if configured, then answers with the best directly relevant
sentence from the local pages and includes its source link; if nothing matches,
it says so without returning a provider error. The UI identifies local fallback
and rate-limit states.
Common greetings, "who are you", "how are you", and "can I ask you something?"
questions are answered locally without calling an AI provider. Provider errors
are kept out of visitor replies and API-key-like values are redacted in terminal
diagnostics. A 403 also
prompts the administrator to verify model access and rotate any shared key.

Python 3.10 or later is recommended. Copy `.env.example` to `.env`, add your
Gemini API key, and keep the file private (it is ignored by Git):

```powershell
# If .env does not already exist:
Copy-Item .\.env.example .\.env
# Add these settings to .env, then put your key after the equals sign:
# AI_PROVIDER=gemini
# GEMINI_API_KEY=your-gemini-api-key
python .\chatbot.py
```

The chatbot uses Python's standard library; Pillow is required for local image
matching:

```powershell
python -m pip install -r .\requirements.txt
```

The chat page can match an uploaded JPEG, PNG, WebP, or GIF against the
locally indexed Punjab Tourism page images. Quantized CLIP image and text
models compare image content and landmark labels locally; uploads are not
sent to Gemini. The pinned model assets (about 150 MiB total) download
automatically on the first image-index or matching request. Add CLIP
embeddings to an existing page index after getting
authorization to collect the site's images:

```powershell
python .\index_tourism_images.py --allow-robots-unavailable
```

`--allow-robots-unavailable` is only for site administrators or people with
explicit site-owner authorization. The crawler stays on the Punjab Tourism
host, limits image downloads to 10 MiB each, and respects available robots
rules. To rebuild the HTML and image list from scratch instead, run
`python .\scrape_tourism.py --allow-robots-unavailable --max-pages 200`.

Open `http://127.0.0.1:8000/` or start with `.\run.bat`, which keeps the console
open and prints the app and health URLs. Use `Ctrl+C` in the terminal to stop
the server.
Gemini is the default provider and the model is read from `GEMINI_MODEL` in
`.env` (default: `gemini-3.8-flash`). You can choose a different provider or
model with `AI_PROVIDER`, `GEMINI_MODEL`, or the `--provider` / `--model`
options. On Gemini HTTP 503 overload responses, the chatbot retries and
automatically falls back to `gemini-3.6-flash`, then `gemini-3.5-flash-lite`.
On provider failure, configured Groq is tried before local corpus fallback.
Restart the chatbot after changing its code or settings. To prefer Groq, set
`AI_PROVIDER=groq` and `GROQ_API_KEY`. `GET /api/diagnostics` reports provider
and Photon network reachability/latency without exposing keys or visitor text;
it does not verify provider quota or credentials.

Hybrid retrieval is off by default. To enable it, set `HYBRID_SEARCH=1` in
`.env` and restart. The first hybrid search downloads a pinned, quantized
multilingual E5 ONNX model and tokenizer (about 135 MB, roughly 129 MiB total);
embeddings run locally, and an unavailable model falls back to BM25. Evaluate
the checked-in 40-question set with:

```powershell
python .\evaluate_retrieval.py
# Optional model-enabled evaluation; may download the pinned assets:
python .\evaluate_retrieval.py --hybrid
```

Run the chatbot tests with:

```powershell
python -m unittest discover -s .\tests -v
```

The chatbot API sends the visitor's question, recent conversation, and selected
page excerpts to the configured AI provider for generation. The full JSONL
corpus stays on the local server. Avoid sending personal or sensitive
information in visitor questions.

## Refresh the source data

The multi-source importer reads official URLs from `siteslink.md`, crawls each
HTTPS host independently, and adds English and Urdu Wikipedia background
articles as secondary sources. TDCP uses rendered Chromium pages because its
public pages are a JavaScript app without server-rendered text. Install the
optional browser dependencies once:

```powershell
python -m venv .venv                 # only if .venv does not exist yet
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements-scraping.txt
python -m playwright install chromium
```

Run the importer from that activated environment so its TDCP subprocess uses
the same Python installation. A single incomplete page response is retried up
to three times; if it still fails, that page is skipped and the rest of the
site crawl continues.

Official records carry a source name and priority (the order
listed in `siteslink.md`); Wikipedia records are marked secondary and the
chatbot favors official evidence when both kinds are relevant. Each site's
`robots.txt` is checked; a missing or blocked policy file stops that site's
crawl. Unvisited records stay in the corpus when a page cap is used, and
records from unreachable sources are retained.

Run a full refresh with:

```powershell
python .\ingest_sources.py --no-proxy --max-pages-per-site 100
```

`--only-host tdcp.gop.pk` refreshes TDCP alone; `--skip-official` refreshes
Wikipedia only. `--no-proxy` bypasses HTTP proxy environment settings for this
run.

If a site's `robots.txt` returns 403 or an HTML error page, it is skipped by
default. Only run `--allow-robots-unavailable` when you have site-owner
authorization for those sources; the importer still obeys any valid policy.

The importer does not download images from these additional sites, so image
matching remains limited to image records already in the corpus. Use
`--skip-wikipedia` to refresh only the listed official domains. The standalone
`scrape_tourism.py` command below remains available for a single-site crawl.
The provided DTS link is HTTP-only from this environment; those records are
marked as HTTP in their metadata and should be checked against another official
source before relying on changing details.

This Python scraper collects public HTML pages from
`https://tourism.punjab.gov.pk/` and writes one chatbot-ready JSON document per
line. It stays on the configured HTTPS host, follows `robots.txt`, uses a
one-second default delay, limits response sizes, and does not submit forms or
download documents. It can index same-host raster images for image matching
(up to 10 MiB each).

## Run

Python 3.10 or later is recommended. From this folder:

```powershell
python .\scrape_tourism.py --max-pages 200
```

The output is `data\tourism_pages.jsonl`. Each JSONL row contains `url`,
`canonical_url`, `title`, `description`,
`headings`, `text`, page/image CLIP embeddings, `images` (same-site image URL,
alt text, and visual hash), `content_sha256`, and `crawled_at`.

This standalone command replaces `data\tourism_pages.jsonl` with the Punjab
Tourism site crawl. If you have already built the multi-site corpus with
`ingest_sources.py`, do not run this command against the default output path;
it would replace the other sites' records. For a test crawl, use a separate
output such as `--output .\data\sample.jsonl`. Use `ingest_sources.py` to
refresh the merged multi-site corpus.

The site currently returns HTTP 403 for `/robots.txt`. By default, the scraper
stops rather than guessing the site's crawling policy. If you administer this
site and have government/site-owner authorization to collect its public pages,
explicitly allow that unavailable policy file:

```powershell
python .\scrape_tourism.py --allow-robots-unavailable --max-pages 200
```

If `robots.txt` becomes available and disallows a page, the scraper will still
skip that page. Other useful options:

```powershell
python .\scrape_tourism.py --allow-robots-unavailable --max-pages 50 --delay 2 --output .\data\sample.jsonl
```

Keep the crawl limit and delay conservative, and coordinate repeated or
production crawls with the site administrator.

## Load the data

Each line is a separate JSON object, so it can be streamed without loading the
whole dataset into memory:

```python
import json

with open("data/tourism_pages.jsonl", encoding="utf-8") as pages_file:
    for line in pages_file:
        page = json.loads(line)
        print(page["title"], page["url"], page["text"][:300])
```

## Test

```powershell
python -m unittest discover -s .\tests -v
```
