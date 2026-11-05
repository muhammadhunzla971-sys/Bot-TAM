# Operations and troubleshooting

## Requirements

- Windows PowerShell and Python 3.10 or later (the current environment uses
  Python 3.13).
- Internet access for the configured AI provider, Photon geocoding, map tiles,
  Leaflet CDN, allowlisted Google Maps short-link redirects, and first-time
  model downloads.
- `requirements.txt` dependencies for image-matching features.

## Install and start (Windows)

From the project directory:

```powershell
python -m pip install -r .\requirements.txt
Copy-Item .\.env.example .\.env
```

Edit `.env` and set a valid provider key. The example placeholder is not a
working key:

```dotenv
AI_PROVIDER=gemini
GEMINI_API_KEY=your-real-key
GEMINI_MODEL=gemini-3.8-flash
```

Start the server in a terminal:

```powershell
python .\chatbot.py
```

Or run `.\run.bat`; it prints the app and health URLs, keeps the console open,
and displays startup failures before pausing.

Keep that terminal open while using the app. Open
`http://127.0.0.1:8000/` in the same machine's browser. Stop the server with
`Ctrl+C`. If port 8000 is occupied, start on another port and use that port in
the browser:

```powershell
python .\chatbot.py --port 8001
```

To select Groq, configure `AI_PROVIDER=groq` and `GROQ_API_KEY` in `.env`, or
use `--provider groq`. `--model` can override the selected provider's default
model. Process environment variables take precedence over `.env`.
When both provider keys are configured, Gemini is tried first by default and
Groq is used as a fallback. `AI_PROVIDER=groq` makes Groq the preferred provider.

## Optional hybrid retrieval

BM25 is the default and requires no model download. To enable multilingual
semantic retrieval, add `HYBRID_SEARCH=1` to `.env` and restart the server.
The first hybrid search downloads a pinned quantized E5 ONNX file and tokenizer
(about 135 MB, roughly 129 MiB total) from Hugging Face; questions and corpus
text remain local.
If the files cannot be downloaded or loaded, the server logs a generic warning
and continues with BM25. To evaluate either mode:

```powershell
python .\evaluate_retrieval.py
python .\evaluate_retrieval.py --hybrid
```

The evaluation reports the unchanged legacy 40-case metrics and the expanded
100-case metrics separately, including elapsed evaluation time. The added
cases include Urdu, Roman Urdu, out-of-scope, and location questions. The
hybrid command may download the model if it is not already present.

The BM25 mode remains the default. Hybrid retrieval is opt-in because it has
higher evaluation latency even when its retrieval scores improve; missing or
unavailable model assets fall back to BM25.

## Health checks

With the server running:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Expected output includes `"ok": true`, page count, image count, and count of
images with visual embeddings. A successful health response verifies that the
server and corpus loaded; it does **not** verify the AI provider quota, map
tiles, CDN access, or geocoder.

Check provider/geocoder network reachability and latency with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/diagnostics
```

Diagnostics use body-free HTTPS `HEAD` probes and never return keys or
questions. Reachability does not verify provider credentials or quota.

The nearby-place list requires verified registry coordinates. Currently none
of the seeded places have verified coordinates, so the interface correctly
reports that distances are unavailable. Do not add a pin based only on a
Photon result or an unverified map link.

General-knowledge answers without a cited source are labeled unverified and
include a warning that no independent source was checked. Recognized specific
opening-hour, price, or distance claims are suppressed when no source is
available.

The trip planner uses selected places from `GET /api/places` and sends their
IDs plus a day count to `POST /api/itinerary`. Its source excerpts come from
the local tourism corpus. Suggested day periods are planning prompts—not
confirmed opening hours—and the planner does not invent prices, travel times,
or route order.

## Common failures

| Symptom | Likely cause | What to check |
|---|---|---|
| Browser says `ERR_CONNECTION_REFUSED` | Server process is not running, exited during startup, or browser uses the wrong port | Start `python .\chatbot.py`; read startup errors in the same terminal; verify the URL/port |
| Startup says the port is in use | Another process is listening on port 8000 | Use `--port 8001`, then open `127.0.0.1:8001` |
| Page opens but answers are generic or unavailable | Missing key, invalid key, provider error, or exhausted quota | Check `.env`, provider setting, terminal diagnostics, and provider quota; restart after configuration changes |
| Gemini reports HTTP 429 | Rate limit or free-tier quota reached | Wait for quota reset or configure valid provider capacity/Groq |
| Answer says local fallback | No AI key is configured, or the provider chain was unavailable/rate-limited | Check `.env`, provider credentials, and provider quota; the tourism-page answer remains local and cited |
| Chat page works but location map is missing | Internet/CDN/tile service failure or Photon returned no coordinate | Check browser network access; the name-based Google Maps link may still work |
| Several locations are shown | Photon or the local registry found similarly ranked results | Select a listed result; Photon pins remain approximate, and unverified registry entries trigger a name-and-district Photon lookup |
| Nearby distances are unavailable | No registered place has verified coordinates | Confirm coordinates from an authoritative source before marking any place verified |
| Browser location is unavailable | Permission was denied, geolocation is unsupported, or the page is not in a secure context | Use localhost/HTTPS and allow location access; distance calculations are optional |
| Trip planner returns no itinerary | No valid places are selected, the selection/day count is invalid, or the local corpus lacks place details | Select registered places and check the corpus/API response |
| Google Maps link is rejected | Link is not HTTPS, is outside the Maps host allowlist, or cannot be safely resolved | Paste a supported HTTPS Google Maps link; coordinates parsed from a link are not independently verified |
| Image feature fails on first use | Missing dependency, unavailable model host, or model checksum/download issue | Install requirements and check internet access; the model downloads on first image-index/match request |
| Hybrid retrieval falls back to BM25 | `HYBRID_SEARCH` is disabled or a pinned E5 asset could not be loaded | Set `HYBRID_SEARCH=1` if desired; check the generic startup/request warning and internet access |
| Chat server cannot load data | Missing or malformed JSONL file | Check `data\tourism_pages.jsonl`; validate each non-empty line is a JSON object with string `url` and `text` |

Do not paste API keys into chat, screenshots, issue reports, or committed files.
The `.env` file is listed in `.gitignore`; keep it private.

## Refreshing the tourism corpus

The crawler should only be used by the site administrator or with explicit
site-owner authorization. It defaults to a one-second delay, stays on the
configured HTTPS host, and checks the site's `robots.txt`. It follows only
same-host HTTPS redirects for `robots.txt`; insecure or cross-host redirects
stop the crawl. A 404 is treated as an empty policy file. If `robots.txt`
returns HTTP 403, it stops by default. Only an authorized operator may
explicitly allow that unavailable policy file:

```powershell
python .\scrape_tourism.py --max-pages 200
```

With site-owner authorization when `robots.txt` is unavailable:

```powershell
python .\scrape_tourism.py --allow-robots-unavailable --max-pages 200
```

Useful options include `--delay`, `--timeout`, `--max-images`, and `--output`.
The crawler writes to a temporary sibling file and replaces the chosen output
only after a successful crawl. If crawling fails, the existing corpus remains
untouched.

To add embeddings to an existing authorized image index:

```powershell
python .\index_tourism_images.py --allow-robots-unavailable
```

The explicit authorization flag is required only when the site's robots file
is unavailable; obey any available robots rules.

## Tests

```powershell
python -m unittest discover -s .\tests -v
```

Run the test command after code, data-format, or dependency changes.

The optional Playwright viewport smoke test checks 390x844, 768x1024,
1024x500, and 1440x900. If its browser runtime cannot launch, manually check
each size in a browser: confirm the chat input and map are visible, the page
has no horizontal scrollbar, and the chat remains usable at the short height.
