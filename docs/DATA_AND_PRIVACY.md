# Data, external services, and privacy

## Project data

`data/tourism_pages.jsonl` stores indexed public tourism pages, including page
URLs, titles, descriptions, headings, extracted text, image URLs and
embeddings. It is read locally by the chatbot; the complete corpus is not sent
to the AI provider. The scraper can also retain crawl timestamps and a hash of
page text.

`data/models/` contains quantized CLIP assets and, when enabled, the multilingual
E5 ONNX model/tokenizer. Model files are downloaded from pinned Hugging Face
URLs, checked against expected SHA-256 hashes, and ignored by Git. The E5
assets are about 135 MB (roughly 129 MiB) total and download only when
`HYBRID_SEARCH=1` is configured or the operator runs
`evaluate_retrieval.py --hybrid`. The upstream
`intfloat/multilingual-e5-small` model card declares MIT; its Xenova ONNX export
does not declare a separate license in its repository metadata.

## Visitor data flow

- Text generation: when provider keys are configured, the current question,
  recent conversation history, and selected page excerpts are sent to the
  selected provider (Gemini by default), then to the alternate provider if
  configured and the selected provider fails. Provider responses are validated
  and citations are limited to the retrieved source set. Do not send sensitive
  or personal information.
- Local fallback: if no provider is configured or all configured providers
  fail, the server can return a matching local sentence and marks this in the
  UI. This is not equivalent to a provider-generated answer.
- Hybrid retrieval: the optional E5 ONNX model embeds the visitor's query and
  local page excerpts on the local machine. This feature does not send the
  question or corpus to Hugging Face or another embedding service. Hugging Face
  is contacted only to download the pinned model/tokenizer assets when hybrid
  search is enabled and the assets are not already present; if unavailable,
  retrieval falls back to local BM25.
- Grounding check: generated factual sentences are compared against the cited
  retrieved excerpts locally. When support is insufficient, the generated
  answer is replaced by a source excerpt.
- Response cache: up to 128 responses are retained in process memory for at
  most 60 seconds. Cache keys are hashes of normalized questions, conversation
  history, provider/model, and retrieved source URLs; cached data is not written
  to disk.
- Location lookup: when the question is detected as a map/location request,
  the extracted place name is sent from the server to Photon
  (`photon.komoot.io`) to request coordinates. Photon is an OpenStreetMap-based
  service; no API key is used by this project. For answers without local
  tourism-site evidence, the UI may link to a Google web search for the
  visitor's question; the server does not fetch or verify those search results.
  If results are ambiguous, selecting a Photon candidate displays its
  approximate coordinates locally. Selecting an unverified registry candidate
  sends that selected place name and district to Photon for a narrower lookup.
  This still uses the existing Photon service; the results are labeled
  approximate and external to the Punjab Tourism website. Registry candidate
  source links point to the corresponding public Punjab Tourism page.
- Google Maps links: direct links are parsed locally. If a user supplies a
  supported short Maps link, the server contacts only HTTPS hosts on the
  allowlist to follow at most three redirects; each hostname and connected IP
  is checked to reject non-public addresses. For place-only links, the
  extracted place name is sent to Photon. The link text is not logged or sent
  to the AI provider, and coordinates from a link are labeled unverified.
- Map display: the browser loads SRI-pinned Leaflet assets from unpkg and map
  tiles from OpenStreetMap's tile servers. The department vicinity map and
  result maps require internet access for those resources; tile failures show
  a status message.
- Diagnostics: `GET /api/diagnostics` sends body-free HTTPS `HEAD` probes to
  configured provider hosts and Photon. It returns reachability and latency,
  not credentials, questions, key validity, provider quota, or tile status.
- Image matching: the uploaded image is received by the local server and
  compared locally to stored CLIP embeddings. The upload is not sent to Gemini
  or Groq. Image URL matching results reference public indexed website
  resources.

## Browser location and trip planning

Browser geolocation is requested only after the visitor selects the
"Use my location" control. Coordinates stay in page memory and are used for
client-side distance calculations; they are not sent to this server or
persisted. A directions link may include the current browser coordinates as
its Google Maps origin. Creating the link does not navigate or transmit
anything; the visitor must activate it to open Google Maps.

`GET /api/places` returns registry records with coordinates only for
explicitly verified entries. `POST /api/itinerary` receives selected place IDs
and day count, not browser coordinates. It returns matching local corpus
excerpts and generic suggested time periods.

## Logging and secrets

The HTTP handler logs request time, client address, method and path, but not
chat text or query-string content. Provider failures are written to the server
terminal with secrets redacted. `.env` is ignored by Git.
Operators should still avoid sharing raw logs, keys, or visitor data.

## Collection and service policies

The crawler is restricted to a configured HTTPS host, respects available
`robots.txt`, uses a default one-second delay, and limits page/image payloads.
An unavailable robots file is not permission to crawl: use the override only
with explicit site-owner authorization. Coordinate recurring or large crawls
with the site administrator.

OpenStreetMap data requires attribution, which is included in the map controls.
Review the current usage policies of the tile host, Photon, unpkg, the Google
Maps hosts used for short-link redirects, and each configured AI provider
before production or high-volume use. “No API key” does not mean unlimited
use, guaranteed availability, or no external data transfer.

## Data retention boundaries

The current application has no explicit conversation database or chat-history
retention layer. Conversation history lives in browser memory for the active
page and is sent with subsequent requests; provider calls may be subject to the
provider's own terms and retention settings. The local HTTP server does not
persist chat messages in its request log.

## Before public deployment

- Publish a clear privacy notice describing the AI, Photon, map-link redirect,
  CDN, and tile service data flows.
- Set provider retention/training controls appropriate to the deployment.
- Add access controls, TLS, rate limits, and an incident/log retention policy.
- Consider self-hosted or contractually approved map assets/geocoding if
  external transfer is not acceptable.
