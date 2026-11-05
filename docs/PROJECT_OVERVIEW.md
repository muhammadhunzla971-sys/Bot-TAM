# Project overview and analysis

## Purpose

The Punjab Tourism Assistant is a small, local web application for asking
questions about information indexed from the Punjab Tourism website. It can
return source-page links, map links and coordinates for place queries, and
visual matches for uploaded tourism images.

## Current user features

- Chat in English or Urdu/Roman Urdu. Common greetings and a few identity or
  tourism-department questions have local responses.
- Search the locally stored tourism pages and answer from retrieved excerpts.
  Title/phrase-weighted BM25 uses an explicit confidence threshold; when
  optional hybrid search is enabled, multilingual ONNX embeddings are fused
  with BM25. Provider-generated answers include citations selected from
  retrieved pages, and unsupported generated claims are replaced with a source
  excerpt.
- Use Gemini by default, Groq as fallback when configured, and then a local
  corpus answer. Setting `AI_PROVIDER=groq` makes Groq preferred. The UI labels
  local/provider fallback and rate-limit states. Answers
  without local evidence are labeled as outside the Punjab Tourism website and
  include a clearly labeled web-search link; the chatbot does not verify those
  search results.
- Ask for a place's map/location to receive an OpenStreetMap map, coordinates
  when the Photon service returns a usable match, and a Google Maps link. If
  geocoding does not return a result, the name-based Google Maps search link
  remains available. When multiple candidates are similarly ranked, the user
  can select an approximate Photon result to reveal its map, or choose an
  unverified registry entry to retry with its district. Registry choices retain
  their official tourism-page source link.
- Paste an HTTPS Google Maps link to parse coordinates or geocode its place
  query. Link coordinates are labeled unverified; Photon lookups remain
  approximate. Short links are followed only on allowlisted Maps hosts, for at
  most three HTTPS redirects, after public-IP checks.
- Match an uploaded JPEG, PNG, WebP, or GIF against indexed site images. The
  upload is processed locally by quantized CLIP models; it is not sent to the
  text-generation provider. A map link for an image match requires the user to
  confirm the landmark first.
- See an approximate map vicinity for the Tourism Department address. The
  building marker was removed because its exact coordinates are unverified.
- View maps rendered with pinned Leaflet assets (SRI). If map tiles or the
  Leaflet CDN fail, a status message explains that place details and links
  remain available.
- Browse verified nearby places and calculate straight-line distances in the
  browser. Geolocation is requested only after the user acts, is not sent to
  this server, and is shared with Google Maps only when the user opens a
  directions link. The current registry has no verified coordinates, so
  nearby distances remain unavailable until coordinates are reviewed.
- Build a source-grounded itinerary from selected registry places. It uses
  indexed tourism-page excerpts and generic suggested day periods, not
  unverified opening hours, prices, travel times, or route order.
- Operators can check provider/geocoder endpoint reachability and latency at
  `GET /api/diagnostics`; this does not verify provider credentials or quota.

## Application components

| Path | Responsibility |
|---|---|
| `chatbot.py` | Corpus loading/search, HTTP routes, question handling, provider calls, map/geocoder lookup, local fallback |
| `static/index.html` | Chat interface, approximate department vicinity map, dynamic result maps, source links, image-match cards |
| `scrape_tourism.py` | Authorized same-host HTML and image crawler; builds the JSONL corpus |
| `index_tourism_images.py` | Adds CLIP visual and text embeddings to an existing image index |
| `image_matching.py` | Image validation, model asset management, embeddings, place labels, similarity ranking |
| `hybrid_search.py` | Pinned multilingual ONNX model assets and local text embeddings for optional hybrid retrieval |
| `evaluate_retrieval.py` | Reproducible hit@3, no-answer, and location metrics for the 40-question evaluation set |
| `data/tourism_pages.jsonl` | Runtime corpus: page text, metadata, source URLs, image data and embeddings |
| `data/models/` | Downloaded, pinned, quantized model/tokenizer assets; ignored by Git |
| `tests/` | Unit and local HTTP integration tests for chatbot, image matching, and scraper helpers |
| `requirements.txt` | Pillow, ONNX Runtime, NumPy, and tokenizer dependencies |
| `.env.example` | Example AI-provider and optional `HYBRID_SEARCH` configuration; copy to ignored `.env` for local settings |

## Request and answer flow

1. The browser sends the current question and a short conversation history to
   `POST /api/chat`.
2. The server validates the JSON request. If it contains a supported Google
   Maps link, the link is handled before normal retrieval: coordinates are
   marked unverified, and place names are looked up via the registry/Photon.
3. Otherwise, the server searches up to five corpus documents using
   title/phrase-weighted BM25 and a confidence threshold. Optional multilingual
   ONNX embeddings are combined with BM25 by reciprocal rank fusion when
   `HYBRID_SEARCH=1`; the server falls back to BM25 if the model is unavailable.
   If a place alias is recognized, its official tourism page is selected
   before the general corpus search is considered. Map questions prefer verified
   entries from `data/places.json`; otherwise the server asks Photon for up to
   five candidates filtered to the configured Pakistan/Punjab bounds. Ambiguous
   results are displayed with area details; selecting a Photon result renders
   its approximate map, while selecting an unverified registry entry performs
   a new lookup using its name and district. Verified registry coordinates may
   be shown as verified; unverified entries link to their official source page
   but never provide an exact pin.
4. With configured keys, the server tries the selected provider, then the
   alternate provider when configured, and finally a local sentence answer.
   It filters citations to the retrieved corpus sources. Responses are cached
   in memory by normalized question, recent history, and retrieved source URLs
   for at most 60 seconds (128 entries).
5. The response may include source links, map links, coordinates and indexed
   images. The browser renders these with Leaflet and OpenStreetMap tiles.

Image uploads use `POST /api/image`. The server validates the upload and
compares its local CLIP embedding to stored image embeddings; it returns up to
three page-level matches. If embeddings are not present in the JSONL index,
the indexer must be run before matching can work.

## Runtime/API surface

- `GET /` and `GET /index.html`: serve the chat page.
- `GET /api/health`: report process health and corpus/image counts.
- `GET /api/diagnostics`: report provider/geocoder network reachability and
  latency without returning keys or visitor questions.
- `POST /api/chat`: answer a text question and optionally return location data.
- `POST /api/image`: compare an uploaded image to the indexed tourism images.
- `GET /api/places`: return registered places and expose coordinates only for
  entries marked verified.
- `POST /api/itinerary`: build a bounded itinerary from selected registered
  place IDs and indexed source excerpts.

The default server bind is `127.0.0.1`, which keeps it accessible only from the
same machine. The chat API accepts up to 2,000 characters per question, up to
10 history messages, and a 64 KiB JSON body. Image uploads are limited to
10 MiB plus multipart overhead; decoded images are limited to 40 million
pixels.

Browser geolocation remains in page memory. It is not included in chat,
nearby-place, or itinerary API requests. A Google Maps directions URL can
contain the browser location as its origin, but only after the user activates
the directions link.

## Important scope distinction

The current map feature is a place-name geocoder, not a general web or Google
Maps search agent. It does not open arbitrary websites, follow arbitrary
Google Maps URLs, resolve shortened links, or verify a place against the
website named in a question. Geocoder results are labeled approximate; only
database entries explicitly marked verified may be labeled verified. The
seeded registry currently has no verified coordinates, so nearby distances
cannot yet be calculated. See the
[audit](AUDIT_REPORT.md) and [roadmap](ROADMAP.md) for remaining limitations.
