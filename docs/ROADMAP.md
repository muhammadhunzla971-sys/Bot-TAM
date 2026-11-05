# Suggested updates and optional features

Completed phases are listed separately below. Items under priority headings
remain proposals until the corresponding code and tests exist.

## Completed

- **Phase 2 — Intent routing and parsing:** Deterministic info, location,
  directions, image, and small-talk classification; alias-aware place
  extraction preserves matched spelling and explicit region context.
  English, Urdu-script, Roman Urdu, follow-up, and map-link classification
  have focused tests.
- **Phase 3 — Safe map-link input:** HTTPS Google Maps URLs from the
  allowlisted map hosts can yield coordinates or a place query. Short links
  use at most three HTTPS redirects, with public-IP checks; parsed coordinates
  are explicitly unverified, while Photon results remain approximate.
- **Phase 4 — Reliability and degraded mode:** Gemini falls back to configured
  Groq and then local corpus answers; a 128-entry, 60-second response cache,
  provider/geocoder reachability diagnostics, explicit fallback status, Leaflet
  SRI/tile-error messaging, and `run.bat` are in place.
- **Phase 5 — Retrieval quality:** BM25 now weights title/heading phrases and
  gates weak matches through a confidence threshold. Optional multilingual E5
  ONNX embeddings are fused with BM25 using reciprocal rank fusion and fall
  back to BM25 when unavailable. Cited generated claims are checked against
  source text, with unsupported answers replaced by an extractive source
  excerpt. `tests/eval_set.jsonl` and `evaluate_retrieval.py` report hit@3,
  no-answer accuracy, and location accuracy over 40 English, Urdu, Roman Urdu,
  ambiguous, and unanswerable cases.
- **Phase 6 — Nearby places and trip planning:** Added a verified-coordinate
  registry API, browser-only geolocation and straight-line distance display,
  explicit Google Maps directions links, copy/zoom controls, confirmation
  before mapping image matches, and a source-grounded itinerary API/UI.
  Coordinates remain unavailable until authoritative verification; itinerary
  suggestions do not claim opening hours, prices, travel times, or route order.
- **Phase 7 — Documentation:** Updated the overview, audit status, operations,
  privacy notes, and roadmap to describe the implemented phases and preserve
  outstanding work as proposals.
- **Phase 8 — Location confidence:** Ambiguous results now offer selectable
  choices. Photon candidates reveal an explicitly approximate map only after
  selection; unverified registry candidates use a place-and-district Photon
  lookup, and only verified registry coordinates can produce verified pins.

## P0 — Correctness and supportability

1. **Verify the department map pin.** Confirm the actual building location with
   the site administrator or an authoritative source; make clear whether it is
   an approximate building marker.
## P1 — Retrieval and answer quality

- Add an optional source excerpt/“why this answer” display so users can verify
  factual answers.
- Add a user-reviewed authoritative source for verified place coordinates so
  nearby-distance calculations can be enabled safely.

## P1 — Reliability and observability

- Add response-time, error-rate, and model-quota metrics with privacy-aware
  aggregation.
- Add browser-facing API error details that distinguish server, provider,
  geocoder and network failures.

## P2 — User experience

- Add a source-page preview with title, excerpt, and crawl date.
- Add chat export or conversation reset controls; retain history only with
  clear user consent.
- Add keyboard and screen-reader testing for dynamic maps, status messages, and
  image-match cards.

## P2 — Data and operations

- Add a corpus freshness report and safe backup/restore procedure before a
  crawl or image-index rebuild.
- Record scraper/indexer summary metadata separately from page content.
- Add a controlled schedule for authorized data refreshes, with crawl budgets
  and alerting when robots policy changes.
- Add a documented production architecture before making the service
  network-accessible; keep secrets in a managed secret store, not a committed
  `.env`.
- Add dependency/model update review steps and verify license and usage terms
  for redistributable model and map assets.

## Optional larger features (require separate scope decisions)

- **Multi-source tourism answers:** index additional official destination
  sources with per-source provenance and independent crawl permissions.
- **General web search:** only with an explicitly selected search provider,
  safe URL handling, source attribution, and clear separation from the local
  tourism corpus.
- **Directions and routing:** integrate a routing service or user-selected
  external directions link; this is distinct from knowing a place's
  coordinates.
- **Admin dashboard:** authorized corpus refresh, provider status, and image
  index metrics, protected by authentication.
- **Offline mode:** locally hosted Leaflet assets, cached map data subject to
  licensing/size limits, and local-only answer behavior.

## Proposed release gates

- Location results are tested for ambiguous names, no result, out-of-region
  candidates, and non-Latin labels.
- Supplied links cannot cause arbitrary server-side URL fetching.
- The fixed marker is verified or explicitly labeled approximate.
- AI provider outages and quota errors are understandable to users.
- A public deployment has authentication, TLS, rate limits, privacy notices,
  and an operational owner.
