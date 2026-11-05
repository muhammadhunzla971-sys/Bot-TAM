# Project audit report

**Scope:** Static review of the current application, crawler, image pipeline,
and tests. This is a practical code and product audit, not a penetration test
or a guarantee that the application is free of defects.

**Review date:** 2026-11-04

## Summary

The project has a coherent local-first design: a small standard-library HTTP
server, bounded request sizes, a local corpus, filtered source citations,
image upload validation, pinned model checksums, and conservative crawler
host/robots checks. The test suite exercises core routes, provider fallback,
source filtering, image matching, and crawler URL/content behavior.

The principal outstanding issue is product correctness: arbitrary website
map contents are not fetched, and the places registry has no confirmed
coordinates. The department vicinity map has no marker because the exact
office/building point has not been verified. Public deployment would also
require additional operational and security controls. Phase 5 adds
title/phrase-weighted BM25 with an explicit confidence threshold, optional
locally executed multilingual ONNX hybrid retrieval, a citation-grounding
fallback, and a 40-case retrieval evaluation. Phase 6 added verified-only
nearby-place data, browser-only geolocation, source-grounded trip planning,
and confirmation-before-map behavior for image matches. Phase 7 reconciles
the project documentation with the implemented behavior and keeps remaining
features explicitly identified as proposals. Phase 8 adds explicit selection
for ambiguous location results without upgrading approximate results to
verified pins.

## Findings and recommendations

### High priority

1. **Arbitrary third-party website lookup is not implemented.**
   Google Maps URLs from the allowlist are parsed and short links are resolved
   with bounded HTTPS redirects and public-IP checks. This does not fetch
   arbitrary websites or verify user-provided coordinates. Place-only links
   use Photon and remain approximate unless the registry explicitly verifies
   a location.
   **Recommendation:** Keep arbitrary URL fetching out of scope unless an
   approved source allowlist and SSRF protections are designed.

2. **Geocoder remains approximate.**
   `lookup_location()` now requests up to five Photon results, asks Photon for
   a language-appropriate label, bounds results to Punjab, filters outside
   Pakistan/Punjab where identified, and returns candidate details when
   results are ambiguous. Visitors can select a Photon result to see its
   approximate map, or select an unverified registry result for a narrower
   name-and-district lookup. Its confidence rule is heuristic, and places
   missing region metadata cannot always be conclusively classified.
   **Recommendation:** Treat all Photon coordinates as approximate and verify
   landmark/business identity with the authoritative place source before
   promoting a record to verified.

3. **Exact office coordinate is still unknown.**
   Photon did not return a result for the full published address. A nearby
   Planning-and-Development-area feature is not sufficient evidence for the
   office entrance or building; the UI now shows the approximate vicinity
   without a marker and links to an address search.
   **Recommendation:** Confirm the building pin with the site administrator or
   an authoritative map link before adding an exact marker.

### Medium priority

4. **External map and geocoding dependencies affect availability and privacy.**
   The page loads SRI-pinned Leaflet from unpkg, map tiles from OpenStreetMap
   tile servers, and the backend sends place names to Photon. Provider and
   geocoder diagnostics measure endpoint reachability and latency; they do not
   verify provider credentials, quota, or tile availability. These services
   require internet access and have their own usage policies.
   **Recommendation:** Keep these limitations visible and review service
   policies before production or high-volume use.

5. **Provider quota and outages can reduce answer quality.**
   The configured chain tries the selected provider (Gemini by default), then
   Groq when available, then a local corpus answer. The UI identifies provider
   fallback and quota/rate-limit states. Diagnostics check network reachability
   only, not key validity or quota. During troubleshooting on 2026-11-04,
   Gemini returned HTTP 429.
   **Recommendation:** Configure an alternate provider if desired and monitor
   provider quota through its operator console.

6. **Static front-end assets and map tiles remain third-party hosted.**
   Leaflet CSS/JavaScript are loaded from pinned unpkg assets with SRI. A CDN
   outage prevents map rendering; tile request failures show a user-facing
   status, but the maps themselves still require internet access.
   **Recommendation:** Consider local vendoring if offline map rendering
   becomes a requirement.

7. **Server is a local development server, not a production deployment.**
   The default bind is loopback, but `--host` can expose the service. The
   built-in HTTP server has no authentication, TLS termination, rate limiting,
   or deployment-grade request logging/monitoring. A publicly reachable
   instance could be abused to consume AI quota and server resources.
   **Recommendation:** Keep loopback binding for local use. Before any
   network-facing deployment, add an authenticated reverse proxy, TLS,
   rate/concurrency limits, deployment-grade server configuration, and
   monitoring.

### Low priority / quality

8. **Map-link coordinates are user-provided, not verified.**
   HTTPS URLs from allowlisted Google Maps hosts are parsed; short-link
   resolution is restricted to HTTPS with a three-redirect limit and public-IP
   checks. Coordinates parsed from a link are labeled unverified. Place-only
   links are separately geocoded through Photon and labeled approximate.
   **Recommendation:** Keep the distinction between parsed coordinates,
   approximate geocoder results, and verified registry coordinates visible.

9. **Outside-corpus answers do not have retrieved citations.**
   The interface now identifies these as outside the Punjab Tourism website
   and offers a clearly labeled web-search link, but the server does not fetch
   or verify third-party search results. The link is a discovery aid, not a
   citation.
   **Recommendation:** If verified external citations are required, add an
   explicitly approved search provider and fetch/validation policy rather than
   presenting generated text as sourced research.

10. **Operational documentation was previously concentrated in the root
   README.**
   Setup, crawling, privacy, troubleshooting, and feature plans were not
   separated by audience.
   **Recommendation:** This `docs/` folder adds linked overview, audit,
   operations, privacy, and roadmap documents; keep them updated alongside
   behavior changes.

11. **Retrieval confidence and answer grounding are heuristic.**
   Title/phrase boosts and lexical/embedding thresholds improve ranking, but
   the 40-case evaluation set is small and not a substitute for review of
   current corpus coverage. Sentence-level token overlap can miss valid
   paraphrases or accept a partly supported claim.
   **Recommendation:** Review evaluation misses after corpus refreshes and
   expand the labeled set with reviewed examples before relying on these
   thresholds for higher-stakes answers.

## Positive controls observed

- Default loopback bind and explicit port range validation.
- Bounded chat request body, question length, history length, and image upload.
- Image format and pixel-count validation before image decoding.
- Model assets are pinned and SHA-256 checked before use.
- AI citations are restricted to source IDs from retrieved documents.
- Weak retrieval matches follow an explicit no-evidence path; generated claims
  with insufficient cited-text overlap are replaced by local source excerpts.
- Optional multilingual embedding retrieval is local-only, pinned by SHA-256,
  and falls back to BM25 when its model is unavailable.
- Crawler requires HTTPS, stays on the allowed host, filters file types,
  limits responses, checks `robots.txt`, and uses a request delay.
- Provider API-key-like values are redacted from provider error output.
- Chat request logs omit visitor question content.
- Tests cover core local API, provider fallback/citation behavior, location
  lookup validation, image matching and basic scraper behavior.
- `tests/eval_set.jsonl` covers 40 multilingual, ambiguous, and unanswerable
  retrieval cases; `evaluate_retrieval.py` reports hit@3, no-answer accuracy,
  and location accuracy.
- Browser geolocation is retained in browser memory and is only placed into an
  external directions link that the visitor must activate. No verified place
  coordinates currently exist, so the nearby feature reports distances as
  unavailable rather than guessing.
- Trip plans accept registered IDs and use local corpus excerpts; suggested
  day periods are not represented as confirmed opening hours, prices, travel
  times, or route order.

Latest evaluation on the checked-in set: BM25 scored 0.857 hit@3, 0.900
no-answer accuracy, and 0.900 location accuracy; hybrid search scored 0.886,
0.975, and 0.950 respectively. These are retrieval metrics on a small,
corpus-specific set—not provider answer accuracy or a general quality guarantee.

## Suggested acceptance checks for the next release

- Confirm the fixed office pin against an authoritative coordinate.
- Add tests for ambiguous/non-Punjab place names, geocoder no-result and
  provider-error behavior; map-link parsing and SSRF guard tests are in place.
- Manually verify the map tile/CDN fallback and quota/degraded UI in a browser.
- Verify geolocation permission denial and directions behavior in a browser;
  populate the registry only after authoritative coordinate review.
- Re-run `python .\evaluate_retrieval.py` after refreshing the page corpus and
  review misses; use `--hybrid` to compare optional multilingual retrieval.
- Keep the server bound to localhost unless a secured deployment is designed.
