# Complete Project Audit Report

**Project:** Punjab Tourism Assistant  
**Audit date:** 2026-11-04  
**Scope:** Current workspace code, UI, local data/configuration shape, tests,
documentation, retrieval evaluation, and live local HTTP endpoints.  
**Audit type:** Practical product, reliability, data-flow, and code-quality
review. This is not a formal penetration test or a guarantee that no defects
exist.

## Executive summary

The main local chat flow, maps UI, image matching, place registry, itinerary
endpoint, and fallback paths are present and exercised by the test suite. The
application is running locally and returned a healthy corpus response during
this audit.

**One confirmed high-priority defect remains:** the Groq request sends a literal
placeholder instead of the configured authorization credential. As a result,
Groq cannot authenticate; the advertised Gemini-to-Groq fallback will continue
to the local answer when it reaches Groq. The existing provider-fallback test
does not check the actual outgoing authorization header.

The audit also found a map-popup handling concern, two crawler reliability/
scope issues, and product limitations around unverified coordinates and
browser-level regression testing. No application changes were made as part of
this report.

## Findings

Severity reflects user impact and likelihood in ordinary operation.

| ID | Severity | Finding | Evidence / impact | Recommended action |
|---|---|---|---|---|
| A-01 | **High** | Groq authentication header is a placeholder | `chatbot.py:1827-1831` sets `Authorization` to a fixed masked string instead of using the configured key. Groq requests therefore cannot authenticate, so the configured fallback does not work. Current tests verify provider ordering/fallback behavior but not the outgoing credential header. | Send the configured value as a Bearer credential and add a unit test asserting the header is built from the supplied key without printing or exposing it. |
| A-02 | **Medium** | Dynamic map-popup text is passed to Leaflet as an HTML string | `static/index.html:527-529` builds the normal location popup from the displayed place name. That name can originate in Photon, an external geocoder response. Leaflet interprets string popup content as HTML, unlike the newer ambiguous-candidate path, which passes a DOM node. | Create popup content as a DOM element and assign untrusted text with `textContent`; add a focused rendering test if UI test infrastructure is added. |
| A-03 | **Medium** | `robots.txt` retrieval does not use the crawler's same-host redirect guard | `scrape_tourism.py:208-219` opens the robots URL with the default opener. Page and image fetches use `SameHostRedirectHandler` (`scrape_tourism.py:243-272`), but the robots fetch does not. A redirected robots request can therefore leave the configured host, contrary to the crawler's documented same-host boundary. | Apply equivalent HTTPS/host redirect validation to robots retrieval or reject redirects. Add a redirect-policy test. |
| A-04 | **Medium** | A failed crawl can replace the existing corpus with a partial file | `scrape_tourism.py:306-310` opens the destination in write/truncate mode before crawling finishes; records are flushed incrementally. A later network or processing failure leaves the prior corpus replaced by a partial output. The operations guide advises making a backup, but the operation itself is not atomic. | Write to a temporary sibling file and replace the destination only after a successful crawl; preserve the old corpus on failure. |
| A-05 | **Low / product limitation** | Exact tourism locations and nearby distances are unavailable | `data/places.json` currently has no verified coordinates. `/api/places` correctly returns null coordinates for all 10 places, and the department map intentionally shows a vicinity without an exact marker. Consequently, nearby distance calculations cannot currently return place results. | Keep uncertainty visible. Add coordinates only after an authoritative source is supplied and reviewed. |
| A-06 | **Low / validation gap** | Responsive layout has no automated real-browser viewport test | Responsive CSS and JavaScript syntax were checked, and the live page serves the updated CSS. A Playwright viewport measurement could not run because its browser process failed to launch in this environment. Thus layout behavior at phone, tablet, short-height, and desktop sizes is not established by automated browser evidence. | Run manual browser checks at narrow and short viewports, or configure a working browser test runner and add a small layout smoke test. |

## Flow review

### Chat, retrieval, and answer provenance

- The chat route validates JSON shape, question length, and conversation
  history, then resolves map links and intents before normal answer generation.
- The default retrieval path is BM25; optional local ONNX hybrid retrieval is
  available and falls back to BM25 when unavailable.
- Provider citations are restricted to retrieved source IDs. Unsupported
  generated answers may be replaced by an extractive local answer.
- Unrelated general-knowledge questions can receive a generated response
  without a source citation; the UI labels those answers as outside the
  Punjab Tourism website and offers an unchecked search link. This is
  disclosed behavior, not a verified external-research flow.
- Retrieval and grounding remain heuristic. The evaluation set has 40
  corpus-specific cases and cannot establish general answer correctness.

### Locations and maps

- Google Maps link parsing is limited to supported hosts; short-link handling
  is bounded and checks public destinations before following redirects.
- Direct coordinates from a supplied link are marked unverified; Photon
  results are approximate; registry coordinates are surfaced as verified only
  when the record is explicitly verified and coordinates validate.
- Ambiguous results now offer user choices. Photon maps remain approximate;
  unverified registry choices trigger a district-narrowed Photon lookup.
- There are 10 registry entries and zero verified coordinate records. The
  department map is an approximate vicinity without an exact marker. This is a
  known data limitation, not evidence that the map pin is correct.
- Leaflet and OpenStreetMap tile availability require internet access. Photon
  queries disclose the requested place name to the external geocoder.

### Geolocation and itinerary

- Browser geolocation is requested only after user action and is held in page
  memory. The itinerary request contains selected place IDs and day count, not
  browser coordinates.
- Directions URLs may include the browser location as origin when the visitor
  activates the directions link.
- Itinerary output uses a local corpus excerpt and generic suggested day
  periods; it does not establish opening hours, prices, driving times, or
  route order.
- The live sample itinerary returned a sourced Derawar Fort entry.

### Image matching

- Upload type, byte size, dimensions, and image decoding are validated before
  matching. The upload is processed locally and is not included in text
  provider requests.
- Matches are similarity rankings, not certainty that the uploaded image
  depicts a landmark. The UI requires confirmation before mapping an image
  match.
- CLIP assets are pinned and checksum-checked. First-time model acquisition
  and CPU inference may make the first request noticeably slow.

### Operations and privacy

- The server defaults to loopback, uses bounded request sizes, and omits chat
  text from request-path logs. Provider calls do send the current question,
  recent history, and selected excerpts to the configured provider, as
  described in the privacy documentation.
- The built-in HTTP server is for local development. Binding it to a
  network-facing address without authentication, TLS, rate limits, and
  deployment monitoring is not recommended.
- No crawler was run during this audit. No `.env` values or provider
  credentials were read or included in this report.

## Checks performed

| Check | Result |
|---|---|
| `python -m unittest discover -s .\tests -v` | **Passed: 86 tests** |
| `python -m py_compile .\chatbot.py .\image_matching.py .\hybrid_search.py .\scrape_tourism.py .\index_tourism_images.py .\evaluate_retrieval.py .\tests\test_chatbot.py .\tests\test_scrape_tourism.py` | **Passed** |
| `python .\evaluate_retrieval.py` | **40 cases:** hit@3 30/35 (**0.857**); no-answer 36/40 (**0.900**); location 18/20 (**0.900**) |
| `python .\evaluate_retrieval.py --hybrid` | **40 cases:** hit@3 31/35 (**0.886**); no-answer 39/40 (**0.975**); location 19/20 (**0.950**) |
| Inline JavaScript syntax and responsive-rule presence check | **Passed** |
| Live `GET /`, `GET /api/health`, `GET /api/places`, and sample `POST /api/itinerary` | **Passed**; health reported 88 pages and 232 images; places returned 10 entries, 0 verified |
| Browser viewport automation | **Not completed**; the Playwright browser process could not be launched in this environment |

## Test coverage gaps

- No assertion currently checks that `call_groq()` sends the configured
  authorization value; this allowed A-01 to pass unnoticed.
- The normal location-popup path has no browser-level test for safe rendering
  of third-party place labels.
- Scraper tests cover URL normalization and content extraction, but not
  same-host redirect behavior for `robots.txt` or preservation of the old
  output file after a crawl failure.
- Tests and retrieval metrics use a small local fixture/corpus and do not
  validate current external provider availability, live geocoder quality,
  OpenStreetMap tile availability, or all viewport layouts.

## Recommended fix order

1. Fix and test Groq authentication so provider failover is real.
2. Make Leaflet popup rendering safe for external place labels.
3. Enforce same-host redirect policy for `robots.txt`.
4. Make corpus refresh atomic so crawl failures do not replace good data.
5. Add a reproducible browser viewport smoke test and keep manual map,
   geolocation-permission, and mobile checks in release verification.

## Conclusion

The core local app and test suite are functioning, but the project is **not
issue-free**: A-01 affects the advertised provider fallback, and A-02 through
A-04 are confirmed hardening/reliability issues. Exact map coordinates remain
unverified by design. The checks above provide useful confidence in covered
flows, not proof that every browser, provider, or external service works in
every environment.
