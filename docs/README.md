# Project documentation

This folder documents the current Punjab Tourism Assistant and its possible
next steps. The code and data remain the source of truth; recommendations below
are proposals, not features already implemented.

| Document | What it covers |
|---|---|
| [Project overview](PROJECT_OVERVIEW.md) | Features, components, architecture, and data flow |
| [Audit report](AUDIT_REPORT.md) | Current strengths, limitations, and prioritized findings |
| [Operations guide](OPERATIONS.md) | Windows setup, configuration, commands, and troubleshooting |
| [Data and privacy](DATA_AND_PRIVACY.md) | Data collection, external services, and visitor-data handling |
| [Roadmap](ROADMAP.md) | Suggested updates, optional features, and priorities |

## Current project at a glance

- Python HTTP server with a browser-based chat UI; default address is
  `http://127.0.0.1:8000/`.
- The local JSONL corpus contains Punjab Tourism website pages. The last
  observed health response reported 88 pages and 232 indexed images.
- Retrieval uses title/phrase-weighted BM25 and a confidence threshold.
  Optional multilingual E5 ONNX embeddings are fused locally; a 40-question
  evaluation script reports hit@3, no-answer, and location accuracy.
- Gemini is the default text provider; Groq is also supported. Without a
  configured provider key, the assistant can still use local, extractive
  answers for matching corpus pages.
- Leaflet and OpenStreetMap display the approximate department vicinity and
  chatbot place results. The exact office marker is omitted because its
  coordinates are unverified; place lookup uses the keyless Photon geocoder.
- Nearby distances require verified coordinates, which are currently absent.
  Browser location stays client-side; trip planning uses selected place IDs
  and local tourism-page excerpts.
- Ambiguous location candidates are selectable: Photon results are marked
  approximate, and unverified registry entries are narrowed by district before
  another geocoder lookup.
- Local CLIP image embeddings power optional tourism image matching.

The server is designed for local development, not as-is for public deployment.
See the audit and operations documents before exposing it to a network.
