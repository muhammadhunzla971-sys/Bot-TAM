import json
import http.client
import io
import os
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch
from io import BytesIO
from PIL import Image

from chatbot import (
    BOT_IDENTITY_ANSWER,
    Corpus,
    DEFAULT_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    BOT_CONTEXT_ANSWER,
    GEMINI_API_URL,
    GEMINI_FALLBACK_MODELS,
    INVITATION_ANSWER,
    MapLinkError,
    NO_ANSWER,
    ResponseCache,
    TOURISM_DEPARTMENT_ANSWER,
    call_groq,
    call_gemini,
    has_unsourced_specific_details,
    answer_is_grounded,
    has_corpus_evidence,
    is_tourism_department_question,
    contextualize_place_question,
    extract_place_query,
    find_place_in_text,
    google_maps_source,
    load_places,
    load_env_file,
    lookup_location,
    local_answer,
    make_handler,
    probe_endpoint,
    response_cache_key,
    resolve_google_maps_link,
    route_intent,
)
from hybrid_search import HybridModelUnavailable
from image_matching import (
    difference_hash,
    image_label_text,
    image_place_name,
    text_embedding,
    visual_embedding,
)
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "pages.jsonl"
        documents = [
            {
                "title": "Murree hill station",
                "description": "Scenic mountain destination",
                "headings": [{"level": "h1", "text": "Murree"}],
                "text": "Murree is a scenic hill station with pine forests.",
                "url": "https://tourism.punjab.gov.pk/murree",
            },
            {
                "title": "Lahore heritage",
                "description": "Historical architecture",
                "headings": [],
                "text": "Lahore has historic forts and Mughal architecture.",
                "url": "https://tourism.punjab.gov.pk/lahore",
            },
        ]
        self.path.write_text(
            "".join(json.dumps(item) + "\n" for item in documents),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_search_returns_matching_page(self):
        corpus = Corpus(self.path)
        results = corpus.search("Tell me about Murree hill station")
        self.assertEqual(len(results), 1)
        self.assertIn("murree", results[0]["url"])

    def test_search_returns_no_results_for_unmatched_query(self):
        corpus = Corpus(self.path)
        self.assertEqual(corpus.search("volcanoes on mars"), [])

    def test_title_phrase_boost_ranks_the_exact_title_match(self):
        documents = [
            {
                "title": "General Punjab page",
                "url": "https://tourism.punjab.gov.pk/general",
                "text": "Mughal architecture is briefly mentioned on this page.",
            },
            {
                "title": "Mughal architecture",
                "url": "https://tourism.punjab.gov.pk/mughal",
                "text": "This page describes a historic period.",
            },
        ]
        self.path.write_text(
            "".join(json.dumps(item) + "\n" for item in documents),
            encoding="utf-8",
        )
        corpus = Corpus(self.path)
        self.assertEqual(
            corpus.search("Mughal architecture", limit=1)[0]["url"],
            "https://tourism.punjab.gov.pk/mughal",
        )

    def test_hybrid_search_fuses_semantic_rank_with_bm25(self):
        corpus = Corpus(self.path, hybrid_enabled=True)

        def fake_embeddings(texts, role):
            return [[1.0, 0.0], [0.0, 1.0]] if role == "passage" else [[0.0, 1.0]]

        with patch("chatbot.embed_texts", side_effect=fake_embeddings):
            results = corpus.search("قلعہ مری", limit=1)
            self.assertEqual(results[0]["url"], "https://tourism.punjab.gov.pk/lahore")
            self.assertTrue(corpus.has_evidence("قلعہ مری", results))
        self.assertEqual(corpus.retrieval_mode, "hybrid")

    def test_unavailable_hybrid_model_degrades_to_bm25(self):
        corpus = Corpus(self.path, hybrid_enabled=True)
        with (
            patch(
                "chatbot.embed_texts",
                side_effect=HybridModelUnavailable("no model"),
            ),
            patch("chatbot.sys.stderr", io.StringIO()) as log_stream,
        ):
            results = corpus.search("Murree hill station")
        self.assertEqual(results[0]["url"], "https://tourism.punjab.gov.pk/murree")
        self.assertEqual(corpus.retrieval_mode, "bm25")
        self.assertNotIn("Murree", log_stream.getvalue())

    def test_rejects_malformed_jsonl(self):
        self.path.write_text("{bad json}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
            Corpus(self.path)


class GroundingTests(unittest.TestCase):
    def test_grounding_accepts_supported_claims_and_rejects_extra_facts(self):
        documents = [
            {
                "title": "Murree",
                "url": "https://tourism.punjab.gov.pk/murree",
                "text": "Murree is a scenic hill station with pine forests.",
            }
        ]
        sources = [{"title": "Murree", "url": documents[0]["url"]}]
        self.assertTrue(
            answer_is_grounded(
                "Murree is a scenic hill station.", sources, documents
            )
        )
        self.assertFalse(
            answer_is_grounded(
                "Murree is a scenic hill station. Admission costs 500 rupees.",
                sources,
                documents,
            )
        )

    def test_confidence_threshold_rejects_incidental_token_overlap(self):
        documents = [
            {
                "title": "Tourism Department",
                "url": "https://tourism.punjab.gov.pk/",
                "text": "The department promotes tourism and culture.",
            }
        ]
        self.assertFalse(
            has_corpus_evidence("explain quantum computing department", documents)
        )


class EnvFileTests(unittest.TestCase):
    def test_loads_optional_hybrid_search_setting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text("HYBRID_SEARCH=1\n", encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["HYBRID_SEARCH"], "1")

    def test_loads_model_from_env_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text("GROQ_MODEL=openai/gpt-oss-120b\n", encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["GROQ_MODEL"], DEFAULT_GROQ_MODEL)

    def test_process_environment_overrides_env_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text("GROQ_MODEL=model-from-file\n", encoding="utf-8")
            with patch.dict("os.environ", {"GROQ_MODEL": "model-from-process"}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["GROQ_MODEL"], "model-from-process")


class ResponseCacheTests(unittest.TestCase):
    def test_cache_is_bounded_lru_and_returns_copies(self):
        cache = ResponseCache(max_entries=2, ttl_seconds=30)
        with patch("chatbot.time.monotonic", return_value=10):
            cache.set("first", {"answer": "one", "sources": []})
            cache.set("second", {"answer": "two", "sources": []})
            self.assertEqual(cache.get("first")["answer"], "one")
            cache.set("third", {"answer": "three", "sources": []})
            self.assertIsNone(cache.get("second"))
            cached = cache.get("first")
            cached["answer"] = "changed"
            self.assertEqual(cache.get("first")["answer"], "one")

    def test_cache_entries_expire_after_ttl(self):
        cache = ResponseCache(max_entries=2, ttl_seconds=5)
        with patch("chatbot.time.monotonic", side_effect=[10, 14, 15]):
            cache.set("answer", {"answer": "cached", "sources": []})
            self.assertIsNotNone(cache.get("answer"))
            self.assertIsNone(cache.get("answer"))

    def test_cache_key_includes_normalized_question_history_and_source_ids(self):
        page = {
            "url": "https://tourism.punjab.gov.pk/murree",
            "text": "Murree is a hill station.",
        }
        same_key = response_cache_key(
            "Tell me about Murree!",
            [],
            [page],
            {"gemini": DEFAULT_GEMINI_MODEL},
        )
        self.assertEqual(
            same_key,
            response_cache_key(
                "tell me about murree",
                [],
                [page],
                {"gemini": DEFAULT_GEMINI_MODEL},
            ),
        )
        self.assertNotEqual(
            same_key,
            response_cache_key(
                "tell me about murree",
                [],
                [{"url": "https://tourism.punjab.gov.pk/murree-and-gallies"}],
                {"gemini": DEFAULT_GEMINI_MODEL},
            ),
        )
        self.assertNotEqual(
            same_key,
            response_cache_key(
                "tell me about murree",
                [{"role": "user", "content": "previous context"}],
                [page],
                {"gemini": DEFAULT_GEMINI_MODEL},
            ),
        )

    @patch("chatbot.urlopen")
    def test_diagnostic_probe_uses_head_without_a_request_body(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        mock_urlopen.return_value = FakeResponse()
        result = probe_endpoint("https://example.test/")
        request = mock_urlopen.call_args.args[0]
        self.assertTrue(result["reachable"])
        self.assertEqual(request.get_method(), "HEAD")
        self.assertIsNone(request.data)


class LocalAnswerTests(unittest.TestCase):
    def test_handles_common_greetings_and_small_talk(self):
        for question in (
            "hi",
            "Hi!",
            "hello",
            "whoare you",
            "who are u",
            "who r u",
            "tum kya hu",
            "how are you",
            "can i ask you something?",
            "can I ask you anything",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(local_answer(question))

    def test_invitation_has_a_friendly_response(self):
        self.assertEqual(
            local_answer("can i ask you something?"),
            INVITATION_ANSWER,
        )

    def test_identity_questions_work_in_roman_urdu(self):
        self.assertIn("Punjab Tourism Assistant", local_answer("tum kya hu"))
        self.assertIn("Punjab Tourism Assistant", local_answer("aap kon ho"))

    def test_explains_virtual_assistant_context(self):
        self.assertEqual(
            local_answer("what are u doing right now and from where"),
            BOT_CONTEXT_ANSWER,
        )

    def test_does_not_consume_tourism_questions_as_small_talk(self):
        self.assertIsNone(local_answer("Did you know anything about tourism?"))

    def test_identifies_tourism_department_questions_and_common_typo(self):
        self.assertTrue(is_tourism_department_question("what is tourism department"))
        self.assertTrue(is_tourism_department_question("what is tourism deprt"))
        self.assertFalse(is_tourism_department_question("where is tourism department"))


class LocationLookupTests(unittest.TestCase):
    def setUp(self):
        self.places = load_places()

    @patch("chatbot.urlopen")
    def test_looks_up_coordinates_and_readable_place_name(self, mock_urlopen):
        response_body = {
            "features": [
                {
                    "geometry": {"coordinates": [74.185, 28.775]},
                    "properties": {
                        "name": "Derawar Fort",
                        "city": "Bahawalpur",
                        "country": "Pakistan",
                        "countrycode": "pk",
                        "state": "Punjab",
                        "district": "Bahawalpur",
                    },
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(response_body).encode()

        mock_urlopen.return_value = FakeResponse()
        result = lookup_location("Derawar Fort", self.places)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["precision"], "approximate")
        self.assertEqual(result["latitude"], 28.775)
        self.assertEqual(result["longitude"], 74.185)
        self.assertEqual(result["label"], "Derawar Fort")
        self.assertEqual(result["district"], "Bahawalpur")
        self.assertIn("photon.komoot.io", mock_urlopen.call_args.args[0].full_url)
        self.assertIn("limit=5", mock_urlopen.call_args.args[0].full_url)
        self.assertIn("bbox=", mock_urlopen.call_args.args[0].full_url)
        self.assertIn("lang=en", mock_urlopen.call_args.args[0].full_url)

    @patch("chatbot.urlopen")
    def test_rejects_invalid_coordinates(self, mock_urlopen):
        response_body = {
            "features": [
                {
                    "geometry": {"coordinates": [181, 91]},
                    "properties": {"name": "Invalid place", "countrycode": "pk"},
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(response_body).encode()

        mock_urlopen.return_value = FakeResponse()
        self.assertEqual(
            lookup_location("Invalid place", self.places)["status"], "not_found"
        )

    @patch("chatbot.urlopen")
    def test_exact_alias_uses_verified_registry_coordinates_without_geocoding(
        self, mock_urlopen
    ):
        verified_place = {
            **self.places[0],
            "lat": 31.588,
            "lng": 74.31,
            "verified": True,
        }
        result = lookup_location("Badshahi Masjid", [verified_place])
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["precision"], "verified")
        self.assertEqual(result["latitude"], 31.588)
        mock_urlopen.assert_not_called()

    @patch("chatbot.urlopen")
    def test_unverified_registry_place_uses_canonical_name_for_photon(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps({"features": []}).encode()

        mock_urlopen.return_value = FakeResponse()
        result = lookup_location("Badshahi Masjid", self.places)
        self.assertEqual(result["status"], "not_found")
        self.assertIn("Badshahi+Mosque", mock_urlopen.call_args.args[0].full_url)

    @patch("chatbot.urlopen")
    def test_ambiguous_photon_results_ask_for_district(self, mock_urlopen):
        features = []
        for offset, city in enumerate(("Lahore", "Kasur")):
            features.append({
                "geometry": {"coordinates": [74.3 + offset * 0.1, 31.5]},
                "properties": {
                    "name": "Shrine",
                    "city": city,
                    "district": city,
                    "country": "Pakistan",
                    "countrycode": "pk",
                    "state": "Punjab",
                },
            })

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps({"features": features}).encode()

        mock_urlopen.return_value = FakeResponse()
        result = lookup_location("Shrine", self.places)
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertIn("district", result["message"])
        self.assertEqual(result["candidates"][0]["precision"], "approximate")
        self.assertEqual(result["candidates"][0]["latitude"], 31.5)
        self.assertEqual(result["candidates"][0]["longitude"], 74.3)

    def test_ambiguous_registry_only_exposes_verified_candidate_coordinates(self):
        first = {
            **self.places[0],
            "name_en": "Shrine East",
            "aliases": ["Shared Shrine"],
            "district": "Lahore",
        }
        second = {
            **self.places[1],
            "name_en": "Shrine West",
            "aliases": ["Shared Shrine"],
            "district": "Kasur",
        }

        result = lookup_location("Shared Shrine", [first, second])
        self.assertEqual(result["status"], "ambiguous")
        self.assertTrue(all("latitude" not in item for item in result["candidates"]))
        self.assertTrue(
            all(item["source_url"].startswith("https://tourism.punjab.gov.pk/")
                for item in result["candidates"])
        )

        first.update({"lat": 31.5, "lng": 74.3, "verified": True})
        second.update({"lat": 31.2, "lng": 74.1, "verified": True})
        verified_result = lookup_location("Shared Shrine", [first, second])
        self.assertEqual(verified_result["status"], "ambiguous")
        self.assertEqual(verified_result["candidates"][0]["precision"], "verified")
        self.assertEqual(verified_result["candidates"][0]["latitude"], 31.5)
        self.assertEqual(verified_result["candidates"][0]["longitude"], 74.3)

    @patch("chatbot.urlopen")
    def test_registry_district_context_resolves_shared_alias(self, mock_urlopen):
        first = {
            **self.places[0],
            "name_en": "Shared Shrine",
            "aliases": ["Shared Shrine"],
            "district": "Lahore",
        }
        second = {
            **self.places[1],
            "name_en": "Shared Shrine",
            "aliases": ["Shared Shrine"],
            "district": "Kasur",
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps({"features": []}).encode()

        mock_urlopen.return_value = FakeResponse()
        result = lookup_location("Shared Shrine, Kasur", [first, second])
        self.assertEqual(result["status"], "not_found")
        requested_url = mock_urlopen.call_args.args[0].full_url
        self.assertIn("Kasur", requested_url)
        self.assertIn("Punjab", requested_url)

    @patch("chatbot.urlopen")
    def test_filters_non_punjab_and_out_of_pakistan_candidates(self, mock_urlopen):
        features = [
            {
                "geometry": {"coordinates": [74.3, 31.5]},
                "properties": {
                    "name": "Target place",
                    "country": "Pakistan",
                    "countrycode": "pk",
                    "state": "Sindh",
                },
            },
            {
                "geometry": {"coordinates": [74.3, 31.5]},
                "properties": {
                    "name": "Target place",
                    "country": "India",
                    "countrycode": "in",
                    "state": "Punjab",
                },
            },
        ]

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps({"features": features}).encode()

        mock_urlopen.return_value = FakeResponse()
        self.assertEqual(
            lookup_location("Target place", self.places)["status"], "not_found"
        )

    @patch("chatbot.urlopen")
    def test_photon_failure_returns_safe_error_state_without_exception_text(
        self, mock_urlopen
    ):
        mock_urlopen.side_effect = HTTPError(
            "https://photon.komoot.io/api/",
            503,
            "service unavailable",
            {},
            None,
        )
        with patch("chatbot.sys.stderr") as stderr:
            result = lookup_location("unknown place", self.places)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("unknown place", str(stderr.write.call_args))
        self.assertNotIn("503", str(stderr.write.call_args))

    @patch("chatbot.urlopen")
    def test_no_photon_match_returns_explicit_no_result(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return b'{"features":[]}'

        mock_urlopen.return_value = FakeResponse()
        result = lookup_location("Unknown place", self.places)
        self.assertEqual(result["status"], "not_found")
        self.assertIn("Punjab, Pakistan", result["message"])

    @patch("chatbot.urlopen")
    def test_uses_urdu_photon_name_for_urdu_query(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps({
                    "features": [{
                        "geometry": {"coordinates": [74.3, 31.5]},
                        "properties": {
                            "name": "بادشاہی مسجد",
                            "name_en": "Badshahi Mosque",
                            "country": "Pakistan",
                            "countrycode": "pk",
                            "state": "Punjab",
                        },
                    }]
                }).encode()

        mock_urlopen.return_value = FakeResponse()
        result = lookup_location("بادشاہی مسجد", self.places)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["label"], "بادشاہی مسجد")
        self.assertIn("lang=ur", mock_urlopen.call_args.args[0].full_url)

    def test_fuzzy_place_alias_is_found_inside_a_question(self):
        place = find_place_in_text(
            "badshai mosque ki pictures dikhao", load_places()
        )
        self.assertIsNotNone(place)
        self.assertEqual(place["id"], "badshahi-mosque")


class IntentRoutingTests(unittest.TestCase):
    def test_routes_primary_intents_in_english_urdu_and_roman_urdu(self):
        examples = (
            ("Show photos of Badshahi Mosque", "image"),
            ("بادشاہی مسجد کی تصاویر دکھائیں", "image"),
            ("Badshahi Masjid ki tasveer dikhao", "image"),
            ("Where is Badshahi Mosque?", "location"),
            ("بادشاہی مسجد کہاں ہے؟", "location"),
            ("Badshahi Masjid ka naqsha dikhao", "location"),
            ("How do I reach Badshahi Mosque?", "directions"),
            ("بادشاہی مسجد کا راستہ بتائیں", "directions"),
            ("Badshahi Masjid ka raasta batao", "directions"),
            ("Hello", "smalltalk"),
            ("Tell me about Badshahi Mosque", "info"),
        )
        for question, expected in examples:
            with self.subTest(question=question):
                self.assertEqual(route_intent(question), expected)

    def test_contextual_follow_up_in_urdu_inherits_previous_place(self):
        question = contextualize_place_question(
            "اس کی تصاویر دکھائیں",
            [{"role": "user", "content": "بادشاہی مسجد کہاں ہے؟"}],
        )
        self.assertIn("بادشاہی مسجد", question)
        self.assertEqual(route_intent(question), "image")

    def test_extracts_alias_with_explicit_region_and_preserves_spelling(self):
        self.assertEqual(
            extract_place_query(
                "Share directions to Badshahi Masjid in Lahore, Punjab",
                load_places(),
            ),
            "Badshahi Masjid in Lahore, Punjab",
        )
        self.assertEqual(
            extract_place_query("badshai mosque kahan hai", load_places()),
            "badshai mosque",
        )

    def test_classifies_google_maps_link_without_treating_it_as_place_name(self):
        link = "https://www.google.com/maps/search/?api=1&query=Badshahi+Mosque"
        self.assertEqual(route_intent(link), "location")
        self.assertIsNone(extract_place_query(link, load_places()))

    def test_directions_intent_produces_a_google_maps_search_source(self):
        source = google_maps_source(
            "How do I reach Badshahi Mosque?",
            "Badshahi Mosque",
            route_intent("How do I reach Badshahi Mosque?"),
        )
        self.assertIsNotNone(source)
        self.assertIn("maps/search", source["url"])


class GoogleMapsLinkTests(unittest.TestCase):
    class FakeResponse:
        def __init__(self, status, location=None):
            self.status = status
            self.location = location

        def getheader(self, name):
            return self.location if name.casefold() == "location" else None

    class FakeSocket:
        def getpeername(self):
            return ("8.8.8.8", 443)

    class FakeConnection:
        def __init__(self, response):
            self.response = response
            self.sock = GoogleMapsLinkTests.FakeSocket()
            self.requests = []

        def connect(self):
            pass

        def request(self, method, target, headers=None):
            self.requests.append((method, target, headers))

        def getresponse(self):
            return self.response

        def close(self):
            pass

    def test_parses_coordinate_and_place_url_forms(self):
        examples = (
            (
                "https://www.google.com/maps/search/?api=1&query=31.5881%2C74.3093",
                (31.5881, 74.3093),
                None,
            ),
            (
                "https://maps.google.com/maps/@31.5881,74.3093,16z",
                (31.5881, 74.3093),
                None,
            ),
            (
                "https://www.google.com/maps/place/Badshahi+Mosque/data=!3d31.5881!4d74.3093",
                (31.5881, 74.3093),
                None,
            ),
            (
                "https://www.google.com/maps/place/Badshahi+Mosque",
                None,
                "Badshahi Mosque",
            ),
            (
                "https://google.com/maps?query=Badshahi+Mosque",
                None,
                "Badshahi Mosque",
            ),
        )
        for url, coordinates, place_query in examples:
            with self.subTest(url=url):
                result = resolve_google_maps_link(url)
                self.assertEqual(result["coordinates"], coordinates)
                self.assertEqual(result["place_query"], place_query)

    def test_resolves_short_link_through_allowlisted_https_redirects(self):
        connections = [
            self.FakeConnection(
                self.FakeResponse(
                    302,
                    "https://www.google.com/maps/search/?api=1&query=31.5%2C74.3",
                )
            ),
            self.FakeConnection(self.FakeResponse(200)),
        ]
        with (
            patch(
                "chatbot._open_public_map_connection",
                side_effect=connections,
            ),
        ):
            result = resolve_google_maps_link("https://maps.app.goo.gl/abc123")
        self.assertEqual(result["coordinates"], (31.5, 74.3))
        self.assertEqual(len(connections), 2)

    def test_rejects_non_allowlisted_map_host(self):
        result = resolve_google_maps_link("https://evil.example/maps?q=31,74")
        self.assertIn("Unsupported map link", result["error"])

    def test_rejects_allowlisted_hostname_resolving_to_private_ip(self):
        answer = [(2, 1, 6, "", ("127.0.0.1", 443))]
        with patch("chatbot.socket.getaddrinfo", return_value=answer):
            result = resolve_google_maps_link("https://maps.app.goo.gl/abc123")
        self.assertIn("Unsupported map link", result["error"])

    def test_blocks_private_ip_redirect_before_connecting_again(self):
        connection = self.FakeConnection(
            self.FakeResponse(302, "https://127.0.0.1/maps?q=31,74")
        )
        with (
            patch("chatbot._open_public_map_connection", return_value=connection),
        ):
            result = resolve_google_maps_link("https://goo.gl/maps/abc123")
        self.assertIn("Unsupported map link", result["error"])
        self.assertEqual(len(connection.requests), 1)

    def test_blocks_allowlisted_redirect_with_private_dns_result(self):
        connection = self.FakeConnection(
            self.FakeResponse(
                302,
                "https://maps.google.com/maps?query=Badshahi+Mosque",
            )
        )
        with (
            patch(
                "chatbot._open_public_map_connection",
                side_effect=[connection, MapLinkError("unsupported")],
            ),
        ):
            result = resolve_google_maps_link("https://goo.gl/maps/abc123")
        self.assertIn("Unsupported map link", result["error"])
        self.assertEqual(len(connection.requests), 1)

    def test_rejects_private_connected_ip_before_sending_request(self):
        connection = self.FakeConnection(self.FakeResponse(200))
        connection.sock.getpeername = lambda: ("127.0.0.1", 443)
        with (
            patch("chatbot._open_public_map_connection", return_value=connection),
        ):
            result = resolve_google_maps_link("https://maps.app.goo.gl/abc123")
        self.assertIn("Unsupported map link", result["error"])
        self.assertEqual(connection.requests, [])

    def test_rejects_malformed_google_maps_url(self):
        result = resolve_google_maps_link(
            "https://maps.google.com:99999/maps?q=Badshahi+Mosque"
        )
        self.assertIn("couldn't parse", result["error"])

    def test_rejects_short_link_after_redirect_limit(self):
        redirect = self.FakeResponse(
            302, "https://maps.google.com/maps?query=Badshahi+Mosque"
        )
        connections = [self.FakeConnection(redirect) for _ in range(4)]
        with (
            patch(
                "chatbot._open_public_map_connection",
                side_effect=connections,
            ),
        ):
            result = resolve_google_maps_link("https://maps.app.goo.gl/abc123")
        self.assertIn("redirected too many times", result["error"])
        self.assertEqual(len(connections), 4)


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        location_lookup = patch(
            "chatbot.lookup_location",
            return_value={
                "status": "not_found",
                "precision": "search only",
                "message": "No matching place was found.",
                "candidates": [],
            },
        )
        location_lookup.start()
        self.addCleanup(location_lookup.stop)
        self.temp_dir = tempfile.TemporaryDirectory()
        data_path = Path(self.temp_dir.name) / "pages.jsonl"
        image_buffer = BytesIO()
        sample_image = Image.new("RGB", (160, 120), "white")
        for x in range(25, 135):
            for y in range(20, 100):
                sample_image.putpixel((x, y), (20, 90, 45))
        sample_image.save(image_buffer, format="PNG")
        self.sample_image = image_buffer.getvalue()
        documents = [
            {
                "title": "Punjab Tourism Department",
                "url": "https://tourism.punjab.gov.pk/",
                "text": (
                    "The Tourism Department works with affiliated departments "
                    "to promote, preserve, enhance, and harness tourism potential."
                ),
            },
            {
                "title": "Cholistan - Gateway to Adventure",
                "url": "https://tourism.punjab.gov.pk/cholistan",
                "text": (
                    "In the middle of the Cholistan Desert stands Derawar Fort, "
                    "approximately 95 km south of Bahawalpur. "
                    "This is the largest and best preserved chain of forts in the desert."
                ),
                "images": [
                    {
                        "url": "https://tourism.punjab.gov.pk/system/files/fort.png",
                        "alt": "Derawar Fort",
                        "dhash": difference_hash(self.sample_image),
                        "embedding": visual_embedding(self.sample_image),
                        "label_embedding": text_embedding(
                            image_label_text(
                                "Cholistan - Gateway to Adventure",
                                "Derawar Fort",
                                "https://tourism.punjab.gov.pk/system/files/fort.png",
                            )
                        ),
                    }
                ],
                "text_embedding": text_embedding(
                    "A photograph of Cholistan - Gateway to Adventure"
                ),
            },
            {
                "title": "Punjab tourism",
                "url": "https://tourism.punjab.gov.pk/aboutus",
                "text": "Punjab has many tourism destinations.",
            },
            {
                "title": "Badshahi Mosque | Punjab Tourism",
                "url": "https://tourism.punjab.gov.pk/badshahi-mosque",
                "text": (
                    "Badshahi Mosque is located in front of Alamgiri Gate of "
                    "Lahore Fort. The mosque was completed by 1674 AD."
                ),
                "images": [
                    {
                        "url": "https://tourism.punjab.gov.pk/sites/files/badshahi01.jpg",
                        "alt": "",
                        "dhash": difference_hash(self.sample_image),
                        "embedding": visual_embedding(self.sample_image),
                        "label_embedding": text_embedding(
                            "a photograph of Badshahi Mosque"
                        ),
                    }
                ],
            },
        ]
        data_path.write_text(
            "".join(json.dumps(document) + "\n" for document in documents),
            encoding="utf-8",
        )
        self.corpus = Corpus(data_path)
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.corpus, DEFAULT_MODEL, "", "gemini"),
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp_dir.cleanup()

    def post_question(self, question, server=None, history=None):
        server = server or self.server
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        connection.request(
            "POST",
            "/api/chat",
            json.dumps({"question": question, "history": history or []}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        connection.close()
        return response.status, result

    def post_itinerary(self, payload, server=None):
        server = server or self.server
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        connection.request(
            "POST",
            "/api/itinerary",
            json.dumps(payload),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        connection.close()
        return response.status, result

    def get_path(self, path, server=None):
        server = server or self.server
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read()
        connection.close()
        result = json.loads(body) if response.getheader("Content-Type", "").startswith(
            "application/json"
        ) else None
        return response.status, result

    def post_image(self, image_bytes, server=None):
        server = server or self.server
        boundary = "tourism-image-test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="image"; filename="upload.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        connection.request(
            "POST",
            "/api/image",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        connection.close()
        return response.status, result

    def test_identity_question_works_without_groq_key(self):
        for question in ("who are u", "who r u", "tum kya hu"):
            with self.subTest(question=question):
                status, result = self.post_question(question)
                self.assertEqual(status, 200)
                self.assertIn("Punjab Tourism Assistant", result["answer"])
                self.assertEqual(result["sources"], [])

    def test_tourism_question_still_requires_groq_key(self):
        status, result = self.post_question("Tell me about Punjab tourism")
        self.assertEqual(status, 200)
        self.assertIn("Punjab has many tourism destinations.", result["answer"])
        self.assertEqual(result["sources"][0]["url"], "https://tourism.punjab.gov.pk/aboutus")
        self.assertEqual(result["answer_origin"], "local fallback")
        self.assertIn("No AI provider key is configured", result["source_note"])

    def test_tourism_department_question_is_answered_from_official_page(self):
        for question in ("what is tourism department", "what is tourism deprt"):
            with self.subTest(question=question):
                status, result = self.post_question(question)
                self.assertEqual(status, 200)
                self.assertEqual(result["answer"], TOURISM_DEPARTMENT_ANSWER)
                self.assertEqual(
                    result["sources"][0]["url"],
                    "https://tourism.punjab.gov.pk/",
                )

    def test_derawar_question_is_answered_from_local_pages_without_api_key(self):
        status, result = self.post_question("where is derawar fort")
        self.assertEqual(status, 200)
        self.assertIn("middle of the Cholistan Desert", result["answer"])
        self.assertEqual(result["answer_origin"], "local fallback")
        self.assertIn("No AI provider key is configured", result["source_note"])
        self.assertEqual(
            result["sources"][0]["url"],
            "https://tourism.punjab.gov.pk/cholistan",
        )
        self.assertIn("Google Maps: Derawar Fort", result["maps"][0]["title"])
        self.assertIn(
            "query=Derawar+Fort",
            result["maps"][0]["url"],
        )

    def test_why_famous_question_answers_from_page_and_links_to_exact_place(self):
        status, result = self.post_question("why is Derawar Fort famous")
        self.assertEqual(status, 200)
        self.assertIn("largest and best preserved chain of forts", result["answer"])
        self.assertIn("query=Derawar+Fort", result["maps"][0]["url"])

    def test_location_lookup_adds_coordinates_and_coordinate_google_maps_link(self):
        with patch(
            "chatbot.lookup_location",
            return_value={
                "status": "found",
                "precision": "approximate",
                "latitude": 28.775,
                "longitude": 74.185,
                "label": "Derawar Fort, Bahawalpur, Pakistan",
            },
        ):
            status, result = self.post_question(
                "Google Maps link and exact location for Derawar Fort"
            )
        self.assertEqual(status, 200)
        self.assertEqual(result["location"]["latitude"], 28.775)
        self.assertIn("query=28.775%2C74.185", result["maps"][0]["url"])

    def test_google_maps_coordinate_link_is_labeled_unverified(self):
        with patch("chatbot.lookup_location") as lookup:
            status, result = self.post_question(
                "https://www.google.com/maps/search/?api=1&query=31.5881%2C74.3093"
            )
        lookup.assert_not_called()
        self.assertEqual(status, 200)
        self.assertEqual(result["location"]["precision"], "coordinates parsed from link")
        self.assertEqual(result["location"]["latitude"], 31.5881)
        self.assertIn("not independently verified", result["answer"])
        self.assertEqual(result["maps"][0]["url"], (
            "https://www.google.com/maps/search/?api=1&query=31.5881%2C74.3093"
        ))

    def test_google_maps_place_link_is_geocoded_as_approximate(self):
        with patch(
            "chatbot.lookup_location",
            return_value={
                "status": "found",
                "precision": "approximate",
                "latitude": 31.5881,
                "longitude": 74.3093,
                "label": "Badshahi Mosque, Lahore, Pakistan",
                "sources": [],
            },
        ) as lookup:
            status, result = self.post_question(
                "https://www.google.com/maps/place/Badshahi+Mosque"
            )
        self.assertEqual(status, 200)
        self.assertEqual(lookup.call_args.args[0], "Badshahi Mosque")
        self.assertEqual(result["location"]["precision"], "approximate")
        self.assertIn("approximate", result["answer"])

    def test_diagnostics_reports_provider_and_geocoder_status_without_keys(self):
        secret_gemini = "gemini-secret-test-key"
        secret_groq = "groq-secret-test-key"
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(
                self.corpus,
                DEFAULT_GEMINI_MODEL,
                secret_gemini,
                "gemini",
                provider_keys={"gemini": secret_gemini, "groq": secret_groq},
                provider_models={
                    "gemini": DEFAULT_GEMINI_MODEL,
                    "groq": DEFAULT_GROQ_MODEL,
                },
            ),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch(
                "chatbot.probe_endpoint",
                return_value={"reachable": True, "latency_ms": 4.2},
            ) as probe:
                status, result = self.get_path("/api/diagnostics", server)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        serialized = json.dumps(result)
        self.assertEqual(status, 200)
        self.assertTrue(result["providers"]["gemini"]["configured"])
        self.assertTrue(result["providers"]["groq"]["reachable"])
        self.assertTrue(result["geocoder"]["reachable"])
        self.assertNotIn(secret_gemini, serialized)
        self.assertNotIn(secret_groq, serialized)
        self.assertNotIn("question", serialized)
        self.assertEqual(probe.call_count, 3)

    def test_request_log_omits_query_text(self):
        with patch("chatbot.sys.stderr", io.StringIO()) as log_stream:
            status, _ = self.get_path(
                "/api/health?question=visitor-private-text"
            )
        self.assertEqual(status, 404)
        self.assertNotIn("visitor-private-text", log_stream.getvalue())

    def test_places_endpoint_exposes_only_verified_coordinates(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(
                self.corpus,
                DEFAULT_MODEL,
                "",
                "gemini",
                places=[
                    {
                        "id": "verified",
                        "name_en": "Verified Place",
                        "name_ur": "",
                        "district": "Lahore",
                        "category": "Fort",
                        "source_url": "https://tourism.punjab.gov.pk/verified",
                        "verified": True,
                        "lat": 31.5,
                        "lng": 74.3,
                    },
                    {
                        "id": "unverified",
                        "name_en": "Unverified Place",
                        "name_ur": "",
                        "district": "Lahore",
                        "category": "Fort",
                        "source_url": "https://tourism.punjab.gov.pk/unverified",
                        "verified": False,
                        "lat": 31.6,
                        "lng": 74.4,
                    },
                ],
            ),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, result = self.get_path("/api/places", server)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(status, 200)
        self.assertEqual(result["places"][0]["latitude"], 31.5)
        self.assertEqual(result["places"][0]["longitude"], 74.3)
        self.assertIsNone(result["places"][1]["latitude"])
        self.assertIsNone(result["places"][1]["longitude"])
        self.assertFalse(result["places"][1]["verified"])

    def test_itinerary_uses_source_excerpt_and_rejects_unknown_place(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.corpus, DEFAULT_MODEL, "", "gemini"),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, result = self.post_itinerary(
                {"days": 1, "place_ids": ["derawar-fort"]}, server
            )
            bad_status, bad_result = self.post_itinerary(
                {"days": 1, "place_ids": ["unknown-place"]}, server
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        item = result["days"][0]["items"][0]
        self.assertEqual(status, 200)
        self.assertEqual(item["place_name"], "Derawar Fort")
        self.assertIn("Derawar Fort", item["description"])
        self.assertEqual(
            item["source"]["url"],
            "https://tourism.punjab.gov.pk/cholistan",
        )
        self.assertIn("opening hours, prices", result["note"])
        self.assertEqual(bad_status, 400)
        self.assertIn("unavailable", bad_result["error"])

    def test_itinerary_allows_free_days_and_avoids_generic_trip_page(self):
        self.corpus.documents.append(
            {
                "title": "Plan Your Trip | Tourism, Archaeology and Museums Department",
                "url": "https://tourism.punjab.gov.pk/plan-your-trip",
                "text": (
                    "Plan your next trip. Please head to the TDCP website "
                    "for further information: https://tdcp.gop.pk/tours. "
                    "8th Floor, Tower C, Old P&D Building, Lahore."
                ),
            }
        )
        status, result = self.post_itinerary(
            {"days": 2, "place_ids": ["lahore-fort"]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(result["days"]), 2)
        item = result["days"][0]["items"][0]
        self.assertEqual(item["place_name"], "Lahore Fort")
        self.assertIn("No dedicated Punjab Tourism information", item["description"])
        self.assertEqual(result["days"][1]["items"], [])
        self.assertNotIn("tdcp.gop.pk", item["source"]["url"])
        self.assertIn("google.com/maps/search", item["source"]["url"])

    def test_lahore_fort_image_fallback_has_no_generic_trip_source(self):
        status, result = self.post_question("show Lahore Fort pictures")
        self.assertEqual(status, 200)
        self.assertEqual(result["images"], [])
        self.assertEqual(result["sources"], [])

    def test_noor_mahal_image_request_filters_homepage_carousel_images(self):
        homepage = next(
            document
            for document in self.corpus.documents
            if document["url"] == "https://tourism.punjab.gov.pk/"
        )
        homepage["text"] = "Rohtas Fort. Badshahi Mosque. Derawar Fort. Noor Mahal."
        homepage["images"] = [
            {
                "url": "https://tourism.punjab.gov.pk/files/Banner_Rohtas_Fort.png",
                "alt": "",
            },
            {
                "url": "https://tourism.punjab.gov.pk/files/Banner_Badshahi_Mosque.png",
                "alt": "",
            },
            {
                "url": "https://tourism.punjab.gov.pk/files/Banner_Derawar_Fort.png",
                "alt": "",
            },
            {
                "url": "https://tourism.punjab.gov.pk/files/Banner_Noor_Mahal_0.png",
                "alt": "",
            },
        ]
        status, result = self.post_question("show Noor Mahal pictures")
        self.assertEqual(status, 200)
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0]["place_name"], "Noor Mahal")
        self.assertIn("Noor_Mahal", result["images"][0]["image_url"])

    def test_response_cache_avoids_repeating_provider_call_for_same_sources(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.corpus, DEFAULT_MODEL, "gemini-test-key", "gemini"),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with patch(
            "chatbot.call_gemini",
            return_value={
                "answer": "Derawar Fort is in Cholistan.",
                "sources": [
                    {
                        "title": "Cholistan - Gateway to Adventure",
                        "url": "https://tourism.punjab.gov.pk/cholistan",
                    }
                ],
            },
        ) as call_model:
            try:
                first_status, first = self.post_question(
                    "Tell me about Derawar Fort", server
                )
                second_status, second = self.post_question(
                    "tell me about derawar fort!", server
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
        self.assertEqual(first_status, 200)
        self.assertEqual(second_status, 200)
        self.assertEqual(first["answer"], second["answer"])
        call_model.assert_called_once()

    def test_unsupported_generated_claim_is_replaced_with_corpus_excerpt(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.corpus, DEFAULT_MODEL, "gemini-test-key", "gemini"),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch(
                "chatbot.call_gemini",
                return_value={
                    "answer": (
                        "Derawar Fort stands in the Cholistan Desert. "
                        "Entry costs 500 rupees."
                    ),
                    "sources": [
                        {
                            "title": "Cholistan",
                            "url": "https://tourism.punjab.gov.pk/cholistan",
                        }
                    ],
                },
            ):
                status, result = self.post_question("Tell me about Derawar Fort", server)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(status, 200)
        self.assertNotIn("500 rupees", result["answer"])
        self.assertEqual(result["answer_origin"], "grounded local excerpt")
        self.assertIn("replaced with a source excerpt", result["source_note"])

    def test_gemini_failure_uses_groq_before_local_fallback(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(
                self.corpus,
                DEFAULT_GEMINI_MODEL,
                "gemini-test-key",
                "gemini",
                provider_keys={
                    "gemini": "gemini-test-key",
                    "groq": "groq-test-key",
                },
                provider_models={
                    "gemini": DEFAULT_GEMINI_MODEL,
                    "groq": DEFAULT_GROQ_MODEL,
                },
            ),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with (
                patch(
                    "chatbot.call_gemini",
                    side_effect=RuntimeError("Gemini API returned HTTP 429"),
                ) as gemini,
                patch(
                    "chatbot.call_groq",
                    return_value={
                        "answer": "Derawar Fort is in Cholistan.",
                        "sources": [
                            {
                                "title": "Cholistan",
                                "url": "https://tourism.punjab.gov.pk/cholistan",
                            }
                        ],
                    },
                ) as groq,
            ):
                status, result = self.post_question(
                    "Tell me about Derawar Fort", server
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(status, 200)
        self.assertEqual(result["answer"], "Derawar Fort is in Cholistan.")
        self.assertIn("quota or rate limit", result["source_note"])
        self.assertIn("Groq", result["source_note"])
        gemini.assert_called_once()
        groq.assert_called_once()

    def test_unsupported_map_link_returns_clear_message_without_geocoding(self):
        with patch("chatbot.lookup_location") as lookup:
            status, result = self.post_question(
                "https://evil.example/maps?q=31,74"
            )
        self.assertEqual(status, 200)
        self.assertIn("Unsupported map link", result["answer"])
        lookup.assert_not_called()

    def test_general_question_without_tourism_match_uses_configured_provider(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.corpus, DEFAULT_MODEL, "test-key", "gemini"),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch(
                "chatbot.call_gemini",
                return_value={"answer": "A general knowledge answer.", "sources": []},
            ) as call_model:
                status, result = self.post_question("explain quantum computing", server)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(status, 200)
        self.assertEqual(result["answer"], "A general knowledge answer.")
        self.assertEqual(call_model.call_args.args[2], [])

    def test_external_answer_does_not_make_unsourced_hours_prices_or_distances(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.corpus, DEFAULT_MODEL, "test-key", "gemini"),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch(
                "chatbot.call_gemini",
                return_value={
                    "answer": (
                        "It is open daily from 9 AM to 5 PM, entry costs Rs 500, "
                        "and it is 3 km away."
                    ),
                    "sources": [],
                },
            ):
                status, result = self.post_question(
                    "Tell me about an unrelated attraction on Mars.", server
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(status, 200)
        self.assertEqual(
            result["answer"],
            "I can't provide specific opening hours, prices, or distances without a source.",
        )
        self.assertEqual(result["sources"], [])
        self.assertEqual(
            result["source_note"],
            "Unverified: This answer is not supported by the Punjab Tourism website, "
            "and no independent source was checked. Please verify it before relying "
            "on it; opening hours, prices, and distances will not be stated unless "
            "a source is cited.",
        )

    def test_follow_up_picture_request_returns_official_page_image(self):
        status, result = self.post_question(
            "iski pic dikhao",
            history=[
                {"role": "user", "content": "where is derawar fort"},
                {"role": "assistant", "content": "It is in the Cholistan Desert."},
            ],
        )
        self.assertEqual(status, 200)
        self.assertIn("official Punjab Tourism website", result["answer"])
        self.assertEqual(result["images"][0]["place_name"], "Derawar Fort")
        self.assertEqual(
            result["images"][0]["image_url"],
            "https://tourism.punjab.gov.pk/system/files/fort.png",
        )
        self.assertEqual(
            result["images"][0]["page_url"],
            "https://tourism.punjab.gov.pk/cholistan",
        )

    def test_badshahi_picture_request_only_returns_its_official_page_image(self):
        status, result = self.post_question("show badshai mosque pictures")
        self.assertEqual(status, 200)
        self.assertEqual(
            result["sources"][0]["url"],
            "https://tourism.punjab.gov.pk/badshahi-mosque",
        )
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0]["place_name"], "Badshahi Mosque")
        self.assertEqual(
            result["images"][0]["page_url"],
            "https://tourism.punjab.gov.pk/badshahi-mosque",
        )

    def test_unknown_landmark_does_not_send_incidental_corpus_matches_to_ai(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.corpus, DEFAULT_MODEL, "test-key", "gemini"),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch(
                "chatbot.call_gemini",
                return_value={"answer": "The Eiffel Tower is in Paris.", "sources": []},
            ) as call_model:
                status, result = self.post_question("Tell me about the Eiffel Tower", server)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(status, 200)
        self.assertEqual(call_model.call_args.args[2], [])
        self.assertEqual(result["answer_origin"], "external general knowledge")
        self.assertIn("Unverified:", result["source_note"])
        self.assertIn("no independent source was checked", result["source_note"])
        self.assertIn("google.com/search", result["external_search_url"])

    def test_provider_outage_uses_local_page_answer_instead_of_error(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.corpus, DEFAULT_MODEL, "test-key", "gemini"),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        log_stream = io.StringIO()
        try:
            with (
                patch("chatbot.call_gemini", side_effect=RuntimeError("provider offline")),
                patch("chatbot.sys.stderr", log_stream),
            ):
                status, result = self.post_question("where is derawar fort", server)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(status, 200)
        self.assertIn("middle of the Cholistan Desert", result["answer"])
        self.assertEqual(result["answer_origin"], "local fallback")
        self.assertIn("AI providers were unavailable", result["source_note"])
        self.assertIn("Provider gemini failed", log_stream.getvalue())
        self.assertNotIn("where is derawar fort", log_stream.getvalue())

    def test_uploaded_image_matches_indexed_tourism_image(self):
        status, result = self.post_image(self.sample_image)
        self.assertEqual(status, 200)
        self.assertIn("Cholistan - Gateway to Adventure", result["answer"])
        self.assertEqual(
            result["matches"][0]["image_url"],
            "https://tourism.punjab.gov.pk/system/files/fort.png",
        )
        self.assertEqual(
            result["matches"][0]["page_url"],
            "https://tourism.punjab.gov.pk/cholistan",
        )
        self.assertIn("Derawar Fort", result["answer"])
        self.assertEqual(result["matches"][0]["place_name"], "Derawar Fort")
        self.assertEqual(result["matches"][0]["confidence"], 100)

    def test_homepage_lahore_photo_has_specific_landmark_name(self):
        self.assertEqual(
            image_place_name(
                "Welcome to Tourism, Archaeology and Museums Department",
                "",
                "https://tourism.punjab.gov.pk/sites/tourism.punjab.gov.pk/files/lahore-1.jpg",
            ),
            "Minar-e-Pakistan, Lahore",
        )

    def test_generic_homepage_alt_uses_landmark_name_from_image_filename(self):
        self.assertEqual(
            image_place_name(
                "Welcome to Tourism, Archaeology and Museums Department",
                "Tourism, Archaeology and Museums Department",
                "https://tourism.punjab.gov.pk/sites/tourism.punjab.gov.pk/files/Banner%20_%20Rohtas%20Fort_3.png",
            ),
            "Rohtas Fort",
        )

    def test_resized_uploaded_image_still_matches(self):
        resized_buffer = BytesIO()
        with Image.open(BytesIO(self.sample_image)) as image:
            image.resize((320, 240)).save(resized_buffer, format="JPEG", quality=82)
        status, result = self.post_image(resized_buffer.getvalue())
        self.assertEqual(status, 200)
        self.assertTrue(result["matches"])

    def test_uploaded_non_image_is_rejected(self):
        status, result = self.post_image(b"not an image")
        self.assertEqual(status, 400)
        self.assertIn("valid supported image", result["error"])


class GroqResponseTests(unittest.TestCase):
    def test_unsourced_detail_guard_recognizes_english_urdu_and_roman_urdu(self):
        examples = (
            "It is open daily from 9 AM to 5 PM.",
            "The distance is 3 km.",
            "یہاں ٹکٹ 500 روپے ہے اور فاصلہ 2 کلومیٹر ہے۔",
            "Yahan ki qeemat 500 rupay hai aur fasla 3 km.",
        )
        for answer in examples:
            with self.subTest(answer=answer):
                self.assertTrue(has_unsourced_specific_details(answer))

    @patch("chatbot.urlopen")
    def test_answers_general_question_without_corpus_sources(self, mock_urlopen):
        response_body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "answer": "A general knowledge answer.",
                                        "source_ids": [],
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(response_body).encode()

        mock_urlopen.return_value = FakeResponse()
        result = call_gemini(
            "Explain quantum computing",
            [],
            [],
            api_key="test-gemini-key",
            model=DEFAULT_GEMINI_MODEL,
        )
        self.assertEqual(
            result,
            {"answer": "A general knowledge answer.", "sources": []},
        )
        request = mock_urlopen.call_args.args[0]
        prompt = json.loads(request.data)["systemInstruction"]["parts"][0]["text"]
        self.assertIn("general knowledge questions", prompt)

    @patch("chatbot.urlopen")
    def test_provider_error_detail_is_reported_without_leaking_api_key(self, mock_urlopen):
        secret = "test-key-123"
        body = json.dumps(
            {"error": {"message": f"Model access denied for {secret}."}}
        ).encode()
        mock_urlopen.side_effect = HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            403,
            "Forbidden",
            {},
            io.BytesIO(body),
        )

        with self.assertRaises(RuntimeError) as error:
            call_groq(
                "Tell me about tourism",
                [],
                [{"title": "Punjab", "url": "https://tourism.punjab.gov.pk/", "text": "Tourism"}],
                api_key=secret,
                model=DEFAULT_MODEL,
            )

        self.assertIn("HTTP 403", str(error.exception))
        self.assertIn("Model access denied", str(error.exception))
        self.assertIn("revoke it and configure a new key", str(error.exception))
        self.assertNotIn(secret, str(error.exception))

    @patch("chatbot.urlopen")
    def test_gemini_request_uses_configured_model_and_returns_cited_answer(self, mock_urlopen):
        response_body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "answer": "Murree is a scenic hill station.",
                                        "source_ids": [1],
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(response_body).encode()

        mock_urlopen.return_value = FakeResponse()
        result = call_gemini(
            "Tell me about Murree",
            [],
            [
                {
                    "title": "Murree",
                    "url": "https://tourism.punjab.gov.pk/murree",
                    "text": "Murree is a scenic hill station.",
                }
            ],
            api_key="test-gemini-key",
            model=DEFAULT_GEMINI_MODEL,
        )

        self.assertEqual(result["answer"], "Murree is a scenic hill station.")
        self.assertEqual(
            result["sources"],
            [{"title": "Murree", "url": "https://tourism.punjab.gov.pk/murree"}],
        )
        request = mock_urlopen.call_args.args[0]
        self.assertIn(f"/{DEFAULT_GEMINI_MODEL}:generateContent", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "test-gemini-key")
        self.assertEqual(json.loads(request.data)["generationConfig"]["responseMimeType"], "application/json")

    @patch("chatbot.urlopen")
    def test_gemini_provider_error_redacts_api_key(self, mock_urlopen):
        secret = "AIzaSyTestFakeKey_123456789012345"
        body = json.dumps(
            {"error": {"message": f"API key {secret} is not allowed."}}
        ).encode()
        mock_urlopen.side_effect = HTTPError(
            f"{GEMINI_API_URL}/model:generateContent",
            403,
            "Forbidden",
            {},
            io.BytesIO(body),
        )

        with self.assertRaises(RuntimeError) as error:
            call_gemini(
                "Tell me about tourism",
                [],
                [
                    {
                        "title": "Punjab",
                        "url": "https://tourism.punjab.gov.pk/",
                        "text": "Tourism",
                    }
                ],
                api_key=secret,
                model=DEFAULT_GEMINI_MODEL,
            )

        self.assertIn("Gemini API returned HTTP 403", str(error.exception))
        self.assertNotIn(secret, str(error.exception))

    @patch("chatbot.time.sleep")
    @patch("chatbot.urlopen")
    def test_gemini_retries_once_after_temporary_unavailability(
        self, mock_urlopen, mock_sleep
    ):
        response_body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {"answer": "Punjab tourism.", "source_ids": [1]}
                                )
                            }
                        ]
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(response_body).encode()

        mock_urlopen.side_effect = [
            HTTPError(
                f"{GEMINI_API_URL}/model:generateContent",
                503,
                "Service Unavailable",
                {},
                io.BytesIO(b'{"error":{"message":"High demand"}}'),
            ),
            FakeResponse(),
        ]

        result = call_gemini(
            "Tell me about tourism",
            [],
            [{"title": "Punjab", "url": "https://tourism.punjab.gov.pk/", "text": "Tourism"}],
            api_key="test-key",
            model=DEFAULT_GEMINI_MODEL,
        )

        self.assertEqual(result["answer"], "Punjab tourism.")
        self.assertEqual(mock_urlopen.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("chatbot.time.sleep")
    @patch("chatbot.urlopen")
    def test_gemini_quota_error_is_not_retried(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = HTTPError(
            f"{GEMINI_API_URL}/{DEFAULT_GEMINI_MODEL}:generateContent",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b'{"error":{"message":"Quota exceeded"}}'),
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            call_gemini(
                "Tell me about tourism",
                [],
                [],
                api_key="test-key",
                model=DEFAULT_GEMINI_MODEL,
            )
        mock_urlopen.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("chatbot.time.sleep")
    @patch("chatbot.urlopen")
    def test_gemini_tries_each_fallback_after_retries(
        self, mock_urlopen, _mock_sleep
    ):
        response_body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {"answer": "Punjab tourism.", "source_ids": [1]}
                                )
                            }
                        ]
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(response_body).encode()

        unavailable = lambda: HTTPError(
            f"{GEMINI_API_URL}/model:generateContent",
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b'{"error":{"message":"High demand"}}'),
        )
        mock_urlopen.side_effect = [
            unavailable(),
            unavailable(),
            unavailable(),
            unavailable(),
            FakeResponse(),
        ]

        result = call_gemini(
            "Tell me about tourism",
            [],
            [{"title": "Punjab", "url": "https://tourism.punjab.gov.pk/", "text": "Tourism"}],
            api_key="test-key",
            model=DEFAULT_GEMINI_MODEL,
        )

        self.assertEqual(result["answer"], "Punjab tourism.")
        requested_models = [
            request.args[0].full_url.split("/models/")[1].split(":")[0]
            for request in mock_urlopen.call_args_list
        ]
        self.assertEqual(
            requested_models,
            [
                DEFAULT_GEMINI_MODEL,
                DEFAULT_GEMINI_MODEL,
                GEMINI_FALLBACK_MODELS[0],
                GEMINI_FALLBACK_MODELS[0],
                GEMINI_FALLBACK_MODELS[1],
            ],
        )

    @patch("chatbot.urlopen")
    def test_only_returns_sources_from_retrieved_corpus(self, mock_urlopen):
        response_body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"answer": "Murree is a hill station.", "source_ids": [1, 999]}
                        )
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(response_body).encode()

        mock_urlopen.return_value = FakeResponse()
        result = call_groq(
            "Where is Murree?",
            [],
            [{"title": "Murree", "url": "https://tourism.punjab.gov.pk/murree", "text": "Hill station"}],
            api_key="test-key",
            model="test-model",
        )

        self.assertEqual(result["answer"], "Murree is a hill station.")
        self.assertEqual(
            result["sources"],
            [{"title": "Murree", "url": "https://tourism.punjab.gov.pk/murree"}],
        )
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(json.loads(request.data)["model"], "test-model")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")

    @patch("chatbot.urlopen")
    def test_empty_groq_key_skips_request_without_logging_sensitive_data(
        self, mock_urlopen
    ):
        log_stream = io.StringIO()
        with (
            patch("chatbot.sys.stderr", log_stream),
            self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY is not configured"),
        ):
            call_groq(
                "private question text",
                [],
                [],
                api_key="",
                model="test-model",
            )

        mock_urlopen.assert_not_called()
        self.assertIn("groq", log_stream.getvalue().casefold())
        self.assertNotIn("private question text", log_stream.getvalue())
        self.assertNotIn("test-key", log_stream.getvalue())

    @patch("chatbot.urlopen")
    def test_unreferenced_model_answer_is_filtered(self, mock_urlopen):
        response_body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"answer": "The unverified answer.", "source_ids": []}
                        )
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(response_body).encode()

        mock_urlopen.return_value = FakeResponse()
        result = call_groq(
            "Question?",
            [],
            [{"title": "Page", "url": "https://tourism.punjab.gov.pk/page", "text": "Facts"}],
            api_key="test-key",
            model="test-model",
        )
        self.assertEqual(result, {"answer": NO_ANSWER, "sources": []})


if __name__ == "__main__":
    unittest.main()
