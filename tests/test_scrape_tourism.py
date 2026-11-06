import json
from http.client import IncompleteRead
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from scrape_tourism import (
    PageParser,
    SameHostRedirectHandler,
    build_parser,
    crawl,
    fetch_page,
    load_robots,
    normalize_url,
)


class PageParserTests(unittest.TestCase):
    def test_extracts_content_and_ignores_scripts(self):
        parser = PageParser()
        parser.feed(
            "<html><head><title> Visit Punjab </title>"
            '<meta name="description" content=" Official tourism information ">'
            "</head><body><main><h1>Explore Punjab</h1>"
            "<p>Discover <b>heritage</b> and nature.</p>"
            "<script>secret()</script><a href='/murree'>Murree</a>"
            '<img src="/system/files/murree.jpg" alt="Murree hills">'
            '<img srcset="/small.jpg 320w, /large.jpg 1000w" alt="Valley">'
            "</main></body></html>"
        )

        document = parser.document()

        self.assertEqual(document["title"], "Visit Punjab")
        self.assertEqual(document["description"], "Official tourism information")
        self.assertEqual(document["headings"], [{"level": "h1", "text": "Explore Punjab"}])
        self.assertIn("Discover heritage and nature.", document["text"])
        self.assertNotIn("secret", document["text"])
        self.assertEqual(parser.links, ["/murree"])
        self.assertEqual(
            document["images"],
            [
                {"src": "/system/files/murree.jpg", "alt": "Murree hills"},
                {"src": "/large.jpg", "alt": "Valley"},
            ],
        )

    def test_page_fetch_retries_an_incomplete_http_response(self):
        class FakeHeaders:
            def get_content_type(self):
                return "text/html"

            def get_content_charset(self):
                return "utf-8"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def geturl(self):
                return "https://tourism.punjab.gov.pk/"

            def read(self, _limit):
                return b"<html><body>Punjab tourism</body></html>"

        class FakeOpener:
            def __init__(self):
                self.attempts = 0

            def open(self, _request, timeout):
                self.attempts += 1
                if self.attempts == 1:
                    raise IncompleteRead(b"partial")
                return FakeResponse()

        opener = FakeOpener()
        with (
            patch("scrape_tourism.build_opener", return_value=opener),
            patch("scrape_tourism.time.sleep"),
        ):
            result = fetch_page(
                "https://tourism.punjab.gov.pk/",
                "test-agent",
                "tourism.punjab.gov.pk",
                3,
            )

        self.assertEqual(opener.attempts, 2)
        self.assertEqual(
            result,
            (
                "https://tourism.punjab.gov.pk/",
                "<html><body>Punjab tourism</body></html>",
            ),
        )

    def test_normalize_url_keeps_only_same_site_https_html_paths(self):
        host = "tourism.punjab.gov.pk"
        self.assertEqual(
            normalize_url("https://tourism.punjab.gov.pk/murree/?utm_source=x#top", host),
            "https://tourism.punjab.gov.pk/murree",
        )
        self.assertIsNone(normalize_url("https://example.com/murree", host))
        self.assertIsNone(normalize_url("http://tourism.punjab.gov.pk/murree", host))
        self.assertIsNone(normalize_url("https://tourism.punjab.gov.pk/file.pdf", host))

    def test_normalize_image_url_requires_https_same_host_raster_image(self):
        from scrape_tourism import normalize_image_url

        host = "tourism.punjab.gov.pk"
        self.assertEqual(
            normalize_image_url(
                "https://tourism.punjab.gov.pk/system/files/photo.jpg?itok=abc#top",
                host,
            ),
            "https://tourism.punjab.gov.pk/system/files/photo.jpg?itok=abc",
        )
        self.assertIsNone(normalize_image_url("http://tourism.punjab.gov.pk/photo.jpg", host))
        self.assertIsNone(normalize_image_url("https://example.com/photo.jpg", host))
        self.assertIsNone(normalize_image_url("https://tourism.punjab.gov.pk/logo.svg", host))


class CrawlOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output = Path(self.temp_dir.name) / "corpus.jsonl"
        self.base_url = "https://tourism.punjab.gov.pk/"
        self.args = build_parser().parse_args(
            ["--base-url", self.base_url, "--output", str(self.output),
             "--max-pages", "2", "--delay", "0"]
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_failed_crawl_preserves_existing_corpus_and_cleans_temporary_output(self):
        original = b'{"previous":"corpus"}\n'
        self.output.write_bytes(original)
        with (
            patch("scrape_tourism.load_robots"),
            patch(
                "scrape_tourism.fetch_page",
                side_effect=[
                    (
                        self.base_url,
                        '<html><body><a href="/second">Next</a></body></html>',
                    ),
                    RuntimeError("simulated crawl failure"),
                ],
            ),
            patch("scrape_tourism.text_embedding", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "simulated crawl failure"),
        ):
            crawl(self.args)

        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_successful_crawl_atomically_replaces_existing_corpus(self):
        self.output.write_bytes(b"old corpus\n")
        with (
            patch("scrape_tourism.load_robots"),
            patch(
                "scrape_tourism.fetch_page",
                return_value=(
                    self.base_url,
                    "<html><head><title>Tourism</title></head>"
                    "<body><p>Punjab tourism.</p></body></html>",
                ),
            ),
            patch("scrape_tourism.text_embedding", return_value=[]),
        ):
            self.assertEqual(crawl(self.args), 0)

        records = self.output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0])["text"], "Punjab tourism.")
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_empty_crawl_preserves_existing_corpus(self):
        original = b'{"previous":"corpus"}\n'
        self.output.write_bytes(original)
        with (
            patch("scrape_tourism.load_robots"),
            patch("scrape_tourism.fetch_page", return_value=None),
            self.assertRaisesRegex(RuntimeError, "produced no HTML pages"),
        ):
            crawl(self.args)

        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])


class RobotsRedirectTests(unittest.TestCase):
    def test_html_robots_response_is_rejected_unless_authorized_override_is_set(self):
        class FakeHeaders:
            def get_content_type(self):
                return "text/html"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return b"<!doctype html><html><body>Application shell</body></html>"

        class FakeOpener:
            def open(self, _request, timeout):
                return FakeResponse()

        with patch("scrape_tourism.build_opener", return_value=FakeOpener()):
            with self.assertRaisesRegex(RuntimeError, "HTML page instead of a crawl policy"):
                load_robots("https://tdcp.gop.pk/", "test-agent", False, 3)

        with patch("scrape_tourism.build_opener", return_value=FakeOpener()):
            parser = load_robots("https://tdcp.gop.pk/", "test-agent", True, 3)
        self.assertTrue(parser.can_fetch("test-agent", "https://tdcp.gop.pk/"))

    def test_robots_fetch_uses_same_host_https_redirect_policy(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return b"User-agent: *\nDisallow: /private\n"

        class FakeOpener:
            def open(self, _request, timeout):
                self.timeout = timeout
                return FakeResponse()

        opener = FakeOpener()
        with patch("scrape_tourism.build_opener", return_value=opener) as build:
            parser = load_robots(
                "https://tourism.punjab.gov.pk/",
                "test-agent",
                False,
                3,
            )

        handler = build.call_args.args[0]
        self.assertIsInstance(handler, SameHostRedirectHandler)
        self.assertEqual(handler.allowed_host, "tourism.punjab.gov.pk")
        self.assertEqual(opener.timeout, 3)
        self.assertFalse(parser.can_fetch("test-agent", "https://tourism.punjab.gov.pk/private"))

    def test_robots_redirect_blocks_other_hosts_and_insecure_targets(self):
        handler = SameHostRedirectHandler("tourism.punjab.gov.pk")
        request = Request("https://tourism.punjab.gov.pk/robots.txt")

        same_host = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://tourism.punjab.gov.pk/robots-v2.txt",
        )
        self.assertEqual(
            same_host.full_url,
            "https://tourism.punjab.gov.pk/robots-v2.txt",
        )
        for target in (
            "https://example.com/robots.txt",
            "http://tourism.punjab.gov.pk/robots.txt",
        ):
            with self.subTest(target=target), self.assertRaises(HTTPError):
                handler.redirect_request(request, None, 302, "Found", {}, target)


if __name__ == "__main__":
    unittest.main()
