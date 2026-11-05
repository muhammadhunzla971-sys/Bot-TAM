from pathlib import Path
import re
from threading import Thread
import unittest
from http.server import ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]

try:
    from playwright.sync_api import sync_playwright
    from playwright._impl._driver import compute_driver_executable
except ImportError:
    sync_playwright = None
    compute_driver_executable = None


class FrontendSafetyTests(unittest.TestCase):
    def test_dynamic_map_popup_uses_text_content_not_html(self):
        source = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        payload = '<img src=x onerror=alert(1)>'

        self.assertIn("const popup = document.createElement(\"span\");", source)
        self.assertIn(
            "popup.textContent = `${name.textContent} (${location.precision})`;",
            source,
        )
        self.assertIn(".bindPopup(popup);", source)
        self.assertNotRegex(source, r"\.innerHTML\s*=")
        self.assertNotRegex(source, r"\.insertAdjacentHTML\s*\(")
        self.assertNotRegex(source, r"\.bindPopup\(\s*[`'\"]")
        self.assertIn("Unverified answer (no cited source)", source)
        self.assertNotIn(payload, source)

    def test_nearby_search_uses_labeled_browser_side_photon_fallback(self):
        source = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('new URL("https://photon.komoot.io/api/")', source)
        self.assertIn("properties.countrycode", source)
        self.assertIn("Your location stays in this browser.", source)
        self.assertIn("not verified site coordinates", source)
        self.assertIn("Approximate OpenStreetMap location", source)

    def test_itinerary_days_are_independent_of_selected_place_count(self):
        source = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("updateTripDayOptions", source)
        self.assertIn("Extra days are left open;", source)
        self.assertIn("No selected stops — keep this day free for rest or travel.", source)

    def test_dashboard_layout_adapts_to_mobile_and_keeps_keyboard_focus_visible(self):
        source = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);", source)
        self.assertIn("@media (max-width: 700px)", source)
        self.assertIn("grid-template-columns: minmax(0, 1fr);", source)
        self.assertIn(":focus-visible", source)
        self.assertIn('class="brand-mark" aria-hidden="true"', source)


@unittest.skipIf(sync_playwright is None, "Playwright is not installed.")
class ResponsiveViewportSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from chatbot import Corpus, DEFAULT_DATA, DEFAULT_GEMINI_MODEL, make_handler

        if compute_driver_executable is not None:
            node_executable, _ = compute_driver_executable()
            if not Path(node_executable).is_file():
                raise unittest.SkipTest(
                    "Playwright's bundled Node runtime is missing."
                )
        try:
            cls.playwright = sync_playwright().start()
            cls.browser = cls.playwright.chromium.launch(headless=True)
        except Exception as error:
            if hasattr(cls, "playwright"):
                cls.playwright.stop()
            raise unittest.SkipTest(
                f"Playwright browser runtime could not start: {error}"
            ) from error
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(Corpus(DEFAULT_DATA), DEFAULT_GEMINI_MODEL, "", "gemini"),
        )
        cls.server_thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "server"):
            cls.server.shutdown()
            cls.server.server_close()
            cls.server_thread.join()
        if hasattr(cls, "browser"):
            cls.browser.close()
        if hasattr(cls, "playwright"):
            cls.playwright.stop()

    def test_chat_and_map_fit_common_viewports_without_horizontal_overflow(self):
        viewports = ((390, 844), (768, 1024), (1024, 500), (1440, 900))
        for width, height in viewports:
            with self.subTest(viewport=(width, height)):
                page = self.browser.new_page(viewport={"width": width, "height": height})
                page.route(
                    "**/*",
                    lambda route: route.continue_()
                    if route.request.url.startswith(self.base_url)
                    else route.abort(),
                )
                response = page.goto(self.base_url, wait_until="domcontentloaded")
                self.assertIsNotNone(response)
                self.assertEqual(response.status, 200)
                self.assertTrue(
                    page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth"
                    ),
                    f"Horizontal overflow at {width}x{height}",
                )
                for selector in ("#question", "#site-map"):
                    element = page.locator(selector)
                    self.assertTrue(element.is_visible(), f"{selector} hidden at {width}x{height}")
                    bounds = element.bounding_box()
                    self.assertIsNotNone(bounds)
                    self.assertGreater(bounds["height"], 0)
                    self.assertGreaterEqual(bounds["y"], 0)
                    self.assertLessEqual(
                        bounds["y"] + bounds["height"],
                        height,
                        f"{selector} extends below viewport at {width}x{height}",
                    )
                if width == 390:
                    popup = page.evaluate(
                        """() => {
                          const payload = '<img src=x onerror=alert(1)>';
                          window.L = {
                            map: () => ({setView() { return this; }}),
                            tileLayer: () => ({
                              on() {},
                              addTo() {}
                            }),
                            marker: () => ({
                              addTo() { return this; },
                              bindPopup(content) {
                                window.__capturedPopup = content;
                                return this;
                              }
                            })
                          };
                          addMessage('assistant', '', [], [], {
                            status: 'found',
                            latitude: 31,
                            longitude: 74,
                            precision: 'approximate',
                            label: payload
                          });
                          return {
                            text: window.__capturedPopup.textContent,
                            containsImage: Boolean(
                              window.__capturedPopup.querySelector('img')
                            )
                          };
                        }"""
                    )
                    self.assertEqual(
                        popup["text"],
                        '<img src=x onerror=alert(1)> (approximate)',
                    )
                    self.assertFalse(popup["containsImage"])
                page.close()


if __name__ == "__main__":
    unittest.main()
