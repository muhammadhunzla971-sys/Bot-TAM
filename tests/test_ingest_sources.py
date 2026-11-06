from pathlib import Path
import tempfile
import unittest

from ingest_sources import merge_records, normalize_dts_url, read_sites


class SourceListTests(unittest.TestCase):
    def test_dts_http_normalizer_accepts_only_the_user_listed_host(self):
        self.assertEqual(
            normalize_dts_url("http://dts.gov.pk/services/?ref=home#top"),
            "http://dts.gov.pk/services",
        )
        self.assertIsNone(normalize_dts_url("https://dts.gov.pk/services"))
        self.assertIsNone(normalize_dts_url("http://example.com/services"))

    def test_reads_sites_and_upgrades_http_seeds_to_https(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.md"
            path.write_text(
                "# Official sites\nhttps://tourism.punjab.gov.pk/\n"
                "http://dts.gov.pk/\nhttps://tourism.punjab.gov.pk/\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_sites(path),
                ["https://tourism.punjab.gov.pk/", "http://dts.gov.pk/"],
            )

    def test_merge_updates_rediscovered_urls_and_preserves_other_records(self):
        existing = [
            {
                "url": "https://tourism.punjab.gov.pk/old",
                "images": [{"url": "https://tourism.punjab.gov.pk/photo.jpg", "embedding": [1, 2]}],
                "text_embedding": [3, 4],
            },
            {"url": "https://tourism.punjab.gov.pk/still-valid"},
            {"url": "https://dts.gov.pk/old"},
        ]
        updates = [
            {"url": "https://tourism.punjab.gov.pk/old", "text": "fresh", "images": []},
        ]
        merged = merge_records(existing, updates)
        self.assertEqual(
            [record.get("text", "old") for record in merged],
            ["fresh", "old", "old"],
        )
        self.assertEqual(merged[0]["images"][0]["embedding"], [1, 2])
        self.assertEqual(merged[0]["text_embedding"], [3, 4])


if __name__ == "__main__":
    unittest.main()
