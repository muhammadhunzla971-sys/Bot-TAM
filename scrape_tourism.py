#!/usr/bin/env python3
"""Crawl public HTML pages on the Punjab Tourism website into JSONL."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from http.client import IncompleteRead, RemoteDisconnected
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from urllib.robotparser import RobotFileParser
from typing import TextIO

from image_matching import (
    difference_hash,
    image_label_text,
    text_embedding,
    visual_embedding,
)


DEFAULT_BASE_URL = "https://tourism.punjab.gov.pk/"
DEFAULT_USER_AGENT = (
    "PunjabTourismChatbotDataCollector/1.0 "
    "(public-page indexing; contact: site administrator)"
)
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
PAGE_FETCH_ATTEMPTS = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
BLOCKED_EXTENSIONS = {
    ".7z", ".avi", ".bmp", ".css", ".csv", ".doc", ".docx", ".gif",
    ".gz", ".ico", ".jpeg", ".jpg", ".js", ".json", ".mp3", ".mp4",
    ".pdf", ".png", ".ppt", ".pptx", ".rar", ".svg", ".tar", ".webp", ".xls",
    ".xlsx", ".xml", ".zip",
}
IGNORED_TAGS = {"script", "style", "noscript", "svg", "template"}
BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
    "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2",
    "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol",
    "p", "section", "table", "td", "th", "tr", "ul",
}


@contextmanager
def atomic_text_output(output: Path) -> Iterator[TextIO]:
    temporary_output = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_output = Path(file.name)
            yield file
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_output, output)
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class PageParser(HTMLParser):
    """Extract useful text and same-site links without retaining markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.description = ""
        self.canonical = ""
        self.headings: list[dict[str, str]] = []
        self.body_parts: list[str] = []
        self.links: list[str] = []
        self.images: list[dict[str, str]] = []
        self._active_heading: dict[str, str] | None = None
        self._in_title = False
        self._in_head = False
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        if tag == "head":
            self._in_head = True
            return
        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return

        if tag == "title":
            self._in_title = True
        elif tag == "img":
            source = (
                attributes.get("data-src")
                or attributes.get("data-lazy-src")
                or attributes.get("data-original")
                or attributes.get("src")
            )
            if attributes.get("data-srcset") or attributes.get("srcset"):
                candidates = []
                for candidate in (attributes.get("data-srcset") or attributes["srcset"]).split(","):
                    parts = candidate.strip().split()
                    if parts:
                        width = re.match(r"(\d+)w$", parts[-1]) if len(parts) > 1 else None
                        candidates.append((int(width.group(1)) if width else 0, parts[0]))
                if candidates:
                    source = max(candidates)[1]
            if source:
                self.images.append(
                    {
                        "src": source,
                        "alt": clean_text(attributes.get("alt", "")),
                    }
                )
        elif tag == "meta" and attributes.get("name", "").lower() == "description":
            self.description = clean_text(attributes.get("content", ""))
        elif tag == "link" and "canonical" in attributes.get("rel", "").lower().split():
            self.canonical = attributes.get("href", "")
        elif re.fullmatch(r"h[1-6]", tag):
            self._active_heading = {"level": tag, "text": ""}
            self.headings.append(self._active_heading)
        elif tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])

        if tag in BLOCK_TAGS:
            self.body_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._in_head = False
            return
        if tag in IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
        elif re.fullmatch(r"h[1-6]", tag):
            self._active_heading = None
        if tag in BLOCK_TAGS:
            self.body_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        if self._in_head:
            return
        text = clean_text(data)
        if text:
            self.body_parts.append(text + " ")
            if self._active_heading is not None:
                self._active_heading["text"] = clean_text(
                    self._active_heading["text"] + " " + text
                )

    def document(self) -> dict[str, object]:
        body = re.sub(r"[ \t]+\n", "\n", "".join(self.body_parts))
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        headings = [
            {"level": heading["level"], "text": heading["text"]}
            for heading in self.headings
            if heading["text"]
        ]
        return {
            "title": clean_text(" ".join(self.title_parts)),
            "description": self.description,
            "headings": headings,
            "text": body,
            "images": self.images,
        }


class SameHostRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_host: str) -> None:
        super().__init__()
        self.allowed_host = allowed_host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(urljoin(req.full_url, newurl))
        if target.scheme != "https" or target.hostname != self.allowed_host:
            raise HTTPError(newurl, code, "Cross-site or insecure redirect blocked", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def normalize_url(url: str, allowed_host: str) -> str | None:
    parts = urlsplit(urldefrag(url)[0])
    if parts.scheme != "https" or parts.hostname != allowed_host or parts.username:
        return None
    path = parts.path or "/"
    suffix = Path(path).suffix.lower()
    if suffix in BLOCKED_EXTENSIONS:
        return None
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(("https", allowed_host, path, "", ""))


def normalize_image_url(url: str, allowed_host: str) -> str | None:
    parts = urlsplit(urldefrag(url)[0])
    if (
        parts.scheme != "https"
        or parts.hostname != allowed_host
        or parts.username
        or parts.password
        or Path(parts.path).suffix.lower() not in IMAGE_EXTENSIONS
    ):
        return None
    return urlunsplit(("https", allowed_host, parts.path, parts.query, ""))


def load_robots(
    base_url: str,
    user_agent: str,
    allow_unavailable: bool,
    timeout: int,
    bypass_proxy: bool = False,
) -> RobotFileParser:
    robots_url = urljoin(base_url, "/robots.txt")
    robots_parts = urlsplit(robots_url)
    if robots_parts.scheme != "https" or not robots_parts.hostname:
        raise RuntimeError("robots.txt URL must be HTTPS and include a host.")
    parser = RobotFileParser(robots_url)
    request = Request(robots_url, headers={"User-Agent": user_agent})
    try:
        opener = build_opener(
            *(
                [ProxyHandler({}), SameHostRedirectHandler(robots_parts.hostname.lower())]
                if bypass_proxy
                else [SameHostRedirectHandler(robots_parts.hostname.lower())]
            )
        )
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(256 * 1024)
            headers = getattr(response, "headers", None)
            content_type = (
                headers.get_content_type().lower()
                if headers is not None and hasattr(headers, "get_content_type")
                else ""
            )
            if content_type == "text/html" or payload.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                if not allow_unavailable:
                    raise RuntimeError(
                        "robots.txt returned an HTML page instead of a crawl policy; "
                        "the crawler stopped. Only use --allow-robots-unavailable "
                        "with site-owner authorization."
                    )
                print(
                    "WARNING: robots.txt returned HTML; proceeding only because "
                    "--allow-robots-unavailable was supplied. Verify site-owner authorization.",
                    file=sys.stderr,
                )
                parser.parse([])
            else:
                parser.parse(payload.decode("utf-8", "replace").splitlines())
    except HTTPError as error:
        if error.code == 404:
            parser.parse([])
        elif error.code == 403 and allow_unavailable:
            print(
                "WARNING: robots.txt returned 403; proceeding only because "
                "--allow-robots-unavailable was supplied. Verify site-owner authorization.",
                file=sys.stderr,
            )
            parser.parse([])
        elif error.code == 403:
            raise RuntimeError(
                "robots.txt returned 403. The crawler will stop by default; "
                "only use --allow-robots-unavailable if you administer the site "
                "and have authorization to crawl it."
            ) from error
        else:
            raise RuntimeError(
                f"Could not read robots.txt safely (HTTP {error.code})."
            ) from error
    except (TimeoutError, URLError) as error:
        raise RuntimeError(f"Could not read robots.txt: {error}") from error
    return parser


def fetch_page(
    url: str, user_agent: str, allowed_host: str, timeout: int,
    bypass_proxy: bool = False,
) -> tuple[str, str] | None:
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        },
    )
    last_error: IncompleteRead | RemoteDisconnected | None = None
    for attempt in range(PAGE_FETCH_ATTEMPTS):
        opener = build_opener(
            *( [ProxyHandler({}), SameHostRedirectHandler(allowed_host)]
               if bypass_proxy else [SameHostRedirectHandler(allowed_host)] )
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                final_url = normalize_url(response.geturl(), allowed_host)
                if final_url is None:
                    raise RuntimeError(
                        f"Redirected to a URL outside the allowed site: {response.geturl()}"
                    )
                content_type = response.headers.get_content_type().lower()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    return None
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise RuntimeError(f"Page exceeded the 5 MiB response limit: {url}")
                charset = response.headers.get_content_charset() or "utf-8"
                return final_url, payload.decode(charset, "replace")
        except (IncompleteRead, RemoteDisconnected) as error:
            last_error = error
            if attempt + 1 >= PAGE_FETCH_ATTEMPTS:
                raise
            time.sleep(0.25 * (2 ** attempt))
    if last_error is not None:
        raise last_error
    return None


def fetch_image(
        url: str, user_agent: str, allowed_host: str, timeout: int,
        bypass_proxy: bool = False,
) -> tuple[str, bytes] | None:
        opener = build_opener(
            *( [ProxyHandler({}), SameHostRedirectHandler(allowed_host)]
               if bypass_proxy else [SameHostRedirectHandler(allowed_host)] )
        )
        request = Request(url, headers={"User-Agent": user_agent, "Accept": "image/*"})
        with opener.open(request, timeout=timeout) as response:
            final_url = normalize_image_url(response.geturl(), allowed_host)
            if final_url is None:
                raise RuntimeError(f"Image redirected to a non-image or outside URL: {url}")
            if response.headers.get_content_maintype().lower() != "image":
                return None
            payload = response.read(MAX_IMAGE_BYTES + 1)
            if len(payload) > MAX_IMAGE_BYTES:
                raise RuntimeError(f"Image exceeded the 10 MiB response limit: {url}")
            return final_url, payload


def crawl(args: argparse.Namespace) -> int:
    base_parts = urlsplit(args.base_url)
    if base_parts.scheme != "https" or not base_parts.hostname:
        raise RuntimeError("--base-url must be an HTTPS URL.")
    if base_parts.username or base_parts.password:
        raise RuntimeError("--base-url must not include credentials.")
    host = base_parts.hostname.lower()
    start_url = normalize_url(args.base_url, host)
    if start_url is None:
        raise RuntimeError("--base-url is not a valid HTTPS URL.")

    robots = load_robots(
        args.base_url, args.user_agent, args.allow_robots_unavailable, args.timeout,
        args.no_proxy,
    )
    queue = deque([start_url])
    queued = {start_url}
    visited: set[str] = set()
    written = 0
    indexed_images = 0
    image_features: dict[str, tuple[str, list[int]]] = {}
    last_request = 0.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with atomic_text_output(output) as file:
        while queue and written < args.max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            if not robots.can_fetch(args.user_agent, url):
                print(f"Blocked by robots.txt: {url}", file=sys.stderr)
                continue

            wait = args.delay - (time.monotonic() - last_request)
            if last_request and wait > 0:
                time.sleep(wait)
            last_request = time.monotonic()
            try:
                fetched = fetch_page(url, args.user_agent, host, args.timeout, args.no_proxy)
            except HTTPError as error:
                if error.code == 404:
                    print(f"Not found (skipping): {url}", file=sys.stderr)
                    continue
                raise RuntimeError(f"Page request failed (HTTP {error.code}): {url}") from error
            except (IncompleteRead, RemoteDisconnected) as error:
                print(
                    f"Incomplete response after {PAGE_FETCH_ATTEMPTS} attempts (skipping): {url}: {error}",
                    file=sys.stderr,
                )
                continue
            except (TimeoutError, URLError) as error:
                raise RuntimeError(f"Page request failed for {url}: {error}") from error
            if fetched is None:
                print(f"Non-HTML page (skipping): {url}", file=sys.stderr)
                continue

            final_url, html = fetched
            page = PageParser()
            page.feed(html)
            document = page.document()
            image_records: list[dict[str, str]] = []
            title = str(document.get("title") or final_url)
            if not args.skip_text_embeddings:
                document["text_embedding"] = text_embedding(
                    f"A photograph of {title}. {document.get('description', '')}"
                )
            seen_images: set[str] = set()
            for image in document["images"]:
                image_url = normalize_image_url(
                    urljoin(final_url, image["src"]), host
                )
                if (
                    not image_url
                    or image_url in seen_images
                    or not robots.can_fetch(args.user_agent, image_url)
                ):
                    continue
                seen_images.add(image_url)
                if image_url in image_features:
                    image_hash, image_embedding = image_features[image_url]
                    image_records.append(
                        {
                            "url": image_url,
                            "alt": image["alt"],
                            "dhash": image_hash,
                            "embedding": image_embedding,
                            "label_embedding": text_embedding(
                                image_label_text(title, image["alt"], image_url)
                            ),
                        }
                    )
                    continue
                if indexed_images >= args.max_images:
                    continue
                wait = args.delay - (time.monotonic() - last_request)
                if last_request and wait > 0:
                    time.sleep(wait)
                last_request = time.monotonic()
                try:
                    fetched_image = fetch_image(
                        image_url, args.user_agent, host, args.timeout, args.no_proxy
                    )
                    if fetched_image is None:
                        continue
                    final_image_url, image_bytes = fetched_image
                    image_records.append(
                        {
                            "url": final_image_url,
                            "alt": image["alt"],
                            "dhash": difference_hash(image_bytes),
                            "embedding": visual_embedding(image_bytes),
                            "label_embedding": text_embedding(
                                image_label_text(title, image["alt"], final_image_url)
                            ),
                        }
                    )
                    features = (
                        image_records[-1]["dhash"],
                        image_records[-1]["embedding"],
                    )
                    image_features[image_url] = features
                    image_features[final_image_url] = features
                    indexed_images += 1
                except (HTTPError, OSError, RuntimeError, TimeoutError, ValueError) as error:
                    print(f"Image skipped ({image_url}): {error}", file=sys.stderr)
            document["images"] = image_records
            canonical = normalize_url(urljoin(final_url, page.canonical), host) if page.canonical else None
            record = {
                "url": final_url,
                "canonical_url": canonical,
                **document,
                "content_sha256": sha256(document["text"].encode("utf-8")).hexdigest(),
                "crawled_at": datetime.now(timezone.utc).isoformat(),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            written += 1

            for href in page.links:
                target = normalize_url(urljoin(final_url, href), host)
                if target and target not in queued and target not in visited:
                    queued.add(target)
                    queue.append(target)
            print(f"[{written}/{args.max_pages}] {final_url}")
        if written == 0:
            raise RuntimeError(
                "Crawl produced no HTML pages; the existing output was preserved."
            )

    print(f"Saved {written} HTML pages to {output}")
    print(f"Indexed {indexed_images} same-site images")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl public Punjab Tourism HTML pages into chatbot-ready JSONL."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="HTTPS site URL to crawl")
    parser.add_argument("--output", default="data/tourism_pages.jsonl", help="Output JSONL path")
    parser.add_argument("--max-pages", type=int, default=200, help="Maximum HTML pages (default: 200)")
    parser.add_argument(
        "--max-images", type=int, default=500,
        help="Maximum same-site reference images to index (default: 500)",
    )
    parser.add_argument(
        "--skip-text-embeddings", action="store_true",
        help="Do not load the image model to create title embeddings",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (default: 1)")
    parser.add_argument("--timeout", type=int, default=25, help="Request timeout in seconds (default: 25)")
    parser.add_argument(
        "--no-proxy", action="store_true",
        help="Connect directly, ignoring HTTP(S)_PROXY environment variables",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Descriptive crawler user-agent")
    parser.add_argument(
        "--allow-robots-unavailable",
        action="store_true",
        help="Proceed if robots.txt returns 403; only use with site-owner authorization",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_pages < 1:
        parser.error("--max-pages must be at least 1")
    if args.max_images < 0:
        parser.error("--max-images cannot be negative")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")
    try:
        return crawl(args)
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
