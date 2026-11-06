#!/usr/bin/env python3
"""Build a multi-source tourism corpus from configured official sites and Wikipedia.

Official sites in siteslink.md are crawled first using scrape_tourism.py, which
checks each host's robots.txt and stays on that host. Wikipedia is fetched via
the MediaWiki API and is always marked as secondary evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from hashlib import sha256
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urldefrag, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, install_opener, urlopen
from urllib.robotparser import RobotFileParser

from scrape_tourism import (
    BLOCKED_EXTENSIONS,
    DEFAULT_USER_AGENT,
    MAX_RESPONSE_BYTES,
    PageParser,
    atomic_text_output,
    load_robots,
    normalize_url,
)


ROOT = Path(__file__).resolve().parent
SITES_FILE = ROOT / "siteslink.md"
DEFAULT_OUTPUT = ROOT / "data" / "tourism_pages.jsonl"
WIKIPEDIA_QUERIES = {
    "en": (
        "tourism in Punjab Pakistan",
        "tourist attractions in Punjab Pakistan",
        "tourism in Lahore",
        "Walled City of Lahore",
        "Punjabi culture Pakistan",
        "Sufi shrines Punjab Pakistan",
    ),
    "ur": (
        "پنجاب پاکستان سیاحت",
        "پنجاب کے سیاحتی مقامات",
        "لاہور سیاحت",
        "اندرون لاہور",
        "پنجابی ثقافت",
        "پنجاب مزارات",
    ),
}
WIKIPEDIA_PRIORITY = 100
MAX_API_BYTES = 2 * 1024 * 1024


def read_sites(path: Path) -> list[str]:
    sites: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if not re.match(r"^https?://", value, re.IGNORECASE):
            continue
        # Preserve the exact HTTP-only DTS source the user supplied; upgrade other seeds.
        if not re.match(r"^http://dts\.gov\.pk(?:/|$)", value, re.IGNORECASE):
            value = re.sub(r"^http://", "https://", value, flags=re.IGNORECASE)
        sites.append(value.rstrip("/") + "/")
    return list(dict.fromkeys(sites))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {error}") from error
        if isinstance(value, dict) and isinstance(value.get("url"), str):
            records.append(value)
    return records


def request_json(url: str, user_agent: str, timeout: int) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(MAX_API_BYTES + 1)
            break
        except HTTPError as error:
            if error.code != 429 and error.code < 500 or attempt == 4:
                raise
            retry_after = error.headers.get("Retry-After", "")
            try:
                wait = min(60.0, max(1.0, float(retry_after)))
            except ValueError:
                wait = min(60.0, 2.0 ** (attempt + 1))
            print(f"Wikipedia API HTTP {error.code}; retrying in {wait:g}s", file=sys.stderr)
            time.sleep(wait)
    if len(payload) > MAX_API_BYTES:
        raise RuntimeError("Wikipedia API response exceeded the 2 MiB limit.")
    result = json.loads(payload.decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("Wikipedia API returned an unexpected response.")
    return result


def wikipedia_records(delay: float, timeout: int, user_agent: str) -> list[dict[str, object]]:
    if "contact:" in user_agent:
        user_agent = "PunjabTourismAssistantCorpus/1.0 (public source indexing; contact: project administrator)"
    records: list[dict[str, object]] = []
    for language, queries in WIKIPEDIA_QUERIES.items():
        api_url = f"https://{language}.wikipedia.org/w/api.php"
        titles: dict[int, str] = {}
        for query in queries:
            params = urlencode({
                "action": "query", "list": "search", "srsearch": query,
                "srnamespace": 0, "srlimit": 5, "format": "json", "utf8": 1,
            })
            response = request_json(f"{api_url}?{params}", user_agent, timeout)
            query_data = response.get("query", {})
            results = query_data.get("search", []) if isinstance(query_data, dict) else []
            for item in results:
                if isinstance(item, dict) and isinstance(item.get("pageid"), int):
                    titles[item["pageid"]] = str(item.get("title", ""))
            time.sleep(delay)

        page_ids = list(titles)
        for offset in range(0, len(page_ids), 20):
            params = urlencode({
                "action": "query", "pageids": "|".join(map(str, page_ids[offset:offset + 20])),
                "prop": "extracts", "explaintext": 1, "exsectionformat": "plain",
                "format": "json", "utf8": 1,
            })
            response = request_json(f"{api_url}?{params}", user_agent, timeout)
            query_data = response.get("query", {})
            pages = query_data.get("pages", {}) if isinstance(query_data, dict) else {}
            if not isinstance(pages, dict):
                continue
            for raw_page in pages.values():
                if not isinstance(raw_page, dict):
                    continue
                extract = re.sub(r"\s+", " ", str(raw_page.get("extract", ""))).strip()
                page_title = str(raw_page.get("title", "")).strip()
                if not page_title or len(extract) < 100:
                    continue
                title_path = quote(page_title.replace(" ", "_"), safe="()_,'")
                page_url = f"https://{language}.wikipedia.org/wiki/{title_path}"
                records.append({
                    "url": page_url,
                    "canonical_url": page_url,
                    "title": f"Wikipedia (secondary): {page_title}",
                    "description": f"{language.upper()} Wikipedia background reference; secondary source. Verify details against official Punjab government sources.",
                    "headings": [],
                    "text": extract[:18000],
                    "source_name": f"Wikipedia ({language.upper()})",
                    "source_group": "secondary",
                    "source_priority": WIKIPEDIA_PRIORITY,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                })
            time.sleep(delay)
    return records


def site_label(url: str) -> str:
    host = url.split("//", 1)[-1].split("/", 1)[0].lower()
    labels = {
        "tourism.punjab.gov.pk": "Punjab Tourism Department",
        "tdcp.gop.pk": "Tourism Development Corporation of Punjab",
        "walledcitylahore.gop.pk": "Walled City of Lahore Authority",
        "auqaf.punjab.gov.pk": "Auqaf and Religious Affairs Department Punjab",
        "dts.gov.pk": "Department of Tourist Services",
    }
    return labels.get(host, host)


def run_site_crawl(
    site: str, priority: int, *, max_pages: int, delay: float, timeout: int,
    user_agent: str, allow_robots_unavailable: bool, no_proxy: bool,
) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="tourism-source-") as temporary:
        partial = Path(temporary) / "source.jsonl"
        command = [
            sys.executable, str(ROOT / "scrape_tourism.py"),
            "--base-url", site, "--output", str(partial),
            "--max-pages", str(max_pages), "--max-images", "0",
            "--skip-text-embeddings",
            "--delay", str(delay), "--timeout", str(timeout),
            "--user-agent", user_agent,
        ]
        if no_proxy:
            command.append("--no-proxy")
        if allow_robots_unavailable:
            command.append("--allow-robots-unavailable")
        child_env = os.environ.copy()
        if no_proxy:
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                child_env.pop(name, None)
        result = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            detail = (result.stderr or "crawl failed").strip()
            raise RuntimeError(detail[-1200:])
        records = read_jsonl(partial)
    label = site_label(site)
    for record in records:
        record["source_name"] = label
        record["source_group"] = "official"
        record["source_priority"] = priority
    return records


def run_tdcp_browser_crawl(
    site: str, priority: int, *, max_pages: int, delay: float, timeout: int,
    user_agent: str, allow_robots_unavailable: bool, no_proxy: bool,
) -> list[dict[str, object]]:
    """Render TDCP's public React pages because its HTML shell has no page text."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError(
            "TDCP is JavaScript-rendered. Install requirements-scraping.txt and "
            "run 'python -m playwright install chromium' to collect it."
        ) from error

    host = "tdcp.gop.pk"
    start = normalize_url(site, host)
    if start is None:
        raise RuntimeError("TDCP seed URL is not a valid same-host HTTPS URL.")
    robots = load_robots(site, user_agent, allow_robots_unavailable, timeout, no_proxy)
    queue = [start]
    queued = {start}
    visited: set[str] = set()
    records: list[dict[str, object]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(user_agent=user_agent, locale="en-PK")
        page = context.new_page()
        page.set_default_navigation_timeout(timeout * 1000)
        page.set_default_timeout(timeout * 1000)
        try:
            while queue and len(visited) < max_pages:
                url = queue.pop(0)
                if url in visited:
                    continue
                visited.add(url)
                if not robots.can_fetch(user_agent, url):
                    print(f"Blocked by robots.txt: {url}", file=sys.stderr)
                    continue
                if records:
                    time.sleep(delay)
                try:
                    response = page.goto(url, wait_until="domcontentloaded")
                    if response is None or response.status >= 400:
                        print(f"TDCP page unavailable (skipping): {url}", file=sys.stderr)
                        continue
                    try:
                        page.wait_for_load_state("networkidle", timeout=min(timeout, 8) * 1000)
                    except Exception:
                        pass
                    final_url = normalize_url(page.url, host)
                    if final_url is None:
                        print(f"TDCP redirected outside the allowed host (skipping): {url}", file=sys.stderr)
                        continue
                    title = (page.title() or final_url).strip()
                    body_text = page.locator("body").inner_text().strip()
                    body_text = re.sub(r"\n[ \t]+", "\n", body_text)
                    body_text = re.sub(r"\n{3,}", "\n\n", body_text)
                    headings = page.locator("h1,h2,h3,h4,h5,h6").evaluate_all(
                        "els => els.map(el => ({level: el.tagName.toLowerCase(), text: el.innerText.trim()})).filter(x => x.text)"
                    )
                    links = page.locator("a[href]").evaluate_all("els => els.map(el => el.href)")
                    if len(body_text) >= 80:
                        records.append({
                            "url": final_url,
                            "canonical_url": final_url,
                            "title": title,
                            "description": "",
                            "headings": headings,
                            "text": body_text[:24000],
                            "images": [],
                            "content_sha256": sha256(body_text.encode("utf-8")).hexdigest(),
                            "crawled_at": datetime.now(timezone.utc).isoformat(),
                            "source_name": "Tourism Development Corporation of Punjab",
                            "source_group": "official",
                            "source_priority": priority,
                        })
                    for href in links:
                        target = normalize_url(str(href), host)
                        if target and target not in queued and target not in visited:
                            queued.add(target)
                            queue.append(target)
                    print(f"[TDCP {len(records)}/{max_pages}] {final_url}")
                except Exception as error:
                    print(f"TDCP page skipped ({url}): {error}", file=sys.stderr)
        finally:
            context.close()
            browser.close()
    if not records:
        raise RuntimeError("TDCP browser crawl produced no readable public pages.")
    return records


class SameHostDtsRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(urljoin(req.full_url, newurl))
        if target.scheme != "http" or target.hostname != "dts.gov.pk":
            raise HTTPError(newurl, code, "DTS redirect outside its supplied HTTP origin blocked", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def normalize_dts_url(url: str) -> str | None:
    parts = urlsplit(urldefrag(url)[0])
    if parts.scheme != "http" or parts.hostname != "dts.gov.pk" or parts.username or parts.password:
        return None
    path = parts.path or "/"
    if Path(path).suffix.lower() in BLOCKED_EXTENSIONS:
        return None
    return urlunsplit(("http", "dts.gov.pk", path.rstrip("/") or "/", "", ""))


def crawl_dts_http(
    site: str, priority: int, *, max_pages: int, delay: float, timeout: int,
    user_agent: str, allow_robots_unavailable: bool, no_proxy: bool,
) -> list[dict[str, object]]:
    """Crawl only the explicit user-listed HTTP DTS host, with strict host limits."""
    start = normalize_dts_url(site)
    if start is None:
        raise RuntimeError("DTS URL must use the supplied http://dts.gov.pk origin.")
    robots_url = "http://dts.gov.pk/robots.txt"
    parser = RobotFileParser(robots_url)
    opener = build_opener(
        *( [ProxyHandler({}), SameHostDtsRedirectHandler()]
           if no_proxy else [SameHostDtsRedirectHandler()] )
    )
    request = Request(robots_url, headers={"User-Agent": user_agent})
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(256 * 1024)
            if response.headers.get_content_type().lower() == "text/html":
                if not allow_robots_unavailable:
                    raise RuntimeError("DTS robots.txt returned HTML; the crawl was stopped.")
                print("WARNING: DTS robots.txt is HTML; using the explicitly authorized override.", file=sys.stderr)
                parser.parse([])
            else:
                parser.parse(payload.decode("utf-8", "replace").splitlines())
    except HTTPError as error:
        if error.code == 404:
            parser.parse([])
        elif error.code == 403 and allow_robots_unavailable:
            print("WARNING: DTS robots.txt returned 403; using the authorized override.", file=sys.stderr)
            parser.parse([])
        else:
            raise RuntimeError(f"Could not read DTS robots.txt (HTTP {error.code}).") from error

    queue = deque([start])
    queued = {start}
    visited: set[str] = set()
    records: list[dict[str, object]] = []
    last_request = 0.0
    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        if not parser.can_fetch(user_agent, url):
            print(f"Blocked by DTS robots.txt: {url}", file=sys.stderr)
            continue
        wait = delay - (time.monotonic() - last_request)
        if last_request and wait > 0:
            time.sleep(wait)
        last_request = time.monotonic()
        page_request = Request(url, headers={"User-Agent": user_agent, "Accept": "text/html"})
        try:
            with opener.open(page_request, timeout=timeout) as response:
                final_url = normalize_dts_url(response.geturl())
                if final_url is None:
                    raise RuntimeError("DTS page redirected outside the supplied HTTP origin.")
                if response.headers.get_content_type().lower() not in {"text/html", "application/xhtml+xml"}:
                    continue
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("DTS page exceeded the 5 MiB response limit.")
                charset = response.headers.get_content_charset() or "utf-8"
                html = payload.decode(charset, "replace")
        except HTTPError as error:
            print(f"DTS page skipped (HTTP {error.code}): {url}", file=sys.stderr)
            continue
        parser_page = PageParser()
        parser_page.feed(html)
        document = parser_page.document()
        text = str(document.get("text", ""))
        if len(text) >= 80:
            records.append({
                "url": final_url,
                "canonical_url": normalize_dts_url(urljoin(final_url, parser_page.canonical)) if parser_page.canonical else final_url,
                **document,
                "images": [],
                "content_sha256": sha256(text.encode("utf-8")).hexdigest(),
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "source_name": "Department of Tourist Services (HTTP source)",
                "source_group": "official",
                "source_priority": priority,
                "transport_security": "http; verify details against a secure official source",
            })
        for href in parser_page.links:
            target = normalize_dts_url(urljoin(final_url, href))
            if target and target not in queued and target not in visited:
                queued.add(target)
                queue.append(target)
        print(f"[DTS {len(records)}/{max_pages}] {final_url}", flush=True)
    if not records:
        raise RuntimeError("DTS HTTP crawl produced no readable public HTML pages.")
    return records


def merge_records(
    existing: Iterable[dict[str, object]], updates: Iterable[dict[str, object]]
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for record in existing:
        url = str(record.get("url", ""))
        merged[url] = record
    for record in updates:
        url = str(record["url"])
        previous = merged.get(url)
        if previous:
            if not record.get("images") and previous.get("images"):
                record["images"] = previous["images"]
            if "text_embedding" not in record and "text_embedding" in previous:
                record["text_embedding"] = previous["text_embedding"]
        merged[url] = record
    return list(merged.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=SITES_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-pages-per-site", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--skip-wikipedia", action="store_true")
    parser.add_argument("--skip-official", action="store_true",
                        help="Refresh Wikipedia only, preserving the official-site records.")
    parser.add_argument("--only-host", help="Crawl one configured official host, such as tdcp.gop.pk.")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Ignore broken HTTP(S)_PROXY environment settings and connect directly.")
    parser.add_argument("--allow-robots-unavailable", action="store_true",
                        help="Proceed when robots.txt is blocked or invalid; only use with site-owner authorization.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_pages_per_site < 1 or args.delay < 0 or args.timeout < 1:
        raise SystemExit("Page limit and timeout must be positive; delay cannot be negative.")
    if args.no_proxy:
        install_opener(build_opener(ProxyHandler({})))
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(name, None)
    sites = read_sites(args.sites)
    if not sites:
        raise SystemExit(f"No HTTP(S) sites found in {args.sites}")
    existing = read_jsonl(args.output)
    updates: list[dict[str, object]] = []
    failures: list[str] = []
    for priority, site in enumerate(sites, 1):
        if args.skip_official:
            break
        host = site.split("//", 1)[-1].split("/", 1)[0].lower()
        if args.only_host and host != args.only_host.lower():
            continue
        try:
            if host == "tdcp.gop.pk":
                records = run_tdcp_browser_crawl(
                    site, priority, max_pages=args.max_pages_per_site,
                    delay=args.delay, timeout=args.timeout, user_agent=args.user_agent,
                    allow_robots_unavailable=args.allow_robots_unavailable,
                    no_proxy=args.no_proxy,
                )
            elif host == "dts.gov.pk" and site.startswith("http://"):
                records = crawl_dts_http(
                    site, priority, max_pages=args.max_pages_per_site,
                    delay=args.delay, timeout=args.timeout, user_agent=args.user_agent,
                    allow_robots_unavailable=args.allow_robots_unavailable,
                    no_proxy=args.no_proxy,
                )
            else:
                records = run_site_crawl(
                    site, priority, max_pages=args.max_pages_per_site,
                    delay=args.delay, timeout=args.timeout, user_agent=args.user_agent,
                    allow_robots_unavailable=args.allow_robots_unavailable,
                    no_proxy=args.no_proxy,
                )
            updates.extend(records)
            print(f"Collected {len(records)} official pages from {host} (priority {priority})")
        except (OSError, RuntimeError, HTTPError, URLError) as error:
            failures.append(f"{host}: {error}")
            print(f"Source skipped; existing records retained for {host}: {error}", file=sys.stderr)

    if not args.skip_wikipedia:
        try:
            wiki = wikipedia_records(args.delay, args.timeout, args.user_agent)
            updates.extend(wiki)
            print(f"Collected {len(wiki)} secondary Wikipedia articles")
        except (OSError, RuntimeError, HTTPError, URLError, ValueError) as error:
            failures.append(f"Wikipedia: {error}")
            print(f"Wikipedia skipped; existing Wikipedia records retained: {error}", file=sys.stderr)

    if not updates:
        print("No new sources could be collected; the existing corpus was preserved.", file=sys.stderr)
        return 1
    merged = merge_records(existing, updates)
    if not merged:
        print("No sources were collected; the existing output was preserved.", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_output(args.output) as output:
        for record in merged:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(merged)} records to {args.output}")
    if failures:
        print(f"Completed with {len(failures)} source(s) unavailable; see messages above.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
