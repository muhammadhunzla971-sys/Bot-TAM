#!/usr/bin/env python3
"""Add CLIP visual embeddings to an existing authorized tourism image index."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from image_matching import image_label_text, text_embedding, visual_embedding
from scrape_tourism import (
    DEFAULT_BASE_URL,
    DEFAULT_USER_AGENT,
    fetch_image,
    load_robots,
    normalize_image_url,
)


def image_records(pages: list[dict[str, object]]) -> Iterator[tuple[str, dict[str, object]]]:
    host = urlsplit(DEFAULT_BASE_URL).hostname
    seen: set[str] = set()
    for page in pages:
        images = page.get("images", [])
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict) or not isinstance(image.get("url"), str):
                continue
            image_url = normalize_image_url(image["url"], host)
            if image_url and image_url not in seen:
                seen.add(image_url)
                yield image_url, image


def index_images(
    path: Path,
    *,
    allow_robots_unavailable: bool,
    delay: float,
    timeout: int,
) -> int:
    try:
        pages = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read the page index: {path}") from error
    if not pages or not all(isinstance(page, dict) for page in pages):
        raise ValueError(f"The page index is empty or malformed: {path}")

    user_agent = DEFAULT_USER_AGENT
    robots = load_robots(
        DEFAULT_BASE_URL, user_agent, allow_robots_unavailable, timeout
    )
    host = urlsplit(DEFAULT_BASE_URL).hostname
    images = list(image_records(pages))
    embeddings: dict[str, list[int]] = {}
    label_embeddings: dict[str, list[int]] = {}
    last_request = 0.0
    for index, (image_url, _) in enumerate(images, start=1):
        existing_embedding = next(
            (
                page_image.get("embedding")
                for page in pages
                for page_image in (
                    page.get("images", [])
                    if isinstance(page.get("images", []), list)
                    else []
                )
                if isinstance(page_image, dict)
                and page_image.get("url") == image_url
                and isinstance(page_image.get("embedding"), list)
                and len(page_image["embedding"]) == 512
            ),
            None,
        )
        if existing_embedding is not None:
            embeddings[image_url] = existing_embedding
            print(f"[{index}/{len(images)}] Reusing visual embedding {image_url}")
            continue
        if not robots.can_fetch(user_agent, image_url):
            print(f"Blocked by robots.txt: {image_url}", file=sys.stderr)
            continue
        wait = delay - (time.monotonic() - last_request)
        if last_request and wait > 0:
            time.sleep(wait)
        last_request = time.monotonic()
        try:
            fetched = fetch_image(image_url, user_agent, host, timeout)
            if fetched is None:
                continue
            final_url, payload = fetched
            embedding = visual_embedding(payload)
            embeddings[image_url] = embedding
            embeddings[final_url] = embedding
        except (HTTPError, OSError, RuntimeError, TimeoutError, URLError, ValueError) as error:
            print(f"Image skipped ({image_url}): {error}", file=sys.stderr)
            continue
        print(f"[{index}/{len(images)}] Indexed {image_url}")

    if images and not embeddings:
        raise RuntimeError("No reference images could be processed; the existing index was left unchanged.")

    for page in pages:
        page_images = page.get("images", [])
        if not isinstance(page_images, list):
            continue
        title = str(page.get("title") or page.get("url") or "Punjab tourism")
        description = str(page.get("description") or "")
        page["text_embedding"] = text_embedding(
            f"A photograph of {title}. {description}"
        )
        for image in page_images:
            if not isinstance(image, dict):
                continue
            embedding = embeddings.get(image.get("url", ""))
            if embedding is not None:
                image["embedding"] = embedding
            label = image_label_text(
                title, str(image.get("alt", "")), str(image.get("url", ""))
            )
            if label not in label_embeddings:
                label_embeddings[label] = text_embedding(label)
            image["label_embedding"] = label_embeddings[label]

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            for page in pages:
                file.write(json.dumps(page, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
    return len(embeddings)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build local CLIP embeddings for an authorized tourism image index."
    )
    parser.add_argument("--data", type=Path, default=Path("data/tourism_pages.jsonl"))
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument(
        "--allow-robots-unavailable",
        action="store_true",
        help="Proceed only with explicit site-owner authorization when robots.txt is unavailable",
    )
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")
    try:
        count = index_images(
            args.data,
            allow_robots_unavailable=args.allow_robots_unavailable,
            delay=args.delay,
            timeout=args.timeout,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Saved visual embeddings for {count} reference images to {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
