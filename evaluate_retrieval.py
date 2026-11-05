#!/usr/bin/env python3
"""Evaluate corpus retrieval against the checked-in multilingual question set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from chatbot import Corpus, find_place_in_text, load_places


ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_SET = ROOT / "tests" / "eval_set.jsonl"
DEFAULT_EXTENDED_EVAL_SET = ROOT / "tests" / "eval_set_extended.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "tourism_pages.jsonl"


def load_eval_set(
    path: Path, expected_count: int | None = 40
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid evaluation JSONL at line {line_number}.") from error
            if (
                not isinstance(case, dict)
                or not isinstance(case.get("question"), str)
                or not isinstance(case.get("expected_urls"), list)
                or not isinstance(case.get("answerable"), bool)
                or not isinstance(case.get("location"), bool)
            ):
                raise ValueError(f"Invalid evaluation case at line {line_number}.")
            if case["answerable"] != bool(case["expected_urls"]):
                raise ValueError(
                    f"Answerability and expected URLs disagree at line {line_number}."
                )
            cases.append(case)
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(
            f"Expected {expected_count} evaluation cases, found {len(cases)}."
        )
    return cases


def evaluate(
    corpus: Corpus,
    cases: list[dict[str, object]],
    places: list[dict[str, object]],
) -> dict[str, object]:
    started = time.perf_counter()
    hit_count = answerable_count = 0
    no_answer_correct = 0
    location_correct = location_count = 0
    for case in cases:
        question = str(case["question"])
        expected_urls = set(case["expected_urls"])
        answerable = bool(case["answerable"])
        results = corpus.search(question, limit=3)
        place = find_place_in_text(question, places)
        if place:
            page = next(
                (
                    document
                    for document in corpus.documents
                    if document.get("url") == place["source_url"]
                ),
                None,
            )
            results = [page] if page else []
        result_urls = [str(document.get("url", "")) for document in results]
        predicted_answerable = (
            bool(results)
            if place
            else corpus.has_evidence(question, results)
        )
        if answerable:
            answerable_count += 1
            if expected_urls.intersection(result_urls):
                hit_count += 1
        if predicted_answerable == answerable:
            no_answer_correct += 1
        if case["location"]:
            location_count += 1
            if result_urls and result_urls[0] in expected_urls:
                location_correct += 1

    return {
        "retrieval_mode": corpus.retrieval_mode,
        "cases": len(cases),
        "hit_at_3": {
            "correct": hit_count,
            "total": answerable_count,
            "score": round(hit_count / answerable_count, 3) if answerable_count else 0,
        },
        "no_answer_accuracy": {
            "correct": no_answer_correct,
            "total": len(cases),
            "score": round(no_answer_correct / len(cases), 3) if cases else 0,
        },
        "location_accuracy": {
            "correct": location_correct,
            "total": location_count,
            "score": round(location_correct / location_count, 3) if location_count else 0,
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument(
        "--extra-eval-set", type=Path, default=DEFAULT_EXTENDED_EVAL_SET
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable optional ONNX embeddings; model files download on first search.",
    )
    args = parser.parse_args()
    baseline_cases = load_eval_set(args.eval_set)
    extra_cases = load_eval_set(args.extra_eval_set, expected_count=60)
    expanded_cases = baseline_cases + extra_cases
    corpus = Corpus(args.data, hybrid_enabled=args.hybrid)
    places = load_places()
    print(
        json.dumps(
            {
                "baseline_40": evaluate(corpus, baseline_cases, places),
                "expanded_100": evaluate(corpus, expanded_cases, places),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
