"""Prepare Xquik tweet exports for the sentiment notebooks."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


OUTPUT_COLUMNS = ["tweet", "sentiment"]

TEXT_ALIASES = ("tweet", "text", "full_text", "tweet_text", "content", "body")
SENTIMENT_ALIASES = ("sentiment", "label", "polarity", "target")

SENTIMENT_LABELS = {
    "positive": "Positive",
    "pos": "Positive",
    "4": "Positive",
    "1": "Positive",
    "negative": "Negative",
    "neg": "Negative",
    "0": "Negative",
    "-1": "Negative",
    "neutral": "Neutral",
    "neu": "Neutral",
    "2": "Neutral",
}


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _index_row(row: Mapping[str, object]) -> dict[str, object]:
    return {_normalize_key(str(key)): value for key, value in row.items()}


def _first_value(indexed_row: Mapping[str, object], aliases: Sequence[str]) -> str:
    for alias in aliases:
        value = indexed_row.get(alias)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return ""


def normalize_sentiment(value: object) -> str | None:
    return SENTIMENT_LABELS.get(str(value).strip().lower())


def normalize_xquik_row(row: Mapping[str, object]) -> dict[str, str] | None:
    indexed_row = _index_row(row)
    tweet = _first_value(indexed_row, TEXT_ALIASES)
    sentiment = normalize_sentiment(_first_value(indexed_row, SENTIMENT_ALIASES))
    if not tweet or sentiment is None:
        return None
    return {"tweet": tweet, "sentiment": sentiment}


def normalize_xquik_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    normalized_rows = []
    for row in rows:
        normalized_row = normalize_xquik_row(row)
        if normalized_row is not None:
            normalized_rows.append(normalized_row)
    return normalized_rows


def convert_csv(input_path: Path, output_path: Path) -> int:
    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = normalize_xquik_rows(reader)

    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Xquik exports to notebook training data.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    count = convert_csv(args.input_csv, args.output_csv)
    print(f"Wrote {count} labeled tweet rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
