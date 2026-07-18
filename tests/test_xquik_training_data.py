import csv
import json
import tempfile
import unittest
from pathlib import Path

from xquik_training_data import (
    OUTPUT_COLUMNS,
    convert_csv,
    normalize_sentiment,
    normalize_xquik_rows,
)


class XquikTrainingDataTest(unittest.TestCase):
    def test_normalizes_common_sentiment_labels(self):
        self.assertEqual(normalize_sentiment("pos"), "Positive")
        self.assertEqual(normalize_sentiment("0"), "Negative")
        self.assertEqual(normalize_sentiment("neutral"), "Neutral")
        self.assertEqual(normalize_sentiment("unknown"), None)

    def test_maps_xquik_text_aliases_and_skips_unlabeled_rows(self):
        rows = normalize_xquik_rows(
            [
                {"full_text": "Great support", "label": "positive"},
                {"tweet_text": "Missing label", "label": ""},
                {"content": "", "label": "negative"},
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Tweet"], "Great support")
        self.assertEqual(rows[0]["Polarity"], "Positive")

    def test_converts_csv_with_notebook_header(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "xquik.csv"
            output_path = Path(tmp_dir) / "twitter_trainingdata.csv"
            source_path.write_text(
                "tweet_id,created_at,full_text,sentiment\n"
                "1,2026-07-05T12:00:00Z,Useful analytics,4\n"
                "2,2026-07-05T13:00:00Z,Slow dashboard,-1\n",
                encoding="utf-8",
            )

            count = convert_csv(source_path, output_path)

            self.assertEqual(count, 2)
            with output_path.open(newline="", encoding="utf-8") as output:
                reader = csv.DictReader(output)
                self.assertEqual(OUTPUT_COLUMNS, ["Tweet", "Polarity"])
                self.assertEqual(reader.fieldnames, ["Tweet", "Polarity"])
                rows = list(reader)

        self.assertEqual(rows[0]["Polarity"], "Positive")
        self.assertEqual(rows[1]["Polarity"], "Negative")

    def test_notebook_sampling_supports_small_datasets(self):
        notebook_path = Path(__file__).parents[1] / "twitter_analytics.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )

        self.assertIn("df.sample(n = min(70000, len(df))", source)


if __name__ == "__main__":
    unittest.main()
