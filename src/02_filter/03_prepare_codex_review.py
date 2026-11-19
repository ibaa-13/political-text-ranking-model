"""Create a party-blind review CSV to upload to Codex/Luna (no API call).

Example (run from the project root in PowerShell):
    python ./src/02_filter/03_prepare_codex_review.py
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qwen_csv", type=Path, nargs="?", default=Path("output/filtered_texts/ideological_qwen_results.csv"))
    p.add_argument("review_csv", type=Path, nargs="?", default=Path("output/filtered_texts/codex_review.csv"))
    args = p.parse_args()
    data = pd.read_csv(args.qwen_csv)
    if not {"para_id", "text", "qwen_label"}.issubset(data.columns):
        raise ValueError("Qwen CSV must contain para_id, text, and qwen_label columns")
    review = pd.DataFrame({
        "review_id": range(1, len(data) + 1),
        "text": data["text"].fillna(""),
        "codex_label": "",
        "codex_note": "",
    })
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.review_csv, index=False)
    print(f"Prepared {len(review)} blind review rows: {args.review_csv}")


if __name__ == "__main__":
    main()
