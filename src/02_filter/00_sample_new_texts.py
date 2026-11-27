"""
Draw a NEW random sample of excerpts for a second Luna round, without Qwen or a GPU.

It skips every excerpt already used in round 1 and writes a CSV in the format that
02_luna_api.py expects (para_id, text, qwen_label, plus metadata). qwen_label is
left empty because Qwen is not run on this sample.

Run from the repository root:
  python ./src/02_filter/00_sample_new_texts.py --n-texts 1500
Then:
  python ./src/02_filter/02_luna_api.py output/round2/sample.csv output/round2/ideological_luna_results.csv
  python ./src/03_comparision/01_compare_texts.py --input output/round2/ideological_luna_results.csv --output output/round2/comparisons
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, default=ROOT / "data/processed/paragraphs.parquet")
    p.add_argument("--used", type=Path, default=ROOT / "output/filtered_texts/ideological_luna_results.csv",
                   help="round-1 file whose para_ids are excluded")
    p.add_argument("--n-texts", type=int, default=1500)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, default=ROOT / "output/round2/sample.csv")
    a = p.parse_args()

    data = pd.read_parquet(a.parquet)
    used = set(pd.read_csv(a.used).para_id) if a.used.exists() else set()
    pool = data[~data.para_id.isin(used)]
    sample = pool.sample(n=a.n_texts, random_state=a.seed).reset_index(drop=True)
    sample["qwen_label"] = ""
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(a.output, index=False)
    print(f"excluded {len(used)} round-1 excerpts; wrote {len(sample)} new excerpts -> {a.output}")
    print(sample.party.value_counts().to_string())


if __name__ == "__main__":
    main()
