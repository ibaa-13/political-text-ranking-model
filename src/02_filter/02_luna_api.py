"""Add OpenAI API Luna labels to Qwen's CSV, one independent request per text.

Example (run from the project root in PowerShell, with OPENAI_API_KEY set):
    python ./src/02_filter/02_luna_api.py
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI


MODEL = "gpt-6-luna"
PROMPT_VERSION = "single-text-v1"
INSTRUCTIONS = (
    "Classify this congressional excerpt. Answer YES only if the text clearly "
    "expresses a substantive position placeable on the US liberal-conservative "
    "left-right scale. Topic, bill summary, or party alone is insufficient. Mixed, "
    "unclear, descriptive, procedural, or ceremonial texts are NO. Output only the "
    "single label YES or NO, with no explanation."
)


def classify_one(client, text):
    for attempt in range(3):
        try:
            response = client.responses.create(
                model=MODEL,
                reasoning={"effort": "none"},
                instructions=INSTRUCTIONS,
                input=f"Excerpt:\n{text}\n\nLabel:",
                max_output_tokens=16,
                store=False,
            )
            match = re.fullmatch(r"\s*(YES|NO)[.!]?\s*", response.output_text, re.I)
            if not match:
                raise ValueError(f"Unexpected Luna response: {response.output_text!r}")
            return match.group(1).upper()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qwen_csv", type=Path, nargs="?", default=Path("output/filtered_texts/ideological_qwen_results.csv"))
    p.add_argument("output_csv", type=Path, nargs="?", default=Path("output/filtered_texts/ideological_luna_results.csv"))
    args = p.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment first.")
    data = pd.read_csv(args.qwen_csv)
    if not {"para_id", "text", "qwen_label"}.issubset(data.columns):
        raise ValueError("Qwen CSV must contain para_id, text, and qwen_label columns")

    data["luna_label"] = ""
    data["luna_prompt_version"] = PROMPT_VERSION
    if args.output_csv.exists():
        saved = pd.read_csv(args.output_csv)
        if {"para_id", "luna_label", "luna_prompt_version"}.issubset(saved.columns):
            prior = saved[saved["luna_prompt_version"] == PROMPT_VERSION].drop_duplicates("para_id")
            labels = prior.set_index("para_id")["luna_label"].to_dict()
            data["luna_label"] = data["para_id"].map(labels).fillna("")
    data["luna_model"] = MODEL

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    for i in range(len(data)):
        if data.at[i, "luna_label"] in {"YES", "NO"}:
            continue
        data.at[i, "luna_label"] = classify_one(client, str(data.at[i, "text"]))
        data.to_csv(args.output_csv, index=False)
        print(f"Luna API: {i + 1}/{len(data)}", flush=True)


if __name__ == "__main__":
    main()
