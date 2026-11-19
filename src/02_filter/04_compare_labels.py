"""Join Qwen, API Luna, and Codex/Luna labels and report agreement.

Example (run from the project root in PowerShell, after filling the Codex CSV):
    python ./src/02_filter/04_compare_labels.py

Also writes CSV files for label totals and party composition beside the agreement summary.
"""

import argparse
from pathlib import Path

import pandas as pd


def kappa(a, b):
    valid = pd.DataFrame({"a": a, "b": b}).dropna()
    if valid.empty:
        return float("nan")
    observed = (valid["a"] == valid["b"]).mean()
    pa = valid["a"].value_counts(normalize=True)
    pb = valid["b"].value_counts(normalize=True)
    expected = sum(pa.get(label, 0) * pb.get(label, 0) for label in ("YES", "NO"))
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qwen_csv", type=Path, nargs="?", default=Path("output/filtered_texts/ideological_qwen_results.csv"))
    p.add_argument("luna_api_csv", type=Path, nargs="?", default=Path("output/filtered_texts/ideological_luna_results.csv"))
    p.add_argument("codex_review_csv", type=Path, nargs="?", default=Path("output/filtered_texts/codex_review.csv"))
    p.add_argument("final_csv", type=Path, nargs="?", default=Path("output/filtered_texts/final_comparison.csv"))
    p.add_argument("summary_csv", type=Path, nargs="?", default=Path("output/tables/label_agreement.csv"))
    args = p.parse_args()

    qwen = pd.read_csv(args.qwen_csv)
    api = pd.read_csv(args.luna_api_csv)
    codex = pd.read_csv(args.codex_review_csv)
    required = {"para_id", "qwen_label", "text", "party"}
    if not required.issubset(qwen.columns):
        raise ValueError("Qwen CSV must contain para_id, qwen_label, text, and party columns")
    if not {"para_id", "luna_label"}.issubset(api.columns):
        raise ValueError("Luna API CSV must contain para_id and luna_label columns")
    if not {"review_id", "codex_label"}.issubset(codex.columns):
        raise ValueError("Codex review CSV must contain review_id and codex_label")
    if len(codex) != len(qwen):
        raise ValueError("Codex review row count differs from the Qwen input")

    qwen = qwen.copy()
    qwen["review_id"] = range(1, len(qwen) + 1)
    api_labels = api[["para_id", "luna_label"]].drop_duplicates("para_id")
    final = qwen.merge(api_labels, on="para_id", how="left", validate="one_to_one")
    final = final.merge(codex[["review_id", "codex_label"]], on="review_id", how="left", validate="one_to_one")
    for col in ("qwen_label", "luna_label", "codex_label"):
        final[col] = final[col].astype("string").str.upper().str.strip()
    final["party"] = final["party"].astype("string").str.upper().str.strip()

    label_columns = {
        "Qwen": "qwen_label",
        "Luna API": "luna_label",
        "Codex": "codex_label",
    }
    label_totals = []
    yes_counts = {}
    for evaluator, column in label_columns.items():
        valid_labels = final[column][final[column].isin(["YES", "NO"])]
        yes_count = int((valid_labels == "YES").sum())
        no_count = int((valid_labels == "NO").sum())
        yes_counts[evaluator] = yes_count
        label_totals.append({
            "evaluator": evaluator,
            "n_labeled": len(valid_labels),
            "yes_count": yes_count,
            "no_count": no_count,
            "yes_pct": 100 * yes_count / len(valid_labels) if len(valid_labels) else float("nan"),
        })

    party_breakdown = []
    for evaluator, column in label_columns.items():
        for label in ("YES", "NO"):
            group = final[final[column] == label]
            known_party = group[group["party"].isin(["D", "R"])]
            n_known = len(known_party)
            democrats = int((known_party["party"] == "D").sum())
            republicans = int((known_party["party"] == "R").sum())
            party_breakdown.append({
                "evaluator": evaluator,
                "label": label,
                "n_texts": len(group),
                "n_with_D_or_R_party": n_known,
                "democrats_count": democrats,
                "democrats_pct": 100 * democrats / n_known if n_known else float("nan"),
                "republicans_count": republicans,
                "republicans_pct": 100 * republicans / n_known if n_known else float("nan"),
            })

    comparisons = [
        ("qwen_luna", "Qwen vs Luna API", "qwen_label", "luna_label"),
        ("luna_codex", "Luna API vs Codex", "luna_label", "codex_label"),
        ("qwen_codex", "Qwen vs Codex", "qwen_label", "codex_label"),
    ]
    summary = []
    for slug, name, a, b in comparisons:
        valid = final[[a, b]].dropna()
        valid = valid[valid[a].isin(["YES", "NO"]) & valid[b].isin(["YES", "NO"])]
        agreements = int((valid[a] == valid[b]).sum()) if len(valid) else 0
        agreement = agreements / len(valid) if len(valid) else float("nan")
        final[f"{slug}_agree"] = final[a].eq(final[b]).where(final[a].isin(["YES", "NO"]) & final[b].isin(["YES", "NO"]))
        summary.append({
            "comparison": name,
            "n_compared": len(valid),
            "agreements": agreements,
            "agreement_pct": 100 * agreement if len(valid) else float("nan"),
            "cohen_kappa": kappa(valid[a], valid[b]) if len(valid) else float("nan"),
            "qwen_yes_total": yes_counts["Qwen"],
            "luna_yes_total": yes_counts["Luna API"],
            "codex_yes_total": yes_counts["Codex"],
        })
    final["all_three_agree"] = final[["qwen_label", "luna_label", "codex_label"]].nunique(axis=1, dropna=True).eq(1) & final[["qwen_label", "luna_label", "codex_label"]].notna().all(axis=1)
    args.final_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.final_csv, index=False)
    agreement_summary = pd.DataFrame(summary)
    totals_summary = pd.DataFrame(label_totals)
    party_summary = pd.DataFrame(party_breakdown)
    agreement_summary.to_csv(args.summary_csv, index=False)
    party_csv = args.summary_csv.with_name(f"{args.summary_csv.stem}_party_breakdown.csv")
    totals_csv = args.summary_csv.with_name(f"{args.summary_csv.stem}_label_totals.csv")
    party_summary.to_csv(party_csv, index=False)
    totals_summary.to_csv(totals_csv, index=False)
    print("Label totals (yes_pct is percent of labeled texts):")
    print(totals_summary.to_string(index=False))
    print("Pairwise agreement:")
    print(agreement_summary.to_string(index=False))
    print("Party mix within each model's YES/NO labels (percent of rows with D/R party):")
    print(party_summary.to_string(index=False))
    print(f"Wrote {args.final_csv}, {args.summary_csv}, {totals_csv}, and {party_csv}")


if __name__ == "__main__":
    main()
