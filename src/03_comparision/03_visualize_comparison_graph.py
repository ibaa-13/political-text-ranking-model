"""Create compact diagnostics from the saved comparison graph and CSVs.

Run from the project root:
    python src/03_comparision/03_visualize_comparison_graph.py
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


PARTY_COLORS = {"D": "#3977c3", "R": "#d6534f"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("output/comparisons/comparison_graph.pkl"))
    parser.add_argument("--texts", type=Path, default=Path("output/comparisons/texts.csv"))
    parser.add_argument("--comparisons", type=Path, default=Path("output/comparisons/comparisons.csv"))
    parser.add_argument("--image", type=Path, default=Path("output/comparisons/comparison_diagnostics.png"))
    parser.add_argument("--summary", type=Path, default=Path("output/comparisons/pairwise_node_summary.csv"))
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--min-comparisons", type=int, default=5)
    args = parser.parse_args()
    if args.top_n < 1 or args.min_comparisons < 1:
        parser.error("--top-n and --min-comparisons must be positive")

    texts = pd.read_csv(args.texts)
    comparisons = pd.read_csv(args.comparisons)
    with args.graph.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.DiGraph) or not graph.is_directed():
        raise TypeError("The pickle must contain a directed NetworkX graph")
    if not {"text_id", "ideology_score", "party"}.issubset(texts.columns):
        raise ValueError("texts.csv must contain text_id, ideology_score, and party columns")
    if not {"text_A_id", "text_B_id", "decision"}.issubset(comparisons.columns):
        raise ValueError("comparisons.csv must contain text_A_id, text_B_id, and decision columns")

    summary = texts[["text_id", "party", "ideology_score"]].copy()
    summary["text_id"] = pd.to_numeric(summary["text_id"], errors="raise").astype(int)
    summary["ideology_score"] = pd.to_numeric(summary["ideology_score"], errors="coerce")
    for column in ("wins", "losses", "ties", "unclear"):
        summary[column] = 0
    row_for_id = summary.set_index("text_id").index
    known_ids = set(row_for_id)

    for row in comparisons.itertuples(index=False):
        a, b = int(row.text_A_id), int(row.text_B_id)
        if a not in known_ids or b not in known_ids:
            continue
        decision = str(row.decision).strip().upper()
        if decision == "A":
            summary.loc[summary.text_id == a, "wins"] += 1
            summary.loc[summary.text_id == b, "losses"] += 1
        elif decision == "B":
            summary.loc[summary.text_id == b, "wins"] += 1
            summary.loc[summary.text_id == a, "losses"] += 1
        elif decision == "TIE":
            summary.loc[summary.text_id.isin((a, b)), "ties"] += 1
        elif decision == "UNCLEAR":
            summary.loc[summary.text_id.isin((a, b)), "unclear"] += 1

    # In this graph, each decisive edge points from the losing text to the winning text.
    summary["graph_win_degree"] = summary.text_id.map(dict(graph.in_degree())).fillna(0).astype(int)
    summary["graph_loss_degree"] = summary.text_id.map(dict(graph.out_degree())).fillna(0).astype(int)
    summary["n_comparisons"] = summary[["wins", "losses", "ties", "unclear"]].sum(axis=1)
    decisive_or_tie = summary["wins"] + summary["losses"] + summary["ties"]
    summary["pairwise_win_fraction"] = np.where(
        decisive_or_tie > 0,
        (summary["wins"] + 0.5 * summary["ties"]) / decisive_or_tie,
        np.nan,
    )
    summary["score_rank"] = summary["ideology_score"].rank(pct=True)
    summary["pairwise_rank"] = summary["pairwise_win_fraction"].rank(pct=True)
    rank_data = summary[["score_rank", "pairwise_rank"]].dropna()
    if len(rank_data) > 1 and rank_data.nunique().min() > 1:
        rho = float(np.corrcoef(rank_data["score_rank"], rank_data["pairwise_rank"])[0, 1])
    else:
        rho = float("nan")

    args.image.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values("pairwise_win_fraction", ascending=False).to_csv(args.summary, index=False)

    fig, (ax_scatter, ax_extremes) = plt.subplots(
        1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True
    )
    summary["party_group"] = summary.party.astype(str).str.upper().str.strip()
    for party in ("D", "R"):
        group = summary[summary.party_group == party].dropna(subset=["ideology_score", "pairwise_win_fraction"])
        if group.empty:
            continue
        ax_scatter.scatter(
            group.ideology_score,
            group.pairwise_win_fraction,
            s=26 + 3.0 * np.sqrt(group.n_comparisons),
            color=PARTY_COLORS[party],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.4,
            label=f"{party} (n={len(group)})",
        )
    unknown = summary[~summary.party_group.isin(PARTY_COLORS)].dropna(
        subset=["ideology_score", "pairwise_win_fraction"]
    )
    if not unknown.empty:
        ax_scatter.scatter(
            unknown.ideology_score,
            unknown.pairwise_win_fraction,
            s=26 + 3.0 * np.sqrt(unknown.n_comparisons),
            color="#888888",
            alpha=0.65,
            edgecolors="white",
            linewidths=0.4,
            label=f"Other/unknown (n={len(unknown)})",
        )

    # Label only the texts whose pairwise result differs most from their score rank.
    discordant = summary.dropna(subset=["score_rank", "pairwise_rank"]).copy()
    discordant["rank_gap"] = (discordant.score_rank - discordant.pairwise_rank).abs()
    for row in discordant.nlargest(min(8, len(discordant)), "rank_gap").itertuples():
        ax_scatter.annotate(str(row.text_id), (row.ideology_score, row.pairwise_win_fraction),
                            xytext=(4, 3), textcoords="offset points", fontsize=7)
    rho_label = "undefined" if pd.isna(rho) else f"{rho:.2f}"
    ax_scatter.set(
        xlabel="Luna individual ideology score (0 = left, 100 = right)",
        ylabel="Observed pairwise win fraction",
        title=f"Individual score vs pairwise results\nSpearman rank correlation = {rho_label} (n={len(rank_data)})",
        xlim=(-2, 102),
        ylim=(-0.04, 1.04),
    )
    ax_scatter.grid(alpha=0.18)
    ax_scatter.legend(frameon=False, loc="best")
    ax_scatter.text(
        0.01, -0.17,
        "Win fraction = (wins + 0.5 × ties) / (wins + losses + ties); UNCLEAR excluded.\n"
        "Bubble size reflects the number of observed comparisons. Labels show largest rank gaps.",
        transform=ax_scatter.transAxes, fontsize=8, va="top",
    )

    eligible = summary[summary.n_comparisons >= args.min_comparisons].dropna(
        subset=["pairwise_win_fraction"]
    )
    if eligible.empty:
        eligible = summary[summary.n_comparisons > 0].dropna(subset=["pairwise_win_fraction"])
    k = min(args.top_n, len(eligible) // 2)
    if k:
        extremes = pd.concat([
            eligible.nsmallest(k, "pairwise_win_fraction"),
            eligible.nlargest(k, "pairwise_win_fraction"),
        ]).drop_duplicates("text_id").sort_values("pairwise_win_fraction")
        y_positions = np.arange(len(extremes))
        colors = [PARTY_COLORS.get(p, "#888888") for p in extremes.party_group]
        ax_extremes.barh(y_positions, extremes.pairwise_win_fraction, color=colors, alpha=0.85)
        ax_extremes.set_yticks(y_positions, [f"{int(r.text_id)} ({r.party_group})" for r in extremes.itertuples()])
        for y, row in zip(y_positions, extremes.itertuples()):
            ax_extremes.text(row.pairwise_win_fraction + 0.015, y,
                             f"{int(row.n_comparisons)} comparisons", va="center", fontsize=7)
        ax_extremes.set_xlim(0, 1.28)
        ax_extremes.set_xlabel("Observed pairwise win fraction")
        ax_extremes.set_title(f"Lowest and highest results\n(at least {args.min_comparisons} comparisons)")
        ax_extremes.grid(axis="x", alpha=0.18)
        ax_extremes.set_axisbelow(True)
    else:
        ax_extremes.axis("off")
        ax_extremes.text(0.5, 0.5, "Not enough completed comparisons to rank texts yet.",
                         ha="center", va="center", wrap=True)

    fig.suptitle(
        f"Pairwise graph diagnostics: {len(comparisons):,} comparisons, "
        f"{graph.number_of_nodes():,} graph nodes, {graph.number_of_edges():,} directed edges",
        fontsize=14,
    )
    fig.savefig(args.image, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Image: {args.image}\nPer-text summary: {args.summary}\nSpearman rank correlation: {rho_label}")


if __name__ == "__main__":
    main()
