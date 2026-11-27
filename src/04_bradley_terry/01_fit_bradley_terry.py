"""
Fit a Bradley-Terry model with ties (Davidson 1970) to the Luna pairwise comparisons
and validate the scores.

Model, for texts i and j with scores s_i, s_j and tie parameter nu > 0:
    D = exp(s_i) + exp(s_j) + nu * exp((s_i + s_j) / 2)
    P(i judged more conservative) = exp(s_i) / D
    P(tie)                        = nu * exp((s_i + s_j) / 2) / D
UNCLEAR answers are dropped. A small L2 penalty keeps scores finite for texts that
(almost) always win or lose; scores are centred so that they sum to zero.

Inputs (from 03_comparision):  output/comparisons/texts.csv, comparisons.csv
Outputs:
  output/bradley_terry/bt_scores.csv        one row per text: BT score, bootstrap SE, metadata
  output/bradley_terry/bt_validation.csv    correlations, party AUC, held-out pair accuracy
  output/bradley_terry/bt_validation.png    BT score vs DW-NOMINATE and vs Luna direct score

Run from the repository root:
  python ./src/04_bradley_terry/01_fit_bradley_terry.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]


def load(texts_path, comps_path):
    texts = pd.read_csv(texts_path).sort_values("text_id").reset_index(drop=True)
    c = pd.read_csv(comps_path)
    c = c[c.decision.isin(["A", "B", "TIE"])].copy()
    idx = {t: k for k, t in enumerate(texts.text_id)}
    c["a"] = c.text_A_id.map(idx)
    c["b"] = c.text_B_id.map(idx)
    # outcome from A's point of view: 1 = A more conservative, 0 = B, 0.5 = tie
    c["y"] = c.decision.map({"A": 1.0, "B": 0.0, "TIE": 0.5})
    return texts, c.reset_index(drop=True)


def neg_loglik(params, a, b, y, n, lam):
    s, log_nu = params[:n], params[n]
    nu = np.exp(log_nu)
    sa, sb = s[a], s[b]
    m = np.maximum(sa, sb)                                  # for numerical stability
    ea, eb, et = np.exp(sa - m), np.exp(sb - m), nu * np.exp((sa + sb) / 2 - m)
    D = ea + eb + et
    win_a, win_b, tie = y == 1.0, y == 0.0, y == 0.5
    ll = (np.log(ea[win_a] / D[win_a]).sum() + np.log(eb[win_b] / D[win_b]).sum()
          + np.log(et[tie] / D[tie]).sum())
    # gradient
    pa, pb, pt = ea / D, eb / D, et / D
    ga = np.where(win_a, 1, 0) + np.where(tie, 0.5, 0) - (pa + 0.5 * pt)
    gb = np.where(win_b, 1, 0) + np.where(tie, 0.5, 0) - (pb + 0.5 * pt)
    grad_s = np.bincount(a, ga, n) + np.bincount(b, gb, n)
    grad_nu = (tie.astype(float) - pt).sum()
    obj = -ll + lam * (s ** 2).sum()
    grad = np.concatenate([-grad_s + 2 * lam * s, [-grad_nu]])
    return obj, grad


def fit(a, b, y, n, lam=0.01):
    x0 = np.zeros(n + 1); x0[n] = np.log(0.1)
    r = minimize(neg_loglik, x0, args=(a, b, y, n, lam), jac=True, method="L-BFGS-B")
    s = r.x[:n] - r.x[:n].mean()
    return s, float(np.exp(r.x[n]))


def pair_accuracy(score, a, b, y):
    """Share of decisive comparisons (no ties) where the higher-scored text won."""
    d = y != 0.5
    pred = score[a[d]] > score[b[d]]
    return float((pred == (y[d] == 1.0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=ROOT / "output/comparisons/texts.csv")
    ap.add_argument("--comparisons", default=ROOT / "output/comparisons/comparisons.csv")
    ap.add_argument("--out", default=ROOT / "output/bradley_terry")
    ap.add_argument("--lam", type=float, default=0.01, help="L2 penalty on scores")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    texts, c = load(args.texts, args.comparisons)
    n = len(texts)
    a, b, y = c.a.to_numpy(), c.b.to_numpy(), c.y.to_numpy()
    print(f"{n} texts, {len(c)} usable comparisons ({(y == 0.5).sum()} ties)")

    s, nu = fit(a, b, y, n, args.lam)
    print(f"tie parameter nu = {nu:.3f}")

    # bootstrap over comparisons -> standard error of each score
    boot = np.empty((args.bootstrap, n))
    for k in range(args.bootstrap):
        i = rng.integers(0, len(c), len(c))
        boot[k] = fit(a[i], b[i], y[i], n, args.lam)[0]
    texts["bt_score"] = s
    texts["bt_se"] = boot.std(axis=0)
    texts["n_comparisons"] = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)

    # held-out pair accuracy: 5-fold CV over comparisons
    folds = rng.permutation(len(c)) % 5
    cv_bt, cv_luna = [], []
    luna = texts.ideology_score.to_numpy()
    for f in range(5):
        tr, te = folds != f, folds == f
        s_tr, _ = fit(a[tr], b[tr], y[tr], n, args.lam)
        cv_bt.append(pair_accuracy(s_tr, a[te], b[te], y[te]))
        # Luna direct scores: equal scores count as a miss
        d = te & (y != 0.5)
        cv_luna.append(float(((luna[a[d]] > luna[b[d]]) == (y[d] == 1.0)).mean()))

    is_r = (texts.party == "R").astype(int)
    rows = [
        ("texts", n),
        ("usable comparisons (A/B/TIE)", len(c)),
        ("tie parameter nu", round(nu, 3)),
        ("held-out pair accuracy, Bradley-Terry (5-fold CV)", round(np.mean(cv_bt), 3)),
        ("held-out pair accuracy, Luna direct score", round(np.mean(cv_luna), 3)),
        ("Spearman, BT vs Luna direct score", round(spearmanr(s, luna)[0], 3)),
        ("Pearson, BT vs DW-NOMINATE (all)", round(pearsonr(s, texts.nominate_dim1)[0], 3)),
        ("Pearson, Luna direct vs DW-NOMINATE (all)", round(pearsonr(luna, texts.nominate_dim1)[0], 3)),
    ]
    for p in ["D", "R"]:
        m = (texts.party == p).to_numpy()
        rows.append((f"Pearson, BT vs DW-NOMINATE (within {p})",
                     round(pearsonr(s[m], texts.nominate_dim1[m])[0], 3)))
        rows.append((f"Pearson, Luna direct vs DW-NOMINATE (within {p})",
                     round(pearsonr(luna[m], texts.nominate_dim1[m])[0], 3)))
    rows += [
        ("party AUC, BT score", round(roc_auc_score(is_r, s), 3)),
        ("party AUC, Luna direct score", round(roc_auc_score(is_r, luna), 3)),
        ("median bootstrap SE of BT score", round(float(np.median(texts.bt_se)), 3)),
        ("SD of BT scores", round(float(s.std()), 3)),
    ]
    val = pd.DataFrame(rows, columns=["statistic", "value"])
    val.to_csv(out / "bt_validation.csv", index=False)
    print(val.to_string(index=False))

    keep = ["text_id", "bt_score", "bt_se", "n_comparisons", "ideology_score", "party",
            "speaker", "speakerid", "congress", "chamber", "nominate_dim1", "text"]
    texts[keep].sort_values("bt_score").to_csv(out / "bt_scores.csv", index=False)

    plot(texts, out / "bt_validation.png")
    print(f"outputs -> {out}")


def plot(t, path):
    C = {"D": "#2a78d6", "R": "#e34948"}
    NAME = {"D": "Democrats", "R": "Republicans"}
    INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = [("nominate_dim1", "DW-NOMINATE of the speaker (roll-call votes)"),
              ("ideology_score", "Luna direct score (0 = left, 100 = right)")]
    for ax, (col, xlabel) in zip(axes, panels):
        for p in ["D", "R"]:
            d = t[t.party == p]
            ax.scatter(d[col], d.bt_score, s=34, color=C[p], alpha=.8, label=NAME[p],
                       edgecolor="white", linewidth=1)
        r = pearsonr(t[col], t.bt_score)[0] if col == "nominate_dim1" else spearmanr(t[col], t.bt_score)[0]
        stat = "Pearson r" if col == "nominate_dim1" else "Spearman ρ"
        ax.set_xlabel(xlabel)
        ax.set_title(f"{stat} = {r:.2f}", loc="left", color=INK, fontsize=11)
    axes[0].set_ylabel("Bradley–Terry score  (← liberal · conservative →)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Bradley–Terry scores from 5,000 Luna comparisons ({len(t)} texts)",
                 x=0.06, ha="left", color=INK, fontsize=13)
    fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
