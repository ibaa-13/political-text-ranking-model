"""
Step 5: a text scorer that ranks texts Luna never saw.

Each text is turned into a sentence embedding (a list of 768 numbers that summarises
its meaning). A linear scorer f(x) = w . x is trained on the Luna comparisons so that
the text Luna judged more conservative gets the higher score:
    P(i more conservative than j) = sigmoid(f(x_i) - f(x_j))      (binary cross-entropy)
This is logistic regression on the difference of the two embeddings.

Models compared (all evaluated on speakers held out from training):
  pairwise_embed  : the scorer above, on sentence embeddings            (main model)
  ridge_bt_embed  : ridge regression from embeddings to Bradley-Terry scores
                    (BT fitted on training comparisons only)
  pairwise_tfidf  : the pairwise scorer on TF-IDF word features
  luna_direct     : Luna's own 0-100 score (reference: needs an API call per text)

Evaluation: 5-fold cross-validation grouped by speaker, so no speaker is in both
training and test. Test pairs are Luna comparisons between two held-out texts.

Transfer: the final pairwise_embed model (trained on all 173 texts) scores the
technology speeches of the 108th-110th and 115th Congresses (not in the main corpus).
There are no Luna labels for these, so we only check party separation (AUC).

Run from the repository root:
  python ./src/05_ranker/01_train_ranker.py
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # small and fast on a laptop CPU
C_GRID = [0.01, 0.1, 1.0, 10.0]
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]

# reuse the Bradley-Terry fit from step 4
_spec = importlib.util.spec_from_file_location(
    "bt", ROOT / "src/04_bradley_terry/01_fit_bradley_terry.py")
bt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bt)


def embed(texts, cache):
    cache = Path(str(cache).replace(".npy", f"_{EMB_MODEL.split('/')[-1]}.npy"))
    if Path(cache).exists():
        e = np.load(cache)
        if len(e) == len(texts):
            return e
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL)
    e = m.encode(list(texts), batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, e)
    return e


def pair_data(X, a, b, y):
    """Difference features for decisive pairs, plus the flipped copy (keeps it symmetric)."""
    d = y != 0.5
    diff = X[a[d]] - X[b[d]]
    lab = (y[d] == 1.0).astype(int)
    return np.vstack([diff, -diff]), np.concatenate([lab, 1 - lab])


def fit_pairwise(X, a, b, y, C):
    Xd, yd = pair_data(X, a, b, y)
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=5000)
    clf.fit(Xd, yd)
    return lambda Z: Z @ clf.coef_.ravel()


def fit_ridge_bt(X, a, b, y, n, alpha):
    s, _ = bt.fit(a, b, y, n)
    seen = np.zeros(n, bool); seen[a] = True; seen[b] = True
    r = Ridge(alpha=alpha).fit(X[seen], s[seen])
    return r.predict


def acc(score, a, b, y):
    return bt.pair_accuracy(score, a, b, y)


def subset(c, keep):
    """Comparisons where both texts are in `keep` (boolean mask over texts)."""
    m = keep[c.a.to_numpy()] & keep[c.b.to_numpy()]
    return c.a.to_numpy()[m], c.b.to_numpy()[m], c.y.to_numpy()[m]


def choose(fit_fn, grid, X, c, train_mask, groups, n):
    """Pick the regularisation strength by inner 3-fold CV over training speakers."""
    idx = np.where(train_mask)[0]
    best, best_acc = grid[0], -1
    for g in grid:
        accs = []
        for itr, ite in GroupKFold(3).split(idx, groups=groups[idx]):
            tr = np.zeros(n, bool); tr[idx[itr]] = True
            te = np.zeros(n, bool); te[idx[ite]] = True
            f = fit_fn(X, *subset(c, tr), g)
            accs.append(acc(f(X), *subset(c, te)))
        if np.mean(accs) > best_acc:
            best, best_acc = g, np.mean(accs)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=ROOT / "output/comparisons/texts.csv")
    ap.add_argument("--comparisons", default=ROOT / "output/comparisons/comparisons.csv")
    ap.add_argument("--bt_scores", default=ROOT / "output/bradley_terry/bt_scores.csv")
    ap.add_argument("--tech", default=ROOT / "data/external/tech_speeches.parquet")
    ap.add_argument("--out", default=ROOT / "output/ranker")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    texts, c = bt.load(args.texts, args.comparisons)
    n = len(texts)
    groups = texts.speakerid.to_numpy()
    bt_full = pd.read_csv(args.bt_scores).set_index("text_id").bt_score.reindex(texts.text_id).to_numpy()
    luna = texts.ideology_score.to_numpy().astype(float)

    E = embed(texts.text, ROOT / "data/processed/emb_comparison_texts.npy")
    tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    T = tfidf.fit_transform(texts.text).toarray()   # vocabulary only, no labels used

    models = {
        "pairwise_embed": (E, lambda X, a, b, y, g: fit_pairwise(X, a, b, y, g), C_GRID),
        "ridge_bt_embed": (E, lambda X, a, b, y, g: fit_ridge_bt(X, a, b, y, n, g), ALPHA_GRID),
        "pairwise_tfidf": (T, lambda X, a, b, y, g: fit_pairwise(X, a, b, y, g), C_GRID),
    }
    oof = {m: np.full(n, np.nan) for m in models}       # out-of-fold scores per text
    pair_acc = {m: [] for m in list(models) + ["luna_direct", "bt_upper_bound"]}
    n_test_pairs = []

    for k, (tr_i, te_i) in enumerate(GroupKFold(5).split(np.arange(n), groups=groups)):
        tr = np.zeros(n, bool); tr[tr_i] = True
        te = ~tr
        a_te, b_te, y_te = subset(c, te)
        n_test_pairs.append(int((y_te != 0.5).sum()))
        for name, (X, fit_fn, grid) in models.items():
            g = choose(fit_fn, grid, X, c, tr, groups, n)
            f = fit_fn(X, *subset(c, tr), g)
            sc = f(X)
            # each fold's model has its own scale: standardise with the training texts'
            # mean and SD before pooling out-of-fold scores across folds
            oof[name][te] = (sc[te] - sc[tr].mean()) / sc[tr].std()
            pair_acc[name].append(acc(sc, a_te, b_te, y_te))
            print(f"fold {k}  {name:15s} reg={g:<6} test pair acc={pair_acc[name][-1]:.3f}")
        pair_acc["luna_direct"].append(acc(luna, a_te, b_te, y_te))
        # BT fitted on ALL comparisons, incl. the test ones: an upper bound, not a fair model
        pair_acc["bt_upper_bound"].append(acc(bt_full, a_te, b_te, y_te))

    is_r = (texts.party == "R").to_numpy().astype(int)
    nom = texts.nominate_dim1.to_numpy()
    rows = []
    for name in list(models) + ["luna_direct", "bt_upper_bound"]:
        sc = oof.get(name, luna if name == "luna_direct" else bt_full)
        row = {"model": name,
               "held-out pair accuracy": round(np.mean(pair_acc[name]), 3),
               "Spearman with BT scores": round(spearmanr(sc, bt_full)[0], 3),
               "party AUC": round(roc_auc_score(is_r, sc), 3),
               "Pearson with DW-NOMINATE": round(pearsonr(sc, nom)[0], 3)}
        for p in ["D", "R"]:
            m = (texts.party == p).to_numpy()
            row[f"Pearson with DW-NOMINATE, within {p}"] = round(pearsonr(sc[m], nom[m])[0], 3)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "ranker_comparison.csv", index=False)
    print(f"\ntest pairs per fold: {n_test_pairs}")
    print(res.to_string(index=False))

    # ---------- final model on all texts, then transfer to technology speeches ----------
    g = choose(models["pairwise_embed"][1], C_GRID, E, c, np.ones(n, bool), groups, n)
    final = fit_pairwise(E, c.a.to_numpy(), c.b.to_numpy(), c.y.to_numpy(), g)
    texts["ranker_score"] = final(E)
    texts["ranker_score_out_of_fold"] = oof["pairwise_embed"]
    texts[["text_id", "party", "speaker", "nominate_dim1", "ideology_score",
           "ranker_score", "ranker_score_out_of_fold"]].assign(bt_score=bt_full).to_csv(
        out / "ranker_scores_comparison_texts.csv", index=False)

    tech_rows = []
    if Path(args.tech).exists():
        tech = pd.read_parquet(args.tech)
        tech = tech[tech.congress.isin([108, 109, 110, 115])].reset_index(drop=True)
        Et = embed(tech.text, ROOT / "data/processed/emb_tech_transfer.npy")
        tech["ranker_score"] = final(Et)
        dr = tech[tech.party.isin(["D", "R"])]
        auc_t = roc_auc_score((dr.party == "R").astype(int), dr.ranker_score)
        tech_rows.append(("technology segments scored (108-110, 115)", len(tech)))
        tech_rows.append(("party AUC on technology segments (D vs R)", round(auc_t, 3)))
        for cg, d in dr.groupby("congress"):
            tech_rows.append((f"party AUC, congress {cg}",
                              round(roc_auc_score((d.party == "R").astype(int), d.ranker_score), 3)))
        # speaker level: average score per (surname, party, chamber), at least 5 segments
        spk = (dr.groupby(["speaker_lastname", "party", "chamber"]).ranker_score
                 .agg(["mean", "size"]).reset_index())
        spk = spk[spk["size"] >= 5]
        tech_rows.append(("speakers with >= 5 technology segments", len(spk)))
        tech_rows.append(("party AUC, speaker averages",
                          round(roc_auc_score((spk.party == "R").astype(int), spk["mean"]), 3)))
        pd.DataFrame(tech_rows, columns=["statistic", "value"]).to_csv(out / "tech_transfer.csv", index=False)
        tech[["segment_id", "congress", "year", "chamber", "party", "speaker_lastname",
              "cpc_classes", "ranker_score"]].assign(cpc_classes=tech.cpc_classes.map(list)).to_csv(
            out / "ranker_scores_tech.csv", index=False)
        print(pd.DataFrame(tech_rows, columns=["statistic", "value"]).to_string(index=False))
    else:
        print(f"no technology file at {args.tech}; transfer step skipped")
        dr = None

    plot(res, texts, dr, out / "ranker_results.png")
    print(f"outputs -> {out}")


def plot(res, texts, tech, path):
    C = {"D": "#2a78d6", "R": "#e34948"}
    NAME = {"D": "Democrats", "R": "Republicans"}
    INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # (1) held-out pair accuracy by model
    label = {"pairwise_embed": "Embedding ranker (ours)", "ridge_bt_embed": "Ridge on BT scores",
             "pairwise_tfidf": "TF-IDF ranker", "luna_direct": "Luna direct score*",
             "bt_upper_bound": "Bradley–Terry (upper bound)*"}
    r = res.iloc[::-1]
    ax = axes[0]
    col = ["#2a78d6" if m == "pairwise_embed" else "#9a9a9a" for m in r.model]
    ax.barh([label[m] for m in r.model], r["held-out pair accuracy"], color=col, height=.6)
    for i, v in enumerate(r["held-out pair accuracy"]):
        ax.text(v + .005, i, f"{v:.2f}", va="center", color=INK, fontsize=10)
    ax.axvline(.5, color=MUTED, lw=1, ls="--")
    ax.set_xlim(.45, 1.02)
    ax.set_xlabel("Accuracy on pairs of unseen speakers  (dashed = chance)")
    ax.set_title("Who predicts Luna's choice?", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", visible=False)

    # (2) out-of-fold ranker score vs DW-NOMINATE
    ax = axes[1]
    for p in ["D", "R"]:
        d = texts[texts.party == p]
        ax.scatter(d.nominate_dim1, d.ranker_score_out_of_fold, s=30, color=C[p], alpha=.8,
                   label=NAME[p], edgecolor="white", linewidth=1)
    rr = pearsonr(texts.nominate_dim1, texts.ranker_score_out_of_fold)[0]
    ax.set_xlabel("DW-NOMINATE of the speaker")
    ax.set_ylabel("Ranker score (speaker not seen in training)")
    ax.set_title(f"Unseen speakers: Pearson r = {rr:.2f}", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper left")

    # (3) technology transfer
    ax = axes[2]
    if tech is not None:
        bins = np.linspace(tech.ranker_score.quantile(.005), tech.ranker_score.quantile(.995), 40)
        for p in ["D", "R"]:
            ax.hist(tech[tech.party == p].ranker_score, bins=bins, color=C[p], alpha=.55,
                    label=f"{NAME[p]} (n = {(tech.party == p).sum():,})", edgecolor="white", linewidth=.6)
        auc = roc_auc_score((tech.party == "R").astype(int), tech.ranker_score)
        ax.set_title(f"Technology speeches 2004–08 and 2017–18: AUC = {auc:.2f}",
                     loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Ranker score  (← liberal · conservative →)")
        ax.set_ylabel("Segments")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.03, "* not a fair competitor: Luna direct needs an API call per text; the BT upper bound "
             "was fitted on the test comparisons too.", fontsize=9, color=MUTED)
    fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
