"""Pick the model for the explanation phases from phase 1 (accuracy) and phase 2 (heatmap quality) results.

Criteria are fixed in advance and weighted equally (no cherry-picking):
    accuracy        FF++ test AUC, Celeb-DF test AUC                       (higher is better)
    faithfulness    Grad-CAM++ gain over a random map, per dataset:
                        deletion gain  = random AUC - Grad-CAM++ AUC       (higher is better)
                        insertion gain = Grad-CAM++ AUC - random AUC       (higher is better)
    face focus      share of Grad-CAM++ activation inside the face regions (higher is better)
    cost            GFLOPs per image                                       (lower is better)
Each criterion is ranked across models (1 = best); the overall score is the mean rank.

Inputs: notebook 04's outputs/gradcam/ folder and (optionally) notebook 03's outputs/compare/summary.csv.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classifier.model import BASELINES
from src.xai.gradcam import BACKGROUND

# Fixed categorical order: colour follows the model, never its rank
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRID = dict(color="#d9d9d4", linewidth=0.6)


def _colors(models):
    known = list(BASELINES.values())
    order = [m for m in known if m in models] + sorted(m for m in models if m not in known)
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(order)}, order


def short(name: str) -> str:
    """Chart label without the citation in brackets, e.g. 'MesoNet-4 (Afchar et al., 2018)' -> 'MesoNet-4'."""
    return re.sub(r"\s*\(.*\)", "", name)


def _style(ax):
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(gradcam_dir, summary_csv=None):
    g = Path(gradcam_dir)
    faith = pd.read_csv(g / "faithfulness.csv")
    faith = faith[~faith.empty_cam.astype(bool)] if "empty_cam" in faith else faith
    data = {"faith": faith, "tests": pd.read_csv(g / "faithfulness_tests.csv"), "regions": pd.read_csv(g / "regions.csv")}
    data["summary"] = pd.read_csv(summary_csv) if summary_csv and Path(summary_csv).exists() else None
    return data


def faithfulness_gains(faith, tests):
    """Per model and dataset: Grad-CAM++ gain over random (mean, 95% CI from paired per-image differences, p)."""
    rows = []
    for (model, ds), g in faith.groupby(["model", "dataset"]):
        piv = g.pivot_table(index="file", columns="cam", values=["deletion_auc", "insertion_auc"])
        if "random" not in piv["deletion_auc"] or "gradcam++" not in piv["deletion_auc"]:
            continue
        for metric, diff in (("deletion", piv["deletion_auc"]["random"] - piv["deletion_auc"]["gradcam++"]),
                             ("insertion", piv["insertion_auc"]["gradcam++"] - piv["insertion_auc"]["random"])):
            diff = diff.dropna()
            half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
            t = tests[(tests.model == model) & (tests.dataset == ds) & (tests.metric == f"{metric}_auc")
                      & (tests.a == "gradcam++") & (tests.b == "random")]
            p = float(t.p_value.iloc[0]) if len(t) else np.nan
            rows.append({"model": model, "dataset": ds, "metric": metric, "gain": diff.mean(),
                         "ci_low": diff.mean() - half, "ci_high": diff.mean() + half, "p_value": p,
                         "significant": bool(p < 0.05 and diff.mean() > 0)})
    return pd.DataFrame(rows)


def face_focus(regions):
    """Share of Grad-CAM++ activation inside the face (1 - background/border share), correct predictions."""
    r = regions[(regions.cam == "gradcam++") & regions.correct.astype(bool)]
    return (1 - r.groupby(["model", "dataset"])[BACKGROUND].mean()).rename("face_share").reset_index()


def accuracy_table(summary):
    if summary is None:
        return None
    s = summary.copy()
    if "backbone" in s:  # same display names as notebook 04's files, even if labels were edited in between
        s["model"] = s.backbone.map(lambda b: BASELINES.get(b, b))
    for ds in ("FF++", "CelebDF"):
        ci = s.get(f"{ds}_auc_ci")
        if ci is not None:
            lo_hi = ci.str.strip("[]").str.split(",", expand=True).astype(float)
            s[f"{ds}_lo"], s[f"{ds}_hi"] = lo_hi[0], lo_hi[1]
    return s


def scorecard(gains, focus, acc):
    card = pd.DataFrame(index=sorted(gains.model.unique()))
    if acc is not None:
        a = acc.set_index("model")
        for ds in ("FF++", "CelebDF"):
            if f"{ds}_auc" in a:
                card[f"{ds} AUC"] = a[f"{ds}_auc"]
        card["GFLOPs"] = a["gflops"]
    for (ds, metric), g in gains.groupby(["dataset", "metric"]):
        card[f"{ds} {metric} gain"] = g.set_index("model").gain
    card["significant vs random"] = gains.groupby("model").significant.sum().astype(int).astype(str) + \
        "/" + gains.groupby("model").size().astype(str)
    card["face focus"] = focus.groupby("model").face_share.mean()
    ranked = card.drop(columns=["significant vs random"])
    ranks = ranked.rank(ascending=False, method="min")
    if "GFLOPs" in ranks:
        ranks["GFLOPs"] = ranked["GFLOPs"].rank(ascending=True, method="min")
    card["mean rank"] = ranks.mean(axis=1)
    return card.sort_values("mean rank"), ranks


def plot_accuracy(acc, colors, order):
    dss = [d for d in ("FF++", "CelebDF") if f"{d}_auc" in acc]
    fig, axes = plt.subplots(1, len(dss), figsize=(6 * len(dss), 0.45 * len(order) + 1.4))
    for ax, ds in zip(np.atleast_1d(axes), dss):
        a = acc.set_index("model").reindex([m for m in order if m in set(acc.model)])
        y = np.arange(len(a))[::-1]
        for yi, (m, r) in zip(y, a.iterrows()):
            if f"{ds}_lo" in a:
                ax.plot([r[f"{ds}_lo"], r[f"{ds}_hi"]], [yi, yi], color=colors[m], linewidth=2)
            ax.plot(r[f"{ds}_auc"], yi, "o", color=colors[m], markersize=8, markeredgecolor="white", markeredgewidth=2)
            ax.annotate(f"{r[f'{ds}_auc']:.3f}", (r[f"{ds}_auc"], yi), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color="#5f5e5a")
        ax.set_yticks(y, [short(m) for m in a.index], fontsize=9)
        ax.set_title(f"{ds} test AUC (95% CI)", fontsize=10, loc="left")
        _style(ax)
    fig.suptitle("Phase 1 · detection accuracy", fontsize=11, x=0.01, ha="left")
    plt.tight_layout()
    return fig


def plot_gains(gains, colors, order):
    panels = [(ds, m) for ds in ("FF++", "CelebDF") for m in ("deletion", "insertion")
              if len(gains[(gains.dataset == ds) & (gains.metric == m)])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.45 * len(order) + 1.6), sharey=True)
    for ax, (ds, metric) in zip(np.atleast_1d(axes), panels):
        g = gains[(gains.dataset == ds) & (gains.metric == metric)].set_index("model")
        models = [m for m in order if m in g.index]
        y = np.arange(len(models))[::-1]
        for yi, m in zip(y, models):
            r = g.loc[m]
            ax.barh(yi, r.gain, height=0.62, color=colors[m], alpha=1 if r.significant else 0.35,
                    hatch=None if r.significant else "///", edgecolor="white", linewidth=0)
            ax.plot([r.ci_low, r.ci_high], [yi, yi], color="#3a3a37", linewidth=1)
            ax.annotate("" if r.significant else "n.s.", (max(r.ci_high, 0), yi), xytext=(4, -3),
                        textcoords="offset points", fontsize=8, color="#5f5e5a")
        ax.axvline(0, color="#8a8a85", linewidth=0.8)
        ax.set_yticks(y, [short(m) for m in models], fontsize=9)
        ax.set_title(f"{ds} · {metric} gain over random", fontsize=10, loc="left")
        ax.set_xlabel("AUC difference (higher = more faithful)", fontsize=8)
        _style(ax)
    fig.suptitle("Phase 2 · how much better Grad-CAM++ is than a random map (bar = mean, line = 95% CI; "
                 "faded + n.s. = not significant)", fontsize=11, x=0.01, ha="left")
    plt.tight_layout()
    return fig


def plot_tradeoff(card, colors):
    if "CelebDF AUC" not in card:
        return None
    gain_cols = [c for c in card.columns if c.endswith("gain")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (ycol, ylabel) in zip(axes, (("faithfulness", "mean Grad-CAM++ gain over random"),
                                         ("face focus", "share of activation on the face"))):
        yvals = card[gain_cols].mean(axis=1) if ycol == "faithfulness" else card["face focus"]
        for m, r in card.iterrows():
            size = 40 + 18 * r.get("GFLOPs", 1)
            ax.scatter(r["CelebDF AUC"], yvals[m], s=size, color=colors[m], edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(short(m), (r["CelebDF AUC"], yvals[m]), xytext=(7, 4), textcoords="offset points",
                        fontsize=8, color="#3a3a37")
        ax.margins(x=0.18, y=0.12)
        ax.set_xlabel("Celeb-DF test AUC (generalisation)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(**GRID); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"Accuracy vs. {ycol} (bubble = GFLOPs)", fontsize=10, loc="left")
    plt.tight_layout()
    return fig


def plot_ranks(ranks, card, colors):
    r = ranks.loc[card.index]
    fig, ax = plt.subplots(figsize=(1.1 * r.shape[1] + 3, 0.45 * len(r) + 1.6))
    vmax = r.values.max()
    ax.imshow(r.values, cmap="Blues_r", vmin=1, vmax=vmax + 1, aspect="auto")
    ax.set_xticks(range(r.shape[1]), r.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(r)), [f"{short(m)}  (mean {card.loc[m, 'mean rank']:.1f})" for m in r.index], fontsize=9)
    for (i, j), v in np.ndenumerate(r.values):
        dark = v <= (vmax + 1) / 2  # dark cell -> light text, and vice versa
        ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=9, color="white" if dark else "#1f1f1d")
    ax.set_title("Rank per criterion (1 = best) · sorted by mean rank", fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    return fig


def report(gradcam_dir, summary_csv=None, display=print, save_dir=None):
    d = load(gradcam_dir, summary_csv)
    gains = faithfulness_gains(d["faith"], d["tests"])
    focus = face_focus(d["regions"])
    acc = accuracy_table(d["summary"])
    colors, order = _colors(set(gains.model) | (set(acc.model) if acc is not None else set()))
    card, ranks = scorecard(gains, focus, acc)

    figs = {}
    if acc is not None:
        figs["accuracy"] = plot_accuracy(acc, colors, order)
    else:
        display("Notebook 03's summary.csv not found: accuracy criteria skipped (attach 03's output).")
    figs["faithfulness_gain"] = plot_gains(gains, colors, order)
    tr = plot_tradeoff(card, colors)
    if tr is not None:
        figs["tradeoff"] = tr
    figs["ranks"] = plot_ranks(ranks, card, colors)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(Path(save_dir) / f"selection_{name}.png", dpi=150, bbox_inches="tight")
        card.to_csv(Path(save_dir) / "selection_scorecard.csv")
        gains.to_csv(Path(save_dir) / "selection_faithfulness_gains.csv", index=False)
    return card, gains, focus, figs
