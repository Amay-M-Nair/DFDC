"""Phase 5 statistics: paired comparisons between conditions on the same images.

Every image is explained by every model in every condition, so comparisons are paired by image. Binary checks
use McNemar's exact test (a two-sided binomial test on the discordant pairs) and continuous scores the Wilcoxon
signed-rank test (effect size: matched-pairs rank-biserial r). Each difference gets a 95% percentile bootstrap
confidence interval (images resampled with replacement), and p-values are Holm-corrected within a family of
tests. Agreement between models on the artefact type is Fleiss' kappa with a bootstrap interval.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

N_BOOT = 10_000
SEED = 0


def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95) -> tuple:
    """Percentile bootstrap interval for the mean of `values` (NaNs dropped)."""
    x = np.asarray(values, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    means = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(1)
    return tuple(float(q) for q in np.quantile(means, [(1 - level) / 2, (1 + level) / 2]))


def mean_ci(values, digits: int = 3) -> str:
    """"mean [low, high]" for a table cell; "" when there are no values."""
    x = np.asarray(values, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return ""
    lo, hi = bootstrap_ci(x)
    return f"{x.mean():.{digits}f}" + ("" if np.isnan(lo) else f" [{lo:.{digits}f}, {hi:.{digits}f}]")


def paired_test(before, after, binary: bool = None) -> dict:
    """Paired comparison of two conditions on the same images (same order; pairs with a NaN are dropped).
    diff = mean(after) - mean(before). Yes/no checks get McNemar's exact test, scores Wilcoxon; binary=None
    decides from the values (all 0/1 = yes/no)."""
    a, b = np.asarray(before, float), np.asarray(after, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if not len(a):
        return {"n": 0}
    d = b - a
    lo, hi = bootstrap_ci(d)
    row = {"n": len(a), "before": a.mean(), "after": b.mean(), "diff": d.mean(), "ci_low": lo, "ci_high": hi}
    if binary is None:
        binary = set(np.unique(np.r_[a, b])) <= {0.0, 1.0}
    if binary:
        up, down = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
        p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
        row.update(test="McNemar exact", effect=f"{up} up / {down} down", p=p)
    else:
        nz = d[d != 0]
        if len(nz):
            ranks = rankdata(np.abs(nz))
            r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
            p = wilcoxon(nz).pvalue
        else:
            r, p = 0.0, 1.0
        row.update(test="Wilcoxon signed-rank", effect=f"r = {r:+.2f}", p=p)
    return row


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p, float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def compare(scores: pd.DataFrame, pairs, metrics, continuous=(), key: str = "image",
            baseline_llm: str = "baseline") -> pd.DataFrame:
    """Paired tests, one row per (llm, before -> after, metric).

    scores: one row per (llm, condition, image). A condition that only the `baseline_llm` rows have (a no-LLM
    baseline) is compared with every model's answers on the same images. Metrics in `continuous` always get the
    Wilcoxon test (a share that happens to be all 0 or 1 is still a share)."""
    rows = []
    for llm in sorted(set(scores.llm) - {baseline_llm}):
        mine = scores[scores.llm.isin([llm, baseline_llm])]
        for before, after in pairs:
            wa = mine[mine.condition == before].set_index(key)
            wb = mine[mine.condition == after].set_index(key)
            common = wa.index.intersection(wb.index)
            for m in metrics:
                if m not in scores or not len(common):
                    continue
                res = paired_test(wa.loc[common, m], wb.loc[common, m], False if m in continuous else None)
                if res["n"]:
                    rows.append({"llm": llm, "comparison": f"{before} → {after}", "metric": m, **res})
    return pd.DataFrame(rows)


def add_holm(tests: pd.DataFrame, alpha: float = 0.05, higher_is_better: dict = None) -> pd.DataFrame:
    """Holm-adjusted p-values over all rows of `tests` (one family) and a plain-English verdict."""
    tests = tests.copy()
    tests["p_holm"] = holm(tests.p.values) if len(tests) else []
    better = higher_is_better or {}

    def verdict(r):
        if r.p_holm >= alpha or r["diff"] == 0:
            return "no significant difference"
        good = (r["diff"] > 0) == better.get(r.metric, True)
        return "better" if good else "worse"

    tests["verdict"] = tests.apply(verdict, axis=1) if len(tests) else []
    return tests


def fleiss_kappa(labels: pd.DataFrame) -> float:
    """Agreement of several raters (columns) on a category per item (rows); every cell filled."""
    cats = sorted(set(labels.values.ravel()))
    counts = np.array([[list(row).count(c) for c in cats] for row in labels.values])
    n = counts.sum(1)[0]
    p_item = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    p_cat = counts.sum(0) / counts.sum()
    expected = (p_cat ** 2).sum()
    return float((p_item.mean() - expected) / (1 - expected)) if expected < 1 else 1.0


def kappa_ci(labels: pd.DataFrame, n_boot: int = 2000, seed: int = SEED) -> tuple:
    """(kappa, low, high): Fleiss' kappa with a percentile bootstrap interval over items."""
    k = fleiss_kappa(labels)
    if len(labels) < 2:
        return k, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = [fleiss_kappa(labels.iloc[rng.integers(0, len(labels), len(labels))]) for _ in range(n_boot)]
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return k, float(lo), float(hi)
