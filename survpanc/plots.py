import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .cox import kaplan_meier

STAGE = {0: "IB", 1: "IIA", 2: "IIB", 3: "III"}
COL = {0: "#2a9d8f", 1: "#264653", 2: "#e76f51", 3: "#c1121f"}


def fig_cohort(df, thr, path, title):
    ycol = "tumor_volume" if "tumor_volume" in df else "age" if "age" in df else None
    scol = next((c for c in ("stage_grouping", "stage_ordinal") if c in df), None)
    fig, ax = plt.subplots(figsize=(6, 4))
    levels = sorted(df[scol].round().unique()) if scol else [None]
    cmap = plt.get_cmap("viridis", max(len(levels), 2))
    syn_mask = df["synthetic"] == 1 if "synthetic" in df else np.zeros(len(df), bool)
    for i, s in enumerate(levels):
        d = df if s is None else df[df[scol].round() == s]
        sm = syn_mask[d.index] if hasattr(syn_mask, "__getitem__") and len(syn_mask) == len(df) else np.zeros(len(d), bool)
        y = d[ycol] if ycol else np.zeros(len(d))
        lab = "patients" if s is None else f"{scol} = {int(s)}"
        ax.scatter(d.duration[~np.asarray(sm)], y[~np.asarray(sm)], color=cmap(i), s=22, label=lab)
        if np.asarray(sm).any():
            ax.scatter(d.duration[np.asarray(sm)], y[np.asarray(sm)], color=cmap(i), s=22, marker="x", alpha=.6)
    if thr:
        ax.axvline(thr, ls="--", c="gold", label=f"threshold = {thr} d")
    ax.set_xlabel("Survival days"); ax.set_ylabel(ycol or ""); ax.set_title(title)
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_threshold(sweep, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(sweep.threshold_days.astype(str), sweep.balanced_accuracy, color="#264653")
    for i, v in enumerate(sweep.balanced_accuracy):
        ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(.5, c="grey", ls=":"); ax.set_ylim(0, 1)
    ax.set_xlabel("High/low-risk threshold (days)"); ax.set_ylabel("Balanced accuracy (CV)")
    ax.set_title("Naive Bayes threshold selection"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_cindex(results, path):
    """results: {mode: summary DataFrame}"""
    modes = list(results); sets = [s for s in ["clinical", "radiomic", "combined"] if s in next(iter(results.values())).index]
    w = 0.8 / len(modes); fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, m in enumerate(modes):
        s = results[m]
        ax.bar(np.arange(len(sets)) + i * w, s.loc[sets, "mean_cindex"], w, yerr=s.loc[sets, "std_over_folds"],
               label=m, capsize=2)
    ax.axhline(.5, c="grey", ls=":"); ax.set_xticks(np.arange(len(sets)) + w * (len(modes) - 1) / 2)
    ax.set_xticklabels(sets); ax.set_ylabel("10-fold C-index"); ax.set_ylim(.4, 1)
    ax.legend(fontsize=8); ax.set_title("Cox PH concordance by feature set")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_km_groups(panels, path):
    """panels: list of (title, T, E, high_mask, logrank_p)"""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), squeeze=False)
    for ax, (title, T, E, hi, p) in zip(axes[0], panels):
        for mask, name, c in [(~hi, "low risk", "#2a9d8f"), (hi, "high risk", "#e76f51")]:
            t, s = kaplan_meier(T[mask], E[mask]); ax.step(t, s, where="post", c=c, label=f"{name} (n={mask.sum()})")
        ax.set_title(f"{title}\nlog-rank p = {p:.2g}", fontsize=9); ax.set_xlabel("Survival days")
        ax.set_ylabel("Survival probability"); ax.legend(fontsize=8); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_forest(summary, path, top=15):
    s = summary.reindex(summary.coef.abs().sort_values(ascending=False).index[:top]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(s.coef, range(len(s)), xerr=s.ci_width / 2, fmt="s", c="k", mfc="w", capsize=3)
    ax.axvline(0, c="grey", ls="--", lw=.8); ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=7)
    ax.set_xlabel("log hazard ratio per 1 SD (95% CI)"); ax.set_title(f"Top {top} Cox coefficients (combined model)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_partial(curves, path):
    """curves: list of (title, {label: (times, S)})"""
    fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4), squeeze=False)
    for ax, (title, d) in zip(axes[0], curves):
        for lab, (t, s) in d.items():
            ax.plot(t, s, label=lab)
        ax.set_title(title, fontsize=9); ax.set_xlabel("Survival days"); ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
