"""Run the full pipeline:  python main.py [--data my_data.csv] [--out outputs] [--repeats 3]"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from survpanc.data import make_synthetic, load_csv, split_columns, CATEGORICAL_COLS
from survpanc.naive_bayes import threshold_sweep
from survpanc.smote import smote_augment
from survpanc.cox import logrank_test
from survpanc.pipeline import cross_validate, summarise, fit_three_models
from survpanc import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV with duration, event, clinical columns, rad_* columns "
                                   "(default: synthetic cohort)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3, help="repeated CV runs with different splits")
    ap.add_argument("--penalizer", type=float, default=0.1)
    ap.add_argument("--max-ci-width", type=float, default=None,
                    help="drop radiomic features whose 95%% CI is wider than this (paper: 1). Default: use --prune-frac")
    ap.add_argument("--prune-frac", type=float, default=0.456,
                    help="fraction of widest-CI radiomic features to drop (paper removed 410/900)")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[240, 270, 300, 330, 360])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cfg = dict(folds=a.folds, penalizer=a.penalizer, max_ci_width=a.max_ci_width, prune_frac=a.prune_frac)

    df = load_csv(a.data) if a.data else make_synthetic(seed=a.seed)
    clin, rad = split_columns(df)
    n_ev, n_cens = int(df.event.sum()), int((1 - df.event).sum())
    if len(df) < 30 or min(n_ev, n_cens) < 2:
        raise SystemExit(f"Cohort too small for this analysis ({len(df)} patients, {n_ev} deaths, {n_cens} censored). "
                         "Need at least ~30 patients with some censored cases; use the full dataset.")
    if min(n_ev, n_cens) < a.folds:
        print(f"WARNING: only {min(n_ev, n_cens)} patients in the smaller outcome class -> using "
              f"{min(n_ev, n_cens)} CV folds instead of {a.folds}")
        a.folds = min(n_ev, n_cens); cfg["folds"] = a.folds
    src = a.data or "SYNTHETIC cohort (seed=%d)" % a.seed
    print(f"Data: {src} | {len(df)} patients, {len(clin)} clinical, {len(rad)} radiomic features, "
          f"{int((1 - df.event).sum())} censored")

    # 1 ---- Naive Bayes threshold selection ---------------------------------------------------
    nb_cols = [c for c in ["stage_grouping", "chemo_after_sbrt", "tumor_width", "resectability_nccn",
                           "radiation_conventional"] if c in clin] or clin
    sweep = threshold_sweep(df, nb_cols, a.thresholds, seed=a.seed)
    thr = int(sweep.loc[sweep.balanced_accuracy.idxmax(), "threshold_days"])
    print("\nNaive Bayes threshold sweep\n", sweep.round(3).to_string(index=False), f"\n-> best threshold: {thr} days")
    plots.fig_threshold(sweep, f"{a.out}/fig_threshold.png")
    plots.fig_cohort(df.assign(synthetic=0), None, f"{a.out}/fig_cohort.png", "Cohort (before SMOTE)")
    aug = smote_augment(df, clin + rad, CATEGORICAL_COLS, thr, seed=a.seed)
    print(f"SMOTE: {len(df)} real + {int(aug.synthetic.sum())} synthetic patients")
    plots.fig_cohort(aug, thr, f"{a.out}/fig_cohort_smote.png", "After SMOTE (x = synthetic)")

    # 2 ---- Cross-validated Cox models (3 feature sets x 3 augmentation modes) ----------------------
    results, oof_main, kept = {}, None, []
    for mode in ["none", "in_fold", "leaky"]:
        fl = []
        for r in range(a.repeats):
            f, oof = cross_validate(df, clin, rad, thr, cfg, mode, seed=a.seed + 100 * r)
            fl.append(f)
            if mode == "in_fold" and r == 0:
                oof_main = oof
        results[mode] = summarise(fl)
        kept.append(pd.concat(fl).n_radiomic_kept.mean())
        print(f"\nCox 10-fold C-index, SMOTE mode = {mode}  ({time.time() - t0:.0f}s)\n", results[mode].round(3))
    if rad:
        print(f"\nMean radiomic features kept after CI pruning: {kept[1]:.0f} of {len(rad)}")
    else:
        print("\nNo radiomic (rad_*) columns found -> clinical-only analysis.")
    plots.fig_cindex({"no SMOTE": results["none"], "SMOTE in-fold (correct)": results["in_fold"],
                      "SMOTE before CV (paper, leaky)": results["leaky"]}, f"{a.out}/fig_cindex.png")

    # 3 ---- Risk stratification: out-of-fold scores (honest) vs in-sample (paper style) ----------------
    real = df.loc[oof_main.index]
    hi_oof = (oof_main > oof_main.median()).values
    T, E = real.duration.values, real.event.values
    _, p_oof = logrank_test(T[~hi_oof], E[~hi_oof], T[hi_oof], E[hi_oof])
    models, sets = fit_three_models(aug, clin, rad, a.penalizer, a.max_ci_width, a.prune_frac)   # final models on augmented data
    fkey = "combined" if "combined" in models else "clinical"
    comb = models[fkey]
    eta_in = comb.predict_log_hazard(df[sets[fkey]])
    hi_in = eta_in > np.median(eta_in)
    D, V = df.duration.values, df.event.values
    _, p_in = logrank_test(D[~hi_in], V[~hi_in], D[hi_in], V[hi_in])
    print(f"\nLog-rank p (out-of-fold risk groups): {p_oof:.3g} | (in-sample, optimistic): {p_in:.3g}")
    plots.fig_km_groups([("Out-of-fold risk groups (honest)", T, E, hi_oof, p_oof),
                         ("In-sample risk groups (paper-style)", D, V, hi_in, p_in)], f"{a.out}/fig_km_groups.png")

    # 4 ---- Interpretation -----------------------------------------------------------------------
    summ = comb.summary()
    summ.sort_values("coef", key=abs, ascending=False).head(25).to_csv(f"{a.out}/top_coefficients.csv")
    plots.fig_forest(summ, f"{a.out}/fig_forest.png")
    tgrid = np.arange(0, int(df.duration.max()) + 1, 10)
    base = pd.DataFrame([df[sets[fkey]].mean()] * 1)

    def curve(**mods):
        x = base.copy()
        for k, v in mods.items():
            x[k] = v
        return tgrid, comb.predict_survival(x, tgrid)[0]
    curves = []
    cols = sets[fkey]
    if "chemo_after_sbrt" in cols:
        curves.append(("Effect of chemotherapy (others at mean)",
                       {"no chemo": curve(chemo_after_sbrt=0), "chemo": curve(chemo_after_sbrt=1)}))
    for scol in ("stage_grouping", "stage_ordinal"):
        if scol in cols:
            lv = sorted(df[scol].round().unique())[:6]
            curves.append((f"Effect of {scol} (others at mean)", {f"{scol}={int(v)}": curve(**{scol: v}) for v in lv}))
    for bcol in ("tp53_mutated", "kras_g12d", "smad4_altered"):
        if bcol in cols:
            curves.append((f"Effect of {bcol} (others at mean)", {f"{bcol}=0": curve(**{bcol: 0}), f"{bcol}=1": curve(**{bcol: 1})}))
            break
    if curves:
        plots.fig_partial(curves, f"{a.out}/fig_partial_effects.png")

    # 5 ---- Save ------------------------------------------------------------------------------------
    out = dict(data=src, n_patients=len(df), best_threshold_days=thr, cfg=cfg, repeats=a.repeats,
               cindex={m: results[m].round(4).to_dict("index") for m in results},
               logrank_p_out_of_fold=p_oof, logrank_p_in_sample=p_in,
               radiomic_features_kept_mean=kept[1] if rad else 0, runtime_s=round(time.time() - t0, 1))
    if hasattr(df, "attrs") and df.attrs.get("true_signal_features"):
        top = set(summ.reindex(summ.coef.abs().sort_values(ascending=False).index[:25]).index)
        out["true_radiomic_signal_in_top25"] = len(top & set(df.attrs["true_signal_features"]))
    json.dump(out, open(f"{a.out}/results.json", "w"), indent=2)
    sweep.to_csv(f"{a.out}/nb_threshold_sweep.csv", index=False)
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {a.out}/")


if __name__ == "__main__":
    main()
