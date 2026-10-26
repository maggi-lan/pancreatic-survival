"""End-to-end experiment: NB threshold -> SMOTE -> Cox (clinical / radiomic / combined) -> risk groups."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .cox import CoxPH, concordance_index
from .data import CATEGORICAL_COLS
from .smote import smote_augment


def prune_radiomic(train, rad_cols, penalizer, max_ci_width=None, prune_frac=0.456):
    """Paper step: fit Cox on all radiomic features and remove the high-variance ones (wide error bands).
    If `max_ci_width` is given, drop features whose 95% CI is wider than it (paper: > 1). Otherwise drop the
    `prune_frac` widest-CI fraction (paper removed 410 of 900 = 45.6%). The fraction rule is the default
    because with n~80-100 and penalizer 0.1 the absolute CI widths all land in a narrow band, so a fixed
    cut-off can remove everything or nothing."""
    m = CoxPH(penalizer).fit(train[rad_cols], train.duration, train.event)
    w = m.ci_width_
    keep_mask = w <= max_ci_width if max_ci_width is not None else w <= np.quantile(w, 1 - prune_frac)
    keep = [c for c, k in zip(rad_cols, keep_mask) if k]
    return keep if keep else list(rad_cols)


def fit_three_models(train, clin, rad, penalizer, max_ci_width=None, prune_frac=0.456):
    """Clinical / radiomic / combined Cox models. With no radiomic columns only the clinical model is fitted."""
    if not rad:
        sets = {"clinical": clin}
    else:
        keep = prune_radiomic(train, rad, penalizer, max_ci_width, prune_frac)
        sets = {"clinical": clin, "radiomic": keep, "combined": clin + keep}
    models = {k: CoxPH(penalizer).fit(train[c], train.duration, train.event) for k, c in sets.items()}
    return models, sets


def cross_validate(df, clin, rad, thr, cfg, mode, seed):
    """10-fold CV C-index. mode:
         'none'    - no oversampling
         'in_fold' - SMOTE applied to each training fold only (leak-free; test folds are real patients)
         'leaky'   - SMOTE on the whole dataset before splitting (what the paper did; test folds contain
                     synthetic points derived from training patients -> optimistic)
    Returns (per-fold C-index DataFrame, out-of-fold standardised risk scores of real patients [combined])."""
    feats = clin + rad
    work = df.assign(synthetic=0)
    if mode == "leaky":
        work = smote_augment(df, feats, CATEGORICAL_COLS, thr, k=cfg.get("smote_k", 5), seed=seed)
    skf = StratifiedKFold(cfg["folds"], shuffle=True, random_state=seed)
    rows, oof = [], pd.Series(np.nan, index=work.index)
    for f, (tr, te) in enumerate(skf.split(work, work.event)):
        train, test = work.iloc[tr], work.iloc[te]
        if mode == "in_fold":
            train = smote_augment(train.drop(columns="synthetic"), feats, CATEGORICAL_COLS, thr,
                                  k=cfg.get("smote_k", 5), seed=seed + f)
        models, sets = fit_three_models(train, clin, rad, cfg["penalizer"], cfg["max_ci_width"], cfg["prune_frac"])
        row = {"fold": f, "n_radiomic_kept": len(sets.get("radiomic", []))}
        for k, m in models.items():
            row[k] = concordance_index(test.duration, test.event, m.predict_log_hazard(test[sets[k]]))
        rows.append(row)
        # standardise held-out scores with the *training* score distribution -> comparable across folds
        key = "combined" if "combined" in models else "clinical"
        tr_s = models[key].predict_log_hazard(train[sets[key]])
        oof.iloc[te] = (models[key].predict_log_hazard(test[sets[key]]) - tr_s.mean()) / tr_s.std()
    return pd.DataFrame(rows), oof[work.synthetic == 0]


def summarise(folds_by_repeat):
    allf = pd.concat(folds_by_repeat)
    return pd.DataFrame({k: [allf[k].mean(), allf[k].std()] for k in ["clinical", "radiomic", "combined"] if k in allf},
                        index=["mean_cindex", "std_over_folds"]).T
