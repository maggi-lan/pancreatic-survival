"""SMOTE for survival data (numpy implementation; same k=5 nearest-neighbour scheme as imbalanced-learn).

Synthetic patient = a + u*(b-a) between a minority sample `a` and one of its k nearest minority
neighbours `b` (u~U(0,1)), applied to features AND survival days; the event flag is inherited
from the nearer parent; categorical columns are rounded back to valid values.
"""
import numpy as np
import pandas as pd
from .naive_bayes import risk_labels


def smote_augment(df, feature_cols, cat_cols, thr, k=5, seed=0):
    """Oversample the high-risk class (died before `thr`) until it matches the low-risk class."""
    rng = np.random.default_rng(seed)
    y = risk_labels(df.duration.values, df.event.values, thr)
    hi, lo = np.where(y == 1)[0], np.where(y == 0)[0]
    n_new = len(lo) - len(hi)
    out = df.copy()
    out["synthetic"] = 0
    if len(hi) < 2 or n_new <= 0:
        return out
    cols = list(feature_cols) + ["duration"]
    M = df.iloc[hi][cols].values.astype(float)
    sd = M.std(axis=0); sd[sd == 0] = 1
    Ms = (M - M.mean(0)) / sd                       # distances in standardised space
    G = Ms @ Ms.T
    sq = np.diag(G)
    d2 = sq[:, None] + sq[None, :] - 2 * G
    np.fill_diagonal(d2, np.inf)
    kk = min(k, len(hi) - 1)
    nbrs = np.argsort(d2, axis=1)[:, :kk]
    ev = df.iloc[hi]["event"].values
    rows = []
    for _ in range(n_new):
        a = rng.integers(len(hi)); b = nbrs[a, rng.integers(kk)]; u = rng.random()
        rows.append(np.append(M[a] + u * (M[b] - M[a]), ev[a] if u < 0.5 else ev[b]))
    new = pd.DataFrame(rows, columns=cols + ["event"])
    for c in cat_cols:
        if c in new:
            new[c] = new[c].round()
    new["duration"] = np.maximum(1, new["duration"].round()).astype(int)
    new["event"] = new["event"].astype(int)
    new["synthetic"] = 1
    return pd.concat([out, new[list(out.columns)]], ignore_index=True)
