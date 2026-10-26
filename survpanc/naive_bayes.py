"""Discretiser + multinomial Naive Bayes with Laplace smoothing (as in Section 4 of the paper)."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold


class Discretizer:
    """Features with <= max_cat distinct values are treated as categories; others get quantile bins."""

    def __init__(self, n_bins=3, max_cat=5):
        self.n_bins, self.max_cat = n_bins, max_cat

    def fit(self, X):
        X = np.asarray(X, float)
        self.maps_, self.edges_, self.V_ = [], [], []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            if len(u) <= self.max_cat:
                self.maps_.append({v: i for i, v in enumerate(u)}); self.edges_.append(None)
                self.V_.append(len(u))
            else:
                e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                self.maps_.append(None); self.edges_.append(e); self.V_.append(len(e) + 1)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        out = np.zeros(X.shape, int)
        for j in range(X.shape[1]):
            if self.maps_[j] is not None:   # unseen category -> nearest known value
                keys = np.array(list(self.maps_[j]))
                out[:, j] = [self.maps_[j][keys[np.abs(keys - v).argmin()]] for v in X[:, j]]
            else:
                out[:, j] = np.searchsorted(self.edges_[j], X[:, j], side="right")
        return out


class NaiveBayes:
    """phi_{dk|y} = (1 + count(x_d=k, y)) / (V_d + count(y));   phi_y = count(y=1)/n."""

    def fit(self, Xd, y, V):
        Xd, y = np.asarray(Xd), np.asarray(y)
        self.V_ = V
        self.phi_y_ = y.mean()
        self.phi_ = {}
        for c in (0, 1):
            m = y == c
            self.phi_[c] = [
                (1 + np.bincount(Xd[m, d], minlength=V[d])) / (V[d] + m.sum()) for d in range(Xd.shape[1])
            ]
        return self

    def predict(self, Xd):
        Xd = np.asarray(Xd)
        lp = {c: np.log(self.phi_y_ if c == 1 else 1 - self.phi_y_) for c in (0, 1)}
        s = {c: lp[c] + sum(np.log(self.phi_[c][d][Xd[:, d]]) for d in range(Xd.shape[1])) for c in (0, 1)}
        return (s[1] > s[0]).astype(int)


def risk_labels(duration, event, thr):
    """1 = died before thr (high risk), 0 = survived >= thr (low risk), -1 = censored before thr (unknown)."""
    duration, event = np.asarray(duration), np.asarray(event)
    y = np.full(len(duration), -1)
    y[(duration < thr) & (event == 1)] = 1
    y[duration >= thr] = 0
    return y


def threshold_sweep(df, clinical_cols, thresholds=(240, 270, 300, 330, 360), n_splits=10, n_repeats=3, seed=0):
    """Balanced accuracy of the Naive Bayes classifier for each survival-day threshold."""
    rows = []
    for thr in thresholds:
        y_all = risk_labels(df.duration.values, df.event.values, thr)
        m = y_all >= 0
        X, y = df.loc[m, clinical_cols].values, y_all[m]
        k = min(n_splits, np.bincount(y, minlength=2).min())
        if k < 2:
            rows.append((thr, np.nan, int(y.sum()), int((1 - y).sum()))); continue
        accs = []
        for tr, te in RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=seed).split(X, y):
            disc = Discretizer().fit(X[tr])
            nb = NaiveBayes().fit(disc.transform(X[tr]), y[tr], disc.V_)
            accs.append(balanced_accuracy_score(y[te], nb.predict(disc.transform(X[te]))))
        rows.append((thr, float(np.mean(accs)), int(y.sum()), int((1 - y).sum())))
    return pd.DataFrame(rows, columns=["threshold_days", "balanced_accuracy", "n_high_risk", "n_low_risk"])
