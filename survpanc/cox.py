"""Survival-analysis primitives implemented from scratch with numpy/scipy:
Cox proportional hazards (L2-penalised, Breslow ties), Kaplan-Meier, log-rank test, Harrell's C-index.
"""
import numpy as np
from scipy.stats import chi2


# ----------------------------------------------------------------------------- metrics
def concordance_index(T, E, risk):
    """Harrell's C: over comparable pairs (i has an event, T_j > T_i), fraction where the model gives
    patient i the higher risk. Risk ties count 0.5. 1.0 = perfect, 0.5 = random."""
    T, E, r = np.asarray(T, float), np.asarray(E, bool), np.asarray(risk, float)
    num = den = 0.0
    for i in np.where(E)[0]:
        m = T > T[i]
        den += m.sum()
        num += (r[i] > r[m]).sum() + 0.5 * (r[i] == r[m]).sum()
    return num / den if den > 0 else np.nan


def kaplan_meier(T, E):
    """S_KM(t) = prod_{t_i <= t} (n_i - d_i)/n_i.  Returns (times, survival) starting at (0, 1)."""
    T, E = np.asarray(T, float), np.asarray(E, int)
    times = np.unique(T[E == 1])
    surv, s = [], 1.0
    for t in times:
        n = (T >= t).sum()
        d = ((T == t) & (E == 1)).sum()
        s *= 1 - d / n
        surv.append(s)
    return np.concatenate([[0.0], times]), np.concatenate([[1.0], surv])


def logrank_test(T1, E1, T2, E2):
    """Two-group log-rank test. Returns (chi-square statistic, p-value) with 1 d.o.f."""
    T = np.concatenate([T1, T2]).astype(float)
    E = np.concatenate([E1, E2]).astype(int)
    g = np.concatenate([np.zeros(len(T1)), np.ones(len(T2))])
    O1 = E1.sum() if hasattr(E1, "sum") else sum(E1)
    exp = var = 0.0
    for t in np.unique(T[E == 1]):
        at = T >= t
        n, n1 = at.sum(), (at & (g == 0)).sum()
        d = ((T == t) & (E == 1)).sum()
        exp += d * n1 / n
        if n > 1:
            var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    stat = (O1 - exp) ** 2 / var
    return float(stat), float(chi2.sf(stat, 1))


# ----------------------------------------------------------------------------- Cox model
class CoxPH:
    """h(t|x) = h0(t) exp(beta^T x), fitted by Newton-Raphson on the Breslow partial likelihood with a
    ridge penalty:   minimise  -loglik/n + 0.5 * penalizer * ||beta||^2   (same scaling as lifelines'
    CoxPHFitter(penalizer=..., l1_ratio=0)). Features are z-scored internally, so `coef_` is
    log hazard ratio per 1 SD of the feature.
    """

    def __init__(self, penalizer=0.1, max_iter=100, tol=1e-10):
        self.penalizer, self.max_iter, self.tol = penalizer, max_iter, tol

    # -- helpers ------------------------------------------------------------
    def _prep(self, X, T, E):
        order = np.argsort(-T, kind="stable")          # descending time -> risk sets are prefixes
        self._X, self._T, self._E = X[order], T[order], E[order]
        groups, i, n = [], 0, len(T)
        while i < n:
            j = i
            while j < n and self._T[j] == self._T[i]:
                j += 1
            groups.append((i, j, i + np.where(self._E[i:j] == 1)[0]))
            i = j
        self._groups = groups

    def _terms(self, beta, hess=True):
        """Partial log-likelihood, its gradient and (optionally) the observed information matrix
        info = sum_events d_t * Cov_w(x | risk set at t)  (Breslow ties). Written with stacked,
        centred risk-set rows so the p x p matrix is one matmul (fast when p >> n)."""
        X, p = self._X, self._X.shape[1]
        eta = X @ beta
        eta = eta - eta.max()                           # numerical stability (cancels in the likelihood)
        w = np.exp(eta)
        cw = np.cumsum(w)                               # S0 over prefixes (risk sets are prefixes)
        S1 = np.cumsum(X * w[:, None], axis=0)
        ll, g, rows = 0.0, np.zeros(p), []
        for s, e, ev in self._groups:
            d = len(ev)
            if not d:
                continue
            S0, m = cw[e - 1], S1[e - 1] / cw[e - 1]
            ll += eta[ev].sum() - d * np.log(S0)
            g += X[ev].sum(0) - d * m
            if hess:
                rows.append(np.sqrt(d * w[:e] / S0)[:, None] * (X[:e] - m))
        info = None
        if hess and rows:
            R = np.vstack(rows)
            info = R.T @ R
        elif hess:
            info = np.zeros((p, p))
        return ll, g, info

    def _obj(self, beta):
        return -self._terms(beta, hess=False)[0] / self._n + 0.5 * self.penalizer * beta @ beta

    # -- API ----------------------------------------------------------------
    def fit(self, X, T, E):
        self.columns_ = list(getattr(X, "columns", range(np.shape(X)[1])))
        X, T, E = np.asarray(X, float), np.asarray(T, float), np.asarray(E, int)
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = (X - self.mu_) / self.sd_
        self._n, p = len(T), Xs.shape[1]
        self._prep(Xs, T, E)

        beta = np.zeros(p)
        f = self._obj(beta)
        for it in range(self.max_iter):
            ll, g, info = self._terms(beta)
            grad = -g / self._n + self.penalizer * beta
            H = info / self._n + self.penalizer * np.eye(p)
            step = np.linalg.solve(H, grad)
            t = 1.0
            while t > 1e-12:                            # backtracking line search
                fn = self._obj(beta - t * step)
                if fn <= f - 1e-4 * t * (grad @ step):
                    break
                t /= 2
            beta, df_ = beta - t * step, f - fn
            f = fn
            if abs(df_) < self.tol * (1 + abs(f)) and np.abs(t * step).max() < 1e-6:
                break
        self.n_iter_ = it + 1
        self.coef_ = beta
        ll, g, info = self._terms(beta)
        H = info / self._n + self.penalizer * np.eye(p)
        self.se_ = np.sqrt(np.diag(np.linalg.inv(H)) / self._n)   # same convention as lifelines
        self.ci_width_ = 2 * 1.96 * self.se_
        self.log_likelihood_ = ll
        self._baseline(beta)
        return self

    def _baseline(self, beta):
        """Breslow estimator of the cumulative baseline hazard (baseline = all z-scored features at 0)."""
        w = np.exp(self._X @ beta)
        S0, times, haz = 0.0, [], []
        for s, e, ev in self._groups:
            S0 += w[s:e].sum()
            if len(ev):
                times.append(self._T[s]); haz.append(len(ev) / S0)
        idx = np.argsort(times)
        self.base_times_ = np.array(times)[idx]
        self.base_cumhaz_ = np.cumsum(np.array(haz)[idx])

    def _z(self, X):
        return (np.asarray(X, float) - self.mu_) / self.sd_

    def predict_log_hazard(self, X):
        """Linear predictor eta = beta^T z (higher = higher risk)."""
        return self._z(X) @ self.coef_

    def predict_survival(self, X, times):
        """S(t|x) = exp(-H0(t) * exp(eta)). Returns array (n_samples, n_times)."""
        eta = self.predict_log_hazard(X)
        i = np.searchsorted(self.base_times_, times, side="right") - 1
        H0 = np.where(i >= 0, self.base_cumhaz_[np.clip(i, 0, None)], 0.0)
        return np.exp(-np.outer(np.exp(eta), H0))

    def summary(self):
        import pandas as pd
        return pd.DataFrame({"coef": self.coef_, "se": self.se_, "ci_width": self.ci_width_,
                             "hazard_ratio": np.exp(self.coef_)}, index=self.columns_)
