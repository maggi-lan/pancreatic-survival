import numpy as np, pandas as pd
from survpanc.cox import CoxPH, concordance_index, kaplan_meier, logrank_test
from survpanc.naive_bayes import Discretizer, NaiveBayes, risk_labels
from survpanc.smote import smote_augment
from survpanc.data import make_synthetic, split_columns, CATEGORICAL_COLS


def test_concordance_perfect_and_reversed():
    T = np.array([1, 2, 3, 4]); E = np.ones(4)
    assert concordance_index(T, E, [4, 3, 2, 1]) == 1.0
    assert concordance_index(T, E, [1, 2, 3, 4]) == 0.0
    assert concordance_index(T, E, [1, 1, 1, 1]) == 0.5


def test_kaplan_meier_hand_example():
    # times 1(event) 2(censored) 3(event): S(1)=3/4? use n=3: S(1)=2/3, S(3)=2/3*(0/1)=0
    t, s = kaplan_meier([1, 2, 3], [1, 0, 1])
    assert np.allclose(t, [0, 1, 3]) and np.allclose(s, [1, 2 / 3, 0])


def test_logrank_separates_groups_and_not_identical():
    rng = np.random.default_rng(0)
    a = rng.exponential(1, 200); b = rng.exponential(3, 200)
    _, p_diff = logrank_test(a, np.ones(200, int), b, np.ones(200, int))
    _, p_same = logrank_test(a[:100], np.ones(100, int), a[100:], np.ones(100, int))
    assert p_diff < 1e-6 and p_same > 0.01


def test_cox_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 4)); T = rng.exponential(1, 40).round(1) + 0.1; E = rng.binomial(1, .7, 40)
    m = CoxPH(penalizer=0.1); m._n = 40
    m._prep((X - X.mean(0)) / X.std(0), T, E)
    b = rng.normal(size=4) * 0.3
    _, g, info = m._terms(b)
    num = np.array([(m._terms(b + 1e-6 * np.eye(4)[j], False)[0] - m._terms(b - 1e-6 * np.eye(4)[j], False)[0]) / 2e-6
                    for j in range(4)])
    assert np.allclose(g, num, atol=1e-4)
    # information matrix == -d(grad)/d(beta)
    num_h = np.array([(m._terms(b + 1e-5 * np.eye(4)[j])[1] - m._terms(b - 1e-5 * np.eye(4)[j])[1]) / 2e-5
                      for j in range(4)])
    assert np.allclose(info, -num_h, atol=1e-3)


def test_cox_recovers_true_coefficients():
    rng = np.random.default_rng(2)
    n = 4000; X = rng.normal(size=(n, 3)); beta = np.array([0.8, -0.5, 0.0])
    T = rng.exponential(1, n) / np.exp(X @ beta); C = rng.exponential(2, n)
    m = CoxPH(penalizer=0.0).fit(X, np.minimum(T, C), (T <= C).astype(int))
    assert np.allclose(m.coef_ / 1.0, beta, atol=0.08)       # features already ~unit variance
    # survival curves are valid
    S = m.predict_survival(X[:5], np.array([0.1, 1.0, 5.0]))
    assert (np.diff(S, axis=1) <= 1e-12).all() and (S > 0).all() and (S <= 1).all()


def test_ridge_shrinks_coefficients():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 5)); T = rng.exponential(1, 100) / np.exp(X[:, 0]); E = np.ones(100, int)
    a = CoxPH(penalizer=0.01).fit(X, T, E).coef_; b = CoxPH(penalizer=1.0).fit(X, T, E).coef_
    assert np.linalg.norm(b) < np.linalg.norm(a)


def test_naive_bayes_laplace_formula():
    Xd = np.array([[0], [0], [1], [1]]); y = np.array([1, 1, 0, 0])
    nb = NaiveBayes().fit(Xd, y, [2])
    assert np.allclose(nb.phi_[1][0], [(1 + 2) / (2 + 2), (1 + 0) / (2 + 2)])
    assert nb.phi_y_ == 0.5


def test_smote_balances_classes_and_keeps_valid_values():
    df = make_synthetic(n=80, n_radiomic=30, seed=5)
    clin, rad = split_columns(df)
    aug = smote_augment(df, clin + rad, CATEGORICAL_COLS, thr=270, seed=0)
    y = risk_labels(aug.duration.values, aug.event.values, 270)
    syn = aug[aug.synthetic == 1]
    assert (y == 1).sum() >= (y == 0).sum() - 1 or len(syn) == 0
    assert set(syn.chemo_after_sbrt.unique()) <= {0, 1} and (syn.duration >= 1).all()
    assert (aug.iloc[:80].drop(columns="synthetic").values == df.values).all()   # originals untouched


def test_synthetic_cohort_shape():
    df = make_synthetic()
    clin, rad = split_columns(df)
    assert len(df) == 80 and len(clin) == 12 and len(rad) == 900
