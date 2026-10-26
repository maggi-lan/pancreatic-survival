"""Data loading and a synthetic stand-in cohort.

The original Stanford dataset (80 patients, MRI radiomics + clinical data) is private (NDA),
so this module can (a) load your own CSV, or (b) generate a *synthetic* cohort with the same
structure: 12 clinical features, 900 correlated radiomic features, survival days + event flag.
Synthetic results demonstrate that the pipeline works; they are NOT clinical findings.
"""
import numpy as np
import pandas as pd

CLINICAL_COLS = [
    "radiation_conventional", "resectability_nccn", "chemo_after_sbrt", "tumor_location",
    "stage_grouping", "primary_tumor_T", "kps", "gender",
    "tumor_depth", "tumor_height", "tumor_width", "tumor_volume",
]
# columns that are categorical / binary (rounded after SMOTE interpolation)
CATEGORICAL_COLS = [
    "radiation_conventional", "resectability_nccn", "chemo_after_sbrt", "tumor_location",
    "stage_grouping", "primary_tumor_T", "kps", "gender",
    # TCGA-PAAD clinical file (see prepare_data.py)
    "gender_male", "stage_ordinal", "kras_mutated", "kras_g12d", "tp53_mutated", "smad4_altered", "is_pdac",
    "t_stage", "n_stage", "grade_ordinal", "adjuvant_therapy", "radiation_therapy",
]
RADIOMIC_PREFIX = "rad_"


def make_synthetic(n=80, n_radiomic=900, n_latent=40, n_signal=6, seed=0):
    """Synthetic cohort with a known ground-truth Weibull-Cox hazard."""
    rng = np.random.default_rng(seed)
    stage = rng.choice([0, 1, 2, 3], n, p=[0.03, 0.17, 0.10, 0.70])
    t_stage = rng.choice([2, 3, 4], n, p=[0.2, 0.5, 0.3])
    resect = rng.choice([0, 1, 2], n, p=[0.10, 0.35, 0.55])
    loc = rng.choice([0, 1, 2], n)
    chemo = rng.binomial(1, 0.45, n)
    rad_type = rng.binomial(1, 0.35, n)
    kps = rng.choice([60, 70, 80, 90, 100], n, p=[0.05, 0.15, 0.30, 0.35, 0.15])
    gender = rng.binomial(1, 0.5, n)
    width = np.clip(rng.normal(4.0, 1.0, n), 1.5, 8.0)
    depth = width * rng.uniform(0.7, 1.0, n)
    height = width * rng.uniform(0.7, 1.0, n)
    volume = np.pi / 6 * width * depth * height
    clin = pd.DataFrame({
        "radiation_conventional": rad_type, "resectability_nccn": resect, "chemo_after_sbrt": chemo,
        "tumor_location": loc, "stage_grouping": stage, "primary_tumor_T": t_stage, "kps": kps,
        "gender": gender, "tumor_depth": depth, "tumor_height": height, "tumor_width": width,
        "tumor_volume": volume,
    })

    # Radiomic features: many strongly correlated columns driven by a few latent factors.
    Z = rng.normal(size=(n, n_latent))
    A = rng.normal(size=(n_latent, n_radiomic)) * (rng.random((n_latent, n_radiomic)) < 0.25)
    Xr = Z @ A / np.sqrt(n_latent * 0.25) + 0.5 * rng.normal(size=(n, n_radiomic))
    rad = pd.DataFrame(Xr, columns=[f"{RADIOMIC_PREFIX}{i:04d}" for i in range(n_radiomic)])

    # Ground-truth linear predictor
    z = lambda v: (v - v.mean()) / v.std()
    eta_clin = (0.30 * stage + 0.25 * (t_stage - 3) + 0.20 * resect - 0.7 * chemo
                + 0.30 * z(volume) - 0.02 * (kps - 80))
    signal_idx = rng.choice(n_radiomic, n_signal, replace=False)
    signs = rng.choice([-1.0, 1.0], n_signal)
    eta_rad = (z(Xr[:, signal_idx]) * (0.35 * signs)).sum(axis=1)
    eta = eta_clin + eta_rad

    k = 1.5                                    # Weibull shape
    scale = 700 / np.log(2) ** (1 / k)        # median ~700 days at eta=0
    T = scale * (-np.log(rng.random(n))) ** (1 / k) * np.exp(-eta / k)
    C = rng.uniform(300, 1300, n)              # administrative censoring
    duration = np.maximum(1, np.round(np.minimum(T, C))).astype(int)
    event = (T <= C).astype(int)

    df = pd.concat([clin, rad], axis=1)
    df["duration"] = duration
    df["event"] = event
    df.attrs["true_signal_features"] = [rad.columns[i] for i in signal_idx]
    return df


def load_csv(path, duration_col="duration", event_col="event"):
    """Load a real dataset. Needs `duration` (days), `event` (1=death, 0=censored),
    the 12 clinical columns named in CLINICAL_COLS (extra/missing ones are tolerated, see README)
    and radiomic columns prefixed with `rad_`."""
    df = pd.read_csv(path).rename(columns={duration_col: "duration", event_col: "event"})
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values; impute or drop them first.")
    return df


def split_columns(df):
    rad = [c for c in df.columns if c.startswith(RADIOMIC_PREFIX)]
    clin = [c for c in df.columns if c not in rad and c not in ("duration", "event", "synthetic")]
    return clin, rad
