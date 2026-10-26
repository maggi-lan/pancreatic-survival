"""Convert the raw TCGA-PAAD clinical CSV into the table main.py expects.

    python prepare_data.py --input data/raw/pancreas_raw.csv --out data/pancreas.csv

Columns are matched by prefix (case-insensitive) so truncated/variant header names work:
Patient_ID, Gender, Age_at_Diagn*, Histological_T*, Tumor_Stage, Survival_Statu*, Overall_Surviv*,
KRAS_Mutatio*, TP53_Mutatio*, SMAD4_Status.

Output columns: duration (days), event (1=DECEASED), age, gender_male, stage_ordinal, kras_mutated,
kras_g12d, tp53_mutated, smad4_altered  (+ is_pdac if the cohort mixes histologies).
"""
import argparse, re
import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.4375
STAGE_ORDER = ["IA", "IB", "IIA", "IIB", "III", "IV"]


def find(df, prefix):
    hits = [c for c in df.columns if c.lower().startswith(prefix.lower())]
    if not hits:
        raise SystemExit(f"Could not find a column starting with '{prefix}'. Columns are: {list(df.columns)}")
    return hits[0]


def stage_to_ordinal(s):
    m = re.search(r"\b(I{1,3}V?|IV)([ABC]?)\b", str(s).upper().replace("STAGE", "").strip())
    key = (m.group(1) + m.group(2)) if m else None
    if key is None:
        return np.nan
    if key in STAGE_ORDER:
        return STAGE_ORDER.index(key)
    base = {"I": "IA", "II": "IIA", "IV": "IV"}.get(key)          # coarse stage without sub-letter
    return STAGE_ORDER.index(base) if base else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/pancreas.csv")
    ap.add_argument("--time-unit", choices=["auto", "months", "days"], default="auto")
    a = ap.parse_args()

    raw = pd.read_csv(a.input)
    c = {k: find(raw, p) for k, p in dict(
        id="Patient_ID", gender="Gender", age="Age", hist="Histological", stage="Tumor_Stage",
        status="Survival_Statu", time="Overall_Surviv", kras="KRAS", tp53="TP53", smad4="SMAD4").items()}
    print("Column mapping:", c)

    t = pd.to_numeric(raw[c["time"]], errors="coerce")
    unit = a.time_unit
    if unit == "auto":
        name = c["time"].lower()
        unit = "months" if "month" in name else "days" if "day" in name else ("months" if t.max() < 200 else "days")
        print(f"Time unit not given -> assuming '{unit}' (max value {t.max():.1f}). Override with --time-unit.")
    days = t * DAYS_PER_MONTH if unit == "months" else t

    status = raw[c["status"]].astype(str).str.upper().str.strip()
    if not set(status.unique()) <= {"DECEASED", "LIVING"}:
        print("WARNING: unexpected survival-status values:", status.unique())
    out = pd.DataFrame({
        "duration": days.round().astype("Int64"),
        "event": (status == "DECEASED").astype(int),
        "age": pd.to_numeric(raw[c["age"]], errors="coerce"),
        "gender_male": (raw[c["gender"]].astype(str).str.upper().str.strip() == "MALE").astype(int),
        "stage_ordinal": raw[c["stage"]].map(stage_to_ordinal),
        "kras_mutated": (~raw[c["kras"]].astype(str).str.upper().str.strip().isin(["WT", "WILDTYPE", "NAN", ""])).astype(int),
        "kras_g12d": (raw[c["kras"]].astype(str).str.upper().str.strip() == "G12D").astype(int),
        "tp53_mutated": (~raw[c["tp53"]].astype(str).str.upper().str.strip().isin(["WT", "WILDTYPE", "NAN", ""])).astype(int),
        "smad4_altered": (~raw[c["smad4"]].astype(str).str.upper().str.strip().isin(["WILDTYPE", "WT", "NAN", ""])).astype(int),
    })
    hist = raw[c["hist"]].astype(str)
    if hist.nunique() > 1:
        out["is_pdac"] = hist.str.contains("ductal", case=False).astype(int)

    before = len(out)
    bad = out[["duration", "age", "stage_ordinal"]].isna().any(axis=1) | (out.duration <= 0)
    if bad.any():
        print(f"Dropping {bad.sum()} rows with missing/invalid time, age or stage")
    out = out[~bad].copy()
    out["duration"] = out.duration.astype(int)
    for col in out.columns:                       # drop constant columns (carry no information)
        if col not in ("duration", "event") and out[col].nunique() < 2:
            print(f"Dropping constant column {col}"); out = out.drop(columns=col)

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nWrote {a.out}: {len(out)}/{before} patients, {int(out.event.sum())} deaths, "
          f"{int((1 - out.event).sum())} censored, median follow-up {out.duration.median():.0f} days")
    print(out.describe().T[["mean", "min", "max"]].round(2))
    if len(out) < 60:
        print("NOTE: small cohort - expect very wide confidence intervals.")


if __name__ == "__main__":
    main()
