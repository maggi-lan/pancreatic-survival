"""Convert a cBioPortal clinical-data TSV (TCGA pancreatic cohort) into data/pancreas.csv for main.py.

    python prepare_cbioportal.py --input data/raw/clinical.tsv --out data/pancreas.csv

Columns are found by keyword (cBioPortal header names), e.g. "Overall Survival (Months)",
"Overall Survival Status", "Diagnosis Age", "Sex", "American Joint Committee on Cancer Tumor Stage Code",
"Mutation Count", "Fraction Genome Altered", "Adjuvant Postoperative Targeted Therapy Administered Indicator", ...
It prints what it matched and what it could not find. Columns with > --max-missing missing values are dropped,
then patients with a missing value in the remaining columns are dropped.
"""
import argparse, os, re
import numpy as np
import pandas as pd
from prepare_data import stage_to_ordinal

NA = ["", "NA", "N/A", "—", "-", "[Not Available]", "[Not Applicable]", "[Unknown]", "[Not Evaluated]", "[Discrepancy]"]
DAYS_PER_MONTH = 30.4375


def find(df, *words, exclude=()):
    for c in df.columns:
        lc = c.lower()
        if all(w in lc for w in words) and not any(x in lc for x in exclude):
            return c
    return None


def yes_no(s):
    s = s.astype(str).str.upper().str.strip()
    return s.map({"YES": 1, "NO": 0, "Y": 1, "N": 0}).astype(float)


def num_from_code(s, prefix):
    """'T3' -> 3, 'N1' -> 1, 'TX'/'NX' -> NaN, 'T3a' -> 3."""
    return pd.to_numeric(s.astype(str).str.upper().str.extract(rf"{prefix}(\d)")[0], errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/pancreas.csv")
    ap.add_argument("--max-missing", type=float, default=0.20, help="drop feature columns with more missing than this")
    ap.add_argument("--keep-all-histologies", action="store_true", help="do not restrict to adenocarcinoma")
    a = ap.parse_args()

    raw = pd.read_csv(a.input, sep="\t", na_values=NA, keep_default_na=True, dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    print(f"Read {len(raw)} rows x {raw.shape[1]} columns")

    # ---- outcome ---------------------------------------------------------------------------------
    tcol = find(raw, "overall survival", "month")
    scol = find(raw, "overall survival", "status") or find(raw, "vital", "status")
    if tcol is None or scol is None:
        cands = [c for c in raw.columns if any(k in c.lower() for k in ("surviv", "death", "vital", "follow"))]
        raise SystemExit(f"Could not find overall-survival columns (time: {tcol}, status: {scol}).\n"
                         f"Survival-like columns in this file: {cands}\nTell me these names and I'll adapt the script.")
    print(f"Outcome: time = '{tcol}' (months -> days), status = '{scol}'")
    time_days = pd.to_numeric(raw[tcol], errors="coerce") * DAYS_PER_MONTH
    st = raw[scol].astype(str).str.upper()
    event = np.where(st.str.contains("DECEASED|DEAD|^1"), 1, np.where(st.str.contains("LIVING|ALIVE|^0"), 0, np.nan))

    out = pd.DataFrame({"duration": time_days.round(), "event": event})
    feats = {}

    def add(name, series, src):
        if series is not None:
            feats[name] = series; print(f"  feature {name:22s} <- '{src}'")

    # ---- features --------------------------------------------------------------------------------
    c = find(raw, "diagnosis age");                       add("age", pd.to_numeric(raw[c], errors="coerce") if c else None, c)
    c = find(raw, "sex");                                 add("gender_male", (raw[c].str.upper().str.strip() == "MALE").where(raw[c].notna()).astype(float) if c else None, c)
    c = find(raw, "tumor stage code") or find(raw, "primary tumor", "pathologic");  add("t_stage", num_from_code(raw[c], "T") if c else None, c)
    c = find(raw, "lymph node", "stage code") or find(raw, "lymph node", "pathologic");  add("n_stage", num_from_code(raw[c], "N") if c else None, c)
    c = find(raw, "disease stage") or find(raw, "neoplasm", "stage", exclude=("code", "version")); add("stage_ordinal", raw[c].map(stage_to_ordinal) if c else None, c)
    c = find(raw, "histologic grade");                    add("grade_ordinal", num_from_code(raw[c], "G") if c else None, c)
    c = find(raw, "mutation count");                      add("mutation_count", pd.to_numeric(raw[c], errors="coerce") if c else None, c)
    c = find(raw, "fraction genome");                     add("fraction_genome_altered", pd.to_numeric(raw[c], errors="coerce") if c else None, c)
    c = find(raw, "adjuvant", "postoperative", exclude=("neoadjuvant",)) or find(raw, "adjuvant", "therapy", exclude=("neoadjuvant",));  add("adjuvant_therapy", yes_no(raw[c]) if c else None, c)
    c = find(raw, "radiation", "therapy", exclude=("adjuvant",)) or find(raw, "radiation");  add("radiation_therapy", yes_no(raw[c]) if c else None, c)
    for k, v in feats.items():
        out[k] = v

    # ---- cohort filter -----------------------------------------------------------------------------
    hcol = find(raw, "cancer type detailed") or find(raw, "histolog", exclude=("grade",))
    if hcol and not a.keep_all_histologies:
        m = raw[hcol].astype(str).str.contains("adenocarcinoma", case=False)
        if 60 <= m.sum() < len(raw):
            print(f"Keeping {m.sum()}/{len(raw)} patients with adenocarcinoma histology (use --keep-all-histologies to disable)")
            out = out[m.values]

    # ---- missing data ------------------------------------------------------------------------------
    n0 = len(out)
    out = out[out.duration.notna() & out.event.notna() & (out.duration > 0)]
    print(f"{len(out)}/{n0} patients have usable survival time and status")
    drop = [c for c in out.columns if c not in ("duration", "event") and out[c].isna().mean() > a.max_missing]
    for c in drop:
        print(f"  dropping '{c}' ({out[c].isna().mean():.0%} missing)")
    out = out.drop(columns=drop)
    n1 = len(out); out = out.dropna()
    if len(out) < n1:
        print(f"Dropped {n1 - len(out)} patients with a missing feature value")
    for c in list(out.columns):
        if c not in ("duration", "event") and out[c].nunique() < 2:
            print(f"  dropping constant column '{c}'"); out = out.drop(columns=c)
    out["duration"] = out.duration.astype(int); out["event"] = out.event.astype(int)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nWrote {a.out}: {len(out)} patients, {int(out.event.sum())} deaths, {int((1 - out.event).sum())} censored, "
          f"median follow-up {out.duration.median():.0f} days, features: {[c for c in out.columns if c not in ('duration','event')]}")
    if len(out) < 60:
        print("NOTE: small cohort - expect wide confidence intervals.")


if __name__ == "__main__":
    main()
