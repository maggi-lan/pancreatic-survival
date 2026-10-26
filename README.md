# Pancreatic Cancer Prognosis from Clinical + Radiomic Data

Machine Learning mini-project: Reimplementation of _"Pancreatic cancer prognosis using clinical and
radiomic data"_ (A. Jamalian, CS229 Final Report, 2020).

**Team:** Malavika Arun (PES1UG24CS257), Mahilan (PES1UG24CS256)

> **Data notice.** The paper's Stanford dataset (80 patients, MRI radiomics) is private (NDA), so by default this
> repo runs on a **synthetic cohort** with the same structure (80 patients, 12 clinical + 900 correlated radiomic
> features, survival days + censoring). The numbers it produces demonstrate that the pipeline works; they are
> **not clinical findings** and are not comparable to the paper's 0.67 / 0.76 / 0.82. To use real data, see
> "Using your own dataset".

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python tests/run_tests.py                              # 9 unit tests (~5 s)
python main.py                                         # full pipeline, ~2-3 min, writes ./outputs
python main.py --repeats 1                             # quick run (~1 min)
python main.py --data data/my_cohort.csv               # your own data
```

Python 3.9+. No `lifelines` / `imbalanced-learn` needed: Cox PH, Kaplan-Meier, log-rank, C-index, Naive Bayes
and SMOTE are implemented from scratch in `survpanc/` (see below) and unit-tested.

## Pipeline (mirrors the paper)

| Step                                                                                                                                                                     | Paper             | Code                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ------------------------- |
| 1. Pick the high/low-risk survival threshold (240-360 d) with a Naive Bayes classifier on discretised clinical features (Laplace smoothing), scored by balanced accuracy | Sec. 4-5, Table 1 | `naive_bayes.py`          |
| 2. SMOTE (k=5) to oversample the high-risk group (features **and** survival days)                                                                                        | Sec. 5            | `smote.py`                |
| 3. Cox PH with L2 penalty 0.1 on clinical (12), radiomic (pruned) and combined features; 10-fold CV, Harrell's C-index                                                   | Table 2           | `cox.py`, `pipeline.py`   |
| 4. Radiomic pruning: fit on all 900, drop the wide-error-band features (paper dropped 410/900)                                                                           | Sec. 5            | `pipeline.prune_radiomic` |
| 5. Median-hazard split -> Kaplan-Meier curves + log-rank test                                                                                                            | Fig. 4            | `main.py`                 |
| 6. Top-15 coefficient forest plot; survival curves varying chemotherapy / stage                                                                                          | Fig. 5-6          | `plots.py`                |

### Deliberate differences from the paper

1. **SMOTE inside each training fold.** The paper augments the whole dataset _before_ cross-validation, so
   synthetic test points are interpolations of training patients (leakage). We run three variants: no SMOTE,
   SMOTE in-fold (correct, headline), and SMOTE-before-CV (paper's way) to quantify the inflation.
2. **Risk groups from out-of-fold scores.** In-sample median splits make the log-rank test look overwhelming
   (p = 1e-22 here). We report both; the out-of-fold p-value is the honest one.
3. **Pruning by fraction by default** (`--prune-frac 0.456`) rather than "CI width > 1". With n ~ 100 and
   penalizer 0.1, all CI widths fall in a narrow band (about 1.27-1.36 here), so a fixed cut-off removes everything
   or nothing. A fixed cut-off is available via `--max-ci-width 1.0`.
4. Censored patients who left follow-up before the threshold have unknown risk class, so they are excluded from
   the Naive Bayes / SMOTE labelling (they stay in the Cox fits).
5. Only L2 regularisation is implemented (the paper found L1 did not converge and chose `l1_ratio=0`).

## Results on the synthetic cohort (3 x 10-fold CV, seed 0)

| Feature set                | No SMOTE | SMOTE in-fold (correct) | SMOTE before CV (paper-style) |
| -------------------------- | -------- | ----------------------- | ----------------------------- |
| Clinical (12)              | 0.51     | 0.51                    | 0.55                          |
| Radiomic (490 kept of 900) | 0.56     | 0.57                    | 0.73                          |
| Clinical + radiomic (502)  | 0.60     | 0.58                    | 0.74                          |

How to read it:

- With only 80 patients and 900 features, honest out-of-sample C-index is modest (fold std ~ 0.15). Even the
  _true_ clinical linear predictor used to generate the data only reaches C = 0.63 on this cohort, so the weak
  clinical result is a property of small, noisy data, not a bug.
- Combined >= radiomic > clinical in the honest columns, but the differences are within one fold-std, so we do
  not claim they are significant.
- The paper-style protocol inflates radiomic/combined C-index by about 0.15, which supports caution about the
  paper's 0.82.
- Naive Bayes threshold selection is weak here (balanced accuracy 0.48-0.58; best 300 days), as expected
  from five coarse clinical features.

Outputs in `outputs/`: `results.json`, `nb_threshold_sweep.csv`, `top_coefficients.csv`, `run_log.txt`, and figures
`fig_cohort*.png`, `fig_threshold.png`, `fig_cindex.png`, `fig_km_groups.png`, `fig_forest.png`,
`fig_partial_effects.png`.

## Running on a cBioPortal clinical TSV (TCGA pancreatic cohort)

```bash
cp ~/Downloads/<file>.tsv data/raw/clinical.tsv
python prepare_cbioportal.py --input data/raw/clinical.tsv --out data/pancreas.csv
python main.py --data data/pancreas.csv --out outputs_real
```

`prepare_cbioportal.py` finds columns by their cBioPortal names (Overall Survival (Months) / Status, Diagnosis Age, Sex,
AJCC tumor stage code, Mutation Count, Fraction Genome Altered, adjuvant/radiation therapy ...), converts months to
days, keeps adenocarcinoma only, drops feature columns with more than 20% missing values (`--max-missing`), then drops
patients with remaining gaps. It prints exactly what it used and dropped - copy that into the write-up.
`main.py` refuses to run on fewer than 30 patients. `tests/fixtures/*` are randomly generated files used only to test
the converters; they are not patient data.

## Running on the TCGA-PAAD clinical CSV

The public file we use has columns `Patient_ID, Gender, Age_at_Diagn*, Histological_T*, Tumor_Stage,
Survival_Statu*, Overall_Surviv*, KRAS_Mutatio*, TP53_Mutatio*, SMAD4_Status` (clinical + mutation data, no imaging).

```bash
mkdir -p data/raw && cp <your file> data/raw/pancreas_raw.csv
python prepare_data.py --input data/raw/pancreas_raw.csv --out data/pancreas.csv --time-unit months
python main.py --data data/pancreas.csv
```

- `prepare_data.py` matches columns by prefix, turns DECEASED/LIVING into the event flag, converts survival
  **months to days** (x30.44; check the unit of your column and pass `--time-unit days` if it is already days),
  encodes stage as an ordinal (IA=0 ... IV=5) and mutation columns as 0/1 flags.
- With no `rad_*` columns the pipeline runs **clinical-only** (the radiomic/combined models and pruning are
  skipped). Say this in the write-up: there is no imaging data, so the paper's radiomic comparison cannot be
  reproduced on this file. Add gene-expression columns named `rad_*` if you want a high-dimensional block.
- SMOTE only adds samples when the high-risk class (died before the threshold) is the _minority_. If it is not on
  your data, 0 synthetic patients are added and the three SMOTE modes give identical results. Try other cut-offs
  with e.g. `--thresholds 180 270 365 540`.
- `tests/fixtures/mock_raw_format_check.csv` is a randomly generated file in this format, used only to test the
  script. It is not real patient data.

## Using your own dataset

CSV with one row per patient and no missing values:

- `duration` - days from diagnosis to death or last follow-up
- `event` - 1 = death, 0 = censored
- clinical columns (numeric/ordinal-encoded; names in `survpanc/data.py::CLINICAL_COLS`; the Naive Bayes step uses
  `stage_grouping, chemo_after_sbrt, tumor_width, resectability_nccn, radiation_conventional` if present)
- radiomic columns whose names start with `rad_`

If your data has no radiomic columns the radiomic/combined models are not meaningful; run the clinical model only
and say so in the write-up (public sources to look at: TCGA-PAAD clinical data, TCIA).

## Repository layout

```
prepare_cbioportal.py  cBioPortal clinical TSV -> data/pancreas.csv
prepare_data.py        small TCGA-PAAD clinical CSV -> data/pancreas.csv
main.py                 entry point (args: --data --out --repeats --folds --penalizer --prune-frac ...)
survpanc/data.py        synthetic cohort + CSV loader
survpanc/naive_bayes.py discretiser, Naive Bayes (Laplace), threshold sweep
survpanc/smote.py       SMOTE for survival data
survpanc/cox.py         CoxPH (Newton, Breslow, ridge), Kaplan-Meier, log-rank, C-index
survpanc/pipeline.py    pruning, CV with SMOTE modes
survpanc/plots.py       figures
tests/                  unit tests (Cox gradient vs finite differences, coefficient recovery, KM by hand, ...)
outputs/                results from the default run
```

## Reference

A. Jamalian, "CS229 Project Final Report: Pancreatic cancer prognosis using clinical and radiomic data", 2020.
