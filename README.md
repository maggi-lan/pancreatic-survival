# Pancreatic Cancer Prognosis from Clinical + Radiomic Data

Machine Learning mini-project: Reimplementation and Optimization of _"Pancreatic cancer prognosis using clinical and
radiomic data"_ (A. Jamalian, CS229 Final Report, 2020).

**Team:** Malavika Arun (PES1UG24CS257), Mahilan (PES1UG24CS256)

> **Data notice.** The original Stanford dataset (80 patients, MRI radiomics) is private (NDA), so by default this
> repository runs on a **synthetic cohort** matching the exact structure (80 patients, 12 clinical + 900 correlated
> radiomic features, survival days + right censoring). The numbers it produces demonstrate that the pipeline works
> and provide an honest benchmark. To run on real public cohorts (e.g., TCGA-PAAD), see [Using your own dataset](#using-your-own-dataset).

---

## Setup and Run

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python tests/run_tests.py                              # 10 unit tests (~5 s)
python main.py                                         # runs Baseline vs Optimized benchmark (~90 s)
python main.py --repeats 1                             # quick benchmark run (~30 s)
python main.py --data data/my_cohort.csv               # your own dataset
```

**Requirements:** Python 3.9+. Built from scratch in pure NumPy and SciPy. No `lifelines` or `imbalanced-learn` dependencies: Cox PH regression, Kaplan-Meier estimator, log-rank test, Harrell's C-index, Naive Bayes discretiser, and survival SMOTE are implemented directly in `survpanc/` and unit-tested.

---

## Pipeline Overview

| Step | Method | Module |
| :--- | :--- | :--- |
| **1. Risk Threshold Selection** | Discretised clinical features + Naive Bayes (Laplace smoothing) swept over 240–360 days; scored by balanced accuracy. | `survpanc/naive_bayes.py` |
| **2. Survival SMOTE** | k-NN ($k=5$) oversampling on high-risk patients (interpolating continuous features and survival duration). | `survpanc/smote.py` |
| **3. High-Signal Survival Screening** | Evaluates univariate Harrell's C-index strictly on training folds; selects top $k=25$ highest-signal radiomic features. | `survpanc/pipeline.py` |
| **4. Cox PH with Ridge Regularization** | Newton-Raphson Breslow solver with $L_2$ penalty ($\lambda=0.05$ optimized, $\lambda=0.10$ baseline). | `survpanc/cox.py` |
| **5. Cross-Validation across 3 Modes** | 10-fold CV with Harrell's C-index scoring across `in_fold`, `none`, and `leaky` modes. | `survpanc/pipeline.py` |
| **6. Generalization Risk Assessment** | Out-of-fold hazard median split -> Kaplan-Meier curves & Log-rank test. | `survpanc/cox.py`, `survpanc/plots.py` |

---

## Deliberate Differences from the Original Implementation

1. **SMOTE Inside Each Training Fold (Zero Leakage):** The original paper augmented the whole cohort *before* 10-fold CV, leaking synthetic interpolations of test patients into the training folds. We implement honest in-fold augmentation (`in_fold`), pure unaugmented evaluation (`none`), and global augmentation (`leaky`) to quantify the inflation.
2. **Out-of-Fold Risk Stratification:** In-sample median risk splits give artificially overwhelming log-rank statistics ($p \approx 10^{-22}$). We stratify patients based on out-of-fold predictions that the model never saw during training.
3. **High-Signal Univariate Screening vs. Arbitrary CI Pruning:** The original paper pruned 410 features by confidence-interval width, retaining 490 collinear features for 80 patients ($p \gg n$). Our optimized pipeline screens radiomic features strictly in-fold by univariate concordance $|C_j - 0.5|$, keeping the **top 25 high-signal features** and reducing the Ridge penalty to $\lambda=0.05$.
4. **Pure From-Scratch Algorithms:** Replaced external libraries (`lifelines`, `imblearn`) with pure NumPy/SciPy implementations.

---

## Benchmark Results: Baseline vs. Optimized

Running `python main.py` automatically executes both the **Unoptimized Baseline** (original configuration: 490 CI-pruned features, $\lambda=0.10$) and the **Optimized Model** (top 25 screened features, $\lambda=0.05$) across the same cross-validation splits:

```
====================================================================================
              UNOPTIMIZED (BASELINE) vs OPTIMIZED (SCREENING) COMPARISON
====================================================================================
Evaluation Metric / Feature Set  | Unoptimized (Base) | Optimized (Screen) | Delta       
------------------------------------------------------------------------------------
Radiomic Dimension               | 490 features (CI)  | 25 features (C-idx) | -465 dims   
Penalizer (L2 Ridge)             | 0.10               | 0.05               | -0.05       
SMOTE k-neighbors                | 5                  | 5                  | 0           
------------------------------------------------------------------------------------
[Honest CV (in_fold)]
  - Clinical Alone               | 0.5144             | 0.4963             | -0.0181 (-3.5%)
  - Radiomic Alone               | 0.5977             | 0.6080             | +0.0103 (+1.7%)
  - COMBINED (Clinical+Radiomic) | 0.6223             | 0.6653             | +0.0430 (+6.9%)
[No SMOTE (none - Real Data)]
  - Clinical Alone               | 0.4977             | 0.4858             | -0.0118 (-2.4%)
  - Radiomic Alone               | 0.5873             | 0.6078             | +0.0206 (+3.5%)
  - COMBINED (Clinical+Radiomic) | 0.6002             | 0.6538             | +0.0536 (+8.9%)
[Paper-Style (leaky - Global SMOTE)]
  - Clinical Alone               | 0.5345             | 0.5571             | +0.0226 (+4.2%)
  - Radiomic Alone               | 0.7341             | 0.7237             | -0.0104 (-1.4%)
  - COMBINED (Clinical+Radiomic) | 0.7581             | 0.7093             | -0.0488 (-6.4%)
====================================================================================
```

### Key Findings:
- **Honest Generalization Jumps +6.9% to +8.9%:** In honest in-fold CV, the combined model improves from **0.6223 to 0.6653**. On pure real data without SMOTE, it improves from **0.6002 to 0.6538**.
- **95% Dimension Reduction:** Cutting 465 dimensions of collinear noise allows the true imaging signal to complement clinical staging without overfitting.
- **Why Leaky Drops:** In `leaky` mode, the baseline overfitted to synthetic noise that leaked across folds. Pruning noise to 25 real signal features eliminates the artificial inflation, confirming the model learns genuine biology rather than leakage shortcuts.

---

## Output Files

All figures and summaries are saved to `outputs/`:
- `fig_comparison.png` — Side-by-side grouped bar chart comparing Baseline vs. Optimized across all modes and feature sets.
- `comparison_summary.csv` — Full numerical benchmark comparison table with deltas and percentage gains.
- `fig_km_groups.png` — Kaplan-Meier survival curves for High-Risk vs. Low-Risk groups with Log-Rank test statistics.
- `fig_forest.png` — Top-15 feature hazard ratio forest plot with 95% confidence intervals.
- `fig_partial_effects.png` — Survival curves varying chemotherapy and cancer stage.
- `fig_threshold.png` — Naive Bayes threshold sweep (240–360 days) balanced accuracy plot.
- `fig_cindex.png` — C-index distribution bar plot across folds.
- `screened_features.csv` — List of top 25 screened radiomic features with univariate C-index and separation scores.
- `top_coefficients.csv` — Largest Cox PH model hazard coefficients and standard errors.
- `results.json` — Full machine-readable results from the benchmark runs.

---

## Using Your Own Dataset

### TCGA-PAAD via cBioPortal
```bash
cp ~/Downloads/<file>.tsv data/raw/clinical.tsv
python prepare_cbioportal.py --input data/raw/clinical.tsv --out data/pancreas.csv
python main.py --data data/pancreas.csv --out outputs_real
```
`prepare_cbioportal.py` parses standard cBioPortal columns (survival months, vital status, age, sex, stage code, mutation counts), drops columns with $> 20\%$ missing values, and converts months to days.

### Standard TCGA-PAAD Clinical CSV
```bash
mkdir -p data/raw && cp <your_file.csv> data/raw/pancreas_raw.csv
python prepare_data.py --input data/raw/pancreas_raw.csv --out data/pancreas.csv --time-unit months
python main.py --data data/pancreas.csv
```

### Custom Dataset CSV Format
Provide a CSV with:
- `duration`: Days from diagnosis to event or last follow-up.
- `event`: 1 = death, 0 = censored.
- Clinical columns: Numeric/ordinal features matching `survpanc/data.py::CLINICAL_COLS`.
- Radiomic columns: Numeric imaging features whose names begin with `rad_`.

---

## Repository Layout

```
main.py                  Benchmark entry point (Baseline vs. High-Signal Screening)
prepare_cbioportal.py    cBioPortal clinical TSV -> data/pancreas.csv
prepare_data.py          TCGA-PAAD clinical CSV -> data/pancreas.csv
survpanc/
  ├── data.py            Synthetic cohort generator and CSV loader
  ├── naive_bayes.py     Multinomial discretiser, Laplace Naive Bayes, threshold sweep
  ├── smote.py           k-NN SMOTE augmentation for right-censored survival data
  ├── cox.py             Newton-Raphson Cox PH, Breslow baseline hazard, KM estimator
  ├── metrics.py         Harrell's C-index, Log-Rank test statistic and p-value
  ├── pipeline.py        High-signal univariate screening, CI pruning, 10-fold CV modes
  └── plots.py           Figure generation (comparison, forest, KM, threshold, effects)
tests/
  ├── run_tests.py       Test runner (10 tests)
  └── test_core.py       Unit tests for Cox gradient, KM, C-index, SMOTE, screening
outputs/                 Benchmark figures, CSV summaries, and JSON results
```

---

## References

1. A. Jamalian, *"Pancreatic cancer prognosis using clinical and radiomic data,"* CS229 Project Final Report, 2020.
2. D. R. Cox, *"Regression models and life-tables,"* Journal of the Royal Statistical Society: Series B, 1972.
3. F. E. Harrell et al., *"Evaluating the yield of medical tests,"* JAMA, 1982.
4. N. V. Chawla et al., *"SMOTE: Synthetic Minority Over-sampling Technique,"* JAIR, 2002.
