# Model Improvement Report

**Date:** 2026-08-25 (UTC)
**Scope:** Experimental investigation of whether the production LightGBM model can be improved.
**Production model status:** **UNCHANGED.** `05_MODEL_TRAINING/models/final_model.joblib` was opened read-only throughout and its modification time and size were asserted identical after every experiment.
**Outcome:** **Candidate found — NOT promoted to production.**

---

## 1. Baseline

All baseline figures are **read from the existing artefact**
`06_MODEL_EVALUATION/evaluation_report.json`. None were re-derived or entered by hand.

| Metric | Held-out test (patient-grouped) | Temporal holdout |
|---|--:|--:|
| n | 311,967 | 199,767 |
| Positive rate | 0.19551 | 0.15032 |
| Threshold | 0.50 | 0.50 |
| **PR-AUC** | **0.76166** | **0.68597** |
| ROC-AUC | 0.90189 | 0.88594 |
| **Sensitivity / recall** | **0.54543** | **0.40234** |
| Specificity | 0.96856 | 0.98666 |
| Precision (PPV) | 0.80830 | 0.84213 |
| **F1** | **0.65134** | **0.54453** |
| Accuracy | 0.88583 | 0.89882 |
| Brier | 0.085373 | 0.077675 |

The report also records `pr_auc_ci95 = [0.75819, 0.76450]` for the test partition —
a **95% CI half-width of ≈0.0032**. This is used throughout as the yardstick for
whether a PR-AUC difference is meaningful: a change smaller than ~0.003 is
inside the noise of the baseline's own estimate.

Model: LightGBM, 30 features, 1,111,803 training rows from 46,905 patients,
isotonic calibration fitted on the validation partition.

### Harness validation (EXP-00)

Before any experiment ran, the harness re-fitted the production configuration
from scratch and reproduced the published numbers **exactly**:

| | Published | Reproduced | Δ |
|---|--:|--:|--:|
| Test PR-AUC | 0.76166 | 0.76166 | **0.00000** |
| Test accuracy | 0.88583 | 0.88583 | 0.00000 |
| Test recall | 0.54543 | 0.54543 | 0.00000 |
| Temporal PR-AUC | 0.68597 | 0.68597 | **0.00000** |
| Temporal recall | 0.40234 | 0.40234 | 0.00000 |

Partitions matched exactly (train 1,111,803 / val 154,426 / test 311,967).
Every comparison below is therefore like-for-like.

---

## 2. Research Question

Can the production model be improved on **PR-AUC, sensitivity and F1** —
without sacrificing specificity unacceptably — and **does any such improvement
survive the temporal holdout**, where the model is fitted on past cultures and
tested on future ones?

Accuracy is explicitly *not* the objective. At a 19.55% positive rate, a
predict-everything-susceptible model would score 80.4% accuracy while being
clinically useless.

---

## 3. Experimental Method

### Protocol

* Partitions come from the **same functions and seed** as production
  (`patient_grouped_split`, `temporal_split`, seed 42), so no comparison is
  confounded by a different split.
* **Selection never touches the test partition.** Thresholds are selected on a
  patient-disjoint half of validation; class weights and feature sets on the
  validation partition; hyperparameters by GroupKFold CV on a patient-level
  subsample of the training partition.
* The test partition and temporal holdout are evaluated **once**, on an
  already-selected candidate.
* The categorical encoder is fitted on the training fold only.
* Temporal evaluation replicates the production method exactly: refit on the
  temporal training partition, evaluate **uncalibrated** on the temporal test
  partition (this is what `scripts/08_evaluate_models.py` does, so the baseline
  temporal figures are comparable).

### A note on threshold selection and calibration

The production calibrator is fitted on the *whole* validation partition. Choosing
a threshold on those same rows would be optimistic, because the calibrator has
already seen them. The validation partition is therefore split **by patient**
into two disjoint halves: the isotonic calibrator is fitted on half A, and the
threshold is swept and chosen on half B.

---

## 4. Threshold Optimization (EXP-01)

### Methodological point first

**PR-AUC and ROC-AUC are threshold-independent.** Across all eleven thresholds
the measured PR-AUC on the selection half was **constant at 0.76864** (one
unique value across 11 rows). No claim is made anywhere in this report that
changing the threshold improves PR-AUC — it cannot, and the experiment records
that it does not.

### Sweep on the held-out validation half

| Threshold | Accuracy | Precision | Recall | Specificity | F1 |
|--:|--:|--:|--:|--:|--:|
| 0.20 | 0.83090 | 0.54309 | 0.79820 | 0.83875 | 0.64638 |
| 0.25 | 0.85215 | 0.59247 | 0.75731 | 0.87492 | 0.66482 |
| 0.30 | 0.87252 | 0.65985 | 0.70505 | 0.91273 | 0.68170 |
| **0.35** | **0.88211** | **0.71035** | **0.66044** | **0.93534** | **0.68449** ← max F1 |
| 0.40 | 0.88404 | 0.72630 | 0.64365 | 0.94176 | 0.68248 |
| 0.45 | 0.88742 | 0.76082 | 0.61048 | 0.95392 | 0.67741 |
| 0.50 | 0.88942 | 0.81705 | 0.55267 | 0.97029 | 0.65934 | ← production |
| 0.55 | 0.88893 | 0.83911 | 0.52748 | 0.97572 | 0.64777 |
| 0.60 | 0.88889 | 0.83921 | 0.52715 | 0.97575 | 0.64754 |
| 0.65 | 0.88745 | 0.87299 | 0.48998 | 0.98288 | 0.62767 |
| 0.70 | 0.88455 | 0.89289 | 0.45877 | 0.98678 | 0.60612 |

Two operating points were selected **on validation only**:
* **t = 0.35** — maximises F1.
* **t = 0.30** — maximises recall subject to specificity ≥ 0.90.

### Applied once to the untouched partitions

| Threshold | Test acc | Test recall | Test spec | Test F1 | Temporal recall | Temporal spec | Temporal F1 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.50 (production) | 0.88583 | 0.54543 | 0.96856 | 0.65134 | 0.40234 | 0.98666 | 0.54453 |
| **0.35 (candidate)** | 0.87787 | **0.65426** | 0.93222 | **0.67688** | **0.53518** | 0.95359 | **0.59546** |
| 0.30 (alternative) | 0.86932 | **0.69333** | 0.91209 | 0.67476 | **0.56725** | 0.94198 | 0.59862 |

At t = 0.35: recall **+0.10883** on test and **+0.13284** on temporal; F1
**+0.02554** and **+0.05093**. Cost: specificity −0.03634 and −0.03307,
accuracy −0.00796 and −0.01526. **The gain is larger on the temporal holdout
than on the random test split**, which is the opposite of the pattern an
overfitted result would show.

---

## 5. Class Weight Experiments (EXP-02)

Weights 0.75 → 3.00 applied as `class_weight={0:1, 1:w}`, each refitted on the
full training partition, selected on validation PR-AUC.

| Weight | Val PR-AUC | Val ROC-AUC | Val recall | Val specificity | Val F1 | Val Brier |
|--:|--:|--:|--:|--:|--:|--:|
| 0.75 | **0.77198** | 0.90292 | 0.49881 | 0.98047 | 0.63174 | 0.08578 |
| 1.00 (production) | 0.77136 | 0.90298 | 0.53139 | 0.97399 | 0.64868 | **0.08460** |
| 1.25 | 0.77097 | 0.90287 | 0.56524 | 0.96574 | 0.66258 | 0.08509 |
| 1.50 | 0.77114 | 0.90299 | 0.61603 | 0.94907 | 0.67490 | 0.08656 |
| 2.00 | 0.76990 | 0.90305 | 0.66090 | 0.93019 | 0.67852 | 0.09139 |
| 2.50 | 0.76959 | 0.90299 | 0.69402 | 0.91455 | 0.67857 | 0.09744 |
| 3.00 | 0.76852 | 0.90285 | 0.73323 | 0.88716 | 0.66735 | 0.10415 |

**PR-AUC is flat.** The full range across a fourfold change in weight is 0.00346
— comparable to the baseline's own CI half-width of 0.0032. ROC-AUC varies by
0.0002. What actually changes is the recall/specificity trade-off, sliding
recall from 0.499 to 0.733 — **the same trade-off the threshold produces**.

### The validation winner did not transfer

Weight 0.75 had the best validation PR-AUC, so it was carried to the untouched
partitions:

| | Test PR-AUC | Test recall | Test spec | Temporal PR-AUC | Temporal recall |
|---|--:|--:|--:|--:|--:|
| w = 1.00 (production) | **0.76166** | **0.54543** | 0.96856 | **0.68597** | **0.40234** |
| w = 0.75 (val winner) | 0.76132 | 0.53353 | 0.97170 | 0.68595 | 0.37997 |

The validation advantage of +0.00062 reversed to **−0.00034 on test**, and
temporal recall fell by 0.022. This is noise, correctly identified as such by
the protocol. **Decision: REJECT.**

Class weighting is also strictly worse than thresholding as a mechanism for the
same effect: it requires refitting, and it degrades calibration monotonically
(Brier 0.0846 → 0.1042 from w=1 to w=3), whereas a threshold change leaves the
probabilities untouched.

---

## 6. Hyperparameter Optimization (EXP-03)

24-iteration `RandomizedSearchCV`, GroupKFold(3) by patient, scored by average
precision, on a 249,995-row / 10,401-patient subsample of the training partition.
Search space covered learning rate, n_estimators, num_leaves, max_depth,
min_child_samples, subsample, colsample_bytree, reg_alpha and reg_lambda.
Search time 519 s.

| | CV PR-AUC |
|---|--:|
| Searched winner | 0.76255 |
| Production parameters, identical CV protocol | 0.76118 ± **0.00522** |

The apparent gain of +0.00137 is **less than one third of the CV standard
deviation**. Carried to the untouched partitions anyway:

| | Val PR-AUC | Test PR-AUC | Test recall | Temporal PR-AUC | Temporal recall |
|---|--:|--:|--:|--:|--:|
| Production | **0.77136** | **0.76166** | **0.54543** | **0.68597** | **0.40234** |
| Searched | 0.76923 | 0.75889 | 0.53889 | 0.68361 | 0.40005 |

The searched configuration was **worse on validation, worse on test and worse on
temporal**. **Decision: REJECT.** The production hyperparameters — themselves
the output of a 12-iteration search in `scripts/07_train_models.py` — are at or
near a local optimum that a further modest search does not beat.

---

## 7. Feature Engineering (EXP-04) and Feature Selection (EXP-05)

### Finding that reframed the experiment

Every feature the brief proposes as a candidate **already exists** in the
analytic dataset. `scripts/04_build_analytic_dataset.py` engineered **109
prior-history columns**; feature selection kept **16** of them in the final 30
and discarded **93**.

| Brief's proposed feature | Existing column(s) | In production 30? |
|---|---|:--:|
| Number of previous antibiotic exposures | `prior_med_n_drugdays_30d/180d` | yes |
| Number of previous resistance events | `prior_res_n_events_1095d` | yes |
| Number of previous cultures | `prior_n_cultures` | no (discarded) |
| Time since previous antibiotic exposure | `prior_med_days_since_last` | no (discarded) |
| Time since previous resistant event | `prior_res_days_since_last` | no (discarded) |
| Same-organism previous resistance | `prior_res_same_org_abx_n` | yes |
| Same-antibiotic previous resistance | `prior_res_same_abx_n` | yes |
| Organism-specific historical resistance rate | `prior_same_org_nonsusc_rate` | yes |
| Antibiotic-specific historical resistance rate | `prior_same_abx_nonsusc_rate` | yes |

The honest experiment is therefore not "invent new features" but **"does adding
back the discarded history help?"**

### Leakage verification

Every history column is built by the same shifted-cumulative construction —
`cumsum() - own_value` and `order_time.shift(1)` — verified at
`scripts/04_build_analytic_dataset.py:515-583`, with same-day (`t == 0`)
medication rows excluded at line 321. Each row therefore sees only events
**strictly before its own culture**. No feature was created in this experiment,
so no new leakage surface was introduced.

An independent single-feature AUC screen was run on the training partition as a
cross-check — a discarded column able to predict its own label almost perfectly
would betray a shift error:

| Feature | Single-feature AUC | % missing |
|---|--:|--:|
| `prior_nonsusc_rate` | 0.68342 | 53.85 |
| `prior_res_n_events_365d` | 0.59131 | 0.00 |
| `prior_res_n_events_total` | 0.59040 | 0.00 |
| `prior_res_same_org_n` | 0.58337 | 0.00 |
| `prior_res_same_abx_n_365d` | 0.58298 | 0.00 |
| `prior_n_cultures` | 0.56926 | 0.00 |
| … (14 screened) | … | … |
| `prior_res_same_abx_days_since` | 0.29508 | 90.62 |

**Maximum 0.683; zero features above the 0.90 leakage-suspicion line.** The
history construction is sound.

### Sets compared (validation only)

| Set | n features | Val PR-AUC | Val ROC-AUC | Val recall | Val F1 |
|---|--:|--:|--:|--:|--:|
| **A — production 30** | 30 | **0.77136** | 0.90298 | 0.53139 | 0.64868 |
| B — A + 14 brief-named history features | 44 | 0.76954 | 0.90210 | 0.53037 | 0.64681 |
| C — A + all 93 discarded history features | 123 | 0.76985 | 0.90245 | 0.53242 | 0.64886 |

**The production 30-feature set won.** Neither B nor C reached test or temporal
evaluation, because the protocol forbids promoting a set that lost on
validation. Set C is the upper bound on what the entire engineered history
space can offer, and it is *below* the 30-feature set — quadrupling the feature
count adds nothing but variance. **Decision: REJECT; the existing 30-feature set
is retained unchanged.**

---

## 8. Temporal Validation (EXP-06)

The decisive test. Every candidate that survived selection was evaluated on the
temporal holdout (fit on the earliest 80% of orders, test on the most recent,
with straddling patients dropped).

| Experiment | Temporal PR-AUC | Δ vs baseline | Temporal recall | Δ | Temporal F1 | Δ |
|---|--:|--:|--:|--:|--:|--:|
| Baseline (t=0.50) | 0.68597 | — | 0.40234 | — | 0.54453 | — |
| **Threshold 0.35** | 0.68597 | **0.00000** | **0.53518** | **+0.13284** | **0.59546** | **+0.05093** |
| Threshold 0.30 | 0.68597 | 0.00000 | 0.56725 | +0.16491 | 0.59862 | +0.05409 |
| Class weight 0.75 | 0.68595 | −0.00002 | 0.37997 | −0.02237 | — | — |
| Hyperparameter winner | 0.68361 | −0.00236 | 0.40005 | −0.00229 | — | — |

Only the threshold change improves temporal performance, and it does so
**without touching temporal PR-AUC** — because it does not change the model.
Both rejected experiments were neutral-to-worse temporally, consistent with
their test-set behaviour.

Note the temporal recall gain (+0.133) exceeds the test recall gain (+0.109).
The temporal partition has a lower positive rate (0.150 vs 0.196) and the model
is more conservative on it, so lowering the threshold recovers proportionally
more of the missed positives.

---

## 9. Calibration (EXP-07, EXP-07b)

Brier score conflates calibration with discrimination, so Expected and Maximum
Calibration Error were computed separately (20 quantile bins, test partition).

| Variant | ECE | MCE | Brier | PR-AUC | ROC-AUC | Distinct probabilities |
|---|--:|--:|--:|--:|--:|--:|
| Uncalibrated | 0.006403 | 0.026280 | 0.085431 | 0.76701 | 0.90198 | 142,215 |
| **Isotonic (production)** | **0.002875** | **0.010856** | **0.085373** | 0.76166 | 0.90189 | **301** |
| Platt / sigmoid | 0.032936 | 0.076984 | 0.086993 | 0.76701 | 0.90198 | 108,292 |

### Reading this correctly

A naive reading of PR-AUC alone would say isotonic calibration *costs* 0.00535
PR-AUC and should be removed. The decomposition shows that conclusion is wrong:

* Isotonic **more than halves calibration error** — ECE 0.006403 → 0.002875
  (−55%), MCE 0.026280 → 0.010856 (−59%). It is doing exactly its job.
* Its PR-AUC cost is a **tie artefact, not a loss of information**. Isotonic
  regression is a step function: it collapses 142,215 distinct probabilities to
  **301** (ratio 0.0021). PR-AUC penalises ties. ROC-AUC — which handles ties
  differently — barely moves (0.90198 → 0.90189, −0.00009), confirming the
  underlying ranking information is intact.
* Platt scaling preserves ranking exactly (identical PR-AUC, monotone strictly
  increasing) but is **badly calibrated** — ECE 0.032936, five times worse than
  uncalibrated and eleven times worse than isotonic.

**Conclusion: the production choice of isotonic calibration is correct and is
retained.** No change recommended.

### One actionable consequence

The 301-level collapse has a visible effect in the application, which ranks 54
antibiotics by calibrated probability: distinct agents receive **identical
probabilities and therefore tie in the ranking**. This was observed
independently during user acceptance testing — in UAT Test Case C, Ertapenem,
Imipenem and Meropenem all showed P = 0.9904, and Cefoxitin, Ceftazidime and
Fosfomycin all showed 0.0806. This is not a defect in the calibrator; it is an
expected property of isotonic regression surfacing in the product.

**Recommendation (application-level, not model-level):** break ranking ties in
`08_RECOMMENDATION_ENGINE/antibiotic_ranking.py` using the *uncalibrated* score
as a secondary sort key. This preserves the calibrated probability that is
displayed to the clinician while restoring a deterministic, information-bearing
order. Not implemented here — it is an application change, outside the scope of
this model experiment.

---

## 10. Results

Full log: `experiment_log.csv` (9 rows).
Comparison table: `14_REPORT_MATERIAL/tables/model_improvement_comparison.csv`.

| Model / Experiment | Accuracy | PR-AUC | Recall | Specificity | F1 | Temporal PR-AUC | Temporal Recall |
|---|--:|--:|--:|--:|--:|--:|--:|
| **BASELINE — production (t=0.50)** | 0.88583 | 0.76166 | 0.54543 | 0.96856 | 0.65134 | 0.68597 | 0.40234 |
| **Threshold 0.35 (candidate)** | 0.87787 | 0.76166 | **0.65426** | 0.93222 | **0.67688** | 0.68597 | **0.53518** |
| Threshold 0.30 (alternative) | 0.86932 | 0.76166 | **0.69333** | 0.91209 | 0.67476 | 0.68597 | **0.56725** |
| Class weight 0.75 | 0.88603 | 0.76132 | 0.53353 | 0.97170 | 0.64671 | 0.68595 | 0.37997 |
| Hyperparameter search winner | 0.88561 | 0.75889 | 0.53889 | 0.96987 | 0.64815 | 0.68361 | 0.40005 |
| Calibration: uncalibrated | 0.88590 | 0.76701 | 0.52646 | 0.97325 | 0.64339 | 0.68597 | 0.40234 |
| Calibration: Platt / sigmoid | 0.88591 | 0.76701 | 0.54330 | 0.96917 | 0.65060 | 0.68597 | 0.40234 |

Feature sets B (44) and C (123) are recorded in the experiment log with
validation results only; both lost on validation and were correctly never
evaluated on test or temporal.

### Figures

`14_REPORT_MATERIAL/figures/model_improvement/`

1. `01_threshold_vs_recall.png`
2. `02_threshold_vs_specificity.png`
3. `03_threshold_vs_f1.png`
4. `04_precision_recall_curves.png`
5. `05_baseline_vs_candidate.png`
6. `06_temporal_comparison.png`
7. `07_calibration_curve.png`

---

## 11. Best Candidate

**The production LightGBM model, operated at a decision threshold of 0.35
instead of 0.50.**

Saved as `05_MODEL_TRAINING/models/candidate_improved_model.joblib` with
`metadata.candidate = True` and `metadata.promoted_to_production = False`.

The fitted model, encoder, feature list and isotonic calibrator in the candidate
bundle are **the same objects as production** — the bundle was loaded from
`final_model.joblib` and only `threshold` was changed. The production file's
modification time and size were asserted unchanged after saving.

**Candidate model found — NOT promoted to production.**

---

## 12. Why It Is Better

Against the six-point selection rule:

1. **No leakage.** Nothing was retrained and no feature was added. The threshold
   is applied to an already-computed probability.
2. **PR-AUC improved or meaningfully maintained.** Exactly maintained at
   0.76166 — it is the same model, and PR-AUC is threshold-independent.
3. **No unacceptable loss of specificity.** Test specificity 0.96856 → 0.93222
   (−0.036); temporal 0.98666 → 0.95359 (−0.033). Specificity remains above
   0.93 on both.
4. **Temporal performance improved.** Recall +0.13284, F1 +0.05093, PR-AUC
   unchanged. The temporal gain *exceeds* the test gain.
5. **Reproducible.** Deterministic given the model and threshold; no stochastic
   fitting is involved.
6. **Methodologically defensible.** Selected by max F1 on a patient-disjoint
   half of the validation partition, with the calibrator fitted on the other
   half. The test partition and temporal holdout were each touched once.

The clinical argument is the substantive one. The task is detecting
non-susceptibility, and the production operating point misses **45.5%** of truly
non-susceptible results. At t = 0.35 that falls to **34.6%** on test and from
59.8% to **46.5%** on temporal data. In this setting a false negative — calling
a resistant isolate susceptible — is the expensive error, and the baseline
threshold is tuned against the wrong one.

---

## 13. Trade-offs

The candidate is **not a free improvement**:

| | Test | Temporal |
|---|--:|--:|
| Recall | **+0.10883** | **+0.13284** |
| F1 | **+0.02554** | **+0.05093** |
| Specificity | −0.03634 | −0.03307 |
| Accuracy | −0.00796 | −0.01526 |
| Precision (PPV) | 0.80830 → 0.70111 | — |
| PR-AUC | 0.00000 | 0.00000 |

In absolute terms on the test partition, moving from 0.50 to 0.35 converts
about **6,600 false negatives into true positives** while creating roughly
**9,100 additional false positives**. Precision falls from 0.808 to 0.701 — of
agents flagged non-susceptible, roughly 30% would not be, against 19% today.

Whether that is the right trade is a **clinical judgement, not a statistical
one**. In this application the consequence of a false positive is that a
clinician sees an agent ranked lower with a resistance flag they can overrule;
the consequence of a false negative is an agent presented as a reasonable option
when the laboratory would call it resistant. Those are not symmetric, which is
the argument for the lower threshold — but the decision belongs to the project
owner and a clinical reviewer.

---

## 14. Whether It Should Replace the Existing Model

**It should not be promoted automatically, and it has not been.**

Three reasons:

1. **It is not a better model.** It is the same model at a different operating
   point. Nothing about the fitted estimator improved — three independent
   experiments (class weight, hyperparameters, features) all failed to move
   discrimination. Promoting it would be a **deployment policy change**, and
   should be described as one.
2. **The trade-off is clinical.** The statistics establish what the trade costs;
   they do not establish that it is worth paying. That requires someone
   accountable for the clinical consequence.
3. **It changes what the application displays.** The threshold drives the
   Susceptible / Non-susceptible label on every one of the 54 ranked agents, the
   Dashboard's "Predicted susceptible" count, and the ground-truth agreement
   figures. All published UAT and evaluation numbers are quoted at t = 0.50 and
   would need regenerating.

If it is approved, the promotion path is: set `threshold = 0.35` on the
production bundle, re-run `scripts/08_evaluate_models.py`, re-run the UAT, and
regenerate the documentation.

**A defensible middle option:** keep t = 0.50 as the label threshold and expose
the 0.35 operating point in the application as a selectable "high-sensitivity"
view. The ranking itself is threshold-independent, so this changes only the
labels, and it lets a clinician see both.

---

## 15. Limitations

1. **Only the operating point moved.** No experiment improved discrimination.
   PR-AUC across every arm sat within 0.7589–0.7670 on test — a spread of
   0.0081, barely more than the baseline's own CI width. The model is at a
   plateau for this feature set and this data.
2. **The threshold was selected by max F1**, which weights precision and recall
   equally. That is a statistical convention, not a clinical one. A
   cost-weighted objective (the evaluation report already tabulates
   `cost_fn_3x_fp` and `cost_fn_5x_fp`) would justify a different point —
   probably lower.
3. **The hyperparameter search was deliberately modest** — 24 iterations on a
   250,000-row subsample. A larger search might find something, though the
   consistent failure of three separate approaches makes a large gain unlikely.
4. **Feature engineering was not attempted from raw data.** 109 history features
   already existed and adding all 93 discarded ones made validation performance
   *worse*, so effort was better spent verifying that than inventing more. A
   genuinely new signal would likely need a new data source, not a new
   transformation of this one.
5. **Single institution, no external validation.** Unchanged from the baseline.
6. **The temporal holdout is one split** at the 80th percentile of order time.
   A rolling-origin evaluation would estimate temporal degradation more robustly.
7. **No statistical test of the recall difference.** The improvement is large
   relative to the metric's scale and consistent across two independent
   partitions, but no formal significance test was run.

---

## Final Decision

### **A — Genuine improvement**, scoped precisely to the operating point.

The threshold change satisfies all six criteria of the selection rule: it
introduces no leakage, maintains PR-AUC exactly, keeps specificity above 0.93,
**improves temporal performance more than test performance**, is deterministic,
and was selected without touching the test set.

**With an equally important qualification: for the model itself the result is C
— no meaningful improvement.** Class weighting, hyperparameter search and
feature expansion each produced a selection-set "winner" that failed to transfer
to untouched data. That consistency across three independent methods is itself
the finding: the production LightGBM is at a genuine performance plateau, and
its existing configuration is well chosen.

The calibration investigation returned the same verdict — the production choice
of isotonic is correct, and the apparent PR-AUC "cost" is a tie artefact rather
than a loss of information.

**Production model status: UNCHANGED. Candidate saved separately, not promoted,
awaiting explicit approval.**
