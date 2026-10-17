# Changelog

All notable changes to this project. Dates are UTC.

The pipeline's generated documents (`docs/*.md`, `README.md`,
`PROJECT_MEMORY.md`, `MODEL_CARD.md`, `EDA_REPORT.md`) are regenerated from
artefacts rather than edited, so they are not listed individually here — only
the changes that caused them to change are.

---

## [2026-08-24] — User acceptance testing of the application

The completed application was acceptance-tested end to end. **No model was
retrained, no feature or hyperparameter was changed, and no measured
performance number was altered.** The regression suite passed identically
before and after (111 tests). Result: **PASS** — 57/59 tests passed, 2 failed
and were fixed, 0 outstanding.

### Fixed

**BUG-01 — the wrong drug's FDA label could be displayed as the dosage
reference (high, patient-safety relevant).**
`10_API_SERVICES/drug_api.py` fell back to an unfielded openFDA full-text
search when the fielded `openfda.generic_name` query returned nothing, and
accepted whatever came back without checking it was the requested drug. Because
the candidate ranking prefers an oral product that has a dosage section, an
unrelated label won outright. Six of the 54 candidate agents were affected:
**Doripenem, a carbapenem, displayed the dosing instructions for LETERMOVIR, an
antiviral**; Ticarcillin displayed Ertapenem's; and four combination agents
displayed SILICEA, a homeopathic pellet product. Doripenem and Ticarcillin
contain no separator character and so were affected independently of BUG-02 —
Doripenem was rank 7 in the application's default case, so the path was reached
in ordinary use.

Fixed by `_identity_tokens()`, `_label_identity()` and `_label_matches_drug()`,
applied in `get_drug_label()` so candidate labels are filtered to those actually
naming the requested ingredient before ranking. When nothing matches, the
function returns `ok=False` and the application shows "Insufficient information
for dosage reference." — the behaviour the project already requires for
unavailable data. Wrong-drug labels: 6 → 0, with no correct label lost.

**BUG-02 — combination antibiotics had neither dosage nor cost data (medium).**
`_NAME_OVERRIDES` was written with hyphenated keys (`piperacillin-tazobactam`)
while the deployed model's vocabulary spells combinations with a slash
(`Piperacillin/Tazobactam`), so ten agents missed the lookup and were queried
under a name openFDA does not index. Because `price_api.get_price()` reuses
`normalise_drug_name()`, NADAC cost lookup failed identically. Affected agents
included Piperacillin/Tazobactam and Trimethoprim/Sulfamethoxazole, two of the
most frequently used agents in the cohort.

Fixed by making the lookup separator-agnostic — `/`, `-` and " and " all fold to
a single space, with exact matches still taking precedence — plus four mappings
the table never covered, each verified against the live openFDA endpoint
(`cephalexin/cephalothin`, `imipenem/ebactam`, `meropenem/vaborbactam`, and the
cohort's own misspelling `ticacarcillin/clavulanic acid`).

Combined effect: genuinely correct dosage references **39/54 (72 %) → 47/54
(87 %)**; cost coverage **27/54 (50 %) → 33/54 (61 %)**.

### Added

- `12_TESTS/USER_ACCEPTANCE_TEST_REPORT.md` — the full 19-section UAT report.
- `12_TESTS/uat_report.json` — machine-readable UAT artefact, so the UAT numbers
  in `README.md` and `PROJECT_MEMORY.md` are generated rather than transcribed.
- `14_REPORT_MATERIAL/USER_TESTING_SUMMARY.md` — concise summary for the
  academic report.
- `14_REPORT_MATERIAL/tables/user_testing_summary.csv` — test-by-test table.
- `scripts/12_generate_project_docs.py` — a "User acceptance testing" section in
  both README and PROJECT_MEMORY, a `USER ACCEPTANCE TESTING` row in the status
  block, and automatic carry-through of deferred UAT bugs into "Known problems".

### Known, deferred

- **BUG-03 (low)** — `_confidence_band()` derives its band from `abs(p − 0.5)`
  alone, so an agent with zero training examples can read "confidence: high"
  beside an evidence band of "none — extrapolated". Both facts are displayed, so
  nothing is hidden, but the labels conflict. Deferred: changing it alters
  clinical presentation on every option card and should be a deliberate design
  decision rather than a change made during a test cycle.
- **BUG-04 (low)** — `use_container_width` is used in 20 places and is
  deprecated for removal. Still functional in the pinned Streamlit 1.62 and the
  warning never reaches the UI. Deferred to a standalone change before any
  Streamlit upgrade.

### Verified, unchanged

Guardrails flag but never remove (confirmed in all seven test cases); the
genomic module honestly reports that no genomic data exists; offline mode with
the response cache also neutralised produced 0 exceptions, all 9 tabs and an
intact prediction; the clinician review form refuses to record a decision
without an explicit confirmation and a reviewer identifier; and no prescriptive
wording exists anywhere in the application.

---

## [2026-08-24] — Report-readiness: EDA, verification, diagrams, consistency audit

No model was retrained and no feature was added or removed. This phase made the
existing project correct, reproducible, verified and report-ready.

### Added

**Exploratory data analysis (Phase 4 — previously NOT STARTED)**
- `scripts/04b_exploratory_analysis.py` — complete 15-part EDA producing 14
  figures and 21 tables. Enforces an explicit scope rule: structural
  description uses the full analytic dataset, anything touching the TARGET uses
  the training partition only.
- `scripts/14_generate_eda_report.py` → `02_DATA_ANALYSIS/EDA_REPORT.md`
  (19 sections, every number read from the EDA artefact).
- `02_DATA_ANALYSIS/notebooks/03_exploratory_data_analysis.ipynb`, plus seven
  more notebooks covering audit, cleaning/leakage, feature engineering, feature
  selection, training, evaluation and XAI. All eight execute end to end.

**Verification**
- `scripts/17_verify_system.py` — 59 executed checks across explainable AI,
  recommendation-engine edge cases, external services, the genomic module and
  the application. Writes `docs/SYSTEM_VERIFICATION.md`.
  **59/59 pass.**
- `scripts/15_consistency_check.py` — project-wide audit that loads the
  authoritative artefacts and scans every document for numeric contradictions,
  stale status lines, unsafe language, fabricated API hosts, hard-coded
  credentials, hard-coded paths and broken links. Exits non-zero on any error
  so it can run in CI. **Currently reports no inconsistencies.**
- `12_TESTS/test_application.py` — executes the Streamlit app and both extra
  pages with `AppTest` and asserts on what they render. Suite is now
  **111 tests, all passing**.

**Report material**
- `scripts/16_generate_diagrams.py` — seven report-ready diagrams, every
  annotation read from a pipeline artefact: `methodology_workflow.png`,
  `system_architecture.png`, `feature_engineering_workflow.png`,
  `feature_selection_methodology.png`, `recommendation_workflow.png`,
  `application_workflow.png`, `human_in_the_loop_workflow.png`.
- `scripts/18_assemble_report_material.py` — reorganises `14_REPORT_MATERIAL/`
  into `diagrams/`, `figures/{DATASET,EDA,FEATURE_SELECTION,MODEL,XAI}/` and
  `tables/{DATASET,EDA,FEATURE_SELECTION,MODEL}/`, and writes `INDEX.md`
  listing all 93 artefacts with their description and producing script.

### Fixed

- **The manual-entry sidebar collected history the model never used.**
  `components/case.py` enumerated plausible-sounding history fields
  (`prior_n_cultures`, `prior_med_n_drugdays_90d`, …) that are not among the 30
  selected features, so clinician-entered history was silently discarded and
  the "assumed fields" notice could never fire. The history inputs are now
  derived from `bundle.features` at run time, and an explicit
  "history unknown" toggle distinguishes *confirmed no prior exposure* (a real,
  informative zero) from *not known* (defaulted, and reported as assumed).
- **`MODEL_CARD.md` was missing the medical safety disclaimer.** It now opens
  with it, and the consistency checker asserts its presence.
- **Gram-stain intrinsic-resistance rules were keyed on an enumerated genus
  list**, so a rare non-fermenter fell through: the ranking offered vancomycin
  and linezolid as the top two options for *Achromobacter xylosoxidans*, a
  gram-negative. Rules are now keyed on `organism_group`, which every organism
  has, with a second subclass-level layer so benzylpenicillin is flagged for
  gram-negatives while ampicillin correctly is not. Regression tests cover
  organisms that appear in no genus rule.
- **The ground-truth panel showed duplicate antibiotic rows** for polymicrobial
  cultures, because a culture order was treated as one isolate. The application
  now exposes an isolate selector and filters the AST panel to the chosen
  organism.
- **The EDA outlier check reported 706,738 impossible haemoglobin values.** The
  `*_hgb` columns are exported in **mg/dL, not g/dL**; the bound was wrong, not
  the data. Corrected, and recorded as a caveat: units in this export are not
  uniform across analytes.
- **matplotlib 3.11 renamed `boxplot(labels=)` to `tick_labels=`.**

### Dataset findings (no data was modified)

- **Duplicated COLUMNS in the source laboratory export.** 6 `first_*` /
  `last_*` columns carry byte-identical values under different analyte names
  (`last_hgb` == `last_plt` == `last_lymphocytes` == `last_neutrophils`). Four
  distinct analytes cannot agree to the digit on every row. All are 96-99%
  missing and were already removed by the feature-selection missingness filter,
  so no corrupted value ever reached the model — but the defect is now on
  record, and any future use of `first_*`/`last_*` laboratory features must
  account for it.
- **The organism × antibiotic space is 78% empty** — only 21.6% of possible
  combinations have any test at all, and 4.1% have 50 or more. This is
  panel-selection bias made visible and is the strongest single justification
  for the unified-model architecture.
- **The antibiogram independently validates the cleaning pipeline**: textbook
  intrinsic resistance appears at or near 1.00 exactly where microbiology says
  it should (Klebsiella/ampicillin, Proteus/nitrofurantoin and tetracycline,
  AmpC organisms/cefazolin, S. aureus/penicillin).
- **Prior resistance to the same organism-antibiotic pair raises
  non-susceptibility from 16.4% to 68.8%** (4.20× relative risk), measured
  univariately before any model was fitted.

### Documentation

- `PROJECT_MEMORY.md` now records **EDA: COMPLETE**, carries an EDA findings
  section, and reports **GENOMIC MODULE: COMPLETE / INACTIVE**.
- `04_FEATURE_SELECTION/FEATURE_SELECTION_REPORT.md` gained Section 13,
  explaining the final 30-feature set and — as required — why no laboratory
  value and no age feature survived selection.
- `scripts/run_all.py` now covers all 20 stages including EDA, verification,
  diagrams, report assembly and the consistency audit.

---

## [2026-08-23] — Initial build: full pipeline, application and documentation

### Added

**Data audit and documentation (Phases 1–2)**
- `scripts/01_dataset_audit.py` — streaming column-level profile of all five raw
  files (1.7 GB, 16.4 M rows) without loading any of them fully into memory.
  Emits `02_DATA_ANALYSIS/reports/dataset_audit.json` and an MD5 integrity
  manifest.
- `scripts/00_verify_raw_integrity.py` — verifies the raw files are byte-identical
  to the audit snapshot.
- `scripts/02_relationship_and_leakage_analysis.py` — measures table grain,
  join-key uniqueness and relationship cardinality without executing a merge;
  runs a dedicated leakage investigation.
- `scripts/03_generate_data_docs.py` — generates `docs/DATA_DICTIONARY.md` and
  the entity-relationship diagram from the audit JSON.
- `scripts/05_generate_analysis_docs.py` — generates `DATA_RELATIONSHIPS.md`,
  `DATA_LEAKAGE_ANALYSIS.md` and `ML_PROBLEM_DEFINITION.md`.

**Preprocessing and feature engineering (Phases 3 & 5)**
- `src/preprocessing/organism_normalizer.py` — normalises 309 raw organism
  strings to 245 clean species names; extracts colony morphotype as a feature and
  **strips embedded resistance phenotype tokens** as leakage.
- `src/preprocessing/antibiotic_taxonomy.py` — class / subclass / WHO AWaRe tier
  for all 54 agents observed in the cohort (100 % coverage).
- `src/preprocessing/medication_mapping.py` — maps all 89 raw medication names
  (brand names, salts, diluents) to ingredient and antibiotic class.
- `scripts/04_build_analytic_dataset.py` — cleaning, temporal-safe history
  feature engineering and validated assembly into a 1,578,196 × 194 analytic
  table. Every merge is routed through `validated_merge()`, which refuses to run
  if the join would fan out or lose rows.

**Feature selection (Phase 6)**
- `scripts/06_feature_selection.py` — seven methods, eight candidate feature
  sets, a feature-count sweep, an organism-blind ablation, and a nested-vs-global
  cross-validation check that quantifies selection optimism.
- `scripts/10_generate_feature_docs.py` — generates the feature-selection report
  and `docs/FINAL_FEATURE_SET.md`.

**Modelling (Phases 7–8)**
- `src/models/dataset.py` — the feature/target/split contract, patient-grouped
  and temporal splitters, patient-level subsampling.
- `src/models/preprocessing.py` — separate tree and linear preprocessing
  pipelines, so the model families are compared fairly.
- `src/models/bundle.py` — `ModelBundle`: model, calibrator, encoder, feature
  list, training defaults, organism × antibiotic support counts and measured
  metrics in one serialised artefact.
- `scripts/07_train_models.py` — six model families, a four-arm class-imbalance
  experiment, `RandomizedSearchCV` with `StratifiedGroupKFold`, refit on the full
  training partition, isotonic calibration on validation.
- `scripts/08_evaluate_models.py` — the only script that reads the test
  partition. Headline metrics with bootstrap CIs, threshold analysis under three
  cost ratios, per-antibiotic / per-organism / subgroup breakdowns, temporal
  holdout, secondary 3-class model, and a unified-vs-per-agent architecture
  comparison.

**Explainability (Phase 9)**
- `src/explainability/shap_explainer.py` — one SHAP implementation shared by the
  feature-selection stage, the XAI report and the live application.
- `scripts/09_explainability.py` — global importance, beeswarm, dependence
  plots, five local waterfall explanations (including the model's failures), and
  a mechanism check that history features move predictions in the clinically
  expected direction.

**Decision support (Phases 10–12)**
- `08_RECOMMENDATION_ENGINE/recommendation_rules.py` — intrinsic resistance
  (EUCAST/CLSI), site suitability (FDA labelling, PK/PD), stewardship tier (WHO
  AWaRe 2023), evidence-strength and allergy guardrails. Flags, never filters.
- `08_RECOMMENDATION_ENGINE/antibiotic_ranking.py` — scores every candidate
  agent, attaches SHAP factors and guardrails, ranks for clinician review.
- `08_RECOMMENDATION_ENGINE/dosage_reference.py` — relays FDA label dosage text
  verbatim; never computes a dose.
- `08_RECOMMENDATION_ENGINE/cost_service.py` — relays live CMS NADAC unit
  prices; never estimates or extrapolates.
- `09_GENOMIC_AMR/genomic_analysis.py` — 21-marker reference map and a plug-in
  point for a future genotype export. Inactive on this dataset, and says so.
- `10_API_SERVICES/` — verified clients for openFDA drug label, RxNorm/RxNav and
  CMS NADAC, all behind a single `ApiResult` envelope with provenance,
  disk caching, timeouts and a global offline switch.

**Application (Phase 13)**
- `11_APPLICATION/app.py` — Streamlit decision-support interface: nine tabs over
  one isolate, with cases loadable from the held-out test partition so the
  prototype demonstrates on genuine patients.
- `pages/1_Model_Card.py`, `pages/2_Data_and_Methodology.py`.

**Testing and documentation (Phases 14–15)**
- `12_TESTS/` — 90 tests covering raw-data immutability, dataset grain, split
  integrity, temporal safety of every history feature, preprocessing correctness,
  guardrail behaviour and the safety framing of the recommendation layer.
- `scripts/11_generate_model_docs.py`, `scripts/12_generate_project_docs.py` —
  generate `MODEL_EVALUATION.md`, `XAI_REPORT.md`, `MODEL_CARD.md`, `README.md`
  and `PROJECT_MEMORY.md` from measured artefacts.
- `scripts/run_all.py` — ordered pipeline runner with stage selection.

### Dataset changes
- Raw data is **referenced in place, never copied or modified**. Only a manifest
  (size, row count, MD5) is stored in `01_DATA/raw/`.
- 634,468 rows dropped as negative cultures (no susceptibility result).
- 4,602 rows dropped as `Inconclusive` / `Synergism` — not single-agent
  phenotypes.
- 23,784 rows dropped resolving 22,708 duplicated (order, organism, antibiotic)
  keys, of which 11,785 carried conflicting results. Rule: most-resistant-wins.
- 1,034,527 same-day (`t == 0`) prior-medication rows excluded as temporally
  unsafe.
- 3,961,688 exact-duplicate prior-medication rows collapsed to distinct
  drug-days.
- Labs collapsed from 1,408,677 rows to one row per culture order by median of
  non-missing values.

### Model changes
- Architecture chosen: **one unified model with the antibiotic as a feature**,
  after comparing it against per-antibiotic and multi-label alternatives.
- Target defined as binary Susceptible vs Non-susceptible (Resistant +
  Intermediate); a 3-class model is trained and reported as a secondary
  analysis.
- Class imbalance resolved empirically: **no resampling**. SMOTE measurably
  harmed performance (PR-AUC 0.677 vs 0.765 for doing nothing) because
  interpolating between ordinal-coded organism and antibiotic codes synthesises
  entities that do not exist.
- Final model: **LightGBM**, 30 features, isotonic-calibrated on validation.

### Fixed
- **openFDA returned the wrong label for several agents.** A generic-name search
  for "ciprofloxacin" returned the *ophthalmic* solution first, whose dosage
  section describes drops for corneal ulcers and never mentions renal
  adjustment. `drug_api.get_drug_label()` now retrieves up to 25 candidate labels
  and re-ranks them so a systemic (oral / IV / IM) product with an actual dosage
  section is selected.
- **`numpy.shuffle` on an Arrow-backed string array** warned and could duplicate
  entries in `subsample_by_patient()`. Patient ids are now materialised to a
  plain object array first.
- **The prior-culture history counted only positive cultures**, because it was
  computed after the negative cultures were dropped. It now runs over all
  751,075 culture orders, so a patient with ten prior negative cultures is
  distinguishable from one with none.
- **The selection-leakage check compared the wrong things.** The first version
  put `SelectKBest` inside a CV pipeline, which measures a *different selector*
  rather than the optimism of the one actually used. Replaced with a
  like-for-like nested-vs-global comparison: the real selector is re-derived
  inside each fold and scored against the globally chosen list. Measured
  optimism: +0.001 PR-AUC.

### Notes on things deliberately NOT done
- **No genomic predictions.** The datasets contain no genomic data; the module is
  built and inactive rather than fabricated.
- **No dose calculation.** Even with complete patient data the project would
  relay the label and leave the calculation to the prescriber. Creatinine
  clearance is not computed because the dataset has no weight.
- **No course-cost calculation.** It would require a dose, which the project does
  not generate.
- **RxNav Drug Interaction API not integrated** — verified retired, returns
  "Not found" for valid RxCUIs. Showing nothing is better than showing data from
  an unmaintained source.
- **RFECV not used** — with ~180 candidates and 1.1 M training rows it would
  refit XGBoost several hundred times for an answer the feature-count sweep
  already provides.
- **Chi-square not used** for feature selection — it requires non-negative
  discrete counts, which ordinal-coded categoricals and standardised labs are
  not.
- **Kernel SVM not used** — roughly O(n²) training cost is intractable at 1.1 M
  rows. The margin family is represented by a linear SVM trained with stochastic
  gradient descent.
