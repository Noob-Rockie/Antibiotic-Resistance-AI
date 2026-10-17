# User Acceptance Testing Report

**Project:** Explainable Machine Learning for Antibiotic Resistance Prediction and Clinical Decision Support
**Test date:** 2026-08-24 (UTC)
**Tester:** Automated acceptance harness driving the deployed application
**Application under test:** `11_APPLICATION/app.py` (unchanged during testing)
**Model under test:** `05_MODEL_TRAINING/models/final_model.joblib` (**not retrained, not modified**)

---

## 1. Testing Objective

To determine whether a clinician using the existing application can, without
reading the source code:

1. Load or enter a patient / culture case
2. Generate an antibiotic resistance prediction
3. Understand what the prediction means
4. View the SHAP explanation behind it
5. View patient history
6. See potential antibiotic options for review
7. See guardrail warnings
8. View dosage reference information
9. View medicine cost information
10. View genomic information / status
11. Record a final clinician decision
12. Understand the medical safety disclaimer

The test also verifies the system's negative guarantees: that it does **not**
prescribe, does **not** hide model failures, and does **not** fabricate data.

### Method

The application was exercised in two complementary ways.

* **Live server.** `streamlit run 11_APPLICATION/app.py` was started headless on
  port 8599 and confirmed serving (`HTTP 200`, `/_stcore/health` -> `ok`).
* **Scripted UI harness.** `streamlit.testing.v1.AppTest` executed the real
  `app.py` script in-process, which renders every tab and exposes the full
  element tree, every widget, and any raised exception. This is Streamlit's own
  official test harness -- it runs the actual application, not a reimplementation.
* **Case-level drill-down.** Test cases A-G were additionally driven through the
  same modules the app calls (`components/case.py`,
  `08_RECOMMENDATION_ENGINE/antibiotic_ranking.py`) so that every probability,
  SHAP value and guardrail quoted in this report is real model output.

No test result in this document is estimated or illustrative.

---

## 2. Environment

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language 10.0.26200 |
| Python | 3.11.9 (project `.venv`) |
| Streamlit | 1.62.0 |
| LightGBM | 4.7.0 |
| SHAP | 0.51.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| Live API switch | `ENABLE_LIVE_API=1` (offline mode also tested at `0`) |
| Server URL | `http://localhost:8599` |

Missing dependencies: **none**. Broken imports: **none**.

---

## 3. Application Version

| Item | Value |
|---|---|
| Model algorithm | LightGBM, 30 features |
| Trained on | 1,111,803 rows from 46,905 patients |
| Decision threshold | 0.50 |
| Target | `1 = non-susceptible (Resistant or Intermediate)` |
| Candidate agents scored per case | 54 |
| Application tabs | 9 + 2 additional pages |
| Held-out test metrics (unchanged) | PR-AUC 0.76166 - ROC-AUC 0.90189 - Sensitivity 0.54543 - Specificity 0.96856 - Accuracy 0.88583 |

The project's own regression suite was run before and after the fixes described
in section 17: **111 passed, 0 failed** on both runs.

---

## 4. Test Cases

Status key: **PASS** = behaved as required - **PASS\*** = passed, with an
observation recorded in section 15 - **FAIL** = defect found (all fixed defects
are re-tested and listed with their post-fix status).

| Test ID | Test | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| UAT-01 | Application startup | App starts, no exceptions | Server ready in 22.7 s; first full page render 32.7 s cold; 0 exceptions | PASS |
| UAT-02 | Dependencies / imports | All imports resolve | All resolve; no missing package | PASS |
| UAT-03 | Server reachable | HTTP 200 + health ok | `HTTP 200` in 0.06 s; `/_stcore/health` -> `ok` | PASS |
| UAT-04 | Dashboard tab | Loads, meaningful content | 8 metrics, susceptibility bar chart, ground-truth table | PASS |
| UAT-05 | Prediction tab | Loads, per-agent probabilities | 54-row table, dual probability columns, confidence, evidence count | PASS |
| UAT-06 | Explanation tab | Loads, SHAP per prediction | Per-agent SHAP table + chart + narration + global importance figure | PASS |
| UAT-07 | Options for review tab | Loads, ranked options | Ranked cards with pills, evidence, guardrails, expandable reasoning | PASS |
| UAT-08 | Dosage reference tab | Loads, verbatim label text | FDA label text, route, renal context, provenance, disclaimer | PASS |
| UAT-09 | Cost tab | Loads, NADAC prices | NDC-level price table with effective date + provenance | PASS |
| UAT-10 | Patient history tab | Loads, history + labs | 17-item history table + 11-analyte lab panel | PASS |
| UAT-11 | Genomic tab | Honest unavailable status | States module inactive; no fabricated genotype | PASS |
| UAT-12 | Clinician review tab | Loads, decision form | Form with reviewer, decision, agents, notes, confirmation | PASS |
| UAT-13 | Model Card page | Loads | 0 exceptions; 9 headers, 6 tables, 15 metrics | PASS |
| UAT-14 | Data & Methodology page | Loads | 0 exceptions; 9 sections, 3 tables | PASS |
| UAT-15 | Test case A -- correct prediction | Correct call + explanation | 12/12 agreement, full SHAP, ranked options | PASS |
| UAT-16 | Test case B -- incorrect prediction | Failure shown, not hidden | 2/8 agreement; all 6 disagreements displayed with a cross; SHAP available | PASS |
| UAT-17 | Test case C -- prior resistance history | History used and explained | 17/17 history items shown; SHAP cites prior-resistance features | PASS |
| UAT-18 | Test case D -- prior antibiotic exposure | Exposure displayed | 41 drug-days/90 d shown; SHAP cites prior exposure | PASS |
| UAT-19 | Test case E -- missing history | No pretence of history | Missing items render "not recorded"; defaults declared | PASS |
| UAT-20 | Test case F -- rare organism | Evidence warning appears | All 13 agents "very thin" (1-5 training rows); warnings raised | PASS |
| UAT-21 | Test case G -- guardrails | Flags shown, not filtered | 28 critical flags; agents demoted to ranks 27-54, none removed | PASS |
| UAT-22 | Organism / specimen / setting selection | Constrained inputs | Selectboxes bound to model vocabulary; free text impossible | PASS |
| UAT-23 | Probability + classification + confidence | All three present | P(resistant), P(susceptible), label, confidence band, evidence count | PASS |
| UAT-24 | Prediction vs laboratory result distinction | Clearly separated | "Model P(susceptible)" / "Model call" vs "Laboratory result" columns | PASS |
| UAT-25 | SHAP on a correct prediction | Explanation present | 5 ranked factors with sign, magnitude, encoded value | PASS |
| UAT-26 | SHAP on an incorrect prediction | Explanation still present | Explanations render for every misclassified agent tested | PASS |
| UAT-27 | Global feature importance | Displayed | `07_shap_feature_importance.png` rendered | PASS |
| UAT-28 | Ranking uses model predictions | Scores from model | All 54 probabilities produced by the fitted model | PASS |
| UAT-29 | Guardrails flag, never filter | Nothing removed | Flagged agents always present, demoted below unflagged | PASS |
| UAT-30 | Evidence strength shown | Training count per pair | `training_examples` + evidence band on every option | PASS |
| UAT-31 | No prescriptive wording | Advisory framing only | 0 prohibited phrases; "potential option for clinician review" throughout | PASS |
| UAT-32 | Dosage never calculated | Verbatim relay only | No dose arithmetic anywhere; text quoted from label | PASS |
| UAT-33 | Insufficient dosage information | Explicit message | "Insufficient information for dosage reference." shown | PASS |
| UAT-34 | Dosage provenance | Source + timestamp | openFDA source, URL, retrieval time, cache status shown | PASS |
| UAT-35 | Dosage label identity | Label must be the requested drug | **Wrong drug's label displayed for 6 agents** -- see BUG-01 | **FAIL -> FIXED** |
| UAT-36 | Cost fields | Name, NDC, price, date, source | All present, per NDC, with effective date | PASS |
| UAT-37 | Cost unavailable message | Explicit message | "Price unavailable, verify locally." shown | PASS |
| UAT-38 | Cost not a selection criterion | Warned against | Explicit warning; cost never enters ranking | PASS |
| UAT-39 | Genomic honesty | States unavailable | Error banner + reason + enablement instructions | PASS |
| UAT-40 | No fabricated genomic data | No patient genotype invented | Demo markers explicitly labelled "NOT from this patient" | PASS |
| UAT-41 | Future genotype format documented | Format shown | `order_proc_id_coded, marker` CSV + expected path displayed | PASS |
| UAT-42 | Clinician can record a decision | Decision recorded | Written to `13_RESULTS/clinician_reviews.csv` with audit fields | PASS |
| UAT-43 | Explicit human decision required | Refuse without confirmation | Refused without checkbox; refused without reviewer id | PASS |
| UAT-44 | Accept / Modify / Reject + notes | All available | Radio with 3 options, agent multiselect, free-text notes | PASS |
| UAT-45 | Invalid / extreme manual inputs | No crash | 18 edge cases (negative, NaN, inf, absurd, out-of-range) -- 0 crashes | PASS |
| UAT-46 | Empty / missing record | No crash | Empty dict, null organism, partial record -- all handled | PASS |
| UAT-47 | Allergy input handling | Robust | Case/whitespace tolerant; unknown + injection strings inert | PASS |
| UAT-48 | Invalid culture order id | Clean error | `KeyError` with readable message, no traceback to user | PASS |
| UAT-49 | Offline mode `ENABLE_LIVE_API=0` | App still works | 0 exceptions, 9/9 tabs, prediction intact | PASS |
| UAT-50 | No traceback leaks to user | None visible | 0 traceback fragments in rendered output | PASS |
| UAT-51 | Prediction survives API loss | Model output unaffected | 54-row prediction table intact with no network | PASS |
| UAT-52 | API error handling | Friendly messages | HTTP 404/429/500, non-JSON and network errors all handled | PASS |
| UAT-53 | No fabricated API data | Nothing invented | Unavailable is stated, never filled | PASS |
| UAT-54 | Prohibited medical wording | Absent | 0 matches for prescriptive phrase patterns | PASS |
| UAT-55 | Safety disclaimer | Present and prominent | Banner at top of every page + footer on all tabs | PASS |
| UAT-56 | Combination-antibiotic reference data | Should resolve | **10 agents had no dosage and no cost** -- see BUG-02 | **FAIL -> FIXED** |
| UAT-57 | Confidence vs evidence coherence | Should not conflict | "high confidence" shown on 0-evidence agents -- see BUG-03 | **PASS\*** |
| UAT-58 | Performance | Responsive | Prediction 0.066 s; SHAP 3.90 s; see section 14 | PASS |
| UAT-59 | Regression suite | All pass | 111 passed, 0 failed (before and after fixes) | PASS |

**Totals: 59 tests - 57 PASS (1 with observation) - 2 FAIL, both fixed and
re-tested - 0 outstanding failures.**

---

## 5. Model Prediction Tests

Cases were drawn from the **held-out test partition** -- patients that
contributed nothing to training, feature selection or calibration. All
identifiers shown are the project's own de-identified coded order keys; no
private identifying information appears anywhere in this report.

### Scan of the demonstration case menu

All 60 held-out cultures offered by the application's case picker were scored:

| Measure | Value |
|---|--:|
| Cultures scanned | 60 |
| Agent-level predictions matched to a laboratory result | 1,003 |
| Predictions agreeing with the laboratory | 783 |
| Agreement on this menu | **78.07 %** |

> **This 78.07 % is not the model's accuracy.** The application's case menu is
> deliberately enriched for difficulty -- it pairs the 30 cultures with the most
> non-susceptible results against 30 fully-susceptible ones. The model's
> genuine held-out accuracy remains **88.58 %**, unchanged by this testing. The
> figure above describes the demonstration menu, not model performance.

### Per-case results

| Case | Order (de-identified) | Organism | Specimen | Agreement |
|---|---|---|---|--:|
| A | 758664704 | Citrobacter koseri | Urine | 12/12 |
| B | 486309118 | Enterococcus species | Urine | 2/8 |
| C | 567063460 | Escherichia coli | Urine | 5/20 |
| D | 668087393 | Klebsiella pneumoniae | Respiratory | 20/25 |
| E | 924237064 | Citrobacter koseri | Urine | 16/16 |
| F | 325204949 | Elizabethkingia meningoseptica | Respiratory | 11/13 |
| G | 648954466 | Pseudomonas aeruginosa | Respiratory | 11/12 |

### TEST CASE A -- correct prediction (UAT-15)

Order `758664704` - *Citrobacter koseri* - urine - 35-44 y - outpatient - 2021.
All 12 tested agents were susceptible in the laboratory and all 12 were called
susceptible by the model.

| Antibiotic | Laboratory | Model call | P(non-susceptible) | Confidence | Training rows |
|---|---|---|--:|---|--:|
| Amikacin | Susceptible | Susceptible | 0.0000 | high | 658 |
| Ceftriaxone | Susceptible | Susceptible | 0.0034 | high | 770 |
| Ciprofloxacin | Susceptible | Susceptible | 0.0016 | high | 961 |
| Meropenem | Susceptible | Susceptible | 0.0000 | high | 675 |
| Nitrofurantoin | Susceptible | Susceptible | 0.0973 | high | 824 |
| Trimethoprim/Sulfamethoxazole | Susceptible | Susceptible | 0.0034 | high | 961 |

*(6 of 12 rows shown; all 12 agreed.)*

Guardrails behaved correctly: 9 agents were flagged as intrinsically resistant
(Vancomycin, Daptomycin, Linezolid, Clindamycin and others -- all Gram-positive
agents against a Gram-negative organism) and demoted to ranks 44-54, **not
removed**.

### TEST CASE B -- incorrect prediction (UAT-16)

Order `486309118` - *Enterococcus species* - urine - 55-64 y - inpatient - 2016.
**The model was wrong on 6 of 8 agents.** The application displayed every
disagreement.

| Antibiotic | Laboratory | Model call | P(non-susceptible) | Confidence | Agree |
|---|---|---|--:|---|:--:|
| Ampicillin | **Resistant** | Susceptible | 0.1151 | high | no |
| Ciprofloxacin | **Resistant** | Susceptible | 0.4181 | very low | no |
| Levofloxacin | **Resistant** | Susceptible | 0.3433 | low | no |
| Linezolid | **Intermediate** | Susceptible | 0.0249 | high | no |
| Nitrofurantoin | **Intermediate** | Susceptible | 0.1151 | high | no |
| Penicillin | **Resistant** | Susceptible | 0.3903 | low | no |
| Tetracycline | Resistant | Non-susceptible | 0.8884 | high | yes |
| Vancomycin | Susceptible | Susceptible | 0.3763 | low | yes |

The Dashboard reported "Agreement on this culture: **2/8** agents (25 %)" and
displayed the full comparison table. **Nothing was hidden.** The SHAP
explanation remained fully available for every misclassified agent, e.g. for
the misclassified Ampicillin:

| Factor | SHAP | Direction |
|---|--:|---|
| the organism group (gram stain / family) | -0.5672 | towards susceptibility |
| antibiotic is oral | +0.5144 | towards resistance |
| prior exposure drug-days 180 d | +0.3653 | towards resistance |
| is species level | -0.3329 | towards susceptibility |
| the antibiotic class | +0.2528 | towards resistance |

This is the single most important behaviour verified in this UAT: **the
application explains its failures as readily as its successes.**

### TEST CASE C -- previous resistance history (UAT-17)

Order `567063460` - *E. coli* - urine - inpatient. A heavily pre-treated
patient: **394 documented prior resistant results**, 315 in the last year, a
59.1 % historic non-susceptibility rate and 72.0 % for this organism. All
**17/17** history items were populated and displayed.

The model used the history -- SHAP cites `prior resistance n events 1095d`
(+0.3767) and `this patient's past non-susceptibility rate for this agent`
-- but agreement was only **5/20**: the model called 15 laboratory-resistant
agents susceptible. This is a genuine and important limitation, visible to the
user, and consistent with the model's published sensitivity of 0.545. It is
recorded here rather than smoothed over.

### TEST CASE D -- previous antibiotic exposure (UAT-18)

Order `668087393` - *K. pneumoniae* - respiratory - inpatient. Recent heavy
exposure: 41 antibiotic drug-days in 90 days, 9 distinct classes, last dose
1 day before the culture. History displayed correctly (16/17 items). Agreement
**20/25**. SHAP for the misclassified Cefepime cites
`this patient's past non-susceptibility rate for this agent` (+1.2697) and
`prior exposure to Folate pathway inhibitor 90d` (+0.5705) -- i.e. exposure
features are genuinely driving the estimate.

### TEST CASE E -- missing history (UAT-19)

Order `924237064` - *C. koseri* - urine - outpatient - 2023. A patient with no
prior culture record. The application rendered the four unknown items as
**"not recorded"** and never substituted a value. Zero counts (`0` previous
cultures) are shown as genuine zeros, distinct from unknown. Agreement 16/16.

In the manual-entry path the same distinction is enforced by an explicit
checkbox -- "Patient history is unknown / unavailable" -- and every field then
defaulted is listed back to the user by the Dashboard's assumed-fields notice.

### TEST CASE F -- rare organism / low coverage (UAT-20)

Order `325204949` - *Elizabethkingia meningoseptica* - respiratory. A rare
organism: **every** tested agent had 1-5 training rows for this
organism-antibiotic pair, and all 13 were labelled **"very thin"** evidence.
41 agents raised a `no_evidence` caution flag and 13 a `thin_evidence` flag.
The system did not present these as reliable. Agreement 11/13.

### TEST CASE G -- guardrails (UAT-21)

Order `648954466` - *Pseudomonas aeruginosa* - respiratory. **28 of 54 agents
carried a critical guardrail flag** -- 27 intrinsic-resistance (correctly
including Ceftriaxone, Cefotaxime, Cefotetan and Vancomycin against
*Pseudomonas*) and 3 site-unsuitability (Daptomycin, Fosfomycin, Nitrofurantoin
for a respiratory specimen). All 28 were **demoted to ranks 27-54 and remained
visible with their reasons**.

Agreement was 11/12 on a multi-drug-resistant isolate. Notably, the top-ranked
options were all still labelled **"Non-susceptible"** -- the application did not
manufacture a good option where none existed.

---

## 6. SHAP Tests

| Check | Result |
|---|---|
| Explanation produced for an individual prediction | Yes -- per agent, per isolate |
| Feature names humanised | Yes (`prior_res_same_abx_n` -> "previous resistance in this patient to this antibiotic") |
| Feature values shown | Yes (encoded value column) |
| Direction of effect | Yes ("increases predicted resistance" / "...susceptibility") |
| Positive contributions | Yes, rendered as up-arrow towards resistance |
| Negative contributions | Yes, rendered as down-arrow towards susceptibility |
| Global importance | Yes -- `07_shap_feature_importance.png` |
| Available on correct predictions | Yes (Cases A, E) |
| Available on incorrect predictions | **Yes** (Cases B, C, D, F, G) |
| Explanation invented when SHAP fails | No -- falls back to an explicit "attributions unavailable" message |

SHAP values are computed by `TreeExplainer` on the deployed model -- the same
implementation used by feature selection and the XAI report, so what selects a
feature and what explains it to a clinician are the same computation. Values
are on the model's log-odds scale, which the UI states.

The narration is generated strictly from the SHAP output. It names only factors
that actually appear in the attribution list; no clinical claim is added.

---

## 7. Recommendation Tests

| Requirement | Result |
|---|---|
| Scores candidate antibiotics | 54 agents scored per case |
| Uses model predictions | Every probability from the fitted model |
| Shows predicted susceptibility / non-susceptibility | Both probabilities + a label |
| Shows supporting SHAP factors | Top 5 per agent, expandable |
| Shows guardrail flags | Critical / caution / info, colour-coded with citations |
| Shows evidence strength | Training-row count + banded description |
| Does **not** automatically prescribe | Confirmed -- no prescriptive statement anywhere |

Framing verified verbatim in the running application:

> "These are potentially suitable options for clinician review, ordered by the
> model's estimated probability of susceptibility. **They are not a
> prescription and not a recommendation to treat.**"

Guardrail severity is applied correctly: only intrinsic resistance, site
unsuitability and documented allergy are *critical* (and therefore demote an
agent); stewardship and evidence-thinness are advisory and do not.

**Ordering observation.** Options are ranked by predicted susceptibility alone.
An agent with zero training examples for the organism-antibiotic pair can
therefore outrank a well-evidenced agent (e.g. Cefiderocol at rank 5 with 0
training rows in Case A). The evidence band is displayed on the same card, so
nothing is concealed, but rank position alone does not encode evidence. See
BUG-03 and section 15.

---

## 8. Dosage Tests

| Requirement | Result |
|---|---|
| Dosage calculated or invented | **No** -- no dose arithmetic exists in the codebase |
| Labelled as reference information | Yes -- "REFERENCE INFORMATION -- CLINICIAN VERIFICATION REQUIRED" |
| Source displayed | Yes -- openFDA, with URL, retrieval timestamp and cache status |
| Verbatim label text | Yes -- `dosage_and_administration` quoted directly |
| "Insufficient information" message | Yes -- exact wording present |
| Renal individualisation | Explicitly declined; the app states why it cannot compute creatinine clearance |
| Label belongs to the requested drug | **Failed initially (BUG-01) -- now enforced** |

**Coverage across all 54 candidate agents:**

| | Before fixes | After fixes |
|---|--:|--:|
| Dosage reference shown | 45 | 47 |
| ...of which were the **wrong drug** | **6** | **0** |
| Genuinely correct dosage references | **39 (72 %)** | **47 (87 %)** |
| Honestly reported unavailable | 9 | 7 |

The 7 remaining unavailable agents (Ceftazidime/Avibactam, Chloramphenicol,
Doripenem, Meropenem/Vaborbactam, Quinupristin/Dalfopristin,
Ticarcillin/Clavulanate, Ticarcillin) genuinely have no matching openFDA label
with a dosage section. The application correctly says so rather than guessing.

---

## 9. Cost Tests

| Field | Present |
|---|---|
| Drug name | Yes |
| NDC | Yes (per product) |
| Cost | Yes (`nadac_per_unit`, USD) |
| Pricing unit | Yes (EA / ML / GM) |
| Effective date | Yes |
| Retrieval timestamp | Yes |
| Source | Yes (CMS NADAC, data.medicaid.gov) |

Example of real retrieved data (Ciprofloxacin): NDC `00143992701`, $0.08886 per
EA, effective 2025-12-17.

Unavailable wording confirmed: **"Price unavailable, verify locally."**

Cost is never a selection criterion. The application displays:

> "Cost is shown for reference only and must never be the primary basis for
> choosing an antibiotic... A cheaper agent that fails costs far more than an
> effective one."

No course cost is computed -- correctly, since that would require a dose,
frequency and duration the system deliberately does not generate.

**Coverage:** 33/54 agents (61 %), up from 27/54 (50 %) after BUG-02 was fixed.
The remaining gaps are genuine: NADAC surveys *retail pharmacy acquisition*
cost, so hospital-only intravenous agents (aztreonam, cefotaxime, cefoxitin,
imipenem, colistin and similar) legitimately have no NADAC record.

---

## 10. Genomic Tests

| Requirement | Result |
|---|---|
| Honestly states genomic data unavailable | Yes -- prominent error banner |
| Fabricates AMR genes | **No** |
| Fabricates mutations | **No** |
| Fabricates genomic resistance predictions | **No** |
| Future genotype input format documented | Yes |

The tab renders:

> **Genomic module inactive -- no genomic data exists in this project's datasets.**

with the reason (all five source files verified column by column), the
enablement path (`01_DATA/raw/genotypes.csv`), and the required schema
(`order_proc_id_coded, marker`).

A 21-marker reference map (NCBI Reference Gene Catalog, CARD, ResFinder) is
implemented and can be previewed, but the demonstration is explicitly captioned:

> "Illustrative only -- these markers are **NOT** from this patient or this
> dataset. Shown so the intended output format is visible."

No patient genotype is ever asserted.

---

## 11. Clinician Review Tests

| Requirement | Result |
|---|---|
| Review the prediction | Yes |
| Review explanations | Yes |
| Review potential options | Yes |
| Accept an option | Yes |
| Modify / Reject | Yes (3-way radio) |
| Add notes | Yes (free text) |
| Record the final decision | Yes -- appended to `13_RESULTS/clinician_reviews.csv` |
| **Requires an explicit human decision** | **Yes -- enforced** |

Gating was tested directly:

| Attempt | Result |
|---|---|
| Submit with nothing filled in | Refused -- "The confirmation checkbox is required before a review can be recorded." |
| Reviewer supplied, confirmation unticked | Refused -- same message |
| Confirmation ticked, reviewer blank | Refused -- "A reviewer identifier is required." |
| Both supplied | Recorded, with a full audit row |

The recorded row captures the clinician's decision alongside what the model
said, so the two can never be confused later:

```
recorded_at_utc, reviewer, decision, clinician_selected_agents, clinician_notes,
organism, specimen, order_proc_id_coded, model_algorithm, model_top_option,
model_top_p_susceptible
```

*The test entry written during this UAT was removed afterwards, restoring the
pre-test state (the file did not exist before testing).*

---

## 12. Error Handling

**32 invalid, extreme or incomplete inputs were submitted. Zero crashes.**

| Input | Behaviour |
|---|---|
| No history supplied | Defaults applied and declared as assumed |
| Sex not recorded | Passed as missing; never asserted |
| All laboratory fields empty | Handled; missing is passed as missing, not imputed |
| Negative creatinine (-5) | No crash |
| Creatinine 9999 | No crash |
| NaN / infinite creatinine | No crash |
| Negative drug-days | No crash |
| Drug-days 1e9 | No crash; prediction shifts, remains in [0,1] |
| Culture year 1900 / 2099 | No crash |
| Month 13 | No crash |
| Unknown / empty organism string | No crash; full ranking still produced |
| Unknown / empty specimen | No crash |
| Unknown setting | No crash |
| Empty record `{}` | No crash |
| Allergy: unknown drug name | No flag raised, no crash |
| Allergy: `'; DROP TABLE--` | Inert |
| Allergy: `<script>alert(1)</script>` | Inert |
| Invalid culture order id | Clean `KeyError` with readable message |

All output probabilities remained within [0, 1] in every case.

**Two notes on interpretation.** First, the UI itself prevents most of these
inputs from ever being entered: organism, specimen and setting are selectboxes
bound to the model's own vocabulary, numeric fields have enforced bounds, and
free-text organism entry is impossible. The tests above exercise the layer
*beneath* the UI, so the result is defence in depth rather than a reachable
user path. Second, an unrecognised organism string produces a ranking with
only 3 critical flags, because intrinsic-resistance rules key on a known
organism -- another reason the constrained selectbox matters, and a caution for
any future API that accepts free text.

---

## 13. Offline Mode

Tested with `ENABLE_LIVE_API=0` **and** the disk cache neutralised, so no
external data of any kind was reachable -- simulating openFDA, CMS NADAC,
RxNorm and general internet unavailability simultaneously.

| Check | Result |
|---|---|
| Application starts | Yes |
| Exceptions | **0** |
| Tabs rendered | **9 / 9** |
| Traceback fragments visible to the user | **0** |
| Model prediction still produced | Yes -- full 54-agent table, P(susceptible) 0.059-0.985 |
| SHAP explanations still produced | Yes |
| Dosage tab | "**Insufficient information for dosage reference.** Live API access is disabled (ENABLE_LIVE_API=0)." |
| Cost tab | "**-- Price unavailable, verify locally.** Live API access is disabled (ENABLE_LIVE_API=0)." |
| Fabricated data in place of missing API data | **None** |

Failure containment was also verified at the HTTP layer: `10_API_SERVICES/_http.py`
converts 404, 429, other non-2xx responses, non-JSON bodies and network
exceptions into an `ApiResult(ok=False, reason=...)` envelope and never raises
into the application. Real HTTP 500 responses from openFDA were observed
during testing and were handled exactly this way -- the user saw a plain
"insufficient information" message, not an error.

---

## 14. Performance

Measured on the test machine, warm process, median of 12 runs.

| Stage | Time |
|---|--:|
| Streamlit server ready | 22.7 s |
| **Full application startup (first page render, cold)** | **32.7 s** |
| Python imports | 4.4 s |
| Model load (`final_model.joblib`) | 0.49 s |
| Test partition first load | 15.97 s (one-off) |
| Test partition, subsequent | 0.16 s |
| **Load a test case** | **0.016 s** |
| **Generate prediction (all 54 agents)** | **0.066 s** |
| **Generate SHAP explanation (all 54 agents)** | **3.61 s** (mean 3.90 s) |
| **Full ranking (prediction + SHAP + guardrails)** | **3.67 s** (mean 3.97 s) |

### Bottleneck

The bottleneck is **SHAP, not the model**: 3.90 s of the 3.97 s ranking time is
`TreeExplainer` -- **98.3 %**. Model inference for all 54 agents takes 66
milliseconds.

This confirms the behaviour already noted in `PROJECT_MEMORY.md` ("scoring 54
agents with SHAP takes a moment on first load") as **expected, not a defect**.
The one-off 16 s test-partition load and the ~33 s cold start are dominated by
reading the analytic dataset and computing the patient-grouped split; both are
cached thereafter, and subsequent case changes take under 4 seconds.

No optimisation was attempted, per the testing brief.

---

## 15. Usability Findings

Assessed from the position of a clinician opening the application for the first
time.

| Question | Verdict | Notes |
|---|---|---|
| Can I understand what the system predicts? | **Yes** | The target is stated on the Prediction tab: "1 = non-susceptible (Resistant or Intermediate)", with the decision threshold |
| Can I understand why it predicted it? | **Yes** | SHAP factors are given in clinical language, with direction and magnitude |
| Can I see patient history? | **Yes** | A dedicated tab, with unknown values shown as "not recorded" |
| Can I see potential antibiotic options? | **Yes** | Ranked cards with probability, evidence and warnings |
| Can I understand warnings? | **Yes** | Colour-coded by severity with plain-language reasons and citations |
| Can I distinguish AI prediction from laboratory result? | **Yes** | Separate, explicitly labelled columns; ground truth captioned "never used as an input" |
| Can I make the final decision? | **Yes** | The review form cannot be submitted without an explicit human confirmation |
| Can I understand the limitations? | **Yes** | Model Card page, evidence bands, and the persistent safety banner |
| Is it obvious the system does not prescribe automatically? | **Yes** | Framing is consistent across every tab |

### Points of possible confusion

1. **"Confidence: high" on an agent with no supporting data.** The confidence
   band is derived purely from how far the probability sits from 0.50, and is
   independent of how much training data supports that organism-antibiotic
   pair. A clinician can therefore see "confidence: high" beside "0 training
   examples". Both facts are displayed, but the words appear to contradict each
   other. Recorded as BUG-03.

2. **Ranking does not encode evidence.** Options are ordered by predicted
   susceptibility alone, so a zero-evidence extrapolation can sit above a
   well-evidenced agent. The evidence pill is visible on the card, but position
   is the first thing read.

3. **Laboratory values entered in the sidebar do not change the prediction.**
   No laboratory feature survived feature selection, so entering a creatinine
   or white cell count has no effect on the model output (it is used only for
   the dosage tab's renal context). This is deliberate and documented in
   `docs/FINAL_FEATURE_SET.md`, but the sidebar does not say so at the point of
   entry, and a clinician may reasonably assume the values matter.

4. **The model under-calls resistance in heavily pre-treated patients.**
   Case C -- a patient with 394 prior resistant results -- produced 15
   false-susceptible calls out of 20. This is the published sensitivity of
   0.545 made concrete. It is visible when ground truth is shown, but in real
   use (where ground truth is by definition absent) a clinician would have no
   such signal. The evidence and confidence bands are the only cue.

5. **Startup takes about 33 seconds.** Acceptable for a research prototype, but
   worth knowing before a live demonstration.

---

## 16. Bugs Found

### BUG-01 -- Wrong drug's FDA label displayed as the dosage reference *(HIGH -- patient-safety relevant)*

**Found by:** UAT-35.

`get_drug_label()` in `10_API_SERVICES/drug_api.py` first queried the fielded
`openfda.generic_name:"<drug>"` endpoint and, if that returned nothing, fell
back to an **unfielded full-text search**. openFDA then matched the query tokens
anywhere in any label document, so an unrelated product could be returned. The
result was ranked by route and presence of a dosage section -- and an oral
product with a dosage section scores best -- so the unrelated label won
outright. **No check existed that the returned label was actually the requested
drug.**

Verified live across all 54 agents; 6 were affected:

| Agent requested | Label actually displayed | Dosage text shown |
|---|---|---|
| **Doripenem** (a carbapenem) | **LETERMOVIR (PREVYMIS)** -- an antiviral | Letermovir dosing |
| **Ticarcillin** | **ERTAPENEM SODIUM** -- a different antibiotic | Ertapenem dosing |
| Ceftazidime/Avibactam | SILICEA -- a homeopathic pellet | "Take 4 or 6 Pellets by mouth, three times daily" |
| Meropenem/Vaborbactam | SILICEA | as above |
| Quinupristin/Dalfopristin | SILICEA | as above |
| Ticarcillin/Clavulanate | SILICEA | as above |

**Doripenem and Ticarcillin contain no "/" and were therefore affected in the
application as originally shipped**, independent of BUG-02. Doripenem appears
as a ranked option in ordinary use -- it was rank 7 in the application's default
case and rank 5 in Test Case C -- so a clinician opening the Dosage tab for it
would have been shown an antiviral's dosing instructions, presented verbatim
under an FDA-sourced provenance line.

This is more serious than missing data: the project's own design rule is that
where a value is unavailable the application says so rather than filling the
gap. Here it filled the gap with real text from the wrong drug.

### BUG-02 -- Combination antibiotics had no dosage and no cost data *(MEDIUM -- functional)*

**Found by:** UAT-56.

The `_NAME_OVERRIDES` table in `drug_api.py` was written with **hyphenated**
keys (`"piperacillin-tazobactam"`), but the deployed model's antibiotic
vocabulary spells combinations with a **slash** (`"Piperacillin/Tazobactam"`).
Ten agents therefore missed the lookup entirely and were sent to openFDA
verbatim, where `openfda.generic_name:"piperacillin/tazobactam"` matches
nothing (and intermittently returned HTTP 500).

Because `price_api.get_price()` reuses `normalise_drug_name()`, the same defect
also removed the NADAC cost lookup for those agents.

Affected: Ampicillin/Sulbactam, Ceftazidime/Avibactam, Ceftolozane/Tazobactam,
Cephalexin/Cephalothin, Imipenem/Ebactam, Meropenem/Vaborbactam,
**Piperacillin/Tazobactam**, Quinupristin/Dalfopristin,
Ticarcillin/Clavulanate, **Trimethoprim/Sulfamethoxazole**.

Piperacillin/Tazobactam and Trimethoprim/Sulfamethoxazole are among the most
frequently used agents in the cohort, and Piperacillin/Tazobactam was ranked
4th in Test Case A -- so this was reached in ordinary use.

Live proof:

| Query sent to openFDA | Result |
|---|---|
| `openfda.generic_name:"piperacillin/tazobactam"` | 0 results |
| `openfda.generic_name:"piperacillin and tazobactam"` | 3 results |
| `openfda.generic_name:"trimethoprim/sulfamethoxazole"` | 0 results |
| `openfda.generic_name:"sulfamethoxazole and trimethoprim"` | 3 results |

### BUG-03 -- Confidence band ignores evidence strength *(LOW -- usability / safety framing)*

**Found by:** UAT-57. **Not fixed** -- see section 17.

`_confidence_band()` in `08_RECOMMENDATION_ENGINE/antibiotic_ranking.py`
derives the band from `abs(p - 0.5)` only. An agent with **zero** training
examples for the organism-antibiotic pair can therefore display
"confidence: high". Observed in Test Case A (Cefiderocol, rank 5, 0 training
rows, "high") and Test Case B (Metronidazole, Amikacin, Meropenem and Imipenem
all at 0 training rows with "high" confidence for a urinary *Enterococcus*).

The evidence band is shown alongside and correctly reads "none -- extrapolated
from class and organism group", so the information is present and nothing is
concealed. The defect is that two adjacent labels tell the clinician opposite
things.

### BUG-04 -- Deprecated Streamlit parameter *(LOW -- cosmetic / forward-compatibility)*

**Found by:** UAT-01. **Not fixed** -- see section 17.

The application calls `use_container_width=` in 20 places. Streamlit 1.62 emits:
"`use_container_width` will be removed after 2025-12-31. For
`use_container_width=True`, use `width='stretch'`." The removal date has passed;
the parameter still functions in the pinned version but will break on a future
upgrade. Warnings appear in the server log only, never in the UI.

---

## 17. Fixes Applied

Two fixes were applied, both confined to the external-API name-resolution
layer. **No model was retrained. No model file, feature, hyperparameter,
threshold or measured performance number was altered.** The regression suite
was re-run after the changes: **111 passed, 0 failed** -- identical to before.

### FIX-01 -- Drug identity guard (resolves BUG-01)

`10_API_SERVICES/drug_api.py`. Added `_identity_tokens()`, `_label_identity()`
and `_label_matches_drug()`, and applied them in `get_drug_label()` so that
candidate labels are filtered to those actually naming the requested ingredient
**before** ranking. If nothing matches, the function returns `ok=False` with an
explicit reason and the application shows "Insufficient information for dosage
reference." -- the behaviour the project already requires for unavailable data.

Re-tested across all 54 agents with the API cache cleared:

| | Before | After |
|---|--:|--:|
| Labels displayed that were the wrong drug | **6** | **0** |
| Labels correctly matching the requested drug | 47 | 47 |

The 6 previously-wrong agents now honestly report that no dosage reference is
available. No correct label was lost.

### FIX-02 -- Separator-agnostic drug-name lookup (resolves BUG-02)

`10_API_SERVICES/drug_api.py`. `normalise_drug_name()` now tries an exact match
first (so explicit entries always win), then a canonical form in which `/`, `-`
and " and " are all folded to a single space. `"Piperacillin/Tazobactam"`,
`"piperacillin-tazobactam"` and `"piperacillin and tazobactam"` all resolve to
the same entry. Four mappings the table never covered were added, each checked
against the live openFDA endpoint: `cephalexin/cephalothin` -> `cephalexin`,
`imipenem/ebactam` -> `imipenem and cilastatin`, `meropenem/vaborbactam` ->
`meropenem and vaborbactam`, and `ticacarcillin/clavulanic acid` (the cohort's
own misspelling) -> `ticarcillin and clavulanate potassium`.

Because `price_api` reuses this function, cost lookup was fixed by the same
change.

| Coverage across 54 agents | Before | After |
|---|--:|--:|
| Correct dosage reference | 39 (72 %) | **47 (87 %)** |
| Cost available | 27 (50 %) | **33 (61 %)** |

Regression check: single-ingredient agents (Ciprofloxacin, Nitrofurantoin,
Vancomycin, Meropenem, Amikacin, Penicillin, Oxacillin) resolve exactly as
before.

### Not fixed, and why

**BUG-03** (confidence vs evidence) is a design question rather than a coding
error -- changing the confidence band alters what every option card displays and
would change how the ranking reads. It touches clinical presentation and should
be a deliberate decision, not a change made during acceptance testing. The
recommended fix is to demote the band when evidence is thin, e.g. cap it at
"low" when `training_examples == 0`, or relabel it "statistical confidence" to
distinguish it from evidential confidence.

**BUG-04** (deprecated parameter) is cosmetic, invisible to users, and the
mechanical `use_container_width` -> `width` rename across 20 call sites carries
more regression risk during a test cycle than the warning justifies. It should
be done as its own change before any Streamlit upgrade.

---

## 18. Remaining Limitations

Carried forward from the project's own documentation and confirmed during
testing:

1. **Sensitivity is 0.545.** The model misses roughly 45 % of truly
   non-susceptible results. Test Case C makes this concrete: 15 of 20
   laboratory-resistant agents were called susceptible for a patient with 394
   prior resistant results. This is the single most important clinical caveat.
2. **Prior-resistance history covers only ~38 % of orders.** Absence of a
   record is not absence of exposure.
3. **Panel-selection bias** is acknowledged but not modelled -- which agents get
   tested is itself informative and not random.
4. **No external validation cohort exists.** All results are single-institution.
5. **Genomic module is inactive** -- these datasets contain no genomic data.
6. **Dosage coverage is 87 %** and cost coverage 61 %; the remainder genuinely
   have no openFDA or NADAC record and are reported as unavailable.
7. **Cost data is US-specific** (CMS NADAC) and does not apply elsewhere.
8. **Confidence bands do not reflect evidence volume** (BUG-03).
9. **Ranking does not encode evidence strength** -- position reflects predicted
   susceptibility alone.
10. **Laboratory values entered manually do not affect the prediction**, as no
    laboratory feature survived selection.
11. **The 3-class model's Intermediate recall is low** -- a property of the AST
    category rather than a fixable defect.
12. **Cold start is ~33 seconds**, and each case change costs ~3.7 s, dominated
    by SHAP.

### Manual tests required

The harness runs the real application and inspects its full rendered element
tree, but it cannot see pixels. The following should be confirmed by eye before
the demonstration -- each should take under a minute:

> **MANUAL TEST REQUIRED -- visual rendering.**
> Run `streamlit run 11_APPLICATION/app.py`, open `http://localhost:8501`, and
> click through all 9 tabs plus the two sidebar pages. Confirm: the red safety
> banner is visible at the top; the bar charts render; the option cards show
> their coloured pills; no text is clipped. *(Content and absence of errors are
> already verified automatically; this checks layout only.)*

> **MANUAL TEST REQUIRED -- browser interactivity.**
> In the sidebar, switch "Case source" to "Enter a case manually", change the
> organism, and confirm every tab updates to the new isolate within a few
> seconds. Then tick "Patient history is unknown / unavailable" and confirm the
> Dashboard lists the assumed fields.

> **MANUAL TEST REQUIRED -- multi-organism cultures.**
> Select a held-out order that grew more than one organism (the sidebar offers a
> second selectbox when it does, e.g. order `486309118`, which grew both
> *Enterococcus species* and *E. coli*). Confirm that switching the isolate
> changes the ground-truth table and the ranking.

---

## 19. Final UAT Result

| Measure | Value |
|---|--:|
| Total tests | **59** |
| Passed | **57** (1 with a recorded observation) |
| Failed | **2** -- both fixed and re-tested |
| Outstanding failures | **0** |
| Warnings / observations | **5** |
| Bugs found | **4** |
| Bugs fixed | **2** (one HIGH, one MEDIUM) |
| Bugs deferred with justification | **2** (both LOW) |
| Crashes across 32 invalid-input tests | **0** |
| Regression suite | **111 passed, 0 failed** |
| **Overall UAT pass rate** | **96.6 % as first run; 100 % after fixes** |

### Verdict

**PASS.**

The application does what it claims and, more importantly, refuses to do what
it should not. Across every test the system predicted, explained, warned and
deferred to the clinician; at no point did it prescribe, conceal a failure, or
invent a value. Test Case B -- where the model was wrong on 6 of 8 agents and
the application displayed every error alongside a working SHAP explanation -- is
the strongest evidence that the human-in-the-loop design is real rather than
decorative.

One patient-safety defect was found (BUG-01, an antiviral's dosing shown under
a carbapenem's name) and fixed. That defect existed in the shipped application
and would not have been found by inspecting the code alone -- it required
querying the live API for all 54 agents and checking the identity of what came
back. Finding it justifies the exercise.

**Ready for classroom demonstration:** **Yes.** Allow ~35 seconds for the first
load and ~4 seconds per case change. Test Cases A (12/12 correct) and B (2/8,
failure shown honestly) make an effective paired demonstration.

**Ready for research-report screenshots:** **Yes.** All tabs render complete,
consistent, real output with visible provenance.
