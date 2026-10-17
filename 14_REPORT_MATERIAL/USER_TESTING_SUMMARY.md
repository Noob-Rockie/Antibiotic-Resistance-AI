# User Testing Summary

*Concise summary of user acceptance testing, suitable for inclusion in the
academic report. Full detail: `12_TESTS/USER_ACCEPTANCE_TEST_REPORT.md`.
Test-by-test table: `14_REPORT_MATERIAL/tables/user_testing_summary.csv`.*

---

## Purpose and method

User acceptance testing was carried out on 2026-08-24 to establish whether the
completed clinical decision support application is usable by a clinician and
whether it honours the safety constraints the project sets for itself. The
model was neither retrained nor modified at any point; testing was applied to
the system as built.

The application was driven in three ways. First, it was launched normally
(`streamlit run 11_APPLICATION/app.py`) and confirmed to serve. Second, the
real application script was executed through `streamlit.testing.v1.AppTest` —
Streamlit's own test harness — which renders every tab and exposes the complete
element tree, so each tab's actual content and any raised exception could be
inspected. Third, seven representative cases were driven through the same
modules the application calls, so that every probability, SHAP value and
guardrail reported here is genuine model output rather than a description of
intended behaviour.

Cases were drawn from the **held-out test partition**, whose patients
contributed nothing to training, feature selection or calibration. All
identifiers used are the project's own de-identified coded order keys.

---

## Headline result

| Measure | Value |
|---|--:|
| Tests executed | 59 |
| Passed | 57 |
| Failed | 2 (both fixed and re-tested) |
| Outstanding failures | 0 |
| Bugs found | 4 |
| Bugs fixed | 2 |
| Crashes across 32 invalid-input tests | 0 |
| Existing regression suite | 111 passed, 0 failed |
| **Pass rate** | **96.6 % first run; 100 % after fixes** |

**Verdict: PASS.** The application is suitable for classroom demonstration and
for research-report screenshots.

---

## What was verified

All nine tabs and both auxiliary pages loaded with meaningful, case-consistent
content and no Python traceback. Seven case types were tested: a correct
prediction, an incorrect prediction, a case with extensive prior resistance, a
case with heavy prior antibiotic exposure, a case with no history, a rare
organism with almost no training coverage, and a multi-drug-resistant isolate
exercising the clinical guardrails.

The system's negative guarantees held throughout. It never prescribed: every
option is framed as a "potential option for clinician review", and a source
scan found no prescriptive phrasing. It never removed an option: agents
carrying a critical guardrail flag — intrinsic resistance, unsuitable specimen
site, or documented allergy — were demoted below unflagged options but remained
visible with their reasons, verified in all seven cases. It never fabricated
data: the genomic module honestly reports that these datasets contain no
genomic data, and where an external lookup fails the application says so rather
than filling the gap. The clinician review form cannot be submitted without an
explicit human confirmation and a reviewer identifier, so the human-in-the-loop
step is enforced rather than decorative.

Under `ENABLE_LIVE_API=0`, with the response cache additionally neutralised so
that openFDA, CMS NADAC and RxNorm were all unreachable, the application
started cleanly, rendered all nine tabs, produced the full prediction and SHAP
output, and displayed the required "Insufficient information for dosage
reference." and "Price unavailable, verify locally." messages. No traceback
reached the user in any failure mode tested.

### The most important finding

Test Case B is the clearest evidence that the explainability design is real.
For a urinary *Enterococcus* isolate the model was **wrong on six of eight
agents**, calling laboratory-resistant Ampicillin, Ciprofloxacin, Levofloxacin
and Penicillin susceptible. The application displayed every disagreement in its
ground-truth comparison table, reported the agreement rate as 2/8, and
continued to provide a complete SHAP explanation for each misclassified agent.
**The application explains its failures as readily as its successes.**

Test Case C reinforces the clinical caveat quantitatively: for a patient with
394 documented prior resistant results, the model agreed with the laboratory on
only 5 of 20 agents, calling 15 resistant agents susceptible. This is the
model's published sensitivity of 0.545 made concrete, and it is the single most
important limitation for any reader of this project.

---

## Defects found

Two defects were found and fixed; two are documented and deliberately deferred.

**BUG-01 (high, patient-safety relevant) — the wrong drug's FDA label was shown
as the dosage reference.** When the fielded openFDA query returned nothing, the
client fell back to an unfielded full-text search and accepted whatever came
back, with no check that the label belonged to the requested drug. Across all
54 agents, six were affected: **Doripenem, a carbapenem, displayed the dosing
instructions for LETERMOVIR, an antiviral**; Ticarcillin displayed Ertapenem's;
and four combination agents displayed SILICEA, a homeopathic pellet product
("Take 4 or 6 Pellets by mouth"). Doripenem contains no separator character and
was therefore affected in the application as originally built — it appeared as
rank 7 in the default case, so the path was reachable in ordinary use. This was
more serious than missing data, because it filled a gap with real text from the
wrong product under an FDA provenance line. Fixed by adding a drug-identity
guard that discards any label not naming the requested ingredient; mismatches
went from 6 to 0 with no correct label lost.

**BUG-02 (medium, functional) — combination antibiotics had neither dosage nor
cost data.** The drug-name override table was written with hyphenated keys
(`piperacillin-tazobactam`) while the deployed model's vocabulary uses slashes
(`Piperacillin/Tazobactam`), so ten agents missed the lookup and were queried
under a name openFDA does not index. Because the cost client reuses the same
function, NADAC lookup failed identically. Piperacillin/Tazobactam and
Trimethoprim/Sulfamethoxazole are among the most frequently used agents in the
cohort. Fixed by making the lookup separator-agnostic and adding four verified
mappings.

Together these fixes raised genuinely correct dosage coverage from **39/54
(72 %) to 47/54 (87 %)** and cost coverage from **27/54 (50 %) to 33/54
(61 %)**, while eliminating all six wrong-drug labels. Both fixes are confined
to the external-API name-resolution layer; no model, feature, threshold or
performance figure was altered, and the regression suite passed identically
(111 tests) before and after.

**BUG-03 (low) — the confidence band ignores evidence strength.** The band is
computed from the probability's distance from 0.50 alone, so an agent with zero
training examples for the organism-antibiotic pair can display "confidence:
high" beside an evidence band reading "none — extrapolated". Both facts are
shown, so nothing is concealed, but the two labels appear to contradict each
other. Deferred deliberately: changing it alters clinical presentation on every
option card and should be a considered design decision rather than a change
made during a test cycle.

**BUG-04 (low) — deprecated Streamlit parameter.** `use_container_width` is
used in 20 places and is scheduled for removal; it still functions in the
pinned version and the warning is never visible to users. Deferred as a
standalone change to be made before any Streamlit upgrade.

---

## Performance

| Stage | Time |
|---|--:|
| Application startup (first render, cold) | 32.7 s |
| Load a held-out test case | 0.016 s |
| Generate prediction, all 54 agents | 0.066 s |
| Generate SHAP explanation, all 54 agents | 3.61 s (median) |
| Full ranking (prediction + SHAP + guardrails) | 3.67 s (median) |

The bottleneck is **SHAP, not the model**: `TreeExplainer` accounts for 98.3 %
of ranking time, while inference for all 54 agents takes 66 milliseconds. This
confirms the behaviour already anticipated in the project documentation as
expected rather than defective. Cold start is dominated by the one-off load of
the analytic dataset and the patient-grouped split, both cached thereafter.

---

## Limitations confirmed by testing

Sensitivity of 0.545 means roughly 45 % of truly non-susceptible results are
missed, demonstrated directly in Test Case C. Prior-resistance history covers
only about 38 % of orders, so absence of a record is not absence of exposure.
Ranking is ordered by predicted susceptibility alone and does not encode
evidence strength, so a zero-evidence extrapolation can outrank a
well-evidenced agent, though the evidence band is always displayed alongside.
Laboratory values entered manually do not influence the prediction, because no
laboratory feature survived feature selection. Dosage and cost coverage remain
incomplete at 87 % and 61 % respectively, with the remainder correctly reported
as unavailable. There is no external validation cohort, and the genomic module
remains inactive because these datasets contain no genomic data.

---

## Conclusion

The application performs the twelve tasks it was tested against and upholds the
constraints it claims: it predicts, explains, warns, and defers to the
clinician, without prescribing, concealing failure, or inventing data. The
patient-safety defect found during testing existed in the shipped application
and could not have been found by code inspection alone — it required querying
the live external API for all 54 agents and verifying the identity of every
response. Its discovery and correction is the principal practical outcome of
this testing phase.
