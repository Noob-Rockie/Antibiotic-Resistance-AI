"""Model Insights — model development, performance, and limitations.

Every metric below is read live from the project's own evaluation
artefacts (the trained model bundle, `06_MODEL_EVALUATION/evaluation_report.json`,
`05_MODEL_TRAINING/experiments/experiment_log.csv`, and
`12_TESTS/uat_report.json`). Nothing here is hardcoded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components.resources import get_bundle  # noqa: E402
from components.ui import safety_banner       # noqa: E402

bundle = get_bundle()

st.title("Model Insights")
safety_banner()

meta = bundle.metadata
test = bundle.metrics.get("test")
val = bundle.metrics.get("validation", {})
temporal = bundle.metrics.get("temporal_test")

# ===========================================================================
# MODEL DEVELOPMENT
# ===========================================================================
st.header("Model development")
st.markdown(
    "Multiple model families were evaluated on the same feature set and "
    "validation partition before selecting a final algorithm.")

exp_log = PROJECT_ROOT / "05_MODEL_TRAINING" / "experiments" / "experiment_log.csv"
if exp_log.exists():
    df = pd.read_csv(exp_log)
    fam = df[df.get("notes") == "model family comparison"]
    if not fam.empty:
        show = fam[["algorithm", "accuracy", "precision", "recall_sensitivity",
                   "specificity", "f1", "roc_auc", "pr_auc", "brier",
                   "train_seconds"]].copy()
        show = show.sort_values("pr_auc", ascending=False)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(
            f"Final selected model: **{meta.get('algorithm', '—')}** — "
            "selected primarily on PR-AUC (the metric best suited to this "
            "class-imbalanced task), with training time and calibration "
            "quality as secondary considerations. Source: "
            "`05_MODEL_TRAINING/experiments/experiment_log.csv`.")
    else:
        st.info("Model family comparison rows not found in the experiment log.")
else:
    st.info("Experiment log not available.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Algorithm", meta.get("algorithm", "—"))
c2.metric("Features", meta.get("n_features", len(bundle.features)))
c3.metric("Training rows", f"{meta.get('n_train_rows', 0):,}")
c4.metric("Training patients", f"{meta.get('n_train_patients', 0):,}")
with st.expander("Hyperparameters"):
    st.json(meta.get("hyperparameters", {}))

st.divider()

# ===========================================================================
# HELD-OUT TEST PERFORMANCE
# ===========================================================================
st.header("1. Held-out test performance")
st.caption("Primary held-out metric — a random, patient-grouped split. No "
          "patient appears in both train and test.")
shown = test if test is not None else val
if shown:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC", shown.get("pr_auc", "—"),
             help="Primary selection metric. No-skill baseline ≈ the "
                  "positive-class prevalence (~0.195).")
    c2.metric("ROC-AUC", shown.get("roc_auc", "—"))
    c3.metric("Sensitivity (non-susceptible)", shown.get("recall_sensitivity", "—"))
    c4.metric("Specificity", shown.get("specificity", "—"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (PPV)", shown.get("precision_ppv", shown.get("precision", "—")))
    c2.metric("F1", shown.get("f1", "—"))
    c3.metric("Accuracy", shown.get("accuracy", "—"))
    c4.metric("Brier score", shown.get("brier", "—"),
             help="Lower is better — probability calibration quality.")
    if shown.get("pr_auc_ci95"):
        st.caption(f"PR-AUC 95% bootstrap CI: {shown['pr_auc_ci95']} · "
                  f"ROC-AUC 95% CI: {shown.get('roc_auc_ci95')}")
else:
    st.info("Held-out test metrics are not in the bundle yet.")

evalp = PROJECT_ROOT / "06_MODEL_EVALUATION" / "figures" / "01_roc_pr_calibration.png"
if evalp.exists():
    st.image(str(evalp), use_container_width=True)

st.divider()

# ===========================================================================
# TEMPORAL HOLDOUT PERFORMANCE
# ===========================================================================
st.header("2. Temporal holdout performance")
st.caption("Trained on the past, tested on the future — the "
          "deployment-realistic figure. Patients spanning the cutoff are "
          "removed from the test side.")
if temporal:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", temporal.get("accuracy", "—"))
    c2.metric("PR-AUC", temporal.get("pr_auc", "—"),
             delta=(round(temporal["pr_auc"] - shown["pr_auc"], 4) if shown else None))
    c3.metric("ROC-AUC", temporal.get("roc_auc", "—"))
    c4.metric("Sensitivity", temporal.get("recall_sensitivity", "—"))
    st.caption("Sensitivity and PR-AUC both drop under temporal drift "
              "relative to the random-split figures above — the honest "
              "expectation for a model deployed prospectively.")
else:
    st.info("Temporal holdout metrics are not in the bundle yet.")

st.divider()

# ===========================================================================
# THREE-CLASS EXPERIMENT
# ===========================================================================
st.header("3. Three-class experiment")
st.caption("A secondary experiment predicting Susceptible / Intermediate / "
          "Resistant directly, rather than the primary binary target.")
eval_report = PROJECT_ROOT / "06_MODEL_EVALUATION" / "evaluation_report.json"
three_class = None
if eval_report.exists():
    three_class = json.loads(eval_report.read_text(encoding="utf-8")).get("three_class")
if three_class:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", three_class.get("accuracy", "—"))
    c2.metric("Macro F1", three_class.get("macro_f1", "—"))
    c3.metric("Weighted F1", three_class.get("weighted_f1", "—"))
    cls_report = three_class.get("classification_report", {})
    rows = [{"Class": k, "Precision": v.get("precision"),
            "Recall": v.get("recall"), "F1": v.get("f1-score"),
            "Support": v.get("support")}
           for k, v in cls_report.items()
           if k in ("Susceptible", "Intermediate", "Resistant")]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if three_class.get("conclusion"):
        st.info(three_class["conclusion"])
else:
    st.info("Three-class experiment results not available in "
           "`06_MODEL_EVALUATION/evaluation_report.json`.")

st.divider()

# ===========================================================================
# USER ACCEPTANCE TESTING
# ===========================================================================
st.header("4. User Acceptance Testing")
st.warning("This section reports clinician-facing acceptance testing, "
          "**not** a model accuracy metric.")
uat_path = PROJECT_ROOT / "12_TESTS" / "uat_report.json"
if uat_path.exists():
    uat = json.loads(uat_path.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Test result", uat.get("result", "—"))
    c2.metric("Tests passed", f"{uat.get('tests_passed', '—')}/{uat.get('tests_total', '—')}")
    c3.metric("Cases scanned", uat.get("cases_scanned", "—"))
    if uat.get("demo_menu_agreement_rate") is not None:
        st.metric("Demo-menu agreement rate",
                 f"{uat['demo_menu_agreement_rate']:.2%}",
                 help=uat.get("demo_menu_caveat"))
        if uat.get("demo_menu_caveat"):
            st.caption(uat["demo_menu_caveat"])
else:
    st.info("User acceptance test report not available.")

st.divider()

# ===========================================================================
# INTENDED USE / LIMITATIONS / FAIRNESS / ETHICS
# ===========================================================================
st.header("Intended use")
st.markdown("""
**Intended.** A research and educational decision-support prototype that
estimates the probability a laboratory would report an identified isolate as
non-susceptible to a given antibiotic, and presents that estimate — with its
explanation, its evidence base and its guardrails — to a qualified clinician
during the window between organism identification and the AST result.

**Not intended.**
- Autonomous prescribing, or any use where the output is acted on without a
  qualified clinician reviewing it.
- A substitute for antimicrobial susceptibility testing.
- Use outside the population it was fitted on (see Limitations).
- Prediction of clinical cure, mortality, or any outcome other than the
  laboratory susceptibility category.
- Dose selection. The application relays FDA label text verbatim and never
  computes a dose.
""")

st.header("Limitations")
st.markdown("""
1. **Single-centre, single-country data.** All cultures come from one US
   academic health system.
2. **Three specimen types only.** Urine, blood and respiratory.
3. **Adults only.** The youngest age band is 18-24 years.
4. **Panel-selection bias.** An isolate only has a label for the agents the
   laboratory chose to test.
5. **Labels are laboratory categories, not outcomes.**
6. **No genomic data.** The genomic module is implemented but inactive.
7. **Partial history coverage.** Absence of a record is not evidence of
   absence of exposure.
8. **Deidentified sex.** No sex-specific claim can be made or checked.
9. **Temporal drift.** Performance on a temporal holdout is lower than on a
   random patient split — this is out-of-scope drift to monitor, not a
   contradiction.
""")

st.header("Fairness and subgroup performance")
sub = PROJECT_ROOT / "06_MODEL_EVALUATION" / "tables"
found = False
for col in ("age", "gender_code", "ordering_mode", "culture_description",
           "organism_group"):
    p = sub / f"subgroup_{col}.csv"
    if p.exists():
        found = True
        with st.expander(f"By {col}"):
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
if not found:
    st.info("Subgroup tables are not available.")

st.header("Ethical considerations")
st.markdown("""
- **Automation bias is the principal risk.** The interface never presents a
  single answer, always shows evidence and confidence, and requires an
  explicit clinician decision.
- **Under-prediction of resistance is the costly error.**
- **Over-prediction of resistance drives escalation.**
- **The training data reflects historical prescribing.**
""")

card = PROJECT_ROOT / "MODEL_CARD.md"
if card.exists():
    with st.expander("Full MODEL_CARD.md"):
        st.markdown(card.read_text(encoding="utf-8"))
