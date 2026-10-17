"""Model card page — what the model is, how well it works, and where it fails."""
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

from components.ui import inject_css, page_config, safety_banner  # noqa: E402
from src.models.bundle import ModelBundle  # noqa: E402

page_config("Model Card", "📋")
inject_css()

st.title("📋 Model Card")
safety_banner()

try:
    bundle = ModelBundle.load()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

meta = bundle.metadata
test = bundle.metrics.get("test")
val = bundle.metrics.get("validation", {})
temporal = bundle.metrics.get("temporal_test")

st.header("Model details")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Algorithm", meta.get("algorithm", "—"))
c2.metric("Features", meta.get("n_features", len(bundle.features)))
c3.metric("Training rows", f"{meta.get('n_train_rows', 0):,}")
c4.metric("Training patients", f"{meta.get('n_train_patients', 0):,}")

st.markdown(f"""
- **Task** — binary classification: `{meta.get('target_definition', '—')}`
- **Prediction moment** — organism identification is available; the
  susceptibility result is not.
- **Split** — {meta.get('split', '—')}
- **Class-imbalance strategy** — {meta.get('imbalance_strategy', '—')}
- **Random seed** — {meta.get('random_seed', '—')}
- **Dataset version** — {meta.get('dataset_version', '—')}
- **Trained** — {meta.get('saved_at_utc', '—')}
""")

with st.expander("Hyperparameters"):
    st.json(meta.get("hyperparameters", {}))

st.header("Performance")
if test is None:
    st.warning("Held-out test metrics are not in the bundle yet. Run "
               "`python scripts/08_evaluate_models.py`. Validation metrics are "
               "shown below in the meantime.")
    shown, label = val, "Validation (unseen patients)"
else:
    shown, label = test, "Held-out TEST partition (used once)"

st.subheader(label)
c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC", shown.get("pr_auc", "—"),
          help="Primary selection metric. The no-skill baseline equals the "
               "positive-class prevalence (~0.195).")
c2.metric("ROC-AUC", shown.get("roc_auc", "—"))
c3.metric("Sensitivity (non-susceptible)", shown.get("recall_sensitivity", "—"))
c4.metric("Specificity", shown.get("specificity", "—"))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision (PPV)", shown.get("precision_ppv",
                                       shown.get("precision", "—")))
c2.metric("F1", shown.get("f1", "—"))
c3.metric("Accuracy", shown.get("accuracy", "—"))
c4.metric("Brier score", shown.get("brier", "—"),
          help="Lower is better. Measures probability calibration, which "
               "matters because the application ranks agents by probability.")

if shown.get("pr_auc_ci95"):
    st.caption(f"PR-AUC 95% bootstrap CI: {shown['pr_auc_ci95']} · "
               f"ROC-AUC 95% CI: {shown.get('roc_auc_ci95')}")

if temporal:
    st.subheader("Temporal generalisation — fitted on the past, tested on the future")
    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC", temporal.get("pr_auc", "—"),
              delta=(round(temporal["pr_auc"] - shown["pr_auc"], 4)
                     if test else None))
    c2.metric("ROC-AUC", temporal.get("roc_auc", "—"))
    c3.metric("Sensitivity", temporal.get("recall_sensitivity", "—"))
    st.caption("This is the deployment-realistic figure. It trains only on "
               "earlier cultures and tests on the most recent ones, with "
               "patients spanning the cut-off removed from the test side.")

evalp = PROJECT_ROOT / "06_MODEL_EVALUATION" / "figures" / \
    "01_roc_pr_calibration.png"
if evalp.exists():
    st.image(str(evalp), use_container_width=True)

cm = PROJECT_ROOT / "06_MODEL_EVALUATION" / "figures" / "02_confusion_matrix.png"
if cm.exists():
    st.image(str(cm), use_container_width=True)

per_abx = PROJECT_ROOT / "06_MODEL_EVALUATION" / "tables" / \
    "per_antibiotic_performance.csv"
if per_abx.exists():
    st.subheader("Where the model works — and where it does not")
    d = pd.read_csv(per_abx)
    st.dataframe(d, use_container_width=True, hide_index=True)
    st.caption("Compare PR-AUC against `positive_rate`: an agent whose PR-AUC "
               "barely exceeds its prevalence is one where the model adds "
               "little over guessing the base rate.")

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
   academic health system. Resistance epidemiology is intensely local; the
   model should be recalibrated before use anywhere else.
2. **Three specimen types only.** Urine, blood and respiratory. Wound, tissue,
   CSF, stool and device isolates are absent, and the model must not be applied
   to them.
3. **Adults only.** The youngest age band is 18-24 years. There is no
   paediatric or neonatal representation.
4. **Panel-selection bias.** An isolate only has a label for the agents the
   laboratory chose to test, and that choice depends on the organism, the
   specimen and local protocol. The model learns the resistance pattern
   *conditional on being tested*, which is not the same as the pattern that
   would be seen if everything were tested.
5. **Labels are laboratory categories, not outcomes.** Susceptible in vitro
   does not guarantee clinical success; resistance breakpoints also changed
   during the 2007-2024 span.
6. **No genomic data.** The source datasets contain no gene, mutation or AMR
   marker. The genomic module is implemented but inactive, and the model makes
   no genotypic claims.
7. **Partial history coverage.** Prior resistance history exists for only ~38%
   of culture orders and prior medications for ~73%. Absence of a record is not
   evidence of absence of exposure — care outside this health system is
   invisible.
8. **Deidentified sex.** The sex variable is coded 0/1 with no legend, so no
   sex-specific claim can be made or checked.
9. **Temporal drift.** Performance measured on a temporal holdout is lower than
   on a random patient split. A deployed model would need monitoring and
   periodic refitting.
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
            st.dataframe(pd.read_csv(p), use_container_width=True,
                         hide_index=True)
if not found:
    st.info("Subgroup tables are produced by `scripts/08_evaluate_models.py`.")
else:
    st.caption("A large PR-AUC spread across subgroups means the model serves "
               "some groups better than others. Note that PR-AUC is not "
               "comparable across groups with different prevalence — read each "
               "group's PR-AUC against its own `positive_rate`.")

st.header("Ethical considerations")
st.markdown("""
- **Automation bias is the principal risk.** A ranked list is persuasive. The
  interface therefore never presents a single answer, always shows the evidence
  base and confidence, and requires an explicit clinician decision.
- **Under-prediction of resistance is the costly error.** It leads to an
  ineffective antibiotic. The threshold analysis in
  `docs/MODEL_EVALUATION.md` reports operating points for asymmetric costs so a
  deploying site can choose deliberately.
- **Over-prediction of resistance drives escalation.** Pushing clinicians
  towards broader agents accelerates the resistance the tool is meant to help
  with. Both errors are reported, neither is optimised away silently.
- **The training data reflects historical prescribing.** Where past practice was
  inequitable, a model fitted to it can perpetuate that. Subgroup results are
  published above rather than summarised into a single number.
""")

card = PROJECT_ROOT / "MODEL_CARD.md"
if card.exists():
    with st.expander("Full MODEL_CARD.md"):
        st.markdown(card.read_text(encoding="utf-8"))
