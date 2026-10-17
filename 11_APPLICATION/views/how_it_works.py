"""How It Works — the prediction pipeline, and the full data/methodology audit trail."""
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

from components.ui import safety_banner  # noqa: E402

st.title("How It Works")
safety_banner()

st.markdown("""
```
Clinical / Laboratory Data
        ↓
Feature Processing
        ↓
LightGBM Prediction
        ↓
Resistance Probability
        ↓
Antimicrobial Ranking
        ↓
SHAP Explanation
        ↓
Clinical Decision Support
```
""")

st.markdown("""
This system does not replace:
- Laboratory antimicrobial susceptibility testing
- Clinical expertise
- Medical guidelines
- Clinician judgement

It provides AI-assisted predictive insight based on learned historical data
patterns, for review by a qualified clinician.
""")

st.divider()
st.header("Dataset and methodology")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


audit = load_json(PROJECT_ROOT / "02_DATA_ANALYSIS" / "reports" /
                  "dataset_audit.json")
build = load_json(PROJECT_ROOT / "01_DATA" / "processed" / "build_report.json")
leak = load_json(PROJECT_ROOT / "02_DATA_ANALYSIS" / "reports" / "leakage.json")

tabs = st.tabs(["Source data", "Cleaning decisions", "Leakage controls",
               "Pipeline", "Documents"])

# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("The five source files")
    if audit:
        rows = [{
            "Dataset": d["key"],
            "File": d["file_name"],
            "Size (MB)": round(d["file_size_mb"], 1),
            "Rows": d["n_rows"],
            "Columns": d["n_columns"],
            "Exact duplicate rows": d["n_exact_duplicate_rows"],
        } for d in audit["datasets"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                    hide_index=True)
        st.caption("Raw files are referenced in place and never modified.")
    diag = PROJECT_ROOT / "docs" / "diagrams" / "dataset_relationships.png"
    if diag.exists():
        st.subheader("How the tables relate")
        st.image(str(diag), use_container_width=True)

    if build:
        st.subheader("From raw rows to the modelling table")
        s1 = build["step1_cohort"]
        f = build["step7_final"]
        steps = [
            ("Raw cohort rows", s1["n_rows_raw"]),
            ("− negative cultures (no susceptibility result)",
             -s1["n_dropped_no_label_negative_culture"]),
            ("− Inconclusive / Synergism (not a phenotype)",
             -s1["n_dropped_non_phenotype_result"]),
            ("− duplicate (order, organism, antibiotic) resolved",
             -s1["n_dropped_duplicate_resolution"]),
            ("= modelling rows", f["n_rows"]),
        ]
        st.dataframe(pd.DataFrame(steps, columns=["Step", "Rows"]),
                    use_container_width=True, hide_index=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modelling rows", f"{f['n_rows']:,}")
        c2.metric("Culture orders", f"{f['n_orders']:,}")
        c3.metric("Patients", f"{f['n_patients']:,}")
        c4.metric("Non-susceptible rate", f"{f['target_positive_rate']:.1%}")

# --------------------------------------------------------------------------
with tabs[1]:
    if not build:
        st.info("Build report not available.")
    else:
        s1 = build["step1_cohort"]
        st.subheader("Organism names")
        st.markdown(f"""
The raw `organism` column holds **{s1['organism_raw_distinct']} distinct
strings** from a laboratory dictionary. After normalisation there are
**{s1['organism_clean_distinct']}** clean species-level names. The cleaning
removes deprecated `ZZZ`-prefixed entries, reconciles `(GROUP B)` against
`{{GROUP B}}`, collapses `… COMPLEX` onto the species, and extracts colony
morphotype into its own feature.
""")
        st.subheader("Duplicate resolution")
        st.markdown(f"""
**{s1['n_rows_in_duplicate_key_groups']:,} rows** sat in
**{s1['n_duplicate_key_groups']:,}** duplicated
(order, organism, antibiotic) groups, of which
**{s1['n_duplicate_groups_with_conflicting_labels']:,}** carried *conflicting*
susceptibility results.

Resolution rule: **{s1['dedup_rule']}**. {s1['n_dropped_duplicate_resolution']:,}
rows were removed.
""")
        st.subheader("Laboratory results")
        s3 = build["step3_labs"]
        st.markdown(f"""
The labs export carries **{s3['rows_per_order']['mean']} rows per culture order
on average** (max {s3['rows_per_order']['max']}).

{s3['collapse_rule']}
""")
        st.subheader("Prior medications")
        s4 = build["step4_prior_med"]
        st.markdown(f"""
{s4['dedup_note']}

**{s4['n_rows_same_day_t_eq_0']:,} rows** sat at exactly 0 days from the
culture. {s4['same_day_exclusion_reason']}
""")

# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Leakage found and removed")
    if build and leak:
        s1 = build["step1_cohort"]
        st.error(f"**Resistance phenotype embedded in the organism name — "
                f"{s1['organism_leaked_phenotype_rows']:,} rows.**")
        st.markdown(
            "Values such as `STAPH AUREUS {MRSA}`, "
            "`KLEBSIELLA PNEUMONIAE (CARBAPENEM RESISTANT)` and "
            "`ENTEROCOCCUS FAECIUM - VANCO RESISTANT` restate the label in "
            "the organism field. The normaliser strips them into a separate "
            "field which is then dropped.")
        st.success("**Prior resistance history — investigated and cleared.**")
        rt = leak["resistance_time_to_culture"]
        st.markdown(f"""
Minimum time before the culture is **{rt['min']} day**, with
**{rt['n_le_zero']}** events at or after the culture — cleared as a leakage
risk.
""")
        st.info("**Split-level leakage.** Every split is grouped by patient. "
               "No patient appears in more than one partition — asserted by "
               "`12_TESTS/test_leakage_and_splits.py`.")

    doc = PROJECT_ROOT / "docs" / "DATA_LEAKAGE_ANALYSIS.md"
    if doc.exists():
        with st.expander("Full leakage analysis"):
            st.markdown(doc.read_text(encoding="utf-8"))

# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Pipeline")
    st.markdown("""
```
Raw CSVs (read-only, referenced in place)
        ↓  scripts/01_dataset_audit.py
Dataset audit  →  docs/DATA_DICTIONARY.md
        ↓  scripts/02_relationship_and_leakage_analysis.py
Relationships + leakage  →  docs/DATA_RELATIONSHIPS.md, docs/DATA_LEAKAGE_ANALYSIS.md
        ↓  scripts/04_build_analytic_dataset.py
Cleaning + feature engineering  →  01_DATA/processed/analytic_dataset.parquet
        ↓  scripts/06_feature_selection.py
Seven selection methods + experiment  →  config/final_features.json
        ↓  scripts/07_train_models.py
6 model families, imbalance study, tuning, calibration  →  final_model.joblib
        ↓  scripts/08_evaluate_models.py
Held-out test evaluation  →  docs/MODEL_EVALUATION.md
        ↓  scripts/09_explainability.py
SHAP global + local  →  07_EXPLAINABLE_AI/
        ↓
Recommendation engine  →  Streamlit application  →  CLINICIAN
```
""")
    st.caption("Every stage writes a JSON report, and every document in "
              "`docs/` is generated from those reports rather than typed by "
              "hand, so no published number can drift from the data.")

# --------------------------------------------------------------------------
with tabs[4]:
    docs = sorted((PROJECT_ROOT / "docs").glob("*.md"))
    if not docs:
        st.info("No documents generated yet.")
    for d in docs:
        with st.expander(d.name):
            st.markdown(d.read_text(encoding="utf-8"))
