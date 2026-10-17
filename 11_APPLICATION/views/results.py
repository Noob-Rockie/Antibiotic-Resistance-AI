"""Results — the clinical report for one analyzed isolate.

Every number on this page comes straight from `rank_antibiotics()` (real
model prediction + real SHAP), stored in `st.session_state` by
`views/new_analysis.py`. This page renders it; it never computes a
prediction itself.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components import case as case_mod                             # noqa: E402
from components.resources import (cached_cost, cached_dosage,       # noqa: E402
                                  get_bundle, get_modules)
from components.ui import (SAFETY_DISCLAIMER, assumed_fields_notice,  # noqa: E402
                           aware_pill, evidence_pill, pill, provenance,
                           render_flag, risk_badge_html, safety_banner,
                           unavailable)

bundle = get_bundle()
MOD = get_modules()

st.title("Antimicrobial Resistance Analysis")

result = st.session_state.get("analysis_result")
if result is None:
    st.info(
        "**No analysis has been run yet.**\n\n"
        "Start a new clinical analysis to identify an organism, enter "
        "clinical context, and generate a resistance profile.")
    if st.button("Start New Analysis", type="primary"):
        st.switch_page("views/new_analysis.py")
    st.stop()

record = st.session_state["case_record"]
assumed = st.session_state.get("case_assumed", [])
actual_results = st.session_state.get("case_actual_results")
options = st.session_state["analysis_options"]

organism = record.get("organism_clean")
specimen = record.get("culture_description")

c1, c2 = st.columns(2)
c1.metric("Organism", organism or "—")
c2.metric("Analysis Status", "Complete")
st.caption(f"Specimen: {specimen or '—'}")

safety_banner(compact=True)
assumed_fields_notice(assumed)

st.markdown("#### AI Prediction Summary")
st.markdown(
    "The model has generated predicted resistance probabilities for "
    "supported antimicrobial agents.")
st.markdown(f"> {result['framing']}\n>\n> {result['safety_notice']}")

# ===========================================================================
# RANKING TABLE
# ===========================================================================
st.markdown("#### Antimicrobial Ranking")
st.caption(
    "Ranked by predicted susceptibility. A lower predicted resistance risk "
    "is not an instruction to prescribe — every option requires clinician "
    "review, and flagged agents (guardrails below) are ranked, not removed.")

table = options[["rank", "antibiotic", "antibiotic_subclass",
                 "p_nonsusceptible", "confidence_band",
                 "has_critical_flag"]].copy()
table["Risk Category"] = table["p_nonsusceptible"].apply(
    lambda p: risk_badge_html(p))
table["Clinical Review"] = table["has_critical_flag"].apply(
    lambda f: "🛑 Guardrail flag — review required" if f else "No guardrail flags")
table = table.rename(columns={
    "rank": "Rank", "antibiotic": "Antimicrobial Agent",
    "antibiotic_subclass": "Class",
    "p_nonsusceptible": "Predicted Resistance Probability",
    "confidence_band": "Confidence"})

st.write(
    table[["Rank", "Antimicrobial Agent", "Class",
          "Predicted Resistance Probability", "Risk Category",
          "Confidence", "Clinical Review"]]
    .to_html(escape=False, index=False,
            formatters={"Predicted Resistance Probability": "{:.1%}".format}),
    unsafe_allow_html=True)

st.divider()

# ===========================================================================
# PER-AGENT DETAIL
# ===========================================================================
st.markdown("#### Options for review — Antibiotic Detail View")
st.caption("Select an agent below to review the AI's explanation, evidence, "
          "and reference information for that option. These are options for "
          "clinician review, not a prescription.")

n_show = st.slider("How many options to expand below", 3, min(20, len(options)),
                   min(8, len(options)))
for _, o in options.head(n_show).iterrows():
    cls = "optcard flagged" if o["has_critical_flag"] else "optcard"
    pills = (aware_pill(o["aware_category"])
             + evidence_pill(o["evidence_strength"])
             + (pill("oral option available", "#e0f2fe", "#075985")
                if o["route_oral_available"] else "")
             + (pill("urinary agent only", "#fef3c7", "#92400e")
                if o["urinary_only"] else ""))
    st.markdown(
        f'<div class="{cls}">'
        f'<h4 style="margin:0 0 0.35rem 0;">{o["rank"]}. {o["antibiotic"]}'
        f'<span style="font-weight:400;color:#64748b;font-size:0.85rem;"> — '
        f'{o["antibiotic_subclass"]}</span></h4>'
        f'<div style="margin-bottom:0.5rem;">{pills}</div>'
        f'{risk_badge_html(o["p_nonsusceptible"])} &nbsp;·&nbsp; '
        f'predicted susceptibility: {o["p_susceptible"]:.0%} &nbsp;·&nbsp; '
        f'{o["training_examples"]:,} training examples for this '
        f'organism-agent pair'
        f'</div>', unsafe_allow_html=True)

    with st.expander(f"Detail — {o['antibiotic']}",
                     expanded=bool(o["has_critical_flag"])):
        tab_clinical, tab_technical, tab_dosage, tab_cost = st.tabs(
            ["Clinical View", "Technical Details", "Dosage reference", "Cost"])

        with tab_clinical:
            st.caption("AI Prediction Rationale — Explanation")
            st.markdown(f"**Model reasoning.** {o['explanation']}")
            if o["top_factors"]:
                st.markdown("**Factors increasing predicted resistance**")
                up = [f for f in o["top_factors"] if f["shap"] > 0]
                if up:
                    for f in up:
                        st.markdown(f"- ▲ {MOD['ranking'].humanise(f['feature'])}")
                else:
                    st.caption("None among the top contributing factors.")
                st.markdown("**Factors decreasing predicted resistance**")
                down = [f for f in o["top_factors"] if f["shap"] < 0]
                if down:
                    for f in down:
                        st.markdown(f"- ▼ {MOD['ranking'].humanise(f['feature'])}")
                else:
                    st.caption("None among the top contributing factors.")
            else:
                unavailable("Per-factor explanation unavailable for this "
                           "prediction.",
                           "SHAP could not be computed for this model/row "
                           "combination. The probability above is unaffected.")
            if o["flags"]:
                st.markdown("**Guardrails:**")
                for f in o["flags"]:
                    render_flag(f)
            else:
                st.caption("No guardrail flags raised for this agent.")

        with tab_technical:
            st.caption("Contributions are SHAP values computed on the "
                      "deployed model for this specific isolate — not "
                      "generic feature importances.")
            if o["top_factors"]:
                fdf = pd.DataFrame(o["top_factors"])
                fdf["Factor"] = fdf["feature"].map(MOD["ranking"].humanise)
                fdf["Effect on predicted resistance"] = fdf["shap"].round(4)
                st.dataframe(
                    fdf[["Factor", "Effect on predicted resistance",
                        "direction", "feature", "value"]]
                    .rename(columns={"direction": "Direction",
                                     "feature": "Model feature",
                                     "value": "Encoded value"}),
                    use_container_width=True, hide_index=True)
                st.caption("Positive values push the prediction towards "
                          "resistance; negative values push it towards "
                          "susceptibility. Values are on the model's "
                          "log-odds scale.")
            else:
                st.caption("No SHAP values available for this row.")
            c1, c2 = st.columns(2)
            c1.metric("Confidence band", o["confidence_band"])
            c2.metric("Training examples (organism-agent pair)",
                      f"{o['training_examples']:,}")

        with tab_dosage:
            st.caption("Dosage reference — clinician verification required")
            cr = record.get("median_cr")
            cr = (None if cr is None or (isinstance(cr, float) and np.isnan(cr))
                  else float(cr))
            d = cached_dosage(o["antibiotic"], cr)
            if not d["available"]:
                unavailable(d["message"], d.get("reason"))
                provenance(d.get("source"), d.get("source_url"),
                          d.get("retrieved_at_utc"))
            else:
                st.warning(d["clinician_notice"])
                st.markdown("**Dosage and administration (FDA label, verbatim)**")
                st.text_area("label text", d["dosage_and_administration"],
                            height=220, label_visibility="collapsed",
                            key=f"dose_{o['antibiotic']}")
                provenance(d["source"], d["source_url"], d["retrieved_at_utc"],
                          d.get("from_cache", False))

        with tab_cost:
            st.caption("Medicine cost reference")
            c = cached_cost(o["antibiotic"])
            if not c["available"]:
                unavailable(f"Price unavailable, verify locally.",
                           c.get("reason"))
            else:
                rows = [{
                    "Product (NDC description)": p["description"],
                    "Price (USD)": p["price_per_unit_usd"],
                    "Per": p["pricing_unit"],
                    "Generic/Brand": p["generic_or_brand"],
                } for p in c["products"][:3]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                            hide_index=True)
                st.warning(c["cost_notice"])
                provenance(c["source"], c["source_url"],
                          c["retrieved_at_utc"], c.get("from_cache", False),
                          extra=f"basis: {c['price_basis']}")

st.divider()

# ===========================================================================
# PATIENT HISTORY
# ===========================================================================
with st.expander("Patient history used by the model"):
    st.caption(
        "Patient history — every item below is restricted to events that "
        "occurred strictly BEFORE this culture.")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("**Prior cultures, resistance and antibiotic exposure**")
        st.dataframe(case_mod.patient_history_view(record),
                    use_container_width=True, hide_index=True, height=380)
    with c2:
        st.markdown("**Laboratory results (14 days before the culture)**")
        st.dataframe(case_mod.labs_view(record), use_container_width=True,
                    hide_index=True, height=380)
        st.caption("'not available' means the test was not performed in "
                  "that window — informative, and passed to the model as "
                  "missing rather than imputed.")

# ===========================================================================
# GENOMIC
# ===========================================================================
with st.expander("Genomic — gene-based AMR analysis"):
    st.caption("Genomic AMR marker analysis")
    status = MOD["genomic"].availability_report()
    if not status["available"]:
        st.error("**Genomic module inactive — no genomic data exists in "
                 "this project's datasets.**")
        st.markdown(status["reason"])
        st.caption(f"A reference map of {status['reference_map_size']} AMR "
                  f"markers is implemented and ready. {status['reference_note']}")
    else:
        markers = MOD["genomic"].markers_for_order(
            record.get("order_proc_id_coded"))
        if not markers:
            st.info("A genotype table is present but contains no markers "
                   "for this culture order.")
        for a in markers:
            st.markdown(f"**{a['marker']} — {a['gene_family']}**")
            st.markdown(f"Mechanism: {a['mechanism']}")
            st.info(a["implication"])

# ===========================================================================
# HELD-OUT GROUND TRUTH (demonstration cases only)
# ===========================================================================
if actual_results is not None:
    with st.expander("Ground truth for this held-out culture (demonstration)"):
        st.caption("The laboratory's actual AST result, shown so the "
                  "prediction can be checked. It is never used as an input.")
        merged = actual_results.merge(
            options[["antibiotic", "p_susceptible", "predicted_label"]],
            on="antibiotic", how="left")
        merged["correct"] = np.where(
            merged["predicted_label"].isna(), "—",
            np.where((merged["non_susceptible"] == 1) ==
                    (merged["predicted_label"] == "Non-susceptible"),
                    "✔", "✘"))
        st.dataframe(
            merged.rename(columns={
                "antibiotic": "Antibiotic",
                "laboratory_result": "Laboratory result",
                "p_susceptible": "Model P(susceptible)",
                "predicted_label": "Model call",
                "correct": "Agreement"})
            [["Antibiotic", "Laboratory result", "Model P(susceptible)",
              "Model call", "Agreement"]],
            use_container_width=True, hide_index=True)
        n_ok = int((merged["correct"] == "✔").sum())
        n_tot = int((merged["correct"] != "—").sum())
        if n_tot:
            st.info(f"Agreement on this culture: **{n_ok}/{n_tot}** agents "
                   f"({n_ok/n_tot:.0%}). A single culture is far too small "
                   "to judge a model — see Model Insights for the held-out "
                   "test metrics.")

st.divider()

# ===========================================================================
# CLINICIAN REVIEW
# ===========================================================================
st.markdown("#### Clinician Review — the decision stays with you")
st.markdown(
    """
```
AI prediction  →  Explanation  →  Potential options  →  CLINICIAN REVIEW
                                                              ↓
                                             Accept / Modify / Reject  →  Notes
```
""")
st.caption("Nothing is actioned by this application. The record below is a "
          "review artefact for audit and for the research report.")

with st.form("clinician_review"):
    c1, c2 = st.columns(2)
    reviewer = c1.text_input("Reviewing clinician (name or identifier)")
    decision = c2.radio("Decision on the model's ranking",
                        ["Accept", "Modify", "Reject"], horizontal=True)
    top_default = (options[~options["has_critical_flag"]]["antibiotic"]
                  .tolist() or options["antibiotic"].tolist())[:1]
    selected = st.multiselect(
        "Agent(s) the clinician considers appropriate",
        options["antibiotic"].tolist(), default=top_default)
    reason = st.text_area(
        "Clinical reasoning / notes",
        placeholder="e.g. Model ranks nitrofurantoin first, but this is a "
                    "bacteraemic urinary source, so a systemic agent is "
                    "required.")
    confirm = st.checkbox(
        "I confirm I am a qualified healthcare professional and that this "
        "decision is my own clinical judgement, informed but not "
        "determined by the model output.")
    submitted = st.form_submit_button("Record review", type="primary")

if submitted:
    if not confirm:
        st.error("The confirmation checkbox is required before a review "
                 "can be recorded.")
    elif not reviewer.strip():
        st.error("A reviewer identifier is required.")
    else:
        entry = {
            "recorded_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds"),
            "reviewer": reviewer.strip(),
            "decision": decision,
            "clinician_selected_agents": selected,
            "clinician_notes": reason.strip(),
            "organism": organism,
            "specimen": specimen,
            "order_proc_id_coded": record.get("order_proc_id_coded"),
            "model_algorithm": bundle.metadata.get("algorithm"),
            "model_top_option": (options.iloc[0]["antibiotic"]
                                 if len(options) else None),
            "model_top_p_susceptible": (float(options.iloc[0]["p_susceptible"])
                                        if len(options) else None),
        }
        st.session_state.setdefault("reviews", []).append(entry)
        out = PROJECT_ROOT / "13_RESULTS" / "clinician_reviews.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{**entry,
                             "clinician_selected_agents": "; ".join(selected)}])
        row.to_csv(out, mode="a", header=not out.exists(), index=False)
        st.success(f"Review recorded and appended to `{out}`.")

if st.session_state.get("reviews"):
    st.markdown("##### Reviews recorded in this session")
    st.dataframe(pd.DataFrame(st.session_state["reviews"]),
                use_container_width=True, hide_index=True)

st.divider()
st.caption(SAFETY_DISCLAIMER)
