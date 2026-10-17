"""New Clinical Analysis — guided organism -> clinical context -> analyze wizard."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components import case as case_mod                    # noqa: E402
from components.organisms import search_organisms          # noqa: E402
from components.resources import get_bundle, get_modules, get_sample_orders  # noqa: E402
from components.ui import stepper_html                     # noqa: E402

bundle = get_bundle()
MOD = get_modules()

STEPS = ["Identify Organism", "Clinical Context", "Analyze"]
ss = st.session_state
ss.setdefault("na_step", 0)
ss.setdefault("na_mode", "search")
ss.setdefault("na_organism", None)

st.title("New Clinical Analysis")
st.markdown(stepper_html(STEPS, ss["na_step"]), unsafe_allow_html=True)


def _go(step: int) -> None:
    ss["na_step"] = step
    st.rerun()


# ===========================================================================
# STEP 1 — IDENTIFY ORGANISM
# ===========================================================================
if ss["na_step"] == 0:
    st.subheader("Identify Organism")
    ss["na_mode"] = st.radio(
        "How was the organism identified?",
        ["search", "sample"], index=0 if ss["na_mode"] == "search" else 1,
        format_func=lambda m: ("Search bacterial organism"
                               if m == "search" else
                               "Load a real de-identified culture (demonstration)"),
        horizontal=True)

    if ss["na_mode"] == "search":
        query = st.text_input(
            "Search bacterial organism…", value="",
            placeholder="e.g. Escherichia coli, Klebsiella, Staphylococcus…",
            help="Matches are restricted to the organisms the trained model "
                 "actually learned from during training.")
        matches = search_organisms(bundle, query)
        if query.strip() and not matches:
            st.error(
                "**This organism is not supported by the current trained "
                "model.**\n\n"
                "Recommended action: select an organism included in the "
                "supported, dataset-defined vocabulary "
                f"({len(bundle.organisms)} organisms). Try a shorter or "
                "differently spelled search term.")
            ss["na_organism"] = None
        else:
            shown = matches[:25]
            picked = st.radio(
                f"{len(matches)} matching organism(s)"
                + (" — showing first 25" if len(matches) > 25 else ""),
                shown, index=0 if shown else None,
                key=f"organism_pick_{hash(query)}") if shown else None
            ss["na_organism"] = picked
        ss["na_sample_order_id"] = None
    else:
        samples = get_sample_orders()
        labels = [
            f"{r.organism}  ·  {r.specimen}  ·  {r.age}  ·  "
            f"{int(r.n_agents_tested)} agents tested  ·  "
            f"{int(r.n_nonsusceptible)} non-susceptible"
            for r in samples.itertuples()]
        pick = st.selectbox("Held-out culture order", range(len(labels)),
                            format_func=lambda i: labels[i])
        order_id = samples.iloc[pick]["order_proc_id_coded"]
        organisms_here = case_mod.organisms_in_order(order_id)
        chosen_organism = organisms_here[0]
        if len(organisms_here) > 1:
            chosen_organism = st.selectbox(
                f"This culture grew {len(organisms_here)} organisms — choose one",
                organisms_here)
        st.caption(
            "This case is a real, de-identified culture from the model's "
            "held-out test partition — patients the model never saw during "
            "training — shown for demonstration.")
        ss["na_organism"] = chosen_organism
        ss["na_sample_order_id"] = order_id

    if ss["na_organism"]:
        st.success(f"Organism: **{ss['na_organism']}**")

    c1, c2 = st.columns([1, 5])
    if c1.button("Next", type="primary", disabled=not ss["na_organism"]):
        _go(1)

# ===========================================================================
# STEP 2 — CLINICAL CONTEXT
# ===========================================================================
elif ss["na_step"] == 1:
    st.subheader("Clinical Context")
    st.caption(f"Organism: **{ss['na_organism']}**")

    if ss["na_mode"] == "sample":
        st.info(
            "Clinical context for this case comes directly from the real "
            "held-out culture record and does not require manual entry.")
        c1, c2 = st.columns(2)
        if c1.button("Back"):
            _go(0)
        if c2.button("Next", type="primary"):
            _go(2)
    else:
        st.markdown("##### Microbiology")
        c1, c2 = st.columns(2)
        specimen = c1.selectbox("Specimen type", case_mod.SPECIMENS,
                                help="Used by the model (`culture_description`).")
        morphotype = c2.selectbox(
            "Colony morphotype",
            ["standard", "mucoid", "non_mucoid", "small_colony_variant"],
            help="Used by the model (`organism_morphotype`).")
        today = datetime.now(timezone.utc)
        order_year = st.number_input(
            "Culture year", 2007, 2030, min(today.year, 2024),
            help="Used by the model (`order_year`).")

        st.markdown("##### Clinical History")
        st.caption("Only the history fields the deployed model actually uses "
                   "are collected here.")
        history_unknown = st.checkbox(
            "Patient history is unknown / unavailable",
            help="Entering 0 means 'confirmed no prior exposure', which is a "
                 "real and informative value. Tick this instead when the "
                 "history simply is not known — the model will fall back to "
                 "training-set defaults and the results page will list every "
                 "field it had to assume.")
        history = {}
        if not history_unknown:
            for key, label, lo, hi, default in case_mod.enterable_history(bundle):
                history[key] = st.number_input(label, lo, hi, default,
                                               key=f"hist_{key}")
        else:
            st.caption("History fields will be defaulted and reported as assumed.")

        st.markdown("##### Patient Context")
        c1, c2 = st.columns(2)
        gender_choice = c1.selectbox(
            "Recorded binary sex code", ["Not recorded", "0", "1"],
            help="Used by the model (`gender_code`). The source data "
                 "de-identifies sex to the codes 0 and 1 and provides no "
                 "legend — the mapping to sex is unknown and is never "
                 "asserted by this application.")
        setting = c2.selectbox(
            "Care setting", case_mod.SETTINGS,
            help="Reference only — not used by the resistance prediction "
                 "model.")
        age_band = st.selectbox(
            "Age band", case_mod.AGE_BANDS, index=5,
            help="Reference only — not used by the resistance prediction "
                 "model. No feature in the final model represents patient "
                 "age.")

        with st.expander("Laboratory results (14-day window) — reference only"):
            st.caption(
                "These values are not used by the resistance prediction "
                "model. They are collected only to give renal context on "
                "the Dosage reference in your results.")
            labs = {}
            for label, key in (("White cell count (x10^9/L)", "median_wbc"),
                               ("Creatinine (mg/dL)", "median_cr"),
                               ("Urea nitrogen (mg/dL)", "median_bun"),
                               ("Haemoglobin (g/dL)", "median_hgb"),
                               ("Platelets (x10^9/L)", "median_plt"),
                               ("Sodium (mmol/L)", "median_na"),
                               ("Lactate (mmol/L)", "median_lactate")):
                v = st.text_input(label, value="", key=f"lab_{key}")
                labs[key] = float(v) if v.strip() else None

        allergy_input = st.text_input(
            "Documented allergies (comma separated)", value="",
            placeholder="e.g. penicillin",
            help="Allergies are clinician-entered and are used only to raise "
                 "a flag. They never change a model prediction.")

        ss["na_ctx"] = dict(
            age_band=age_band, gender_code=(None if gender_choice == "Not recorded"
                                            else int(gender_choice)),
            specimen=specimen, setting=setting, morphotype=morphotype,
            order_year=int(order_year), labs=labs, history=history,
            allergies=[a.strip() for a in allergy_input.split(",") if a.strip()])

        c1, c2 = st.columns(2)
        if c1.button("Back"):
            _go(0)
        if c2.button("Next", type="primary"):
            _go(2)

# ===========================================================================
# STEP 3 — ANALYZE
# ===========================================================================
else:
    st.subheader("Analyze")
    st.markdown("**Clinical information validated**")
    st.write(f"- Organism: **{ss['na_organism']}**")
    if ss["na_mode"] == "sample":
        st.write("- Source: real de-identified held-out culture")
    else:
        ctx = ss.get("na_ctx", {})
        st.write(f"- Specimen: **{ctx.get('specimen', '—')}**")
        st.write(f"- Colony morphotype: **{ctx.get('morphotype', '—')}**")
        st.write(f"- Culture year: **{ctx.get('order_year', '—')}**")

    c1, c2 = st.columns(2)
    if c1.button("Back"):
        _go(1)

    if c2.button("Analyze Resistance Profile", type="primary"):
        with st.status("Analyzing Antimicrobial Resistance Profile",
                       expanded=True) as status:
            status.write("✓ Validating inputs")
            if ss["na_mode"] == "sample":
                record, actual_results = case_mod.sample_case(
                    ss["na_sample_order_id"], ss["na_organism"])
                assumed = []
                allergies = []
            else:
                ctx = ss["na_ctx"]
                record, assumed = case_mod.manual_record(
                    bundle, age_band=ctx["age_band"],
                    gender_code=ctx["gender_code"], specimen=ctx["specimen"],
                    setting=ctx["setting"], organism=ss["na_organism"],
                    morphotype=ctx["morphotype"], order_year=ctx["order_year"],
                    order_month=datetime.now(timezone.utc).month,
                    labs=ctx["labs"], history=ctx["history"])
                actual_results = None
                allergies = ctx["allergies"]
            status.write("✓ Preparing model features")

            result = MOD["ranking"].rank_antibiotics(
                dict(record), bundle, allergies=list(allergies))
            status.write("● Predicting resistance probabilities")
            options = pd.DataFrame(result["options"])
            status.write("○ Generating explainability analysis")
            status.update(label="Analysis complete", state="complete")

        ss["case_record"] = record
        ss["case_assumed"] = assumed
        ss["case_actual_results"] = actual_results
        ss["analysis_options"] = options
        ss["analysis_result"] = result
        ss["allergies"] = allergies
        st.switch_page("views/results.py")
