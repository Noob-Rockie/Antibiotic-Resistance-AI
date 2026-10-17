"""Antibiotic Resistance Clinical Decision Support — research prototype.

Run:  streamlit run 11_APPLICATION/app.py

Design stance
-------------
The application is organised around one isolate at a time. A clinician
starts a new analysis, identifies an organism, supplies clinical context,
and reviews a ranked report built entirely from real model output.

Nothing here generates a number. Predictions come from the trained model,
explanations from SHAP on that model, dosage text verbatim from the FDA
label, prices from live CMS NADAC records, and guardrails from cited
references. Where a value is unavailable the application says so rather
than filling the gap.

This file only bootstraps the model/module cache and wires up navigation.
Page content lives in `views/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components.resources import get_bundle, get_modules  # noqa: E402
from components.ui import inject_css, nav_status, page_config  # noqa: E402

page_config("Antibiotic Resistance CDS")
inject_css()

try:
    bundle = get_bundle()
    model_ready = True
except FileNotFoundError as exc:
    bundle = None
    model_ready = False
    model_error = str(exc)

if model_ready:
    MOD = get_modules()

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="cds-brand-title">Antimicrobial Resistance</div>'
    '<div class="cds-brand-sub">Clinical Decision Support</div>',
    unsafe_allow_html=True)

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon="📊", default=True),
    st.Page("views/new_analysis.py", title="New Clinical Analysis", icon="🩺"),
    st.Page("views/results.py", title="Results", icon="📋"),
    st.Page("views/model_insights.py", title="Model Insights", icon="📈"),
    st.Page("views/how_it_works.py", title="How It Works", icon="ℹ️"),
]
nav = st.navigation(pages, position="sidebar")

if not model_ready:
    st.error(
        "**Prediction could not be completed.**\n\n"
        f"Reason: the trained model could not be loaded ({model_error}).\n\n"
        "Recommended action: run `python scripts/07_train_models.py` to "
        "produce `05_MODEL_TRAINING/models/final_model.joblib`, then reload "
        "this application.")
    st.stop()

nav.run()

st.sidebar.divider()
st.sidebar.markdown(nav_status(model_ready), unsafe_allow_html=True)
