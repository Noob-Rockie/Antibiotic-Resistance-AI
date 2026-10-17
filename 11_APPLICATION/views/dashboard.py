"""Dashboard — landing page and system overview."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components.resources import get_bundle  # noqa: E402
from components.ui import safety_banner, stat_card_html  # noqa: E402

bundle = get_bundle()

st.markdown('<div class="cds-eyebrow">Dashboard</div>', unsafe_allow_html=True)
st.title("Antimicrobial Resistance")
st.caption("Clinical Decision Support · AI-Assisted Resistance Prediction")

c1, c2 = st.columns([1, 4])
with c1:
    if st.button("Start New Analysis", type="primary", use_container_width=True):
        st.switch_page("views/new_analysis.py")

st.write("")
st.markdown('<div class="cds-eyebrow">System overview</div>', unsafe_allow_html=True)
cols = st.columns(4)
cols[0].markdown(stat_card_html("ML Model", bundle.metadata.get("algorithm", "—")),
                  unsafe_allow_html=True)
cols[1].markdown(stat_card_html("Explainability", "SHAP Enabled",
                                "TreeSHAP, per-prediction"),
                  unsafe_allow_html=True)
cols[2].markdown(stat_card_html("Supported Organisms", f"{len(bundle.organisms)}",
                                "Dataset-defined vocabulary"),
                  unsafe_allow_html=True)
cols[3].markdown(stat_card_html("Model Status", "Ready",
                                f"{len(bundle.antibiotics)} antimicrobial agents"),
                  unsafe_allow_html=True)

st.write("")
st.markdown('<div class="cds-eyebrow">How it works</div>', unsafe_allow_html=True)
steps = [
    "Select bacterial organism",
    "Enter required clinical context",
    "Analyze resistance profile",
    "Review predicted resistance probabilities",
    "Review ranked antimicrobial options",
    "Review AI explanation",
    "Make clinician-led decision",
]
for i, s in enumerate(steps, start=1):
    st.markdown(f"**{i}.** {s}")

st.divider()
safety_banner()
