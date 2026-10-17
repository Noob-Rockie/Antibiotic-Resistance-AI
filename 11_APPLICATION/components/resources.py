"""Cached model/module loading shared by app.py and every view.

Kept out of app.py itself because Streamlit executes each `st.Page` target
as its own script; a plain module import here (rather than re-running
app.py) is what lets every view share the same cached resources.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from src.models.bundle import ModelBundle             # noqa: E402
from src.utils.module_loader import load_stage_module  # noqa: E402


@st.cache_resource(show_spinner="Loading the trained model …")
def get_bundle():
    return ModelBundle.load()


@st.cache_resource(show_spinner=False)
def get_modules():
    return {
        "ranking": load_stage_module("08_RECOMMENDATION_ENGINE",
                                     "antibiotic_ranking"),
        "dosage": load_stage_module("08_RECOMMENDATION_ENGINE",
                                    "dosage_reference"),
        "cost": load_stage_module("08_RECOMMENDATION_ENGINE", "cost_service"),
        "genomic": load_stage_module("09_GENOMIC_AMR", "genomic_analysis"),
    }


@st.cache_data(show_spinner="Loading held-out example cases …")
def get_sample_orders():
    from components import case as case_mod
    return case_mod.list_sample_orders(60)


@st.cache_data(show_spinner=False)
def cached_dosage(antibiotic: str, creatinine):
    return get_modules()["dosage"].get_dosage_reference(
        antibiotic, {"median_cr": creatinine})


@st.cache_data(show_spinner=False)
def cached_cost(antibiotic: str):
    return get_modules()["cost"].get_cost(antibiotic)
