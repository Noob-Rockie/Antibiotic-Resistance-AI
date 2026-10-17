"""Execute the Streamlit application for real and assert on what it renders.

`AppTest` runs each script in-process, so a crash in any view, a missing
artefact or a broken column reference fails here rather than in front of a
user. The app is now a multi-page `st.navigation` app (Dashboard / New
Clinical Analysis / Results / Model Insights / How It Works), so each test
targets whichever file actually renders the content it checks, seeding
`session_state` with a REAL computed record + ranking result (built the
same way the app itself builds one) where a view depends on an
already-run analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
APP_DIR = PROJECT_ROOT / "11_APPLICATION"
sys.path.insert(0, str(APP_DIR))

from config.paths import MODELS_DIR  # noqa: E402

APP = APP_DIR / "app.py"
VIEWS = APP_DIR / "views"
MODEL = MODELS_DIR / "final_model.joblib"

requires_model = pytest.mark.skipif(
    not MODEL.exists(),
    reason="no trained model; run scripts/07_train_models.py")

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _blob(at) -> str:
    """Every bit of text an AppTest run rendered, across element kinds that
    are safe to introspect across Streamlit testing versions."""
    parts = []
    for attr in ("markdown", "caption", "title", "header", "subheader",
                "text", "info", "success", "warning", "error"):
        try:
            for el in getattr(at, attr):
                parts.append(str(getattr(el, "value", el)))
        except Exception:
            continue
    return " ".join(parts)


def _real_analysis():
    """A real record + real ranking result, built exactly as the app does,
    for seeding a view's session_state directly (bypassing the wizard UI)."""
    from components import case as case_mod
    from src.models.bundle import ModelBundle
    from src.utils.module_loader import load_stage_module

    bundle = ModelBundle.load()
    ranking = load_stage_module("08_RECOMMENDATION_ENGINE",
                                "antibiotic_ranking")
    samples = case_mod.list_sample_orders(6)
    order_id = samples.iloc[0]["order_proc_id_coded"]
    organisms_here = case_mod.organisms_in_order(order_id)
    record, actual_results = case_mod.sample_case(order_id, organisms_here[0])
    result = ranking.rank_antibiotics(dict(record), bundle, allergies=[])
    options = pd.DataFrame(result["options"])
    return record, actual_results, result, options


@pytest.fixture(scope="module")
def app():
    """The entry point — renders the default page (Dashboard)."""
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    return at


@pytest.fixture(scope="module")
def results_app():
    """The Results view, seeded with a real analysis (as if the wizard had
    just completed one) so its content actually renders."""
    record, actual_results, result, options = _real_analysis()
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=600)
    at.session_state["case_record"] = record
    at.session_state["case_assumed"] = []
    at.session_state["case_actual_results"] = actual_results
    at.session_state["analysis_options"] = options
    at.session_state["analysis_result"] = result
    at.session_state["allergies"] = []
    at.run()
    return at


@requires_model
def test_app_runs_without_exception(app):
    assert not app.exception, (
        "the application raised: "
        + "; ".join(str(e.value) for e in app.exception))


@requires_model
def test_results_view_runs_without_exception(results_app):
    assert not results_app.exception, (
        "the Results view raised: "
        + "; ".join(str(e.value) for e in results_app.exception))


@requires_model
def test_safety_disclaimer_is_rendered(app):
    blob = _blob(app).lower()
    assert "research" in blob and "prototype" in blob
    assert "does not replace qualified healthcare professionals" in blob
    assert "must be reviewed by an appropriately qualified healthcare" in blob


@requires_model
def test_no_prescriptive_language_is_rendered(app, results_app):
    blob = (_blob(app) + " " + _blob(results_app)).lower()
    for phrase in ("you should take", "we recommend prescribing",
                  "prescribe this", "the correct antibiotic is",
                  "best antibiotic"):
        assert phrase not in blob, f"prescriptive phrasing rendered: {phrase!r}"


@requires_model
def test_all_functional_areas_present(app, results_app):
    """Every capability the old 9-tab layout exposed must still exist
    somewhere in the redesigned navigation — nothing silently dropped."""
    blob = (_blob(app) + " " + _blob(results_app)).lower()
    for needle in ("dashboard", "prediction", "explanation",
                  "options for review", "dosage", "cost",
                  "patient history", "genomic", "clinician review"):
        assert needle in blob, f"functional area missing from redesign: {needle}"


@requires_model
def test_options_are_framed_for_clinician_review(results_app):
    blob = _blob(results_app).lower()
    assert "clinician review" in blob


@requires_model
def test_genomic_tab_reports_absence_honestly(results_app):
    blob = _blob(results_app).lower()
    errors = " ".join(str(e.value).lower() for e in results_app.error)
    assert ("no genomic data" in blob or "no genomic data" in errors), \
        "the genomic section must state plainly that no genomic data exists"


@requires_model
def test_unsupported_organism_is_reported_clearly():
    at = AppTest.from_file(str(VIEWS / "new_analysis.py"), default_timeout=600)
    at.run()
    search_boxes = [w for w in at.text_input
                   if "Search bacterial organism" in (w.label or "")]
    assert search_boxes, "organism search box not found on Step 1"
    search_boxes[0].set_value("Definitely Not A Real Organism XYZ").run()
    blob = _blob(at)
    assert "not supported by the current trained model" in blob


@requires_model
def test_model_insights_page_runs():
    at = AppTest.from_file(str(VIEWS / "model_insights.py"), default_timeout=600)
    at.run()
    assert not at.exception, (
        "Model Insights view raised: "
        + "; ".join(str(e.value) for e in at.exception))
    blob = _blob(at).lower()
    assert "limitations" in blob
    assert "out-of-scope" in blob or "not intended" in blob


def test_how_it_works_page_runs():
    at = AppTest.from_file(str(VIEWS / "how_it_works.py"), default_timeout=600)
    at.run()
    assert not at.exception, (
        "How It Works view raised: "
        + "; ".join(str(e.value) for e in at.exception))
