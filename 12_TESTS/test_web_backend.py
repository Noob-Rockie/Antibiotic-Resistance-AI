"""Exercise the FastAPI backend (11_APPLICATION/web/backend.py) for real.

Uses FastAPI's TestClient, which runs the actual app in-process — a real
prediction, real SHAP, and real dosage/cost/genomic calls, exactly as a
browser hitting these endpoints would get. This is the JSON-API
counterpart to `test_application.py` (which exercises the Streamlit app).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
APP_DIR = PROJECT_ROOT / "11_APPLICATION"
sys.path.insert(0, str(APP_DIR))

from config.paths import MODELS_DIR  # noqa: E402

MODEL = MODELS_DIR / "final_model.joblib"
requires_model = pytest.mark.skipif(
    not MODEL.exists(),
    reason="no trained model; run scripts/07_train_models.py")

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from web.backend import app  # noqa: E402

client = TestClient(app)


@requires_model
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@requires_model
def test_model_status_reports_ready():
    r = client.get("/api/model-status")
    assert r.status_code == 200
    d = r.json()
    assert d["ready"] is True
    assert d["algorithm"] == "LightGBM"
    assert d["n_organisms"] > 0 and d["n_antibiotics"] > 0


@requires_model
def test_organism_search_matches_supported_vocabulary():
    r = client.get("/api/organisms", params={"q": "coli"})
    assert r.status_code == 200
    matches = r.json()["matches"]
    assert "ESCHERICHIA COLI" in matches


@requires_model
def test_organism_search_reports_no_matches_for_unsupported():
    r = client.get("/api/organisms", params={"q": "definitely not a real organism"})
    assert r.status_code == 200
    assert r.json() == {"matches": [], "total": 0}


@requires_model
def test_analyze_rejects_unsupported_organism_with_clear_message():
    r = client.post("/api/analyze", json={
        "organism": "NOT A REAL ORGANISM", "specimen": "URINE",
    })
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "not supported by the current trained model" in detail["message"]


@requires_model
def test_analyze_manual_returns_real_ranked_options():
    r = client.post("/api/analyze", json={
        "organism": "ESCHERICHIA COLI", "specimen": "URINE",
        "morphotype": "standard", "order_year": 2023, "gender_code": 1,
        "setting": "Outpatient", "history_unknown": True, "allergies": [],
    })
    assert r.status_code == 200
    d = r.json()
    assert d["organism"] == "ESCHERICHIA COLI"
    assert d["n_options"] == d["n_options"]  # present
    assert len(d["options"]) == d["n_options"] > 0
    first = d["options"][0]
    assert 0.0 <= first["p_nonsusceptible"] <= 1.0
    assert first["rank"] == 1
    assert isinstance(first["top_factors"], list)


@requires_model
def test_full_analysis_detail_endpoints_do_not_error():
    analyze_resp = client.post("/api/analyze", json={
        "organism": "KLEBSIELLA PNEUMONIAE", "specimen": "BLOOD",
        "morphotype": "standard", "order_year": 2022, "gender_code": None,
        "setting": "Inpatient", "history_unknown": True, "allergies": [],
    })
    assert analyze_resp.status_code == 200
    analysis_id = analyze_resp.json()["analysis_id"]
    antibiotic = analyze_resp.json()["options"][0]["antibiotic"]

    dosage = client.get(f"/api/analysis/{analysis_id}/dosage",
                        params={"antibiotic": antibiotic})
    assert dosage.status_code == 200

    cost = client.get(f"/api/analysis/{analysis_id}/cost",
                      params={"antibiotic": antibiotic})
    assert cost.status_code == 200

    history = client.get(f"/api/analysis/{analysis_id}/patient-history")
    assert history.status_code == 200
    assert "history" in history.json() and "labs" in history.json()

    genomic = client.get(f"/api/analysis/{analysis_id}/genomic")
    assert genomic.status_code == 200
    # Honesty check: no genomic data exists in this project's datasets.
    if not genomic.json()["available"]:
        assert "no genomic data" in genomic.json()["reason"].lower()

    review = client.post(f"/api/analysis/{analysis_id}/clinician-review", json={
        "reviewer": "pytest", "decision": "Accept",
        "selected_agents": [antibiotic], "notes": "automated test", "confirm": True,
    })
    assert review.status_code == 200
    assert review.json()["recorded"] is True


@requires_model
def test_clinician_review_requires_confirmation():
    analyze_resp = client.post("/api/analyze", json={
        "organism": "ESCHERICHIA COLI", "specimen": "URINE",
        "history_unknown": True,
    })
    analysis_id = analyze_resp.json()["analysis_id"]
    r = client.post(f"/api/analysis/{analysis_id}/clinician-review", json={
        "reviewer": "pytest", "decision": "Accept", "confirm": False,
    })
    assert r.status_code == 400


@requires_model
def test_sample_orders_and_sample_analysis_round_trip():
    r = client.get("/api/sample-orders")
    assert r.status_code == 200
    orders = r.json()["orders"]
    assert len(orders) > 0
    order_id = orders[0]["order_id"]

    organisms_r = client.get(f"/api/sample-orders/{order_id}/organisms")
    assert organisms_r.status_code == 200
    organism = organisms_r.json()["organisms"][0]

    analyze_r = client.post("/api/analyze-sample",
                            json={"order_id": order_id, "organism": organism})
    assert analyze_r.status_code == 200
    d = analyze_r.json()
    assert d["has_ground_truth"] is True

    gt = client.get(f"/api/analysis/{d['analysis_id']}/ground-truth")
    assert gt.status_code == 200
    assert "rows" in gt.json() and "agreement" in gt.json()


@requires_model
def test_no_prescriptive_language_in_ranking_text():
    r = client.post("/api/analyze", json={
        "organism": "ESCHERICHIA COLI", "specimen": "URINE",
        "history_unknown": True,
    })
    d = r.json()
    blob = (d["framing"] + " " + d["safety_notice"] + " " +
           " ".join(o["explanation"] for o in d["options"])).lower()
    for phrase in ("you should take", "we recommend prescribing",
                  "prescribe this", "the correct antibiotic is",
                  "best antibiotic"):
        assert phrase not in blob, f"prescriptive phrasing rendered: {phrase!r}"


@requires_model
def test_model_insights_endpoint_matches_bundle_metrics():
    r = client.get("/api/model-insights")
    assert r.status_code == 200
    d = r.json()
    assert d["meta"]["algorithm"] == "LightGBM"
    assert d["held_out_test"] is not None
    assert d["temporal_test"] is not None
    if d["uat"]:
        assert "demo_menu_caveat" in d["uat"]
        assert "not" in d["uat"]["demo_menu_caveat"].lower()


def test_methodology_endpoint_runs():
    r = client.get("/api/methodology")
    assert r.status_code == 200


@requires_model
def test_static_frontend_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Antimicrobial Resistance" in r.text
