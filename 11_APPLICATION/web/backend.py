"""FastAPI backend for the Antibiotic Resistance CDS web frontend.

This is a thin JSON wrapper around the exact same backend the Streamlit
app uses — the trained LightGBM bundle, the recommendation engine, SHAP,
dosage/cost services and the genomic module. No prediction, ranking, or
explanation logic lives here; every number returned comes from calling
straight into `05_MODEL_TRAINING` / `08_RECOMMENDATION_ENGINE` /
`09_GENOMIC_AMR` / `src`, unchanged.

Run:  .venv/Scripts/python -m uvicorn 11_APPLICATION.web.backend:app --port 8000
  or: .venv/Scripts/python 11_APPLICATION/web/backend.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

WEB_DIR = Path(__file__).resolve().parent
APP_DIR = WEB_DIR.parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components import case as case_mod                    # noqa: E402
from components.organisms import search_organisms           # noqa: E402
from src.models.bundle import ModelBundle                   # noqa: E402
from src.utils.module_loader import load_stage_module        # noqa: E402


# ---------------------------------------------------------------------------
# Model / module load — once, at process start
# ---------------------------------------------------------------------------
try:
    BUNDLE = ModelBundle.load()
    MODEL_ERROR: str | None = None
except FileNotFoundError as exc:
    BUNDLE = None
    MODEL_ERROR = str(exc)

MOD = {} if BUNDLE is None else {
    "ranking": load_stage_module("08_RECOMMENDATION_ENGINE", "antibiotic_ranking"),
    "dosage": load_stage_module("08_RECOMMENDATION_ENGINE", "dosage_reference"),
    "cost": load_stage_module("08_RECOMMENDATION_ENGINE", "cost_service"),
    "genomic": load_stage_module("09_GENOMIC_AMR", "genomic_analysis"),
}

_SAMPLE_ORDERS: pd.DataFrame | None = None
_DOSAGE_CACHE: dict = {}
_COST_CACHE: dict = {}

# ---------------------------------------------------------------------------
# MongoDB — analyses and clinician reviews persist here so they survive a
# restart / redeploy on a host with an ephemeral filesystem (e.g. Railway).
# Falls back to an in-memory dict / local CSV when no database is configured,
# so local development still works without MongoDB running.
# ---------------------------------------------------------------------------
MONGODB_URI = (os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URL")
              or os.environ.get("MONGO_PUBLIC_URL"))
_DB = None
if MONGODB_URI and MongoClient is not None:
    try:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        _DB = _client["antibiotic_resistance_cds"]
    except Exception:
        _DB = None

_ANALYSES_COL = _DB["analyses"] if _DB is not None else None
_REVIEWS_COL = _DB["clinician_reviews"] if _DB is not None else None
_ANALYSES: dict[str, dict] = {}

DEMO_GENOMIC_MARKERS = ["blaKPC-2", "blaNDM-1", "blaCTX-M-15", "mecA", "vanA",
                        "ermB", "qnrS1", "mcr-1", "gyrA"]


def _sample_orders() -> pd.DataFrame:
    global _SAMPLE_ORDERS
    if _SAMPLE_ORDERS is None:
        _SAMPLE_ORDERS = case_mod.list_sample_orders(60)
    return _SAMPLE_ORDERS


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to native JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj


def require_model() -> None:
    if BUNDLE is None:
        raise HTTPException(
            503,
            {"message": "Prediction could not be completed.",
             "reason": f"The trained model could not be loaded ({MODEL_ERROR}).",
             "recommended_action": ("Run `python scripts/07_train_models.py` "
                                    "to produce "
                                    "05_MODEL_TRAINING/models/final_model.joblib, "
                                    "then restart this server.")})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Antibiotic Resistance CDS API")

CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")
                if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "mongo_connected": _DB is not None}


@app.get("/api/model-status")
def model_status():
    if BUNDLE is None:
        return {"ready": False, "reason": MODEL_ERROR}
    return {
        "ready": True,
        "algorithm": BUNDLE.metadata.get("algorithm"),
        "n_organisms": len(BUNDLE.organisms),
        "n_antibiotics": len(BUNDLE.antibiotics),
        "n_features": len(BUNDLE.features),
        "n_train_rows": BUNDLE.metadata.get("n_train_rows"),
        "n_train_patients": BUNDLE.metadata.get("n_train_patients"),
        "target_definition": BUNDLE.metadata.get("target_definition"),
        "decision_threshold": BUNDLE.threshold,
    }


@app.get("/api/organisms")
def organisms(q: str = ""):
    require_model()
    matches = search_organisms(BUNDLE, q)
    return {"matches": matches[:50], "total": len(matches)}


@app.get("/api/history-fields")
def history_fields():
    require_model()
    return {"fields": [
        {"key": k, "label": label, "min": lo, "max": hi, "default": default}
        for k, label, lo, hi, default in case_mod.enterable_history(BUNDLE)]}


@app.get("/api/specimens")
def specimens():
    return {"specimens": case_mod.SPECIMENS, "settings": case_mod.SETTINGS,
            "morphotypes": ["standard", "mucoid", "non_mucoid",
                           "small_colony_variant"],
            "age_bands": case_mod.AGE_BANDS}


@app.get("/api/sample-orders")
def sample_orders():
    require_model()
    df = _sample_orders()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "order_id": str(r["order_proc_id_coded"]),
            "organism": r["organism"], "specimen": r["specimen"],
            "setting": r["setting"], "age": r["age"],
            "n_agents_tested": int(r["n_agents_tested"]),
            "n_nonsusceptible": int(r["n_nonsusceptible"]),
        })
    return _sanitize({"orders": rows})


@app.get("/api/sample-orders/{order_id}/organisms")
def sample_order_organisms(order_id: str):
    require_model()
    return {"organisms": case_mod.organisms_in_order(order_id)}


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------
class ManualAnalyzeRequest(BaseModel):
    organism: str
    specimen: str
    morphotype: str = "standard"
    order_year: int = datetime.now(timezone.utc).year
    gender_code: int | None = None
    setting: str = "Inpatient"
    age_band: str = case_mod.AGE_BANDS[5]
    history: dict[str, float] = Field(default_factory=dict)
    history_unknown: bool = False
    labs: dict[str, float | None] = Field(default_factory=dict)
    allergies: list[str] = Field(default_factory=list)


class SampleAnalyzeRequest(BaseModel):
    order_id: str
    organism: str


def _options_payload(options_records: list[dict]) -> list[dict]:
    return _sanitize(options_records)


def _store_analysis(record: dict, actual_results: pd.DataFrame | None,
                    result: dict, assumed: list, allergies: list) -> str:
    analysis_id = uuid.uuid4().hex
    doc = {
        "_id": analysis_id,
        "record": _sanitize(dict(record)),
        "actual_results": (_sanitize(actual_results.to_dict("records"))
                          if actual_results is not None else None),
        "result": _sanitize(result),
        "assumed": assumed,
        "allergies": allergies,
        "created_at": datetime.now(timezone.utc),
    }
    if _ANALYSES_COL is not None:
        _ANALYSES_COL.insert_one(doc)
    else:
        _ANALYSES[analysis_id] = doc
    return analysis_id


def _analysis_response(analysis_id: str) -> dict:
    a = _ANALYSES[analysis_id]
    record, result = a["record"], a["result"]
    return _sanitize({
        "analysis_id": analysis_id,
        "organism": record.get("organism_clean"),
        "specimen": record.get("culture_description"),
        "assumed_fields": a["assumed"],
        "n_options": result.get("n_options"),
        "n_with_critical_flag": result.get("n_with_critical_flag"),
        "framing": result.get("framing"),
        "safety_notice": result.get("safety_notice"),
        "model": result.get("model"),
        "has_ground_truth": a["actual_results"] is not None,
        "options": _options_payload(result.get("options", [])),
    })


@app.post("/api/analyze")
def analyze(req: ManualAnalyzeRequest):
    require_model()
    if req.organism not in set(BUNDLE.organisms):
        raise HTTPException(400, {
            "message": "This organism is not supported by the current "
                      "trained model.",
            "recommended_action": "Select an organism from the supported, "
                                  "dataset-defined vocabulary."})
    history = {} if req.history_unknown else req.history
    record, assumed = case_mod.manual_record(
        BUNDLE, age_band=req.age_band, gender_code=req.gender_code,
        specimen=req.specimen, setting=req.setting, organism=req.organism,
        morphotype=req.morphotype, order_year=req.order_year,
        order_month=datetime.now(timezone.utc).month,
        labs=req.labs, history=history)
    result = MOD["ranking"].rank_antibiotics(
        dict(record), BUNDLE, allergies=list(req.allergies))
    analysis_id = _store_analysis(record, None, result, assumed,
                                  list(req.allergies))
    return _analysis_response(analysis_id)


@app.post("/api/analyze-sample")
def analyze_sample(req: SampleAnalyzeRequest):
    require_model()
    try:
        record, actual_results = case_mod.sample_case(req.order_id, req.organism)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    result = MOD["ranking"].rank_antibiotics(dict(record), BUNDLE, allergies=[])
    analysis_id = _store_analysis(record, actual_results, result, [], [])
    return _analysis_response(analysis_id)


def _get_analysis(analysis_id: str) -> dict:
    if _ANALYSES_COL is not None:
        a = _ANALYSES_COL.find_one({"_id": analysis_id})
    else:
        a = _ANALYSES.get(analysis_id)
    if a is None:
        raise HTTPException(404, "Unknown or expired analysis_id. Start a "
                                 "new analysis.")
    return a


@app.get("/api/analysis/{analysis_id}/dosage")
def analysis_dosage(analysis_id: str, antibiotic: str):
    require_model()
    a = _get_analysis(analysis_id)
    record = a["record"]
    cr = record.get("median_cr")
    cr = None if cr is None or (isinstance(cr, float) and np.isnan(cr)) else float(cr)
    key = (antibiotic, cr)
    if key not in _DOSAGE_CACHE:
        _DOSAGE_CACHE[key] = MOD["dosage"].get_dosage_reference(
            antibiotic, {"median_cr": cr})
    return _sanitize(_DOSAGE_CACHE[key])


@app.get("/api/analysis/{analysis_id}/cost")
def analysis_cost(analysis_id: str, antibiotic: str):
    require_model()
    _get_analysis(analysis_id)
    if antibiotic not in _COST_CACHE:
        _COST_CACHE[antibiotic] = MOD["cost"].get_cost(antibiotic)
    return _sanitize(_COST_CACHE[antibiotic])


@app.get("/api/analysis/{analysis_id}/patient-history")
def analysis_patient_history(analysis_id: str):
    a = _get_analysis(analysis_id)
    record = a["record"]
    history = case_mod.patient_history_view(record).to_dict("records")
    labs = case_mod.labs_view(record).to_dict("records")
    return {"history": _sanitize(history), "labs": _sanitize(labs)}


@app.get("/api/analysis/{analysis_id}/genomic")
def analysis_genomic(analysis_id: str):
    require_model()
    a = _get_analysis(analysis_id)
    status = MOD["genomic"].availability_report()
    if not status["available"]:
        return {"available": False, "reason": status["reason"],
               "how_to_enable": status["how_to_enable"],
               "expected_path": status["expected_path"],
               "reference_map_size": status["reference_map_size"],
               "reference_note": status["reference_note"],
               "demo_markers": DEMO_GENOMIC_MARKERS}
    markers = MOD["genomic"].markers_for_order(
        a["record"].get("order_proc_id_coded"))
    return {"available": True, "markers": _sanitize(markers)}


@app.post("/api/genomic/annotate")
def genomic_annotate(body: dict):
    require_model()
    markers = body.get("markers", [])
    return {"markers": _sanitize(MOD["genomic"].annotate_markers(markers))}


@app.get("/api/analysis/{analysis_id}/ground-truth")
def analysis_ground_truth(analysis_id: str):
    a = _get_analysis(analysis_id)
    if a["actual_results"] is None:
        raise HTTPException(404, "This analysis has no held-out ground truth "
                                 "(it was built from manual entry).")
    actual = pd.DataFrame(a["actual_results"])
    options = pd.DataFrame(a["result"]["options"])
    merged = actual.merge(
        options[["antibiotic", "p_susceptible", "predicted_label"]],
        on="antibiotic", how="left")
    merged["correct"] = np.where(
        merged["predicted_label"].isna(), "unknown",
        np.where((merged["non_susceptible"] == 1) ==
                (merged["predicted_label"] == "Non-susceptible"),
                "agree", "disagree"))
    n_ok = int((merged["correct"] == "agree").sum())
    n_tot = int((merged["correct"] != "unknown").sum())
    return {
        "rows": _sanitize(merged.to_dict("records")),
        "agreement": {"n_agree": n_ok, "n_total": n_tot,
                     "rate": (n_ok / n_tot) if n_tot else None},
    }


class ClinicianReviewRequest(BaseModel):
    reviewer: str
    decision: str
    selected_agents: list[str] = Field(default_factory=list)
    notes: str = ""
    confirm: bool


@app.post("/api/analysis/{analysis_id}/clinician-review")
def clinician_review(analysis_id: str, req: ClinicianReviewRequest):
    a = _get_analysis(analysis_id)
    if not req.confirm:
        raise HTTPException(400, "The confirmation checkbox is required "
                                 "before a review can be recorded.")
    if not req.reviewer.strip():
        raise HTTPException(400, "A reviewer identifier is required.")
    record, result = a["record"], a["result"]
    options = result.get("options", [])
    entry = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reviewer": req.reviewer.strip(),
        "decision": req.decision,
        "clinician_selected_agents": "; ".join(req.selected_agents),
        "clinician_notes": req.notes.strip(),
        "organism": record.get("organism_clean"),
        "specimen": record.get("culture_description"),
        "order_proc_id_coded": record.get("order_proc_id_coded"),
        "model_algorithm": BUNDLE.metadata.get("algorithm"),
        "model_top_option": options[0]["antibiotic"] if options else None,
        "model_top_p_susceptible": options[0]["p_susceptible"] if options else None,
    }
    if _REVIEWS_COL is not None:
        _REVIEWS_COL.insert_one(dict(entry))
        return {"recorded": True, "storage": "mongodb"}
    out = PROJECT_ROOT / "13_RESULTS" / "clinician_reviews.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([entry]).to_csv(out, mode="a", header=not out.exists(), index=False)
    return {"recorded": True, "storage": "csv", "path": str(out)}


# ---------------------------------------------------------------------------
# Model insights
# ---------------------------------------------------------------------------
@app.get("/api/model-insights")
def model_insights():
    require_model()
    meta = BUNDLE.metadata
    metrics = BUNDLE.metrics

    family = []
    exp_log = PROJECT_ROOT / "05_MODEL_TRAINING" / "experiments" / "experiment_log.csv"
    if exp_log.exists():
        df = pd.read_csv(exp_log)
        fam = df[df.get("notes") == "model family comparison"]
        if not fam.empty:
            cols = ["algorithm", "accuracy", "precision", "recall_sensitivity",
                    "specificity", "f1", "roc_auc", "pr_auc", "brier",
                    "train_seconds"]
            family = _sanitize(
                fam[cols].sort_values("pr_auc", ascending=False)
                .to_dict("records"))

    three_class = None
    eval_report = PROJECT_ROOT / "06_MODEL_EVALUATION" / "evaluation_report.json"
    if eval_report.exists():
        three_class = json.loads(eval_report.read_text(encoding="utf-8")).get(
            "three_class")

    uat = None
    uat_path = PROJECT_ROOT / "12_TESTS" / "uat_report.json"
    if uat_path.exists():
        uat = json.loads(uat_path.read_text(encoding="utf-8"))

    subgroups = {}
    sub_dir = PROJECT_ROOT / "06_MODEL_EVALUATION" / "tables"
    for col in ("age", "gender_code", "ordering_mode", "culture_description",
               "organism_group"):
        p = sub_dir / f"subgroup_{col}.csv"
        if p.exists():
            subgroups[col] = _sanitize(pd.read_csv(p).to_dict("records"))

    per_antibiotic = None
    p = sub_dir / "per_antibiotic_performance.csv"
    if p.exists():
        per_antibiotic = _sanitize(pd.read_csv(p).to_dict("records"))

    return _sanitize({
        "meta": meta,
        "held_out_test": metrics.get("test"),
        "validation": metrics.get("validation"),
        "temporal_test": metrics.get("temporal_test"),
        "three_class": three_class,
        "uat": uat,
        "model_family_comparison": family,
        "subgroups": subgroups,
        "per_antibiotic": per_antibiotic,
    })


@app.get("/api/methodology")
def methodology():
    def load_json(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    audit = load_json(PROJECT_ROOT / "02_DATA_ANALYSIS" / "reports" /
                      "dataset_audit.json")
    build = load_json(PROJECT_ROOT / "01_DATA" / "processed" / "build_report.json")
    leak = load_json(PROJECT_ROOT / "02_DATA_ANALYSIS" / "reports" / "leakage.json")
    docs = []
    docs_dir = PROJECT_ROOT / "docs"
    if docs_dir.exists():
        for d in sorted(docs_dir.glob("*.md")):
            docs.append({"name": d.name, "content": d.read_text(encoding="utf-8")})
    return {"audit": audit, "build": build, "leakage": leak, "docs": docs}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
STATIC_DIR = WEB_DIR / "static"


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
