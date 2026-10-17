"""Save the candidate as a SEPARATE bundle. The production model is untouched.

The candidate is the SAME fitted model, encoder and calibrator as production.
The only difference is the decision threshold: 0.50 -> 0.35, selected on a
held-out half of the validation partition. Nothing is retrained, so the
candidate cannot differ from production in any way except where the cut falls.
"""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
from exp_common import EXP_DIR, FINAL_MODEL, baseline, log
from src.models.bundle import ModelBundle

CAND = FINAL_MODEL.parent / "candidate_improved_model.joblib"
NEW_THRESHOLD = 0.35

prod_hash_before = FINAL_MODEL.stat().st_mtime, FINAL_MODEL.stat().st_size
b = ModelBundle.load(FINAL_MODEL)
log(f"loaded production bundle: threshold={b.threshold} algo={b.metadata.get('algorithm')}")

e1 = json.loads((EXP_DIR / "exp_01_threshold.json").read_text(encoding="utf-8"))
a = e1["applied"]["max_f1"]
base = baseline()

b.threshold = NEW_THRESHOLD
b.metrics["test_at_candidate_threshold"] = a["test"]
b.metrics["temporal_at_candidate_threshold"] = a["temporal"]
b.metrics["test_at_production_threshold"] = base["test_metrics"]
b.metrics["temporal_at_production_threshold"] = base["temporal_metrics"]
b.metadata["candidate"] = True
b.metadata["promoted_to_production"] = False
b.metadata["candidate_change"] = (
    "Decision threshold 0.50 -> 0.35. The fitted model, encoder, feature list "
    "and isotonic calibrator are byte-identical to production; only the "
    "operating point differs. Selected on a patient-disjoint half of the "
    "validation partition (max F1), never on test.")
b.metadata["candidate_experiment"] = "05_MODEL_TRAINING/experiments/model_improvement"
b.save(CAND)
log(f"saved candidate -> {CAND}")

prod_hash_after = FINAL_MODEL.stat().st_mtime, FINAL_MODEL.stat().st_size
assert prod_hash_before == prod_hash_after, "PRODUCTION MODEL WAS MODIFIED"
log(f"production model untouched: mtime/size unchanged {prod_hash_after}")

reloaded = ModelBundle.load(CAND)
log(f"reloaded candidate: threshold={reloaded.threshold} "
    f"features={len(reloaded.features)} candidate={reloaded.metadata.get('candidate')} "
    f"promoted={reloaded.metadata.get('promoted_to_production')}")
log("\nCandidate model found - NOT promoted to production.")
