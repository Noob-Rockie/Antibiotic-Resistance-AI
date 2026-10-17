"""EXP-00: reproduce the production baseline to validate the harness.

If this does not land on the published test PR-AUC of 0.76166 then every later
comparison is meaningless, so nothing else runs until this matches.
"""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.isotonic import IsotonicRegression
from exp_common import Data, EXP_DIR, baseline, full_metrics, lgbm, log

log("EXP-00  reproducing the production baseline")
b = baseline()
log(f"  published test     PR-AUC={b['test_metrics']['pr_auc']}  "
    f"acc={b['test_metrics']['accuracy']}  recall={b['test_metrics']['recall_sensitivity']}")
log(f"  published temporal PR-AUC={b['temporal_metrics']['pr_auc']}  "
    f"recall={b['temporal_metrics']['recall_sensitivity']}")

d = Data()
log(f"  hyperparameters from the production bundle: {d.best_params}")
g = d.encode_grouped()
log(f"  partitions: train={len(g['ytr']):,}  val={len(g['yva']):,}  test={len(g['yte']):,}")

t0 = time.time()
model = lgbm(d.best_params)
model.fit(g["Xtr"], g["ytr"])
fit_s = time.time() - t0
log(f"  fitted in {fit_s:.1f}s")

p_val = model.predict_proba(g["Xva"])[:, 1]
calib = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_val, g["yva"])
p_te = calib.predict(model.predict_proba(g["Xte"])[:, 1])
m_te = full_metrics(g["yte"], p_te, 0.5)
log(f"  REPRODUCED test    PR-AUC={m_te['pr_auc']}  acc={m_te['accuracy']}  "
    f"recall={m_te['recall_sensitivity']}  spec={m_te['specificity']}")

t = d.encode_temporal()
t0 = time.time()
tm = lgbm(d.best_params); tm.fit(t["Xtr"], t["ytr"])
p_t = tm.predict_proba(t["Xte"])[:, 1]          # baseline temporal is UNcalibrated
m_t = full_metrics(t["yte"], p_t, 0.5)
log(f"  REPRODUCED temporal PR-AUC={m_t['pr_auc']}  acc={m_t['accuracy']}  "
    f"recall={m_t['recall_sensitivity']}  ({time.time()-t0:.1f}s)")

d_pr = abs(m_te["pr_auc"] - b["test_metrics"]["pr_auc"])
d_tpr = abs(m_t["pr_auc"] - b["temporal_metrics"]["pr_auc"])
log(f"\n  |delta| test PR-AUC = {d_pr:.5f}   |delta| temporal PR-AUC = {d_tpr:.5f}")
ok = d_pr < 0.002 and d_tpr < 0.002
log(f"  HARNESS VALID: {ok}")

json.dump({"published_test": b["test_metrics"], "reproduced_test": m_te,
           "published_temporal": b["temporal_metrics"], "reproduced_temporal": m_t,
           "delta_test_pr_auc": round(d_pr, 6),
           "delta_temporal_pr_auc": round(d_tpr, 6),
           "harness_valid": bool(ok), "fit_seconds": round(fit_s, 1)},
          open(EXP_DIR / "exp_00_baseline.json", "w"), indent=1)
np.save(EXP_DIR / "_p_test_baseline.npy", p_te)
np.save(EXP_DIR / "_p_temporal_baseline.npy", p_t)
log("  wrote exp_00_baseline.json")
