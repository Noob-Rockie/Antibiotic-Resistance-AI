"""EXP-02: LightGBM positive-class weighting.

Selection on the VALIDATION partition by PR-AUC; the selected candidate is then
evaluated once on the untouched test partition and on the temporal holdout.

Note on interpretation: for a probability model, class weighting and threshold
movement address the same trade-off. Weighting changes what the model fits;
thresholding changes only where the cut is placed. Both are reported so the two
mechanisms can be told apart.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from exp_common import Data, EXP_DIR, baseline, full_metrics, lgbm, log

WEIGHTS = [0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]
log("EXP-02  class-weight optimisation")
b = baseline(); d = Data(); g = d.encode_grouped()

rows = []
fitted = {}
for w in WEIGHTS:
    cw = None if w == 1.0 else {0: 1.0, 1: float(w)}
    t0 = time.time()
    m = lgbm(d.best_params, class_weight=cw)
    m.fit(g["Xtr"], g["ytr"])
    p_va = m.predict_proba(g["Xva"])[:, 1]
    mv = full_metrics(g["yva"], p_va, 0.5)
    mv.update(weight=w, fit_seconds=round(time.time() - t0, 1))
    rows.append(mv); fitted[w] = (m, p_va)
    log(f"  w={w:<5} VAL PR-AUC={mv['pr_auc']:.5f} ROC={mv['roc_auc']:.5f} "
        f"rec={mv['recall_sensitivity']:.5f} spec={mv['specificity']:.5f} "
        f"f1={mv['f1']:.5f} acc={mv['accuracy']:.5f} brier={mv['brier']:.5f}")

val = pd.DataFrame(rows)
val.to_csv(EXP_DIR / "exp_02_class_weight_validation.csv", index=False)
best_w = float(val.loc[val["pr_auc"].idxmax(), "weight"])
log(f"\n  best validation PR-AUC at weight={best_w} "
    f"(baseline w=1.00 -> {float(val.loc[val.weight==1.0,'pr_auc'].iloc[0]):.5f})")

# --- evaluate the selected candidate + the baseline weight on untouched data
t = d.encode_temporal()
out = {"weights": WEIGHTS, "validation": rows, "selected_weight": best_w,
       "applied": {}}
for w in sorted({1.0, best_w}):
    model, p_va = fitted[w]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        p_va, g["yva"])
    p_te = cal.predict(model.predict_proba(g["Xte"])[:, 1])
    mte = full_metrics(g["yte"], p_te, 0.5)
    cw = None if w == 1.0 else {0: 1.0, 1: float(w)}
    tm = lgbm(d.best_params, class_weight=cw); tm.fit(t["Xtr"], t["ytr"])
    mt = full_metrics(t["yte"], tm.predict_proba(t["Xte"])[:, 1], 0.5)
    out["applied"][f"w={w}"] = {"test": mte, "temporal": mt}
    log(f"  w={w:<5} TEST PR-AUC={mte['pr_auc']:.5f} rec={mte['recall_sensitivity']:.5f} "
        f"spec={mte['specificity']:.5f} f1={mte['f1']:.5f} | "
        f"TEMPORAL PR-AUC={mt['pr_auc']:.5f} rec={mt['recall_sensitivity']:.5f}")

json.dump(out, open(EXP_DIR / "exp_02_class_weight.json", "w"), indent=1)
log("EXP-02 done")
