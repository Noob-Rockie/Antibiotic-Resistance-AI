"""EXP-07: calibration.

Calibration changes how well probabilities match observed frequencies. It is
NOT a classification-accuracy technique, and is reported separately from
discrimination for that reason.

Protocol: the calibrator is fitted on the VALIDATION partition (the production
protocol) and assessed on the untouched test partition, where the fit is honest.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from exp_common import Data, EXP_DIR, FIG_DIR, baseline, full_metrics, lgbm, log

log("EXP-07  calibration")
b = baseline(); d = Data(); g = d.encode_grouped()
m = lgbm(d.best_params); m.fit(g["Xtr"], g["ytr"])
p_va = m.predict_proba(g["Xva"])[:, 1]
p_te_raw = m.predict_proba(g["Xte"])[:, 1]

iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va, g["yva"])
plat = LogisticRegression(max_iter=1000).fit(p_va.reshape(-1, 1), g["yva"])

variants = {
    "uncalibrated": p_te_raw,
    "isotonic (production)": iso.predict(p_te_raw),
    "platt / sigmoid": plat.predict_proba(p_te_raw.reshape(-1, 1))[:, 1],
}
out = {}
for name, p in variants.items():
    mm = full_metrics(g["yte"], p, 0.5)
    out[name] = mm
    log(f"  [{name:22s}] TEST Brier={mm['brier']:.6f}  PR-AUC={mm['pr_auc']:.5f}  "
        f"ROC={mm['roc_auc']:.5f}  acc={mm['accuracy']:.5f}")

fig, ax = plt.subplots(figsize=(6.4, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
for (name, p), c in zip(variants.items(), ["#6b7280", "#2563eb", "#db2777"]):
    ft, fp = calibration_curve(g["yte"], p, n_bins=20, strategy="quantile")
    ax.plot(fp, ft, "o-", ms=4, color=c, lw=1.8,
            label=f"{name} (Brier {out[name]['brier']:.5f})")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Calibration on the held-out test partition")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG_DIR / "07_calibration_curve.png", dpi=160,
                                facecolor="white")
plt.close(fig); log("  figure -> 07_calibration_curve.png")
json.dump(out, open(EXP_DIR / "exp_07_calibration.json", "w"), indent=1)
log("EXP-07 done")
