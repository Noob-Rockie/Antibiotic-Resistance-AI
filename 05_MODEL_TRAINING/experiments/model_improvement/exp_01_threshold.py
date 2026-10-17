"""EXP-01: threshold optimisation.

Threshold is a DEPLOYMENT choice, not a model change: PR-AUC and ROC-AUC are
threshold-independent and are reported here unchanged to make that explicit.

Selection protocol (no test-set involvement):
  * validation partition is split BY PATIENT into two disjoint halves
  * isotonic calibrator fitted on half A
  * threshold swept and chosen on half B
  * the chosen thresholds are then applied ONCE to the untouched test partition
    and to the temporal holdout
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from exp_common import (Data, EXP_DIR, FIG_DIR, baseline, full_metrics, lgbm,
                        log, val_halves)

GRID = [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
log("EXP-01  threshold optimisation")
b = baseline()
d = Data(); g = d.encode_grouped()

model = lgbm(d.best_params); model.fit(g["Xtr"], g["ytr"])
p_va_raw = model.predict_proba(g["Xva"])[:, 1]

# --- honest split of the validation partition --------------------------------
A, B = val_halves(g["pat_va"])
log(f"  validation split by patient: calibration half={A.sum():,} rows, "
    f"threshold-selection half={B.sum():,} rows")
cal_A = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
    p_va_raw[A], g["yva"][A])
p_B = cal_A.predict(p_va_raw[B]); y_B = g["yva"][B]

rows = []
for t in GRID:
    m = full_metrics(y_B, p_B, t); m["partition"] = "validation (held-out half)"
    rows.append(m)
sweep = pd.DataFrame(rows)
sweep.to_csv(EXP_DIR / "exp_01_threshold_sweep_validation.csv", index=False)
log("\n  threshold sweep on the held-out validation half:")
log(sweep[["threshold","accuracy","precision_ppv","recall_sensitivity",
           "specificity","f1"]].to_string(index=False))
log(f"\n  PR-AUC is threshold-independent and constant at "
    f"{sweep['pr_auc'].iloc[0]:.5f} across every threshold "
    f"(unique values: {sweep['pr_auc'].nunique()}).")

# --- candidate operating points, chosen on validation only -------------------
best_f1 = float(sweep.loc[sweep["f1"].idxmax(), "threshold"])
# a recall-oriented point that still keeps specificity >= 0.90
ok = sweep[sweep["specificity"] >= 0.90]
recall_pt = float(ok.loc[ok["recall_sensitivity"].idxmax(), "threshold"]) if len(ok) else best_f1
log(f"\n  selected on validation:  max-F1 threshold={best_f1}   "
    f"recall-oriented (specificity>=0.90) threshold={recall_pt}")

# --- apply ONCE to test and temporal ----------------------------------------
cal_full = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
    p_va_raw, g["yva"])                       # production calibration protocol
p_te = cal_full.predict(model.predict_proba(g["Xte"])[:, 1])

t = d.encode_temporal()
tm = lgbm(d.best_params); tm.fit(t["Xtr"], t["ytr"])
p_t = tm.predict_proba(t["Xte"])[:, 1]

out = {"grid": GRID, "selected_max_f1": best_f1,
       "selected_recall_oriented": recall_pt,
       "pr_auc_is_threshold_independent": True,
       "validation_sweep": rows, "applied": {}}
log("\n  applied to the untouched partitions:")
for name, th in [("baseline_0.50", 0.50), ("max_f1", best_f1),
                 ("recall_oriented", recall_pt)]:
    mte = full_metrics(g["yte"], p_te, th)
    mt  = full_metrics(t["yte"], p_t, th)
    out["applied"][name] = {"threshold": th, "test": mte, "temporal": mt}
    log(f"    t={th:.2f} [{name:16s}] TEST acc={mte['accuracy']:.5f} "
        f"rec={mte['recall_sensitivity']:.5f} spec={mte['specificity']:.5f} "
        f"f1={mte['f1']:.5f} | TEMPORAL rec={mt['recall_sensitivity']:.5f} "
        f"spec={mt['specificity']:.5f} f1={mt['f1']:.5f}")

json.dump(out, open(EXP_DIR / "exp_01_threshold.json", "w"), indent=1)

# --- figures ----------------------------------------------------------------
def fig_line(ycol, title, fname, ylab):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(sweep["threshold"], sweep[ycol], "o-", color="#2563eb", lw=2)
    ax.axvline(0.50, color="#6b7280", ls="--", lw=1.2, label="production threshold 0.50")
    ax.axvline(best_f1, color="#dc2626", ls=":", lw=1.6, label=f"max-F1 {best_f1:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=160, facecolor="white")
    plt.close(fig); log(f"  figure -> {fname}")

fig_line("recall_sensitivity", "Threshold vs recall (validation half)",
         "01_threshold_vs_recall.png", "Recall / sensitivity")
fig_line("specificity", "Threshold vs specificity (validation half)",
         "02_threshold_vs_specificity.png", "Specificity")
fig_line("f1", "Threshold vs F1 (validation half)",
         "03_threshold_vs_f1.png", "F1")

fig, ax = plt.subplots(figsize=(7.2, 5))
for nm, yy, pp, col in [("Test partition", g["yte"], p_te, "#2563eb"),
                        ("Temporal holdout", t["yte"], p_t, "#db2777")]:
    pr, rc, _ = precision_recall_curve(yy, pp)
    ax.plot(rc, pr, lw=2, color=col,
            label=f"{nm} (PR-AUC {average_precision_score(yy, pp):.4f})")
    ax.axhline(float(np.mean(yy)), color=col, ls=":", lw=1,
               label=f"{nm} prevalence {np.mean(yy):.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-recall curves — production model"); ax.legend(fontsize=8)
ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(FIG_DIR / "04_precision_recall_curves.png", dpi=160, facecolor="white")
plt.close(fig); log("  figure -> 04_precision_recall_curves.png")
log("EXP-01 done")
