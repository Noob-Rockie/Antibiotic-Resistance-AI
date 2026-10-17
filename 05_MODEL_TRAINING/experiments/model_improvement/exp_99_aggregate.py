"""Aggregate every experiment into the log, the comparison table and figures."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp_common import EXP_DIR, FIG_DIR, baseline, log
ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "14_REPORT_MATERIAL" / "tables"

def J(n):
    p = EXP_DIR / n
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

b = baseline(); bt, btemp = b["test_metrics"], b["temporal_metrics"]
e1, e2, e3, e4, e7 = (J("exp_01_threshold.json"), J("exp_02_class_weight.json"),
                      J("exp_03_hyperparams.json"), J("exp_04_features.json"),
                      J("exp_07_calibration.json"))
rows, comp = [], []

def add(eid, etype, params, fset, vpr, te, tp, decision, notes):
    rows.append({
        "Experiment_ID": eid, "Experiment_Type": etype, "Parameters": params,
        "Feature_Set": fset,
        "Validation_PR_AUC": vpr,
        "Test_PR_AUC": te["pr_auc"], "Temporal_PR_AUC": tp["pr_auc"],
        "Test_Accuracy": te["accuracy"], "Temporal_Accuracy": tp["accuracy"],
        "Test_Recall": te["recall_sensitivity"], "Temporal_Recall": tp["recall_sensitivity"],
        "Test_Specificity": te["specificity"], "Temporal_Specificity": tp["specificity"],
        "Test_F1": te["f1"], "Temporal_F1": tp["f1"],
        "Decision": decision, "Notes": notes})

def addc(label, te, tp):
    comp.append({"Model/Experiment": label, "Accuracy": te["accuracy"],
                 "PR-AUC": te["pr_auc"], "Recall": te["recall_sensitivity"],
                 "Specificity": te["specificity"], "F1": te["f1"],
                 "Temporal PR-AUC": tp["pr_auc"],
                 "Temporal Recall": tp["recall_sensitivity"]})

add("BASE-00", "baseline", "threshold=0.50", "production 30", 0.77136, bt, btemp,
    "reference", "Production LightGBM, read from 06_MODEL_EVALUATION/evaluation_report.json")
addc("BASELINE — production LightGBM (t=0.50)", bt, btemp)

if e1:
    for nm, a in e1["applied"].items():
        if nm == "baseline_0.50": continue
        d = "ACCEPT (candidate)" if nm == "max_f1" else "ACCEPT (alternative operating point)"
        add(f"THR-{nm}", "threshold", f"threshold={a['threshold']}", "production 30",
            0.77136, a["test"], a["temporal"], d,
            "Threshold selected on a held-out half of validation; PR-AUC/ROC-AUC unchanged by construction")
        addc(f"Threshold {a['threshold']:.2f} ({nm})", a["test"], a["temporal"])
if e2:
    v = {r["weight"]: r for r in e2["validation"]}
    for k, a in e2["applied"].items():
        w = float(k.split("=")[1])
        if w == 1.0: continue
        add(f"CW-{w}", "class_weight", f"class_weight={{0:1,1:{w}}}", "production 30",
            v[w]["pr_auc"], a["test"], a["temporal"], "REJECT",
            "PR-AUC flat across weights; trade-off duplicates thresholding and degrades Brier")
        addc(f"Class weight {w}", a["test"], a["temporal"])
if e3:
    for nm, a in e3["applied"].items():
        if nm == "production": continue
        add("HP-searched", "hyperparameters", json.dumps(a["params"]), "production 30",
            a["validation"]["pr_auc"], a["test"], a["temporal"], "SEE REPORT",
            f"24-iteration GroupKFold search; CV {e3['best_cv_pr_auc']} vs production {e3['production_params_cv_pr_auc']}")
        addc("Hyperparameter search winner", a["test"], a["temporal"])
NOT_RUN = {"pr_auc": None, "accuracy": None, "recall_sensitivity": None,
           "specificity": None, "f1": None}
if e4:
    for nm, mv in e4["validation"].items():
        if nm == "A_production_30": continue
        applied = e4["applied"].get(nm)
        te = applied["test"] if applied else NOT_RUN
        tp = applied["temporal"] if applied else NOT_RUN
        dec = "REJECT" if not applied else "SEE REPORT"
        note = (f"validation PR-AUC {mv['pr_auc']} vs production "
                f"{e4['validation']['A_production_30']['pr_auc']}; "
                + ("lost on validation, so never evaluated on test/temporal "
                   "(selection protocol)" if not applied else "validation winner"))
        add(f"FS-{nm}", "features", "production hyperparameters",
            f"{nm} (n={mv['n_features']})", mv["pr_auc"], te, tp, dec, note)
        if applied:
            addc(f"Feature set {nm} (n={mv['n_features']})", te, tp)

if e7:
    b_un = e7.get("uncalibrated"); b_iso = e7.get("isotonic (production)")
    b_pl = e7.get("platt / sigmoid")
    tp_base = btemp   # temporal baseline is already uncalibrated
    for nm, mm in [("uncalibrated", b_un), ("platt / sigmoid", b_pl)]:
        if not mm: continue
        add(f"CAL-{nm}", "calibration", f"calibration={nm}", "production 30",
            None, mm, tp_base, "SEE REPORT",
            "Temporal baseline is already uncalibrated, so temporal is unchanged "
            "by the calibration choice")
        addc(f"Calibration: {nm}", mm, tp_base)

pd.DataFrame(rows).to_csv(EXP_DIR / "experiment_log.csv", index=False)
cdf = pd.DataFrame(comp)
cdf.to_csv(TABLES / "model_improvement_comparison.csv", index=False)
log(f"wrote experiment_log.csv ({len(rows)} rows)")
log(f"wrote model_improvement_comparison.csv ({len(cdf)} rows)")
log("\n" + cdf.to_string(index=False))

# ---- figure 5: baseline vs candidate on the test partition ------------------
if e1:
    cand = e1["applied"]["max_f1"]
    labels = ["Accuracy", "PR-AUC", "Recall", "Specificity", "F1"]
    base_v = [bt["accuracy"], bt["pr_auc"], bt["recall_sensitivity"],
              bt["specificity"], bt["f1"]]
    cand_v = [cand["test"]["accuracy"], cand["test"]["pr_auc"],
              cand["test"]["recall_sensitivity"], cand["test"]["specificity"],
              cand["test"]["f1"]]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - w/2, base_v, w, label="Baseline (t=0.50)", color="#6b7280")
    ax.bar(x + w/2, cand_v, w, label=f"Candidate (t={cand['threshold']:.2f})",
           color="#2563eb")
    for i, (a, c) in enumerate(zip(base_v, cand_v)):
        ax.text(i - w/2, a + .012, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, c + .012, f"{c:.3f}", ha="center", fontsize=8,
                fontweight="bold" if c > a else "normal")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.08)
    ax.set_title("Baseline vs candidate — held-out test partition")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_baseline_vs_candidate.png",
                                    dpi=160, facecolor="white"); plt.close(fig)
    log("figure -> 05_baseline_vs_candidate.png")

    # ---- figure 6: temporal comparison --------------------------------------
    base_t = [btemp["accuracy"], btemp["pr_auc"], btemp["recall_sensitivity"],
              btemp["specificity"], btemp["f1"]]
    cand_t = [cand["temporal"]["accuracy"], cand["temporal"]["pr_auc"],
              cand["temporal"]["recall_sensitivity"],
              cand["temporal"]["specificity"], cand["temporal"]["f1"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - w/2, base_t, w, label="Baseline (t=0.50)", color="#6b7280")
    ax.bar(x + w/2, cand_t, w, label=f"Candidate (t={cand['threshold']:.2f})",
           color="#db2777")
    for i, (a, c) in enumerate(zip(base_t, cand_t)):
        ax.text(i - w/2, a + .012, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, c + .012, f"{c:.3f}", ha="center", fontsize=8,
                fontweight="bold" if c > a else "normal")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.08)
    ax.set_title("Baseline vs candidate — TEMPORAL holdout (fit on past, test on future)")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_temporal_comparison.png",
                                    dpi=160, facecolor="white"); plt.close(fig)
    log("figure -> 06_temporal_comparison.png")
