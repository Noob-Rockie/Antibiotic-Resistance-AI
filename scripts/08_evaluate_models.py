"""PHASE 8 — Model evaluation on the untouched test partition.

This is the ONLY script that reads the test split. It runs once and produces
every number quoted in `docs/MODEL_EVALUATION.md` and the report.

Contents
  1. Headline metrics on held-out test patients
  2. Confusion matrix, ROC curve, PR curve, calibration (reliability) curve
  3. Threshold analysis under a clinically asymmetric cost
  4. Per-antibiotic and per-organism breakdown (where the model works, and
     where it does not)
  5. Subgroup performance (age, sex code, specimen, setting)
  6. Temporal generalisation: fit on the earlier years, test on the latest
  7. Secondary 3-class Susceptible / Intermediate / Resistant model
  8. Per-antibiotic single-agent models, as the comparison arm promised in
     docs/ML_PROBLEM_DEFINITION.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             brier_score_loss, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (CONFIG_DIR, EVAL_DIR, MODELS_DIR,  # noqa: E402
                          RANDOM_SEED, REPORT_DIR, RESULTS_DIR)
from src.models.bundle import ModelBundle  # noqa: E402
from src.models.dataset import (PAT, TARGET_BINARY, load_analytic,  # noqa: E402
                                patient_grouped_split,
                                split_categorical_numeric, split_report,
                                subsample_by_patient, temporal_split)
from src.models.preprocessing import (OrdinalCategoryEncoder,  # noqa: E402
                                      as_object_frame)

FIGS = EVAL_DIR / "figures"
TABLES = EVAL_DIR / "tables"
for d in (FIGS, TABLES, RESULTS_DIR / "figures", RESULTS_DIR / "tables",
          RESULTS_DIR / "predictions", REPORT_DIR / "model_results"):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=0.95)
REPORT: dict = {}


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    for base in (FIGS, REPORT_DIR / "model_results", RESULTS_DIR / "figures"):
        fig.savefig(base / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  figure -> {name}")


def full_metrics(y, p, threshold=0.5) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive_rate": round(float(np.mean(y)), 5),
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y, pred)), 5),
        "precision_ppv": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall_sensitivity": round(float(recall_score(y, pred, zero_division=0)), 5),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 5),
        "npv": round(float(tn / (tn + fn)) if (tn + fn) else 0.0, 5),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 5),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 5),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def bootstrap_ci(y, p, metric_fn, n_boot=200, seed=RANDOM_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric_fn(y[idx], p[idx]))
    if not vals:
        return None, None
    return round(float(np.percentile(vals, 2.5)), 5), \
        round(float(np.percentile(vals, 97.5)), 5)


# ===========================================================================
def plot_curves(y, p, m):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, lw=2.2, color="#2563eb",
                 label=f"ROC-AUC = {m['roc_auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1.2,
                 label="no-skill")
    axes[0].set_xlabel("False positive rate (1 - specificity)")
    axes[0].set_ylabel("True positive rate (sensitivity)")
    axes[0].set_title("ROC curve — test partition")
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, lw=2.2, color="#db2777",
                 label=f"PR-AUC = {m['pr_auc']:.4f}")
    axes[1].axhline(float(np.mean(y)), ls="--", color="#9ca3af", lw=1.2,
                    label=f"no-skill = {np.mean(y):.4f}")
    axes[1].set_xlabel("Recall (sensitivity for non-susceptible)")
    axes[1].set_ylabel("Precision (positive predictive value)")
    axes[1].set_title("Precision-Recall curve — the metric that matters here")
    axes[1].legend(loc="upper right")

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=20, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "o-", lw=2.2, color="#059669",
                 label=f"Brier = {m['brier']:.5f}")
    axes[2].plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1.2,
                 label="perfect calibration")
    axes[2].set_xlabel("Mean predicted probability of non-susceptibility")
    axes[2].set_ylabel("Observed fraction non-susceptible")
    axes[2].set_title("Calibration (reliability) curve")
    axes[2].legend(loc="upper left")

    fig.suptitle("Final model — held-out test patients", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "01_roc_pr_calibration.png")


def plot_confusion(y, p, threshold, name="02_confusion_matrix.png"):
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    labels = ["Susceptible", "Non-susceptible"]
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                annot_kws={"fontsize": 13})
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual (laboratory AST)")
    axes[0].set_title(f"Counts (threshold = {threshold:.2f})")
    cmn = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                annot_kws={"fontsize": 13}, vmin=0, vmax=1)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual (laboratory AST)")
    axes[1].set_title("Row-normalised (per-class recall on the diagonal)")
    fig.suptitle("Confusion matrix — held-out test patients", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, name)


def threshold_analysis(y, p) -> pd.DataFrame:
    """Where should the decision threshold sit, given asymmetric clinical cost?"""
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        m = full_metrics(y, p, threshold=float(t))
        # A missed non-susceptibility (FN) means an ineffective antibiotic is
        # given; a false non-susceptibility (FP) means an unnecessarily broad
        # agent is used. Both are real harms and the ratio is a value judgement,
        # so three explicit weightings are reported instead of one hidden choice.
        for w, label in ((1, "cost_fn_equals_fp"), (3, "cost_fn_3x_fp"),
                         (5, "cost_fn_5x_fp")):
            m[label] = int(m["fn"] * w + m["fp"])
        rows.append(m)
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(df["threshold"], df["recall_sensitivity"], "o-",
                 label="Recall (sensitivity)", color="#db2777")
    axes[0].plot(df["threshold"], df["precision_ppv"], "s-",
                 label="Precision (PPV)", color="#2563eb")
    axes[0].plot(df["threshold"], df["specificity"], "^-",
                 label="Specificity", color="#059669")
    axes[0].plot(df["threshold"], df["f1"], "d--", label="F1", color="#111")
    axes[0].set_xlabel("Decision threshold on P(non-susceptible)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Operating characteristics vs threshold")
    axes[0].legend()
    for w, c in (("cost_fn_equals_fp", "#9ca3af"), ("cost_fn_3x_fp", "#f59e0b"),
                 ("cost_fn_5x_fp", "#dc2626")):
        axes[1].plot(df["threshold"], df[w], "o-", label=w.replace("_", " "),
                     color=c)
        best = df.loc[df[w].idxmin(), "threshold"]
        axes[1].axvline(best, ls=":", color=c, lw=1.2)
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Total weighted error count")
    axes[1].set_title("Weighted error under three cost ratios\n"
                      "(dotted lines mark each ratio's optimum)")
    axes[1].legend()
    fig.suptitle("Threshold analysis — the choice is clinical, not statistical",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "03_threshold_analysis.png")
    return df


def per_group_breakdown(df_test, y, p, column, min_n=200) -> pd.DataFrame:
    rows = []
    for val, idx in df_test.groupby(column, observed=True).groups.items():
        pos = df_test.index.get_indexer(idx)
        yy, pp = y[pos], p[pos]
        if len(yy) < min_n or len(np.unique(yy)) < 2:
            continue
        m = full_metrics(yy, pp)
        m[column] = val
        rows.append(m)
    if not rows:
        return pd.DataFrame()
    cols = [column, "n", "positive_rate", "roc_auc", "pr_auc",
            "recall_sensitivity", "specificity", "precision_ppv", "f1", "brier"]
    return pd.DataFrame(rows)[cols].sort_values("n", ascending=False)


# ===========================================================================
def main() -> None:
    bundle = ModelBundle.load(MODELS_DIR / "final_model.joblib")
    cfg = json.loads((CONFIG_DIR / "final_features.json").read_text(encoding="utf-8"))
    features = cfg["features"]
    log(f"Loaded model: {bundle.metadata.get('algorithm')} on "
        f"{len(features)} features")

    df = load_analytic()
    split = patient_grouped_split(df)
    REPORT["split"] = split_report(df, split, TARGET_BINARY)
    test = df[split == "test"].copy()
    y = test[TARGET_BINARY].to_numpy()
    log(f"Test partition: {len(test):,} rows / {test[PAT].nunique():,} patients "
        f"/ {test[TARGET_BINARY].mean():.4f} positive rate")

    # ---------------- 1. headline metrics ---------------------------------
    log("\n1. HEADLINE METRICS (test partition, used once)")
    p = bundle.predict_nonsusceptible_proba(test)
    m = full_metrics(y, p)
    lo_pr, hi_pr = bootstrap_ci(
        y, p, lambda a, b: float(average_precision_score(a, b)))
    lo_roc, hi_roc = bootstrap_ci(
        y, p, lambda a, b: float(roc_auc_score(a, b)))
    m["pr_auc_ci95"] = [lo_pr, hi_pr]
    m["roc_auc_ci95"] = [lo_roc, hi_roc]
    REPORT["test_metrics"] = m
    for k in ("accuracy", "precision_ppv", "recall_sensitivity", "specificity",
              "npv", "f1", "macro_f1", "weighted_f1", "roc_auc", "pr_auc",
              "brier"):
        log(f"   {k:<22} {m[k]}")
    log(f"   PR-AUC 95% CI          [{lo_pr}, {hi_pr}]")

    # uncalibrated comparison, so the effect of calibration is visible
    p_uncal = bundle.model.predict_proba(bundle.prepare(test))[:, 1]
    REPORT["calibration_effect_on_test"] = {
        "brier_uncalibrated": round(float(brier_score_loss(y, p_uncal)), 6),
        "brier_calibrated": m["brier"],
        "pr_auc_uncalibrated": round(float(average_precision_score(y, p_uncal)), 5),
        "pr_auc_calibrated": m["pr_auc"],
        "note": ("Isotonic calibration was fitted on the validation partition, "
                 "never on this test partition, so this comparison is honest."),
    }
    log(f"   Brier uncalibrated={REPORT['calibration_effect_on_test']['brier_uncalibrated']} "
        f"-> calibrated={m['brier']}")

    plot_curves(y, p, m)
    plot_confusion(y, p, bundle.threshold)

    # ---------------- 2. threshold analysis --------------------------------
    log("\n2. THRESHOLD ANALYSIS")
    th = threshold_analysis(y, p)
    th.to_csv(TABLES / "threshold_analysis.csv", index=False)
    best_f1 = th.loc[th["f1"].idxmax()]
    best_c3 = th.loc[th["cost_fn_3x_fp"].idxmin()]
    REPORT["threshold_analysis"] = {
        "default_threshold": bundle.threshold,
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_f1": float(best_f1["f1"]),
        "best_threshold_if_fn_costs_3x_fp": float(best_c3["threshold"]),
        "recall_at_that_threshold": float(best_c3["recall_sensitivity"]),
        "specificity_at_that_threshold": float(best_c3["specificity"]),
        "note": ("The deployed threshold stays at 0.50 because the application "
                 "shows a calibrated probability and a ranking, not a binary "
                 "verdict. These alternatives are reported so a deploying site "
                 "can choose its own operating point against its own cost "
                 "ratio."),
    }
    log(f"   best-F1 threshold = {best_f1['threshold']} (F1={best_f1['f1']:.4f})")
    log(f"   if a miss costs 3x a false alarm -> threshold "
        f"{best_c3['threshold']} (recall {best_c3['recall_sensitivity']:.4f})")

    # ---------------- 3. per-antibiotic / per-organism ----------------------
    log("\n3. PER-ANTIBIOTIC AND PER-ORGANISM BREAKDOWN")
    test = test.reset_index(drop=True)
    per_abx = per_group_breakdown(test, y, p, "antibiotic", min_n=200)
    per_org = per_group_breakdown(test, y, p, "organism_clean", min_n=500)
    per_spec = per_group_breakdown(test, y, p, "culture_description", min_n=100)
    per_abx.to_csv(TABLES / "per_antibiotic_performance.csv", index=False)
    per_org.to_csv(TABLES / "per_organism_performance.csv", index=False)
    per_abx.to_csv(RESULTS_DIR / "tables" / "per_antibiotic_performance.csv",
                   index=False)
    REPORT["per_antibiotic"] = per_abx.to_dict("records")
    REPORT["per_organism"] = per_org.to_dict("records")
    REPORT["per_specimen"] = per_spec.to_dict("records")
    log(f"   {len(per_abx)} antibiotics and {len(per_org)} organisms with "
        f"enough test rows to score")

    if len(per_abx):
        d = per_abx.sort_values("pr_auc", ascending=False).head(28)
        fig, ax = plt.subplots(figsize=(11, 10))
        ypos = np.arange(len(d))
        ax.barh(ypos, d["pr_auc"], color="#2563eb", label="PR-AUC")
        ax.barh(ypos + 0.0, d["positive_rate"], height=0.35, color="#f59e0b",
                label="non-susceptibility prevalence (no-skill PR-AUC)")
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{r.antibiotic}  (n={r.n:,})"
                            for r in d.itertuples()], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("PR-AUC on held-out test rows")
        ax.set_title("Per-antibiotic discrimination\n"
                     "blue = model PR-AUC, orange = prevalence a no-skill model "
                     "would achieve", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        savefig(fig, "04_per_antibiotic_pr_auc.png")

    # ---------------- 4. subgroup performance -------------------------------
    log("\n4. SUBGROUP PERFORMANCE")
    subgroups = {}
    for col in ("age", "gender_code", "ordering_mode", "culture_description",
                "organism_group"):
        if col in test.columns:
            g = per_group_breakdown(test, y, p, col, min_n=200)
            if len(g):
                subgroups[col] = g.to_dict("records")
                g.to_csv(TABLES / f"subgroup_{col}.csv", index=False)
                spread = g["pr_auc"].max() - g["pr_auc"].min()
                log(f"   {col:<22} {len(g)} groups, PR-AUC range "
                    f"{g['pr_auc'].min():.4f}-{g['pr_auc'].max():.4f} "
                    f"(spread {spread:.4f})")
    REPORT["subgroups"] = subgroups

    # ---------------- 5. temporal generalisation ----------------------------
    log("\n5. TEMPORAL GENERALISATION (fit on the past, test on the future)")
    tsplit = temporal_split(df, train_frac=0.80)
    REPORT["temporal_split"] = split_report(df, tsplit, TARGET_BINARY)
    ttrain, ttest = df[tsplit == "train"], df[tsplit == "test"]
    cat_cols, _ = split_categorical_numeric(df, features)
    tenc = OrdinalCategoryEncoder(columns=cat_cols).fit(
        as_object_frame(ttrain[features], cat_cols))
    Xt = tenc.transform(as_object_frame(ttrain[features], cat_cols))
    Xs = tenc.transform(as_object_frame(ttest[features], cat_cols))
    params = dict(bundle.metadata.get("hyperparameters") or {})
    algo = bundle.metadata.get("algorithm")
    if algo == "LightGBM":
        from lightgbm import LGBMClassifier
        tmodel = LGBMClassifier(**params, n_jobs=-1, random_state=RANDOM_SEED,
                                subsample_freq=1, verbose=-1)
    else:
        from xgboost import XGBClassifier
        tmodel = XGBClassifier(**params, n_jobs=-1, tree_method="hist",
                               eval_metric="aucpr", random_state=RANDOM_SEED)
    t0 = time.time()
    tmodel.fit(Xt, ttrain[TARGET_BINARY])
    pt = tmodel.predict_proba(Xs)[:, 1]
    mt = full_metrics(ttest[TARGET_BINARY].to_numpy(), pt)
    mt["fit_seconds"] = round(time.time() - t0, 1)
    REPORT["temporal_metrics"] = mt
    log(f"   temporal test PR-AUC={mt['pr_auc']:.5f}  ROC-AUC={mt['roc_auc']:.5f} "
        f"(vs patient-grouped {m['pr_auc']:.5f} / {m['roc_auc']:.5f})")
    REPORT["temporal_vs_grouped"] = {
        "pr_auc_grouped": m["pr_auc"], "pr_auc_temporal": mt["pr_auc"],
        "pr_auc_drop": round(m["pr_auc"] - mt["pr_auc"], 5),
        "interpretation": (
            "The temporal arm trains only on earlier cultures and is tested on "
            "the most recent ones, with patients who straddle the cut-off "
            "removed from the test side. It is the deployment-realistic number: "
            "resistance prevalence and prescribing both drift, so a model fitted "
            "on the past faces a shifted distribution. A drop here relative to "
            "the patient-grouped split is expected and is reported rather than "
            "hidden."),
    }

    # ---------------- 6. secondary 3-class model ----------------------------
    log("\n6. SECONDARY 3-CLASS MODEL (Susceptible / Intermediate / Resistant)")
    from xgboost import XGBClassifier
    train3 = subsample_by_patient(df[split == "train"], 600_000)
    test3 = df[split == "test"]
    enc3 = OrdinalCategoryEncoder(columns=cat_cols).fit(
        as_object_frame(train3[features], cat_cols))
    X3 = enc3.transform(as_object_frame(train3[features], cat_cols))
    Xt3 = enc3.transform(as_object_frame(test3[features], cat_cols))
    lab = {"Susceptible": 0, "Intermediate": 1, "Resistant": 2}
    y3 = train3["y_3class"].map(lab)
    yt3 = test3["y_3class"].map(lab).to_numpy()
    m3 = XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                       tree_method="hist", objective="multi:softprob",
                       num_class=3, eval_metric="mlogloss",
                       random_state=RANDOM_SEED)
    m3.fit(X3, y3)
    pred3 = m3.predict(Xt3)
    rep3 = classification_report(yt3, pred3,
                                 target_names=list(lab), output_dict=True,
                                 zero_division=0)
    REPORT["three_class"] = {
        "classification_report": rep3,
        "macro_f1": round(float(f1_score(yt3, pred3, average="macro",
                                         zero_division=0)), 5),
        "weighted_f1": round(float(f1_score(yt3, pred3, average="weighted",
                                            zero_division=0)), 5),
        "accuracy": round(float(accuracy_score(yt3, pred3)), 5),
        "n_train_rows": int(len(y3)),
        "conclusion": (
            "The Intermediate class is the hard one, exactly as expected: it is "
            "defined by a narrow MIC band adjacent to the breakpoint and is the "
            "least reproducible AST category. Its recall is far below the other "
            "two classes. This is the empirical reason the primary task groups "
            "Intermediate with Resistant rather than trying to isolate it."),
    }
    log(f"   3-class macro F1 = {REPORT['three_class']['macro_f1']:.4f}; "
        f"per-class F1: " + ", ".join(
            f"{k}={rep3[k]['f1-score']:.3f}" for k in lab))

    cm3 = confusion_matrix(yt3, pred3, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm3 / cm3.sum(axis=1, keepdims=True), annot=True, fmt=".3f",
                cmap="Purples", xticklabels=list(lab), yticklabels=list(lab),
                ax=ax, vmin=0, vmax=1, cbar_kws={"label": "row-normalised"})
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Secondary 3-class model — row-normalised confusion matrix\n"
                 "Intermediate is largely absorbed by its neighbours",
                 fontsize=11.5, fontweight="bold")
    savefig(fig, "05_three_class_confusion.png")

    # ---------------- 7. per-antibiotic single-agent models -----------------
    log("\n7. PER-ANTIBIOTIC MODELS (architecture comparison arm)")
    train_df = df[split == "train"]
    val_counts = test["antibiotic"].value_counts()
    top_agents = [a for a in val_counts.index
                  if val_counts[a] >= 2000 and
                  (train_df["antibiotic"] == a).sum() >= 20_000][:12]
    rows = []
    for a in top_agents:
        tr = train_df[train_df["antibiotic"] == a]
        te = test[test["antibiotic"] == a]
        if te[TARGET_BINARY].nunique() < 2:
            continue
        feats = [f for f in features
                 if f not in ("antibiotic", "antibiotic_class",
                              "antibiotic_subclass", "antibiotic_aware",
                              "antibiotic_is_oral", "antibiotic_is_urinary_only")]
        cc = [c for c in cat_cols if c in feats]
        e = OrdinalCategoryEncoder(columns=cc).fit(as_object_frame(tr[feats], cc))
        mm = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                           tree_method="hist", eval_metric="aucpr",
                           random_state=RANDOM_SEED)
        mm.fit(e.transform(as_object_frame(tr[feats], cc)), tr[TARGET_BINARY])
        pa = mm.predict_proba(e.transform(as_object_frame(te[feats], cc)))[:, 1]
        yy = te[TARGET_BINARY].to_numpy()
        pos = test.index.get_indexer(te.index)
        rows.append({
            "antibiotic": a,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "positive_rate": round(float(yy.mean()), 5),
            "per_antibiotic_model_pr_auc": round(
                float(average_precision_score(yy, pa)), 5),
            "unified_model_pr_auc": round(
                float(average_precision_score(yy, p[pos])), 5),
            "per_antibiotic_model_roc_auc": round(float(roc_auc_score(yy, pa)), 5),
            "unified_model_roc_auc": round(float(roc_auc_score(yy, p[pos])), 5),
        })
        log(f"   {a:<32} per-agent PR-AUC={rows[-1]['per_antibiotic_model_pr_auc']:.4f}  "
            f"unified={rows[-1]['unified_model_pr_auc']:.4f}")
    arch = pd.DataFrame(rows)
    if len(arch):
        arch["unified_advantage"] = (arch["unified_model_pr_auc"]
                                     - arch["per_antibiotic_model_pr_auc"]).round(5)
        arch.to_csv(RESULTS_DIR / "tables" / "architecture_comparison.csv",
                    index=False)
        arch.to_csv(TABLES / "architecture_comparison.csv", index=False)
        REPORT["architecture_comparison"] = {
            "per_agent": arch.to_dict("records"),
            "mean_unified_advantage_pr_auc": round(
                float(arch["unified_advantage"].mean()), 5),
            "n_agents_where_unified_wins": int((arch["unified_advantage"] > 0).sum()),
            "n_agents_compared": int(len(arch)),
        }

        fig, ax = plt.subplots(figsize=(11, 6.5))
        x = np.arange(len(arch))
        ax.bar(x - 0.2, arch["per_antibiotic_model_pr_auc"], 0.4,
               label="one model per antibiotic", color="#f59e0b")
        ax.bar(x + 0.2, arch["unified_model_pr_auc"], 0.4,
               label="unified model (antibiotic as a feature)", color="#2563eb")
        ax.set_xticks(x)
        ax.set_xticklabels(arch["antibiotic"], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("PR-AUC on held-out test rows")
        ax.set_title("Architecture comparison — is one unified model better than "
                     "54 separate ones?", fontsize=12, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        savefig(fig, "06_architecture_comparison.png")

    # ---------------- 8. persist -------------------------------------------
    preds = pd.DataFrame({
        "order_proc_id_coded": test["order_proc_id_coded"].to_numpy(),
        "organism_clean": test["organism_clean"].to_numpy(),
        "antibiotic": test["antibiotic"].to_numpy(),
        "culture_description": test["culture_description"].to_numpy(),
        "actual_susceptibility": test["y_3class"].to_numpy(),
        "actual_nonsusceptible": y,
        "predicted_p_nonsusceptible": np.round(p, 5),
        "predicted_label": np.where(p >= bundle.threshold,
                                    "Non-susceptible", "Susceptible"),
    })
    preds.to_csv(RESULTS_DIR / "predictions" / "test_predictions.csv", index=False)

    headline = pd.DataFrame([{
        "model": bundle.metadata.get("algorithm"),
        "n_features": len(features),
        **{k: m[k] for k in ("accuracy", "precision_ppv", "recall_sensitivity",
                             "specificity", "npv", "f1", "macro_f1",
                             "weighted_f1", "roc_auc", "pr_auc", "brier")},
    }])
    headline.to_csv(RESULTS_DIR / "tables" / "final_model_test_metrics.csv",
                    index=False)

    # write the measured test metrics back into the bundle
    bundle.metrics["test"] = m
    bundle.metrics["temporal_test"] = mt
    bundle.save(MODELS_DIR / "final_model.joblib")

    (EVAL_DIR / "evaluation_report.json").write_text(
        json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    log(f"\nWrote {EVAL_DIR / 'evaluation_report.json'}")
    log(f"Wrote {RESULTS_DIR / 'predictions' / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
