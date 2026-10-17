"""PHASE 7 — Model training, hyperparameter tuning and imbalance handling.

Trains and compares five model families on the selected feature set, tunes the
two strongest, resolves the class-imbalance strategy empirically, refits the
winner on the full training partition, calibrates it on the validation
partition and saves a `ModelBundle`.

The test partition is NOT touched here. It is used exactly once, in
`scripts/08_evaluate_models.py`.

Outputs
  05_MODEL_TRAINING/models/final_model.joblib
  05_MODEL_TRAINING/experiments/experiment_log.csv
  05_MODEL_TRAINING/experiments/training_report.json
  13_RESULTS/tables/model_comparison.csv
  13_RESULTS/tables/imbalance_comparison.csv
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              RandomForestClassifier)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (CONFIG_DIR, EXPERIMENTS_DIR, MODELS_DIR,  # noqa: E402
                          RANDOM_SEED, RESULTS_DIR, ensure_dirs)
from src.models.bundle import ModelBundle  # noqa: E402
from src.models.dataset import (PAT, TARGET_BINARY, load_analytic,  # noqa: E402
                                patient_grouped_split,
                                split_categorical_numeric, split_report,
                                subsample_by_patient)
from src.models.preprocessing import (OrdinalCategoryEncoder,  # noqa: E402
                                      as_object_frame, linear_preprocessor)

SEARCH_ROWS = 250_000     # patient-level subsample for hyperparameter search
COMPARE_ROWS = 500_000    # patient-level subsample for the model comparison
REPORT: dict = {}
EXPERIMENTS: list = []


def log(m):
    print(m, flush=True)


def metrics(y, p, threshold=0.5, train_seconds=None) -> dict:
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 5),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall_sensitivity": round(float(recall_score(y, pred, zero_division=0)), 5),
        "specificity": round(tn / (tn + fp), 5) if (tn + fp) else None,
        "f1": round(float(f1_score(y, pred, zero_division=0)), 5),
        "macro_f1": round(float(f1_score(y, pred, average="macro",
                                         zero_division=0)), 5),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted",
                                            zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "train_seconds": round(train_seconds, 2) if train_seconds else None,
    }


def record(exp_id, algorithm, feature_set, hyperparams, cv, m, notes,
           dataset_version, n_train, n_features):
    EXPERIMENTS.append({
        "experiment_id": exp_id,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": dataset_version,
        "feature_set": feature_set,
        "n_features": n_features,
        "n_train_rows": n_train,
        "algorithm": algorithm,
        "hyperparameters": json.dumps(hyperparams, default=str),
        "cv_method": cv,
        **{k: v for k, v in m.items()},
        "notes": notes,
    })


# ===========================================================================
def build_models(y_train, n_cat, n_num, cat_cols, num_cols) -> dict:
    """Five model families, each configured for THIS dataset's shape.

    Deliberately excluded: a kernel SVM. Its training cost is roughly O(n^2) in
    the number of samples, which is intractable at 1.1M rows; the margin model
    is represented instead by a linear SVM trained with stochastic gradient
    descent, which is the same hypothesis class at a feasible cost.
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    return {
        "LogisticRegression": {
            "estimator": Pipeline([
                ("prep", linear_preprocessor(cat_cols, num_cols)),
                ("clf", LogisticRegression(max_iter=1500, C=1.0,
                                           class_weight="balanced",
                                           solver="lbfgs", n_jobs=-1,
                                           random_state=RANDOM_SEED)),
            ]),
            "needs_raw": True,
            "rationale": "Interpretable linear baseline. Establishes how much of "
                         "the signal is additive before any interaction is "
                         "modelled.",
        },
        "LinearSVM_SGD": {
            "estimator": Pipeline([
                ("prep", linear_preprocessor(cat_cols, num_cols)),
                ("clf", SGDClassifier(loss="modified_huber", penalty="l2",
                                      alpha=1e-5, max_iter=30, tol=1e-4,
                                      class_weight="balanced", n_jobs=-1,
                                      random_state=RANDOM_SEED)),
            ]),
            "needs_raw": True,
            "rationale": "Margin-based model at a tractable cost. "
                         "`modified_huber` loss gives calibrated-ish "
                         "probabilities directly, avoiding a second "
                         "calibration wrapper.",
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_leaf=20,
                max_features="sqrt", n_jobs=-1, random_state=RANDOM_SEED,
                class_weight="balanced_subsample"),
            "needs_raw": False,
            "rationale": "Bagged trees. Low variance, handles the mixed feature "
                         "types, but cannot extrapolate and is memory-hungry at "
                         "this row count.",
        },
        "HistGradientBoosting": {
            "estimator": HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.08, max_depth=None,
                max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
                random_state=RANDOM_SEED, class_weight="balanced"),
            "needs_raw": False,
            "rationale": "scikit-learn's histogram gradient boosting. Included "
                         "instead of the classic `GradientBoostingClassifier`, "
                         "whose exact-split implementation would take hours on "
                         "1.1M rows for a strictly worse result.",
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                n_estimators=600, max_depth=8, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_lambda=1.5, gamma=0.0, n_jobs=-1, tree_method="hist",
                eval_metric="aucpr", random_state=RANDOM_SEED,
                scale_pos_weight=pos_weight),
            "needs_raw": False,
            "rationale": "Gradient boosting with native missing-value routing — "
                         "important here because lab missingness is informative "
                         "rather than random.",
        },
        "LightGBM": {
            "estimator": LGBMClassifier(
                n_estimators=600, learning_rate=0.08, num_leaves=63,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.5, n_jobs=-1,
                random_state=RANDOM_SEED, class_weight="balanced",
                verbose=-1),
            "needs_raw": False,
            "rationale": "Leaf-wise boosting; the fastest of the tree family at "
                         "this scale and a useful check that the XGBoost result "
                         "is not implementation-specific.",
        },
    }


# ===========================================================================
def imbalance_experiment(Xtr, ytr, Xva, yva, dataset_version, n_features):
    """Resolve the imbalance strategy with evidence rather than reflex."""
    from imblearn.under_sampling import RandomUnderSampler
    from xgboost import XGBClassifier

    log("\nCLASS IMBALANCE — comparing strategies")
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    rows = []

    def run(name, X, y, **kw):
        m = XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                          tree_method="hist", eval_metric="aucpr",
                          random_state=RANDOM_SEED, **kw)
        t0 = time.time()
        m.fit(X, y)
        el = time.time() - t0
        p = m.predict_proba(Xva)[:, 1]
        r = metrics(yva, p, train_seconds=el)
        r["strategy"] = name
        r["n_train_rows"] = int(len(y))
        rows.append(r)
        log(f"  {name:<28} PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  "
            f"recall={r['recall_sensitivity']:.4f}  F1={r['f1']:.4f}  "
            f"Brier={r['brier']:.4f}")
        record(f"IMB-{len(rows):02d}", "XGBoost", "D_consensus_final",
               {"imbalance": name}, "holdout-validation", r,
               f"imbalance strategy: {name}", dataset_version, len(y), n_features)
        return r

    run("none (baseline)", Xtr, ytr)
    run("scale_pos_weight", Xtr, ytr, scale_pos_weight=pos_weight)

    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    Xr, yr = rus.fit_resample(Xtr, ytr)
    run("random undersampling", Xr, yr)

    # SMOTE is evaluated on a subsample and reported honestly rather than
    # applied by reflex: most of these features are ordinal-coded categoricals
    # and sparse counts, and interpolating between them manufactures organisms
    # and antibiotics that do not exist.
    try:
        from imblearn.over_sampling import SMOTE
        n = min(120_000, len(Xtr))
        idx = np.random.default_rng(RANDOM_SEED).choice(len(Xtr), n, replace=False)
        Xs = Xtr.iloc[idx].fillna(Xtr.median(numeric_only=True)).fillna(0)
        ys = ytr.iloc[idx]
        t0 = time.time()
        Xsm, ysm = SMOTE(random_state=RANDOM_SEED, k_neighbors=5).fit_resample(Xs, ys)
        smote_time = time.time() - t0
        r = run("SMOTE (subsample)", Xsm, ysm)
        r["smote_resample_seconds"] = round(smote_time, 1)
    except Exception as exc:  # never let an optional arm break the run
        log(f"  SMOTE arm failed: {type(exc).__name__}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tables" / "imbalance_comparison.csv", index=False)
    best = df.sort_values("pr_auc", ascending=False).iloc[0]
    REPORT["imbalance"] = {
        "positive_rate": round(float(ytr.mean()), 6),
        "scale_pos_weight_value": round(pos_weight, 4),
        "results": rows,
        "selected": str(best["strategy"]),
        "rationale": (
            "The positive class is 19.5% — imbalanced but not extreme, so the "
            "question is which correction actually helps rather than whether to "
            "apply one. Strategies were compared on identical held-out "
            "validation patients. SMOTE is included for completeness but is a "
            "poor fit for this feature space: interpolating between "
            "ordinal-coded categoricals synthesises organism and antibiotic "
            "codes that correspond to no real entity, and interpolating sparse "
            "prior-exposure counts invents patient histories. Random "
            "undersampling discards a large share of the majority class and "
            "with it the informative lab and history patterns that only appear "
            "in susceptible isolates."),
    }
    return str(best["strategy"]), pos_weight


# ===========================================================================
def tune(name, estimator, param_dist, Xtr, ytr, groups, n_iter=12):
    log(f"\nTUNING {name} ({n_iter} candidates x 3-fold StratifiedGroupKFold)")
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator, param_dist, n_iter=n_iter, scoring="average_precision",
        cv=cv, random_state=RANDOM_SEED, n_jobs=1, refit=True, verbose=0)
    t0 = time.time()
    search.fit(Xtr, ytr, groups=groups)
    el = time.time() - t0
    log(f"  best CV PR-AUC = {search.best_score_:.5f}  ({el:.0f}s)")
    log(f"  best params    = {search.best_params_}")
    return search, el


# ===========================================================================
def main() -> None:
    ensure_dirs()
    for d in (RESULTS_DIR / "tables", EXPERIMENTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((CONFIG_DIR / "final_features.json").read_text(encoding="utf-8"))
    features = cfg["features"]
    log(f"Final feature set: {len(features)} features")

    df = load_analytic()
    dataset_version = f"analytic_dataset v1 ({len(df)} rows x {df.shape[1]} cols)"
    split = patient_grouped_split(df)
    REPORT["split"] = split_report(df, split, TARGET_BINARY)
    REPORT["feature_set"] = cfg

    cat_cols, num_cols = split_categorical_numeric(df, features)
    train_full = df[split == "train"]
    val = df[split == "val"]
    log(f"train={len(train_full):,} rows / {train_full[PAT].nunique():,} patients | "
        f"val={len(val):,} rows / {val[PAT].nunique():,} patients")

    # ---- encoders fitted on TRAINING data only -----------------------------
    Xtr_raw_full = as_object_frame(train_full[features], cat_cols)
    Xva_raw = as_object_frame(val[features], cat_cols)
    encoder = OrdinalCategoryEncoder(columns=cat_cols).fit(Xtr_raw_full)
    Xtr_full = encoder.transform(Xtr_raw_full)
    Xva = encoder.transform(Xva_raw)
    ytr_full = train_full[TARGET_BINARY]
    yva = val[TARGET_BINARY].to_numpy()

    # ---- subsamples for the search / comparison stages ---------------------
    cmp_idx = subsample_by_patient(train_full, COMPARE_ROWS).index
    srch_idx = subsample_by_patient(train_full, SEARCH_ROWS).index
    Xtr_cmp, ytr_cmp = Xtr_full.loc[cmp_idx], ytr_full.loc[cmp_idx]
    Xtr_raw_cmp = Xtr_raw_full.loc[cmp_idx]
    Xtr_srch, ytr_srch = Xtr_full.loc[srch_idx], ytr_full.loc[srch_idx]
    groups_srch = train_full.loc[srch_idx, PAT].to_numpy()
    REPORT["compute_budget"] = {
        "comparison_rows": int(len(cmp_idx)),
        "search_rows": int(len(srch_idx)),
        "final_fit_rows": int(len(train_full)),
        "note": ("Model comparison and hyperparameter search run on "
                 "patient-level subsamples so that the search space can be "
                 "explored properly; the selected model is then refitted on the "
                 "full training partition."),
    }

    # =====================================================================
    # 1. MODEL FAMILY COMPARISON
    # =====================================================================
    log("\nMODEL COMPARISON (identical training rows, identical validation set)")
    models = build_models(ytr_cmp, len(cat_cols), len(num_cols), cat_cols, num_cols)
    rows = []
    for i, (name, spec) in enumerate(models.items(), 1):
        est = spec["estimator"]
        X = Xtr_raw_cmp if spec["needs_raw"] else Xtr_cmp
        Xv = Xva_raw if spec["needs_raw"] else Xva
        t0 = time.time()
        try:
            est.fit(X, ytr_cmp)
        except Exception as exc:
            log(f"  {name:<24} FAILED: {type(exc).__name__}: {exc}")
            continue
        el = time.time() - t0
        p = est.predict_proba(Xv)[:, 1]
        m = metrics(yva, p, train_seconds=el)
        m["model"] = name
        m["rationale"] = spec["rationale"]
        rows.append(m)
        log(f"  {name:<24} PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
            f"F1={m['f1']:.4f}  recall={m['recall_sensitivity']:.4f}  "
            f"spec={m['specificity']:.4f}  {el:.0f}s")
        record(f"CMP-{i:02d}", name, "D_consensus_final",
               est.get_params() if not spec["needs_raw"] else {"pipeline": name},
               "holdout-validation", m, "model family comparison",
               dataset_version, len(ytr_cmp), len(features))

    comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    cols = ["model", "n_features"] if "n_features" in comparison else ["model"]
    comparison.insert(1, "n_features", len(features))
    comparison.to_csv(RESULTS_DIR / "tables" / "model_comparison.csv", index=False)
    REPORT["model_comparison"] = comparison.to_dict("records")
    log(f"\n  best family by PR-AUC: {comparison.iloc[0]['model']}")

    # =====================================================================
    # 2. CLASS IMBALANCE
    # =====================================================================
    strategy, pos_weight = imbalance_experiment(
        Xtr_cmp, ytr_cmp, Xva, yva, dataset_version, len(features))

    # =====================================================================
    # 3. HYPERPARAMETER TUNING for the two strongest tree models
    # =====================================================================
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    xgb_space = {
        "max_depth": [5, 6, 7, 8, 10],
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "n_estimators": [300, 500, 800],
        "min_child_weight": [1, 5, 10, 25],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1.5, 5.0],
        "gamma": [0.0, 0.5, 2.0],
    }
    xgb_base = XGBClassifier(
        n_jobs=-1, tree_method="hist", eval_metric="aucpr",
        random_state=RANDOM_SEED,
        scale_pos_weight=pos_weight if strategy == "scale_pos_weight" else 1.0)
    xgb_search, xgb_time = tune("XGBoost", xgb_base, xgb_space,
                                Xtr_srch, ytr_srch, groups_srch, n_iter=12)

    lgb_space = {
        "num_leaves": [31, 63, 127, 255],
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "n_estimators": [300, 500, 800],
        "min_child_samples": [20, 40, 80],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1.5, 5.0],
    }
    lgb_base = LGBMClassifier(n_jobs=-1, random_state=RANDOM_SEED,
                              subsample_freq=1, verbose=-1,
                              class_weight="balanced"
                              if strategy != "none (baseline)" else None)
    lgb_search, lgb_time = tune("LightGBM", lgb_base, lgb_space,
                                Xtr_srch, ytr_srch, groups_srch, n_iter=12)

    tuned = {}
    for name, search, el in (("XGBoost", xgb_search, xgb_time),
                             ("LightGBM", lgb_search, lgb_time)):
        p = search.best_estimator_.predict_proba(Xva)[:, 1]
        m = metrics(yva, p, train_seconds=el)
        m["model"] = f"{name} (tuned)"
        tuned[name] = {"search": search, "metrics": m}
        log(f"  {name} tuned on validation: PR-AUC={m['pr_auc']:.4f}  "
            f"ROC-AUC={m['roc_auc']:.4f}")
        record(f"TUNE-{name}", name, "D_consensus_final", search.best_params_,
               "RandomizedSearchCV 3-fold StratifiedGroupKFold", m,
               f"best CV PR-AUC {search.best_score_:.5f}",
               dataset_version, len(ytr_srch), len(features))

    REPORT["tuning"] = {
        name: {"best_params": v["search"].best_params_,
               "best_cv_pr_auc": round(float(v["search"].best_score_), 5),
               "validation_metrics": v["metrics"],
               "search_seconds": round(
                   xgb_time if name == "XGBoost" else lgb_time, 1)}
        for name, v in tuned.items()}

    winner = max(tuned.items(), key=lambda kv: kv[1]["metrics"]["pr_auc"])
    winner_name, winner_info = winner
    log(f"\nWINNER: {winner_name} (validation PR-AUC "
        f"{winner_info['metrics']['pr_auc']:.5f})")

    # =====================================================================
    # 4. REFIT THE WINNER ON THE FULL TRAINING PARTITION
    # =====================================================================
    log("\nRefitting the winner on the full training partition ...")
    best_params = winner_info["search"].best_params_
    if winner_name == "XGBoost":
        final = XGBClassifier(
            **best_params, n_jobs=-1, tree_method="hist", eval_metric="aucpr",
            random_state=RANDOM_SEED,
            scale_pos_weight=pos_weight if strategy == "scale_pos_weight" else 1.0)
    else:
        final = LGBMClassifier(
            **best_params, n_jobs=-1, random_state=RANDOM_SEED, subsample_freq=1,
            verbose=-1,
            class_weight="balanced" if strategy != "none (baseline)" else None)
    t0 = time.time()
    final.fit(Xtr_full, ytr_full)
    final_time = time.time() - t0
    p_val = final.predict_proba(Xva)[:, 1]
    m_full = metrics(yva, p_val, train_seconds=final_time)
    log(f"  full-training-set validation: PR-AUC={m_full['pr_auc']:.5f}  "
        f"ROC-AUC={m_full['roc_auc']:.5f}  Brier={m_full['brier']:.5f}  "
        f"({final_time:.0f}s)")
    record("FINAL-01", winner_name, "D_consensus_final", best_params,
           "full training partition", m_full,
           "winner refitted on all training rows", dataset_version,
           len(ytr_full), len(features))

    # =====================================================================
    # 5. CALIBRATION
    # =====================================================================
    # The application shows probabilities to a clinician and ranks agents by
    # them, so calibration is a requirement, not a nicety. Isotonic regression
    # is fitted on the VALIDATION split (never on training, never on test).
    log("\nCalibrating on the validation partition (isotonic) ...")
    calib = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calib.fit(p_val, yva)
    p_cal = calib.predict(p_val)
    m_cal = metrics(yva, p_cal)
    log(f"  Brier before={m_full['brier']:.5f}  after={m_cal['brier']:.5f}")
    REPORT["calibration"] = {
        "method": "isotonic regression fitted on the validation partition",
        "brier_before": m_full["brier"], "brier_after": m_cal["brier"],
        "pr_auc_before": m_full["pr_auc"], "pr_auc_after": m_cal["pr_auc"],
        "caveat": ("Isotonic calibration is fitted on the validation split, so "
                   "validation Brier after calibration is optimistic by "
                   "construction. The honest figure is the calibration measured "
                   "on the untouched test partition in "
                   "docs/MODEL_EVALUATION.md."),
    }

    # =====================================================================
    # 6. BUNDLE
    # =====================================================================
    log("\nBuilding the model bundle ...")
    pair = (train_full.groupby(["organism_clean", "antibiotic"], observed=True)
            .size())
    pair_support = {f"{o}||{a}": int(v) for (o, a), v in pair.items()}

    defaults = {}
    for c in features:
        s = train_full[c]
        if c in cat_cols:
            mode = s.mode(dropna=True)
            defaults[c] = mode.iloc[0] if len(mode) else None
        else:
            defaults[c] = float(pd.to_numeric(s, errors="coerce").median())

    bundle = ModelBundle(
        model=final, encoder=encoder, features=features, categorical=cat_cols,
        defaults=defaults, pair_support=pair_support,
        antibiotics=sorted(df["antibiotic"].dropna().unique().tolist()),
        organisms=sorted(df["organism_clean"].dropna().unique().tolist()),
        calibrator=calib, threshold=0.5,
        metrics={"validation": m_full, "validation_calibrated": m_cal},
        metadata={
            "algorithm": winner_name,
            "hyperparameters": best_params,
            "imbalance_strategy": strategy,
            "n_train_rows": int(len(ytr_full)),
            "n_train_patients": int(train_full[PAT].nunique()),
            "n_features": len(features),
            "random_seed": RANDOM_SEED,
            "dataset_version": dataset_version,
            "target": TARGET_BINARY,
            "target_definition": "1 = non-susceptible (Resistant or Intermediate)",
            "split": "patient-grouped 70/10/20 on anon_id",
        })
    path = bundle.save(MODELS_DIR / "final_model.joblib")
    log(f"  saved {path}")

    # =====================================================================
    pd.DataFrame(EXPERIMENTS).to_csv(
        EXPERIMENTS_DIR / "experiment_log.csv", index=False)
    (EXPERIMENTS_DIR / "training_report.json").write_text(
        json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    log(f"  wrote {EXPERIMENTS_DIR / 'experiment_log.csv'} "
        f"({len(EXPERIMENTS)} experiments)")
    log(f"  wrote {EXPERIMENTS_DIR / 'training_report.json'}")


if __name__ == "__main__":
    main()
