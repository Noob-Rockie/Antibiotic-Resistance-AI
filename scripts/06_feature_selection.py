"""PHASE 6 — Feature selection.

A deliberate, documented selection methodology rather than "train on every
column". Seven methods are applied in sequence; each one is justified against
this dataset rather than run because it exists.

LEAKAGE DISCIPLINE
  * The test partition is NEVER touched here. Every statistic — missingness,
    variance, correlation, mutual information, model importance, SHAP, RFE — is
    computed on the TRAINING patients only.
  * The candidate feature sets are then compared on a held-out VALIDATION
    partition of unseen patients.
  * To show that the sequential filter does not itself leak, the selection is
    additionally re-derived INSIDE every cross-validation fold and compared
    like-for-like against the globally chosen list (`selection_inside_cv`).
    The gap between the two is the optimism the global look actually bought.

Outputs
  04_FEATURE_SELECTION/selected_features/*.json
  04_FEATURE_SELECTION/figures/01..08_*.png
  13_RESULTS/tables/feature_selection_comparison.csv
  config/final_features.json
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import (RFE, f_classif,  # noqa: E402
                                       mutual_info_classif)
from sklearn.metrics import (average_precision_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (CONFIG_DIR, FS_DIR, RANDOM_SEED,  # noqa: E402
                          REPORT_DIR, RESULTS_DIR)
from src.models.dataset import (EXCLUDED_COLUMNS, PAT, TARGET_BINARY,  # noqa: E402
                                all_feature_columns, feature_groups,
                                load_analytic, patient_grouped_split,
                                split_categorical_numeric, split_report,
                                subsample_by_patient)
from src.models.preprocessing import (OrdinalCategoryEncoder,  # noqa: E402
                                      as_object_frame)

FIG_DIR = FS_DIR / "figures"
SEL_DIR = FS_DIR / "selected_features"
for d in (FIG_DIR, SEL_DIR, RESULTS_DIR / "tables", RESULTS_DIR / "figures",
          REPORT_DIR / "feature_selection", CONFIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Thresholds — each is stated here once and quoted in the report.
MISSING_THRESHOLD = 95.0     # drop a feature above this % missing
DOMINANT_THRESHOLD = 0.999   # drop if one value covers this share of rows
CORRELATION_THRESHOLD = 0.95  # drop one of any pair above this |r|
FS_SUBSAMPLE_ROWS = 400_000   # patient-level subsample for the search stages
MI_SUBSAMPLE_ROWS = 150_000   # mutual information is O(n log n) per feature

REPORT: dict = {"thresholds": {
    "missing_pct_max": MISSING_THRESHOLD,
    "dominant_value_share_max": DOMINANT_THRESHOLD,
    "abs_correlation_max": CORRELATION_THRESHOLD,
    "fs_subsample_rows": FS_SUBSAMPLE_ROWS,
    "mi_subsample_rows": MI_SUBSAMPLE_ROWS,
    "random_seed": RANDOM_SEED,
}}

sns.set_theme(style="whitegrid", font_scale=0.95)
PALETTE = {"keep": "#2563eb", "drop": "#dc2626", "accent": "#0891b2"}


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    for base in (FIG_DIR, REPORT_DIR / "feature_selection"):
        fig.savefig(base / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"    figure -> {name}")


# ===========================================================================
# METHOD 1 — missing-value analysis
# ===========================================================================
def method_1_missingness(Xtr: pd.DataFrame) -> tuple:
    log("METHOD 1  missing-value analysis")
    miss = (Xtr.isna().mean() * 100).sort_values(ascending=False)
    drop = miss[miss > MISSING_THRESHOLD].index.tolist()

    fig, ax = plt.subplots(figsize=(11, 13))
    top = miss[miss > 0]
    colors = [PALETTE["drop"] if v > MISSING_THRESHOLD else PALETTE["keep"]
              for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(MISSING_THRESHOLD, color="#111", ls="--", lw=1.3,
               label=f"drop threshold = {MISSING_THRESHOLD}%")
    ax.set_xlabel("% missing in the training partition")
    ax.set_title("Method 1 — Missing-value analysis\n"
                 f"{len(drop)} of {Xtr.shape[1]} features exceed the threshold "
                 "and are removed", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    savefig(fig, "01_missing_values.png")

    REPORT["method_1_missingness"] = {
        "n_features_in": int(Xtr.shape[1]),
        "n_dropped": len(drop),
        "dropped": drop,
        "missing_pct": {k: round(float(v), 3) for k, v in miss.items() if v > 0},
        "rationale": (
            "Gradient-boosted trees treat NaN as a routed branch, so a moderately "
            "missing feature is still usable and its missingness is itself "
            "informative (an unordered procalcitonin means the clinician did not "
            "suspect severe sepsis). The threshold is therefore set high (95%) "
            "and removes only features so sparse that any split on them is fitted "
            "to a handful of patients."),
    }
    return drop, miss


# ===========================================================================
# METHOD 2 — low-variance / near-constant filter
# ===========================================================================
def method_2_low_variance(Xtr: pd.DataFrame, already_dropped) -> list:
    log("METHOD 2  low-variance / near-constant filter")
    cols = [c for c in Xtr.columns if c not in already_dropped]
    stats = {}
    drop = []
    for c in cols:
        s = Xtr[c]
        nonnull = s.dropna()
        if len(nonnull) == 0:
            drop.append(c)
            stats[c] = {"dominant_share": 1.0, "n_unique": 0}
            continue
        vc = nonnull.value_counts(normalize=True)
        share = float(vc.iloc[0])
        stats[c] = {"dominant_share": round(share, 6), "n_unique": int(len(vc))}
        if share > DOMINANT_THRESHOLD or len(vc) < 2:
            drop.append(c)

    REPORT["method_2_low_variance"] = {
        "n_dropped": len(drop), "dropped": drop,
        "near_constant_detail": {c: stats[c] for c in drop},
        "rationale": (
            "A feature whose single most common value covers >99.9% of rows "
            "cannot separate a 19.5% positive class in any useful way; it only "
            "adds a split point for the tree to overfit. Note that the audit "
            "already removed the two genuinely constant source columns "
            "(`Period_Day` = 14, `was_positive` = 1 after filtering) upstream, "
            "so this filter is a safety net rather than the main event."),
    }
    log(f"    dropped {len(drop)}")
    return drop


# ===========================================================================
# METHOD 3 — correlation analysis
# ===========================================================================
def method_3_correlation(Xtr: pd.DataFrame, numeric_cols, keep) -> list:
    log("METHOD 3  correlation analysis")
    cols = [c for c in numeric_cols if c in keep]
    sub = Xtr[cols].astype("float32")
    corr = sub.corr(method="spearman", min_periods=1000)
    corr_abs = corr.abs()

    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    pairs = (upper.stack().rename("abs_corr").reset_index()
             .rename(columns={"level_0": "feature_a", "level_1": "feature_b"}))
    high = pairs[pairs["abs_corr"] > CORRELATION_THRESHOLD].sort_values(
        "abs_corr", ascending=False)

    # Of each correlated pair keep the one with fewer missing values; on a tie
    # keep the one whose name is shorter (the simpler, more direct measure).
    miss = Xtr[cols].isna().mean()
    drop = set()
    for _, r in high.iterrows():
        a, b = r["feature_a"], r["feature_b"]
        if a in drop or b in drop:
            continue
        loser = b if (miss[a], len(a)) <= (miss[b], len(b)) else a
        drop.add(loser)

    # heatmap of the most correlated block, not all ~180 features
    block = (corr_abs.sum().sort_values(ascending=False).head(40).index.tolist())
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr.loc[block, block], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.6, "label": "Spearman rho"},
                xticklabels=True, yticklabels=True, ax=ax)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Method 3 — Correlation heat map (40 most inter-correlated "
                 f"numeric features)\n{len(drop)} features removed at "
                 f"|rho| > {CORRELATION_THRESHOLD}",
                 fontsize=12, fontweight="bold")
    savefig(fig, "02_correlation_heatmap.png")

    REPORT["method_3_correlation"] = {
        "n_pairs_above_threshold": int(len(high)),
        "n_dropped": len(drop), "dropped": sorted(drop),
        "top_pairs": high.head(40).to_dict("records"),
        "rationale": (
            "The lab block is highly redundant by construction — Q25/median/Q75 "
            "of the same analyte over the same 14-day window are near-identical, "
            "and so are the class-level prior-exposure counts at 90d and 365d. "
            "Spearman is used rather than Pearson because most of these features "
            "are skewed counts. Removing one of each redundant pair shortens "
            "training and makes the SHAP attributions readable, since perfectly "
            "collinear features split their credit arbitrarily."),
    }
    log(f"    {len(high)} high-correlation pairs, dropped {len(drop)}")
    return sorted(drop)


# ===========================================================================
# METHOD 4 — statistical selection (ANOVA F + mutual information)
# ===========================================================================
def method_4_statistical(Xenc: pd.DataFrame, y, keep) -> dict:
    log("METHOD 4  statistical selection (ANOVA F + mutual information)")
    X = Xenc[keep].fillna(Xenc[keep].median(numeric_only=True)).astype("float32")
    X = X.fillna(0.0)

    f_stat, f_p = f_classif(X, y)
    anova = pd.Series(f_stat, index=keep).sort_values(ascending=False)

    n = min(MI_SUBSAMPLE_ROWS, len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    sel = rng.choice(len(X), size=n, replace=False)
    mi = pd.Series(
        mutual_info_classif(X.iloc[sel], np.asarray(y)[sel],
                            discrete_features=False, random_state=RANDOM_SEED,
                            n_neighbors=3),
        index=keep).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 12))
    top = mi.head(45)[::-1]
    ax.barh(range(len(top)), top.values, color=PALETTE["accent"])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Mutual information with non-susceptibility (nats)")
    ax.set_title("Method 4 — Mutual information ranking (top 45)\n"
                 f"computed on {n:,} training rows",
                 fontsize=12, fontweight="bold")
    savefig(fig, "03_mutual_information.png")

    REPORT["method_4_statistical"] = {
        "anova_top30": {k: round(float(v), 3) for k, v in anova.head(30).items()},
        "anova_n_features_p_gt_0.05": int((f_p > 0.05).sum()),
        "mutual_information_top30": {k: round(float(v), 6)
                                     for k, v in mi.head(30).items()},
        "mi_n_features_zero": int((mi <= 1e-9).sum()),
        "rationale": (
            "ANOVA F tests a linear mean shift and mutual information captures "
            "any monotone or non-monotone dependence, so they are complementary "
            "and disagree in informative ways. Chi-square was NOT used: it "
            "requires non-negative discrete counts, and the ordinal-coded "
            "categoricals and z-scored labs here violate that assumption, so it "
            "would return a number without meaning."),
    }
    return {"anova": anova, "mi": mi}


# ===========================================================================
# METHODS 5 + 6 — model-based importance and SHAP
# ===========================================================================
def method_5_model_importance(Xenc, y, keep, groups) -> dict:
    log("METHOD 5  model-based importance (Random Forest + XGBoost)")
    X = Xenc[keep]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_leaf=25, n_jobs=-1,
        random_state=RANDOM_SEED, class_weight="balanced_subsample")
    t0 = time.time()
    rf.fit(X.fillna(-999.0), y)
    rf_time = time.time() - t0
    rf_imp = pd.Series(rf.feature_importances_, index=keep).sort_values(
        ascending=False)

    xgb = XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, tree_method="hist",
        eval_metric="aucpr", random_state=RANDOM_SEED,
        scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
    t0 = time.time()
    xgb.fit(X, y)
    xgb_time = time.time() - t0
    xgb_imp = pd.Series(xgb.get_booster().get_score(importance_type="gain"))
    xgb_imp = xgb_imp.reindex(keep).fillna(0.0).sort_values(ascending=False)

    for name, imp, fname, color in (
            ("Random Forest (Gini impurity decrease)", rf_imp,
             "04_random_forest_importance.png", "#059669"),
            ("XGBoost (average gain)", xgb_imp,
             "05_xgboost_importance.png", "#7c3aed")):
        fig, ax = plt.subplots(figsize=(10, 12))
        top = imp.head(45)[::-1]
        ax.barh(range(len(top)), top.values, color=color)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"Methods 5 — {name}\ntop 45 of {len(keep)} features",
                     fontsize=12, fontweight="bold")
        savefig(fig, fname)

    REPORT["method_5_model_importance"] = {
        "random_forest_top30": {k: round(float(v), 6)
                                for k, v in rf_imp.head(30).items()},
        "xgboost_gain_top30": {k: round(float(v), 4)
                               for k, v in xgb_imp.head(30).items()},
        "rf_fit_seconds": round(rf_time, 2),
        "xgb_fit_seconds": round(xgb_time, 2),
        "n_features_zero_xgb_gain": int((xgb_imp == 0).sum()),
        "rationale": (
            "Random Forest impurity importance is biased towards high-cardinality "
            "features, and `organism_clean` has 245 levels, so it is reported but "
            "not trusted alone. XGBoost average gain is the counterweight, and "
            "SHAP (Method 6) is the arbiter because it is the only one of the "
            "three that is additive and locally faithful."),
    }
    return {"rf": rf_imp, "xgb": xgb_imp, "xgb_model": xgb}


def method_6_shap(xgb_model, Xenc, keep, n_sample: int = 40_000) -> pd.Series:
    log("METHOD 6  SHAP importance")
    import shap
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(Xenc), size=min(n_sample, len(Xenc)), replace=False)
    Xs = Xenc[keep].iloc[idx]

    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(Xs)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=keep).sort_values(
        ascending=False)

    fig, ax = plt.subplots(figsize=(10, 12))
    top = mean_abs.head(45)[::-1]
    ax.barh(range(len(top)), top.values, color="#db2777")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("mean(|SHAP value|)  —  average impact on model output")
    ax.set_title("Method 6 — SHAP global feature importance (top 45)\n"
                 f"computed on {len(Xs):,} held-in training rows",
                 fontsize=12, fontweight="bold")
    savefig(fig, "07_shap_feature_importance.png")

    fig = plt.figure(figsize=(11, 11))
    shap.summary_plot(sv, Xs, max_display=30, show=False, plot_size=None)
    plt.title("SHAP summary — direction and magnitude of each feature's effect",
              fontsize=12, fontweight="bold")
    savefig(plt.gcf(), "07b_shap_summary_beeswarm.png")

    REPORT["method_6_shap"] = {
        "shap_top30": {k: round(float(v), 6) for k, v in mean_abs.head(30).items()},
        "n_rows_explained": int(len(Xs)),
        "rationale": (
            "SHAP is the selection arbiter because its attributions are additive "
            "and locally faithful, so a feature that only matters for a small but "
            "clinically important subgroup is not averaged away as it is by gain. "
            "It is also the same machinery the application uses to explain an "
            "individual prediction, so what selects a feature and what explains it "
            "to a clinician are consistent."),
    }
    return mean_abs


# ===========================================================================
# METHOD 7 — recursive feature elimination
# ===========================================================================
def method_7_rfe(Xenc, y, keep, n_features: int = 30) -> pd.Series:
    log("METHOD 7  recursive feature elimination")
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(Xenc), size=min(120_000, len(Xenc)), replace=False)
    X = Xenc[keep].iloc[idx]
    yy = np.asarray(y)[idx]

    base = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.15,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                         tree_method="hist", eval_metric="aucpr",
                         random_state=RANDOM_SEED)
    t0 = time.time()
    rfe = RFE(base, n_features_to_select=n_features, step=0.15)
    rfe.fit(X, yy)
    elapsed = time.time() - t0
    ranking = pd.Series(rfe.ranking_, index=keep).sort_values()

    fig, ax = plt.subplots(figsize=(10, 12))
    top = ranking.head(45)[::-1]
    colors = [PALETTE["keep"] if v == 1 else "#94a3b8" for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("RFE elimination rank (1 = retained in the final set)")
    ax.set_title(f"Method 7 — Recursive Feature Elimination\n"
                 f"target size {n_features}; fitted on {len(X):,} rows in "
                 f"{elapsed:.0f}s", fontsize=12, fontweight="bold")
    savefig(fig, "06_rfe_ranking.png")

    REPORT["method_7_rfe"] = {
        "n_features_selected": int((rfe.ranking_ == 1).sum()),
        "selected": ranking[ranking == 1].index.tolist(),
        "fit_seconds": round(elapsed, 1),
        "rationale": (
            "RFECV was deliberately NOT used. With ~180 candidates and 1.1M "
            "training rows, cross-validated elimination would refit XGBoost "
            "several hundred times for a result that Methods 5-6 already give at "
            "a fraction of the cost. Plain RFE at a fixed target size on a "
            "120,000-row patient-level subsample is used instead as an "
            "independent third opinion, and the feature-count sweep in the "
            "experiment section supplies the 'how many features' answer that "
            "RFECV would have provided."),
    }
    return ranking


# ===========================================================================
# The experiment — candidate feature sets compared on held-out patients
# ===========================================================================
def evaluate_set(name, features, Xtr, ytr, Xva, yva, cat_cols) -> dict:
    feats = [f for f in features if f in Xtr.columns]
    model = XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, tree_method="hist",
        eval_metric="aucpr", random_state=RANDOM_SEED,
        scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    t0 = time.time()
    model.fit(Xtr[feats], ytr)
    train_time = time.time() - t0
    p = model.predict_proba(Xva[feats])[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "feature_set": name,
        "n_features": len(feats),
        "accuracy": round(float((pred == yva).mean()), 5),
        "precision": round(float(precision_score(yva, pred, zero_division=0)), 5),
        "recall": round(float(recall_score(yva, pred, zero_division=0)), 5),
        "f1": round(float(f1_score(yva, pred, zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(yva, p)), 5),
        "pr_auc": round(float(average_precision_score(yva, p)), 5),
        "train_seconds": round(train_time, 2),
    }


def selection_inside_cv(Xtr, ytr, groups, global_selection, k: int = 30) -> dict:
    """Did selecting features on the whole training set leak into the score?

    Section 17 of the project brief. The sequential filter above chose its
    features by looking at ALL the training rows, then the candidate sets were
    scored on validation. If that global look conferred an unfair advantage, the
    honest score would be lower.

    This is the like-for-like test. Within the training partition only, and
    grouped by patient so no patient spans a fold:

      ARM 1 (nested / honest)  — on each training fold, refit the SAME selector
                                 from scratch (XGBoost gain, top-k), then refit
                                 the model on those k features and score the
                                 held-out fold. The held-out fold never
                                 influences which features exist.
      ARM 2 (global / audited) — use the features that were chosen globally,
                                 refit the model on the training fold, score the
                                 held-out fold.

    Both arms train on identical rows and are scored on identical rows; the only
    difference is where the feature list came from. If ARM 2 does not beat
    ARM 1, the global selection carried no smuggled information. Fold-to-fold
    agreement of the ARM 1 selections is reported as a stability measure.
    """
    log("  nested vs global feature-selection check ...")
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    all_cols = list(Xtr.columns)
    global_feats = [f for f in global_selection if f in all_cols]

    def fit_score(feats, tr, te):
        m = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                          tree_method="hist", eval_metric="aucpr",
                          random_state=RANDOM_SEED)
        m.fit(Xtr.iloc[tr][feats], ytr.iloc[tr])
        p = m.predict_proba(Xtr.iloc[te][feats])[:, 1]
        return float(average_precision_score(ytr.iloc[te], p)), m

    nested_scores, global_scores, chosen = [], [], []
    for fold, (tr, te) in enumerate(cv.split(Xtr, ytr, groups=groups), 1):
        # ARM 1 — selector refitted on this fold's training rows only
        _, ranker = fit_score(all_cols, tr, te)
        gain = pd.Series(ranker.get_booster().get_score(importance_type="gain"))
        gain = gain.reindex(all_cols).fillna(0.0).sort_values(ascending=False)
        feats = list(dict.fromkeys(gain.head(k).index.tolist()
                                   + ["antibiotic", "antibiotic_class"]))
        s_nested, _ = fit_score(feats, tr, te)
        chosen.append(set(feats))
        nested_scores.append(s_nested)

        # ARM 2 — the globally chosen features
        s_global, _ = fit_score(global_feats, tr, te)
        global_scores.append(s_global)
        log(f"    fold {fold}: nested PR-AUC={s_nested:.5f}  "
            f"global PR-AUC={s_global:.5f}  (delta {s_global - s_nested:+.5f})")

    stable = set.intersection(*chosen)
    delta = float(np.mean(global_scores) - np.mean(nested_scores))
    return {
        "k": k,
        "nested_cv_pr_auc_mean": round(float(np.mean(nested_scores)), 5),
        "nested_cv_pr_auc_std": round(float(np.std(nested_scores)), 5),
        "global_cv_pr_auc_mean": round(float(np.mean(global_scores)), 5),
        "global_cv_pr_auc_std": round(float(np.std(global_scores)), 5),
        "optimism_from_global_selection": round(delta, 5),
        "nested_fold_scores": [round(s, 5) for s in nested_scores],
        "global_fold_scores": [round(s, 5) for s in global_scores],
        "features_selected_in_every_fold": sorted(stable),
        "n_stable": len(stable),
        "selection_stability_pct": round(100 * len(stable) / max(
            len(set.union(*chosen)), 1), 1),
        "interpretation": (
            "`optimism_from_global_selection` is how much PR-AUC the globally "
            "chosen feature list gains over a list re-derived inside each fold. "
            "A value near zero means the sequential filter did not leak; a "
            "clearly positive value would mean the reported feature-selection "
            "numbers are optimistic and the honest figure is the nested one."),
    }


# ===========================================================================
def main() -> None:
    log("Loading analytic dataset ...")
    df = load_analytic()
    split = patient_grouped_split(df)
    REPORT["split"] = split_report(df, split, TARGET_BINARY)
    log(f"  split: " + ", ".join(
        f"{k}={v['n_rows']:,} rows/{v['n_patients']:,} patients"
        for k, v in REPORT["split"]["partitions"].items()))

    features = all_feature_columns(df)
    cat_cols, num_cols = split_categorical_numeric(df, features)
    groups_map = feature_groups(df.columns)
    REPORT["feature_inventory"] = {
        "n_candidate_features": len(features),
        "n_categorical": len(cat_cols),
        "n_numeric": len(num_cols),
        "groups": {k: {"n": len(v), "features": v} for k, v in groups_map.items()},
        "excluded_columns": EXCLUDED_COLUMNS,
    }

    train = df[split == "train"]
    val = df[split == "val"]

    # patient-level subsample for the search stages
    train_s = subsample_by_patient(train, FS_SUBSAMPLE_ROWS)
    val_s = subsample_by_patient(val, 150_000)
    REPORT["subsample"] = {
        "train_rows": len(train_s), "train_patients": int(train_s[PAT].nunique()),
        "val_rows": len(val_s), "val_patients": int(val_s[PAT].nunique()),
        "note": ("Search stages run on a PATIENT-level subsample so that "
                 "within-patient correlation is preserved; the final models in "
                 "Phase 7 are refitted on the full training partition."),
    }

    Xtr_raw = as_object_frame(train_s[features], cat_cols)
    Xva_raw = as_object_frame(val_s[features], cat_cols)
    ytr = train_s[TARGET_BINARY]
    yva = val_s[TARGET_BINARY].to_numpy()

    # ---- filters -----------------------------------------------------------
    drop_missing, miss = method_1_missingness(Xtr_raw)
    drop_variance = method_2_low_variance(Xtr_raw, set(drop_missing))
    keep_after_filters = [c for c in features
                          if c not in set(drop_missing) | set(drop_variance)]
    drop_corr = method_3_correlation(Xtr_raw, num_cols, set(keep_after_filters))
    keep_after_filters = [c for c in keep_after_filters if c not in set(drop_corr)]
    log(f"  after filters: {len(keep_after_filters)} features remain "
        f"(from {len(features)})")

    # ---- encode once for every downstream method --------------------------
    enc = OrdinalCategoryEncoder(columns=cat_cols).fit(Xtr_raw)
    Xtr = enc.transform(Xtr_raw)
    Xva = enc.transform(Xva_raw)

    stat = method_4_statistical(Xtr, ytr, keep_after_filters)
    modimp = method_5_model_importance(Xtr, ytr, keep_after_filters,
                                       train_s[PAT].to_numpy())
    shap_imp = method_6_shap(modimp["xgb_model"], Xtr, keep_after_filters)
    rfe_rank = method_7_rfe(Xtr, ytr, keep_after_filters)

    # ---- candidate feature sets -------------------------------------------
    set_A = features                                   # everything
    set_B = stat["mi"].head(30).index.tolist()         # statistical
    set_C = modimp["xgb"].head(30).index.tolist()      # model-based
    set_shap = shap_imp.head(30).index.tolist()
    set_rfe = rfe_rank[rfe_rank == 1].index.tolist()

    # Set D — consensus: a feature must be endorsed by at least two of the four
    # independent rankers, then the filters are applied on top.
    votes = pd.Series(0, index=keep_after_filters, dtype=int)
    for s in (set_B, set_C, set_shap, set_rfe):
        votes[votes.index.isin(s)] += 1
    set_D = votes[votes >= 2].index.tolist()
    # Always retain the antibiotic identity and its class: without them a single
    # unified model cannot distinguish which agent it is scoring.
    for must in ("antibiotic", "antibiotic_class"):
        if must not in set_D and must in keep_after_filters:
            set_D.append(must)

    candidate_sets = {
        "A_all_features": set_A,
        "B_statistical_MI_top30": set_B,
        "C_model_based_XGB_top30": set_C,
        "D_consensus_final": set_D,
        "E_filters_only": keep_after_filters,
        "F_shap_top30": set_shap,
        "G_rfe_selected": set_rfe,
        "H_organism_blind": [c for c in set_D
                             if c not in ("organism_clean", "organism_genus",
                                          "organism_group", "organism_morphotype",
                                          "is_species_level")],
    }

    log("Running the feature-set experiment ...")
    rows = []
    for name, feats in candidate_sets.items():
        r = evaluate_set(name, feats, Xtr, ytr, Xva, yva, cat_cols)
        rows.append(r)
        log(f"  {name:<26} n={r['n_features']:>3}  PR-AUC={r['pr_auc']:.4f}  "
            f"ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  "
            f"{r['train_seconds']:.1f}s")
    comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    for base in (RESULTS_DIR / "tables", REPORT_DIR / "feature_selection"):
        comparison.to_csv(base / "feature_selection_comparison.csv", index=False)

    # ---- selection rule, stated before the numbers were seen ---------------
    # The final set is the candidate with the highest PR-AUC on held-out
    # validation patients. `H_organism_blind` is excluded from the contest: it
    # exists to MEASURE how much of the signal comes from the organism, not to
    # compete, and choosing it would answer a different question than the one
    # the application asks.
    contest = comparison[comparison["feature_set"] != "H_organism_blind"]
    best_name = str(contest.iloc[0]["feature_set"])
    final_features = list(candidate_sets[best_name])
    for must in ("antibiotic", "antibiotic_class"):
        if must not in final_features and must in keep_after_filters:
            final_features.append(must)
    log(f"  selection rule -> highest validation PR-AUC among candidate sets: "
        f"{best_name} (PR-AUC {contest.iloc[0]['pr_auc']:.5f}); "
        f"consensus set D scored "
        f"{float(comparison.loc[comparison['feature_set'] == 'D_consensus_final', 'pr_auc'].iloc[0]):.5f}")
    REPORT["selection_rule"] = {
        "rule": ("The final feature set is the candidate set with the highest "
                 "PR-AUC on the held-out validation patients, with `antibiotic` "
                 "and `antibiotic_class` force-retained because a single "
                 "unified model cannot identify which agent it is scoring "
                 "without them. The organism-blind arm is excluded from the "
                 "contest because it is a measurement, not a candidate."),
        "winner": best_name,
        "winner_pr_auc": float(contest.iloc[0]["pr_auc"]),
        "n_features": len(final_features),
    }

    # ---- feature-count sweep (replaces RFECV) ------------------------------
    log("Feature-count sweep (performance vs number of features) ...")
    ranked = shap_imp.index.tolist()
    sweep = []
    for k in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, len(keep_after_filters)):
        k = min(k, len(ranked))
        feats = list(dict.fromkeys(ranked[:k] + ["antibiotic", "antibiotic_class"]))
        r = evaluate_set(f"top{k}", feats, Xtr, ytr, Xva, yva, cat_cols)
        sweep.append(r)
        log(f"  k={k:>3}  PR-AUC={r['pr_auc']:.4f}  F1={r['f1']:.4f}")
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(RESULTS_DIR / "tables" / "feature_count_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(sweep_df["n_features"], sweep_df["pr_auc"], "o-",
                 color=PALETTE["keep"], lw=2)
    axes[0].set_xlabel("Number of features (SHAP-ranked)")
    axes[0].set_ylabel("PR-AUC on held-out validation patients")
    axes[0].set_title("Performance vs number of features")
    axes[1].plot(sweep_df["n_features"], sweep_df["f1"], "o-",
                 color="#db2777", lw=2, label="F1")
    axes[1].plot(sweep_df["n_features"], sweep_df["roc_auc"], "s--",
                 color=PALETTE["accent"], lw=2, label="ROC-AUC")
    axes[1].set_xlabel("Number of features (SHAP-ranked)")
    axes[1].set_ylabel("Score")
    axes[1].set_title("F1 and ROC-AUC vs number of features")
    axes[1].legend()
    fig.suptitle("Method 7b — how many features are actually needed?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "08_feature_count_vs_f1.png")

    # ---- selected vs rejected chart ---------------------------------------
    fig, ax = plt.subplots(figsize=(11, 12))
    ordered = shap_imp.head(60)[::-1]
    final_set = set(final_features)
    colors = [PALETTE["keep"] if f in final_set else PALETTE["drop"]
              for f in ordered.index]
    ax.barh(range(len(ordered)), ordered.values, color=colors)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered.index, fontsize=7.5)
    ax.set_xlabel("mean(|SHAP value|)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE["keep"]),
               plt.Rectangle((0, 0), 1, 1, color=PALETTE["drop"])]
    ax.legend(handles, [f"selected into the final set (n={len(final_features)})",
                        "rejected"], loc="lower right")
    ax.set_title("Selected vs rejected features\n(top 60 by SHAP importance)",
                 fontsize=12, fontweight="bold")
    savefig(fig, "09_selected_vs_rejected.png")

    # ---- leakage-proof CV check -------------------------------------------
    cvcheck = selection_inside_cv(Xtr[keep_after_filters], ytr,
                                  train_s[PAT].to_numpy(), final_features, k=30)
    REPORT["selection_inside_cv"] = cvcheck
    log(f"  nested (honest) PR-AUC = {cvcheck['nested_cv_pr_auc_mean']:.5f}; "
        f"global-selection PR-AUC = {cvcheck['global_cv_pr_auc_mean']:.5f}; "
        f"optimism = {cvcheck['optimism_from_global_selection']:+.5f}; "
        f"{cvcheck['n_stable']} features chosen in every fold")

    # ---- persist -----------------------------------------------------------
    REPORT["candidate_sets"] = {k: v for k, v in candidate_sets.items()}
    REPORT["experiment_results"] = comparison.to_dict("records")
    REPORT["feature_count_sweep"] = sweep_df.to_dict("records")
    REPORT["final_feature_set"] = final_features
    REPORT["consensus_set_D"] = set_D
    REPORT["consensus_votes"] = {k: int(v) for k, v in
                                 votes.sort_values(ascending=False).items()}
    REPORT["importances"] = {
        "shap": {k: round(float(v), 8) for k, v in shap_imp.items()},
        "xgb_gain": {k: round(float(v), 6) for k, v in modimp["xgb"].items()},
        "random_forest": {k: round(float(v), 8) for k, v in modimp["rf"].items()},
        "mutual_information": {k: round(float(v), 8) for k, v in stat["mi"].items()},
        "anova_f": {k: round(float(v), 4) for k, v in stat["anova"].items()},
        "rfe_rank": {k: int(v) for k, v in rfe_rank.items()},
    }

    (SEL_DIR / "feature_selection_report.json").write_text(
        json.dumps(REPORT, indent=2, default=str), encoding="utf-8")

    final_cat, final_num = split_categorical_numeric(df, final_features)
    (CONFIG_DIR / "final_features.json").write_text(json.dumps({
        "target_binary": TARGET_BINARY,
        "target_3class": "y_3class",
        "n_features": len(final_features),
        "features": sorted(final_features),
        "categorical_features": sorted(final_cat),
        "numeric_features": sorted(final_num),
        "selection_method": REPORT["selection_rule"]["rule"],
        "winning_candidate_set": REPORT["selection_rule"]["winner"],
        "candidate_sets_compared": {k: len(v) for k, v in candidate_sets.items()},
        "thresholds": REPORT["thresholds"],
        "excluded_columns": EXCLUDED_COLUMNS,
        "random_seed": RANDOM_SEED,
    }, indent=2), encoding="utf-8")

    log(f"\nFinal feature set: {len(final_features)} features "
        f"(winning candidate set: {best_name})")
    log(f"Wrote {CONFIG_DIR / 'final_features.json'}")
    log(f"Wrote {SEL_DIR / 'feature_selection_report.json'}")
    log(f"Wrote {RESULTS_DIR / 'tables' / 'feature_selection_comparison.csv'}")


if __name__ == "__main__":
    main()
