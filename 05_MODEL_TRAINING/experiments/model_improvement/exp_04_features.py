"""EXP-04 / EXP-05: history feature engineering and selection.

Finding that shapes this experiment: every feature the brief proposes as a
candidate ALREADY EXISTS in the analytic dataset. 109 prior-history columns were
engineered in scripts/04_build_analytic_dataset.py; feature selection kept 16 of
them and discarded 93. So the honest experiment is not "invent new features" —
it is "does adding back the discarded history actually help?".

Leakage: every one of these columns is built by the same shifted-cumulative
construction (`cumsum() - own_value`, `order_time.shift(1)`) verified at
scripts/04_build_analytic_dataset.py:515-583, so each row sees only events
STRICTLY BEFORE its own culture. No new feature is created here, so no new
leakage surface is introduced. A single-feature AUC screen is run anyway as an
independent check: a discarded history column that could predict its own label
almost perfectly would betray a shift error.

Sets compared (validation only):
  A  the production 30 features
  B  A + the 14 history features the brief names explicitly
  C  A + all 93 discarded history features  (upper bound on what history offers)
  D  a reduced set, only if A/B/C justify looking at one
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from exp_common import Data, EXP_DIR, baseline, full_metrics, lgbm, log
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.models.dataset import TARGET_BINARY

log("EXP-04/05  history features")
b = baseline()
d = Data()                       # production 30
PROD = list(d.features)
allcols = list(d.df.columns)
hist_all = [c for c in allcols if c.startswith(("prior_", "has_prior"))]
discarded = [c for c in hist_all if c not in PROD]

# the features the brief names, mapped onto columns that actually exist
BRIEF = [
    "prior_med_n_drugdays_90d",            # previous antibiotic exposures
    "prior_med_n_distinct_ingredients_90d",
    "prior_res_n_events_total",            # previous resistance events
    "prior_res_n_events_365d",
    "prior_n_cultures",                    # previous cultures
    "prior_n_positive_cultures",
    "prior_med_days_since_last",           # time since previous exposure
    "prior_med_same_class_days_since",
    "prior_res_days_since_last",           # time since previous resistant event
    "prior_res_same_abx_days_since",
    "prior_res_same_org_n",                # same-organism previous resistance
    "prior_res_same_abx_n_365d",           # same-antibiotic previous resistance
    "prior_nonsusc_rate",                  # historical resistance rate (patient)
    "prior_same_abx_n_tests",
]
BRIEF = [c for c in BRIEF if c in allcols]
log(f"  history columns: {len(hist_all)} total, {len(PROD)&0} , "
    f"{len([c for c in hist_all if c in PROD])} already in the production set, "
    f"{len(discarded)} discarded by feature selection")
log(f"  brief-named candidates present in the data: {len(BRIEF)}")

# ---- leakage screen: single-feature AUC on the TRAINING partition only -------
tr = d.df[d.split == "train"]
ytr = tr[TARGET_BINARY].to_numpy()
screen = []
for c in BRIEF:
    v = pd.to_numeric(tr[c], errors="coerce").to_numpy()
    ok = ~np.isnan(v)
    auc = roc_auc_score(ytr[ok], v[ok]) if ok.sum() > 1000 and len(np.unique(ytr[ok])) > 1 else np.nan
    screen.append({"feature": c, "single_feature_auc": round(float(auc), 5),
                   "pct_missing": round(float(100 * (~ok).mean()), 2)})
sc = pd.DataFrame(screen).sort_values("single_feature_auc", ascending=False)
sc.to_csv(EXP_DIR / "exp_04_leakage_screen.csv", index=False)
log("\n  single-feature AUC screen (training partition; >0.90 would signal leakage):")
log(sc.to_string(index=False))
suspicious = sc[sc["single_feature_auc"] > 0.90]
log(f"  suspicious features: {len(suspicious)}")

SETS = {
    "A_production_30": PROD,
    "B_plus_brief_named": PROD + [c for c in BRIEF if c not in PROD],
    "C_plus_all_discarded_history": PROD + discarded,
}
results, fitted = {}, {}
for name, feats in SETS.items():
    t0 = time.time()
    dd = Data(features=feats); dd.split = d.split
    gg = dd.encode_grouped()
    m = lgbm(d.best_params); m.fit(gg["Xtr"], gg["ytr"])
    p_va = m.predict_proba(gg["Xva"])[:, 1]
    mv = full_metrics(gg["yva"], p_va, 0.5)
    mv["n_features"] = len(feats); mv["fit_seconds"] = round(time.time()-t0, 1)
    results[name] = mv; fitted[name] = (dd, gg, m, p_va)
    log(f"  [{name:30s}] n={len(feats):3d}  VAL PR-AUC={mv['pr_auc']:.5f} "
        f"ROC={mv['roc_auc']:.5f} rec={mv['recall_sensitivity']:.5f} "
        f"f1={mv['f1']:.5f}  ({mv['fit_seconds']}s)")

best = max(results, key=lambda k: results[k]["pr_auc"])
gain = results[best]["pr_auc"] - results["A_production_30"]["pr_auc"]
log(f"\n  best on validation: {best} (PR-AUC gain vs production "
    f"{gain:+.5f})")

out = {"n_history_columns": len(hist_all), "n_discarded": len(discarded),
       "brief_named_present": BRIEF, "leakage_screen": screen,
       "n_suspicious": int(len(suspicious)),
       "validation": results, "best_on_validation": best,
       "validation_pr_auc_gain": round(float(gain), 5), "applied": {}}

# ---- evaluate production set and the validation winner on untouched data ----
for name in sorted({"A_production_30", best}):
    dd, gg, m, p_va = fitted[name]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va, gg["yva"])
    mte = full_metrics(gg["yte"], cal.predict(m.predict_proba(gg["Xte"])[:, 1]), 0.5)
    tt = dd.encode_temporal()
    tm = lgbm(d.best_params); tm.fit(tt["Xtr"], tt["ytr"])
    mt = full_metrics(tt["yte"], tm.predict_proba(tt["Xte"])[:, 1], 0.5)
    out["applied"][name] = {"n_features": len(SETS[name]), "test": mte, "temporal": mt}
    log(f"  [{name:30s}] TEST PR-AUC={mte['pr_auc']:.5f} rec={mte['recall_sensitivity']:.5f} "
        f"| TEMPORAL PR-AUC={mt['pr_auc']:.5f} rec={mt['recall_sensitivity']:.5f}")

json.dump(out, open(EXP_DIR / "exp_04_features.json", "w"), indent=1, default=str)
log("EXP-04/05 done")
