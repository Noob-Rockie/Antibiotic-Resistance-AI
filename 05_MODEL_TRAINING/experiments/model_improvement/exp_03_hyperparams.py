"""EXP-03: modest LightGBM hyperparameter search.

Deliberately small: the production model was already tuned by
scripts/07_train_models.py (RandomizedSearchCV, 12 iterations). The question
here is only whether a further modest search finds meaningfully better
settings, not whether an exhaustive search could.

Selection: GroupKFold CV by patient on a patient-level subsample, scored by
average precision. The winner is refitted on the FULL training partition and
evaluated once on the untouched test partition and the temporal holdout.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from exp_common import Data, EXP_DIR, baseline, full_metrics, lgbm, log
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.paths import RANDOM_SEED
from src.models.dataset import PAT, TARGET_BINARY, subsample_by_patient

N_ITER, SEARCH_ROWS = 24, 250_000
log("EXP-03  hyperparameter search")
b = baseline(); d = Data(); g = d.encode_grouped()

# patient-level subsample of the TRAINING partition only
train_df = d.df[d.split == "train"]
sub = subsample_by_patient(train_df, SEARCH_ROWS)
from src.models.preprocessing import as_object_frame
Xs = g["enc"].transform(as_object_frame(sub[d.features], d.cat_cols))
ys = sub[TARGET_BINARY].to_numpy(); gs = sub[PAT].to_numpy()
log(f"  search on {len(ys):,} rows / {len(np.unique(gs)):,} patients, "
    f"{N_ITER} iterations, GroupKFold(3) by patient")

space = {
    "n_estimators":     [300, 500, 800, 1200],
    "learning_rate":    [0.02, 0.03, 0.05, 0.08],
    "num_leaves":       [63, 127, 255, 511],
    "max_depth":        [-1, 8, 12, 16],
    "min_child_samples":[10, 20, 50, 100],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "reg_alpha":        [0.0, 0.1, 0.5, 1.0],
    "reg_lambda":       [0.0, 0.5, 1.5, 5.0],
}
t0 = time.time()
search = RandomizedSearchCV(
    lgbm({}), space, n_iter=N_ITER, scoring="average_precision",
    cv=GroupKFold(n_splits=3), random_state=RANDOM_SEED, n_jobs=1,
    refit=False, verbose=0)
search.fit(Xs, ys, groups=gs)
log(f"  search finished in {time.time()-t0:.0f}s")
log(f"  best CV PR-AUC={search.best_score_:.5f}  params={search.best_params_}")

# baseline params scored under the IDENTICAL CV protocol, for a fair comparison
from sklearn.model_selection import cross_val_score
base_cv = cross_val_score(lgbm(d.best_params), Xs, ys, groups=gs,
                          cv=GroupKFold(n_splits=3),
                          scoring="average_precision", n_jobs=1)
log(f"  production params under the same CV: {base_cv.mean():.5f} "
    f"(+/-{base_cv.std():.5f})")

out = {"n_iter": N_ITER, "search_rows": int(len(ys)),
       "best_params": search.best_params_,
       "best_cv_pr_auc": round(float(search.best_score_), 5),
       "production_params_cv_pr_auc": round(float(base_cv.mean()), 5),
       "production_params_cv_std": round(float(base_cv.std()), 5),
       "cv_results": [], "applied": {}}
cvr = pd.DataFrame(search.cv_results_)[["mean_test_score","std_test_score","params"]]
cvr = cvr.sort_values("mean_test_score", ascending=False)
cvr.to_csv(EXP_DIR / "exp_03_cv_results.csv", index=False)

# refit winner on the FULL training partition, evaluate once
t = d.encode_temporal()
for name, params in [("production", d.best_params), ("searched", search.best_params_)]:
    m = lgbm(params); m.fit(g["Xtr"], g["ytr"])
    p_va = m.predict_proba(g["Xva"])[:, 1]
    mv = full_metrics(g["yva"], p_va, 0.5)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va, g["yva"])
    mte = full_metrics(g["yte"], cal.predict(m.predict_proba(g["Xte"])[:, 1]), 0.5)
    tm = lgbm(params); tm.fit(t["Xtr"], t["ytr"])
    mt = full_metrics(t["yte"], tm.predict_proba(t["Xte"])[:, 1], 0.5)
    out["applied"][name] = {"params": params, "validation": mv,
                            "test": mte, "temporal": mt}
    log(f"  [{name:10s}] VAL PR-AUC={mv['pr_auc']:.5f} | TEST PR-AUC={mte['pr_auc']:.5f} "
        f"rec={mte['recall_sensitivity']:.5f} | TEMPORAL PR-AUC={mt['pr_auc']:.5f} "
        f"rec={mt['recall_sensitivity']:.5f}")

json.dump(out, open(EXP_DIR / "exp_03_hyperparams.json", "w"), indent=1, default=str)
log("EXP-03 done")
