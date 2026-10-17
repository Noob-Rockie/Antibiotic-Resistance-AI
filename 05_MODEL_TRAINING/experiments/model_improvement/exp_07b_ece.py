"""EXP-07b: decompose the calibration result.

Brier score conflates calibration with discrimination, so it cannot settle
whether isotonic calibration is earning its cost. Expected Calibration Error
(ECE) and Maximum Calibration Error (MCE) measure calibration alone.

Also quantifies WHY isotonic costs PR-AUC: it is a step function, so it maps
many distinct probabilities onto the same value, destroying the tie-breaking
the application relies on when it ranks 54 antibiotics.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from exp_common import Data, EXP_DIR, full_metrics, lgbm, log

def ece_mce(y, p, n_bins=20):
    """Quantile-binned expected / maximum calibration error."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    e, m, N = 0.0, 0.0, len(y)
    for b in range(n_bins):
        s = idx == b
        if s.sum() == 0: continue
        gap = abs(y[s].mean() - p[s].mean())
        e += (s.sum() / N) * gap
        m = max(m, gap)
    return float(e), float(m)

log("EXP-07b  calibration decomposition")
d = Data(); g = d.encode_grouped()
m = lgbm(d.best_params); m.fit(g["Xtr"], g["ytr"])
p_va = m.predict_proba(g["Xva"])[:, 1]
p_raw = m.predict_proba(g["Xte"])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va, g["yva"])
plat = LogisticRegression(max_iter=1000).fit(p_va.reshape(-1, 1), g["yva"])

variants = {"uncalibrated": p_raw,
            "isotonic (production)": iso.predict(p_raw),
            "platt / sigmoid": plat.predict_proba(p_raw.reshape(-1, 1))[:, 1]}
out = {}
y = g["yte"]
for name, p in variants.items():
    e, mx = ece_mce(y, p)
    mm = full_metrics(y, p, 0.5)
    n_unique = int(len(np.unique(np.round(p, 6))))
    out[name] = {"ece": round(e, 6), "mce": round(mx, 6),
                 "brier": mm["brier"], "pr_auc": mm["pr_auc"],
                 "roc_auc": mm["roc_auc"],
                 "n_distinct_probabilities": n_unique}
    log(f"  [{name:22s}] ECE={e:.6f}  MCE={mx:.6f}  Brier={mm['brier']:.6f}  "
        f"PR-AUC={mm['pr_auc']:.5f}  distinct probabilities={n_unique:,}")

base = out["uncalibrated"]
iso_o = out["isotonic (production)"]
log(f"\n  isotonic vs uncalibrated:")
log(f"    ECE     {base['ece']:.6f} -> {iso_o['ece']:.6f}  "
    f"({iso_o['ece']-base['ece']:+.6f})")
log(f"    Brier   {base['brier']:.6f} -> {iso_o['brier']:.6f}  "
    f"({iso_o['brier']-base['brier']:+.6f})")
log(f"    PR-AUC  {base['pr_auc']:.5f} -> {iso_o['pr_auc']:.5f}  "
    f"({iso_o['pr_auc']-base['pr_auc']:+.5f})")
log(f"    distinct probabilities {base['n_distinct_probabilities']:,} -> "
    f"{iso_o['n_distinct_probabilities']:,}  "
    f"(collapse ratio {iso_o['n_distinct_probabilities']/base['n_distinct_probabilities']:.4f})")
json.dump(out, open(EXP_DIR / "exp_07b_ece.json", "w"), indent=1)
log("EXP-07b done")
