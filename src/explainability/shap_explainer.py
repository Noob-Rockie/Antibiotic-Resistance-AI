"""SHAP explainability — one implementation, used by the app and the reports.

Why SHAP and not something else
-------------------------------
Three properties matter for a clinical decision-support prototype:

  * **Local faithfulness.** A clinician needs to know why *this* isolate got
    *this* number, not which features matter on average. Split-gain importance
    cannot answer that; SHAP can.
  * **Additivity.** The contributions for one prediction sum to the difference
    between that prediction and the model's base rate. That makes an
    explanation auditable rather than suggestive.
  * **Consistency.** If a model changes so that a feature matters more, its
    SHAP attribution cannot go down — a guarantee that impurity importance does
    not provide.

`TreeExplainer` computes exact Shapley values for tree ensembles in polynomial
time, so no sampling approximation is involved for the deployed model.

What SHAP does NOT give
-----------------------
A SHAP value is an attribution of the MODEL's output, not a causal claim about
the patient. "Prior exposure to fluoroquinolones raised the predicted
resistance" means the model's estimate moved; it does not establish that the
exposure caused the resistance. Every user-facing string in this project is
worded accordingly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _get_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values for the positive class, as (n_rows, n_features)."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):        # older API: one array per class
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                # (n, features, classes)
        sv = sv[:, :, -1]
    return sv


def expected_value(model) -> float:
    import shap
    ev = shap.TreeExplainer(model).expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.asarray(ev).ravel()[-1]
    return float(ev)


def global_importance(model, X: pd.DataFrame, n_sample: int = 40_000,
                      seed: int = 42) -> tuple:
    """Mean |SHAP| per feature, plus the raw values for plotting."""
    if len(X) > n_sample:
        idx = np.random.default_rng(seed).choice(len(X), n_sample, replace=False)
        X = X.iloc[idx]
    sv = _get_shap_values(model, X)
    imp = pd.Series(np.abs(sv).mean(axis=0),
                    index=X.columns).sort_values(ascending=False)
    return imp, sv, X


def explain_instance(model, X_row: pd.DataFrame, top_k: int = 8) -> dict:
    """Full additive decomposition of a single prediction."""
    sv = _get_shap_values(model, X_row)[0]
    base = expected_value(model)
    order = np.argsort(-np.abs(sv))[:top_k]
    contributions = [{
        "feature": X_row.columns[j],
        "value": (None if pd.isna(X_row.iloc[0, j]) else float(X_row.iloc[0, j])),
        "shap": float(sv[j]),
        "direction": ("increases predicted resistance" if sv[j] > 0
                      else "increases predicted susceptibility"),
    } for j in order]
    return {
        "base_value_log_odds": base,
        "sum_of_contributions": float(sv.sum()),
        "prediction_log_odds": float(base + sv.sum()),
        "top_contributions": contributions,
        "n_features": int(X_row.shape[1]),
        "additivity_note": ("base value + all contributions = the model's "
                            "log-odds output for this row; the top "
                            f"{top_k} are shown."),
    }


def explain_batch(model, X: pd.DataFrame, top_k: int = 5) -> list:
    """Per-row top contributions for a batch (used by the ranking engine)."""
    if len(X) == 0:
        return []
    sv = _get_shap_values(model, X)
    cols = list(X.columns)
    out = []
    for i in range(sv.shape[0]):
        order = np.argsort(-np.abs(sv[i]))[:top_k]
        out.append([{
            "feature": cols[j],
            "value": (None if pd.isna(X.iloc[i, j]) else float(X.iloc[i, j])),
            "shap": float(sv[i, j]),
            "direction": ("increases predicted resistance" if sv[i, j] > 0
                          else "increases predicted susceptibility"),
        } for j in order])
    return out
