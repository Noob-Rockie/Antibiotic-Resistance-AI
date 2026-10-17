"""PHASE 9 — Explainable AI on the deployed model.

Produces the global and local explanations that the application shows and the
report cites. All figures come from the model that is actually saved in the
bundle, on held-out test patients.

Outputs
  07_EXPLAINABLE_AI/figures/*.png
  07_EXPLAINABLE_AI/xai_report.json
  14_REPORT_MATERIAL/xai/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (MODELS_DIR, RANDOM_SEED, REPORT_DIR,  # noqa: E402
                          RESULTS_DIR, XAI_DIR)
from src.explainability.shap_explainer import (expected_value,  # noqa: E402
                                               explain_instance,
                                               global_importance)
from src.models.bundle import ModelBundle  # noqa: E402
from src.models.dataset import (TARGET_BINARY, load_analytic,  # noqa: E402
                                patient_grouped_split)

FIGS = XAI_DIR / "figures"
for d in (FIGS, REPORT_DIR / "xai", RESULTS_DIR / "figures"):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=0.95)
REPORT: dict = {}


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    for base in (FIGS, REPORT_DIR / "xai"):
        fig.savefig(base / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  figure -> {name}")


def main() -> None:
    bundle = ModelBundle.load(MODELS_DIR / "final_model.joblib")
    df = load_analytic()
    split = patient_grouped_split(df)
    test = df[split == "test"].reset_index(drop=True)
    X = bundle.prepare(test)
    y = test[TARGET_BINARY].to_numpy()
    log(f"Explaining {bundle.metadata.get('algorithm')} on "
        f"{len(test):,} held-out test rows")

    # ---------------- 1. global importance ---------------------------------
    log("\n1. GLOBAL EXPLANATION")
    imp, sv, Xs = global_importance(bundle.model, X, n_sample=40_000,
                                    seed=RANDOM_SEED)
    base = expected_value(bundle.model)
    REPORT["global"] = {
        "n_rows_explained": int(len(Xs)),
        "base_value_log_odds": round(base, 6),
        "mean_abs_shap": {k: round(float(v), 8) for k, v in imp.items()},
        "top_15": imp.head(15).index.tolist(),
    }
    log("   top 10 by mean|SHAP|: " + ", ".join(imp.head(10).index))

    fig, ax = plt.subplots(figsize=(10, 11))
    top = imp.head(30)[::-1]
    ax.barh(range(len(top)), top.values, color="#db2777")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8.5)
    ax.set_xlabel("mean(|SHAP value|)  —  average impact on the model output")
    ax.set_title("Global feature importance on held-out test patients\n"
                 f"{bundle.metadata.get('algorithm')}, "
                 f"{len(Xs):,} rows explained", fontsize=12.5,
                 fontweight="bold")
    savefig(fig, "01_shap_global_bar.png")

    fig = plt.figure(figsize=(11, 10))
    shap.summary_plot(sv, Xs, max_display=25, show=False, plot_size=None)
    plt.title("SHAP summary — magnitude AND direction of every feature",
              fontsize=12.5, fontweight="bold")
    plt.xlabel("SHAP value (impact on predicted log-odds of non-susceptibility)")
    savefig(plt.gcf(), "02_shap_summary_beeswarm.png")

    # ---------------- 2. dependence plots ----------------------------------
    log("\n2. DEPENDENCE PLOTS for the strongest features")
    for i, feat in enumerate(imp.head(6).index, 1):
        fig = plt.figure(figsize=(7.5, 5.2))
        shap.dependence_plot(feat, sv, Xs, show=False,
                             interaction_index="auto")
        plt.title(f"How `{feat}` moves the prediction", fontsize=11.5,
                  fontweight="bold")
        savefig(plt.gcf(), f"03_{i}_dependence_{feat[:40]}.png")

    # ---------------- 3. local explanations --------------------------------
    log("\n3. LOCAL EXPLANATIONS (individual predictions)")
    p = bundle.predict_nonsusceptible_proba(test)
    cases = {
        "confident_resistant_correct": np.where((p > 0.9) & (y == 1))[0],
        "confident_susceptible_correct": np.where((p < 0.05) & (y == 0))[0],
        "false_negative_missed_resistance": np.where((p < 0.2) & (y == 1))[0],
        "false_positive": np.where((p > 0.8) & (y == 0))[0],
        "borderline": np.where((p > 0.45) & (p < 0.55))[0],
    }
    rng = np.random.default_rng(RANDOM_SEED)
    local = {}
    for label, idx in cases.items():
        if len(idx) == 0:
            continue
        i = int(rng.choice(idx))
        row = X.iloc[[i]]
        expl = explain_instance(bundle.model, row, top_k=10)
        expl["case_type"] = label
        expl["p_nonsusceptible"] = round(float(p[i]), 5)
        expl["actual_nonsusceptible"] = int(y[i])
        expl["organism"] = str(test.loc[i, "organism_clean"])
        expl["antibiotic"] = str(test.loc[i, "antibiotic"])
        expl["specimen"] = str(test.loc[i, "culture_description"])
        local[label] = expl
        log(f"   {label:<34} {expl['organism'][:28]:<30} "
            f"{expl['antibiotic'][:22]:<24} p={p[i]:.3f} actual={y[i]}")

        ex = shap.Explanation(
            values=np.array([c["shap"] for c in expl["top_contributions"]]),
            base_values=base,
            data=np.array([c["value"] if c["value"] is not None else np.nan
                           for c in expl["top_contributions"]]),
            feature_names=[c["feature"] for c in expl["top_contributions"]])
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(ex, max_display=10, show=False)
        plt.title(f"{label.replace('_', ' ').title()}\n"
                  f"{expl['organism']} · {expl['antibiotic']} · "
                  f"{expl['specimen']} — model P(non-susceptible)="
                  f"{p[i]:.3f}, laboratory said "
                  f"{'non-susceptible' if y[i] else 'susceptible'}",
                  fontsize=10.5, fontweight="bold")
        savefig(plt.gcf(), f"04_local_{label}.png")

    REPORT["local_explanations"] = local

    # ---------------- 4. class-level SHAP behaviour ------------------------
    log("\n4. FEATURE BEHAVIOUR BY ANTIBIOTIC CLASS")
    cls = test.loc[Xs.index, "antibiotic_class"] if "antibiotic_class" in test \
        else None
    by_class = {}
    if cls is not None:
        svdf = pd.DataFrame(sv, columns=Xs.columns, index=Xs.index)
        for c, idx in cls.groupby(cls, observed=True).groups.items():
            sub = svdf.loc[idx]
            if len(sub) < 400:
                continue
            by_class[str(c)] = {
                "n_rows": int(len(sub)),
                "top_5": np.abs(sub).mean().sort_values(
                    ascending=False).head(5).round(6).to_dict(),
            }
        REPORT["by_antibiotic_class"] = by_class
        log(f"   summarised {len(by_class)} antibiotic classes")

    # ---------------- 5. does the model use history the way we expect? -----
    log("\n5. SANITY CHECK — direction of the history features")
    checks = {}
    y_expl = y[Xs.index.to_numpy()] if len(Xs) < len(y) else y
    numeric_cols = [c for c in Xs.columns
                    if Xs[c].dtype.kind in "fiu" and Xs[c].nunique() > 5]
    for feat in ("prior_same_abx_nonsusc_rate", "prior_res_same_abx_n",
                 "prior_res_same_org_abx_n",
                 "prior_res_same_org_abx_days_since",
                 "prior_med_same_class_days_since",
                 "prior_res_n_events_1095d",
                 "prior_same_org_nonsusc_rate"):
        if feat not in Xs.columns:
            continue
        vals = Xs[feat].to_numpy()
        shaps = sv[:, list(Xs.columns).index(feat)]
        ok = ~np.isnan(vals)
        if ok.sum() < 500:
            continue
        corr = float(np.corrcoef(vals[ok], shaps[ok])[0, 1])

        # A contradicting SHAP sign has two very different explanations, and
        # they must be told apart. Either the model learned a shortcut (bad), or
        # a strongly correlated partner feature already carries the signal and
        # this one is taking a compensating correction (benign). So also record:
        #   (a) the feature's RAW marginal association with the outcome, which
        #       is model-independent, and
        #   (b) its strongest correlation with another selected feature.
        pos = vals[ok] > 0
        yy = np.asarray(y_expl)[ok] if len(y_expl) == len(vals) else None
        raw = None
        if yy is not None and pos.sum() > 100 and (~pos).sum() > 100:
            raw = {
                "outcome_rate_when_feature_gt_0": round(float(yy[pos].mean()), 5),
                "outcome_rate_when_feature_eq_0": round(float(yy[~pos].mean()), 5),
                "mean_value_when_nonsusceptible": round(
                    float(vals[ok][yy == 1].mean()), 5),
                "mean_value_when_susceptible": round(
                    float(vals[ok][yy == 0].mean()), 5),
            }
        partners = {}
        for other in numeric_cols:
            if other == feat:
                continue
            both = ok & (~np.isnan(Xs[other].to_numpy()))
            if both.sum() < 500:
                continue
            c = float(np.corrcoef(vals[both], Xs[other].to_numpy()[both])[0, 1])
            if abs(c) > 0.5:
                partners[other] = round(c, 4)
        expected = "negative" if "days_since" in feat else "positive"
        sign_ok = (corr < 0) if expected == "negative" else (corr > 0)
        checks[feat] = {
            "corr_value_vs_shap": round(corr, 4),
            "expected_sign": expected,
            "shap_sign_as_expected": bool(sign_ok),
            "n": int(ok.sum()),
            "raw_marginal_association": raw,
            "correlated_selected_features_abs_gt_0.5": dict(
                sorted(partners.items(), key=lambda kv: -abs(kv[1]))[:5]),
        }
        flag = "" if sign_ok else "   <-- contradicts expectation"
        log(f"   {feat:<40} corr(value, SHAP) = {corr:+.3f}{flag}")
        if not sign_ok and raw:
            log(f"      raw association is "
                f"{'CORRECT' if raw['outcome_rate_when_feature_gt_0'] > raw['outcome_rate_when_feature_eq_0'] else 'ALSO INVERTED'}: "
                f"outcome rate {raw['outcome_rate_when_feature_eq_0']:.4f} when "
                f"0 vs {raw['outcome_rate_when_feature_gt_0']:.4f} when >0; "
                f"top correlated partner(s): "
                f"{list(checks[feat]['correlated_selected_features_abs_gt_0.5'])[:2]}")

    REPORT["direction_checks"] = {
        "correlations": checks,
        "interpretation": (
            "A positive correlation between a prior-resistance feature's value "
            "and its SHAP contribution means the model raises predicted "
            "resistance as that history grows - the clinically expected "
            "direction. `days_since` features should correlate NEGATIVELY: the "
            "longer ago the prior event, the less it should push towards "
            "resistance."),
        "how_to_read_a_contradiction": (
            "A contradicting SHAP sign is a prompt to investigate, not a "
            "verdict. Two causes look identical in the SHAP column and are "
            "distinguished by the extra fields recorded here. (1) The model "
            "learned a shortcut - in which case the feature's RAW marginal "
            "association with the outcome will also be inverted or absent, and "
            "the finding is serious. (2) A strongly correlated partner feature "
            "already carries the signal and this one is absorbing a "
            "compensating correction - in which case the raw association is "
            "still in the clinically expected direction and the correlated "
            "partner is listed. SHAP splits credit between collinear features "
            "arbitrarily, so this is expected behaviour rather than a defect, "
            "but it does mean an individual attribution for one member of a "
            "correlated group should not be read in isolation."),
    }

    (XAI_DIR / "xai_report.json").write_text(
        json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    log(f"\nWrote {XAI_DIR / 'xai_report.json'}")


if __name__ == "__main__":
    main()
