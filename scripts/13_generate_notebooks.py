"""Generate the analysis notebooks.

The notebooks are deliberately thin: the pipeline logic lives in `scripts/` and
`src/` where it is tested and reusable, and each notebook LOADS the artefacts
that stage produced and walks through them. That way a reader can follow the
analysis interactively without the notebooks becoming a second, untested,
silently-diverging copy of the pipeline.

Every notebook is runnable as-is once its stage has been executed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (ANALYSIS_DIR, FE_DIR, FS_DIR, PROJECT_ROOT,  # noqa: E402
                          TRAIN_DIR, XAI_DIR)

HEADER = """import sys, json
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

ROOT = Path.cwd()
while not (ROOT / "config" / "paths.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
pd.set_option("display.max_columns", 200, "display.width", 200)

def load(rel):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def show(rel, width=1000):
    p = ROOT / rel
    display(Image(str(p), width=width)) if p.exists() else print("missing:", rel)

print("project root:", ROOT)"""


def nb(cells) -> nbf.NotebookNode:
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return n


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def write(path: Path, notebook) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(path))
    print("wrote", path)


# ===========================================================================
def notebook_01():
    return nb([
        md("# 01 — Dataset Audit\n\n"
           "**Phase 1.** Nothing is assumed about the data here: every number "
           "comes from `scripts/01_dataset_audit.py`, which profiled all five "
           "raw files column by column without loading any of them fully into "
           "memory.\n\n"
           "Run `python scripts/01_dataset_audit.py` first if the artefacts are "
           "missing."),
        code(HEADER),
        md("## File inventory"),
        code('audit = load("02_DATA_ANALYSIS/reports/dataset_audit.json")\n'
             'inv = pd.DataFrame([{\n'
             '    "dataset": d["key"], "file": d["file_name"],\n'
             '    "size_MB": round(d["file_size_mb"], 1), "rows": d["n_rows"],\n'
             '    "cols": d["n_columns"],\n'
             '    "exact_duplicate_rows": d["n_exact_duplicate_rows"],\n'
             '} for d in audit["datasets"]])\n'
             'inv'),
        md("Note `prior_med`: 40.3% of its rows are exact duplicates. That is "
           "not corruption — it is the same drug administered more than once on "
           "the same day. The pipeline counts exposure in distinct drug-days, "
           "so those rows are collapsed rather than discarded."),
        md("## Column profiles\n\nPick any dataset to inspect."),
        code('DATASET = "cohort"   # cohort | demographics | labs | resistance | prior_med\n'
             'd = next(x for x in audit["datasets"] if x["key"] == DATASET)\n'
             'prof = pd.DataFrame([{\n'
             '    "column": c["column"], "missing_%": c["pct_missing"],\n'
             '    "n_unique": c["n_unique"], "numeric": c["is_numeric"],\n'
             '    "dominant_value_share": c.get("dominant_value_share"),\n'
             '    "examples": ", ".join(map(str, c["sample_values"][:2])),\n'
             '} for c in d["column_profiles"]])\n'
             'prof'),
        md("## The label column"),
        code('c = next(c for c in\n'
             '         next(x for x in audit["datasets"] if x["key"] == "cohort")["column_profiles"]\n'
             '         if c["column"] == "susceptibility")\n'
             'vc = pd.Series(c["value_counts"]).sort_values(ascending=False)\n'
             'display(vc.to_frame("rows").assign(pct=lambda d: (100*d.rows/d.rows.sum()).round(3)))\n'
             'print(f"missing (negative cultures): {c[\'pct_missing\']}%")'),
        md("`Inconclusive` and `Synergism` are **not** susceptibility "
           "phenotypes — one is an uninterpretable test, the other a two-drug "
           "interaction assay. Both are removed rather than forced into a class "
           "they do not belong to."),
        md("## Findings that changed the pipeline"),
        code('for d in audit["datasets"]:\n'
             '    const = [c["column"] for c in d["column_profiles"]\n'
             '             if c.get("dominant_value_share") == 1.0]\n'
             '    if const:\n'
             '        print(f"{d[\'key\']}: constant column(s) -> {const}")\n'
             'print()\n'
             'pm = next(x for x in audit["datasets"] if x["key"] == "prior_med")\n'
             'nm = next(c for c in pm["column_profiles"] if c["column"] == "medication_name")\n'
             'mc = next(c for c in pm["column_profiles"] if c["column"] == "medication_category")\n'
             'print(f"medication_name unique:     {nm[\'n_unique\']}")\n'
             'print(f"medication_category unique: {mc[\'n_unique\']}")\n'
             'print("-> a strict 1:1 surrogate key, NOT a therapeutic class. "\n'
             '      "Excluded; a real class is derived from the drug name.")'),
        md("Full write-up: [`docs/DATA_DICTIONARY.md`](../../docs/DATA_DICTIONARY.md)"),
    ])


def notebook_02():
    return nb([
        md("# 02 — Data Cleaning, Relationships and Leakage\n\n"
           "**Phases 2 and 3.** How the five tables relate, what had to be "
           "cleaned, and — most importantly — every leakage pathway that was "
           "investigated and what was done about it."),
        code(HEADER),
        md("## Relationships — measured, never assumed"),
        code('rel = load("02_DATA_ANALYSIS/reports/relationships.json")\n'
             'pd.DataFrame([{\n'
             '    "from": r["left"], "to": r["right"],\n'
             '    "cardinality": r["cardinality"],\n'
             '    "cohort orders matched": f\'{r["pct_left_keys_matched"]}%\',\n'
             '    "max rows per key (right)": r["right_max_rows_per_key"],\n'
             '} for r in rel["relationships"]])'),
        md("Only `demographics` is 1:1. Everything else would **multiply** the "
           "label rows if joined raw — up to 748× for `prior_med`. Each is "
           "aggregated to one row per key first."),
        code('show("docs/diagrams/dataset_relationships.png", 1200)'),
        md("## Why splits must be grouped by patient"),
        code('ps = rel["patient_structure"]\n'
             'for k, v in ps.items():\n'
             '    print(f"{k:<38} {v}")'),
        md("More than half of patients contribute several cultures. A random "
           "row split would put the same patient — often the same organism — on "
           "both sides, and the test set would partly be measuring "
           "memorisation."),
        md("## Cleaning decisions"),
        code('b = load("01_DATA/processed/build_report.json")\n'
             's1 = b["step1_cohort"]\n'
             'steps = pd.DataFrame([\n'
             '    ("raw cohort rows", s1["n_rows_raw"]),\n'
             '    ("- negative cultures (no result)", -s1["n_dropped_no_label_negative_culture"]),\n'
             '    ("- Inconclusive / Synergism", -s1["n_dropped_non_phenotype_result"]),\n'
             '    ("- duplicate key resolution", -s1["n_dropped_duplicate_resolution"]),\n'
             '    ("= modelling rows", b["step7_final"]["n_rows"]),\n'
             '], columns=["step", "rows"])\n'
             'display(steps)\n'
             'print("\\ndedup rule:", s1["dedup_rule"])\n'
             'print("duplicate groups:", f\'{s1["n_duplicate_key_groups"]:,}\',\n'
             '      "| with CONFLICTING labels:", f\'{s1["n_duplicate_groups_with_conflicting_labels"]:,}\')'),
        md("## Organism normalisation"),
        code('from src.preprocessing.organism_normalizer import parse_organism\n'
             'demo = ["STAPH AUREUS {MRSA}",\n'
             '        "KLEBSIELLA PNEUMONIAE (CARBAPENEM RESISTANT)",\n'
             '        "ZZZENTEROBACTER AEROGENES",\n'
             '        "MUCOID PSEUDOMONAS AERUGINOSA",\n'
             '        "STREPTOCOCCUS AGALACTIAE {GROUP B}",\n'
             '        "STREPTOCOCCUS AGALACTIAE (GROUP B)",\n'
             '        "ENTEROBACTER CLOACAE COMPLEX",\n'
             '        "ENTEROCOCCUS FAECIUM -  VANCO RESISTANT"]\n'
             'pd.DataFrame([{"raw": d, **parse_organism(d)} for d in demo])'),
        md("### The leakage this exposed\n\n"
           "`leaked_phenotype` is the resistance answer written into the "
           "organism *name*. It is stripped and never becomes a feature."),
        code('print(f\'rows affected: {s1["organism_leaked_phenotype_rows"]:,}\')\n'
             'display(pd.DataFrame(s1["organism_leak_evidence"]).T)'),
        md("Read the `VANCOMYCIN_RESISTANT_vs_Glycopeptide` row: rows carrying "
           "that token are **100%** resistant to glycopeptides against a 2.8% "
           "baseline. That is the label, restated."),
        md("## Prior resistance — the candidate that was cleared"),
        code('lk = load("02_DATA_ANALYSIS/reports/leakage.json")\n'
             'rt = lk["resistance_time_to_culture"]\n'
             'print("days before the index culture:")\n'
             'for k in ("min", "median", "max", "n_le_zero"):\n'
             '    print(f"  {k:<12} {rt[k]}")\n'
             'print()\n'
             'm = lk["cohort_rows_matching_resistance_same_order_org_abx"]\n'
             'print(f"matched on (order, organism, antibiotic): {m[\'pct\']}% of rows")\n'
             'for label, key in (("matched", "target_breakdown_of_matched"),\n'
             '                   ("unmatched", "target_breakdown_of_unmatched")):\n'
             '    d = {k: v for k, v in m[key].items()\n'
             '         if k in ("Resistant", "Susceptible", "Intermediate")}\n'
             '    tot = sum(d.values())\n'
             '    print(f"  {label:<10} resistant: {100*d[\'Resistant\']/tot:.1f}%")'),
        md("Minimum 1 day before the culture, zero events at or after it, only "
           "4% of rows match, and matched rows are 64% resistant rather than "
           "100%. Strongly predictive, and genuinely history.\n\n"
           "Full analysis: "
           "[`docs/DATA_LEAKAGE_ANALYSIS.md`](../../docs/DATA_LEAKAGE_ANALYSIS.md)"),
    ])


def notebook_03():
    from _eda_notebook_cells import CELLS, INTRO, OUTRO, SETUP
    cells = [md(INTRO), code(HEADER), code(SETUP)]
    for heading, body in CELLS:
        cells.append(md(heading))
        cells.append(code(body))
    cells.append(md(OUTRO))
    return nb(cells)


def notebook_04():
    return nb([
        md("# 04 — Feature Engineering\n\n"
           "**Phase 5.** How 10 raw cohort columns plus four companion tables "
           "became a 194-column analytic matrix, and the temporal rule that "
           "governs every history feature."),
        code(HEADER),
        md("## The strict temporal rule\n\n"
           "> Every history feature is derived only from events that occurred "
           "**strictly before** the index culture.\n\n"
           "Three separate mechanisms enforce it:"),
        code('b = load("01_DATA/processed/build_report.json")\n'
             'print("1. PRIOR RESISTANCE")\n'
             'print("  ", b["step5_prior_resistance"]["temporal_safety_note"])\n'
             'print("\\n2. PRIOR MEDICATION")\n'
             'print("  ", b["step4_prior_med"]["same_day_exclusion_reason"])\n'
             'print(f\'   rows dropped: {b["step4_prior_med"]["n_dropped_same_day"]:,}\')\n'
             'print("\\n3. DERIVED CULTURE HISTORY")\n'
             'print("  ", b["step6_prior_culture_history"]["method"])'),
        md("## Feature groups"),
        code('fs = load("04_FEATURE_SELECTION/selected_features/feature_selection_report.json")\n'
             'pd.DataFrame([{"group": g, "n_features": d["n"],\n'
             '               "examples": ", ".join(d["features"][:5])}\n'
             '              for g, d in fs["feature_inventory"]["groups"].items()])'),
        md("## Domain mappings built for this dataset"),
        code('from src.preprocessing.antibiotic_taxonomy import ANTIBIOTIC_CLASS\n'
             'from src.preprocessing.medication_mapping import MEDICATION_MAP\n'
             'tax = pd.DataFrame([{"antibiotic": k, "class": v[0],\n'
             '                     "subclass": v[1], "AWaRe": v[2]}\n'
             '                    for k, v in ANTIBIOTIC_CLASS.items()])\n'
             'print(f"{len(tax)} agents across {tax[\'class\'].nunique()} classes")\n'
             'display(tax.head(15))\n'
             'print(f"\\n{len(MEDICATION_MAP)} medication names mapped to "\n'
             '      f"{len({v[0] for v in MEDICATION_MAP.values()})} ingredients")'),
        md("## Labs: three partial rows per order, collapsed to one"),
        code('s3 = b["step3_labs"]\n'
             'print(s3["collapse_rule"])\n'
             'print()\n'
             'print("rows per order:", s3["rows_per_order"])\n'
             'print("\\nPeriod_Day:", s3["period_day_dropped_reason"])'),
        md("## Every join was validated before it ran"),
        code('pd.DataFrame(b["step7_joins"])[\n'
             '    ["join", "how", "rows_before", "rows_after",\n'
             '     "right_side_unique_on_keys", "pct_left_rows_matched"]]'),
        md("`rows_before == rows_after` on every line. `validated_merge()` "
           "raises rather than allowing a fan-out or a silent row loss."),
        md("## The resulting table"),
        code('f = b["step7_final"]\n'
             'for k, v in f.items():\n'
             '    print(f"{k:<24} {v:,}" if isinstance(v, int) else f"{k:<24} {v}")'),
    ])


def notebook_05():
    return nb([
        md("# 05 — Feature Selection\n\n"
           "**Phase 6.** Seven methods, eight candidate feature sets, and a "
           "check on whether the selection process itself leaked.\n\n"
           "Run `python scripts/06_feature_selection.py` first."),
        code(HEADER),
        code('fs = load("04_FEATURE_SELECTION/selected_features/feature_selection_report.json")\n'
             'print("thresholds:")\n'
             'for k, v in fs["thresholds"].items():\n'
             '    print(f"  {k:<26} {v}")\n'
             'print("\\nsplit:")\n'
             'display(pd.DataFrame(fs["split"]["partitions"]).T)'),
        md("## Methods 1–3: the filters"),
        code('for key, label in (("method_1_missingness", "Missing values"),\n'
             '                   ("method_2_low_variance", "Low variance"),\n'
             '                   ("method_3_correlation", "Correlation")):\n'
             '    m = fs[key]\n'
             '    print(f"{label}: {m[\'n_dropped\']} dropped")\n'
             '    print("  ", m["rationale"][:300], "...")\n'
             '    print()'),
        code('show("04_FEATURE_SELECTION/figures/01_missing_values.png", 750)\n'
             'show("04_FEATURE_SELECTION/figures/02_correlation_heatmap.png", 1000)'),
        md("## Methods 4–7: the rankers"),
        code('imp = fs["importances"]\n'
             'rank = pd.DataFrame({\n'
             '    "SHAP": pd.Series(imp["shap"]).rank(ascending=False),\n'
             '    "XGB_gain": pd.Series(imp["xgb_gain"]).rank(ascending=False),\n'
             '    "RandomForest": pd.Series(imp["random_forest"]).rank(ascending=False),\n'
             '    "MutualInfo": pd.Series(imp["mutual_information"]).rank(ascending=False),\n'
             '    "ANOVA_F": pd.Series(imp["anova_f"]).rank(ascending=False),\n'
             '    "RFE": pd.Series(imp["rfe_rank"]),\n'
             '}).sort_values("SHAP")\n'
             'rank.head(25).astype(int)'),
        md("Where the rankers **disagree** is the interesting part. Univariate "
           "methods (mutual information, ANOVA F) cannot see that a feature is "
           "only useful in combination with the antibiotic being scored, so "
           "they systematically under-rank the interaction features."),
        code('for f in ["04_random_forest_importance.png", "05_xgboost_importance.png",\n'
             '          "03_mutual_information.png", "06_rfe_ranking.png",\n'
             '          "07_shap_feature_importance.png", "07b_shap_summary_beeswarm.png"]:\n'
             '    show(f"04_FEATURE_SELECTION/figures/{f}", 800)'),
        md("## The experiment"),
        code('comp = pd.read_csv(ROOT / "13_RESULTS/tables/feature_selection_comparison.csv")\n'
             'comp.sort_values("pr_auc", ascending=False)'),
        code('a = comp.set_index("feature_set")\n'
             'print(f"all {int(a.loc[\'A_all_features\',\'n_features\'])} features:      "\n'
             '      f"PR-AUC {a.loc[\'A_all_features\',\'pr_auc\']:.4f}")\n'
             'print(f"selected 30 features:     PR-AUC {a.loc[\'C_model_based_XGB_top30\',\'pr_auc\']:.4f}")\n'
             'print(f"organism-blind ablation:  PR-AUC {a.loc[\'H_organism_blind\',\'pr_auc\']:.4f}")\n'
             'print()\n'
             'print("-> fewer features win, and the organism carries a large share")\n'
             'print("   of the signal. The ablation quantifies exactly what a purely")\n'
             'print("   empiric, pre-identification model would lose.")'),
        md("## How many features are actually needed?"),
        code('sweep = pd.read_csv(ROOT / "13_RESULTS/tables/feature_count_sweep.csv")\n'
             'display(sweep)\n'
             'show("04_FEATURE_SELECTION/figures/08_feature_count_vs_f1.png", 1200)'),
        md("## Did selecting on the whole training set leak?\n\n"
           "Project brief §17. The honest test: re-derive the selection inside "
           "every CV fold and compare like-for-like against the globally chosen "
           "list."),
        code('cv = fs["selection_inside_cv"]\n'
             'display(pd.DataFrame({"nested (honest)": cv["nested_fold_scores"],\n'
             '                      "global": cv["global_fold_scores"]},\n'
             '                     index=[f"fold {i}" for i in range(1, 4)]))\n'
             'print(f"mean nested: {cv[\'nested_cv_pr_auc_mean\']:.5f}")\n'
             'print(f"mean global: {cv[\'global_cv_pr_auc_mean\']:.5f}")\n'
             'print(f"optimism:    {cv[\'optimism_from_global_selection\']:+.5f} PR-AUC")\n'
             'print()\n'
             'print(cv["interpretation"])'),
        md("## The final set"),
        code('cfg = load("config/final_features.json")\n'
             'print(cfg["selection_method"])\n'
             'print(f"\\n{cfg[\'n_features\']} features:")\n'
             'for f in cfg["features"]:\n'
             '    print("  -", f)\n'
             'show("04_FEATURE_SELECTION/figures/09_selected_vs_rejected.png", 850)'),
    ])


def notebook_06():
    return nb([
        md("# 06 — Model Training\n\n"
           "**Phase 7.** Six model families, an evidence-based resolution of "
           "class imbalance, grouped cross-validated tuning, and calibration.\n\n"
           "Run `python scripts/07_train_models.py` first."),
        code(HEADER),
        md("## Model family comparison\n\n"
           "Identical training rows, identical feature set, identical held-out "
           "validation patients. Only the learner differs."),
        code('tr = load("05_MODEL_TRAINING/experiments/training_report.json")\n'
             'c = pd.DataFrame(tr["model_comparison"])\n'
             'display(c[["model", "pr_auc", "roc_auc", "f1", "recall_sensitivity",\n'
             '           "specificity", "brier", "train_seconds"]]\n'
             '        .sort_values("pr_auc", ascending=False))\n'
             'print()\n'
             'for r in tr["model_comparison"]:\n'
             '    print(f"{r[\'model\']}:\\n   {r.get(\'rationale\',\'\')}\\n")'),
        code('fig, ax = plt.subplots(figsize=(9, 5))\n'
             'd = c.sort_values("pr_auc")\n'
             'ax.barh(d["model"], d["pr_auc"], color="#2563eb")\n'
             'ax.set_xlabel("PR-AUC on held-out validation patients")\n'
             'ax.set_title("Model family comparison")\n'
             'plt.show()'),
        md("The gap between the linear baseline and the boosted trees is the "
           "clearest single result: roughly a quarter of the achievable signal "
           "lives in interactions between the antibiotic, the organism and the "
           "patient's history, which an additive model cannot reach."),
        md("## Class imbalance — resolved by experiment"),
        code('imb = tr["imbalance"]\n'
             'print(f"positive rate: {imb[\'positive_rate\']:.4f}")\n'
             'display(pd.DataFrame(imb["results"])[\n'
             '    ["strategy", "n_train_rows", "pr_auc", "roc_auc",\n'
             '     "recall_sensitivity", "precision", "f1", "brier"]]\n'
             '    .sort_values("pr_auc", ascending=False))\n'
             'print(f"\\nSELECTED: {imb[\'selected\']}\\n")\n'
             'print(imb["rationale"])'),
        md("**The headline finding: SMOTE actively harmed this problem.** "
           "Interpolating between ordinal-coded organism and antibiotic codes "
           "manufactures entities that do not exist, and interpolating sparse "
           "prior-exposure counts invents patient histories. The right answer "
           "was to leave the class distribution alone and move the threshold."),
        md("## Hyperparameter tuning"),
        code('for name, t in tr["tuning"].items():\n'
             '    print(f"{name}: best CV PR-AUC {t[\'best_cv_pr_auc\']:.5f} "\n'
             '          f"({t[\'search_seconds\']:.0f}s)")\n'
             '    print("  ", t["best_params"])\n'
             '    print()'),
        md("## Calibration"),
        code('cal = tr["calibration"]\n'
             'print(cal["method"])\n'
             'print(f"Brier  {cal[\'brier_before\']} -> {cal[\'brier_after\']}")\n'
             'print(f"PR-AUC {cal[\'pr_auc_before\']} -> {cal[\'pr_auc_after\']}")\n'
             'print()\n'
             'print(cal["caveat"])'),
        md("## Experiment log\n\nEvery run, for reproducibility."),
        code('log = pd.read_csv(ROOT / "05_MODEL_TRAINING/experiments/experiment_log.csv")\n'
             'log[["experiment_id", "algorithm", "cv_method", "n_train_rows",\n'
             '     "pr_auc", "roc_auc", "f1", "notes"]]'),
    ])


def notebook_07():
    return nb([
        md("# 07 — Model Evaluation\n\n"
           "**Phase 8.** The test partition was read exactly once, by "
           "`scripts/08_evaluate_models.py`. Nothing in feature selection, "
           "model selection, tuning or calibration touched it."),
        code(HEADER),
        code('ev = load("06_MODEL_EVALUATION/evaluation_report.json")\n'
             'm = ev["test_metrics"]\n'
             'pd.Series({k: m[k] for k in\n'
             '           ["n", "positive_rate", "pr_auc", "roc_auc",\n'
             '            "recall_sensitivity", "specificity", "precision_ppv",\n'
             '            "npv", "f1", "macro_f1", "weighted_f1", "accuracy",\n'
             '            "brier"]}).to_frame("value")'),
        md("Accuracy is reported but never used for selection: predicting "
           "\"susceptible\" for every row would already score ~0.80."),
        code('show("06_MODEL_EVALUATION/figures/01_roc_pr_calibration.png", 1300)\n'
             'show("06_MODEL_EVALUATION/figures/02_confusion_matrix.png", 1000)'),
        md("## Threshold analysis — a clinical choice, not a statistical one"),
        code('th = pd.read_csv(ROOT / "06_MODEL_EVALUATION/tables/threshold_analysis.csv")\n'
             'display(th[["threshold", "recall_sensitivity", "specificity",\n'
             '            "precision_ppv", "f1", "fn", "fp"]])\n'
             'show("06_MODEL_EVALUATION/figures/03_threshold_analysis.png", 1300)\n'
             'print()\n'
             'for k, v in ev["threshold_analysis"].items():\n'
             '    print(f"{k}: {v}")'),
        md("## Where the model works — and where it does not"),
        code('pa = pd.DataFrame(ev["per_antibiotic"]).sort_values("pr_auc", ascending=False)\n'
             'pa["lift_over_prevalence"] = (pa["pr_auc"] / pa["positive_rate"]).round(2)\n'
             'display(pa[["antibiotic", "n", "positive_rate", "pr_auc", "roc_auc",\n'
             '            "recall_sensitivity", "specificity", "lift_over_prevalence"]])\n'
             'show("06_MODEL_EVALUATION/figures/04_per_antibiotic_pr_auc.png", 900)'),
        md("Read `pr_auc` against `positive_rate`. An agent whose PR-AUC barely "
           "exceeds its prevalence is one where the model adds little over "
           "guessing the base rate."),
        md("## Subgroup performance"),
        code('for col, recs in ev["subgroups"].items():\n'
             '    d = pd.DataFrame(recs)\n'
             '    print(f"=== {col} (PR-AUC spread "\n'
             '          f"{d[\'pr_auc\'].max()-d[\'pr_auc\'].min():.4f}) ===")\n'
             '    display(d[[col, "n", "positive_rate", "pr_auc", "roc_auc",\n'
             '               "recall_sensitivity", "specificity"]])'),
        md("PR-AUC is not comparable across groups with different prevalence — "
           "read each group's PR-AUC against its own `positive_rate`. ROC-AUC is "
           "prevalence-independent and is the fairer cross-group comparison."),
        md("## Temporal generalisation — the deployment-realistic number"),
        code('tv = ev["temporal_vs_grouped"]\n'
             'tm = ev["temporal_metrics"]\n'
             'display(pd.DataFrame({\n'
             '    "patient-grouped": {k: m[k] for k in\n'
             '                        ["pr_auc","roc_auc","recall_sensitivity",\n'
             '                         "specificity","brier"]},\n'
             '    "temporal":        {k: tm[k] for k in\n'
             '                        ["pr_auc","roc_auc","recall_sensitivity",\n'
             '                         "specificity","brier"]}}))\n'
             'print(tv["interpretation"])'),
        md("## Secondary 3-class model"),
        code('tc = ev["three_class"]\n'
             'cr = tc["classification_report"]\n'
             'display(pd.DataFrame({k: cr[k] for k in\n'
             '                      ["Susceptible","Intermediate","Resistant"]}).T)\n'
             'show("06_MODEL_EVALUATION/figures/05_three_class_confusion.png", 700)\n'
             'print(tc["conclusion"])'),
        md("## Architecture comparison — unified vs one model per antibiotic"),
        code('ac = ev.get("architecture_comparison")\n'
             'if ac:\n'
             '    display(pd.DataFrame(ac["per_agent"]))\n'
             '    print(f"unified wins on {ac[\'n_agents_where_unified_wins\']}"\n'
             '          f"/{ac[\'n_agents_compared\']} agents; mean advantage "\n'
             '          f"{ac[\'mean_unified_advantage_pr_auc\']:+.4f} PR-AUC")\n'
             '    show("06_MODEL_EVALUATION/figures/06_architecture_comparison.png", 1000)'),
        md("The full reading of this result — including where it argues "
           "*against* the chosen architecture — is in "
           "[`docs/MODEL_EVALUATION.md`](../../docs/MODEL_EVALUATION.md) §11."),
    ])


def notebook_08():
    return nb([
        md("# 08 — Explainable AI\n\n"
           "**Phase 9.** SHAP on the deployed model, over held-out test "
           "patients. The same `src/explainability/shap_explainer.py` code runs "
           "here, in the report and in the live application."),
        code(HEADER),
        code('xai = load("07_EXPLAINABLE_AI/xai_report.json")\n'
             'g = xai["global"]\n'
             'print(f"base value (log-odds): {g[\'base_value_log_odds\']}")\n'
             'print(f"rows explained: {g[\'n_rows_explained\']:,}")\n'
             'pd.Series(g["mean_abs_shap"]).head(25).to_frame("mean |SHAP|")'),
        code('show("07_EXPLAINABLE_AI/figures/01_shap_global_bar.png", 850)\n'
             'show("07_EXPLAINABLE_AI/figures/02_shap_summary_beeswarm.png", 1000)'),
        md("The beeswarm is the more informative plot: it shows **direction** as "
           "well as magnitude, so a feature whose high values push towards "
           "resistance is visibly distinguishable from one whose high values "
           "push the other way."),
        md("## Dependence — how a single feature moves the prediction"),
        code('import glob\n'
             'for p in sorted(glob.glob(str(ROOT / "07_EXPLAINABLE_AI/figures/03_*.png"))):\n'
             '    display(Image(p, width=700))'),
        md("## Individual predictions — including the failures\n\n"
           "An explanation method is only useful if it also explains mistakes."),
        code('for label, e in xai["local_explanations"].items():\n'
             '    print("=" * 78)\n'
             '    print(label.replace("_", " ").upper())\n'
             '    print(f"{e[\'organism\']} | {e[\'antibiotic\']} | {e[\'specimen\']}")\n'
             '    print(f"model P(non-susceptible) = {e[\'p_nonsusceptible\']}; "\n'
             '          f"laboratory said "\n'
             '          f"{\'non-susceptible\' if e[\'actual_nonsusceptible\'] else \'susceptible\'}")\n'
             '    display(pd.DataFrame(e["top_contributions"])[\n'
             '        ["feature", "value", "shap", "direction"]])\n'
             '    show(f"07_EXPLAINABLE_AI/figures/04_local_{label}.png", 800)'),
        md("## Mechanism check — do the history features behave as expected?"),
        code('dc = xai["direction_checks"]\n'
             'rows = []\n'
             'for f, v in dc["correlations"].items():\n'
             '    rows.append({"feature": f,\n'
             '                 "corr(value, SHAP)": v["corr_value_vs_shap"],\n'
             '                 "expected": v["expected_sign"],\n'
             '                 "as expected": v["shap_sign_as_expected"]})\n'
             'display(pd.DataFrame(rows))\n'
             'print(dc["interpretation"])'),
        md("### Investigating the one that disagrees"),
        code('for f, v in dc["correlations"].items():\n'
             '    if v["shap_sign_as_expected"]:\n'
             '        continue\n'
             '    print(f"--- {f} ---")\n'
             '    print("raw marginal association (model-independent):")\n'
             '    for k, val in (v["raw_marginal_association"] or {}).items():\n'
             '        print(f"   {k:<38} {val}")\n'
             '    print("\\ncorrelated partners in the selected set:")\n'
             '    for k, c in v["correlated_selected_features_abs_gt_0.5"].items():\n'
             '        print(f"   {k:<40} r = {c:+.3f}")\n'
             'print()\n'
             'print(dc["how_to_read_a_contradiction"])'),
        md("The raw association is in the clinically expected direction, and the "
           "feature has strongly correlated partners that already carry the "
           "signal. This is **credit-splitting between collinear features**, not "
           "a learned shortcut — but it does mean a single SHAP attribution "
           "within a correlated group should not be read in isolation."),
        md("## Which evidence the model uses, by antibiotic class"),
        code('bc = xai.get("by_antibiotic_class", {})\n'
             'pd.DataFrame([{"class": k, "n_rows": v["n_rows"],\n'
             '               "top features": ", ".join(list(v["top_5"])[:5])}\n'
             '              for k, v in sorted(bc.items(),\n'
             '                                 key=lambda kv: -kv[1]["n_rows"])])'),
        md("Different classes are driven by different evidence — which is "
           "exactly what a unified model with the antibiotic as a feature is "
           "supposed to do."),
    ])


def main() -> None:
    write(ANALYSIS_DIR / "notebooks" / "01_dataset_audit.ipynb", notebook_01())
    write(ANALYSIS_DIR / "notebooks" / "02_data_cleaning_and_leakage.ipynb",
          notebook_02())
    write(ANALYSIS_DIR / "notebooks" / "03_exploratory_data_analysis.ipynb",
          notebook_03())
    write(FE_DIR / "notebooks" / "04_feature_engineering.ipynb", notebook_04())
    write(FS_DIR / "notebooks" / "05_feature_selection.ipynb", notebook_05())
    write(TRAIN_DIR / "notebooks" / "06_model_training.ipynb", notebook_06())
    write(TRAIN_DIR / "notebooks" / "07_model_evaluation.ipynb", notebook_07())
    write(XAI_DIR / "notebooks" / "08_explainable_ai.ipynb", notebook_08())


if __name__ == "__main__":
    main()
