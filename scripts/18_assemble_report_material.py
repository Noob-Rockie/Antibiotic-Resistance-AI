"""Assemble 14_REPORT_MATERIAL/ into the structure an academic report needs.

Collects every figure and table the pipeline produced into one predictable
tree, with descriptive filenames, and writes an index so a reader knows what
each artefact is and which script produced it.

  14_REPORT_MATERIAL/
  ├── diagrams/            workflow and architecture diagrams
  ├── figures/
  │   ├── DATASET/         dataset structure, relationships, missingness
  │   ├── EDA/             the 14 exploratory figures
  │   ├── FEATURE_SELECTION/
  │   ├── MODEL/           ROC, PR, calibration, confusion, comparisons
  │   └── XAI/             SHAP global, beeswarm, dependence, local
  ├── tables/
  │   ├── DATASET/  EDA/  FEATURE_SELECTION/  MODEL/
  └── INDEX.md
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (ANALYSIS_DIR, CONFIG_DIR, DATA_DIR,  # noqa: E402
                          DOCS_DIR, EVAL_DIR, EXPERIMENTS_DIR, FS_DIR,
                          PROCESSED_DIR, PROJECT_ROOT, REPORT_DIR,
                          RESULTS_DIR, XAI_DIR)

FIGS = REPORT_DIR / "figures"
TABS = REPORT_DIR / "tables"
DIAG = REPORT_DIR / "diagrams"

copied: list = []


def copy(src: Path, dest_dir: Path, rename: str | None = None,
         note: str = "", source_script: str = "") -> None:
    if not src.exists():
        print(f"  MISSING  {src}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (rename or src.name)
    # Diagrams are generated straight into 14_REPORT_MATERIAL/diagrams, so the
    # source and destination can already be the same file. Index it, don't copy.
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    copied.append({
        "artefact": str(dest.relative_to(REPORT_DIR)),
        "source": str(src.relative_to(PROJECT_ROOT)),
        "produced_by": source_script,
        "description": note,
    })


def main() -> None:
    print("Assembling report material ...")

    # ---------------- DATASET ------------------------------------------
    copy(DOCS_DIR / "diagrams" / "dataset_relationships.png", DIAG,
         note="Entity-relationship diagram with measured cardinalities",
         source_script="scripts/03_generate_data_docs.py")
    copy(ANALYSIS_DIR / "figures" / "01_missing_values.png",
         FIGS / "DATASET", "01_missing_values.png",
         "Missing-value analysis across all 194 columns",
         "scripts/04b_exploratory_analysis.py")

    for name, note in (
            ("01_dataset_overview.csv", "Dataset summary table"),
            ("02_missing_values.csv", "Missing count and percentage per column"),
            ("03_duplicate_analysis_raw_files.csv",
             "Exact duplicate rows in each source file")):
        copy(ANALYSIS_DIR / "tables" / name, TABS / "DATASET", note=note,
             source_script="scripts/04b_exploratory_analysis.py")
    copy(DATA_DIR / "raw" / "RAW_DATA_MANIFEST.md", TABS / "DATASET",
         "raw_data_manifest.md", "Source file sizes, row counts and MD5 hashes",
         "scripts/01_dataset_audit.py")

    # ---------------- EDA ----------------------------------------------
    eda_notes = {
        "01_missing_values.png": "Missing-value analysis",
        "02_target_distribution.png": "Target distribution and class imbalance",
        "03_antibiotic_distribution.png":
            "Antibiotic distribution — the ragged panel",
        "04_organism_distribution.png": "Organism and organism-group distribution",
        "05_specimen_distribution.png": "Specimen type distribution",
        "06_prior_antibiotic_exposure.png": "Previous antibiotic exposure",
        "06b_demographics.png": "Age band and de-identified sex code",
        "07_prior_resistance_history.png": "Previous resistance history",
        "08_temporal_resistance_trend.png": "Non-susceptibility over time",
        "09_antibiotic_organism_heatmap.png": "Cohort antibiogram",
        "10_correlation_heatmap.png": "Correlation among continuous features",
        "11_laboratory_distributions.png": "Laboratory value distributions",
        "12_history_feature_distributions.png": "History count distributions",
        "13_outlier_boxplots.png": "Outlier structure of the laboratory values",
    }
    for name, note in eda_notes.items():
        copy(ANALYSIS_DIR / "figures" / name, FIGS / "EDA", note=note,
             source_script="scripts/04b_exploratory_analysis.py")
    for p in sorted((ANALYSIS_DIR / "tables").glob("*.csv")):
        copy(p, TABS / "EDA", note="EDA table",
             source_script="scripts/04b_exploratory_analysis.py")

    # ---------------- FEATURE ENGINEERING / SELECTION -------------------
    copy(DIAG / "feature_engineering_workflow.png", DIAG,
         note="Raw tables to analytic matrix",
         source_script="scripts/16_generate_diagrams.py")
    copy(DIAG / "feature_selection_methodology.png", DIAG,
         note="Seven methods, 185 candidates to 30 features",
         source_script="scripts/16_generate_diagrams.py")

    fs_notes = {
        "01_missing_values.png": "Method 1 — missing-value filter",
        "02_correlation_heatmap.png": "Method 3 — correlation filter",
        "03_mutual_information.png": "Method 4 — mutual information ranking",
        "04_random_forest_importance.png": "Method 5 — Random Forest importance",
        "05_xgboost_importance.png": "Method 5 — XGBoost gain",
        "06_rfe_ranking.png": "Method 7 — recursive feature elimination",
        "07_shap_feature_importance.png": "Method 6 — SHAP importance",
        "07b_shap_summary_beeswarm.png": "Method 6 — SHAP beeswarm",
        "08_feature_count_vs_f1.png": "Performance vs number of features",
        "09_selected_vs_rejected.png": "Selected vs rejected features",
    }
    for name, note in fs_notes.items():
        copy(FS_DIR / "figures" / name, FIGS / "FEATURE_SELECTION", note=note,
             source_script="scripts/06_feature_selection.py")
    for name, note in (
            ("feature_selection_comparison.csv",
             "Candidate feature sets compared on held-out validation patients"),
            ("feature_count_sweep.csv",
             "Performance as features are added in SHAP order")):
        copy(RESULTS_DIR / "tables" / name, TABS / "FEATURE_SELECTION",
             note=note, source_script="scripts/06_feature_selection.py")
    copy(CONFIG_DIR / "final_features.json", TABS / "FEATURE_SELECTION",
         "final_feature_list.json", "The 30 selected features, machine-readable",
         "scripts/06_feature_selection.py")

    # ---------------- MODEL ----------------------------------------------
    model_notes = {
        "01_roc_pr_calibration.png": "ROC curve, precision-recall curve and "
                                     "calibration curve",
        "02_confusion_matrix.png": "Confusion matrix, counts and row-normalised",
        "03_threshold_analysis.png": "Operating characteristics vs threshold",
        "04_per_antibiotic_pr_auc.png": "Per-antibiotic discrimination",
        "05_three_class_confusion.png": "Secondary 3-class model",
        "06_architecture_comparison.png": "Unified vs per-antibiotic models",
    }
    for name, note in model_notes.items():
        copy(EVAL_DIR / "figures" / name, FIGS / "MODEL", note=note,
             source_script="scripts/08_evaluate_models.py")

    for src, dest_name, note in (
            (RESULTS_DIR / "tables" / "model_comparison.csv",
             "model_comparison.csv", "Six model families compared"),
            (RESULTS_DIR / "tables" / "imbalance_comparison.csv",
             "imbalance_comparison.csv", "Four class-imbalance strategies"),
            (RESULTS_DIR / "tables" / "final_model_test_metrics.csv",
             "final_model_test_metrics.csv", "Headline held-out test metrics"),
            (RESULTS_DIR / "tables" / "architecture_comparison.csv",
             "architecture_comparison.csv", "Unified vs per-agent PR-AUC"),
            (RESULTS_DIR / "tables" / "per_antibiotic_performance.csv",
             "per_antibiotic_performance.csv", "Test performance per agent"),
            (EVAL_DIR / "tables" / "per_organism_performance.csv",
             "per_organism_performance.csv", "Test performance per organism"),
            (EVAL_DIR / "tables" / "threshold_analysis.csv",
             "threshold_analysis.csv", "Metrics at every decision threshold"),
            (EXPERIMENTS_DIR / "experiment_log.csv", "experiment_log.csv",
             "Every training experiment, for reproducibility")):
        copy(src, TABS / "MODEL", dest_name, note,
             "scripts/07_train_models.py / 08_evaluate_models.py")
    for p in sorted((EVAL_DIR / "tables").glob("subgroup_*.csv")):
        copy(p, TABS / "MODEL", note="Subgroup performance",
             source_script="scripts/08_evaluate_models.py")

    # ---------------- XAI ------------------------------------------------
    for p in sorted((XAI_DIR / "figures").glob("*.png")):
        note = ("Global SHAP importance" if p.name.startswith("01") else
                "SHAP beeswarm summary" if p.name.startswith("02") else
                "SHAP dependence plot" if p.name.startswith("03") else
                "Individual prediction explanation (waterfall)")
        copy(p, FIGS / "XAI", note=note,
             source_script="scripts/09_explainability.py")

    # ---------------- CDS diagrams ---------------------------------------
    for name, note in (
            ("system_architecture.png", "Layered system architecture"),
            ("application_workflow.png", "What a user does in the application"),
            ("recommendation_workflow.png",
             "How one ranked option is produced"),
            ("human_in_the_loop_workflow.png",
             "Where the clinician decides"),
            ("methodology_workflow.png",
             "End-to-end research methodology")):
        copy(DIAG / name, DIAG, note=note,
             source_script="scripts/16_generate_diagrams.py")

    # ---------------- INDEX ----------------------------------------------
    build = json.loads((PROCESSED_DIR / "build_report.json").read_text(
        encoding="utf-8"))
    ev = json.loads((EVAL_DIR / "evaluation_report.json").read_text(
        encoding="utf-8"))
    cfg = json.loads((CONFIG_DIR / "final_features.json").read_text(
        encoding="utf-8"))
    m = ev["test_metrics"]

    groups: dict = {}
    for c in copied:
        parts = c["artefact"].replace("\\", "/").split("/")
        key = "/".join(parts[:-1]) or "(root)"   # directory, not the file
        groups.setdefault(key, []).append(c)

    L = ["# Report Material — Index\n",
         "> Auto-generated by `scripts/18_assemble_report_material.py`. Every "
         "artefact here is a copy of a file the pipeline produced; nothing was "
         "drawn or tabulated by hand.\n",
         "## Headline numbers, for captions\n",
         "| Quantity | Value |", "|---|--:|",
         f"| Records (culture × organism × antibiotic) | "
         f"{build['step7_final']['n_rows']:,} |",
         f"| Patients | {build['step7_final']['n_patients']:,} |",
         f"| Positive culture orders | {build['step7_final']['n_orders']:,} |",
         f"| Antibiotics | 54 |",
         f"| Organisms (cleaned) | "
         f"{build['step1_cohort']['organism_clean_distinct']} |",
         f"| Non-susceptible rate | "
         f"{100*build['step7_final']['target_positive_rate']:.2f}% |",
         f"| Final features | {cfg['n_features']} |",
         f"| Final model | LightGBM |",
         f"| Test PR-AUC | {m['pr_auc']} |",
         f"| Test ROC-AUC | {m['roc_auc']} |",
         f"| Test sensitivity | {m['recall_sensitivity']} |",
         f"| Test specificity | {m['specificity']} |",
         f"| Temporal-holdout PR-AUC | {ev['temporal_metrics']['pr_auc']} |",
         ""]

    for key in sorted(groups):
        L.append(f"## `{key}`\n")
        L.append("| File | Description | Produced by |")
        L.append("|---|---|---|")
        for c in sorted(groups[key], key=lambda x: x["artefact"]):
            fname = c["artefact"].replace("\\", "/").split("/")[-1]
            L.append(f"| `{fname}` | {c['description']} | "
                     f"`{c['produced_by']}` |")
        L.append("")

    L.append("## Regenerating everything\n")
    L.append("```bash\npython scripts/run_all.py        # full pipeline\n"
             "python scripts/run_all.py --docs # documents and figures only\n"
             "```\n")

    (REPORT_DIR / "INDEX.md").write_text("\n".join(L), encoding="utf-8")
    (REPORT_DIR / "report_material_index.json").write_text(
        json.dumps(copied, indent=2), encoding="utf-8")

    print(f"\nCopied {len(copied)} artefacts into {REPORT_DIR}")
    for key in sorted(groups):
        print(f"  {key:<32} {len(groups[key])} files")
    print(f"Wrote {REPORT_DIR / 'INDEX.md'}")


if __name__ == "__main__":
    main()
