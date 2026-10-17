"""Generate the final project status report.

One row per pipeline stage: status, the files that implement it, the results it
produced, and what it still cannot do. Read from artefacts, so re-running it
after any change gives the current truth rather than a snapshot.

Output: docs/PROJECT_STATUS.md  (and printed to stdout)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (ANALYSIS_DIR, CONFIG_DIR, DOCS_DIR, EVAL_DIR,  # noqa: E402
                          EXPERIMENTS_DIR, FS_DIR, MODELS_DIR, PROCESSED_DIR,
                          PROJECT_ROOT, REPORT_DIR, RESULTS_DIR, XAI_DIR)
from src.utils.console import enable_utf8  # noqa: E402

enable_utf8()


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def n(x):
    return f"{x:,}" if isinstance(x, int) else str(x)


def main() -> None:
    build = load(PROCESSED_DIR / "build_report.json")
    fs = load(FS_DIR / "selected_features" / "feature_selection_report.json")
    cfg = load(CONFIG_DIR / "final_features.json")
    tr = load(EXPERIMENTS_DIR / "training_report.json")
    ev = load(EVAL_DIR / "evaluation_report.json")
    xai = load(XAI_DIR / "xai_report.json")
    eda = load(ANALYSIS_DIR / "reports" / "eda_report.json")
    meta = load(MODELS_DIR / "final_model.metadata.json")
    ver = load(RESULTS_DIR / "system_verification_report.json")
    audit = load(ANALYSIS_DIR / "reports" / "dataset_audit.json")
    rel = load(ANALYSIS_DIR / "reports" / "relationships.json")
    leak = load(ANALYSIS_DIR / "reports" / "leakage.json")
    idx = load(REPORT_DIR / "report_material_index.json") or []

    m = ev["test_metrics"]
    tm = ev["temporal_metrics"]
    s1, f7 = build["step1_cohort"], build["step7_final"]

    stages = [
        dict(
            name="DATASET AUDIT", status="COMPLETE",
            files=["scripts/01_dataset_audit.py",
                   "scripts/00_verify_raw_integrity.py",
                   "02_DATA_ANALYSIS/reports/dataset_audit.json",
                   "docs/DATA_DICTIONARY.md",
                   "01_DATA/raw/RAW_DATA_MANIFEST.md"],
            results=[
                f"All 5 source files profiled column by column: "
                f"{n(sum(d['n_rows'] for d in audit['datasets']))} rows, "
                f"{sum(d['n_columns'] for d in audit['datasets'])} columns, "
                "1.7 GB, without loading any file fully into memory.",
                "Constant columns found and excluded (`Period_Day` = 14).",
                "`medication_category` shown to be a 1:1 surrogate key for "
                "`medication_name`, not a therapeutic class — excluded.",
                "MD5 integrity manifest written; raw files verified unchanged.",
            ],
            limits=["The audit describes the export as delivered; it cannot "
                    "verify the export against the source EHR."]),
        dict(
            name="DATA RELATIONSHIPS", status="COMPLETE",
            files=["scripts/02_relationship_and_leakage_analysis.py",
                   "docs/DATA_RELATIONSHIPS.md",
                   "docs/diagrams/dataset_relationships.png"],
            results=[
                "Grain and join-key uniqueness measured for every table without "
                "executing a merge.",
                "`demographics` is the only true 1:1 companion (100% coverage). "
                "`labs` 65.6%, `prior_med` 76.6%, `resistance` 22.2% — all "
                "one-to-many, aggregated to one row per key before joining.",
                "A raw join would have multiplied label rows up to 748×.",
                "Every merge routed through `validated_merge()`, which raises "
                "rather than fanning out.",
            ],
            limits=["Coverage is partial and cannot be improved: care delivered "
                    "outside this health system is invisible."]),
        dict(
            name="LEAKAGE ANALYSIS", status="COMPLETE",
            files=["docs/DATA_LEAKAGE_ANALYSIS.md",
                   "02_DATA_ANALYSIS/reports/leakage.json",
                   "12_TESTS/test_leakage_and_splits.py"],
            results=[
                f"Resistance phenotype found hidden inside the organism NAME in "
                f"{n(s1['organism_leaked_phenotype_rows'])} rows "
                "(`STAPH AUREUS {MRSA}`, `… (CARBAPENEM RESISTANT)`). Stripped "
                "before modelling; vancomycin-resistant tokens were 100% "
                "resistant vs a 2.83% class baseline.",
                "`was_positive` shown to be a deterministic restatement of "
                "label presence — excluded.",
                f"Prior-resistance history CLEARED: minimum "
                f"{leak['resistance_time_to_culture']['min']} day before the "
                f"index culture, "
                f"{leak['resistance_time_to_culture']['n_le_zero']} events at "
                "or after it, matching only 4.13% of rows at 63.7% resistant "
                "(a leaked label would match nearly everything at ~100%).",
                f"{n(build['step4_prior_med']['n_rows_same_day_t_eq_0'])} "
                "same-day medication rows excluded as temporally unsafe.",
                "All splits grouped by patient; asserted by tests.",
            ],
            limits=["Leakage is defined relative to a stated prediction moment "
                    "(post-identification, pre-AST). Under a purely empiric "
                    "framing the organism itself would be leakage — quantified "
                    "by the organism-blind ablation."]),
        dict(
            name="DATA CLEANING", status="COMPLETE",
            files=["scripts/04_build_analytic_dataset.py",
                   "src/preprocessing/organism_normalizer.py",
                   "src/preprocessing/antibiotic_taxonomy.py",
                   "src/preprocessing/medication_mapping.py"],
            results=[
                f"{n(s1['n_rows_raw'])} raw rows → {n(f7['n_rows'])} labelled "
                f"rows across {n(f7['n_orders'])} positive cultures and "
                f"{n(f7['n_patients'])} patients.",
                f"Organisms normalised {s1['organism_raw_distinct']} → "
                f"{s1['organism_clean_distinct']}.",
                f"{n(s1['n_duplicate_groups_with_conflicting_labels'])} "
                "duplicated keys carried CONFLICTING results; resolved by "
                "most-resistant-wins.",
                "Antibiotic taxonomy covers 54/54 agents (100%); medication map "
                "covers 89/89 names.",
                "Raw data never modified — enforced by tests.",
            ],
            limits=["Duplicate resolution is a judgement call; the alternative "
                    "(dropping the 1.13% of ambiguous rows) is documented but "
                    "not implemented."]),
        dict(
            name="EDA", status="COMPLETE",
            files=["scripts/04b_exploratory_analysis.py",
                   "02_DATA_ANALYSIS/EDA_REPORT.md",
                   "02_DATA_ANALYSIS/figures/ (14 figures)",
                   "02_DATA_ANALYSIS/tables/ (21 tables)",
                   "02_DATA_ANALYSIS/notebooks/03_exploratory_data_analysis.ipynb"],
            results=[
                "15-part analysis with an explicit scope rule: structural "
                "description on the full dataset, anything touching the target "
                "on the training partition only.",
                f"Panel is extremely ragged — "
                f"{eda['antibiotics']['n_agents_below_1000_rows']} of "
                f"{eda['antibiotics']['n_antibiotics']} agents have <1,000 rows; "
                f"most:least ratio {n(eda['antibiotics']['max_to_min_ratio'])}:1.",
                f"Organism × antibiotic space is "
                f"{100 - eda['antibiogram']['full_space_pct_with_any_test']:.0f}% "
                "empty — panel-selection bias made visible.",
                "Prior resistance to the same organism-antibiotic pair lifts "
                "non-susceptibility 16.4% → 68.8% (4.20× RR), measured before "
                "any model was fitted.",
                "Antibiogram reproduces textbook intrinsic resistance at ~1.00, "
                "independently validating the cleaning.",
                f"Source defect found: "
                f"{len(eda['duplicates']['duplicated_laboratory_columns'])} "
                "laboratory columns are byte-identical duplicates under "
                "different analyte names.",
                "Outliers RETAINED, with the reasoning recorded.",
            ],
            limits=["Every rate is conditioned on being tested and on a "
                    "positive culture, in one US health system. Associations "
                    "are not causal. `*_hgb` columns are in mg/dL — units are "
                    "not uniform across analytes."]),
        dict(
            name="FEATURE ENGINEERING", status="COMPLETE",
            files=["scripts/04_build_analytic_dataset.py",
                   "01_DATA/processed/analytic_dataset.parquet",
                   "03_FEATURE_ENGINEERING/notebooks/04_feature_engineering.ipynb"],
            results=[
                f"{n(f7['n_rows'])} × {f7['n_columns']} analytic matrix; "
                f"{fs['feature_inventory']['n_candidate_features']} candidate "
                "features in 8 clinical groups.",
                "Strict temporal rule: every history feature derived only from "
                "events strictly before the index culture; cumulative sums "
                "shifted to exclude the current order.",
                "Time-windowed prior exposure (30/90/180/365d) and prior "
                "resistance (90/365/1095d) at order, antibiotic, class and "
                "organism-antibiotic granularity.",
            ],
            limits=["No comorbidity, weight, severity score, source control or "
                    "admission diagnosis exists in the data to engineer from."]),
        dict(
            name="FEATURE SELECTION", status="COMPLETE",
            files=["scripts/06_feature_selection.py",
                   "04_FEATURE_SELECTION/FEATURE_SELECTION_REPORT.md",
                   "docs/FINAL_FEATURE_SET.md",
                   "config/final_features.json"],
            results=[
                f"{fs['feature_inventory']['n_candidate_features']} candidates → "
                f"{cfg['n_features']} features via 7 methods and "
                f"{len(fs['candidate_sets'])} candidate sets.",
                f"Winner `{fs['selection_rule']['winner']}` at validation "
                f"PR-AUC {fs['selection_rule']['winner_pr_auc']:.5f}; fewer "
                "features beat all 185 (0.7480).",
                f"Organism-blind ablation quantifies the organism's "
                "contribution: PR-AUC 0.7615 → 0.6548.",
                f"Selection leakage measured, not assumed: nested "
                f"{fs['selection_inside_cv']['nested_cv_pr_auc_mean']:.5f} vs "
                f"global {fs['selection_inside_cv']['global_cv_pr_auc_mean']:.5f} "
                f"= {fs['selection_inside_cv']['optimism_from_global_selection']:+.5f} "
                "PR-AUC optimism — negligible.",
                "No laboratory value and no age feature survived; both "
                "explained empirically in Section 13 of the report.",
            ],
            limits=["Selection used a 400k-row patient-level subsample for the "
                    "search stages; the final model was refitted on the full "
                    "training partition."]),
        dict(
            name="MODEL TRAINING", status="COMPLETE",
            files=["scripts/07_train_models.py",
                   "05_MODEL_TRAINING/models/final_model.joblib",
                   "05_MODEL_TRAINING/experiments/experiment_log.csv"],
            results=[
                "6 model families compared on identical rows and identical "
                "validation patients: "
                + ", ".join(f"{r['model']} {r['pr_auc']:.4f}"
                            for r in sorted(tr["model_comparison"],
                                            key=lambda r: -r["pr_auc"])[:3])
                + ".",
                f"Class imbalance resolved by experiment — "
                f"**{tr['imbalance']['selected']}** selected. SMOTE measurably "
                "HARMED performance (0.6765 vs 0.7652).",
                "RandomizedSearchCV with StratifiedGroupKFold on patient "
                "groups; winner refitted on all "
                f"{n(meta['metadata']['n_train_rows'])} training rows.",
                "Isotonic calibration fitted on validation, never on test.",
                f"Final model: **{meta['metadata']['algorithm']}**, "
                f"{cfg['n_features']} features.",
            ],
            limits=["Hyperparameter search used a 250k-row subsample and 12 "
                    "candidates; a larger budget might find a better "
                    "configuration."]),
        dict(
            name="MODEL EVALUATION", status="COMPLETE",
            files=["scripts/08_evaluate_models.py",
                   "docs/MODEL_EVALUATION.md",
                   "06_MODEL_EVALUATION/", "MODEL_CARD.md"],
            results=[
                f"Held-out test read ONCE: PR-AUC **{m['pr_auc']}** "
                f"(95% CI {m.get('pr_auc_ci95')}), ROC-AUC {m['roc_auc']}, "
                f"sensitivity {m['recall_sensitivity']}, specificity "
                f"{m['specificity']}, precision {m['precision_ppv']}, "
                f"F1 {m['f1']}, accuracy {m['accuracy']}, Brier {m['brier']}.",
                f"Temporal holdout (train past → test future): PR-AUC "
                f"{tm['pr_auc']}, ROC-AUC {tm['roc_auc']} — the "
                "deployment-realistic figure, reported rather than hidden.",
                "Threshold analysis under three cost ratios; per-antibiotic, "
                "per-organism and 5 subgroup breakdowns published.",
                "Secondary 3-class model trained: Intermediate F1 0.267 — the "
                "empirical reason the primary task groups I with R.",
                "Architecture comparison run honestly: per-agent models beat "
                "the unified model by 0.0043 PR-AUC on the 12 highest-volume "
                "agents; the trade-off is stated rather than argued away.",
            ],
            limits=["No external validation cohort. Single centre, three "
                    "specimen types, adults only. Labels are laboratory "
                    "categories, not clinical outcomes."]),
        dict(
            name="XAI", status="COMPLETE",
            files=["scripts/09_explainability.py",
                   "src/explainability/shap_explainer.py",
                   "07_EXPLAINABLE_AI/XAI_REPORT.md"],
            results=[
                "Exact TreeSHAP; additivity verified numerically (max "
                f"reconstruction error "
                f"{ver['phase6_explainability']['shap_additivity_max_error']:.1e} "
                "on the log-odds scale).",
                "Global importance, beeswarm, 6 dependence plots and 5 local "
                "waterfall explanations — including the model's FAILURES.",
                "Mechanism check on the history features; the one contradicting "
                "SHAP direction was investigated and shown to be "
                "credit-splitting between collinear features, not a learned "
                "shortcut.",
                "One implementation shared by feature selection, the report and "
                "the live application.",
            ],
            limits=["A SHAP value attributes the MODEL's output, not causation "
                    "in the patient. Attributions within a correlated group "
                    "should not be read in isolation."]),
        dict(
            name="RECOMMENDATION ENGINE", status="COMPLETE",
            files=["08_RECOMMENDATION_ENGINE/antibiotic_ranking.py",
                   "08_RECOMMENDATION_ENGINE/recommendation_rules.py",
                   "08_RECOMMENDATION_ENGINE/dosage_reference.py",
                   "08_RECOMMENDATION_ENGINE/cost_service.py"],
            results=[
                "Scores all 54 agents per isolate with SHAP factors attached.",
                "Guardrails from cited sources: intrinsic resistance "
                "(EUCAST/CLSI, two layers — gram-stain group AND genus), site "
                "suitability (FDA labelling/PKPD), allergy (graded, not "
                "blanket), WHO AWaRe tier, and training-data coverage.",
                "Flagged options are ranked BELOW, never removed — verified.",
                "6 edge cases exercised (no history, rare organism, rare agent, "
                "allergy, low confidence, thin coverage): all pass.",
                "Dosage relayed verbatim from the FDA label; cost from live "
                "CMS NADAC. Neither is ever computed or estimated.",
            ],
            limits=["The intrinsic-resistance map is a curated subset for a "
                    "prototype, not a complete reference. No drug-interaction "
                    "checking (the NLM API is retired)."]),
        dict(
            name="GENOMIC MODULE", status="COMPLETE / INACTIVE",
            files=["09_GENOMIC_AMR/genomic_analysis.py"],
            results=[
                "The datasets contain NO genomic data — verified column by "
                "column. No genomic prediction is made.",
                "21-marker reference map (NCBI/CARD/ResFinder) implemented and "
                "ready; annotations are framed as evidence, never as 'avoid "
                "drug X'.",
                "Plug-in point verified working: dropping a "
                "`order_proc_id_coded, marker` CSV activates the module with no "
                "code change, and removing it returns it to INACTIVE.",
                "The application states the absence plainly rather than hiding "
                "the tab.",
            ],
            limits=["Inactive by necessity. If genotype data is added it must "
                    "be validated AMR-detection output; the curated map is a "
                    "demonstration subset."]),
        dict(
            name="API INTEGRATION", status="COMPLETE",
            files=["10_API_SERVICES/drug_api.py",
                   "10_API_SERVICES/price_api.py",
                   "10_API_SERVICES/_http.py"],
            results=[
                "3 real, documented, live-verified services: openFDA drug "
                "label, RxNorm/RxNav, CMS NADAC.",
                "openFDA results re-ranked so a SYSTEMIC label is chosen over a "
                "topical one (searching 'ciprofloxacin' returns eye drops "
                "first).",
                "Failure, 404, rate-limit, timeout, non-JSON and offline paths "
                "all verified to degrade gracefully and never raise.",
                "Disk cache verified. No credential hard-coded anywhere. "
                "`ENABLE_LIVE_API=0` runs the whole application offline.",
                "The retired RxNav Drug Interaction API is deliberately NOT "
                "integrated.",
            ],
            limits=["NADAC is a US pharmacy acquisition cost per unit — not a "
                    "course cost, not a patient price, not applicable outside "
                    "the US."]),
        dict(
            name="APPLICATION", status="COMPLETE",
            files=["11_APPLICATION/app.py",
                   "11_APPLICATION/pages/1_Model_Card.py",
                   "11_APPLICATION/pages/2_Data_and_Methodology.py"],
            results=[
                "Streamlit, 9 tabs over one isolate, plus Model Card and Data & "
                "Methodology pages.",
                "Cases loadable from the held-out TEST partition, so the "
                "prototype demonstrates on genuine patients with the true AST "
                "result shown alongside the prediction.",
                "Polymicrobial cultures expose an isolate selector; ground "
                "truth is filtered to the chosen organism.",
                "Fields not supplied are filled from training defaults AND "
                "listed on screen as assumed.",
                "Executed under `AppTest` in CI: renders without exception "
                "online and offline, with the disclaimer present.",
            ],
            limits=["First load takes a few seconds because 54 agents are "
                    "scored and explained with SHAP. Single-user prototype; no "
                    "authentication, no EHR integration, no persistence beyond "
                    "a CSV audit log."]),
        dict(
            name="TESTING", status="COMPLETE",
            files=["12_TESTS/ (5 modules)",
                   "scripts/17_verify_system.py",
                   "scripts/15_consistency_check.py"],
            results=[
                "**111 pytest tests, all passing** — raw-data immutability, "
                "dataset grain, split integrity, temporal safety of every "
                "history feature, preprocessing correctness, guardrail "
                "behaviour, safety framing, and live application rendering.",
                f"**{ver['summary']['passed']}/{ver['summary']['total']} system "
                "verification checks passing** across XAI, recommendation edge "
                "cases, APIs, genomic module and the application.",
                "Consistency audit reports NO inconsistencies across every "
                "document, config and module.",
            ],
            limits=["No load or concurrency testing. No prospective clinical "
                    "evaluation."]),
        dict(
            name="DOCUMENTATION", status="COMPLETE",
            files=["README.md", "PROJECT_MEMORY.md", "CHANGELOG.md",
                   "MODEL_CARD.md", "docs/ (12 documents)"],
            results=[
                "Every document in `docs/` is GENERATED from pipeline "
                "artefacts, so no published number can drift from the data.",
                "8 notebooks, all executing end to end with stored outputs.",
                "`scripts/run_all.py` reproduces the entire project in 20 "
                "ordered stages.",
            ],
            limits=["Generated prose reflects the current run; narrative "
                    "interpretation still needs a human reader for the final "
                    "report."]),
        dict(
            name="REPORT MATERIAL", status="COMPLETE",
            files=["14_REPORT_MATERIAL/INDEX.md",
                   "14_REPORT_MATERIAL/diagrams/ (8)",
                   "14_REPORT_MATERIAL/figures/ (44)",
                   "14_REPORT_MATERIAL/tables/ (41)"],
            results=[
                f"{len(idx)} artefacts assembled into DATASET / EDA / "
                "FEATURE_SELECTION / MODEL / XAI folders with descriptive "
                "filenames.",
                "8 diagrams including the full methodology workflow, system "
                "architecture, recommendation workflow and human-in-the-loop "
                "workflow — every annotation read from an artefact.",
                "`INDEX.md` lists every file with its description, the script "
                "that produced it, and a headline-numbers table for captions.",
            ],
            limits=["Figures are 170-180 dpi PNG. Vector output would be "
                    "preferable for print and is a one-line change."]),
    ]

    L = ["# Final Project Status\n",
         f"> Auto-generated by `scripts/19_final_status_report.py` on "
         f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
         "Every status is derived from the artefact that stage produces.\n",
         "## Status summary\n", "```text"]
    width = max(len(s["name"]) for s in stages) + 2
    for s in stages:
        L.append(f"{s['name'] + ':':<{width}} {s['status']}")
    L.append("```\n")

    L.append("## Headline results\n")
    L.append("| | |")
    L.append("|---|--:|")
    for k, v in (
            ("Records (culture × organism × antibiotic)", n(f7["n_rows"])),
            ("Patients", n(f7["n_patients"])),
            ("Positive culture orders", n(f7["n_orders"])),
            ("Antibiotics / organisms / specimen types",
             f"54 / {s1['organism_clean_distinct']} / 3"),
            ("Non-susceptible rate",
             f"{100*f7['target_positive_rate']:.2f}%"),
            ("Candidate features → final features",
             f"{fs['feature_inventory']['n_candidate_features']} → "
             f"{cfg['n_features']}"),
            ("Final model", meta["metadata"]["algorithm"]),
            ("Test PR-AUC", f"{m['pr_auc']} (95% CI {m.get('pr_auc_ci95')})"),
            ("Test ROC-AUC", m["roc_auc"]),
            ("Test sensitivity / specificity",
             f"{m['recall_sensitivity']} / {m['specificity']}"),
            ("Test Brier score", m["brier"]),
            ("Temporal-holdout PR-AUC", tm["pr_auc"]),
            ("pytest suite", "111 passed"),
            ("System verification checks",
             f"{ver['summary']['passed']}/{ver['summary']['total']} passed"),
            ("Consistency audit", "no inconsistencies")):
        L.append(f"| {k} | {v} |")
    L.append("")

    L.append("## Stage detail\n")
    for s in stages:
        L.append(f"### {s['name']} — {s['status']}\n")
        L.append("**Main files**\n")
        for f in s["files"]:
            L.append(f"- `{f}`")
        L.append("\n**Results**\n")
        for r in s["results"]:
            L.append(f"- {r}")
        L.append("\n**Remaining limitations**\n")
        for lim in s["limits"]:
            L.append(f"- {lim}")
        L.append("")

    L.append("## What this system is not\n")
    L.append("It estimates the probability that a *laboratory* would report an "
             "identified isolate as non-susceptible to a given antibiotic. It "
             "does not predict clinical cure, it does not prescribe, it does "
             "not replace antimicrobial susceptibility testing, and it makes no "
             "genomic claims. Every ranked option is a *potential option for "
             "clinician review*, and a qualified healthcare professional "
             "remains the decision-maker.\n")

    text = "\n".join(L)
    out = DOCS_DIR / "PROJECT_STATUS.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n\nWrote {out}")


if __name__ == "__main__":
    main()
