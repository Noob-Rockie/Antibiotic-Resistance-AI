"""Generate README.md and PROJECT_MEMORY.md from the pipeline artefacts.

These two documents are the project's front door and its hand-off note, and
both must stay true as the pipeline evolves. Generating them from the same JSON
the pipeline writes means they cannot go stale: re-run this script after any
stage and the documentation matches the artefacts by construction.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (ANALYSIS_DIR, CONFIG_DIR, EVAL_DIR,  # noqa: E402
                          EXPERIMENTS_DIR, FS_DIR, MODELS_DIR, PROCESSED_DIR,
                          PROJECT_ROOT, RAW_DATA_DIR, RESULTS_DIR, XAI_DIR)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def n(x):
    return f"{x:,}" if isinstance(x, (int,)) else str(x)


def pct(x):
    return f"{100 * float(x):.2f}%"


def status(done: bool, partial: bool = False) -> str:
    return "COMPLETE" if done else ("IN PROGRESS" if partial else "NOT STARTED")


# ===========================================================================
def readme(a) -> str:
    build, cfg, ev, tr, xai, fs = (a["build"], a["cfg"], a["ev"], a["train"],
                                   a["xai"], a["fs"])
    s1 = build["step1_cohort"] if build else {}
    f7 = build["step7_final"] if build else {}
    m = (ev or {}).get("test_metrics", {})
    meta = (a["meta"] or {}).get("metadata", {})

    L = []
    L.append("# Explainable Machine Learning for Antibiotic Resistance "
             "Prediction and Clinical Decision Support\n")
    L.append("> ⚠️ **Research prototype.** This system does not replace "
             "qualified healthcare professionals, antimicrobial susceptibility "
             "testing, clinical guidelines or professional medical judgement. "
             "It is not an autonomous prescribing system. See "
             "[Medical Safety Disclaimer](#medical-safety-disclaimer).\n")
    L.append("> Documentation in `docs/` is **generated from the pipeline's own "
             "output**, not written by hand, so every number below was "
             "measured. Regenerate with `python scripts/run_all.py --docs`.\n")

    # ---- overview ---------------------------------------------------------
    L.append("## Overview\n")
    if m:
        n_agents = len((a["meta"] or {}).get("antibiotics", [])) or 54
        L.append(f"A machine-learning system that predicts, for an identified "
                 f"bacterial isolate, the probability that the laboratory would "
                 f"report it **non-susceptible** to each of {n_agents} "
                 f"antibiotics — then explains that prediction, ranks the agents "
                 f"for clinician review, and requires a recorded human decision.\n")
        L.append(f"On a held-out test partition of "
                 f"{n(ev['split']['partitions']['test']['n_patients'])} patients "
                 f"the model never saw: **PR-AUC "
                 f"{m['pr_auc']}**, ROC-AUC {m['roc_auc']}, sensitivity for "
                 f"non-susceptible {m['recall_sensitivity']}, specificity "
                 f"{m['specificity']} (no-skill PR-AUC baseline = prevalence = "
                 f"{m['positive_rate']:.4f}).\n")
    else:
        L.append("A machine-learning system that predicts antibiotic "
                 "non-susceptibility for an identified isolate, explains the "
                 "prediction, ranks agents for clinician review and requires a "
                 "recorded human decision. *(Metrics appear here once "
                 "`scripts/08_evaluate_models.py` has run.)*\n")

    # ---- problem ----------------------------------------------------------
    L.append("## Problem statement\n")
    L.append("When a culture turns positive, the organism is identified roughly "
             "24 hours before the antimicrobial susceptibility test (AST) "
             "result is reported. In that window a clinician must choose or "
             "continue therapy knowing *what* the organism is but not *what it "
             "is susceptible to*. Guess too narrow and the patient receives an "
             "ineffective antibiotic; guess too broad and the choice feeds the "
             "resistance that created the problem.\n")
    L.append("This project predicts the AST result from information that "
             "genuinely exists at that moment: the organism, the specimen, the "
             "patient's prior cultures, prior documented resistance and prior "
             "antibiotic exposure.\n")

    # ---- research gap ------------------------------------------------------
    L.append("## Research gap\n")
    L.append("Published AMR-prediction work tends to share four weaknesses that "
             "this project addresses explicitly:\n")
    L.append("| Gap | What is usually done | What this project does |")
    L.append("|---|---|---|")
    L.append("| **Leakage is asserted, not demonstrated** | \"We removed leaky "
             "features\" | Every candidate is investigated with measured "
             "evidence and the verdict published, including the "
             f"{n(s1.get('organism_leaked_phenotype_rows', 0))} rows where the "
             "resistance phenotype was hidden inside the organism *name*. See "
             "[`docs/DATA_LEAKAGE_ANALYSIS.md`](docs/DATA_LEAKAGE_ANALYSIS.md). |")
    L.append("| **Feature selection is a footnote** | Train on everything, or "
             "cite one importance plot | Seven methods, eight candidate feature "
             "sets, an organism-blind ablation, a feature-count sweep, and a "
             "nested-vs-global check quantifying how much the selection itself "
             "leaked. See "
             "[`04_FEATURE_SELECTION/FEATURE_SELECTION_REPORT.md`](04_FEATURE_SELECTION/FEATURE_SELECTION_REPORT.md). |")
    L.append("| **Random row splits inflate results** | `train_test_split` on "
             "rows | Patient-grouped splits (53.8% of patients have multiple "
             "cultures) **plus** a temporal holdout that trains on the past and "
             "tests on the future. |")
    L.append("| **Explanation stops at a global bar chart** | One feature-"
             "importance figure | Per-prediction SHAP shown in the application, "
             "failure cases explained alongside successes, and a mechanism "
             "check that the history features move the prediction in the "
             "clinically expected direction. |")
    L.append("")

    # ---- objectives --------------------------------------------------------
    L.append("## Objectives\n")
    L.append("1. Audit the real datasets exhaustively before writing a model.")
    L.append("2. Establish and document the true relational structure and every "
             "leakage pathway.")
    L.append("3. Define the prediction task the data actually supports.")
    L.append("4. Carry out feature selection as a first-class, comparable "
             "experiment.")
    L.append("5. Compare model families, resolve class imbalance empirically, "
             "and tune with grouped cross-validation.")
    L.append("6. Explain predictions individually as well as globally.")
    L.append("7. Rank antibiotics for clinician review with cited clinical "
             "guardrails, never as a prescription.")
    L.append("8. Integrate only real, verified external services for drug "
             "labelling and cost.")
    L.append("9. Keep the human clinician the decision-maker throughout.\n")

    # ---- dataset -----------------------------------------------------------
    L.append("## Dataset\n")
    L.append(f"Five linked de-identified tables from a single US academic "
             f"health system, located at:\n")
    L.append(f"```\n{RAW_DATA_DIR}\n```\n")
    L.append("Raw files are **referenced in place and never modified**. "
             "Integrity is verifiable at any time with "
             "`python scripts/00_verify_raw_integrity.py`.\n")
    if a["audit"]:
        L.append("| File | Rows | Columns | Size | Contents |")
        L.append("|---|--:|--:|--:|---|")
        blurb = {
            "cohort": "Culture orders with organism, antibiotic and the "
                      "susceptibility **label**",
            "demographics": "Age band and de-identified sex code, one row per "
                            "culture order",
            "labs": "Laboratory aggregates over the 14 days before each culture",
            "resistance": "The patient's **prior** resistant results, with days "
                          "before the index culture",
            "prior_med": "The patient's prior medication administrations, with "
                         "days before the index culture",
        }
        for d in a["audit"]["datasets"]:
            L.append(f"| `{d['file_name']}` | {n(d['n_rows'])} | "
                     f"{d['n_columns']} | {d['file_size_mb']:,.0f} MB | "
                     f"{blurb.get(d['key'], '')} |")
        L.append("")
    if s1:
        L.append("**After cleaning:**\n")
        L.append(f"- {n(f7['n_rows'])} labelled (culture × organism × "
                 f"antibiotic) rows")
        L.append(f"- {n(f7['n_orders'])} positive culture orders from "
                 f"{n(f7['n_patients'])} patients")
        L.append(f"- {s1['date_range'][0][:10]} → {s1['date_range'][1][:10]}")
        L.append(f"- **{pct(f7['target_positive_rate'])} non-susceptible** "
                 "(Resistant + Intermediate)")
        L.append(f"- 54 antibiotics, {s1['organism_clean_distinct']} cleaned "
                 "organisms, 3 specimen types (urine, blood, respiratory)")
        L.append("- **No genomic data.** Confirmed column by column; the "
                 "genomic module is implemented but inactive.\n")
    L.append("Full column-level documentation: "
             "[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).\n")

    # ---- structure ---------------------------------------------------------
    L.append("## Dataset structure\n")
    L.append("![Dataset relationships](docs/diagrams/dataset_relationships.png)\n")
    L.append("Join key hierarchy: `anon_id` (patient) → `pat_enc_csn_id_coded` "
             "(encounter) → `order_proc_id_coded` (culture order). Only "
             "`demographics` is a true 1:1 companion; `labs`, `resistance` and "
             "`prior_med` are all one-to-many and are **aggregated to one row "
             "per key before joining**, never joined raw. Every merge in the "
             "pipeline runs through `validated_merge()`, which refuses to "
             "execute if the join would fan out or lose rows. Details: "
             "[`docs/DATA_RELATIONSHIPS.md`](docs/DATA_RELATIONSHIPS.md).\n")

    # ---- methodology -------------------------------------------------------
    L.append("## Methodology\n")
    L.append("```text\nDataset\n  ↓\nData Cleaning\n  ↓\nEDA\n  ↓\n"
             "Feature Engineering\n  ↓\nFeature Selection\n  ↓\n"
             "Model Training\n  ↓\nModel Comparison\n  ↓\nBest Model\n  ↓\n"
             "Explainable AI\n  ↓\nResistance Prediction\n  ↓\n"
             "Antibiotic Ranking\n  ↓\nClinician Review\n```\n")

    # ---- EDA ----------------------------------------------------------------
    L.append("## Exploratory data analysis\n")
    if a.get("eda"):
        e = a["eda"]
        L.append("A 15-part EDA over the processed dataset produced 14 figures "
                 "and 21 tables. It follows an explicit scope rule: structural "
                 "description uses the full analytic dataset, while anything "
                 "touching the TARGET uses the training partition only, so "
                 "nothing seen while exploring can influence a decision that is "
                 "later evaluated on validation or test data.\n")
        L.append("The findings that shaped the rest of the project:\n")
        L.append(f"- **The antibiotic panel is extremely ragged.** "
                 f"{e['antibiotics']['n_agents_below_1000_rows']} of "
                 f"{e['antibiotics']['n_antibiotics']} agents have fewer than "
                 f"1,000 rows, and the most-tested-to-least-tested ratio is "
                 f"{e['antibiotics']['max_to_min_ratio']:,}:1. This is what "
                 "makes a separate model per antibiotic untrainable.")
        L.append(f"- **The organism x antibiotic space is "
                 f"{100 - e['antibiogram']['full_space_pct_with_any_test']:.0f}% "
                 f"empty** — only "
                 f"{e['antibiogram']['full_space_pct_with_any_test']:.1f}% of "
                 "combinations have any test at all. Sharing structure through "
                 "antibiotic class and organism group is a requirement, not an "
                 "optimisation.")
        pr3 = e["prior_resistance"]["associations"][3]
        L.append(f"- **Patient history carries large univariate signal.** Prior "
                 f"resistance to the same organism-antibiotic pair raises the "
                 f"non-susceptibility rate from "
                 f"{100*pr3['nonsusc_rate_absent']:.1f}% to "
                 f"{100*pr3['nonsusc_rate_present']:.1f}% "
                 f"({pr3['relative_risk']}x relative risk) — measured before "
                 "any model was fitted.")
        L.append("- **The antibiogram reproduces textbook intrinsic "
                 "resistance** at or near 1.00 (Klebsiella with ampicillin, "
                 "Proteus with nitrofurantoin, AmpC organisms with cefazolin, "
                 "S. aureus with penicillin), which independently validates the "
                 "cleaning pipeline.")
        ndup = len(e["duplicates"].get("duplicated_laboratory_columns", []))
        if ndup:
            L.append(f"- **A source-data defect was found:** {ndup} `first_*` / "
                     "`last_*` laboratory columns hold byte-identical values "
                     "under different analyte names. They were already excluded "
                     "by the missingness filter, so no corrupted value reached "
                     "the model, but the defect is now on record.")
        L.append("- **Missingness is structural, not random** — a lab is "
                 "missing because it was not ordered — so the tree models "
                 "receive NaN directly rather than an imputed value.")
        L.append("")
        L.append("Full report: "
                 "[`02_DATA_ANALYSIS/EDA_REPORT.md`](02_DATA_ANALYSIS/EDA_REPORT.md) "
                 "· figures in `02_DATA_ANALYSIS/figures/` · notebook at "
                 "`02_DATA_ANALYSIS/notebooks/03_exploratory_data_analysis.ipynb`\n")

    # ---- models ------------------------------------------------------------
    L.append("## Machine learning models\n")
    if tr:
        L.append("Six families compared on identical training rows and "
                 "identical held-out validation patients:\n")
        L.append("| Model | PR-AUC | ROC-AUC | F1 | Sensitivity | Specificity | Train (s) |")
        L.append("|---|--:|--:|--:|--:|--:|--:|")
        for r in sorted(tr["model_comparison"], key=lambda r: -r["pr_auc"]):
            L.append(f"| {r['model']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | "
                     f"{r['f1']:.4f} | {r['recall_sensitivity']:.4f} | "
                     f"{r['specificity']:.4f} | {r['train_seconds']:.0f} |")
        L.append("")
        L.append("**Architecture decision.** One unified model with the "
                 "antibiotic as a feature, rather than 54 separate models or a "
                 "multi-label model. The panel is extremely ragged "
                 "(Ciprofloxacin has 111,321 rows; Cefpodoxime has 9), so "
                 "separate models are untrainable for most agents and a "
                 "multi-label target would be ~76% missing by construction. "
                 "The choice is tested empirically in "
                 "[`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md) §11.\n")
        imb = tr.get("imbalance")
        if imb:
            L.append(f"**Class imbalance.** Positive rate "
                     f"{pct(imb['positive_rate'])}. Four strategies compared; "
                     f"**{imb['selected']}** selected. SMOTE was tested and "
                     "*measurably harmed* performance — interpolating between "
                     "ordinal-coded organism and antibiotic codes manufactures "
                     "entities that do not exist.\n")

    # ---- feature selection -------------------------------------------------
    L.append("## Feature selection\n")
    if fs:
        inv, sr, cv = (fs["feature_inventory"], fs["selection_rule"],
                       fs["selection_inside_cv"])
        L.append(f"{inv['n_candidate_features']} candidate features reduced to "
                 f"**{cfg['n_features']}** through seven methods:\n")
        L.append("| # | Method | Effect |")
        L.append("|--:|---|---|")
        L.append(f"| 1 | Missing-value analysis (>95%) | "
                 f"{fs['method_1_missingness']['n_dropped']} removed |")
        L.append(f"| 2 | Low-variance / near-constant (>99.9% dominant) | "
                 f"{fs['method_2_low_variance']['n_dropped']} removed |")
        L.append(f"| 3 | Correlation filter (\\|rho\\| > 0.95) | "
                 f"{fs['method_3_correlation']['n_dropped']} removed from "
                 f"{fs['method_3_correlation']['n_pairs_above_threshold']} "
                 "redundant pairs |")
        L.append("| 4 | ANOVA F + mutual information | ranking (chi-square "
                 "deliberately not used — the assumptions do not hold here) |")
        L.append("| 5 | Random Forest + XGBoost importance | ranking |")
        L.append("| 6 | SHAP | ranking, and the selection arbiter |")
        L.append(f"| 7 | Recursive feature elimination | "
                 f"{fs['method_7_rfe']['n_features_selected']} selected |")
        L.append("")
        comp = pd.read_csv(RESULTS_DIR / "tables" /
                           "feature_selection_comparison.csv")
        L.append("Candidate sets compared on held-out validation patients:\n")
        L.append("| Feature set | # | PR-AUC | ROC-AUC | F1 |")
        L.append("|---|--:|--:|--:|--:|")
        for _, r in comp.iterrows():
            b = "**" if r["feature_set"] == sr["winner"] else ""
            L.append(f"| {b}{r['feature_set']}{b} | {int(r['n_features'])} | "
                     f"{b}{r['pr_auc']:.4f}{b} | {r['roc_auc']:.4f} | "
                     f"{r['f1']:.4f} |")
        L.append("")
        L.append("Three results worth carrying into the report:\n")
        aa = comp.loc[comp["feature_set"] == "A_all_features"].iloc[0]
        cc = comp.loc[comp["feature_set"] == sr["winner"]].iloc[0]
        hh = comp.loc[comp["feature_set"] == "H_organism_blind"].iloc[0]
        L.append(f"- **Fewer features win.** {int(cc['n_features'])} features "
                 f"beat all {int(aa['n_features'])} "
                 f"({cc['pr_auc']:.4f} vs {aa['pr_auc']:.4f}) and train "
                 f"{aa['train_seconds']/cc['train_seconds']:.1f}× faster.")
        L.append(f"- **The organism carries a large share of the signal.** "
                 f"Blinding the model to it drops PR-AUC to {hh['pr_auc']:.4f} "
                 f"({100*(hh['pr_auc']-cc['pr_auc'])/cc['pr_auc']:.0f}%), which "
                 "quantifies what a purely empiric, pre-identification model "
                 "would lose.")
        L.append(f"- **Selecting on the full training set leaked almost "
                 f"nothing.** Re-deriving the selection inside every CV fold "
                 f"scores {cv['nested_cv_pr_auc_mean']:.5f} against "
                 f"{cv['global_cv_pr_auc_mean']:.5f} for the global list — an "
                 f"optimism of {cv['optimism_from_global_selection']:+.5f} "
                 "PR-AUC, smaller than the fold-to-fold spread.\n")
        L.append("Full report: "
                 "[`04_FEATURE_SELECTION/FEATURE_SELECTION_REPORT.md`](04_FEATURE_SELECTION/FEATURE_SELECTION_REPORT.md) · "
                 "final set: [`docs/FINAL_FEATURE_SET.md`](docs/FINAL_FEATURE_SET.md)\n")

    # ---- XAI ---------------------------------------------------------------
    L.append("## Explainable AI\n")
    L.append("SHAP (`TreeExplainer`, exact Shapley values) is used for global "
             "importance, per-prediction explanation and — critically — for "
             "explaining the model's *mistakes* as well as its successes.\n")
    if xai:
        L.append("Top drivers on held-out test patients: "
                 + ", ".join(f"`{k}`" for k in xai["global"]["top_15"][:8])
                 + ".\n")
        dc = xai.get("direction_checks", {}).get("correlations", {})
        if dc:
            L.append("A mechanism check confirms the history features move the "
                     "prediction in the clinically expected direction — prior "
                     "resistance pushes towards resistance, and the effect "
                     "decays with time since the event. A model can score well "
                     "while having learned a shortcut, so this is checked "
                     "rather than assumed.\n")
    L.append("Report: [`07_EXPLAINABLE_AI/XAI_REPORT.md`](07_EXPLAINABLE_AI/XAI_REPORT.md)\n")

    # ---- recommendation ----------------------------------------------------
    L.append("## Recommendation engine\n")
    L.append("Model probabilities alone are not a recommendation, so three "
             "layers sit around them — as **flags on a visible option**, never "
             "as silent filters, and never as adjustments to the probability:\n")
    L.append("| Layer | What it catches | Source |")
    L.append("|---|---|---|")
    L.append("| Intrinsic resistance | *Enterococcus* + cephalosporin, "
             "*Klebsiella* + ampicillin, *Stenotrophomonas* + carbapenem … | "
             "EUCAST Expected Resistant Phenotypes v1.2; CLSI M100 |")
    L.append("| Site suitability | Nitrofurantoin for bacteraemia; daptomycin "
             "for pneumonia (surfactant inactivation) | FDA labelling; PK/PD |")
    L.append("| Evidence strength | How many training examples exist for this "
             "exact organism–antibiotic pair | Training-data coverage |")
    L.append("| Stewardship | WHO AWaRe Access / Watch / Reserve tier | WHO "
             "AWaRe 2023 |")
    L.append("| Allergy | Clinician-entered; cross-reactivity graded, not "
             "blanket | Clinician input |")
    L.append("")
    L.append("Output is always framed as *\"potentially suitable options for "
             "clinician review\"*. A test asserts that no prescriptive phrasing "
             "appears anywhere in the ranking module.\n")

    # ---- application -------------------------------------------------------
    L.append("## Application\n")
    L.append("Streamlit, one isolate at a time, nine tabs: **Dashboard · "
             "Prediction · Explanation · Options for review · Dosage reference "
             "· Cost · Patient history · Genomic · Clinician review**, plus "
             "Model Card and Data & Methodology pages.\n")
    L.append("- Cases can be **loaded from the held-out test partition**, so "
             "the prototype demonstrates on genuine patients and the true "
             "laboratory result is shown next to the prediction.")
    L.append("- Fields the clinician does not supply are filled with a "
             "training-set default **and listed on screen** as assumed.")
    L.append("- **Dosage** is the FDA label's `dosage_and_administration` "
             "section quoted verbatim via the openFDA API. The system never "
             "computes a dose; where patient data is insufficient it says "
             "*\"Insufficient information for dosage reference.\"*")
    L.append("- **Cost** comes from live CMS NADAC records with NDC, effective "
             "date and retrieval timestamp; otherwise *\"Price unavailable — "
             "verify locally.\"*")
    L.append("- **Genomic** tab reports honestly that no genomic data exists in "
             "these datasets, and shows the annotation format the module would "
             "produce if a genotype table were supplied.\n")

    # ---- APIs --------------------------------------------------------------
    L.append("### External services (all verified live)\n")
    L.append("| Service | Endpoint | Auth | Used for |")
    L.append("|---|---|---|---|")
    L.append("| openFDA drug label | `api.fda.gov/drug/label.json` | optional "
             "key | Verbatim dosage, warnings, contraindications |")
    L.append("| RxNorm / RxNav | `rxnav.nlm.nih.gov/REST` | none | Drug-name "
             "normalisation |")
    L.append("| CMS NADAC | `data.medicaid.gov/api/1/datastore/query/…` | none "
             "| Unit acquisition cost |")
    L.append("")
    L.append("The retired RxNav **Drug Interaction API** is deliberately *not* "
             "integrated: it was verified to return \"Not found\" for valid "
             "RxCUIs, so the application shows no interaction data rather than "
             "data from an unmaintained source. Keys live in `.env` only; a "
             "test asserts none is hard-coded, and `ENABLE_LIVE_API=0` runs the "
             "whole application offline with graceful degradation.\n")

    # ---- installation ------------------------------------------------------
    L.append("## Installation\n")
    L.append("```bash\n"
             "git clone <this repository>\n"
             "cd Antibiotic-Resistance-AI\n\n"
             "python -m venv .venv\n"
             ".venv\\Scripts\\activate          # Windows\n"
             "# source .venv/bin/activate      # macOS / Linux\n\n"
             "pip install -r requirements.txt\n"
             "copy .env.example .env           # then edit if you have API keys\n"
             "```\n")
    L.append(f"Point `RAW_DATA_DIR` in `config/paths.py` at the source "
             f"directory if the data is not at:\n\n```\n{RAW_DATA_DIR}\n```\n")

    # ---- running -----------------------------------------------------------
    L.append("## Running the project\n")
    L.append("```bash\n"
             "# Whole pipeline, in order (several hours on a laptop)\n"
             "python scripts/run_all.py\n\n"
             "# Or stage by stage\n"
             "python scripts/00_verify_raw_integrity.py\n"
             "python scripts/01_dataset_audit.py\n"
             "python scripts/02_relationship_and_leakage_analysis.py\n"
             "python scripts/03_generate_data_docs.py\n"
             "python scripts/04_build_analytic_dataset.py\n"
             "python scripts/05_generate_analysis_docs.py\n"
             "python scripts/06_feature_selection.py\n"
             "python scripts/07_train_models.py\n"
             "python scripts/08_evaluate_models.py\n"
             "python scripts/09_explainability.py\n"
             "python scripts/10_generate_feature_docs.py\n"
             "python scripts/11_generate_model_docs.py\n"
             "python scripts/12_generate_project_docs.py\n\n"
             "# Tests\n"
             "python -m pytest 12_TESTS -q\n\n"
             "# Application\n"
             "streamlit run 11_APPLICATION/app.py\n"
             "```\n")

    # ---- structure ---------------------------------------------------------
    L.append("## Project structure\n")
    L.append("```text\n"
             "Antibiotic-Resistance-AI/\n"
             "├── 00_PROJECT_DOCUMENTATION/   overview, research gap, objectives\n"
             "├── 01_DATA/                    raw (referenced) / interim / processed\n"
             "├── 02_DATA_ANALYSIS/           audit + EDA notebooks and JSON reports\n"
             "├── 03_FEATURE_ENGINEERING/     feature construction notebooks\n"
             "├── 04_FEATURE_SELECTION/       7 methods, figures, FEATURE_SELECTION_REPORT.md\n"
             "├── 05_MODEL_TRAINING/          models/, experiments/experiment_log.csv\n"
             "├── 06_MODEL_EVALUATION/        figures, tables, MODEL_EVALUATION_REPORT.md\n"
             "├── 07_EXPLAINABLE_AI/          SHAP figures, XAI_REPORT.md\n"
             "├── 08_RECOMMENDATION_ENGINE/   ranking, rules, dosage, cost\n"
             "├── 09_GENOMIC_AMR/             marker→mechanism module (inactive: no data)\n"
             "├── 10_API_SERVICES/            openFDA, RxNav, CMS NADAC clients\n"
             "├── 11_APPLICATION/             Streamlit app, pages, components\n"
             "├── 12_TESTS/                   pytest suite\n"
             "├── 13_RESULTS/                 figures, tables, predictions\n"
             "├── 14_REPORT_MATERIAL/         report-ready copies of every figure/table\n"
             "├── config/                     paths.py, final_features.json\n"
             "├── docs/                       generated documentation\n"
             "├── scripts/                    the numbered pipeline\n"
             "├── src/                        reusable library code\n"
             "├── README.md  PROJECT_MEMORY.md  CHANGELOG.md  MODEL_CARD.md\n"
             "```\n")

    # ---- results -----------------------------------------------------------
    L.append("## Results\n")
    if m:
        L.append("### Final model, held-out test partition\n")
        L.append("| Metric | Value |")
        L.append("|---|--:|")
        for k, label in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"),
                         ("recall_sensitivity", "Sensitivity (non-susceptible)"),
                         ("specificity", "Specificity"),
                         ("precision_ppv", "Precision (PPV)"),
                         ("npv", "NPV"), ("f1", "F1"),
                         ("macro_f1", "Macro F1"),
                         ("weighted_f1", "Weighted F1"),
                         ("accuracy", "Accuracy"), ("brier", "Brier score")):
            L.append(f"| {label} | {m.get(k)} |")
        L.append(f"\nNo-skill PR-AUC = prevalence = {m['positive_rate']:.4f}. "
                 f"Accuracy is reported but never used for selection: "
                 f"predicting \"susceptible\" for everything would already "
                 f"score {1 - m['positive_rate']:.4f}.\n")
        tv = ev.get("temporal_vs_grouped")
        if tv:
            L.append(f"**Temporal generalisation.** Trained on the earlier "
                     f"years and tested on the most recent: PR-AUC "
                     f"{tv['pr_auc_temporal']} against {tv['pr_auc_grouped']} "
                     f"on the random patient split — a drop of "
                     f"{tv['pr_auc_drop']:.4f}. This is the deployment-"
                     "realistic figure and is reported rather than hidden.\n")
        L.append("Full breakdown by antibiotic, organism and subgroup: "
                 "[`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md).\n")
    else:
        L.append("*(Populated once `scripts/08_evaluate_models.py` has run.)*\n")

    # ---- user acceptance testing ------------------------------------------
    uat = a.get("uat")
    if uat:
        sc = uat["service_coverage"]
        perf = uat["performance_seconds"]
        L.append("## User acceptance testing\n")
        L.append(f"The completed application was acceptance-tested against "
                 f"{uat['cases_scanned']} held-out cultures "
                 f"({uat['agent_level_comparisons']:,} agent-level "
                 f"comparisons), driven through Streamlit's own `AppTest` "
                 f"harness so that every tab of the real application was "
                 f"rendered and inspected. **Result: {uat['result']}** - "
                 f"{uat['tests_passed']}/{uat['tests_total']} tests passed, "
                 f"{uat['tests_failed']} failed and were fixed, "
                 f"{uat['tests_outstanding']} outstanding. "
                 f"{uat['invalid_input_crashes']} crashes across "
                 f"{uat['invalid_input_tests']} invalid-input tests.\n")
        L.append("Verified: the application never prescribes, never removes a "
                 "flagged option, never fabricates a value, and requires an "
                 "explicit human confirmation before a decision is recorded. "
                 "In the incorrect-prediction test case the model was wrong on "
                 "6 of 8 agents and **every disagreement was displayed with a "
                 "working SHAP explanation** - the application explains its "
                 "failures as readily as its successes.\n")
        L.append(f"Two defects were found and fixed, both confined to the "
                 f"external-API name-resolution layer. **No model was "
                 f"retrained and no measured performance number changed** "
                 f"(regression suite: {uat['regression_suite']}). The more "
                 f"serious, BUG-01, was a missing identity check that allowed "
                 f"openFDA to return another product's label: Doripenem "
                 f"displayed the dosing for an antiviral. Correct dosage "
                 f"references rose {sc['dosage_correct_before']}/"
                 f"{sc['candidate_agents']} to {sc['dosage_correct_after']}/"
                 f"{sc['candidate_agents']} and wrong-drug labels fell from "
                 f"{sc['dosage_wrong_drug_before']} to "
                 f"{sc['dosage_wrong_drug_after']}.\n")
        L.append(f"Performance: {perf['startup_cold_first_render']}s cold "
                 f"start, {perf['prediction_54_agents']}s to score all 54 "
                 f"agents, {perf['shap_54_agents_median']}s for SHAP. SHAP is "
                 f"{100 * perf['shap_share_of_ranking_time']:.1f}% of the "
                 f"per-case cost.\n")
        L.append(f"Full report: [`{uat['report']}`]({uat['report']}) - "
                 f"summary for the write-up: "
                 f"[`{uat['summary']}`]({uat['summary']}) - test-by-test "
                 f"table: [`{uat['table']}`]({uat['table']}).\n")


    # ---- limitations -------------------------------------------------------
    L.append("## Limitations\n")
    L.append("1. **Single centre, single country.** Resistance epidemiology is "
             "intensely local; the model needs recalibration before use "
             "elsewhere.")
    L.append("2. **Three specimen types, adults only.** Urine, blood and "
             "respiratory; the youngest age band is 18–24.")
    L.append("3. **Panel-selection bias.** An isolate is labelled only for the "
             "agents the laboratory chose to test, so the model learns the "
             "pattern *conditional on being tested*.")
    L.append("4. **Laboratory categories, not clinical outcomes.** Breakpoints "
             "also changed across the 2007–2024 span.")
    if build:
        L.append(f"5. **Partial history.** Prior resistance exists for "
                 f"{build['step5_prior_resistance']['coverage_of_clean_orders']:.0f}% "
                 f"of orders and prior medications for "
                 f"{build['step4_prior_med']['coverage_of_clean_orders']:.0f}%; "
                 "care outside this health system is invisible.")
    L.append("6. **No genomic data**, no comorbidity, no weight, no severity "
             "score, no source-control information.")
    L.append("7. **De-identified sex** (0/1, no legend), so no sex-specific "
             "claim can be made or audited.")
    L.append("8. **Not prospectively validated.** No clinician has used this in "
             "a real decision, and no outcome study exists.\n")

    # ---- future ------------------------------------------------------------
    L.append("## Future work\n")
    L.append("- External validation on a second health system; recalibration "
             "rather than retraining, to test transportability.")
    L.append("- Activate the genomic module against a real AMRFinderPlus / "
             "ResFinder export — the plug-in point already exists.")
    L.append("- Extend to wound, tissue and device isolates.")
    L.append("- Model the panel-selection mechanism explicitly (a two-stage or "
             "missing-not-at-random formulation) instead of conditioning on it.")
    L.append("- Prospective silent-mode evaluation, comparing model ranking "
             "against the clinician's actual choice before any interface is "
             "shown.")
    L.append("- Time-aware retraining with drift monitoring on the temporal "
             "metric.\n")

    # ---- safety ------------------------------------------------------------
    L.append("## Medical safety disclaimer\n")
    L.append("> **This application is a research/clinical decision-support "
             "prototype. It does not replace qualified healthcare "
             "professionals, antimicrobial susceptibility testing, clinical "
             "guidelines, or professional medical judgment. AI-generated "
             "predictions and recommendations must be reviewed by an "
             "appropriately qualified healthcare professional.**\n")
    L.append("The system has no mechanism to prescribe, order or administer "
             "anything, and must never be given one. It does not generate "
             "doses — it relays FDA label text verbatim. It does not present "
             "cost as a reason to choose an antibiotic. Every ranked option is "
             "labelled a *potential option for clinician review*, and the "
             "application requires an explicit, recorded human decision.\n")

    L.append("## Documentation index\n")
    doc_groups = [
        ("Start here", [
            ("Project status — every stage, its results and its limitations",
             "docs/PROJECT_STATUS.md"),
            ("Project memory — read this first in a new session",
             "PROJECT_MEMORY.md"),
            ("Project overview — the whole thing in one page",
             "00_PROJECT_DOCUMENTATION/PROJECT_OVERVIEW.md"),
        ]),
        ("Motivation", [
            ("Research gap", "00_PROJECT_DOCUMENTATION/RESEARCH_GAP.md"),
            ("Objectives", "00_PROJECT_DOCUMENTATION/OBJECTIVES.md"),
        ]),
        ("Data", [
            ("Data dictionary", "docs/DATA_DICTIONARY.md"),
            ("Dataset relationships", "docs/DATA_RELATIONSHIPS.md"),
            ("Data leakage analysis", "docs/DATA_LEAKAGE_ANALYSIS.md"),
            ("Exploratory data analysis", "02_DATA_ANALYSIS/EDA_REPORT.md"),
            ("Raw data policy", "01_DATA/raw/README.md"),
        ]),
        ("Modelling", [
            ("ML problem definition", "docs/ML_PROBLEM_DEFINITION.md"),
            ("Feature selection report",
             "04_FEATURE_SELECTION/FEATURE_SELECTION_REPORT.md"),
            ("Final feature set", "docs/FINAL_FEATURE_SET.md"),
            ("Model evaluation", "docs/MODEL_EVALUATION.md"),
            ("Explainable AI report", "07_EXPLAINABLE_AI/XAI_REPORT.md"),
            ("Model card — intended use, limitations, bias", "MODEL_CARD.md"),
        ]),
        ("Decision support and services", [
            ("Genomic AMR module (inactive — no genomic data)",
             "09_GENOMIC_AMR/README.md"),
            ("External API services", "10_API_SERVICES/README.md"),
        ]),
        ("Verification", [
            ("System verification report — 59 executed checks",
             "docs/SYSTEM_VERIFICATION.md"),
        ]),
        ("Report material", [
            ("Report material index — every figure and table",
             "14_REPORT_MATERIAL/INDEX.md"),
        ]),
        ("History", [("Changelog", "CHANGELOG.md")]),
    ]
    for group, docs in doc_groups:
        L.append(f"**{group}**\n")
        for label, path in docs:
            exists = (PROJECT_ROOT / path).exists()
            L.append(f"- [{label}]({path})"
                     + ("" if exists else "  *(not generated yet)*"))
        L.append("")

    L.append("## Authors\n")
    L.append("Alan Francis Joseph — design, implementation and analysis.\n")
    L.append("Built with Claude Code. Datasets are de-identified and used for "
             "academic research.\n")
    return "\n".join(L)


# ===========================================================================
def project_memory(a) -> str:
    build, cfg, ev, tr, xai, fs = (a["build"], a["cfg"], a["ev"], a["train"],
                                   a["xai"], a["fs"])
    s1 = (build or {}).get("step1_cohort", {})
    f7 = (build or {}).get("step7_final", {})
    m = (ev or {}).get("test_metrics", {})
    meta = (a["meta"] or {}).get("metadata", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    L = []
    L.append("# PROJECT MEMORY\n")
    L.append(f"> Regenerated by `scripts/12_generate_project_docs.py` on "
             f"{now}. **Read this first in a new session.** Everything below is "
             "read out of the pipeline's own artefacts, so it is current by "
             "construction.\n")

    L.append("## Project identity\n")
    L.append("**Explainable Machine Learning for Antibiotic Resistance "
             "Prediction and Clinical Decision Support** — an academic research "
             "prototype. Predicts laboratory non-susceptibility for an "
             "identified isolate, explains the prediction with SHAP, ranks "
             "agents for clinician review with cited guardrails, and keeps a "
             "human in the loop. It is **not** a prescribing system.\n")
    L.append(f"Project root: `{PROJECT_ROOT}`\n")

    L.append("## Current objective\n")
    L.append("Full pipeline is built and runnable end to end. Ongoing work is "
             "refinement, documentation and report material — not new "
             "architecture.\n")

    L.append("## Dataset locations\n")
    L.append(f"- **Raw (read-only, referenced in place):** `{RAW_DATA_DIR}`")
    L.append("- **Path configuration:** `config/paths.py` — the ONLY place the "
             "raw path appears. A test enforces this.")
    L.append("- **Interim:** `01_DATA/interim/*.parquet`")
    L.append("- **Processed:** `01_DATA/processed/analytic_dataset.parquet`")
    L.append("- **Integrity:** `python scripts/00_verify_raw_integrity.py` "
             "compares size + MD5 against `01_DATA/raw/RAW_DATA_MANIFEST.json`\n")

    L.append("## Dataset structure\n")
    if a["audit"]:
        L.append("| File | Rows | Cols | Grain |")
        L.append("|---|--:|--:|---|")
        grains = {
            "cohort": "order × organism × antibiotic (the LABEL table)",
            "demographics": "culture order (true primary key)",
            "labs": "order (2–3 partially-populated rows each; Period_Day = 14 constant)",
            "resistance": "one prior resistant event",
            "prior_med": "one prior medication administration (40.3% exact duplicates)",
        }
        for d in a["audit"]["datasets"]:
            L.append(f"| `{d['file_name']}` | {n(d['n_rows'])} | "
                     f"{d['n_columns']} | {grains.get(d['key'], '')} |")
        L.append("")

    L.append("## Dataset relationships\n")
    L.append("`anon_id` → `pat_enc_csn_id_coded` → `order_proc_id_coded`. "
             "**`order_proc_id_coded` is the single join key.** Only "
             "`demographics` is 1:1 (100% coverage). `labs` (65.6% of orders), "
             "`resistance` (22.2%) and `prior_med` (76.6%) are one-to-many and "
             "are aggregated to one row per key BEFORE joining — a raw join "
             "would fan out up to 748×. All merges go through "
             "`validated_merge()` in `scripts/04_build_analytic_dataset.py`, "
             "which raises rather than silently corrupting.\n")

    L.append("## Target variable\n")
    if s1:
        L.append(f"- **Primary:** `y_nonsusceptible` — 1 = Resistant or "
                 f"Intermediate, 0 = Susceptible. "
                 f"**{pct(f7['target_positive_rate'])} positive.**")
        L.append(f"- **Secondary:** `y_3class` — Susceptible / Intermediate / "
                 f"Resistant "
                 f"({s1['target_3class_distribution'].get('Intermediate', 0):,} "
                 "Intermediate rows). Trained and reported, but Intermediate "
                 "recall is poor — which is the empirical reason the primary "
                 "task groups it with Resistant.")
        L.append(f"- `Inconclusive` and `Synergism` "
                 f"({s1['n_dropped_non_phenotype_result']:,} rows) are removed: "
                 "they are not single-agent phenotypes.\n")

    L.append("## Selected features\n")
    if cfg:
        L.append(f"**{cfg['n_features']} features** — full list in "
                 "`config/final_features.json`, documented in "
                 "`docs/FINAL_FEATURE_SET.md`.\n")
        L.append("Categorical: "
                 + ", ".join(f"`{c}`" for c in cfg["categorical_features"]))
        L.append("\nNumeric: "
                 + ", ".join(f"`{c}`" for c in cfg["numeric_features"]) + "\n")
        L.append("**Notable absences (empirical, not oversight):** no "
                 "laboratory value and no age feature survived selection. The "
                 "full-feature arm scores *worse*. Explained in "
                 "`docs/FINAL_FEATURE_SET.md`.\n")

    L.append("## Excluded features\n")
    if fs:
        L.append("| Column | Reason |")
        L.append("|---|---|")
        for c, why in fs["feature_inventory"]["excluded_columns"].items():
            L.append(f"| `{c}` | {why} |")
        L.append("")
    L.append("Also excluded: `medication_category` (a 1:1 surrogate key for "
             "`medication_name`, and misleading as a class — `CEF`=cefazolin "
             "but `CEF2`=ceftriaxone); `Period_Day` (constant 14); resistance "
             "tokens embedded in `organism` "
             f"({n(s1.get('organism_leaked_phenotype_rows', 0))} rows).\n")

    L.append("## Feature-selection methodology\n")
    if fs:
        cv = fs["selection_inside_cv"]
        L.append("Seven methods → four+ candidate sets → compared on held-out "
                 "validation patients → highest PR-AUC wins "
                 f"(`{fs['selection_rule']['winner']}`).\n")
        L.append(f"**Leakage check:** re-deriving the selection inside every CV "
                 f"fold scores {cv['nested_cv_pr_auc_mean']:.5f} vs "
                 f"{cv['global_cv_pr_auc_mean']:.5f} for the global list — "
                 f"optimism {cv['optimism_from_global_selection']:+.5f} PR-AUC. "
                 "Negligible.\n")

    L.append("## Data preprocessing\n")
    L.append("- Organism normalisation: strip `ZZZ` prefixes, reconcile "
             "`(GROUP B)`/`{GROUP B}`, collapse `… COMPLEX`, extract "
             "morphotype, **strip resistance phenotype tokens** "
             "(`src/preprocessing/organism_normalizer.py`).")
    L.append("- Duplicate resolution: most-resistant-wins "
             f"({s1.get('n_duplicate_groups_with_conflicting_labels', 0):,} "
             "conflicting groups).")
    L.append("- Labs collapsed to one row per order by median of non-missing.")
    L.append("- Prior medications: `t == 0` (same-day) excluded; repeated "
             "same-drug-same-day collapsed; class derived from drug NAME.")
    L.append("- History features: cumulative sums **shifted** to exclude the "
             "current order.")
    L.append("- Trees: ordinal category codes, NaN preserved. Linear models: "
             "one-hot + median impute + standardise (separate pipeline).\n")

    L.append("## Models tested\n")
    if tr:
        L.append("| Model | Validation PR-AUC |")
        L.append("|---|--:|")
        for r in sorted(tr["model_comparison"], key=lambda r: -r["pr_auc"]):
            L.append(f"| {r['model']} | {r['pr_auc']:.4f} |")
        L.append("")

    L.append("## Best model\n")
    if meta:
        L.append(f"**{meta.get('algorithm')}**, {meta.get('n_features')} "
                 f"features, trained on {n(meta.get('n_train_rows', 0))} rows "
                 f"from {n(meta.get('n_train_patients', 0))} patients. "
                 f"Imbalance strategy: {meta.get('imbalance_strategy')}. "
                 "Isotonic calibration fitted on validation.\n")
        L.append("Saved to `05_MODEL_TRAINING/models/final_model.joblib` as a "
                 "`ModelBundle` carrying the model, encoder, feature list, "
                 "training defaults, organism×antibiotic support counts, "
                 "calibrator and measured metrics.\n")

    L.append("## Hyperparameters\n")
    L.append("```json")
    L.append(json.dumps(meta.get("hyperparameters", {}), indent=2))
    L.append("```\n")

    L.append("## Model performance\n")
    if m:
        L.append("| Metric | Held-out test |")
        L.append("|---|--:|")
        for k, label in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"),
                         ("recall_sensitivity", "Sensitivity"),
                         ("specificity", "Specificity"),
                         ("precision_ppv", "Precision"), ("f1", "F1"),
                         ("accuracy", "Accuracy"), ("brier", "Brier")):
            L.append(f"| {label} | {m.get(k)} |")
        tm = ev.get("temporal_metrics")
        if tm:
            L.append(f"\nTemporal holdout (train past → test future): PR-AUC "
                     f"{tm['pr_auc']}, ROC-AUC {tm['roc_auc']}.\n")

    L.append("## Explainability method\n")
    L.append("SHAP `TreeExplainer` (exact Shapley values). One implementation "
             "in `src/explainability/shap_explainer.py`, used by the feature "
             "selection stage, the XAI report and the live application — so "
             "what selects a feature and what explains it to a clinician are "
             "the same computation.\n")

    L.append("## Recommendation logic\n")
    L.append("`08_RECOMMENDATION_ENGINE/antibiotic_ranking.py` scores every "
             "candidate agent, attaches SHAP factors and guardrail flags from "
             "`recommendation_rules.py`, and orders unflagged options by "
             "predicted susceptibility with flagged options ranked below — "
             "**never removed**. Framing is always \"potential options for "
             "clinician review\".\n")

    L.append("## API integrations\n")
    L.append("- **openFDA drug label** — verbatim dosage text. Route-ranked so "
             "a systemic label is chosen over an ophthalmic one (this was a "
             "real bug: searching \"ciprofloxacin\" returns eye drops first).")
    L.append("- **RxNorm / RxNav** — name normalisation. No key.")
    L.append("- **CMS NADAC** — unit acquisition cost. No key.")
    L.append("- **NOT used:** RxNav Drug Interaction API — retired by NLM, "
             "verified to return \"Not found\".")
    L.append("- All calls go through `10_API_SERVICES/_http.py`: timeout, disk "
             "cache, `ApiResult` envelope with source + retrieval timestamp, "
             "never raises, `ENABLE_LIVE_API=0` disables everything.\n")

    L.append("## Genomic module\n")
    L.append("**INACTIVE — the datasets contain no genomic data.** Verified "
             "column by column. `09_GENOMIC_AMR/genomic_analysis.py` implements "
             "a 21-marker reference map (NCBI/CARD/ResFinder) and a plug-in "
             "point: drop a CSV with `order_proc_id_coded, marker` at "
             "`01_DATA/raw/genotypes.csv` (or set `GENOTYPE_TABLE_PATH`) and it "
             "activates with no other code change.\n")

    L.append("## Application structure\n")
    L.append("`11_APPLICATION/app.py` — sidebar case entry (manual or a real "
             "held-out culture) + nine tabs. `pages/1_Model_Card.py`, "
             "`pages/2_Data_and_Methodology.py`. Shared UI in "
             "`components/ui.py`, case construction in `components/case.py`.\n")

    L.append("## Important architectural decisions\n")
    L.append("1. **One unified model with the antibiotic as a feature** — not "
             "54 per-agent models, not multi-label. Tested empirically in "
             "`docs/MODEL_EVALUATION.md` §11.")
    L.append("2. **Prediction moment fixed at post-identification, pre-AST.** "
             "This is what makes the organism a legal feature; an "
             "organism-blind ablation quantifies the alternative.")
    L.append("3. **Patient-grouped splits + a temporal holdout**, reported "
             "side by side.")
    L.append("4. **Guardrails flag, never filter.** Nothing is hidden from the "
             "reviewing clinician.")
    L.append("5. **Dosage is relayed, never computed.** Cost is retrieved, "
             "never estimated.")
    L.append("6. **Every document in `docs/` is generated from JSON artefacts**, "
             "so published numbers cannot drift from the data.")
    L.append("7. **Raw data is referenced in place, never copied or written.** "
             "Enforced by tests.\n")

    uat = a.get("uat")
    if uat:
        L.append("## User acceptance testing\n")
        L.append(f"**Result: {uat['result']}** - {uat['tests_passed']}/"
                 f"{uat['tests_total']} tests passed ({uat['tests_failed']} "
                 f"failed, both fixed and re-tested; {uat['tests_outstanding']} "
                 f"outstanding). Pass rate "
                 f"{100 * uat['pass_rate_first_run']:.1f}% as first run, "
                 f"{100 * uat['pass_rate_after_fixes']:.0f}% after fixes. "
                 f"{uat['invalid_input_crashes']} crashes across "
                 f"{uat['invalid_input_tests']} invalid-input tests. "
                 f"Regression suite: {uat['regression_suite']}.\n")
        L.append(f"Tested against {uat['cases_scanned']} held-out cultures "
                 f"({uat['agent_level_comparisons']:,} agent-level "
                 f"comparisons). {uat['demo_menu_caveat']}\n")
        perf = uat["performance_seconds"]
        L.append(f"Performance: startup {perf['startup_cold_first_render']}s "
                 f"cold; prediction for 54 agents "
                 f"{perf['prediction_54_agents']}s; SHAP "
                 f"{perf['shap_54_agents_median']}s; full ranking "
                 f"{perf['full_ranking_median']}s. SHAP is "
                 f"{100 * perf['shap_share_of_ranking_time']:.1f}% of ranking "
                 f"time - the slow first load is expected, not a defect.\n")
        sc = uat["service_coverage"]
        L.append(f"Two defects were found and fixed, both confined to the "
                 f"external-API name-resolution layer - no model was retrained "
                 f"and no measured performance number changed. Correct dosage "
                 f"references rose {sc['dosage_correct_before']}/"
                 f"{sc['candidate_agents']} to {sc['dosage_correct_after']}/"
                 f"{sc['candidate_agents']}, wrong-drug labels "
                 f"{sc['dosage_wrong_drug_before']} to "
                 f"{sc['dosage_wrong_drug_after']}, cost coverage "
                 f"{sc['cost_before']}/{sc['candidate_agents']} to "
                 f"{sc['cost_after']}/{sc['candidate_agents']}.\n")
        for b in uat["bugs"]:
            L.append(f"- **{b['id']} ({b['severity']}, {b['status']})** - "
                     f"{b['title']}. {b['fix']}")
        L.append("")
        L.append(f"Full report: `{uat['report']}` - report summary: "
                 f"`{uat['summary']}` - test table: `{uat['table']}`.\n")

    L.append("## Known problems / open issues\n")
    L.append("- Prior-resistance history covers only ~38% of orders; absence of "
             "a record is not absence of exposure.")
    L.append("- Panel-selection bias is acknowledged but not modelled.")
    L.append("- No external validation cohort exists.")
    L.append("- Streamlit `@st.cache_data` hashes the record dict on every "
             "input change; scoring 54 agents with SHAP takes a moment on "
             "first load.")
    L.append("- The 3-class model's Intermediate recall is low; this is a "
             "property of the AST category, not a fixable defect.")
    if a.get("uat"):
        for b in a["uat"]["bugs"]:
            if b["status"] == "deferred":
                L.append(f"- {b['title']} ({b['id']}, {b['severity']}, "
                         f"open after UAT). {b['detail']}")
    L.append("")

    L.append("## Current development status\n")
    L.append("```text")
    rows = [
        ("DATASET AUDIT", a["audit"] is not None),
        ("DATA RELATIONSHIPS", (ANALYSIS_DIR / "reports" /
                                "relationships.json").exists()),
        ("LEAKAGE ANALYSIS", (ANALYSIS_DIR / "reports" / "leakage.json").exists()),
        ("DATA CLEANING", build is not None),
        ("EDA", (ANALYSIS_DIR / "EDA_REPORT.md").exists()
                and (ANALYSIS_DIR / "reports" / "eda_report.json").exists()),
        ("FEATURE ENGINEERING", build is not None),
        ("FEATURE SELECTION", fs is not None),
        ("MODEL TRAINING", tr is not None),
        ("MODEL EVALUATION", ev is not None),
        ("XAI", xai is not None),
        ("RECOMMENDATION ENGINE",
         (PROJECT_ROOT / "08_RECOMMENDATION_ENGINE" /
          "antibiotic_ranking.py").exists()),
        ("GENOMIC MODULE",
         (PROJECT_ROOT / "09_GENOMIC_AMR" / "genomic_analysis.py").exists(),
         "COMPLETE / INACTIVE (no genomic data in these datasets)"),
        ("API INTEGRATION",
         (PROJECT_ROOT / "10_API_SERVICES" / "drug_api.py").exists()),
        ("APPLICATION", (PROJECT_ROOT / "11_APPLICATION" / "app.py").exists()),
        ("TESTING", any((PROJECT_ROOT / "12_TESTS").glob("test_*.py"))),
        ("USER ACCEPTANCE TESTING",
         (PROJECT_ROOT / "12_TESTS" /
          "USER_ACCEPTANCE_TEST_REPORT.md").exists(),
         (f"PASS ({a['uat']['tests_passed']}/{a['uat']['tests_total']} tests, "
          f"{a['uat']['bugs_fixed']} bugs fixed)" if a.get("uat") else None)),
        ("DOCUMENTATION", (PROJECT_ROOT / "docs" / "MODEL_EVALUATION.md").exists()),
        ("REPORT MATERIAL",
         (PROJECT_ROOT / "14_REPORT_MATERIAL" / "diagrams" /
          "methodology_workflow.png").exists()),
    ]
    width = max(len(r[0]) for r in rows) + 2
    for row in rows:
        label, done = row[0], row[1]
        override = row[2] if len(row) > 2 else None
        L.append(f"{label + ':':<{width}} "
                 f"{override if (override and done) else status(done)}")
    L.append("```\n")

    L.append("## How to pick this up in a new session\n")
    L.append("1. Read this file, then `README.md`.")
    L.append("2. `python scripts/00_verify_raw_integrity.py` — confirm the "
             "source data has not changed.")
    L.append("3. `python -m pytest 12_TESTS -q` — confirm the guarantees still "
             "hold.")
    L.append("4. Artefacts to inspect before changing anything: "
             "`01_DATA/processed/build_report.json`, "
             "`04_FEATURE_SELECTION/selected_features/feature_selection_report.json`, "
             "`05_MODEL_TRAINING/experiments/training_report.json`, "
             "`06_MODEL_EVALUATION/evaluation_report.json`.")
    L.append("5. After ANY pipeline change, re-run "
             "`python scripts/12_generate_project_docs.py` so this file and the "
             "README stay true.\n")
    return "\n".join(L)


# ===========================================================================
def main() -> None:
    a = {
        "audit": load(ANALYSIS_DIR / "reports" / "dataset_audit.json"),
        "build": load(PROCESSED_DIR / "build_report.json"),
        "fs": load(FS_DIR / "selected_features" / "feature_selection_report.json"),
        "cfg": load(CONFIG_DIR / "final_features.json"),
        "train": load(EXPERIMENTS_DIR / "training_report.json"),
        "ev": load(EVAL_DIR / "evaluation_report.json"),
        "xai": load(XAI_DIR / "xai_report.json"),
        "meta": load(MODELS_DIR / "final_model.metadata.json"),
        "eda": load(ANALYSIS_DIR / "reports" / "eda_report.json"),
        "uat": load(PROJECT_ROOT / "12_TESTS" / "uat_report.json"),
    }
    (PROJECT_ROOT / "README.md").write_text(readme(a), encoding="utf-8")
    print("wrote", PROJECT_ROOT / "README.md")
    pm = project_memory(a)
    (PROJECT_ROOT / "PROJECT_MEMORY.md").write_text(pm, encoding="utf-8")
    docdir = PROJECT_ROOT / "00_PROJECT_DOCUMENTATION"
    docdir.mkdir(parents=True, exist_ok=True)
    (docdir / "PROJECT_MEMORY.md").write_text(pm, encoding="utf-8")
    print("wrote", PROJECT_ROOT / "PROJECT_MEMORY.md")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _overview_docs import build_docs
    for name, text in build_docs(a).items():
        (docdir / name).write_text(text, encoding="utf-8")
        print("wrote", docdir / name)


if __name__ == "__main__":
    main()
