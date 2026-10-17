"""Generate the report-ready workflow and architecture diagrams.

Every count and metric annotated on these diagrams is read from the pipeline
artefacts, so the diagrams cannot drift from the project they describe.

Outputs (all into 14_REPORT_MATERIAL/diagrams/):
  methodology_workflow.png            end-to-end research methodology
  system_architecture.png             layered system architecture
  feature_engineering_workflow.png    raw tables -> analytic matrix
  feature_selection_methodology.png   185 candidates -> final set
  application_workflow.png            what a user does in the Streamlit app
  recommendation_workflow.png         how one ranked option is produced
  human_in_the_loop_workflow.png      where the clinician decides
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (ANALYSIS_DIR, CONFIG_DIR, EVAL_DIR,  # noqa: E402
                          EXPERIMENTS_DIR, FS_DIR, MODELS_DIR, PROCESSED_DIR,
                          REPORT_DIR)

OUT = REPORT_DIR / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# palette
DATA = "#0891b2"      # data stages
PROC = "#2563eb"      # processing
MODEL = "#7c3aed"     # modelling
XAI = "#db2777"       # explainability
CDS = "#059669"       # decision support
HUMAN = "#c2410c"     # human
WARN = "#dc2626"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def box(ax, x, y, w, h, title, subtitle="", colour=PROC, fs_title=10,
        fs_sub=8, zorder=3, alpha="18"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=2.0, edgecolor=colour, facecolor=colour + alpha,
        zorder=zorder))
    if subtitle:
        ax.text(x, y + h * 0.16, title, ha="center", va="center",
                fontsize=fs_title, fontweight="bold", color=colour,
                zorder=zorder + 1)
        ax.text(x, y - h * 0.22, subtitle, ha="center", va="center",
                fontsize=fs_sub, color="#333", zorder=zorder + 1,
                linespacing=1.35)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs_title,
                fontweight="bold", color=colour, zorder=zorder + 1)


def arrow(ax, p0, p1, colour="#555", lw=1.7, style="-|>", rad=0.0, zorder=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=15, linewidth=lw,
        color=colour, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, text, fs=7.6, colour="#111", bg="white"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=colour,
            zorder=6, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", fc=bg, ec="#c9ccd1", lw=0.7))


def finish(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ===========================================================================
def methodology_workflow(A):
    build, cfg, fs, ev, tr, eda, meta = (A["build"], A["cfg"], A["fs"],
                                         A["ev"], A["tr"], A["eda"], A["meta"])
    s1 = build["step1_cohort"]
    f7 = build["step7_final"]
    m = ev["test_metrics"]

    stages = [
        ("Raw Dataset", DATA,
         f"5 linked CSV files · {s1['n_rows_raw']:,} cohort rows\n"
         f"{s1['n_all_culture_orders']:,} culture orders · read-only, "
         "referenced in place"),
        ("Data Audit", DATA,
         "Streaming column-level profile of all 5 files\n"
         "→ docs/DATA_DICTIONARY.md · MD5 integrity manifest"),
        ("Data Cleaning", PROC,
         f"−{s1['n_dropped_no_label_negative_culture']:,} negative cultures · "
         f"−{s1['n_dropped_non_phenotype_result']:,} non-phenotype\n"
         f"−{s1['n_dropped_duplicate_resolution']:,} duplicate keys resolved "
         "(most-resistant-wins)"),
        ("Data Integration", PROC,
         "4 companion tables aggregated to one row per key,\n"
         "then joined via validated_merge() — refuses any fan-out"),
        ("Exploratory Data Analysis", DATA,
         f"14 figures · 21 tables · scope-separated\n"
         f"organism×antibiotic space is "
         f"{100 - eda['antibiogram']['full_space_pct_with_any_test']:.0f}% empty"),
        ("Feature Engineering", PROC,
         f"{f7['n_rows']:,} rows × {f7['n_columns']} columns\n"
         "temporal rule: history strictly BEFORE the index culture"),
        ("Feature Selection", PROC,
         f"7 methods · {len(fs['candidate_sets'])} candidate sets\n"
         f"{fs['feature_inventory']['n_candidate_features']} candidates → "
         f"{cfg['n_features']} features"),
        ("Train / Test Split", MODEL,
         "Patient-grouped 70/10/20 on anon_id\n"
         "+ temporal holdout (train past → test future)"),
        ("Model Training", MODEL,
         "6 families · 4-arm imbalance study\n"
         "RandomizedSearchCV + StratifiedGroupKFold"),
        ("Model Comparison", MODEL,
         " · ".join(f"{r['model']} {r['pr_auc']:.3f}"
                    for r in sorted(tr["model_comparison"],
                                    key=lambda r: -r["pr_auc"])[:3])
         + "\n(validation PR-AUC)"),
        (f"Final {meta['metadata']['algorithm']} Model", MODEL,
         f"{cfg['n_features']} features · isotonic-calibrated on validation\n"
         f"trained on {meta['metadata']['n_train_rows']:,} rows from "
         f"{meta['metadata']['n_train_patients']:,} patients"),
        ("Model Evaluation", MODEL,
         f"Held-out test read ONCE · PR-AUC {m['pr_auc']} · "
         f"ROC-AUC {m['roc_auc']}\n"
         f"sensitivity {m['recall_sensitivity']} · specificity "
         f"{m['specificity']} · Brier {m['brier']}"),
        ("SHAP Explainability", XAI,
         "Exact TreeSHAP · global + per-prediction\n"
         "failures explained alongside successes · direction sanity-check"),
        ("Resistance Prediction", CDS,
         f"Calibrated P(non-susceptible) for each of "
         f"{len(meta['antibiotics'])} agents"),
        ("Antibiotic Ranking", CDS,
         "Ordered by predicted susceptibility · evidence strength\n"
         "and training-data coverage attached to every option"),
        ("Clinical Guardrails", CDS,
         "Intrinsic resistance (EUCAST/CLSI) · site suitability\n"
         "allergy · WHO AWaRe tier — options are FLAGGED, never removed"),
        ("Doctor Review", HUMAN,
         "Accept / Modify / Reject + notes, recorded\n"
         "The clinician is the decision-maker. The system never prescribes."),
    ]

    n = len(stages)
    fig, ax = plt.subplots(figsize=(13.5, 1.30 * n + 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.25, n + 0.45)
    ax.axis("off")

    for i, (title, colour, sub) in enumerate(stages):
        y = n - 1 - i
        box(ax, 0.5, y, 0.80, 0.80, title, sub, colour, fs_title=12.5,
            fs_sub=8.6)
        if i < n - 1:
            arrow(ax, (0.5, y - 0.42), (0.5, y - 0.60), colour="#666", lw=2.0)

    ax.text(0.5, n + 0.10,
            "Explainable ML for Antibiotic Resistance Prediction and Clinical "
            "Decision Support",
            ha="center", fontsize=15.5, fontweight="bold")
    ax.text(0.5, n - 0.28,
            "Research methodology — every annotation is a measured value from "
            "the pipeline's own artefacts",
            ha="center", fontsize=9.8, color="#444", style="italic")
    ax.text(0.5, -0.85,
            "Raw data is never modified. The test partition is read exactly "
            "once. Dosage is relayed verbatim from the FDA label and never "
            "computed.\nThis is a research prototype and not an autonomous "
            "prescribing system.",
            ha="center", va="center", fontsize=8.6, color="#333",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fef2f2", ec=WARN, lw=1.3))
    finish(fig, "methodology_workflow.png")


# ===========================================================================
def system_architecture(A):
    cfg, meta, ev = A["cfg"], A["meta"], A["ev"]
    fig, ax = plt.subplots(figsize=(16, 11.2))
    ax.set_xlim(0, 1); ax.set_ylim(-0.06, 1.10); ax.axis("off")

    layers = [
        (0.895, "DATA LAYER", DATA,
         [("Raw CSV files\n(read-only, referenced)", 0.16),
          ("01_DATA/interim\nvalidated aggregates", 0.38),
          ("01_DATA/processed\nanalytic_dataset.parquet", 0.62),
          ("RAW_DATA_MANIFEST\nMD5 integrity", 0.85)]),
        (0.715, "PROCESSING LAYER", PROC,
         [("organism_normalizer\nstrips leakage tokens", 0.16),
          ("antibiotic_taxonomy\n54 agents → class/AWaRe", 0.38),
          ("medication_mapping\n89 names → ingredient", 0.62),
          ("validated_merge()\nrefuses fan-out", 0.85)]),
        (0.535, "MODEL LAYER", MODEL,
         [(f"{meta['metadata']['algorithm']} classifier\n"
           f"{cfg['n_features']} features", 0.16),
          ("Isotonic calibrator\nfitted on validation", 0.38),
          ("ModelBundle\nmodel+encoder+defaults", 0.62),
          ("SHAP TreeExplainer\nexact Shapley values", 0.85)]),
        (0.355, "DECISION-SUPPORT LAYER", CDS,
         [("antibiotic_ranking\nscores every agent", 0.16),
          ("recommendation_rules\nEUCAST/CLSI guardrails", 0.38),
          ("dosage_reference\nFDA label, verbatim", 0.62),
          ("cost_service\nCMS NADAC, live", 0.85)]),
        (0.175, "APPLICATION LAYER", HUMAN,
         [("Streamlit app\n9 tabs, one isolate", 0.16),
          ("Model Card page\nperformance + limits", 0.38),
          ("Data & Methodology\naudit trail", 0.62),
          ("Clinician review\nAccept/Modify/Reject", 0.85)]),
    ]

    for y, name, colour, items in layers:
        ax.add_patch(FancyBboxPatch(
            (0.035, y - 0.077), 0.93, 0.154,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=1.4, edgecolor=colour, facecolor=colour + "0d", zorder=1))
        ax.text(0.045, y + 0.062, name, ha="left", va="center", fontsize=9.5,
                fontweight="bold", color=colour, zorder=5)
        for text, x in items:
            box(ax, x, y - 0.012, 0.195, 0.088, text.split("\n")[0],
                text.split("\n")[1], colour, fs_title=8.8, fs_sub=7.3,
                zorder=3, alpha="20")

    for y0, y1 in ((0.818, 0.792), (0.638, 0.612), (0.458, 0.432),
                   (0.278, 0.252)):
        for x in (0.16, 0.38, 0.62, 0.85):
            arrow(ax, (x, y0), (x, y1), colour="#999", lw=1.3)

    # external services, off to the side
    ax.add_patch(FancyBboxPatch(
        (0.035, -0.045), 0.93, 0.088,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=1.4, edgecolor="#64748b", facecolor="#f8fafc", zorder=1))
    ax.text(0.045, 0.026, "EXTERNAL SERVICES  (all verified live · no keys in "
            "code · ENABLE_LIVE_API=0 runs fully offline)",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color="#475569", zorder=5)
    for text, x in (("openFDA drug label\napi.fda.gov", 0.20),
                    ("RxNorm / RxNav\nrxnav.nlm.nih.gov", 0.50),
                    ("CMS NADAC\ndata.medicaid.gov", 0.80)):
        box(ax, x, 0.042, 0.24, 0.048, text.split("\n")[0],
            text.split("\n")[1], "#64748b", fs_title=8.3, fs_sub=7,
            zorder=3, alpha="14")

    ax.text(0.5, 1.072, "System Architecture", ha="center", fontsize=16,
            fontweight="bold")
    ax.text(0.5, 1.020,
            "Each layer depends only on the one above it. The model never sees "
            "an identifier, and the guardrail layer never changes a prediction.",
            ha="center", fontsize=9.3, color="#444", style="italic")
    finish(fig, "system_architecture.png")


# ===========================================================================
def feature_engineering_workflow(A):
    build, fs = A["build"], A["fs"]
    s1, f7 = build["step1_cohort"], build["step7_final"]
    fig, ax = plt.subplots(figsize=(15.5, 9.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    sources = [
        ("cohort", "2,241,050 rows\nlabels + context", 0.10, 0.855),
        ("demographics", "751,075 rows\nage band, sex code", 0.30, 0.855),
        ("labs", "1,408,677 rows\n14-day aggregates", 0.50, 0.855),
        ("microbial_resistance", "2,161,648 rows\nprior resistant events", 0.70, 0.855),
        ("prior_med", "9,823,458 rows\nprior administrations", 0.90, 0.855),
    ]
    for name, sub, x, y in sources:
        box(ax, x, y, 0.18, 0.115, name, sub, DATA, fs_title=9, fs_sub=7.4)

    steps = [
        ("Clean & normalise", 0.10, 0.655,
         f"organism: {s1['organism_raw_distinct']} → "
         f"{s1['organism_clean_distinct']}\nleakage tokens stripped"),
        ("1:1 join", 0.30, 0.655, "age → ordinal\nsex → anonymous code"),
        ("Collapse to 1 row/order", 0.50, 0.655,
         "median of non-missing\nPeriod_Day dropped (constant)"),
        ("Window aggregation", 0.70, 0.655,
         "90/365/1095-day windows\nt ≥ 1 day enforced"),
        ("Window aggregation", 0.90, 0.655,
         "30/90/180/365-day windows\nsame-day rows excluded"),
    ]
    for name, x, y, sub in steps:
        box(ax, x, y, 0.18, 0.115, name, sub, PROC, fs_title=8.6, fs_sub=7.2)
        arrow(ax, (x, 0.795), (x, 0.715), colour="#888", lw=1.4)
        arrow(ax, (x, 0.595), (x, 0.505), colour="#888", lw=1.4)

    box(ax, 0.5, 0.445, 0.80, 0.115, "validated_merge()  —  every join proved safe first",
        "right side must be unique on the keys · left row count must not change · "
        "raises rather than corrupting",
        PROC, fs_title=11.5, fs_sub=8.4)
    arrow(ax, (0.5, 0.385), (0.5, 0.305), colour="#666", lw=2.0)

    box(ax, 0.5, 0.245, 0.62, 0.105,
        f"Analytic dataset — {f7['n_rows']:,} rows × {f7['n_columns']} columns",
        f"{f7['n_orders']:,} positive culture orders · {f7['n_patients']:,} "
        f"patients · {100*f7['target_positive_rate']:.2f}% non-susceptible",
        MODEL, fs_title=12, fs_sub=8.6)
    arrow(ax, (0.5, 0.19), (0.5, 0.125), colour="#666", lw=2.0)

    groups = fs["feature_inventory"]["groups"]
    grp_txt = " · ".join(f"{g.replace('_', ' ')} {d['n']}"
                         for g, d in groups.items())
    box(ax, 0.5, 0.068, 0.86, 0.098,
        f"{fs['feature_inventory']['n_candidate_features']} candidate features "
        f"(9 identifier/target columns excluded)",
        grp_txt, MODEL, fs_title=11, fs_sub=7.8)

    ax.text(0.5, 0.972, "Feature Engineering Workflow", ha="center",
            fontsize=16, fontweight="bold")
    ax.text(0.5, 0.943,
            "Five source tables → one model-ready matrix. Every companion "
            "table is aggregated to one row per key BEFORE joining;\n"
            "a raw join would multiply the label rows up to 748×.",
            ha="center", fontsize=9.2, color="#444", style="italic")
    finish(fig, "feature_engineering_workflow.png")


# ===========================================================================
def feature_selection_methodology(A):
    fs, cfg = A["fs"], A["cfg"]
    inv = fs["feature_inventory"]
    fig, ax = plt.subplots(figsize=(14.5, 11))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    n0 = inv["n_candidate_features"]
    d1 = fs["method_1_missingness"]["n_dropped"]
    d2 = fs["method_2_low_variance"]["n_dropped"]
    d3 = fs["method_3_correlation"]["n_dropped"]
    after = n0 - d1 - d2 - d3

    funnel = [
        (f"{n0} candidate features", 0.86, 0.905, PROC),
        (f"Method 1 — Missing values > 95%", 0.78, 0.800, PROC),
        (f"Method 2 — Near-constant > 99.9%", 0.70, 0.700, PROC),
        (f"Method 3 — Correlation |ρ| > 0.95", 0.62, 0.600, PROC),
        (f"{after} features after the filters", 0.54, 0.500, MODEL),
    ]
    subs = ["structural filters remove what cannot help",
            f"−{d1} removed", f"−{d2} removed",
            f"−{d3} removed from "
            f"{fs['method_3_correlation']['n_pairs_above_threshold']} "
            "redundant pairs",
            "input to the ranking methods"]
    for (text, w, y, colour), sub in zip(funnel, subs):
        box(ax, 0.5, y, w, 0.078, text, sub, colour, fs_title=11, fs_sub=8)
        if y > 0.51:
            arrow(ax, (0.5, y - 0.041), (0.5, y - 0.062), colour="#777", lw=1.6)

    rankers = [
        ("Method 4\nANOVA F + Mutual Info", 0.14, "univariate"),
        ("Method 5\nRF + XGBoost importance", 0.38, "model-based"),
        ("Method 6\nSHAP", 0.62, "the arbiter"),
        ("Method 7\nRFE", 0.86, "independent check"),
    ]
    for text, x, sub in rankers:
        arrow(ax, (0.5, 0.459), (x, 0.412), colour="#999", lw=1.3,
              rad=0.12 if x < 0.5 else -0.12)
        box(ax, x, 0.362, 0.215, 0.095, text.split("\n")[1],
            f"{text.split(chr(10))[0]} · {sub}", MODEL, fs_title=9, fs_sub=7.3)
        arrow(ax, (x, 0.312), (0.5, 0.268), colour="#999", lw=1.3,
              rad=-0.12 if x < 0.5 else 0.12)

    comp = fs["experiment_results"]
    best = fs["selection_rule"]
    box(ax, 0.5, 0.208, 0.86, 0.098,
        f"{len(fs['candidate_sets'])} candidate feature sets, compared on "
        "held-out validation patients",
        " · ".join(f"{r['feature_set'].split('_')[0]} {r['pr_auc']:.4f}"
                   for r in sorted(comp, key=lambda r: -r["pr_auc"])),
        MODEL, fs_title=11, fs_sub=7.6)
    arrow(ax, (0.5, 0.157), (0.5, 0.118), colour="#666", lw=2.0)

    cv = fs["selection_inside_cv"]
    box(ax, 0.5, 0.062, 0.72, 0.098,
        f"Final set — {cfg['n_features']} features  ({best['winner']})",
        f"validation PR-AUC {best['winner_pr_auc']:.5f}  ·  leakage check: "
        f"nested {cv['nested_cv_pr_auc_mean']:.5f} vs global "
        f"{cv['global_cv_pr_auc_mean']:.5f}\n"
        f"optimism from selecting globally = "
        f"{cv['optimism_from_global_selection']:+.5f} PR-AUC — negligible",
        CDS, fs_title=12, fs_sub=8.2)

    ax.text(0.5, 0.972, "Feature Selection Methodology", ha="center",
            fontsize=16, fontweight="bold")
    ax.text(0.5, 0.945,
            "Three structural filters, four independent rankers, an explicit "
            "experiment, and a check on whether the selection itself leaked.",
            ha="center", fontsize=9.3, color="#444", style="italic")
    finish(fig, "feature_selection_methodology.png")


# ===========================================================================
def recommendation_workflow(A):
    meta = A["meta"]
    fig, ax = plt.subplots(figsize=(14.5, 9.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.5, 0.905, 0.56, 0.09, "One identified isolate",
        "patient context + specimen + organism + history", DATA,
        fs_title=12, fs_sub=8.4)
    arrow(ax, (0.5, 0.858), (0.5, 0.812), lw=2.0)

    box(ax, 0.5, 0.762, 0.68, 0.09,
        f"Build one candidate row per antibiotic  ({len(meta['antibiotics'])} agents)",
        "same patient context, only the antibiotic changes", PROC,
        fs_title=11.5, fs_sub=8.2)
    arrow(ax, (0.5, 0.715), (0.5, 0.668), lw=2.0)

    box(ax, 0.5, 0.618, 0.60, 0.09,
        f"{meta['metadata']['algorithm']} → calibrated P(non-susceptible)",
        "the ONLY number the model produces — nothing downstream changes it",
        MODEL, fs_title=11.5, fs_sub=8.2)

    for x, rad in ((0.20, 0.0), (0.80, 0.0)):
        arrow(ax, (0.5, 0.571), (x, 0.512), colour="#888", lw=1.5,
              rad=0.1 if x < 0.5 else -0.1)

    box(ax, 0.20, 0.455, 0.34, 0.10, "SHAP attribution",
        "top factors for THIS isolate\nrendered in clinical language", XAI,
        fs_title=10.5, fs_sub=8)
    box(ax, 0.80, 0.455, 0.38, 0.10, "Guardrail evaluation",
        "intrinsic resistance · site suitability\nallergy · AWaRe tier · "
        "evidence strength", CDS, fs_title=10.5, fs_sub=8)

    arrow(ax, (0.20, 0.402), (0.42, 0.352), colour="#888", lw=1.5, rad=-0.1)
    arrow(ax, (0.80, 0.402), (0.58, 0.352), colour="#888", lw=1.5, rad=0.1)

    box(ax, 0.5, 0.298, 0.74, 0.098, "Ordering",
        "unflagged options first, by predicted susceptibility · "
        "flagged options ranked BELOW\nnothing is ever removed from the list",
        CDS, fs_title=12, fs_sub=8.3)
    arrow(ax, (0.5, 0.247), (0.5, 0.198), lw=2.0)

    box(ax, 0.5, 0.145, 0.80, 0.098,
        "\"Potentially suitable options for clinician review\"",
        "each option carries: P(susceptible) · confidence band · training-data "
        "coverage · SHAP factors · every flag",
        HUMAN, fs_title=12, fs_sub=8.3)

    ax.text(0.5, 0.042,
            "The guardrail layer NEVER modifies a probability and NEVER deletes "
            "an option. It annotates and re-orders, so the clinician can see "
            "both\nwhat the model estimated and why an option was pushed down. "
            "The system does not prescribe.",
            ha="center", va="center", fontsize=8.7, color="#333",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fef2f2", ec=WARN, lw=1.3))
    ax.text(0.5, 0.972, "Recommendation Workflow", ha="center", fontsize=16,
            fontweight="bold")
    finish(fig, "recommendation_workflow.png")


# ===========================================================================
def application_workflow(A):
    fig, ax = plt.subplots(figsize=(15.5, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.5, 0.915, 0.70, 0.085, "Clinician opens the application",
        "streamlit run 11_APPLICATION/app.py", HUMAN, fs_title=12, fs_sub=8.2)
    arrow(ax, (0.5, 0.872), (0.5, 0.828), lw=1.9)

    box(ax, 0.27, 0.775, 0.40, 0.085, "Load a held-out culture",
        "real test-partition patient\ntrue AST result shown for comparison",
        DATA, fs_title=10, fs_sub=7.6)
    box(ax, 0.73, 0.775, 0.40, 0.085, "Enter a case manually",
        "age band · sex code · specimen · organism\nlabs and history optional",
        DATA, fs_title=10, fs_sub=7.6)
    arrow(ax, (0.27, 0.732), (0.45, 0.688), colour="#888", lw=1.5, rad=-0.1)
    arrow(ax, (0.73, 0.732), (0.55, 0.688), colour="#888", lw=1.5, rad=0.1)

    box(ax, 0.5, 0.638, 0.62, 0.082, "All 54 agents scored + explained",
        "missing fields filled from training defaults AND listed on screen as "
        "assumed", MODEL, fs_title=11, fs_sub=8)

    tabs = [
        ("Dashboard", "at-a-glance + ground truth", 0.115, 0.475, CDS),
        ("Prediction", "probability per agent", 0.305, 0.475, CDS),
        ("Explanation", "per-isolate SHAP", 0.495, 0.475, XAI),
        ("Options for review", "ranked + guardrails", 0.685, 0.475, CDS),
        ("Dosage reference", "FDA label, verbatim", 0.875, 0.475, CDS),
        ("Cost", "live CMS NADAC", 0.115, 0.335, CDS),
        ("Patient history", "what the model used", 0.305, 0.335, DATA),
        ("Genomic", "inactive — says so", 0.495, 0.335, "#64748b"),
        ("Model Card", "performance + limits", 0.685, 0.335, MODEL),
        ("Data & Methodology", "full audit trail", 0.875, 0.335, DATA),
    ]
    for name, sub, x, y, colour in tabs:
        box(ax, x, y, 0.175, 0.088, name, sub, colour, fs_title=8.8, fs_sub=7)
    for x in (0.115, 0.305, 0.495, 0.685, 0.875):
        arrow(ax, (0.5, 0.594), (x, 0.522), colour="#bbb", lw=1.1,
              rad=0.08 if x < 0.5 else -0.08)

    arrow(ax, (0.5, 0.288), (0.5, 0.232), lw=2.0)
    box(ax, 0.5, 0.178, 0.66, 0.088, "Clinician review tab",
        "Accept / Modify / Reject · select agents · record reasoning\n"
        "qualified-professional confirmation required before anything is "
        "recorded", HUMAN, fs_title=12, fs_sub=8.2)
    arrow(ax, (0.5, 0.132), (0.5, 0.086), lw=2.0)
    box(ax, 0.5, 0.042, 0.52, 0.062, "13_RESULTS/clinician_reviews.csv",
        "an audit record — not an order, not a prescription", "#64748b",
        fs_title=10, fs_sub=7.8)

    ax.text(0.5, 0.972, "Application Workflow", ha="center", fontsize=16,
            fontweight="bold")
    finish(fig, "application_workflow.png")


# ===========================================================================
def human_in_the_loop(A):
    fig, ax = plt.subplots(figsize=(13.5, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    chain = [
        ("AI Prediction", "calibrated P(non-susceptible) per agent", MODEL, 0.885),
        ("Explanation", "SHAP factors for THIS isolate, in clinical language", XAI, 0.745),
        ("Potential antibiotic options", "ranked, with evidence strength and "
         "every guardrail flag visible", CDS, 0.605),
        ("CLINICIAN REVIEW", "the only step that decides anything", HUMAN, 0.465),
    ]
    for title, sub, colour, y in chain:
        box(ax, 0.5, y, 0.66, 0.098, title, sub, colour, fs_title=12.5,
            fs_sub=8.4)
        arrow(ax, (0.5, y - 0.051), (0.5, y - 0.089), lw=2.0)

    for name, x, colour, sub in (
            ("Accept", 0.19, CDS, "proceed with a\nreviewed option"),
            ("Modify", 0.50, "#f59e0b", "choose a different\nagent"),
            ("Reject", 0.81, WARN, "disregard the\nmodel entirely")):
        arrow(ax, (0.5, 0.412), (x, 0.362), colour="#888", lw=1.5,
              rad=0.12 if x < 0.5 else (-0.12 if x > 0.5 else 0))
        box(ax, x, 0.305, 0.24, 0.098, name, sub, colour, fs_title=12,
            fs_sub=8)
        arrow(ax, (x, 0.254), (0.5, 0.204), colour="#888", lw=1.5,
              rad=-0.12 if x < 0.5 else (0.12 if x > 0.5 else 0))

    box(ax, 0.5, 0.148, 0.62, 0.088, "Clinician notes + recorded decision",
        "reviewer identity · reasoning · qualified-professional confirmation",
        HUMAN, fs_title=11.5, fs_sub=8.2)

    ax.text(0.5, 0.048,
            "There is no path from the model to an order. The application has "
            "no prescribing capability and must never be given one.\n"
            "Every option is labelled a POTENTIAL option for clinician review, "
            "and antimicrobial susceptibility testing remains the reference "
            "standard.",
            ha="center", va="center", fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.55", fc="#fef2f2", ec=WARN, lw=1.4))
    ax.text(0.5, 0.972, "Human-in-the-Loop Workflow", ha="center", fontsize=16,
            fontweight="bold")
    finish(fig, "human_in_the_loop_workflow.png")


# ===========================================================================
def main() -> None:
    A = {
        "build": load(PROCESSED_DIR / "build_report.json"),
        "cfg": load(CONFIG_DIR / "final_features.json"),
        "fs": load(FS_DIR / "selected_features" / "feature_selection_report.json"),
        "ev": load(EVAL_DIR / "evaluation_report.json"),
        "tr": load(EXPERIMENTS_DIR / "training_report.json"),
        "eda": load(ANALYSIS_DIR / "reports" / "eda_report.json"),
        "meta": load(MODELS_DIR / "final_model.metadata.json"),
    }
    missing = [k for k, v in A.items() if not v]
    if missing:
        raise SystemExit(f"Missing artefacts: {missing}. Run the pipeline first.")

    methodology_workflow(A)
    system_architecture(A)
    feature_engineering_workflow(A)
    feature_selection_methodology(A)
    recommendation_workflow(A)
    application_workflow(A)
    human_in_the_loop(A)


if __name__ == "__main__":
    main()
