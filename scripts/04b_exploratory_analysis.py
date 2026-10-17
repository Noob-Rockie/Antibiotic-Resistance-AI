"""PHASE 4 — Complete exploratory data analysis.

Answers the questions a reader of the report will ask before believing any model
result: what is in this cohort, how clean is it, what does resistance actually
look like, and how much signal do the history features carry on their own.

SCOPE RULE (stated once, applied throughout)
  * **Structural description** — record counts, cardinalities, missingness,
    duplicates, distributions of inputs — is computed on the FULL analytic
    dataset, because describing the data is not a modelling decision.
  * **Anything involving the TARGET** — resistance rates by organism/agent/year,
    the antibiogram, history-vs-outcome associations — is computed on the
    TRAINING partition only, so that nothing seen while exploring can influence
    a decision that is later evaluated on validation or test data.
  Every figure and table states which of the two it used.

Outputs
  02_DATA_ANALYSIS/figures/01..10_*.png      (+ copies in 14_REPORT_MATERIAL)
  02_DATA_ANALYSIS/tables/*.csv              (+ copies in 14_REPORT_MATERIAL)
  02_DATA_ANALYSIS/reports/eda_report.json   machine-readable, feeds EDA_REPORT.md
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (ANALYSIS_DIR, PROCESSED_DIR, REPORT_DIR,  # noqa: E402
                          RESULTS_DIR)
from src.models.dataset import (PAT, TARGET_BINARY, load_analytic,  # noqa: E402
                                patient_grouped_split)

FIGS = ANALYSIS_DIR / "figures"
TABLES = ANALYSIS_DIR / "tables"
RM_FIGS = REPORT_DIR / "figures" / "EDA"
RM_TABLES = REPORT_DIR / "tables" / "EDA"
for d in (FIGS, TABLES, RM_FIGS, RM_TABLES, ANALYSIS_DIR / "reports"):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=0.95)
BLUE, RED, AMBER, GREEN, PURPLE = ("#2563eb", "#dc2626", "#f59e0b",
                                   "#059669", "#7c3aed")
REPORT: dict = {}


def log(m):
    print(m, flush=True)


def savefig(fig, name):
    fig.savefig(FIGS / name, dpi=170, bbox_inches="tight", facecolor="white")
    shutil.copy2(FIGS / name, RM_FIGS / name)
    plt.close(fig)
    log(f"  figure -> {name}")


def savetable(df, name, index=False):
    df.to_csv(TABLES / name, index=index)
    shutil.copy2(TABLES / name, RM_TABLES / name)
    log(f"  table  -> {name}")


def annotate(ax, values, fmt="{:.1%}", dy=0.0, fontsize=8.5, horizontal=False):
    for i, v in enumerate(values):
        if horizontal:
            ax.text(v + dy, i, fmt.format(v), va="center", ha="left",
                    fontsize=fontsize)
        else:
            ax.text(i, v + dy, fmt.format(v), ha="center", va="bottom",
                    fontsize=fontsize)


# ===========================================================================
def section_1_overview(df, tr, audit, build):
    log("\n1. DATASET OVERVIEW")
    ov = {
        "n_records": int(len(df)),
        "n_patients": int(df[PAT].nunique()),
        "n_culture_orders": int(df["order_proc_id_coded"].nunique()),
        "n_organisms": int(df["organism_clean"].nunique()),
        "n_organism_groups": int(df["organism_group"].nunique()),
        "n_antibiotics": int(df["antibiotic"].nunique()),
        "n_antibiotic_classes": int(df["antibiotic_class"].nunique()),
        "n_specimen_types": int(df["culture_description"].nunique()),
        "date_min": str(df["order_time"].min())[:10],
        "date_max": str(df["order_time"].max())[:10],
        "n_columns_analytic_table": int(df.shape[1]),
        "n_candidate_features": int(df.shape[1] - 9),  # 9 id/target columns
        "target_variable": TARGET_BINARY,
        "target_definition": "1 = non-susceptible (Resistant or Intermediate), "
                             "0 = Susceptible",
        "mean_rows_per_order": round(len(df) / df["order_proc_id_coded"].nunique(), 3),
        "mean_orders_per_patient": round(
            df.groupby(PAT, observed=True)["order_proc_id_coded"].nunique().mean(), 3),
        "raw_rows_before_cleaning": int(build["step1_cohort"]["n_rows_raw"]),
        "raw_culture_orders_including_negative": int(
            build["step1_cohort"]["n_all_culture_orders"]),
    }
    ov["training_partition"] = {
        "n_records": int(len(tr)),
        "n_patients": int(tr[PAT].nunique()),
        "n_culture_orders": int(tr["order_proc_id_coded"].nunique()),
    }
    REPORT["overview"] = ov
    for k, v in ov.items():
        if not isinstance(v, dict):
            log(f"   {k:<38} {v:,}" if isinstance(v, int) else f"   {k:<38} {v}")

    savetable(pd.DataFrame([{"property": k, "value": v} for k, v in ov.items()
                            if not isinstance(v, dict)]),
              "01_dataset_overview.csv")
    return ov


# ===========================================================================
def section_2_missing(df):
    log("\n2. MISSING-VALUE ANALYSIS")
    miss = pd.DataFrame({
        "column": df.columns,
        "n_missing": df.isna().sum().to_numpy(),
    })
    miss["pct_missing"] = (100 * miss["n_missing"] / len(df)).round(4)
    miss = miss.sort_values("pct_missing", ascending=False).reset_index(drop=True)
    nonzero = miss[miss["pct_missing"] > 0]
    savetable(miss, "02_missing_values.csv")

    fig, axes = plt.subplots(1, 2, figsize=(17, 11),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    top = nonzero.head(45).iloc[::-1]
    colors = [RED if v > 90 else AMBER if v > 50 else BLUE
              for v in top["pct_missing"]]
    axes[0].barh(range(len(top)), top["pct_missing"], color=colors)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top["column"], fontsize=7)
    axes[0].set_xlabel("% missing")
    axes[0].set_title(f"45 columns with the most missing data\n"
                      f"({len(nonzero)} of {len(miss)} columns have any)",
                      fontsize=11.5, fontweight="bold")
    axes[0].set_xlim(0, 100)

    bins = [(0, 0, "complete"), (0, 25, "0-25%"), (25, 50, "25-50%"),
            (50, 75, "50-75%"), (75, 95, "75-95%"), (95, 100.1, ">95%")]
    counts = [int((miss["pct_missing"] == 0).sum())]
    counts += [int(((miss["pct_missing"] > lo) & (miss["pct_missing"] <= hi)).sum())
               for lo, hi, _ in bins[1:]]
    labels = [b[2] for b in bins]
    bar_colors = [GREEN, BLUE, BLUE, AMBER, AMBER, RED]
    axes[1].bar(labels, counts, color=bar_colors)
    for i, c in enumerate(counts):
        axes[1].text(i, c, str(c), ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("Number of columns")
    axes[1].set_title("How missingness is distributed across columns",
                      fontsize=11.5, fontweight="bold")
    axes[1].margins(y=0.15)
    fig.suptitle("Missing-value analysis — full analytic dataset "
                 f"({len(df):,} rows)", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "01_missing_values.png")

    REPORT["missing"] = {
        "n_columns": int(len(miss)),
        "n_columns_complete": int((miss["pct_missing"] == 0).sum()),
        "n_columns_with_missing": int(len(nonzero)),
        "n_columns_above_95pct": int((miss["pct_missing"] > 95).sum()),
        "top_20": nonzero.head(20).to_dict("records"),
        "distribution": dict(zip(labels, counts)),
        "note": ("Missingness here is structural, not random. Laboratory "
                 "columns are missing when the test was not ordered; "
                 "`*_days_since` columns are missing when the event never "
                 "happened. Both are informative, so the tree models receive "
                 "NaN directly rather than an imputed value."),
    }
    log(f"   {len(nonzero)} of {len(miss)} columns have missing values; "
        f"{int((miss['pct_missing'] > 95).sum())} above 95%")


# ===========================================================================
def section_3_duplicates(df, audit, build):
    log("\n3. DUPLICATE ANALYSIS")
    s1 = build["step1_cohort"]
    raw_dups = [{"file": d["file_name"],
                 "rows": d["n_rows"],
                 "exact_duplicate_rows": d["n_exact_duplicate_rows"],
                 "pct": d["pct_exact_duplicate_rows"]}
                for d in audit["datasets"]]

    # --- duplicated COLUMNS in the source export ---------------------------
    # Four "last_*" laboratory columns carry byte-identical values under four
    # different analyte names, and so do four "first_*" columns. The audit hinted
    # at it (identical missingness AND identical unique counts); this confirms it.
    lab_like = [c for c in df.columns
                if c.startswith(("last_", "first_", "median_", "Q25_", "Q75_"))]
    dup_cols, seen = [], {}
    for c in lab_like:
        v = pd.to_numeric(df[c], errors="coerce")
        sig = (int(v.notna().sum()), round(float(v.sum(skipna=True)), 3))
        if sig in seen:
            a = pd.to_numeric(df[seen[sig]], errors="coerce")
            both = a.notna() & v.notna()
            if both.any() and bool((a[both] == v[both]).all()):
                dup_cols.append({"column": c, "identical_to": seen[sig],
                                 "n_compared": int(both.sum())})
                continue
        seen[sig] = c

    key = ["order_proc_id_coded", "organism_clean", "antibiotic"]
    n_dup_final = int(df.duplicated(key).sum())
    order_org = df.groupby("order_proc_id_coded", observed=True)[
        "organism_clean"].nunique()
    pat_orders = df.groupby(PAT, observed=True)["order_proc_id_coded"].nunique()

    dup = {
        "raw_exact_duplicates": raw_dups,
        "cohort_duplicate_key_groups": int(s1["n_duplicate_key_groups"]),
        "cohort_rows_in_duplicate_groups": int(s1["n_rows_in_duplicate_key_groups"]),
        "cohort_groups_with_conflicting_labels": int(
            s1["n_duplicate_groups_with_conflicting_labels"]),
        "rows_removed_by_resolution": int(s1["n_dropped_duplicate_resolution"]),
        "resolution_rule": s1["dedup_rule"],
        "duplicates_remaining_in_analytic_table": n_dup_final,
        "orders_with_multiple_organisms": int((order_org > 1).sum()),
        "max_organisms_in_one_order": int(order_org.max()),
        "patients_with_multiple_orders": int((pat_orders > 1).sum()),
        "max_orders_for_one_patient": int(pat_orders.max()),
        "duplicated_laboratory_columns": dup_cols,
        "duplicated_columns_finding": (
            "The source laboratory export contains duplicated COLUMNS: "
            "`last_hgb`, `last_plt`, `last_lymphocytes` and `last_neutrophils` "
            "hold byte-identical values under four different analyte names, and "
            "the four matching `first_*` columns do the same. Four distinct "
            "analytes cannot legitimately agree to the digit on every row, so "
            "this is a defect in the export, not a coincidence. It was detected "
            "by comparing columns directly after the audit flagged that they "
            "shared identical missingness AND identical unique-value counts. "
            "Impact on this project: none. All eight columns are 96-99% missing "
            "and were removed by the feature-selection missingness filter long "
            "before modelling, so no corrupted value reaches the deployed model. "
            "It is reported because a future version that tried to use `first_*` "
            "or `last_*` laboratory features would silently train on four copies "
            "of one unidentified analyte."),
        "action_taken": (
            "The 40.3% exact duplicates in the prior-medication file are the "
            "same drug administered more than once on the same day; exposure is "
            "counted in distinct drug-days, so they were collapsed rather than "
            "discarded. In the cohort, every duplicated "
            "(order, organism, antibiotic) key carried CONFLICTING results — "
            "one specimen yielding two strains of the same species with "
            "different susceptibility. These were resolved by "
            "most-resistant-wins, the conservative clinical reading. The "
            "analytic table now contains zero duplicate keys, asserted by "
            "`12_TESTS/test_data_integrity.py`."),
    }
    REPORT["duplicates"] = dup
    savetable(pd.DataFrame(raw_dups), "03_duplicate_analysis_raw_files.csv")
    log(f"   raw exact duplicates: prior_med "
        f"{raw_dups[-1]['exact_duplicate_rows']:,} ({raw_dups[-1]['pct']}%)")
    log(f"   conflicting cohort key groups: "
        f"{dup['cohort_groups_with_conflicting_labels']:,}")
    log(f"   duplicates remaining in the analytic table: {n_dup_final}")


# ===========================================================================
def section_4_target(df, tr):
    log("\n4. TARGET DISTRIBUTION")
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))

    order3 = ["Susceptible", "Intermediate", "Resistant"]
    vc = df["y_3class"].value_counts().reindex(order3)
    axes[0].bar(order3, vc.values, color=[GREEN, AMBER, RED])
    for i, v in enumerate(vc.values):
        axes[0].text(i, v, f"{v:,}\n{100*v/len(df):.2f}%", ha="center",
                     va="bottom", fontsize=9)
    axes[0].set_ylabel("Rows")
    axes[0].set_title("Three AST categories")
    axes[0].margins(y=0.20)

    b = df[TARGET_BINARY].value_counts().sort_index()
    axes[1].bar(["Susceptible\n(0)", "Non-susceptible\n(1) = R + I"], b.values,
                color=[GREEN, RED])
    for i, v in enumerate(b.values):
        axes[1].text(i, v, f"{v:,}\n{100*v/len(df):.2f}%", ha="center",
                     va="bottom", fontsize=9.5)
    axes[1].set_ylabel("Rows")
    axes[1].set_title("Binary target used by the model")
    axes[1].margins(y=0.20)

    ratio = b.iloc[0] / b.iloc[1]
    axes[2].pie(b.values, labels=["Susceptible", "Non-susceptible"],
                colors=[GREEN, RED], autopct="%1.2f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 10})
    axes[2].set_title(f"Class imbalance — {ratio:.2f} : 1\n"
                      f"a majority-class-only model would score "
                      f"{100*b.iloc[0]/len(df):.1f}% accuracy")

    fig.suptitle(f"Target distribution — full analytic dataset "
                 f"({len(df):,} rows)", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "02_target_distribution.png")

    tgt = pd.DataFrame([
        {"category": k, "rows": int(v), "pct": round(100 * v / len(df), 4)}
        for k, v in vc.items()])
    savetable(tgt, "04_target_distribution.csv")

    REPORT["target"] = {
        "three_class": {k: int(v) for k, v in vc.items()},
        "binary": {"susceptible": int(b.iloc[0]),
                   "non_susceptible": int(b.iloc[1])},
        "positive_rate_full": round(float(df[TARGET_BINARY].mean()), 6),
        "positive_rate_train": round(float(tr[TARGET_BINARY].mean()), 6),
        "imbalance_ratio": round(float(ratio), 4),
        "majority_class_accuracy": round(float(b.iloc[0] / len(df)), 6),
        "note": ("Imbalanced but not extremely so. The consequence for "
                 "evaluation is that accuracy is nearly uninformative "
                 f"({100*b.iloc[0]/len(df):.1f}% for a model that always says "
                 "susceptible), which is why PR-AUC is the selection metric."),
    }
    log(f"   positive rate {df[TARGET_BINARY].mean():.4f} (full), "
        f"{tr[TARGET_BINARY].mean():.4f} (train)")


# ===========================================================================
def section_5_antibiotics(df, tr):
    log("\n5. ANTIBIOTIC DISTRIBUTION")
    abx = (df.groupby("antibiotic", observed=True)
           .agg(n_rows=("antibiotic", "size"),
                n_orders=("order_proc_id_coded", "nunique"),
                n_organisms=("organism_clean", "nunique"))
           .join(df.groupby("antibiotic", observed=True)["antibiotic_class"].first())
           .join(tr.groupby("antibiotic", observed=True)[TARGET_BINARY]
                 .mean().rename("nonsusc_rate_train"))
           .sort_values("n_rows", ascending=False))
    abx["pct_of_rows"] = (100 * abx["n_rows"] / len(df)).round(4)
    abx["nonsusc_rate_train"] = abx["nonsusc_rate_train"].round(5)
    savetable(abx.reset_index(), "05_antibiotic_distribution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(18, 11),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    top = abx.head(30).iloc[::-1]
    axes[0].barh(range(len(top)), top["n_rows"], color=BLUE)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top.index, fontsize=8)
    axes[0].set_xlabel("Tested organism-antibiotic pairs")
    axes[0].set_title("30 most-tested agents", fontsize=11.5, fontweight="bold")
    for i, v in enumerate(top["n_rows"]):
        axes[0].text(v, i, f" {v:,}", va="center", fontsize=7)

    counts = abx["n_rows"].sort_values()
    axes[1].barh(range(len(counts)), counts.values,
                 color=[RED if v < 1000 else AMBER if v < 10000 else BLUE
                        for v in counts.values])
    axes[1].set_yticks(range(len(counts)))
    axes[1].set_yticklabels(counts.index, fontsize=5.6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Rows (log scale)")
    axes[1].axvline(1000, color="#111", ls="--", lw=1.2)
    axes[1].axvline(20000, color=GREEN, ls="--", lw=1.2)
    axes[1].set_title(f"All {len(abx)} agents on a log scale\n"
                      "red < 1,000 rows · dashed line = 1,000 · "
                      "green = 20,000 (the per-agent-model threshold)",
                      fontsize=10.5, fontweight="bold")
    fig.suptitle("Antibiotic distribution — the panel is extremely ragged",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "03_antibiotic_distribution.png")

    rare = abx[abx["n_rows"] < 1000]
    very_rare = abx[abx["n_rows"] < 100]
    REPORT["antibiotics"] = {
        "n_antibiotics": int(len(abx)),
        "n_classes": int(df["antibiotic_class"].nunique()),
        "most_tested": abx.head(10)["n_rows"].astype(int).to_dict(),
        "least_tested": abx.tail(10)["n_rows"].astype(int).to_dict(),
        "n_agents_below_1000_rows": int(len(rare)),
        "n_agents_below_100_rows": int(len(very_rare)),
        "rare_agents": rare["n_rows"].astype(int).to_dict(),
        "max_to_min_ratio": int(abx["n_rows"].max() // max(abx["n_rows"].min(), 1)),
        "why_unified_model": (
            f"The most-tested agent has {abx['n_rows'].max():,} rows and the "
            f"least-tested has {abx['n_rows'].min()}. "
            f"{len(rare)} of {len(abx)} agents have fewer than 1,000 rows and "
            f"{len(very_rare)} have fewer than 100. A separate model per "
            "antibiotic is therefore untrainable for most of the panel, and a "
            "multi-label target would be ~76% missing by construction because "
            "each isolate is only tested against the agents the laboratory "
            "selected. A single unified model with the antibiotic and its class "
            "as input features uses every row and lets a rare agent inherit "
            "structure from its class — which is the only way to score an agent "
            "the laboratory did not test."),
    }
    log(f"   {len(abx)} agents; {len(rare)} have <1,000 rows, "
        f"{len(very_rare)} have <100")


# ===========================================================================
def section_6_organisms(df, tr):
    log("\n6. ORGANISM DISTRIBUTION")
    org = (df.groupby("organism_clean", observed=True)
           .agg(n_rows=("organism_clean", "size"),
                n_orders=("order_proc_id_coded", "nunique"),
                n_agents_tested=("antibiotic", "nunique"))
           .join(df.groupby("organism_clean", observed=True)["organism_group"].first())
           .join(tr.groupby("organism_clean", observed=True)[TARGET_BINARY]
                 .mean().rename("nonsusc_rate_train"))
           .sort_values("n_rows", ascending=False))
    org["pct_of_rows"] = (100 * org["n_rows"] / len(df)).round(4)
    org["nonsusc_rate_train"] = org["nonsusc_rate_train"].round(5)
    savetable(org.reset_index(), "06_organism_distribution.csv")

    grp = (df.groupby("organism_group", observed=True)
           .agg(n_rows=("organism_group", "size"),
                n_species=("organism_clean", "nunique"),
                n_orders=("order_proc_id_coded", "nunique"))
           .join(tr.groupby("organism_group", observed=True)[TARGET_BINARY]
                 .mean().rename("nonsusc_rate_train"))
           .sort_values("n_rows", ascending=False))
    savetable(grp.reset_index(), "06b_organism_group_distribution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    top = org.head(25).iloc[::-1]
    axes[0].barh(range(len(top)), top["n_rows"], color=PURPLE)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels([s[:42] for s in top.index], fontsize=8)
    axes[0].set_xlabel("Tested organism-antibiotic pairs")
    axes[0].set_title("25 most frequent organisms", fontsize=11.5,
                      fontweight="bold")
    for i, v in enumerate(top["n_rows"]):
        axes[0].text(v, i, f" {v:,}", va="center", fontsize=7)

    g = grp.iloc[::-1]
    axes[1].barh(range(len(g)), g["n_rows"], color=BLUE)
    axes[1].set_yticks(range(len(g)))
    axes[1].set_yticklabels([f"{i}\n({int(r.n_species)} species)"
                             for i, r in g.iterrows()], fontsize=8)
    axes[1].set_xlabel("Rows")
    axes[1].set_title("By organism group (gram stain / family)",
                      fontsize=11.5, fontweight="bold")
    for i, v in enumerate(g["n_rows"]):
        axes[1].text(v, i, f" {v:,}", va="center", fontsize=7.5)
    fig.suptitle(f"Organism distribution — {df['organism_clean'].nunique()} "
                 f"cleaned species across {df['organism_group'].nunique()} groups",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "04_organism_distribution.png")

    top5_share = 100 * org.head(5)["n_rows"].sum() / len(df)
    REPORT["organisms"] = {
        "n_species": int(len(org)),
        "n_groups": int(len(grp)),
        "top_10": org.head(10)["n_rows"].astype(int).to_dict(),
        "top_5_share_pct": round(float(top5_share), 3),
        "n_species_below_100_rows": int((org["n_rows"] < 100).sum()),
        "by_group": {str(k): {"n_rows": int(r["n_rows"]),
                              "n_species": int(r["n_species"]),
                              "nonsusc_rate_train": (
                                  None if pd.isna(r["nonsusc_rate_train"])
                                  else round(float(r["nonsusc_rate_train"]), 5))}
                     for k, r in grp.iterrows()},
        "note": (f"The cohort is dominated by a handful of species: the five "
                 f"most common account for {top5_share:.1f}% of all rows, with "
                 "Escherichia coli alone the largest by a wide margin. "
                 f"{int((org['n_rows'] < 100).sum())} species have fewer than "
                 "100 rows. This is why `organism_group` is carried alongside "
                 "`organism_clean`: a rare species can still be scored through "
                 "its group's intrinsic-resistance structure."),
    }
    log(f"   {len(org)} species, {len(grp)} groups; top 5 = {top5_share:.1f}% of rows")


# ===========================================================================
def section_7_specimen(df, tr):
    log("\n7. SPECIMEN DISTRIBUTION")
    spec = (df.groupby("culture_description", observed=True)
            .agg(n_rows=("culture_description", "size"),
                 n_orders=("order_proc_id_coded", "nunique"),
                 n_patients=(PAT, "nunique"),
                 n_organisms=("organism_clean", "nunique"),
                 n_agents=("antibiotic", "nunique"))
            .join(tr.groupby("culture_description", observed=True)[TARGET_BINARY]
                  .mean().rename("nonsusc_rate_train"))
            .sort_values("n_rows", ascending=False))
    savetable(spec.reset_index(), "07_specimen_distribution.csv")

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
    axes[0].bar(spec.index, spec["n_rows"], color=BLUE)
    for i, v in enumerate(spec["n_rows"]):
        axes[0].text(i, v, f"{v:,}\n{100*v/len(df):.1f}%", ha="center",
                     va="bottom", fontsize=9)
    axes[0].set_ylabel("Rows"); axes[0].set_title("Tested pairs by specimen")
    axes[0].margins(y=0.2)

    axes[1].bar(spec.index, spec["n_orders"], color=PURPLE)
    for i, v in enumerate(spec["n_orders"]):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Culture orders")
    axes[1].set_title("Positive culture orders by specimen")
    axes[1].margins(y=0.2)

    axes[2].bar(spec.index, spec["nonsusc_rate_train"], color=RED)
    annotate(axes[2], spec["nonsusc_rate_train"].to_numpy())
    axes[2].axhline(float(tr[TARGET_BINARY].mean()), color="#111", ls="--",
                    lw=1.3, label=f"cohort mean {tr[TARGET_BINARY].mean():.1%}")
    axes[2].set_ylabel("Non-susceptibility rate")
    axes[2].set_title("Non-susceptibility by specimen (training partition)")
    axes[2].legend(); axes[2].margins(y=0.2)
    fig.suptitle("Specimen distribution — this cohort contains urine, blood "
                 "and respiratory isolates only", fontsize=13.5,
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "05_specimen_distribution.png")

    REPORT["specimen"] = {
        "types": {str(k): {"n_rows": int(r["n_rows"]),
                           "n_orders": int(r["n_orders"]),
                           "n_patients": int(r["n_patients"]),
                           "nonsusc_rate_train": round(float(r["nonsusc_rate_train"]), 5)}
                  for k, r in spec.iterrows()},
        "note": ("Only three specimen types exist in this dataset. There are no "
                 "wound, tissue, CSF, stool or device isolates, so the model "
                 "must not be applied to them. Urine dominates, which shapes "
                 "which organisms and agents the model has seen most."),
    }
    for k, r in spec.iterrows():
        log(f"   {k:<14} {int(r['n_rows']):>9,} rows  "
            f"non-susceptible {r['nonsusc_rate_train']:.4f}")


# ===========================================================================
def section_8_demographics(df, tr):
    log("\n8. DEMOGRAPHIC ANALYSIS")
    age = (df.groupby("age", observed=True)
           .agg(n_rows=("age", "size"), n_patients=(PAT, "nunique"))
           .join(tr.groupby("age", observed=True)[TARGET_BINARY]
                 .mean().rename("nonsusc_rate_train"))
           .sort_index())
    sex = (df.groupby("gender_code", observed=True, dropna=False)
           .agg(n_rows=("gender_code", "size"), n_patients=(PAT, "nunique"))
           .join(tr.groupby("gender_code", observed=True)[TARGET_BINARY]
                 .mean().rename("nonsusc_rate_train")))
    sex.index = [("not recorded" if pd.isna(i) else f"code {int(i)}")
                 for i in sex.index]
    savetable(age.reset_index(), "08_age_distribution.csv")
    savetable(sex.reset_index().rename(columns={"index": "gender_code"}),
              "08b_sex_code_distribution.csv")

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
    axes[0].bar(range(len(age)), age["n_rows"], color=BLUE)
    axes[0].set_xticks(range(len(age)))
    axes[0].set_xticklabels(age.index, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Rows")
    ax2 = axes[0].twinx()
    ax2.plot(range(len(age)), age["nonsusc_rate_train"], "o-", color=RED, lw=2)
    ax2.set_ylabel("Non-susceptibility rate", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    axes[0].set_title("Age band: volume (bars) and\nnon-susceptibility (line)")

    axes[1].bar(sex.index, sex["n_rows"], color=PURPLE)
    for i, v in enumerate(sex["n_rows"]):
        axes[1].text(i, v, f"{v:,}\n{100*v/len(df):.1f}%", ha="center",
                     va="bottom", fontsize=9)
    axes[1].set_ylabel("Rows")
    axes[1].set_title("De-identified sex code\n(mapping to sex NOT disclosed)")
    axes[1].margins(y=0.2)

    vals = sex["nonsusc_rate_train"].fillna(0).to_numpy()
    axes[2].bar(sex.index, vals, color=RED)
    annotate(axes[2], vals)
    axes[2].axhline(float(tr[TARGET_BINARY].mean()), color="#111", ls="--",
                    lw=1.3, label=f"cohort mean {tr[TARGET_BINARY].mean():.1%}")
    axes[2].set_ylabel("Non-susceptibility rate")
    axes[2].set_title("Non-susceptibility by sex code\n(training partition)")
    axes[2].legend(); axes[2].margins(y=0.2)
    fig.suptitle("Demographics — the only patient attributes this dataset "
                 "provides", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "06b_demographics.png")

    REPORT["demographics"] = {
        "age_bands": {str(k): {"n_rows": int(r["n_rows"]),
                               "n_patients": int(r["n_patients"]),
                               "nonsusc_rate_train": round(float(r["nonsusc_rate_train"]), 5)}
                      for k, r in age.iterrows()},
        "sex_code": {str(k): {"n_rows": int(r["n_rows"]),
                              "nonsusc_rate_train": (
                                  None if pd.isna(r["nonsusc_rate_train"])
                                  else round(float(r["nonsusc_rate_train"]), 5))}
                     for k, r in sex.iterrows()},
        "constraints": (
            "The dataset provides exactly two patient attributes: a nine-level "
            "age BAND (no exact age) and a binary sex code. The source supplies "
            "no legend for the sex code, so its mapping to sex is unknown and "
            "is never asserted anywhere in this project. There is no weight, "
            "height, ethnicity, comorbidity, severity score or admission "
            "diagnosis. The youngest band is 18-24 years, so the cohort is "
            "adults only and the model must not be used in paediatrics."),
        "interpretation_warning": (
            "Differences in non-susceptibility across these groups are "
            "descriptive associations in one health system. They are confounded "
            "by which specimens are collected from whom, which organisms those "
            "specimens grow, and which agents the laboratory then tests. No "
            "causal or biological claim is made from them."),
    }
    log(f"   {len(age)} age bands (18-24 upward, adults only); "
        f"sex code 0/1 with no disclosed mapping")


# ===========================================================================
def section_9_prior_exposure(df, tr):
    log("\n9. PREVIOUS ANTIBIOTIC EXPOSURE")
    windows = [("prior_med_n_drugdays_30d", "30 days"),
               ("prior_med_n_drugdays_90d", "90 days"),
               ("prior_med_n_drugdays_180d", "180 days"),
               ("prior_med_n_drugdays_365d", "365 days")]
    cov = []
    for col, label in windows:
        if col not in df.columns:
            continue
        v = df[col].fillna(0)
        cov.append({"window": label, "column": col,
                    "pct_rows_with_exposure": round(100 * float((v > 0).mean()), 3),
                    "mean_drugdays_when_exposed": round(float(v[v > 0].mean()), 3),
                    "median_drugdays_when_exposed": float(v[v > 0].median()),
                    "max_drugdays": float(v.max())})
    cov_df = pd.DataFrame(cov)
    savetable(cov_df, "09_prior_exposure_by_window.csv")

    cls_cols = [c for c in df.columns
                if c.startswith("prior_med_class__") and c.endswith("__365d")]
    cls = []
    for c in cls_cols:
        name = c.replace("prior_med_class__", "").replace("__365d", "")
        name = name.replace("_", " ")
        v = df[c].fillna(0)
        exposed = v > 0
        cls.append({"antibiotic_class": name,
                    "pct_rows_exposed_365d": round(100 * float(exposed.mean()), 4),
                    "n_rows_exposed": int(exposed.sum()),
                    "nonsusc_rate_exposed_train": round(
                        float(tr.loc[tr[c].fillna(0) > 0, TARGET_BINARY].mean()), 5)
                    if (tr[c].fillna(0) > 0).sum() > 200 else None})
    cls_df = pd.DataFrame(cls).sort_values("pct_rows_exposed_365d",
                                           ascending=False)
    savetable(cls_df, "09b_prior_exposure_by_class.csv")

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    ax = axes[0, 0]
    ax.bar(cov_df["window"], cov_df["pct_rows_with_exposure"], color=BLUE)
    for i, v in enumerate(cov_df["pct_rows_with_exposure"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylabel("% of rows with any prior antibiotic")
    ax.set_title("Prior exposure coverage by look-back window")
    ax.margins(y=0.18)

    ax = axes[0, 1]
    d = cls_df.head(14).iloc[::-1]
    ax.barh(range(len(d)), d["pct_rows_exposed_365d"], color=PURPLE)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["antibiotic_class"], fontsize=8)
    ax.set_xlabel("% of rows with exposure in the last 365 days")
    ax.set_title("Prior exposure by antibiotic class")

    ax = axes[1, 0]
    v = df["prior_med_n_drugdays_90d"].fillna(0)
    v = v[v > 0]
    ax.hist(v, bins=np.arange(0, min(v.max(), 60) + 2), color=BLUE,
            edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Distinct antibiotic drug-days in the previous 90 days")
    ax.set_ylabel("Rows (log scale)")
    ax.set_title("Exposure intensity, among rows with any exposure")

    ax = axes[1, 1]
    dsl = df["prior_med_days_since_last"].dropna()
    ax.hist(dsl[dsl <= 730], bins=60, color=GREEN, edgecolor="white")
    ax.set_xlabel("Days since the patient's last recorded antibiotic")
    ax.set_ylabel("Rows")
    ax.set_title("Recency of the last antibiotic (first 2 years shown)")

    fig.suptitle("Previous antibiotic exposure — the classical driver of "
                 "subsequent resistance", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "06_prior_antibiotic_exposure.png")

    REPORT["prior_exposure"] = {
        "by_window": cov,
        "by_class_365d": cls_df.to_dict("records"),
        "pct_rows_with_any_prior_med_record": round(
            100 * float((df["has_prior_med_history"] == 1).mean()), 3),
        "note": ("Coverage is partial by construction: a prior-medication "
                 "record exists for 72.6% of rows, and care delivered outside "
                 "this health system is invisible. Absence of a record is "
                 "therefore not evidence of absence of exposure, and the model "
                 "carries an explicit `has_prior_med_history` indicator so it "
                 "can distinguish 'no exposure' from 'no data'."),
    }
    for r in cov:
        log(f"   {r['window']:<9} {r['pct_rows_with_exposure']:>6.2f}% of rows "
            f"have exposure")


# ===========================================================================
def section_10_prior_resistance(df, tr):
    log("\n10. PREVIOUS RESISTANCE HISTORY")
    feats = [
        ("has_prior_resistance_history", "Any prior resistance record exists"),
        ("prior_res_n_events_1095d", "Any resistant result in the last 3 years"),
        ("prior_res_same_abx_n", "Prior resistance to THIS antibiotic"),
        ("prior_res_same_org_abx_n", "Prior resistance to THIS organism+antibiotic"),
        ("prior_same_abx_n_nonsusc", "Prior non-susceptible result for THIS agent"),
        ("prior_same_org_nonsusc_rate", "Prior non-susceptibility for THIS organism"),
    ]
    rows = []
    for col, label in feats:
        if col not in df.columns:
            continue
        v_all = df[col].fillna(0)
        v_tr = tr[col].fillna(0)
        has = v_tr > 0
        if has.sum() < 100 or (~has).sum() < 100:
            continue
        r_yes = float(tr.loc[has, TARGET_BINARY].mean())
        r_no = float(tr.loc[~has, TARGET_BINARY].mean())
        rows.append({
            "feature": col, "description": label,
            "pct_rows_present": round(100 * float((v_all > 0).mean()), 3),
            "n_rows_present_train": int(has.sum()),
            "nonsusc_rate_absent": round(r_no, 5),
            "nonsusc_rate_present": round(r_yes, 5),
            "relative_risk": round(r_yes / r_no, 4) if r_no else None,
        })
    hist = pd.DataFrame(rows)
    savetable(hist, "10_prior_resistance_history.csv")

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    ax = axes[0, 0]
    d = hist.iloc[::-1]
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["nonsusc_rate_absent"], 0.4, color=GREEN, label="absent")
    ax.barh(y + 0.2, d["nonsusc_rate_present"], 0.4, color=RED, label="present")
    ax.set_yticks(y)
    ax.set_yticklabels([w[:46] for w in d["description"]], fontsize=8)
    ax.set_xlabel("Non-susceptibility rate (training partition)")
    ax.set_title("History item absent vs present")
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    ax.barh(y, d["relative_risk"], color=PURPLE)
    ax.set_yticks(y)
    ax.set_yticklabels([w[:46] for w in d["description"]], fontsize=8)
    ax.axvline(1.0, color="#111", ls="--", lw=1.3, label="no association")
    ax.set_xlabel("Relative risk (present ÷ absent)")
    ax.set_title("Strength of the univariate association")
    for i, v in enumerate(d["relative_risk"]):
        ax.text(v, i, f" {v:.2f}x", va="center", fontsize=8.5)
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    v = df["prior_res_n_events_1095d"].fillna(0)
    v = v[v > 0]
    ax.hist(v[v <= 60], bins=30, color=RED, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Documented resistant results in the previous 3 years")
    ax.set_ylabel("Rows (log scale)")
    ax.set_title("How much prior resistance a patient carries")

    ax = axes[1, 1]
    ds = df["prior_res_same_org_abx_days_since"].dropna()
    if len(ds):
        ax.hist(ds[ds <= 1825], bins=50, color=AMBER, edgecolor="white")
        ax.set_xlabel("Days since prior resistance to this organism+antibiotic")
        ax.set_ylabel("Rows")
        ax.set_title(f"Recency of the most specific history signal\n"
                     f"(available for {100*len(ds)/len(df):.1f}% of rows)")
    fig.suptitle("Previous resistance history — the strongest non-organism "
                 "predictor in this dataset", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "07_prior_resistance_history.png")

    REPORT["prior_resistance"] = {
        "associations": rows,
        "pct_rows_with_any_prior_resistance_record": round(
            100 * float((df["has_prior_resistance_history"] == 1).mean()), 3),
        "note": ("These are univariate associations computed before any model "
                 "was fitted, which matters: it shows the signal the model "
                 "later exploits is present in the data rather than "
                 "manufactured by the learner. The most specific feature — "
                 "prior resistance to this exact organism-antibiotic pair — "
                 "carries the largest relative risk but is available for only a "
                 "small minority of rows."),
    }
    for r in rows:
        log(f"   {r['description'][:50]:<52} {r['nonsusc_rate_absent']:.3f} -> "
            f"{r['nonsusc_rate_present']:.3f}  (RR {r['relative_risk']})")


# ===========================================================================
def section_11_temporal(df, tr):
    log("\n11. TEMPORAL ANALYSIS")
    yearly = (tr.groupby("order_year", observed=True)
              .agg(n_rows=("order_year", "size"),
                   n_orders=("order_proc_id_coded", "nunique"),
                   nonsusc_rate=(TARGET_BINARY, "mean")))
    yearly = yearly[yearly["n_rows"] >= 500]
    savetable(yearly.reset_index(), "11_temporal_trend.csv")

    top_classes = (tr["antibiotic_class"].value_counts().head(5).index.tolist())
    by_cls = (tr[tr["antibiotic_class"].isin(top_classes)]
              .groupby(["order_year", "antibiotic_class"], observed=True)
              [TARGET_BINARY].agg(["mean", "size"]).reset_index())
    by_cls = by_cls[by_cls["size"] >= 200]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    axes[0].plot(yearly.index, yearly["nonsusc_rate"], "o-", color=RED, lw=2.2)
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Non-susceptibility rate")
    axes[0].set_title("Non-susceptibility over time")
    axes[0].grid(True, alpha=0.35)

    axes[1].bar(yearly.index, yearly["n_rows"], color=BLUE)
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Tested pairs")
    axes[1].set_title("Cohort volume over time")

    for cls in top_classes:
        d = by_cls[by_cls["antibiotic_class"] == cls]
        if len(d) > 3:
            axes[2].plot(d["order_year"], d["mean"], "o-", lw=1.7, ms=3.5,
                         label=cls)
    axes[2].set_xlabel("Year"); axes[2].set_ylabel("Non-susceptibility rate")
    axes[2].set_title("By antibiotic class (5 most-tested)")
    axes[2].legend(fontsize=7.5)
    axes[2].grid(True, alpha=0.35)

    fig.suptitle("Temporal trend — ONE US academic health system, training "
                 "partition only.\nThese trends describe this cohort and must "
                 "not be read as national or population-wide resistance rates.",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "08_temporal_resistance_trend.png")

    REPORT["temporal"] = {
        "by_year": yearly.round(5).reset_index().to_dict("records"),
        "year_min": int(yearly.index.min()),
        "year_max": int(yearly.index.max()),
        "rate_first_year": round(float(yearly["nonsusc_rate"].iloc[0]), 5),
        "rate_last_year": round(float(yearly["nonsusc_rate"].iloc[-1]), 5),
        "absolute_change": round(float(yearly["nonsusc_rate"].iloc[-1]
                                       - yearly["nonsusc_rate"].iloc[0]), 5),
        "peak_year": int(yearly["nonsusc_rate"].idxmax()),
        "peak_rate": round(float(yearly["nonsusc_rate"].max()), 5),
        "scope_warning": (
            "This dataset is a single US academic health system. The trend "
            "describes THIS cohort's tested isolates. It is not a national "
            "resistance rate, not a population prevalence, and not "
            "generalisable to other institutions. It is also shaped by changes "
            "in which agents the laboratory chose to test and by breakpoint "
            "revisions during the period."),
        "why_it_matters": (
            "A visible drift is the justification for reporting a temporal "
            "holdout alongside the random patient split: a model fitted on "
            "earlier years faces a shifted distribution in later ones."),
    }
    log(f"   {yearly.index.min()}-{yearly.index.max()}: "
        f"{yearly['nonsusc_rate'].iloc[0]:.1%} -> "
        f"{yearly['nonsusc_rate'].iloc[-1]:.1%}")


# ===========================================================================
def section_12_antibiogram(df, tr, min_tests=50):
    log("\n12. ANTIBIOTIC x ORGANISM ANALYSIS")
    top_org = tr["organism_clean"].value_counts().head(15).index.tolist()
    top_abx = tr["antibiotic"].value_counts().head(20).index.tolist()
    sub = tr[tr["organism_clean"].isin(top_org) & tr["antibiotic"].isin(top_abx)]
    rate = sub.pivot_table(index="organism_clean", columns="antibiotic",
                           values=TARGET_BINARY, aggfunc="mean")
    cnt = sub.pivot_table(index="organism_clean", columns="antibiotic",
                          values=TARGET_BINARY, aggfunc="size")
    rate = rate.reindex(index=top_org, columns=top_abx)
    cnt = cnt.reindex(index=top_org, columns=top_abx)
    masked = rate.mask(cnt.fillna(0) < min_tests)

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(masked, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1,
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "non-susceptibility rate"},
                annot_kws={"fontsize": 7})
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(labelsize=8)
    ax.set_title(
        "Cohort antibiogram — non-susceptibility rate by organism and agent\n"
        f"Training partition. Cells with fewer than {min_tests} tests are left "
        "BLANK rather than shown as if reliable;\nblank therefore also means "
        "the laboratory rarely tests that pair, which is itself informative.",
        fontsize=12, fontweight="bold")
    savefig(fig, "09_antibiotic_organism_heatmap.png")
    savetable(masked.round(4).reset_index(), "12_antibiogram_rates.csv")
    savetable(cnt.fillna(0).astype(int).reset_index(),
              "12b_antibiogram_test_counts.csv")

    # sparsity across the whole space, not just the displayed block
    full_cnt = tr.pivot_table(index="organism_clean", columns="antibiotic",
                              values=TARGET_BINARY, aggfunc="size")
    total_cells = int(full_cnt.shape[0] * full_cnt.shape[1])
    tested = int(full_cnt.notna().sum().sum())
    reliable = int((full_cnt.fillna(0) >= min_tests).sum().sum())

    REPORT["antibiogram"] = {
        "displayed_organisms": top_org,
        "displayed_antibiotics": top_abx,
        "min_tests_to_display": min_tests,
        "displayed_cells": int(masked.size),
        "displayed_cells_shown": int(masked.notna().sum().sum()),
        "displayed_pct_shown": round(
            100 * float(masked.notna().sum().sum() / masked.size), 2),
        "full_space_cells": total_cells,
        "full_space_cells_with_any_test": tested,
        "full_space_pct_with_any_test": round(100 * tested / total_cells, 3),
        "full_space_cells_with_50_plus_tests": reliable,
        "full_space_pct_reliable": round(100 * reliable / total_cells, 3),
        "validation_note": (
            "This figure independently validates the cleaning pipeline. "
            "Textbook intrinsic resistance appears at or near 1.00 exactly "
            "where microbiology says it should: Klebsiella with ampicillin, "
            "Proteus with nitrofurantoin and tetracycline, the AmpC organisms "
            "(Enterobacter, Citrobacter freundii, Klebsiella aerogenes) with "
            "cefazolin, cefoxitin and amoxicillin-clavulanate, Serratia with "
            "nitrofurantoin, and Staphylococcus aureus with penicillin. If "
            "labels or organism names had been mishandled, these patterns would "
            "not survive."),
        "sparsity_note": (
            f"Only {100*tested/total_cells:.1f}% of the full organism x "
            f"antibiotic space has ANY test at all, and only "
            f"{100*reliable/total_cells:.1f}% has {min_tests} or more. This is "
            "panel-selection bias made visible, and it is the central reason a "
            "unified model that shares structure through antibiotic class and "
            "organism group was chosen over independent per-agent models."),
    }
    log(f"   full space: {tested:,}/{total_cells:,} cells tested "
        f"({100*tested/total_cells:.1f}%); {reliable:,} with >={min_tests} tests")


# ===========================================================================
NUMERIC_PANEL = [
    ("median_wbc", "White cell count", "x10^9/L", 0, 40),
    ("median_cr", "Creatinine", "mg/dL", 0, 8),
    ("median_bun", "Urea nitrogen", "mg/dL", 0, 120),
    ("median_hgb", "Haemoglobin", "g/dL", 4, 20),
    ("median_plt", "Platelets", "x10^9/L", 0, 800),
    ("median_na", "Sodium", "mmol/L", 115, 160),
    ("median_lactate", "Lactate", "mmol/L", 0, 12),
    ("median_procalcitonin", "Procalcitonin", "ng/mL", 0, 25),
]

HISTORY_PANEL = [
    ("prior_med_n_drugdays_90d", "Antibiotic drug-days, 90d"),
    ("prior_med_n_drugdays_365d", "Antibiotic drug-days, 365d"),
    ("prior_res_n_events_1095d", "Resistant results, 3y"),
    ("prior_n_cultures", "Previous cultures"),
    ("prior_n_positive_cultures", "Previous positive cultures"),
    ("prior_same_abx_n_tests", "Previous tests of this agent"),
]


def section_13_distributions(df, tr):
    log("\n13. FEATURE DISTRIBUTIONS")
    panel = [p for p in NUMERIC_PANEL if p[0] in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    stats = []
    for ax, (col, label, unit, lo, hi) in zip(axes.ravel(), panel):
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        clipped = v[(v >= lo) & (v <= hi)]
        ax.hist(clipped, bins=60, color=BLUE, edgecolor="white")
        ax.axvline(float(v.median()), color=RED, lw=1.8,
                   label=f"median {v.median():.2f}")
        ax.set_xlabel(f"{label} ({unit})")
        ax.set_ylabel("Rows")
        ax.set_title(f"{label}\n{100*(1-len(v)/len(df)):.0f}% missing",
                     fontsize=10)
        ax.legend(fontsize=7.5)
        q = v.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        stats.append({"feature": col, "description": label, "unit": unit,
                      "n_present": int(len(v)),
                      "pct_missing": round(100 * (1 - len(v) / len(df)), 3),
                      "min": float(v.min()), "p1": float(q.loc[0.01]),
                      "q1": float(q.loc[0.25]), "median": float(q.loc[0.5]),
                      "q3": float(q.loc[0.75]), "p99": float(q.loc[0.99]),
                      "max": float(v.max()), "mean": round(float(v.mean()), 4),
                      "std": round(float(v.std()), 4),
                      "pct_outside_display_range": round(
                          100 * float(((v < lo) | (v > hi)).mean()), 4)})
    fig.suptitle("Laboratory value distributions — full analytic dataset\n"
                 "x-axis clipped to a physiologically plausible display range; "
                 "no value was removed from the data",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "11_laboratory_distributions.png")
    savetable(pd.DataFrame(stats), "13_numeric_feature_summary.csv")

    hpanel = [p for p in HISTORY_PANEL if p[0] in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    hstats = []
    for ax, (col, label) in zip(axes.ravel(), hpanel):
        v = df[col].fillna(0)
        pos = v[v > 0]
        cap = float(pos.quantile(0.99)) if len(pos) else 1.0
        ax.hist(pos[pos <= cap], bins=40, color=PURPLE, edgecolor="white")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Rows (log scale)")
        ax.set_title(f"{label}\n{100*float((v > 0).mean()):.1f}% of rows > 0",
                     fontsize=10)
        hstats.append({"feature": col, "description": label,
                       "pct_rows_zero": round(100 * float((v == 0).mean()), 3),
                       "mean_when_positive": round(float(pos.mean()), 4)
                       if len(pos) else None,
                       "median_when_positive": float(pos.median()) if len(pos) else None,
                       "p99_when_positive": round(cap, 3),
                       "max": float(v.max())})
    fig.suptitle("Patient-history count distributions — heavily zero-inflated "
                 "and long-tailed\n(log y-axis; zeros excluded from the "
                 "histogram, reported in each title)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "12_history_feature_distributions.png")
    savetable(pd.DataFrame(hstats), "13b_history_feature_summary.csv")

    REPORT["distributions"] = {
        "laboratory": stats,
        "history": hstats,
        "note": ("The laboratory values are right-skewed with long tails, which "
                 "is expected for inflammatory and renal markers in an infected "
                 "population. The history counts are heavily zero-inflated: for "
                 "most rows the patient has no recorded prior event at all. "
                 "Neither shape is a problem for gradient-boosted trees, which "
                 "split on rank rather than magnitude — which is also why no "
                 "log transform or scaling was applied for the tree models."),
    }
    log(f"   summarised {len(stats)} laboratory and {len(hstats)} history features")


# ===========================================================================
def section_14_correlation(df):
    log("\n14. CORRELATION ANALYSIS")
    lab_cols = [c for c, *_ in NUMERIC_PANEL if c in df.columns]
    hist_cols = [c for c, _ in HISTORY_PANEL if c in df.columns]
    extra = [c for c in ("prior_med_n_broadspectrum_drugdays_90d",
                         "prior_res_same_abx_n", "prior_res_same_org_abx_n",
                         "prior_same_abx_nonsusc_rate",
                         "prior_same_org_nonsusc_rate")
             if c in df.columns]
    cols = lab_cols + hist_cols + extra

    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman", min_periods=2000)

    fig, ax = plt.subplots(figsize=(13.5, 11.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.4,
                linecolor="white", ax=ax, annot_kws={"fontsize": 6.5},
                cbar_kws={"shrink": 0.7, "label": "Spearman rho"})
    ax.tick_params(labelsize=7.5)
    ax.set_title(
        "Correlation among CONTINUOUS features only (Spearman)\n"
        "Laboratory values and patient-history counts. Categorical variables "
        "are deliberately EXCLUDED:\ntheir integer codes are arbitrary labels, "
        "so a correlation computed on them would be meaningless.",
        fontsize=11.5, fontweight="bold")
    savefig(fig, "10_correlation_heatmap.png")
    savetable(corr.round(4).reset_index().rename(columns={"index": "feature"}),
              "14_correlation_matrix.csv")

    upper = corr.where(mask.T)
    pairs = (upper.stack().rename("spearman_rho").reset_index()
             .rename(columns={"level_0": "feature_a", "level_1": "feature_b"}))
    pairs["abs"] = pairs["spearman_rho"].abs()
    strong = pairs[pairs["abs"] > 0.6].sort_values("abs", ascending=False)
    savetable(strong.drop(columns="abs").round(4),
              "14b_strongly_correlated_pairs.csv")

    REPORT["correlation"] = {
        "features_included": cols,
        "method": "Spearman rank correlation",
        "why_these_features": (
            "Correlation is only meaningful between variables with a genuine "
            "ordering. The continuous laboratory values and the ordered "
            "patient-history counts qualify. The nine categorical features "
            "(organism, antibiotic, class, specimen, morphotype and so on) do "
            "NOT: their integer codes are arbitrary labels assigned by the "
            "encoder, so a Pearson or Spearman coefficient computed on them "
            "would be an artefact of the encoding order and is not reported. "
            "Spearman rather than Pearson because every one of these features "
            "is skewed or zero-inflated."),
        "n_strongly_correlated_pairs_above_0.6": int(len(strong)),
        "strongest_pairs": strong.head(12).drop(columns="abs").round(4)
        .to_dict("records"),
        "consequence": (
            "The redundancy visible here is what the feature-selection "
            "correlation filter acted on: 35 features were removed at "
            "|rho| > 0.95. The moderately-correlated pairs that remain are the "
            "reason a single SHAP attribution within a correlated group should "
            "not be read in isolation — see the XAI report."),
    }
    log(f"   {len(cols)} continuous features; "
        f"{len(strong)} pairs with |rho| > 0.6")


# ===========================================================================
# Physiologically possible range for each analyte, used ONLY to count and
# report data-entry artefacts. Nothing is removed on the basis of these bounds;
# they exist so the report can state how much of the tail is real biology and
# how much is a unit or transcription error.
PLAUSIBLE_RANGE = {
    "median_wbc": (0.0, 500.0),          # extreme leukaemic counts do reach 500
    "median_cr": (0.0, 30.0),
    "median_bun": (0.0, 300.0),
    # The *_hgb quartile columns are exported in mg/dL, not g/dL: the median
    # is 11,600 (= 11.6 g/dL) and the 1st-99th percentile span 6,900-15,800.
    # Bounding them on a g/dL scale would flag every value as impossible,
    # which is a unit error in the ANALYSIS, not in the data.
    "median_hgb": (1000.0, 30000.0),     # mg/dL
    "median_plt": (0.0, 3000.0),
    "median_na": (90.0, 200.0),
    "median_lactate": (0.0, 40.0),
    "median_procalcitonin": (0.0, 1000.0),
}


def section_15_outliers(df):
    log("\n15. OUTLIER ANALYSIS")
    rows = []
    for col, label, unit, lo, hi in NUMERIC_PANEL:
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(v) < 1000:
            continue
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = (v < lo_f) | (v > hi_f)
        p_lo, p_hi = PLAUSIBLE_RANGE.get(col, (-np.inf, np.inf))
        implausible = (v < p_lo) | (v > p_hi)
        rows.append({
            "feature": col, "description": label, "unit": unit,
            "n_present": int(len(v)),
            "iqr_lower_fence": round(float(lo_f), 4),
            "iqr_upper_fence": round(float(hi_f), 4),
            "n_iqr_outliers": int(out.sum()),
            "pct_iqr_outliers": round(100 * float(out.mean()), 4),
            "max_observed": float(v.max()),
            "plausible_range": f"{p_lo:g} - {p_hi:g}",
            "n_physiologically_implausible": int(implausible.sum()),
            "pct_physiologically_implausible": round(
                100 * float(implausible.mean()), 6),
        })
    out_df = pd.DataFrame(rows)
    savetable(out_df, "15_outlier_analysis.csv")
    n_impl = int(out_df["n_physiologically_implausible"].sum())
    affected = out_df[out_df["n_physiologically_implausible"] > 0]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    data, labels = [], []
    for col, label, unit, lo, hi in NUMERIC_PANEL:
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(v) < 1000:
            continue
        # standardise so the panel is comparable on one axis
        data.append(((v - v.median()) / (v.quantile(0.75) - v.quantile(0.25) or 1))
                    .clip(-8, 25))
        labels.append(f"{label}\n({unit})")
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True,
                    flierprops={"marker": ".", "markersize": 1.5,
                                "alpha": 0.25, "markerfacecolor": RED,
                                "markeredgecolor": "none"})
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.55)
    ax.axhline(0, color="#111", lw=1)
    ax.set_ylabel("Robust z-score  (value − median) / IQR")
    ax.tick_params(labelsize=8)
    ax.set_title("Outlier structure of the laboratory values\n"
                 "Robust standardisation so all analytes share one axis. "
                 "Every point shown was RETAINED — see the note below.",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "13_outlier_boxplots.png")

    REPORT["outliers"] = {
        "by_feature": rows,
        "method": ("Tukey IQR fences (Q1 - 1.5xIQR, Q3 + 1.5xIQR) for the "
                   "statistical tail, plus an explicit per-analyte "
                   "physiological range check for data-entry artefacts."),
        "action_taken": "RETAINED — nothing was removed, transformed or capped.",
        "n_physiologically_implausible_total": n_impl,
        "affected_analytes": affected[
            ["feature", "description", "max_observed", "plausible_range",
             "n_physiologically_implausible",
             "pct_physiologically_implausible"]].to_dict("records"),
        "justification": (
            "Two separate populations sit in these tails and they were treated "
            "differently in the reporting, not in the data.\n\n"
            "(1) STATISTICAL OUTLIERS — values beyond the IQR fences but "
            "physiologically real. In an infected inpatient population these "
            "are the clinical signal, not noise: a creatinine of 8 mg/dL or a "
            "lactate of 12 mmol/L identifies exactly the severely unwell "
            "patients whose resistance risk differs most. Trimming them would "
            "delete the informative tail, so they are kept.\n\n"
            "(2) PHYSIOLOGICALLY IMPOSSIBLE VALUES — checked explicitly against "
            f"a per-analyte range. {n_impl} values across {len(affected)} "
            "analyte(s) fall outside any possible human range.\n\n"
            "One finding is worth recording because it nearly produced a false "
            "alarm. The `*_hgb` columns are exported in **mg/dL, not g/dL**: "
            "the median is 11,600 and the 1st-99th percentile span is "
            "6,900-15,800, i.e. 11.6 g/dL and 6.9-15.8 g/dL. Bounded on a g/dL "
            "scale, every one of the 706,738 haemoglobin values would have been "
            "flagged as impossible. The error was in the analysis assumption, "
            "not in the data, and the range check was corrected rather than the "
            "data being 'cleaned'. Units in this export are not uniform across "
            "analytes and should be confirmed per column before any laboratory "
            "feature is used."),
        "why_retained_anyway": (
            "Three reasons, in order of importance. (a) NO laboratory feature "
            "survived feature selection, so not one of these values reaches the "
            "deployed model — the practical impact on results is exactly zero. "
            "(b) The model family is gradient-boosted trees, which split on "
            "rank rather than magnitude, so an impossible value sits in the "
            "same terminal bin as a merely extreme one and changes nothing. "
            "(c) Deleting rows on the basis of one analyte would discard the "
            "other, valid measurements on the same row, and imputing them would "
            "manufacture data. Reporting the count is the honest option.\n\n"
            "If a future version DID use laboratory features, these values "
            "would need to be nulled (not the rows dropped) before training. "
            "That is recorded as a limitation, not left as a silent assumption."),
        "note_on_display": (
            "The histogram x-axes in figure 11 are clipped to a plausible "
            "display range so the bulk of each distribution is visible. That is "
            "a plotting choice only; the underlying data is unclipped and every "
            "row reaches the pipeline."),
    }
    for r in rows:
        log(f"   {r['description'][:22]:<24} IQR outliers "
            f"{r['pct_iqr_outliers']:>6.2f}%  max {r['max_observed']:>9.1f}  "
            f"implausible {r['n_physiologically_implausible']}")
    log(f"   -> {n_impl} physiologically impossible values across "
        f"{len(affected)} analyte(s); retained, and documented as a limitation")


# ===========================================================================
def main() -> None:
    audit = json.loads((ANALYSIS_DIR / "reports" / "dataset_audit.json")
                       .read_text(encoding="utf-8"))
    build = json.loads((PROCESSED_DIR / "build_report.json")
                       .read_text(encoding="utf-8"))

    df = load_analytic()
    split = patient_grouped_split(df)
    tr = df[split == "train"]
    log(f"Loaded {len(df):,} rows; training partition {len(tr):,} rows "
        f"({tr[PAT].nunique():,} patients)")

    REPORT["scope"] = {
        "structural_analyses_on": "full analytic dataset",
        "target_related_analyses_on": "training partition only",
        "n_rows_full": int(len(df)),
        "n_rows_train": int(len(tr)),
        "rule": ("Describing the data is not a modelling decision, so counts, "
                 "cardinalities, missingness, duplicates and input "
                 "distributions use the full analytic dataset. Anything that "
                 "touches the TARGET uses the training partition only, so that "
                 "nothing seen during exploration can influence a decision "
                 "later evaluated on validation or test data."),
    }

    section_1_overview(df, tr, audit, build)
    section_2_missing(df)
    section_3_duplicates(df, audit, build)
    section_4_target(df, tr)
    section_5_antibiotics(df, tr)
    section_6_organisms(df, tr)
    section_7_specimen(df, tr)
    section_8_demographics(df, tr)
    section_9_prior_exposure(df, tr)
    section_10_prior_resistance(df, tr)
    section_11_temporal(df, tr)
    section_12_antibiogram(df, tr)
    section_13_distributions(df, tr)
    section_14_correlation(df)
    section_15_outliers(df)

    out = ANALYSIS_DIR / "reports" / "eda_report.json"
    out.write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    log(f"\nWrote {out}")
    log(f"Figures: {FIGS}  (copied to {RM_FIGS})")
    log(f"Tables:  {TABLES}  (copied to {RM_TABLES})")


if __name__ == "__main__":
    main()
