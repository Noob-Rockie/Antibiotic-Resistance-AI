"""Cell content for the EDA notebook, kept separate so the long code strings
do not have to be escaped inside the main generator."""
from __future__ import annotations

INTRO = """# 03 — Exploratory Data Analysis

**Phase 4.** Complete EDA over the processed analytic dataset.

**Scope rule.** Structural description (counts, missingness, duplicates, input
distributions) uses the FULL analytic dataset; anything involving the TARGET
uses the TRAINING partition only, so nothing seen here can influence a decision
later evaluated on validation or test data.

Run `python scripts/04b_exploratory_analysis.py` first.
Narrative write-up: [`../EDA_REPORT.md`](../EDA_REPORT.md)."""

SETUP = '''EDA = "02_DATA_ANALYSIS"
eda = load(f"{EDA}/reports/eda_report.json")
print(eda["scope"]["rule"])
print()
print(f'full dataset : {eda["scope"]["n_rows_full"]:,} rows')
print(f'training     : {eda["scope"]["n_rows_train"]:,} rows')'''

CELLS = [
    ("## 1. Dataset overview", '''ov = eda["overview"]
pd.DataFrame([{"property": k, "value": v} for k, v in ov.items()
              if not isinstance(v, dict)])'''),

    ("## 2. Missing-value analysis", '''show(f"{EDA}/figures/01_missing_values.png", 1300)
ms = eda["missing"]
print(f'{ms["n_columns_with_missing"]} of {ms["n_columns"]} columns have '
      f'missing data; {ms["n_columns_above_95pct"]} are above 95%')
display(pd.DataFrame(ms["top_20"]).head(15))
print(ms["note"])'''),

    ("## 3. Duplicate analysis", '''dup = eda["duplicates"]
print("Exact duplicate rows in the SOURCE files:")
display(pd.DataFrame(dup["raw_exact_duplicates"]))
print("\\nDuplicate KEYS in the label table:")
for k in ("cohort_duplicate_key_groups",
          "cohort_groups_with_conflicting_labels",
          "rows_removed_by_resolution",
          "duplicates_remaining_in_analytic_table"):
    print(f"  {k:<48} {dup[k]:,}")
print("\\nresolution rule:", dup["resolution_rule"])
print()
print(dup["action_taken"])'''),

    ("### Duplicated COLUMNS — a defect found in the source export",
     '''display(pd.DataFrame(dup["duplicated_laboratory_columns"]))
print(dup["duplicated_columns_finding"])'''),

    ("## 4. Target distribution", '''show(f"{EDA}/figures/02_target_distribution.png", 1300)
tg = eda["target"]
display(pd.Series(tg["three_class"]).to_frame("rows"))
print(f'imbalance ratio       {tg["imbalance_ratio"]} : 1')
print(f'positive rate (full)  {tg["positive_rate_full"]:.4f}')
print(f'positive rate (train) {tg["positive_rate_train"]:.4f}')
print()
print(tg["note"])'''),

    ("## 5. Antibiotic distribution", '''show(f"{EDA}/figures/03_antibiotic_distribution.png", 1400)
ab = eda["antibiotics"]
print(f'{ab["n_antibiotics"]} agents in {ab["n_classes"]} classes')
print(f'below 1,000 rows: {ab["n_agents_below_1000_rows"]}')
print(f'below 100 rows:   {ab["n_agents_below_100_rows"]}')
print(f'most:least ratio: {ab["max_to_min_ratio"]:,} : 1')
display(pd.Series(ab["least_tested"]).to_frame("rows"))'''),

    ("### Why one unified model instead of 54 separate ones",
     'print(ab["why_unified_model"])'),

    ("## 6. Organism distribution", '''show(f"{EDA}/figures/04_organism_distribution.png", 1400)
org = eda["organisms"]
display(pd.Series(org["top_10"]).to_frame("rows"))
display(pd.DataFrame(org["by_group"]).T)
print(org["note"])'''),

    ("## 7. Specimen distribution", '''show(f"{EDA}/figures/05_specimen_distribution.png", 1300)
display(pd.DataFrame(eda["specimen"]["types"]).T)
print(eda["specimen"]["note"])'''),

    ("## 8. Demographic analysis", '''show(f"{EDA}/figures/06b_demographics.png", 1300)
dm = eda["demographics"]
display(pd.DataFrame(dm["age_bands"]).T)
display(pd.DataFrame(dm["sex_code"]).T)
print(dm["constraints"])
print()
print("WARNING:", dm["interpretation_warning"])'''),

    ("## 9. Previous antibiotic exposure",
     '''show(f"{EDA}/figures/06_prior_antibiotic_exposure.png", 1300)
px = eda["prior_exposure"]
display(pd.DataFrame(px["by_window"]))
display(pd.DataFrame(px["by_class_365d"]).head(12))
print(px["note"])'''),

    ("## 10. Previous resistance history\n\nThe key EDA result: this signal is "
     "present in the data **before any model is fitted**.",
     '''show(f"{EDA}/figures/07_prior_resistance_history.png", 1300)
pr = eda["prior_resistance"]
display(pd.DataFrame(pr["associations"])[
    ["description", "pct_rows_present", "nonsusc_rate_absent",
     "nonsusc_rate_present", "relative_risk"]])
print(pr["note"])'''),

    ("## 11. Temporal analysis", '''show(f"{EDA}/figures/08_temporal_resistance_trend.png", 1400)
tm = eda["temporal"]
display(pd.DataFrame(tm["by_year"]))
print(f'{tm["rate_first_year"]:.1%} -> {tm["rate_last_year"]:.1%} '
      f'({tm["absolute_change"]:+.4f})')
print()
print("SCOPE WARNING:", tm["scope_warning"])'''),

    ("## 12. Antibiotic x organism analysis",
     '''show(f"{EDA}/figures/09_antibiotic_organism_heatmap.png", 1500)
bg = eda["antibiogram"]
print(f'full space: {bg["full_space_cells_with_any_test"]:,} of '
      f'{bg["full_space_cells"]:,} cells tested '
      f'({bg["full_space_pct_with_any_test"]}%)')
print()
print(bg["sparsity_note"])
print()
print("VALIDATION:", bg["validation_note"])'''),

    ("## 13. Feature distributions", '''show(f"{EDA}/figures/11_laboratory_distributions.png", 1400)
display(pd.DataFrame(eda["distributions"]["laboratory"]))
show(f"{EDA}/figures/12_history_feature_distributions.png", 1300)
display(pd.DataFrame(eda["distributions"]["history"]))
print(eda["distributions"]["note"])'''),

    ("## 14. Correlation analysis", '''show(f"{EDA}/figures/10_correlation_heatmap.png", 1100)
co = eda["correlation"]
print(co["why_these_features"])
print()
display(pd.DataFrame(co["strongest_pairs"]))
print(co["consequence"])'''),

    ("## 15. Outlier analysis", '''show(f"{EDA}/figures/13_outlier_boxplots.png", 1200)
ol = eda["outliers"]
display(pd.DataFrame(ol["by_feature"]))
print("ACTION:", ol["action_taken"])
print()
print(ol["justification"])
print()
print(ol["why_retained_anyway"])'''),
]

OUTRO = ("---\n\nFull narrative write-up, including implications for feature "
         "engineering and feature selection: "
         "[`../EDA_REPORT.md`](../EDA_REPORT.md)")
