"""PHASES 3 + 5 — data cleaning and feature engineering.

Builds the model-ready analytic table from the five raw files. Every decision
made here is driven by a measured fact from the Phase 1/2 audit and is recorded
in `01_DATA/processed/build_report.json` so the documentation and the report can
quote real numbers.

Grain of the output: one row per (culture order x organism x antibiotic tested).

Outputs
  01_DATA/interim/cohort_clean.parquet        cleaned labels + context
  01_DATA/interim/labs_by_order.parquet       labs collapsed to one row/order
  01_DATA/interim/prior_med_by_order.parquet  prior-exposure features per order
  01_DATA/interim/prior_res_*.parquet         prior-resistance features
  01_DATA/processed/analytic_dataset.parquet  final feature matrix
  01_DATA/processed/build_report.json         every count, every decision

STRICT TEMPORAL RULE
Every history feature is derived only from events that occurred STRICTLY BEFORE
the index culture (`*_time_to_culturetime >= 1` day, or an earlier
`order_time_jittered_utc`). Same-day (`== 0`) medication rows are excluded
because a same-day administration cannot be distinguished from empiric therapy
started immediately after the culture was drawn.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (INTERIM_DIR, NA_VALUES, PROCESSED_DIR,  # noqa: E402
                          RAW_FILES, ensure_dirs)
from src.preprocessing.antibiotic_taxonomy import (  # noqa: E402
    ANTIBIOTIC_CLASS, antibiotic_class, antibiotic_subclass, aware_category,
    coverage_report as abx_coverage, is_oral, is_urinary_only)
from src.preprocessing.medication_mapping import (  # noqa: E402
    BROAD_SPECTRUM_CLASSES, MEDICATION_MAP,
    coverage_report as med_coverage, medication_class, medication_ingredient)
from src.preprocessing.organism_normalizer import parse_organism  # noqa: E402

ORDER = "order_proc_id_coded"
PAT = "anon_id"
ENC = "pat_enc_csn_id_coded"

# Susceptibility categories that carry a usable resistance phenotype.
SEVERITY = {"Susceptible": 0, "Intermediate": 1, "Resistant": 2}
# Excluded outright: not a susceptibility phenotype.
NON_PHENOTYPE = {"Inconclusive", "Synergism"}

# Exposure look-back windows in days.
WINDOWS = (30, 90, 180, 365)
RES_WINDOWS = (90, 365, 1095)

REPORT: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# STEP 1 — cohort: labels, organism cleaning, deduplication
# ---------------------------------------------------------------------------
def build_cohort() -> tuple:
    log("STEP 1  cohort ...")
    df = pd.read_csv(RAW_FILES["cohort"], na_values=NA_VALUES, low_memory=False,
                     dtype={PAT: str, ENC: str, ORDER: str})
    rep = {"n_rows_raw": len(df)}

    df["order_time"] = pd.to_datetime(df["order_time_jittered_utc"],
                                      errors="coerce", utc=True)
    rep["n_order_time_unparsed"] = int(df["order_time"].isna().sum())

    # Order-level census of EVERY culture, positive and negative. Prior-culture
    # history must be counted over all cultures a patient has had, not only the
    # positive ones that survive into the modelling table - a patient with ten
    # prior negative urine cultures is clinically different from one with none.
    all_orders = (df.groupby([PAT, ORDER], observed=True)
                  .agg(order_time=("order_time", "first"),
                       was_positive=("was_positive", "max"))
                  .reset_index())
    all_orders["was_positive"] = pd.to_numeric(
        all_orders["was_positive"], errors="coerce").fillna(0).astype("int8")
    rep["n_all_culture_orders"] = len(all_orders)
    rep["n_positive_culture_orders"] = int(all_orders["was_positive"].sum())

    # -- 1a. drop unlabelled rows (negative cultures) ------------------------
    n_before = len(df)
    df = df[df["susceptibility"].notna()].copy()
    rep["n_dropped_no_label_negative_culture"] = n_before - len(df)

    # -- 1b. drop non-phenotype result categories ----------------------------
    n_before = len(df)
    df = df[~df["susceptibility"].isin(NON_PHENOTYPE)].copy()
    rep["n_dropped_non_phenotype_result"] = n_before - len(df)
    rep["non_phenotype_categories_removed"] = sorted(NON_PHENOTYPE)

    # -- 1c. was_positive is now constant -> zero-variance, excluded ---------
    rep["was_positive_unique_after_filter"] = \
        df["was_positive"].astype(str).unique().tolist()

    # -- 1d. organism normalisation + leakage extraction ---------------------
    log("  normalising organism names ...")
    distinct = df["organism"].dropna().unique()
    table = {v: parse_organism(v) for v in distinct}
    parsed = pd.DataFrame([table[v] for v in df["organism"]], index=df.index)
    df = pd.concat([df, parsed], axis=1)

    rep["organism_raw_distinct"] = int(len(distinct))
    rep["organism_clean_distinct"] = int(df["organism_clean"].nunique())
    rep["organism_leaked_phenotype_rows"] = int(df["leaked_phenotype"].notna().sum())
    rep["organism_leaked_phenotype_breakdown"] = {
        str(k): int(v) for k, v in
        df["leaked_phenotype"].value_counts(dropna=True).items()}

    # Quantify the leak BEFORE dropping it, so the leakage doc can cite it.
    leak_rows = df[df["leaked_phenotype"].notna()]
    rep["organism_leak_evidence"] = {}
    for tag, cls in (("CARBAPENEM_RESISTANT", "Carbapenem"),
                     ("MRSA", "Cephalosporin"),
                     ("VANCOMYCIN_RESISTANT", "Glycopeptide")):
        sub = leak_rows[leak_rows["leaked_phenotype"].astype(str).str.contains(tag)]
        sub = sub[sub["antibiotic"].map(antibiotic_class) == cls]
        base = df[df["antibiotic"].map(antibiotic_class) == cls]
        if len(sub):
            rep["organism_leak_evidence"][f"{tag}_vs_{cls}"] = {
                "n_rows_with_token": int(len(sub)),
                "pct_resistant_when_token_present":
                    round(100 * float((sub["susceptibility"] == "Resistant").mean()), 2),
                "pct_resistant_overall_for_class":
                    round(100 * float((base["susceptibility"] == "Resistant").mean()), 2),
            }
    df = df.drop(columns=["leaked_phenotype"])

    # -- 1e. antibiotic taxonomy ---------------------------------------------
    rep["antibiotic_taxonomy_coverage"] = abx_coverage(df["antibiotic"].unique())
    df["antibiotic_class"] = df["antibiotic"].map(antibiotic_class)
    df["antibiotic_subclass"] = df["antibiotic"].map(antibiotic_subclass)
    df["antibiotic_aware"] = df["antibiotic"].map(aware_category)
    df["antibiotic_is_oral"] = df["antibiotic"].map(is_oral).astype("int8")
    df["antibiotic_is_urinary_only"] = \
        df["antibiotic"].map(is_urinary_only).astype("int8")

    # -- 1f. deduplicate (order, organism_clean, antibiotic) -----------------
    # Every duplicated key in this dataset carries CONFLICTING results (8,882 of
    # 8,882 groups). Resolution rule: most-resistant-wins (R > I > S), the
    # conservative clinical reading when one specimen yields two strains of the
    # same species with different susceptibility.
    df["severity"] = df["susceptibility"].map(SEVERITY).astype("int8")
    key = [ORDER, "organism_clean", "antibiotic"]
    n_before = len(df)
    dup_mask = df.duplicated(key, keep=False)
    rep["n_rows_in_duplicate_key_groups"] = int(dup_mask.sum())
    rep["n_duplicate_key_groups"] = int(df.loc[dup_mask, key].drop_duplicates().shape[0])
    conflict = df[dup_mask].groupby(key, observed=True)["severity"].nunique()
    rep["n_duplicate_groups_with_conflicting_labels"] = int((conflict > 1).sum())

    df = (df.sort_values("severity", ascending=False)
            .drop_duplicates(key, keep="first")
            .sort_index())
    rep["n_dropped_duplicate_resolution"] = n_before - len(df)
    rep["dedup_rule"] = "most-resistant-wins (Resistant > Intermediate > Susceptible)"

    # -- 1g. targets ----------------------------------------------------------
    df["y_nonsusceptible"] = (df["severity"] >= 1).astype("int8")
    df["y_3class"] = df["susceptibility"]
    rep["target_binary_distribution"] = {
        "susceptible": int((df["y_nonsusceptible"] == 0).sum()),
        "non_susceptible": int((df["y_nonsusceptible"] == 1).sum()),
    }
    rep["target_binary_positive_rate"] = round(
        float(df["y_nonsusceptible"].mean()), 6)
    rep["target_3class_distribution"] = {
        str(k): int(v) for k, v in df["y_3class"].value_counts().items()}

    # -- 1h. temporal context features ---------------------------------------
    df["order_year"] = df["order_time"].dt.year.astype("int16")
    df["order_month"] = df["order_time"].dt.month.astype("int8")
    df["order_dayofweek"] = df["order_time"].dt.dayofweek.astype("int8")
    df["order_hour"] = df["order_time"].dt.hour.astype("int8")

    keep = [PAT, ENC, ORDER, "order_time", "order_year", "order_month",
            "order_dayofweek", "order_hour", "ordering_mode",
            "culture_description", "organism_clean", "organism_genus",
            "organism_group", "organism_morphotype", "is_species_level",
            "antibiotic", "antibiotic_class", "antibiotic_subclass",
            "antibiotic_aware", "antibiotic_is_oral",
            "antibiotic_is_urinary_only", "susceptibility", "severity",
            "y_nonsusceptible", "y_3class"]
    df = df[keep]

    rep["n_rows_clean"] = len(df)
    rep["n_orders_clean"] = int(df[ORDER].nunique())
    rep["n_patients_clean"] = int(df[PAT].nunique())
    rep["date_range"] = [str(df["order_time"].min()), str(df["order_time"].max())]
    REPORT["step1_cohort"] = rep
    log(f"  -> {len(df):,} labelled rows, {rep['n_orders_clean']:,} orders "
        f"(of {rep['n_all_culture_orders']:,} cultures; "
        f"{rep['n_all_culture_orders'] - rep['n_positive_culture_orders']:,} "
        f"were negative), {rep['n_patients_clean']:,} patients")
    df.to_parquet(INTERIM_DIR / "cohort_clean.parquet", index=False)
    return df, all_orders


# ---------------------------------------------------------------------------
# STEP 2 — demographics (validated 1:1 join)
# ---------------------------------------------------------------------------
AGE_ORDER = ["18-24 years", "25-34 years", "35-44 years", "45-54 years",
             "55-64 years", "65-74 years", "75-84 years", "85-89 years",
             "above 90"]
AGE_MIDPOINT = {"18-24 years": 21, "25-34 years": 29.5, "35-44 years": 39.5,
                "45-54 years": 49.5, "55-64 years": 59.5, "65-74 years": 69.5,
                "75-84 years": 79.5, "85-89 years": 87, "above 90": 93}


def build_demographics(order_ids: set) -> pd.DataFrame:
    log("STEP 2  demographics ...")
    d = pd.read_csv(RAW_FILES["demographics"], na_values=NA_VALUES,
                    dtype={PAT: str, ENC: str, ORDER: str})
    rep = {"n_rows": len(d), "n_unique_orders": int(d[ORDER].nunique())}
    rep["is_one_row_per_order"] = bool(d[ORDER].is_unique)
    d = d[[ORDER, "age", "gender"]].copy()
    d["age_band_ordinal"] = d["age"].map(
        {v: i for i, v in enumerate(AGE_ORDER)}).astype("float32")
    d["age_midpoint_years"] = d["age"].map(AGE_MIDPOINT).astype("float32")
    d["gender_code"] = pd.to_numeric(d["gender"], errors="coerce").astype("float32")
    rep["age_band_values"] = {str(k): int(v) for k, v in d["age"].value_counts().items()}
    rep["gender_code_values"] = {str(k): int(v)
                                 for k, v in d["gender"].value_counts(dropna=False).items()}
    rep["gender_note"] = (
        "gender is de-identified to the codes 0 and 1; the source provides no "
        "legend, so the mapping to sex is NOT known and is never asserted. The "
        "column is used as an anonymous binary covariate named gender_code.")
    rep["n_missing_gender"] = int(d["gender_code"].isna().sum())
    rep["coverage_of_clean_orders"] = round(
        100 * len(set(d[ORDER]) & order_ids) / len(order_ids), 4)
    REPORT["step2_demographics"] = rep
    return d[[ORDER, "age", "age_band_ordinal", "age_midpoint_years", "gender_code"]]


# ---------------------------------------------------------------------------
# STEP 3 — labs collapsed to one row per order
# ---------------------------------------------------------------------------
def build_labs(order_ids: set) -> pd.DataFrame:
    log("STEP 3  labs ...")
    lab = pd.read_csv(RAW_FILES["labs"], na_values=NA_VALUES,
                      dtype={PAT: str, ENC: str, ORDER: str}, low_memory=False)
    rep = {"n_rows": len(lab), "n_unique_orders": int(lab[ORDER].nunique())}

    # Period_Day is constant (14) -> zero variance. It documents the look-back
    # window rather than varying across rows, so it is dropped as a feature.
    rep["period_day_values"] = {str(k): int(v)
                                for k, v in lab["Period_Day"].value_counts().items()}
    rep["period_day_dropped_reason"] = (
        "constant value 14 -> zero variance. It defines the lab aggregation "
        "window (14 days before the culture) and is documented, not modelled.")
    lab = lab.drop(columns=["Period_Day", PAT, ENC])

    numeric_cols = [c for c in lab.columns if c != ORDER]
    for c in numeric_cols:
        lab[c] = pd.to_numeric(lab[c], errors="coerce").astype("float32")

    rows_per_order = lab.groupby(ORDER).size()
    rep["rows_per_order"] = {
        "mean": round(float(rows_per_order.mean()), 4),
        "max": int(rows_per_order.max()),
        "n_orders_with_multiple_rows": int((rows_per_order > 1).sum()),
    }
    rep["collapse_rule"] = (
        "The labs export contains 2-3 partially-populated rows per culture "
        "order (max 106) that are not exact duplicates: each row carries a "
        "different subset of the analytes. Rows are collapsed to one row per "
        "order by taking the median of the non-missing values per column, which "
        "coalesces the disjoint subsets and resolves the (rare) disagreements "
        "without inventing a value.")

    log("  collapsing labs to one row per order ...")
    lab = lab.groupby(ORDER, sort=False)[numeric_cols].median().reset_index()
    lab["has_labs"] = np.int8(1)

    rep["n_rows_after_collapse"] = len(lab)
    rep["missingness_pct"] = {
        c: round(100 * float(lab[c].isna().mean()), 3) for c in numeric_cols}
    rep["coverage_of_clean_orders"] = round(
        100 * len(set(lab[ORDER]) & order_ids) / len(order_ids), 4)
    REPORT["step3_labs"] = rep
    log(f"  -> {len(lab):,} orders with labs "
        f"({rep['coverage_of_clean_orders']}% of clean orders)")
    lab.to_parquet(INTERIM_DIR / "labs_by_order.parquet", index=False)
    return lab


# ---------------------------------------------------------------------------
# STEP 4 — prior medication exposure
# ---------------------------------------------------------------------------
def build_prior_med(order_ids: set) -> tuple:
    log("STEP 4  prior medication exposure ...")
    pm = pd.read_csv(RAW_FILES["prior_med"], na_values=NA_VALUES,
                     usecols=[ORDER, "medication_name",
                              "medication_time_to_culturetime"],
                     dtype={ORDER: str, "medication_name": str})
    rep = {"n_rows_raw": len(pm)}
    pm["t"] = pd.to_numeric(pm["medication_time_to_culturetime"],
                            errors="coerce").astype("float32")
    pm = pm.drop(columns=["medication_time_to_culturetime"])

    # -- 4a. exclude same-day administrations (temporal safety) --------------
    n_before = len(pm)
    rep["n_rows_same_day_t_eq_0"] = int((pm["t"] == 0).sum())
    pm = pm[pm["t"] >= 1].copy()
    rep["n_dropped_same_day"] = n_before - len(pm)
    rep["same_day_exclusion_reason"] = (
        "A medication recorded 0 days from the culture cannot be distinguished "
        "from empiric therapy started immediately AFTER the specimen was "
        "collected. Including it would leak the treating clinician's response "
        "to the very episode being predicted, so all t == 0 rows are excluded.")

    # -- 4b. collapse repeated same-day administrations ----------------------
    n_before = len(pm)
    pm = pm.drop_duplicates([ORDER, "medication_name", "t"])
    rep["n_dropped_repeat_same_drug_same_day"] = n_before - len(pm)
    rep["dedup_note"] = (
        "40.3% of the raw prior-medication rows are exact duplicates: the same "
        "drug on the same day for the same order, i.e. repeated administrations "
        "within a day. Exposure is counted in distinct drug-days, so these are "
        "collapsed.")

    # -- 4c. map to ingredient + class ---------------------------------------
    rep["medication_map_coverage"] = med_coverage(pm["medication_name"].unique())
    pm["ingredient"] = pm["medication_name"].map(medication_ingredient)
    pm["med_class"] = pm["medication_name"].map(medication_class)
    pm = pm.drop(columns=["medication_name"])

    classes = sorted({v[1] for v in MEDICATION_MAP.values()})
    rep["medication_classes"] = classes

    # -- 4d. order-level aggregate features ----------------------------------
    log("  aggregating order-level exposure ...")
    feats = pd.DataFrame({ORDER: pm[ORDER].unique()})
    for w in WINDOWS:
        sub = pm[pm["t"] <= w]
        agg = sub.groupby(ORDER).agg(
            **{f"prior_med_n_drugdays_{w}d": ("t", "size"),
               f"prior_med_n_distinct_ingredients_{w}d": ("ingredient", "nunique"),
               f"prior_med_n_distinct_classes_{w}d": ("med_class", "nunique")})
        feats = feats.merge(agg.reset_index(), on=ORDER, how="left")
        bs = sub[sub["med_class"].isin(BROAD_SPECTRUM_CLASSES)]
        bsagg = bs.groupby(ORDER).size().rename(
            f"prior_med_n_broadspectrum_drugdays_{w}d").reset_index()
        feats = feats.merge(bsagg, on=ORDER, how="left")

    dmin = pm.groupby(ORDER)["t"].min().rename("prior_med_days_since_last")
    feats = feats.merge(dmin.reset_index(), on=ORDER, how="left")
    feats["prior_med_total_drugdays"] = feats[ORDER].map(
        pm.groupby(ORDER).size()).astype("float32")

    # -- 4e. per-class exposure (wide, one column per class per window) ------
    log("  building per-class exposure matrix ...")
    class_windows = (90, 365)
    for w in class_windows:
        sub = pm[pm["t"] <= w]
        wide = (sub.groupby([ORDER, "med_class"]).size().unstack(fill_value=0))
        wide.columns = [f"prior_med_class__{c.replace(' ', '_').replace('/', '_')}__{w}d"
                        for c in wide.columns]
        feats = feats.merge(wide.reset_index(), on=ORDER, how="left")

    # days since last exposure per class (used for the same-class row feature)
    last_by_class = (pm.groupby([ORDER, "med_class"])["t"].min()
                       .unstack())
    last_by_class.columns = [f"prior_med_dayssince__{c}" for c in last_by_class.columns]
    last_by_class = last_by_class.reset_index()

    num_cols = [c for c in feats.columns if c != ORDER]
    feats[num_cols] = feats[num_cols].fillna(0).astype("float32")
    feats.loc[feats["prior_med_days_since_last"] == 0,
              "prior_med_days_since_last"] = np.nan
    feats["has_prior_med_history"] = np.int8(1)

    rep["n_orders_with_history"] = len(feats)
    rep["coverage_of_clean_orders"] = round(
        100 * len(set(feats[ORDER]) & order_ids) / len(order_ids), 4)
    rep["n_features"] = len(num_cols) + 1
    REPORT["step4_prior_med"] = rep
    log(f"  -> {len(feats):,} orders, {rep['n_features']} features")
    feats.to_parquet(INTERIM_DIR / "prior_med_by_order.parquet", index=False)
    last_by_class.to_parquet(INTERIM_DIR / "prior_med_dayssince_by_class.parquet",
                             index=False)
    del pm
    gc.collect()
    return feats, last_by_class


# ---------------------------------------------------------------------------
# STEP 5 — prior resistance history
# ---------------------------------------------------------------------------
def build_prior_resistance(order_ids: set) -> dict:
    log("STEP 5  prior resistance history ...")
    res = pd.read_csv(RAW_FILES["resistance"], na_values=NA_VALUES,
                      usecols=[ORDER, "organism", "antibiotic",
                               "resistant_time_to_culturetime"],
                      dtype={ORDER: str})
    rep = {"n_rows_raw": len(res)}
    res["t"] = pd.to_numeric(res["resistant_time_to_culturetime"],
                             errors="coerce").astype("float32")
    res = res.drop(columns=["resistant_time_to_culturetime"])
    rep["t_min"] = float(res["t"].min())
    rep["n_rows_t_le_0"] = int((res["t"] <= 0).sum())
    rep["temporal_safety_note"] = (
        "Every prior-resistance event is at least 1 day before the index "
        "culture (measured minimum = 1.0, zero rows at t <= 0), so this table "
        "is genuine patient history and not the index result. Confirmed "
        "independently: joining it to the cohort on "
        "(order, organism, antibiotic) matches 4.13% of rows and those rows are "
        "63.7% Resistant - high, but far from the 100% that a leaked label "
        "would produce.")
    res = res[res["t"] >= 1].copy()

    distinct = res["organism"].dropna().unique()
    table = {v: parse_organism(v)["organism_clean"] for v in distinct}
    res["organism_clean"] = res["organism"].map(table)
    res["res_class"] = res["antibiotic"].map(antibiotic_class)
    rep["n_antibiotics"] = int(res["antibiotic"].nunique())
    rep["antibiotics"] = sorted(res["antibiotic"].dropna().unique().tolist())

    out = {}

    # -- 5a. order-level summary ---------------------------------------------
    o = pd.DataFrame({ORDER: res[ORDER].unique()})
    for w in RES_WINDOWS:
        sub = res[res["t"] <= w]
        agg = sub.groupby(ORDER).agg(
            **{f"prior_res_n_events_{w}d": ("t", "size"),
               f"prior_res_n_distinct_abx_{w}d": ("antibiotic", "nunique"),
               f"prior_res_n_distinct_classes_{w}d": ("res_class", "nunique"),
               f"prior_res_n_distinct_organisms_{w}d": ("organism_clean", "nunique")})
        o = o.merge(agg.reset_index(), on=ORDER, how="left")
    o["prior_res_n_events_total"] = o[ORDER].map(res.groupby(ORDER).size())
    o["prior_res_days_since_last"] = o[ORDER].map(res.groupby(ORDER)["t"].min())
    o["has_prior_resistance_history"] = np.int8(1)
    numc = [c for c in o.columns if c not in (ORDER, "prior_res_days_since_last",
                                              "has_prior_resistance_history")]
    o[numc] = o[numc].fillna(0).astype("float32")
    out["order"] = o

    # -- 5b. (order, antibiotic) level ---------------------------------------
    oa = res.groupby([ORDER, "antibiotic"]).agg(
        prior_res_same_abx_n=("t", "size"),
        prior_res_same_abx_days_since=("t", "min")).reset_index()
    for w in RES_WINDOWS:
        c = res[res["t"] <= w].groupby([ORDER, "antibiotic"]).size().rename(
            f"prior_res_same_abx_n_{w}d").reset_index()
        oa = oa.merge(c, on=[ORDER, "antibiotic"], how="left")
    out["order_abx"] = oa

    # -- 5c. (order, antibiotic class) level ---------------------------------
    oc = res.groupby([ORDER, "res_class"]).agg(
        prior_res_same_class_n=("t", "size"),
        prior_res_same_class_days_since=("t", "min")).reset_index()
    for w in RES_WINDOWS:
        c = res[res["t"] <= w].groupby([ORDER, "res_class"]).size().rename(
            f"prior_res_same_class_n_{w}d").reset_index()
        oc = oc.merge(c, on=[ORDER, "res_class"], how="left")
    out["order_class"] = oc

    # -- 5d. (order, organism, antibiotic) level -----------------------------
    ooa = res.groupby([ORDER, "organism_clean", "antibiotic"]).agg(
        prior_res_same_org_abx_n=("t", "size"),
        prior_res_same_org_abx_days_since=("t", "min")).reset_index()
    out["order_org_abx"] = ooa

    # -- 5e. (order, organism) level -----------------------------------------
    oo = res.groupby([ORDER, "organism_clean"]).agg(
        prior_res_same_org_n=("t", "size"),
        prior_res_same_org_n_distinct_abx=("antibiotic", "nunique")).reset_index()
    out["order_org"] = oo

    rep["n_orders_with_history"] = int(res[ORDER].nunique())
    rep["coverage_of_clean_orders"] = round(
        100 * len(set(res[ORDER]) & order_ids) / len(order_ids), 4)
    REPORT["step5_prior_resistance"] = rep
    log(f"  -> {rep['n_orders_with_history']:,} orders with prior resistance "
        f"({rep['coverage_of_clean_orders']}% of clean orders)")
    for k, v in out.items():
        v.to_parquet(INTERIM_DIR / f"prior_res_{k}.parquet", index=False)
    del res
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# STEP 6 — prior culture history derived from the cohort itself (time-aware)
# ---------------------------------------------------------------------------
def build_prior_culture_history(cohort: pd.DataFrame,
                                all_orders: pd.DataFrame) -> tuple:
    log("STEP 6  prior culture history (strictly earlier orders only) ...")
    rep = {}

    # --- order level, over ALL cultures (positive and negative) -------------
    per_order = (cohort.groupby(ORDER, observed=True)
                 .agg(n_tests=("y_nonsusceptible", "size"),
                      n_nonsusc=("y_nonsusceptible", "sum")))
    o = all_orders.copy()
    o["n_tests"] = o[ORDER].map(per_order["n_tests"]).fillna(0).astype("float32")
    o["n_nonsusc"] = o[ORDER].map(per_order["n_nonsusc"]).fillna(0).astype("float32")
    o = o.sort_values([PAT, "order_time", ORDER])

    g = o.groupby(PAT, sort=False)
    o["prior_n_cultures"] = g.cumcount().astype("float32")
    o["prior_n_positive_cultures"] = (
        g["was_positive"].cumsum() - o["was_positive"]).astype("float32")
    o["prior_n_tests"] = (g["n_tests"].cumsum() - o["n_tests"]).astype("float32")
    o["prior_n_nonsusc"] = (g["n_nonsusc"].cumsum() - o["n_nonsusc"]).astype("float32")
    o["prior_nonsusc_rate"] = np.where(
        o["prior_n_tests"] > 0, o["prior_n_nonsusc"] / o["prior_n_tests"], np.nan
    ).astype("float32")
    o["prior_positive_culture_rate"] = np.where(
        o["prior_n_cultures"] > 0,
        o["prior_n_positive_cultures"] / o["prior_n_cultures"], np.nan
    ).astype("float32")
    prev_time = g["order_time"].shift(1)
    o["prior_days_since_last_culture"] = (
        (o["order_time"] - prev_time).dt.total_seconds() / 86400).astype("float32")
    o["has_prior_culture"] = (o["prior_n_cultures"] > 0).astype("int8")

    rep["n_all_orders_considered"] = len(o)
    rep["pct_orders_with_prior_culture"] = round(
        100 * float(o.loc[o[ORDER].isin(set(cohort[ORDER])),
                          "has_prior_culture"].mean()), 4)
    rep["method"] = (
        "Per patient, ALL culture orders (positive and negative, 751,075 in "
        "total) are sorted by order_time and every counter is a cumulative sum "
        "SHIFTED to exclude the current order, so no feature can see its own "
        "outcome. Negative cultures are included in the denominator because a "
        "history of repeated negative cultures is clinically informative and "
        "excluding it would bias the prior-culture counts upward. Orders that "
        "share an identical timestamp are ordered by order id, and the current "
        "order is always excluded regardless.")

    order_hist = o[[ORDER, "prior_n_cultures", "prior_n_positive_cultures",
                    "prior_n_tests", "prior_n_nonsusc", "prior_nonsusc_rate",
                    "prior_positive_culture_rate",
                    "prior_days_since_last_culture", "has_prior_culture"]]

    # --- (order, organism) level: has this patient grown this organism before? -
    oo = (cohort.groupby([PAT, ORDER, "organism_clean"], observed=True)
          .agg(order_time=("order_time", "first"),
               n_tests=("y_nonsusceptible", "size"),
               n_nonsusc=("y_nonsusceptible", "sum"))
          .reset_index()
          .sort_values([PAT, "organism_clean", "order_time", ORDER]))
    g2 = oo.groupby([PAT, "organism_clean"], sort=False, observed=True)
    oo["prior_same_org_n_cultures"] = g2.cumcount().astype("float32")
    oo["prior_same_org_n_nonsusc"] = (
        g2["n_nonsusc"].cumsum() - oo["n_nonsusc"]).astype("float32")
    oo["prior_same_org_n_tests"] = (
        g2["n_tests"].cumsum() - oo["n_tests"]).astype("float32")
    oo["prior_same_org_nonsusc_rate"] = np.where(
        oo["prior_same_org_n_tests"] > 0,
        oo["prior_same_org_n_nonsusc"] / oo["prior_same_org_n_tests"], np.nan
    ).astype("float32")
    prev2 = g2["order_time"].shift(1)
    oo["prior_same_org_days_since"] = (
        (oo["order_time"] - prev2).dt.total_seconds() / 86400).astype("float32")

    rep["pct_rows_with_prior_same_organism"] = round(
        100 * float((oo["prior_same_org_n_cultures"] > 0).mean()), 4)

    org_hist = oo[[ORDER, "organism_clean", "prior_same_org_n_cultures",
                   "prior_same_org_n_tests", "prior_same_org_n_nonsusc",
                   "prior_same_org_nonsusc_rate", "prior_same_org_days_since"]]

    # --- (order, antibiotic) level: prior result for this exact agent ---------
    oa = (cohort[[PAT, ORDER, "order_time", "antibiotic", "y_nonsusceptible"]]
          .sort_values([PAT, "antibiotic", "order_time", ORDER]))
    g3 = oa.groupby([PAT, "antibiotic"], sort=False, observed=True)
    oa["prior_same_abx_n_tests"] = g3.cumcount().astype("float32")
    oa["prior_same_abx_n_nonsusc"] = (
        g3["y_nonsusceptible"].cumsum() - oa["y_nonsusceptible"]).astype("float32")
    oa["prior_same_abx_nonsusc_rate"] = np.where(
        oa["prior_same_abx_n_tests"] > 0,
        oa["prior_same_abx_n_nonsusc"] / oa["prior_same_abx_n_tests"], np.nan
    ).astype("float32")
    abx_hist = oa[[ORDER, "antibiotic", "prior_same_abx_n_tests",
                   "prior_same_abx_n_nonsusc", "prior_same_abx_nonsusc_rate"]]
    abx_hist = abx_hist.drop_duplicates([ORDER, "antibiotic"])

    rep["pct_rows_with_prior_same_antibiotic_test"] = round(
        100 * float((oa["prior_same_abx_n_tests"] > 0).mean()), 4)

    REPORT["step6_prior_culture_history"] = rep
    log(f"  -> {rep['pct_orders_with_prior_culture']}% of orders have a prior "
        f"culture; {rep['pct_rows_with_prior_same_antibiotic_test']}% of rows "
        f"have a prior test of the same agent")
    return order_hist, org_hist, abx_hist


# ---------------------------------------------------------------------------
# STEP 7 — assemble, with a validated join at every step
# ---------------------------------------------------------------------------
def validated_merge(left: pd.DataFrame, right: pd.DataFrame, on, how: str,
                    name: str, expect_right_unique: bool = True) -> pd.DataFrame:
    """Merge only after proving the right side cannot fan out the left side."""
    on = [on] if isinstance(on, str) else list(on)
    n_before = len(left)
    right_unique = not right.duplicated(on).any()
    if expect_right_unique and not right_unique:
        raise ValueError(
            f"Refusing to merge '{name}': right side is not unique on {on}. "
            f"{int(right.duplicated(on).sum()):,} duplicate keys would fan out "
            f"the left table.")
    matched = left[on].merge(right[on].drop_duplicates(), on=on, how="inner")
    out = left.merge(right, on=on, how=how)
    if len(out) != n_before:
        raise ValueError(
            f"Merge '{name}' changed the row count: {n_before:,} -> {len(out):,}")
    REPORT.setdefault("step7_joins", []).append({
        "join": name, "keys": on, "how": how,
        "rows_before": n_before, "rows_after": len(out),
        "right_side_unique_on_keys": bool(right_unique),
        "left_rows_matched": int(len(matched)),
        "pct_left_rows_matched": round(100 * len(matched) / n_before, 4),
    })
    log(f"  join {name:<28} {len(matched):>9,}/{n_before:,} matched "
        f"({100*len(matched)/n_before:5.1f}%)")
    return out


def assemble(cohort, demo, labs, pmed, pmed_dayssince, pres,
             order_hist, org_hist, abx_hist) -> pd.DataFrame:
    log("STEP 7  assembling the analytic table ...")
    df = cohort
    df = validated_merge(df, demo, ORDER, "left", "demographics")
    df = validated_merge(df, labs, ORDER, "left", "labs")
    df = validated_merge(df, pmed, ORDER, "left", "prior_med_order")
    df = validated_merge(df, pres["order"], ORDER, "left", "prior_res_order")
    df = validated_merge(df, pres["order_abx"], [ORDER, "antibiotic"], "left",
                         "prior_res_order_abx")
    df = validated_merge(df, pres["order_org"], [ORDER, "organism_clean"], "left",
                         "prior_res_order_org")
    df = validated_merge(df, pres["order_org_abx"],
                         [ORDER, "organism_clean", "antibiotic"], "left",
                         "prior_res_order_org_abx")
    df = validated_merge(df, order_hist, ORDER, "left", "prior_culture_order")
    df = validated_merge(df, org_hist, [ORDER, "organism_clean"], "left",
                         "prior_culture_org")
    df = validated_merge(df, abx_hist, [ORDER, "antibiotic"], "left",
                         "prior_culture_abx")

    # class-level prior resistance must be matched on the row's own antibiotic
    # class, so it is merged on (order, antibiotic_class)
    ocl = pres["order_class"].rename(columns={"res_class": "antibiotic_class"})
    ocl = ocl[ocl["antibiotic_class"].notna()]
    df = validated_merge(df, ocl, [ORDER, "antibiotic_class"], "left",
                         "prior_res_order_class")

    # -- same-class prior MEDICATION exposure (row-level lookup) -------------
    log("  deriving same-class prior medication exposure ...")
    ds = pmed_dayssince.set_index(ORDER)
    ds.columns = [c.replace("prior_med_dayssince__", "") for c in ds.columns]
    aligned = ds.reindex(df[ORDER].to_numpy())
    cls = df["antibiotic_class"].to_numpy()
    col_pos = {c: i for i, c in enumerate(aligned.columns)}
    idx = np.array([col_pos.get(c, -1) for c in cls])
    vals = aligned.to_numpy(dtype="float32")
    rows = np.arange(len(df))
    same = np.full(len(df), np.nan, dtype="float32")
    ok = idx >= 0
    same[ok] = vals[rows[ok], idx[ok]]
    df = df.copy()  # de-fragment before the final column additions
    df["prior_med_same_class_days_since"] = same
    for w in (90, 365):
        df[f"prior_med_same_class_exposed_{w}d"] = (
            (df["prior_med_same_class_days_since"] <= w).astype("float32"))
        df.loc[df["prior_med_same_class_days_since"].isna(),
               f"prior_med_same_class_exposed_{w}d"] = 0.0

    # -- indicator columns and count fills -----------------------------------
    for col, fill in (("has_labs", 0), ("has_prior_med_history", 0),
                      ("has_prior_resistance_history", 0)):
        df[col] = df[col].fillna(fill).astype("int8")

    count_cols = [c for c in df.columns
                  if c.startswith(("prior_res_", "prior_med_"))
                  and ("days_since" not in c)]
    df[count_cols] = df[count_cols].fillna(0).astype("float32")

    REPORT["step7_final"] = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_orders": int(df[ORDER].nunique()),
        "n_patients": int(df[PAT].nunique()),
        "target_positive_rate": round(float(df["y_nonsusceptible"].mean()), 6),
    }
    return df


def main() -> None:
    ensure_dirs()
    cohort, all_orders = build_cohort()
    order_ids = set(cohort[ORDER])

    demo = build_demographics(order_ids)
    labs = build_labs(order_ids)
    pmed, pmed_ds = build_prior_med(order_ids)
    pres = build_prior_resistance(order_ids)
    order_hist, org_hist, abx_hist = build_prior_culture_history(
        cohort, all_orders)

    df = assemble(cohort, demo, labs, pmed, pmed_ds, pres,
                  order_hist, org_hist, abx_hist)

    out = PROCESSED_DIR / "analytic_dataset.parquet"
    df.to_parquet(out, index=False)
    log(f"\nWrote {out}  ({len(df):,} rows x {df.shape[1]} columns)")

    (PROCESSED_DIR / "build_report.json").write_text(
        json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    log(f"Wrote {PROCESSED_DIR / 'build_report.json'}")


if __name__ == "__main__":
    main()
