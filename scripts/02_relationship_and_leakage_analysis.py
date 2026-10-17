"""PHASE 2 - Dataset relationship analysis + data leakage investigation.

Answers, from the real data only:
  * What is the grain (primary key) of each table?
  * Which columns are the join keys, and are they unique / complete?
  * What is the cardinality of each relationship (1:1, 1:N, N:M)?
  * Which variables are only knowable AFTER the prediction event (leakage)?

Never performs a blind join: every candidate join is validated first and the
validation numbers are written out for the documentation.

Output: 02_DATA_ANALYSIS/reports/relationships.json
        02_DATA_ANALYSIS/reports/leakage.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import ANALYSIS_DIR, NA_VALUES, RAW_FILES  # noqa: E402

OUT = ANALYSIS_DIR / "reports"
OUT.mkdir(parents=True, exist_ok=True)

ORDER_KEY = "order_proc_id_coded"
ENC_KEY = "pat_enc_csn_id_coded"
PAT_KEY = "anon_id"
TRIPLE = [PAT_KEY, ENC_KEY, ORDER_KEY]


def load_keys(path: Path, extra: list | None = None) -> pd.DataFrame:
    """Load only the identifier columns (+ extras) - keeps memory small."""
    cols = TRIPLE + (extra or [])
    return pd.read_csv(path, usecols=cols, na_values=NA_VALUES, dtype=str,
                       low_memory=False)


def grain_report(name: str, df: pd.DataFrame, candidate_grains: list) -> dict:
    rep = {"table": name, "n_rows": len(df), "candidate_grains": []}
    for g in candidate_grains:
        if not all(c in df.columns for c in g):
            continue
        n_unique = len(df[g].drop_duplicates())
        rep["candidate_grains"].append({
            "columns": g,
            "n_unique": int(n_unique),
            "n_rows": len(df),
            "is_unique_key": bool(n_unique == len(df)),
            "avg_rows_per_key": round(len(df) / n_unique, 4),
        })
    return rep


def relationship(a_name, a_keys, b_name, b_keys) -> dict:
    """Validate a candidate join between two key columns. No merge performed."""
    a_set, b_set = set(a_keys.dropna()), set(b_keys.dropna())
    a_dupmax = int(a_keys.value_counts().max()) if len(a_keys) else 0
    b_dupmax = int(b_keys.value_counts().max()) if len(b_keys) else 0
    inter = a_set & b_set
    if a_dupmax == 1 and b_dupmax == 1:
        card = "1:1"
    elif a_dupmax == 1:
        card = "1:N"
    elif b_dupmax == 1:
        card = "N:1"
    else:
        card = "N:M"
    return {
        "left": a_name, "right": b_name,
        "left_n_rows": int(len(a_keys)), "right_n_rows": int(len(b_keys)),
        "left_n_unique_keys": len(a_set), "right_n_unique_keys": len(b_set),
        "left_max_rows_per_key": a_dupmax, "right_max_rows_per_key": b_dupmax,
        "n_keys_in_both": len(inter),
        "pct_left_keys_matched": round(100 * len(inter) / len(a_set), 4) if a_set else None,
        "pct_right_keys_matched": round(100 * len(inter) / len(b_set), 4) if b_set else None,
        "left_only_keys": len(a_set - b_set),
        "right_only_keys": len(b_set - a_set),
        "cardinality": card,
    }


def main() -> None:
    print("Loading identifier columns ...", flush=True)
    cohort = load_keys(RAW_FILES["cohort"],
                       ["organism", "antibiotic", "susceptibility",
                        "was_positive", "culture_description", "ordering_mode",
                        "order_time_jittered_utc"])
    demo = load_keys(RAW_FILES["demographics"], ["age", "gender"])
    labs = load_keys(RAW_FILES["labs"], ["Period_Day"])
    res = load_keys(RAW_FILES["resistance"],
                    ["organism", "antibiotic", "resistant_time_to_culturetime",
                     "order_time_jittered_utc"])
    print("  loaded", flush=True)

    # ---------------- grain -------------------------------------------------
    grains = [
        grain_report("cohort", cohort, [
            [ORDER_KEY], [ORDER_KEY, "organism"],
            [ORDER_KEY, "organism", "antibiotic"], TRIPLE]),
        grain_report("demographics", demo, [[PAT_KEY], [ORDER_KEY], TRIPLE]),
        grain_report("labs", labs, [[ORDER_KEY], [ORDER_KEY, "Period_Day"], TRIPLE]),
        grain_report("resistance", res, [
            [ORDER_KEY], [ORDER_KEY, "organism", "antibiotic"],
            [ORDER_KEY, "organism", "antibiotic",
             "resistant_time_to_culturetime"]]),
    ]

    # prior_med is huge -> stream just what is needed
    print("Streaming prior_med keys ...", flush=True)
    pm_order_counts = {}
    pm_rows = 0
    pm_cat = {}
    pm_time_min, pm_time_max = None, None
    for ch in pd.read_csv(RAW_FILES["prior_med"],
                          usecols=[ORDER_KEY, "medication_category",
                                   "medication_time_to_culturetime"],
                          chunksize=2_000_000, na_values=NA_VALUES,
                          dtype={ORDER_KEY: str, "medication_category": str},
                          low_memory=False):
        pm_rows += len(ch)
        for k, v in ch[ORDER_KEY].value_counts().items():
            pm_order_counts[k] = pm_order_counts.get(k, 0) + int(v)
        for k, v in ch["medication_category"].value_counts().items():
            pm_cat[k] = pm_cat.get(k, 0) + int(v)
        t = pd.to_numeric(ch["medication_time_to_culturetime"], errors="coerce")
        if t.notna().any():
            lo, hi = float(t.min()), float(t.max())
            pm_time_min = lo if pm_time_min is None else min(pm_time_min, lo)
            pm_time_max = hi if pm_time_max is None else max(pm_time_max, hi)
        print(f"  prior_med {pm_rows:,}", flush=True)

    grains.append({
        "table": "prior_med", "n_rows": pm_rows,
        "candidate_grains": [{
            "columns": [ORDER_KEY], "n_unique": len(pm_order_counts),
            "n_rows": pm_rows, "is_unique_key": False,
            "avg_rows_per_key": round(pm_rows / len(pm_order_counts), 4),
            "max_rows_per_key": max(pm_order_counts.values()),
        }],
    })

    # ---------------- relationships ----------------------------------------
    pm_keys = pd.Series(
        np.repeat(list(pm_order_counts.keys()), list(pm_order_counts.values())))
    rels = [
        relationship("cohort", cohort[ORDER_KEY], "demographics", demo[ORDER_KEY]),
        relationship("cohort", cohort[ORDER_KEY], "labs", labs[ORDER_KEY]),
        relationship("cohort", cohort[ORDER_KEY], "resistance", res[ORDER_KEY]),
        relationship("cohort", cohort[ORDER_KEY], "prior_med", pm_keys),
    ]
    del pm_keys

    # patient-level fan-out (needed for grouped splitting)
    pat_orders = cohort.groupby(PAT_KEY)[ORDER_KEY].nunique()
    enc_orders = cohort.groupby(ENC_KEY)[ORDER_KEY].nunique()
    patient_structure = {
        "n_patients": int(cohort[PAT_KEY].nunique()),
        "n_encounters": int(cohort[ENC_KEY].nunique()),
        "n_culture_orders": int(cohort[ORDER_KEY].nunique()),
        "n_rows_cohort": int(len(cohort)),
        "orders_per_patient_mean": round(float(pat_orders.mean()), 4),
        "orders_per_patient_max": int(pat_orders.max()),
        "patients_with_multiple_orders": int((pat_orders > 1).sum()),
        "pct_patients_with_multiple_orders":
            round(100 * float((pat_orders > 1).mean()), 4),
        "orders_per_encounter_mean": round(float(enc_orders.mean()), 4),
    }

    json.dump({"grains": grains, "relationships": rels,
               "patient_structure": patient_structure,
               "prior_med_category_counts": pm_cat,
               "prior_med_time_to_culture_range": [pm_time_min, pm_time_max]},
              (OUT / "relationships.json").open("w"), indent=2)
    print("wrote relationships.json", flush=True)

    # ---------------- leakage ----------------------------------------------
    leak = {}

    leak["target_distribution"] = {
        str(k): int(v)
        for k, v in cohort["susceptibility"].value_counts(dropna=False).items()}
    leak["was_positive_vs_target"] = {
        str(k): {str(kk): int(vv) for kk, vv in v.items()}
        for k, v in pd.crosstab(cohort["was_positive"],
                                cohort["susceptibility"].fillna("<NA>")).iterrows()}
    leak["organism_missing_when_was_positive_0"] = int(
        cohort.loc[cohort["was_positive"] == "0", "organism"].isna().sum())
    leak["n_rows_was_positive_0"] = int((cohort["was_positive"] == "0").sum())

    # resistance history timing: is it strictly BEFORE the culture?
    rt = pd.to_numeric(res["resistant_time_to_culturetime"], errors="coerce")
    leak["resistance_time_to_culture"] = {
        "min": float(rt.min()), "max": float(rt.max()),
        "median": float(rt.median()),
        "n_le_zero": int((rt <= 0).sum()),
        "n_missing": int(rt.isna().sum()),
        "quantiles": {str(q): float(rt.quantile(q))
                      for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
    }
    leak["prior_med_time_to_culture"] = {"min": pm_time_min, "max": pm_time_max}
    leak["labs_period_day_values"] = {
        str(k): int(v) for k, v in labs["Period_Day"].value_counts().items()}

    # Does the resistance table contain the CURRENT culture's own result?
    # If so, joining it on (order, organism, antibiotic) is direct label leakage.
    cur = cohort[[ORDER_KEY, "organism", "antibiotic", "susceptibility"]].copy()
    r2 = res[[ORDER_KEY, "organism", "antibiotic"]].drop_duplicates().copy()
    r2["_in_resistance_table"] = 1
    merged = cur.merge(r2, how="left", on=[ORDER_KEY, "organism", "antibiotic"])
    leak["cohort_rows_matching_resistance_same_order_org_abx"] = {
        "n_matched": int(merged["_in_resistance_table"].notna().sum()),
        "n_total": int(len(merged)),
        "pct": round(100 * float(merged["_in_resistance_table"].notna().mean()), 4),
        "target_breakdown_of_matched": {
            str(k): int(v) for k, v in
            merged.loc[merged["_in_resistance_table"].notna(), "susceptibility"]
            .value_counts(dropna=False).items()},
        "target_breakdown_of_unmatched": {
            str(k): int(v) for k, v in
            merged.loc[merged["_in_resistance_table"].isna(), "susceptibility"]
            .value_counts(dropna=False).items()},
    }

    # order_time consistency between cohort and resistance table
    ct = cohort[[ORDER_KEY, "order_time_jittered_utc"]].drop_duplicates(ORDER_KEY)
    rtm = res[[ORDER_KEY, "order_time_jittered_utc"]].drop_duplicates(ORDER_KEY)
    chk = ct.merge(rtm, on=ORDER_KEY, suffixes=("_cohort", "_res"))
    leak["order_time_matches_between_cohort_and_resistance"] = {
        "n_compared": int(len(chk)),
        "n_identical": int((chk["order_time_jittered_utc_cohort"]
                            == chk["order_time_jittered_utc_res"]).sum()),
    }

    json.dump(leak, (OUT / "leakage.json").open("w"), indent=2)
    print("wrote leakage.json", flush=True)


if __name__ == "__main__":
    main()
