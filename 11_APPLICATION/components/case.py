"""Turn clinician input (or a real held-out case) into a model-ready record.

Two entry paths, both producing the same dict shape:

  * `manual_record(...)` — built from the sidebar form. Anything the clinician
    does not supply is filled from the TRAINING-SET default carried in the model
    bundle, and the list of fields that were filled this way is returned so the
    UI can say so out loud.

  * `sample_case(...)` — a real row from the held-out TEST partition, with its
    true laboratory result available for comparison. This lets the prototype be
    demonstrated on genuine data instead of invented numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dataset import (PAT, TARGET_BINARY, load_analytic,  # noqa: E402
                                patient_grouped_split)

AGE_BANDS = ["18-24 years", "25-34 years", "35-44 years", "45-54 years",
             "55-64 years", "65-74 years", "75-84 years", "85-89 years",
             "above 90"]
AGE_MIDPOINT = {"18-24 years": 21, "25-34 years": 29.5, "35-44 years": 39.5,
                "45-54 years": 49.5, "55-64 years": 59.5, "65-74 years": 69.5,
                "75-84 years": 79.5, "85-89 years": 87, "above 90": 93}
SPECIMENS = ["URINE", "BLOOD", "RESPIRATORY"]
SETTINGS = ["Inpatient", "Outpatient"]

# Patient-history fields the model ACTUALLY uses that a clinician could
# plausibly supply at the bedside, with the label shown in the sidebar.
#
# This list is deliberately written against the FINAL feature set rather than
# the full analytic table. An earlier version enumerated plausible-sounding
# fields (`prior_n_cultures`, `prior_med_n_drugdays_90d` …) that are not among
# the 30 selected features, so the sidebar collected numbers the model never
# saw and the "assumed fields" notice never fired. Anything listed here is
# checked against `bundle.features` at run time by `enterable_history()`.
HISTORY_INPUTS = [
    ("prior_med_n_drugdays_30d",
     "Antibiotic drug-days in the last 30 days", 0, 200, 0),
    ("prior_med_n_drugdays_180d",
     "Antibiotic drug-days in the last 180 days", 0, 800, 0),
    ("prior_med_n_broadspectrum_drugdays_90d",
     "Broad-spectrum antibiotic drug-days, last 90 days", 0, 400, 0),
    ("prior_med_class__Fluoroquinolone__90d",
     "Fluoroquinolone drug-days, last 90 days", 0, 200, 0),
    ("prior_med_class__Folate_pathway_inhibitor__90d",
     "Trimethoprim/sulfonamide drug-days, last 90 days", 0, 200, 0),
    ("prior_res_n_events_1095d",
     "Documented resistant results in the last 3 years", 0, 300, 0),
    ("prior_res_same_abx_n",
     "Prior resistant results for the agent being considered", 0, 100, 0),
    ("prior_same_org_n_tests",
     "Previous susceptibility tests for THIS organism", 0, 300, 0),
]

# Non-history fields the sidebar collects directly.
DIRECT_INPUTS = {
    "age", "age_band_ordinal", "age_midpoint_years", "gender_code",
    "culture_description", "ordering_mode", "organism_clean", "organism_genus",
    "organism_group", "organism_morphotype", "is_species_level",
    "order_year", "order_month", "order_dayofweek", "order_hour",
}


def enterable_history(bundle) -> list:
    """The history inputs that are genuinely features of the deployed model."""
    feats = set(bundle.features)
    return [h for h in HISTORY_INPUTS if h[0] in feats]


def clinician_enterable(bundle) -> set:
    """Every field the sidebar can supply that the model actually consumes."""
    return (DIRECT_INPUTS | {h[0] for h in enterable_history(bundle)}) & \
        (set(bundle.features) | DIRECT_INPUTS)


def organism_group_for(name: str) -> str:
    from src.preprocessing.organism_normalizer import parse_organism
    return parse_organism(name)["organism_group"] or "Other_Unclassified"


def manual_record(bundle, *, age_band: str, gender_code, specimen: str,
                  setting: str, organism: str, morphotype: str,
                  order_year: int, order_month: int,
                  labs: dict, history: dict) -> tuple:
    """Build a record dict; return (record, list_of_assumed_fields)."""
    from src.preprocessing.organism_normalizer import parse_organism

    parsed = parse_organism(organism)
    rec: dict = {
        "age": age_band,
        "age_band_ordinal": float(AGE_BANDS.index(age_band)),
        "age_midpoint_years": float(AGE_MIDPOINT[age_band]),
        "gender_code": (np.nan if gender_code is None else float(gender_code)),
        "culture_description": specimen,
        "ordering_mode": setting,
        "organism_clean": parsed["organism_clean"] or organism,
        "organism_genus": parsed["organism_genus"],
        "organism_group": parsed["organism_group"],
        "organism_morphotype": morphotype,
        "is_species_level": int(parsed["is_species_level"]),
        "order_year": int(order_year),
        "order_month": int(order_month),
        "order_dayofweek": 2,
        "order_hour": 12,
    }
    for k, v in labs.items():
        if v is not None:
            rec[k] = float(v)
    for k, v in history.items():
        if v is not None:
            rec[k] = float(v)

    enterable = clinician_enterable(bundle)
    assumed = []
    for f in bundle.features:
        if f not in rec:
            rec[f] = bundle.defaults.get(f, np.nan)
            if f in enterable:
                assumed.append(f)
    return rec, sorted(assumed)


_CACHE: dict = {}


def load_test_partition(max_rows: int = 60_000) -> pd.DataFrame:
    """Held-out test rows, for demonstrating on genuine cases."""
    if "test" in _CACHE:
        return _CACHE["test"]
    df = load_analytic()
    split = patient_grouped_split(df)
    test = df[split == "test"]
    if len(test) > max_rows:
        rng = np.random.default_rng(7)
        orders = np.asarray(test["order_proc_id_coded"].unique(), dtype=object)
        rng.shuffle(orders)
        keep = set(orders[:max(1, max_rows // 12)])
        test = test[test["order_proc_id_coded"].isin(keep)]
    _CACHE["test"] = test.reset_index(drop=True)
    return _CACHE["test"]


def list_sample_orders(n: int = 40) -> pd.DataFrame:
    """A menu of real held-out culture orders the user can load."""
    test = load_test_partition()
    g = (test.groupby("order_proc_id_coded", observed=True)
         .agg(organism=("organism_clean", "first"),
              specimen=("culture_description", "first"),
              setting=("ordering_mode", "first"),
              age=("age", "first"),
              n_agents_tested=("antibiotic", "nunique"),
              n_nonsusceptible=(TARGET_BINARY, "sum"),
              prior_cultures=("prior_n_cultures", "first"),
              order_time=("order_time", "first"))
         .reset_index())
    g = g[g["n_agents_tested"] >= 6]
    # a spread of easy and hard cases rather than only the common ones
    g = g.sort_values("n_nonsusceptible", ascending=False)
    hard = g.head(n // 2)
    easy = g[g["n_nonsusceptible"] == 0].head(n - len(hard))
    out = pd.concat([hard, easy]).sort_values("organism")
    return out.reset_index(drop=True)


def sample_case(order_id: str, organism: str | None = None) -> tuple:
    """Return (record_dict, actual_results_frame) for a real test-set order.

    A culture order can grow more than one organism (13,214 of them do), and the
    AST panel is reported per isolate. The record is therefore one ISOLATE, and
    the ground-truth table is filtered to that same isolate — otherwise the same
    antibiotic appears twice with different results and the comparison against
    the prediction is meaningless.
    """
    test = load_test_partition()
    rows = test[test["order_proc_id_coded"] == order_id]
    if rows.empty:
        raise KeyError(f"Order {order_id} is not in the loaded test sample.")

    if organism is None:
        organism = rows["organism_clean"].iloc[0]
    isolate = rows[rows["organism_clean"] == organism]
    if isolate.empty:
        isolate = rows

    rec = isolate.iloc[0].to_dict()
    actual = (isolate[["antibiotic", "y_3class", TARGET_BINARY]]
              .rename(columns={"y_3class": "laboratory_result",
                               TARGET_BINARY: "non_susceptible"})
              .drop_duplicates("antibiotic")
              .sort_values("antibiotic")
              .reset_index(drop=True))
    return rec, actual


def organisms_in_order(order_id: str) -> list:
    """Every organism isolated from one culture order."""
    test = load_test_partition()
    rows = test[test["order_proc_id_coded"] == order_id]
    return sorted(rows["organism_clean"].dropna().unique().tolist())


def patient_history_view(record: dict) -> pd.DataFrame:
    """Readable summary of the history features behind a prediction."""
    fields = [
        ("Previous culture orders", "prior_n_cultures", "count"),
        ("Previous positive cultures", "prior_n_positive_cultures", "count"),
        ("Days since previous culture", "prior_days_since_last_culture", "days"),
        ("Previous susceptibility tests", "prior_n_tests", "count"),
        ("Previous non-susceptible results", "prior_n_nonsusc", "count"),
        ("Historic non-susceptibility rate", "prior_nonsusc_rate", "rate"),
        ("Previous isolates of this organism", "prior_same_org_n_cultures", "count"),
        ("Non-susceptibility rate for this organism",
         "prior_same_org_nonsusc_rate", "rate"),
        ("Documented resistant results (all time)", "prior_res_n_events_total",
         "count"),
        ("Documented resistant results, last 365 days",
         "prior_res_n_events_365d", "count"),
        ("Days since last documented resistance", "prior_res_days_since_last",
         "days"),
        ("Distinct antibiotics with prior resistance",
         "prior_res_n_distinct_abx_1095d", "count"),
        ("Antibiotic drug-days, last 90 days", "prior_med_n_drugdays_90d",
         "count"),
        ("Antibiotic drug-days, last 365 days", "prior_med_n_drugdays_365d",
         "count"),
        ("Distinct antibiotic classes, last 90 days",
         "prior_med_n_distinct_classes_90d", "count"),
        ("Broad-spectrum drug-days, last 90 days",
         "prior_med_n_broadspectrum_drugdays_90d", "count"),
        ("Days since last antibiotic", "prior_med_days_since_last", "days"),
    ]
    rows = []
    for label, key, kind in fields:
        v = record.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            shown = "not recorded"
        elif kind == "rate":
            shown = f"{float(v):.1%}"
        elif kind == "days":
            shown = f"{float(v):,.0f} days"
        else:
            shown = f"{float(v):,.0f}"
        rows.append({"History item": label, "Value": shown, "Field": key})
    return pd.DataFrame(rows)


def labs_view(record: dict) -> pd.DataFrame:
    panel = [
        ("White cell count", "median_wbc", "x10^9/L"),
        ("Neutrophils", "median_neutrophils", "%"),
        ("Lymphocytes", "median_lymphocytes", "%"),
        ("Haemoglobin", "median_hgb", "g/dL"),
        ("Platelets", "median_plt", "x10^9/L"),
        ("Sodium", "median_na", "mmol/L"),
        ("Bicarbonate", "median_hco3", "mmol/L"),
        ("Urea nitrogen", "median_bun", "mg/dL"),
        ("Creatinine", "median_cr", "mg/dL"),
        ("Lactate", "median_lactate", "mmol/L"),
        ("Procalcitonin", "median_procalcitonin", "ng/mL"),
    ]
    rows = []
    for label, key, unit in panel:
        v = record.get(key)
        rows.append({
            "Analyte": label,
            "Median over 14 days before culture":
                "not available" if v is None or (isinstance(v, float) and np.isnan(v))
                else f"{float(v):,.2f}",
            "Unit": unit,
        })
    return pd.DataFrame(rows)
