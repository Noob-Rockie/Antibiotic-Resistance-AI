"""Shared dataset contract: what is a feature, what is not, and how we split.

This module is the single source of truth used by feature selection, model
training, evaluation, explainability and the Streamlit application. If a column
is not declared here it never reaches a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import PROCESSED_DIR, RANDOM_SEED  # noqa: E402

PAT = "anon_id"
ORDER = "order_proc_id_coded"

# --- columns that are never features ----------------------------------------
# (column, reason) — mirrored into docs/DATA_LEAKAGE_ANALYSIS.md
EXCLUDED_COLUMNS: dict[str, str] = {
    "anon_id": "Patient identifier — memorisable; used only to group the split.",
    "pat_enc_csn_id_coded": "Encounter identifier — no clinical content.",
    "order_proc_id_coded": "Order identifier — used only to link a prediction to its culture.",
    "order_time": "Raw timestamp — replaced by year/month/day-of-week/hour.",
    "susceptibility": "Target (raw label).",
    "severity": "Target (ordinal encoding of the label).",
    "y_nonsusceptible": "Target (binary).",
    "y_3class": "Target (3-class).",
    "age": "Raw age band string — replaced by age_band_ordinal and age_midpoint_years.",
}

TARGET_BINARY = "y_nonsusceptible"
TARGET_3CLASS = "y_3class"

# --- feature groups (used for the group-ablation experiments) ---------------
CATEGORICAL_FEATURES = [
    "ordering_mode", "culture_description", "organism_clean", "organism_genus",
    "organism_group", "organism_morphotype", "antibiotic", "antibiotic_class",
    "antibiotic_subclass", "antibiotic_aware",
]

ORGANISM_FEATURES = ["organism_clean", "organism_genus", "organism_group",
                     "organism_morphotype", "is_species_level"]

ANTIBIOTIC_FEATURES = ["antibiotic", "antibiotic_class", "antibiotic_subclass",
                       "antibiotic_aware", "antibiotic_is_oral",
                       "antibiotic_is_urinary_only"]

CONTEXT_FEATURES = ["ordering_mode", "culture_description", "order_year",
                    "order_month", "order_dayofweek", "order_hour"]

PATIENT_FEATURES = ["age_band_ordinal", "age_midpoint_years", "gender_code"]


def feature_groups(columns) -> dict:
    """Assign every modellable column to exactly one clinical group."""
    cols = [c for c in columns if c not in EXCLUDED_COLUMNS]
    groups: dict[str, list] = {
        "patient": [], "context": [], "microbiology": [], "antibiotic": [],
        "laboratory": [], "prior_medication": [], "prior_resistance": [],
        "prior_culture": [], "other": [],
    }
    lab_tokens = ("wbc", "neutrophils", "lymphocytes", "hgb", "plt", "na",
                  "hco3", "bun", "cr", "lactate", "procalcitonin")
    for c in cols:
        if c in PATIENT_FEATURES:
            groups["patient"].append(c)
        elif c in CONTEXT_FEATURES:
            groups["context"].append(c)
        elif c in ORGANISM_FEATURES:
            groups["microbiology"].append(c)
        elif c in ANTIBIOTIC_FEATURES:
            groups["antibiotic"].append(c)
        elif c == "has_labs" or any(c.endswith(f"_{t}") or f"_{t}" in c
                                    for t in lab_tokens):
            groups["laboratory"].append(c)
        elif c.startswith("prior_med"):
            groups["prior_medication"].append(c)
        elif c.startswith("prior_res"):
            groups["prior_resistance"].append(c)
        elif c.startswith("prior_") or c == "has_prior_culture":
            groups["prior_culture"].append(c)
        else:
            groups["other"].append(c)
    return {k: v for k, v in groups.items() if v}


def load_analytic(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or (PROCESSED_DIR / "analytic_dataset.parquet"))


def all_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS]


def split_categorical_numeric(df: pd.DataFrame, features) -> tuple:
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    num = [c for c in features if c not in cat]
    return cat, num


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def patient_grouped_split(df: pd.DataFrame, test_size: float = 0.20,
                          val_size: float = 0.10,
                          seed: int = RANDOM_SEED) -> pd.Series:
    """Assign each row to train / val / test, grouped so no patient is split.

    Returns a Series of {'train','val','test'} aligned to df.index.
    """
    groups = df[PAT].to_numpy()
    idx = np.arange(len(df))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_i, test_i = next(gss.split(idx, groups=groups))

    # val_size is expressed as a fraction of the WHOLE dataset
    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr_rel, va_rel = next(gss2.split(rest_i, groups=groups[rest_i]))

    split = pd.Series("train", index=df.index, dtype=object)
    split.iloc[test_i] = "test"
    split.iloc[rest_i[va_rel]] = "val"

    # hard guarantee: no patient crosses a partition
    check = df.assign(_s=split.to_numpy()).groupby(PAT)["_s"].nunique()
    if int(check.max()) != 1:
        raise AssertionError(
            f"{int((check > 1).sum())} patients appear in more than one split")
    return split


def temporal_split(df: pd.DataFrame, train_frac: float = 0.80) -> pd.Series:
    """Train on the earlier orders, test on the most recent ones.

    Patients whose orders straddle the cut-off are removed from the test side so
    the temporal check is not silently contaminated by patient overlap.
    """
    cutoff = df["order_time"].quantile(train_frac)
    split = pd.Series(np.where(df["order_time"] < cutoff, "train", "test"),
                      index=df.index, dtype=object)
    train_patients = set(df.loc[split == "train", PAT])
    overlap = (split == "test") & df[PAT].isin(train_patients)
    split[overlap] = "drop"
    return split


def split_report(df: pd.DataFrame, split: pd.Series, target: str) -> dict:
    out = {"cutoff_note": None, "partitions": {}}
    for part in ("train", "val", "test", "drop"):
        m = split == part
        if not m.any():
            continue
        out["partitions"][part] = {
            "n_rows": int(m.sum()),
            "n_patients": int(df.loc[m, PAT].nunique()),
            "n_orders": int(df.loc[m, ORDER].nunique()),
            "positive_rate": round(float(df.loc[m, target].mean()), 6),
            "date_min": str(df.loc[m, "order_time"].min()),
            "date_max": str(df.loc[m, "order_time"].max()),
        }
    return out


def subsample_by_patient(df: pd.DataFrame, n_rows: int,
                         seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Take a patient-level random subsample of approximately n_rows rows.

    Sampling whole patients (not rows) keeps every within-patient correlation
    intact, so the subsample behaves like a smaller version of the same problem
    rather than a leakier one.
    """
    if len(df) <= n_rows:
        return df
    rng = np.random.default_rng(seed)
    # `.unique()` on an Arrow-backed string column returns an ArrowStringArray,
    # which numpy cannot shuffle safely (it warns and can duplicate entries).
    # Materialise a plain object array first.
    patients = np.asarray(df[PAT].unique(), dtype=object)
    rng.shuffle(patients)
    sizes = df.groupby(PAT, observed=True).size()
    cum = sizes.reindex(patients).cumsum()
    keep = set(cum[cum <= n_rows].index)
    if not keep:
        keep = {patients[0]}
    return df[df[PAT].isin(keep)]
