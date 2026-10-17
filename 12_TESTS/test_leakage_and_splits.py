"""The guarantees that stop this project from fooling itself."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import CONFIG_DIR, PROCESSED_DIR  # noqa: E402
from src.models.dataset import (EXCLUDED_COLUMNS, PAT, TARGET_BINARY,  # noqa: E402
                                load_analytic, patient_grouped_split,
                                temporal_split)

ANALYTIC = PROCESSED_DIR / "analytic_dataset.parquet"
FEATURES_JSON = CONFIG_DIR / "final_features.json"

requires_dataset = pytest.mark.skipif(
    not ANALYTIC.exists(), reason="analytic dataset not built")
requires_features = pytest.mark.skipif(
    not FEATURES_JSON.exists(), reason="feature selection not run")


@pytest.fixture(scope="module")
def df():
    return load_analytic()


@requires_dataset
def test_no_patient_appears_in_two_partitions(df):
    split = patient_grouped_split(df)
    n_partitions = df.assign(_s=split.to_numpy()).groupby(PAT)["_s"].nunique()
    assert int(n_partitions.max()) == 1, (
        f"{int((n_partitions > 1).sum())} patients cross a partition boundary")


@requires_dataset
def test_split_is_deterministic(df):
    a = patient_grouped_split(df)
    b = patient_grouped_split(df)
    assert (a.to_numpy() == b.to_numpy()).all()


@requires_dataset
def test_split_proportions_are_sane(df):
    split = patient_grouped_split(df)
    frac = split.value_counts(normalize=True)
    assert 0.60 < frac["train"] < 0.80
    assert 0.14 < frac["test"] < 0.26
    assert 0.05 < frac["val"] < 0.16


@requires_dataset
def test_every_partition_has_both_classes(df):
    split = patient_grouped_split(df)
    for part in ("train", "val", "test"):
        assert df.loc[split == part, TARGET_BINARY].nunique() == 2


@requires_dataset
def test_temporal_split_has_no_patient_overlap(df):
    tsplit = temporal_split(df, train_frac=0.80)
    train_pat = set(df.loc[tsplit == "train", PAT])
    test_pat = set(df.loc[tsplit == "test", PAT])
    assert not (train_pat & test_pat), (
        f"{len(train_pat & test_pat)} patients appear on both sides of the "
        "temporal cut-off")


@requires_dataset
def test_temporal_split_is_actually_temporal(df):
    tsplit = temporal_split(df, train_frac=0.80)
    assert (df.loc[tsplit == "train", "order_time"].max()
            <= df.loc[tsplit == "test", "order_time"].min())


@requires_features
def test_final_feature_set_excludes_every_leaky_column():
    cfg = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    overlap = set(cfg["features"]) & set(EXCLUDED_COLUMNS)
    assert not overlap, f"leaky/excluded columns selected as features: {overlap}"


@requires_features
def test_target_columns_are_not_features():
    cfg = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    for t in ("y_nonsusceptible", "y_3class", "severity", "susceptibility"):
        assert t not in cfg["features"]


@requires_features
def test_identifiers_are_not_features():
    cfg = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    for ident in ("anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"):
        assert ident not in cfg["features"]


@requires_features
def test_was_positive_is_not_a_feature():
    """It is constant among labelled rows AND is a culture result."""
    cfg = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    assert "was_positive" not in cfg["features"]


@requires_dataset
def test_prior_resistance_events_are_strictly_before_the_culture():
    """The raw history table must contain no event at or after the index time."""
    rep = json.loads((PROCESSED_DIR / "build_report.json").read_text(
        encoding="utf-8"))
    assert rep["step5_prior_resistance"]["n_rows_t_le_0"] == 0
    assert rep["step5_prior_resistance"]["t_min"] >= 1.0


@requires_dataset
def test_same_day_medications_were_excluded():
    rep = json.loads((PROCESSED_DIR / "build_report.json").read_text(
        encoding="utf-8"))
    s4 = rep["step4_prior_med"]
    assert s4["n_dropped_same_day"] == s4["n_rows_same_day_t_eq_0"]
    assert s4["n_rows_same_day_t_eq_0"] > 0


@requires_dataset
def test_prior_test_counts_exclude_the_current_order(df):
    """At a patient's first POSITIVE culture, prior test counts must be zero.

    Note the subtlety: `prior_n_cultures` counts ALL prior cultures including
    negative ones, and the analytic table holds only positive cultures, so a
    patient's first row here can legitimately report prior cultures. What can
    never be non-zero is `prior_n_tests` / `prior_n_nonsusc`, because only a
    positive culture produces susceptibility tests. If the cumulative history
    were not shifted, the current order's own tests would appear here.
    """
    first = (df.sort_values([PAT, "order_time", "order_proc_id_coded"])
             .drop_duplicates(PAT, keep="first"))
    assert (first["prior_n_tests"] == 0).all(), (
        "a patient's first positive culture reports prior susceptibility "
        "tests — the cumulative history is not shifted correctly")
    assert (first["prior_n_nonsusc"] == 0).all()


@requires_dataset
def test_prior_culture_counts_are_monotonic_within_a_patient(df):
    """History can only grow as time moves forward."""
    sample_patients = df[PAT].drop_duplicates().head(3000)
    sub = (df[df[PAT].isin(set(sample_patients))]
           .sort_values([PAT, "order_time", "order_proc_id_coded"]))
    per_order = sub.drop_duplicates([PAT, "order_proc_id_coded"])
    diffs = per_order.groupby(PAT)["prior_n_cultures"].diff().dropna()
    assert (diffs >= 0).all(), (
        "prior culture counts decrease over time within a patient")


@requires_dataset
def test_prior_same_antibiotic_history_excludes_the_current_row(df):
    """The first time an agent is tested for a patient, its history is empty."""
    first = (df.sort_values([PAT, "antibiotic", "order_time",
                             "order_proc_id_coded"])
             .drop_duplicates([PAT, "antibiotic"], keep="first"))
    assert (first["prior_same_abx_n_tests"] == 0).all()
    assert (first["prior_same_abx_n_nonsusc"] == 0).all()
