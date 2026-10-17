"""Data-integrity guarantees. These are the checks that would catch a silent
corruption of the pipeline, so they assert on the real artefacts, not mocks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import PROCESSED_DIR, RAW_FILES, RAW_REF_DIR  # noqa: E402

ANALYTIC = PROCESSED_DIR / "analytic_dataset.parquet"
MANIFEST = RAW_REF_DIR / "RAW_DATA_MANIFEST.json"

requires_dataset = pytest.mark.skipif(
    not ANALYTIC.exists(),
    reason="analytic dataset not built; run scripts/04_build_analytic_dataset.py")


def test_raw_files_exist():
    for key, path in RAW_FILES.items():
        assert path.exists(), f"raw file missing: {key} -> {path}"


def test_no_source_file_writes_to_a_raw_path():
    """No module may send a write call to a raw-data path.

    The check is a static scan for a write-style call whose target is derived
    from `RAW_FILES` / `RAW_DATA_DIR` / `RAW_REF_DIR` is excluded because that
    directory holds the manifest, not the data.
    """
    import re
    write_call = re.compile(
        r"\b(?:to_csv|to_parquet|write_text|write_bytes|to_excel|savetxt)"
        r"\s*\(\s*(?:RAW_FILES|RAW_DATA_DIR)\b"
        r"|open\s*\(\s*(?:RAW_FILES|RAW_DATA_DIR)[^)]*?['\"][waxr]?\+?[wa]")
    offenders = []
    for py in PROJECT_ROOT.rglob("*.py"):
        if ".venv" in py.parts or "12_TESTS" in py.parts:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        if write_call.search(text):
            offenders.append(str(py.relative_to(PROJECT_ROOT)))
    assert not offenders, f"raw data would be written by: {offenders}"


def test_raw_paths_are_only_read_through_the_central_config():
    """Nothing may hard-code the raw directory outside config/paths.py."""
    offenders = []
    needle = "OneDrive\\Desktop"
    for py in PROJECT_ROOT.rglob("*.py"):
        if ".venv" in py.parts or "12_TESTS" in py.parts:
            continue
        if py.name == "paths.py" and py.parent.name == "config":
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        if needle in text and "Antibiotic Resistance Prediction" in text:
            offenders.append(str(py.relative_to(PROJECT_ROOT)))
    assert not offenders, (
        f"raw data path hard-coded outside config/paths.py: {offenders}")


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not generated yet")
def test_raw_file_sizes_match_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key, rec in manifest["files"].items():
        path = RAW_FILES[key]
        assert path.stat().st_size == rec["size_bytes"], (
            f"{key} changed size since the audit: "
            f"{path.stat().st_size:,} vs {rec['size_bytes']:,}")


@requires_dataset
def test_analytic_dataset_shape_and_target():
    df = pd.read_parquet(ANALYTIC, columns=["y_nonsusceptible", "severity",
                                            "susceptibility"])
    assert len(df) > 1_000_000
    assert set(df["y_nonsusceptible"].unique()) == {0, 1}
    assert set(df["susceptibility"].unique()) == {
        "Susceptible", "Intermediate", "Resistant"}
    # binary target must be exactly (severity >= 1)
    assert (df["y_nonsusceptible"] == (df["severity"] >= 1).astype("int8")).all()


@requires_dataset
def test_no_inconclusive_or_synergism_rows_survive():
    df = pd.read_parquet(ANALYTIC, columns=["susceptibility"])
    assert not df["susceptibility"].isin(["Inconclusive", "Synergism"]).any()


@requires_dataset
def test_dataset_grain_is_unique():
    df = pd.read_parquet(ANALYTIC, columns=["order_proc_id_coded",
                                            "organism_clean", "antibiotic"])
    dups = df.duplicated(["order_proc_id_coded", "organism_clean",
                          "antibiotic"]).sum()
    assert dups == 0, f"{dups:,} duplicate (order, organism, antibiotic) rows"


@requires_dataset
def test_no_resistance_phenotype_leaked_into_organism_names():
    """The MRSA / carbapenem-resistant tokens must not survive cleaning."""
    df = pd.read_parquet(ANALYTIC, columns=["organism_clean"])
    names = df["organism_clean"].dropna().astype(str).unique()
    bad = [n for n in names
           if any(t in n.upper() for t in
                  ("MRSA", "RESISTANT", "ESBL", "VRE", "MSSA"))]
    assert not bad, f"resistance tokens survived into organism_clean: {bad}"


@requires_dataset
def test_history_features_are_never_negative():
    df = pd.read_parquet(ANALYTIC)
    count_cols = [c for c in df.columns
                  if c.startswith(("prior_res_n", "prior_med_n", "prior_n"))
                  and "rate" not in c]
    for c in count_cols:
        assert (df[c].dropna() >= 0).all(), f"{c} has negative counts"


@requires_dataset
def test_build_report_matches_dataset():
    rep = json.loads((PROCESSED_DIR / "build_report.json").read_text(
        encoding="utf-8"))
    df = pd.read_parquet(ANALYTIC, columns=["y_nonsusceptible"])
    assert rep["step7_final"]["n_rows"] == len(df)
    assert abs(rep["target_binary_positive_rate"]
               if "target_binary_positive_rate" in rep
               else rep["step1_cohort"]["target_binary_positive_rate"]
               - float(df["y_nonsusceptible"].mean())) < 1e-6
