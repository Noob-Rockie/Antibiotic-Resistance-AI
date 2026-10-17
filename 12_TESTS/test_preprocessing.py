"""Unit tests for the domain preprocessing: organism cleaning, taxonomies."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.antibiotic_taxonomy import (  # noqa: E402
    ANTIBIOTIC_CLASS, antibiotic_class, aware_category, coverage_report,
    is_oral, is_urinary_only)
from src.preprocessing.medication_mapping import (  # noqa: E402
    MEDICATION_MAP, medication_class, medication_ingredient)
from src.preprocessing.organism_normalizer import parse_organism  # noqa: E402
from src.models.preprocessing import OrdinalCategoryEncoder  # noqa: E402


# --- organism normalisation ------------------------------------------------
@pytest.mark.parametrize("raw,expected_clean,expected_leak", [
    ("STAPH AUREUS {MRSA}", "STAPHYLOCOCCUS AUREUS", "MRSA"),
    ("KLEBSIELLA PNEUMONIAE (CARBAPENEM RESISTANT)", "KLEBSIELLA PNEUMONIAE",
     "CARBAPENEM_RESISTANT"),
    ("ENTEROCOCCUS FAECIUM -  VANCO RESISTANT", "ENTEROCOCCUS FAECIUM",
     "VANCOMYCIN_RESISTANT"),
    ("ESCHERICHIA COLI", "ESCHERICHIA COLI", None),
])
def test_resistance_tokens_are_extracted_and_stripped(raw, expected_clean,
                                                      expected_leak):
    r = parse_organism(raw)
    assert r["organism_clean"] == expected_clean
    assert r["leaked_phenotype"] == expected_leak


def test_deprecated_zzz_prefix_is_resolved_to_the_current_name():
    assert parse_organism("ZZZENTEROBACTER AEROGENES")["organism_clean"] == \
        "KLEBSIELLA AEROGENES"


def test_bracket_variants_collapse_to_one_level():
    a = parse_organism("STREPTOCOCCUS AGALACTIAE (GROUP B)")["organism_clean"]
    b = parse_organism("STREPTOCOCCUS AGALACTIAE {GROUP B}")["organism_clean"]
    assert a == b == "STREPTOCOCCUS AGALACTIAE"


def test_complex_suffix_collapses():
    assert parse_organism("ENTEROBACTER CLOACAE COMPLEX")["organism_clean"] == \
        parse_organism("ENTEROBACTER CLOACAE")["organism_clean"]


@pytest.mark.parametrize("raw,morphotype", [
    ("MUCOID PSEUDOMONAS AERUGINOSA", "mucoid"),
    ("PSEUDOMONAS AERUGINOSA (NON-MUCOID CF)", "non_mucoid"),
    ("STAPH AUREUS(COLONY VARIANT - SMALL COLONY OR OTHER MORPHOTYPE)",
     "small_colony_variant"),
    ("ESCHERICHIA COLI", "standard"),
])
def test_morphotype_is_preserved_not_discarded(raw, morphotype):
    """Morphotype is observed at identification, so it is a legal feature."""
    assert parse_organism(raw)["organism_morphotype"] == morphotype


def test_morphotype_does_not_change_the_species():
    assert parse_organism("MUCOID PSEUDOMONAS AERUGINOSA")["organism_clean"] == \
        "PSEUDOMONAS AERUGINOSA"


@pytest.mark.parametrize("raw,group", [
    ("ESCHERICHIA COLI", "GN_Enterobacterales"),
    ("PSEUDOMONAS AERUGINOSA", "GN_NonFermenter"),
    ("STAPHYLOCOCCUS AUREUS", "GP_Cocci"),
    ("BACTEROIDES FRAGILIS", "Anaerobe"),
    ("MYCOBACTERIUM ABSCESSUS", "Mycobacterium"),
    ("CANDIDA ALBICANS", "Fungus"),
    ("COAG NEGATIVE STAPHYLOCOCCUS", "GP_Cocci"),
    ("NON FERMENTING GRAM NEGATIVE RODS", "GN_NonFermenter"),
])
def test_organism_group_assignment(raw, group):
    assert parse_organism(raw)["organism_group"] == group


def test_null_organism_is_handled():
    r = parse_organism(None)
    assert r["organism_clean"] is None and r["organism_group"] is None


# --- antibiotic taxonomy ----------------------------------------------------
def test_taxonomy_covers_every_agent_in_the_cohort():
    """The 54 agents observed in the data must all be classified."""
    from config.paths import PROCESSED_DIR
    p = PROCESSED_DIR / "analytic_dataset.parquet"
    if not p.exists():
        pytest.skip("analytic dataset not built")
    agents = pd.read_parquet(p, columns=["antibiotic"])["antibiotic"].unique()
    rep = coverage_report(agents)
    assert rep["unmapped"] == [], f"unmapped antibiotics: {rep['unmapped']}"
    assert rep["coverage_pct"] == 100.0


def test_class_assignments_are_correct():
    assert antibiotic_class("Meropenem") == "Carbapenem"
    assert antibiotic_class("Ciprofloxacin") == "Fluoroquinolone"
    assert antibiotic_class("Vancomycin") == "Glycopeptide"
    assert antibiotic_class("Cefepime") == "Cephalosporin"
    assert aware_category("Linezolid") == "Reserve"
    assert aware_category("Ampicillin") == "Access"


def test_urinary_only_agents():
    assert is_urinary_only("Nitrofurantoin") == 1
    assert is_urinary_only("Fosfomycin") == 1
    assert is_urinary_only("Meropenem") == 0


def test_oral_flag():
    assert is_oral("Ciprofloxacin") == 1
    assert is_oral("Meropenem") == 0


# --- medication mapping -----------------------------------------------------
def test_brand_names_resolve_to_their_ingredient():
    assert medication_ingredient("Zithromax") == "Azithromycin"
    assert medication_ingredient("Bactrim Ds") == "Trimethoprim/Sulfamethoxazole"
    assert medication_ingredient("Keflex") == "Cephalexin"
    assert medication_ingredient("Macrobid") == "Nitrofurantoin"
    assert medication_ingredient("Flagyl") == "Metronidazole"


def test_salt_and_diluent_variants_collapse():
    for v in ("Ciprofloxacin", "Ciprofloxacin Hcl", "Ciprofloxacin In", "Cipro"):
        assert medication_ingredient(v) == "Ciprofloxacin"
        assert medication_class(v) == "Fluoroquinolone"


def test_medication_map_covers_all_89_source_names():
    assert len(MEDICATION_MAP) == 89, (
        f"the source file has 89 distinct medication_name values; the map has "
        f"{len(MEDICATION_MAP)}")


def test_medication_classes_align_with_antibiotic_classes():
    """Prior exposure to class X must line up with susceptibility of class X."""
    med_classes = {v[1] for v in MEDICATION_MAP.values()}
    abx_classes = {v[0] for v in ANTIBIOTIC_CLASS.values()}
    shared = med_classes & abx_classes
    assert "Fluoroquinolone" in shared
    assert "Carbapenem" in shared
    assert "Cephalosporin" in shared
    assert "Glycopeptide" in shared


# --- encoder ----------------------------------------------------------------
def test_ordinal_encoder_maps_unseen_categories_to_minus_one():
    train = pd.DataFrame({"org": ["A", "B", "A"], "num": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"org": ["A", "C"], "num": [1.0, 9.0]})
    enc = OrdinalCategoryEncoder(columns=["org"]).fit(train)
    out = enc.transform(test)
    assert out["org"].iloc[0] >= 0        # seen category
    assert out["org"].iloc[1] == -1.0     # unseen category


def test_ordinal_encoder_is_fitted_only_on_training_data():
    train = pd.DataFrame({"org": ["A", "B"]})
    enc = OrdinalCategoryEncoder(columns=["org"]).fit(train)
    assert set(enc.mapping_["org"]) == {"A", "B"}
