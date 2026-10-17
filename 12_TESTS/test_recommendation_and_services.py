"""Safety behaviour of the decision-support layer and the external services.

These tests exist because the failure modes they cover are the ones that would
actually matter: a prototype that quietly invents a dose, quietly invents a
price, or quietly hides an option from the reviewing clinician.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.module_loader import load_stage_module  # noqa: E402

rules = load_stage_module("08_RECOMMENDATION_ENGINE", "recommendation_rules")
dosage = load_stage_module("08_RECOMMENDATION_ENGINE", "dosage_reference")
cost = load_stage_module("08_RECOMMENDATION_ENGINE", "cost_service")
genomic = load_stage_module("09_GENOMIC_AMR", "genomic_analysis")
ranking = load_stage_module("08_RECOMMENDATION_ENGINE", "antibiotic_ranking")


# --- intrinsic resistance: genus / species level ----------------------------
@pytest.mark.parametrize("organism,antibiotic,abx_class", [
    ("ENTEROCOCCUS FAECALIS", "Ceftriaxone", "Cephalosporin"),
    ("KLEBSIELLA PNEUMONIAE", "Ampicillin", "Penicillin"),
    ("PROTEUS MIRABILIS", "Nitrofurantoin", "Nitrofuran"),
    ("PSEUDOMONAS AERUGINOSA", "Ertapenem", "Carbapenem"),
    ("STENOTROPHOMONAS MALTOPHILIA", "Meropenem", "Carbapenem"),
    ("ACHROMOBACTER XYLOSOXIDANS", "Aztreonam", "Monobactam"),
])
def test_known_intrinsic_resistance_is_flagged(organism, antibiotic, abx_class):
    flags = rules.intrinsic_resistance_flags(organism, antibiotic, abx_class)
    assert flags, f"no intrinsic-resistance flag for {organism} / {antibiotic}"
    assert flags[0]["severity"] == rules.CRITICAL


# --- intrinsic resistance: gram-stain level, keyed on organism_group --------
# This is the layer that a genus enumeration always eventually misses. It is
# tested against organisms that are NOT named anywhere in the genus rules,
# because that is exactly the case that failed in development: the ranking
# offered vancomycin and linezolid as top options for a gram-negative
# non-fermenter (Achromobacter xylosoxidans).
@pytest.mark.parametrize("organism,group,antibiotic,abx_class", [
    ("ACHROMOBACTER XYLOSOXIDANS", "GN_NonFermenter", "Vancomycin", "Glycopeptide"),
    ("ACHROMOBACTER XYLOSOXIDANS", "GN_NonFermenter", "Linezolid", "Oxazolidinone"),
    ("ACHROMOBACTER XYLOSOXIDANS", "GN_NonFermenter", "Daptomycin", "Lipopeptide"),
    ("ELIZABETHKINGIA SPECIES", "GN_NonFermenter", "Vancomycin", "Glycopeptide"),
    ("ESCHERICHIA COLI", "GN_Enterobacterales", "Vancomycin", "Glycopeptide"),
    ("RAOULTELLA ORNITHINOLYTICA", "GN_Enterobacterales", "Linezolid", "Oxazolidinone"),
    ("STAPHYLOCOCCUS AUREUS", "GP_Cocci", "Aztreonam", "Monobactam"),
    ("STREPTOCOCCUS PNEUMONIAE", "GP_Cocci", "Colistin", "Polymyxin"),
    ("CORYNEBACTERIUM STRIATUM", "GP_Rod", "Aztreonam", "Monobactam"),
    ("BACTEROIDES FRAGILIS", "Anaerobe", "Gentamicin", "Aminoglycoside"),
])
def test_gram_stain_level_intrinsic_resistance_is_flagged(
        organism, group, antibiotic, abx_class):
    flags = rules.intrinsic_resistance_flags(organism, antibiotic, abx_class,
                                             organism_group=group)
    assert flags, (f"{organism} ({group}) was NOT flagged against {antibiotic} "
                   "— a gram-stain-level intrinsic-resistance rule is missing")
    assert flags[0]["severity"] == rules.CRITICAL


def test_narrow_penicillins_are_flagged_for_gram_negatives_by_subclass():
    """Benzylpenicillin must be flagged, but ampicillin must NOT be."""
    flagged = rules.intrinsic_resistance_flags(
        "ESCHERICHIA COLI", "Penicillin", "Penicillin",
        organism_group="GN_Enterobacterales",
        antibiotic_subclass="Natural penicillin")
    assert flagged, "benzylpenicillin not flagged for a gram-negative"

    not_flagged = rules.intrinsic_resistance_flags(
        "ESCHERICHIA COLI", "Ampicillin", "Penicillin",
        organism_group="GN_Enterobacterales",
        antibiotic_subclass="Aminopenicillin")
    assert not not_flagged, (
        "ampicillin was flagged for E. coli — aminopenicillins do treat some "
        "Enterobacterales, so the rule must be at subclass level")


def test_fungal_isolates_flag_every_antibacterial():
    for abx, cls in (("Meropenem", "Carbapenem"), ("Vancomycin", "Glycopeptide"),
                     ("Ciprofloxacin", "Fluoroquinolone")):
        flags = rules.intrinsic_resistance_flags(
            "CANDIDA ALBICANS", abx, cls, organism_group="Fungus")
        assert flags, f"{abx} not flagged for a fungal isolate"


@pytest.mark.parametrize("organism,group,antibiotic,abx_class", [
    ("ESCHERICHIA COLI", "GN_Enterobacterales", "Ciprofloxacin", "Fluoroquinolone"),
    ("STAPHYLOCOCCUS AUREUS", "GP_Cocci", "Vancomycin", "Glycopeptide"),
    ("KLEBSIELLA PNEUMONIAE", "GN_Enterobacterales", "Meropenem", "Carbapenem"),
    ("ESCHERICHIA COLI", "GN_Enterobacterales", "Nitrofurantoin", "Nitrofuran"),
    ("PSEUDOMONAS AERUGINOSA", "GN_NonFermenter", "Ceftazidime", "Cephalosporin"),
])
def test_sensible_pairs_are_not_flagged(organism, group, antibiotic, abx_class):
    assert not rules.intrinsic_resistance_flags(
        organism, antibiotic, abx_class, organism_group=group)


# --- site suitability -------------------------------------------------------
def test_nitrofurantoin_is_flagged_for_blood():
    flags = rules.site_flags("BLOOD", "Nitrofurantoin")
    assert flags and flags[0]["severity"] == rules.CRITICAL


def test_daptomycin_is_flagged_for_respiratory():
    flags = rules.site_flags("RESPIRATORY", "Daptomycin")
    assert flags and "surfactant" in flags[0]["detail"].lower()


def test_nitrofurantoin_is_fine_for_urine():
    assert not rules.site_flags("URINE", "Nitrofurantoin")


# --- allergy ----------------------------------------------------------------
def test_penicillin_allergy_flags_penicillins_critically():
    flags = rules.allergy_flags("Penicillin", ["penicillin"])
    assert flags and flags[0]["severity"] == rules.CRITICAL


def test_penicillin_allergy_flags_cephalosporins_as_caution_not_critical():
    """Cross-reactivity is low; over-flagging drives unnecessary escalation."""
    flags = rules.allergy_flags("Cephalosporin", ["penicillin"])
    assert flags and flags[0]["severity"] == rules.CAUTION


def test_no_allergy_no_flag():
    assert not rules.allergy_flags("Cephalosporin", [])


# --- evidence strength ------------------------------------------------------
def test_zero_training_examples_raises_a_flag():
    flags = rules.evidence_flags(0)
    assert flags and flags[0]["kind"] == "no_evidence"


def test_thin_evidence_raises_a_flag():
    assert rules.evidence_flags(120)[0]["kind"] == "thin_evidence"


def test_ample_evidence_raises_no_flag():
    assert rules.evidence_flags(50_000) == []


# --- stewardship ------------------------------------------------------------
def test_reserve_agents_are_cautioned():
    flags = rules.stewardship_flags("Linezolid", "Reserve")
    assert flags[0]["severity"] == rules.CAUTION


def test_access_agents_are_informational_only():
    assert rules.stewardship_flags("Ampicillin", "Access")[0]["severity"] == \
        rules.INFO


# --- framing ----------------------------------------------------------------
def test_ranking_module_never_uses_prescriptive_language():
    text = (Path(PROJECT_ROOT / "08_RECOMMENDATION_ENGINE" /
                 "antibiotic_ranking.py").read_text(encoding="utf-8"))
    banned = ["you should take", "prescribe ", "recommended dose of",
              "give the patient"]
    lowered = text.lower()
    for phrase in banned:
        assert phrase not in lowered, f"prescriptive phrasing found: {phrase!r}"


def test_safety_notice_is_present_and_explicit():
    notice = ranking.SAFETY_NOTICE.lower()
    assert "not a treatment decision" in notice
    assert "qualified healthcare professional" in notice
    assert "susceptibility testing" in notice


# --- dosage -----------------------------------------------------------------
def test_dosage_returns_insufficient_information_when_offline(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_API", "0")
    http = load_stage_module("10_API_SERVICES", "_http")
    assert http.live_enabled() is False
    r = dosage.get_dosage_reference("Ciprofloxacin", {})
    if not r["available"]:
        assert r["message"] == "Insufficient information for dosage reference."
        assert r["reason"]


def test_dosage_never_computes_creatinine_clearance():
    ctx = dosage.renal_context({"median_cr": 2.4})
    assert ctx["creatinine_clearance_computable"] is False
    assert "weight" in ctx["why_not"].lower()


def test_dosage_module_contains_no_hardcoded_dose_strings():
    text = Path(PROJECT_ROOT / "08_RECOMMENDATION_ENGINE" /
                "dosage_reference.py").read_text(encoding="utf-8")
    import re
    # a literal like "500 mg" or "1 g every 8 hours" would mean an invented dose
    assert not re.search(r'"\s*\d+\s*(mg|g|mcg)\b', text, re.I), \
        "a literal dose string appears in the dosage module"


# --- cost -------------------------------------------------------------------
def test_cost_message_when_unavailable_is_the_specified_wording():
    assert cost.UNAVAILABLE_MESSAGE == "Price unavailable — verify locally."


def test_cost_never_computes_a_course_cost(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_API", "0")
    r = cost.get_cost("Ciprofloxacin")
    if r["available"]:
        assert r["course_cost_calculated"] is False


# --- genomic ----------------------------------------------------------------
def test_genomic_module_reports_unavailable_for_this_dataset():
    status = genomic.availability_report()
    assert status["available"] is False
    assert "no genomic data" in status["reason"].lower()


def test_genomic_annotation_never_asserts_avoidance():
    ann = genomic.annotate_markers(["blaKPC-2", "mecA", "vanA"])
    for a in ann:
        assert "supports clinician consideration" in a["implication"].lower()
        assert "do not use" not in a["implication"].lower()
        assert "must be avoided" not in a["implication"].lower()


def test_unknown_marker_is_reported_as_unknown_not_guessed():
    a = genomic.annotate_markers(["totallyMadeUpGene9"])[0]
    assert a["gene_family"] == "unrecognised"
    assert a["affected_classes"] == []


def test_marker_normalisation_handles_allele_and_format_variants():
    for variant in ("blaKPC-2", "blaKPC", "BLAKPC3", "bla_kpc_2"):
        ann = genomic.annotate_markers([variant])[0]
        assert "KPC" in ann["gene_family"], f"{variant} failed to normalise"


# --- API plumbing -----------------------------------------------------------
def test_no_api_key_is_hardcoded():
    for name in ("drug_api.py", "price_api.py", "_http.py", "genomic_api.py"):
        p = PROJECT_ROOT / "10_API_SERVICES" / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        import re
        assert not re.search(r'api_key\s*=\s*["\'][A-Za-z0-9]{8,}', text), \
            f"a literal API key appears in {name}"


def test_live_api_can_be_disabled_globally(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_API", "0")
    http = load_stage_module("10_API_SERVICES", "_http")
    res = http.get_json("https://api.fda.gov/drug/label.json", {"limit": 1},
                        source="test", use_cache=False)
    assert res.ok is False
    assert "disabled" in res.reason.lower()
