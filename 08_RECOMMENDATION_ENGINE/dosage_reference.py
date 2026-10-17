"""Dosage REFERENCE — relayed from the FDA label, never generated.

THE RULE
This module does not calculate, adjust, estimate or paraphrase a dose. It
retrieves the `dosage_and_administration` section of the FDA Structured Product
Label through `10_API_SERVICES/drug_api.py`, quotes it verbatim, and states
where it came from and when it was retrieved.

If the label cannot be retrieved, or the patient information needed to interpret
it is missing, the module returns `available=False` with an explicit reason and
the application prints:

    "Insufficient information for dosage reference."

It never falls back to a plausible-looking number.

WHY RENAL FUNCTION MATTERS HERE
Most antibacterials in this cohort are renally cleared, and the FDA label
usually specifies dose adjustment by creatinine clearance. This dataset provides
creatinine (`median_cr`, `last_cr`, `first_cr`) but NOT weight, height or race,
and Cockcroft-Gault requires weight. A creatinine clearance therefore cannot be
computed from the available data, and this module does not pretend otherwise: it
surfaces the creatinine value it has, flags whether the label mentions renal
adjustment, and requires the clinician to perform the adjustment.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.module_loader import load_stage_module  # noqa: E402

CLINICIAN_NOTICE = (
    "REFERENCE INFORMATION — CLINICIAN VERIFICATION REQUIRED. This text is "
    "quoted verbatim from the US FDA product label. It is not individualised "
    "for this patient, has not been checked against local formulary or national "
    "guidelines, and does not account for renal or hepatic impairment, weight, "
    "pregnancy, age extremes, allergy or drug interactions. A qualified "
    "prescriber must verify and, where necessary, adjust it."
)


def _drug_api():
    return load_stage_module("10_API_SERVICES", "drug_api")


def renal_context(record: dict) -> dict:
    """What the dataset can and cannot say about renal function."""
    cr = None
    for key in ("last_cr", "median_cr", "first_cr", "creatinine"):
        v = record.get(key)
        if v is not None and v == v:  # not NaN
            cr = float(v)
            break
    ctx = {
        "creatinine_mg_dl": cr,
        "creatinine_source": None if cr is None else "14-day pre-culture lab aggregate",
        "creatinine_clearance_computable": False,
        "why_not": ("Creatinine clearance cannot be computed: Cockcroft-Gault "
                    "requires body weight, and this dataset contains no weight, "
                    "height or exact age (age is banded). The clinician must "
                    "calculate it."),
    }
    if cr is not None:
        ctx["note"] = ("Serum creatinine is shown for context only. Any renal "
                       "dose adjustment must be calculated by the clinician "
                       "using the patient's actual weight and age.")
        if cr >= 1.5:
            ctx["flag"] = ("Serum creatinine is raised. Renal dose adjustment "
                           "is likely to be required — check the label section "
                           "below.")
    return ctx


def get_dosage_reference(antibiotic: str, record: dict | None = None) -> dict:
    """Return label dosage text plus the context needed to read it safely."""
    record = record or {}
    api = _drug_api()
    res = api.get_dosage_reference(antibiotic)

    if not res.ok:
        return {
            "available": False,
            "antibiotic": antibiotic,
            "message": "Insufficient information for dosage reference.",
            "reason": res.reason,
            "source": res.source or "openFDA drug label",
            "source_url": res.source_url,
            "retrieved_at_utc": res.retrieved_at_utc,
            "clinician_notice": CLINICIAN_NOTICE,
        }

    d = res.data
    ctx = renal_context(record)
    missing = []
    if ctx["creatinine_mg_dl"] is None:
        missing.append("renal function (no creatinine available for this patient)")
    if record.get("weight_kg") in (None, "", 0):
        missing.append("body weight")
    if record.get("age") in (None, ""):
        missing.append("exact age (the dataset provides an age band only)")

    return {
        "available": True,
        "antibiotic": antibiotic,
        "generic_name": d.get("generic_name"),
        "brand_name": d.get("brand_name"),
        "manufacturer": d.get("manufacturer"),
        "routes": d.get("route", []),
        "dosage_forms": d.get("dosage_forms", []),
        "dosage_and_administration": d.get("dosage_and_administration"),
        "text_truncated": d.get("text_truncated", False),
        "warnings_excerpt": d.get("warnings_excerpt"),
        "contraindications_excerpt": d.get("contraindications_excerpt"),
        "label_mentions_renal_adjustment": d.get("renal_impairment_mentioned"),
        "label_mentions_hepatic_adjustment": d.get("hepatic_impairment_mentioned"),
        "label_mentions_paediatric": d.get("pediatric_mentioned"),
        "renal_context": ctx,
        "missing_patient_information": missing,
        "individualisation_possible": len(missing) == 0,
        "individualisation_note": (
            "This project does NOT individualise the dose. Even with complete "
            "patient data it would relay the label and leave the calculation to "
            "the prescriber."
            if not missing else
            "Patient information required to interpret this label is missing: "
            + ", ".join(missing) + "."),
        "source": res.source,
        "source_url": res.source_url,
        "retrieved_at_utc": res.retrieved_at_utc,
        "from_cache": res.from_cache,
        "api_disclaimer": res.disclaimer,
        "clinician_notice": CLINICIAN_NOTICE,
    }


if __name__ == "__main__":
    import json
    for abx in ("Ciprofloxacin", "Ertapenem", "Nitrofurantoin"):
        r = get_dosage_reference(abx, {"median_cr": 1.9})
        print(f"\n=== {abx} === available={r['available']}")
        if r["available"]:
            print("  generic:", r["generic_name"],
                  "| renal adj mentioned:", r["label_mentions_renal_adjustment"])
            print("  missing:", r["missing_patient_information"])
            print("  dosage[:200]:", (r["dosage_and_administration"] or "")[:200])
        else:
            print(" ", r["message"], "|", r["reason"])
