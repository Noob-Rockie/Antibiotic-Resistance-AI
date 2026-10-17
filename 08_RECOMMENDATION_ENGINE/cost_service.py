"""Medicine cost — retrieved from CMS NADAC, never estimated.

THE RULE
Every price shown originates from a live CMS NADAC record with its own NDC,
effective date and retrieval timestamp. If NADAC has no record for an agent, the
service returns `available=False` and the application prints:

    "Price unavailable — verify locally."

It never interpolates from a similar drug, never converts a unit price into a
course cost, and never displays a number without its source.

WHAT NADAC IS, AND IS NOT
NADAC is the US National Average Drug Acquisition Cost: a weekly survey-based
estimate of what retail community pharmacies pay to *acquire* a drug, published
by the Centers for Medicare & Medicaid Services. It is:
  * per PRICING UNIT (each tablet, each mL, each gram) — not per course;
  * an acquisition cost — not a retail price, insurance cost or patient co-pay;
  * United States specific — it says nothing about prices anywhere else.

Because a course cost depends on dose, frequency and duration — none of which
this project generates (see `dosage_reference.py`) — no course cost is
calculated. Displaying unit cost with an explicit basis is the honest maximum.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.module_loader import load_stage_module  # noqa: E402

UNAVAILABLE_MESSAGE = "Price unavailable — verify locally."

COST_NOTICE = (
    "Cost is shown for reference only and must never be the primary basis for "
    "choosing an antibiotic. NADAC is a US pharmacy acquisition cost per "
    "pricing unit, not a course cost and not a patient-facing price; it does "
    "not apply outside the United States. A cheaper agent that fails costs far "
    "more than an expensive one that works."
)


def _price_api():
    return load_stage_module("10_API_SERVICES", "price_api")


def get_cost(antibiotic: str, prefer_oral: bool = False) -> dict:
    res = _price_api().get_price(antibiotic)
    if not res.ok:
        return {
            "available": False,
            "antibiotic": antibiotic,
            "message": UNAVAILABLE_MESSAGE,
            "reason": res.reason,
            "source": res.source or "CMS NADAC (data.medicaid.gov)",
            "source_url": res.source_url,
            "retrieved_at_utc": res.retrieved_at_utc,
            "cost_notice": COST_NOTICE,
        }

    d = res.data
    products = d["products"]
    oral_units = {"EA"}
    shown = ([p for p in products if p["pricing_unit"] in oral_units] or products) \
        if prefer_oral else products

    return {
        "available": True,
        "antibiotic": antibiotic,
        "n_products_matched": d["n_products_matched"],
        "n_products_shown": len(shown),
        "products": shown[:25],
        "cheapest": shown[0] if shown else None,
        "summary_by_pricing_unit": d["summary_by_pricing_unit"],
        "price_basis": d["price_basis"],
        "currency": "USD",
        "course_cost_calculated": False,
        "course_cost_note": (
            "No course cost is calculated. A course cost requires a dose, "
            "frequency and duration, and this project deliberately does not "
            "generate any of those — it only relays FDA label text for a "
            "clinician to interpret."),
        "source": res.source,
        "source_url": res.source_url,
        "retrieved_at_utc": res.retrieved_at_utc,
        "from_cache": res.from_cache,
        "api_disclaimer": res.disclaimer,
        "cost_notice": COST_NOTICE,
    }


def get_costs(antibiotics) -> dict:
    return {a: get_cost(a) for a in antibiotics}


if __name__ == "__main__":
    for abx in ("Ciprofloxacin", "Ertapenem", "Ceftazidime/Avibactam",
                "Nitrofurantoin"):
        r = get_cost(abx)
        if r["available"]:
            c = r["cheapest"]
            print(f"{abx:26s} cheapest ${c['price_per_unit_usd']:.5f} per "
                  f"{c['pricing_unit']}  ({c['description']})  "
                  f"eff. {c['effective_date']}")
        else:
            print(f"{abx:26s} {r['message']}  ({r['reason']})")
