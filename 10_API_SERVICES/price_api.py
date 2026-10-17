"""Medicine cost client — CMS NADAC via the data.medicaid.gov datastore API.

VERIFIED ENDPOINT (checked live during development)

  GET https://data.medicaid.gov/api/1/datastore/query/<datasetId>/0
      ?limit=..&conditions[0][property]=ndc_description
      &conditions[0][value]=<DRUG>%25&conditions[0][operator]=like

  Dataset discovery:
  GET https://data.medicaid.gov/api/1/search/?fulltext=NADAC&page-size=20

  Docs:  https://data.medicaid.gov/api-docs
  Auth:  none required.
  Data:  NADAC = National Average Drug Acquisition Cost. A weekly survey-based
         US national average of what retail pharmacies pay to acquire a drug,
         published by CMS. Price is per *pricing unit* (EA / ML / GM), NOT per
         course of therapy and NOT a patient-facing retail price.

WHAT THIS MODULE WILL NOT DO
  * It never estimates, converts, or extrapolates a price.
  * It never presents a NADAC unit cost as a treatment cost.
  * If no NADAC record exists for a drug, the caller must display
    "Price unavailable - verify locally." No placeholder number is produced.
"""
from __future__ import annotations

import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _http import ApiResult, get_json  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module  # noqa: E402

MEDICAID_SEARCH = "https://data.medicaid.gov/api/1/search/"
DATASTORE = "https://data.medicaid.gov/api/1/datastore/query/{ds}/0"

NADAC_DISCLAIMER = (
    "Source: CMS National Average Drug Acquisition Cost (NADAC), "
    "data.medicaid.gov. NADAC is a US pharmacy ACQUISITION cost per pricing "
    "unit from a weekly national survey. It is not a retail price, not a "
    "patient co-pay, not a total course cost, and is not applicable outside "
    "the United States. Verify cost locally before it influences any "
    "prescribing decision."
)

_DATASET_CACHE: dict[str, str] = {}


def _drug_api():
    """Lazy import of the sibling module (avoids a circular import at load)."""
    return import_module("drug_api") if "drug_api" in sys.modules else \
        __import__("drug_api")


def find_current_nadac_dataset() -> ApiResult:
    """Discover the most recent yearly NADAC dataset id from the CMS catalogue."""
    if "current" in _DATASET_CACHE:
        return ApiResult(ok=True, data=_DATASET_CACHE["current"],
                         source="data.medicaid.gov catalogue",
                         from_cache=True,
                         retrieved_at_utc=datetime.now(timezone.utc).isoformat(
                             timespec="seconds"))

    res = get_json(MEDICAID_SEARCH, {"fulltext": "NADAC", "page-size": 30},
                   source="data.medicaid.gov catalogue")
    if not res.ok:
        return res

    best_year, best_id = None, None
    for item in ((res.data or {}).get("results") or {}).values():
        title = item.get("title", "")
        if not title.startswith("NADAC (National Average Drug Acquisition Cost)"):
            continue
        year_token = title.rsplit(" ", 1)[-1]
        if year_token.isdigit():
            year = int(year_token)
            if best_year is None or year > best_year:
                best_year, best_id = year, item.get("identifier")

    if best_id is None:
        return ApiResult(ok=False, source=res.source, source_url=res.source_url,
                         retrieved_at_utc=res.retrieved_at_utc,
                         reason="No yearly NADAC dataset found in the CMS "
                                "catalogue.")
    _DATASET_CACHE["current"] = best_id
    res.data = best_id
    res.notes.append(f"Using NADAC {best_year} dataset {best_id}.")
    return res


def get_price(antibiotic: str, limit: int = 60) -> ApiResult:
    """Look up NADAC unit prices for an antibiotic by NDC description prefix."""
    try:
        drug_api = _drug_api()
        search_term = drug_api.normalise_drug_name(antibiotic)
    except Exception:
        search_term = str(antibiotic).strip().lower()
    # NADAC descriptions are single-ingredient strings; use the first token of a
    # combination name so e.g. "sulfamethoxazole and trimethoprim" still matches.
    search_term = search_term.split(" and ")[0].split("-")[0].strip().upper()

    ds = find_current_nadac_dataset()
    if not ds.ok:
        return ds

    params = {
        "limit": limit,
        "conditions[0][property]": "ndc_description",
        "conditions[0][value]": f"{search_term}%",
        "conditions[0][operator]": "like",
    }
    res = get_json(DATASTORE.format(ds=ds.data), params,
                   source="CMS NADAC (data.medicaid.gov)")
    if not res.ok:
        return res

    rows = (res.data or {}).get("results") or []
    if not rows:
        return ApiResult(ok=False, source=res.source, source_url=res.source_url,
                         retrieved_at_utc=res.retrieved_at_utc,
                         reason=f"No NADAC record for '{antibiotic}'.")

    products = []
    for r in rows:
        try:
            unit_price = float(r["nadac_per_unit"])
        except (KeyError, TypeError, ValueError):
            continue
        products.append({
            "description": r.get("ndc_description"),
            "ndc": r.get("ndc"),
            "price_per_unit_usd": round(unit_price, 5),
            "pricing_unit": r.get("pricing_unit"),
            "effective_date": r.get("effective_date"),
            "as_of_date": r.get("as_of_date"),
            "otc": r.get("otc"),
            "generic_or_brand": {"G": "Generic", "B": "Brand"}.get(
                r.get("classification_for_rate_setting"),
                r.get("classification_for_rate_setting")),
        })
    if not products:
        return ApiResult(ok=False, source=res.source, source_url=res.source_url,
                         retrieved_at_utc=res.retrieved_at_utc,
                         reason=f"NADAC returned rows for '{antibiotic}' but "
                                f"none carried a usable unit price.")

    products.sort(key=lambda p: p["price_per_unit_usd"])
    by_unit: dict[str, list] = {}
    for p in products:
        by_unit.setdefault(p["pricing_unit"] or "?", []).append(
            p["price_per_unit_usd"])

    res.data = {
        "antibiotic_queried": antibiotic,
        "nadac_search_term": search_term,
        "n_products_matched": (res.data or {}).get("count", len(products)),
        "n_products_returned": len(products),
        "products": products,
        "summary_by_pricing_unit": {
            unit: {
                "n": len(v),
                "min_usd": round(min(v), 5),
                "median_usd": round(statistics.median(v), 5),
                "max_usd": round(max(v), 5),
            } for unit, v in by_unit.items()
        },
        "price_basis": "USD per pricing unit (acquisition cost, not retail)",
    }
    res.disclaimer = NADAC_DISCLAIMER
    return res


if __name__ == "__main__":  # smoke test
    for abx in ("Ciprofloxacin", "Ertapenem", "Vancomycin", "Nitrofurantoin"):
        r = get_price(abx)
        if r.ok:
            s = r.data["summary_by_pricing_unit"]
            print(f"{abx:20s} OK  {r.data['n_products_returned']} products  {s}")
        else:
            print(f"{abx:20s} --  {r.reason}")
