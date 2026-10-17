"""Drug information client — openFDA drug label API + RxNorm/RxNav.

VERIFIED ENDPOINTS (checked live during development; see docs/API_INTEGRATION.md)

  1. openFDA drug label
     GET https://api.fda.gov/drug/label.json?search=...&limit=...
     Docs: https://open.fda.gov/apis/drug/label/
     Auth: optional API key (OPENFDA_API_KEY) — raises the rate limit only.
     Returns the FDA-approved Structured Product Label, including the
     `dosage_and_administration` section. This is the ONLY source of dosage text
     used anywhere in this project. Dosage strings are quoted verbatim from the
     label and are never generated, paraphrased, or inferred by the model.

  2. RxNorm / RxNav
     GET https://rxnav.nlm.nih.gov/REST/rxcui.json?name=<drug>
     GET https://rxnav.nlm.nih.gov/REST/rxcui/<rxcui>/allrelated.json
     Docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/
     Auth: none. Used to normalise an antibiotic name to a stable RxCUI.

NOT USED — and why
  * RxNav Drug Interaction API (`/REST/interaction/interaction.json`) was
    retired by the NLM and now returns "Not found" for valid RxCUIs. It is
    therefore NOT integrated; the application shows no interaction data rather
    than data from an unmaintained source. Verified during development.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _http import ApiResult, get_json  # noqa: E402

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
RXNAV_BASE = os.getenv("RXNAV_BASE_URL", "https://rxnav.nlm.nih.gov/REST").rstrip("/")

OPENFDA_DISCLAIMER = (
    "Source: openFDA drug label API (US FDA Structured Product Labeling). "
    "Do not rely on openFDA to make decisions regarding medical care. "
    "Label text is US-specific and may not match local formulary or national "
    "guidelines."
)

# The cohort records antibiotics by short generic name (e.g. "Ertapenem",
# "Sulfamethoxazole-Trimethoprim"). Map the few names whose openFDA generic
# spelling differs. Anything not listed is queried under its own name.
#
# NOTE ON SEPARATORS. This table was originally written with hyphenated keys,
# but the deployed model's antibiotic vocabulary spells combinations with a
# SLASH ("Piperacillin/Tazobactam"). A slash key therefore missed the table and
# was sent to openFDA verbatim, where `openfda.generic_name:"piperacillin/
# tazobactam"` matches nothing - so ten combination agents silently had no
# dosage reference, and `price_api` (which reuses this function) had no NADAC
# match either. Lookup is now separator-agnostic: `/`, `-` and " and " are all
# folded to one space before the table is consulted, so both spellings resolve.
_NAME_OVERRIDES = {
    "sulfamethoxazole-trimethoprim": "sulfamethoxazole and trimethoprim",
    "trimethoprim-sulfamethoxazole": "sulfamethoxazole and trimethoprim",
    "ampicillin-sulbactam": "ampicillin and sulbactam",
    "piperacillin-tazobactam": "piperacillin and tazobactam",
    "amoxicillin-clavulanate": "amoxicillin and clavulanate potassium",
    "amoxicillin/clavulanic acid": "amoxicillin and clavulanate potassium",
    "ticarcillin-clavulanate": "ticarcillin and clavulanate potassium",
    "ceftazidime-avibactam": "ceftazidime and avibactam",
    "ceftolozane-tazobactam": "ceftolozane and tazobactam",
    "imipenem-cilastatin": "imipenem and cilastatin",
    "quinupristin-dalfopristin": "quinupristin and dalfopristin",
    "penicillin": "penicillin g potassium",
    "penicillin g": "penicillin g potassium",
    "oxacillin/nafcillin": "oxacillin",
    # Combination names in the cohort vocabulary that no hyphenated key covered.
    # Each target below was checked against the live openFDA endpoint.
    "cephalexin/cephalothin": "cephalexin",
    "imipenem/ebactam": "imipenem and cilastatin",
    "meropenem/vaborbactam": "meropenem and vaborbactam",
    # "Ticacarcillin" is the cohort's own misspelling of ticarcillin.
    "ticacarcillin/clavulanic acid": "ticarcillin and clavulanate potassium",
}


def _canonical_key(name: str) -> str:
    """Separator-agnostic lookup key.

    "Piperacillin/Tazobactam", "piperacillin-tazobactam" and "piperacillin and
    tazobactam" all reduce to "piperacillin tazobactam", so the override table
    resolves whichever spelling the caller happens to hold.
    """
    key = re.sub(r"\s+", " ", str(name)).strip().lower()
    key = key.replace("/", " ").replace("-", " ")
    key = re.sub(r"\s+and\s+", " ", key)
    return re.sub(r"\s+", " ", key).strip()


_CANONICAL_OVERRIDES = {_canonical_key(k): v for k, v in _NAME_OVERRIDES.items()}


def normalise_drug_name(name: str) -> str:
    """Map a cohort antibiotic label to the name openFDA indexes it under.

    Exact match first (so an explicit entry always wins), then the
    separator-agnostic form. An unmapped name is returned unchanged, which is
    the correct behaviour for the many agents openFDA indexes under their own
    single-word generic name.
    """
    key = re.sub(r"\s+", " ", str(name)).strip().lower()
    if key in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[key]
    return _CANONICAL_OVERRIDES.get(_canonical_key(key), key)


def _with_key(params: dict) -> dict:
    key = os.getenv("OPENFDA_API_KEY", "").strip()
    if key:
        params = dict(params, api_key=key)
    return params


# Route preference. A generic-name search returns every product with that
# ingredient, and for several agents in this cohort the first hit is a topical
# formulation: searching "ciprofloxacin" returns the OPHTHALMIC solution, whose
# dosage section describes drops for corneal ulcers and never mentions renal
# adjustment. For a systemic infection that label is the wrong document, so
# results are re-ranked by route before one is chosen.
SYSTEMIC_ROUTES = ("INTRAVENOUS", "ORAL", "INTRAMUSCULAR")
TOPICAL_ROUTES = ("OPHTHALMIC", "OTIC", "TOPICAL", "NASAL", "RECTAL",
                  "VAGINAL", "DENTAL", "INHALATION", "IRRIGATION")


def _route_score(label: dict) -> int:
    """Lower is better: prefer systemic labels, push topical ones to the back."""
    routes = [str(r).upper() for r in
              (label.get("openfda", {}) or {}).get("route", []) or []]
    if any(r in routes for r in SYSTEMIC_ROUTES):
        # oral/IV first, and prefer a label that is ONLY systemic
        penalty = 1 if any(r in routes for r in TOPICAL_ROUTES) else 0
        return penalty
    if not routes:
        return 5          # unknown route — usable but not preferred
    if any(r in routes for r in TOPICAL_ROUTES):
        return 9          # topical only — almost never what we want
    return 5


def _has_dosage(label: dict) -> int:
    return 0 if label.get("dosage_and_administration") else 1


# --- identity guard --------------------------------------------------------
# When the fielded `openfda.generic_name` query finds nothing, `get_drug_label`
# falls back to an UNFIELDED full-text search. openFDA then matches the tokens
# anywhere in any label, so an unrelated product can come back and — if it is
# oral and carries a dosage section — win the systemic-first ranking outright.
# Observed live during user acceptance testing: "Doripenem" returned LETERMOVIR
# (an antiviral), "Ticarcillin" returned ERTAPENEM, and four combination agents
# returned SILICEA, a homeopathic pellet. Its dosage text ("Take 4 or 6 Pellets
# by mouth") was displayed verbatim as the dosage reference for a carbapenem.
#
# Relaying the wrong drug's label is worse than relaying nothing, so a candidate
# is now accepted only if it actually names the requested ingredient. Nothing
# matching means "Insufficient information for dosage reference." — which is the
# behaviour this project already requires when a value is unavailable.
_IDENTITY_STOPWORDS = {"acid", "sodium", "potassium", "hydrochloride", "and",
                       "injection", "tablet", "capsule", "solution", "for"}


def _identity_tokens(name) -> set:
    """Meaningful ingredient tokens from a drug name."""
    return {t for t in re.split(r"[^a-z]+", str(name or "").lower())
            if len(t) >= 5 and t not in _IDENTITY_STOPWORDS}


def _label_identity(label: dict) -> set:
    of = label.get("openfda", {}) or {}
    toks: set = set()
    for key in ("generic_name", "brand_name", "substance_name"):
        for v in (of.get(key) or []):
            toks |= _identity_tokens(v)
    return toks


def _label_matches_drug(label: dict, *names) -> bool:
    """True if the label names at least one ingredient the caller asked for."""
    wanted: set = set()
    for n in names:
        wanted |= _identity_tokens(n)
    if not wanted:
        return True          # nothing to check against - do not block
    got = _label_identity(label)
    if not got:
        return False         # unidentifiable label - do not trust it
    return bool(wanted & got)


def get_drug_label(antibiotic: str, prefer_systemic: bool = True) -> ApiResult:
    """Fetch the FDA label record for an antibiotic (generic-name search).

    Several candidate labels are retrieved and re-ranked so that a systemic
    (oral / IV / IM) product with an actual dosage section is chosen over a
    topical one.
    """
    generic = normalise_drug_name(antibiotic)
    params = _with_key({
        "search": f'openfda.generic_name:"{generic}"',
        "limit": 25,
    })
    res = get_json(OPENFDA_LABEL_URL, params, source="openFDA drug label")
    if not res.ok:
        # Fall back to a broader full-text search before giving up.
        res = get_json(OPENFDA_LABEL_URL,
                       _with_key({"search": generic, "limit": 25}),
                       source="openFDA drug label")
    if not res.ok:
        res.reason = res.reason or f"No FDA label found for '{antibiotic}'."
        return res

    results = (res.data or {}).get("results") or []
    if not results:
        return ApiResult(ok=False, source=res.source, source_url=res.source_url,
                         retrieved_at_utc=res.retrieved_at_utc,
                         reason=f"No FDA label found for '{antibiotic}'.")

    # Discard labels that are not actually this drug before ranking anything.
    n_returned = len(results)
    results = [l for l in results if _label_matches_drug(l, generic, antibiotic)]
    if not results:
        return ApiResult(
            ok=False, source=res.source, source_url=res.source_url,
            retrieved_at_utc=res.retrieved_at_utc,
            reason=(f"openFDA returned {n_returned} label(s) for "
                    f"'{antibiotic}', none of which name that ingredient. "
                    f"No dosage reference is shown rather than text from "
                    f"another product."))

    if prefer_systemic:
        ranked = sorted(results, key=lambda l: (_has_dosage(l), _route_score(l)))
    else:
        ranked = sorted(results, key=_has_dosage)
    chosen = ranked[0]

    res.data = chosen
    res.disclaimer = OPENFDA_DISCLAIMER
    chosen_routes = (chosen.get("openfda", {}) or {}).get("route", []) or []
    res.notes.append(
        f"Queried openFDA generic_name = '{generic}'; {n_returned} labels "
        f"returned, {len(results)} confirmed to name this ingredient, selected "
        f"route(s) {chosen_routes or 'unspecified'} out of a systemic-first "
        f"ranking.")
    return res


def _first_section(label: dict, *keys) -> str | None:
    for k in keys:
        val = label.get(k)
        if isinstance(val, list) and val:
            return "\n\n".join(str(v).strip() for v in val if str(v).strip())
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def get_dosage_reference(antibiotic: str, max_chars: int = 4000) -> ApiResult:
    """Return the VERBATIM `dosage_and_administration` section from the label.

    This function never composes a dose. It only relays label text, its source
    URL and its retrieval time. If the label has no dosage section, `ok=False`
    and the caller must display "Insufficient information for dosage reference."
    """
    res = get_drug_label(antibiotic)
    if not res.ok:
        return res

    label = res.data or {}
    text = _first_section(label,
                          "dosage_and_administration",
                          "dosage_and_administration_table")
    if not text:
        return ApiResult(
            ok=False, source=res.source, source_url=res.source_url,
            retrieved_at_utc=res.retrieved_at_utc, from_cache=res.from_cache,
            reason="The FDA label for this product contains no "
                   "dosage_and_administration section.")

    openfda = label.get("openfda", {}) or {}
    truncated = len(text) > max_chars
    payload = {
        "antibiotic_queried": antibiotic,
        "generic_name": (openfda.get("generic_name") or [None])[0],
        "brand_name": (openfda.get("brand_name") or [None])[0],
        "manufacturer": (openfda.get("manufacturer_name") or [None])[0],
        "route": openfda.get("route") or [],
        "dosage_forms": openfda.get("dosage_form") or [],
        "spl_id": label.get("id"),
        "effective_time": label.get("effective_time"),
        "dosage_and_administration": text[:max_chars] + ("…" if truncated else ""),
        "text_truncated": truncated,
        "renal_impairment_mentioned": bool(
            re.search(r"renal|creatinine|CrCl|dialysis", text, re.I)),
        "hepatic_impairment_mentioned": bool(
            re.search(r"hepatic|liver", text, re.I)),
        "pediatric_mentioned": bool(
            re.search(r"pediatric|paediatric|children", text, re.I)),
        "warnings_excerpt": (_first_section(label, "warnings_and_cautions",
                                            "warnings", "boxed_warning")
                             or "")[:1200] or None,
        "contraindications_excerpt": (_first_section(label, "contraindications")
                                      or "")[:800] or None,
    }
    res.data = payload
    res.disclaimer = OPENFDA_DISCLAIMER
    return res


def get_rxcui(antibiotic: str) -> ApiResult:
    """Resolve an antibiotic name to an RxNorm concept unique identifier."""
    generic = normalise_drug_name(antibiotic)
    res = get_json(f"{RXNAV_BASE}/rxcui.json", {"name": generic},
                   source="RxNorm / RxNav (NLM)")
    if not res.ok:
        return res
    ids = ((res.data or {}).get("idGroup") or {}).get("rxnormId") or []
    if not ids:
        return ApiResult(ok=False, source=res.source, source_url=res.source_url,
                         retrieved_at_utc=res.retrieved_at_utc,
                         reason=f"RxNorm has no concept named '{generic}'.")
    res.data = {"rxcui": ids[0], "all_rxcui": ids, "queried_name": generic}
    res.disclaimer = ("Source: RxNorm / RxNav, U.S. National Library of Medicine. "
                      "Provided for terminology normalisation only.")
    return res


if __name__ == "__main__":  # smoke test
    for abx in ("Ciprofloxacin", "Ertapenem", "Sulfamethoxazole-Trimethoprim"):
        d = get_dosage_reference(abx)
        r = get_rxcui(abx)
        print(f"{abx:35s} label_ok={d.ok!s:5s} rxcui_ok={r.ok!s:5s} "
              f"{'' if d.ok else d.reason}")
        if d.ok:
            print("   generic:", d.data["generic_name"],
                  "| dosage chars:", len(d.data["dosage_and_administration"]))
