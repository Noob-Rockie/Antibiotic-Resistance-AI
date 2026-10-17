# 10_API_SERVICES — External service clients

Three real, publicly documented services. Each was verified live during
development, and each is re-verified by `scripts/17_verify_system.py`
(see [`../docs/SYSTEM_VERIFICATION.md`](../docs/SYSTEM_VERIFICATION.md), Phase 8).

## The services in use

| Service | Endpoint | Auth | Used for | Client |
|---|---|---|---|---|
| **openFDA drug label** | `https://api.fda.gov/drug/label.json` | Optional key (raises rate limit only) | Verbatim `dosage_and_administration`, warnings, contraindications | `drug_api.py` |
| **RxNorm / RxNav** (NLM) | `https://rxnav.nlm.nih.gov/REST` | None | Normalising a drug name to a stable RxCUI | `drug_api.py` |
| **CMS NADAC** | `https://data.medicaid.gov/api/1/datastore/query/{dataset}/0` | None | Unit acquisition cost per NDC | `price_api.py` |

Documentation:
[openFDA](https://open.fda.gov/apis/drug/label/) ·
[RxNav](https://lhncbc.nlm.nih.gov/RxNav/APIs/) ·
[data.medicaid.gov](https://data.medicaid.gov/api-docs)

## Deliberately NOT integrated

**RxNav Drug Interaction API** (`/REST/interaction/interaction.json`). The NLM
retired it, and it was verified during development to return `Not found` for
valid RxCUIs. The application therefore shows **no** drug-interaction data
rather than data from an unmaintained source. This is enforced by a consistency
check: the endpoint may only appear in the codebase under a `NOT USED` heading.

No other API is referenced. A consistency check asserts that every URL host
appearing in this folder is one of the verified real endpoints above.

## Design rules, all enforced in `_http.py`

Every outbound call goes through one function, `get_json()`, so these
properties hold uniformly:

1. **Never raises into the caller.** Network errors, 404, 429, non-JSON
   responses and timeouts all return `ApiResult(ok=False, reason=...)`. A
   failing lookup degrades the UI; it never breaks a prediction.
2. **Always has a timeout** (12 s default).
3. **Always carries provenance.** Every result records `source`,
   `source_url` and `retrieved_at_utc`, and the application displays them
   beside the value. No number is ever shown without its source.
4. **Disk-cached** under `01_DATA/interim/api_cache/`, keyed on the URL plus
   parameters, with a TTL from `API_CACHE_TTL_HOURS` (default 24 h). A corrupt
   cache entry falls through to a live fetch rather than failing.
5. **No credential in code.** Keys come from the environment only. A test and a
   consistency check both scan for hard-coded credentials.
6. **One global offline switch.** `ENABLE_LIVE_API=0` disables all network I/O;
   clients then return an explicit "live API disabled" reason and the
   application renders its fallback messages. The whole app is verified to run
   in this mode.

## The `ApiResult` envelope

```python
@dataclass
class ApiResult:
    ok: bool                 # False is a NORMAL state, not an error
    data: Any = None
    source: str = ""         # human-readable service name
    source_url: str = ""     # the exact URL that was called
    retrieved_at_utc: str = ""
    from_cache: bool = False
    reason: str = ""         # why ok is False
    disclaimer: str = ""     # the service's own terms, shown in the UI
```

## A real bug worth recording

A generic-name search on openFDA returns **every** product with that
ingredient, and for several agents in this cohort the first hit is a topical
formulation. Searching `ciprofloxacin` returned the **ophthalmic** solution,
whose dosage section describes drops for corneal ulcers and never mentions
renal adjustment — the wrong document entirely for a systemic infection.

`get_drug_label()` now retrieves up to 25 candidate labels and re-ranks them so
a systemic product (oral / IV / IM) with an actual dosage section is selected,
pushing topical-only labels to the back. A verification check asserts that the
returned label's route is systemic.

## What these services are NOT used for

- **Dosage is relayed, never computed.** `08_RECOMMENDATION_ENGINE/dosage_reference.py`
  quotes the FDA label section verbatim and adds nothing. Creatinine clearance
  is not calculated, because Cockcroft-Gault needs a body weight and this
  dataset has none.
- **Cost is retrieved, never estimated.** NADAC is a US pharmacy *acquisition*
  cost per pricing unit — not a retail price, not a co-pay, not a course cost,
  and not applicable outside the United States. No course cost is calculated,
  because that would require a dose the project deliberately does not generate.
- **Neither ever influences a model prediction.** They are display-layer
  reference information for a clinician.

## Files

| File | Purpose |
|---|---|
| `_http.py` | Shared HTTP plumbing: `ApiResult`, caching, timeouts, offline switch |
| `drug_api.py` | openFDA drug label + RxNorm/RxNav, with systemic-route ranking |
| `price_api.py` | CMS NADAC dataset discovery and price lookup |
| `README.md` | This document |

## Configuration

Copy `.env.example` to `.env` at the project root:

```bash
OPENFDA_API_KEY=            # optional; raises the rate limit only
RXNAV_BASE_URL=https://rxnav.nlm.nih.gov/REST
ENABLE_LIVE_API=1           # set to 0 to run entirely offline
API_CACHE_TTL_HOURS=24
```

`.env` is gitignored. Never commit it.
