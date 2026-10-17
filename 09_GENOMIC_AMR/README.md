# 09_GENOMIC_AMR — Gene-based AMR module

## Status: IMPLEMENTED BUT INACTIVE

**The datasets in this project contain no genomic data.** All five source files
were audited column by column (see [`../docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md));
none contains a sequence, an assembly accession, a gene name, a mutation call,
an AMR marker, or any genotype field. The available microbiology signal is
strictly phenotypic: organism identity plus a disc/MIC-derived susceptibility
category.

Consequently:

- **No genomic prediction is made anywhere in this project.**
- The resistance model contains **no genomic features**.
- The application's Genomic tab states the absence plainly rather than hiding
  the tab or filling it with plausible-looking output.

This is verified by tests
(`12_TESTS/test_recommendation_and_services.py::test_genomic_module_reports_unavailable_for_this_dataset`)
and by the system verification run (`docs/SYSTEM_VERIFICATION.md`, Phase 9).

## What the module does provide

### 1. An honest availability check

```python
from src.utils.module_loader import load_stage_module
gen = load_stage_module("09_GENOMIC_AMR", "genomic_analysis")
gen.availability_report()
# {'available': False, 'reason': 'No genomic data is present ...', ...}
```

The application calls this before rendering anything, so the absence is
reported rather than inferred.

### 2. A curated marker → mechanism → implication map

21 well-established AMR markers, each mapped to its gene family, resistance
mechanism, affected antibiotic classes, typical organisms and a
genotype–phenotype confidence rating.

```python
gen.annotate_markers(["blaKPC-2", "mecA", "vanA"])
```

**Provenance.** Curated from the public AMR reference databases:

- NCBI Reference Gene Catalog / AMRFinderPlus —
  <https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/>
- CARD, the Comprehensive Antibiotic Resistance Database —
  <https://card.mcmaster.ca/>
- ResFinder (CGE, DTU) — <https://cge.food.dtu.dk/services/ResFinder/>

It is a **demonstration subset for a research prototype**, not a substitute for
running a validated AMR-detection pipeline, and it is labelled as such wherever
it is displayed.

### 3. Evidence framing, never a verdict

Gene presence does not equal phenotypic resistance — expression, copy number,
promoter mutations, efflux and porin loss all modulate it, and
genotype–phenotype concordance varies widely by gene and organism. Every
annotation is therefore phrased:

> *"Supports clinician consideration of reduced reliability for: … Genotype is
> supporting evidence only and does not establish phenotypic resistance —
> confirm with susceptibility testing."*

A test asserts that no annotation ever says "do not use" or "must be avoided".
Unrecognised markers return `gene_family: "unrecognised"` with an empty
implication rather than a guess.

## How to activate the module with real genotype data

No code change is required. Provide a CSV with at least these two columns:

| Column | Meaning |
|---|---|
| `order_proc_id_coded` | The culture order the isolate came from — the same join key the rest of the pipeline uses |
| `marker` | One detected marker per row (e.g. AMRFinderPlus / ResFinder / CARD-RGI output) |

Place it at:

```
01_DATA/raw/genotypes.csv
```

or point the `GENOTYPE_TABLE_PATH` environment variable at it. The module and
the application then activate automatically.

```csv
order_proc_id_coded,marker
928257722,blaCTX-M-15
928257722,aac(6')-Ib-cr
697566032,blaKPC-2
```

This plug-in path is **verified working**: the system verification run drops a
temporary genotype table in, confirms the module activates and annotates, then
removes it and confirms the module returns to INACTIVE.

## Architectural note

The module was built now, while there is no data for it, specifically so that
adding genomic data later is a data task rather than a redesign. Genotype
information enters as an *additional evidence layer* alongside the phenotypic
model — it does not replace the model, and it does not alter a predicted
probability. That mirrors how the clinical guardrails in
`08_RECOMMENDATION_ENGINE/` work.

## Files

| File | Purpose |
|---|---|
| `genomic_analysis.py` | Availability check, marker reference map, annotation and per-order lookup |
| `README.md` | This document |
