"""Gene-based AMR module.

STATUS ON THE CURRENT DATASET: INACTIVE — no genomic data exists.

The five source files were audited column by column (see docs/DATA_DICTIONARY.md).
None of them contains a sequence, an assembly accession, a gene name, a mutation
call, an AMR marker, or any genotype field. The available microbiology signal is
strictly phenotypic: organism identity plus a disc/MIC-derived susceptibility
category. Therefore this project makes NO genomic predictions, and the resistance
model contains NO genomic features.

WHAT THIS MODULE PROVIDES INSTEAD

  1. `genomic_data_available()` — a single, honest availability check that the
     application calls before rendering the Genomic tab. It returns False for the
     current dataset, and the UI states that plainly.

  2. A plug-in point (`load_genotype_table`) so a future genotype export — an
     AMRFinderPlus / ResFinder / CARD-RGI result table keyed by the same
     `order_proc_id_coded` — can be dropped in without touching any other layer.

  3. `annotate_markers()` — when genotype data IS supplied, it maps each detected
     marker to its resistance mechanism and the antibiotic classes that mechanism
     is known to affect, and returns that as EVIDENCE for a clinician.

CRITICAL SEMANTICS: the annotation is evidence, not a verdict. Gene presence does
not equal phenotypic resistance — expression, copy number, promoter mutations,
efflux and porin loss all modulate it, and genotype-phenotype concordance varies
widely by gene and organism. Every row returned by this module is labelled
`"implication": "supports consideration of ..."` and never
`"do not use antibiotic X"`. Rendering code must preserve that framing.

REFERENCE MAP PROVENANCE
The marker map below encodes well-established, textbook resistance mechanisms as
documented by the public AMR reference databases:
  * NCBI Reference Gene Catalog / AMRFinderPlus
    https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/
  * CARD — Comprehensive Antibiotic Resistance Database
    https://card.mcmaster.ca/
  * ResFinder (CGE, DTU)
    https://cge.food.dtu.dk/services/ResFinder/
It is a small curated subset intended for prototype demonstration, NOT a
substitute for running a validated AMR-detection pipeline, and it is flagged as
such wherever it is displayed.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# A genotype table would live here. Absent for the current dataset.
GENOTYPE_PATH_ENV = "GENOTYPE_TABLE_PATH"
DEFAULT_GENOTYPE_PATH = PROJECT_ROOT / "01_DATA" / "raw" / "genotypes.csv"

REFERENCE_NOTE = (
    "Curated from the NCBI Reference Gene Catalog, CARD and ResFinder. "
    "Demonstration subset for a research prototype — not a validated AMR "
    "detection pipeline. Genotype does not equal phenotype; confirm with "
    "antimicrobial susceptibility testing."
)


@dataclass
class MarkerAnnotation:
    marker: str
    gene_family: str
    mechanism: str
    affected_classes: list
    typical_organisms: list
    implication: str
    confidence: str
    reference: str

    def to_dict(self) -> dict:
        return asdict(self)


# marker -> (gene family, mechanism, affected antibiotic classes,
#            typical organisms, genotype-phenotype confidence)
MARKER_MAP: dict[str, dict] = {
    "meca": dict(
        family="mecA / SCCmec", confidence="high",
        mechanism="Encodes PBP2a, a penicillin-binding protein with low "
                  "affinity for beta-lactams; cell-wall synthesis continues "
                  "while beta-lactams are bound.",
        classes=["Penicillins", "Anti-staphylococcal penicillins",
                 "Cephalosporins (except anti-MRSA agents)", "Carbapenems"],
        organisms=["Staphylococcus aureus", "Staphylococcus epidermidis"],
        reference="NCBI Reference Gene Catalog; CARD ARO:3000617"),
    "mecc": dict(
        family="mecC / SCCmec XI", confidence="high",
        mechanism="mecA homologue encoding a divergent PBP2a; same beta-lactam "
                  "evasion mechanism, frequently missed by mecA-only PCR.",
        classes=["Penicillins", "Cephalosporins (except anti-MRSA agents)",
                 "Carbapenems"],
        organisms=["Staphylococcus aureus"],
        reference="NCBI Reference Gene Catalog"),
    "vana": dict(
        family="van operon (VanA)", confidence="high",
        mechanism="Remodels the peptidoglycan precursor terminus from D-Ala-D-Ala "
                  "to D-Ala-D-Lac, collapsing glycopeptide binding affinity.",
        classes=["Glycopeptides (vancomycin)", "Lipoglycopeptides (teicoplanin)"],
        organisms=["Enterococcus faecium", "Enterococcus faecalis"],
        reference="CARD ARO:3000010; ResFinder"),
    "vanb": dict(
        family="van operon (VanB)", confidence="high",
        mechanism="D-Ala-D-Lac precursor substitution, inducible by vancomycin; "
                  "teicoplanin activity is often retained.",
        classes=["Glycopeptides (vancomycin)"],
        organisms=["Enterococcus faecium", "Enterococcus faecalis"],
        reference="CARD ARO:3000013"),
    "blakpc": dict(
        family="KPC carbapenemase (Ambler class A)", confidence="high",
        mechanism="Serine carbapenemase hydrolysing essentially all beta-lactams "
                  "including carbapenems; inhibited by avibactam and vaborbactam.",
        classes=["Penicillins", "Cephalosporins", "Carbapenems", "Monobactams"],
        organisms=["Klebsiella pneumoniae", "Escherichia coli",
                   "Enterobacter cloacae"],
        reference="NCBI Reference Gene Catalog; CARD ARO:3000059"),
    "blandm": dict(
        family="NDM metallo-beta-lactamase (Ambler class B)", confidence="high",
        mechanism="Zinc-dependent metallo-carbapenemase; hydrolyses carbapenems "
                  "and is NOT inhibited by avibactam, vaborbactam or relebactam. "
                  "Aztreonam is not a substrate but is usually co-resisted.",
        classes=["Penicillins", "Cephalosporins", "Carbapenems"],
        organisms=["Klebsiella pneumoniae", "Escherichia coli",
                   "Acinetobacter baumannii"],
        reference="CARD ARO:3000589"),
    "blaoxa48": dict(
        family="OXA-48-like carbapenemase (Ambler class D)", confidence="moderate",
        mechanism="Weak carbapenemase with strong penicillinase activity; "
                  "carbapenem MICs are often only modestly raised, so phenotypic "
                  "detection is unreliable in isolation.",
        classes=["Penicillins", "Carbapenems (variable)"],
        organisms=["Klebsiella pneumoniae", "Escherichia coli"],
        reference="CARD ARO:3001068"),
    "blavim": dict(
        family="VIM metallo-beta-lactamase (Ambler class B)", confidence="high",
        mechanism="Zinc-dependent metallo-carbapenemase, not inhibited by "
                  "serine-beta-lactamase inhibitors.",
        classes=["Penicillins", "Cephalosporins", "Carbapenems"],
        organisms=["Pseudomonas aeruginosa", "Klebsiella pneumoniae"],
        reference="CARD ARO:3000578"),
    "blaimp": dict(
        family="IMP metallo-beta-lactamase (Ambler class B)", confidence="high",
        mechanism="Zinc-dependent metallo-carbapenemase.",
        classes=["Penicillins", "Cephalosporins", "Carbapenems"],
        organisms=["Pseudomonas aeruginosa", "Enterobacterales"],
        reference="CARD ARO:3000020"),
    "blactxm": dict(
        family="CTX-M extended-spectrum beta-lactamase", confidence="high",
        mechanism="ESBL hydrolysing oxyimino-cephalosporins; inhibited by "
                  "clavulanate and tazobactam. Carbapenems generally retain "
                  "activity in the absence of a carbapenemase.",
        classes=["Penicillins", "3rd/4th-generation cephalosporins",
                 "Monobactams"],
        organisms=["Escherichia coli", "Klebsiella pneumoniae"],
        reference="CARD ARO:3001989"),
    "blashv": dict(
        family="SHV beta-lactamase", confidence="moderate",
        mechanism="Broad-spectrum or extended-spectrum beta-lactamase depending "
                  "on the allele; allele-level typing is required.",
        classes=["Penicillins", "Cephalosporins (allele-dependent)"],
        organisms=["Klebsiella pneumoniae"],
        reference="CARD ARO:3001070"),
    "blatem": dict(
        family="TEM beta-lactamase", confidence="moderate",
        mechanism="Classically a narrow-spectrum penicillinase; specific point "
                  "mutations confer an extended spectrum.",
        classes=["Penicillins", "Cephalosporins (allele-dependent)"],
        organisms=["Escherichia coli", "Haemophilus influenzae"],
        reference="CARD ARO:3000873"),
    "blaampc": dict(
        family="AmpC cephalosporinase", confidence="moderate",
        mechanism="Inducible or derepressed class C cephalosporinase; not "
                  "inhibited by clavulanate. Cefepime and carbapenems are "
                  "usually stable.",
        classes=["Penicillins", "Cephamycins",
                 "3rd-generation cephalosporins"],
        organisms=["Enterobacter cloacae", "Citrobacter freundii",
                   "Serratia marcescens"],
        reference="CARD ARO:3000076"),
    "erm": dict(
        family="erm rRNA methyltransferase", confidence="high",
        mechanism="Methylates 23S rRNA at A2058, blocking macrolide, "
                  "lincosamide and streptogramin B binding (MLSb phenotype); "
                  "may be inducible (D-test positive).",
        classes=["Macrolides", "Lincosamides (clindamycin)",
                 "Streptogramin B"],
        organisms=["Staphylococcus aureus", "Streptococcus pneumoniae"],
        reference="CARD ARO:3000560"),
    "qnr": dict(
        family="Qnr pentapeptide repeat protein", confidence="moderate",
        mechanism="Plasmid-mediated quinolone resistance; shields DNA gyrase "
                  "from fluoroquinolones. Usually raises MIC without reaching "
                  "the clinical resistance breakpoint on its own.",
        classes=["Fluoroquinolones"],
        organisms=["Escherichia coli", "Klebsiella pneumoniae"],
        reference="CARD ARO:3000419"),
    "gyra": dict(
        family="gyrA quinolone-resistance-determining region mutation",
        confidence="high",
        mechanism="Target-site substitution (commonly S83L / D87N in E. coli) "
                  "reducing fluoroquinolone binding to DNA gyrase.",
        classes=["Fluoroquinolones"],
        organisms=["Escherichia coli", "Salmonella spp.",
                   "Pseudomonas aeruginosa"],
        reference="CARD ARO:3003288"),
    "sul1": dict(
        family="sul dihydropteroate synthase", confidence="high",
        mechanism="Alternative sulfonamide-insensitive dihydropteroate "
                  "synthase; often carried on class 1 integrons alongside other "
                  "resistance cassettes.",
        classes=["Sulfonamides"],
        organisms=["Enterobacterales"],
        reference="CARD ARO:3000410"),
    "dfra": dict(
        family="dfrA dihydrofolate reductase", confidence="high",
        mechanism="Trimethoprim-insensitive dihydrofolate reductase.",
        classes=["Trimethoprim"],
        organisms=["Enterobacterales"],
        reference="CARD ARO:3002858"),
    "tetm": dict(
        family="tet(M) ribosomal protection protein", confidence="high",
        mechanism="Ribosomal protection protein that dislodges tetracycline "
                  "from the 30S subunit.",
        classes=["Tetracyclines"],
        organisms=["Streptococcus spp.", "Enterococcus spp."],
        reference="CARD ARO:3000186"),
    "aac6ib": dict(
        family="aac(6')-Ib aminoglycoside acetyltransferase", confidence="high",
        mechanism="Acetylates aminoglycosides; the -cr variant additionally "
                  "acetylates ciprofloxacin.",
        classes=["Aminoglycosides", "Fluoroquinolones (cr variant only)"],
        organisms=["Enterobacterales"],
        reference="CARD ARO:3002547"),
    "mcr1": dict(
        family="mcr-1 phosphoethanolamine transferase", confidence="high",
        mechanism="Modifies lipid A, reducing colistin binding; plasmid-borne "
                  "and horizontally transmissible.",
        classes=["Polymyxins (colistin)"],
        organisms=["Escherichia coli", "Klebsiella pneumoniae"],
        reference="CARD ARO:3003689"),
}


def _norm(marker: str) -> str:
    """Normalise a marker name so bla-KPC-2, blaKPC_2 and BLAKPC all match."""
    s = "".join(ch for ch in str(marker).lower() if ch.isalnum())
    # strip a trailing allele number: blakpc2 -> blakpc
    while s and s[-1].isdigit() and s[:-1] in MARKER_MAP:
        s = s[:-1]
    if s in MARKER_MAP:
        return s
    # prefix match for families such as erm(A)/ermB/ermC, tetM/tetO
    for key in MARKER_MAP:
        if s.startswith(key):
            return key
    return s


def genomic_data_available() -> bool:
    """True only if a genotype table is actually present. False = say so."""
    return genotype_table_path().exists()


def genotype_table_path() -> Path:
    return Path(os.getenv(GENOTYPE_PATH_ENV, str(DEFAULT_GENOTYPE_PATH)))


def availability_report() -> dict:
    """Structured status the Streamlit page renders verbatim."""
    path = genotype_table_path()
    return {
        "available": path.exists(),
        "expected_path": str(path),
        "reason": (
            "No genomic data is present in this project's datasets. The five "
            "source files (cohort, demographics, labs, microbial_resistance, "
            "prior_med) were audited column by column and contain no sequence, "
            "gene, mutation or AMR-marker field. The resistance model is "
            "therefore phenotype-based only and makes no genomic claims."
        ) if not path.exists() else "Genotype table detected.",
        "how_to_enable": (
            "Provide a CSV at the path above with columns "
            "`order_proc_id_coded, marker` (one row per detected marker, e.g. "
            "AMRFinderPlus / ResFinder / CARD-RGI output), or set the "
            f"{GENOTYPE_PATH_ENV} environment variable to point at one. This "
            "module and the application will activate automatically; no other "
            "code changes are needed."
        ),
        "reference_map_size": len(MARKER_MAP),
        "reference_note": REFERENCE_NOTE,
    }


def load_genotype_table():
    """Load the genotype table if one exists, else return None."""
    path = genotype_table_path()
    if not path.exists():
        return None
    import pandas as pd
    df = pd.read_csv(path, dtype=str)
    required = {"order_proc_id_coded", "marker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Genotype table {path} is missing required column(s): "
            f"{sorted(missing)}. Expected at least {sorted(required)}.")
    return df


def annotate_markers(markers) -> list:
    """Map detected markers to mechanism + affected classes (evidence only)."""
    out = []
    for m in markers:
        key = _norm(m)
        info = MARKER_MAP.get(key)
        if info is None:
            out.append(MarkerAnnotation(
                marker=str(m), gene_family="unrecognised",
                mechanism="This marker is not in the prototype reference map. "
                          "No mechanism can be asserted for it.",
                affected_classes=[], typical_organisms=[],
                implication="No interpretation available — refer to a validated "
                            "AMR database or clinical microbiology.",
                confidence="unknown", reference=REFERENCE_NOTE).to_dict())
            continue
        classes = ", ".join(info["classes"])
        out.append(MarkerAnnotation(
            marker=str(m),
            gene_family=info["family"],
            mechanism=info["mechanism"],
            affected_classes=info["classes"],
            typical_organisms=info["organisms"],
            implication=(
                f"Supports clinician consideration of reduced reliability for: "
                f"{classes}. Genotype is supporting evidence only and does not "
                f"establish phenotypic resistance — confirm with susceptibility "
                f"testing."),
            confidence=info["confidence"],
            reference=info["reference"]).to_dict())
    return out


def markers_for_order(order_proc_id: str) -> list:
    """Return annotated markers for one culture order, or [] if unavailable."""
    df = load_genotype_table()
    if df is None:
        return []
    hits = df.loc[df["order_proc_id_coded"].astype(str) == str(order_proc_id),
                  "marker"].tolist()
    return annotate_markers(hits)


if __name__ == "__main__":
    print(json.dumps(availability_report(), indent=2))
    print("\nExample annotation (demonstration input, not from this dataset):")
    print(json.dumps(annotate_markers(["blaKPC-2", "mecA", "unknownGeneX"]),
                     indent=2))
