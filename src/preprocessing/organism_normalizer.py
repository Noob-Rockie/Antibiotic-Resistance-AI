"""Normalise the 314 raw `organism` strings into clean, modellable fields.

The raw column is a laboratory information-system dictionary dump and carries
four kinds of noise, all confirmed by counting the real values:

  1. Deprecated dictionary entries prefixed `ZZ` / `ZZZ`
     (16 distinct names, 11,725 rows) — e.g. `ZZZENTEROBACTER AEROGENES`,
     which is the same organism as `KLEBSIELLA AEROGENES` under its old name.
  2. Inconsistent bracketing — `STREPTOCOCCUS AGALACTIAE (GROUP B)` and
     `STREPTOCOCCUS AGALACTIAE {GROUP B}` are two separate levels of the same
     categorical (12,718 + 6,170 rows).
  3. Colony morphotype baked into the name — `MUCOID PSEUDOMONAS AERUGINOSA`,
     `PSEUDOMONAS AERUGINOSA (NON-MUCOID CF)`, `STAPH AUREUS(COLONY VARIANT …)`.
     Morphotype is observed at identification, so it is legitimately available
     at prediction time and is extracted into its own feature.
  4. **Resistance phenotype baked into the name** — `STAPH AUREUS {MRSA}`,
     `KLEBSIELLA PNEUMONIAE (CARBAPENEM RESISTANT)`,
     `ENTEROCOCCUS FAECIUM -  VANCO RESISTANT` and similar.
     22,119 rows (1.375% of positive cultures) carry such a token.

Point 4 is a **data-leakage vector**, not a formatting problem. A carbapenem-
resistant designation on the identification line is derived from the same
susceptibility workflow that produces the label, so a model that sees it would
be reading the answer for carbapenem rows. This module therefore *strips* the
resistance qualifier and returns it separately, so the pipeline can exclude it
explicitly and the leakage report can quantify it. See
docs/DATA_LEAKAGE_ANALYSIS.md.
"""
from __future__ import annotations

import re

# --- resistance phenotype tokens (LEAKAGE - stripped, never a feature) -------
RESISTANCE_TOKEN_PATTERNS = [
    (re.compile(r"\bMRSA\b"), "MRSA"),
    (re.compile(r"\bMSSA\b"), "MSSA"),
    (re.compile(r"\bCARBAPENEM\s+RESISTANT\b"), "CARBAPENEM_RESISTANT"),
    (re.compile(r"\bVANCO(?:MYCIN)?\s+RESISTANT\b"), "VANCOMYCIN_RESISTANT"),
    (re.compile(r"\bVRE\b"), "VANCOMYCIN_RESISTANT"),
    (re.compile(r"\bESBL\b"), "ESBL"),
    (re.compile(r"\bMETHICILLIN\s+RESISTANT\b"), "MRSA"),
]

# --- colony morphotype (observed at ID - safe to use as a feature) -----------
MORPHOTYPE_PATTERNS = [
    (re.compile(r"\bNON-?MUCOID\b"), "non_mucoid"),
    (re.compile(r"\bMUCOID\b"), "mucoid"),
    (re.compile(r"COLONY VARIANT|SMALL COLONY|MORPHOTYPE"), "small_colony_variant"),
]

# --- genus -> broad clinical group ------------------------------------------
GENUS_GROUP = {
    # Enterobacterales
    "ESCHERICHIA": "GN_Enterobacterales", "KLEBSIELLA": "GN_Enterobacterales",
    "PROTEUS": "GN_Enterobacterales", "ENTEROBACTER": "GN_Enterobacterales",
    "CITROBACTER": "GN_Enterobacterales", "SERRATIA": "GN_Enterobacterales",
    "MORGANELLA": "GN_Enterobacterales", "PROVIDENCIA": "GN_Enterobacterales",
    "HAFNIA": "GN_Enterobacterales", "RAOULTELLA": "GN_Enterobacterales",
    "PANTOEA": "GN_Enterobacterales", "CRONOBACTER": "GN_Enterobacterales",
    "KLUYVERA": "GN_Enterobacterales", "SALMONELLA": "GN_Enterobacterales",
    "SHIGELLA": "GN_Enterobacterales", "YERSINIA": "GN_Enterobacterales",
    "EDWARDSIELLA": "GN_Enterobacterales", "LECLERCIA": "GN_Enterobacterales",
    "PLESIOMONAS": "GN_Enterobacterales", "EWINGELLA": "GN_Enterobacterales",
    "BUTTIAUXELLA": "GN_Enterobacterales", "RAHNELLA": "GN_Enterobacterales",
    # Non-fermenting gram negatives
    "PSEUDOMONAS": "GN_NonFermenter", "ACINETOBACTER": "GN_NonFermenter",
    "STENOTROPHOMONAS": "GN_NonFermenter", "BURKHOLDERIA": "GN_NonFermenter",
    "ACHROMOBACTER": "GN_NonFermenter", "ALCALIGENES": "GN_NonFermenter",
    "CHRYSEOBACTERIUM": "GN_NonFermenter", "ELIZABETHKINGIA": "GN_NonFermenter",
    "RALSTONIA": "GN_NonFermenter", "SPHINGOMONAS": "GN_NonFermenter",
    "COMAMONAS": "GN_NonFermenter", "OLIGELLA": "GN_NonFermenter",
    "BREVUNDIMONAS": "GN_NonFermenter", "DELFTIA": "GN_NonFermenter",
    "MYROIDES": "GN_NonFermenter", "PANDORAEA": "GN_NonFermenter",
    "CUPRIAVIDUS": "GN_NonFermenter", "INQUILINUS": "GN_NonFermenter",
    "AEROMONAS": "GN_NonFermenter", "VIBRIO": "GN_NonFermenter",
    "SHEWANELLA": "GN_NonFermenter", "PSYCHROBACTER": "GN_NonFermenter",
    # Fastidious gram negatives
    "HAEMOPHILUS": "GN_Fastidious", "NEISSERIA": "GN_Fastidious",
    "MORAXELLA": "GN_Fastidious", "CAMPYLOBACTER": "GN_Fastidious",
    "PASTEURELLA": "GN_Fastidious", "AGGREGATIBACTER": "GN_Fastidious",
    "EIKENELLA": "GN_Fastidious", "KINGELLA": "GN_Fastidious",
    "CARDIOBACTERIUM": "GN_Fastidious", "CAPNOCYTOPHAGA": "GN_Fastidious",
    "BORDETELLA": "GN_Fastidious", "BRUCELLA": "GN_Fastidious",
    "LEGIONELLA": "GN_Fastidious", "HELICOBACTER": "GN_Fastidious",
    # Gram positive cocci
    "STAPHYLOCOCCUS": "GP_Cocci", "STAPH": "GP_Cocci",
    "STREPTOCOCCUS": "GP_Cocci", "ENTEROCOCCUS": "GP_Cocci",
    "AEROCOCCUS": "GP_Cocci", "ABIOTROPHIA": "GP_Cocci",
    "GRANULICATELLA": "GP_Cocci", "GEMELLA": "GP_Cocci",
    "LEUCONOSTOC": "GP_Cocci", "LACTOCOCCUS": "GP_Cocci",
    "PEDIOCOCCUS": "GP_Cocci", "MICROCOCCUS": "GP_Cocci",
    "KOCURIA": "GP_Cocci", "ALLOIOCOCCUS": "GP_Cocci",
    # Gram positive rods
    "CORYNEBACTERIUM": "GP_Rod", "LISTERIA": "GP_Rod", "BACILLUS": "GP_Rod",
    "LACTOBACILLUS": "GP_Rod", "NOCARDIA": "GP_Rod", "ACTINOMYCES": "GP_Rod",
    "ROTHIA": "GP_Rod", "ARCANOBACTERIUM": "GP_Rod", "ERYSIPELOTHRIX": "GP_Rod",
    "BREVIBACTERIUM": "GP_Rod", "DERMABACTER": "GP_Rod", "TURICELLA": "GP_Rod",
    "CELLULOSIMICROBIUM": "GP_Rod", "MICROBACTERIUM": "GP_Rod",
    "PAENIBACILLUS": "GP_Rod", "TSUKAMURELLA": "GP_Rod", "GORDONIA": "GP_Rod",
    "RHODOCOCCUS": "GP_Rod",
    # Anaerobes
    "BACTEROIDES": "Anaerobe", "CLOSTRIDIUM": "Anaerobe",
    "CLOSTRIDIOIDES": "Anaerobe", "FUSOBACTERIUM": "Anaerobe",
    "PREVOTELLA": "Anaerobe", "PEPTOSTREPTOCOCCUS": "Anaerobe",
    "VEILLONELLA": "Anaerobe", "PROPIONIBACTERIUM": "Anaerobe",
    "CUTIBACTERIUM": "Anaerobe", "PARABACTEROIDES": "Anaerobe",
    "EGGERTHELLA": "Anaerobe", "FINEGOLDIA": "Anaerobe",
    "ANAEROCOCCUS": "Anaerobe", "PORPHYROMONAS": "Anaerobe",
    "BIFIDOBACTERIUM": "Anaerobe", "ACTINOTIGNUM": "Anaerobe",
    "PARVIMONAS": "Anaerobe", "ATOPOBIUM": "Anaerobe",
    # Mycobacteria & fungi
    "MYCOBACTERIUM": "Mycobacterium", "MYCOBACTEROIDES": "Mycobacterium",
    "CANDIDA": "Fungus", "ASPERGILLUS": "Fungus", "CRYPTOCOCCUS": "Fungus",
    "SACCHAROMYCES": "Fungus", "RHODOTORULA": "Fungus", "TRICHOSPORON": "Fungus",
    "FUSARIUM": "Fungus", "SCEDOSPORIUM": "Fungus", "EXOPHIALA": "Fungus",
    "MALASSEZIA": "Fungus", "PENICILLIUM": "Fungus", "GEOTRICHUM": "Fungus",
}

# Descriptive (non-binomial) lab results that have no genus.
DESCRIPTIVE_GROUP = [
    (re.compile(r"COAG.?NEG.*STAPH|COAGULASE NEGATIVE"), "GP_Cocci",
     "COAGULASE NEGATIVE STAPHYLOCOCCUS"),
    (re.compile(r"VIRIDANS"), "GP_Cocci", "VIRIDANS GROUP STREPTOCOCCI"),
    (re.compile(r"BETA STREPTOCOCCUS|BETA-HEMOLYTIC STREP"), "GP_Cocci",
     "BETA HEMOLYTIC STREPTOCOCCUS"),
    (re.compile(r"NON.?FERMENT"), "GN_NonFermenter",
     "NON-FERMENTING GRAM NEGATIVE RODS"),
    (re.compile(r"LACTOSE FERMENTING GRAM NEGATIVE"), "GN_Enterobacterales",
     "LACTOSE FERMENTING GRAM NEGATIVE RODS"),
    (re.compile(r"GRAM NEGATIVE ROD"), "GN_Unspecified",
     "GRAM NEGATIVE RODS"),
    (re.compile(r"GRAM POSITIVE ROD"), "GP_Rod", "GRAM POSITIVE RODS"),
    (re.compile(r"GRAM POSITIVE COCC"), "GP_Cocci", "GRAM POSITIVE COCCI"),
    (re.compile(r"YEAST"), "Fungus", "YEAST"),
    (re.compile(r"MOLD|MOULD"), "Fungus", "MOLD"),
    (re.compile(r"DIPHTHEROID"), "GP_Rod", "DIPHTHEROIDS"),
]

# Old dictionary names that are the same taxon under a newer name.
SYNONYMS = {
    "ENTEROBACTER AEROGENES": "KLEBSIELLA AEROGENES",
    "STAPH AUREUS": "STAPHYLOCOCCUS AUREUS",
    "CLOSTRIDIUM DIFFICILE": "CLOSTRIDIOIDES DIFFICILE",
    "PROPIONIBACTERIUM ACNES": "CUTIBACTERIUM ACNES",
}

_BRACKETS = re.compile(r"[\{\}\[\]]")
_PARENS = re.compile(r"\([^)]*\)")
_MULTISPACE = re.compile(r"\s+")
_TRAILING_QUALIFIER = re.compile(
    r"\s*-\s*(NOT|COAG|VANCO|CARBAPENEM|FLUOROQUINOLONE|UNIDENTIFIED|ALPHA|BETA)\b.*$")


def parse_organism(raw) -> dict:
    """Split a raw organism string into clean fields.

    Returns a dict with:
      organism_clean        species-level name, phenotype tokens removed
      organism_genus        first word of the cleaned name
      organism_group        broad clinical group (GN_Enterobacterales, GP_Cocci …)
      organism_morphotype   mucoid / non_mucoid / small_colony_variant / standard
      leaked_phenotype      resistance token found in the raw name, or None
                            (EXCLUDED from features — recorded for the audit)
      is_species_level      1 if the name resolves to genus + species
    """
    if raw is None or (isinstance(raw, float)) or str(raw).strip() == "" \
            or str(raw).lower() == "nan":
        return {"organism_clean": None, "organism_genus": None,
                "organism_group": None, "organism_morphotype": None,
                "leaked_phenotype": None, "is_species_level": 0}

    s = str(raw).upper().strip()
    s = _BRACKETS.sub(lambda m: "(" if m.group() in "{[" else ")", s)

    # 1. deprecated dictionary prefix
    s = re.sub(r"^Z{2,}\s*", "", s)

    # 2. resistance phenotype -> extracted and REMOVED (leakage control)
    leaked = None
    for pat, tag in RESISTANCE_TOKEN_PATTERNS:
        if pat.search(s):
            leaked = tag if leaked is None else f"{leaked}+{tag}"
            s = pat.sub(" ", s)

    # 3. morphotype -> extracted and REMOVED (kept as its own feature)
    morphotype = "standard"
    for pat, tag in MORPHOTYPE_PATTERNS:
        if pat.search(s):
            morphotype = tag
            s = pat.sub(" ", s)
            break

    # 4. drop remaining parenthetical qualifiers and trailing notes
    s = _PARENS.sub(" ", s)
    s = _TRAILING_QUALIFIER.sub("", s)
    s = s.replace(" CF", " ").replace("()", " ")
    s = re.sub(r"[^A-Z0-9 ./-]", " ", s)
    s = _MULTISPACE.sub(" ", s).strip(" -.")

    # 5. descriptive (non-binomial) results
    for pat, group, label in DESCRIPTIVE_GROUP:
        if pat.search(s):
            return {"organism_clean": label, "organism_genus": label.split()[0],
                    "organism_group": group, "organism_morphotype": morphotype,
                    "leaked_phenotype": leaked, "is_species_level": 0}

    # 6. drop rank noise words so "ENTEROBACTER CLOACAE COMPLEX" collapses onto
    #    "ENTEROBACTER CLOACAE"
    tokens = [t for t in s.split()
              if t not in {"COMPLEX", "GROUP", "GP.", "GP", "SUBSP.", "SUBSP",
                           "SP.", "SP", "SPP.", "SPP", "STR"}]
    s = " ".join(tokens).strip()
    s = SYNONYMS.get(s, s)

    genus = s.split()[0] if s else None
    group = GENUS_GROUP.get(genus, "Other_Unclassified")
    species_level = int(len(s.split()) >= 2 and "SPECIES" not in s)
    if not species_level and genus:
        s = f"{genus} SPECIES"
        s = SYNONYMS.get(s, s)

    return {"organism_clean": s or None, "organism_genus": genus,
            "organism_group": group, "organism_morphotype": morphotype,
            "leaked_phenotype": leaked, "is_species_level": species_level}


def normalize_series(series):
    """Vectorised-ish wrapper: parse each distinct value once, then map."""
    import pandas as pd
    distinct = series.dropna().astype(str).unique()
    table = {v: parse_organism(v) for v in distinct}
    out = pd.DataFrame(
        [table.get(v, parse_organism(None)) for v in series.astype("object")],
        index=series.index)
    return out


if __name__ == "__main__":
    demo = ["ESCHERICHIA COLI", "STAPH AUREUS {MRSA}",
            "KLEBSIELLA PNEUMONIAE (CARBAPENEM RESISTANT)",
            "ZZZENTEROBACTER AEROGENES", "MUCOID PSEUDOMONAS AERUGINOSA",
            "PSEUDOMONAS AERUGINOSA (NON-MUCOID CF)",
            "STREPTOCOCCUS AGALACTIAE {GROUP B}",
            "STREPTOCOCCUS AGALACTIAE (GROUP B)",
            "ENTEROCOCCUS FAECIUM -  VANCO RESISTANT",
            "COAG NEGATIVE STAPHYLOCOCCUS",
            "ENTEROBACTER CLOACAE COMPLEX", "ENTEROBACTER CLOACAE",
            "NON FERMENTING GRAM NEGATIVE RODS", "ENTEROCOCCUS SPECIES",
            "STAPH AUREUS(COLONY VARIANT - SMALL COLONY OR OTHER MORPHOTYPE)"]
    for d in demo:
        r = parse_organism(d)
        print(f"{d[:52]:<54} -> {str(r['organism_clean']):<38} "
              f"{r['organism_group']:<22} {r['organism_morphotype']:<20} "
              f"leak={r['leaked_phenotype']}")
