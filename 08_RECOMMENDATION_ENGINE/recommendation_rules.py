"""Clinical guardrails applied AROUND the model — never inside it.

Design principle
----------------
The model estimates one thing: the probability that the laboratory would call
this isolate non-susceptible to this agent. It knows nothing about whether the
agent can reach the site of infection, whether the organism is intrinsically
resistant regardless of what any panel says, or whether the agent is a
last-line reserve that should not be spent casually.

Those are separate, well-documented facts, and they are applied here as
**flags on a ranked option**, not as filters that silently delete options and
not as adjustments to the predicted probability. A clinician sees the model's
number and the flag side by side and decides. Nothing in this module changes a
prediction.

Sources
-------
* Intrinsic resistance: EUCAST "Expected Resistant Phenotypes" (v1.2, 2023) and
  CLSI M100 intrinsic-resistance tables.
* Site suitability: FDA-approved labelling for the agents concerned (the same
  labels the application retrieves live through `10_API_SERVICES/drug_api.py`)
  and standard pharmacokinetic/pharmacodynamic constraints.
* Stewardship tier: WHO AWaRe classification 2023.

This is a curated subset for a research prototype. It is not a complete
intrinsic-resistance reference and is displayed as such.
"""
from __future__ import annotations

import re

# --- severity of a flag ------------------------------------------------------
CRITICAL = "critical"   # would be clinically wrong regardless of the AST
CAUTION = "caution"     # depends on context; clinician must weigh it
INFO = "info"           # stewardship / practical note


# ---------------------------------------------------------------------------
# 1. Intrinsic resistance — organism pattern -> (antibiotic classes, agents)
# ---------------------------------------------------------------------------
# Each entry: regex over the cleaned organism name -> what it is intrinsically
# resistant to, plus a short mechanism note.
INTRINSIC_RESISTANCE = [
    {
        "organism": r"^ENTEROCOCCUS|^ENTEROCOCCUS SPECIES",
        "classes": {"Cephalosporin", "Cephalosporin/inhibitor", "Monobactam"},
        "agents": {"Trimethoprim/Sulfamethoxazole", "Clindamycin"},
        "note": "Enterococci are intrinsically resistant to all cephalosporins "
                "and aztreonam (low-affinity PBP5), and to clindamycin. "
                "Sulfamethoxazole/trimethoprim may test susceptible in vitro but "
                "fails in vivo because enterococci scavenge exogenous folate.",
    },
    {
        "organism": r"^KLEBSIELLA",
        "classes": set(),
        "agents": {"Ampicillin", "Ticarcillin"},
        "note": "Klebsiella spp. carry a chromosomal SHV beta-lactamase and are "
                "intrinsically resistant to aminopenicillins.",
    },
    {
        "organism": r"^PROTEUS|^PROVIDENCIA|^MORGANELLA",
        "classes": {"Polymyxin"},
        "agents": {"Nitrofurantoin", "Tetracycline", "Doxycycline",
                   "Minocycline", "Tigecycline", "Colistin"},
        "note": "Proteae (Proteus, Providencia, Morganella) are intrinsically "
                "resistant to polymyxins, nitrofurantoin and tetracyclines "
                "through lipid A modification and efflux.",
    },
    {
        "organism": r"^SERRATIA",
        "classes": {"Polymyxin"},
        "agents": {"Nitrofurantoin", "Colistin"},
        "note": "Serratia marcescens is intrinsically resistant to polymyxins "
                "and nitrofurantoin.",
    },
    {
        "organism": r"^ENTEROBACTER|^CITROBACTER FREUNDII|^KLEBSIELLA AEROGENES|^HAFNIA",
        "classes": set(),
        "agents": {"Ampicillin", "Amoxicillin/Clavulanic Acid",
                   "Ampicillin/Sulbactam", "Cefazolin", "Cefoxitin",
                   "Cefotetan", "Cephalexin/Cephalothin"},
        "note": "AmpC-producing Enterobacterales are intrinsically resistant to "
                "aminopenicillins, first-generation cephalosporins and "
                "cephamycins, and can derepress AmpC on therapy with a "
                "third-generation cephalosporin.",
    },
    {
        "organism": r"^PSEUDOMONAS AERUGINOSA",
        "classes": {"Tetracycline", "Glycopeptide", "Lincosamide",
                    "Macrolide", "Oxazolidinone", "Folate pathway inhibitor",
                    "Nitrofuran"},
        "agents": {"Ampicillin", "Amoxicillin/Clavulanic Acid",
                   "Ampicillin/Sulbactam", "Cefazolin", "Ceftriaxone",
                   "Cefotaxime", "Cefoxitin", "Ertapenem",
                   "Trimethoprim/Sulfamethoxazole", "Nitrofurantoin"},
        "note": "P. aeruginosa is intrinsically resistant to most beta-lactams "
                "other than the antipseudomonal agents, and to ertapenem, "
                "through AmpC, efflux (MexAB-OprM) and low outer-membrane "
                "permeability.",
    },
    {
        "organism": r"^STENOTROPHOMONAS",
        "classes": {"Carbapenem", "Aminoglycoside"},
        "agents": set(),
        "note": "S. maltophilia carries the L1 metallo-beta-lactamase and is "
                "intrinsically resistant to all carbapenems.",
    },
    {
        "organism": r"^ACINETOBACTER",
        "classes": {"Monobactam"},
        "agents": {"Ertapenem", "Ampicillin", "Cefazolin"},
        "note": "Acinetobacter spp. are intrinsically resistant to ertapenem and "
                "aztreonam.",
    },
    {
        "organism": r"^BURKHOLDERIA",
        "classes": {"Aminoglycoside", "Polymyxin"},
        "agents": {"Ampicillin", "Ertapenem"},
        "note": "Burkholderia cepacia complex is intrinsically resistant to "
                "polymyxins and aminoglycosides.",
    },
    {
        "organism": r"^ACHROMOBACTER|^ALCALIGENES",
        "classes": set(),
        "agents": {"Ertapenem", "Aztreonam"},
        "note": "Achromobacter xylosoxidans is intrinsically resistant to "
                "aztreonam and ertapenem.",
    },
    {
        "organism": r"^LACTOBACILLUS|^LEUCONOSTOC|^PEDIOCOCCUS",
        "classes": {"Glycopeptide"},
        "agents": set(),
        "note": "Lactobacillus, Leuconostoc and Pediococcus are intrinsically "
                "vancomycin resistant (D-Ala-D-Lac peptidoglycan terminus).",
    },
]


# ---------------------------------------------------------------------------
# 1b. Gram-stain-level intrinsic resistance, keyed on `organism_group`
# ---------------------------------------------------------------------------
# These are the broadest and most important rules, and they must NOT be driven
# by an enumerated list of genus names: the enumeration will always be
# incomplete, and the failure is silent. `organism_group` is produced for every
# organism by `src/preprocessing/organism_normalizer.py`, so keying on it means
# a rare non-fermenter such as Achromobacter xylosoxidans is covered by exactly
# the same rule as Escherichia coli.
GROUP_INTRINSIC_RESISTANCE = {
    "GN_Enterobacterales": {
        "classes": {"Glycopeptide", "Lipopeptide", "Oxazolidinone",
                    "Lincosamide", "Macrolide", "Streptogramin"},
        # Aminopenicillins DO treat some Enterobacterales, so the exclusion is
        # written at subclass level rather than banning "Penicillin" wholesale.
        "subclasses": {"Natural penicillin", "Antistaphylococcal penicillin"},
        "note": "Gram-negative bacteria are intrinsically resistant to "
                "glycopeptides (vancomycin), lipopeptides (daptomycin), "
                "oxazolidinones (linezolid), lincosamides and macrolides — "
                "these agents are too large or too hydrophobic to cross the "
                "gram-negative outer membrane.",
    },
    "GN_NonFermenter": {
        "classes": {"Glycopeptide", "Lipopeptide", "Oxazolidinone",
                    "Lincosamide", "Macrolide", "Streptogramin",
                    "Nitroimidazole"},
        "subclasses": {"Natural penicillin", "Antistaphylococcal penicillin",
                       "Aminopenicillin", "Aminopenicillin + inhibitor",
                       "1st generation", "2nd generation", "Cephamycin"},
        "note": "Gram-negative non-fermenters are intrinsically resistant to "
                "the gram-positive agents (glycopeptides, lipopeptides, "
                "oxazolidinones, lincosamides, macrolides), which cannot cross "
                "the outer membrane, and to metronidazole, which requires "
                "anaerobic nitroreduction.",
    },
    "GN_Fastidious": {
        "classes": {"Glycopeptide", "Lipopeptide", "Oxazolidinone",
                    "Streptogramin"},
        "subclasses": {"Antistaphylococcal penicillin"},
        "note": "Fastidious gram-negative organisms are intrinsically "
                "resistant to glycopeptides, lipopeptides and oxazolidinones.",
    },
    "GN_Unspecified": {
        "classes": {"Glycopeptide", "Lipopeptide", "Oxazolidinone",
                    "Lincosamide", "Macrolide"},
        "subclasses": {"Natural penicillin", "Antistaphylococcal penicillin"},
        "note": "Gram-negative organisms are intrinsically resistant to the "
                "gram-positive agents listed, which cannot cross the outer "
                "membrane.",
    },
    "GP_Cocci": {
        "classes": {"Monobactam", "Polymyxin"},
        "note": "Gram-positive bacteria are intrinsically resistant to "
                "aztreonam (which binds only gram-negative PBP3) and to "
                "polymyxins (which target the gram-negative outer membrane).",
    },
    "GP_Rod": {
        "classes": {"Monobactam", "Polymyxin"},
        "note": "Gram-positive bacteria are intrinsically resistant to "
                "aztreonam and polymyxins.",
    },
    "Anaerobe": {
        "classes": {"Aminoglycoside"},
        "note": "Anaerobes are intrinsically resistant to aminoglycosides, "
                "whose uptake requires oxygen-dependent active transport.",
    },
    "Fungus": {
        "classes": set(),   # every antibacterial; handled as a special case
        "note": "This is a fungal isolate. Antibacterial agents have no "
                "activity against fungi; an antifungal is required. No "
                "antifungal is modelled by this project.",
        "all_antibacterials": True,
    },
    "Mycobacterium": {
        "classes": {"Penicillin", "Beta-lactam/inhibitor", "Cephalosporin",
                    "Cephalosporin/inhibitor", "Monobactam", "Glycopeptide",
                    "Nitrofuran", "Polymyxin"},
        "note": "Mycobacteria are intrinsically resistant to most conventional "
                "antibacterials because of their mycolic-acid cell wall. "
                "Treatment requires specific antimycobacterial regimens, which "
                "this project does not model.",
    },
}


# ---------------------------------------------------------------------------
# 2. Site suitability — specimen type -> agents that do not belong there
# ---------------------------------------------------------------------------
SITE_RULES = {
    "URINE": {
        "unsuitable": {
            "Moxifloxacin": "Moxifloxacin achieves poor urinary concentrations "
                            "and is not indicated for urinary tract infection.",
            "Tigecycline": "Tigecycline achieves low urinary and serum "
                           "concentrations; not indicated for UTI.",
            "Daptomycin": "Daptomycin is not indicated for urinary tract "
                          "infection.",
        },
        "note": "Urinary agents (nitrofurantoin, fosfomycin) are appropriate "
                "for uncomplicated cystitis but do not treat pyelonephritis or "
                "bacteraemia because they do not achieve tissue or serum "
                "concentrations.",
    },
    "BLOOD": {
        "unsuitable": {
            "Nitrofurantoin": "Nitrofurantoin achieves no meaningful serum "
                              "concentration and must not be used for "
                              "bacteraemia.",
            "Fosfomycin": "Oral fosfomycin achieves no meaningful serum "
                          "concentration and must not be used for bacteraemia.",
            "Tigecycline": "Tigecycline achieves low serum concentrations; "
                           "increased all-cause mortality has been reported. "
                           "Avoid as monotherapy in bacteraemia.",
        },
        "note": "Bloodstream infection requires an agent with reliable serum "
                "concentrations and, usually, intravenous administration.",
    },
    "RESPIRATORY": {
        "unsuitable": {
            "Daptomycin": "Daptomycin is inactivated by pulmonary surfactant "
                          "and must not be used for pneumonia.",
            "Nitrofurantoin": "Nitrofurantoin does not reach the lung; not "
                              "indicated for respiratory infection.",
            "Fosfomycin": "Oral fosfomycin does not reach the lung; not "
                          "indicated for respiratory infection.",
        },
        "note": "Respiratory isolates may represent colonisation rather than "
                "infection, particularly in ventilated or cystic-fibrosis "
                "patients. Correlate with the clinical picture before treating.",
    },
}


# ---------------------------------------------------------------------------
# 3. Stewardship tier (WHO AWaRe 2023)
# ---------------------------------------------------------------------------
AWARE_NOTE = {
    "Access": ("WHO AWaRe: Access — first- or second-choice agent with a "
               "narrower spectrum and lower resistance potential. Preferred "
               "where clinically adequate."),
    "Watch": ("WHO AWaRe: Watch — higher resistance potential. Reserve for "
              "specific indications and de-escalate when the AST allows."),
    "Reserve": ("WHO AWaRe: Reserve — last-resort agent for confirmed or "
                "suspected multi-drug-resistant infection. Use should be "
                "protected and, in most institutions, requires infectious "
                "diseases approval."),
}


# ---------------------------------------------------------------------------
def intrinsic_resistance_flags(organism: str, antibiotic: str,
                               antibiotic_class: str,
                               organism_group: str | None = None,
                               antibiotic_subclass: str | None = None) -> list:
    """Intrinsic-resistance flags for one organism-agent pair.

    Two layers, and both matter:
      * gram-stain / family level, keyed on `organism_group` — broad, complete,
        and impossible to miss for a rare genus;
      * species or genus level, keyed on the organism name — the specific
        exceptions (Klebsiella + ampicillin, Stenotrophomonas + carbapenem).
    """
    flags = []
    src = "EUCAST Expected Resistant Phenotypes v1.2; CLSI M100"

    # --- layer 1: gram stain / family -------------------------------------
    grp = GROUP_INTRINSIC_RESISTANCE.get(organism_group or "")
    if grp:
        hit = (grp.get("all_antibacterials")
               or antibiotic_class in grp["classes"]
               or (antibiotic_subclass
                   and antibiotic_subclass in grp.get("subclasses", set())))
        if hit:
            flags.append({
                "severity": CRITICAL,
                "kind": "intrinsic_resistance",
                "message": (f"{organism or organism_group} is expected to be "
                            f"intrinsically resistant to {antibiotic}."),
                "detail": grp["note"],
                "source": src,
            })

    # --- layer 2: genus / species -----------------------------------------
    if organism:
        org = str(organism).upper()
        for rule in INTRINSIC_RESISTANCE:
            if not re.search(rule["organism"], org):
                continue
            if (antibiotic in rule["agents"]
                    or antibiotic_class in rule["classes"]):
                flags.append({
                    "severity": CRITICAL,
                    "kind": "intrinsic_resistance",
                    "message": (f"{organism} is expected to be intrinsically "
                                f"resistant to {antibiotic}."),
                    "detail": rule["note"],
                    "source": src,
                })
    # de-duplicate identical messages from the two layers
    seen, out = set(), []
    for f in flags:
        key = (f["kind"], f["message"])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def site_flags(specimen: str, antibiotic: str) -> list:
    rules = SITE_RULES.get(str(specimen).upper())
    if not rules:
        return []
    out = []
    if antibiotic in rules["unsuitable"]:
        out.append({
            "severity": CRITICAL,
            "kind": "site_unsuitable",
            "message": f"{antibiotic} is not appropriate for a "
                       f"{str(specimen).lower()} isolate.",
            "detail": rules["unsuitable"][antibiotic],
            "source": "FDA product labelling; standard PK/PD constraints",
        })
    return out


def stewardship_flags(antibiotic: str, aware_category: str) -> list:
    note = AWARE_NOTE.get(aware_category)
    if not note:
        return []
    severity = CAUTION if aware_category == "Reserve" else INFO
    return [{
        "severity": severity,
        "kind": "stewardship",
        "message": f"{antibiotic} is a WHO AWaRe '{aware_category}' agent.",
        "detail": note,
        "source": "WHO AWaRe classification of antibiotics, 2023",
    }]


def evidence_flags(n_training_rows: int, min_rows: int = 500) -> list:
    """Warn when the model has thin evidence for this organism-agent pair."""
    if n_training_rows is None:
        return []
    if n_training_rows == 0:
        return [{
            "severity": CAUTION, "kind": "no_evidence",
            "message": "This organism-antibiotic combination never appeared in "
                       "the training data.",
            "detail": "The laboratory does not routinely test this pair, so the "
                      "prediction is an extrapolation from the antibiotic's "
                      "class and the organism's group rather than from direct "
                      "evidence. Treat it as a hypothesis, not a result.",
            "source": "Training-data coverage check",
        }]
    if n_training_rows < min_rows:
        return [{
            "severity": CAUTION, "kind": "thin_evidence",
            "message": f"Only {n_training_rows:,} training examples exist for "
                       f"this organism-antibiotic combination.",
            "detail": f"Below the {min_rows:,}-row threshold at which this "
                      "project considers a pair well evidenced. The predicted "
                      "probability is correspondingly less reliable.",
            "source": "Training-data coverage check",
        }]
    return []


def allergy_flags(antibiotic_class: str, allergies) -> list:
    """Flag a documented allergy. Allergies are clinician-entered, not modelled."""
    if not allergies:
        return []
    out = []
    beta_lactam = {"Penicillin", "Beta-lactam/inhibitor", "Cephalosporin",
                   "Cephalosporin/inhibitor", "Carbapenem",
                   "Carbapenem/inhibitor"}
    for a in allergies:
        a_l = str(a).strip().lower()
        if not a_l:
            continue
        if a_l in ("penicillin", "penicillins", "beta-lactam", "beta lactam"):
            if antibiotic_class in beta_lactam:
                sev = CRITICAL if antibiotic_class in (
                    "Penicillin", "Beta-lactam/inhibitor") else CAUTION
                out.append({
                    "severity": sev, "kind": "allergy",
                    "message": f"Documented {a} allergy; this agent is a "
                               f"{antibiotic_class}.",
                    "detail": "Cross-reactivity between penicillins and modern "
                              "cephalosporins/carbapenems is low (<1-3%) but "
                              "the severity of the index reaction must be "
                              "established before the agent is used.",
                    "source": "Clinician-entered allergy record",
                })
        elif a_l and a_l in str(antibiotic_class).lower():
            out.append({
                "severity": CRITICAL, "kind": "allergy",
                "message": f"Documented {a} allergy matches this agent's class "
                           f"({antibiotic_class}).",
                "detail": "Do not use without allergy assessment.",
                "source": "Clinician-entered allergy record",
            })
    return out


def collect_flags(*, organism: str, antibiotic: str, antibiotic_class: str,
                  aware_category: str, specimen: str,
                  organism_group: str | None = None,
                  antibiotic_subclass: str | None = None,
                  n_training_rows: int | None = None,
                  allergies=None) -> list:
    """All guardrail flags for one ranked option, most severe first."""
    flags = []
    flags += intrinsic_resistance_flags(organism, antibiotic, antibiotic_class,
                                        organism_group, antibiotic_subclass)
    flags += site_flags(specimen, antibiotic)
    flags += allergy_flags(antibiotic_class, allergies)
    flags += evidence_flags(n_training_rows)
    flags += stewardship_flags(antibiotic, aware_category)
    order = {CRITICAL: 0, CAUTION: 1, INFO: 2}
    return sorted(flags, key=lambda f: order.get(f["severity"], 3))


def has_critical(flags) -> bool:
    return any(f["severity"] == CRITICAL for f in flags)
