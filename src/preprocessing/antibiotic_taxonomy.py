"""Antibiotic taxonomy for the 54 agents that actually appear in this cohort.

Every key below was taken verbatim from
`microbiology_cultures_cohort.antibiotic` — including the source system's
spelling quirks (`Ticacarcillin/Clavulanic Acid`, `Imipenem/Ebactam`,
`Cephalexin/Cephalothin`). Nothing here is invented and nothing is a guess about
an agent that is not in the data.

Why a taxonomy is needed
------------------------
1. `antibiotic` is a 54-level categorical. Class membership is a much lower
   variance signal that lets the model generalise from a well-tested agent to a
   rarely-tested one in the same class (e.g. Doripenem, n=2,299, borrows from
   the carbapenem class).
2. Cross-resistance is organised by mechanism, so class is the right unit for
   "prior resistance to this drug class" history features.
3. The recommendation layer needs class to avoid ranking five members of one
   class as five independent "options" for a clinician.

Sources for the class assignments: standard antimicrobial classification as used
by CLSI M100 and the WHO AWaRe classification. `awarelabel` follows the WHO
AWaRe 2023 list (Access / Watch / Reserve) where the agent is listed.
"""
from __future__ import annotations

# agent -> (class, subclass, WHO AWaRe category)
ANTIBIOTIC_CLASS: dict[str, tuple] = {
    # --- penicillins ---------------------------------------------------------
    "Penicillin":                    ("Penicillin", "Natural penicillin", "Access"),
    "Ampicillin":                    ("Penicillin", "Aminopenicillin", "Access"),
    "Oxacillin":                     ("Penicillin", "Antistaphylococcal penicillin", "Access"),
    "Piperacillin":                  ("Penicillin", "Ureidopenicillin", "Watch"),
    "Ticarcillin":                   ("Penicillin", "Carboxypenicillin", "Watch"),
    # --- beta-lactam + beta-lactamase inhibitor ------------------------------
    "Amoxicillin/Clavulanic Acid":   ("Beta-lactam/inhibitor", "Aminopenicillin + inhibitor", "Access"),
    "Ampicillin/Sulbactam":          ("Beta-lactam/inhibitor", "Aminopenicillin + inhibitor", "Access"),
    "Piperacillin/Tazobactam":       ("Beta-lactam/inhibitor", "Ureidopenicillin + inhibitor", "Watch"),
    "Ticacarcillin/Clavulanic Acid": ("Beta-lactam/inhibitor", "Carboxypenicillin + inhibitor", "Watch"),
    # --- cephalosporins ------------------------------------------------------
    "Cefazolin":                     ("Cephalosporin", "1st generation", "Access"),
    "Cephalexin/Cephalothin":        ("Cephalosporin", "1st generation", "Access"),
    "Cefuroxime":                    ("Cephalosporin", "2nd generation", "Watch"),
    "Cefoxitin":                     ("Cephalosporin", "Cephamycin", "Watch"),
    "Cefotetan":                     ("Cephalosporin", "Cephamycin", "Watch"),
    "Ceftriaxone":                   ("Cephalosporin", "3rd generation", "Watch"),
    "Cefotaxime":                    ("Cephalosporin", "3rd generation", "Watch"),
    "Ceftazidime":                   ("Cephalosporin", "3rd generation (antipseudomonal)", "Watch"),
    "Cefpodoxime":                   ("Cephalosporin", "3rd generation (oral)", "Watch"),
    "Cefepime":                      ("Cephalosporin", "4th generation", "Watch"),
    "Ceftaroline":                   ("Cephalosporin", "5th generation (anti-MRSA)", "Reserve"),
    "Cefiderocol":                   ("Cephalosporin", "Siderophore cephalosporin", "Reserve"),
    "Ceftazidime/Avibactam":         ("Cephalosporin/inhibitor", "3rd gen + inhibitor", "Reserve"),
    "Ceftolozane/Tazobactam":        ("Cephalosporin/inhibitor", "Antipseudomonal + inhibitor", "Reserve"),
    # --- carbapenems / monobactam -------------------------------------------
    "Ertapenem":                     ("Carbapenem", "Group 1 carbapenem", "Watch"),
    "Meropenem":                     ("Carbapenem", "Group 2 carbapenem", "Watch"),
    "Imipenem":                      ("Carbapenem", "Group 2 carbapenem", "Watch"),
    "Doripenem":                     ("Carbapenem", "Group 2 carbapenem", "Watch"),
    "Meropenem/Vaborbactam":         ("Carbapenem/inhibitor", "Carbapenem + inhibitor", "Reserve"),
    "Imipenem/Ebactam":              ("Carbapenem/inhibitor", "Carbapenem + inhibitor", "Reserve"),
    "Aztreonam":                     ("Monobactam", "Monobactam", "Watch"),
    # --- fluoroquinolones ----------------------------------------------------
    "Ciprofloxacin":                 ("Fluoroquinolone", "2nd generation", "Watch"),
    "Levofloxacin":                  ("Fluoroquinolone", "3rd generation", "Watch"),
    "Moxifloxacin":                  ("Fluoroquinolone", "4th generation", "Watch"),
    # --- aminoglycosides -----------------------------------------------------
    "Gentamicin":                    ("Aminoglycoside", "Aminoglycoside", "Access"),
    "Tobramycin":                    ("Aminoglycoside", "Aminoglycoside", "Access"),
    "Amikacin":                      ("Aminoglycoside", "Aminoglycoside", "Access"),
    "Streptomycin":                  ("Aminoglycoside", "Aminoglycoside", "Watch"),
    # --- gram-positive agents ------------------------------------------------
    "Vancomycin":                    ("Glycopeptide", "Glycopeptide", "Watch"),
    "Daptomycin":                    ("Lipopeptide", "Lipopeptide", "Reserve"),
    "Linezolid":                     ("Oxazolidinone", "Oxazolidinone", "Reserve"),
    "Quinupristin/Dalfopristin":     ("Streptogramin", "Streptogramin A+B", "Reserve"),
    # --- macrolides / lincosamides ------------------------------------------
    "Erythromycin":                  ("Macrolide", "14-membered macrolide", "Watch"),
    "Clarithromycin":                ("Macrolide", "14-membered macrolide", "Watch"),
    "Clindamycin":                   ("Lincosamide", "Lincosamide", "Access"),
    # --- tetracyclines -------------------------------------------------------
    "Tetracycline":                  ("Tetracycline", "Tetracycline", "Access"),
    "Doxycycline":                   ("Tetracycline", "Tetracycline", "Access"),
    "Minocycline":                   ("Tetracycline", "Tetracycline", "Watch"),
    "Tigecycline":                   ("Glycylcycline", "Glycylcycline", "Reserve"),
    # --- other ---------------------------------------------------------------
    "Trimethoprim/Sulfamethoxazole": ("Folate pathway inhibitor", "Sulfonamide + DHFR inhibitor", "Access"),
    "Nitrofurantoin":                ("Nitrofuran", "Nitrofuran", "Access"),
    "Fosfomycin":                    ("Fosfomycin", "Phosphonic acid", "Access"),
    "Metronidazole":                 ("Nitroimidazole", "Nitroimidazole", "Access"),
    "Colistin":                      ("Polymyxin", "Polymyxin", "Reserve"),
    "Chloramphenicol":               ("Phenicol", "Phenicol", "Access"),
    # --- present ONLY in the prior-resistance history table ------------------
    # These two are never a prediction target (they do not appear in the cohort
    # `antibiotic` column) but they do appear in
    # `microbiology_cultures_microbial_resistance`, so the history features need
    # a class for them.
    "Rifampin":                      ("Rifamycin", "Rifamycin", "Watch"),
    "Ethambutol":                    ("Antimycobacterial", "Antimycobacterial", "Watch"),
}

# Agents that can appear as a prediction target (i.e. observed in the cohort's
# `antibiotic` column). Rifampin/Ethambutol are deliberately excluded.
HISTORY_ONLY_AGENTS = {"Rifampin", "Ethambutol"}

# Agents that can be given orally with reliable bioavailability. Used only to
# annotate a ranked option in the UI, never to change a model prediction.
ORAL_OPTION = {
    "Amoxicillin/Clavulanic Acid", "Cephalexin/Cephalothin", "Cefuroxime",
    "Cefpodoxime", "Ciprofloxacin", "Levofloxacin", "Moxifloxacin",
    "Trimethoprim/Sulfamethoxazole", "Nitrofurantoin", "Fosfomycin",
    "Doxycycline", "Minocycline", "Tetracycline", "Clindamycin",
    "Linezolid", "Metronidazole", "Clarithromycin", "Erythromycin",
    "Ampicillin", "Penicillin",
}

# Agents licensed/used for urinary tract infection where urinary concentration
# is what matters. Nitrofurantoin and fosfomycin are urine-only agents.
URINARY_ONLY = {"Nitrofurantoin", "Fosfomycin"}


def antibiotic_class(name) -> str:
    return ANTIBIOTIC_CLASS.get(name, ("Unmapped", "Unmapped", "Unclassified"))[0]


def antibiotic_subclass(name) -> str:
    return ANTIBIOTIC_CLASS.get(name, ("Unmapped", "Unmapped", "Unclassified"))[1]


def aware_category(name) -> str:
    return ANTIBIOTIC_CLASS.get(name, ("Unmapped", "Unmapped", "Unclassified"))[2]


def is_oral(name) -> int:
    return int(name in ORAL_OPTION)


def is_urinary_only(name) -> int:
    return int(name in URINARY_ONLY)


def coverage_report(observed_names) -> dict:
    """Report which observed antibiotics are missing from the taxonomy."""
    observed = {n for n in observed_names if n and str(n) != "nan"}
    mapped = observed & set(ANTIBIOTIC_CLASS)
    return {
        "n_observed": len(observed),
        "n_mapped": len(mapped),
        "coverage_pct": round(100 * len(mapped) / len(observed), 2) if observed else None,
        "unmapped": sorted(observed - set(ANTIBIOTIC_CLASS)),
        "n_classes": len({v[0] for v in ANTIBIOTIC_CLASS.values()}),
    }
