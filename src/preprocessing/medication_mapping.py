"""Map the 89 raw `medication_name` values onto ingredient + antibiotic class.

WHY THIS EXISTS — a finding from the audit
------------------------------------------
`microbiology_cultures_prior_med` ships a `medication_category` column that
looks like a therapeutic class. It is not. Counting the real data shows a
strict 1:1 mapping — 89 distinct `medication_name` values and 89 distinct
`medication_category` values, with zero categories covering more than one name
and zero names covering more than one category. The codes are also actively
misleading as classes:

    CEF  -> Cefazolin              CEF1 -> Cefazolin In Dextrose
    CEF2 -> Ceftriaxone            CEF3 -> Cefpodoxime

so `CEF*` spans first-generation and third-generation cephalosporins, while
`Keflex` (cephalexin, also 1st generation) sits under `CEP`. `medication_category`
is therefore a redundant surrogate key for `medication_name`, and this project
excludes it and derives a real class from the name instead.

The name field also mixes brand names (`Zithromax`, `Bactrim Ds`, `Augmentin`,
`Keflex`, `Levaquin`, `Cipro`, `Macrobid`, `Xifaxan`, `Flagyl`, `Hiprex`,
`Zyvox`) with generics, and salt/diluent suffixes (`In Dextrose`, `Hcl`,
`Sodium`, `-Sod.Chloride(Iso)`). Both are normalised away here so that
`Ciprofloxacin Hcl`, `Ciprofloxacin In`, `Cipro` and `Ciprofloxacin` all
collapse to the same prior-exposure signal.

Class labels are kept consistent with `antibiotic_taxonomy.ANTIBIOTIC_CLASS` so
that "prior exposure to class X" lines up with "susceptibility of agent in
class X".
"""
from __future__ import annotations

# raw medication_name -> (normalised ingredient, antibiotic class)
MEDICATION_MAP: dict[str, tuple] = {
    # --- beta-lactam + inhibitor --------------------------------------------
    "Piperacillin-Tazobactam-Dextrs": ("Piperacillin/Tazobactam", "Beta-lactam/inhibitor"),
    "Piperacillin-Tazobactam":        ("Piperacillin/Tazobactam", "Beta-lactam/inhibitor"),
    "Amoxicillin-Pot Clavulanate":    ("Amoxicillin/Clavulanic Acid", "Beta-lactam/inhibitor"),
    "Augmentin":                      ("Amoxicillin/Clavulanic Acid", "Beta-lactam/inhibitor"),
    # --- penicillins ---------------------------------------------------------
    "Amoxicillin":                    ("Amoxicillin", "Penicillin"),
    "Ampicillin":                     ("Ampicillin", "Penicillin"),
    "Ampicillin Sodium":              ("Ampicillin", "Penicillin"),
    "Penicillin":                     ("Penicillin", "Penicillin"),
    "Dicloxacillin":                  ("Dicloxacillin", "Penicillin"),
    # --- cephalosporins ------------------------------------------------------
    "Cefazolin":                      ("Cefazolin", "Cephalosporin"),
    "Cefazolin In Dextrose":          ("Cefazolin", "Cephalosporin"),
    "Cephalexin":                     ("Cephalexin", "Cephalosporin"),
    "Keflex":                         ("Cephalexin", "Cephalosporin"),
    "Cefadroxil":                     ("Cefadroxil", "Cephalosporin"),
    "Cefuroxime Axetil":              ("Cefuroxime", "Cephalosporin"),
    "Cefoxitin":                      ("Cefoxitin", "Cephalosporin"),
    "Ceftriaxone":                    ("Ceftriaxone", "Cephalosporin"),
    "Cefpodoxime":                    ("Cefpodoxime", "Cephalosporin"),
    "Cefdinir":                       ("Cefdinir", "Cephalosporin"),
    "Ceftazidime":                    ("Ceftazidime", "Cephalosporin"),
    "Ceftazidime-Dextrose":           ("Ceftazidime", "Cephalosporin"),
    "Cefepime":                       ("Cefepime", "Cephalosporin"),
    "Cefepime In":                    ("Cefepime", "Cephalosporin"),
    # --- carbapenems / monobactam -------------------------------------------
    "Ertapenem":                      ("Ertapenem", "Carbapenem"),
    "Meropenem":                      ("Meropenem", "Carbapenem"),
    "Aztreonam":                      ("Aztreonam", "Monobactam"),
    "Aztreonam In":                   ("Aztreonam", "Monobactam"),
    # --- fluoroquinolones ----------------------------------------------------
    "Ciprofloxacin":                  ("Ciprofloxacin", "Fluoroquinolone"),
    "Ciprofloxacin Hcl":              ("Ciprofloxacin", "Fluoroquinolone"),
    "Ciprofloxacin In":               ("Ciprofloxacin", "Fluoroquinolone"),
    "Cipro":                          ("Ciprofloxacin", "Fluoroquinolone"),
    "Ciprofloxacin-Dexamethasone":    ("Ciprofloxacin", "Fluoroquinolone"),
    "Levofloxacin":                   ("Levofloxacin", "Fluoroquinolone"),
    "Levofloxacin In":                ("Levofloxacin", "Fluoroquinolone"),
    "Levaquin":                       ("Levofloxacin", "Fluoroquinolone"),
    "Moxifloxacin":                   ("Moxifloxacin", "Fluoroquinolone"),
    "Moxifloxacin-Sod.Chloride(Iso)": ("Moxifloxacin", "Fluoroquinolone"),
    "Ofloxacin":                      ("Ofloxacin", "Fluoroquinolone"),
    "Gatifloxacin":                   ("Gatifloxacin", "Fluoroquinolone"),
    # --- aminoglycosides -----------------------------------------------------
    "Gentamicin":                     ("Gentamicin", "Aminoglycoside"),
    "Gentamicin In Nacl":             ("Gentamicin", "Aminoglycoside"),
    "Gentamicin-Sodium Citrate":      ("Gentamicin", "Aminoglycoside"),
    "Tobramycin Sulfate":             ("Tobramycin", "Aminoglycoside"),
    "Tobramycin-Dexamethasone":       ("Tobramycin", "Aminoglycoside"),
    "Amikacin":                       ("Amikacin", "Aminoglycoside"),
    # --- gram-positive agents ------------------------------------------------
    "Vancomycin":                     ("Vancomycin", "Glycopeptide"),
    "Vancomycin In":                  ("Vancomycin", "Glycopeptide"),
    "Vancomycin In Dextrose":         ("Vancomycin", "Glycopeptide"),
    "Vancomycin-Diluent Combo":       ("Vancomycin", "Glycopeptide"),
    "Linezolid":                      ("Linezolid", "Oxazolidinone"),
    "Linezolid In Dextrose":          ("Linezolid", "Oxazolidinone"),
    "Zyvox":                          ("Linezolid", "Oxazolidinone"),
    "Tedizolid":                      ("Tedizolid", "Oxazolidinone"),
    # --- macrolides / lincosamides ------------------------------------------
    "Azithromycin":                   ("Azithromycin", "Macrolide"),
    "Zithromax":                      ("Azithromycin", "Macrolide"),
    "Clarithromycin":                 ("Clarithromycin", "Macrolide"),
    "Erythromycin":                   ("Erythromycin", "Macrolide"),
    "Erythromycin Ethylsuccinate":    ("Erythromycin", "Macrolide"),
    "Clindamycin Hcl":                ("Clindamycin", "Lincosamide"),
    "Clindamycin In":                 ("Clindamycin", "Lincosamide"),
    "Clindamycin Phosphate":          ("Clindamycin", "Lincosamide"),
    # --- tetracyclines -------------------------------------------------------
    "Doxycycline Hyclate":            ("Doxycycline", "Tetracycline"),
    "Doxycycline Monohydrate":        ("Doxycycline", "Tetracycline"),
    "Minocycline":                    ("Minocycline", "Tetracycline"),
    # --- folate pathway ------------------------------------------------------
    "Sulfamethoxazole-Trimethoprim":  ("Trimethoprim/Sulfamethoxazole", "Folate pathway inhibitor"),
    "Bactrim":                        ("Trimethoprim/Sulfamethoxazole", "Folate pathway inhibitor"),
    "Bactrim Ds":                     ("Trimethoprim/Sulfamethoxazole", "Folate pathway inhibitor"),
    "Trimethoprim":                   ("Trimethoprim", "Folate pathway inhibitor"),
    "Silver Sulfadiazine":            ("Silver sulfadiazine", "Folate pathway inhibitor"),
    "Dapsone":                        ("Dapsone", "Folate pathway inhibitor"),
    # --- urinary agents ------------------------------------------------------
    "Nitrofurantoin":                 ("Nitrofurantoin", "Nitrofuran"),
    "Nitrofurantoin Macrocrystal":    ("Nitrofurantoin", "Nitrofuran"),
    "Macrobid":                       ("Nitrofurantoin", "Nitrofuran"),
    "Macrodantin":                    ("Nitrofurantoin", "Nitrofuran"),
    "Fosfomycin Tromethamine":        ("Fosfomycin", "Fosfomycin"),
    "Methenamine Hippurate":          ("Methenamine", "Urinary antiseptic"),
    "Methenamine Mandelate":          ("Methenamine", "Urinary antiseptic"),
    "Hiprex":                         ("Methenamine", "Urinary antiseptic"),
    # --- nitroimidazoles / anti-anaerobe ------------------------------------
    "Metronidazole":                  ("Metronidazole", "Nitroimidazole"),
    "Metronidazole In Nacl":          ("Metronidazole", "Nitroimidazole"),
    "Flagyl":                         ("Metronidazole", "Nitroimidazole"),
    "Fidaxomicin":                    ("Fidaxomicin", "Macrocyclic"),
    # --- rifamycins / antimycobacterials ------------------------------------
    "Rifampin":                       ("Rifampin", "Rifamycin"),
    "Rifabutin":                      ("Rifabutin", "Rifamycin"),
    "Rifaximin":                      ("Rifaximin", "Rifamycin"),
    "Xifaxan":                        ("Rifaximin", "Rifamycin"),
    "Isoniazid":                      ("Isoniazid", "Antimycobacterial"),
    "Ethambutol":                     ("Ethambutol", "Antimycobacterial"),
    # --- last-line gram-negative --------------------------------------------
    "Colistin":                       ("Colistin", "Polymyxin"),
}

# Classes considered "broad-spectrum" for a summary exposure feature. Prior
# broad-spectrum exposure is the classical driver of subsequent resistance.
BROAD_SPECTRUM_CLASSES = {
    "Beta-lactam/inhibitor", "Carbapenem", "Cephalosporin",
    "Fluoroquinolone", "Glycopeptide", "Monobactam", "Polymyxin",
    "Oxazolidinone",
}


def medication_ingredient(name) -> str:
    return MEDICATION_MAP.get(name, (str(name), "Unmapped"))[0]


def medication_class(name) -> str:
    return MEDICATION_MAP.get(name, (str(name), "Unmapped"))[1]


def coverage_report(observed_names) -> dict:
    observed = {n for n in observed_names if n and str(n) != "nan"}
    mapped = observed & set(MEDICATION_MAP)
    return {
        "n_observed": len(observed),
        "n_mapped": len(mapped),
        "coverage_pct": round(100 * len(mapped) / len(observed), 2) if observed else None,
        "unmapped": sorted(observed - set(MEDICATION_MAP)),
        "n_ingredients": len({v[0] for v in MEDICATION_MAP.values()}),
        "n_classes": len({v[1] for v in MEDICATION_MAP.values()}),
    }
