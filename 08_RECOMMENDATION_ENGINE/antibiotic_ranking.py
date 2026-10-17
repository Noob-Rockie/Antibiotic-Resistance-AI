"""Rank antibiotics for clinician review from real model output.

WHAT THIS DOES
  1. Takes one isolate record (patient + specimen + organism + history).
  2. Builds one candidate row per antibiotic the model can score.
  3. Runs the trained model to get P(non-susceptible) for each agent.
  4. Attaches the SHAP contributions that produced each number.
  5. Attaches guardrail flags from `recommendation_rules.py`.
  6. Returns the agents ordered by predicted susceptibility, annotated.

WHAT THIS DOES NOT DO
  * It never says "give antibiotic X". Every option is labelled a *potential
    option for clinician review*.
  * It never invents a probability. Every number comes from the fitted model.
  * It never silently removes an option. An agent with a critical guardrail flag
    is demoted below the unflagged options and the reason is shown, so the
    clinician can see both that it was considered and why it was pushed down.
  * It never claims more confidence than the training data supports: every
    option carries the number of training examples behind it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recommendation_rules import collect_flags, has_critical  # noqa: E402
from src.preprocessing.antibiotic_taxonomy import (  # noqa: E402
    HISTORY_ONLY_AGENTS, antibiotic_class, antibiotic_subclass, aware_category,
    is_oral, is_urinary_only)

SAFETY_NOTICE = (
    "Model output is a statistical estimate of laboratory susceptibility, not a "
    "treatment decision. Every option below requires review by a qualified "
    "healthcare professional and confirmation by antimicrobial susceptibility "
    "testing."
)


@dataclass
class RankedOption:
    rank: int
    antibiotic: str
    antibiotic_class: str
    antibiotic_subclass: str
    aware_category: str
    p_susceptible: float
    p_nonsusceptible: float
    predicted_label: str
    confidence_band: str
    training_examples: int
    evidence_strength: str
    route_oral_available: bool
    urinary_only: bool
    flags: list = field(default_factory=list)
    has_critical_flag: bool = False
    top_factors: list = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _confidence_band(p_nonsusc: float) -> str:
    """Describe how decisive the model's estimate is, in plain language."""
    d = abs(p_nonsusc - 0.5)
    if d >= 0.35:
        return "high"
    if d >= 0.2:
        return "moderate"
    if d >= 0.1:
        return "low"
    return "very low (near the decision boundary)"


def _evidence_strength(n: int) -> str:
    if n == 0:
        return "none — extrapolated from class and organism group"
    if n < 100:
        return "very thin"
    if n < 500:
        return "thin"
    if n < 5000:
        return "moderate"
    return "strong"


def candidate_antibiotics(bundle, restrict_to=None) -> list:
    agents = [a for a in bundle.antibiotics if a not in HISTORY_ONLY_AGENTS]
    if restrict_to:
        agents = [a for a in agents if a in set(restrict_to)]
    return agents


def build_candidate_frame(record: dict, agents, bundle) -> pd.DataFrame:
    """One row per candidate agent, sharing the isolate's patient context."""
    base = {k: v for k, v in record.items() if k != "antibiotic"}
    rows = []
    for a in agents:
        r = dict(base)
        r["antibiotic"] = a
        r["antibiotic_class"] = antibiotic_class(a)
        r["antibiotic_subclass"] = antibiotic_subclass(a)
        r["antibiotic_aware"] = aware_category(a)
        r["antibiotic_is_oral"] = is_oral(a)
        r["antibiotic_is_urinary_only"] = is_urinary_only(a)
        rows.append(r)
    df = pd.DataFrame(rows)
    for f in bundle.features:
        if f not in df.columns:
            df[f] = bundle.defaults.get(f, np.nan)
    return df


def shap_contributions(bundle, X_encoded: pd.DataFrame, top_k: int = 5) -> list:
    """Per-row SHAP attributions, via the shared explainability module.

    Failure here degrades the UI (no per-factor breakdown) but must never break
    a prediction, so any explainer error returns empty attributions rather than
    propagating.
    """
    try:
        from src.explainability.shap_explainer import explain_batch
        return explain_batch(bundle.model, X_encoded, top_k=top_k)
    except Exception:
        return [[] for _ in range(len(X_encoded))]


# --- readable names for the SHAP factors shown to a clinician ---------------
FEATURE_LABELS = {
    "antibiotic": "the antibiotic being considered",
    "antibiotic_class": "the antibiotic class",
    "antibiotic_subclass": "the antibiotic subclass",
    "antibiotic_aware": "the WHO AWaRe tier of the agent",
    "organism_clean": "the identified organism",
    "organism_genus": "the organism genus",
    "organism_group": "the organism group (gram stain / family)",
    "organism_morphotype": "the colony morphotype",
    "culture_description": "the specimen type",
    "ordering_mode": "inpatient vs outpatient setting",
    "age_band_ordinal": "patient age band",
    "age_midpoint_years": "patient age",
    "gender_code": "recorded binary sex code",
    "order_year": "year of the culture",
    "prior_res_same_org_abx_n": "previous resistance in this patient to this "
                                "exact organism-antibiotic pair",
    "prior_res_same_abx_n": "previous resistance in this patient to this antibiotic",
    "prior_res_same_class_n": "previous resistance in this patient to this "
                              "antibiotic class",
    "prior_res_n_events_total": "total previous resistant results for this patient",
    "prior_same_abx_nonsusc_rate": "this patient's past non-susceptibility rate "
                                   "for this agent",
    "prior_same_org_nonsusc_rate": "this patient's past non-susceptibility rate "
                                   "for this organism",
    "prior_nonsusc_rate": "this patient's overall past non-susceptibility rate",
    "prior_med_same_class_days_since": "days since this patient last received "
                                       "this antibiotic class",
    "prior_med_n_drugdays_90d": "antibiotic exposure in the last 90 days",
    "prior_med_n_drugdays_365d": "antibiotic exposure in the last year",
    "prior_n_cultures": "number of previous cultures for this patient",
    "prior_n_positive_cultures": "number of previous positive cultures",
    "median_cr": "median creatinine over the 14 days before the culture",
    "median_wbc": "median white cell count over the 14 days before the culture",
}


def humanise(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    f = feature.replace("prior_med_class__", "prior exposure to ")
    f = f.replace("prior_res_", "prior resistance ").replace("prior_med_", "prior exposure ")
    f = f.replace("__", " ").replace("_", " ")
    return f


def narrate(option_name: str, factors: list, p_susc: float) -> str:
    """A short, model-derived sentence — no template claims beyond the SHAP."""
    if not factors:
        return (f"The model estimates a {p_susc:.0%} probability that this isolate "
                f"is susceptible to {option_name}. Per-factor attributions are "
                f"unavailable for this prediction.")
    up = [f for f in factors if f["shap"] > 0][:2]
    down = [f for f in factors if f["shap"] < 0][:2]
    parts = [f"The model estimates a {p_susc:.0%} probability of susceptibility "
             f"to {option_name}."]
    if up:
        parts.append("Pushing the estimate towards resistance: "
                     + "; ".join(humanise(f["feature"]) for f in up) + ".")
    if down:
        parts.append("Pushing it towards susceptibility: "
                     + "; ".join(humanise(f["feature"]) for f in down) + ".")
    return " ".join(parts)


def rank_antibiotics(record: dict, bundle, *, allergies=None,
                     restrict_to=None, top_k_factors: int = 5,
                     explain: bool = True) -> dict:
    """Score and rank every candidate agent for one isolate."""
    agents = candidate_antibiotics(bundle, restrict_to)
    if not agents:
        return {"options": [], "safety_notice": SAFETY_NOTICE,
                "error": "No candidate antibiotics available in this model."}

    cand = build_candidate_frame(record, agents, bundle)
    p_nonsusc = bundle.predict_nonsusceptible_proba(cand)

    factors = ([[]] * len(cand))
    if explain:
        factors = shap_contributions(bundle, bundle.prepare(cand), top_k_factors)

    organism = record.get("organism_clean")
    organism_group = record.get("organism_group")
    specimen = record.get("culture_description")

    options = []
    for i, a in enumerate(agents):
        pn = float(p_nonsusc[i])
        ps = 1.0 - pn
        n_support = bundle.support_for(organism, a)
        flags = collect_flags(
            organism=organism, antibiotic=a,
            antibiotic_class=antibiotic_class(a),
            aware_category=aware_category(a), specimen=specimen,
            organism_group=organism_group,
            antibiotic_subclass=antibiotic_subclass(a),
            n_training_rows=n_support, allergies=allergies)
        options.append(RankedOption(
            rank=0, antibiotic=a,
            antibiotic_class=antibiotic_class(a),
            antibiotic_subclass=antibiotic_subclass(a),
            aware_category=aware_category(a),
            p_susceptible=round(ps, 4),
            p_nonsusceptible=round(pn, 4),
            predicted_label=("Susceptible" if pn < bundle.threshold
                             else "Non-susceptible"),
            confidence_band=_confidence_band(pn),
            training_examples=n_support,
            evidence_strength=_evidence_strength(n_support),
            route_oral_available=bool(is_oral(a)),
            urinary_only=bool(is_urinary_only(a)),
            flags=flags,
            has_critical_flag=has_critical(flags),
            top_factors=factors[i] if i < len(factors) else [],
            explanation=narrate(a, factors[i] if i < len(factors) else [], ps),
        ))

    # Ordering: unflagged options first (best predicted susceptibility first),
    # then flagged options. Nothing is removed.
    options.sort(key=lambda o: (o.has_critical_flag, -o.p_susceptible))
    for i, o in enumerate(options, 1):
        o.rank = i

    n_flagged = sum(1 for o in options if o.has_critical_flag)
    return {
        "options": [o.to_dict() for o in options],
        "n_options": len(options),
        "n_with_critical_flag": n_flagged,
        "organism": organism,
        "organism_group": organism_group,
        "specimen": specimen,
        "model": {
            "algorithm": bundle.metadata.get("algorithm"),
            "target_definition": bundle.metadata.get("target_definition"),
            "decision_threshold": bundle.threshold,
            "validation_pr_auc": bundle.metrics.get("validation", {}).get("pr_auc"),
            "test_pr_auc": bundle.metrics.get("test", {}).get("pr_auc"),
        },
        "safety_notice": SAFETY_NOTICE,
        "framing": ("These are potentially suitable options for clinician "
                    "review, ordered by the model's estimated probability of "
                    "susceptibility. They are not a prescription and not a "
                    "recommendation to treat."),
    }
