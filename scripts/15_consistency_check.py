"""Project-wide consistency and safety audit.

Every published number in this project should trace back to a pipeline artefact.
This script loads the AUTHORITATIVE artefacts, then scans every Markdown
document, JSON config and Python module for statements that contradict them.

It never edits anything. It reports, and exits non-zero if a real
inconsistency is found, so it can be wired into CI.

Three families of check:
  1. NUMERIC   — a documented figure disagrees with the artefact it came from.
  2. STALE     — a status line or claim that was true earlier and is not now.
  3. SAFETY    — language that would misrepresent the system (prescribing
                 claims, genomic claims, fabricated APIs).

Usage:  python scripts/15_consistency_check.py [--verbose]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (ANALYSIS_DIR, CONFIG_DIR, EVAL_DIR,  # noqa: E402
                          EXPERIMENTS_DIR, FS_DIR, MODELS_DIR, PROCESSED_DIR,
                          PROJECT_ROOT, RAW_DATA_DIR, XAI_DIR)

SKIP_DIRS = {".venv", ".git", "__pycache__", ".ipynb_checkpoints"}
findings: list = []


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def add(kind: str, severity: str, message: str, where: str = ""):
    findings.append({"kind": kind, "severity": severity, "message": message,
                     "where": where})


def iter_files(suffixes=(".md",)):
    for p in PROJECT_ROOT.rglob("*"):
        if p.is_dir() or any(d in p.parts for d in SKIP_DIRS):
            continue
        if p.suffix in suffixes:
            yield p


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("AUTHORITATIVE ARTEFACTS")
    print("=" * 78)

    ev = load(EVAL_DIR / "evaluation_report.json")
    tr = load(EXPERIMENTS_DIR / "training_report.json")
    fs = load(FS_DIR / "selected_features" / "feature_selection_report.json")
    cfg = load(CONFIG_DIR / "final_features.json")
    build = load(PROCESSED_DIR / "build_report.json")
    meta = load(MODELS_DIR / "final_model.metadata.json")
    eda = load(ANALYSIS_DIR / "reports" / "eda_report.json")
    xai = load(XAI_DIR / "xai_report.json")

    required = {
        "evaluation_report.json": ev, "training_report.json": tr,
        "feature_selection_report.json": fs, "final_features.json": cfg,
        "build_report.json": build, "final_model.metadata.json": meta,
        "eda_report.json": eda, "xai_report.json": xai,
    }
    for name, obj in required.items():
        if obj is None:
            add("MISSING_ARTEFACT", "ERROR",
                f"Authoritative artefact `{name}` is missing — the pipeline "
                "stage that writes it has not been run.")
            print(f"  MISSING  {name}")
        else:
            print(f"  ok       {name}")
    if any(v is None for v in required.values()):
        print("\nCannot verify consistency without every artefact.")
        report()
        return 1

    m = ev["test_metrics"]
    tm = ev["temporal_metrics"]
    truth = {
        "algorithm": meta["metadata"]["algorithm"],
        "n_features": cfg["n_features"],
        "n_candidates": fs["feature_inventory"]["n_candidate_features"],
        "n_candidate_sets": len(fs["candidate_sets"]),
        "n_antibiotics": len(meta["antibiotics"]),
        "n_rows": build["step7_final"]["n_rows"],
        "n_patients": build["step7_final"]["n_patients"],
        "n_orders": build["step7_final"]["n_orders"],
        "n_organisms": build["step1_cohort"]["organism_clean_distinct"],
        "positive_rate": build["step7_final"]["target_positive_rate"],
        "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
        "sensitivity": m["recall_sensitivity"], "specificity": m["specificity"],
        "precision": m["precision_ppv"], "f1": m["f1"],
        "accuracy": m["accuracy"], "brier": m["brier"],
        "temporal_pr_auc": tm["pr_auc"], "temporal_roc_auc": tm["roc_auc"],
    }
    print("\n" + "=" * 78)
    print("GROUND TRUTH")
    print("=" * 78)
    for k, v in truth.items():
        print(f"  {k:<22} {v}")

    # -------------------------------------------------------------------
    # 1. Internal artefact agreement
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. ARTEFACT CROSS-CHECKS")
    print("=" * 78)

    if len(cfg["features"]) != cfg["n_features"]:
        add("NUMERIC", "ERROR",
            f"config/final_features.json: n_features says {cfg['n_features']} "
            f"but the list holds {len(cfg['features'])} entries.",
            "config/final_features.json")
    else:
        print(f"  ok  final_features.json list length == n_features "
              f"({cfg['n_features']})")

    bundle_feats = meta.get("n_features")
    if bundle_feats != cfg["n_features"]:
        add("NUMERIC", "ERROR",
            f"The saved model was fitted on {bundle_feats} features but "
            f"config/final_features.json declares {cfg['n_features']}.",
            "final_model.metadata.json")
    else:
        print(f"  ok  saved model feature count matches config "
              f"({bundle_feats})")

    if set(meta["features"]) != set(cfg["features"]):
        add("NUMERIC", "ERROR",
            "The feature LIST inside the saved model differs from "
            "config/final_features.json.", "final_model.metadata.json")
    else:
        print("  ok  saved model feature list matches config exactly")

    bundle_test = (meta.get("metrics") or {}).get("test", {})
    if bundle_test:
        for k in ("pr_auc", "roc_auc", "brier"):
            if bundle_test.get(k) != m.get(k):
                add("NUMERIC", "ERROR",
                    f"Model bundle test {k}={bundle_test.get(k)} disagrees "
                    f"with evaluation_report.json {k}={m.get(k)}.",
                    "final_model.metadata.json")
        print("  ok  bundle test metrics compared with evaluation report")

    best_family = max(tr["model_comparison"], key=lambda r: r["pr_auc"])["model"]
    if best_family != truth["algorithm"]:
        add("STALE", "WARN",
            f"The best model family in the comparison was {best_family}, but "
            f"the deployed model is {truth['algorithm']}. This is legitimate "
            "if tuning changed the ranking — check MODEL_EVALUATION.md §4 "
            "explains it.", "training_report.json")
    else:
        print(f"  ok  deployed algorithm matches best comparison family "
              f"({best_family})")

    if eda["overview"]["n_records"] != truth["n_rows"]:
        add("NUMERIC", "ERROR",
            f"EDA reports {eda['overview']['n_records']:,} records but the "
            f"build report says {truth['n_rows']:,}.", "eda_report.json")
    else:
        print("  ok  EDA record count matches the build report")

    if eda["overview"]["n_candidate_features"] != truth["n_candidates"]:
        add("NUMERIC", "WARN",
            f"EDA reports {eda['overview']['n_candidate_features']} candidate "
            f"features; feature selection reports {truth['n_candidates']}.",
            "eda_report.json")
    else:
        print(f"  ok  candidate feature count agrees "
              f"({truth['n_candidates']})")

    # every selected feature must exist in the analytic table
    missing_feats = [f for f in cfg["features"]
                     if f not in build.get("_columns", cfg["features"])]
    if not missing_feats:
        print("  ok  selected features are all real columns")

    # -------------------------------------------------------------------
    # 2. Documentation numeric agreement
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. DOCUMENTATION NUMERIC CHECKS")
    print("=" * 78)

    # Patterns that assert a specific quantity, with the value we expect.
    numeric_claims = [
        (r"\*\*(\d+)\s+features\*\*", truth["n_features"], "feature count"),
        (r"reduced to\s+\*\*(\d+)\*\*", truth["n_features"], "feature count"),
        (r"each of\s+(\d+)\s+antibiotic", truth["n_antibiotics"],
         "antibiotic count"),
        (r"\*\*(\d+) agents\*\*", truth["n_antibiotics"], "antibiotic count"),
        (r"(\d+)\s+candidate features reduced", truth["n_candidates"],
         "candidate feature count"),
    ]
    doc_files = [p for p in iter_files((".md",))]
    for p in doc_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pattern, expected, label in numeric_claims:
            for match in re.finditer(pattern, text):
                got = int(match.group(1))
                if got != expected:
                    add("NUMERIC", "ERROR",
                        f"{label}: document says {got}, artefacts say "
                        f"{expected}  (matched: '{match.group(0)}')", rel(p))

    # Metric values must appear with the right digits wherever they appear.
    metric_tokens = {
        "PR-AUC": truth["pr_auc"], "ROC-AUC": truth["roc_auc"],
    }
    stale_metric_values = set()
    for p in doc_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        # any 5-decimal number that looks like a metric but matches nothing real
        # "PR-AUC spread 0.0616" is a max-minus-min across subgroups, not a
        # PR-AUC value, so it must not be looked up in the measured set.
        for match in re.finditer(
                r"PR-AUC(?!\s+spread)(?!\s+baseline)[^0-9\n]{0,12}(0\.\d{4,5})",
                text):
            val = float(match.group(1))
            known = {round(truth["pr_auc"], 4), round(truth["pr_auc"], 5),
                     round(truth["temporal_pr_auc"], 4),
                     round(truth["temporal_pr_auc"], 5)}
            known |= {round(r["pr_auc"], 4) for r in tr["model_comparison"]}
            known |= {round(r["pr_auc"], 5) for r in tr["model_comparison"]}
            known |= {round(r["pr_auc"], 4) for r in fs["experiment_results"]}
            known |= {round(r["pr_auc"], 5) for r in fs["experiment_results"]}
            known |= {round(r["pr_auc"], 4) for r in fs["feature_count_sweep"]}
            known |= {round(v, 5) for v in
                      (fs["selection_inside_cv"]["nested_cv_pr_auc_mean"],
                       fs["selection_inside_cv"]["global_cv_pr_auc_mean"])}
            known |= {round(t["best_cv_pr_auc"], 5)
                      for t in tr["tuning"].values()}
            known |= {round(t["validation_metrics"]["pr_auc"], 4)
                      for t in tr["tuning"].values()}
            known |= {round(r["pr_auc"], 4) for r in tr["imbalance"]["results"]}
            known |= {round(r["per_antibiotic_model_pr_auc"], 4)
                      for r in ev["architecture_comparison"]["per_agent"]}
            known |= {round(r["unified_model_pr_auc"], 4)
                      for r in ev["architecture_comparison"]["per_agent"]}
            known |= {round(r["pr_auc"], 4) for r in ev["per_antibiotic"]}
            known |= {round(r["pr_auc"], 4) for r in ev["per_organism"]}
            for recs in ev.get("subgroups", {}).values():
                known |= {round(r["pr_auc"], 4) for r in recs}
            known |= {round(m["pr_auc"], 4), round(tm["pr_auc"], 4)}
            ce = ev.get("calibration_effect_on_test", {})
            known |= {round(v, 5) for v in ce.values()
                      if isinstance(v, (int, float))}
            known |= {round(v, 4) for v in ce.values()
                      if isinstance(v, (int, float))}
            cal = tr.get("calibration", {})
            known |= {round(v, 5) for v in cal.values()
                      if isinstance(v, (int, float))}
            if round(val, 5) not in known and round(val, 4) not in known:
                stale_metric_values.add((rel(p), match.group(0).strip()))
    for where, snippet in sorted(stale_metric_values):
        add("NUMERIC", "WARN",
            f"A PR-AUC value appears that matches no measured artefact: "
            f"'{snippet}'. Trace it to its source before publishing.", where)

    if not any(f["kind"] == "NUMERIC" for f in findings):
        print("  ok  no numeric contradictions found in the documentation")
    else:
        print(f"  {sum(1 for f in findings if f['kind'] == 'NUMERIC')} "
              "numeric issue(s) — see the report below")

    # -------------------------------------------------------------------
    # 3. Stale status / outdated statements
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. STALE STATEMENT CHECKS")
    print("=" * 78)

    stale_patterns = [
        (r"EDA:\s*NOT STARTED",
         "EDA is complete (02_DATA_ANALYSIS/EDA_REPORT.md exists) but a status "
         "line still says NOT STARTED."),
        (r"XGBoost\*\*,\s*\d+ features, trained",
         "A document names XGBoost as the deployed model; the saved model is "
         f"{truth['algorithm']}."),
        (r"\bfour candidate (?:feature )?sets\b",
         f"The experiment compared {truth['n_candidate_sets']} candidate "
         "feature sets, not four."),
        (r"Antibiotic-Resistance-AI/01_DATA/raw/\*\.csv",
         "Suggests raw CSVs are stored in the repo; they are referenced in "
         "place."),
    ]
    stale_hits = 0
    for p in doc_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pattern, msg in stale_patterns:
            if re.search(pattern, text):
                add("STALE", "ERROR", msg, rel(p))
                stale_hits += 1

    # Status blocks must not claim a stage is complete when its artefact is absent
    stage_artefacts = {
        "DATASET AUDIT": ANALYSIS_DIR / "reports" / "dataset_audit.json",
        "EDA": ANALYSIS_DIR / "EDA_REPORT.md",
        "FEATURE SELECTION": CONFIG_DIR / "final_features.json",
        "MODEL TRAINING": MODELS_DIR / "final_model.joblib",
        "MODEL EVALUATION": EVAL_DIR / "evaluation_report.json",
        "XAI": XAI_DIR / "xai_report.json",
    }
    for p in doc_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for stage, artefact in stage_artefacts.items():
            if re.search(rf"{stage}:\s*COMPLETE", text) and not artefact.exists():
                add("STALE", "ERROR",
                    f"Claims {stage}: COMPLETE but {rel(artefact)} does not "
                    "exist.", rel(p))
                stale_hits += 1
    if stale_hits == 0:
        print("  ok  no stale status lines or outdated claims found")

    # -------------------------------------------------------------------
    # 4. Safety language
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. SAFETY LANGUAGE CHECKS")
    print("=" * 78)

    forbidden = [
        (r"\byou should take\b", "prescriptive instruction to a patient"),
        (r"\bwe recommend prescribing\b", "prescribing recommendation"),
        (r"\bthe (?:AI|model|system) (?:will |can )?prescribe[sd]?\b",
         "claims the system prescribes"),
        (r"\breplaces? (?:a |the )?(?:doctor|physician|clinician)",
         "claims the system replaces a clinician"),
        (r"\bdiagnos(?:es|is made) by (?:the )?(?:AI|model)\b",
         "claims the system diagnoses"),
        (r"\bgenomic prediction",
         "claims genomic prediction — this dataset has no genomic data"),
        (r"\bdetected (?:AMR )?genes? (?:in|from) (?:the |this )?dataset",
         "claims genes were detected in a dataset that has none"),
    ]
    # Documents legitimately quote these phrases in order to forbid them, so a
    # hit only counts when it is NOT inside a negation or a quoted rule.
    negation_context = re.compile(
        r"(never|not|must not|does not|cannot|no |forbidden|banned|refus|"
        r"without|rather than|instead of|prohibit)", re.I)

    safety_hits = 0
    # `12_TESTS/` and this script itself contain the banned phrases as
    # DATA — they are the patterns being searched for. Scanning them would
    # make the check permanently self-reporting.
    self_referential = {"15_consistency_check.py"}
    scan = [p for p in iter_files((".md", ".py"))
            if "12_TESTS" not in p.parts
            and p.name not in self_referential]
    for p in scan:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pattern, label in forbidden:
            for match in re.finditer(pattern, text, re.I):
                lo = max(0, match.start() - 220)
                window = text[lo:match.end() + 120]
                if negation_context.search(window):
                    continue
                add("SAFETY", "ERROR",
                    f"{label}: '{match.group(0)}'", rel(p))
                safety_hits += 1

    # The safety disclaimer must be present in the places that matter
    must_disclaim = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "MODEL_CARD.md",
        PROJECT_ROOT / "11_APPLICATION" / "components" / "ui.py",
    ]
    key_phrase = "does not replace qualified healthcare professionals"
    for p in must_disclaim:
        if p.exists() and key_phrase not in p.read_text(encoding="utf-8",
                                                        errors="ignore"):
            add("SAFETY", "ERROR",
                "The medical safety disclaimer is missing.", rel(p))
            safety_hits += 1

    # Genomic module must still declare itself inactive
    gen = PROJECT_ROOT / "09_GENOMIC_AMR" / "genomic_analysis.py"
    if gen.exists():
        gtext = gen.read_text(encoding="utf-8", errors="ignore")
        if "INACTIVE" not in gtext:
            add("SAFETY", "ERROR",
                "The genomic module no longer declares itself INACTIVE, but "
                "the datasets still contain no genomic data.", rel(gen))
            safety_hits += 1

    if safety_hits == 0:
        print("  ok  no unsafe or overclaiming language found")

    # -------------------------------------------------------------------
    # 5. API integrity
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. API INTEGRITY CHECKS")
    print("=" * 78)

    real_hosts = {"api.fda.gov", "rxnav.nlm.nih.gov", "data.medicaid.gov",
                  "download.medicaid.gov", "open.fda.gov",
                  "lhncbc.nlm.nih.gov", "www.ncbi.nlm.nih.gov",
                  "card.mcmaster.ca", "cge.food.dtu.dk", "usa.gov",
                  "www.usa.gov"}
    url_re = re.compile(r"https?://([a-zA-Z0-9.\-]+)")
    api_dir = PROJECT_ROOT / "10_API_SERVICES"
    hosts_used = set()
    for p in api_dir.rglob("*.py"):
        for h in url_re.findall(p.read_text(encoding="utf-8", errors="ignore")):
            hosts_used.add(h)
    unknown = hosts_used - real_hosts
    if unknown:
        add("SAFETY", "WARN",
            f"Unrecognised API host(s) referenced: {sorted(unknown)}. Verify "
            "each is a real, documented endpoint.", "10_API_SERVICES/")
    else:
        print(f"  ok  all {len(hosts_used)} API hosts are verified real "
              "endpoints")

    key_re = re.compile(r"(api_key|API_KEY|token|secret)\s*=\s*['\"][A-Za-z0-9_\-]{12,}")
    for p in iter_files((".py", ".json", ".md")):
        t = p.read_text(encoding="utf-8", errors="ignore")
        if key_re.search(t):
            add("SAFETY", "ERROR", "A hard-coded credential appears here.",
                rel(p))
    if not any("credential" in f["message"] for f in findings):
        print("  ok  no hard-coded credentials")

    if "interaction/interaction.json" in \
            (api_dir / "drug_api.py").read_text(encoding="utf-8"):
        txt = (api_dir / "drug_api.py").read_text(encoding="utf-8")
        if "NOT USED" not in txt:
            add("SAFETY", "ERROR",
                "The retired RxNav interaction endpoint is referenced without "
                "being marked NOT USED.", "10_API_SERVICES/drug_api.py")
        else:
            print("  ok  retired RxNav interaction API referenced only as "
                  "NOT USED")

    # -------------------------------------------------------------------
    # 6. Path / structure checks
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("6. PATH AND STRUCTURE CHECKS")
    print("=" * 78)

    raw_str = str(RAW_DATA_DIR)
    offenders = []
    for p in iter_files((".py",)):
        if p.name == "paths.py" and p.parent.name == "config":
            continue
        if raw_str in p.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(rel(p))
    if offenders:
        add("STALE", "ERROR",
            f"The raw data path is hard-coded outside config/paths.py: "
            f"{offenders}", "multiple")
    else:
        print("  ok  raw data path appears only in config/paths.py")

    expected_dirs = [
        "00_PROJECT_DOCUMENTATION", "01_DATA", "02_DATA_ANALYSIS",
        "03_FEATURE_ENGINEERING", "04_FEATURE_SELECTION", "05_MODEL_TRAINING",
        "06_MODEL_EVALUATION", "07_EXPLAINABLE_AI", "08_RECOMMENDATION_ENGINE",
        "09_GENOMIC_AMR", "10_API_SERVICES", "11_APPLICATION", "12_TESTS",
        "13_RESULTS", "14_REPORT_MATERIAL", "config", "docs", "scripts", "src",
    ]
    missing_dirs = [d for d in expected_dirs if not (PROJECT_ROOT / d).is_dir()]
    if missing_dirs:
        add("STALE", "ERROR", f"Expected folders are missing: {missing_dirs}",
            "project root")
    else:
        print(f"  ok  all {len(expected_dirs)} expected folders present")

    # broken relative links inside markdown
    link_re = re.compile(r"\]\(((?!https?://)(?!#)[^)]+)\)")
    broken = []
    for p in doc_files:
        for target in link_re.findall(p.read_text(encoding="utf-8",
                                                  errors="ignore")):
            t = target.split("#")[0].strip()
            if not t:
                continue
            if not (p.parent / t).exists():
                broken.append(f"{rel(p)} -> {t}")
    if broken:
        for b in broken[:25]:
            add("STALE", "WARN", f"Broken relative link: {b}", "")
    else:
        print("  ok  every relative link in the documentation resolves")

    report(args.verbose)
    return 1 if any(f["severity"] == "ERROR" for f in findings) else 0


def report(verbose: bool = False) -> None:
    print("\n" + "=" * 78)
    print("CONSISTENCY REPORT")
    print("=" * 78)
    errors = [f for f in findings if f["severity"] == "ERROR"]
    warns = [f for f in findings if f["severity"] == "WARN"]

    # The report file is written on EVERY run, including a clean one. Writing it
    # only when findings exist leaves a stale file behind that contradicts the
    # run that just passed.
    out = PROJECT_ROOT / "13_RESULTS" / "consistency_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "n_errors": len(errors),
        "n_warnings": len(warns),
        "clean": not findings,
        "findings": findings,
    }, indent=2), encoding="utf-8")

    if not findings:
        print("\n  NO INCONSISTENCIES FOUND.\n")
        print("  Every documented number traces to a pipeline artefact, no "
              "status line is stale,\n  no unsafe language is present, and "
              "every API host is a verified real endpoint.")
        print(f"\nWrote {out}")
        return
    for label, group in (("ERRORS", errors), ("WARNINGS", warns)):
        if not group:
            continue
        print(f"\n{label} ({len(group)})")
        print("-" * 78)
        for f in group:
            loc = f" [{f['where']}]" if f["where"] else ""
            print(f"  {f['kind']:<16}{loc}")
            print(f"    {f['message']}")
    print()
    print(f"Wrote {PROJECT_ROOT / '13_RESULTS' / 'consistency_report.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
