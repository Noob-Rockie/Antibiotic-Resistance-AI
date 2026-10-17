"""Run the whole pipeline in order, or a subset of it.

    python scripts/run_all.py                 # everything
    python scripts/run_all.py --from 06       # resume from feature selection
    python scripts/run_all.py --only 08 09    # just those stages
    python scripts/run_all.py --docs          # regenerate documentation only
    python scripts/run_all.py --list          # show the stages and exit

Each stage is a separate process so a failure is isolated and the traceback is
attributable to one stage.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent

STAGES = [
    ("00", "00_verify_raw_integrity.py", "Verify raw data integrity", False),
    ("01", "01_dataset_audit.py", "PHASE 1 — dataset audit", False),
    ("02", "02_relationship_and_leakage_analysis.py",
     "PHASE 2 — relationships + leakage", False),
    ("03", "03_generate_data_docs.py", "Data dictionary + ER diagram", True),
    ("04", "04_build_analytic_dataset.py",
     "PHASES 3 & 5 — cleaning + feature engineering", False),
    ("04b", "04b_exploratory_analysis.py",
     "PHASE 4 — exploratory data analysis", False),
    ("05", "05_generate_analysis_docs.py",
     "Relationship / leakage / problem-definition docs", True),
    ("06", "06_feature_selection.py", "PHASE 6 — feature selection", False),
    ("07", "07_train_models.py", "PHASE 7 — model training", False),
    ("08", "08_evaluate_models.py", "PHASE 8 — evaluation (reads test once)",
     False),
    ("09", "09_explainability.py", "PHASE 9 — explainable AI", False),
    ("10", "10_generate_feature_docs.py", "Feature-selection docs", True),
    ("11", "11_generate_model_docs.py", "Evaluation, XAI and model-card docs",
     True),
    ("12", "12_generate_project_docs.py", "README + PROJECT_MEMORY", True),
    ("13", "13_generate_notebooks.py", "Analysis notebooks", True),
    ("14", "14_generate_eda_report.py", "EDA_REPORT.md", True),
    ("16", "16_generate_diagrams.py",
     "Methodology and architecture diagrams", True),
    ("17", "17_verify_system.py",
     "PHASES 6-10 — end-to-end system verification", False),
    ("18", "18_assemble_report_material.py",
     "Assemble 14_REPORT_MATERIAL", True),
    ("15", "15_consistency_check.py",
     "Project-wide consistency and safety audit", True),
    ("19", "19_final_status_report.py", "Final project status report", True),
]


def run(stage_id: str, script: str, title: str) -> float:
    print("\n" + "=" * 78)
    print(f"[{stage_id}] {title}")
    print("=" * 78, flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(SCRIPTS / script)],
                          cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n[{stage_id}] FAILED after {elapsed:.0f}s "
              f"(exit code {proc.returncode})")
        raise SystemExit(proc.returncode)
    print(f"\n[{stage_id}] done in {elapsed:.0f}s", flush=True)
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", help="resume from this stage id")
    ap.add_argument("--only", nargs="+", help="run only these stage ids")
    ap.add_argument("--docs", action="store_true",
                    help="run only the documentation-generation stages")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args()

    if args.list:
        for sid, script, title, is_doc in STAGES:
            print(f"  {sid}  {title:<52} {'[docs]' if is_doc else ''}")
        return

    stages = STAGES
    if args.docs:
        stages = [s for s in STAGES if s[3]]
    elif args.only:
        wanted = set(args.only)
        stages = [s for s in STAGES if s[0] in wanted]
    elif args.start:
        idx = next((i for i, s in enumerate(STAGES) if s[0] == args.start), None)
        if idx is None:
            raise SystemExit(f"Unknown stage id: {args.start}")
        stages = STAGES[idx:]

    print(f"Running {len(stages)} stage(s):")
    for sid, _, title, _ in stages:
        print(f"  {sid}  {title}")

    total = 0.0
    timings = []
    for sid, script, title, _ in stages:
        elapsed = run(sid, script, title)
        timings.append((sid, title, elapsed))
        total += elapsed

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    for sid, title, el in timings:
        print(f"  {sid}  {title:<52} {el:8.0f}s")
    print(f"  {'':<4}{'TOTAL':<52} {total:8.0f}s "
          f"({total/60:.1f} min)")
    print("\nNext: python -m pytest 12_TESTS -q")
    print("      streamlit run 11_APPLICATION/app.py")


if __name__ == "__main__":
    main()
