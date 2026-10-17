"""Central path configuration.

Raw data is READ-ONLY and is referenced in place at its original location.
It is never copied, moved, or modified by any code in this project.
"""
from pathlib import Path

# --- Project root -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Raw data (READ-ONLY, referenced in place) -------------------------------
RAW_DATA_DIR = Path(
    r"C:\Users\alanf\OneDrive\Desktop"
    r"\Antibiotic Resistance Prediction and Clinical Decision Support"
)

RAW_FILES = {
    "cohort":      RAW_DATA_DIR / "microbiology_cultures_cohort.csv",
    "demographics": RAW_DATA_DIR / "microbiology_cultures_demographics.csv",
    "labs":        RAW_DATA_DIR / "microbiology_cultures_labs.csv",
    "resistance":  RAW_DATA_DIR / "microbiology_cultures_microbial_resistance.csv",
    "prior_med":   RAW_DATA_DIR / "microbiology_cultures_prior_med.csv",
}

# --- Project data tiers ------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "01_DATA"
RAW_REF_DIR   = DATA_DIR / "raw"          # manifest + checksums only
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Stage directories -------------------------------------------------------
ANALYSIS_DIR   = PROJECT_ROOT / "02_DATA_ANALYSIS"
FE_DIR         = PROJECT_ROOT / "03_FEATURE_ENGINEERING"
FS_DIR         = PROJECT_ROOT / "04_FEATURE_SELECTION"
TRAIN_DIR      = PROJECT_ROOT / "05_MODEL_TRAINING"
EVAL_DIR       = PROJECT_ROOT / "06_MODEL_EVALUATION"
XAI_DIR        = PROJECT_ROOT / "07_EXPLAINABLE_AI"
RECO_DIR       = PROJECT_ROOT / "08_RECOMMENDATION_ENGINE"
GENOMIC_DIR    = PROJECT_ROOT / "09_GENOMIC_AMR"
API_DIR        = PROJECT_ROOT / "10_API_SERVICES"
APP_DIR        = PROJECT_ROOT / "11_APPLICATION"
RESULTS_DIR    = PROJECT_ROOT / "13_RESULTS"
REPORT_DIR     = PROJECT_ROOT / "14_REPORT_MATERIAL"
CONFIG_DIR     = PROJECT_ROOT / "config"
DOCS_DIR       = PROJECT_ROOT / "docs"

MODELS_DIR      = TRAIN_DIR / "models"
EXPERIMENTS_DIR = TRAIN_DIR / "experiments"

# --- Global reproducibility --------------------------------------------------
RANDOM_SEED = 42

# Strings that the source export uses to encode a missing value.
NA_VALUES = ["Null", "NULL", "null", "NaN", "nan", "NA", "N/A", "", "None"]


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, EXPERIMENTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
