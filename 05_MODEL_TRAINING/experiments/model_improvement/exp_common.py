"""Shared harness for the model-improvement experiments.

Guarantees, enforced here so no individual experiment can violate them:

  * The production bundle at 05_MODEL_TRAINING/models/final_model.joblib is
    opened READ-ONLY and never written.
  * Partitions come from the SAME functions and seed the production pipeline
    used (`patient_grouped_split`, `temporal_split`), so every number here is
    comparable with 06_MODEL_EVALUATION/evaluation_report.json.
  * The TEST partition is loaded but must only be touched by the final
    evaluation of an already-selected candidate. Selection uses validation.
  * The categorical encoder is fitted on the TRAINING fold only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config.paths import RANDOM_SEED                                  # noqa: E402
from src.models.dataset import (PAT, TARGET_BINARY, load_analytic,    # noqa: E402
                                patient_grouped_split,
                                split_categorical_numeric, temporal_split)
from src.models.preprocessing import (OrdinalCategoryEncoder,         # noqa: E402
                                      as_object_frame)
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = ROOT / "14_REPORT_MATERIAL" / "figures" / "model_improvement"
FIG_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_EVAL = ROOT / "06_MODEL_EVALUATION" / "evaluation_report.json"
FINAL_MODEL = ROOT / "05_MODEL_TRAINING" / "models" / "final_model.joblib"


def log(m):
    print(m, flush=True)


def full_metrics(y, p, threshold=0.5) -> dict:
    """Identical definition to scripts/08_evaluate_models.py::full_metrics."""
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive_rate": round(float(np.mean(y)), 5),
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 5),
        "precision_ppv": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall_sensitivity": round(float(recall_score(y, pred, zero_division=0)), 5),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 5),
        "npv": round(float(tn / (tn + fn)) if (tn + fn) else 0.0, 5),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 5),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 5),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def baseline() -> dict:
    """The authoritative baseline, READ FROM the existing evaluation artefact."""
    return json.loads(BASELINE_EVAL.read_text(encoding="utf-8"))


def production_features() -> list:
    """The deployed 30-feature list, read from the production bundle metadata."""
    meta = json.loads(
        (FINAL_MODEL.with_suffix(".metadata.json")).read_text(encoding="utf-8"))
    return list(meta["features"]), dict(meta["metadata"].get("hyperparameters") or {})


class Data:
    """Loads once, encodes once, reused by every experiment."""

    def __init__(self, features=None):
        t0 = time.time()
        self.df = load_analytic()
        self.features, self.best_params = production_features()
        if features is not None:
            self.features = list(features)
        self.cat_cols, self.num_cols = split_categorical_numeric(
            self.df, self.features)
        self.split = patient_grouped_split(self.df)
        self.y = self.df[TARGET_BINARY].to_numpy()
        log(f"  loaded {len(self.df):,} rows in {time.time()-t0:.1f}s; "
            f"{len(self.features)} features")

    # -- patient-grouped partitions ----------------------------------------
    def encode_grouped(self):
        m_tr, m_va, m_te = (self.split == "train"), (self.split == "val"), (self.split == "test")
        tr, va, te = self.df[m_tr], self.df[m_va], self.df[m_te]
        enc = OrdinalCategoryEncoder(columns=self.cat_cols).fit(
            as_object_frame(tr[self.features], self.cat_cols))
        X = lambda d: enc.transform(as_object_frame(d[self.features], self.cat_cols))
        return dict(
            enc=enc,
            Xtr=X(tr), ytr=self.y[m_tr.to_numpy()],
            Xva=X(va), yva=self.y[m_va.to_numpy()],
            Xte=X(te), yte=self.y[m_te.to_numpy()],
            groups_tr=tr[PAT].to_numpy(), pat_va=va[PAT].to_numpy(),
        )

    # -- temporal partitions (fit on the past, test on the future) ----------
    def encode_temporal(self):
        ts = temporal_split(self.df, train_frac=0.80)
        ttr, tte = self.df[ts == "train"], self.df[ts == "test"]
        enc = OrdinalCategoryEncoder(columns=self.cat_cols).fit(
            as_object_frame(ttr[self.features], self.cat_cols))
        X = lambda d: enc.transform(as_object_frame(d[self.features], self.cat_cols))
        return dict(
            Xtr=X(ttr), ytr=ttr[TARGET_BINARY].to_numpy(),
            Xte=X(tte), yte=tte[TARGET_BINARY].to_numpy(),
            groups_tr=ttr[PAT].to_numpy(),
        )


def lgbm(params: dict, class_weight=None, seed=RANDOM_SEED):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params, n_jobs=-1, random_state=seed,
                          subsample_freq=1, verbose=-1,
                          class_weight=class_weight)


def val_halves(pat_va, seed=RANDOM_SEED):
    """Split the validation partition BY PATIENT into two disjoint halves.

    Needed because the threshold must not be chosen on the same rows that fitted
    the calibrator, or the selected operating point is optimistic.
    """
    pats = np.unique(pat_va)
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    half = set(pats[: len(pats) // 2])
    a = np.array([p in half for p in pat_va])
    return a, ~a
