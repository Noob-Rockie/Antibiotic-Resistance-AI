"""The serialised artefact that everything downstream consumes.

A `ModelBundle` is deliberately more than a pickled estimator. Anything the
application needs in order to make an honest prediction travels with the model:

  model               fitted estimator (probability output)
  calibrator          optional isotonic calibrator fitted on the validation split
  encoder             the categorical encoder fitted on the training fold
  features            exact ordered feature list the model expects
  categorical         which of those are categorical
  defaults            training-set medians / modes, used to fill a field the
                      clinician has not entered (and recorded as "assumed")
  pair_support        training row count per (organism, antibiotic), so the UI
                      can say how much evidence sits behind a prediction
  antibiotics         the agents the model may be asked to score
  metrics             held-out performance, so the UI never quotes a number
                      that was not measured
  metadata            seed, dataset fingerprint, versions, training timestamp

Loading a bundle therefore reconstructs the full inference contract, and no
part of the application has to re-derive it (or guess it).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import MODELS_DIR  # noqa: E402


@dataclass
class ModelBundle:
    model: object
    encoder: object
    features: list
    categorical: list
    defaults: dict
    pair_support: dict = field(default_factory=dict)
    antibiotics: list = field(default_factory=list)
    organisms: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    calibrator: object = None
    threshold: float = 0.5

    # -- inference ----------------------------------------------------------
    def prepare(self, records: pd.DataFrame) -> pd.DataFrame:
        """Align an arbitrary record frame to the model's exact feature list."""
        X = records.copy()
        for f in self.features:
            if f not in X.columns:
                X[f] = self.defaults.get(f, np.nan)
        X = X[self.features]
        for c in self.categorical:
            X[c] = X[c].astype("object")
        return self.encoder.transform(X)

    def predict_nonsusceptible_proba(self, records: pd.DataFrame) -> np.ndarray:
        X = self.prepare(records)
        p = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)

    def support_for(self, organism: str, antibiotic: str) -> int:
        return int(self.pair_support.get(f"{organism}||{antibiotic}", 0))

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.setdefault(
            "saved_at_utc",
            datetime.now(timezone.utc).isoformat(timespec="seconds"))
        joblib.dump(self, path, compress=3)
        side = path.with_suffix(".metadata.json")
        side.write_text(json.dumps({
            "features": self.features,
            "categorical": self.categorical,
            "n_features": len(self.features),
            "antibiotics": self.antibiotics,
            "n_organisms": len(self.organisms),
            "metrics": self.metrics,
            "metadata": self.metadata,
            "threshold": self.threshold,
        }, indent=2, default=str), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path | None = None) -> "ModelBundle":
        path = Path(path) if path else (MODELS_DIR / "final_model.joblib")
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run "
                f"`python scripts/07_train_models.py` first.")
        return joblib.load(path)


def default_bundle_path() -> Path:
    return MODELS_DIR / "final_model.joblib"
