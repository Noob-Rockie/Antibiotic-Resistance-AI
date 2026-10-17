"""PHASE 1 - Complete dataset audit.

Profiles every raw file WITHOUT loading it fully into memory and without
touching the source files. Emits a machine-readable JSON profile that the
documentation generators and later phases consume.

Output: 02_DATA_ANALYSIS/reports/dataset_audit.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import ANALYSIS_DIR, NA_VALUES, RAW_FILES, RAW_REF_DIR  # noqa: E402

CHUNK = 500_000
MAX_TRACKED_UNIQUE = 20_000   # cap on exact unique-value tracking per column
TOP_N = 40                    # value-counts retained for low-cardinality cols


def file_md5(path: Path, buf: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(buf):
            h.update(block)
    return h.hexdigest()


class ColumnProfile:
    """Streaming profile for one column."""

    def __init__(self, name: str):
        self.name = name
        self.n = 0
        self.n_missing = 0
        self.values: Counter = Counter()
        self.unique_overflow = False
        self.numeric_ok = True
        self.num_min = None
        self.num_max = None
        self.num_sum = 0.0
        self.num_sumsq = 0.0
        self.num_count = 0
        self.sample: list = []

    def update(self, s: pd.Series) -> None:
        self.n += len(s)
        self.n_missing += int(s.isna().sum())
        nonnull = s.dropna()
        if len(self.sample) < 5 and len(nonnull):
            self.sample.extend(nonnull.head(5 - len(self.sample)).astype(str).tolist())

        if not self.unique_overflow:
            self.values.update(nonnull.astype(str).tolist())
            if len(self.values) > MAX_TRACKED_UNIQUE:
                self.unique_overflow = True
                # keep only the most common so memory stays bounded
                self.values = Counter(dict(self.values.most_common(TOP_N)))

        if self.numeric_ok:
            num = pd.to_numeric(nonnull, errors="coerce")
            if len(nonnull) and num.isna().any():
                self.numeric_ok = False
                self.num_count = 0
            elif len(num):
                self.num_count += len(num)
                self.num_sum += float(num.sum())
                self.num_sumsq += float((num.astype("float64") ** 2).sum())
                lo, hi = float(num.min()), float(num.max())
                self.num_min = lo if self.num_min is None else min(self.num_min, lo)
                self.num_max = hi if self.num_max is None else max(self.num_max, hi)

    def to_dict(self) -> dict:
        d = {
            "column": self.name,
            "n_rows": self.n,
            "n_missing": self.n_missing,
            "pct_missing": round(100 * self.n_missing / self.n, 4) if self.n else None,
            "n_unique": (f">{MAX_TRACKED_UNIQUE}" if self.unique_overflow
                         else len(self.values)),
            "unique_exact": not self.unique_overflow,
            "sample_values": self.sample[:5],
            "is_numeric": bool(self.numeric_ok and self.num_count),
        }
        if d["is_numeric"]:
            mean = self.num_sum / self.num_count
            var = max(self.num_sumsq / self.num_count - mean**2, 0.0)
            d["numeric_summary"] = {
                "min": self.num_min, "max": self.num_max,
                "mean": round(mean, 4), "std": round(var**0.5, 4),
            }
        if not self.unique_overflow and len(self.values) <= TOP_N:
            d["value_counts"] = dict(self.values.most_common(TOP_N))
        else:
            d["top_values"] = dict(self.values.most_common(15))
        # constant / near-constant detection
        if self.values:
            top = self.values.most_common(1)[0][1]
            nonmissing = self.n - self.n_missing
            d["dominant_value_share"] = round(top / nonmissing, 4) if nonmissing else None
        return d


def profile_file(key: str, path: Path) -> dict:
    print(f"\n=== {key}: {path.name} ===", flush=True)
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    profiles = {c: ColumnProfile(c) for c in cols}

    n_rows = 0
    dup_hashes: Counter = Counter()
    # key-uniqueness tracking for the shared identifier triple
    id_cols = [c for c in ("anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded")
               if c in cols]
    id_counter: Counter = Counter()
    id_overflow = False

    reader = pd.read_csv(path, chunksize=CHUNK, na_values=NA_VALUES,
                         keep_default_na=True, dtype=str, low_memory=False)
    for i, chunk in enumerate(reader):
        n_rows += len(chunk)
        for c in cols:
            profiles[c].update(chunk[c])
        # exact full-row duplicate detection via hashing
        h = pd.util.hash_pandas_object(chunk.fillna("\x00NA"), index=False)
        dup_hashes.update(h.tolist())
        if id_cols and not id_overflow:
            joined = chunk[id_cols].astype(str).agg("|".join, axis=1)
            id_counter.update(joined.tolist())
            if len(id_counter) > 4_000_000:
                id_overflow = True
        print(f"  chunk {i+1}: {n_rows:,} rows", flush=True)

    n_dup = sum(c - 1 for c in dup_hashes.values() if c > 1)
    result = {
        "key": key,
        "file_name": path.name,
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "file_size_mb": round(path.stat().st_size / 1024**2, 2),
        "md5": file_md5(path),
        "n_rows": n_rows,
        "n_columns": len(cols),
        "columns": cols,
        "n_exact_duplicate_rows": n_dup,
        "pct_exact_duplicate_rows": round(100 * n_dup / n_rows, 4) if n_rows else None,
        "column_profiles": [profiles[c].to_dict() for c in cols],
    }
    if id_cols and not id_overflow:
        n_id_unique = len(id_counter)
        result["identifier_triple"] = {
            "columns": id_cols,
            "n_unique_combinations": n_id_unique,
            "rows_per_combination_mean": round(n_rows / n_id_unique, 4),
            "is_primary_key": n_id_unique == n_rows,
            "max_rows_for_one_combination": max(id_counter.values()),
        }
    return result


def main() -> None:
    out_dir = ANALYSIS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {"generated_by": "scripts/01_dataset_audit.py", "datasets": []}
    for key, path in RAW_FILES.items():
        audit["datasets"].append(profile_file(key, path))
    out = out_dir / "dataset_audit.json"
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    # Integrity manifest so raw-data drift is detectable later.
    manifest = {
        "source_directory": str(Path(audit["datasets"][0]["file_path"]).parent),
        "note": "Raw files are referenced in place and never modified.",
        "files": {
            d["key"]: {
                "file_name": d["file_name"],
                "size_bytes": d["file_size_bytes"],
                "md5": d["md5"],
                "n_rows": d["n_rows"],
                "n_columns": d["n_columns"],
            }
            for d in audit["datasets"]
        },
    }
    RAW_REF_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_REF_DIR / "RAW_DATA_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    lines = ["# Raw data manifest\n",
             "> Generated by `scripts/01_dataset_audit.py`. "
             "Verify with `python scripts/00_verify_raw_integrity.py`.\n",
             f"**Source directory:** `{manifest['source_directory']}`\n",
             "| Dataset | File | Size (bytes) | Rows | Cols | MD5 |",
             "|---|---|--:|--:|--:|---|"]
    for k, v in manifest["files"].items():
        lines.append(f"| `{k}` | `{v['file_name']}` | {v['size_bytes']:,} | "
                     f"{v['n_rows']:,} | {v['n_columns']} | `{v['md5']}` |")
    (RAW_REF_DIR / "RAW_DATA_MANIFEST.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RAW_REF_DIR / 'RAW_DATA_MANIFEST.md'}")


if __name__ == "__main__":
    main()
