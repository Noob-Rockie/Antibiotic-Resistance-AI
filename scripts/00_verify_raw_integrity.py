"""Verify that no raw source file has been modified since the audit.

Compares current size + MD5 against 01_DATA/raw/RAW_DATA_MANIFEST.json.
Exit code 0 = unchanged, 1 = drift detected.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import RAW_FILES, RAW_REF_DIR  # noqa: E402

MANIFEST = RAW_REF_DIR / "RAW_DATA_MANIFEST.json"


def md5(path: Path, buf: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(buf):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}. Run scripts/01_dataset_audit.py first.")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = True
    for key, path in RAW_FILES.items():
        rec = manifest["files"].get(key)
        if rec is None:
            print(f"[NEW ] {key}: not in manifest")
            ok = False
            continue
        if not path.exists():
            print(f"[GONE] {key}: {path}")
            ok = False
            continue
        size = path.stat().st_size
        digest = md5(path)
        if size == rec["size_bytes"] and digest == rec["md5"]:
            print(f"[OK  ] {key}: unchanged ({size:,} bytes)")
        else:
            print(f"[DIFF] {key}: size {size:,} vs {rec['size_bytes']:,}, "
                  f"md5 {digest} vs {rec['md5']}")
            ok = False
    print("\nRAW DATA INTEGRITY:", "VERIFIED" if ok else "DRIFT DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
