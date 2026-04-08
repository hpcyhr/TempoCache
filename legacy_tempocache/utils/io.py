
"""tempocache/utils/io.py – artifact export helpers."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def export_json(data: Any, path: str) -> None:
    """Write *data* as pretty-printed JSON."""
    _ensure_dir(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [export] JSON → {path}")


def export_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Write a list of dicts as a CSV file.  Handles heterogeneous key sets."""
    if not rows:
        return
    _ensure_dir(path)
    # Union of all keys, preserving insertion order
    keys: list = []
    seen: set = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})
    print(f"  [export] CSV  → {path}")