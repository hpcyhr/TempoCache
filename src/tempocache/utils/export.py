"""Export helpers for diagnostics and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, payload: Mapping) -> Path:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return p


def dump_csv_rows(path: str | Path, rows: Iterable[Mapping]) -> Path:
    rows = list(rows)
    p = ensure_parent(path)
    if not rows:
        with p.open("w", encoding="utf-8") as f:
            f.write("")
        return p
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p

