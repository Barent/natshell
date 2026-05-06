#!/usr/bin/env python3
"""Summarize a CSV file: column stats, null counts, value distributions."""
import csv
import sys
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: csv_summary.py <file.csv>")
    sys.exit(1)

path = sys.argv[1]
max_values = int(sys.argv[2]) if len(sys.argv) > 2 else 10

with open(path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        print("Empty or headerless file.")
        sys.exit(0)
    cols = reader.fieldnames
    counts: dict[str, Counter] = {c: Counter() for c in cols}
    nulls: dict[str, int] = {c: 0 for c in cols}
    total = 0
    for row in reader:
        total += 1
        for col in cols:
            val = row.get(col, "")
            if val == "" or val is None:
                nulls[col] += 1
            else:
                counts[col][val] += 1

print(f"File: {path}")
print(f"Rows: {total}, Columns: {len(cols)}")
print(f"Columns: {cols}\n")

for col in cols:
    unique = len(counts[col])
    null_pct = nulls[col] / total * 100 if total else 0
    print(f"  [{col}]  unique={unique}  nulls={nulls[col]} ({null_pct:.1f}%)")
    for val, n in counts[col].most_common(max_values):
        pct = n / total * 100
        print(f"    {val!r}: {n} ({pct:.1f}%)")
    if unique > max_values:
        print(f"    ... ({unique - max_values} more unique values)")
    print()
