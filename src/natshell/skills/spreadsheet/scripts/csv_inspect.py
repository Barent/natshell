#!/usr/bin/env python3
"""Inspect a CSV file: dialect, header, first 5 rows, row count."""
import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    print("Usage: csv_inspect.py <path>")
    sys.exit(1)

with open(path, newline="", encoding="utf-8-sig") as f:
    sample = f.read(8192)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        dialect = csv.excel
        has_header = True

    f.seek(0)
    reader = csv.DictReader(f, dialect=dialect)
    rows = []
    total = 0
    for row in reader:
        total += 1
        if total <= 5:
            rows.append(dict(row))

print(f"File: {path}")
print(f"Delimiter: {repr(dialect.delimiter)}")
print(f"Has header: {has_header}")
print(f"Columns: {list(rows[0].keys()) if rows else 'unknown'}")
print(f"First 5 rows ({total} total scanned, actual row count may be higher):")
for i, row in enumerate(rows, 1):
    print(f"  {i}: {row}")
