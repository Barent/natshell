#!/usr/bin/env python3
"""Convert an XLSX file to CSV. Requires openpyxl."""
import csv
import sys

try:
    import openpyxl
except ImportError:
    print("openpyxl not installed. Run: pip install --user openpyxl")
    sys.exit(1)

if len(sys.argv) < 3:
    print("Usage: xlsx_to_csv.py <input.xlsx> <output.csv> [sheet_name]")
    sys.exit(1)

xlsx_path = sys.argv[1]
csv_path = sys.argv[2]
sheet_name = sys.argv[3] if len(sys.argv) > 3 else None

wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
ws = wb[sheet_name] if sheet_name else wb.active
rows_written = 0

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in ws.iter_rows(values_only=True):
        writer.writerow(["" if v is None else str(v) for v in row])
        rows_written += 1

wb.close()
print(f"Wrote {rows_written} rows to {csv_path}")
