---
name: spreadsheet
description: Read, write, and transform CSV and XLSX files. Use for Excel, tabular data, bulk row edits, and column transforms.
---

# Spreadsheet skill

## When to use
- User mentions CSV, Excel, XLSX, or tabular/spreadsheet data
- User wants to filter, transform, aggregate, or reformat rows/columns
- User wants to convert between CSV and XLSX
- User wants to inspect, summarize, or fix a data file

## When NOT to use
- The file is a database (SQLite, PostgreSQL) — use data-analysis skill or execute_shell with sqlite3
- The file is a JSON/log — use data-analysis skill
- The user wants to edit a Word document — use docx skill

## Procedure
1. Identify the file path and format (CSV or XLSX). Run `scripts/csv_inspect.py` or `scripts/xlsx_to_csv.py` first to understand structure.
2. For CSV: use Python stdlib `csv` module via `run_code`.
3. For XLSX: check if `openpyxl` is available. If not, install via `execute_shell`: `pip install --user openpyxl`. Then use `run_code`.
4. Read the header row first before writing any transforms.
5. Write output to a new file (don't overwrite the original unless the user asks).
6. Report row count and first few rows to confirm correctness.

## Recipes

**Inspect CSV (dialect, header, first 5 rows):**
```python
import csv, sys
path = "/path/to/file.csv"
with open(path, newline="", encoding="utf-8-sig") as f:
    dialect = csv.Sniffer().sniff(f.read(4096))
    f.seek(0)
    reader = csv.DictReader(f, dialect=dialect)
    rows = []
    for i, row in enumerate(reader):
        if i >= 5:
            break
        rows.append(row)
print("Header:", reader.fieldnames)
print("Rows:", rows)
```

**Filter CSV rows where column equals value:**
```python
import csv
with open("in.csv", newline="") as fin, open("out.csv", "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if row["status"] == "active":
            writer.writerow(row)
```

**Read XLSX sheet:**
```python
import openpyxl
wb = openpyxl.load_workbook("/path/to/file.xlsx", read_only=True, data_only=True)
ws = wb.active
for row in ws.iter_rows(values_only=True):
    print(row)
wb.close()
```

**Write XLSX:**
```python
import openpyxl
wb = openpyxl.Workbook()
ws = wb.active
ws.append(["Name", "Score"])
ws.append(["Alice", 95])
wb.save("/path/to/output.xlsx")
```

**Convert CSV to XLSX:**
Use the bundled `scripts/xlsx_to_csv.py` (in reverse) or write directly:
```python
import csv, openpyxl
wb = openpyxl.Workbook()
ws = wb.active
with open("data.csv", newline="") as f:
    for row in csv.reader(f):
        ws.append(row)
wb.save("data.xlsx")
```

## Pitfalls
- XLSX files with formulas: `data_only=True` reads the cached value (may be stale if never opened in Excel). Never modify formulas by editing cell values — use `openpyxl` formula strings.
- CSV encoding: use `utf-8-sig` to handle BOM from Excel exports.
- Large files (>100k rows): prefer `read_only=True` in openpyxl; use `csv.DictReader` (streaming) not `readlines()`.
- `run_code` Python is stdlib-only. `openpyxl` is a third-party package — install via `execute_shell` first.
- Newlines inside CSV cells require the `csv` module — never parse CSV by hand with `split(",")`.
- Do not overwrite the original file without explicit user consent.

## References
- `scripts/csv_inspect.py` — inspect CSV dialect, header, row count, first 5 rows
- `scripts/xlsx_to_csv.py` — convert XLSX to CSV
