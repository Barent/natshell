---
name: data-analysis
description: Inspect, filter, and summarize JSON, CSV, and log data using jq, awk, and pandas.
---

# Data analysis skill

## When to use
- User wants to query, filter, or aggregate structured data (JSON, CSV, NDJSON, log files)
- User wants statistics: counts, sums, averages, distributions
- User wants to join, reshape, or transform datasets

## When NOT to use
- The data is in a spreadsheet file (XLSX) — use spreadsheet skill
- The data is in a relational database — use execute_shell with `sqlite3` or `psql`
- The task is PDF extraction — use pdf skill

## Procedure
1. **Sample the data first**: `head -n 20 /path/to/file` before processing large files.
2. For JSON: use `jq` via execute_shell.
3. For CSV: use `awk` for simple column operations; pandas via run_code for multi-step transforms.
4. For logs: use grep + awk for pattern extraction, then aggregate.
5. Report summary statistics (row count, null counts, value ranges) before presenting results.

## Recipes

**Count lines / rows:**
```bash
wc -l data.csv
jq 'length' data.json
```

**Inspect JSON structure:**
```bash
jq 'keys' data.json          # top-level keys
jq '.[0]' data.json          # first element
jq 'type' data.json          # "array" | "object"
```

**Filter JSON array:**
```bash
jq '.[] | select(.status == "active")' data.json
jq '[.[] | select(.score > 80)]' data.json
```

**Extract fields:**
```bash
jq '.[] | {name, score}' data.json
jq -r '.[] | [.name, .score] | @csv' data.json   # CSV output
```

**Group and count in JSON:**
```bash
jq 'group_by(.category) | map({category: .[0].category, count: length})' data.json
```

**Sum a CSV column with awk:**
```bash
awk -F',' 'NR>1 {sum += $3} END {print sum}' data.csv   # sum column 3 (skip header)
```

**Count distinct values in a CSV column:**
```bash
awk -F',' 'NR>1 {print $2}' data.csv | sort | uniq -c | sort -rn
```

**Filter CSV rows by column value:**
```bash
awk -F',' '$3 == "active" {print}' data.csv
```

**Pandas: full summary (run_code):**
```python
import csv
from collections import Counter, defaultdict

path = "/path/to/data.csv"
with open(path, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print(f"Rows: {len(rows)}")
if rows:
    print(f"Columns: {list(rows[0].keys())}")
    # Value counts for first column
    col = list(rows[0].keys())[0]
    counts = Counter(r[col] for r in rows)
    print(f"\nTop values in {col!r}:")
    for val, n in counts.most_common(10):
        print(f"  {val}: {n}")
```

**NDJSON (newline-delimited JSON):**
```bash
jq -s '.' data.ndjson          # parse as array
jq -s 'length' data.ndjson     # count objects
jq -sc '.[] | select(.level == "error")' data.ndjson  # filter
```

**Extract log timestamps and messages:**
```bash
grep "ERROR" app.log | awk '{print $1, $2, $0}' | head -20
```

## Pitfalls
- `jq` is not always installed — check with `command -v jq`. If missing, install via package manager or use Python's `json` module.
- `awk` column indices are 1-based (not 0).
- CSV files with quoted commas or newlines inside fields break naive `awk` — use `csv` module via run_code instead.
- Large files (>1 GB): sample first with `head -n 100000` before full processing.
- `pandas` is a third-party package not available in run_code — use stdlib `csv` + `collections` instead.

## References
- `references/jq-recipes.md` — comprehensive jq examples
- `scripts/csv_summary.py` — column stats, null counts, value distributions
