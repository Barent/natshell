#!/usr/bin/env python3
"""Merge multiple PDFs into one. Requires pypdf."""
import sys

try:
    import pypdf
except ImportError:
    print("pypdf not installed. Run: pip install --user pypdf")
    sys.exit(1)

if len(sys.argv) < 4:
    print("Usage: pdf_merge.py <output.pdf> <input1.pdf> <input2.pdf> [...]")
    sys.exit(1)

output = sys.argv[1]
inputs = sys.argv[2:]

merger = pypdf.PdfMerger()
for path in inputs:
    merger.append(path)
    print(f"  Added: {path}")

merger.write(output)
merger.close()
print(f"Merged {len(inputs)} PDFs → {output}")
