#!/usr/bin/env python3
"""Extract text from a .docx file showing paragraph/run structure. Requires python-docx."""
import sys

try:
    import docx
except ImportError:
    print("python-docx not installed. Run: pip install --user python-docx")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: docx_text.py <file.docx>")
    sys.exit(1)

path = sys.argv[1]
doc = docx.Document(path)

print(f"File: {path}")
print(f"Paragraphs: {len(doc.paragraphs)}, Tables: {len(doc.tables)}\n")

for i, para in enumerate(doc.paragraphs):
    style = para.style.name
    runs = [f"[{r.text!r}]" for r in para.runs if r.text]
    if para.text.strip():
        print(f"Para {i+1} ({style}): {para.text[:120]}")
        if len(para.runs) > 1:
            print(f"  Runs: {' '.join(runs)}")
