#!/usr/bin/env python3
"""Replace text in a .docx file at the run level (preserves styles). Requires python-docx."""
import sys

try:
    import docx
except ImportError:
    print("python-docx not installed. Run: pip install --user python-docx")
    sys.exit(1)

if len(sys.argv) < 5:
    print("Usage: docx_replace.py <input.docx> <output.docx> <old_text> <new_text>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]
old_text = sys.argv[3]
new_text = sys.argv[4]

doc = docx.Document(input_path)
replaced = 0

for para in doc.paragraphs:
    for run in para.runs:
        if old_text in run.text:
            count = run.text.count(old_text)
            run.text = run.text.replace(old_text, new_text)
            replaced += count

for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    if old_text in run.text:
                        count = run.text.count(old_text)
                        run.text = run.text.replace(old_text, new_text)
                        replaced += count

doc.save(output_path)
print(f"Replaced {replaced} occurrence(s) of {old_text!r} → {new_text!r}")
print(f"Saved to: {output_path}")
