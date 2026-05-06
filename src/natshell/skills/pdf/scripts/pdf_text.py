#!/usr/bin/env python3
"""Extract text from a PDF. Requires pypdf."""
import sys

try:
    import pypdf
except ImportError:
    print("pypdf not installed. Run: pip install --user pypdf")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: pdf_text.py <file.pdf> [start_page] [end_page]")
    print("  Pages are 1-indexed. Omit to extract all pages.")
    sys.exit(1)

path = sys.argv[1]
start = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0  # convert to 0-indexed
end = int(sys.argv[3]) if len(sys.argv) > 3 else None       # exclusive end

reader = pypdf.PdfReader(path)
total = len(reader.pages)
pages = reader.pages[start:end]

print(f"File: {path}  Total pages: {total}")
if not pages:
    print("No pages in range.")
    sys.exit(0)

for i, page in enumerate(pages, start + 1):
    text = page.extract_text() or ""
    if text.strip():
        print(f"\n--- Page {i} ---")
        print(text)
    else:
        print(f"\n--- Page {i} --- (no text extracted — may be scanned/image)")
