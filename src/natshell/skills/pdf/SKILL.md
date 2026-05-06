---
name: pdf
description: Extract text, merge, split, and search PDF documents. Use whenever the user mentions a PDF file.
---

# PDF skill

## When to use
- User asks to read, extract, or search content in a PDF
- User wants to merge multiple PDFs into one
- User wants to split a PDF into pages or ranges
- User asks about page count, metadata, or structure of a PDF

## When NOT to use
- The file is a Word document (.docx) — use docx skill
- The user wants to edit or annotate a PDF visually (use a GUI app)
- The file is a spreadsheet — use spreadsheet skill

## Procedure
1. Try `pypdf` (pure Python) via `run_code` first.
2. If `pypdf` is not installed, install via `execute_shell`: `pip install --user pypdf`
3. If the PDF is scanned (text extraction returns empty), try `pdftotext` (poppler) via `execute_shell`.
4. If still empty and OCR is needed, inform the user and suggest `tesseract` + `ocrmypdf`.
5. For merge/split, always write to a new file — never overwrite the original.

## Recipes

**Extract text from all pages:**
```python
import pypdf
reader = pypdf.PdfReader("/path/to/file.pdf")
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"--- Page {i+1} ---")
    print(text)
```

**Extract text from page range:**
```python
import pypdf
reader = pypdf.PdfReader("/path/to/file.pdf")
# Pages 3-5 (0-indexed: 2-4)
for page in reader.pages[2:5]:
    print(page.extract_text())
```

**Get page count and metadata:**
```python
import pypdf
reader = pypdf.PdfReader("/path/to/file.pdf")
print(f"Pages: {len(reader.pages)}")
print(f"Metadata: {reader.metadata}")
```

**Merge PDFs:**
```python
import pypdf
merger = pypdf.PdfMerger()
for path in ["/path/a.pdf", "/path/b.pdf", "/path/c.pdf"]:
    merger.append(path)
merger.write("/path/merged.pdf")
merger.close()
print("Merged.")
```

**Split: extract pages 3-7 to a new file:**
```python
import pypdf
reader = pypdf.PdfReader("/path/to/file.pdf")
writer = pypdf.PdfWriter()
for page in reader.pages[2:7]:  # 0-indexed
    writer.add_page(page)
with open("/path/pages_3_to_7.pdf", "wb") as f:
    writer.write(f)
```

**Fallback: pdftotext via execute_shell (requires poppler):**
```bash
pdftotext /path/to/file.pdf -  # stdout
pdftotext -f 3 -l 7 /path/to/file.pdf output.txt  # page range
```

**Check if poppler is installed:**
```bash
command -v pdftotext && echo "available" || echo "not found"
```

## Pitfalls
- Scanned PDFs contain images, not text — `pypdf` extraction returns empty strings. Use OCR fallback.
- Password-protected PDFs require a password: `reader = pypdf.PdfReader(path, password="secret")`.
- `pypdf` is a third-party package. Install via `execute_shell` before using in `run_code`.
- Some PDFs have malformed cross-reference tables — pypdf handles most but may fail on corrupt files.
- Do not overwrite the original PDF without explicit user consent.
- OCR (tesseract/ocrmypdf) can take minutes for large files — warn the user.

## References
- `scripts/pdf_text.py` — extract text with optional page range
- `scripts/pdf_merge.py` — merge a list of PDFs into one
