---
name: docx
description: Read and edit Microsoft Word documents (.docx). Use for document editing and content extraction.
---

# Docx skill

## When to use
- User asks to read, edit, search, or extract content from a `.docx` file
- User wants to replace text, add paragraphs, or extract headings/tables from a Word document
- User wants to convert `.docx` content to plain text

## When NOT to use
- The file is an old binary `.doc` file — see pitfalls below
- The file is a PDF — use pdf skill
- The user wants to edit a spreadsheet — use spreadsheet skill

## Procedure
1. Check if `python-docx` is available. If not, install: `pip install --user python-docx`
2. Use `run_code` with `python-docx` to read, inspect, or modify the file.
3. Always read/inspect the document structure first before editing.
4. For text replacement, edit at the **run level** (not paragraph level) to preserve styles.
5. Write to a new file path unless the user explicitly asks to overwrite.

## Recipes

**Extract all text:**
```python
import docx
doc = docx.Document("/path/to/file.docx")
for para in doc.paragraphs:
    if para.text.strip():
        print(para.text)
```

**List headings:**
```python
import docx
doc = docx.Document("/path/to/file.docx")
for para in doc.paragraphs:
    if para.style.name.startswith("Heading"):
        print(f"[{para.style.name}] {para.text}")
```

**Extract tables:**
```python
import docx
doc = docx.Document("/path/to/file.docx")
for i, table in enumerate(doc.tables):
    print(f"Table {i+1}:")
    for row in table.rows:
        print([cell.text for cell in row.cells])
```

**Replace text (preserving styles — run level):**
```python
import docx
doc = docx.Document("/path/to/file.docx")
old_text = "FooBar Inc."
new_text = "Acme Corp."
for para in doc.paragraphs:
    for run in para.runs:
        if old_text in run.text:
            run.text = run.text.replace(old_text, new_text)
doc.save("/path/to/output.docx")
```

**Add a paragraph:**
```python
import docx
doc = docx.Document("/path/to/file.docx")
doc.add_paragraph("New paragraph text here.")
doc.save("/path/to/output.docx")
```

**Convert old .doc to .docx via LibreOffice:**
```bash
libreoffice --headless --convert-to docx /path/to/file.doc --outdir /tmp/
```

## Pitfalls
- Old binary `.doc` files cannot be read by `python-docx`. Convert to `.docx` first using LibreOffice.
- Replacing text at the paragraph level (via `para.text = ...`) strips all formatting. Always edit at the run level.
- Text may be split across multiple runs (e.g., "Foo**Bar**" could be 3 runs). Use `scripts/docx_text.py` to inspect.
- `python-docx` is a third-party package — install via `execute_shell` before `run_code`.
- Do not overwrite the original without explicit user consent.

## References
- `scripts/docx_text.py` — extract full document text with run/paragraph structure
- `scripts/docx_replace.py` — safe run-level text replacement
