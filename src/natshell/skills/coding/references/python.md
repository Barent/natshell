# Python reference

## Project layout signals
- `pyproject.toml` or `setup.py` → project root
- `src/` layout: source is under `src/<package>/`
- `tests/` at root level; may also be `test/`

## Run tests
```bash
pytest                     # all tests
pytest tests/test_foo.py   # single file
pytest -k "test_bar"       # by name filter
pytest --lf                # last-failed only
pytest -x                  # stop on first failure
```

## Check syntax / type errors
```bash
python3 -m py_compile path/to/file.py
mypy src/                  # if mypy is installed
ruff check src/            # if ruff is installed
```

## Common idioms
- Use `dataclasses.dataclass` for plain data containers
- Prefer `pathlib.Path` over `os.path`
- Use `logging.getLogger(__name__)` — never print for library code
- `from __future__ import annotations` for PEP 563 deferred annotations
- Type hints: `str | None`, `list[str]`, `dict[str, Any]`

## Gotchas
- Mutable default arguments: `def f(x=[])` — use `None` and set inside body
- `import *` pollutes namespace — avoid in library code
- `asyncio.run()` cannot be called from inside a running event loop
- `f.readlines()` loads entire file; prefer iteration: `for line in f:`
