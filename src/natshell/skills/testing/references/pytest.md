# pytest reference

## Common flags
```bash
pytest -v                  # verbose test names
pytest -s                  # don't capture output (show print())
pytest -x                  # stop on first failure
pytest --lf                # run last-failed tests only
pytest --ff                # run failed first, then rest
pytest -k "foo and not bar" # filter by keyword expression
pytest --tb=short          # shorter tracebacks
pytest --tb=long           # full tracebacks
pytest -p no:warnings      # suppress warnings
pytest --co -q             # collect (list) tests without running
```

## Markers
```bash
pytest -m slow             # run only tests marked @pytest.mark.slow
pytest -m "not slow"       # exclude slow tests
```

## Fixtures
```python
@pytest.fixture
def my_fixture():
    # setup
    yield value
    # teardown

def test_something(my_fixture):
    assert my_fixture == expected
```

## Parametrize
```python
@pytest.mark.parametrize("x,expected", [(1, 2), (3, 4)])
def test_add(x, expected):
    assert add(x, 1) == expected
```

## Mock
```python
from unittest.mock import patch, MagicMock
with patch("mymodule.some_function") as mock_fn:
    mock_fn.return_value = 42
    result = code_under_test()
    mock_fn.assert_called_once_with("arg")
```

## Assert exceptions
```python
with pytest.raises(ValueError, match="bad input"):
    function_that_raises()
```
