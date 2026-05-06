# Jest reference

## Common flags
```bash
npx jest --no-coverage     # skip coverage (faster)
npx jest --watchAll=false  # non-interactive (for CI)
npx jest tests/foo.test.js # run single file
npx jest -t "my test name" # filter by test name
npx jest --verbose         # detailed output
npx jest --bail            # stop on first failure
```

## Matchers
```javascript
expect(value).toBe(expected)           // strict equality
expect(value).toEqual(expected)        // deep equality
expect(value).toBeNull()
expect(value).toBeTruthy()
expect(fn).toThrow("message")
expect(arr).toContain(item)
expect(str).toMatch(/regex/)
expect(mock).toHaveBeenCalledWith(arg)
expect(mock).toHaveBeenCalledTimes(n)
```

## Mocking
```javascript
jest.mock("./module")                  // auto-mock
const fn = jest.fn().mockReturnValue(42)
jest.spyOn(obj, "method").mockImplementation(() => 42)
```

## Async tests
```javascript
test("async", async () => {
    const result = await fetchData()
    expect(result).toEqual(expected)
})
```

## beforeEach / afterEach
```javascript
beforeEach(() => { /* setup */ })
afterEach(() => { /* teardown */ })
```
