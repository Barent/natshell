# Go reference

## Project layout signals
- `go.mod` → module root (also the project root)
- `cmd/` for main packages, `pkg/` or `internal/` for libraries
- `_test.go` suffix for test files

## Run tests
```bash
go test ./...              # all packages
go test ./pkg/foo/...      # specific subtree
go test -run TestFooBar    # by name
go test -v                 # verbose
go test -count=1           # disable test caching
```

## Build / check
```bash
go build ./...             # compile all packages
go vet ./...               # static analysis
go fmt ./...               # format
golangci-lint run          # if installed
```

## Common idioms
- Error handling: `if err != nil { return fmt.Errorf("context: %w", err) }`
- Use `errors.Is` / `errors.As` for error comparison, not string matching
- Prefer interfaces over concrete types in function signatures
- `defer f.Close()` immediately after opening a resource
- Context propagation: accept `ctx context.Context` as first arg for I/O-bound functions

## Gotchas
- Goroutine leak: always cancel contexts and drain channels
- Slice aliasing: `b := a[1:3]` shares the backing array with `a`
- `nil` interface vs typed nil: a typed nil pointer returned as an interface is non-nil
- `range` over a map is non-deterministic; sort keys for stable output
