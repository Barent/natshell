# Rust reference

## Project layout signals
- `Cargo.toml` → crate root
- `src/lib.rs` or `src/main.rs` → source entry point
- `tests/` for integration tests; `#[cfg(test)]` modules for unit tests

## Run tests
```bash
cargo test                 # all tests
cargo test test_foo        # by name filter
cargo test -- --nocapture  # show println output
cargo nextest run          # if nextest is installed (faster)
```

## Build / check
```bash
cargo check                # fast type check (no binary)
cargo build                # debug build
cargo build --release      # release build
cargo clippy               # lints
cargo fmt                  # format
```

## Common idioms
- Prefer `Result<T, E>` and `Option<T>` over panics in library code
- Use `?` operator for error propagation
- `Clone` only when necessary; prefer references
- `String` owns data; `&str` is a borrowed slice
- `Vec<T>` owns elements; `&[T]` is a borrowed slice

## Gotchas
- Borrow checker: cannot have mutable and immutable borrows simultaneously
- `unwrap()` / `expect()` panic at runtime — only use in tests or infallible paths
- Trait objects `dyn Trait` require `Box<dyn Trait>` for heap allocation
- `Copy` types are implicitly cloned; non-`Copy` types are moved
- Lifetimes: if the compiler complains, add lifetime annotations to function signatures
