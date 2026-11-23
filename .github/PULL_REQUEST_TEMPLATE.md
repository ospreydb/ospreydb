## Summary

<!-- Describe what this PR changes and why. Link any related issue or RFC/ADR. -->

## Verification Checklist

- [ ] `cargo fmt --all -- --check` passes with zero formatting diffs
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` passes with zero warnings
- [ ] `cargo test --workspace --release` passes (and unit/integration tests cover the new behavior)
- [ ] Checked `AGENTS.md` invariants: 64-byte alignment (`#[repr(C, align(64))]`), no `.unwrap()` / `.expect()` / `panic!()` outside `#[cfg(test)]`, no heap allocations inside `osprey-simd` hot loops
