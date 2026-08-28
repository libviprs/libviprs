.PHONY: ci fmt clippy test miri loom doc

## Run all CI workflows locally (mirrors .github/workflows/ci.yml)
ci: fmt clippy test doc miri loom
	@echo ""
	@echo "All CI checks passed."

## Build the docs and fail on any broken intra-doc link (Docs job).
## Runs with every feature so the gated surface resolves (issue #146).
doc:
	@echo "==> cargo doc (deny broken intra-doc links)"
	RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps --all-features

## Check formatting (Check & Lint job)
fmt:
	@echo "==> cargo fmt --check"
	cargo fmt -- --check

## Run clippy with and without pdfium feature (Check & Lint job)
clippy:
	@echo "==> cargo clippy"
	RUSTFLAGS="-Dwarnings" cargo clippy --all-targets -- -D warnings
	@echo "==> cargo clippy (pdfium)"
	RUSTFLAGS="-Dwarnings" cargo clippy --all-targets --features pdfium -- -D warnings

## Run tests (Test job)
test:
	@echo "==> cargo test"
	RUSTFLAGS="-Dwarnings" cargo test

## Run Miri (Miri job — requires nightly)
##
## This is the local mirror of the `miri` job in
## `.github/workflows/merge-gate.yml`, and the flags have to match it or a local
## green and a hosted green stop meaning the same thing.
## `tests/miri_invocation_parity.rs` holds the two in step, and the workflow
## carries the reasoning for each flag. The short version: `-A deprecated` is
## what lets the crate compile under nightly at all (#643), and
## `--cfg sha2_backend="soft"` keeps the run off sha2's aarch64 NEON path, which
## aborts it on a Stacked Borrows violation about 30 seconds in (#707).
##
## `cargo +nightly` has to resolve to something at or past the 1.97 MSRV. As of
## 2026-08-28, `nightly-2026-08-20` is rustc 1.100.0 and works; an older nightly
## pinned in `rustup` will fail to build the crate rather than fail Miri.
miri:
	@echo "==> cargo +nightly miri test"
	RUSTFLAGS='-A deprecated --cfg sha2_backend="soft"' cargo +nightly miri test

## Run Loom concurrency tests (Loom job)
loom:
	@echo "==> loom tests"
	RUSTFLAGS="--cfg loom" cargo test --lib loom_tests
