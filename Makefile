.PHONY: ci fmt clippy test miri loom doc

## Run all CI workflows locally (mirrors .github/workflows/ci.yml)
ci: fmt clippy test doc miri loom
	@echo ""
	@echo "All CI checks passed."

## Build the docs and fail on any broken or private intra-doc link (Docs job).
## Runs with every feature so the gated surface resolves (issue #146).
## `private_intra_doc_links` is warn-by-default, so a public doc linking a
## `pub(crate)` item rendered as literal brackets and got past this target.
doc:
	@echo "==> cargo doc (deny broken and private intra-doc links)"
	RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links -D rustdoc::private_intra_doc_links" cargo doc --no-deps --all-features

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
miri:
	@echo "==> cargo +nightly miri test"
	cargo +nightly miri test

## Run Loom concurrency tests (Loom job)
loom:
	@echo "==> loom tests"
	RUSTFLAGS="--cfg loom" cargo test --lib loom_tests
