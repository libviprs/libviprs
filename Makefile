.PHONY: ci fmt clippy test miri loom

## Run all CI workflows locally (mirrors .github/workflows/ci.yml)
ci: fmt clippy test miri loom
	@echo ""
	@echo "All CI checks passed."

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
