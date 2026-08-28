.PHONY: ci fmt clippy test miri loom doc

## Run all CI workflows locally (mirrors .github/workflows/ci.yml)
ci: fmt clippy test doc miri loom
	@echo ""
	@echo "All CI checks passed."

## Build the docs and fail on any bad intra-doc link (Docs job).
## Runs with every feature so the gated surface resolves (issue #146).
## `private_intra_doc_links` and `redundant_explicit_links` are both
## warn-by-default: a link to a `pub(crate)` item renders as inert bracketed
## text on docs.rs, and a redundant explicit target is noise that hides the
## next real warning. Both passed the gate until issues #697 and #795.
## `tests/doc_link_gate.rs` holds this line and the workflow's together.
doc:
	@echo "==> cargo doc (deny broken, private and redundant intra-doc links)"
	RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links -D rustdoc::private_intra_doc_links -D rustdoc::redundant_explicit_links" cargo doc --no-deps --all-features

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

## The nightly `make miri` uses. It is a dated toolchain rather than the
## floating `nightly` on purpose, and the reason is the whole of #675's first
## review finding: bare `+nightly` on this machine resolves to 1.96.0-nightly
## (2026-03-13), which is *below* the crate's 1.97 MSRV, so cargo refuses to
## build before Miri is ever reached. The workflow file used to carry a comment
## saying Miri "cannot run on the dev machine" for exactly that reason, and
## replacing that sentence with "it runs again" while leaving the recipe on
## `+nightly` swapped one unverified claim for another.
##
## Override it when a newer nightly is what you want:
##
##   make miri MIRI_TOOLCHAIN=nightly
##
## The recipe checks whatever it resolves to against the MSRV read out of
## `Cargo.toml`, so a toolchain that cannot work says so in one line instead of
## printing the MSRV refusal once per target.
MIRI_TOOLCHAIN ?= nightly-2026-08-20

## Run Miri (Miri job)
##
## This is the local mirror of the `miri` job in
## `.github/workflows/merge-gate.yml`. The flags have to match it, and
## `tests/miri_invocation_parity.rs` fails if they drift. The workflow carries
## the reasoning for each one; the short version is that `-A deprecated` is what
## lets the crate compile under nightly at all (#643), and
## `--cfg sha2_backend="soft"` keeps the run off sha2's aarch64 NEON path, which
## aborts it on a Stacked Borrows violation about 30 seconds in (#707).
##
## What this cannot mirror is the compiler. The hosted job resolves whatever
## `dtolnay/rust-toolchain@nightly` gives it on the day, and this pins a date,
## so the two run the same command and the same flags on different nightlies.
## Treat a local green as evidence about the crate, not as a prediction of the
## hosted run.
miri:
	@echo "==> cargo +$(MIRI_TOOLCHAIN) miri test"
	@msrv=$$(sed -n 's/^rust-version = "\(.*\)"/\1/p' Cargo.toml | head -1); \
	have=$$(rustc +$(MIRI_TOOLCHAIN) --version 2>/dev/null | awk '{print $$2}'); \
	if [ -z "$$have" ]; then \
		echo "error: toolchain '$(MIRI_TOOLCHAIN)' is not installed. Install it, or pick another with 'make miri MIRI_TOOLCHAIN=<name>'." >&2; \
		exit 1; \
	fi; \
	if [ "$$(printf '%s\n%s\n' "$$msrv" "$$have" | sort -V | head -1)" != "$$msrv" ]; then \
		echo "error: toolchain '$(MIRI_TOOLCHAIN)' is rustc $$have, below this crate's MSRV of $$msrv. Cargo refuses to build it, so Miri never runs. Pick a newer nightly with 'make miri MIRI_TOOLCHAIN=<name>'." >&2; \
		exit 1; \
	fi; \
	cargo +$(MIRI_TOOLCHAIN) miri --version >/dev/null 2>&1 || { \
		echo "error: miri is not installed for '$(MIRI_TOOLCHAIN)'. Run 'rustup component add miri --toolchain $(MIRI_TOOLCHAIN)'." >&2; \
		exit 1; \
	}
	RUSTFLAGS='-A deprecated --cfg sha2_backend="soft"' cargo +$(MIRI_TOOLCHAIN) miri test

## Run Loom concurrency tests (Loom job)
loom:
	@echo "==> loom tests"
	RUSTFLAGS="--cfg loom" cargo test --lib loom_tests
