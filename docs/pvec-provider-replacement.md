# Replacing the DWG provider

A provider is replaceable if it can be held to acadrust's answers. The seam is `pvec::DwgProvider` (`docs/pvec-provider-api.md`) and the
harness is generic over it, so a second provider is one more call and needs no trait of its own.

## What a replacement must do

1. Implement `DwgProvider`: `probe`, `open`, `name`, `caps`, `layouts`, `layers`, `entities`, and the optional facets it can report.
   Declare in `caps()` exactly the facets it reports; a facet it cannot read is `CannotReport`, not an empty answer.
2. Hand out one block definition at a time, stamp each entity with the INSERT chain it was reached through, and leave INSERTs unexpanded.
3. Report knots with their repeats, weights and periodicity for splines, as `SourceFact`s, never invented.
4. Be pure in the sense of the build: no LibreDWG, no proprietary converter, no native library on the default path
   (`docs/pvec-providers.md`).

## The command that proves it

`pvec::differential` runs any number of providers over the same fixtures and compares per entity: structural parameters, the N3.5 class and
reason, and the PVec semantic type. Two providers agreeing is `DUAL_ORACLE_EQUIVALENT` and never source truth; a run with one provider is
partial, not a pass. The accepted differences are `tests/fixtures/pvec/provider_deltas.toml`, each with a written reason.

```text
cargo test --features pvec-convert --test pvec_provider_replacement --test pvec_provider_contract --test pvec_provider_boundary
```

The same fixtures drive the end to end suites: `--test pvec_known_truth`, `--test pvec_dual_oracle` and `--test pvec_corpus_regression`.
`docs/provider-differential.md` explains the report and how a delta is written.

## Today

`AcadrustProvider` is the only implementation. A second one over `acadsharp-rs` waits on a released wire version 3 (libviprs-dep#161 merged on 2026-11-01, the release has not
been cut), so until then the harness reports partial. `tests/pvec_1079_corpus_matrix.rs` runs the matrix over the pinned corpus revision.
