# Provider differential report

`pvec::differential` (feature `pvec-normalized`) runs any number of `DwgProvider`s over the same fixtures
and compares them (issue #1395).

```rust
let mut m = Matrix::new("r2-open");                 // the corpus revision, printed on every result line
m.run::<AcadrustProvider>(&fixtures)?;              // one call per provider that is compiled in
m.absent("acadsharp-rs", "no DwgProvider adapter"); // one per provider that is not, with why
let deltas = Deltas::parse(include_str!("provider_deltas.toml"))?;
assert!(m.verdict(&deltas).passed(), "{}", m.report(&deltas));
```

## What is compared

Each fixture's model space is flattened through the provider (`BlockResolver`), so an INSERT chain is
compared as the leaves it produces. One row per leaf, keyed `<instance path>#<handle>` (a repeated key,
such as a MINSERT cell, gets `~n`), and one row per INSERT that could not be expanded.

| field | meaning |
|---|---|
| `present` | the entity exists for one provider and not the other |
| `class_name`, `layer` | what the provider calls it and where it sits |
| `bucket`, `reason` | the N3.5 class tag and reason code, per entity, never in aggregate |
| `pvec_type` | the PVec semantic type of N6.2's table |
| `param:<name>` | a structural parameter (`end.x`, `radius`, `v3.bulge`) or the composed map (`matrix.tx`), bit for bit |

There is no epsilon: libviprs defines none, and two readers of the same bytes should agree exactly. A
difference that is a real and accepted one goes in `tests/fixtures/pvec/provider_deltas.toml` with a
written reason.

## What a result means

* **Equivalent** prints `agreement=DUAL_ORACLE_EQUIVALENT` next to the fixture's own `evidence=` class.
  Two providers agreeing is never source truth, and the harness never writes `KNOWN_TRUTH_EXACT` itself:
  that class comes in with a fixture from EPIC O and is only echoed.
* **Partial** (fewer than two providers ran, or one was declared absent) names the missing provider and
  says that no differential comparison was performed. It is not a pass, whatever the one provider said.
* **Failed** lists the unlisted differences and the stale entries. An entry that matches no difference
  fails the run, so the list cannot rot.

Every `result` line carries `platform=<arch>-<os>` and `corpus=<revision>`. The ACadSharp-backed provider
and the `O2.x` oracle converters are x86_64, so a run that includes them executes in a container on a native
x64 host (`tools/nas.sh`, MARS), never under emulation, and the line says which platform it was.

## What is not here yet

* The second provider adapter (an ACadSharp-backed `DwgProvider`) waits on libviprs-dep#161. Until then a
  real run is `PARTIAL`, and the suite's second provider is a test double that wraps acadrust and changes
  one thing.
* The corpus is the two provider fixtures. The known-truth set arrives with the loader of #1050.
