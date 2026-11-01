# DWG providers behind `pvec::DwgProvider`

The analytic path (Epic N) reads a drawing through one seam, `pvec::DwgProvider`. Nothing past that
seam names a reader.

## The default is acadrust, and nothing defaults to acadsharp-rs

* **acadrust** is the default provider and the only one Epic N builds on (`pvec::dwg_acadrust::AcadrustProvider`,
  feature `dwg-acadrust`). It is pure Rust, needs no native library and is the provider every test and
  every report names first. The decision is on the closed #1239.
* **acadsharp-rs** is opt-in and documented as such. It is the ACadSharp decoder behind
  `cad::CadDecoder` (the tile path), reached only through the feature `dwg-acadsharp`, which is off by
  default and is not enabled by any other feature. It owns a native library, so a build with the feature
  off never fetches or links it. It is not a `pvec::DwgProvider` today: a second `DwgProvider` over it
  waits on libviprs-dep#161 and a released wire version 3.
* Cargo's `default` feature set is empty, so `cargo build` has no DWG reader at all, and no `src/pvec`
  code mentions acadsharp-rs.

`tests/pvec_1079_provider_default.rs` enforces all three: the default feature closure holds neither
provider's dependency, `acadsharp-rs` is an optional dependency that only `dwg-acadsharp` enables, and no
non-comment line of `src/pvec` names it. The same test checks that this page keeps saying so, and that its
own checks fail when `Cargo.toml` is mutated to default to acadsharp-rs.

## Switching or adding a provider

A provider implements `DwgProvider` and nothing else. It is held to acadrust's answers with the harness in
[provider-differential.md](provider-differential.md), which is generic over `DwgProvider`, so a second
provider is one more `Matrix::run` and needs no trait of its own.

LibreDWG is the planned open alternative for a second lineage. It is GPL, so it stays outside this crate
behind a process boundary, the way `pvec-oracle-libredwg` does for the corpus: a provider over it reads
the JSON the container writes, and nothing links it. That is a future adapter, not something this crate
ships.
