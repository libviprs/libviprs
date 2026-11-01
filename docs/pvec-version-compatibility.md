# PVec and provider version compatibility

This is libviprs' side of version compatibility: what a build does when it meets a PVec container older or
newer than the one it was compiled against, and which source drawings it will take. PVec publishes its own rules
(negotiation and unknown-data skipping, `docs/CONTAINER.md` in `libviprs/pvec`); this page says what the pinned
build of them does from here, and `tests/pvec_version_compat.rs` holds it to the code.

Every outcome is one of four words: **reads**, **writes**, **refuses**, **reads with a reduced capability set**.

The pin is `tests/fixtures/pvec_compat.json`. The test fails if the document and the pin disagree in either
direction, and if the pin and the code disagree: the PVec revision in `Cargo.toml`, the capability set an artifact
written by this build declares, the DWG versions the provider takes, and the provider revision.

## The PVec container

This build links `pvec` at revision `b9f441618153d256777121a6a1518b15e7f42099` and writes container version 1.0.

| direction | container version | outcome |
|---|---|---|
| older container | 0.x | refuses |
| current container | 1.0 | reads |
| current container | 1.0 | writes |
| older container, written by this build | 0.x | refuses |
| newer minor | 1.1 and up | reads |
| newer major | 2.0 and up | refuses |

- **Older container.** There is none to read: a major below 1 is refused with `UnsupportedVersion`, and this build
  writes only at the pinned version, never at an older one.
- **Newer minor.** A header one minor ahead opens. What it does with the blocks of a newer minor is PVec's rule
  (an unknown block is skipped only where its descriptor marks it skippable, and then the file opens degraded), and
  that is covered by PVec's own suite for negotiation (P4.7), not re-tested here: the facade this build links writes
  no block of a kind it has not heard of, so I cannot build the file that would test it.
- **Newer major.** Refused with `UnsupportedVersion`, naming the major found and the one supported. Nothing is
  best-effort parsed.
- **A capability this build cannot interpret.** A container that declares a required capability the reader does not
  implement is refused by PVec's negotiation (`NEGOTIATE-01`), naming the capability id, and a build with the conversion
  API refuses a semantic profile that needs a capability it does not write at preflight (the allowed-capability gate,
  `docs/pvec-diagnostics.md`). Reading such a container needs a hand-built manifest, which the facade does not offer, so
  that outcome is PVec's to test.

## What this build writes

The capabilities this build can write are the ones the mapping table (`pvec::mapping::NORMALIZED_TO_PVEC`, ten kinds of
normalized geometry) assigns to its kinds, ids 5 to 12 of PVec's registry, listed in the pin as `capabilities_written`.
An artifact declares a subset of them, and the test checks that: it converts a mixed drawing (lines, arcs, circles,
polylines and splines) and compares the manifest with the pin. The funnel pushes curves and no stream, so PVec derives
the capability list from the curves it is given and a drawing that holds only some of the kinds declares fewer.

## Source providers

| provider | source version | minimum | outcome |
|---|---|---|---|
| acadrust (default) | AC1014 to AC1032 (R14, 2000, 2004, 2007, 2010, 2013, 2018) | AC1014 | reads |
| acadrust (default) | below AC1014, or an unknown stamp | AC1014 | refuses |

- The provider is `acadrust` at revision `72af48970e306ef1d50dd751b227df2b4fc3fb99` of the `libviprs/acadrust` fork.
  `acadsharp-rs` is opt-in (the `dwg-acadsharp` feature) and takes the same DWG versions through `cad::CadDecoder`
  (`docs/vector-providers.md`); it is not the default.
- A drawing below the minimum is refused at conversion start with the typed open error, before any traversal, and
  nothing is written.
- Each entity mapping needs the provider facets it reads (`docs/pvec-providers.md`); none of them raises the
  minimum DWG version.
- The vector PDF path (`src/pdf.rs` and `pvec::pdf_vector`) is not a drawing provider in this sense: it reads the page's
  content stream with `lopdf` at the version `Cargo.lock` pins, and what it loses against a DWG is in `docs/pvec-dwg-pdf-differential.md`.
