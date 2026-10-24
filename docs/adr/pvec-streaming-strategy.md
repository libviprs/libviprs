# ADR: PVec generation uses a bounded spool with an external sort

Status: Frozen by N1.8 (#1047) for phase 6 on the stand-in measurements in the spike below. The
strategy is settled, and one open item (PVec's real writer inputs, P1.5 and P1.6) can reopen it,
see Open items. Decides N1.7 (#1046), part of the integration epic #1039.
Evidence: [docs/spikes/bounded-memory.md](../spikes/bounded-memory.md).

## Context

PVec generation has to take entities in whatever order a source yields them and
produce a spatially ordered result. libviprs already does the equivalent for tiles
in `src/pmtiles/writer.rs`: tiles arrive in any order, the index is spilled to an
append-only log, sorted externally at finalize behind a 24 MB buffer, and
published through a staged file, an `fsync` and an atomic rename. The question was
whether PVec can stream directly instead, which depends on whether the layout
needs a value that traversal order cannot supply.

I measured three strategies on a stand-in layout (a Morton-keyed index over the
global extents) at 1, 4 and 16 million synthetic entities, native x86_64 on MARS.
The numeric policy for any geometry comparison is P1.2's. This ADR defines no tolerance of its own,
and libviprs defines no epsilon of its own either.

## Decision

Phase 6 implements the bounded spool with an external sort: payload in arrival
order, a raw record per entity in a spool, a fixed-size sort buffer, sorted runs,
a merge with a capped fan-in, and a staged publish. Phase 6 does not choose direct
streaming or two-pass without new measurements.

## Numbers

Native x86_64, `rust:1.99.1-bookworm` on MARS. Peak RSS in MB, CPU in ms:

| entities | direct RSS | twopass RSS | spool RSS | direct CPU | twopass CPU | spool CPU |
|---|---|---|---|---|---|---|
| 1,000,000 | 41.3 to 41.5 | 26.2 | 28.0 to 28.4 | 110 to 160 | 110 to 270 | 110 to 210 |
| 4,000,000 | 155.7 to 156.1 | 94.9 to 95.1 | 29.3 to 29.7 | 460 to 570 | 460 to 530 | 470 to 690 |
| 16,000,000 | 612.0 to 612.2 | 368.2 to 369.1 | 29.8 to 30.1 | 2,210 to 2,300 | 2,000 to 2,220 | 3,390 to 3,410 |

Two-pass re-reads 1,920,025,008 bytes at 16M (pass one 395 to 415 ms, 1.9 MB peak).
The spool writes 512,000,000 bytes of spool and 384,000,000 bytes of runs at 16M,
with a 25,165,824-byte sort buffer. Wall times are in the spike and are not used
here, because the host was heavily loaded.

## Consequences

- Memory is bounded independent of entity count: about 30 MB at 16M entities, the
  same as at 1M. Temporary disk is not independent of entity count: 56 bytes per
  entity for the spool and one level of runs.
- The strategy costs roughly 1.5 times the CPU of direct streaming at 16M.
- It does not require a source to be traversable twice or deterministically.
- Phase 6 needs a rule for deleting the spool after the last pass; the prototype
  does not exercise that.

## Open items

- The inputs PVec's real container and index need before the first entity is
  written are not known until P1.5 and P1.6 land. If none depends on a later
  entity, direct streaming becomes viable and needs a new measurement.
- A layout that needs payloads in spatial order needs a random-read pass over the
  spool, which was not measured.
- The cost of a second decode of a real DWG, which is the missing figure for
  two-pass.
- Not measured: the cost of a second decode of a real DWG (the missing figure for `twopass`), more
  than a few repetitions at 16M entities, an idle host, and any arm64 figure. CPU time varied by up
  to about 40 percent between runs at 4M, so the CPU ranking is weak. Peak RSS was stable and is the
  figure the decision rests on.
