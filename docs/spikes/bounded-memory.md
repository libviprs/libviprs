# Bounded-memory PVec generation and the spool fallback (N1.7)

The question: does the final spatial layout need global knowledge that the order
a source gets traversed in cannot supply? If it does, a writer cannot stream
straight to its output and has to do what the PMTiles writer already does, which
is spill, then sort, then publish.

I prototyped three strategies, ran each on synthetic streams of 1, 4 and 16
million entities, and measured peak resident memory, CPU time, wall time and
bytes on disk.

## What this measures, and what it does not

PVec's container and spatial layout are not defined yet (P1.5 and P1.6 in
`libviprs/pvec` are open, and that repository holds a README). So I did not
write to a PVec writer. The harness writes to a **stand-in**: an opaque payload
region plus a spatial index ordered by a Morton key of each entity's centre,
quantised over the **global extents** of the whole stream. That is the shape of
requirement I am worried about (a key that depends on a value only known after
the last entity), not a claim about PVec's actual layout. Every conclusion below
is about the stand-in, and the section "What is still open" says which PVec facts
could change it.

The "entities" are synthetic: 48 to 160 bytes of opaque payload, 70 percent of
centres clustered in 64 clusters and the rest uniform, from a seeded generator
so every strategy sees the same stream. No decoder was involved.

Files: `spool_spike.rs` (std only), `run.sh` (the exact run), `results-x64.txt`
(the raw output).

## The three strategies

| strategy | what it does |
|---|---|
| `direct` | One pass. Payload goes out in arrival order. An index record per entity is kept in memory (40 bytes: centre, offset, length, id, key). Keys cannot be computed until the extents are complete, so they are computed at the end, sorted in memory, and written. |
| `twopass` | Pass one walks the source only to find the extents. Pass two walks it again, computes each key immediately, writes the payload and keeps a 24-byte index entry in memory, then sorts and writes. |
| `spool` | One traversal. Payload goes out in arrival order and a 32-byte raw record goes to a spool file. Then the spool is read back in chunks of 1,048,576 records (a 24 MiB sort buffer, the same figure as the PMTiles writer), each chunk is keyed, sorted and written as a run, and the runs are merged with a fan-in capped at 128. More than 128 runs costs another merge pass and not more file handles. |

All three produce the same index file. I hashed it (FNV-1a 64) at every size and
the three hashes agree at 1M (`1b0e00ee314ae5d3`), 4M (`547e705491a5ee05`) and
16M (`c07fa3b5feae76f0`), which is the check that the spool is not winning by
doing less. A fourth run forced a two-level merge (4M entities, 16,384-record
runs, 245 runs, 2 merge passes) and produced the 4M hash again.

## Platform

Every figure here is **native x86_64**, from `uname -m` inside the container on
MARS (`devshell@192.168.0.100`) through `tools/nas.sh`, image `rust:1.99.1-bookworm`
(rustc 1.99.1), built with `rustc --edition 2021 -O`. I ran no arm64 numbers and
no emulated x86_64 numbers, so there is no arm64 column.

Peak memory is `VmHWM` read from `/proc/self/status` at exit, one process per
strategy per run. It counts resident pages of the process, not the page cache the
output files sit in.

## Results

Entity counts are exact. Payload bytes written are identical for all strategies
(103,981,904 at 1M, 415,972,320 at 4M, 1,664,025,008 at 16M) and so is the final
index (24 bytes per entity: 24 MB, 96 MB, 384 MB).

Peak RSS, in MB (1 MB = 1,048,576 bytes), min to max over the repetitions:

| entities | direct | twopass | spool |
|---|---|---|---|
| 1,000,000 (3 runs) | 41.3 to 41.5 | 26.2 | 28.0 to 28.4 |
| 4,000,000 (3 runs) | 155.7 to 156.1 | 94.9 to 95.1 | 29.3 to 29.7 |
| 16,000,000 (2 runs) | 612.0 to 612.2 | 368.2 to 369.1 | 29.8 to 30.1 |

CPU time (user plus system), in ms, per run:

| entities | direct | twopass | spool |
|---|---|---|---|
| 1,000,000 | 110, 160, 140 | 270, 170, 110 | 210, 150, 110 |
| 4,000,000 | 570, 460, 490 | 530, 460, 500 | 560, 470, 690 |
| 16,000,000 | 2,210, 2,300 | 2,220, 2,000 | 3,410, 3,390 |

Wall time, in ms, per run. **Do not read these as a ranking.** MARS is a shared
host and its load average was around 380 on 24 cores when I started. The same
binary on the same input varied by ten times between repetitions (direct at 4M:
7,490, 3,783, 490 ms). I decided nothing from wall time and I am recording it
because the acceptance criteria ask for it:

| entities | direct | twopass | spool |
|---|---|---|---|
| 1,000,000 | 796, 356, 150 | 839, 863, 114 | 1,240, 652, 1,779 |
| 4,000,000 | 7,490, 3,783, 490 | 918, 1,444, 775 | 889, 477, 698 |
| 16,000,000 | 27,086, 17,191 | 32,137, 2,954 | 45,652, 23,024 |

Strategy-specific numbers:

| | 1M | 4M | 16M |
|---|---|---|---|
| `twopass` pass-one wall time (ms) | 25 to 31 | 91, 88, 109 | 415, 395 |
| `twopass` pass-one peak RSS (MB) | 1.8 to 1.9 | 1.9 | 1.9 to 2.0 |
| `twopass` bytes re-read (regenerated) | 119,981,904 | 479,972,320 | 1,920,025,008 |
| `spool` bytes written to the spool | 32,000,000 | 128,000,000 | 512,000,000 |
| `spool` bytes written as runs (and merge output) | 24,000,000 | 96,000,000 | 384,000,000 |
| `spool` runs / merge passes | 1 / 0 | 4 / 1 | 16 / 1 |
| `spool` sort buffer (bytes) | 25,165,824 | 25,165,824 | 25,165,824 |

The two-level merge run (4M entities, 16,384-record runs) held a 393,216-byte
sort buffer, wrote 192,000,000 bytes of runs over its two merge passes, and
peaked at 13.5 MB.

## What the numbers say

**Memory.** `direct` and `twopass` grow linearly with the entity count. `direct`
costs about 39 bytes an entity (612 MB at 16M) and `twopass` about 24 (368 MB).
`spool` is flat: 28.0 to 28.4 MB at 1M and 29.8 to 30.1 MB at 16M, a sixteen-fold
increase in entities for about 1.7 MB. That ceiling is the 24 MiB sort buffer plus
fixed buffers (up to 128 merge readers of 64 KiB, plus write buffers), and it does
not depend on how many entities there are. I am not extrapolating beyond 16M;
the linear fits are two or three points each.

**Disk.** The spool's temporary disk is **not** independent of entity count, and I
do not want to claim it is. It is 32 bytes an entity for the spool plus 24 bytes
an entity for the runs (single merge level), so 56 bytes an entity, 896,000,000
bytes at 16M. What is bounded is the **memory ceiling**, and the disk cost is a
fixed price per entity. The acceptance wording asks for both to be bounded
independently of entity count, and only the memory ceiling can be. The prototype
deletes run files after each merge and leaves the spool file in place until the
directory is removed, so it does not demonstrate zero leftover disk. A production
writer would delete the spool after the last pass, which I did not exercise.

**CPU.** The spool costs about 1.5 times the CPU of `direct` at 16M (3.4 s against
2.2 s), which is the second read of the spool and the merge. `twopass` costs about
the same CPU as `direct` here, but its second pass is regenerating a synthetic
stream.

**The cost I could not measure.** `twopass` re-walks the source. Here that is a
generator and pass one took 0.4 s for 16M entities. For a real source it is a
second full decode of a drawing, and I have no figure for that. It also assumes a
second traversal yields the same entities in the same order, which the decoder
contract does not promise anywhere I have read.

## Which writer inputs are unavailable in traversal order

For the stand-in: the **global extents** (the Morton key's quantisation range).
They are not known when the first entity is written, which is why `direct` has to
hold an index record per entity until the end. Nothing else the stand-in writes
depends on later entities, because the payload goes out in arrival order.

For PVec itself I cannot give the list, because its layout and index are not
defined yet. The missing measurement is the P1.5 and P1.6 output: which values the
container header, the spatial index and the payload ownership model need before
the first entity can be written (extents, entity count, quantisation origin,
offsets of regions). Until that exists, "none are unavailable", which is the
finding that would permit direct streaming, is not established.

## Decision

Phase 6 implements the **bounded spool with an external sort**, in the shape of
`src/pmtiles/writer.rs`: a spool in arrival order, a fixed sort buffer, run files,
a capped merge fan-in and a staged publish. Phase 6 should not choose `direct` or
`twopass` without new measurements.

Reasoning, with the numbers beside it:

- `direct` is 612 MB at 16M entities and grows by about 39 bytes each, because the
  only index key that matters needs a value that is not known until the end.
- `twopass` is 368 MB at 16M and grows by about 24 bytes each. It also needs the
  source walked twice, and I have no cost for that on a real decoder.
- `spool` is 30 MB at 16M, the same as at 1M. It costs about 1.5 times the CPU of
  `direct` and 56 bytes an entity of temporary disk.

Where the choice was ambiguous (the spool is not clearly faster, and wall time here
cannot rank anything), I took the option that does not depend on a number I could
not measure: a spool does not re-run a decoder and does not assume traversal is
repeatable, and its memory does not depend on the entity count.

## What is still open

- PVec's real writer inputs (P1.5, P1.6), as above. If they are all known in
  traversal order, `direct` becomes viable and needs a fresh measurement.
- A layout that needs the **payload** in spatial order, not only the index, needs
  another pass that reads payloads back by offset. I measured arrival-order
  payload only. Random reads from the spool were not measured.
- The cost of a second decode of a real DWG, which is the missing figure for
  `twopass`.
- Fewer than three repetitions at 16M, and a loaded host. Memory figures were
  stable to within 0.3 percent between runs, but CPU time varied by up to about 40
  percent between runs at 4M (spool: 470 to 690 ms).
- No arm64 figures.
