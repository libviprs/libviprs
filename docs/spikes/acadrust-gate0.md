# Gate 0: acadrust over the AutoCAD-authored corpus (N4.1)

Gate 0 of #1072, the condition from the decision on #1239: before the first provider ships on
acadrust, I run it over the AutoCAD-authored saves and the upstream MIT samples named in
`ACAD-ENGINE-DECISION-PLAN.md`. An open failure on any AutoCAD-authored file flips the first
provider back to the existing decoder with the traits unchanged, and I stop and report instead of
shipping. This is that run.

Gate 0: PASS. All 22 AutoCAD-authored files open through `AcadrustProvider`, no `OPEN_FAILED` in Tier A.

`OPEN_FAILED` means `AcadrustProvider::open` returned an error, or the process running it died,
for that file. A reader panic counts, because the provider catches it and reports it as an open
error. The table below is `acadrust-gate0/results-x64.tsv`, and `tests/spike_acadrust_gate0.rs`
fails if the two disagree, if a Tier A row is `OPEN_FAILED`, or if a corpus file has no row.

## What I ran

- acadrust: libviprs/acadrust at `72af48970e306ef1d50dd751b227df2b4fc3fb99` (fork `main`), the same
  full rev `Cargo.toml` pins for the `dwg-acadrust` feature. A test checks the two stay equal.
- Provider: `AcadrustProvider` at libviprs `e19b68bc` (branch `feat/1072-provider-traits`), release build,
  through `examples/dwg_gate0.rs`, one process per file so a crash is one row.
- Host: MARS, native x86_64, `rust:1.99.1-bookworm`, in a container, by `acadrust-gate0/run.sh`.
- Corpus: ACadSharp `samples/` at `d7dc111023477d8a9fffc2153139459c95b4f345` (MIT). Tier A is the
  AutoCAD-authored set and decides. Tier B is the 50 `g11_*`/`g13_*` regression files
  already in `tests/fixtures/cad/g13` (ACadSharp `DwgWriter`), and Tier S the two spike fixtures
  of N4.1; B and S are reported and decide nothing. Each row carries the file's sha256 in the TSV.
- What a row says: `open`; how many layers, layouts and block records the provider exposed;
  how many entities a traversal of the model layout and of every scope yielded; how many of
  those came back `Unmapped` (counted with the class name, not dropped); and whether a second
  traversal and a traversal after a fresh open produced the same sequence.

## Tier A

| file | version | open | layers | layouts | blocks | model entities | entities (all scopes) | unmapped |
|---|---|---|---|---|---|---|---|---|
| AecObjects.dwg | AC1032 | OPEN_OK | 7 | 2 | 6 | 4 | 15 | 9 |
| BLOCKALIGNMENTPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 4 | 1 | 2 | 0 |
| BLOCKBASEPOINTPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 4 | 1 | 2 | 0 |
| BLOCKFLIPPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 6 | 2 | 8 | 0 |
| BLOCKLINEARPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 5 | 2 | 4 | 0 |
| BLOCKLOOKUPPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 24 | 4 | 59 | 0 |
| BLOCKPOINTPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 6 | 3 | 12 | 0 |
| BLOCKPOLARPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 9 | 2 | 22 | 0 |
| BLOCKROTATIONPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 7 | 4 | 20 | 0 |
| BLOCKVISIBILITYPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 16 | 4 | 65 | 0 |
| BLOCKXYPARAMETER.dwg | AC1032 | OPEN_OK | 1 | 3 | 7 | 2 | 25 | 0 |
| empty.dwg | AC1032 | OPEN_OK | 1 | 3 | 3 | 0 | 0 | 0 |
| export_sample.dwg | AC1032 | OPEN_OK | 4 | 9 | 32 | 22 | 455 | 173 |
| geoloc.dwg | AC1032 | OPEN_OK | 1 | 3 | 3 | 0 | 0 | 0 |
| sample_AC1014.dwg | AC1014 | OPEN_OK | 21 | 4 | 27 | 163 | 341 | 187 |
| sample_AC1015.dwg | AC1015 | OPEN_OK | 21 | 4 | 27 | 163 | 341 | 187 |
| sample_AC1018.dwg | AC1018 | OPEN_OK | 21 | 4 | 27 | 163 | 341 | 187 |
| sample_AC1021.dwg | AC1021 | OPEN_OK | 21 | 4 | 27 | 163 | 341 | 187 |
| sample_AC1024.dwg | AC1024 | OPEN_OK | 19 | 4 | 27 | 163 | 341 | 187 |
| sample_AC1027.dwg | AC1027 | OPEN_OK | 19 | 4 | 27 | 163 | 341 | 188 |
| sample_AC1032.dwg | AC1032 | OPEN_OK | 19 | 4 | 27 | 163 | 341 | 188 |
| sample_base.dwg | AC1032 | OPEN_OK | 19 | 4 | 27 | 163 | 341 | 188 |

The plan says Tier A is 23 files with eleven dynamic-block drawings. At that commit upstream has
ten `BLOCK*.dwg` (22 files in all), so I ran what is there and did not go looking for a missing one...

Cross-version invariance, the plan's best oracle: the seven saves are one drawing saved by
AutoCAD at seven versions, and the provider gives 163 model-space entities and 341 in all scopes
for every one of them. The unmapped count is 187 up to AC1024 and 188 for AC1027 and AC1032, and
the one extra is a SPLINE, see Findings.

## AutoCAD's own statement of the counts

AutoCAD wrote an ASCII DXF beside each save in the same SAVEAS session. `dxf_census.py` counts
from that text, with no DWG reader involved, and the provider's counts are held against it by
`acadrust_exposes_the_counts_autocads_own_dxf_states`. All six saves that have a DXF twin agree
exactly (AC1014 has no twin, AC1009 has no DWG):

| save | layers (DXF / provider) | block records (DXF / provider) | model-space entities (DXF / provider) |
|---|---|---|---|
| sample_AC1015.dwg | 21 / 21 | 27 / 27 | 163 / 163 |
| sample_AC1018.dwg | 21 / 21 | 27 / 27 | 163 / 163 |
| sample_AC1021.dwg | 21 / 21 | 27 / 27 | 163 / 163 |
| sample_AC1024.dwg | 19 / 19 | 27 / 27 | 163 / 163 |
| sample_AC1027.dwg | 19 / 19 | 27 / 27 | 163 / 163 |
| sample_AC1032.dwg | 19 / 19 | 27 / 27 | 163 / 163 |

That is a count and nothing more: it says the provider did not skip layers, blocks or
model-space entities, and says nothing about whether the geometry of any of them is right.

## Tier B and S, for the record

Tier B: 49 of 50 open. `g13_ac1009.dwg` is `OPEN_FAILED` (acadrust: not a drawing this provider reads: version stamp "AC1009"). That one is
a version-stamp refusal by `probe`, AC1009 being R12, which the existing decoder's supported list
(AC1014 to AC1032) also leaves out, so it is not a regression against what the tile path takes.
Tier S: 2 of 2 open. Rows are in the TSV.

## Findings

- **One spline reads differently in R2013 and later.** In `sample_AC1027.dwg` and `sample_AC1032.dwg`
  the second SPLINE (handle `434`) comes back from acadrust as a fit-point spline: 2 fit points,
  no control points, no knots. The same drawing saved as AC1024 reads it with 4 control points and
  8 knots, and AutoCAD's own DXF for all three versions says 4 control points and no fit points.
  I did not open the DWG bytes to confirm the file has no control polygon, so I cannot tell a reader
  limitation from how R2013 stores a fit-point spline. The provider reports it `Unmapped` with the
  reason, which is the behaviour I wanted either way: a curve that cannot be reported is not rebuilt
  from points. N4.3 decides what to do with it.
- **Four entities of `AecObjects.dwg` have no entity behind their handle.** A block lists four handles
  the document has no entity for, so they come back `Unmapped` with class `UNKNOWN`. Likely
  proxy objects, but I did not chase it...
- **`geoloc.dwg` yields no entities at all** (0 in model space, 0 in every scope). I have no second
  reading of that file, so I cannot say whether it is empty or the reader loses its contents. The
  plan names it for georeferencing, which is not entity data.
- **Many classes are not mapped yet.** Of 341 entities in the AC1032 save, 188 are `Unmapped`
  (POINT, MTEXT, TEXT, MULTILEADER, SOLID, HATCH, dimensions, and others, listed per file in the TSV).
  That is the extraction work of N4.2 to N4.9 and not a Gate 0 criterion.

## What this does not show

- Geometry. Gate 0 counts and opens. It does not compare a coordinate with AutoCAD's, and the
  ezdxf adjudication in the plan is not run. `vc_layers` and `vc_nested` are ACadSharp-written and
  acadrust-read, so they are cross-family checks and not known truth.
- Anything on arm64. This ran on native x86_64 only. acadrust is pure Rust, so I expect the same,
  and I have not measured it.
- Speed and memory. No timing is recorded. `acadrust` holds a whole `CadDocument` in memory, and
  whether that matters for large drawings is the performance half of the bake-off (#1241).
- Hostile input. A panic is caught and reported, but I fuzzed nothing.
- The decision itself. This clears the flip condition on #1239. It is not the full acadrust against
  ACadSharp comparison, which stays with #1241.

## Reproducing it

`acadrust-gate0/run.sh` rebuilds everything in containers on the docker host and rewrites the TSV;
`python3 -I acadrust-gate0/dxf_census.py <ACadSharp samples dir>` rewrites the census. The run needs
the upstream samples checked out at the pinned commit, and it refuses any other commit.
