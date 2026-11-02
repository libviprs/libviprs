# PVec drift check

A `.pvec` can be checked against the fixture it was generated from. This is the piece that was
missing between the end-to-end path (`tests/pvec_end_to_end.rs`, #1050), which converts and compares
inside one test, and the determinism baseline (`tests/fixtures/pvec-gate/`, #1060), which says that
bytes changed and never what moved. The issue is #1475.

## What it does

`libviprs::pvec::drift` (feature `pvec-write`) reads the artifact back through PVec's public reader
and compares, per entity id:

- a `KNOWN_TRUTH_EXACT` fixture with its `expected.math.json`, the independently authored truth. A
  clean report says the file is the drawing, not just that it equals last time's file;
- every other evidence class with the previously accepted output (`tests/fixtures/pvec-drift/accepted.dump`),
  because it has no truth: `DUAL_ORACLE_EQUIVALENT` is two converters agreeing and
  `REAL_WORLD_DIFFERENTIAL` is regression only. A `QUARANTINED` fixture is compared the same way and
  its report says it does not gate;
- a known-truth fixture is also compared with its accepted output when one is given, so a report can
  say both "not the drawing" and "not what we accepted".

Comparisons use `Policy::V1`. libviprs has no epsilon of its own, and none of PVec's maths is
reimplemented: curves are read, evaluated and bounded through the public `Curve`/`Curve2d` API.

| drift | what it means |
|---|---|
| `moved-vertex` | a vertex, endpoint, centre, control point, knot or weight is somewhere else |
| `changed-radius` | a radius or semi-axis changed |
| `dropped-entity` | the fixture has an entity the artifact lacks |
| `extra-entity` | the artifact has one the fixture lacks |
| `reordered-ids` | the right geometry sits under another entity id |
| `changed-bounds` | the manifest bounds are not the geometry's (exact for curves PVec bounds in closed form, between the endpoints' box and the control hull for splines) |
| `changed-unit` | the manifest unit is not the fixture's |
| `swapped-class` | the entity is another kind of curve than the fixture says (a circle stored as an ellipse, say) |

Entity `i` of the fixture's live entities (layer on and thawed, no `-equivalent-` duplicate) belongs
under id `i`, the converter assigns ids in drawing order. A curve found under another id is a
reorder, not a drop plus an extra.

## Running it

```text
cargo run --features pvec-write --example pvec_drift -- \
    tests/fixtures/pvec-gold/fixtures/circle-001 some.pvec [--json] [--accepted FILE] [--dump]
```

The human report is the default and `--json` prints the machine one (fixed key order, one object).
`--dump` prints the canonical dump instead. Exit status is 0 for no drift (or drift on a fixture that
does not gate), 1 for drift, 2 when the check could not be made. A `viprs` subcommand would live in
libviprs-cli, which is a separate repository with a path dependency on this one, so it is a follow-up
there and not part of this change.

## The committed digests

`tests/fixtures/pvec-drift/accepted.dump` holds the canonical dump of every converted, non-quarantined
fixture, and `tests/fixtures/pvec-gate/drift.baseline` the sha256 of each dump, one line per fixture. A
dump prints every number as its shortest decimal next to its IEEE bits, so the digest moves when one
bit of the stored geometry does. The ordinary suite (`tests/pvec_drift.rs`) converts the corpus and
fails on any difference, and checks that each accepted section still hashes to its committed digest.

This is not the determinism baseline of #1060. That one hashes the artifact bytes, so a change to a
header field moves it and it cannot say what moved. This one hashes the geometry the reader returns,
and when it moves, the report names the entity.

The files are the x86_64 ones. Two fixtures come out in other bits on arm64 (`hypot`, see
[the ADR](adr/pvec-artifact-reproducibility.md)), and their rows live in `drift.aarch64.baseline`
and `accepted.aarch64.dump` next to the base files. The drift check calls both outputs the same
drawing, because they agree far inside `Policy::V1`.

Regenerate after a deliberate change, then read the diff before committing it. Run it on x86_64 first,
which writes the base files, and then on arm64, which writes only the rows that differ:

```text
PVEC_GATE_BLESS=1 cargo test --features pvec-convert --test pvec_drift -- n_1471_the_digest_of_every
```

The suite runs in the mathematical-fidelity gate (`pvec-gate-fidelity` in `ci.yml`), next to the known-truth
suites it shares its comparison with.

## What it catches and what it cannot

It catches any change to the stored geometry, the unit or the bounds of a fixture that converts, and
for known-truth fixtures it says which entity and in what way. It cannot say anything about a fixture
the converter refuses (no artifact to read), about geometry the fixture's truth does not describe
(styles, layers, the report sidecar), or about a bound change that stays inside a spline's control
hull and around its endpoints when no accepted output is given. A periodic spline is stored rewritten,
so it is compared on degree and closure and never claimed exact.
