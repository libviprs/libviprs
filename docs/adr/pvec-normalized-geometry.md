# ADR: Normalized2DGeometry is a thin adapter over PVec's public semantic concepts

Status: Frozen by N1.8 (#1047), except NG-12 (see the freeze record). Decides N1.2 (#1041), part of the integration epic #1039.
Executable proof: `tests/adr_normalized_geometry.rs`, one test per acceptance criterion, which
reads this file as text and fails when a clause below goes missing or says something it must not.

## Context

libviprs has no geometry model. `src/geo.rs` is a pixel-to-geo affine and nothing else, and the
tile path (`cad::Primitive` into `vector::tiler`) tessellates because a tile needs a deviation
budget that depends on zoom. The integration epic adds a second, analytic derivative beside that
path: curves reach PVec as curves. The risk is not that I pick the wrong model. It is that a second
curve library grows in this crate by accident, one helper at a time ("just a bounding box for the
arc", "just an epsilon for the join"), until libviprs owns mathematics that PVec is supposed to own
once for four repositories. Panel finding 1 of the epic review says that must not happen. This ADR
is the specification that forbids it, and it lands before any code.

What I read, and pinned: `docs/ISO_PROFILE.md` and `docs/NUMERIC_POLICY.md` in `libviprs/pvec` at
revision `fdf87de` (main after P3.7, #83; the profile and policy came in with P1.1 and P1.2, ADR 0001 and 0002 there, and P3.8 added the promotion table to the profile). Every name and rule number
below is theirs. When that revision moves, the citations in NG-03 and NG-09 are re-checked in the
same change that moves the pin, and not before.

## Decision

### NG-01 The statement

`Normalized2DGeometry` is a thin adapter over PVec's public semantic concepts and is not a second geometry library.

It holds values that PVec's own constructors have accepted, hands them on, and records what was
done to get them. It does not compute with them. Anything in this crate that would need to compute
with a curve calls PVec, and where PVec has no entry point for it the mapping is
`RequiresPVecCapability` (NG-11). The constructs this rules out are listed in NG-10, each with its
reason, and NG-V1 names the clause a reviewer points at.

### NG-02 The record

What a source entity becomes is exactly this, and nothing is added to it by later issues except
through the component tables of NG-12:

```rust
/// Sketch, not the N3.1 signature. Field types marked (P) are PVec's, referenced not redefined.
pub struct Normalized2DGeometry {
    /// Stable identity and source order, owned by N3.7.
    entity: EntityKey,
    /// A PVec public curve value (P). `None` exactly when `fidelity` is one of the three
    /// classes that produce no geometry (NG-07), and the constructor enforces that.
    geometry: Option<PVecCurve>,
    /// The box PVec reports for `geometry`, conservative and not always tight, input for the
    /// spatial index only (NG-05).
    source_bounds: Option<SourceBounds>,
    /// Optional, behind its own handle, resolved through the store of N3.2 (NG-06).
    provenance: Option<ProvenanceHandle>,
    /// The classification of N3.5, one of six (NG-07).
    fidelity: FidelityClass,
}
```

The record is complete and usable with provenance absent: nothing in `geometry`, `source_bounds`
or `fidelity` is derived from, or points into, the provenance store, and every consumer in N6 and N7
has to work on a record whose `provenance` is `None`. `PVecCurve` stands for PVec's `Curve` enum from `pvec-geometry` (P3.1, landed), which wraps the
kinds of NG-03. The sketch keeps the placeholder because N3.1 decides how the record names it, and
the table below still uses the profile's construct names, not Rust type names.

### NG-03 The semantic type mapping

The type column is the ISO 10303-42 construct name that `ISO_PROFILE.md` gives the PVec capability
(libviprs/pvec@fdf87de), and the section column is the profile section that defines it. A source
shape that has no row maps to nothing and is classified under NG-07, never tessellated.

| source shape | PVec semantic type | profile sections | how a source value becomes one |
|---|---|---|---|
| line | `trimmed_curve` over `line` | section 4 and section 8 | the profile has no separate segment type, a segment is a trimmed line |
| polyline | `polyline` | section 5 | vertices only; a closed polyline repeats its first vertex, as the profile says |
| circular arc | `trimmed_curve` over `circle` | section 6 and section 8 | centre, radius and the two trim parameters with a sense, passed as given |
| circle | `circle` | section 6 | centre and radius, no trim |
| ellipse | `ellipse` | section 7 | placement and the two semi-axes; an elliptical arc is a `trimmed_curve` over it (section 8) |
| B-spline | `b_spline_curve_with_knots` | section 11 | degree, control points, explicit knots with multiplicities, passed through whole |
| NURBS | `rational_b_spline_curve` with `b_spline_curve_with_knots` | section 12 | the same plus one weight per control point; weights present means rational, never demoted |

Consequences the profile already fixes, restated so nobody re-decides them here:

- A distinct source entity class is not a reason for a distinct type. An ARC, a CIRCLE and a bulge
  segment all land on the rows above (profile, "The profile rule").
- A polyline segment that carries a bulge is not a `polyline`. It is a composite of trimmed lines and
  trimmed circles (profile section 9), and the profile lists bulge inside a polyline as unsupported
  for the `polyline` entry.
- A Bezier source has no row in this table because no v1 source produces one; the PDF path of N5 is
  where it appears, and N1.6 adds its row.
- Style, text and hatch are outside this record (profile, "The profile rule").

### NG-04 Coordinates

Coordinates are source-space `f64`, unconverted: no projection, no unit rescale, no recentring, no
rounding, no snapping, and no libviprs-side arithmetic on a coordinate before PVec sees it. The
values PVec's constructors refuse (not finite, subnormal, out of range) are refused there, and the
adapter reports the refusal (NG-07). The constructor's own normalisations, such as the storage of
negative zero, are PVec's and are not repeated here. Georeferencing stays separate from source
geometry (epic, architectural rule); the record carries no CRS and no affine to one.

### NG-05 Source bounds

`source_bounds` is PVec's box for the entity, and it exists for spatial indexing input only (N6.4).
It is carried, not computed: the adapter takes it from `Curve::bounds` for the stored value, which
exists for every kind. It is conservative, not tight. PVec computes a closed-form box for lines,
arcs, ellipses, polylines and Bezier curves up to degree 3, and for splines and higher-degree Bezier
curves it returns the control hull, which encloses the curve but is not tight (its own doc comment
says so). So a consumer treats `source_bounds` as an enclosing box that may be larger than the
curve, which is all an index needs, and never as the curve's extent. Making it tight for a spline
needs the mathematics of that curve, so computing it here would be NG-B1, and no flag is added to
say which kind is tight: that would be the adapter restating PVec's rule. Nothing may expand, round,
union or compare a bound for any purpose except handing it to the index builder.

### NG-06 Provenance

Provenance is optional component data reachable from the record through a handle, never inside the
mathematical entity. `provenance` is an `Option` of an opaque `ProvenanceHandle`. The payload (source
handle, layer, block lineage, provider name) is N3.2's, and it lives in its own store. It follows
from that:

- a record with provenance absent is a complete record (NG-02), and so is an output stream with the
  provenance component skipped;
- no equality, ordering or hashing of a record reads the handle;
- dropping the store changes no geometry byte.

### NG-07 Fidelity class

The `fidelity` field is one of the six classifications that N3.5 implements: `SupportedExact`,
`SupportedEquivalent`, `RequiresPVecCapability`, `RequiresExtension`, `Unsupported`,
`IgnoredByPolicy`. They are named here only so the field's contract is readable; the enum is
not redefined, extended or mirrored by this ADR or by anything in `Normalized2DGeometry`, because
N3.5 requires a single source for it.

`geometry` is `Some` for `SupportedExact`, `SupportedEquivalent` and `RequiresExtension`, and `None`
for `RequiresPVecCapability`, `Unsupported` and `IgnoredByPolicy`. Such a record still exists, so
the six counts sum to the entities encountered, and it carries the reason code N3.5 asks for through
the fidelity value. A display-fidelity result is never `SupportedExact`. Exactness is claimed only
through the conformance classes of EPIC P.

### NG-08 The boundary rule

No provider type appears in a field of `Normalized2DGeometry`, and none appears in any public
signature that returns one. Since #1029 that covers the decoder primitive types too: `cad::Primitive`
and the `Line`, `Arc`, `Spline` structures under it are the decoder's IR, and the adapter consumes
them behind the line and may not name them in a field or a public signature. Conversion goes from the
decoder IR to PVec's values inside one function that returns `Normalized2DGeometry`, and what crosses
out is only the record of NG-02.

### NG-09 Every numeric question is a citation

This ADR defines no tolerance, epsilon or comparison threshold, and contains no number that could
be mistaken for one. Each numeric question the adapter can meet has an answer in the numeric policy,
and the answer is to ask PVec, whose `pvec-numeric` crate is the implementation. Citation form is
the policy's own.

| question | answered by |
|---|---|
| are two values equal, are two points equal | PVec numeric policy v1, NP-05 |
| is a value finite and normal, is it in range | PVec numeric policy v1, NP-01 and PVec numeric policy v1, NP-02 |
| is a length, radius or axis degenerate | PVec numeric policy v1, NP-06 and PVec numeric policy v1, NP-07 |
| is a direction zero or unit | PVec numeric policy v1, NP-10 |
| is a trim a full period, collapsed or ambiguous, for a periodic basis | PVec numeric policy v1, NP-11 and PVec numeric policy v1, NP-12 |
| are the knots, weights and counts acceptable | PVec numeric policy v1, NP-14, PVec numeric policy v1, NP-15 and PVec numeric policy v1, NP-18 |
| is a composite joint closed, is a declared closed curve closed | PVec numeric policy v1, NP-17 and PVec numeric policy v1, NP-20 |
| is an affine map singular, or out of magnitude | PVec numeric policy v1, NP-19 |

The adapter does not pre-validate in order to be helpful. It hands the value to PVec and reports
the refusal PVec returns, because a pre-check here would be a second copy of a rule and the policy
says a rule not in its file does not exist. Counts and resource limits at ingestion are N3.6's and
cite the same table.

### NG-10 Prohibited constructs

Each prohibition has a reason and a replacement. A change that adds one of these to this crate
violates this ADR (NG-V1).

| id | construct | reason | instead |
|---|---|---|---|
| NG-B1 | curve evaluator (point, tangent or derivative at a parameter, extent, length) | a second evaluator is a second definition of every curve, and the two will disagree in the last bits that the conformance classes measure | call PVec's evaluator, or `RequiresPVecCapability` |
| NG-B2 | tessellator (curve to polyline or to tile geometry, inside normalization) | silent tessellation turns an analytic curve into a different locus and loses the fact that it happened; display buffers are PVec's concern | refuse as `Unsupported`, or `RequiresPVecCapability`; the tile path of EPIC L tessellates outside normalization and is untouched |
| NG-B3 | knot insertion, knot removal, degree elevation, any re-parametrisation of a spline | each changes the structural parameters the exactness classes compare, and needs PVec's knot rules | pass the source knots and control points through whole |
| NG-B4 | arc-to-NURBS conversion (or any promotion of one curve kind to another) | the promotion table lives in PVec's `ISO_PROFILE.md` (P3.8, ADR 0012); an ad hoc one here would diverge | map to the row in NG-03, call PVec's `Curve::to_rational_b_spline` when a spline is wanted, or `RequiresExtension` |
| NG-B5 | epsilon, tolerance or comparison threshold of any kind, as a constant, a parameter or a default | the numeric policy is the only definition, and a local one is a bug by its own rule | cite and call, per NG-09 |
| NG-B6 | coordinate equality operator (on points, on records, as `PartialEq` or as a helper) | equality needs the scale term of NP-05, and it is never used to merge or sort; a derived `PartialEq` on `f64` fields would be exact bit comparison passed off as geometric equality | compare through PVec, or compare identities (`EntityKey`) |

The same list bans the helpers that smuggle these in: a bounding box that is not PVec's, a "nearly
closed" test, a "is this arc a full circle" test, a bulge-to-centre computation done here.

### NG-11 When PVec has no entry point

Where a mapping needs mathematics, the adapter calls PVec. Where PVec has no entry point for it, the
mapping is `RequiresPVecCapability`, the entity is reported with a stable reason code and the source
class, and I file the gap against `libviprs/pvec` rather than working around it here. The candidates
I can see already, from the profile and PVec's constructors at the pin (each is a question for N1.3 to
confirm, not a finding):

- a line segment from two points: the profile wants a unit direction and a trim by length, and
  producing either is arithmetic on coordinates; `Line::new` takes a direction
  already, so a segment constructor from two end points is not there;
- a polyline bulge: centre and radius from a chord and a bulge need trigonometry;
- an ellipse from a major-axis vector and a ratio: the first semi-axis is the length of a vector and
  the second is a product, and the profile wants an orthonormal frame;
- the angle convention of an arc's trims: the profile says radians, so if a provider reports another
  unit the conversion has a home that is not here.

An entity in this state is counted and reported; it is never repaired and never approximated.

### NG-12 Where the semantic class and the regions go

Later issues add architectural element classes (`Wall`, `Column`) on entities and regions (`Room`,
`Corridor`) as an optional PVec component stream (libviprs/pvec#84, P3.9, populated by N4.9 and
N6.9). This ADR leaves them a place and keeps them out of the geometry model.

They live in component tables beside the record, keyed by `EntityKey`, in the same way provenance
does, and never as a field of `Normalized2DGeometry`:

```rust
/// Sketch. Types marked (P) are defined by PVec P3.9 and re-exported, not redefined.
pub struct ComponentTables {
    provenance: Table<EntityKey, ProvenanceHandle>,   // N3.2
    element_class: Table<EntityKey, ElementClass>,    // (P) Wall, Column, Unclassified
    regions: Vec<RegionRecord>,                       // (P) Room, Corridor
}
// RegionRecord: kind, ordered closed outer boundary and optional holes, each ring a list of
// EntityKey references; a source tag (layer-rule, block-rule, inferred, explicit) and an
// optional confidence. It embeds no curve parameters.
```

The rules that make this a side table and not a quiet widening of the geometry model:

1. A region record references geometry by `EntityKey` and carries none; a boundary segment created
   by gap closing (N6.9) is an ordinary geometry record that was created in libviprs, and the fact
   that it is derived lives in its provenance component (N3.2), never as a geometry field.
2. Removing the whole class and region side table leaves the geometry records and every output byte
   of the geometry stream byte-identical, and N4.9 and N6.9 each test that.
3. `ElementClass` and the classification of N3.5 answer different questions (what is it in the
   building, against can it be converted) and must not share an enum. A rule match changes the class
   table only.
4. The rules that decide Wall or Room are caller policy (`ElementRules`, N4.9), not adapter logic,
   and the adapter has no opinion on them.
5. Slots are optional by construction: a record and a conversion are valid with no class table at all.

I have not designed the table's storage or the order in which derived entities take their keys;
that is N3.7's problem and I list it in the open items below.

### NG-V1 The clause that says libviprs tessellates a curve itself is a violation

If this crate turns a curve into a polyline, a point sequence or tile geometry anywhere that feeds
`Normalized2DGeometry` or a PVec writer input, that is a specification violation of this ADR. The
clause it breaks is NG-B2, which is why NG-B2 exists as its own row, and the permitted outcomes for
an unsupported curve are `RequiresPVecCapability` or `Unsupported` under NG-07, never a canonical
PVec value built from flattened points. The test of N3.5 that tessellates a curve and expects a
canonical PVec value to fail is this clause in executable form. Tessellation in the tile path
(`vector::tiler`) is outside normalization and is not what this clause governs; the two paths share
no code through the record.

## Options considered

| option | what it means | why not |
|---|---|---|
| A. A libviprs geometry crate, PVec-compatible | own the types and keep them in step with PVec | exactly the second library the panel ruled out, and the numeric policy says nothing defines an epsilon outside its file |
| B. Re-export PVec's types and add helper methods | thin, but with convenience maths on top | the helpers are NG-B1 to NG-B6 under another name; the thin part has to be thin in behaviour too |
| C. A record of PVec values plus identity, bounds, optional provenance and fidelity (chosen) | libviprs carries and classifies, PVec computes | needs PVec entry points that may not exist yet (NG-11), which is the honest cost and gets filed |

## Consequences

- N3.1 implements NG-02 and nothing larger. A reviewer can reject a PR that adds an operation to the
  record type by pointing at NG-01 and the row of NG-10 it matches.
- Conversions can be refused where PVec has no entry point, so some source entities will report
  `RequiresPVecCapability` on day one. That is a visible, counted outcome and the point of the report.
- A cheap conformance check exists today: `tests/adr_normalized_geometry.rs` fails if a
  tolerance-shaped number enters this document, if a mapping row loses its profile name or section,
  or if the violation clause goes missing.

## Status and dependencies

This is a specification, so it can be accepted before its dependencies finish, but several of its
clauses cite work that has not landed, and I do not want that to read as settled:

- N3.1 (#1095) implements NG-02. Until it exists, `PVecCurve`, `EntityKey` and `SourceBounds` are
  placeholders in a sketch. PVec's in-memory API (P3.1) has landed at the pin, but this ADR still
  cites construct names, so the mapping does not depend on a Rust name.
- N3.5 (#1100) defines the six-way classification. NG-07 relies on its shape and on its rule that
  one enum exists.
- N1.3 (#1042) confirms or removes each candidate gap in NG-11 against real DWG entities. N1.5
  (#1044) consumes the promotion table in PVec's `ISO_PROFILE.md` (P3.8) and does not own one; a
  reader of its issue should know P3.8 settled that an ellipse under shear stays an ellipse, and
  that `Curve::to_rational_b_spline` is the way to a spline. N1.6 (#1045) adds the Bezier row.
- N3.2 (#1096) defines the provenance payload and N3.7 (#1102) the entity key and ordering.
- The semantic-class stream (P3.9, libviprs/pvec#84) is open on the PVec side, and N4.9 (#1233) and
  N6.9 (#1234) are open on this side. NG-12 reserves the place against their issue text, not against
  a finished stream version, so the table and the stream's version rules may need a revision when
  P3.9 lands.
- The pinned profile is unreviewed against the licensed text of ISO 10303-42 (its own ADR 0001 says
  so). Nothing here claims more than the profile does.

Open items I am handing on: where derived entities take their `EntityKey` so source order is not
perturbed (N3.7); and whether a closed polyline's repeated first vertex survives the provider boundary
as a vertex or as a flag (N1.3).

## Freeze record

N1.8 (#1047) froze this ADR in place. Every other N1.8 ADR cites it by clause and none restates it:
[pvec-semantic-boundary.md](pvec-semantic-boundary.md) is the frozen decision with N1.3's measured
numbers, and [pvec-provenance.md](pvec-provenance.md) is NG-06 with what N1.4 measured at the decoder
boundary. Evidence: [docs/spikes/dwg-mapping.md](../spikes/dwg-mapping.md).

- NG-11's candidate list was answered by N1.3: the bulge and the ellipse constructor were real gaps
  (libviprs/pvec#91 and #92), the line segment, the bounds and the angle convention were not. The two
  open items below under Status are settled by N1.3 (a closed polyline arrives as a flag with its
  first vertex copied, and PVec's bounds exist for every kind).
- NG-12 is not frozen: P3.9 (libviprs/pvec#84) is open, so the side-table rules stay proposed.
- The PVec pin in NG-03 and NG-09 is still `fdf87de`. N1.3 and N1.5 measured later revisions, and the
  citations are re-checked in the change that moves the pin.
- Numeric questions are P1.2's, and libviprs defines no epsilon of its own.
