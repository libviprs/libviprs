# Which DWG entities are carried

Every entity the conversion meets gets one of six classes (EPIC P's): `SupportedExact`, `SupportedEquivalent`,
`RequiresPVecCapability`, `RequiresExtension`, `Unsupported`, `IgnoredByPolicy`. Only the first two (and
`RequiresExtension`, which keeps the geometry) produce a PVec curve.

## Carried

The ten kinds of normalized geometry in the table of `docs/pvec-iso-mapping.md`: lines, circles, circular arcs, ellipses and
elliptical arcs, polylines and bulge polylines (a composite), Beziers, B-splines and rational B-splines. A planar SPLINE goes
through the exact extractor: a single span becomes a Bezier, a periodic one of one period is unwrapped, a rational one keeps
its weights, and a 3D, malformed or over-limit one is refused with its own reason (`spline-3d`, `spline-malformed`,
`spline-limit-exceeded`).

## Not carried

Counted with the reason code, never reshaped. From the corpus regression (`docs/pvec-corpus-regression.md`), the classes the
mapping does not carry yet are TEXT and MTEXT, HATCH, DIMENSION, IMAGE, MESH, REGION, 3DSOLID, WIPEOUT, OLE2FRAME, ACAD_TABLE and
proxy entities (`unmapped-source-class`), an unresolved xref block (`unresolved-block`), and the degenerate ones (`degenerate-radius`,
`coincident-vertices`, `degenerate-geometry`). Text has a policy of its own (`docs/pvec-text-policy.md`), hatches too
(`docs/pvec-hatch-policy.md`), and units are `docs/pvec-units.md`.

One known gap: a curve PVec's container cannot store deterministically (`WRT-08`) fails the whole conversion today instead of
being counted against its entity (#1446). The corpus regression baseline records the three fixtures it affects.

The codes and their wording are `docs/pvec-diagnostics.md`.
