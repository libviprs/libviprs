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
mapping does not carry yet are TEXT and MTEXT, HATCH, DIMENSION, IMAGE, MESH, REGION, 3DSOLID, WIPEOUT, OLE2FRAME (drawn from its cached preview when asked, see below), ACAD_TABLE and
proxy entities (`unmapped-source-class`), an unresolved xref block (`unresolved-block`), and the degenerate ones (`degenerate-radius`,
`coincident-vertices`, `degenerate-geometry`). Text has a policy of its own (`docs/pvec-text-policy.md`), hatches too
(`docs/pvec-hatch-policy.md`), and units are `docs/pvec-units.md`.

One known gap: a curve PVec's container cannot store deterministically (`WRT-08`) fails the whole conversion today instead of
being counted against its entity (#1446). A plot is the exception for a polyline with arc sides: `PdfPlot` writes it one record per side and draws it (#1717). The corpus regression baseline records the three fixtures it affects.

The codes and their wording are `docs/pvec-diagnostics.md`.

## OLE2FRAME previews

An OLE2FRAME is an embedded object (a spreadsheet table, say) and the drawing stores the picture the
application cached for it. `PvecConversion::ole_previews(true)` (and `PdfPlot::ole_previews(true)`, which a plot needs
with `texts` and `fills` on for the cells and the text) draws that picture at the frame's two corners (issue #1705,
[the ADR](adr/pvec-ole2frame-preview.md)):

- an enhanced metafile is played as vectors by a small bounded player: lines, solid cells and text. Pen and text
  colours and widths are not carried, a stroke takes the entity's own style;
- a raster file or a DIB is placed like an IMAGE, behind the limits of `images` (64 Mpixel an image, 256 images and
  256 Mpixel in all, the size refused from the header before a pixel is allocated);
- a Windows metafile is not played, a frame with no preview draws nothing, and a hostile preview (over 8 MiB, over
  200000 records, over 4096 shapes, a malformed record) is a typed refusal. All of them are a line on the frame's
  row (`OLE2FRAME: not drawn, <why>`) and never an error of the sheet. A record type the player does not know is
  counted in a note.

Off by default, and then a frame is counted as an unmapped class. A linked object draws its cached picture and the
linked file is never opened.
