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
the decision record below):

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

### Decision record (issue #1705)

Status: decided in #1705 (a page made of embedded spreadsheet tables was blank in a plot), with the
limits of #1667 (placed images) as the model for the bounds.

#### Context

An OLE2FRAME is an embedded object: a spreadsheet, a document, a picture. The DWG stores the object
and, next to it, the picture the application cached for display. A drawing whose sheets are made of
such tables plotted blank, because the conversion counted the entity as an unmapped class and stopped
there. The cached picture is what the plotted PDF shows, so it is what a plot has to draw.

`acadrust` (the pinned revision) already reads the cached picture out of the frame's compound file as
one of four things: a raster file, a DIB without its file header, an enhanced metafile (EMF), or a
Windows metafile (WMF). Office writes an EMF inside a WMF wrapper, and that is unwrapped there. The
frames in the drawings this was measured on are Excel worksheets whose preview is an EMF of 10 to 20
KB and a few hundred records, using 22 record types: lines, filled cells and text.

#### Decision

- **O-1 Opt in.** `PvecConversion::ole_previews(bool)` and `PdfPlot::ole_previews(bool)`, off by
  default, as `images`, `texts` and `fills` are. Off, an OLE2FRAME is counted as before and no
  corpus baseline moves.
- **O-2 An EMF is played as vectors.** A bounded player (`pvec::emf::play`) turns `LineTo` into
  polylines, a pattern `BitBlt` or `Rectangle` into a solid cell, and `ExtTextOutW` into a text run.
  The lines are leaves of the sheet like a leader's lines (#1671); the cells and text go in the hatch and
  text streams (#1609), so a plot gets them as fills and as text and not as a picture. Nothing is
  rasterised, no dependency comes in, and a metafile that is the same table at any scale stays sharp.
- **O-3 Placement.** The metafile's bounds, which are inclusive on both edges, are mapped onto the
  frame's two corners, so a table's outer lines land on them. The frame's own plane and any viewport
  or block it is seen through apply as for any entity. The map mode, window and viewport origins and extents, and the world transform (set or modified) are applied to every point, so a picture with a window origin or a non-1:1 ratio lands where it says.
- **O-4 A raster or DIB is placed like an IMAGE (#1667).** A frame is made from the corners and the
  picture goes through the same placement, with the same limits: 64 Mpixel an image, 256 images and 256
  Mpixel in all, the declared size read from the header and refused before a pixel is allocated. A DIB
  of 8, 24 or 32 bits without compression is read by a small decoder here, because the `image`
  features the crate builds with have no BMP codec.
- **O-5 Limits.** A preview over 8 MiB (an EMF) or 32 MiB (a raster) is refused. An EMF is refused for
  more than 200000 records, more than 4096 shapes, a handle past 4096, more than 64 saved device
  contexts, more than 131072 characters of text, a record that is shorter than its header, not a whole number of words or runs past the
  end, a first record that is not a header, or a picture with no width or height. Each is a typed
  `EmfRefusal` and the frame draws nothing. A raster or DIB preview is charged to `max_allocation_bytes`
  (its bytes and four bytes a declared pixel) before it is decoded, and a conversion over the budget is
  refused with `LIMIT_ALLOCATION_BYTES` as for the other stages. The player allocates no more than these limits allow, and a fuzzed record stream does not panic it. The compound file the preview is found in is read by acadrust, whose reader took a stream's declared size on trust until the fork's #76; the pin is at that fix, and a compound file whose chains loop or whose streams declare more than the file holds gives no preview, so the frame has a row that says so.
- **O-6 What the row says.** A frame's row says what became of it: `OLE2FRAME: drawn from its EMF
  preview`, `placed as a raster from its preview`, or `not drawn, <why>` (no preview, a Windows
  metafile, a refusal, a limit). A record type the player does not know is counted in a note, never
  skipped silently. A frame with no preview is a row, not an error.
- **O-7 A linked object** draws its cached picture only. The linked file is never opened, and nothing
  here reads a path.

#### What is not carried

Pen and text colours and pen widths (a stroke and a text take the entity's own style; a cell keeps its
brush colour), bitmaps inside an EMF, curves, clip regions beyond their bounding box,
and a Windows metafile preview, which is not played. A frame is one row per line it draws, as a
leader is.
