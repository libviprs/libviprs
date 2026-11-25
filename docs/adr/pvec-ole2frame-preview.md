# ADR: An OLE2FRAME is drawn from the picture it carries

Status: decided in #1705 (a page made of embedded spreadsheet tables was blank in a plot), with the
limits of #1667 (placed images) as the model for the bounds.

## Context

An OLE2FRAME is an embedded object: a spreadsheet, a document, a picture. The DWG stores the object
and, next to it, the picture the application cached for display. A drawing whose sheets are made of
such tables plotted blank, because the conversion counted the entity as an unmapped class and stopped
there. The cached picture is what the plotted PDF shows, so it is what a plot has to draw.

`acadrust` (the pinned revision) already reads the cached picture out of the frame's compound file as
one of four things: a raster file, a DIB without its file header, an enhanced metafile (EMF), or a
Windows metafile (WMF). Office writes an EMF inside a WMF wrapper, and that is unwrapped there. The
frames in the drawings this was measured on are Excel worksheets whose preview is an EMF of 10 to 20
KB and a few hundred records, using 22 record types: lines, filled cells and text.

## Decision

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
  or block it is seen through apply as for any entity.
- **O-4 A raster or DIB is placed like an IMAGE (#1667).** A frame is made from the corners and the
  picture goes through the same placement, with the same limits: 64 Mpixel an image, 256 images and 256
  Mpixel in all, the declared size read from the header and refused before a pixel is allocated. A DIB
  of 8, 24 or 32 bits without compression is read by a small decoder here, because the `image`
  features the crate builds with have no BMP codec.
- **O-5 Limits.** A preview over 8 MiB (an EMF) or 32 MiB (a raster) is refused. An EMF is refused for
  more than 200000 records, more than 4096 shapes, a handle past 4096, more than 64 saved device
  contexts, a record that is shorter than its header, not a whole number of words or runs past the
  end, a first record that is not a header, or a picture with no width or height. Each is a typed
  `EmfRefusal` and the frame draws nothing. No input panics, and none allocates past these limits.
- **O-6 What the row says.** A frame's row says what became of it: `OLE2FRAME: drawn from its EMF
  preview`, `placed as a raster from its preview`, or `not drawn, <why>` (no preview, a Windows
  metafile, a refusal, a limit). A record type the player does not know is counted in a note, never
  skipped silently. A frame with no preview is a row, not an error.
- **O-7 A linked object** draws its cached picture only. The linked file is never opened, and nothing
  here reads a path.

## What is not carried

Pen and text colours and pen widths (a stroke and a text take the entity's own style; a cell keeps its
brush colour), bitmaps inside an EMF, curves, world transforms, clip regions beyond their bounding box,
and a Windows metafile preview, which is not played. A frame is one row per line it draws, as a
leader is.
