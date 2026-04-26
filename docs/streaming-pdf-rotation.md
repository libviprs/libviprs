# Streaming PDF strip rendering — coordinate-system primer

This document explains the coordinate-system math that powers
`PdfiumStripSource`'s streaming mode, and where the per-rotation device
matrices come from. Read this before touching `strip_matrix` in
`src/pdf.rs`.

## Why the matrix exists

Pdfium offers two render entry points:

- `FPDF_RenderPageBitmap` — the "form-data path". Takes pixel
  coordinates plus a `rotate` parameter; pdfium internally applies the
  page's intrinsic `/Rotate` and writes the pixels into the supplied
  bitmap. This is what the cached-mode source uses (via
  `pdfium-render`'s `set_target_width` builder).

- `FPDF_RenderPageBitmapWithMatrix` — the "matrix path". Takes a 6-element
  affine matrix `[a, b, c, d, e, f]` plus a clipping rectangle and writes
  whatever the matrix-transformed page coords land within the clip. The
  matrix path does **not** auto-apply `/Rotate`; the caller has to compose
  rotation into the matrix.

Streaming mode wants the matrix path because that's the only way to
allocate a strip-sized bitmap (the form-data path's `start_x`/`start_y`
fields are exposed by `pdfium-render` only via internal APIs). The
caller-side matrix composition is what `strip_matrix` does.

## Coordinate systems involved

Three frames matter:

| Frame                     | Origin       | Y direction | Units  |
|---------------------------|--------------|-------------|--------|
| **Page space**            | bottom-left  | y-up        | points |
| **Display space**         | top-left     | y-down      | points |
| **Bitmap pixel space**    | top-left     | y-down      | pixels |

`Page space` is the PDF spec's "default user space" pre-`/Rotate`. A
`/Rotate 270` portrait page has page-space dimensions `(W_pt, H_pt)` =
(short, long), and after the rotation it displays as landscape with
display-space dimensions `(H_pt, W_pt)` = (long, short).

`pdfium-render`'s `PdfPage::width()`/`height()` return **display-space**
dimensions, already swapped for `/Rotate 90/270`. This is the convention
pinned by the comment at `pdf.rs:376-379`.

`Bitmap pixel space` is what the matrix produces — the (x, y) the
matrix outputs are bitmap pixel coordinates, with (0, 0) at top-left
and Y growing downward.

## Per-rotation matrix derivation

For each of the four rotation values, we want a matrix that maps
**page-space coords** to **bitmap pixel coords**, including:

1. Apply `/Rotate` (move page-content corners to their displayed positions).
2. Convert from y-up (page) to y-down (bitmap).
3. Scale by `s = dpi / 72` so points become pixels.
4. Translate by `-y_offset` so the strip rows `[y_offset, y_offset+strip_h)`
   in display-space land at rows `[0, strip_h)` in the strip-sized bitmap.

The PDF matrix convention is `(x', y') = (a·x + c·y + e, b·x + d·y + f)`
applied to `(x, y) ∈ page-space`.

### `/Rotate 0`

Page corner `(x_pg, y_pg) ∈ [0, W_pt] × [0, H_pt]` maps to display
`(x_pg, H_pt - y_pg)` in display-points (y-flip). After the strip
translation:

```
x_bm    =  s·x_pg
y_bm_strip  =  -s·y_pg + s·H_pt - y_off
```

Matrix: `[s, 0, 0, -s, 0, s·H_pt - y_off]`.

### `/Rotate 90` (90° clockwise)

A `/Rotate 90` portrait page (page-space `W_pt × H_pt`) displays as
landscape (display `H_pt × W_pt`). Walking the corners under a 90° CW
rotation about the origin and shifting back into the positive
quadrant:

| Page corner    | Display position |
|----------------|------------------|
| BL (0, 0)      | top-left         |
| BR (W_pt, 0)   | bottom-left      |
| TR (W_pt, H_pt)| bottom-right     |
| TL (0, H_pt)   | top-right        |

Page → display y-down: `x_disp = y_pg`, `y_disp = x_pg`.

In pixels with strip translation:
```
x_bm    =  s·y_pg
y_bm_strip  =  s·x_pg - y_off
```

Matrix: `[0, s, s, 0, 0, -y_off]`.

### `/Rotate 180`

Half-turn rotation: `x_disp = W_pt - x_pg`, `y_disp = y_pg`. In
pixels:
```
x_bm    =  -s·x_pg + s·W_pt
y_bm_strip  =  s·y_pg - y_off
```

Matrix: `[-s, 0, 0, s, s·W_pt, -y_off]`.

### `/Rotate 270` (270° clockwise = 90° CCW)

Display dimensions: landscape (`H_pt × W_pt`). Page → display y-down:
`x_disp = H_pt - y_pg`, `y_disp = W_pt - x_pg`. In pixels:
```
x_bm    =  -s·y_pg + s·H_pt
y_bm_strip  =  -s·x_pg + s·W_pt - y_off
```

Matrix: `[0, -s, -s, 0, s·H_pt, s·W_pt - y_off]`.

## What the bitmap looks like

For all four rotations, the bitmap allocated by
`PdfRenderConfig::set_fixed_size(display_w_px, strip_h_px)` is exactly
the strip's size. The matrix sends only the page content that should
land at rows `[0, strip_h_px)` of that bitmap; everything else falls
outside the clip and is discarded.

This is **why streaming mode bounds source-side memory at one strip,
not the full page** — at no point does the matrix path allocate a
full-page bitmap.

## Why the matrix path was hard to get right

The previous attempt at streaming mode (reverted before v0.3.0)
relied on `pdfium-render::PdfRenderConfig::rotate(...)` to compose
the rotation. `pdfium-render`'s `apply_to_page` (line ~899 of
`render_config.rs` in 0.8.37) implements rotation as a
**translate-then-rotate-then-scale** chain about the page origin.
For `/Rotate 90/270` this produces final matrices in entirely
non-positive coordinate space — without an explicit pre-translation
or alternative composition order, the rendered content lands outside
the default clipping rectangle and the bitmap stays empty. The
practical symptom was 180° wrong output for `/Rotate 90/270` pages
("hand-composed rotation matrices produced 180°-off output for
/Rotate 90/270 pages across every variant we tried" — historical
comment from `streaming.rs`, removed in commit 1e64250).

The fix this codebase ships is to **bypass** `apply_to_page`'s
auto-rotation entirely and hand the matrix straight to pdfium via
`PdfRenderConfig::apply_matrix`. The four matrices above are
**direct** page→bitmap mappings; no per-step composition, no
intermediate frames, no surprises.

## Test coverage

These matrices are pinned in three layers:

1. **Pure-Rust unit tests** (`pdf::tests::strip_matrix_*`) verify
   each matrix's algebra independently — page corners go to expected
   bitmap coordinates without spinning up pdfium.
2. **Cross-product integration tests**
   (`libviprs-tests/tests/pdfium_streaming_rotation_matrix.rs`)
   exercise `/Rotate × strip-position × DPI` over the
   `/Rotate 0` and `/Rotate 270` fixtures shipped with libviprs-tests.
3. **Synthetic fixtures**
   (`libviprs-tests/tests/pdfium_streaming_synthetic_rotations.rs`)
   close the `/Rotate 90` and `/Rotate 180` coverage gap by mutating
   an existing fixture's `/Rotate` via `lopdf`.
4. **Cross-path regression test**
   (`libviprs-tests/tests/pdfium_streaming_regression_rotate.rs`)
   compares streaming-mode output against cached-mode form-data
   output at multiple strip positions — catches systematically-wrong
   matrices that self-consistency tests would miss.

If you change `strip_matrix`, all four layers should run and turn
green (run `cargo test --features pdfium -- --test-threads=1` from
`libviprs-tests/`).

## Related code

- `src/pdf.rs::strip_matrix` — the function this document explains.
- `src/pdf.rs::render_page_strip_with_page` — the call site that
  consumes the matrix and writes a strip-sized bitmap.
- `src/streaming.rs::PdfiumStripSource` (streaming mode) — the public
  API that drives all of the above.
- `src/pdf.rs::pdfium_lock` — the process-wide lock that serialises
  every FPDF call (orthogonal to the matrix math; relevant if you're
  reading the call chain).
