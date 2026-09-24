# Tile codecs on a vector CAD sheet

This is the procedure behind the tile-format numbers in issue #1134, re-run
against the encoder issues #1132 and #1133 shipped. It says what is measured,
how to rerun it, which of #1134's cells survived that change, and what the run
still cannot answer.

#1134 is a record of a hand-driven run from 2026-09-22, filed as an issue
rather than published, because libviprs.org's benchmarks page takes only
entries that join an archived document at the pinned revision by run id, and
that run had no run id: no benchmark cell anywhere in this project can express
a tile-format claim. libviprs-bench#102 is the issue for that and it is still
open, so this document does not make those numbers publishable. It makes them
true again, which is a smaller and different thing.

## Why anything had to be re-run

#1134 measured the JPEG the crate emitted on 2026-09-22, which was this:

```rust
pub(crate) fn encode_jpeg(raster: &Raster, quality: u8) -> Result<Vec<u8>, SinkError> {
    let mut buf = Vec::new();
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buf), quality);
```

`image` 0.25's encoder fixes all three components at 1x1 sampling and borrows
the Annex K Huffman tables as constants, so every tile came out 4:4:4 with
textbook tables. That is what #1134 measured and labelled **"what we emit
today"**, and it is what #1134 recommended replacing. #1132 replaced it. The
tile path now reads

```rust
TileFormat::Jpeg { quality } => encode_jpeg(raster, quality, self.background_rgb()),
```

and `encode_jpeg` calls this crate's own baseline encoder with
`JpegSubsample::Auto`, which is libvips' `VIPS_FOREIGN_SUBSAMPLE_AUTO`: 4:2:0
below quality 90 and 4:4:4 at or above. The tile default is quality 85, so a
tile is now 4:2:0 with Huffman tables built from the tile. That is exactly the
cell #1134 called `420-opt` and priced at 1.85x, which is the whole reason its
JPEG rows stopped describing this crate.

#1133 matters here for a different reason. Before it, `--render --format jpeg`
could not write one tile, because `render_page_pdfium` returns `Rgba8` and
`image`'s JPEG encoder has no RGBA colour type. So #1134's JPEG cells could not
have come from the same command as its PNG and WebP cells, and they did not:
they came from a stitched raster. Every cell here comes from one command shape.

## Setup

| | |
|---|---|
| corpus | `libviprs-tests/tests/fixtures/blueprint.pdf`, sha256 `7053ee18df30d2d3c9a0088f31a8d9c6a698a192ea707d13c09a680742fac493`, 1006356 bytes, one vector AutoCAD sheet |
| render | pdfium 8054 (`libviprs-dep` release `pdfium-8054`, linux-x64 tarball sha256 `b42d1731f07fb73edea38cbd294afe9be4bdcf8e4ed8523de51cd5d12fc8d271`) at 150 dpi to 9932x7020 `Rgba8` |
| plan | 256x256 tiles, no overlap, **15 Deep Zoom levels, 1479 tiles**, 1092 of them at level 14 |
| core / CLI | libviprs 0.5.0 at `365d6d19`, libviprs-cli 0.4.0 at `4eacf25`, rustc 1.98.1 (48a229cea 2026-09-01), `--release` |
| host | HIGARA, Intel Pentium Gold 8505, 6 logical cores, Linux 6.12.30+ x86_64, inside `--platform linux/amd64` containers |
| analysis | `python:3.12-slim`, numpy 2.5.3, Pillow 12.3.0 over libjpeg-turbo (Pillow reports the 6.2 API version) |

#1134's run was on the same box with the same corpus, the same pdfium and the
same CLI revision. rustc moved from 1.97.1 to 1.98.1 between the two, which
moves no bytes here: every byte count below is a property of the encoder's
arithmetic, not of its codegen.

## How to repeat it

One `viprs pyramid` invocation per cell per backend, then one analysis pass.
No harness, and no fixture that is not already in the tree.

```sh
viprs pyramid blueprint.pdf out/<cell>-dir \
    --storage directory --layout deep-zoom \
    --tile-size 256 --overlap 0 --dpi 150 --render \
    --format <fmt> [--quality <q>]

viprs pyramid blueprint.pdf out/<cell>.pmtiles \
    --storage pmtiles \
    --tile-size 256 --overlap 0 --dpi 150 --render \
    --format <fmt> [--quality <q>]
```

with `<cell>` one of `jpeg-q75`, `jpeg-q85`, `jpeg-q95`, `png`, `webp`. Then

```sh
pip install numpy pillow
python3 scripts/tile-fidelity.py out out/../fidelity.json
```

`--render` is what makes the JPEG cells possible at all, and it is what makes
the two backends comparable: the same pdfium raster goes into both.

Three counting rules, written down because #1134 used different ones and the
difference is small enough to look like a measurement.

**Tile bytes** are the `.jpeg` / `.png` / `.webp` files only. **Entries** count
every file and every directory including the root, so `p.dzi` and the two
`.libviprs-job.*` sidecars are in there: 1479 tiles, 1482 files, 17
directories, **1499 entries** for every directory cell. **Allocated bytes** are
`st_blocks * 512`, what the filesystem gave the file rather than what the file
claims to be. **Deduped bytes** are one copy of each distinct payload, keyed on
the sha256 of the file, which is what a store that deduplicates would hold and
what makes the archive column comparable, since PMTiles dedupes by
construction.

## The control that says the two runs are comparable

#1134's PNG total and this run's PNG total do not match, and that has to be
explained before any JPEG row is read across the two documents.

The control is libjpeg. The reference tiles from *this* run, encoded by libjpeg
at the four knob settings #1134 separated, against #1134's own four numbers:

| knob, quality 85 | this run | #1134 | deduped, this run | #1134 |
|---|--:|--:|--:|--:|
| `444-std` | 6 387 650 | 6 387 650 | 4 958 618 | 4 958 618 |
| `444-opt` | 3 990 786 | 3 990 786 | 3 356 210 | 3 356 210 |
| `420-std` | 5 293 159 | 5 293 159 | 4 271 696 | 4 271 696 |
| `420-opt` | 3 460 741 | 3 460 741 | 3 028 053 | 3 028 053 |

Every one of the eight agrees **to the byte**, on a different Pillow and a
different libjpeg-turbo build from the one #1134 used. So the two runs are
looking at the same pixels, and every JPEG comparison across them is exact
rather than approximate.

The distinct-payload counts say the same thing independently: 942 for JPEG at
quality 75 and 85, 943 at 95, 944 for PNG and for lossless WebP, in both runs.
And this run's lossless WebP total, 1 138 638 bytes, is #1134's lossless WebP
total to the byte.

**PNG is the one cell that differs, and the alpha is why.** The render path
hands the sink `Rgba8`, and all 1479 reference tiles here are RGBA with an
alpha of 255 in every sample (the analysis pass asserts that rather than
assuming it). PNG stores that plane and pays about 12% for it; JPEG flattens it
away before encoding and lossless WebP spends nothing on it, which is why those
two agree across the runs and PNG does not. That last step is inferred rather
than measured: what is measured is that the RGB behind every tile is identical,
and that PNG is the only cell where the totals part company.

## What the crate emits now

Bytes, over the whole 1479-tile pyramid.

| cell | tile bytes | deduped | distinct | archive | tree allocated | p50 tile | tiles under one block |
|---|--:|--:|--:|--:|--:|--:|--:|
| `jpeg-q75` | 3 092 360 | 2 666 036 | 942 | 2 669 267 | 6 688 768 | 1 735 | 1327 |
| `jpeg-q85` (the default) | **3 458 216** | 3 025 528 | 942 | 3 028 791 | 7 102 464 | 1 900 | 1239 |
| `jpeg-q95` | 4 853 934 | 4 207 407 | 943 | 4 210 736 | 8 060 928 | 2 636 | 1074 |
| `png` | 5 334 030 | 4 310 006 | 944 | 4 313 294 | 8 388 608 | 2 271 | 1055 |
| `webp` (lossless) | 1 138 638 | 1 082 498 | 944 | 1 085 453 | 6 242 304 | 260 | 1441 |

Against #1134's rows for the same tiles, which the control above says are the
same tiles:

| quality | #1134, `image` 0.25 | now | |
|---|--:|--:|--:|
| 75 | 5 953 742 | 3 092 360 | **1.93x** |
| 85 | 6 379 215 | 3 458 216 | **1.84x** |
| 95 | 7 492 957 | 4 853 934 | **1.54x** |

Quality 95 is the interesting one, because `Auto` leaves it at 4:4:4, so its
1.54x is the Huffman knob alone. #1134 predicted 1.60x for that from
libjpeg-turbo and got it within 4%.

**Our encoder is 2525 bytes smaller than libjpeg at matched settings**, 3 458 216
against `420-opt`'s 3 460 741 over 1479 tiles, which is 0.07%. #1134 listed
"that optimized Huffman behaves the same inside `image` 0.25 as inside
libjpeg-turbo" under **Inferred**, because `image` exposed no Huffman knob to
test it with. It is measured now, and the inference held.

Three of #1134's conclusions invert on these numbers:

- **JPEG is no longer the largest of the three.** It was 1.34x larger than PNG;
  it is now 1.54x smaller than the PNG this pipeline writes, and 1.37x smaller
  than the RGB PNG #1134 measured.
- **Lossless WebP's lead over our JPEG halved**, from 5.60x to 3.04x. It is
  still a lead and nothing here argues with #1134's conclusion that a lossy
  WebP encoder is not worth buying.
- **JPEG's block amplification got worse, not better**, from 1.43x to 2.05x,
  because the tiles got smaller and a 1900-byte tile still occupies a 4 kB
  block. The inode argument #1134 makes about WebP now reaches JPEG: the
  archive holds the same payloads in 3 031 040 bytes of real disk against the
  tree's 7 131 136, which is 2.35x.

The archive checks out the same way #1134's did. Archive size minus the deduped
payload sum is 3231 / 3263 / 3329 / 3288 / 2955 bytes for the five cells, which
is header plus directory and nothing else, and 942 distinct payloads over 1479
tiles is the same 36% duplicate rate. Lossless WebP's p50 tile is 260 bytes and
97.4% of its tiles are under one block, both of which are #1134's figures
unchanged.

## Fidelity

Every lossy cell against the PNG tree, which is lossless and therefore is the
raster the sink was handed. Definitions are in `scripts/tile-fidelity.py`.

| cell | exact tiles | PSNR median | PSNR p10 | IoU mean | IoU median | IoU p10 |
|---|--:|--:|--:|--:|--:|--:|
| `jpeg-q75` | 340 | 48.15 dB | 42.61 dB | 0.97068 | 0.99718 | 0.91259 |
| `jpeg-q85` | 340 | **51.83 dB** | 46.17 dB | 0.98541 | 0.99887 | 0.95684 |
| `jpeg-q95` | 385 | 59.93 dB | 54.43 dB | 0.99402 | 1.0 | 0.98355 |
| `webp` | **1479 / 1479** | n/a | n/a | 1.0 | 1.0 | 1.0 |

The WebP row is the positive control libviprs-bench#102 asks every lossless
cell for: all 1479 tiles decode to the reference exactly, so the comparison is
walking the pyramids it thinks it is.

**The new encoder is very slightly more faithful as well as much smaller.**
#1134's PSNR medians for the old encoder were 48.12, 51.74 and 59.74 dB at the
three qualities; these are 48.15, 51.83 and 59.93. Agreement to a fifth of a dB
at three separate qualities is also what says the two runs compute PSNR the same
way, which is worth having, because **the IoU columns are not comparable across
the two documents.** #1134 describes its ink-mask metric in prose and its script
is not in the tree, so `scripts/tile-fidelity.py` is a reconstruction from that
prose, and it is systematically more generous: 0.98541 mean here against 0.97324
there for what is nearly the same output. Read the IoU columns within this
document and not against that one.

By level, at the default quality. Levels 0 to 8 hold one tile each and level 14
is the full-resolution sheet:

| level | tiles | jpeg q85 | png | webp | q85 PSNR median | q85 IoU mean | q85 IoU p10 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0 to 10 | 17 | 58 203 | 163 511 | 64 102 | | | |
| 11 | 20 | 108 082 | 179 827 | 66 472 | 45.79 dB | 0.96197 | 0.91737 |
| 12 | 70 | 298 702 | 422 004 | 126 074 | 47.32 dB | 0.96036 | 0.87500 |
| 13 | 280 | 807 986 | 1 143 262 | 266 298 | 49.04 dB | 0.95875 | 0.89853 |
| 14 | 1092 | 2 185 243 | 3 425 426 | 615 692 | 52.80 dB | 0.99538 | 0.98724 |

This is where #1134's class table can be read again without re-running it. Its
`jpeg-tuned` column is **2 186 342** bytes at level 14 and **1 216 002** across
levels 11 to 13; the same levels here are 2 185 243 and 1 214 770, within 0.05%
and 0.10%. So that column and its `today` column are now the same encoder, the
ratios between them (1.99x at full resolution, 1.60x deeper) are gone, and
every "vs today" figure in that issue is against a baseline that moved.

The deeper levels score worse on both metrics than the full-resolution one,
which is not a downsampling artefact: a level-13 tile is four level-14 tiles'
worth of linework squeezed into the same 256 pixels, so its ink is a larger
fraction of a smaller number of pixels and there is proportionally more of it
sitting on a DCT block boundary. #1134 found the same shape from the other
direction, in that the lossy WebP quality needed to match went up with depth.

## What #1132 took out of #1134, cell by cell

So nobody has to diff two documents in their head. "Superseded" means the row
describes code that is not in the tree, not that the row was wrong when it was
taken.

| #1134 says | status |
|---|---|
| `jpeg-q85-libviprs`, `image 0.25 (pure Rust)`, 6 379 215 B | **superseded.** That encoder is gone; the cell is 3 458 216 B. |
| `444-std` **(what we emit today)** | **superseded label.** At the default quality a tile is 4:2:0 with tables built from it. |
| `444-opt` is 1.60x, "lossless rearrangement, free" | **realised**, at 1.54x measured here, and not free: the encoder transforms the tile twice and PR #1148 measured 0.93 ms against `image`'s 0.50 ms. |
| the `jpeg-q75` / `q85` / `q95` rows of the naive table | **superseded.** All three name `image` 0.25. |
| `jpeg q85` **ours** as the 1.00x baseline of the matched table | **superseded.** The baseline moved 1.84x, so every "vs today" ratio in that table moved with it. |
| the `today` and `jpeg-tuned` columns of the class table | **collapsed** into one, to within 0.1% at every level. |
| `directory --format jpeg -q 85` 6 503 887 B, `pmtiles` 4 953 402 B | **superseded**, and never a like-for-like with a rendered run. |
| "JPEG amplifies 1.43x" | **superseded**, it is 2.05x, because the tiles got smaller. |
| "**PNG beats JPEG q85 by 1.34x**" | **reversed.** |
| the libviprs JPEG q75 to q100 metric-warning ladder | **superseded.** It was a ladder up the old encoder. |
| "Three cheaper things, in order: 1. optimized Huffman (#1132), 2. fix `--render --format jpeg` (#1133)" | **both done**, in PR #1148. |
| "Inferred: that optimized Huffman behaves the same inside `image` 0.25 as inside libjpeg-turbo" | **measured now**, and it held to 0.07%. |
| **"4:2:0: only behind a chroma check"** | **not done, and still live.** See below. |
| every WebP and PNG number, the lossy-WebP crossover, the libwebp ladder warning, the storage-backend findings, the RSS parity, the inode argument | **untouched.** #1132 changed one encoder. |

## The recommendation that did not ship as written

#1134's colour control found that 4:2:0 saves more on coloured ink and damages
it 5x harder, that ink-mask IoU is blind to the damage, and concluded: **"Enable
4:2:0 behind a chroma check, not unconditionally."**

What shipped is a *quality* check. `subsampled` in `src/encode_jpeg.rs` is

```rust
JpegSubsample::Auto => quality < 90,
```

which is libvips' rule and defensible on its own terms, but it is not the rule
#1134 asked for, and at the tile default of quality 85 it subsamples every tile
whatever colour is in it. PR #1148 measured the cost against our own encoder
rather than against libjpeg-turbo, and it is the cost #1134 predicted: coloured
line art goes from 22719 bytes at 35.01 dB to 15385 bytes at **30.05 dB**, a
4.96 dB drop, while the black-on-white drawing beside it moves 0.02 dB.

Nothing in this document can settle it, because this corpus has no colour in it
to lose: #1134 counted 293 pixels out of 94.5M with any channel spread above
10. It is the one open recommendation out of that issue and it needs the
coloured-layer sheet libviprs-bench#102 also wants.

## What this run does not measure

**No wall clock.** The box was running another lane's gate throughout: up to
six other `libviprs-ci` and `nas-driver` containers at once, with the
one-minute load average between 6 and 63 on six logical cores. So every
wall-clock number this run produced is a number about that gate, and none of
them is published. It did not go quiet at any point during the run, so waiting
for it was not an option either. Byte counts,
entry counts, allocated blocks and fidelity are all independent of contention,
and they are the only columns published here. #1134's storage timings stand on
their own run; they were never comparable to a run like this one anyway,
because #1133 says its JPEG rows could not have used `--render`, and its 468 MB
peak RSS against this run's 842 MB says the same thing from the other side.

**No reference codec beyond the control.** #1134's libwebp and libpng columns
are not reproduced. They were the point when the question was "is our encoder
the problem", and #1132 answered that by replacing the encoder. #1134's own
note that its C-codec *timings* were a 3x harness artefact still stands; its
size figures for those codecs are not contradicted by anything here.

**No colour**, for the reason in the section above.

**One sheet, one renderer, one dpi, one tile size, x86_64 only.**

## Why this is a document and not a published benchmark

These numbers still cannot go on libviprs.org. `ingest.mjs --check` takes only
entries that join an archived document at the pinned revision by run id with
four recomputed integrity digests, and nothing here has a run id, because no
family in libviprs-bench has a tile-format dimension: `BENCH_TILE_FORMAT` and
`BENCH_TILE_SUFFIX` are one pair of constants feeding both sides of the
cross-engine comparison on purpose, and `tests/encoding_claim.rs` asserts that.
libviprs-bench#102 is the issue for parameterising them. It is open, and its
corpus requirement is unmet: it wants a real scanned drawing and a
coloured-layer sheet, and neither is committed anywhere.

So #1134 stays open, and this document is not a substitute for it. What it was
waiting on has not moved. What changed is that half of it had stopped
describing the code, and a record that has stopped describing the code is worse
than no record, because it reads exactly like one that still does.
