# Migrating from libviprs 0.2.0 to 0.3.0

0.3.0 collapses every pyramid entry point into a single `EngineBuilder` and
flips `FsSink::new` to a 2-arg constructor plus a `with_format` builder. This
guide covers the call sites you are most likely to update.

**This file also covers 0.4.0 to 0.5.0, further down.** That section covers the
new `PyramidStorage` type and the PMTiles archive behind it, the two extra
comparisons an archive verify makes, plus five specific renames and removals: the signed and 32-bit `PixelFormat` carriers, the collapsed
allocation refusals, `GifError::BadPageNumber`,
`ConvolutionError::TimesOutOfRange`, and `ConversionError::UnsupportedSampleKind`.
The rest of that release, the colour and rounding changes that move output
bytes without touching a signature, the options-struct and `.v` container
group, and the raster-interpretation group, is the `### Breaking` section of
the `Unreleased` block in [CHANGELOG.md](CHANGELOG.md), which opens with a
preamble naming the issue behind every entry.

## `FsSink`

The third format argument is gone. Set the format via the builder; default is
`TileFormat::Png`.

```rust
// 0.2.0
let sink = FsSink::new("out", plan.clone(), TileFormat::Png);

// 0.3.0
let sink = FsSink::new("out", plan.clone()).with_format(TileFormat::Png);
```

There is no `FsSink::new_with_format`. This file said it "still compiles as a
deprecated alias" for three releases; it was already gone by v0.4.0 and the
crate carries **zero** `#[deprecated]` attributes, so nothing here is a
deprecated alias of anything (issue #950). The 3-arg call is a compile error
and the builder above is the only form.

## Free `generate_pyramid_*` functions → `EngineBuilder`

All five free functions plus `generate_pyramid_resumable` are removed. Replace
each call with `EngineBuilder::new(source, plan, sink)` and pick the engine
via `with_engine(EngineKind::...)` (omit it for `Auto`).

### Monolithic (in-memory)

```rust
// 0.2.0
let result = generate_pyramid(&raster, &plan, &sink, &config)?;

// 0.3.0
let result = EngineBuilder::new(&raster, plan, sink)
    .with_config(config)
    .run()?;
```

### Observed monolithic

```rust
// 0.2.0
let result = generate_pyramid_observed(&raster, &plan, &sink, &config, &observer)?;

// 0.3.0
let result = EngineBuilder::new(&raster, plan, sink)
    .with_config(config)
    .with_observer(observer)
    .run()?;
```

### Streaming

```rust
// 0.2.0
let result = generate_pyramid_streaming(&strip_src, &plan, &sink, &cfg, &observer)?;

// 0.3.0
let result = EngineBuilder::new(strip_src, plan, sink)
    .with_engine(EngineKind::Streaming)
    .with_memory_budget(cfg.memory_budget_bytes)
    .with_budget_policy(cfg.budget_policy)
    .with_observer(observer)
    .run()?;
```

### MapReduce (and the `_auto` variant)

```rust
// 0.2.0
let result = generate_pyramid_mapreduce(&strip_src, &plan, &sink, &cfg, &observer)?;
// or
let result = generate_pyramid_mapreduce_auto(&strip_src, &plan, &sink, &cfg, &observer)?;

// 0.3.0
let result = EngineBuilder::new(strip_src, plan, sink)
    .with_engine(EngineKind::MapReduce)
    .with_observer(observer)
    .run()?;
```

`EngineKind::Auto` (the default) picks Monolithic, Streaming, or MapReduce
based on the source kind and memory budget.

### Resumable

`generate_pyramid_resumable` is absorbed into `EngineBuilder` and works for
every engine, not just the monolithic path.

```rust
// 0.2.0
let result = generate_pyramid_resumable(
    &raster, &plan, &sink, &config, &observer, checkpoint_root,
)?;

// 0.3.0
let result = EngineBuilder::new(&raster, plan, sink)
    .with_config(config)
    .with_observer(observer)
    .with_resume(
        ResumePolicy::resume()
            .with_checkpoint_root(checkpoint_root)
            .with_checkpoint_every(64),
    )
    .run()?;
```

`ResumePolicy::overwrite()`, `::resume()`, and `::verify()` anchor the mode;
`with_checkpoint_every` and `with_checkpoint_root` tune persistence.
`Default` is `Overwrite`.

## Observers / events

`EngineEvent` now covers the full pipeline lifecycle:

- `SourceLoadStarted { source_description }`
- `SourceLoaded { width, height, format: PixelFormat, size_bytes }`
- `PlanCreated { levels, total_tiles, canvas_width, canvas_height }`
- `LevelStarted { level, width, height, tile_count }`
- `TileCompleted { coord }`
- `LevelCompleted { level, tiles_produced }`
- `StripRendered { strip_index, total_strips }` (Streaming)
- `BatchStarted { batch_index, strips_in_batch, total_batches }` (MapReduce)
- `BatchCompleted { batch_index, tiles_produced }` (MapReduce)
- `Finished { total_tiles, levels }`
- `PipelineComplete`

`EngineBuilder::with_observer(impl EngineObserver + 'static)` and
`with_observer_arc(Arc<dyn EngineObserver>)` feed every engine — Monolithic,
Streaming, and MapReduce — so a single observer implementation works against
all of them. `PixelFormat` is now public and re-exported at the crate root.

## Cargo features

| Feature | Default | Purpose |
|---|---|---|
| `pdfium` | off | Vector PDF rendering, `PdfiumStripSource`, `render_page_pdfium*` |
| `pdfium-static` | off | New in 0.3.0 — pulls in `pdfium` plus `pdfium-render/static` for static linking of libpdfium |
| `s3` | off | Gates `ObjectStoreSink` against a user-injected `ObjectStore` |
| `tracing` | off | Structured spans/events |
| `packfile` | off | `PackfileSink` (write tiles into a tar/zip), now with `PackfileSinkBuilder` |

`default = []`, so no features are enabled by default. MSRV is 1.97, edition
2024. That number is `rust-version` in `Cargo.toml` and
`tests/crate_doc_matches_the_crate.rs` holds both files to it; this line named a
floor three minor versions under the manifest's for as long as nothing checked
it (issue #950).

The table above is the feature set as of 0.3.0 and is deliberately left at
that. `README.md` and the crate root carry the current one, both checked
against `[features]`.

# Migrating from libviprs 0.4.0 to 0.5.0

0.5.0 is the largest breaking release this crate has shipped, grouped into
five stories plus a handful of independent items in the `Unreleased` block's
own preamble in [CHANGELOG.md](CHANGELOG.md). This section covers the storage
default flip, the new `TileFormat::Webp` variant, and five specific renames and
removals. For the rest, colour and
rounding changes that
move output bytes without touching a signature, the options-struct and `.v`
container group, and the group where the raster's tag decides instead of its
storage depth, read the preamble and follow its issue numbers into
`### Breaking`.

Three of the five renames below, the folded allocation refusals,
`GifError::BadPageNumber`, and `ConversionError::UnsupportedSampleKind`, name
things that were introduced and removed inside this same release and so
never shipped: GIF, FITS, OpenEXR and Radiance decoding, and JPEG XL and WebP
multi-page loading, did not exist in 0.4.0 at all, and neither did the code
paths their interim error shapes lived on. If you are upgrading from 0.4.0
there is nothing to migrate for those three; they are here because
CHANGELOG.md's `### Breaking` section names them, and a reader working out
whether a removal reaches them deserves to be told "nothing shipped" rather
than silence. The other two, `PixelFormat`'s new carriers and
`ConvolutionError::ZeroTimes`, are real: both existed in 0.4.0 in a form this
release changes.

## PMTiles storage, and where the default actually flips

For a Rust caller this section is additive and you can read it for the new
type rather than for a migration. Nothing here moves what your code writes.
It comes out of [EPIC F](https://github.com/libviprs/libviprs/issues/986),
which put PMTiles v3 in the crate.

Before 0.5.0 a pyramid was always a tree of loose files under `{z}/{x}/{y}`,
because `FsSink` was the only thing that could write one. From 0.5.0 there is a
second shape, and it is the one `PyramidStorage::default()` names: every tile,
the directories that index them and the pyramid's metadata in a single
`.pmtiles` file. At one pyramid it is a convenience. At 100k pyramids of 20k
tiles it is the difference between one file each and about 2 billion files in
total.

The behaviour that actually flips is the `viprs` command line, where
`viprs pyramid input.tif` writes an archive instead of a tree. That is
[libviprs-cli#54](https://github.com/libviprs/libviprs-cli/issues/54), a
different repository on a release of its own. The library stays sink-explicit,
which is what EPIC F's compatibility section asks of it.

### Nothing you wrote stops compiling

`EngineBuilder::new(source, plan, sink)` still writes the sink you hand it and
nothing else. No signature moved, `FsSink` did not move, `Layout` did not move,
and `EngineBuilder` grew no required argument. Your third argument already
names a sink, because it always had to:

```rust
// 0.4.0, and 0.5.0, byte for byte the same tree.
let sink = FsSink::new("output_tiles", plan.clone()).with_format(TileFormat::Png);
let result = EngineBuilder::new(&raster, plan, sink).run()?;
```

One command over your own tree tells you how exposed you are:

```sh
grep -rn 'EngineBuilder::new' src/
```

Every hit names its sink, so **if you are a Rust caller this particular change
costs you nothing** and you can skip to the next section. The release has other
breaks and they are below; this is not one of them. A default is
what a caller gets when they do not choose, and until 0.5.0 there was no way
not to choose.

### Where the choice lives now

`PyramidStorage` is the one place the decision is made, so nothing downstream
has to reinvent it:

```rust
use libviprs::{FsSink, Layout, PyramidStorage};

// The default, which is the part that is new.
assert_eq!(PyramidStorage::default(), PyramidStorage::PmTiles);

// The old behaviour, restored by naming it.
let storage = PyramidStorage::Directory;
let out = storage.output_path("output_tiles");   // `output_tiles`, unchanged
let sink = FsSink::new(&out, plan.clone());
```

For the archive, build a `PmTilesSink` over
`PyramidStorage::PmTiles.output_path(base)` and hand that to `EngineBuilder`
instead. Reading one back is the `pmtiles` module.

The `viprs` command-line flip is
[libviprs-cli#54](https://github.com/libviprs/libviprs-cli/issues/54) and has
its own notes; this file is the library.

### Path and extension rules

| You ask for | `PmTiles` writes | `Directory` writes |
|---|---|---|
| `city` | `city.pmtiles` | `city/` |
| `city.pmtiles` | `city.pmtiles` | `city.pmtiles/` |
| `city.PMTILES` | `city.PMTILES` | `city.PMTILES/` |
| `tiles.v2` | `tiles.v2.pmtiles` | `tiles.v2/` |

The extension is appended, never substituted. `PathBuf::set_extension` replaces
everything after the last dot, so it would turn `tiles.v2` into
`tiles.pmtiles` and drop the `v2`. A base that already ends in `.pmtiles` comes
back untouched, matched without case, because `city.PMTILES` and
`city.PMTILES.pmtiles` are two names for one file on macOS and Windows. If you
want `city.tif` to become `city.pmtiles`, hand it the stem rather than the
whole name.

`output_path` is path arithmetic. It reads nothing, creates nothing and checks
nothing; the sink is what touches the filesystem.

Writing one is atomic. The archive is staged into `<path>.tmp` and its
siblings, flushed with `sync_all` and renamed into place, so nothing exists at
`<path>` until the run finishes and an interrupted run never leaves a short
file wearing the name a complete one would. Two runs aimed at one archive would
share those staging names, so the second sink is refused when it is built.

### Two things an archive will not do

- **A layout that is not addressed by `(z, x, y)`.** An archive keys a tile
  on one `u64` derived from `(z, x, y)`, so `Layout::Xyz` and `Layout::Google`
  both fit and `Layout::DeepZoom`, `Layout::Zoomify` and `Layout::Iiif` do
  not: their level index is a tier rather than a zoom. Those three stay on the
  directory tree. `PyramidStorage::accepts_layout` answers which is which, and
  there is no coordinate migration in either direction: an XYZ tree and an
  archive of the same pyramid hold the same tiles at the same addresses.
- **`TileFormat::Raw`.** The spec has a tile type for PNG and one for JPEG and
  none for raw pixel bytes. Raw tiles keep working; they keep working in a
  directory.

### If you were already reading the tree yourself

A directory of tiles is a stable public artifact and it is not going anywhere,
so code that globs `{z}/{x}/{y}.png` off an `FsSink` run keeps working as long
as it keeps asking for `PyramidStorage::Directory`. What will break is code
that assumed the directory was the *only* thing a run could produce, and the
fix for that is to name the storage rather than to infer it from the path.

## `TileFormat` gains a `Webp` variant, so exhaustive matches stop compiling

This one is a real break for a 0.4.0 caller and it is the easiest to fix
(issue #1123). `TileFormat` is not `#[non_exhaustive]`, so a `match` on it
outside the crate has to cover every variant, and there is now a fourth:

```rust
match format {
    TileFormat::Png => ...,
    TileFormat::Jpeg { quality } => ...,
    TileFormat::Raw => ...,
    TileFormat::Webp => ...,   // add this arm
}
```

That is the entire migration. Nothing existing changed meaning, nothing moved,
and a build that does not match on `TileFormat` is untouched.

The variant was added this way on purpose rather than behind
`#[non_exhaustive]`. Adding the attribute first would have forced every
downstream match to grow a catch-all, and a catch-all is exactly what let five
sites *inside* this crate keep compiling while quietly doing the wrong thing
with a format they had never heard of. The compile errors are the feature.

**`Webp` carries no quality.** The encoder behind it
(`Raster::encode_webp`) is lossless and has no quality knob to point a number
at, so `Webp { quality }` would be an argument thrown away, and a semver time
bomb the day a lossy encoder lands. If you want a knob, the place it will
appear is `webp::Compression`, which is `#[non_exhaustive]` for that reason.

Reading side, one behaviour changes without a signature moving:
`PmTilesPyramidReader::describe()` used to answer `format: None` for a WebP
archive and now answers `Some(TileFormat::Webp)`, because the archive's tile
type determines the variant completely. JPEG is still `None` for a foreign
archive, since its quality lives only in the `vnd.libviprs` namespace.

The same method also gained a way to fail. If an archive's metadata cannot be
parsed **and** it carries a `vnd.libviprs` key, `describe()` now returns
`PyramidReadError::MetadataFromANewerLibviprs` naming the libviprs version that
wrote it, instead of quietly answering all-`None`. An archive with no
`vnd.libviprs` key still describes itself exactly as before, so nothing a
foreign go-pmtiles file does changes.

## An archive verify checks the source size and the overlap

Nothing stops compiling here. What changes is which archives
`ResumeMode::Verify` accepts, and it is strictly fewer (issue #1130).

A `pyramid_verify` run now compares two more things from the archive's
`vnd.libviprs` metadata against the plan in front of it: the source raster's
pixel dimensions, and the overlap the run was planned with. Both were in the
file all along and both were dropped on the way out.

They are worth the break because nothing else in that verify can see them. A
pyramid's level range is its longest side rounded up to a power of two, and
each level's grid is that level's size divided by the tile size and rounded up,
so a 4000x4000 source and a 4096x4096 one at tile 256 plan thirteen identical
levels, identical grids, and exactly the same 349 coordinates. Every check
passed on an archive generated from a different picture. Overlap does not touch
the grid at all, so two plans that disagree about it are byte-identical in
`levels` and have no tile's pixels in common.

**If your archives were written by this crate's `PmTilesSink`, nothing to do.**
The sink has always recorded both, from `plan.image_width` / `plan.image_height`
and `plan.overlap`, so a verify against the plan that wrote the archive passes
exactly as before.

**If an archive carries no `vnd.libviprs.source` object, the verify now
refuses it.** That is an archive assembled from tiles rather than generated
from a raster, and the refusal names what is missing:

```text
Verify: the pyramid does not record the source size it was generated from,
so there is nothing to check this plan against
```

The reasoning is the same one that already applied to the tile size: a pyramid
that will not say how it was made cannot be checked against a plan, and
"cannot be checked" is a refusal rather than a pass.

**There is no workaround, and an earlier draft of this section offered one that
cannot be followed.** It said to verify against a plan built from what the
archive records. The archives this refuses are exactly the ones that record
nothing, so there is nothing to build a plan from. Two routes that do work:
regenerate the archive with this version, which writes the namespace; or check
it structurally instead of against a plan, with
`PmTilesPyramidReader::structural_summary`, which walks the archive's own
consistency and asks nothing about a plan.

## `bytes_read` is 0 for a PMTiles verify of a local archive

Not a break in the API and visible in the numbers, so it is here rather than in
the changelog alone. A verify that takes each tile's stored length out of the
index did not read the payloads, so it reports none. Reporting the summed
lengths would be the same dishonesty the change exists to remove: a run saying
it read an archive it did not read.

Read `EngineResult::tile_evidence` to tell the two apart.
`TileEvidence::LengthsFromTheIndex` is the length path and comes with
`bytes_read: 0`; `TileEvidence::PayloadsRead` means every byte came off the
storage and `bytes_read` is their total. A verify over a remote archive is
always the second.

## `PyramidReader::self_check` is deprecated

Only for implementors of the trait, and it still compiles.

`pyramid_verify` asks `PyramidReader::structural_summary` directly now, so
nothing calls `self_check` and an override of it is dead code that the compiler
is happy with. If you put a structural walk there, move it to
`structural_summary`, which answers the same check and the addressed count from
one walk.

Do not make `structural_summary` delegate back to `self_check` to keep an
override alive. `self_check`'s default already calls `structural_summary`, so
the pair recurses until the stack runs out.

One thing this does **not** fix, worth knowing because it looks like it
should. `tile_coord_to_zxy` ignores `layout`, so an `Xyz` archive and a
`Google` archive of one source are byte-identical apart from a string in the
metadata. The layout check compares that annotation, not the tiles.

## `PixelFormat` gains signed and 32-bit carriers

`PixelFormat` was already `#[non_exhaustive]`, so an exhaustive match on it
was already a compile error before this release; that is not what moves.
Four carriers join it: `Int8`, `Int16`, `Int32` and `Uint32` (all
`NonZeroU16`-banded, the same shape as `Multi8` / `Multi16` / `FloatF32`),
covering libvips's signed and 32-bit integer carriers (issues #516, #517,
#532, #759, #887, #905, #931).

What actually breaks is a byte-width or colour-type assumption. Before this
release `bytes_per_channel() == 4` meant "this is a float format"; now
`Uint32` and `Int32` are 4 bytes too, and neither is a float. Before this
release `bytes_per_channel() == 1` meant "unsigned 8-bit"; `Int8` is 1 byte
and signed. Code that inferred a sample's type from its width needs
`PixelFormat::kind` instead, which answers a `pixel::SampleKind` (`U8`,
`U16`, `U32`, `I8`, `I16`, `I32`, `F32`) through a match with no wildcard arm,
so a future carrier fails to compile here rather than silently reading as the
wrong type:

```rust
// 0.4.0, silently wrong once Int8 / Int16 / Int32 / Uint32 exist
let is_float = format.bytes_per_channel() == 4;

// 0.5.0
let is_float = format.kind() == SampleKind::F32; // or format.is_float()
```

A handful of ops that count or sum pixels change their output format as a
direct consequence: `profile` and `project` now emit `Int32`, and the
histogram and `hough_*` family now emit `Uint32` instead of saturating a
16-bit one. Each is its own `### Breaking` entry in CHANGELOG.md, because the
interesting part is the value each now carries rather than the format name.
`Jp2kError::SignedComponent` is also gone: it existed only because
`PixelFormat` had no signed carrier to decode a signed JPEG 2000 component
into (issue #905).

## `SourceError::AllocLimitExceeded` is the one allocation-refusal shape, and it is new

Native GIF, FITS, OpenEXR and Radiance decoding did not exist in 0.4.0, so
the five per-format refusals CHANGELOG.md's `### Breaking` section describes
folding, `GifError::AllocLimitExceeded`, `FitsError::AllocLimitExceeded`,
`ExrError::AllocLimitExceeded`, `RadianceError::AllocLimitExceeded` and
`JxlError::AllocLimitExceeded`, were an interim shape #632 introduced and
#686 folded away before any of them reached a release. Upgrading from 0.4.0
means matching the final shape directly, with nothing to migrate from:

```rust
Err(SourceError::AllocLimitExceeded {
    geometry: Some(DeclaredGeometry { width, height, .. }),
    needed_bytes,
    ..
}) => ...
```

Both the enum and `DeclaredGeometry` are `#[non_exhaustive]`, so a
destructuring match needs `..` in both places. Simpler still, call
`err.is_alloc_limit()` instead of matching by hand: it answers `true` for
this variant, for `JxlError::DecoderAllocLimitExceeded` (a separate ceiling,
because it is `jxl-oxide`'s own tracker refusing a buffer it does not report
a size for), and for the refusal that JPEG, PNG, single-image TIFF and WebP
report through `image`'s own limiter as `SourceError::Decode`. All three
shapes are an allocation refusal; only two of them are something you can
destructure.

## `SourceError::PageOutOfRange` is the one page-refusal shape, and it is new too

GIF's animated load, and WebP's and JPEG XL's own multi-page support, are all
new since 0.4.0, so this refusal never had a 0.4.0 shape to preserve either.
GIF's own page-window check went through one interim spelling first,
`GifError::BadPageNumber` (its field for how many frames the file held was
called `frames`), before #845 folded it into the same
`SourceError::PageOutOfRange` the WebP and JPEG XL loaders use. Upgrading
from 0.4.0 means matching the final shape directly:

```rust
Err(SourceError::PageOutOfRange { format, page, pages, .. }) => ...
```

`format` is `"gif"`, `"webp"` or `"jxl"`, whichever loader is in play; the
field is called `pages` on this shared variant.

## `ConvolutionError::ZeroTimes` is `TimesOutOfRange`, and this one shipped

`ConvolutionError::TimesOutOfRange { times, min, max }` replaces
`ConvolutionError::ZeroTimes` and refuses both ends of `compass`'s `times`
range instead of only zero (issue #947); `min` is `1` and `max` is `1000`,
matching what `vips compass` itself accepts.

This is the one rename on this page where the old spelling really did ship:
`ConvolutionError::ZeroTimes` shipped in `v0.4.0`. A 0.4.0 caller matching it
by name has to move to `TimesOutOfRange`. `ConvolutionError` is
`#[non_exhaustive]`, so a wildcard arm already compiles today and simply
stops seeing the old variant once you upgrade.

## `ConversionError::UnsupportedSampleKind` never shipped

It was added and removed inside this same release (issue #931), and nothing
in the crate ever constructed it.
`ConversionError::UnsupportedSampleKind` has never been in a release.
If your code matches it, delete the arm; there is nothing to replace it
with, and `ConversionError` being `#[non_exhaustive]` means the match still
compiles either way.

This is unrelated to `BandError::UnsupportedSampleKind`,
`ExtractError::UnsupportedSampleKind`, `JxlError::UnsupportedSampleKind` and
`MosaicError::UnsupportedSampleKind`, which live on different enums and are
not going anywhere.
