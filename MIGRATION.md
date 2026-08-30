# Migrating from libviprs 0.2.0 to 0.3.0

0.3.0 collapses every pyramid entry point into a single `EngineBuilder` and
flips `FsSink::new` to a 2-arg constructor plus a `with_format` builder. This
guide covers the call sites you are most likely to update.

**This file also covers 0.4.0 to 0.5.0, further down.** That section covers
five specific renames and removals: the signed and 32-bit `PixelFormat`
carriers, the collapsed allocation refusals, `GifError::BadPageNumber`,
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
four stories plus a handful of independent items in the `Unreleased` block's
own preamble in [CHANGELOG.md](CHANGELOG.md). This section covers five
specific renames and removals. For the rest, colour and rounding changes that
move output bytes without touching a signature, the options-struct and `.v`
container group, and the group where the raster's tag decides instead of its
storage depth, read the preamble and follow its issue numbers into
`### Breaking`.

Three of the five below, the folded allocation refusals,
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
