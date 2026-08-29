# Migrating from libviprs 0.2.0 to 0.3.0

0.3.0 collapses every pyramid entry point into a single `EngineBuilder` and
flips `FsSink::new` to a 2-arg constructor plus a `with_format` builder. This
guide covers the call sites you are most likely to update.

**This is the 0.2.0 to 0.3.0 guide and nothing else.** There is no equivalent
for 0.4.0 or for the release in progress, which is the largest breaking one so
far: the signed and 32-bit `PixelFormat` carriers, the folded allocation
refusals, and the removed per-format error variants are all in the `Unreleased`
section of [CHANGELOG.md](CHANGELOG.md), under `### Breaking`, and that is where
to look until this file grows the section (issue #961).

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
