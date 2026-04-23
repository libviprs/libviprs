//! Unified fluent entry point for pyramid generation.
//!
//! [`EngineBuilder`] collapses the five free-function entry points
//! ([`generate_pyramid`](crate::generate_pyramid),
//! [`generate_pyramid_observed`](crate::generate_pyramid_observed),
//! [`generate_pyramid_streaming`](crate::generate_pyramid_streaming),
//! [`generate_pyramid_mapreduce`](crate::generate_pyramid_mapreduce),
//! [`generate_pyramid_mapreduce_auto`](crate::generate_pyramid_mapreduce_auto))
//! behind a single typed builder. Call
//! [`EngineBuilder::new(source, plan, sink)`](EngineBuilder::new), chain any
//! combination of `with_*` setters, then [`EngineBuilder::run`] or
//! [`EngineBuilder::run_collect`].
//!
//! Routing to the underlying engine is driven by two inputs:
//!
//! 1. The [`EngineKind`] set via [`EngineBuilder::with_engine`]
//!    (default: [`EngineKind::Auto`]).
//! 2. Whether the source is an in-memory [`Raster`] or a
//!    [`StripSource`](crate::streaming::StripSource).
//!
//! [`EngineKind::Monolithic`] refuses to run against a strip source and
//! surfaces [`EngineError::IncompatibleSource`] instead of silently pulling
//! the whole source into memory.

use std::sync::Arc;

use crate::dedupe::DedupeStrategy;
use crate::engine::{
    BlankTileStrategy, EngineConfig, EngineError, EngineResult, generate_pyramid_observed,
};
use crate::extensions::Extensions;
use crate::observe::{EngineObserver, NoopObserver};
use crate::planner::PyramidPlan;
use crate::raster::Raster;
use crate::resume::{ResumeMode, ResumePolicy};
use crate::retry::{FailurePolicy, RetryPolicy};
use crate::sink::TileSink;
use crate::streaming::{
    BudgetPolicy, RasterStripSource, StreamingConfig, StripSource, generate_pyramid_streaming,
};
use crate::streaming_mapreduce::{MapReduceConfig, generate_pyramid_mapreduce};

// ---------------------------------------------------------------------------
// EngineKind
// ---------------------------------------------------------------------------

/// Which underlying engine implementation [`EngineBuilder::run`] should
/// dispatch to.
///
/// `#[non_exhaustive]` so future engine variants (e.g. `Gpu`, `Distributed`,
/// `Remote`) can be added as minor-version additions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EngineKind {
    /// Pick an engine based on the source and memory budget. The default.
    #[default]
    Auto,
    /// In-memory monolithic engine. Requires an [`EngineSource::Raster`]
    /// source; a [`EngineSource::Strip`] source errors with
    /// [`EngineError::IncompatibleSource`].
    Monolithic,
    /// Sequential streaming engine. Accepts either source kind; in-memory
    /// rasters are wrapped in a [`RasterStripSource`] automatically.
    Streaming,
    /// Parallel map-reduce streaming engine. Accepts either source kind.
    MapReduce,
}

// ---------------------------------------------------------------------------
// EngineSource + IntoEngineSource
// ---------------------------------------------------------------------------

/// Input source for [`EngineBuilder`].
///
/// Callers typically construct this implicitly via
/// [`EngineBuilder::new`]'s `impl IntoEngineSource<'a>` argument — passing a
/// `&Raster` or any `T: StripSource` is enough.
pub enum EngineSource<'a> {
    /// In-memory raster, passed by reference.
    Raster(&'a Raster),
    /// Pull-based strip source, type-erased behind a trait object.
    Strip(Box<dyn StripSource + 'a>),
}

impl<'a> std::fmt::Debug for EngineSource<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raster(_) => f.debug_tuple("EngineSource::Raster").finish(),
            Self::Strip(_) => f.debug_tuple("EngineSource::Strip").finish(),
        }
    }
}

/// Conversion into an [`EngineSource`]. Implemented for `&Raster` and for
/// every `T: StripSource`, so [`EngineBuilder::new`] accepts either kind
/// of source without explicit wrapping.
pub trait IntoEngineSource<'a> {
    fn into_engine_source(self) -> EngineSource<'a>;
}

impl<'a> IntoEngineSource<'a> for &'a Raster {
    fn into_engine_source(self) -> EngineSource<'a> {
        EngineSource::Raster(self)
    }
}

impl<'a, T> IntoEngineSource<'a> for T
where
    T: StripSource + 'a,
{
    fn into_engine_source(self) -> EngineSource<'a> {
        EngineSource::Strip(Box::new(self))
    }
}

// ---------------------------------------------------------------------------
// EngineBuilder
// ---------------------------------------------------------------------------

/// Fluent entry point for pyramid generation.
///
/// Generic over the sink type so `.run()` is monomorphic for single-sink
/// callers; use `EngineBuilder<'a, Box<dyn TileSink>>` when different match
/// arms need to return different concrete sinks.
pub struct EngineBuilder<'a, S: TileSink> {
    source: EngineSource<'a>,
    plan: PyramidPlan,
    sink: S,

    engine_kind: EngineKind,
    observer: Option<Arc<dyn EngineObserver>>,

    // EngineConfig knobs
    concurrency: Option<usize>,
    buffer_size: Option<usize>,
    background_rgb: Option<[u8; 3]>,
    blank_strategy: Option<BlankTileStrategy>,
    failure_policy: Option<FailurePolicy>,
    dedupe: Option<DedupeStrategy>,

    // Resume
    resume: Option<ResumePolicy>,

    // StreamingConfig knobs
    memory_budget_bytes: Option<u64>,
    budget_policy: Option<BudgetPolicy>,

    // Extension hatch (Approach C)
    extensions: Extensions,
}

impl<'a, S: TileSink> std::fmt::Debug for EngineBuilder<'a, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineBuilder")
            .field("source", &self.source)
            .field("engine_kind", &self.engine_kind)
            .field("concurrency", &self.concurrency)
            .field("buffer_size", &self.buffer_size)
            .field("background_rgb", &self.background_rgb)
            .field("blank_strategy", &self.blank_strategy)
            .field("failure_policy", &self.failure_policy)
            .field("dedupe", &self.dedupe)
            .field("resume", &self.resume)
            .field("memory_budget_bytes", &self.memory_budget_bytes)
            .field("budget_policy", &self.budget_policy)
            .field("extensions", &self.extensions)
            .finish_non_exhaustive()
    }
}

impl<'a, S: TileSink> EngineBuilder<'a, S> {
    /// Start a builder rooted at the given source, plan, and sink.
    pub fn new(source: impl IntoEngineSource<'a>, plan: PyramidPlan, sink: S) -> Self {
        Self {
            source: source.into_engine_source(),
            plan,
            sink,
            engine_kind: EngineKind::Auto,
            observer: None,
            concurrency: None,
            buffer_size: None,
            background_rgb: None,
            blank_strategy: None,
            failure_policy: None,
            dedupe: None,
            resume: None,
            memory_budget_bytes: None,
            budget_policy: None,
            extensions: Extensions::new(),
        }
    }

    // --- typed setters (Approach A) ---

    /// Attach an observer receiving every [`EngineEvent`](crate::EngineEvent).
    pub fn with_observer(mut self, observer: impl EngineObserver + 'static) -> Self {
        self.observer = Some(Arc::new(observer));
        self
    }

    /// Attach a pre-boxed observer. Useful when the observer is already
    /// shared between callers or is `Arc<dyn EngineObserver>`-shaped at the
    /// call site.
    pub fn with_observer_arc(mut self, observer: Arc<dyn EngineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Select which engine implementation [`EngineBuilder::run`] will
    /// dispatch to. Defaults to [`EngineKind::Auto`].
    pub fn with_engine(mut self, kind: EngineKind) -> Self {
        self.engine_kind = kind;
        self
    }

    /// Apply every field of an existing [`EngineConfig`] in one call.
    ///
    /// Convenience for callers that already have a fully-constructed
    /// [`EngineConfig`] — typically because they're migrating from the
    /// old `generate_pyramid_observed(source, plan, sink, &config, obs)`
    /// free function. Individual `.with_*` setters called after
    /// `with_config` take precedence.
    ///
    /// Also threads the config's `checkpoint_every` and `checkpoint_root`
    /// into a default [`ResumePolicy::overwrite`] if no explicit policy has
    /// been attached yet — matching the behaviour of the old
    /// `generate_pyramid_resumable` path where those knobs lived directly
    /// on the `EngineConfig`.
    pub fn with_config(mut self, config: EngineConfig) -> Self {
        self.concurrency = Some(config.concurrency);
        self.buffer_size = Some(config.buffer_size);
        self.background_rgb = Some(config.background_rgb);
        self.blank_strategy = Some(config.blank_tile_strategy);
        self.failure_policy = Some(config.failure_policy);
        if let Some(ds) = config.dedupe_strategy {
            self.dedupe = Some(ds);
        }
        // Carry the checkpoint knobs into an existing ResumePolicy (if set)
        // so migrations from `generate_pyramid_resumable(.., &cfg, mode)`
        // don't silently lose the cadence / root that used to live on the
        // config.
        if config.checkpoint_every != 0 || config.checkpoint_root.is_some() {
            let mut policy = self.resume.unwrap_or_else(ResumePolicy::overwrite);
            if config.checkpoint_every != 0 && policy.checkpoint_every() == 0 {
                policy = policy.with_checkpoint_every(config.checkpoint_every);
            }
            if policy.checkpoint_root().is_none() {
                if let Some(root) = config.checkpoint_root {
                    policy = policy.with_checkpoint_root(root);
                }
            }
            self.resume = Some(policy);
        }
        self
    }

    /// Set the [`FailurePolicy`] the engine consults when a tile write
    /// terminally fails. Overrides any earlier [`EngineBuilder::with_retry`]
    /// call.
    pub fn with_failure_policy(mut self, policy: FailurePolicy) -> Self {
        self.failure_policy = Some(policy);
        self
    }

    /// Shorthand for `with_failure_policy(FailurePolicy::RetryThenFail(policy))`.
    pub fn with_retry(self, policy: RetryPolicy) -> Self {
        self.with_failure_policy(FailurePolicy::RetryThenFail(policy))
    }

    /// Control resume / verify behaviour. Only the engine's resumable path
    /// consults this; see [`ResumePolicy`] for the mode selector and the
    /// checkpoint-persistence knobs.
    pub fn with_resume(mut self, policy: ResumePolicy) -> Self {
        self.resume = Some(policy);
        self
    }

    /// Select a content-addressed deduplication strategy. See
    /// [`DedupeStrategy`] for variants.
    pub fn with_dedupe(mut self, strategy: DedupeStrategy) -> Self {
        self.dedupe = Some(strategy);
        self
    }

    /// Control how blank (uniform-colour) tiles are handled.
    pub fn with_blank_strategy(mut self, strategy: BlankTileStrategy) -> Self {
        self.blank_strategy = Some(strategy);
        self
    }

    /// Set the background RGB used to pad edge tiles.
    pub fn with_background_rgb(mut self, rgb: [u8; 3]) -> Self {
        self.background_rgb = Some(rgb);
        self
    }

    /// Worker-thread concurrency (0 = single-threaded on the caller's thread).
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = Some(n);
        self
    }

    /// Capacity of the producer→sink bounded channel.
    pub fn with_buffer_size(mut self, n: usize) -> Self {
        self.buffer_size = Some(n);
        self
    }

    /// Soft memory budget in bytes. Drives strip-height selection in the
    /// streaming and map-reduce engines.
    pub fn with_memory_budget(mut self, bytes: u64) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }

    /// Decide what happens when the requested budget is too tight for the
    /// worst-case minimum aligned strip.
    pub fn with_budget_policy(mut self, policy: BudgetPolicy) -> Self {
        self.budget_policy = Some(policy);
        self
    }

    // --- extension hatch (Approach C) ---

    /// Attach a user-defined extension keyed by its runtime `TypeId`.
    ///
    /// libviprs itself reads zero extensions today; the hatch exists so
    /// third-party consumers (metrics exporters, custom audit logs, bespoke
    /// observer state) can thread context through the pipeline without a
    /// semver bump.
    pub fn with_extension<T: Send + Sync + 'static>(mut self, value: T) -> Self {
        self.extensions.insert(value);
        self
    }

    /// Borrow a previously-inserted extension by type.
    pub fn extension<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.extensions.get::<T>()
    }

    /// Borrow the full extension map — useful for custom observers that want
    /// to read user-supplied context without hard-coding the extension key.
    pub fn extensions(&self) -> &Extensions {
        &self.extensions
    }

    // --- terminal run methods ---

    /// Dispatch to the underlying engine and return the aggregate
    /// [`EngineResult`].
    ///
    /// Consumes the builder; the sink is dropped after the run. Use
    /// [`EngineBuilder::run_collect`] if you need the sink back afterwards
    /// (for example to call `MemorySink::tiles()`).
    pub fn run(self) -> Result<EngineResult, EngineError> {
        let (result, _sink) = self.run_collect()?;
        Ok(result)
    }

    /// Dispatch to the underlying engine and return both the result and the
    /// owned sink.
    pub fn run_collect(self) -> Result<(EngineResult, S), EngineError> {
        let EngineBuilder {
            source,
            plan,
            sink,
            engine_kind,
            observer,
            concurrency,
            buffer_size,
            background_rgb,
            blank_strategy,
            failure_policy,
            dedupe,
            resume,
            memory_budget_bytes,
            budget_policy,
            extensions: _extensions, // libviprs itself reads zero extensions
        } = self;

        // Build the EngineConfig once; the three engines accept slight
        // variations of it but all share the same underlying knob list.
        let mut engine_cfg = build_engine_config(
            concurrency,
            buffer_size,
            background_rgb,
            blank_strategy,
            failure_policy,
            dedupe,
        );

        let observer_ref: &dyn EngineObserver = match &observer {
            Some(arc) => arc.as_ref(),
            None => &NoopObserver,
        };

        let kind = resolve_engine_kind(engine_kind, &source);

        // Unified resume path: every engine kind now flows through the
        // same (skip, cp) + ResumeAwareSink pipeline. Verify is a
        // read-only sibling that delegates to dedicated verify helpers
        // (raster_verify or verify_from_strip_source) and never touches
        // the checkpoint.
        //
        // Invariants preserved:
        //   * Monolithic + Strip is rejected with IncompatibleSource.
        //   * policy.checkpoint_every() / policy.checkpoint_root() are
        //     threaded into engine_cfg before prepare_resume_state runs.
        //   * cp.flush() runs on every non-Verify success; Verify is
        //     read-only and has no checkpoint to persist.
        if let Some(policy) = resume {
            if policy.checkpoint_every() > 0 {
                engine_cfg = engine_cfg.with_checkpoint_every(policy.checkpoint_every());
            }
            if let Some(root) = policy.checkpoint_root() {
                engine_cfg = engine_cfg.with_checkpoint_root(root.to_path_buf());
            }

            if matches!(kind, EngineKind::Monolithic) && matches!(source, EngineSource::Strip(_)) {
                return Err(EngineError::IncompatibleSource {
                    kind: EngineKind::Monolithic,
                    reason: "Monolithic engine requires an in-memory Raster source",
                });
            }

            // Verify: read-only, no skip set, no checkpoint. Route by
            // engine kind to the matching verify helper.
            if matches!(policy.mode(), ResumeMode::Verify) {
                let result = match (kind, source) {
                    (EngineKind::Monolithic, EngineSource::Raster(raster)) => {
                        crate::verify::raster_verify(
                            raster,
                            &plan,
                            &sink,
                            &engine_cfg,
                            observer_ref,
                        )?
                    }
                    (EngineKind::Streaming, EngineSource::Raster(raster))
                    | (EngineKind::MapReduce, EngineSource::Raster(raster)) => {
                        let strip = RasterStripSource::new(raster);
                        crate::verify::verify_from_strip_source(
                            &strip,
                            &plan,
                            &sink,
                            &engine_cfg,
                            observer_ref,
                        )?
                    }
                    (EngineKind::Streaming, EngineSource::Strip(strip))
                    | (EngineKind::MapReduce, EngineSource::Strip(strip)) => {
                        crate::verify::verify_from_strip_source(
                            strip.as_ref(),
                            &plan,
                            &sink,
                            &engine_cfg,
                            observer_ref,
                        )?
                    }
                    (EngineKind::Monolithic, EngineSource::Strip(_)) => {
                        unreachable!("Monolithic + Strip rejected above")
                    }
                    (EngineKind::Auto, _) => {
                        unreachable!("Auto should have been resolved before match")
                    }
                };
                return Ok((result, sink));
            }

            // Overwrite / Resume: shared (skip, cp) setup + ResumeAwareSink.
            let (skip, cp) = prepare_resume_state(&sink, &plan, &engine_cfg, policy.mode())?;
            let wrapped = resume::ResumeAwareSink::new(&sink, &skip, cp.as_ref());

            let mut result = match (kind, source) {
                (EngineKind::Monolithic, EngineSource::Raster(raster)) => {
                    generate_pyramid_observed(raster, &plan, &wrapped, &engine_cfg, observer_ref)?
                }
                (EngineKind::Streaming, EngineSource::Raster(raster)) => {
                    let strip = RasterStripSource::new(raster);
                    let cfg =
                        build_streaming_config(engine_cfg, memory_budget_bytes, budget_policy);
                    generate_pyramid_streaming(&strip, &plan, &wrapped, &cfg, observer_ref)?
                }
                (EngineKind::Streaming, EngineSource::Strip(strip)) => {
                    let cfg =
                        build_streaming_config(engine_cfg, memory_budget_bytes, budget_policy);
                    generate_pyramid_streaming(strip.as_ref(), &plan, &wrapped, &cfg, observer_ref)?
                }
                (EngineKind::MapReduce, EngineSource::Raster(raster)) => {
                    let strip = RasterStripSource::new(raster);
                    let cfg = build_mapreduce_config(
                        memory_budget_bytes,
                        concurrency,
                        buffer_size,
                        background_rgb,
                        blank_strategy,
                    );
                    generate_pyramid_mapreduce(&strip, &plan, &wrapped, &cfg, observer_ref)?
                }
                (EngineKind::MapReduce, EngineSource::Strip(strip)) => {
                    let cfg = build_mapreduce_config(
                        memory_budget_bytes,
                        concurrency,
                        buffer_size,
                        background_rgb,
                        blank_strategy,
                    );
                    generate_pyramid_mapreduce(strip.as_ref(), &plan, &wrapped, &cfg, observer_ref)?
                }
                (EngineKind::Monolithic, EngineSource::Strip(_)) => {
                    unreachable!("Monolithic + Strip rejected above")
                }
                (EngineKind::Auto, _) => {
                    unreachable!("Auto should have been resolved before match")
                }
            };

            // Adjust tiles_produced down by the number of tiles the
            // resume wrapper short-circuited. This restores the
            // pre-unification contract that `tiles_produced` reports
            // "tiles actually written this run" rather than "tiles the
            // engine visited" — the difference matters for Resume runs
            // where most or all tiles skip the write path.
            let skipped = wrapped.skipped_count();
            if skipped > 0 {
                result.tiles_produced = result.tiles_produced.saturating_sub(skipped);
            }

            if let Some(cp) = cp.as_ref() {
                cp.flush().map_err(EngineError::ResumeFailed)?;
            }
            return Ok((result, sink));
        }

        let result = match (kind, source) {
            (EngineKind::Monolithic, EngineSource::Raster(raster)) => {
                generate_pyramid_observed(raster, &plan, &sink, &engine_cfg, observer_ref)?
            }
            (EngineKind::Monolithic, EngineSource::Strip(_)) => {
                return Err(EngineError::IncompatibleSource {
                    kind: EngineKind::Monolithic,
                    reason: "Monolithic engine requires an in-memory Raster source",
                });
            }
            (EngineKind::Streaming, EngineSource::Raster(raster)) => {
                let source = RasterStripSource::new(raster);
                let cfg = build_streaming_config(engine_cfg, memory_budget_bytes, budget_policy);
                generate_pyramid_streaming(&source, &plan, &sink, &cfg, observer_ref)?
            }
            (EngineKind::Streaming, EngineSource::Strip(source)) => {
                let cfg = build_streaming_config(engine_cfg, memory_budget_bytes, budget_policy);
                generate_pyramid_streaming(source.as_ref(), &plan, &sink, &cfg, observer_ref)?
            }
            (EngineKind::MapReduce, EngineSource::Raster(raster)) => {
                let source = RasterStripSource::new(raster);
                let cfg = build_mapreduce_config(
                    memory_budget_bytes,
                    concurrency,
                    buffer_size,
                    background_rgb,
                    blank_strategy,
                );
                generate_pyramid_mapreduce(&source, &plan, &sink, &cfg, observer_ref)?
            }
            (EngineKind::MapReduce, EngineSource::Strip(source)) => {
                let cfg = build_mapreduce_config(
                    memory_budget_bytes,
                    concurrency,
                    buffer_size,
                    background_rgb,
                    blank_strategy,
                );
                generate_pyramid_mapreduce(source.as_ref(), &plan, &sink, &cfg, observer_ref)?
            }
            (EngineKind::Auto, _) => {
                // `resolve_engine_kind` already flattened Auto to a concrete
                // kind; reaching this arm means we missed a case above.
                unreachable!("Auto should have been resolved before match")
            }
        };

        Ok((result, sink))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_engine_config(
    concurrency: Option<usize>,
    buffer_size: Option<usize>,
    background_rgb: Option<[u8; 3]>,
    blank_strategy: Option<BlankTileStrategy>,
    failure_policy: Option<FailurePolicy>,
    dedupe: Option<DedupeStrategy>,
) -> EngineConfig {
    let mut cfg = EngineConfig::default();
    if let Some(n) = concurrency {
        cfg = cfg.with_concurrency(n);
    }
    if let Some(n) = buffer_size {
        cfg = cfg.with_buffer_size(n);
    }
    if let Some(rgb) = background_rgb {
        cfg.background_rgb = rgb;
    }
    if let Some(bts) = blank_strategy {
        cfg = cfg.with_blank_tile_strategy(bts);
    }
    if let Some(fp) = failure_policy {
        cfg = cfg.with_failure_policy(fp);
    }
    if let Some(ds) = dedupe {
        cfg = cfg.with_dedupe_strategy(ds);
    }
    cfg
}

fn build_streaming_config(
    engine: EngineConfig,
    memory_budget_bytes: Option<u64>,
    budget_policy: Option<BudgetPolicy>,
) -> StreamingConfig {
    let defaults = StreamingConfig::default();
    StreamingConfig {
        engine,
        memory_budget_bytes: memory_budget_bytes.unwrap_or(defaults.memory_budget_bytes),
        budget_policy: budget_policy.unwrap_or(defaults.budget_policy),
    }
}

fn build_mapreduce_config(
    memory_budget_bytes: Option<u64>,
    concurrency: Option<usize>,
    buffer_size: Option<usize>,
    background_rgb: Option<[u8; 3]>,
    blank_strategy: Option<BlankTileStrategy>,
) -> MapReduceConfig {
    let mut cfg = MapReduceConfig::default();
    if let Some(b) = memory_budget_bytes {
        cfg.memory_budget_bytes = b;
    }
    if let Some(n) = concurrency {
        cfg.tile_concurrency = n;
    }
    if let Some(n) = buffer_size {
        cfg.buffer_size = n;
    }
    if let Some(rgb) = background_rgb {
        cfg.background_rgb = rgb;
    }
    if let Some(bts) = blank_strategy {
        cfg.blank_tile_strategy = bts;
    }
    cfg
}

/// Flatten [`EngineKind::Auto`] to a concrete variant based on the source.
///
/// * `Raster` → [`EngineKind::Monolithic`] (fastest when the raster fits in
///   memory; the free-function `generate_pyramid_auto` would already have
///   made this choice under its own budget heuristic).
/// * `Strip`  → [`EngineKind::Streaming`].
///
/// Non-`Auto` kinds pass through unchanged.
fn resolve_engine_kind(kind: EngineKind, source: &EngineSource<'_>) -> EngineKind {
    match (kind, source) {
        (EngineKind::Auto, EngineSource::Raster(_)) => EngineKind::Monolithic,
        (EngineKind::Auto, EngineSource::Strip(_)) => EngineKind::Streaming,
        (k, _) => k,
    }
}

// ---------------------------------------------------------------------------
// Resume support for streaming / map-reduce engines
// ---------------------------------------------------------------------------

/// Prepare the (skip_set, checkpoint_state) pair for a streaming or
/// map-reduce resume run. Matches the Monolithic engine's Overwrite /
/// Resume behaviour:
///
/// - [`ResumeMode::Overwrite`] — wipe the output root and start with an
///   empty skip set and a fresh checkpoint state.
/// - [`ResumeMode::Resume`] — load the existing checkpoint (if any) and
///   build the skip set from its `completed_tiles`. Surfaces
///   [`EngineError::PlanHashMismatch`] when the on-disk checkpoint was
///   produced from a different plan.
fn prepare_resume_state(
    sink: &dyn TileSink,
    plan: &PyramidPlan,
    config: &EngineConfig,
    mode: ResumeMode,
) -> Result<
    (
        std::collections::HashSet<crate::planner::TileCoord>,
        Option<crate::engine::CheckpointState>,
    ),
    EngineError,
> {
    match mode {
        ResumeMode::Overwrite => {
            if let Some(root) = crate::engine::resolve_checkpoint_root(config, sink) {
                crate::engine::wipe_directory(&root)
                    .map_err(|e| EngineError::ResumeFailed(crate::resume::ResumeError::from(e)))?;
            }
            let cp = crate::engine::cp_for_sink(sink, plan, config, Vec::new(), Vec::new());
            Ok((std::collections::HashSet::new(), cp))
        }
        ResumeMode::Resume => {
            let expected_hash = crate::resume::compute_plan_hash(plan);
            let (completed, levels) =
                if let Some(root) = crate::engine::resolve_checkpoint_root(config, sink) {
                    match crate::resume::JobCheckpoint::load(&root)? {
                        Some(meta) => {
                            if meta.plan_hash != expected_hash {
                                return Err(EngineError::PlanHashMismatch {
                                    expected: meta.plan_hash.clone(),
                                    got: expected_hash,
                                });
                            }
                            (meta.completed_tiles, meta.levels_completed)
                        }
                        None => (Vec::new(), Vec::new()),
                    }
                } else {
                    (Vec::new(), Vec::new())
                };
            let skip: std::collections::HashSet<crate::planner::TileCoord> =
                completed.iter().copied().collect();
            let cp = crate::engine::cp_for_sink(sink, plan, config, completed, levels);
            Ok((skip, cp))
        }
        ResumeMode::Verify => unreachable!("Verify is rejected above for non-Monolithic engines"),
    }
}

mod resume {
    use std::collections::HashSet;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::engine::{CheckpointState, EngineConfig};
    use crate::planner::TileCoord;
    use crate::sink::{SinkError, Tile, TileSink};

    /// Transparent wrapper that filters `write_tile` calls against a resume
    /// skip set and advances an optional [`CheckpointState`] on every tile
    /// that reaches the inner sink.
    ///
    /// Applied in front of the user-provided sink when the streaming or
    /// map-reduce engines are asked to run in resume mode. Keeps those
    /// engines themselves oblivious to resume semantics — the filtering
    /// happens at the one natural bottleneck (every tile ultimately lands
    /// in `write_tile`).
    pub(super) struct ResumeAwareSink<'a, S: TileSink + ?Sized> {
        pub(super) inner: &'a S,
        pub(super) skip: &'a HashSet<TileCoord>,
        pub(super) cp: Option<&'a CheckpointState>,
        /// Running count of skipped writes, used after `.run()` to adjust
        /// [`crate::engine::EngineResult::tiles_produced`] down so that
        /// callers see "tiles actually written this run" rather than
        /// "tiles the engine visited".
        pub(super) skipped: AtomicU64,
    }

    impl<'a, S: TileSink + ?Sized> ResumeAwareSink<'a, S> {
        pub(super) fn new(
            inner: &'a S,
            skip: &'a HashSet<TileCoord>,
            cp: Option<&'a CheckpointState>,
        ) -> Self {
            Self {
                inner,
                skip,
                cp,
                skipped: AtomicU64::new(0),
            }
        }

        pub(super) fn skipped_count(&self) -> u64 {
            self.skipped.load(Ordering::Relaxed)
        }
    }

    impl<'a, S: TileSink + ?Sized> TileSink for ResumeAwareSink<'a, S> {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            if self.skip.contains(&tile.coord) {
                self.skipped.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            self.inner.write_tile(tile)?;
            if let Some(cp) = self.cp {
                cp.mark_tile_completed(tile.coord)
                    .map_err(|e| SinkError::Other(format!("checkpoint: {e}")))?;
            }
            Ok(())
        }

        fn finish(&self) -> Result<(), SinkError> {
            self.inner.finish()
        }

        fn record_engine_config(&self, config: &EngineConfig) {
            self.inner.record_engine_config(config)
        }

        fn sink_retry_count(&self) -> u64 {
            self.inner.sink_retry_count()
        }

        fn sink_skipped_due_to_failure(&self) -> u64 {
            self.inner.sink_skipped_due_to_failure()
        }

        fn note_sink_skipped(&self) {
            self.inner.note_sink_skipped()
        }

        fn checkpoint_root(&self) -> Option<&Path> {
            self.inner.checkpoint_root()
        }

        fn init_level_count(&self, levels: usize) {
            self.inner.init_level_count(levels)
        }
    }

    // Silence "field never read" lints on the AtomicU64 placeholder if
    // future refactors drop these fields.
    #[allow(dead_code)]
    fn _unused_marker(_: AtomicU64) {}
}
