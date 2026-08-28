//! Unified fluent entry point for pyramid generation.
//!
//! [`EngineBuilder`] is the single typed entry point for pyramid generation.
//! It folds what used to be a family of monolithic, streaming, and MapReduce
//! free functions behind one fluent builder, so callers pick the engine through
//! [`with_engine`](EngineBuilder::with_engine) (see [`EngineKind`]) rather than
//! by choosing a function name. Call
//! [`EngineBuilder::new(source, plan, sink)`](EngineBuilder::new), chain any
//! combination of `with_*` setters, then [`EngineBuilder::run`] or
//! [`EngineBuilder::run_collect`].
//!
//! Routing to the underlying engine is driven by two inputs:
//!
//! 1. The [`EngineKind`] set via [`EngineBuilder::with_engine`]
//!    (default: [`EngineKind::Auto`]).
//! 2. Whether the source is an in-memory [`Raster`] or a
//!    [`StripSource`].
//!
//! [`EngineKind::Monolithic`] refuses to run against a strip source and
//! surfaces [`EngineError::IncompatibleSource`] instead of silently pulling
//! the whole source into memory.
//!
//! # Per-tile operation: current shape and roadmap
//!
//! Every engine kind applies exactly one per-level transform today: the
//! 2×2 box-filter downscale ([`resize::downscale_half`](crate::resize::downscale_half))
//! followed by tile extraction. The full-level cascade that applies it lives
//! in one place, the crate-internal `level_walk::walk_levels_down`, whose
//! `step` closure is the tile-operation parameter (issue #138). The live
//! monolithic run, both read-only verify walks, and the streaming engines'
//! monolithic flush (shared by the sequential and MapReduce drivers via
//! `streaming::flush_monolithic_levels` / `flush_unpaired_accumulators`) all
//! delegate to it; the per-strip halving inside the streaming strip loops
//! and `streaming::propagate_down` is a strip transform and intentionally
//! stays separate.
//!
//! **Remaining roadmap for a second per-tile operation** (rotate, sharpen,
//! colour transform, …):
//!
//! 1. Name the operation as a `TileOp` trait (`fn apply(&self, level:
//!    &Raster) -> Result<Raster, RasterError>`) with a `DownscaleHalf` unit
//!    implementation wrapping today's behaviour, and pass it through the
//!    existing `step` hook of `walk_levels_down` (plus the strip-transform
//!    sites, which must apply the same operation for parity).
//! 2. Surface the operation on this builder via a `with_tile_op` setter,
//!    carried alongside the existing config knobs.

use std::sync::Arc;

use crate::dedupe::DedupeStrategy;
use crate::engine::{
    BlankTileStrategy, EngineConfig, EngineError, EngineResult, generate_pyramid_observed,
};
use crate::extensions::Extensions;
use crate::mapreduce_hot_cache::generate_pyramid_mapreduce_hot_cache;
use crate::observe::{EngineObserver, NoopObserver};
use crate::planner::PyramidPlan;
use crate::raster::Raster;
use crate::resume::{ResumeMode, ResumePolicy};
use crate::retry::{FailurePolicy, RetryPolicy, RetryingSink};
use crate::sink::TileSink;
use crate::streaming::{
    BudgetPolicy, RasterStripSource, StreamingConfig, StripSource, generate_pyramid_streaming,
};
use crate::streaming_mapreduce::{
    LocalWorkExecutor, MapReduceConfig, WorkExecutor, generate_pyramid_mapreduce,
};

/// Checks that a [`PyramidPlan`] is internally well-formed and describes the
/// source it is about to be run against.
///
/// A `PyramidPlan` obtained from [`PyramidPlanner::plan`](crate::PyramidPlanner::plan)
/// always upholds these invariants, but the plan and the source are supplied
/// independently to the engine, so a mismatched pair (or a plan that was
/// mutated after construction) could otherwise reach code that trusts
/// `plan.image_width/height` and a non-empty `plan.levels`. Enforcing the
/// invariants here turns what were out-of-bounds slice copies and an
/// arithmetic underflow into typed, recoverable errors (issue #132).
fn validate_plan_for_source(
    plan: &PyramidPlan,
    source_width: u32,
    source_height: u32,
) -> Result<(), EngineError> {
    if plan.levels.is_empty() {
        return Err(EngineError::InvalidPlan {
            reason: "plan has no levels",
        });
    }
    if plan.image_width != source_width || plan.image_height != source_height {
        return Err(EngineError::PlanSourceMismatch {
            plan_width: plan.image_width,
            plan_height: plan.image_height,
            source_width,
            source_height,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// EngineKind
// ---------------------------------------------------------------------------

/// Which underlying engine implementation [`EngineBuilder::run`] should
/// dispatch to.
///
/// `#[non_exhaustive]` so future engine variants (e.g. `Gpu`, `Distributed`,
/// `Remote`) can be added as minor-version additions.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-parallel).
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
    /// MapReduce engine with an in-memory hot cache: rendering is identical
    /// to [`MapReduce`](Self::MapReduce), but every produced tile is retained
    /// in RAM and the caller's sink receives the whole pyramid as one batched
    /// flush at the end of the run, in canonical `(level, row, col)` order,
    /// followed by a single `finish()`.
    ///
    /// A local-only memory-vs-throughput tradeoff (issue #67): the memory
    /// budget still bounds the MAP/REDUCE working set (including the
    /// pre-flight [`EngineError::BudgetExceeded`] rejection), while the cache
    /// itself holds the full pyramid's raster bytes until the flush. Because
    /// of that deliberate residency, [`Auto`](Self::Auto) never selects this
    /// engine; it is explicit opt-in only. Accepts either source kind.
    /// Byte-identical output to the streaming and MapReduce engines.
    MapReduceHotCache,
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
///
/// **See also:** [interactive example](https://libviprs.org/cli/#pyramid)
/// for an end-to-end runnable program built from these setters.
pub struct EngineBuilder<'a, S: TileSink> {
    source: EngineSource<'a>,
    plan: PyramidPlan,
    sink: S,

    engine_kind: EngineKind,
    observer: Option<Arc<dyn EngineObserver>>,
    executor: Option<Arc<dyn WorkExecutor>>,

    // EngineConfig knobs
    concurrency: Option<usize>,
    buffer_size: Option<usize>,
    background_rgb: Option<[u8; 3]>,
    blank_strategy: Option<BlankTileStrategy>,
    skip_blanks: Option<bool>,
    failure_policy: Option<FailurePolicy>,
    dedupe: Option<DedupeStrategy>,

    // Resume
    resume: Option<ResumePolicy>,

    // Cooperative cancellation
    cancel: Option<crate::cancel::CancelToken>,

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
            .field("has_executor", &self.executor.is_some())
            .field("concurrency", &self.concurrency)
            .field("buffer_size", &self.buffer_size)
            .field("background_rgb", &self.background_rgb)
            .field("blank_strategy", &self.blank_strategy)
            .field("skip_blanks", &self.skip_blanks)
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
            executor: None,
            concurrency: None,
            buffer_size: None,
            background_rgb: None,
            blank_strategy: None,
            skip_blanks: None,
            failure_policy: None,
            dedupe: None,
            resume: None,
            cancel: None,
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

    /// Attach several observers at once; every [`EngineEvent`](crate::EngineEvent)
    /// fans out to each of them, in the order they appear in the vector
    /// (issue #67).
    ///
    /// Composes the observers through a [`FanOutObserver`](crate::FanOutObserver),
    /// so delivery is synchronous, on the thread that produced the event,
    /// exactly as with a single observer; the extension hatch
    /// ([`EngineObserver::on_extensions`])
    /// is forwarded to each of them too. Like every observer setter, this
    /// fills the builder's one observer slot: it replaces anything a prior
    /// [`with_observer`](Self::with_observer) /
    /// [`with_observer_arc`](Self::with_observer_arc) / `with_observers`
    /// call attached, and a later call replaces it in turn.
    /// [`with_observer`](Self::with_observer) remains the single-observer
    /// shorthand.
    pub fn with_observers(mut self, observers: Vec<Arc<dyn EngineObserver>>) -> Self {
        self.observer = Some(Arc::new(crate::observe::FanOutObserver::new(observers)));
        self
    }

    /// Select which engine implementation [`EngineBuilder::run`] will
    /// dispatch to. Defaults to [`EngineKind::Auto`].
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-parallel).
    pub fn with_engine(mut self, kind: EngineKind) -> Self {
        self.engine_kind = kind;
        self
    }

    /// Install a [`WorkExecutor`] at the MapReduce MAP-phase strip-dispatch
    /// seam (issue #67), substituting for the built-in in-process rendering.
    ///
    /// Consulted by [`EngineKind::MapReduce`] and
    /// [`EngineKind::MapReduceHotCache`] (the engines with a MAP phase);
    /// the monolithic and sequential streaming engines have no strip-dispatch
    /// seam and ignore it. Defaults to
    /// [`LocalWorkExecutor`],
    /// which preserves the engine's historical behaviour exactly. See
    /// [`WorkExecutor`] for the dispatch-order and byte-parity contract a
    /// custom executor must uphold.
    pub fn with_executor(mut self, executor: Arc<dyn WorkExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Apply every field of an existing [`EngineConfig`] in one call.
    ///
    /// Convenience for callers that already have a fully-constructed
    /// [`EngineConfig`] — typically because they're migrating from the
    /// old `generate_pyramid_observed(source, plan, sink, &config, obs)`
    /// free function.
    ///
    /// # Precedence
    ///
    /// `with_config` **fills only the fields no earlier setter has set** — an
    /// explicit `.with_*` setter always wins over the config (issue #297).
    /// A field left untouched on the builder takes its value from the config;
    /// a field an earlier setter already set keeps that value, so
    /// `.with_dedupe(..).with_config(cfg)` retains the dedupe strategy and
    /// `.with_cancel(tok).with_config(cfg)` retains the token even when the
    /// config carries none. This is the least-surprising rule and removes the
    /// former order-sensitivity trap, where a coarse `with_config` silently
    /// undid a fine-grained setter (dropping a cancellation token or dedupe
    /// strategy with no signal). Setters applied *after* `with_config` still
    /// overwrite it, so "explicit setter wins" holds in both directions.
    ///
    /// Also threads the config's `checkpoint_every` and `checkpoint_root`
    /// into an *existing* [`ResumePolicy`] — but only when one has already
    /// been attached (via [`EngineBuilder::with_resume`]). It never fabricates
    /// a policy: [`ResumeMode::Overwrite`] is destructive (it wipes the sink
    /// dir), so carrying a checkpoint cadence or root on the config must not
    /// silently enable it. Callers that want a resumable run choose the mode
    /// explicitly; without one, the checkpoint knobs are simply inert.
    pub fn with_config(mut self, config: EngineConfig) -> Self {
        // Fill-if-unset (issue #297): apply each config field only where the
        // builder has no explicit value yet, so an earlier fine-grained `.with_*`
        // setter always survives a later `with_config`. Every builder knob is an
        // `Option<T>`, so `.or(..)` keeps a `Some` a setter already put there and
        // otherwise adopts the config's value. A setter applied *after*
        // `with_config` overwrites the slot with a fresh `Some`, so "explicit
        // setter wins" holds in both directions.
        self.concurrency = self.concurrency.or(Some(config.concurrency));
        self.buffer_size = self.buffer_size.or(Some(config.buffer_size));
        self.background_rgb = self.background_rgb.or(Some(config.background_rgb));
        self.blank_strategy = self
            .blank_strategy
            .take()
            .or(Some(config.blank_tile_strategy));
        self.skip_blanks = self.skip_blanks.or(Some(config.skip_blanks));
        self.failure_policy = self.failure_policy.take().or(Some(config.failure_policy));
        // `dedupe_strategy` / `cancel` are already `Option`s on the config: a
        // `None` there means the config carries no opinion, so it must not clear
        // an earlier `.with_dedupe(..)` / `.with_cancel(..)`. Filling only when
        // the builder slot is still empty preserves the token / strategy whose
        // silent loss (a hung job / a bloated output) was the reported trap.
        self.dedupe = self.dedupe.take().or(config.dedupe_strategy);
        self.cancel = self.cancel.take().or(config.cancel);
        // Carry the checkpoint knobs into an EXPLICITLY-chosen ResumePolicy
        // only, so migrations from `generate_pyramid_resumable(.., &cfg, mode)`
        // don't silently lose the cadence / root that used to live on the
        // config. If no policy was attached we leave `resume` as `None`
        // rather than conjuring a destructive Overwrite (issue #123).
        //
        // A non-zero `config.checkpoint_every` is an *explicit* caller choice
        // and must win over the policy's implicit default — `ResumePolicy::resume()`
        // seeds a coarse fallback cadence (`DEFAULT_RESUME_CHECKPOINT_EVERY`), and
        // gating the override on `policy.checkpoint_every() == 0` used to let that
        // fallback silently swallow the config value. That dropped the caller's
        // cadence on the floor, so a crash between the (rare) fallback flushes
        // left no checkpoint and `--resume` re-rendered everything. Overriding
        // whenever the config sets a value keeps the documented "config threads
        // into the policy" contract and bounds resume rework to one interval.
        if let Some(mut policy) = self.resume.take() {
            if config.checkpoint_every != 0 {
                policy = policy.with_checkpoint_every(config.checkpoint_every);
            }
            if policy.checkpoint_root().is_none()
                && let Some(root) = config.checkpoint_root
            {
                policy = policy.with_checkpoint_root(root);
            }
            self.resume = Some(policy);
        }
        self
    }

    /// Set the [`FailurePolicy`] the engine consults when a tile write
    /// terminally fails. Overrides any earlier [`EngineBuilder::with_retry`]
    /// call.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-retry-max).
    pub fn with_failure_policy(mut self, policy: FailurePolicy) -> Self {
        self.failure_policy = Some(policy);
        self
    }

    /// Shorthand for `with_failure_policy(FailurePolicy::RetryThenFail(policy))`.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-retry-max).
    pub fn with_retry(self, policy: RetryPolicy) -> Self {
        self.with_failure_policy(FailurePolicy::RetryThenFail(policy))
    }

    /// Control resume / verify behaviour. Only the engine's resumable path
    /// consults this; see [`ResumePolicy`] for the mode selector and the
    /// checkpoint-persistence knobs.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-resume),
    /// plus the related [`--overwrite`](https://libviprs.org/cli/#flag-overwrite)
    /// and [`--verify`](https://libviprs.org/cli/#flag-verify) flags.
    pub fn with_resume(mut self, policy: ResumePolicy) -> Self {
        self.resume = Some(policy);
        self
    }

    /// Attach a [`CancelToken`](crate::cancel::CancelToken) so the run can be
    /// cooperatively cancelled from another thread. Every engine kind polls
    /// the token at level / tile / strip boundaries, and the retry backoff
    /// sleeps in short slices so it can be interrupted; a cancelled run stops
    /// and returns [`EngineError::Cancelled`]. See the [`cancel`](crate::cancel)
    /// module for the full polling contract.
    pub fn with_cancel(mut self, token: crate::cancel::CancelToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Select a content-addressed deduplication strategy. See
    /// [`DedupeStrategy`] for variants.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-dedupe-blanks).
    pub fn with_dedupe(mut self, strategy: DedupeStrategy) -> Self {
        self.dedupe = Some(strategy);
        self
    }

    /// Control how blank (uniform-colour) tiles are handled.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-skip-blank).
    pub fn with_blank_strategy(mut self, strategy: BlankTileStrategy) -> Self {
        self.blank_strategy = Some(strategy);
        self
    }

    /// Drop blank (uniform-colour) tiles from the run entirely rather than
    /// writing them, so the output carries strictly fewer files. This is the
    /// builder mirror of [`EngineConfig::skip_blanks`]; skipping takes
    /// precedence over the active [`BlankTileStrategy`].
    ///
    /// Honoured by the monolithic engine only (the streaming and map-reduce
    /// engines currently ignore it; parity is deferred).
    pub fn with_skip_blanks(mut self, skip: bool) -> Self {
        self.skip_blanks = Some(skip);
        self
    }

    /// Set the background RGB used to pad edge tiles.
    pub fn with_background_rgb(mut self, rgb: [u8; 3]) -> Self {
        self.background_rgb = Some(rgb);
        self
    }

    /// Worker-thread concurrency (0 = single-threaded on the caller's thread).
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-concurrency).
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = Some(n);
        self
    }

    /// Capacity of the producer→sink bounded channel.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-buffer-size).
    pub fn with_buffer_size(mut self, n: usize) -> Self {
        self.buffer_size = Some(n);
        self
    }

    /// Soft memory budget in bytes. Drives strip-height selection in the
    /// streaming and map-reduce engines.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-memory-budget).
    pub fn with_memory_budget(mut self, bytes: u64) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }

    /// Decide what happens when the requested budget is too tight for the
    /// worst-case minimum aligned strip.
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-memory-budget).
    pub fn with_budget_policy(mut self, policy: BudgetPolicy) -> Self {
        self.budget_policy = Some(policy);
        self
    }

    // --- extension hatch (Approach C) ---

    /// Attach a user-defined extension keyed by its runtime `TypeId`.
    ///
    /// The hatch lets third-party consumers (metrics exporters, custom audit
    /// logs, bespoke observer state) thread context through the pipeline
    /// without a semver bump. On [`run`](EngineBuilder::run) /
    /// [`run_collect`](EngineBuilder::run_collect) the full map is delivered to
    /// the attached observer via
    /// [`EngineObserver::on_extensions`]
    /// once, before any tile is emitted, so a custom observer can read the
    /// values it needs for the run.
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

    /// Generate a pyramid for a rectangular sub-region `(left, top, width,
    /// height)` of the source — the builder-ergonomic equivalent of
    /// [`generate_pyramid_region`](crate::generate_pyramid_region).
    ///
    /// This is crop-then-pyramid: the source is cropped to the region and the
    /// plan is run against the crop, so the [`PyramidPlan`] must be sized to the
    /// region, not the whole source (a dimension mismatch is
    /// [`EngineError::PlanSourceMismatch`]). The builder's config knobs
    /// (concurrency, blank strategy, skip-blanks, background, dedupe, failure
    /// policy, cancellation) are threaded through; resume, streaming, and
    /// observers are not part of the region path, mirroring the free function.
    ///
    /// # Errors
    ///
    /// Requires an in-memory raster source; a strip source yields
    /// [`EngineError::IncompatibleSource`] (a strip cannot be cropped after the
    /// fact).
    pub fn run_region(
        self,
        left: u32,
        top: u32,
        width: u32,
        height: u32,
    ) -> Result<EngineResult, EngineError> {
        let EngineBuilder {
            source,
            plan,
            sink,
            engine_kind,
            concurrency,
            buffer_size,
            background_rgb,
            blank_strategy,
            skip_blanks,
            failure_policy,
            dedupe,
            cancel,
            ..
        } = self;

        let raster = match source {
            EngineSource::Raster(r) => r,
            EngineSource::Strip(_) => {
                return Err(EngineError::IncompatibleSource {
                    kind: engine_kind,
                    reason: "region generation requires an in-memory raster source, not a strip",
                });
            }
        };

        let mut config = build_engine_config(
            concurrency,
            buffer_size,
            background_rgb,
            blank_strategy,
            skip_blanks,
            failure_policy,
            dedupe,
        );
        config.cancel = cancel;

        crate::engine::generate_pyramid_region(
            raster, &plan, &sink, &config, left, top, width, height,
        )
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
            executor,
            concurrency,
            buffer_size,
            background_rgb,
            blank_strategy,
            skip_blanks,
            failure_policy,
            dedupe,
            resume,
            cancel,
            memory_budget_bytes,
            budget_policy,
            extensions,
        } = self;

        // Build the EngineConfig once; the three engines accept slight
        // variations of it but all share the same underlying knob list.
        let mut engine_cfg = build_engine_config(
            concurrency,
            buffer_size,
            background_rgb,
            blank_strategy,
            skip_blanks,
            failure_policy,
            dedupe,
        );
        // Thread the cooperative-cancellation token onto the config so every
        // engine driver (and the streaming config that embeds it) polls it.
        engine_cfg.cancel = cancel.clone();

        let observer_ref: &dyn EngineObserver = match &observer {
            Some(arc) => arc.as_ref(),
            None => &NoopObserver,
        };

        // The MAP-phase strip-dispatch seam (issue #67). Absent an installed
        // executor, the local in-process renderer preserves the engines'
        // historical behaviour exactly.
        let executor_ref: &dyn WorkExecutor = match &executor {
            Some(arc) => arc.as_ref(),
            None => &LocalWorkExecutor,
        };

        // Deliver the extension hatch to its one reader. The map used to be
        // destructured and discarded, so `with_extension` was decorative:
        // nothing in the pipeline ever read it. Handing it to the observer
        // once, before any tile is emitted, is the wiring that makes the hatch
        // functional — a custom `EngineObserver` clones out the handles it
        // needs (metrics recorders, tracing spans, custom config) for the rest
        // of the run (issue #138). `NoopObserver` and observers that don't
        // override `on_extensions` ignore it at zero cost.
        observer_ref.on_extensions(&extensions);

        // #119: route writes through `RetryingSink` when the configured
        // failure policy carries a `RetryPolicy`. Previously the engine only
        // inspected the `FailurePolicy` *variant* and never read the embedded
        // policy, so `EngineBuilder::with_retry` / `--on-failure retry=N,D`
        // performed zero retries. Wrapping here engages the retry loop inside
        // `write_tile` for every engine kind (monolithic / streaming /
        // map-reduce) on both the plain and resume paths; the engine then
        // sees only the terminal outcome, so its existing `RetryThenSkip`
        // accounting kicks in only after retries are exhausted.
        //
        // The wrapper borrows the sink, so the original `sink` is still
        // returned to the caller (for `run_collect` inspection) and continues
        // to record every tile that lands.
        //
        // A sink the caller already wrapped in `RetryingSink` reports
        // `applies_retry_policy() == true`; we skip re-wrapping it so retries
        // and skip-accounting are not double-counted.
        let retrying: Option<RetryingSink<&S>> = match &engine_cfg.failure_policy {
            FailurePolicy::RetryThenFail(p) | FailurePolicy::RetryThenSkip(p)
                if !sink.applies_retry_policy() =>
            {
                // Share the run's cancel token with the retry loop so an
                // in-flight backoff can be interrupted (#133).
                Some(RetryingSink::new(&sink, p.clone()).with_cancel(engine_cfg.cancel.clone()))
            }
            _ => None,
        };
        // The concrete sink the engine drives its writes through.
        let engine_sink: &dyn TileSink = match &retrying {
            Some(r) => r,
            None => &sink,
        };

        let kind = resolve_engine_kind(engine_kind, &source, &plan, memory_budget_bytes);

        // Validate the plan against the source before any engine driver runs.
        // Both the monolithic and streaming paths trust `plan.image_width/height`
        // and `plan.levels` (canvas embedding, tile extraction, `levels.len() - 1`).
        // A plan built for different dimensions than the supplied source, or a
        // structurally invalid plan, is rejected here with a typed error rather
        // than being allowed to reach an out-of-bounds slice copy or an
        // arithmetic underflow deeper in the pipeline (issue #132).
        let (source_width, source_height) = match &source {
            EngineSource::Raster(raster) => (raster.width(), raster.height()),
            EngineSource::Strip(strip) => (strip.width(), strip.height()),
        };
        validate_plan_for_source(&plan, source_width, source_height)?;

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
        // Resume invariant (issue #290): thread the checkpoint cadence / root
        // onto the config BEFORE any dispatch, so `prepare_resume_state`,
        // `cp_for_sink`, the Verify helpers, and every engine driver all observe
        // them. A `None` resume leaves the config untouched, so the plain path
        // below is unaffected.
        if let Some(policy) = &resume {
            if policy.checkpoint_every() > 0 {
                engine_cfg = engine_cfg.with_checkpoint_every(policy.checkpoint_every());
            }
            if let Some(root) = policy.checkpoint_root() {
                engine_cfg = engine_cfg.with_checkpoint_root(root.to_path_buf());
            }
        }

        // The single (EngineKind × EngineSource) render dispatch, shared by the
        // plain and resume paths (issue #290). Built once — after the checkpoint
        // knobs are threaded and before either path runs — so the ~10-arm match
        // and the eight verbatim `build_mapreduce_config` calls live in ONE place
        // (`RenderDispatch::run`) instead of a copy per path. A new engine kind or
        // source is now a single-site edit.
        let dispatch = RenderDispatch {
            plan: &plan,
            engine_cfg: &engine_cfg,
            observer: observer_ref,
            executor: executor_ref,
            memory_budget_bytes,
            budget_policy,
            concurrency,
            buffer_size,
            background_rgb,
            blank_strategy,
            cancel: cancel.clone(),
        };

        if let Some(policy) = &resume {
            if matches!(kind, EngineKind::Monolithic) && matches!(source, EngineSource::Strip(_)) {
                return Err(EngineError::IncompatibleSource {
                    kind: EngineKind::Monolithic,
                    reason: "Monolithic engine requires an in-memory Raster source",
                });
            }

            // Verify: read-only, no skip set, no checkpoint. Routed by engine
            // kind to the matching verify helper through the single
            // `dispatch_verify` site (issue #290).
            if matches!(policy.mode(), ResumeMode::Verify) {
                let result =
                    dispatch_verify(kind, source, &plan, &sink, &engine_cfg, observer_ref)?;
                return Ok((result, sink));
            }

            // Issue #272 (P0, data corruption) — resolved by sink-side seeding.
            // A resumed dedupe/checksum run used to be refused up front (the
            // #450 stopgap `ResumeUnsupportedWith`) because `ResumeAwareSink`
            // short-circuits already-completed coordinates, leaving the sink's
            // `manifest_refs` / `tile_digests` / `DedupeIndex` empty for every
            // pre-crash tile so `finish()` overwrote `manifest.json` with an
            // incomplete view. That guard is gone: `ResumeAwareSink` now calls
            // `TileSink::seed_completed_tile` for each skipped coordinate, which
            // reconstructs exactly the state an uninterrupted run holds (the
            // dedupe layout is a deterministic function of content + coordinates
            // since #275, and tile production is deterministic), so a resumed
            // `finish()` reproduces a byte-identical manifest + dedupe layout.

            // Refuse a mismatched Resume BEFORE anything touches the output
            // directory. `prepare_resume_state` re-checks the checkpoint under
            // the run lock (that check stays authoritative for the race where
            // another job swaps the checkpoint in between), but by that point
            // `RunLock::acquire` has already created its lock file inside the
            // directory. A refused resume must leave the directory exactly as
            // it found it: no tile output, and no bookkeeping either. This
            // preflight reads the same atomically-renamed checkpoint file the
            // locked check reads, so it never sees a torn header.
            if matches!(policy.mode(), ResumeMode::Resume)
                && let Some(root) = crate::engine::resolve_checkpoint_root(&engine_cfg, &sink)
                && let Some(meta) =
                    crate::resume::JobCheckpoint::load(&root).map_err(EngineError::ResumeFailed)?
                && let Err(current) =
                    crate::resume::verify_checkpoint_contract(&meta, &plan, &engine_cfg, &sink)
            {
                return Err(EngineError::PlanHashMismatch {
                    expected: current,
                    actual: meta.plan_hash,
                });
            }

            // Overwrite / Resume: take the advisory run lock(s) BEFORE any work
            // touches the directory, and hold them for the whole run. This is
            // the engine-side half of issue #126: unique temp filenames alone
            // stop torn renames, but two live jobs can still (a) clobber each
            // other's `completed_tiles` when their periodic checkpoint flushes
            // race, and (b) wipe each other's in-flight output if one runs
            // Overwrite. `RunLock::acquire` is non-blocking, so a second job is
            // refused with `ResumeError::Locked` (surfaced as
            // `EngineError::ResumeFailed`) rather than allowed to race. The
            // guards are dropped when `_run_locks` leaves this block, i.e. after
            // the final checkpoint flush below, and they sit ahead of
            // `prepare_resume_state`, so the Overwrite wipe in that helper only
            // ever runs while every directory it targets is held.
            //
            // A run mutates up to TWO distinct directories, so a single lock is
            // not enough (issues #362/#364/#365/#366):
            //   * the resolved checkpoint root — where the segment appends and
            //     header renames land. `resolve_checkpoint_root` prefers an
            //     explicit `checkpoint_root` over the sink's own dir; guarding it
            //     closes the #276 checkpoint-flush race.
            //   * the sink's own output dir — the target of the Overwrite wipe
            //     (`prepare_resume_state` wipes `sink.checkpoint_root()`) and of
            //     every tile write. Guarding it closes issue #126 hazard (b).
            // When an explicit `checkpoint_root` differs from the sink dir these
            // are two different paths; locking only one leaves the other exposed
            // (guarding just the checkpoint root reopens the #126 output-wipe
            // hazard, guarding just the sink dir reopens #276). We therefore lock
            // the union of the two and hold every guard for the run, so no
            // concurrent job can Overwrite-wipe or append to a directory another
            // run is using. A purely in-memory sink with no checkpoint root
            // contributes no directory and takes no lock — nothing on disk to
            // clobber.
            //
            // The dirs are locked in a deterministic (sorted, de-duplicated)
            // order so two jobs contending the same pair can never each grab one
            // directory and then refuse the other: one job wins both locks, the
            // other is cleanly refused. Acquisition stays non-blocking, so the
            // fixed order introduces no deadlock or blocking wait.
            //
            // De-duplication keys on the CANONICAL path, not the raw `PathBuf`.
            // When the explicit `checkpoint_root` and the sink dir name the SAME
            // physical directory through different spellings (`out` vs `./out`,
            // relative vs absolute, `a/../out`, a symlink alias), a raw-path
            // dedup keeps both entries and `RunLock::acquire` is then called
            // twice on the one `.libviprs-job.lock` file: the second `try_lock`
            // returns `WouldBlock` and the run refuses ITSELF with
            // `ResumeError::Locked`, blaming a nonexistent concurrent job on a
            // perfectly valid single-directory config. Keying on
            // `canonicalize(dir)` collapses the aliases to one lock. The dirs
            // are materialised (the sink dir exists and `acquire`'s
            // `create_dir_all` backs the checkpoint root), so `canonicalize`
            // resolves; `unwrap_or_else(|_| dir.clone())` degrades to the raw
            // path on exotic filesystems where it cannot. We still acquire and
            // report the lock on the original (un-canonicalized) `PathBuf` so
            // the surfaced `Locked { path }` matches the directory as spelled.
            let mut lock_dirs: Vec<std::path::PathBuf> = Vec::new();
            if let Some(dir) = crate::engine::resolve_checkpoint_root(&engine_cfg, &sink) {
                lock_dirs.push(dir);
            }
            if let Some(dir) = sink.checkpoint_root() {
                lock_dirs.push(dir.to_path_buf());
            }
            let canonical_key = |dir: &std::path::PathBuf| -> std::path::PathBuf {
                std::fs::canonicalize(dir).unwrap_or_else(|_| dir.clone())
            };
            lock_dirs.sort_by_cached_key(&canonical_key);
            lock_dirs.dedup_by_key(|dir| canonical_key(dir));
            let _run_locks = lock_dirs
                .iter()
                .map(|dir| crate::resume::RunLock::acquire(dir).map_err(EngineError::ResumeFailed))
                .collect::<Result<Vec<_>, _>>()?;

            // Overwrite / Resume: shared (skip, cp) setup + ResumeAwareSink.
            let (skip, cp) = prepare_resume_state(&sink, &plan, &engine_cfg, policy.mode())?;
            // Route resume writes through the retry wrapper (when configured)
            // so a transient failure is retried before the resume checkpoint
            // records the tile as complete.
            let wrapped = resume::ResumeAwareSink::new(engine_sink, &skip, cp.as_ref());

            // Capture the run outcome rather than propagating it with `?`
            // straight away: the checkpoint must be flushed on *both* the
            // success and the error path (see `flush_checkpoint_dual_path`).
            // The render dispatch is the SAME shared site the plain path uses;
            // only the sink differs (the `ResumeAwareSink` wrapper here).
            let run_result = dispatch.run(kind, source, &wrapped);

            // Resume invariants, each lifted into a named helper so the rule
            // lives in code rather than a prose comment (issue #290):
            //   * every fully-written level is promoted to `levels_completed`,
            //   * the checkpoint is flushed on both the success and error path,
            //   * `tiles_produced` / `bytes_written` exclude short-circuited skips.
            promote_completed_levels(cp.as_ref(), &plan);
            flush_checkpoint_dual_path(cp.as_ref(), &run_result)?;
            let mut result = run_result?;
            adjust_result_for_resume_skips(&mut result, &wrapped);
            return Ok((result, sink));
        }

        let result = dispatch.run(kind, source, engine_sink)?;
        Ok((result, sink))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_engine_config(
    concurrency: Option<usize>,
    buffer_size: Option<usize>,
    background_rgb: Option<[u8; 3]>,
    blank_strategy: Option<BlankTileStrategy>,
    skip_blanks: Option<bool>,
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
    if let Some(skip) = skip_blanks {
        cfg = cfg.skip_blanks(skip);
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

#[allow(clippy::too_many_arguments)]
fn build_mapreduce_config(
    memory_budget_bytes: Option<u64>,
    concurrency: Option<usize>,
    buffer_size: Option<usize>,
    background_rgb: Option<[u8; 3]>,
    blank_strategy: Option<BlankTileStrategy>,
    failure_policy: &FailurePolicy,
    cancel: Option<crate::cancel::CancelToken>,
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
    cfg.failure_policy = failure_policy.clone();
    cfg.cancel = cancel;
    cfg
}

/// Flatten [`EngineKind::Auto`] to a concrete variant based on the source and
/// the caller's memory budget.
///
/// * `Raster` → [`EngineKind::Monolithic`] (fastest when the raster fits in
///   memory), **unless** an explicit memory budget was supplied via
///   [`EngineBuilder::with_memory_budget`] and the monolithic engine's
///   estimated peak (`estimate_peak_memory_for_format`: full canvas plus the
///   first downscaled level) would exceed it. In that case `Auto` selects the
///   bounded-memory [`EngineKind::Streaming`] engine so a large raster under a
///   small budget no longer runs the monolithic path at 2-3x source in RAM
///   and OOMs the container (issue #135).
/// * `Strip`  → [`EngineKind::Streaming`].
///
/// A `None` budget preserves the historical `Raster → Monolithic` choice: the
/// monolithic engine stays the default whenever the caller has not asked for a
/// bounded run.
///
/// Non-`Auto` kinds pass through unchanged.
fn resolve_engine_kind(
    kind: EngineKind,
    source: &EngineSource<'_>,
    plan: &PyramidPlan,
    memory_budget_bytes: Option<u64>,
) -> EngineKind {
    match (kind, source) {
        (EngineKind::Auto, EngineSource::Raster(raster)) => {
            let fits_budget = match memory_budget_bytes {
                Some(budget) => plan.estimate_peak_memory_for_format(raster.format()) <= budget,
                None => true,
            };
            if fits_budget {
                EngineKind::Monolithic
            } else {
                EngineKind::Streaming
            }
        }
        (EngineKind::Auto, EngineSource::Strip(_)) => EngineKind::Streaming,
        (k, _) => k,
    }
}

// ---------------------------------------------------------------------------
// Shared engine dispatch (issue #290)
// ---------------------------------------------------------------------------

/// The inputs the `(EngineKind × EngineSource)` render dispatch needs, bundled
/// so the plain and resume paths can share ONE dispatch site instead of each
/// carrying its own ~10-arm copy of the match (issue #290).
///
/// Before this, the match — and the seven-argument `build_mapreduce_config`
/// call inside four of its arms — was written out verbatim in both paths (and a
/// third, read-only variant for Verify). Every new engine kind or source
/// multiplied the arms across the sites and, being hand-copied, was a standing
/// drift hazard. Threading the shared inputs through this struct lets
/// [`RenderDispatch::run`] be the single site both paths call; the only
/// per-path difference — which sink the tiles are written through (the user
/// sink, or the resume [`ResumeAwareSink`](resume::ResumeAwareSink) wrapper) —
/// stays a parameter.
struct RenderDispatch<'a> {
    plan: &'a PyramidPlan,
    /// The resolved config, already carrying any resume checkpoint knobs.
    engine_cfg: &'a EngineConfig,
    observer: &'a dyn EngineObserver,
    executor: &'a dyn WorkExecutor,
    memory_budget_bytes: Option<u64>,
    budget_policy: Option<BudgetPolicy>,
    // Raw builder knobs consumed only when lowering to a `MapReduceConfig`,
    // which reads them as `Option`s (a `None` keeps the map-reduce default) —
    // exactly as the pre-refactor arms did.
    concurrency: Option<usize>,
    buffer_size: Option<usize>,
    background_rgb: Option<[u8; 3]>,
    blank_strategy: Option<BlankTileStrategy>,
    cancel: Option<crate::cancel::CancelToken>,
}

impl<'a> RenderDispatch<'a> {
    /// Lower to a [`StreamingConfig`] exactly as the pre-refactor arms did.
    fn streaming_config(&self) -> StreamingConfig {
        build_streaming_config(
            self.engine_cfg.clone(),
            self.memory_budget_bytes,
            self.budget_policy,
        )
    }

    /// Lower to a [`MapReduceConfig`] exactly as the pre-refactor arms did —
    /// the seven-argument call that used to be copied verbatim eight times.
    fn mapreduce_config(&self) -> MapReduceConfig {
        build_mapreduce_config(
            self.memory_budget_bytes,
            self.concurrency,
            self.buffer_size,
            self.background_rgb,
            self.blank_strategy,
            &self.engine_cfg.failure_policy,
            self.cancel.clone(),
        )
    }

    /// Run the engine selected by `(kind, source)` against `sink`, returning the
    /// engine's `Result` without touching it. This is the single render
    /// dispatch shared by [`EngineBuilder::run_collect`]'s plain and resume
    /// paths (issue #290).
    ///
    /// `Monolithic` + `Strip` is the one rejected pairing and yields
    /// [`EngineError::IncompatibleSource`]; the resume path rejects it earlier
    /// still (before taking the run lock) so a refused run never touches the
    /// output directory. Reaching the `Auto` arm is a bug — `resolve_engine_kind`
    /// flattens `Auto` before dispatch.
    fn run(
        &self,
        kind: EngineKind,
        source: EngineSource<'_>,
        sink: &dyn TileSink,
    ) -> Result<EngineResult, EngineError> {
        match (kind, source) {
            (EngineKind::Monolithic, EngineSource::Raster(raster)) => {
                generate_pyramid_observed(raster, self.plan, sink, self.engine_cfg, self.observer)
            }
            (EngineKind::Monolithic, EngineSource::Strip(_)) => {
                Err(EngineError::IncompatibleSource {
                    kind: EngineKind::Monolithic,
                    reason: "Monolithic engine requires an in-memory Raster source",
                })
            }
            (EngineKind::Streaming, EngineSource::Raster(raster)) => {
                let strip = RasterStripSource::new(raster);
                let cfg = self.streaming_config();
                generate_pyramid_streaming(&strip, self.plan, sink, &cfg, self.observer)
            }
            (EngineKind::Streaming, EngineSource::Strip(strip)) => {
                let cfg = self.streaming_config();
                generate_pyramid_streaming(strip.as_ref(), self.plan, sink, &cfg, self.observer)
            }
            (EngineKind::MapReduce, EngineSource::Raster(raster)) => {
                let strip = RasterStripSource::new(raster);
                let cfg = self.mapreduce_config();
                generate_pyramid_mapreduce(
                    &strip,
                    self.plan,
                    sink,
                    &cfg,
                    self.observer,
                    self.executor,
                )
            }
            (EngineKind::MapReduce, EngineSource::Strip(strip)) => {
                let cfg = self.mapreduce_config();
                generate_pyramid_mapreduce(
                    strip.as_ref(),
                    self.plan,
                    sink,
                    &cfg,
                    self.observer,
                    self.executor,
                )
            }
            (EngineKind::MapReduceHotCache, EngineSource::Raster(raster)) => {
                let strip = RasterStripSource::new(raster);
                let cfg = self.mapreduce_config();
                generate_pyramid_mapreduce_hot_cache(
                    &strip,
                    self.plan,
                    sink,
                    &cfg,
                    self.observer,
                    self.executor,
                )
            }
            (EngineKind::MapReduceHotCache, EngineSource::Strip(strip)) => {
                let cfg = self.mapreduce_config();
                generate_pyramid_mapreduce_hot_cache(
                    strip.as_ref(),
                    self.plan,
                    sink,
                    &cfg,
                    self.observer,
                    self.executor,
                )
            }
            (EngineKind::Auto, _) => {
                unreachable!("Auto should have been resolved before match")
            }
        }
    }
}

/// The read-only Verify counterpart to [`RenderDispatch::run`]: routes
/// `(kind, source)` to the matching verify helper (issue #290). Verify writes
/// nothing and keeps no checkpoint, so it drives the dedicated
/// [`crate::verify`] walks rather than the generate engines, but the dispatch
/// shape is the same — kept in one place so it cannot drift from the render
/// dispatch's source/kind handling.
fn dispatch_verify(
    kind: EngineKind,
    source: EngineSource<'_>,
    plan: &PyramidPlan,
    sink: &dyn TileSink,
    engine_cfg: &EngineConfig,
    observer: &dyn EngineObserver,
) -> Result<EngineResult, EngineError> {
    match (kind, source) {
        (EngineKind::Monolithic, EngineSource::Raster(raster)) => {
            crate::verify::raster_verify(raster, plan, sink, engine_cfg, observer)
        }
        (EngineKind::Streaming, EngineSource::Raster(raster))
        | (EngineKind::MapReduce, EngineSource::Raster(raster))
        | (EngineKind::MapReduceHotCache, EngineSource::Raster(raster)) => {
            let strip = RasterStripSource::new(raster);
            crate::verify::verify_from_strip_source(&strip, plan, sink, engine_cfg, observer)
        }
        (EngineKind::Streaming, EngineSource::Strip(strip))
        | (EngineKind::MapReduce, EngineSource::Strip(strip))
        | (EngineKind::MapReduceHotCache, EngineSource::Strip(strip)) => {
            crate::verify::verify_from_strip_source(
                strip.as_ref(),
                plan,
                sink,
                engine_cfg,
                observer,
            )
        }
        (EngineKind::Monolithic, EngineSource::Strip(_)) => {
            unreachable!("Monolithic + Strip rejected above")
        }
        (EngineKind::Auto, _) => {
            unreachable!("Auto should have been resolved before match")
        }
    }
}

// ---------------------------------------------------------------------------
// Resume invariants, as named helpers (issue #290)
// ---------------------------------------------------------------------------
//
// The live resume path used to carry each of these rules as a prose comment
// wrapped around inline code. Lifting them into named functions puts the
// invariant in the signature and one body, so the plain and resume paths share
// the render dispatch above without re-describing what resume then owes.

/// Resume invariant — **level promotion**: after the run returns, promote every
/// FULLY-written level to `levels_completed`. The engine driver no longer
/// touches the checkpoint (resume lives entirely in the
/// [`ResumeAwareSink`](resume::ResumeAwareSink)), so promotion happens here once
/// every tile has passed through the wrapper. [`CheckpointState::mark_level_completed`](crate::engine::CheckpointState::mark_level_completed)
/// is gated on every tile of the level being present in `completed_tiles`, so a
/// level left partial by a failed run — or by `RetryThenSkip` dropping a tile —
/// is withheld rather than falsely recorded (issue #125).
fn promote_completed_levels(cp: Option<&crate::engine::CheckpointState<'_>>, plan: &PyramidPlan) {
    if let Some(cp) = cp {
        for level in &plan.levels {
            cp.mark_level_completed(level.level, level.tile_count());
        }
    }
}

/// Resume invariant — **flush on BOTH paths**: persist the checkpoint whether
/// the run succeeded or failed. On the error path this is the only chance to
/// make completed tiles durable — an interrupted run (kill -9 / OOM /
/// preemption surfaced as a sink error, or an exhausted retry budget) would
/// otherwise drop the in-memory [`CheckpointState`](crate::engine::CheckpointState)
/// and force a later `--resume` to re-render everything. A flush error on the
/// success path aborts the run; on the failure path it is swallowed so the
/// original engine error — the reason the run stopped — is what propagates.
fn flush_checkpoint_dual_path(
    cp: Option<&crate::engine::CheckpointState<'_>>,
    run_result: &Result<EngineResult, EngineError>,
) -> Result<(), EngineError> {
    if let Some(cp) = cp {
        match run_result {
            Ok(_) => cp.flush().map_err(EngineError::ResumeFailed)?,
            Err(_) => {
                let _ = cp.flush();
            }
        }
    }
    Ok(())
}

/// Resume invariant — **skipped-count adjustment**: `tiles_produced` and
/// `bytes_written` must report tiles this run actually emitted, not every tile
/// the engine visited. The [`ResumeAwareSink`](resume::ResumeAwareSink)
/// short-circuits already-completed coordinates and returns `Ok` without
/// writing, which the monolithic engine still credits toward both counters, so
/// subtract exactly what the wrapper skipped. `saturating_sub` guards the
/// (impossible-by-construction) underflow.
fn adjust_result_for_resume_skips(
    result: &mut EngineResult,
    wrapped: &resume::ResumeAwareSink<'_>,
) {
    let skipped = wrapped.skipped_count();
    if skipped > 0 {
        result.tiles_produced = result.tiles_produced.saturating_sub(skipped);
    }
    let skipped_bytes = wrapped.skipped_bytes();
    if skipped_bytes > 0 {
        result.bytes_written = result.bytes_written.saturating_sub(skipped_bytes);
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
fn prepare_resume_state<'a>(
    sink: &'a dyn TileSink,
    plan: &PyramidPlan,
    config: &EngineConfig,
    mode: ResumeMode,
) -> Result<
    (
        std::collections::HashSet<crate::planner::TileCoord>,
        Option<crate::engine::CheckpointState<'a>>,
    ),
    EngineError,
> {
    match mode {
        ResumeMode::Overwrite => {
            // Wipe only the sink's OWN output directory. Stale tiles live
            // there, so that is the one place a fresh Overwrite must clear.
            // We deliberately do NOT wipe `config.checkpoint_root`: a caller
            // supplies it to keep metadata out of the output, and it may hold
            // unrelated files that must survive.
            if let Some(sink_dir) = sink.checkpoint_root() {
                crate::engine::wipe_directory(sink_dir)
                    .map_err(|e| EngineError::ResumeFailed(crate::resume::ResumeError::from(e)))?;
            }
            // When an explicit checkpoint_root sits apart from the sink dir,
            // it only ever holds our own checkpoint files. Remove both the
            // JSON header and the append-only segment log (issue #127) so a
            // stale checkpoint can't make this fresh run look resumable, and so
            // the fresh run does not append onto a prior run's coordinate log,
            // leaving every other entry in place.
            if let Some(root) = &config.checkpoint_root {
                let same_as_sink = sink.checkpoint_root().is_some_and(|s| s == root.as_path());
                if !same_as_sink {
                    for marker in [
                        crate::resume::JobCheckpoint::checkpoint_path(root),
                        crate::resume::JobCheckpoint::segments_path(root),
                    ] {
                        match std::fs::remove_file(&marker) {
                            Ok(()) => {}
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                            Err(e) => {
                                return Err(EngineError::ResumeFailed(
                                    crate::resume::ResumeError::from(e),
                                ));
                            }
                        }
                    }
                }
            }
            let cp = crate::engine::cp_for_sink(sink, plan, config, Vec::new(), Vec::new());
            if cp.is_some() {
                // The engine's CheckpointState is the single checkpoint writer
                // for this run (issue #277). Arm the sink's durability tracking
                // so its tile writes are recorded for the checkpoint's
                // `sync_pending` barrier (issue #273).
                sink.arm_durability_tracking();
            }
            Ok((std::collections::HashSet::new(), cp))
        }
        ResumeMode::Resume => {
            let (completed, levels) =
                if let Some(root) = crate::engine::resolve_checkpoint_root(config, sink) {
                    match crate::resume::JobCheckpoint::load(&root)? {
                        Some(meta) => {
                            if let Err(current) =
                                crate::resume::verify_checkpoint_contract(&meta, plan, config, sink)
                            {
                                return Err(EngineError::PlanHashMismatch {
                                    expected: current,
                                    actual: meta.plan_hash.clone(),
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
            if cp.is_some() {
                // The engine's CheckpointState is the single checkpoint writer
                // for this run (issue #277). Arm the sink's durability tracking
                // so its tile writes are recorded for the checkpoint's
                // `sync_pending` barrier (issue #273).
                sink.arm_durability_tracking();
            }
            Ok((skip, cp))
        }
        ResumeMode::Verify => unreachable!("Verify is rejected above for non-Monolithic engines"),
    }
}

mod resume {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::engine::CheckpointState;
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
    pub(super) struct ResumeAwareSink<'a> {
        pub(super) inner: &'a dyn TileSink,
        pub(super) skip: &'a HashSet<TileCoord>,
        pub(super) cp: Option<&'a CheckpointState<'a>>,
        /// Running count of skipped writes, used after `.run()` to adjust
        /// [`crate::engine::EngineResult::tiles_produced`] down so that
        /// callers see "tiles actually written this run" rather than
        /// "tiles the engine visited".
        pub(super) skipped: AtomicU64,
        /// Running sum of the payload bytes of skipped tiles. The monolithic
        /// engine credits `bytes_written` on every `Ok` from the sink — and a
        /// short-circuited skip returns `Ok` without writing anything — so the
        /// engine over-counts by exactly this many bytes. The builder subtracts
        /// it after the run so `bytes_written` reflects bytes actually emitted.
        pub(super) skipped_bytes: AtomicU64,
    }

    impl<'a> ResumeAwareSink<'a> {
        pub(super) fn new(
            inner: &'a dyn TileSink,
            skip: &'a HashSet<TileCoord>,
            cp: Option<&'a CheckpointState<'a>>,
        ) -> Self {
            Self {
                inner,
                skip,
                cp,
                skipped: AtomicU64::new(0),
                skipped_bytes: AtomicU64::new(0),
            }
        }

        pub(super) fn skipped_count(&self) -> u64 {
            self.skipped.load(Ordering::Relaxed)
        }

        pub(super) fn skipped_bytes(&self) -> u64 {
            self.skipped_bytes.load(Ordering::Relaxed)
        }
    }

    impl<'a> TileSink for ResumeAwareSink<'a> {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            if self.skip.contains(&tile.coord) {
                self.skipped.fetch_add(1, Ordering::Relaxed);
                self.skipped_bytes
                    .fetch_add(tile.raster.data().len() as u64, Ordering::Relaxed);
                // Rebuild the sink-side manifest / dedupe / checksum state this
                // pre-crash tile contributes so a resumed `finish()` reproduces
                // the same complete manifest + dedupe layout as an uninterrupted
                // run (issue #272). Deliberately NOT followed by
                // `mark_tile_completed`: the coordinate is already recorded in
                // the checkpoint, and re-marking it would duplicate it in
                // `completed_tiles`. For a plain (non-dedupe/checksum) resume the
                // sink short-circuits this to a no-op, so the skip stays free.
                self.inner.seed_completed_tile(tile)?;
                return Ok(());
            }
            self.inner.write_tile(tile)?;
            if let Some(cp) = self.cp {
                cp.mark_tile_completed(tile.coord)
                    .map_err(SinkError::Checkpoint)?;
            }
            Ok(())
        }

        fn finish(&self) -> Result<(), SinkError> {
            self.inner.finish()
        }

        /// This wrapper owns no engine bookkeeping of its own: every hook
        /// (`record_engine_config`, `sink_retry_count`,
        /// `sink_skipped_due_to_failure`, `note_sink_skipped`,
        /// `checkpoint_root`, `init_level_count`, `content_format`,
        /// `applies_retry_policy`) must reach the wrapped sink unchanged.
        /// Exposing the inner sink through this single hook makes the
        /// trait's defaults forward all of them, so the wrapper can never
        /// again silently drop one, the way it previously dropped
        /// `content_format` by simply not listing it (issue #137).
        fn inner_sink(&self) -> Option<&dyn TileSink> {
            Some(self.inner)
        }
    }

    // Silence "field never read" lints on the AtomicU64 placeholder if
    // future refactors drop these fields.
    #[allow(dead_code)]
    fn _unused_marker(_: AtomicU64) {}
}

// ---------------------------------------------------------------------------
// Retry-wiring tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod retry_wiring_tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::sink::{MemorySink, SinkError, Tile};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A sink that fails its first `fails` `write_tile` calls with a transient
    /// error, then forwards every subsequent call to an inner [`MemorySink`].
    /// Models an object store that intermittently rejects writes.
    struct FlakySink {
        inner: MemorySink,
        fails_left: AtomicU32,
    }

    impl FlakySink {
        fn new(fails: u32) -> Self {
            Self {
                inner: MemorySink::new(),
                fails_left: AtomicU32::new(fails),
            }
        }

        fn written(&self) -> usize {
            self.inner.tile_count()
        }
    }

    impl TileSink for FlakySink {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            loop {
                let cur = self.fails_left.load(Ordering::SeqCst);
                if cur == 0 {
                    break;
                }
                if self
                    .fails_left
                    .compare_exchange(cur, cur - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    return Err(SinkError::Other("transient".into()));
                }
            }
            self.inner.write_tile(tile)
        }
    }

    fn small_source() -> Raster {
        // 4x4 RGB solid so every tile is non-blank and actually written.
        let data = vec![10u8; 4 * 4 * 3];
        Raster::new(4, 4, PixelFormat::Rgb8, data).unwrap()
    }

    fn small_plan() -> PyramidPlan {
        PyramidPlanner::new(4, 4, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    fn fast_policy(max_retries: u32) -> RetryPolicy {
        RetryPolicy::new(max_retries, std::time::Duration::from_micros(1))
            .with_multiplier(1.0)
            .with_max_backoff(std::time::Duration::from_micros(10))
            .with_jitter(false)
    }

    // A transient write failure under `RetryThenFail` must be retried through
    // the builder, not propagated on the first error. Fails on unwired code
    // (the whole run errors out); passes once the RetryingSink is engaged.
    #[test]
    fn builder_retry_then_fail_recovers_transient_failures() {
        let source = small_source();
        let plan = small_plan();
        let sink = FlakySink::new(2);

        let (result, sink) = EngineBuilder::new(&source, plan, sink)
            .with_retry(fast_policy(5))
            .run_collect()
            .expect("transient failures must be retried, not propagated");

        assert!(
            result.retry_count >= 2,
            "expected the configured retry policy to drive at least 2 retries, got {}",
            result.retry_count
        );
        assert_eq!(
            result.skipped_due_to_failure, 0,
            "RetryThenFail must not skip any tile"
        );
        assert!(sink.written() > 0, "every tile must ultimately be written");
    }

    // Under `RetryThenSkip`, a tile must only be skipped after its retries are
    // exhausted. With enough retry budget the transient failures are absorbed
    // and nothing is skipped. Fails on unwired code (tiles dropped on the
    // first error, skip counter non-zero, retry_count zero).
    #[test]
    fn builder_retry_then_skip_only_skips_after_exhaustion() {
        let source = small_source();
        let plan = small_plan();
        let sink = FlakySink::new(2);

        let (result, sink) = EngineBuilder::new(&source, plan, sink)
            .with_failure_policy(FailurePolicy::RetryThenSkip(fast_policy(5)))
            .run_collect()
            .expect("run should succeed");

        assert!(
            result.retry_count >= 2,
            "expected retries to be driven before skipping, got {}",
            result.retry_count
        );
        assert_eq!(
            result.skipped_due_to_failure, 0,
            "no tile should be skipped while retry budget remains"
        );
        assert!(
            sink.written() > 0,
            "tiles recovered by retry must land in the sink"
        );
    }
}

// ---------------------------------------------------------------------------
// Extension-hatch wiring (issue #138)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod extension_wiring_tests {
    use super::*;
    use crate::observe::EngineEvent;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::sink::MemorySink;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A user-supplied context handle stashed on the builder via
    /// `with_extension` and expected back through the observer.
    struct Marker(u64);

    /// Observer that records the value of any [`Marker`] extension delivered to
    /// it, plus whether `on_extensions` fired at all.
    struct CapturingObserver {
        seen_value: Arc<AtomicU64>,
        fired: Arc<AtomicU64>,
    }

    impl EngineObserver for CapturingObserver {
        fn on_event(&self, _event: EngineEvent) {}

        fn on_extensions(&self, extensions: &Extensions) {
            self.fired.fetch_add(1, Ordering::SeqCst);
            if let Some(marker) = extensions.get::<Marker>() {
                self.seen_value.store(marker.0, Ordering::SeqCst);
            }
        }
    }

    fn small_source() -> Raster {
        let data = vec![10u8; 4 * 4 * 3];
        Raster::new(4, 4, PixelFormat::Rgb8, data).unwrap()
    }

    fn small_plan() -> PyramidPlan {
        PyramidPlanner::new(4, 4, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    // The extension hatch must be functional, not decorative: a value inserted
    // via `with_extension` has to reach the attached observer's
    // `on_extensions` hook before the run tiles. On the pre-fix code the
    // builder destructured `extensions: _extensions` and discarded it, so the
    // observer never saw it and this asserts 0 (RED). After wiring the delivery
    // the observer reads the marker back (GREEN).
    #[test]
    fn builder_delivers_extensions_to_observer() {
        let seen_value = Arc::new(AtomicU64::new(0));
        let fired = Arc::new(AtomicU64::new(0));
        let observer = CapturingObserver {
            seen_value: seen_value.clone(),
            fired: fired.clone(),
        };

        let source = small_source();
        let plan = small_plan();
        EngineBuilder::new(&source, plan, MemorySink::new())
            .with_observer(observer)
            .with_extension(Marker(42))
            .run()
            .expect("run must succeed");

        assert_eq!(
            fired.load(Ordering::SeqCst),
            1,
            "the observer's on_extensions hook must fire exactly once per run"
        );
        assert_eq!(
            seen_value.load(Ordering::SeqCst),
            42,
            "the value inserted via with_extension must be delivered to the observer"
        );
    }

    // The hatch is delivered on every routed engine kind, not just the default
    // monolithic path — a metrics/tracing observer must see its context
    // regardless of how `Auto` resolves or which kind the caller pins.
    #[test]
    fn extensions_delivered_across_engine_kinds() {
        for kind in [
            EngineKind::Monolithic,
            EngineKind::Streaming,
            EngineKind::MapReduce,
            EngineKind::MapReduceHotCache,
        ] {
            let seen_value = Arc::new(AtomicU64::new(0));
            let fired = Arc::new(AtomicU64::new(0));
            let observer = CapturingObserver {
                seen_value: seen_value.clone(),
                fired: fired.clone(),
            };

            let source = small_source();
            let plan = small_plan();
            EngineBuilder::new(&source, plan, MemorySink::new())
                .with_engine(kind)
                .with_observer(observer)
                .with_extension(Marker(7))
                .run()
                .unwrap_or_else(|e| panic!("{kind:?} run must succeed: {e:?}"));

            assert_eq!(
                fired.load(Ordering::SeqCst),
                1,
                "{kind:?}: on_extensions must fire once"
            );
            assert_eq!(
                seen_value.load(Ordering::SeqCst),
                7,
                "{kind:?}: the extension value must reach the observer"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint durability on the error path
// ---------------------------------------------------------------------------

#[cfg(test)]
mod checkpoint_durability_tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::resume::{JobCheckpoint, ResumePolicy};
    use crate::sink::{MemorySink, SinkError, Tile, TileSink};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A sink that forwards its first `ok` `write_tile` calls to an inner
    /// [`MemorySink`] and then fails every subsequent call with a permanent
    /// error. Models a process that is killed / runs out of disk part-way
    /// through emitting a pyramid.
    struct FailAfterSink {
        inner: MemorySink,
        ok_left: AtomicU32,
    }

    impl FailAfterSink {
        fn new(ok: u32) -> Self {
            Self {
                inner: MemorySink::new(),
                ok_left: AtomicU32::new(ok),
            }
        }
    }

    impl TileSink for FailAfterSink {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            if self.ok_left.load(Ordering::SeqCst) == 0 {
                return Err(SinkError::Other("permanent failure".into()));
            }
            self.ok_left.fetch_sub(1, Ordering::SeqCst);
            self.inner.write_tile(tile)
        }
    }

    fn solid_source() -> Raster {
        // 8x8 RGB solid so every tile is non-blank and actually written.
        let data = vec![10u8; 8 * 8 * 3];
        Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap()
    }

    fn solid_plan() -> PyramidPlan {
        PyramidPlanner::new(8, 8, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    // A run that dies mid-way must still leave an on-disk checkpoint that
    // records the tiles it completed, so a later `--resume` skips them
    // instead of re-rendering everything. Before the fix the checkpoint was
    // flushed only on the success path, so an interrupted run left nothing
    // on disk and `JobCheckpoint::load` returned `None`. The high cadence
    // here guarantees no periodic flush fired, isolating the error-path
    // flush as the only way the two completed tiles can reach disk.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn interrupted_resume_run_persists_completed_tiles() {
        let dir = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();
        let sink = FailAfterSink::new(2);

        let outcome = EngineBuilder::new(&source, plan, sink)
            .with_resume(
                ResumePolicy::resume()
                    .with_checkpoint_root(dir.path())
                    .with_checkpoint_every(1_000_000),
            )
            .run_collect();

        assert!(
            outcome.is_err(),
            "the permanent sink failure must abort the run"
        );

        let meta = JobCheckpoint::load(dir.path())
            .expect("checkpoint load must not error")
            .expect("an interrupted run must leave a checkpoint on disk");
        assert_eq!(
            meta.completed_tiles.len(),
            2,
            "the checkpoint must record exactly the tiles completed before the failure"
        );
    }

    /// A sink that forwards its first `n - 1` writes to an inner [`FsSink`]
    /// (so they land on disk *and* drive the resume checkpoint) and then
    /// **panics** on the `n`-th write. A panic — unlike a returned
    /// `SinkError` — unwinds straight past the builder's error-path
    /// `cp.flush()`, so the only tiles that survive are the ones a *periodic*
    /// flush already persisted. That makes the checkpoint cadence the sole
    /// thing standing between a crash and a full re-render, which is exactly
    /// the invariant this reproducer guards.
    struct PanicOnNthSink {
        inner: crate::sink::FsSink,
        counter: AtomicU32,
        panic_at: u32,
    }

    impl TileSink for PanicOnNthSink {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            let n = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
            if n == self.panic_at {
                panic!("PanicOnNthSink: deliberate crash at write #{n}");
            }
            self.inner.write_tile(tile)
        }

        fn finish(&self) -> Result<(), SinkError> {
            self.inner.finish()
        }

        fn checkpoint_root(&self) -> Option<&std::path::Path> {
            self.inner.checkpoint_root()
        }
    }

    // Regression: `EngineConfig::with_checkpoint_every` handed to the builder
    // via `with_config` must actually take effect on a Resume run.
    //
    // `ResumePolicy::resume()` seeds a coarse fallback cadence
    // (`DEFAULT_RESUME_CHECKPOINT_EVERY`, 1000). `with_config` previously only
    // adopted the config's cadence when the policy's was still 0, so an
    // explicit `with_checkpoint_every(5)` on the config was silently discarded
    // and the engine flushed every 1000 tiles instead. A crash before the
    // first fallback flush then left *no* checkpoint, forcing `--resume` to
    // re-render every tile — unbounded rework.
    //
    // Here a 64-tile run crashes (panics) at write #30 with the config asking
    // for a flush every 5 tiles. The periodic flush must have persisted the
    // completed tiles, and the count must be bounded to within one cadence
    // interval of the crash point.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn config_checkpoint_every_bounds_resume_rework() {
        let dir = tempfile::tempdir().unwrap();
        // 32x32 @ 4px tiles => top level 8x8 = 64 tiles, plus coarser levels;
        // comfortably more than the 30-write crash point below.
        let data = vec![7u8; 32 * 32 * 3];
        let source = Raster::new(32, 32, PixelFormat::Rgb8, data).unwrap();
        let plan = PyramidPlanner::new(32, 32, 4, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let total = plan.total_tile_count();
        assert!(total > 30, "need > 30 tiles to crash mid-run, got {total}");

        let panic_at = 30u32;
        let cadence = 5u64;

        let sink = PanicOnNthSink {
            inner: crate::sink::FsSink::new(dir.path().join("pyr"), plan.clone()),
            counter: AtomicU32::new(0),
            panic_at,
        };

        // `with_resume` first, then `with_config` — mirroring the documented
        // migration shape — so the config's cadence must thread into the policy.
        let cfg = EngineConfig::default()
            .with_concurrency(1)
            .with_checkpoint_every(cadence);

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            EngineBuilder::new(&source, plan.clone(), sink)
                .with_resume(ResumePolicy::resume())
                .with_config(cfg)
                .run()
        }));
        assert!(outcome.is_err(), "the deliberate panic must abort the run");

        let root = dir.path().join("pyr");
        let meta = JobCheckpoint::load(&root)
            .expect("checkpoint load must not error")
            .expect(
                "a periodic flush must have left a checkpoint on disk; \
                 None means the config cadence was dropped and rework is unbounded",
            );

        let completed = meta.completed_tiles.len() as u64;
        // The write before the crash that landed on a cadence boundary is the
        // floor; everything up to the crash point is the ceiling.
        assert!(
            completed > 0,
            "the checkpoint must record the tiles a periodic flush persisted"
        );
        assert!(
            completed >= (panic_at as u64 - 1) - cadence,
            "expected the checkpoint within one cadence interval of the crash: \
             completed={completed}, crash_at={panic_at}, cadence={cadence}"
        );
        assert!(
            completed <= panic_at as u64,
            "the checkpoint cannot record more tiles than were written: \
             completed={completed}, crash_at={panic_at}"
        );
    }

    // The Resume factory must default to a non-zero flush cadence so that a
    // long run periodically persists progress even when the caller never sets
    // an explicit cadence. Overwrite / Verify keep the "final flush only"
    // default of 0.
    #[test]
    fn resume_policy_defaults_to_nonzero_cadence() {
        assert!(
            ResumePolicy::resume().checkpoint_every() > 0,
            "Resume mode must default to a non-zero checkpoint cadence"
        );
        assert_eq!(
            ResumePolicy::overwrite().checkpoint_every(),
            0,
            "Overwrite must keep the final-flush-only default"
        );
        assert_eq!(
            ResumePolicy::verify().checkpoint_every(),
            0,
            "Verify must keep the final-flush-only default"
        );
    }

    // A plan whose declared image dimensions do not match the source raster
    // must be rejected with a typed error at the engine entry point, not blow
    // up inside `embed_in_canvas`'s slice copy. Here a centred Google plan for
    // a 300x300 image (canvas 512x512, centre offset 106) is paired with a
    // larger 500x500 source; the embed loop writes source rows past the end of
    // the canvas buffer and panics on pre-fix code. After the fix the run
    // returns `EngineError::PlanSourceMismatch` cleanly.
    #[test]
    fn run_collect_rejects_plan_source_dimension_mismatch() {
        let source = Raster::new(500, 500, PixelFormat::Rgb8, vec![10u8; 500 * 500 * 3]).unwrap();
        let plan = PyramidPlanner::new(300, 300, 256, 0, Layout::Google)
            .unwrap()
            .with_centre(true)
            .plan();

        let result = EngineBuilder::new(&source, plan, MemorySink::new()).run_collect();

        match result {
            Err(EngineError::PlanSourceMismatch {
                plan_width,
                plan_height,
                source_width,
                source_height,
            }) => {
                assert_eq!((plan_width, plan_height), (300, 300));
                assert_eq!((source_width, source_height), (500, 500));
            }
            other => panic!("expected PlanSourceMismatch, got {other:?}"),
        }
    }

    // A hand-mutated plan with an empty level list underflows `levels.len() - 1`
    // (debug panic / release silent zero-tile "success"). The engine must
    // reject it with a typed error before it reaches that arithmetic.
    #[test]
    fn run_collect_rejects_plan_with_no_levels() {
        let source = solid_source();
        let mut plan = solid_plan();
        plan.levels.clear();

        let result = EngineBuilder::new(&source, plan, MemorySink::new()).run_collect();

        assert!(
            matches!(result, Err(EngineError::InvalidPlan { .. })),
            "expected InvalidPlan for an empty-levels plan, got {result:?}"
        );
    }

    /// Counts file- and directory-fsyncs so a test can observe whether a sink
    /// *published* a checkpoint (a directory fsync of its base dir) versus
    /// merely fsyncing tile data.
    #[derive(Default)]
    struct SyncCounter {
        files: AtomicU32,
        dirs: AtomicU32,
    }

    impl crate::resume::Durability for SyncCounter {
        fn sync_file(&self, _path: &std::path::Path) -> std::io::Result<()> {
            self.files.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn sync_dir(&self, _path: &std::path::Path) -> std::io::Result<()> {
            self.dirs.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Issue #273 / #277: a builder-driven resume run must (a) have exactly one
    /// component write the checkpoint — the engine's `CheckpointState`, never
    /// the sink — and (b) still fsync tile data, even though the sink is a
    /// PLAIN `FsSink` the caller never flagged for resume. The builder arms the
    /// sink's durability tracking automatically, and the engine's checkpoint
    /// barrier fsyncs the tile files.
    ///
    /// We inject a counting durability into the sink and assert that, after a
    /// builder-driven resume run, the sink issued zero directory fsyncs (it
    /// publishes no checkpoint of its own) while its tile-data fsyncs are
    /// non-zero (the durability barrier ran). The checkpoint nonetheless exists
    /// and records every tile, proving the engine's writer is the single,
    /// complete source.
    ///
    /// RED before the fix: a plain builder-path `FsSink` was never told to
    /// track or fsync anything (`resume_enabled` stayed false and the builder
    /// never set it), so `recorder.files` was zero. GREEN after: the builder
    /// arms durability tracking and the checkpoint barrier fsyncs every tile.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_run_has_single_checkpoint_writer() {
        use crate::sink::{FsSink, TileFormat};
        use std::sync::Arc;

        let source = solid_source();
        let plan = solid_plan();
        let total_tiles: usize = plan
            .levels
            .iter()
            .map(|lp| (lp.cols as usize) * (lp.rows as usize))
            .sum();

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("out_files");

        let recorder = Arc::new(SyncCounter::default());
        // A PLAIN FsSink: the caller does NOT flag it for resume. The builder
        // arms durability tracking on its own (the #273 regression was that it
        // did not, so the builder path never fsynced tiles).
        let sink = FsSink::new(base.clone(), plan.clone())
            .with_format(TileFormat::Raw)
            .with_durability(recorder.clone());

        // A very coarse cadence guarantees no periodic flush fires: only the
        // builder's terminal `CheckpointState::flush` publishes and fsyncs.
        EngineBuilder::new(&source, plan.clone(), sink)
            .with_resume(ResumePolicy::resume().with_checkpoint_every(1_000_000))
            .run_collect()
            .expect("resume run must succeed");

        assert_eq!(
            recorder.dirs.load(Ordering::Relaxed),
            0,
            "the sink must not publish its own checkpoint (directory fsync); the \
             engine's CheckpointState is the sole writer (issue #277)"
        );
        assert!(
            recorder.files.load(Ordering::Relaxed) > 0,
            "the builder must arm durability tracking so tile data is fsynced \
             before the checkpoint certifies it (issue #273)"
        );

        // The engine's writer is the single, complete source of the checkpoint.
        let meta = JobCheckpoint::load(&base)
            .expect("checkpoint load must not error")
            .expect("the engine's CheckpointState must have published a checkpoint");
        assert_eq!(
            meta.completed_tiles.len(),
            total_tiles,
            "the sole checkpoint must record every completed tile"
        );
    }
}

// ---------------------------------------------------------------------------
// FailurePolicy parity across engine kinds (issue #134)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod failure_policy_parity_tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::sink::{MemorySink, SinkError, Tile};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A sink that forwards its first `ok` `write_tile` calls to an inner
    /// [`MemorySink`] and then fails every subsequent call with a *permanent*
    /// error. Models a sink that dies partway through a run; under
    /// `RetryThenSkip` the doomed tiles must be skipped (not propagated) on
    /// every engine kind.
    struct FailAfterSink {
        inner: MemorySink,
        ok_left: AtomicU32,
    }

    impl FailAfterSink {
        fn new(ok: u32) -> Self {
            Self {
                inner: MemorySink::new(),
                ok_left: AtomicU32::new(ok),
            }
        }

        fn written(&self) -> usize {
            self.inner.tile_count()
        }
    }

    impl TileSink for FailAfterSink {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            loop {
                let cur = self.ok_left.load(Ordering::SeqCst);
                if cur == 0 {
                    return Err(SinkError::Other("permanent failure".into()));
                }
                if self
                    .ok_left
                    .compare_exchange(cur, cur - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    return self.inner.write_tile(tile);
                }
            }
        }
    }

    fn source() -> Raster {
        // 8x8 RGB solid so every tile is non-blank and actually written.
        let data = vec![10u8; 8 * 8 * 3];
        Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap()
    }

    fn plan() -> PyramidPlan {
        PyramidPlanner::new(8, 8, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    fn fast_policy(max_retries: u32) -> RetryPolicy {
        RetryPolicy::new(max_retries, std::time::Duration::from_micros(1))
            .with_multiplier(1.0)
            .with_max_backoff(std::time::Duration::from_micros(10))
            .with_jitter(false)
    }

    // Under `RetryThenSkip`, a permanently-failing sink must cause the doomed
    // tiles to be *skipped* — the run completes with a non-zero
    // `skipped_due_to_failure`, not an error — regardless of which engine kind
    // executes it. Before the fix, only the monolithic engine honored the
    // policy; the streaming and MapReduce engines propagated the first
    // terminal write error (FailFast behaviour), so `run_collect` returned
    // `Err`. This test drives each concrete kind and asserts uniform skip
    // semantics.
    fn assert_skip_honored(kind: EngineKind, concurrency: Option<usize>) {
        let ok = 2u32;
        let sink = FailAfterSink::new(ok);
        let src = source();

        let mut builder = EngineBuilder::new(&src, plan(), sink)
            .with_engine(kind)
            .with_failure_policy(FailurePolicy::RetryThenSkip(fast_policy(2)));
        if let Some(c) = concurrency {
            builder = builder.with_concurrency(c);
        }

        let (result, sink) = builder.run_collect().unwrap_or_else(|e| {
            panic!("{kind:?} must skip doomed tiles under RetryThenSkip, got error: {e:?}")
        });

        assert!(
            result.skipped_due_to_failure > 0,
            "{kind:?}: expected at least one tile skipped due to failure, got {}",
            result.skipped_due_to_failure
        );
        assert_eq!(
            sink.written() as u32,
            ok,
            "{kind:?}: exactly the pre-failure tiles must land in the sink"
        );
    }

    #[test]
    fn monolithic_honors_retry_then_skip() {
        assert_skip_honored(EngineKind::Monolithic, None);
    }

    #[test]
    fn streaming_honors_retry_then_skip() {
        assert_skip_honored(EngineKind::Streaming, None);
    }

    #[test]
    fn mapreduce_sequential_honors_retry_then_skip() {
        assert_skip_honored(EngineKind::MapReduce, Some(0));
    }

    #[test]
    fn mapreduce_parallel_honors_retry_then_skip() {
        assert_skip_honored(EngineKind::MapReduce, Some(2));
    }
}

// ---------------------------------------------------------------------------
// `with_config` field-precedence uniformity
// ---------------------------------------------------------------------------

#[cfg(test)]
mod with_config_precedence_tests {
    use super::*;
    use crate::dedupe::DedupeStrategy;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::sink::MemorySink;

    fn source() -> Raster {
        Raster::new(4, 4, PixelFormat::Rgb8, vec![10u8; 4 * 4 * 3]).unwrap()
    }

    fn plan() -> PyramidPlan {
        PyramidPlanner::new(4, 4, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    // Fill-if-unset (issue #297): `with_config` fills only the fields no earlier
    // setter has set, so an explicit `.with_*` setter always survives a later
    // `with_config`. Every field now behaves identically — an earlier
    // `.with_concurrency` / `.with_dedupe` / `.with_skip_blanks` / `.with_cancel`
    // all survive a config that would otherwise overwrite them, while a field no
    // setter touched adopts the config's value.
    //
    // This is the RED->GREEN pin for the clobber fix: before the fix the four
    // "survive" assertions failed (the config's default/`None` wiped each earlier
    // setter, dropping cancellation tokens and dedupe strategies with no signal);
    // after it they hold, and `with_config` still fills the untouched
    // `buffer_size`.
    #[test]
    fn with_config_fills_only_unset_fields_preserving_earlier_setters() {
        let src = source();
        let cancel = crate::cancel::CancelToken::new();

        // A config carrying a non-default buffer_size (a field no setter below
        // touches) but otherwise in its default shape: concurrency 0, no dedupe,
        // no cancel, skip_blanks false.
        let cfg = EngineConfig::default().with_buffer_size(9);

        let builder = EngineBuilder::new(&src, plan(), MemorySink::new())
            .with_concurrency(4)
            .with_dedupe(DedupeStrategy::Blanks)
            .with_skip_blanks(true)
            .with_cancel(cancel)
            .with_config(cfg);

        // Every earlier fine-grained setter survives the later with_config.
        assert_eq!(
            builder.concurrency,
            Some(4),
            "an earlier .with_concurrency must survive with_config"
        );
        assert_eq!(
            builder.dedupe,
            Some(DedupeStrategy::Blanks),
            "an earlier .with_dedupe must survive a config that carries no dedupe"
        );
        assert_eq!(
            builder.skip_blanks,
            Some(true),
            "an earlier .with_skip_blanks(true) must survive a config with skip_blanks=false"
        );
        assert!(
            builder.cancel.is_some(),
            "an earlier .with_cancel must survive a config that carries no cancel token"
        );
        // A field no setter touched still adopts the config's value.
        assert_eq!(
            builder.buffer_size,
            Some(9),
            "with_config must still fill a field no earlier setter set"
        );
    }

    // The flip side of the uniform rule: setters applied *after* `with_config`
    // win, for the same fields the config would otherwise clear.
    #[test]
    fn setters_after_with_config_take_precedence() {
        let src = source();

        let builder = EngineBuilder::new(&src, plan(), MemorySink::new())
            .with_config(EngineConfig::default())
            .with_concurrency(7)
            .with_dedupe(DedupeStrategy::Blanks);

        assert_eq!(builder.concurrency, Some(7));
        assert_eq!(builder.dedupe, Some(DedupeStrategy::Blanks));
    }
}

// ---------------------------------------------------------------------------
// `skip_blanks` through the real EngineBuilder + FsSink acceptance path
// (libviprs-tests issue #87)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod skip_blanks_builder_tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::sink::{FsSink, TileFormat};
    use std::path::Path;

    /// A mostly-white raster whose top-left `patch × patch` quadrant carries a
    /// gradient, so full-resolution tiles split into uniform (blank) tiles over
    /// the white margin and non-uniform tiles over the patch.
    fn mostly_blank_raster(w: u32, h: u32, patch: u32) -> Raster {
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let mut data = vec![255u8; w as usize * h as usize * bpp];
        for y in 0..patch.min(h) {
            for x in 0..patch.min(w) {
                let off = (y as usize * w as usize + x as usize) * bpp;
                data[off] = (x % 256) as u8;
                data[off + 1] = (y % 256) as u8;
                data[off + 2] = ((x + y) % 256) as u8;
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// Recursively count regular files with the `.raw` tile extension under
    /// `root`. FsSink writes one such file per emitted tile, so this is the
    /// on-disk tile-file count the acceptance path actually produces.
    fn count_raw_tile_files(root: &Path) -> usize {
        fn walk(cur: &Path, count: &mut usize) {
            let Ok(entries) = std::fs::read_dir(cur) else {
                return;
            };
            for entry in entries.filter_map(Result::ok) {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, count);
                } else if p.extension().and_then(|e| e.to_str()) == Some("raw") {
                    *count += 1;
                }
            }
        }
        let mut count = 0;
        walk(root, &mut count);
        count
    }

    /// The real acceptance path: build a pyramid through
    /// [`EngineBuilder`] + [`FsSink`] over a mostly-blank source, once with
    /// `with_skip_blanks(true)` and once with it off. Skipping must hand
    /// strictly FEWER tiles to the sink, leaving strictly fewer `.raw` files on
    /// disk, and must report `tiles_skipped > 0`. This exercises the builder's
    /// `with_skip_blanks` setter end to end (the field must survive the builder
    /// into the monolithic engine, which was the regression), and it is
    /// parameterised over concurrency so the parallel consumer's skip branch is
    /// locked in alongside the single-threaded one.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn with_skip_blanks_writes_strictly_fewer_files_through_the_builder() {
        for concurrency in [0usize, 2] {
            let src = mostly_blank_raster(512, 512, 256);
            let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom)
                .unwrap()
                .plan();

            let run = |skip: bool, dir: &Path| -> EngineResult {
                let sink =
                    FsSink::new(dir.to_path_buf(), plan.clone()).with_format(TileFormat::Raw);
                EngineBuilder::new(&src, plan.clone(), sink)
                    .with_engine(EngineKind::Monolithic)
                    .with_concurrency(concurrency)
                    .with_skip_blanks(skip)
                    .run()
                    .expect("builder run must succeed")
            };

            let full_dir = tempfile::tempdir().unwrap();
            let skip_dir = tempfile::tempdir().unwrap();

            let full = run(false, full_dir.path());
            let skipped = run(true, skip_dir.path());

            let full_files = count_raw_tile_files(full_dir.path());
            let skip_files = count_raw_tile_files(skip_dir.path());

            // Setup sanity: the full run wrote one file per produced tile.
            assert_eq!(
                full_files as u64, full.tiles_produced,
                "concurrency={concurrency}: full run must write one file per produced tile"
            );
            assert_eq!(
                skip_files as u64, skipped.tiles_produced,
                "concurrency={concurrency}: skip run must write one file per produced tile"
            );

            assert!(
                skipped.tiles_skipped > 0,
                "concurrency={concurrency}: at least one blank tile must be skipped, got {}",
                skipped.tiles_skipped
            );
            assert!(
                skip_files < full_files,
                "concurrency={concurrency}: skip_blanks must leave strictly fewer .raw files on \
                 disk: {skip_files} vs full {full_files}"
            );
            assert!(
                skipped.tiles_produced < full.tiles_produced,
                "concurrency={concurrency}: skip_blanks must hand strictly fewer tiles to the sink: \
                 {} vs full {}",
                skipped.tiles_produced,
                full.tiles_produced
            );
        }
    }
}

// ---------------------------------------------------------------------------
// `EngineKind::Auto` budget-aware routing (issue #135)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod auto_budget_routing_tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::sink::MemorySink;

    // A raster big enough that the monolithic peak (canvas + first downscaled
    // level) is comfortably larger than the tiny budgets used below.
    fn big_source() -> Raster {
        let data = vec![9u8; 128 * 128 * 3];
        Raster::new(128, 128, PixelFormat::Rgb8, data).unwrap()
    }

    fn big_plan() -> PyramidPlan {
        PyramidPlanner::new(128, 128, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    // Core reproducer for #135: with a memory budget too small for the
    // monolithic peak, `Auto` must flatten to the bounded-memory `Streaming`
    // engine rather than `Monolithic`. Before the fix `resolve_engine_kind`
    // returned `Monolithic` unconditionally for any raster source, ignoring
    // the documented budget — this asserts the corrected routing directly.
    #[test]
    fn auto_selects_streaming_when_budget_below_monolithic_peak() {
        let source = big_source();
        let plan = big_plan();
        let engine_source = EngineSource::Raster(&source);

        let peak = plan.estimate_peak_memory_for_format(source.format());
        assert!(peak > 2, "sanity: monolithic peak must be non-trivial");
        let tight_budget = peak / 2; // strictly below the monolithic peak

        let kind = resolve_engine_kind(EngineKind::Auto, &engine_source, &plan, Some(tight_budget));
        assert_eq!(
            kind,
            EngineKind::Streaming,
            "Auto must pick the bounded-memory engine when the budget \
             ({tight_budget}) cannot fit the monolithic peak ({peak})"
        );
    }

    // A budget that comfortably fits the monolithic peak keeps the fast
    // in-memory path.
    #[test]
    fn auto_keeps_monolithic_when_budget_fits() {
        let source = big_source();
        let plan = big_plan();
        let engine_source = EngineSource::Raster(&source);

        let peak = plan.estimate_peak_memory_for_format(source.format());
        let kind = resolve_engine_kind(EngineKind::Auto, &engine_source, &plan, Some(peak));
        assert_eq!(
            kind,
            EngineKind::Monolithic,
            "a budget >= the monolithic peak must keep the monolithic engine"
        );
    }

    // No explicit budget preserves the historical default: raster sources run
    // the monolithic engine.
    #[test]
    fn auto_defaults_to_monolithic_without_budget() {
        let source = big_source();
        let plan = big_plan();
        let engine_source = EngineSource::Raster(&source);

        let kind = resolve_engine_kind(EngineKind::Auto, &engine_source, &plan, None);
        assert_eq!(kind, EngineKind::Monolithic);
    }

    // End-to-end proof that the routing is honoured by `run()`: a budget too
    // small even for the streaming engine's minimum aligned strip must surface
    // `BudgetExceeded` under the default `Auto`. Only the streaming engine
    // consults the budget, so this error can *only* appear if `Auto` routed
    // away from the monolithic path. Before the fix `Auto` ran the monolithic
    // engine, which ignored the budget and completed the run (no error) — the
    // exact OOM-inducing behaviour #135 describes.
    #[test]
    fn auto_run_honors_budget_and_routes_to_streaming() {
        let source = big_source();
        let plan = big_plan();

        let result = EngineBuilder::new(&source, plan, MemorySink::new())
            .with_memory_budget(1_000)
            .run();

        match result {
            Err(EngineError::BudgetExceeded { budget_bytes, .. }) => {
                assert_eq!(budget_bytes, 1_000);
            }
            other => panic!(
                "Auto with a tiny budget must route to the bounded streaming \
                 engine and surface BudgetExceeded, got {other:?}"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Live-resume bookkeeping (issue #136)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod live_resume_bookkeeping_tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::resume::{JobCheckpoint, ResumePolicy};
    use crate::sink::FsSink;

    fn solid_source() -> Raster {
        // 8x8 RGB solid so every tile is non-blank and actually written.
        let data = vec![10u8; 8 * 8 * 3];
        Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap()
    }

    fn solid_plan() -> PyramidPlan {
        PyramidPlanner::new(8, 8, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    /// Resume is driven entirely by the builder's `ResumeAwareSink`, which used
    /// to skip only the *write* while never touching `levels_completed` and
    /// while the monolithic engine still credited `bytes_written` for every
    /// short-circuited skip (it sees the wrapper's `Ok` and cannot tell a real
    /// write from a skip).
    ///
    /// Before the fix (RED):
    ///   * `levels_completed` stayed empty after a completed run — the live
    ///     path never advanced it.
    ///   * a fully-resumed run reported `bytes_written > 0` even though it
    ///     emitted nothing, because the engine counted the skipped tiles.
    ///
    /// After the fix (GREEN): the builder promotes every fully-emitted level
    /// and subtracts the skipped tiles' bytes, so `levels_completed` is
    /// complete and a no-op resume reports `bytes_written == 0`.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_advances_levels_and_excludes_skipped_bytes() {
        let out = tempfile::tempdir().unwrap();
        let cp = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();
        let level_count = plan.levels.len();

        // Run 1: full render. Seeds the on-disk checkpoint (completed_tiles
        // and, after the fix, levels_completed).
        let sink1 = FsSink::new(out.path().to_path_buf(), plan.clone());
        let r1 = EngineBuilder::new(&source, plan.clone(), sink1)
            .with_engine(EngineKind::Monolithic)
            .with_resume(
                ResumePolicy::resume()
                    .with_checkpoint_root(cp.path())
                    .with_checkpoint_every(1),
            )
            .run()
            .expect("initial render must succeed");
        assert!(
            r1.bytes_written > 0,
            "the first run writes real tiles, so bytes_written must be non-zero"
        );

        // The live resume path must advance levels_completed for every level
        // whose tiles were all emitted.
        let meta = JobCheckpoint::load(cp.path())
            .expect("checkpoint load must not error")
            .expect("a resume run must leave a checkpoint on disk");
        assert_eq!(
            meta.levels_completed.len(),
            level_count,
            "every fully-emitted level must be recorded in levels_completed, \
             got {:?}",
            meta.levels_completed
        );

        // Run 2: resume over the same checkpoint. Every tile is already
        // recorded, so ResumeAwareSink short-circuits all writes.
        let sink2 = FsSink::new(out.path().to_path_buf(), plan.clone());
        let r2 = EngineBuilder::new(&source, plan.clone(), sink2)
            .with_engine(EngineKind::Monolithic)
            .with_resume(
                ResumePolicy::resume()
                    .with_checkpoint_root(cp.path())
                    .with_checkpoint_every(1),
            )
            .run()
            .expect("resume run must succeed");

        assert_eq!(
            r2.tiles_produced, 0,
            "a fully-resumed run must not report any tiles produced"
        );
        assert_eq!(
            r2.bytes_written, 0,
            "tiles skipped on resume must not be counted as bytes_written"
        );
    }
}

// ---------------------------------------------------------------------------
// Resume + dedupe/checksum seeding (issue #272 real fix)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod resume_dedupe_checksum_seeding_tests {
    //! Issue #272 real fix: `ResumeMode::Resume` combined with content
    //! deduplication or per-tile checksums used to be refused by the #450
    //! stopgap. It now runs, because [`ResumeAwareSink`] seeds the sink-side
    //! manifest / dedupe / checksum state for each skipped (pre-crash)
    //! coordinate via [`TileSink::seed_completed_tile`], so a resumed
    //! [`TileSink::finish`] reproduces the same complete `manifest.json`
    //! (`blank_references` + checksum table) and dedupe layout as an
    //! uninterrupted run.
    use super::*;
    use crate::checksum::ChecksumMode;
    use crate::dedupe::DedupeStrategy;
    use crate::manifest::{ChecksumAlgo, ManifestBuilder};
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::resume::ResumePolicy;
    use crate::sink::{FsSink, SinkError, Tile, TileSink};
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Mostly-white raster with a small central feature: most tiles are uniform
    /// white and collapse under `DedupeStrategy::Blanks` into one `_shared/`
    /// blob + many 1-byte placeholders recorded in `blank_references` — the
    /// exact state a resumed run must reconstruct.
    fn blank_heavy_source() -> Raster {
        let (w, h) = (128u32, 128u32);
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let mut data = vec![0xFFu8; w as usize * h as usize * bpp];
        for y in 56..72 {
            for x in 56..72 {
                let off = (y * w as usize + x) * bpp;
                data[off] = 0x10;
                data[off + 1] = 0x20;
                data[off + 2] = 0xF0;
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    fn blank_plan() -> PyramidPlan {
        PyramidPlanner::new(128, 128, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    fn solid_source() -> Raster {
        let data = vec![10u8; 8 * 8 * 3];
        Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap()
    }

    fn solid_plan() -> PyramidPlan {
        PyramidPlanner::new(8, 8, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    fn dedupe_checksum_sink(base: &Path, plan: &PyramidPlan) -> FsSink {
        FsSink::new(base.to_path_buf(), plan.clone())
            .with_dedupe(DedupeStrategy::Blanks)
            .with_manifest(ManifestBuilder::new())
            .with_checksums(ChecksumMode::EmitOnly, ChecksumAlgo::Blake3)
    }

    fn read_manifest_value(base: &Path) -> serde_json::Value {
        let raw = std::fs::read(base.join("manifest.json")).expect("manifest.json must exist");
        serde_json::from_slice(&raw).expect("manifest.json must be valid JSON")
    }

    fn map_field(m: &serde_json::Value, ptr: &[&str]) -> BTreeMap<String, String> {
        let mut cur = m;
        for k in ptr {
            match cur.get(*k) {
                Some(v) => cur = v,
                None => return BTreeMap::new(),
            }
        }
        cur.as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Wraps an inner sink and returns a terminal error after `ok` successful
    /// writes, modelling a crash midway through a run. Forwards `inner_sink`
    /// so every engine hook (checkpoint root, durability arming, seeding) reaches
    /// the wrapped `FsSink`.
    struct FailAfterSink<S: TileSink> {
        inner: S,
        ok_left: AtomicU32,
    }
    impl<S: TileSink> FailAfterSink<S> {
        fn new(inner: S, ok: u32) -> Self {
            Self {
                inner,
                ok_left: AtomicU32::new(ok),
            }
        }
    }
    impl<S: TileSink> TileSink for FailAfterSink<S> {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            if self.ok_left.load(Ordering::SeqCst) == 0 {
                return Err(SinkError::Other("deliberate mid-run failure".into()));
            }
            self.ok_left.fetch_sub(1, Ordering::SeqCst);
            self.inner.write_tile(tile)
        }
        fn finish(&self) -> Result<(), SinkError> {
            self.inner.finish()
        }
        fn inner_sink(&self) -> Option<&dyn TileSink> {
            Some(&self.inner)
        }
    }

    /// Recursively count files under `dir` whose length is exactly one byte
    /// (the deduped placeholders / blank markers).
    fn count_one_byte_files(dir: &Path) -> usize {
        let mut n = 0;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    n += count_one_byte_files(&p);
                } else if std::fs::metadata(&p).map(|m| m.len() == 1).unwrap_or(false) {
                    n += 1;
                }
            }
        }
        n
    }

    /// The headline #272 correctness test: a dedupe + per-tile-checksum run
    /// crashed partway (a terminal sink error persists the checkpoint) and
    /// resumed must reproduce the same `blank_references` map, checksum table
    /// and 1-byte-placeholder layout as a clean single-run reference.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn crash_resume_dedupe_checksum_reproduces_uninterrupted_manifest() {
        let src = blank_heavy_source();
        let plan = blank_plan();
        let total = plan.total_tile_count();
        assert!(total >= 8, "fixture must have enough tiles to crash midway");

        // --- reference: clean run to completion ---
        let ref_dir = tempfile::tempdir().unwrap();
        let ref_base = ref_dir.path().join("out");
        EngineBuilder::new(&src, plan.clone(), dedupe_checksum_sink(&ref_base, &plan))
            .with_engine(EngineKind::Monolithic)
            .with_config(EngineConfig::default().with_concurrency(1))
            .run()
            .expect("clean reference dedupe+checksum run must succeed");
        let ref_manifest = read_manifest_value(&ref_base);
        let ref_blanks = map_field(&ref_manifest, &["blank_references"]);
        let ref_checksums = map_field(&ref_manifest, &["checksums", "per_tile"]);
        assert!(
            !ref_blanks.is_empty(),
            "reference dedupe produced no blank_references — fixture is not deduping"
        );
        assert!(!ref_checksums.is_empty(), "reference recorded no checksums");

        // --- crash run: terminal error after ~half the tiles persists a
        // checkpoint on the engine's error path ---
        let crash_dir = tempfile::tempdir().unwrap();
        let crash_base = crash_dir.path().join("out");
        let crashing =
            FailAfterSink::new(dedupe_checksum_sink(&crash_base, &plan), total as u32 / 2);
        let crash_result = EngineBuilder::new(&src, plan.clone(), &crashing)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::resume().with_checkpoint_every(2))
            .with_config(EngineConfig::default().with_concurrency(1))
            .run();
        assert!(crash_result.is_err(), "the crash run must fail mid-way");

        // --- resume run into the same directory ---
        let resume_result =
            EngineBuilder::new(&src, plan.clone(), dedupe_checksum_sink(&crash_base, &plan))
                .with_engine(EngineKind::Monolithic)
                .with_resume(ResumePolicy::resume().with_checkpoint_every(2))
                .with_config(EngineConfig::default().with_concurrency(1))
                .run()
                .expect("resumed dedupe+checksum run must succeed (seeding, issue #272)");
        assert!(
            resume_result.tiles_produced < total,
            "resume produced all {total} tiles — it never resumed from a \
             checkpoint, so seeding was not exercised (produced {})",
            resume_result.tiles_produced
        );

        // --- resumed manifest + layout must match the reference exactly ---
        let crash_manifest = read_manifest_value(&crash_base);
        assert_eq!(
            map_field(&crash_manifest, &["blank_references"]),
            ref_blanks,
            "resumed blank_references differs — pre-crash placeholders orphaned (#272)"
        );
        assert_eq!(
            map_field(&crash_manifest, &["checksums", "per_tile"]),
            ref_checksums,
            "resumed checksum table differs — pre-crash digests dropped (#272)"
        );
        assert_eq!(
            count_one_byte_files(&crash_base),
            count_one_byte_files(&ref_base),
            "resumed run has a different number of 1-byte placeholders"
        );
    }

    /// Resume + sink-level dedupe on a fresh directory now runs (the #450
    /// stopgap is gone).
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_plus_sink_dedupe_now_runs() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        let sink =
            FsSink::new(out.path().to_path_buf(), plan.clone()).with_dedupe(DedupeStrategy::Blanks);
        let result = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::resume())
            .run();

        assert!(
            result.is_ok(),
            "resume + dedupe must run now that seeding lands (#272): {:?}",
            result.err()
        );
    }

    /// Resume + engine-level dedupe (`EngineBuilder::with_dedupe`) now runs.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_plus_engine_dedupe_now_runs() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone());
        let result = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_dedupe(DedupeStrategy::Blanks)
            .with_resume(ResumePolicy::resume())
            .run();

        assert!(
            result.is_ok(),
            "resume + engine-level dedupe must run now (#272): {:?}",
            result.err()
        );
    }

    /// Resume + per-tile checksums now runs.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_plus_checksum_now_runs() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone())
            .with_manifest(ManifestBuilder::new())
            .with_checksums(ChecksumMode::EmitOnly, ChecksumAlgo::Blake3);
        let result = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::resume())
            .run();

        assert!(
            result.is_ok(),
            "resume + checksums must run now that seeding lands (#272): {:?}",
            result.err()
        );
    }

    /// Resume with NEITHER dedupe nor checksums stays supported and keeps
    /// short-circuiting skipped tiles (no seeding work).
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_without_dedupe_or_checksum_still_runs() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone());
        let result = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::resume())
            .run();

        assert!(
            result.is_ok(),
            "resume without dedupe or checksums must remain supported: {:?}",
            result.err()
        );
    }

    /// A NON-resume run (Overwrite) with dedupe is unaffected.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn non_resume_dedupe_still_runs() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        let sink =
            FsSink::new(out.path().to_path_buf(), plan.clone()).with_dedupe(DedupeStrategy::Blanks);
        let result = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_dedupe(DedupeStrategy::Blanks)
            .with_resume(ResumePolicy::overwrite())
            .run();

        assert!(
            result.is_ok(),
            "a non-resume (Overwrite) dedupe run must still succeed: {:?}",
            result.err()
        );
    }

    /// Verify mode (read-only) with dedupe is unaffected. Seeds a pyramid first
    /// so Verify has something to audit.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn verify_dedupe_still_runs() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        let seed =
            FsSink::new(out.path().to_path_buf(), plan.clone()).with_dedupe(DedupeStrategy::Blanks);
        EngineBuilder::new(&source, plan.clone(), seed)
            .with_engine(EngineKind::Monolithic)
            .with_dedupe(DedupeStrategy::Blanks)
            .with_resume(ResumePolicy::overwrite())
            .run()
            .expect("seed overwrite+dedupe run should succeed");

        let sink =
            FsSink::new(out.path().to_path_buf(), plan.clone()).with_dedupe(DedupeStrategy::Blanks);
        let result = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_dedupe(DedupeStrategy::Blanks)
            .with_resume(ResumePolicy::verify())
            .run();

        assert!(
            result.is_ok(),
            "read-only Verify with dedupe must be unaffected: {:?}",
            result.err()
        );
    }
}

// ---------------------------------------------------------------------------
// Run-lock wiring (issue #126)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod run_lock_wiring_tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::resume::{ResumeError, ResumePolicy, RunLock};
    use crate::sink::FsSink;

    fn solid_source() -> Raster {
        // 8x8 RGB solid so every tile is non-blank and actually written.
        let data = vec![10u8; 8 * 8 * 3];
        Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap()
    }

    fn solid_plan() -> PyramidPlan {
        PyramidPlanner::new(8, 8, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    // Core of issue #126: while one job holds the advisory run lock on an
    // output directory, a second Overwrite job pointed at that same directory
    // must be refused (`ResumeError::Locked`), not allowed to run and wipe the
    // first job's live output.
    //
    // Before the wiring (RED): `EngineBuilder::run` never touched `RunLock`, so
    // the second Overwrite ran to completion and its wipe deleted the sentinel
    // file the "other job" had placed in the directory.
    //
    // After the wiring (GREEN): the run acquires the lock ahead of
    // `prepare_resume_state`, fails fast with `ResumeError::Locked`, and the
    // sentinel survives because the guarded wipe never executes.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn overwrite_is_refused_while_the_output_dir_is_locked() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        // Stand in for a live job that already owns the directory.
        let held = RunLock::acquire(out.path()).expect("first holder acquires the lock");

        // Make the directory "wipe-owned": a `.libviprs-job.json` marker means
        // the Overwrite wipe would consider the directory its own and proceed
        // to clear it. Absent the run lock this run would therefore delete the
        // sentinel below, which is exactly the clobber issue #126 prevents.
        std::fs::write(out.path().join(crate::resume::CHECKPOINT_FILENAME), b"{}").unwrap();

        // A sentinel representing the live job's in-flight output. A concurrent
        // Overwrite that runs its wipe would destroy it.
        let sentinel = out.path().join("live-output.dat");
        std::fs::write(&sentinel, b"do not clobber").unwrap();

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone());
        let err = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::overwrite())
            .run()
            .expect_err("a second job must be refused while the lock is held");

        match err {
            EngineError::ResumeFailed(ResumeError::Locked { path }) => {
                assert_eq!(
                    path,
                    RunLock::lock_path(out.path()),
                    "the refusal must name the lock file for this directory"
                );
            }
            other => panic!("expected ResumeFailed(Locked), got {other:?}"),
        }

        assert!(
            sentinel.exists(),
            "the refused Overwrite must not have wiped the live job's output"
        );

        drop(held);
    }

    // The guard must be released when the run finishes so the directory is free
    // for the next run. A successful Overwrite takes and drops the lock; a fresh
    // `RunLock::acquire` on the same directory afterwards must therefore succeed.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn run_releases_the_lock_when_it_finishes() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone());
        EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::overwrite())
            .run()
            .expect("an unlocked directory must run to completion");

        // If the run had leaked its guard, this non-blocking acquire would fail
        // with `Locked`.
        let reacquired = RunLock::acquire(out.path())
            .expect("the run must release its lock so a later job can acquire it");
        drop(reacquired);
    }

    // Issue #276: when the sink directory and the explicit checkpoint root
    // differ, the run must lock the CHECKPOINT directory (where the segment
    // appends and header renames actually land), not the sink output dir. The
    // lock dir must resolve with the SAME precedence as `resolve_checkpoint_root`
    // (explicit `checkpoint_root` wins over the sink's own dir); before the fix
    // the lock dir resolved sink-first, so a run with `checkpoint_root = cp/`
    // and a sink at `out/` locked `out/` and left `cp/` unguarded.
    //
    // Proof: a stand-in job holds the lock on the checkpoint dir `cp/`. A
    // builder run whose sink is at `out/` but whose checkpoint root is `cp/`
    // must be refused with `Locked` naming `cp/`'s lock file. Pre-fix the run
    // would have locked the free `out/` dir and proceeded (RED); post-fix it
    // collides on `cp/` and fails fast (GREEN).
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn run_locks_the_checkpoint_dir_not_the_sink_dir_when_they_differ() {
        let out = tempfile::tempdir().unwrap();
        let cp = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        // Cross-check the intended precedence directly against the resolver the
        // checkpoint I/O uses: with `checkpoint_root = cp/` and a sink at `out/`,
        // `resolve_checkpoint_root` yields `cp/`. The lock must guard exactly this
        // directory.
        let resolve_sink = FsSink::new(out.path().to_path_buf(), plan.clone());
        let cfg = EngineConfig::default().with_checkpoint_root(cp.path().to_path_buf());
        let resolved = crate::engine::resolve_checkpoint_root(&cfg, &resolve_sink)
            .expect("explicit checkpoint root resolves to Some");
        assert_eq!(
            resolved,
            cp.path(),
            "sanity: resolve_checkpoint_root prefers the explicit checkpoint root"
        );
        assert_ne!(
            RunLock::lock_path(cp.path()),
            RunLock::lock_path(out.path()),
            "sink and checkpoint dirs must be distinct for this test to be meaningful"
        );

        // Stand in for a live job already holding the lock on the CHECKPOINT dir.
        let held = RunLock::acquire(cp.path()).expect("first holder locks the checkpoint dir");

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone());
        let err = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::overwrite().with_checkpoint_root(cp.path().to_path_buf()))
            .run()
            .expect_err("the run must be refused: the checkpoint dir it locks is held");

        match err {
            EngineError::ResumeFailed(ResumeError::Locked { path }) => {
                assert_eq!(
                    path,
                    RunLock::lock_path(&resolved),
                    "the lock must guard the checkpoint dir (resolve_checkpoint_root), i.e. cp/"
                );
                assert_ne!(
                    path,
                    RunLock::lock_path(out.path()),
                    "the lock must NOT guard the sink dir out/ (the pre-fix, wrong-dir behavior)"
                );
            }
            other => panic!("expected ResumeFailed(Locked) on the checkpoint dir, got {other:?}"),
        }

        drop(held);
    }

    // Issues #362/#364/#365/#366 — cross-directory mutual exclusion, direction
    // one. A run mutates BOTH the checkpoint root and the sink's own output dir,
    // but moving the lock onto the checkpoint root (the #276 fix) left the sink
    // dir — the target of the Overwrite wipe (`prepare_resume_state` wipes
    // `sink.checkpoint_root()`) and of every tile write — unguarded whenever an
    // explicit `checkpoint_root` differs from it. That reopened issue #126
    // hazard (b): an Overwrite job locking only `cp/` could wipe the live output
    // of a concurrent job holding `out/`.
    //
    // Proof: a stand-in job holds the lock on the SINK dir `out/`, and `out/`
    // already carries a `.libviprs-job.json` marker (so an unguarded wipe would
    // consider it owned and clear it) plus a sentinel standing in for the live
    // job's in-flight output. A builder run with sink `out/` and checkpoint root
    // `cp/` must be refused with `Locked` naming `out/`'s lock file, and the
    // sentinel must survive.
    //
    // Pre-fix (RED): the run locked only the free `cp/`, then wiped `out/` and
    // destroyed the sentinel. Post-fix (GREEN): the run also locks `out/`,
    // collides, and fails fast before any wipe.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn overwrite_is_refused_while_the_sink_dir_is_locked_with_distinct_checkpoint_root() {
        let out = tempfile::tempdir().unwrap();
        let cp = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        assert_ne!(
            RunLock::lock_path(cp.path()),
            RunLock::lock_path(out.path()),
            "sink and checkpoint dirs must be distinct for this test to be meaningful"
        );

        // Stand in for a live job that already owns the SINK dir out/.
        let held = RunLock::acquire(out.path()).expect("first holder locks the sink dir");

        // Make out/ "wipe-owned" so an unguarded Overwrite wipe would proceed
        // and clear it, and drop a sentinel representing the live job's output.
        std::fs::write(out.path().join(crate::resume::CHECKPOINT_FILENAME), b"{}").unwrap();
        let sentinel = out.path().join("live-output.dat");
        std::fs::write(&sentinel, b"do not clobber").unwrap();

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone());
        let err = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::overwrite().with_checkpoint_root(cp.path().to_path_buf()))
            .run()
            .expect_err("the run must be refused: the sink dir it must wipe-guard is held");

        match err {
            EngineError::ResumeFailed(ResumeError::Locked { path }) => {
                assert_eq!(
                    path,
                    RunLock::lock_path(out.path()),
                    "the run must ALSO lock the sink dir out/ (the Overwrite wipe target), \
                     not just the checkpoint dir cp/"
                );
            }
            other => panic!("expected ResumeFailed(Locked) on the sink dir, got {other:?}"),
        }

        assert!(
            sentinel.exists(),
            "the refused Overwrite must not have wiped the live job's output in out/"
        );

        drop(held);
    }

    // Issues #362/#364/#365/#366 — cross-directory mutual exclusion, direction
    // two (the mirror). Locking both directories must NOT weaken the #276 guard:
    // holding the CHECKPOINT dir `cp/` must still refuse a run whose checkpoint
    // root is `cp/`, even though its sink lives at a distinct `out/`.
    //
    // Proof: a stand-in holds the lock on the checkpoint dir `cp/`. A builder
    // run with sink `out/` and checkpoint root `cp/` must be refused with
    // `Locked` naming `cp/`'s lock file, and a sentinel in `out/` must survive
    // (the run bails before `prepare_resume_state` ever wipes). This direction
    // passes both before and after the two-lock fix — it pins that #276 stays
    // closed — and together with the direction-one test above proves mutual
    // exclusion across the two dirs in BOTH directions.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn overwrite_is_refused_while_the_checkpoint_dir_is_locked_with_distinct_sink_dir() {
        let out = tempfile::tempdir().unwrap();
        let cp = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        assert_ne!(
            RunLock::lock_path(cp.path()),
            RunLock::lock_path(out.path()),
            "sink and checkpoint dirs must be distinct for this test to be meaningful"
        );

        // Stand in for a live job already holding the lock on the CHECKPOINT dir.
        let held = RunLock::acquire(cp.path()).expect("first holder locks the checkpoint dir");

        // A sentinel in out/ that a wipe would destroy — it must survive because
        // the run is refused before `prepare_resume_state` runs.
        std::fs::write(out.path().join(crate::resume::CHECKPOINT_FILENAME), b"{}").unwrap();
        let sentinel = out.path().join("live-output.dat");
        std::fs::write(&sentinel, b"do not clobber").unwrap();

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone());
        let err = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::overwrite().with_checkpoint_root(cp.path().to_path_buf()))
            .run()
            .expect_err("the run must be refused: the checkpoint dir it locks is held");

        match err {
            EngineError::ResumeFailed(ResumeError::Locked { path }) => {
                assert_eq!(
                    path,
                    RunLock::lock_path(cp.path()),
                    "the #276 guard on the checkpoint dir cp/ must still refuse the run"
                );
            }
            other => panic!("expected ResumeFailed(Locked) on the checkpoint dir, got {other:?}"),
        }

        assert!(
            sentinel.exists(),
            "the refused run must not have wiped the sentinel in out/"
        );

        drop(held);
    }

    // Self-lockout regression: an explicit `checkpoint_root` that names the SAME
    // physical directory as the sink through a DIFFERENT spelling must not make
    // the run refuse itself. The union-lock builds two entries — the resolved
    // checkpoint root and the sink dir — and de-duplicates them. A raw-`PathBuf`
    // dedup treats `out/` and its alias `out/.` as distinct, so `RunLock::acquire`
    // fires twice on the one `.libviprs-job.lock`; the second `try_lock` returns
    // `WouldBlock` and the run reports `ResumeError::Locked` against ITSELF —
    // a bogus failure on a supported single-directory config with no concurrent
    // job in sight.
    //
    // Proof: sink at `out/` (otherwise clean, no live holder) plus an Overwrite
    // policy whose checkpoint root is `out/../<out-basename>` — the same
    // directory, spelled through a parent round-trip. A `..` (ParentDir)
    // component is NOT normalised away by `Path`'s own equality (only a bare `.`
    // is), so the two raw paths compare unequal and a raw-`PathBuf` dedup keeps
    // both — while `canonicalize` resolves both to the one physical dir. Pre-fix
    // (RED) this self-collides and returns `Locked`. Post-fix (GREEN) the
    // canonical-key dedup collapses the aliases to a single lock and the run
    // completes.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn distinct_spelling_of_the_sink_dir_as_checkpoint_root_does_not_self_lock() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        // A different spelling of `out/` (a parent round-trip) that resolves to
        // the same physical dir but is NOT `Path`-equal to it.
        let alias = out.path().join("..").join(
            out.path()
                .file_name()
                .expect("tempdir has a final component"),
        );
        assert_ne!(
            RunLock::lock_path(&alias),
            RunLock::lock_path(out.path()),
            "the alias must be a distinct raw path for this test to exercise the dedup"
        );
        assert_eq!(
            std::fs::canonicalize(&alias).unwrap(),
            std::fs::canonicalize(out.path()).unwrap(),
            "the alias must canonicalize to the same physical directory as the sink"
        );

        // No live holder: the only reason a lock could collide here is the run
        // fighting itself over its own directory.
        let sink = FsSink::new(out.path().to_path_buf(), plan.clone());
        EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::overwrite().with_checkpoint_root(alias))
            .run()
            .expect("a self-aliased checkpoint root must run to completion, not self-lock");

        // And the single lock must have been released on completion.
        let reacquired = RunLock::acquire(out.path())
            .expect("the run must release its lock so a later job can acquire it");
        drop(reacquired);
    }
}

// ---------------------------------------------------------------------------
// Resume on-disk side-effect tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod resume_side_effect_tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::resume::{JobCheckpoint, JobMetadata, ResumePolicy};
    use crate::sink::{FsSink, SinkError, Tile, TileFormat, TileSink};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Recursively list every regular file under `root`, relative to `root`,
    /// sorted. Used to assert a refused run has zero on-disk side effects and
    /// to byte-compare a resumed pyramid against a clean one.
    fn list_files(root: &Path) -> Vec<PathBuf> {
        fn walk(root: &Path, cur: &Path, out: &mut Vec<PathBuf>) {
            let Ok(entries) = std::fs::read_dir(cur) else {
                return;
            };
            for entry in entries.filter_map(Result::ok) {
                let p = entry.path();
                if p.is_dir() {
                    walk(root, &p, out);
                } else {
                    out.push(p.strip_prefix(root).unwrap().to_path_buf());
                }
            }
        }
        let mut out = Vec::new();
        walk(root, root, &mut out);
        out.sort();
        out
    }

    /// A raster with gradient content so every tile is non-blank and distinct.
    fn gradient_source(w: u32, h: u32) -> Raster {
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let mut data = vec![0u8; w as usize * h as usize * bpp];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let off = (y * w as usize + x) * bpp;
                data[off] = (x % 251) as u8;
                data[off + 1] = (y % 241) as u8;
                data[off + 2] = ((x ^ y) % 239) as u8;
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// `EngineBuilder::run_region` is the builder-ergonomic equivalent of the
    /// free `generate_pyramid_region`: byte-identical tiles for the same crop.
    #[test]
    fn builder_run_region_matches_generate_pyramid_region() {
        use crate::engine::{EngineConfig, generate_pyramid_region};
        use crate::sink::MemorySink;

        let src = gradient_source(256, 256);
        let (left, top, w, h) = (0u32, 0u32, 100u32, 100u32);
        let plan = PyramidPlanner::new(w, h, 64, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let cfg = EngineConfig::default();

        let free_sink = MemorySink::new();
        let free = generate_pyramid_region(&src, &plan, &free_sink, &cfg, left, top, w, h).unwrap();

        let built_sink = MemorySink::new();
        let built = EngineBuilder::new(&src, plan.clone(), &built_sink)
            .run_region(left, top, w, h)
            .unwrap();

        assert_eq!(free.tiles_produced, built.tiles_produced);
        let mut a = free_sink.tiles();
        let mut b = built_sink.tiles();
        a.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));
        b.sort_by_key(|t| (t.coord.level, t.coord.row, t.coord.col));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.coord, y.coord);
            assert_eq!(x.data, y.data);
        }
    }

    // A Resume refused on a plan-hash mismatch must leave the output
    // directory EXACTLY as it found it: no tiles, no lock file, no segment
    // log. The plan-hash gate therefore runs before the run lock (whose
    // acquisition creates `.libviprs-job.lock`) and before any engine work.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn refused_plan_hash_mismatch_leaves_the_directory_untouched() {
        let out = tempfile::tempdir().unwrap();
        let source = gradient_source(64, 64);
        let plan = PyramidPlanner::new(64, 64, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

        // A pre-existing checkpoint whose hash cannot match the current plan.
        let stale = JobMetadata::new("f".repeat(64), "1970-01-01T00:00:00Z".into());
        JobCheckpoint::save(out.path(), &stale).unwrap();
        let before = list_files(out.path());

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone()).with_format(TileFormat::Png);
        let err = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::resume())
            .run()
            .expect_err("a stale plan hash must refuse to resume");
        assert!(
            matches!(err, EngineError::PlanHashMismatch { .. }),
            "expected PlanHashMismatch, got {err:?}"
        );

        let after = list_files(out.path());
        assert_eq!(
            before, after,
            "a refused Resume must not write tiles, a lock file, or any other file"
        );
    }

    // The plan-hash gate runs BEFORE the run lock is acquired: with a live
    // holder on the directory AND a stale checkpoint, the mismatch refusal
    // must win. Pre-fix the lock was taken first, so this returned
    // ResumeFailed(Locked) and, when the directory was free, materialised
    // `.libviprs-job.lock` before refusing.
    #[test]
    #[cfg_attr(miri, ignore)] // file locking unsupported under Miri isolation
    fn plan_hash_gate_refuses_before_the_run_lock_is_taken() {
        let out = tempfile::tempdir().unwrap();
        let source = gradient_source(64, 64);
        let plan = PyramidPlanner::new(64, 64, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

        let stale = JobMetadata::new("f".repeat(64), "1970-01-01T00:00:00Z".into());
        JobCheckpoint::save(out.path(), &stale).unwrap();

        // Another live job owns the directory for the whole attempt.
        let held = crate::resume::RunLock::acquire(out.path()).expect("test holder acquires");

        let sink = FsSink::new(out.path().to_path_buf(), plan.clone()).with_format(TileFormat::Png);
        let err = EngineBuilder::new(&source, plan.clone(), sink)
            .with_engine(EngineKind::Monolithic)
            .with_resume(ResumePolicy::resume())
            .run()
            .expect_err("stale plan hash must refuse to resume");
        assert!(
            matches!(err, EngineError::PlanHashMismatch { .. }),
            "the plan-hash gate must fire before lock acquisition, got {err:?}"
        );

        drop(held);
    }

    /// Sink wrapper that panics on the Nth `write_tile`, simulating a mid-run
    /// crash. Bookkeeping (checkpoint root, content format, ...) forwards to
    /// the wrapped sink through `inner_sink`.
    struct PanickingSink<S: TileSink> {
        inner: S,
        panic_at: usize,
        writes: AtomicUsize,
    }

    impl<S: TileSink> TileSink for PanickingSink<S> {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            let n = self.writes.fetch_add(1, Ordering::SeqCst) + 1;
            if n == self.panic_at {
                panic!("PanickingSink: deliberate crash at write #{n}");
            }
            self.inner.write_tile(tile)
        }
        fn finish(&self) -> Result<(), SinkError> {
            self.inner.finish()
        }
        fn inner_sink(&self) -> Option<&dyn TileSink> {
            Some(&self.inner)
        }
    }

    // Resume determinism under concurrency: a run that crashes partway and is
    // then resumed at tile_concurrency 4 must land a pyramid byte-identical
    // to a single clean run. Only the checkpoint header and segment log (pure
    // resume bookkeeping) may differ; everything else, including the absence
    // of stray lock or temp files, is compared byte for byte.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn crash_and_resume_at_concurrency_four_matches_a_clean_run_byte_for_byte() {
        let source = gradient_source(96, 64);
        let plan = PyramidPlanner::new(96, 64, 32, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let config = EngineConfig::default().with_concurrency(4);

        // Clean single-run reference.
        let ref_dir = tempfile::tempdir().unwrap();
        let ref_base = ref_dir.path().join("pyramid");
        let ref_sink = FsSink::new(ref_base.clone(), plan.clone()).with_format(TileFormat::Png);
        EngineBuilder::new(&source, plan.clone(), ref_sink)
            .with_engine(EngineKind::Monolithic)
            .with_config(config.clone())
            .run()
            .expect("clean reference run succeeds");

        // Crashing first run: panic partway through, with per-tile checkpoint
        // flushes so the resume genuinely skips prior work instead of
        // re-rendering everything.
        let crash_dir = tempfile::tempdir().unwrap();
        let crash_base = crash_dir.path().join("pyramid");
        let panic_at = (plan.total_tile_count() as usize / 3).max(2);
        let panicking = PanickingSink {
            inner: FsSink::new(crash_base.clone(), plan.clone()).with_format(TileFormat::Png),
            panic_at,
            writes: AtomicUsize::new(0),
        };
        let first_run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            EngineBuilder::new(&source, plan.clone(), &panicking)
                .with_engine(EngineKind::Monolithic)
                .with_config(config.clone())
                .with_resume(ResumePolicy::resume().with_checkpoint_every(1))
                .run()
        }));
        assert!(first_run.is_err(), "the first run must crash mid-pyramid");

        // Resume into the same directory.
        let resume_sink =
            FsSink::new(crash_base.clone(), plan.clone()).with_format(TileFormat::Png);
        EngineBuilder::new(&source, plan.clone(), resume_sink)
            .with_engine(EngineKind::Monolithic)
            .with_config(config.clone())
            .with_resume(ResumePolicy::resume().with_checkpoint_every(1))
            .run()
            .expect("the resume run completes the pyramid");

        // Byte-compare the two trees, ignoring only the resume bookkeeping
        // files. A leftover `.libviprs-job.lock` (or a stray temp file) shows
        // up as a set mismatch and fails the assertion.
        let bookkeeping = [
            std::ffi::OsStr::new(crate::resume::CHECKPOINT_FILENAME),
            std::ffi::OsStr::new(crate::resume::SEGMENTS_FILENAME),
        ];
        let keep =
            |p: &PathBuf| -> bool { !bookkeeping.contains(&p.file_name().unwrap_or_default()) };
        let ref_files: Vec<PathBuf> = list_files(&ref_base).into_iter().filter(keep).collect();
        let crash_files: Vec<PathBuf> = list_files(&crash_base).into_iter().filter(keep).collect();
        assert_eq!(
            ref_files, crash_files,
            "crash+resume must produce exactly the clean run's file set"
        );
        for rel in &ref_files {
            let a = std::fs::read(ref_base.join(rel)).unwrap();
            let b = std::fs::read(crash_base.join(rel)).unwrap();
            assert_eq!(
                a,
                b,
                "crash+resume diverges from the clean run at {}",
                rel.display()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Resume-wrapper forwarding tests (issue #137)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod resume_wrapper_forwarding_tests {
    use super::resume::ResumeAwareSink;
    use crate::planner::{Layout, PyramidPlanner, TileCoord};
    use crate::sink::{FsSink, TileFormat, TileSink};
    use std::collections::HashSet;

    // `ResumeAwareSink` is a transparent decorator: it must expose the inner
    // sink's on-disk format so the resume plan hash never loses the format
    // when writes are routed through the wrapper. Before #137's remainder the
    // wrapper hand-forwarded a subset of bookkeeping methods and simply omitted
    // `content_format`, so this returned `None` even though the inner `FsSink`
    // commits to JPEG. Overriding `inner_sink()` closes that silent gap for
    // every bookkeeping hook at once, not just the ones remembered by hand.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_aware_sink_forwards_content_format() {
        let dir = tempfile::tempdir().unwrap();
        let plan = PyramidPlanner::new(4, 4, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let fs = FsSink::new(dir.path().to_path_buf(), plan)
            .with_format(TileFormat::Jpeg { quality: 80 });
        let skip: HashSet<TileCoord> = HashSet::new();
        let wrapped = ResumeAwareSink::new(&fs, &skip, None);

        assert_eq!(
            wrapped.content_format(),
            Some(TileFormat::Jpeg { quality: 80 }),
            "a transparent resume wrapper must expose the inner sink's on-disk format"
        );
    }

    // The wrapper must likewise surface the inner sink's checkpoint root and
    // the inner sink's format through the single `inner_sink()` hook rather
    // than through a per-method forward that a future edit could drop.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_aware_sink_forwards_checkpoint_root() {
        let dir = tempfile::tempdir().unwrap();
        let plan = PyramidPlanner::new(4, 4, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan();
        let fs = FsSink::new(dir.path().to_path_buf(), plan);
        let skip: HashSet<TileCoord> = HashSet::new();
        let wrapped = ResumeAwareSink::new(&fs, &skip, None);

        assert_eq!(
            wrapped.checkpoint_root(),
            Some(dir.path()),
            "a transparent resume wrapper must expose the inner sink's checkpoint root"
        );
    }
}

// ---------------------------------------------------------------------------
// Explicit checkpoint-root opt-in (issue #137 decision record)
// ---------------------------------------------------------------------------
//
// Issue #137's remainder asked whether `EngineConfig::checkpoint_root` (and
// its `ResumePolicy::with_checkpoint_root` mirror) became redundant once the
// `TileSink::inner_sink` decorator hook (#220/#239) made every in-tree
// wrapper forward `checkpoint_root()` automatically. The answer, recorded
// here, is no: the hook only fixes wrappers the crate owns. An external
// wrapper written by user code can forward nothing but the data path, and
// the explicit root is the one public knob that keeps resume and Verify
// working through it. These tests pin the opt-in as an intentional
// capability; deleting the field or either builder breaks them.

#[cfg(test)]
mod checkpoint_root_public_optin_tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};
    use crate::raster::Raster;
    use crate::resume::{JobCheckpoint, ResumePolicy};
    use crate::sink::{FsSink, SinkError, Tile, TileSink};

    /// Models a third-party wrapper the crate cannot patch: it forwards only
    /// the data path (`write_tile` / `finish`) and overrides neither
    /// `inner_sink()` nor any bookkeeping hook, so `checkpoint_root()` and
    /// `content_format()` fall back to the terminal `None` defaults even
    /// though real tiles land on disk through the inner [`FsSink`].
    struct OpaqueThirdPartySink {
        inner: FsSink,
    }

    impl TileSink for OpaqueThirdPartySink {
        fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
            self.inner.write_tile(tile)
        }
        fn finish(&self) -> Result<(), SinkError> {
            self.inner.finish()
        }
    }

    fn solid_source() -> Raster {
        // 8x8 RGB solid so every tile is non-blank and actually written.
        let data = vec![10u8; 8 * 8 * 3];
        Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap()
    }

    fn solid_plan() -> PyramidPlan {
        PyramidPlanner::new(8, 8, 2, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    fn opaque_sink(out: &std::path::Path, plan: &PyramidPlan) -> OpaqueThirdPartySink {
        OpaqueThirdPartySink {
            inner: FsSink::new(out.to_path_buf(), plan.clone()),
        }
    }

    /// Verify through an opaque wrapper: refused without the opt-in, clean
    /// with it. The refusal half is the counterfactual that proves the
    /// trait-level fix (#220/#239) does NOT make the explicit root
    /// redundant: the wrapper genuinely hides the on-disk root from the
    /// engine.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn verify_through_opaque_wrapper_requires_and_honors_explicit_root() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        // Seed real tiles on disk through the opaque wrapper.
        EngineBuilder::new(&source, plan.clone(), opaque_sink(out.path(), &plan))
            .run()
            .expect("seeding run must succeed");

        // Without the opt-in the engine cannot locate the tiles.
        let err = EngineBuilder::new(&source, plan.clone(), opaque_sink(out.path(), &plan))
            .with_resume(ResumePolicy::verify())
            .run()
            .expect_err("Verify through an opaque wrapper with no explicit root must fail");
        assert!(
            matches!(err, EngineError::VerifyRequiresOnDiskSink),
            "expected VerifyRequiresOnDiskSink, got {err:?}"
        );

        // With the public opt-in the same opaque wrapper verifies cleanly.
        let result = EngineBuilder::new(&source, plan.clone(), opaque_sink(out.path(), &plan))
            .with_resume(ResumePolicy::verify().with_checkpoint_root(out.path()))
            .run()
            .expect("Verify with the explicit checkpoint root must succeed");
        assert_eq!(result.tiles_produced, 0, "Verify is read-only");
    }

    /// The `EngineConfig::checkpoint_root` field itself (not just the
    /// `ResumePolicy` mirror) must keep driving the fallback: `with_config`
    /// threads it into an attached policy that carries no root of its own.
    /// This is the exact field the issue proposed retiring.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn engine_config_checkpoint_root_drives_verify_through_opaque_wrapper() {
        let out = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        EngineBuilder::new(&source, plan.clone(), opaque_sink(out.path(), &plan))
            .run()
            .expect("seeding run must succeed");

        let cfg = EngineConfig::default().with_checkpoint_root(out.path().to_path_buf());
        let result = EngineBuilder::new(&source, plan.clone(), opaque_sink(out.path(), &plan))
            .with_resume(ResumePolicy::verify())
            .with_config(cfg)
            .run()
            .expect("EngineConfig::checkpoint_root must drive Verify through the opaque wrapper");
        assert_eq!(result.tiles_produced, 0, "Verify is read-only");
    }

    /// Resume through an opaque wrapper with an explicit root that sits
    /// apart from the output directory: the checkpoint must land at the
    /// explicit root even though the wrapper hides the sink's directory,
    /// and a second run over the same root must skip every recorded tile.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn resume_through_opaque_wrapper_uses_explicit_root() {
        let out = tempfile::tempdir().unwrap();
        let cp = tempfile::tempdir().unwrap();
        let source = solid_source();
        let plan = solid_plan();

        // Run 1: full resumable render.
        let r1 = EngineBuilder::new(&source, plan.clone(), opaque_sink(out.path(), &plan))
            .with_resume(
                ResumePolicy::resume()
                    .with_checkpoint_root(cp.path())
                    .with_checkpoint_every(1),
            )
            .run()
            .expect("initial resumable render must succeed");
        assert!(r1.tiles_produced > 0, "the first run renders real tiles");

        let meta = JobCheckpoint::load(cp.path())
            .expect("checkpoint load must not error")
            .expect("the explicit root must receive the checkpoint despite the opaque wrapper");
        assert_eq!(
            meta.completed_tiles.len() as u64,
            plan.total_tile_count(),
            "the checkpoint must record every completed tile"
        );

        // Run 2: resume over the same explicit root; every tile short-circuits.
        let r2 = EngineBuilder::new(&source, plan.clone(), opaque_sink(out.path(), &plan))
            .with_resume(
                ResumePolicy::resume()
                    .with_checkpoint_root(cp.path())
                    .with_checkpoint_every(1),
            )
            .run()
            .expect("resume run must succeed");
        assert_eq!(
            r2.tiles_produced, 0,
            "a fully-recorded checkpoint must short-circuit every write"
        );
    }
}
