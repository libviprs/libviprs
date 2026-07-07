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
use crate::retry::{FailurePolicy, RetryPolicy, RetryingSink};
use crate::sink::TileSink;
use crate::streaming::{
    BudgetPolicy, RasterStripSource, StreamingConfig, StripSource, generate_pyramid_streaming,
};
use crate::streaming_mapreduce::{MapReduceConfig, generate_pyramid_mapreduce};

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

    // EngineConfig knobs
    concurrency: Option<usize>,
    buffer_size: Option<usize>,
    background_rgb: Option<[u8; 3]>,
    blank_strategy: Option<BlankTileStrategy>,
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

    /// Select which engine implementation [`EngineBuilder::run`] will
    /// dispatch to. Defaults to [`EngineKind::Auto`].
    ///
    /// **See also:** [interactive example](https://libviprs.org/cli/#flag-parallel).
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
    /// into an *existing* [`ResumePolicy`] — but only when one has already
    /// been attached (via [`EngineBuilder::with_resume`]). It never fabricates
    /// a policy: [`ResumeMode::Overwrite`] is destructive (it wipes the sink
    /// dir), so carrying a checkpoint cadence or root on the config must not
    /// silently enable it. Callers that want a resumable run choose the mode
    /// explicitly; without one, the checkpoint knobs are simply inert.
    pub fn with_config(mut self, config: EngineConfig) -> Self {
        self.concurrency = Some(config.concurrency);
        self.buffer_size = Some(config.buffer_size);
        self.background_rgb = Some(config.background_rgb);
        self.blank_strategy = Some(config.blank_tile_strategy);
        self.failure_policy = Some(config.failure_policy);
        if let Some(ds) = config.dedupe_strategy {
            self.dedupe = Some(ds);
        }
        if config.cancel.is_some() {
            self.cancel = config.cancel;
        }
        // Carry the checkpoint knobs into an EXPLICITLY-chosen ResumePolicy
        // only, so migrations from `generate_pyramid_resumable(.., &cfg, mode)`
        // don't silently lose the cadence / root that used to live on the
        // config. If no policy was attached we leave `resume` as `None`
        // rather than conjuring a destructive Overwrite (issue #123).
        if let Some(mut policy) = self.resume.take() {
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
            cancel,
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
        // Thread the cooperative-cancellation token onto the config so every
        // engine driver (and the streaming config that embeds it) polls it.
        engine_cfg.cancel = cancel.clone();

        let observer_ref: &dyn EngineObserver = match &observer {
            Some(arc) => arc.as_ref(),
            None => &NoopObserver,
        };

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

        let kind = resolve_engine_kind(engine_kind, &source);

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
            // Route resume writes through the retry wrapper (when configured)
            // so a transient failure is retried before the resume checkpoint
            // records the tile as complete.
            let wrapped = resume::ResumeAwareSink::new(engine_sink, &skip, cp.as_ref());

            // Capture the run outcome rather than propagating it with `?`
            // straight away: the checkpoint must be flushed on *both* the
            // success and the error path (see the flush below).
            let run_result: Result<EngineResult, EngineError> = match (kind, source) {
                (EngineKind::Monolithic, EngineSource::Raster(raster)) => {
                    generate_pyramid_observed(raster, &plan, &wrapped, &engine_cfg, observer_ref)
                }
                (EngineKind::Streaming, EngineSource::Raster(raster)) => {
                    let strip = RasterStripSource::new(raster);
                    let cfg =
                        build_streaming_config(engine_cfg, memory_budget_bytes, budget_policy);
                    generate_pyramid_streaming(&strip, &plan, &wrapped, &cfg, observer_ref)
                }
                (EngineKind::Streaming, EngineSource::Strip(strip)) => {
                    let cfg =
                        build_streaming_config(engine_cfg, memory_budget_bytes, budget_policy);
                    generate_pyramid_streaming(strip.as_ref(), &plan, &wrapped, &cfg, observer_ref)
                }
                (EngineKind::MapReduce, EngineSource::Raster(raster)) => {
                    let strip = RasterStripSource::new(raster);
                    let cfg = build_mapreduce_config(
                        memory_budget_bytes,
                        concurrency,
                        buffer_size,
                        background_rgb,
                        blank_strategy,
                        cancel.clone(),
                    );
                    generate_pyramid_mapreduce(&strip, &plan, &wrapped, &cfg, observer_ref)
                }
                (EngineKind::MapReduce, EngineSource::Strip(strip)) => {
                    let cfg = build_mapreduce_config(
                        memory_budget_bytes,
                        concurrency,
                        buffer_size,
                        background_rgb,
                        blank_strategy,
                        cancel.clone(),
                    );
                    generate_pyramid_mapreduce(strip.as_ref(), &plan, &wrapped, &cfg, observer_ref)
                }
                (EngineKind::Monolithic, EngineSource::Strip(_)) => {
                    unreachable!("Monolithic + Strip rejected above")
                }
                (EngineKind::Auto, _) => {
                    unreachable!("Auto should have been resolved before match")
                }
            };

            // Persist the checkpoint whether the run succeeded or failed.
            // On the error path this is the only chance to make completed
            // tiles durable: an interrupted run (kill -9 / OOM / preemption
            // surfaced as a sink error, or an exhausted retry budget) would
            // otherwise drop the in-memory `CheckpointState` here and force a
            // later `--resume` to re-render everything. On the failure path a
            // checkpoint I/O error is swallowed so the original engine error
            // — the reason the run stopped — is what propagates to the
            // caller.
            if let Some(cp) = cp.as_ref() {
                match &run_result {
                    Ok(_) => cp.flush().map_err(EngineError::ResumeFailed)?,
                    Err(_) => {
                        let _ = cp.flush();
                    }
                }
            }

            let mut result = run_result?;

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
            return Ok((result, sink));
        }

        let result = match (kind, source) {
            (EngineKind::Monolithic, EngineSource::Raster(raster)) => {
                generate_pyramid_observed(raster, &plan, engine_sink, &engine_cfg, observer_ref)?
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
                generate_pyramid_streaming(&source, &plan, engine_sink, &cfg, observer_ref)?
            }
            (EngineKind::Streaming, EngineSource::Strip(source)) => {
                let cfg = build_streaming_config(engine_cfg, memory_budget_bytes, budget_policy);
                generate_pyramid_streaming(source.as_ref(), &plan, engine_sink, &cfg, observer_ref)?
            }
            (EngineKind::MapReduce, EngineSource::Raster(raster)) => {
                let source = RasterStripSource::new(raster);
                let cfg = build_mapreduce_config(
                    memory_budget_bytes,
                    concurrency,
                    buffer_size,
                    background_rgb,
                    blank_strategy,
                    cancel.clone(),
                );
                generate_pyramid_mapreduce(&source, &plan, engine_sink, &cfg, observer_ref)?
            }
            (EngineKind::MapReduce, EngineSource::Strip(source)) => {
                let cfg = build_mapreduce_config(
                    memory_budget_bytes,
                    concurrency,
                    buffer_size,
                    background_rgb,
                    blank_strategy,
                    cancel.clone(),
                );
                generate_pyramid_mapreduce(source.as_ref(), &plan, engine_sink, &cfg, observer_ref)?
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
    cfg.cancel = cancel;
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
            // it only ever holds our own `.libviprs-job.json`. Remove just
            // that file so a stale checkpoint can't make this fresh run look
            // resumable — leaving every other entry in place.
            if let Some(root) = &config.checkpoint_root {
                let same_as_sink = sink
                    .checkpoint_root()
                    .is_some_and(|s| s == root.as_path());
                if !same_as_sink {
                    let marker = root.join(crate::resume::CHECKPOINT_FILENAME);
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
            let cp = crate::engine::cp_for_sink(sink, plan, config, Vec::new(), Vec::new());
            Ok((std::collections::HashSet::new(), cp))
        }
        ResumeMode::Resume => {
            let contract = crate::resume::PlanContract::from_engine(config, sink);
            let expected_hash = crate::resume::compute_plan_hash(plan, &contract);
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
                    .map_err(SinkError::Checkpoint)?;
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

        fn applies_retry_policy(&self) -> bool {
            self.inner.applies_retry_policy()
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
}
