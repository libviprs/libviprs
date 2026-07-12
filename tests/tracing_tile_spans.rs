//! Per-tile tracing-span integration coverage (issue libviprs-tests#83).
//!
//! Under the `tracing` feature, `generate_pyramid` must emit one span
//! named `tile` with target `libviprs` (so its `target::name` renders as
//! `libviprs::tile`) for every tile it attempts to write (one span per tile
//! write attempt), and each such span must carry the tile's `x` / `y` /
//! `level` coordinates as fields and nest under the active `libviprs::level`
//! (and, transitively, `libviprs::pipeline`) span.
//!
//! These are the in-repo mirror of the tests-repo `phase3_tracing`
//! span assertions (`emits_span_per_tile`, `tile_span_carries_coords`).
//! They fail on `main` because the tile path emits a trace *event*, not a
//! span, so the subscriber's `on_new_span` never fires for a tile.
//!
//! The whole module is gated on the `tracing` feature; without it the
//! default build never references `tracing` or `tracing-subscriber`.

#![cfg(feature = "tracing")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use libviprs::{
    EngineBuilder, EngineConfig, EngineKind, FailurePolicy, Layout, MemorySink, PixelFormat,
    PyramidPlan, PyramidPlanner, Raster, RetryPolicy, SinkError, Tile, TileSink,
};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::subscriber::with_default;
use tracing::{Event, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::{LookupSpan, Registry};

/// A single captured span: its `target::name`, the fields recorded at
/// creation time (stored as their `Debug`/typed string form), and the
/// qualified names of its ancestors (immediate parent first, walking up to
/// the root) so a test can pin the span's place in the tree.
#[derive(Debug, Clone)]
struct CapturedSpan {
    qualified_name: String,
    fields: HashMap<String, String>,
    ancestors: Vec<String>,
}

/// Thread-safe collector of span records, shareable into a subscriber.
#[derive(Default, Clone)]
struct SpanCollector {
    spans: Arc<Mutex<Vec<CapturedSpan>>>,
}

impl SpanCollector {
    fn new() -> Self {
        Self::default()
    }

    fn snapshot(&self) -> Vec<CapturedSpan> {
        self.spans
            .lock()
            .expect("span collector mutex poisoned")
            .clone()
    }
}

struct FieldCollector<'a>(&'a mut HashMap<String, String>);

impl Visit for FieldCollector<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
}

impl<S> Layer<S> for SpanCollector
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let meta = attrs.metadata();
        let qualified_name = format!("{}::{}", meta.target(), meta.name());
        let mut fields = HashMap::new();
        attrs.record(&mut FieldCollector(&mut fields));

        // Walk the ancestry (parent -> grandparent -> ... -> root) so a test
        // can assert the tile span's nesting without relying on span ids. The
        // registry has already recorded this span's parent by the time our
        // layer's `on_new_span` runs.
        let mut ancestors = Vec::new();
        if let Some(span_ref) = ctx.span(id) {
            let mut cursor = span_ref.parent();
            while let Some(parent) = cursor {
                let pmeta = parent.metadata();
                ancestors.push(format!("{}::{}", pmeta.target(), pmeta.name()));
                cursor = parent.parent();
            }
        }

        self.spans
            .lock()
            .expect("span collector mutex poisoned")
            .push(CapturedSpan {
                qualified_name,
                fields,
                ancestors,
            });
    }

    fn on_event(&self, _event: &Event<'_>, _ctx: Context<'_, S>) {}
}

/// A deterministic non-uniform gradient raster so no tile is detected as
/// blank (which keeps `tiles_skipped == 0` under the default `Emit`
/// strategy and makes `tiles_produced` the exact tile count).
fn gradient_raster(w: u32, h: u32) -> Raster {
    let mut data = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        for x in 0..w {
            data.push((x & 0xff) as u8);
            data.push((y & 0xff) as u8);
            data.push(((x ^ y) & 0xff) as u8);
        }
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).expect("gradient raster construction")
}

/// Runs a Monolithic pyramid with `config`, capturing every span emitted
/// during the run. Returns the captured spans, the plan, and the result.
fn run_capturing_spans(
    w: u32,
    h: u32,
    tile_size: u32,
    config: EngineConfig,
) -> (Vec<CapturedSpan>, PyramidPlan, libviprs::EngineResult) {
    let src = gradient_raster(w, h);
    let planner =
        PyramidPlanner::new(w, h, tile_size, 0, Layout::DeepZoom).expect("planner construction");
    let plan = planner.plan();
    let sink = MemorySink::new();

    let collector = SpanCollector::new();
    let subscriber = Registry::default().with(collector.clone());

    let plan_for_run = plan.clone();
    let result = with_default(subscriber, || {
        EngineBuilder::new(&src, plan_for_run, &sink)
            .with_engine(EngineKind::Monolithic)
            .with_config(config)
            .run()
            .expect("pyramid run")
    });

    (collector.snapshot(), plan, result)
}

fn count_named(spans: &[CapturedSpan], name: &str) -> usize {
    spans.iter().filter(|s| s.qualified_name == name).count()
}

#[test]
fn emits_one_tile_span_per_tile_single_threaded() {
    // Default config => concurrency 0 => single-threaded emit path.
    let (spans, _plan, result) = run_capturing_spans(512, 384, 128, EngineConfig::default());

    let tile_spans = count_named(&spans, "libviprs::tile") as u64;
    // One span per tile write attempt. On the all-success path that equals
    // `tiles_produced`, which already counts every written tile, blanks
    // included: a blank tile bumps `tiles_skipped` and, once its placeholder
    // write succeeds, `tiles_produced` too. Adding `tiles_skipped` would
    // double-count blanks; it only ever matched because this gradient fixture
    // forces `tiles_skipped == 0`.
    let expected = result.tiles_produced;
    assert_eq!(
        tile_spans, expected,
        "expected {expected} libviprs::tile spans (one per tile write attempt = tiles_produced), got {tile_spans}",
    );
    assert!(expected > 0, "sanity: run should have produced tiles");
}

#[test]
fn emits_one_tile_span_per_tile_parallel() {
    // Positive concurrency => parallel producer/consumer emit path.
    let config = EngineConfig::default().with_concurrency(4);
    let (spans, _plan, result) = run_capturing_spans(512, 384, 128, config);

    let tile_spans = count_named(&spans, "libviprs::tile") as u64;
    // One span per tile write attempt = `tiles_produced` on the all-success
    // path (see the single-threaded case: `+ tiles_skipped` would double-count
    // blanks, and only coincides here because the fixture keeps it at 0).
    let expected = result.tiles_produced;
    assert_eq!(
        tile_spans, expected,
        "expected {expected} libviprs::tile spans (one per tile write attempt = tiles_produced) on the parallel path, got {tile_spans}",
    );
    assert!(expected > 0, "sanity: run should have produced tiles");
}

#[test]
fn tile_span_carries_coordinates() {
    let (spans, _plan, _result) = run_capturing_spans(256, 256, 128, EngineConfig::default());

    let tile_spans: Vec<&CapturedSpan> = spans
        .iter()
        .filter(|s| s.qualified_name == "libviprs::tile")
        .collect();
    assert!(
        !tile_spans.is_empty(),
        "expected at least one libviprs::tile span",
    );

    for span in tile_spans {
        for required in &["x", "y", "level"] {
            assert!(
                span.fields.contains_key(*required),
                "libviprs::tile span missing `{required}` field; recorded fields: {:?}",
                span.fields,
            );
        }
    }
}

#[test]
fn tile_span_nests_under_level_and_pipeline() {
    // Single-threaded path: the tile span is created on the same thread that
    // entered the level and pipeline spans, so its ancestry is deterministic.
    // This pins the advertised tile -> level -> pipeline nesting contract.
    let (spans, _plan, _result) = run_capturing_spans(256, 256, 128, EngineConfig::default());

    let tile_spans: Vec<&CapturedSpan> = spans
        .iter()
        .filter(|s| s.qualified_name == "libviprs::tile")
        .collect();
    assert!(
        !tile_spans.is_empty(),
        "expected at least one libviprs::tile span to check its nesting",
    );

    for span in tile_spans {
        assert_eq!(
            span.ancestors.first().map(String::as_str),
            Some("libviprs::level"),
            "a libviprs::tile span's parent must be libviprs::level; ancestry: {:?}",
            span.ancestors,
        );
        assert_eq!(
            span.ancestors.get(1).map(String::as_str),
            Some("libviprs::pipeline"),
            "a libviprs::tile span's grandparent must be libviprs::pipeline; ancestry: {:?}",
            span.ancestors,
        );
    }
}

/// A sink whose every `write_tile` fails with a transient error, so under
/// `FailurePolicy::RetryThenSkip` each tile exhausts its retries and is
/// skipped without ever producing output.
struct AlwaysFailSink;

impl TileSink for AlwaysFailSink {
    fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
        Err(SinkError::Other("always fails".into()))
    }
}

#[test]
fn tile_span_fires_for_retry_then_skip_exhausted_write() {
    // The tile span is entered *before* `sink.write_tile`, so it fires once
    // per tile write attempt, not only per produced tile. A tile whose write
    // exhausts `RetryThenSkip` produces no output (it never bumps
    // `tiles_produced`), yet its span must still reach the subscriber. This
    // pins that the span tracks write attempts, which is why the count model
    // is `tiles_produced` on the success path rather than a narrower "only
    // fully-succeeded tiles" notion.
    let src = gradient_raster(256, 256);
    let planner =
        PyramidPlanner::new(256, 256, 128, 0, Layout::DeepZoom).expect("planner construction");
    let plan = planner.plan();

    let collector = SpanCollector::new();
    let subscriber = Registry::default().with(collector.clone());

    // A fast, jitter-free retry schedule so the doomed writes exhaust quickly.
    let policy = RetryPolicy::new(2, Duration::from_micros(1))
        .with_multiplier(1.0)
        .with_max_backoff(Duration::from_micros(10))
        .with_jitter(false);

    let sink = AlwaysFailSink;
    let result = with_default(subscriber, || {
        EngineBuilder::new(&src, plan, &sink)
            .with_engine(EngineKind::Monolithic)
            .with_failure_policy(FailurePolicy::RetryThenSkip(policy))
            .run()
            .expect("RetryThenSkip must complete the run, not surface an error")
    });

    let tile_spans = count_named(&collector.snapshot(), "libviprs::tile") as u64;
    assert_eq!(
        result.tiles_produced, 0,
        "every tile write fails under this sink, so no tile should be produced",
    );
    assert!(
        tile_spans > 0,
        "a RetryThenSkip-exhausted write must still emit its libviprs::tile span; got {tile_spans}",
    );
}
