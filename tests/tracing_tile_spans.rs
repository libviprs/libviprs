//! Per-tile tracing-span integration coverage (issue libviprs-tests#83).
//!
//! Under the `tracing` feature, `generate_pyramid` must emit one span
//! named `tile` with target `libviprs` (so its `target::name` renders as
//! `libviprs::tile`) for every tile it produces, and each such span must
//! carry the tile's `x` / `y` / `level` coordinates as fields.
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

use libviprs::{
    EngineBuilder, EngineConfig, EngineKind, Layout, MemorySink, PixelFormat, PyramidPlan,
    PyramidPlanner, Raster,
};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::subscriber::with_default;
use tracing::{Event, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::{LookupSpan, Registry};

/// A single captured span: its `target::name` and the fields recorded at
/// creation time (stored as their `Debug`/typed string form).
#[derive(Debug, Clone)]
struct CapturedSpan {
    qualified_name: String,
    fields: HashMap<String, String>,
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
    fn on_new_span(&self, attrs: &Attributes<'_>, _id: &Id, _ctx: Context<'_, S>) {
        let meta = attrs.metadata();
        let qualified_name = format!("{}::{}", meta.target(), meta.name());
        let mut fields = HashMap::new();
        attrs.record(&mut FieldCollector(&mut fields));
        self.spans
            .lock()
            .expect("span collector mutex poisoned")
            .push(CapturedSpan {
                qualified_name,
                fields,
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
    let expected = result.tiles_produced + result.tiles_skipped;
    assert_eq!(
        tile_spans, expected,
        "expected {expected} libviprs::tile spans (tiles_produced + tiles_skipped), got {tile_spans}",
    );
    assert!(expected > 0, "sanity: run should have produced tiles");
}

#[test]
fn emits_one_tile_span_per_tile_parallel() {
    // Positive concurrency => parallel producer/consumer emit path.
    let config = EngineConfig::default().with_concurrency(4);
    let (spans, _plan, result) = run_capturing_spans(512, 384, 128, config);

    let tile_spans = count_named(&spans, "libviprs::tile") as u64;
    let expected = result.tiles_produced + result.tiles_skipped;
    assert_eq!(
        tile_spans, expected,
        "expected {expected} libviprs::tile spans on the parallel path, got {tile_spans}",
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
