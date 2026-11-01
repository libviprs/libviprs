//! Issue #1417 (slice of #1048, N7.4): progress, cancellation and backpressure. A conversion tells
//! an observer its stages, stops at the next checkpoint when the token fires and leaves nothing
//! behind, and runs its classifier and consumer over a bounded queue whose depth it reports. Needs
//! `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, ProvenanceStub, ProviderCaps,
    ProviderError, RawEntity, RawGeometry, SourceUnit,
};
use libviprs::pvec_convert::{PvecConversion, PvecConvertError};
use libviprs::{CancelToken, CollectingObserver, EngineEvent, EngineObserver};

/// 5000 circles, or 300 entities the provider does not map plus one circle, by the bytes.
struct Mock {
    refused: bool,
    caps: ProviderCaps,
}

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

fn stub(handle: u64, class: &str) -> ProvenanceStub {
    ProvenanceStub {
        handle: Some(ItemHandle::new(handle)),
        class_name: class.into(),
        block_path: InstancePath::root(),
        layer: "0".into(),
    }
}

fn circle(n: u64) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Circle {
            center: [3.0 * n as f64, 0.0, 0.0],
            radius: 1.0,
            normal: [0.0, 0.0, 1.0],
        },
        provenance: stub(n + 1, "CIRCLE"),
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(true)
    }

    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        Ok(Mock {
            refused: matches!(source, CadSource::Bytes(b"refused")),
            caps: Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
                c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
            }),
        })
    }

    fn name(&self) -> &'static str {
        "mock"
    }

    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }

    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(vec![LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        }])
    }

    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(Vec::new())
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let EntityScope::Layout(_) = scope else {
            return Err(ProviderError::NoSuchScope("no blocks".into()));
        };
        let items: Vec<RawEntity> = if self.refused {
            std::iter::once(circle(0))
                .chain((1..=300).map(|n| RawEntity {
                    style: EntityStyle::absent(),
                    geometry: RawGeometry::Unmapped {
                        reason: "hatch".into(),
                    },
                    provenance: stub(n + 1, "HATCH"),
                }))
                .collect()
        } else {
            (0..5000).map(circle).collect()
        };
        Ok(Box::new(Cursor(items.into_iter())))
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1417-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn conv<'a>(bytes: &'a [u8], dest: &'a Path) -> PvecConversion<'a> {
    PvecConversion::new(CadSource::Bytes(bytes), dest).unit(SourceUnit::Meters)
}

/// Cancels the token when the first batch is reported, and keeps what it heard.
struct CancelOnBatch {
    token: CancelToken,
    heard: Mutex<Vec<EngineEvent>>,
}

impl EngineObserver for CancelOnBatch {
    fn on_event(&self, event: EngineEvent) {
        if matches!(event, EngineEvent::ConversionBatchNormalized { .. }) {
            self.token.cancel();
        }
        self.heard.lock().unwrap().push(event);
    }
}

/// Sleeps on every diagnostic, which the consumer reports: a deliberately stalled consumer.
struct Slow;

impl EngineObserver for Slow {
    fn on_event(&self, event: EngineEvent) {
        if matches!(event, EngineEvent::ConversionDiagnostic { .. }) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_a_precancelled_run_returns_cancelled_creates_no_file_and_publishes_nothing() {
    let dir = scratch("pre");
    let dest = dir.join("o.pvec");
    let seen = Arc::new(CollectingObserver::new());
    let token = CancelToken::new();
    token.cancel();
    let err = conv(b"many", &dest)
        .observer(seen.clone())
        .cancel(token)
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Cancelled));
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    assert!(!seen.events().contains(&EngineEvent::ConversionPublished));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_a_mid_run_cancel_stops_within_a_batch_and_leaves_no_file() {
    let dir = scratch("mid");
    let dest = dir.join("o.pvec");
    let token = CancelToken::new();
    let observer = Arc::new(CancelOnBatch {
        token: token.clone(),
        heard: Mutex::new(Vec::new()),
    });
    let err = conv(b"many", &dest)
        .queue_capacity(4)
        .observer(observer.clone())
        .cancel(token)
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Cancelled), "{err:?}");
    assert_eq!(
        fs::read_dir(&dir).unwrap().count(),
        0,
        "no .pvec, report or sidecar"
    );
    let heard = observer.heard.lock().unwrap();
    let normalized = heard
        .iter()
        .filter_map(|e| match e {
            EngineEvent::ConversionBatchNormalized { entities } => Some(*entities),
            _ => None,
        })
        .max()
        .expect("at least one batch was reported");
    assert!(normalized < 5000, "{normalized} of 5000 were normalized");
    assert!(!heard.contains(&EngineEvent::ConversionPublished));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_a_cancel_after_the_artifact_is_written_still_publishes_nothing() {
    struct CancelOnWritten(CancelToken);
    impl EngineObserver for CancelOnWritten {
        fn on_event(&self, event: EngineEvent) {
            if matches!(event, EngineEvent::ConversionArtifactWritten { .. }) {
                self.0.cancel();
            }
        }
    }
    let dir = scratch("late");
    let dest = dir.join("o.pvec");
    let token = CancelToken::new();
    let err = conv(b"refused", &dest)
        .observer(Arc::new(CancelOnWritten(token.clone())))
        .cancel(token)
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Cancelled), "{err:?}");
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_a_stalled_consumer_never_sees_more_than_the_configured_capacity_queued() {
    let dir = scratch("stall");
    for capacity in [1usize, 3, 8] {
        let dest = dir.join(format!("o{capacity}.pvec"));
        let out = conv(b"refused", &dest)
            .queue_capacity(capacity)
            .observer(Arc::new(Slow))
            .convert_with::<Mock>()
            .unwrap();
        assert!(
            out.peak_queue_depth as usize <= capacity,
            "peak {} over capacity {capacity}",
            out.peak_queue_depth
        );
        assert_eq!(out.report.diagnostics.len(), 300);
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_two_runs_over_the_same_input_report_the_same_events_in_the_same_order() {
    let dir = scratch("events");
    let run = |name: &str| {
        let dest = dir.join(name);
        let seen = Arc::new(CollectingObserver::new());
        conv(b"refused", &dest)
            .queue_capacity(2)
            .observer(seen.clone())
            .convert_with::<Mock>()
            .unwrap();
        seen.events()
    };
    let (a, b) = (run("a.pvec"), run("b.pvec"));
    assert_eq!(a, b);
    let names: Vec<&str> = a.iter().map(EngineEvent::name).collect();
    assert_eq!(names.first(), Some(&"conversion_source_opened"));
    assert_eq!(names.get(1), Some(&"conversion_view_resolved"));
    assert_eq!(names[names.len() - 2], "conversion_artifact_written");
    assert_eq!(names[names.len() - 1], "conversion_published");
    assert!(names.contains(&"conversion_batch_normalized"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_the_events_carry_the_counts_of_the_run() {
    let dir = scratch("counts");
    let dest = dir.join("o.pvec");
    let seen = Arc::new(CollectingObserver::new());
    let out = conv(b"many", &dest)
        .observer(seen.clone())
        .convert_with::<Mock>()
        .unwrap();
    let events = seen.events();
    assert!(events.contains(&EngineEvent::ConversionSourceOpened {
        provider: "mock".into()
    }));
    assert!(events.contains(&EngineEvent::ConversionViewResolved {
        included: 5000,
        excluded: 0
    }));
    let batches: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            EngineEvent::ConversionBatchNormalized { entities } => Some(*entities),
            _ => None,
        })
        .collect();
    assert_eq!(batches.len(), 5000 / 256);
    assert_eq!(batches.last(), Some(&((5000 / 256) * 256)));
    assert!(batches.windows(2).all(|w| w[1] == w[0] + 256));
    assert!(matches!(
        events[events.len() - 2],
        EngineEvent::ConversionArtifactWritten { entities: 5000, .. }
    ));
    assert_eq!(out.written, 5000);
    let _ = fs::remove_dir_all(&dir);
}
