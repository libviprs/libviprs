//! Issue #1463 (follow-up to #1048, N7.4): a conversion tells its observer when each chunk of the
//! artifact is written, and a cancel that fires on a chunk stops the write there, in the flush
//! as well as between entities. These are red until `write_geometry_to` calls back per chunk and
//! `pvec_convert` forwards it as an event. The event is matched by its `name()`, so the file
//! compiles before the variant exists. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::writer::{LeafConfig, SpillConfig};
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
    let d = std::env::temp_dir().join(format!("libviprs-1417s-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn leaves() -> LeafConfig {
    LeafConfig {
        cell_capacity: 64,
        tier0_max_error: Some(0.05),
    }
}

fn conv<'a>(dest: &'a Path) -> PvecConversion<'a> {
    PvecConversion::new(CadSource::Bytes(b"many"), dest)
        .unit(SourceUnit::Meters)
        .leaves(leaves())
}

fn spill(root: &Path) -> SpillConfig {
    SpillConfig {
        root: root.to_path_buf(),
        budget: 4096,
    }
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

const CHUNK: &str = "conversion_chunk_written";

/// Cancels when the `at`th chunk event arrives, counting from 1.
struct CancelOnChunk {
    token: CancelToken,
    at: usize,
    seen: std::sync::atomic::AtomicUsize,
}

impl EngineObserver for CancelOnChunk {
    fn on_event(&self, event: EngineEvent) {
        if event.name() == CHUNK
            && self.seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1 == self.at
        {
            self.token.cancel();
        }
    }
}

/// Every event of one spilled run, as `name` plus its fields, in order.
fn run_events(tag: &str) -> Vec<String> {
    let (dir, root) = (scratch(tag), scratch(&format!("{tag}root")));
    let observer = Arc::new(CollectingObserver::new());
    conv(&dir.join("o.pvec"))
        .spill(spill(&root))
        .observer(observer.clone())
        .convert_with::<Mock>()
        .unwrap();
    for d in [dir, root] {
        let _ = fs::remove_dir_all(d);
    }
    observer
        .events()
        .iter()
        .map(|e| format!("{} {e:?}", e.name()))
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1463_a_spilled_conversion_reports_each_chunk_before_the_artifact_is_written() {
    let events = run_events("chunks");
    let chunks: Vec<usize> = (0..events.len())
        .filter(|&i| events[i].starts_with(CHUNK))
        .collect();
    assert!(
        chunks.len() >= 2,
        "5000 entities in a 4 KiB budget are more than one chunk, saw {} chunk events",
        chunks.len()
    );
    let written = events
        .iter()
        .position(|e| e.starts_with("conversion_artifact_written"))
        .expect("the artifact is written");
    assert!(
        chunks.iter().all(|&i| i < written),
        "every chunk is reported before the artifact is: {events:#?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1463_two_runs_see_the_same_chunk_events_in_the_same_order() {
    let (a, b) = (run_events("det-a"), run_events("det-b"));
    let only = |v: &[String]| -> Vec<String> {
        v.iter().filter(|e| e.starts_with(CHUNK)).cloned().collect()
    };
    assert!(!only(&a).is_empty(), "no chunk event at all");
    assert_eq!(only(&a), only(&b));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1463_a_cancel_on_the_second_chunk_stops_there_and_leaves_no_spool_file_and_no_artifact() {
    let (dir, root) = (scratch("cancel2"), scratch("cancel2root"));
    let token = CancelToken::new();
    let observer = Arc::new(CancelOnChunk {
        token: token.clone(),
        at: 2,
        seen: Default::default(),
    });
    let result = conv(&dir.join("o.pvec"))
        .spill(spill(&root))
        .cancel(token)
        .observer(observer.clone())
        .convert_with::<Mock>();
    let err = result.expect_err("the cancel on the second chunk has to stop the run");
    assert!(matches!(err, PvecConvertError::Cancelled), "{err:?}");
    assert_eq!(
        observer.seen.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "no chunk is written after the one that cancelled"
    );
    assert!(names(&dir).is_empty(), "{:?}", names(&dir));
    assert!(names(&root).is_empty(), "{:?}", names(&root));
    for d in [dir, root] {
        let _ = fs::remove_dir_all(d);
    }
}

/// The first chunks are pushes; the last one is written by the flush. Cancelling on it is the cancel
/// the existing test does not reach, which fires after the whole spilled write.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1463_a_cancel_on_the_last_chunk_of_the_flush_leaves_no_spool_file_and_no_artifact() {
    let total = run_events("count")
        .iter()
        .filter(|e| e.starts_with(CHUNK))
        .count();
    assert!(total >= 2, "saw {total} chunk events");

    let (dir, root) = (scratch("cancelN"), scratch("cancelNroot"));
    let token = CancelToken::new();
    let observer = Arc::new(CancelOnChunk {
        token: token.clone(),
        at: total,
        seen: Default::default(),
    });
    let err = conv(&dir.join("o.pvec"))
        .spill(spill(&root))
        .cancel(token)
        .observer(observer)
        .convert_with::<Mock>()
        .expect_err("a cancel on the last chunk has to stop the run");
    assert!(matches!(err, PvecConvertError::Cancelled), "{err:?}");
    assert!(names(&dir).is_empty(), "{:?}", names(&dir));
    assert!(names(&root).is_empty(), "{:?}", names(&root));
    for d in [dir, root] {
        let _ = fs::remove_dir_all(d);
    }
}
