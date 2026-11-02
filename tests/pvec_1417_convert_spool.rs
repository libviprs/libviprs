//! Issue #1417 (slice of #1048, N7.4) and #1314 (N6.3): a conversion that spills. With a spill root
//! the artifact is streamed through PVec's spill writer into the partial file, comes out byte for
//! byte as the in-memory conversion's, and a cancel leaves no spool file. Needs `--features
//! pvec-convert`.

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
use libviprs::{CancelToken, EngineEvent, EngineObserver};

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

struct CancelWhenWritten(CancelToken);

impl EngineObserver for CancelWhenWritten {
    fn on_event(&self, event: EngineEvent) {
        if matches!(event, EngineEvent::ConversionArtifactWritten { .. }) {
            self.0.cancel();
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_a_spilled_conversion_writes_the_artifact_and_report_of_the_in_memory_one() {
    let (a, b, root) = (scratch("memory"), scratch("spilled"), scratch("root"));
    let memory = conv(&a.join("o.pvec")).convert_with::<Mock>().unwrap();
    let spilled = conv(&b.join("o.pvec"))
        .spill(spill(&root))
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(
        fs::read(a.join("o.pvec")).unwrap(),
        fs::read(b.join("o.pvec")).unwrap()
    );
    assert_eq!(
        memory.report.to_json_string().unwrap(),
        spilled.report.to_json_string().unwrap()
    );
    assert_eq!(memory.written, spilled.written);
    assert_eq!(names(&a), names(&b));
    assert!(names(&root).is_empty(), "{:?}", names(&root));
    for d in [a, b, root] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_a_spill_root_that_cannot_hold_files_is_refused_not_ignored() {
    let dir = scratch("badroot");
    let not_a_dir = dir.join("file");
    fs::write(&not_a_dir, b"x").unwrap();
    let err = conv(&dir.join("o.pvec"))
        .spill(spill(&not_a_dir.join("below")))
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Write(_)), "{err:?}");
    assert_eq!(names(&dir), ["file"]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_a_cancel_after_the_spilled_write_leaves_no_spool_file_and_no_artifact() {
    let (dir, root) = (scratch("cancel"), scratch("cancelroot"));
    let token = CancelToken::new();
    let err = conv(&dir.join("o.pvec"))
        .spill(spill(&root))
        .cancel(token.clone())
        .observer(Arc::new(CancelWhenWritten(token)))
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Cancelled), "{err:?}");
    assert!(names(&dir).is_empty(), "{:?}", names(&dir));
    assert!(names(&root).is_empty(), "{:?}", names(&root));
    for d in [dir, root] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1417_a_precancelled_spilled_run_leaves_no_file_and_no_spool() {
    let (dir, root) = (scratch("pre"), scratch("preroot"));
    let token = CancelToken::new();
    token.cancel();
    let err = conv(&dir.join("o.pvec"))
        .spill(spill(&root))
        .cancel(token)
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Cancelled));
    assert!(names(&dir).is_empty() && names(&root).is_empty());
    for d in [dir, root] {
        let _ = fs::remove_dir_all(d);
    }
}
