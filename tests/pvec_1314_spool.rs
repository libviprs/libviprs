//! Issue #1314 (N6.3): the spool sink. The funnel streams the artifact to a sink through PVec's
//! `with_spill` and `finish_to` (pvec#235), the publish writes it straight into the partial file,
//! and a cancel during the flush leaves no spool file. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use libviprs::CancelToken;
use libviprs::cad::ItemHandle;
use libviprs::pvec::publish::{
    PublishError, Stage, StreamedPublishError, open_published, publish, publish_streamed,
    report_path,
};
use libviprs::pvec::writer::{
    LeafConfig, SpillConfig, WriteConfig, WriteError, write_geometry, write_geometry_to,
};
use libviprs::pvec::{
    EntityId, EntityStyle, FidelityClass, InstancePath, Normalized2DGeometry, OrderingPolicy,
    ProvenanceStub, RawEntity, RawGeometry, SourceUnit, order_entities,
};

const EXACT: FidelityClass = FidelityClass::SupportedExact;

fn ids(n: usize) -> Vec<EntityId> {
    let raw = (0..n)
        .map(|i| RawEntity {
            style: EntityStyle::absent(),
            geometry: RawGeometry::Line {
                start: [i as f64, 0.0, 0.0],
                end: [i as f64 + 1.0, 0.0, 0.0],
            },
            provenance: ProvenanceStub {
                handle: Some(ItemHandle::new(i as u64 + 1)),
                class_name: "LINE".into(),
                block_path: InstancePath::root(),
                layer: "0".into(),
            },
        })
        .collect();
    order_entities("t", "1", &OrderingPolicy::new(), raw)
        .entities
        .into_iter()
        .map(|o| o.id)
        .collect()
}

fn drawing(n: usize) -> Vec<Normalized2DGeometry> {
    (0..n)
        .map(|i| {
            let x = (i % 100) as f64 * 10.0;
            let y = (i / 100) as f64 * 10.0;
            Normalized2DGeometry::polyline(&[[x, y], [x + 4.0, y + 3.0]], EXACT).unwrap()
        })
        .collect()
}

fn indexed() -> WriteConfig {
    WriteConfig {
        unit: SourceUnit::Millimeters,
        leaves: Some(LeafConfig {
            cell_capacity: 64,
            tier0_max_error: Some(0.05),
        }),
    }
}

fn spill_in(root: &Path, budget: u64) -> SpillConfig {
    SpillConfig {
        root: root.to_path_buf(),
        budget,
    }
}

fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// A sink that cancels the token once `after` bytes have gone through it.
struct CancelAfter<'a> {
    after: usize,
    seen: usize,
    token: &'a CancelToken,
    kept: Vec<u8>,
}

impl Write for CancelAfter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.seen += buf.len();
        self.kept.extend_from_slice(buf);
        if self.seen >= self.after {
            self.token.cancel();
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1314_the_streamed_artifact_is_the_in_memory_one_for_every_budget() {
    let geoms = drawing(3000);
    let all = ids(geoms.len());
    let memory = write_geometry(
        &indexed(),
        all.iter().copied().zip(&geoms),
        Vec::new(),
        &CancelToken::new(),
    )
    .unwrap();
    for budget in [0u64, 1, 4096, 1 << 20, 64 << 20] {
        let root = tempfile::tempdir().unwrap();
        let mut sink = Vec::new();
        let streamed = write_geometry_to(
            &indexed(),
            all.iter().copied().zip(&geoms),
            Vec::new(),
            &spill_in(root.path(), budget),
            &CancelToken::new(),
            &mut sink,
        )
        .unwrap();
        assert_eq!(sink, memory.bytes, "budget {budget}");
        assert_eq!(streamed.length, sink.len() as u64);
        assert_eq!(
            (streamed.written, streamed.skipped),
            (memory.written, memory.skipped)
        );
        assert_eq!(streamed.source_ids, memory.source_ids);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1314_the_spill_files_stay_under_the_root_and_are_gone_after() {
    let geoms = drawing(2000);
    let all = ids(geoms.len());
    let root = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    let mut sink = Vec::new();
    write_geometry_to(
        &indexed(),
        all.into_iter().zip(&geoms),
        Vec::new(),
        &spill_in(root.path(), 4096),
        &CancelToken::new(),
        &mut sink,
    )
    .unwrap();
    assert!(
        entries(root.path()).is_empty(),
        "{:?}",
        entries(root.path())
    );
    assert!(entries(elsewhere.path()).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1314_a_cancel_during_the_flush_returns_cancelled_and_leaves_no_spool_file() {
    let geoms = drawing(4000);
    let all = ids(geoms.len());
    for after in [1usize, 512, 4096] {
        let root = tempfile::tempdir().unwrap();
        let token = CancelToken::new();
        let mut sink = CancelAfter {
            after,
            seen: 0,
            token: &token,
            kept: Vec::new(),
        };
        let r = write_geometry_to(
            &indexed(),
            all.iter().copied().zip(&geoms),
            Vec::new(),
            &spill_in(root.path(), 4096),
            &token,
            &mut sink,
        );
        assert!(
            matches!(r, Err(WriteError::Cancelled)),
            "after {after}: {r:?}"
        );
        assert!(
            entries(root.path()).is_empty(),
            "after {after}: {:?}",
            entries(root.path())
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1314_a_failing_sink_is_a_pvec_error_with_the_io_error_underneath_and_no_spool_left() {
    struct Full(usize);
    impl Write for Full {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.0 < buf.len() {
                return Err(io::Error::other("disk full"));
            }
            self.0 -= buf.len();
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let geoms = drawing(2000);
    let root = tempfile::tempdir().unwrap();
    let r = write_geometry_to(
        &indexed(),
        ids(geoms.len()).into_iter().zip(&geoms),
        Vec::new(),
        &spill_in(root.path(), 4096),
        &CancelToken::new(),
        &mut Full(64),
    );
    let err = r.expect_err("the sink fills up");
    assert!(matches!(err, WriteError::Pvec(_)), "{err:?}");
    assert!(std::error::Error::source(&err).is_some());
    assert!(entries(root.path()).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1314_a_streamed_publish_writes_the_pair_and_leaves_nothing_else() {
    let geoms = drawing(2000);
    let all = ids(geoms.len());
    let dir = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    let memory = write_geometry(
        &indexed(),
        all.iter().copied().zip(&geoms),
        Vec::new(),
        &CancelToken::new(),
    )
    .unwrap();
    publish_streamed(
        &dest,
        b"{\"v\":1}",
        &CancelToken::new(),
        &mut |_| {},
        |out| {
            write_geometry_to(
                &indexed(),
                all.iter().copied().zip(&geoms),
                Vec::new(),
                &spill_in(root.path(), 4096),
                &CancelToken::new(),
                out,
            )
            .map(|_| ())
        },
    )
    .unwrap();
    assert_eq!(fs::read(&dest).unwrap(), memory.bytes);
    assert_eq!(fs::read(report_path(&dest)).unwrap(), b"{\"v\":1}");
    assert_eq!(entries(dir.path()).len(), 2, "{:?}", entries(dir.path()));
    assert!(entries(root.path()).is_empty());
    assert!(open_published(&dest).is_ok());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1314_a_cancel_during_the_flush_of_a_streamed_publish_keeps_the_previous_pair() {
    let geoms = drawing(4000);
    let all = ids(geoms.len());
    let dir = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    let old = write_geometry(
        &indexed(),
        all.iter().copied().take(10).zip(&geoms),
        Vec::new(),
        &CancelToken::new(),
    )
    .unwrap()
    .bytes;
    publish(&dest, &old, b"old", &CancelToken::new(), &mut |_| {}).unwrap();

    let token = CancelToken::new();
    let r = publish_streamed(&dest, b"new", &token, &mut |_| {}, |out| {
        struct Tap<'a, 'b> {
            inner: &'a mut dyn Write,
            token: &'b CancelToken,
            seen: usize,
        }
        impl Write for Tap<'_, '_> {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.seen += buf.len();
                if self.seen > 256 {
                    self.token.cancel();
                }
                self.inner.write(buf)
            }
            fn flush(&mut self) -> io::Result<()> {
                self.inner.flush()
            }
        }
        write_geometry_to(
            &indexed(),
            all.iter().copied().zip(&geoms),
            Vec::new(),
            &spill_in(root.path(), 4096),
            &token,
            &mut Tap {
                inner: out,
                token: &token,
                seen: 0,
            },
        )
        .map(|_| ())
    });
    assert!(
        matches!(r, Err(StreamedPublishError::Produce(WriteError::Cancelled))),
        "{r:?}"
    );
    assert_eq!(fs::read(&dest).unwrap(), old);
    assert_eq!(fs::read(report_path(&dest)).unwrap(), b"old");
    assert_eq!(entries(dir.path()).len(), 2, "{:?}", entries(dir.path()));
    assert!(
        entries(root.path()).is_empty(),
        "{:?}",
        entries(root.path())
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1314_a_streamed_publish_refuses_bytes_that_are_not_an_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    let r: Result<(), StreamedPublishError<io::Error>> =
        publish_streamed(&dest, b"r", &CancelToken::new(), &mut |_| {}, |out| {
            out.write_all(b"not an artifact")
        });
    assert!(
        matches!(
            r,
            Err(StreamedPublishError::Publish(PublishError::NotAnArtifact(
                _
            )))
        ),
        "{r:?}"
    );
    assert!(entries(dir.path()).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1314_a_producer_error_is_returned_as_the_producers_and_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    let mut seen = Vec::new();
    let r: Result<(), StreamedPublishError<io::Error>> = publish_streamed(
        &dest,
        b"r",
        &CancelToken::new(),
        &mut |s| seen.push(s),
        |_| Err(io::Error::other("boom")),
    );
    assert!(matches!(r, Err(StreamedPublishError::Produce(_))), "{r:?}");
    assert!(entries(dir.path()).is_empty());
    assert_eq!(seen.first(), Some(&Stage::ArtifactWriting));
}
