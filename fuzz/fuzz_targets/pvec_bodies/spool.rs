//! The body of `fuzz_pvec_spool` (N7.6, #1070), shared with `tests/pvec_1070_spool_fuzz.rs`.
//!
//! A drawing of polylines (mostly sane coordinates, sometimes anything at all), a spill budget from
//! zero bytes up, a plain or indexed write, and a sink that may fail after any number of bytes. Two
//! things must hold whatever the input: a streamed write that finishes is byte for byte the in-memory
//! one, and a write that ends early leaves no file under the spill root.

use std::io::{self, Write};

use arbitrary::{Result as AResult, Unstructured};
use libviprs::CancelToken;
use libviprs::cad::ItemHandle;
use libviprs::pvec::writer::{
    LeafConfig, SpillConfig, WriteConfig, write_geometry, write_geometry_to,
};
use libviprs::pvec::{
    EntityId, EntityStyle, FidelityClass, InstancePath, Normalized2DGeometry, OrderingPolicy,
    ProvenanceStub, RawEntity, RawGeometry, SourceUnit, order_entities,
};

/// What one input did.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The input did not decode to a drawing, or the drawing had nothing to write.
    NotRun,
    /// A write finished and matched the in-memory artifact. `spilled` is true when the budget was
    /// smaller than the artifact, so the writer had to spill.
    Written { spilled: bool },
    /// The sink failed part way and the spill root was empty afterwards.
    SinkFailed,
}

/// A sink that fails once `limit` bytes have gone through it.
struct FailAfter {
    limit: usize,
    seen: usize,
}

impl Write for FailAfter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.seen + buf.len() > self.limit {
            return Err(io::Error::other("sink full"));
        }
        self.seen += buf.len();
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn coord(u: &mut Unstructured<'_>) -> AResult<f64> {
    if u.ratio(1, 16)? {
        u.arbitrary()
    } else {
        Ok(f64::from(u.int_in_range(0..=255u8)?))
    }
}

fn geometry(u: &mut Unstructured<'_>) -> AResult<Option<Normalized2DGeometry>> {
    let n = u.int_in_range(2..=6usize)?;
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        pts.push([coord(u)?, coord(u)?]);
    }
    Ok(Normalized2DGeometry::polyline(&pts, FidelityClass::SupportedExact).ok())
}

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

fn decode(data: &[u8]) -> AResult<Outcome> {
    let mut u = Unstructured::new(data);
    let indexed = u.arbitrary::<bool>()?;
    let budget = [0u64, 1, 64, 4096, 1 << 20][u.int_in_range(0..=4usize)?];
    let fail_after = if u.ratio(1, 3)? {
        Some(u.int_in_range(0..=2000usize)?)
    } else {
        None
    };
    let want = u.int_in_range(1..=200usize)?;
    let mut geoms = Vec::new();
    for _ in 0..want {
        if let Some(g) = geometry(&mut u)? {
            geoms.push(g);
        }
    }
    if geoms.is_empty() {
        return Ok(Outcome::NotRun);
    }
    let all = ids(geoms.len());
    let config = WriteConfig {
        unit: SourceUnit::Millimeters,
        leaves: indexed.then_some(LeafConfig {
            cell_capacity: 8,
            tier0_max_error: Some(0.05),
        }),
    };
    let memory = write_geometry(
        &config,
        all.iter().copied().zip(&geoms),
        Vec::new(),
        &CancelToken::new(),
    );

    let root = tempfile::tempdir().expect("a scratch directory");
    let spill = SpillConfig {
        root: root.path().to_path_buf(),
        budget,
    };
    let items = all.iter().copied().zip(&geoms);
    let outcome = match (fail_after, memory) {
        (Some(limit), Ok(m)) if limit < m.bytes.len() => {
            let mut sink = FailAfter { limit, seen: 0 };
            let r = write_geometry_to(
                &config,
                items,
                Vec::new(),
                &spill,
                &CancelToken::new(),
                &mut sink,
            );
            assert!(
                r.is_err(),
                "a sink that fails at {limit} bytes let a write finish"
            );
            Outcome::SinkFailed
        }
        (_, Ok(m)) => {
            let mut sink = Vec::new();
            let r = write_geometry_to(
                &config,
                items,
                Vec::new(),
                &spill,
                &CancelToken::new(),
                &mut sink,
            )
            .expect("the in-memory write finished, so the streamed one must");
            assert_eq!(sink, m.bytes, "budget {budget}: streamed bytes differ");
            assert_eq!(r.length, sink.len() as u64);
            assert_eq!((r.written, r.skipped), (m.written, m.skipped));
            assert_eq!(r.source_ids, m.source_ids);
            Outcome::Written {
                spilled: budget < m.bytes.len() as u64,
            }
        }
        (_, Err(_)) => {
            let r = write_geometry_to(
                &config,
                items,
                Vec::new(),
                &spill,
                &CancelToken::new(),
                &mut Vec::new(),
            );
            assert!(
                r.is_err(),
                "the in-memory write refused, the streamed one did not"
            );
            Outcome::NotRun
        }
    };
    let left: Vec<_> = std::fs::read_dir(root.path())
        .expect("the spill root")
        .map(|e| e.expect("an entry").file_name())
        .collect();
    assert!(left.is_empty(), "spill files left behind: {left:?}");
    Ok(outcome)
}

pub fn run(data: &[u8]) -> Outcome {
    decode(data).unwrap_or(Outcome::NotRun)
}
