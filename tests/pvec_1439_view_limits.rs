//! Issue #1439: `ViewProfile::apply` charges the profile's source limits as it walks, so a block
//! that fans out is refused with the limit named, instead of being expanded until memory runs out.
//!
//! The drawing is written by acadrust's `DwgWriter`: `levels` blocks, each inserting the next one
//! twice, instanced once, which is 2^levels leaves. Needs `--features dwg-acadrust`.
#![cfg(feature = "dwg-acadrust")]

use acadrust::entities::{Insert, Line};
use acadrust::tables::BlockRecord;
use acadrust::types::{Handle, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{DwgProvider, Limit, LimitError, SourceLimits, ViewProfile, ViewProfileError};

fn fan_out(levels: usize) -> Vec<u8> {
    let mut doc = CadDocument::new();
    let owners: Vec<Handle> = (0..levels)
        .map(|d| {
            let mut r = BlockRecord::new(format!("F{d}"));
            r.handle = doc.allocate_handle();
            let h = r.handle;
            doc.block_records.add(r).unwrap();
            h
        })
        .collect();
    for (d, owner) in owners.iter().enumerate() {
        let mut leaf = EntityType::Line(Line::from_coords(0.0, 0.0, 0.0, 1.0, 1.0, 0.0));
        leaf.common_mut().owner_handle = *owner;
        doc.add_entity(leaf).unwrap();
        if d + 1 < levels {
            for k in 0..2 {
                let mut i = EntityType::Insert(Insert::new(
                    format!("F{}", d + 1),
                    Vector3::new(k as f64, 0.0, 0.0),
                ));
                i.common_mut().owner_handle = *owner;
                doc.add_entity(i).unwrap();
            }
        }
    }
    doc.add_entity(EntityType::Insert(Insert::new(
        "F0",
        Vector3::new(0.0, 0.0, 0.0),
    )))
    .unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

fn apply(levels: usize, limits: SourceLimits) -> Result<usize, ViewProfileError> {
    let bytes = fan_out(levels);
    let p = AcadrustProvider::open(CadSource::Bytes(&bytes)).unwrap();
    ViewProfile::default_profile()
        .with_limits(limits)
        .apply(&p)
        .map(|v| v.included.len())
}

fn tight() -> SourceLimits {
    let mut l = SourceLimits::default();
    l.max_entities = 10_000;
    l.max_expanded_instances = 10_000;
    l
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_fan_out_past_the_instance_limit_is_refused_naming_it() {
    // 2^18 leaves are far past 10,000 instances and still cheap enough to fail red without
    // taking the machine down.
    match apply(18, tight()) {
        Err(ViewProfileError::Limit(LimitError::Exceeded {
            limit,
            ceiling,
            observed,
        })) => {
            assert!(
                matches!(limit, Limit::ExpandedInstances | Limit::Entities),
                "{limit:?}"
            );
            assert_eq!(ceiling, 10_000);
            assert!(observed > ceiling);
        }
        other => panic!("expected a limit error, got {other:?}"),
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_drawing_inside_the_limits_is_walked_as_before() {
    // Five levels, one line in each block and block d placed 2^d times: 2^5 - 1 leaves.
    assert_eq!(apply(5, tight()).unwrap(), 31);
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_default_limits_still_admit_an_ordinary_drawing() {
    assert!(apply(8, SourceLimits::default()).is_ok());
}
