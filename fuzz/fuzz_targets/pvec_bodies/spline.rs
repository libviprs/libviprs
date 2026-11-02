//! The body of `fuzz_pvec_spline` (N7.6, #1437), shared with `tests/pvec_1437_fuzz_seeds.rs`.
//!
//! Degree, control points, the knot vector, weights, periodic and closed flags, each present,
//! absent or not reportable, with whatever counts and numbers the bytes give: mismatched knot and
//! control counts, repeated knots, decreasing knots, zero and negative weights, NaN and infinity,
//! a degree of four billion. The extractor must refuse with a typed error or hand back a value the
//! record builder takes or refuses in turn; it must not panic or size anything from an unchecked
//! number.

use arbitrary::{Result as AResult, Unstructured};
use libviprs::cad::ItemHandle;
use libviprs::pvec::dwg_spline;
use libviprs::pvec::{
    EntityStyle, InstancePath, Normalized2DGeometry, ProvenanceStub, RawEntity, RawGeometry,
    SourceFact,
};

fn floats(u: &mut Unstructured<'_>) -> AResult<Vec<f64>> {
    let n = u.arbitrary_len::<f64>()?.min(4_096);
    (0..n).map(|_| u.arbitrary()).collect()
}

fn fact<T>(
    u: &mut Unstructured<'_>,
    value: impl FnOnce(&mut Unstructured<'_>) -> AResult<T>,
) -> AResult<SourceFact<T>> {
    Ok(match u.int_in_range(0..=2)? {
        0 => SourceFact::Present(value(u)?),
        1 => SourceFact::Absent,
        _ => SourceFact::NotReportable,
    })
}

fn build(u: &mut Unstructured<'_>) -> AResult<RawEntity> {
    let n = u.arbitrary_len::<[f64; 3]>()?.min(4_096);
    let mut control_points = Vec::with_capacity(n);
    for _ in 0..n {
        control_points.push([
            u.arbitrary()?,
            u.arbitrary()?,
            if u.ratio(7, 8)? { 0.0 } else { u.arbitrary()? },
        ]);
    }
    let helix = u.ratio(1, 16)?;
    Ok(RawEntity {
        geometry: RawGeometry::Spline {
            degree: u.arbitrary()?,
            control_points,
            knots: fact(u, floats)?,
            weights: fact(u, floats)?,
            periodic: fact(u, |u| u.arbitrary())?,
            closed: u.arbitrary()?,
        },
        style: EntityStyle::absent(),
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(1)),
            class_name: if helix { "HELIX" } else { "SPLINE" }.to_string(),
            block_path: InstancePath::root(),
            layer: "0".to_string(),
        },
    })
}

pub fn run(data: &[u8]) {
    let mut u = Unstructured::new(data);
    let Ok(e) = build(&mut u) else { return };
    let _ = dwg_spline::classify(&e);
    match dwg_spline::extract(&e) {
        Ok(s) => {
            let _ = Normalized2DGeometry::from_spline(&s);
        }
        Err(err) => {
            let _ = err.classification("SPLINE");
        }
    }
}
