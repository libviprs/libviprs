//! N7.6 (#1457, slice of #1070): the invariants of the normalization and conversion path, as
//! properties over generated input.
//!
//! Every property is a named case, and every one counts its own invocations and asserts that the
//! count is the case count: a `proptest!` block whose strategy produced nothing passes, which is
//! the bug `tests/pmtiles_proptest.rs` records. Tolerances are `Policy::V1`'s, never a literal of
//! this file's: `close` below builds the comparison from the policy's own terms.
//!
//! The drawings are written by acadrust's `DwgWriter` from generated entity specs, so the whole
//! conversion path is under the properties and not a fixture. Needs `--features pvec-convert`.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

use std::sync::atomic::{AtomicU32, Ordering};

use acadrust::entities::{Arc, Circle, Line, Spline, Text};
use acadrust::types::Vector3;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::fidelity::FidelityClass;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::transform::ComposedTransform;
use libviprs::pvec::{InstanceStep, SourceUnit, curve_bounds};
use libviprs::pvec_convert::PvecConversion;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};
use pvec_numeric::Policy;

const CASES: u32 = 48;

fn step() -> InstanceStep {
    InstanceStep {
        block_name: "P".into(),
        insert_handle: ItemHandle::new(1),
    }
}

/// `a` and `b` agree under `Policy::V1`: the absolute floor, the relative term times the scale
/// of the values, and the ulps of the largest magnitude involved.
fn close(a: f64, b: f64, scale: f64) -> bool {
    let p = Policy::V1;
    let ulps = f64::from(p.ulp) * f64::EPSILON * a.abs().max(b.abs());
    (a - b).abs() <= p.abs + p.rel * scale + ulps
}

fn apply(m: [f64; 6], p: [f64; 2]) -> [f64; 2] {
    [
        m[0] * p[0] + m[1] * p[1] + m[4],
        m[2] * p[0] + m[3] * p[1] + m[5],
    ]
}

/// A matrix with a determinant well away from zero, so composing it is not a question of the
/// singularity rule.
fn matrix() -> impl Strategy<Value = [f64; 6]> {
    (
        [-3.0f64..3.0, -3.0f64..3.0, -3.0f64..3.0, -3.0f64..3.0],
        -100.0f64..100.0,
        -100.0f64..100.0,
    )
        .prop_filter("non-singular", |(m, _, _)| {
            (m[0] * m[3] - m[1] * m[2]).abs() > 0.2
        })
        .prop_map(|(m, tx, ty)| [m[0], m[1], m[2], m[3], tx, ty])
}

fn run<S: Strategy>(
    counter: &AtomicU32,
    strategy: S,
    test: impl Fn(S::Value) -> Result<(), TestCaseError>,
) {
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });
    runner
        .run(&strategy, |v| {
            counter.fetch_add(1, Ordering::Relaxed);
            test(v)
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        counter.load(Ordering::Relaxed),
        CASES,
        "the property did not run every case"
    );
}

static SEQUENTIAL: AtomicU32 = AtomicU32::new(0);

#[test]
fn a_composed_transform_equals_the_sequential_application_of_its_levels() {
    run(
        &SEQUENTIAL,
        (
            proptest::collection::vec(matrix(), 1..5),
            [-50.0f64..50.0, -50.0f64..50.0],
        ),
        |(levels, p)| {
            let composed = ComposedTransform::compose(
                levels
                    .iter()
                    .map(|m| (step(), libviprs::pvec::SourceTransform::new(*m).unwrap())),
            )
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
            // The innermost level (the last pushed) is applied first.
            let sequential = levels.iter().rev().fold(p, |q, m| apply(*m, q));
            let direct = apply(composed.matrix(), p);
            let scale = sequential[0].abs().max(sequential[1].abs()).max(1.0);
            prop_assert!(
                close(sequential[0], direct[0], scale) && close(sequential[1], direct[1], scale)
            );
            Ok(())
        },
    );
}

static ASSOCIATIVE: AtomicU32 = AtomicU32::new(0);

#[test]
fn composition_is_associative() {
    run(
        &ASSOCIATIVE,
        (
            matrix(),
            matrix(),
            matrix(),
            [-20.0f64..20.0, -20.0f64..20.0],
        ),
        |(a, b, c, p)| {
            let t = |m: [f64; 6]| libviprs::pvec::SourceTransform::new(m).unwrap();
            let abc = ComposedTransform::compose([(step(), t(a)), (step(), t(b)), (step(), t(c))])
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            // (a then b) folded into one level, then c, against a, b, c in one nest.
            let ab = ComposedTransform::compose([(step(), t(a)), (step(), t(b))])
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let grouped = ComposedTransform::compose([(step(), t(ab.matrix())), (step(), t(c))])
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let (x, y) = (apply(abc.matrix(), p), apply(grouped.matrix(), p));
            let scale = x[0].abs().max(x[1].abs()).max(1.0);
            prop_assert!(close(x[0], y[0], scale) && close(x[1], y[1], scale));
            Ok(())
        },
    );
}

/// A matrix whose coefficient `at` is not a finite number. Composition must refuse it, so a nest
/// that holds one has no product to compare.
fn hostile_matrix() -> impl Strategy<Value = [f64; 6]> {
    (
        matrix(),
        0usize..6,
        prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
    )
        .prop_map(|(mut m, at, bad)| {
            m[at] = bad;
            m
        })
}

static ASSOCIATIVE_HOSTILE: AtomicU32 = AtomicU32::new(0);

/// Associativity over levels that may be non-finite. Both groupings must agree: either both refuse
/// the nest, or both produce the same matrix. A composition that lets a NaN through produces a
/// matrix that equals nothing, `close` is false on it, and this property goes red; it is the case
/// that fails when the finite check in PVec's `Transform2` is removed.
#[test]
fn composition_is_associative_when_a_level_is_not_finite() {
    run(
        &ASSOCIATIVE_HOSTILE,
        (
            prop_oneof![matrix(), hostile_matrix()],
            prop_oneof![matrix(), hostile_matrix()],
            prop_oneof![matrix(), hostile_matrix()],
            [-20.0f64..20.0, -20.0f64..20.0],
        ),
        |(a, b, c, p)| {
            let nest = |ms: &[[f64; 6]]| {
                ms.iter().try_fold(ComposedTransform::identity(), |acc, m| {
                    acc.push_inner_coefficients(step(), *m)
                })
            };
            let abc = nest(&[a, b, c]);
            let grouped = nest(&[a, b]).and_then(|ab| nest(&[ab.matrix(), c]));
            let any_bad = [a, b, c].iter().any(|m| m.iter().any(|x| !x.is_finite()));
            prop_assert_eq!(
                abc.is_err(),
                any_bad,
                "a nest with a non-finite level must be refused, and only such a nest"
            );
            prop_assert_eq!(abc.is_err(), grouped.is_err());
            if let (Ok(x), Ok(y)) = (abc, grouped) {
                let (x, y) = (apply(x.matrix(), p), apply(y.matrix(), p));
                let scale = x[0].abs().max(x[1].abs()).max(1.0);
                prop_assert!(close(x[0], y[0], scale) && close(x[1], y[1], scale));
            }
            Ok(())
        },
    );
}

static SINGULAR: AtomicU32 = AtomicU32::new(0);

#[test]
fn a_singular_transform_never_produces_geometry() {
    use libviprs::pvec::Normalized2DGeometry;
    run(
        &SINGULAR,
        (
            [-10.0f64..10.0, -10.0f64..10.0],
            -5.0f64..5.0,
            [-10.0f64..10.0, -10.0f64..10.0],
            -50.0f64..50.0,
        ),
        |(row, k, p, t)| {
            // The second row is a multiple of the first: the determinant is zero by construction.
            let m = [row[0], row[1], k * row[0], k * row[1], t, t];
            let line =
                libviprs::pvec::dwg_basic::extract(&line_entity(p, [p[0] + 1.0, p[1] + 2.0]))
                    .map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
            let record = Normalized2DGeometry::from_curve(line.curve, line.classification.class())
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            match ComposedTransform::identity().push_inner_coefficients(step(), m) {
                Err(_) => {}
                Ok(c) => {
                    let promoted = libviprs::pvec::transform::promote(&record, &c);
                    let geometry = promoted.is_ok_and(|p| p.record().geometry().is_some());
                    prop_assert!(!geometry, "a singular map {m:?} produced geometry");
                }
            }
            Ok(())
        },
    );
}

fn line_entity(a: [f64; 2], b: [f64; 2]) -> libviprs::pvec::RawEntity {
    use libviprs::pvec::{EntityStyle, InstancePath, ProvenanceStub, RawEntity, RawGeometry};
    RawEntity {
        geometry: RawGeometry::Line {
            start: [a[0], a[1], 0.0],
            end: [b[0], b[1], 0.0],
        },
        style: EntityStyle::absent(),
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(1)),
            class_name: "LINE".into(),
            block_path: InstancePath::root(),
            layer: "0".into(),
        },
    }
}

/// One generated entity of a drawing.
#[derive(Debug, Clone)]
enum Spec {
    Line([f64; 2], [f64; 2]),
    Arc([f64; 2], f64, f64, f64),
    Circle([f64; 2], f64),
    Spline(Vec<[f64; 2]>, i32),
    Spline3d(Vec<[f64; 3]>),
    Text([f64; 2]),
}

fn coord() -> impl Strategy<Value = f64> {
    -5000.0f64..5000.0
}

fn point() -> impl Strategy<Value = [f64; 2]> {
    [coord(), coord()]
}

fn spec() -> impl Strategy<Value = Spec> {
    prop_oneof![
        (point(), point())
            .prop_filter("distinct", |(a, b)| a != b)
            .prop_map(|(a, b)| Spec::Line(a, b)),
        (point(), 0.5f64..200.0, 0.0f64..3.0, 0.5f64..3.0).prop_map(|(c, r, a, s)| Spec::Arc(
            c,
            r,
            a,
            a + s
        )),
        (point(), 0.5f64..200.0).prop_map(|(c, r)| Spec::Circle(c, r)),
        (proptest::collection::vec(point(), 4..10), 2i32..4).prop_map(|(p, d)| Spec::Spline(p, d)),
        proptest::collection::vec(
            (coord(), coord(), 1.0f64..50.0).prop_map(|(x, y, z)| [x, y, z]),
            4..8
        )
        .prop_map(Spec::Spline3d),
        point().prop_map(Spec::Text),
    ]
}

fn drawing(specs: &[Spec]) -> Vec<u8> {
    let mut doc = CadDocument::new();
    for s in specs {
        let e = match s {
            Spec::Line(a, b) => {
                EntityType::Line(Line::from_coords(a[0], a[1], 0.0, b[0], b[1], 0.0))
            }
            Spec::Arc(c, r, a, b) => EntityType::Arc(Arc::from_coords(c[0], c[1], 0.0, *r, *a, *b)),
            Spec::Circle(c, r) => EntityType::Circle(Circle::from_coords(c[0], c[1], 0.0, *r)),
            Spec::Spline(pts, d) => EntityType::Spline(Spline::from_control_points(
                *d,
                pts.iter().map(|p| Vector3::new(p[0], p[1], 0.0)).collect(),
            )),
            Spec::Spline3d(pts) => EntityType::Spline(Spline::from_control_points(
                3,
                pts.iter().map(|p| Vector3::new(p[0], p[1], p[2])).collect(),
            )),
            Spec::Text(p) => {
                let mut t = Text::new();
                t.value = "t".into();
                t.insertion_point = Vector3::new(p[0], p[1], 0.0);
                t.height = 2.0;
                EntityType::Text(t)
            }
        };
        doc.add_entity(e).expect("entity");
    }
    DwgWriter::write_to_vec(&doc).expect("writes")
}

static EXCLUSIVE: AtomicU32 = AtomicU32::new(0);

#[test]
#[cfg_attr(miri, ignore)]
fn every_entity_lands_in_exactly_one_of_the_six_classes() {
    run(
        &EXCLUSIVE,
        proptest::collection::vec(spec(), 1..12),
        |specs| {
            let bytes = drawing(&specs);
            let dir = tempfile::tempdir().unwrap();
            let f = PvecConversion::new(CadSource::Bytes(&bytes), &dir.path().join("a.pvec"))
                .unit(SourceUnit::Meters)
                .probe()
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let total: u64 = FidelityClass::ALL.iter().map(|c| f.count(*c)).sum();
            prop_assert_eq!(
                f.encountered(),
                specs.len() as u64,
                "every entity is met once"
            );
            prop_assert_eq!(
                total,
                f.encountered(),
                "the six classes partition the entities met"
            );
            Ok(())
        },
    );
}

static BOUNDS: AtomicU32 = AtomicU32::new(0);

#[test]
#[cfg_attr(miri, ignore)]
fn the_reported_bounds_contain_every_exact_entity_and_stay_within_every_box() {
    run(&BOUNDS, proptest::collection::vec(spec(), 1..12), |specs| {
        let bytes = drawing(&specs);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("a.pvec");
        let out = match PvecConversion::new(CadSource::Bytes(&bytes), &dest)
            .unit(SourceUnit::Meters)
            .convert()
        {
            Ok(o) => o,
            // A drawing of only unsupported entities writes nothing, and has no bounds to report.
            Err(libviprs::pvec_convert::PvecConvertError::Write(_)) => return Ok(()),
            Err(e) => return Err(TestCaseError::fail(e.to_string())),
        };
        let reader = open_published(&dest).unwrap();
        let drawing = reader.manifest().bounds;
        // The reported box lies between two unions: it contains the box of every entity whose bound
        // is exact, and it lies within the union of every entity's box, hull bounds included. A
        // spline's box read back is the control polygon's (a hull, which only bounds it from
        // above), while the manifest holds the tighter bound the writer derived, so containing the
        // hull would be a stronger claim than the format makes.
        let (mut exact, mut all) = (None::<[f64; 4]>, None::<[f64; 4]>);
        let grow = |acc: &mut Option<[f64; 4]>, b: [f64; 4]| {
            *acc = Some(acc.map_or(b, |a| {
                [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ]
            }));
        };
        for (_, curve) in reader.entities().unwrap() {
            let (b, method) = curve_bounds(&curve).unwrap();
            let boxed = [b.min().x, b.min().y, b.max().x, b.max().y];
            grow(&mut all, boxed);
            if method == libviprs::pvec::BoundsMethod::Exact {
                grow(&mut exact, boxed);
            }
        }
        if let Some(e) = exact {
            prop_assert!(
                drawing.min[0] <= e[0]
                    && drawing.min[1] <= e[1]
                    && e[2] <= drawing.max[0]
                    && e[3] <= drawing.max[1],
                "the exact entities' box {e:?} is outside the reported {drawing:?}"
            );
        }
        let a = all.expect("an artifact with entities");
        prop_assert!(
            a[0] <= drawing.min[0]
                && a[1] <= drawing.min[1]
                && drawing.max[0] <= a[2]
                && drawing.max[1] <= a[3],
            "the reported {drawing:?} reaches past every entity's box {a:?}"
        );
        prop_assert!(out.written > 0);
        Ok(())
    });
}

static IDENTICAL: AtomicU32 = AtomicU32::new(0);

#[test]
#[cfg_attr(miri, ignore)]
fn the_same_drawing_converts_to_byte_identical_output_twice() {
    run(
        &IDENTICAL,
        proptest::collection::vec(spec(), 1..12),
        |specs| {
            let bytes = drawing(&specs);
            let dir = tempfile::tempdir().unwrap();
            let mut outputs = Vec::new();
            for n in 0..2 {
                let dest = dir.path().join(format!("{n}.pvec"));
                match PvecConversion::new(CadSource::Bytes(&bytes), &dest)
                    .unit(SourceUnit::Meters)
                    .convert()
                {
                    Ok(_) => outputs.push(std::fs::read(&dest).unwrap()),
                    Err(libviprs::pvec_convert::PvecConvertError::Write(_)) => return Ok(()),
                    Err(e) => return Err(TestCaseError::fail(e.to_string())),
                }
            }
            prop_assert_eq!(&outputs[0], &outputs[1]);
            Ok(())
        },
    );
}
