//! Issue #1609: an INSERT whose normal is not the Z axis is drawn as the plan sees it.
//!
//! A block set in a vertical plane is seen edge on from above, so each of its curves is drawn as
//! the segment it projects to, in plan coordinates, and nothing about the PVec format changes. A
//! curve that projects to a point, and anything that is not a curve, is refused and counted as
//! `extrusion-not-applied`, one row each. A block in a tilted plane is an affine image of itself,
//! so its curves map exactly. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1500_plan.rs"]
mod plan;

use std::cell::RefCell;
use std::f64::consts::FRAC_PI_2;
use std::fs;

use libviprs::cad::CadSource;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::ConversionReport;
use libviprs::pvec::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet,
    FacetSource, LayerState, LayoutId, LayoutKind, LayoutState, Point3, ProviderCaps,
    ProviderError, RawEntity, RawGeometry, SourceTransform, SourceUnit,
};
use libviprs::pvec_convert::PvecConversion;
use plan::{scratch, stub};
use pvec_facade::Curve;

thread_local! {
    static NEXT: RefCell<Option<Drawing>> = const { RefCell::new(None) };
}

#[derive(Clone)]
struct Drawing {
    model: Vec<RawEntity>,
    blocks: Vec<(&'static str, Vec<RawEntity>)>,
    caps: ProviderCaps,
}

struct Cursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

impl DwgProvider for Drawing {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(true)
    }

    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        Ok(NEXT
            .with(|n| n.borrow_mut().take())
            .expect("a drawing for this run"))
    }

    fn name(&self) -> &'static str {
        "edge-on"
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
        Ok(["0", "HIDDEN"]
            .into_iter()
            .map(|name| LayerState {
                name: name.into(),
                handle: None,
                off: name == "HIDDEN",
                frozen: false,
                locked: false,
                plottable: true,
            })
            .collect())
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let items = match scope {
            EntityScope::Layout(_) => self.model.clone(),
            EntityScope::BlockDefinition { name, reached_via } => {
                let Some((_, list)) = self.blocks.iter().find(|(n, _)| n == name) else {
                    return Err(ProviderError::NoSuchScope(name.clone()));
                };
                list.iter()
                    .cloned()
                    .map(|mut e| {
                        e.provenance.block_path = reached_via.clone();
                        e
                    })
                    .collect()
            }
            other => return Err(ProviderError::NoSuchScope(format!("{other:?}"))),
        };
        Ok(Box::new(Cursor(items.into_iter())))
    }
}

fn ent(handle: u64, class: &str, layer: &str, geometry: RawGeometry) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry,
        provenance: stub(handle, class, layer),
    }
}

fn line(handle: u64, a: Point3, b: Point3) -> RawEntity {
    ent(handle, "LINE", "0", RawGeometry::Line { start: a, end: b })
}

#[allow(clippy::too_many_arguments)]
fn insert(
    handle: u64,
    block: &str,
    normal: Point3,
    at: [f64; 2],
    elevation: f64,
    z_scale: f64,
    grid: ArrayGrid,
) -> RawEntity {
    ent(
        handle,
        "INSERT",
        "0",
        RawGeometry::Insert(BlockRef {
            block_name: block.into(),
            transform: SourceTransform::new([1.0, 0.0, 0.0, 1.0, at[0], at[1]]).unwrap(),
            normal,
            grid,
            elevation,
            z_scale,
        }),
    )
}

/// The block every case places: a line along its x axis (60), one along its y axis (61), a circle
/// (62), a quarter arc (63), a text (64) and a line on a layer that is off (65).
fn wall() -> Vec<RawEntity> {
    vec![
        line(60, [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]),
        line(61, [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]),
        ent(
            62,
            "CIRCLE",
            "0",
            RawGeometry::Circle {
                center: [1.0, 1.0, 0.0],
                radius: 0.5,
                normal: [0.0, 0.0, 1.0],
            },
        ),
        ent(
            63,
            "ARC",
            "0",
            RawGeometry::Arc {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
                start_angle: 0.0,
                end_angle: FRAC_PI_2,
                normal: [0.0, 0.0, 1.0],
            },
        ),
        ent(
            64,
            "TEXT",
            "0",
            RawGeometry::Unmapped {
                reason: "text".into(),
            },
        ),
        ent(
            65,
            "LINE",
            "HIDDEN",
            RawGeometry::Line {
                start: [0.0, 0.0, 0.0],
                end: [1.0, 0.0, 0.0],
            },
        ),
    ]
}

/// `model` beside one plain line of the drawing's own (1), so there is geometry whatever else is
/// drawn.
fn drawing(mut model: Vec<RawEntity>, blocks: Vec<(&'static str, Vec<RawEntity>)>) -> Drawing {
    model.push(line(1, [100.0, 100.0, 0.0], [101.0, 100.0, 0.0]));
    Drawing {
        model,
        blocks,
        caps: Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
            c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
        }),
    }
}

/// The report and every written curve with the handle of the row it came from.
fn convert(tag: &str, d: Drawing) -> (ConversionReport, Vec<(u64, Curve)>) {
    let dir = scratch(tag);
    let dest = dir.join("o.pvec");
    NEXT.with(|n| *n.borrow_mut() = Some(d));
    let out = PvecConversion::new(CadSource::Bytes(b"edge-on"), &dest)
        .unit(SourceUnit::Meters)
        .convert_with::<Drawing>()
        .unwrap();
    let curves = open_published(&dest).unwrap().entities().unwrap();
    let _ = fs::remove_dir_all(dir);
    let kept: Vec<u64> = out
        .report
        .entities
        .iter()
        .filter(|r| r.semantic_type.is_some())
        .map(|r| r.source_handle.unwrap())
        .collect();
    assert_eq!(kept.len(), curves.len(), "{:#?}", out.report.entities);
    let written = kept
        .into_iter()
        .zip(curves.into_iter().map(|(_, c)| c))
        .collect();
    (out.report, written)
}

/// A trimmed line as its two ends, the smaller one first.
fn ends(c: &Curve) -> [[f64; 2]; 2] {
    let Curve::Trimmed(t) = c else {
        panic!("a trimmed line, not {c:?}");
    };
    let Curve::Line(l) = t.basis() else {
        panic!("a trimmed line, not {c:?}");
    };
    let (o, d, s) = (l.origin(), l.direction(), t.trim().span);
    let mut e = [[o.x, o.y], [o.x + d.x * s, o.y + d.y * s]];
    e.sort_by(|a, b| a.partial_cmp(b).unwrap());
    e
}

fn segments(written: &[(u64, Curve)], handle: u64) -> Vec<[[f64; 2]; 2]> {
    let mut s: Vec<_> = written
        .iter()
        .filter(|(h, _)| *h == handle)
        .map(|(_, c)| ends(c))
        .collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

fn near(got: &[[[f64; 2]; 2]], want: &[[[f64; 2]; 2]]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .flatten()
            .flatten()
            .zip(want.iter().flatten().flatten())
            .all(|(a, b)| (a - b).abs() <= 1e-9 * (1.0 + b.abs()))
}

fn reason(report: &ConversionReport, handle: u64) -> Vec<Option<String>> {
    report
        .entities
        .iter()
        .filter(|r| r.source_handle == Some(handle))
        .map(|r| r.reason.clone())
        .collect()
}

fn x_axis_wall() -> Drawing {
    drawing(
        vec![insert(
            50,
            "WALL",
            [1.0, 0.0, 0.0],
            [2.0, 0.0],
            7.0,
            1.0,
            ArrayGrid::SINGLE,
        )],
        vec![("WALL", wall())],
    )
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_a_block_in_a_vertical_plane_is_drawn_edge_on() {
    // Normal +X: the block's x runs along the plan's y, and its z = 0 sits at plan x = 7.
    let (report, written) = convert("1609-edge-x", x_axis_wall());
    assert!(
        reason(&report, 50).is_empty(),
        "the INSERT is expanded, not skipped: {:?}",
        reason(&report, 50)
    );
    let s60 = segments(&written, 60);
    assert!(near(&s60, &[[[7.0, 2.0], [7.0, 6.0]]]), "{s60:?}");
    let s62 = segments(&written, 62);
    assert!(near(&s62, &[[[7.0, 2.5], [7.0, 3.5]]]), "{s62:?}");
    let s63 = segments(&written, 63);
    assert!(near(&s63, &[[[7.0, 2.0], [7.0, 3.0]]]), "{s63:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_an_edge_on_curve_says_it_is_a_projection() {
    let (report, _) = convert("1609-edge-class", x_axis_wall());
    for h in [60, 62, 63] {
        let rows: Vec<_> = report
            .entities
            .iter()
            .filter(|r| r.source_handle == Some(h))
            .collect();
        assert_eq!(rows.len(), 1, "{rows:#?}");
        assert_eq!(rows[0].fidelity, "SupportedEquivalent", "{:#?}", rows[0]);
        assert!(
            rows[0].warnings.iter().any(|w| w.contains("edge on")),
            "{:#?}",
            rows[0]
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_what_has_no_plan_image_is_refused_one_row_each() {
    let (report, written) = convert("1609-edge-refused", x_axis_wall());
    // The line along the block's y axis is a point from above, and a text is not a curve.
    for h in [61, 64] {
        assert_eq!(
            reason(&report, h),
            [Some("extrusion-not-applied".to_owned())],
            "{h}"
        );
        assert!(segments(&written, h).is_empty());
    }
    // A layer that is off still hides what is on it.
    assert_eq!(reason(&report, 65).len(), 1);
    assert!(segments(&written, 65).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_the_arbitrary_axis_algorithm_places_other_azimuths() {
    // Normal +Y: the OCS x axis is the world's -x, and z = 0 sits at plan y = -3.
    let d = drawing(
        vec![insert(
            50,
            "WALL",
            [0.0, 1.0, 0.0],
            [0.0, 0.0],
            -3.0,
            1.0,
            ArrayGrid::SINGLE,
        )],
        vec![("WALL", wall())],
    );
    let (_, written) = convert("1609-edge-y", d);
    let s60 = segments(&written, 60);
    assert!(near(&s60, &[[[-4.0, -3.0], [0.0, -3.0]]]), "{s60:?}");
    let s62 = segments(&written, 62);
    assert!(near(&s62, &[[[-1.5, -3.0], [-0.5, -3.0]]]), "{s62:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_a_nested_insert_keeps_its_elevation_and_the_outer_z_scale() {
    // The inner INSERT lifts WALL by 5 along OUTER's z, OUTER doubles z, and OUTER's z runs along
    // the plan's x.
    let d = drawing(
        vec![insert(
            50,
            "OUTER",
            [1.0, 0.0, 0.0],
            [0.0, 0.0],
            0.0,
            2.0,
            ArrayGrid::SINGLE,
        )],
        vec![
            (
                "OUTER",
                vec![insert(
                    70,
                    "WALL",
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0],
                    5.0,
                    1.0,
                    ArrayGrid::SINGLE,
                )],
            ),
            ("WALL", wall()),
        ],
    );
    let (report, written) = convert("1609-edge-nested", d);
    assert!(reason(&report, 70).is_empty(), "{:?}", reason(&report, 70));
    let s60 = segments(&written, 60);
    assert!(near(&s60, &[[[10.0, 0.0], [10.0, 4.0]]]), "{s60:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_each_minsert_cell_is_placed_along_the_insert_axes() {
    let d = drawing(
        vec![insert(
            50,
            "WALL",
            [1.0, 0.0, 0.0],
            [2.0, 0.0],
            7.0,
            1.0,
            ArrayGrid {
                rows: 1,
                columns: 2,
                row_spacing: 0.0,
                column_spacing: 10.0,
            },
        )],
        vec![("WALL", wall())],
    );
    let (_, written) = convert("1609-edge-minsert", d);
    let s60 = segments(&written, 60);
    assert!(
        near(
            &s60,
            &[[[7.0, 2.0], [7.0, 6.0]], [[7.0, 12.0], [7.0, 16.0]]]
        ),
        "{s60:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_a_block_in_a_tilted_plane_maps_exactly() {
    // Normal (0, -0.6, 0.8): the OCS x axis is the world's x and the OCS y axis is (0, 0.8, 0.6),
    // so a plan sees the block's y shortened to 0.8 of itself, and an elevation of 5 along the
    // normal moves it by -3 in y.
    let d = drawing(
        vec![insert(
            50,
            "WALL",
            [0.0, -0.6, 0.8],
            [0.0, 0.0],
            5.0,
            1.0,
            ArrayGrid::SINGLE,
        )],
        vec![("WALL", wall())],
    );
    let (report, written) = convert("1609-edge-tilt", d);
    let s60 = segments(&written, 60);
    assert!(near(&s60, &[[[0.0, -3.0], [4.0, -3.0]]]), "{s60:?}");
    let s61 = segments(&written, 61);
    assert!(near(&s61, &[[[0.0, -3.0], [0.0, -0.6]]]), "{s61:?}");
    // The circle is an ellipse in the plan, and nothing about it is approximate.
    let c = written.iter().find(|(h, _)| *h == 62).map(|(_, c)| c);
    assert!(matches!(c, Some(Curve::Ellipse(_))), "{c:?}");
    let row = report
        .entities
        .iter()
        .find(|r| r.source_handle == Some(62))
        .unwrap();
    assert_eq!(row.fidelity, "SupportedExact", "{row:#?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1609_a_block_turned_back_into_the_plan_by_a_nested_insert_maps_exactly() {
    // OUTER stands in the plane x = 0 (normal +X), and the INSERT inside it has normal +Y in
    // OUTER's coordinates, which is the world's Z: WALL's x runs along the plan's -y and its y
    // along the plan's x.
    let d = drawing(
        vec![insert(
            50,
            "OUTER",
            [1.0, 0.0, 0.0],
            [0.0, 0.0],
            0.0,
            1.0,
            ArrayGrid::SINGLE,
        )],
        vec![
            (
                "OUTER",
                vec![insert(
                    70,
                    "WALL",
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0],
                    0.0,
                    1.0,
                    ArrayGrid::SINGLE,
                )],
            ),
            ("WALL", wall()),
        ],
    );
    let (report, written) = convert("1609-edge-back", d);
    let s60 = segments(&written, 60);
    assert!(near(&s60, &[[[0.0, -4.0], [0.0, 0.0]]]), "{s60:?}");
    let s61 = segments(&written, 61);
    assert!(near(&s61, &[[[0.0, 0.0], [3.0, 0.0]]]), "{s61:?}");
    // A whole circle, which a turned map may hand back trimmed to a full turn.
    let c = written.iter().find(|(h, _)| *h == 62).map(|(_, c)| c);
    let c = match c {
        Some(Curve::Circle(c)) => c,
        Some(Curve::Trimmed(t)) if matches!(t.basis(), Curve::Circle(_)) => {
            let Curve::Circle(c) = t.basis() else {
                unreachable!()
            };
            assert!(
                (t.trim().span - std::f64::consts::TAU).abs() < 1e-9,
                "{t:?}"
            );
            c
        }
        other => panic!("a circle, not {other:?}"),
    };
    let o = c.placement().origin();
    assert!(
        (o.x - 1.0).abs() < 1e-9 && (o.y + 1.0).abs() < 1e-9 && (c.radius() - 0.5).abs() < 1e-9,
        "{c:?}"
    );
    for h in [60, 61, 62] {
        let row = report
            .entities
            .iter()
            .find(|r| r.source_handle == Some(h))
            .unwrap();
        assert!(
            !row.warnings.iter().any(|w| w.contains("edge on")),
            "{row:#?}"
        );
    }
}
