//! A polyline with a repeated vertex is plotted, not dropped (#1706).
//!
//! A plotter draws a polyline through its points and a point that repeats the one before it adds
//! nothing: the zero length side is skipped and the rest is drawn. The conversion refuses the whole
//! polyline for it (`COINCIDENT_VERTICES`), so a sheet loses every such polyline, however long the
//! rest of it is. Each case plots one layout holding one polyline and expects its visible sides on
//! the page. The drawings are written with acadrust. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use acadrust::entities::{LwPolyline, Polyline2D, Vertex2D};
use acadrust::objects::ObjectType;
use acadrust::types::{Vector2, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PdfPlot;
use lopdf::Document;
use lopdf::content::Operation;

fn sheet(entity: EntityType) -> Vec<u8> {
    let mut doc = CadDocument::new();
    doc.add_layout("A3").unwrap();
    let layout = doc
        .objects
        .values_mut()
        .find_map(|o| match o {
            ObjectType::Layout(l) if l.name == "A3" => Some(l),
            _ => None,
        })
        .unwrap();
    layout.paper_width = 420.0;
    layout.paper_height = 297.0;
    layout.plot_flags.print_lineweights = true;
    layout.plot_paper_units = 1;
    layout.plot_type = 5;
    layout.plot_scale_numerator = 1.0;
    layout.plot_scale_denominator = 1.0;
    layout.paper_size = "USER_(420.00_x_297.00_MM)".into();
    doc.add_entity_to_layout(entity, "A3").unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

fn plot(tag: &str, entity: EntityType) -> Vec<Operation> {
    let dir = std::env::temp_dir().join(format!("libviprs-1706-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let bytes = sheet(entity);
    let dest: PathBuf = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .convert_with::<libviprs::pvec::dwg_acadrust::AcadrustProvider>()
        .unwrap_or_else(|e| panic!("{tag}: the plot failed: {e}"));
    let doc = Document::load(&dest).unwrap();
    let id = doc.get_pages()[&1];
    doc.get_and_decode_page_content(id).unwrap().operations
}

fn lw(points: &[[f64; 2]], bulges: &[f64], closed: bool) -> EntityType {
    let mut p = LwPolyline::new();
    for (i, q) in points.iter().enumerate() {
        p.add_point_with_bulge(
            Vector2::new(q[0], q[1]),
            bulges.get(i).copied().unwrap_or(0.0),
        );
    }
    p.is_closed = closed;
    EntityType::LwPolyline(p)
}

/// The points of every line segment the page draws: the `m` and `l` operators, in order.
fn drawn(ops: &[Operation]) -> usize {
    ops.iter().filter(|o| o.operator == "l").count()
}

fn strokes(ops: &[Operation]) -> usize {
    ops.iter().filter(|o| o.operator == "S").count()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1706_a_polyline_without_a_repeat_is_drawn() {
    // The control: three sides, three strokes' worth of segments.
    let ops = plot(
        "control",
        lw(
            &[[10.0, 10.0], [100.0, 10.0], [100.0, 80.0], [10.0, 80.0]],
            &[],
            false,
        ),
    );
    assert_eq!(strokes(&ops), 1);
    assert_eq!(drawn(&ops), 3);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1706_an_open_lwpolyline_with_a_repeated_vertex_is_drawn() {
    let ops = plot(
        "open",
        lw(
            &[
                [10.0, 10.0],
                [100.0, 10.0],
                [100.0, 10.0],
                [100.0, 80.0],
                [10.0, 80.0],
            ],
            &[],
            false,
        ),
    );
    assert_eq!(strokes(&ops), 1, "the polyline is dropped");
    assert_eq!(
        drawn(&ops),
        3,
        "the repeat adds no side, the other three stay"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1706_a_closed_lwpolyline_that_repeats_its_first_point_is_drawn() {
    // A closed outline whose last point is its first again: the closing side has no length.
    let ops = plot(
        "closed",
        lw(
            &[
                [10.0, 10.0],
                [100.0, 10.0],
                [100.0, 80.0],
                [10.0, 80.0],
                [10.0, 10.0],
            ],
            &[],
            true,
        ),
    );
    assert_eq!(strokes(&ops), 1, "the outline is dropped");
    assert_eq!(drawn(&ops), 4, "four sides");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1706_a_2d_polyline_with_a_repeated_vertex_is_drawn() {
    let mut p = Polyline2D::new();
    for q in [[10.0, 10.0], [100.0, 10.0], [100.0, 10.0], [100.0, 80.0]] {
        p.add_vertex(Vertex2D::new(Vector3::new(q[0], q[1], 0.0)));
    }
    let ops = plot("2d", EntityType::Polyline2D(p));
    assert_eq!(strokes(&ops), 1, "the polyline is dropped");
    assert_eq!(drawn(&ops), 2);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1706_a_polyline_that_is_one_point_repeated_draws_nothing_and_does_not_fail() {
    // Nothing is left to draw: the plot still succeeds and the page has no stroke.
    let ops = plot(
        "dot",
        lw(&[[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]], &[], false),
    );
    assert_eq!(strokes(&ops), 0);
}

/// Plots on another thread and gives up after `secs`, so a quadratic repair fails the test instead of
/// hanging it.
fn plot_within(tag: &'static str, secs: u64, entity: EntityType) -> Vec<Operation> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(plot(tag, entity));
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .unwrap_or_else(|_| panic!("{tag}: the plot took over {secs} s"))
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1706_two_hundred_thousand_repeats_are_skipped_in_one_pass() {
    // 100000 copies of one point, then 100000 of another: the repair has to be linear.
    let mut points = vec![[10.0, 10.0]; 100_000];
    points.extend(vec![[100.0, 10.0]; 100_000]);
    let ops = plot_within("many", 60, lw(&points, &[], false));
    assert_eq!(strokes(&ops), 1);
    assert_eq!(drawn(&ops), 1, "one side is left");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1706_a_polyline_of_identical_points_is_refused_without_a_panic_or_a_wait() {
    let ops = plot_within("same", 60, lw(&vec![[10.0, 10.0]; 200_000], &[], false));
    assert_eq!(strokes(&ops), 0);
}
