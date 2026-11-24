//! Issue #1670: a pattern hatch is drawn as the lines its pattern makes, clipped to its boundary. The
//! DWG hatch carries the pattern's own line definitions (angle, base point, offset between lines and
//! dash lengths, already scaled and turned), which the PVEC hatch stream does not, so a plot draws them
//! from what the conversion hands back beside the artifact. The drawings are written in memory.
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use acadrust::entities::{
    BoundaryEdge, BoundaryPath, BoundaryPathFlags, Hatch, HatchPattern, HatchPatternLine,
    PolylineEdge,
};
use acadrust::objects::ObjectType;
use acadrust::types::Vector2;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::ViewProfile;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec_convert::PdfPlot;
use lopdf::Document;
use lopdf::content::Operation;

const PT_PER_MM: f64 = 72.0 / 25.4;

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1670h-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn line(angle_deg: f64, offset: [f64; 2], dashes: &[f64]) -> HatchPatternLine {
    HatchPatternLine {
        angle: angle_deg.to_radians(),
        base_point: Vector2::new(0.0, 0.0),
        offset: Vector2::new(offset[0], offset[1]),
        dash_lengths: dashes.to_vec(),
    }
}

fn ring(x0: f64, y0: f64, x1: f64, y1: f64) -> BoundaryEdge {
    BoundaryEdge::Polyline(PolylineEdge::new(
        vec![
            Vector2::new(x0, y0),
            Vector2::new(x1, y0),
            Vector2::new(x1, y1),
            Vector2::new(x0, y1),
        ],
        true,
    ))
}

/// A millimetre A3 sheet `A3` holding one pattern hatch: `outer` as its boundary, `islands` inside it.
/// Returns the strokes the plot draws as (x0, y0, x1, y1) in millimetres, and the plot's notes.
fn plot(
    tag: &str,
    pattern: Vec<HatchPatternLine>,
    outer: [f64; 4],
    islands: &[[f64; 4]],
) -> (Vec<[f64; 4]>, Vec<String>, f64, Vec<Operation>) {
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
    layout.plot_paper_units = 1;
    layout.plot_type = 5;
    layout.plot_scale_numerator = 1.0;
    layout.plot_scale_denominator = 1.0;
    layout.paper_size = "ISO_A3_(420.00_x_297.00_MM)".into();
    layout.plot_flags.print_lineweights = true;
    let mut p = HatchPattern::new("CUSTOM");
    for l in pattern {
        p.add_line(l);
    }
    let mut hatch = Hatch::with_pattern(p);
    let mut outer_path = BoundaryPath::external();
    outer_path.add_edge(ring(outer[0], outer[1], outer[2], outer[3]));
    hatch.add_path(outer_path);
    for i in islands {
        let mut path = BoundaryPath::with_flags(BoundaryPathFlags::new());
        path.add_edge(ring(i[0], i[1], i[2], i[3]));
        hatch.add_path(path);
    }
    doc.add_entity_to_layout(EntityType::Hatch(Box::new(hatch)), "A3")
        .unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dir = scratch(tag);
    let dest = dir.join("out.pdf");
    let started = Instant::now();
    let out = PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .fills(true)
        .convert_with::<AcadrustProvider>()
        .unwrap();
    let took = started.elapsed().as_secs_f64();
    let pdf = Document::load(&dest).unwrap();
    let ops = pdf
        .get_and_decode_page_content(pdf.get_pages()[&1])
        .unwrap()
        .operations;
    let real = |o: &lopdf::Object| {
        o.as_float()
            .map(f64::from)
            .unwrap_or_else(|_| o.as_i64().unwrap() as f64)
    };
    let mut strokes = Vec::new();
    let mut at: Vec<[f64; 2]> = Vec::new();
    for o in &ops {
        match o.operator.as_str() {
            "m" => {
                at = vec![[
                    real(&o.operands[0]) / PT_PER_MM,
                    real(&o.operands[1]) / PT_PER_MM,
                ]]
            }
            "l" => at.push([
                real(&o.operands[0]) / PT_PER_MM,
                real(&o.operands[1]) / PT_PER_MM,
            ]),
            "S" => {
                if at.len() == 2 {
                    strokes.push([at[0][0], at[0][1], at[1][0], at[1][1]]);
                }
                at.clear();
            }
            _ => {}
        }
    }
    let _ = fs::remove_dir_all(dir);
    (strokes, out.notes, took, ops)
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.05
}

// ---- a custom pattern of two lines ---------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_two_line_custom_pattern_is_drawn_inside_its_boundary() {
    // Horizontal lines every 10 mm and vertical lines every 20 mm, in a boundary from (5, 3) to (95, 58).
    let (strokes, _, _, _) = plot(
        "custom",
        vec![line(0.0, [0.0, 10.0], &[]), line(90.0, [0.0, 20.0], &[])],
        [5.0, 3.0, 95.0, 58.0],
        &[],
    );
    let horizontal: Vec<_> = strokes.iter().filter(|s| near(s[1], s[3])).collect();
    let vertical: Vec<_> = strokes.iter().filter(|s| near(s[0], s[2])).collect();
    assert_eq!(horizontal.len(), 5, "{strokes:?}");
    assert_eq!(vertical.len(), 4, "{strokes:?}");
    for h in &horizontal {
        let (lo, hi) = (h[0].min(h[2]), h[0].max(h[2]));
        assert!(
            near(lo, 5.0) && near(hi, 95.0),
            "{h:?} runs the boundary's width"
        );
    }
    for v in &vertical {
        let (lo, hi) = (v[1].min(v[3]), v[1].max(v[3]));
        assert!(
            near(lo, 3.0) && near(hi, 58.0),
            "{v:?} runs the boundary's height"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_dashed_pattern_line_is_cut_into_dashes() {
    // One horizontal line every 10 mm, 4 mm dash and 2 mm gap, in a 30 mm wide boundary: five dashes.
    let (strokes, _, _, _) = plot(
        "dashes",
        vec![line(0.0, [0.0, 10.0], &[4.0, -2.0])],
        [0.0, 5.0, 30.0, 15.0],
        &[],
    );
    let on_ten: Vec<_> = strokes.iter().filter(|s| near(s[1], 10.0)).collect();
    assert_eq!(on_ten.len(), 5, "{strokes:?}");
    for d in &on_ten {
        assert!(near((d[2] - d[0]).abs(), 4.0), "{d:?} is a 4 mm dash");
    }
}

// ---- a standard pattern ----------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_an_ansi_style_pattern_is_45_degree_lines_at_its_spacing() {
    // ANSI31: lines at 45 degrees, 3.175 mm apart, in a 30 mm square.
    let (strokes, _, _, _) = plot(
        "ansi",
        vec![line(45.0, [0.0, 3.175], &[])],
        [10.0, 10.0, 40.0, 40.0],
        &[],
    );
    // the lines cross the square's 30 x sqrt(2) mm shadow on their normal: 42.4 / 3.175 = 13.4
    assert!(
        (12..=14).contains(&strokes.len()),
        "{} lines",
        strokes.len()
    );
    for s in &strokes {
        assert!(
            near((s[2] - s[0]).abs(), (s[3] - s[1]).abs()),
            "{s:?} is at 45 degrees"
        );
    }
}

// ---- an island -----------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_pattern_lines_stop_at_an_island() {
    // Horizontal lines every 10 mm in a 100 mm square with a 20 mm square island at its middle.
    let (strokes, _, _, _) = plot(
        "island",
        vec![line(0.0, [0.0, 5.0], &[])],
        [0.0, 0.0, 100.0, 100.0],
        &[[40.0, 40.0, 60.0, 60.0]],
    );
    let through_island: Vec<_> = strokes
        .iter()
        .filter(|s| s[1] > 40.5 && s[1] < 59.5 && near(s[1], s[3]))
        .collect();
    assert!(!through_island.is_empty(), "{strokes:?}");
    for s in &through_island {
        let (lo, hi) = (s[0].min(s[2]), s[0].max(s[2]));
        assert!(
            hi <= 40.05 || lo >= 59.95,
            "{s:?} crosses the island between x 40 and 60"
        );
    }
    // each of those rows is two pieces
    let on_fifty: Vec<_> = strokes.iter().filter(|s| near(s[1], 50.0)).collect();
    assert_eq!(on_fifty.len(), 2, "{on_fifty:?}");
}

// ---- hostile patterns ----------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_pattern_too_dense_to_draw_is_refused_and_the_hatch_is_filled() {
    // Lines a ten-millionth of a millimetre apart over 100 mm: a hundred million lines.
    let (strokes, notes, took, ops) = plot(
        "dense",
        vec![line(0.0, [0.0, 1e-7], &[])],
        [0.0, 0.0, 100.0, 100.0],
        &[],
    );
    assert!(took < 60.0, "{took} s");
    assert!(strokes.len() < 1000, "{} strokes", strokes.len());
    assert!(notes.iter().any(|n| n == "hatch-too-dense"), "{notes:?}");
    assert!(
        ops.iter().any(|o| o.operator == "f*"),
        "drawn solid instead"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_tiny_dash_is_refused_too() {
    // A 1e-7 mm dash and gap along a 100 mm line: a billion dashes in one line.
    let (strokes, notes, took, _) = plot(
        "tinydash",
        vec![line(0.0, [0.0, 50.0], &[1e-7, -1e-7])],
        [0.0, 0.0, 100.0, 100.0],
        &[],
    );
    assert!(took < 60.0, "{took} s");
    assert!(strokes.len() < 1000, "{} strokes", strokes.len());
    assert!(notes.iter().any(|n| n == "hatch-too-dense"), "{notes:?}");
}
