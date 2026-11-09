//! Issue #1578, L3: a tiling pattern fill is written as a tiling in stream 25, the cell's paths
//! as primitives in pattern space, the steps and the cell box as the PDF gives them, the matrix
//! as the transform to source space. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::pvec::pdf_page_to_pvec;
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use pvec_facade::PvecReader;
use pvec_facade::components::{
    Fill, FillPaintStream, FillRule, FillSource, HatchStream, Tiling, TilingPrimitive,
};

fn tiling_page(
    paint_type: i64,
    cell: &str,
    matrix: Option<&[f64]>,
    content: &str,
) -> (Option<FillPaintStream>, HatchStream) {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let nums = |v: &[f64]| Object::Array(v.iter().map(|x| Object::Real(*x as f32)).collect());
    let mut dict = dictionary! {
        "PatternType" => 1, "PaintType" => paint_type, "TilingType" => 1,
        "BBox" => nums(&[0.0, 0.0, 10.0, 10.0]), "XStep" => 10, "YStep" => 10,
        "Resources" => Dictionary::new(),
    };
    if let Some(m) = matrix {
        dict.set("Matrix", nums(m));
    }
    let mut cell_stream = Stream::new(dict, cell.as_bytes().to_vec());
    cell_stream.allows_compression = false;
    let pattern = doc.add_object(cell_stream);
    let resources = dictionary! {
        "Pattern" => dictionary! { "P1" => pattern },
        "ColorSpace" => dictionary! { "Cs1" => vec![Object::from("Pattern"), "DeviceRGB".into()] },
    };
    let contents = doc.add_object(Stream::new(Dictionary::new(), content.as_bytes().to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
        "Contents" => contents, "Resources" => resources,
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut pdf = Vec::new();
    doc.save_to(&mut pdf).unwrap();
    let page = extract_page_paths_from_bytes(&pdf, 1, &PdfVectorLimits::default()).unwrap();
    let bytes = pdf_page_to_pvec(&page, "test", "0").unwrap();
    let (streams, _) = PvecReader::open(bytes).unwrap().streams().unwrap();
    let find = |cap| streams.iter().find(|s| s.capability == cap);
    (
        find(FillPaintStream::CAPABILITY).map(|s| FillPaintStream::decode(&s.payload).unwrap()),
        HatchStream::decode(&find(HatchStream::CAPABILITY).unwrap().payload).unwrap(),
    )
}

const FILL: &str = "/Pattern cs /P1 scn 10 10 80 80 re f";

fn only_tiling(fills: Option<FillPaintStream>) -> Tiling {
    let fills = fills.expect("stream 25");
    assert_eq!(fills.assignments[0].paint, FillSource::Tiling(0));
    fills.tilings[0].clone()
}

#[test]
fn a_coloured_cell_is_a_tiling_with_its_paths_as_primitives() {
    let (fills, _) = tiling_page(
        1,
        "1 0 0 rg 0 0 5 5 re f 0 0 1 RG 2 w 0 0 m 10 10 l S",
        None,
        FILL,
    );
    let t = only_tiling(fills);
    assert!(t.coloured);
    assert_eq!((t.x_step, t.y_step), (10.0, 10.0));
    assert_eq!(t.cell, [0.0, 0.0, 10.0, 10.0]);
    let s = 1.0 / 72.0;
    assert_eq!(t.transform, [s, 0.0, 0.0, -s, 0.0, 100.0 * s]);
    assert_eq!(t.primitives.len(), 2);
    match &t.primitives[0] {
        TilingPrimitive::Polygon { rule, rgba, points } => {
            assert_eq!(*rule, FillRule::NonZero);
            assert_eq!(*rgba, 0xFF00_00FF);
            assert_eq!(points.len(), 4);
            assert!(points.contains(&[5.0, 5.0]) && points.contains(&[0.0, 0.0]));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        t.primitives[1],
        TilingPrimitive::Polyline {
            rgba: 0x0000_FFFF,
            width: 2.0,
            points: vec![[0.0, 0.0], [10.0, 10.0]]
        }
    );
}

#[test]
fn the_pattern_matrix_is_the_transform_not_the_ctm() {
    let (fills, _) = tiling_page(
        1,
        "0 g 0 0 5 5 re f",
        Some(&[2.0, 0.0, 0.0, 2.0, 10.0, 20.0]),
        &format!("3 0 0 3 5 5 cm {FILL}"),
    );
    let s = 1.0 / 72.0;
    assert_eq!(
        only_tiling(fills).transform,
        [2.0 * s, 0.0, 0.0, -2.0 * s, 10.0 * s, 80.0 * s]
    );
}

#[test]
fn an_uncoloured_cell_takes_the_colour_of_the_scn_in_the_hatch_record() {
    let (fills, hatch) = tiling_page(
        2,
        "0 0 5 5 re f",
        None,
        "/Cs1 cs 0.2 0.4 0.6 /P1 scn 10 10 80 80 re f",
    );
    assert!(!only_tiling(fills).coloured);
    match &hatch.records[0].fill {
        Fill::Solid { rgba } => assert_eq!(*rgba, 0x3366_99FF),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_cell_with_a_curve_and_a_cm_is_flattened_in_pattern_space() {
    let (fills, _) = tiling_page(
        1,
        "q 2 0 0 2 0 0 cm 0 g 0 0 m 1 0 2 1 2 2 c 0 2 l h f Q",
        None,
        FILL,
    );
    let t = only_tiling(fills);
    match &t.primitives[0] {
        TilingPrimitive::Polygon { points, .. } => {
            assert!(points.len() > 4);
            assert!(points.iter().all(|p| p[0] <= 4.0 && p[1] <= 4.0));
            assert!(points.contains(&[4.0, 4.0]));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_cell_this_does_not_read_leaves_the_flat_colour() {
    let (fills, _) = tiling_page(1, "BT /F1 12 Tf (x) Tj ET", None, FILL);
    let fills = fills.expect("the rule is still written");
    assert!(fills.tilings.is_empty());
    assert_eq!(fills.assignments[0].paint, FillSource::Own);
}

#[test]
fn two_fills_of_one_pattern_share_one_tiling() {
    let (fills, _) = tiling_page(1, "0 g 0 0 5 5 re f", None, &format!("{FILL} 0 0 5 5 re f"));
    let fills = fills.unwrap();
    assert_eq!(fills.tilings.len(), 1);
    assert_eq!(fills.assignments.len(), 2);
}
