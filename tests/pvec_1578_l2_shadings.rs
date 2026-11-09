//! Issue #1578, L2: `sh` and shading patterns are read into gradients, a PDF function sampled to
//! colour stops in sRGB, so a converter can write them as PVec gradient fills.

use libviprs::pvec::pdf_shading::{PdfFillPattern, PdfGradientKind};
use libviprs::pvec::pdf_vector::{PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

/// A one page PDF (612 by 792) whose resources `make` builds in the document.
fn build(make: impl FnOnce(&mut Document) -> Dictionary, content: &str) -> PdfVectorPage {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let resources = make(&mut doc);
    let contents = doc.add_object(Stream::new(Dictionary::new(), content.as_bytes().to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
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
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    extract_page_paths_from_bytes(&out, 1, &PdfVectorLimits::default()).unwrap()
}

fn nums(v: &[f64]) -> Object {
    Object::Array(v.iter().map(|x| Object::Real(*x as f32)).collect())
}

fn exp_fn(c0: &[f64], c1: &[f64], n: f64) -> Object {
    Object::Dictionary(dictionary! {
        "FunctionType" => 2, "Domain" => nums(&[0.0, 1.0]),
        "C0" => nums(c0), "C1" => nums(c1), "N" => Object::Real(n as f32),
    })
}

fn axial(function: Object, cs: Object) -> Object {
    Object::Dictionary(dictionary! {
        "ShadingType" => 2, "ColorSpace" => cs,
        "Coords" => nums(&[0.0, 0.0, 100.0, 0.0]),
        "Extend" => vec![Object::Boolean(true), Object::Boolean(false)],
        "Function" => function,
    })
}

fn shading_page(shading: Object, content: &str) -> PdfVectorPage {
    build(
        |_| dictionary! { "Shading" => dictionary! { "Sh1" => shading } },
        content,
    )
}

fn rgb() -> Object {
    "DeviceRGB".into()
}

#[test]
fn an_axial_sh_reads_its_axis_extend_and_two_stops() {
    let p = shading_page(
        axial(exp_fn(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0], 1.0), rgb()),
        "/Sh1 sh",
    );
    assert_eq!(p.shading_paints, 1);
    assert_eq!(p.shadings.len(), 1);
    assert_eq!(p.shading_ops.len(), 1);
    assert_eq!(p.shading_ops[0].shading, 0);
    let s = &p.shadings[0];
    assert_eq!(s.kind, PdfGradientKind::Axial);
    assert_eq!(s.coords, vec![0.0, 0.0, 100.0, 0.0]);
    assert_eq!(s.extend, [true, false]);
    assert_eq!(s.matrix, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let stops: Vec<_> = s.stops.iter().map(|t| (t.offset, t.rgb)).collect();
    assert_eq!(stops, vec![(0.0, [255, 0, 0]), (1.0, [0, 0, 255])]);
}

#[test]
fn sh_takes_the_ctm_in_force_as_the_gradient_to_page_matrix() {
    let p = shading_page(
        axial(exp_fn(&[0.0], &[1.0], 1.0), "DeviceGray".into()),
        "2 0 0 2 10 20 cm /Sh1 sh",
    );
    assert_eq!(p.shadings[0].matrix, [2.0, 0.0, 0.0, 2.0, 10.0, 20.0]);
    assert_eq!(p.shadings[0].stops[1].rgb, [255, 255, 255]);
}

#[test]
fn a_radial_sh_keeps_both_circles() {
    let radial = Object::Dictionary(dictionary! {
        "ShadingType" => 3, "ColorSpace" => rgb(),
        "Coords" => nums(&[10.0, 20.0, 0.0, 30.0, 40.0, 50.0]),
        "Function" => exp_fn(&[0.0, 1.0, 0.0], &[1.0, 1.0, 1.0], 1.0),
    });
    let s = &shading_page(radial, "/Sh1 sh").shadings[0];
    assert_eq!(s.kind, PdfGradientKind::Radial);
    assert_eq!(s.coords, vec![10.0, 20.0, 0.0, 30.0, 40.0, 50.0]);
    assert_eq!(s.extend, [false, false]);
}

#[test]
fn a_stitching_function_keeps_its_joint_and_its_exact_colours() {
    let stitch = build(
        |doc| {
            let f1 = doc.add_object(exp_fn(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 1.0));
            let f2 = doc.add_object(exp_fn(&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0], 1.0));
            let f = Object::Dictionary(dictionary! {
                "FunctionType" => 3, "Domain" => nums(&[0.0, 1.0]),
                "Functions" => vec![Object::from(f1), Object::from(f2)],
                "Bounds" => nums(&[0.5]), "Encode" => nums(&[0.0, 1.0, 0.0, 1.0]),
            });
            dictionary! { "Shading" => dictionary! { "Sh1" => axial(f, rgb()) } }
        },
        "/Sh1 sh",
    );
    let stops: Vec<_> = stitch.shadings[0]
        .stops
        .iter()
        .map(|t| (t.offset, t.rgb))
        .collect();
    assert_eq!(
        stops,
        vec![(0.0, [255, 0, 0]), (0.5, [0, 255, 0]), (1.0, [0, 0, 255])]
    );
}

#[test]
fn a_hard_step_in_a_stitching_function_is_two_stops_at_one_offset() {
    let p = build(
        |doc| {
            let f1 = doc.add_object(exp_fn(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0));
            let f2 = doc.add_object(exp_fn(&[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], 1.0));
            let f = Object::Dictionary(dictionary! {
                "FunctionType" => 3, "Domain" => nums(&[0.0, 1.0]),
                "Functions" => vec![Object::from(f1), Object::from(f2)],
                "Bounds" => nums(&[0.25]), "Encode" => nums(&[0.0, 1.0, 0.0, 1.0]),
            });
            dictionary! { "Shading" => dictionary! { "Sh1" => axial(f, rgb()) } }
        },
        "/Sh1 sh",
    );
    let stops: Vec<_> = p.shadings[0]
        .stops
        .iter()
        .map(|t| (t.offset, t.rgb))
        .collect();
    assert_eq!(
        stops,
        vec![
            (0.0, [255, 0, 0]),
            (0.25, [255, 0, 0]),
            (0.25, [0, 0, 255]),
            (1.0, [0, 0, 255])
        ]
    );
}

#[test]
fn a_sampled_function_is_read_and_a_bend_adds_a_stop() {
    let p = build(
        |doc| {
            // Three 8 bit gray samples 0, 255, 0: a tent, which two stops cannot hold.
            let mut f = Stream::new(
                dictionary! {
                    "FunctionType" => 0, "Domain" => nums(&[0.0, 1.0]),
                    "Range" => nums(&[0.0, 1.0]), "Size" => vec![3.into()],
                    "BitsPerSample" => 8,
                },
                vec![0, 255, 0],
            );
            f.allows_compression = false;
            let f = doc.add_object(f);
            dictionary! { "Shading" => dictionary! {
            "Sh1" => axial(Object::from(f), "DeviceGray".into()) } }
        },
        "/Sh1 sh",
    );
    let s = &p.shadings[0];
    let at_half = s
        .stops
        .iter()
        .find(|t| t.offset == 0.5)
        .expect("a stop at the peak");
    assert_eq!(at_half.rgb, [255, 255, 255]);
    assert!(s.stops.len() <= 256);
    assert!(s.max_error <= 1.0, "{}", s.max_error);
}

#[test]
fn a_postscript_calculator_function_is_evaluated() {
    let p = build(
        |doc| {
            let mut f = Stream::new(
                dictionary! {
                    "FunctionType" => 4, "Domain" => nums(&[0.0, 1.0]),
                    "Range" => nums(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
                },
                b"{ dup 0.5 mul exch 1 exch sub 0 }".to_vec(),
            );
            f.allows_compression = false;
            let f = doc.add_object(f);
            dictionary! { "Shading" => dictionary! { "Sh1" => axial(Object::from(f), rgb()) } }
        },
        "/Sh1 sh",
    );
    let stops: Vec<_> = p.shadings[0]
        .stops
        .iter()
        .map(|t| (t.offset, t.rgb))
        .collect();
    assert_eq!(stops, vec![(0.0, [0, 255, 0]), (1.0, [128, 0, 0])]);
}

#[test]
fn a_curved_function_is_sampled_within_one_level_of_the_function() {
    let p = shading_page(
        axial(exp_fn(&[0.0], &[1.0], 3.0), "DeviceGray".into()),
        "/Sh1 sh",
    );
    let s = &p.shadings[0];
    assert!(
        s.stops.len() > 3 && s.stops.len() <= 256,
        "{}",
        s.stops.len()
    );
    assert!(s.max_error <= 1.0, "{}", s.max_error);
    // The sampled gray at 0.5 is 0.5^3 of 255.
    for pair in s.stops.windows(2) {
        let mid = (pair[0].offset + pair[1].offset) / 2.0;
        let want = (mid.powi(3) * 255.0).round();
        let got = (f64::from(pair[0].rgb[0]) + f64::from(pair[1].rgb[0])) / 2.0;
        assert!((got - want).abs() <= 1.5, "at {mid}: {got} vs {want}");
    }
}

#[test]
fn a_separation_colour_space_goes_through_its_tint_transform() {
    let p = build(
        |doc| {
            let tint = doc.add_object(exp_fn(&[0.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 1.0, 0.0], 1.0));
            let cs = Object::Array(vec![
                "Separation".into(),
                "Spot".into(),
                "DeviceCMYK".into(),
                tint.into(),
            ]);
            dictionary! { "Shading" => dictionary! {
            "Sh1" => axial(exp_fn(&[0.0], &[1.0], 1.0), cs) } }
        },
        "/Sh1 sh",
    );
    let stops: Vec<_> = p.shadings[0].stops.iter().map(|t| t.rgb).collect();
    assert_eq!(stops, vec![[255, 255, 255], [255, 0, 0]]);
}

#[test]
fn an_icc_based_space_is_read_by_its_component_count() {
    let p = build(
        |doc| {
            let icc = doc.add_object(Stream::new(dictionary! { "N" => 3 }, vec![0; 4]));
            let cs = Object::Array(vec!["ICCBased".into(), icc.into()]);
            dictionary! { "Shading" => dictionary! {
            "Sh1" => axial(exp_fn(&[0.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 1.0), cs) } }
        },
        "/Sh1 sh",
    );
    assert_eq!(p.shadings[0].stops[1].rgb, [0, 255, 0]);
}

fn pattern_page(matrix: Option<&[f64]>, content: &str) -> PdfVectorPage {
    let mut pat = dictionary! {
        "PatternType" => 2,
        "Shading" => axial(exp_fn(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0], 1.0), rgb()),
    };
    if let Some(m) = matrix {
        pat.set("Matrix", nums(m));
    }
    build(
        |_| dictionary! { "Pattern" => dictionary! { "P1" => pat } },
        content,
    )
}

#[test]
fn a_shading_pattern_fill_names_its_gradient_and_counts_as_a_pattern() {
    let p = pattern_page(None, "/Pattern cs /P1 scn 0 0 10 10 re f");
    assert_eq!(p.pattern_selections, 1);
    assert_eq!(p.entities[0].fill_pattern, Some(PdfFillPattern::Shading(0)));
    assert_eq!(p.shadings[0].kind, PdfGradientKind::Axial);
    assert_eq!(p.shadings[0].matrix, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    assert!(p.shading_ops.is_empty());
}

#[test]
fn a_pattern_matrix_is_relative_to_the_page_not_to_the_cm_in_force() {
    let p = pattern_page(
        Some(&[1.0, 0.0, 0.0, 1.0, 5.0, 6.0]),
        "3 0 0 3 100 100 cm /Pattern cs /P1 scn 0 0 10 10 re f",
    );
    assert_eq!(p.shadings[0].matrix, [1.0, 0.0, 0.0, 1.0, 5.0, 6.0]);
}

#[test]
fn a_plain_colour_ends_the_pattern_and_q_restores_it() {
    let p = pattern_page(
        None,
        "/Pattern cs /P1 scn q 1 0 0 rg 0 0 5 5 re f Q 0 0 6 6 re f 0 g 0 0 7 7 re f",
    );
    let got: Vec<_> = p
        .entities
        .iter()
        .map(|e| e.fill_pattern.is_some())
        .collect();
    assert_eq!(got, vec![false, true, false]);
}

#[test]
fn a_shading_type_this_does_not_sample_is_counted_and_not_recorded() {
    let function_based = Object::Dictionary(dictionary! {
        "ShadingType" => 1, "ColorSpace" => rgb(),
        "Function" => exp_fn(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], 1.0),
    });
    let p = shading_page(function_based, "/Sh1 sh");
    assert_eq!(p.shading_paints, 1);
    assert!(p.shadings.is_empty() && p.shading_ops.is_empty());
    assert_eq!(p.shadings_not_sampled, 1);
}
