//! Issue #1670: the default plot files. The built-in monochrome plot style table and the built-in PDF
//! PC3 are real files: the table is a compressed .ctb and the PC3 is the JSON kind, both load through
//! the bounded loaders of #1673 and write back byte for byte. The classic compressed PC3 a user may
//! hand over is recognised as a binary kind and kept as it is. The generated .ctb, handed to a plot
//! explicitly, gives the built-in page for every ACI object (a true colour object keeps its own colour
//! under any CTB, which the built-in default does not). Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;

use acadrust::entities::Line;
use acadrust::objects::ObjectType;
use acadrust::types::{Color, LineWeight};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::cad::plot_files::{
    Pc3, Pc3File, PlotDevice, default_pc3, default_plot_style, load_pc3, load_plot_style,
};
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PdfPlot;
use lopdf::Document;

const PT_PER_MM: f64 = 72.0 / 25.4;

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_default_ctb_loads_and_writes_back_byte_for_byte() {
    let bytes = default_plot_style().to_bytes().unwrap();
    assert!(
        bytes.starts_with(b"PIAFILEVERSION_2.0,CTBVER1,compress"),
        "the compressed CTB kind"
    );
    let loaded = load_plot_style("libviprs-monochrome.ctb", &bytes).unwrap();
    assert!(!loaded.is_stb);
    assert_eq!(loaded.to_bytes().unwrap(), bytes);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_default_ctb_has_255_black_entries_at_lineweight_9() {
    let table = default_plot_style();
    assert_eq!(table.aci_entries.len(), 256);
    for aci in 1..=255usize {
        let e = &table.aci_entries[aci];
        assert_eq!(e.color, Some([0, 0, 0]), "ACI {aci} is not black");
        assert_eq!(e.screening, 100, "ACI {aci} is screened");
        assert_eq!(e.lineweight, 9, "ACI {aci} is not lineweight 9");
    }
    // A style's lineweight N is entry N - 1 of the list (#1692): 9 is 0.25 mm.
    assert_eq!(table.lineweights[8], 0.25);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_default_ctb_survives_the_loader_with_its_entries() {
    let bytes = default_plot_style().to_bytes().unwrap();
    let loaded = load_plot_style("libviprs-monochrome.ctb", &bytes).unwrap();
    for aci in 1..=255usize {
        let e = &loaded.aci_entries[aci];
        assert_eq!((e.color, e.lineweight), (Some([0, 0, 0]), 9), "ACI {aci}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_default_pc3_loads_as_the_json_kind_and_writes_back_byte_for_byte() {
    let bytes = default_pc3().to_bytes();
    assert!(bytes.starts_with(b"PIAFILEVERSION_3.0,json\n"));
    let Pc3File::Json(file) = load_pc3(&bytes).unwrap() else {
        panic!("the default PC3 is not the JSON kind");
    };
    assert_eq!(file.to_bytes(), bytes);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_default_pc3_names_the_pdf_device_and_an_a3_page() {
    let pc3 = default_pc3();
    let device = PlotDevice::from_pc3(&pc3).unwrap();
    assert_eq!(
        device,
        PlotDevice {
            paper_mm: [420.0, 297.0],
            printable_mm: [0.0, 0.0, 420.0, 297.0],
            raster_limit_dpi: Some(400.0),
            text_as_text: Some(true),
        }
    );
    let named = pc3.device().unwrap();
    assert!(named.model.contains("PDF"), "{named:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_default_pc3_carries_nothing_from_a_machine_or_a_person() {
    let text = String::from_utf8(default_pc3().to_bytes()).unwrap();
    for needle in [
        "C:\\\\",
        "Program Files",
        "AutoCAD",
        "Users",
        "pathname\" : \"/",
    ] {
        assert!(!text.contains(needle), "the default PC3 holds {needle}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_classic_compressed_pc3_is_a_binary_kind_kept_as_it_is() {
    let mut classic = b"PIAFILEVERSION_2.0,PC3VER1,compress\r\npmzlibcodec".to_vec();
    classic.extend_from_slice(&[0, 1, 2, 3, 250, 251, 252, 253]);
    let Pc3File::Binary(kept) = load_pc3(&classic).unwrap() else {
        panic!("the classic compressed PC3 is not a binary kind");
    };
    assert_eq!(kept.bytes, classic);
    assert_eq!(Pc3File::Binary(kept).to_bytes(), classic);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_ctb_is_still_no_pc3() {
    let ctb = default_plot_style().to_bytes().unwrap();
    assert_eq!(load_pc3(&ctb).unwrap_err().code(), "plot-file-unrecognised");
    assert!(Pc3::parse(&ctb).is_err());
}

// ---- the generated .ctb plots as the built-in default does --------------------------------

fn line(y: f64, color: Color, weight: i16) -> EntityType {
    let mut l = Line::from_coords(10.0, y, 0.0, 100.0, y, 0.0);
    l.common.color = color;
    l.common.line_weight = LineWeight::Value(weight);
    EntityType::Line(l)
}

/// One A3 layout of ACI 1, 5 and 7 lines, a zero lineweight line and a true colour line.
fn drawing() -> Vec<u8> {
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
    layout.plot_flags.plot_plot_styles = true;
    layout.plot_paper_units = 1;
    layout.plot_type = 5;
    layout.plot_scale_numerator = 1.0;
    layout.plot_scale_denominator = 1.0;
    layout.paper_size = "USER_(420.00_x_297.00_MM)".into();
    for e in [
        line(10.0, Color::Index(1), 25),
        line(20.0, Color::Index(5), 50),
        line(30.0, Color::Index(7), 13),
        line(40.0, Color::Index(2), 0),
        line(
            50.0,
            Color::Rgb {
                r: 200,
                g: 30,
                b: 30,
            },
            70,
        ),
    ] {
        doc.add_entity_to_layout(e, "A3").unwrap();
    }
    DwgWriter::write_to_vec(&doc).unwrap()
}

fn strokes(tag: &str, with: Option<&[u8]>) -> Vec<([u8; 3], f64)> {
    let dir: PathBuf =
        std::env::temp_dir().join(format!("libviprs-1670f-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let bytes = drawing();
    let dest = dir.join("out.pdf");
    let mut p = PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"]);
    if let Some(ctb) = with {
        p = p.plot_style_table(load_plot_style("libviprs-monochrome.ctb", ctb).unwrap());
    }
    p.convert_with::<AcadrustProvider>().unwrap();
    let doc = Document::load(&dest).unwrap();
    let ops = doc
        .get_and_decode_page_content(doc.get_pages()[&1])
        .unwrap()
        .operations;
    let real = |o: &lopdf::Object| {
        o.as_float()
            .map(f64::from)
            .unwrap_or_else(|_| o.as_i64().unwrap() as f64)
    };
    let (mut rgb, mut w, mut out) = ([0u8; 3], 0.0, Vec::new());
    for o in &ops {
        match o.operator.as_str() {
            "RG" => {
                let v: Vec<f64> = o.operands.iter().map(real).collect();
                rgb = [v[0], v[1], v[2]].map(|c| (c * 255.0).round() as u8);
            }
            "w" => w = real(&o.operands[0]),
            "S" => out.push((rgb, w)),
            _ => {}
        }
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_generated_ctb_plots_as_the_built_in_default_for_every_aci_object() {
    let ctb = default_plot_style().to_bytes().unwrap();
    let built_in = strokes("builtin", None);
    let generated = strokes("generated", Some(&ctb));
    assert_eq!(built_in.len(), 5, "{built_in:?}");
    assert_eq!(generated.len(), 5, "{generated:?}");
    let quarter = 0.25 * PT_PER_MM;
    for ((b, g), n) in built_in.iter().zip(&generated).zip(1..) {
        assert_eq!(b.0, [0, 0, 0], "built-in stroke {n}");
        assert!((b.1 - quarter).abs() < 0.01, "built-in stroke {n}: {b:?}");
        if n < 5 {
            assert_eq!(g.0, b.0, "stroke {n}");
            assert!((g.1 - b.1).abs() < 0.01, "stroke {n}: {g:?} against {b:?}");
        }
    }
    // The fifth line is true colour: a CTB says nothing about it, the built-in default blackens it.
    assert_eq!(generated[4].0, [200, 30, 30]);
    assert!(
        (generated[4].1 - 0.7 * PT_PER_MM).abs() < 0.01,
        "{generated:?}"
    );
}
