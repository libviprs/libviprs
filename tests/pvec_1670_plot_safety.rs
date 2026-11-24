//! Issue #1670, what a plot may touch and refuse: it writes only to names it made itself and replaces
//! the destination only when the whole plot worked, and it refuses a page that is not a PDF page.
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;

use acadrust::objects::ObjectType;
use acadrust::{CadDocument, DwgWriter};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{SourceLimits, SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PdfPlot, PvecConvertError};

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1670s-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// A drawing with paper layout `A3` of the given paper size in millimetres, and a line on it.
fn drawing(width: f64, height: f64) -> Vec<u8> {
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
    layout.paper_width = width;
    layout.paper_height = height;
    layout.plot_paper_units = 1;
    layout.plot_type = 5;
    layout.plot_scale_numerator = 1.0;
    layout.plot_scale_denominator = 1.0;
    layout.paper_size = format!("USER_({width:.2}_x_{height:.2}_MM)");
    layout.plot_flags.print_lineweights = true;
    let line = acadrust::entities::Line::from_coords(10.0, 10.0, 0.0, 100.0, 50.0, 0.0);
    doc.add_entity_to_layout(acadrust::EntityType::Line(line), "A3")
        .unwrap();
    DwgWriter::write_to_vec(&doc).unwrap()
}

fn plot<'a>(
    bytes: &'a [u8],
    dest: &'a std::path::Path,
    limits: Option<SourceLimits>,
) -> Result<libviprs::pvec_convert::PdfPlotOutput, PvecConvertError> {
    let mut p = PdfPlot::new(CadSource::Bytes(bytes), dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"]);
    if let Some(l) = limits {
        p = p.limits(l);
    }
    p.convert_with::<AcadrustProvider>()
}

fn names(dir: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_directory_the_plot_did_not_make_is_never_removed() {
    // The old scratch name was the destination with the extension `plot-work`.
    let dir = scratch("keepdir");
    let theirs = dir.join("out.plot-work");
    fs::create_dir_all(&theirs).unwrap();
    fs::write(theirs.join("keep.txt"), b"theirs").unwrap();
    plot(&drawing(420.0, 297.0), &dir.join("out.pdf"), None).unwrap();
    assert_eq!(fs::read(theirs.join("keep.txt")).unwrap(), b"theirs");
    assert_eq!(
        names(&dir),
        ["out.pdf", "out.plot-work"],
        "nothing else is left behind"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_failed_plot_leaves_an_existing_output_alone() {
    let dir = scratch("failed");
    let dest = dir.join("out.pdf");
    fs::write(&dest, b"keep me").unwrap();
    let mut limits = SourceLimits::default();
    limits.max_entities = 0;
    assert!(plot(&drawing(420.0, 297.0), &dest, Some(limits)).is_err());
    assert_eq!(fs::read(&dest).unwrap(), b"keep me");
    assert_eq!(
        names(&dir),
        ["out.pdf"],
        "no scratch file or directory is left"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_good_plot_replaces_an_existing_output() {
    let dir = scratch("replace");
    let dest = dir.join("out.pdf");
    fs::write(&dest, b"old").unwrap();
    plot(&drawing(420.0, 297.0), &dest, None).unwrap();
    assert!(fs::read(&dest).unwrap().starts_with(b"%PDF"));
    assert_eq!(names(&dir), ["out.pdf"]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_page_over_the_pdf_limit_is_refused() {
    // A PDF page is at most 14400 points a side, 5080 mm. 6000 mm is over it.
    let dir = scratch("toobig");
    let dest = dir.join("out.pdf");
    let err = plot(&drawing(6000.0, 3000.0), &dest, None).unwrap_err();
    assert_eq!(err.code(), "PAGE_REFUSED", "{err}");
    assert!(!dest.exists());
    assert_eq!(names(&dir), Vec::<String>::new());
    let _ = fs::remove_dir_all(dir);
}
