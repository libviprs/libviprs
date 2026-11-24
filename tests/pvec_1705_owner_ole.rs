//! Issue #1705, the owner's own drawing: a page made of embedded objects draws them.
//!
//! The files are private and are never committed. Point `LIBVIPRS_OWNER_PLOT_DIR` at the directory that
//! holds the drawing (`*.dwg`). With the variable unset, or no drawing in it, the test says so and
//! passes, so it is inert on every machine but the owner's. `LIBVIPRS_OWNER_PLOT_OUT` names a directory
//! to keep the PDFs in (default: a scratch directory). Needs `--features pvec-convert`.
//!
//! The first print layout is plotted twice, without and with `ole_previews`, and the second plot must
//! show more text and more strokes than the first, which is what the embedded objects add.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{DwgProvider, LayoutKind, ViewProfile};
use libviprs::pvec_convert::PdfPlot;
use lopdf::Document;

fn first_dwg(dir: &Path) -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("dwg")))
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// How many text-showing and path-painting operators the first page has.
fn ink(pdf: &Path) -> (usize, usize) {
    let doc = Document::load(pdf).unwrap();
    let id = doc.get_pages()[&1];
    let ops = doc.get_and_decode_page_content(id).unwrap().operations;
    let n = |names: &[&str]| {
        ops.iter()
            .filter(|o| names.contains(&o.operator.as_str()))
            .count()
    };
    (n(&["Tj", "TJ"]), n(&["l", "m"]))
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_the_first_sheet_gains_text_and_strokes_from_its_embedded_objects() {
    let Some(dir) = std::env::var_os("LIBVIPRS_OWNER_PLOT_DIR").map(PathBuf::from) else {
        eprintln!("LIBVIPRS_OWNER_PLOT_DIR is not set: skipped");
        return;
    };
    let Some(dwg) = first_dwg(&dir) else {
        eprintln!("no drawing in LIBVIPRS_OWNER_PLOT_DIR: skipped");
        return;
    };
    let out = std::env::var_os("LIBVIPRS_OWNER_PLOT_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("libviprs-1705-owner-{}", std::process::id()))
        });
    fs::create_dir_all(&out).unwrap();
    let provider = AcadrustProvider::open(CadSource::Path(&dwg)).unwrap();
    let first = provider
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Paper)
        .expect("a print layout")
        .name;
    let mut seen = Vec::new();
    for (tag, on) in [("off", false), ("on", true)] {
        let dest = out.join(format!("{tag}.pdf"));
        PdfPlot::new(CadSource::Path(&dwg), &dest)
            .view_profile(ViewProfile::print_profile(&first))
            .layouts([first.as_str()])
            .texts(true)
            .fills(true)
            .ole_previews(on)
            .convert()
            .unwrap();
        seen.push(ink(&dest));
    }
    assert!(
        seen[1].0 > seen[0].0 && seen[1].1 > seen[0].1,
        "without {:?}, with {:?}",
        seen[0],
        seen[1]
    );
}
