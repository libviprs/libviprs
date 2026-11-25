//! Issue #1704: what a PDF plot writes and keeps while it runs. Pages go to the file as each
//! sheet finishes, the scratch holds one sheet at a time and no per entity report, a plot that stops
//! leaves no partial file and an existing output alone, and the bytes are the same every time.
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use libviprs::cad::CadSource;
use libviprs::observe::{EngineEvent, EngineObserver};
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::{
    CancelToken,
    pvec_convert::{PdfPlot, PdfPlotOutput, PvecConvertError},
};

#[path = "common/qpdf.rs"]
mod qpdf;
#[path = "pvec_1704_support/mod.rs"]
mod support;

fn scratch(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("libviprs-1704s-{tag}-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn plotter<'a>(bytes: &'a [u8], dest: &'a Path, sheets: u32) -> PdfPlot<'a> {
    PdfPlot::new(CadSource::Bytes(bytes), dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("S001"))
        .layouts((1..=sheets).map(|n| format!("S{n:03}")))
}

fn run(bytes: &[u8], dest: &Path, sheets: u32) -> Result<PdfPlotOutput, PvecConvertError> {
    plotter(bytes, dest, sheets).convert_with::<support::Synthetic>()
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// Every file under `dir`, however deep.
fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            files(&p, out);
        } else {
            out.push(p);
        }
    }
}

/// Looks in the destination directory at every event: how many sheet artifacts and how many
/// reports are lying about, at the most.
struct Watch {
    dir: PathBuf,
    artifacts: AtomicUsize,
    reports: AtomicUsize,
    finished: AtomicUsize,
    cancel_after: Option<(usize, CancelToken)>,
}

impl EngineObserver for Watch {
    fn on_event(&self, event: EngineEvent) {
        let mut all = Vec::new();
        files(&self.dir, &mut all);
        let name = |p: &PathBuf| p.file_name().unwrap().to_string_lossy().into_owned();
        let artifacts = all.iter().filter(|p| name(p).ends_with(".pvec")).count();
        let reports = all
            .iter()
            .filter(|p| name(p).ends_with(".report.json"))
            .count();
        self.artifacts.fetch_max(artifacts, Ordering::Relaxed);
        if matches!(event, EngineEvent::ConversionSheetFinished { .. }) {
            // Once a sheet is finished its report is not needed by anyone.
            self.reports.fetch_max(reports, Ordering::Relaxed);
            let n = self.finished.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some((at, cancel)) = &self.cancel_after
                && n == *at
            {
                cancel.cancel();
            }
        }
    }
}

fn watch(dir: &Path, cancel_after: Option<(usize, CancelToken)>) -> Arc<Watch> {
    Arc::new(Watch {
        dir: dir.to_path_buf(),
        artifacts: AtomicUsize::new(0),
        reports: AtomicUsize::new(0),
        finished: AtomicUsize::new(0),
        cancel_after,
    })
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_the_scratch_never_holds_more_than_one_sheet() {
    let dir = scratch("scratch");
    let dest = dir.join("plot.pdf");
    let bytes = support::drawing(6, 50);
    let w = watch(&dir, None);
    let out = plotter(&bytes, &dest, 6)
        .observer(w.clone())
        .convert_with::<support::Synthetic>()
        .unwrap();
    assert_eq!(out.pages, 6);
    assert!(
        w.artifacts.load(Ordering::Relaxed) <= 1,
        "{} sheet artifacts were on disk at once",
        w.artifacts.load(Ordering::Relaxed)
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_a_finished_sheet_leaves_no_report_behind() {
    let dir = scratch("report");
    let dest = dir.join("plot.pdf");
    let bytes = support::drawing(4, 50);
    let w = watch(&dir, None);
    plotter(&bytes, &dest, 4)
        .observer(w.clone())
        .convert_with::<support::Synthetic>()
        .unwrap();
    assert_eq!(
        w.reports.load(Ordering::Relaxed),
        0,
        "a per entity report was still on disk when its sheet finished"
    );
    assert_eq!(names(&dir), ["plot.pdf"]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_a_cancel_part_way_leaves_no_file_and_an_existing_output_alone() {
    let dir = scratch("cancel");
    let dest = dir.join("plot.pdf");
    fs::write(&dest, b"keep me").unwrap();
    let bytes = support::drawing(8, 50);
    let cancel = CancelToken::new();
    let w = watch(&dir, Some((3, cancel.clone())));
    let err = plotter(&bytes, &dest, 8)
        .cancel(cancel)
        .observer(w.clone())
        .convert_with::<support::Synthetic>()
        .unwrap_err();
    assert_eq!(err.code(), "CANCELLED", "{err}");
    assert_eq!(
        w.finished.load(Ordering::Relaxed),
        3,
        "it stopped at the third sheet"
    );
    assert_eq!(fs::read(&dest).unwrap(), b"keep me");
    assert_eq!(
        names(&dir),
        ["plot.pdf"],
        "no temporary file or scratch directory"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_a_deadline_part_way_leaves_no_file() {
    let dir = scratch("deadline");
    let dest = dir.join("plot.pdf");
    let bytes = support::drawing(60, 4_000);
    let err = plotter(&bytes, &dest, 60)
        .deadline(Duration::from_millis(400))
        .convert_with::<support::Synthetic>()
        .unwrap_err();
    assert_eq!(err.code(), "DEADLINE", "{err}");
    assert!(!dest.exists());
    assert_eq!(names(&dir), Vec::<String>::new());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_the_same_plot_is_the_same_bytes_every_time() {
    let dir = scratch("same");
    let bytes = support::drawing(3, 300);
    let (a, b) = (dir.join("a.pdf"), dir.join("b.pdf"));
    run(&bytes, &a, 3).unwrap();
    run(&bytes, &b, 3).unwrap();
    assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_a_streamed_file_has_one_xref_table_and_the_pages_in_order() {
    let dir = scratch("shape");
    let dest = dir.join("plot.pdf");
    let bytes = support::drawing(5, 40);
    run(&bytes, &dest, 5).unwrap();
    let raw = fs::read(&dest).unwrap();
    assert!(raw.starts_with(b"%PDF-1.7"));
    assert!(raw.ends_with(b"%%EOF\n"));
    let text = String::from_utf8_lossy(&raw);
    assert_eq!(
        text.matches("\nxref\n").count(),
        1,
        "one cross reference table"
    );
    assert!(!text.contains("/ObjStm"), "no object streams");
    assert!(
        !text.contains("/ID"),
        "no document id: the output is a function of the drawing"
    );
    // The streams carry their length as an indirect object, written after the stream, which is
    // how a page is written without being held.
    let indirect = text
        .match_indices("/Length ")
        .filter(|(at, _)| {
            let rest = &text[at + 8..];
            let digits = rest.chars().take_while(char::is_ascii_digit).count();
            digits > 0 && rest[digits..].starts_with(" 0 R")
        })
        .count();
    assert!(indirect >= 5, "{indirect} streams with an indirect /Length");
    let doc = lopdf::Document::load(&dest).unwrap();
    assert_eq!(doc.get_pages().len(), 5);
    qpdf::check_pdf(&dest).unwrap();
    let _ = fs::remove_dir_all(dir);
}

/// Every in-use entry of the cross reference table points at the byte where its own
/// `N 0 obj` starts, and `startxref` points at the table.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_every_xref_offset_lands_on_its_object() {
    let dir = scratch("offsets");
    let dest = dir.join("plot.pdf");
    let bytes = support::drawing(4, 30);
    run(&bytes, &dest, 4).unwrap();
    // Streams are binary, so offsets are read against the bytes and not a decoded string.
    let raw = fs::read(&dest).unwrap();
    let find = |needle: &[u8]| {
        raw.windows(needle.len())
            .rposition(|w| w == needle)
            .unwrap()
    };
    let line = |at: usize| {
        let end = raw[at..].iter().position(|b| *b == b'\n').unwrap();
        String::from_utf8(raw[at..at + end].to_vec()).unwrap()
    };
    let tail = find(b"startxref\n") + 10;
    let start: usize = line(tail).parse().unwrap();
    assert_eq!(line(start), "xref", "startxref points at the table");
    let head = line(start + 5);
    let size: usize = head.split(' ').nth(1).unwrap().parse().unwrap();
    let mut at_line = start + 5 + head.len() + 1;
    assert_eq!(line(at_line), "0000000000 65535 f ");
    for n in 1..size {
        at_line += 20;
        let entry = line(at_line);
        assert_eq!(entry.len(), 19, "{entry:?}");
        assert!(entry.ends_with(" 00000 n "), "{entry:?}");
        let at: usize = entry[..10].parse().unwrap();
        assert!(
            raw[at..].starts_with(format!("{n} 0 obj\n").as_bytes()),
            "object {n} is not at byte {at}"
        );
    }
    let _ = fs::remove_dir_all(dir);
}

/// One sheet loads with the structure the plot has always had: one page, the paper's MediaBox,
/// no rotation, a Flate content stream of the sheet's strokes, and the Producer.
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_a_one_sheet_plot_is_structurally_what_it_was() {
    let dir = scratch("one");
    let dest = dir.join("plot.pdf");
    let bytes = support::drawing(1, 3);
    run(&bytes, &dest, 1).unwrap();
    qpdf::check_pdf(&dest).unwrap();
    let doc = lopdf::Document::load(&dest).unwrap();
    let pages = doc.get_pages();
    assert_eq!(pages.len(), 1);
    let page_id = *pages.values().next().unwrap();
    let page = doc.get_dictionary(page_id).unwrap();
    let media: Vec<f64> = page
        .get(b"MediaBox")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_float().map(f64::from).unwrap())
        .collect();
    let (w, h) = (420.0 / 25.4 * 72.0, 297.0 / 25.4 * 72.0);
    assert!(
        (media[2] - w).abs() < 0.01 && (media[3] - h).abs() < 0.01,
        "{media:?}"
    );
    assert_eq!(page.get(b"Rotate").unwrap().as_i64().unwrap(), 0);
    let content = String::from_utf8(doc.get_page_content(page_id).unwrap()).unwrap();
    assert_eq!(
        content.matches("\nS\n").count() + usize::from(content.starts_with("S\n")),
        3
    );
    assert_eq!(content.matches(" m\n").count(), 3);
    assert_eq!(content.matches(" l\n").count(), 3);
    assert!(
        content.contains("0.7087 w"),
        "0.25 mm is 0.7087 pt: {content}"
    );
    let producer = doc
        .trailer
        .get(b"Info")
        .and_then(|o| doc.dereference(o))
        .map(|(_, o)| o.clone())
        .unwrap();
    assert!(
        String::from_utf8_lossy(
            producer
                .as_dict()
                .unwrap()
                .get(b"Producer")
                .unwrap()
                .as_str()
                .unwrap()
        )
        .contains("libviprs")
    );
    let _ = fs::remove_dir_all(dir);
}
