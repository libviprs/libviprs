//! Issue #1670, the owner's own files: every Print Layout of the master floor plan plotted to one PDF
//! with the plot style table that sits beside it, then set against the PDF the owner plotted from the
//! same drawing.
//!
//! The files are private and are never committed. Point `LIBVIPRS_OWNER_PLOT_DIR` at the directory that
//! holds the drawing (`*.dwg`), the plotted reference (`*.pdf`), one `*.ctb` and one `*.pc3`. With the
//! variable unset, or the directory missing any of the four, the tests say so and pass, so they are inert
//! on every machine but the owner's. `LIBVIPRS_OWNER_PLOT_OUT` names a directory to keep the PDFs in
//! (default: a scratch directory that is removed). Needs `--features pvec-convert`.
//!
//! Two plots are made, and the ink recall of each against the reference is measured by
//! `tools/plot_recall.py` at 150 dpi and 3 px (poppler's `pdftoppm` and python3 are needed; the recall
//! assertions say so and pass when they are missing):
//!
//! - default rules plus the hairline option: the print rule leaves no-plot layers out; median recall at
//!   least 93 percent (measured 93.8);
//! - the owner knob: the no-plot layers that the plotted PDF names (its optional content groups) are
//!   drawn too, because they were plottable when it was made; median recall at least 95 percent
//!   (measured 96.2). That is not what the library does by default.
//!
//! `LIBVIPRS_OWNER_PLOT_PAGES=2,4,21` plots only those pages (1-based, as the reference numbers them)
//! for a quicker look; the medians are then over those pages.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use acadrust::plot_style::PlotStyleTable;
use libviprs::cad::CadSource;
use libviprs::observe::{EngineEvent, EngineObserver};
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{
    DwgProvider, LayerOverride, LayerStateRule, LayoutKind, SourceUnit, ViewProfile,
};
use libviprs::pvec_convert::PdfPlot;

/// Pages whose size is within this many points of the reference page's.
const SIZE_TOLERANCE_PT: f64 = 1.0;

struct Owner {
    dwg: PathBuf,
    reference: PathBuf,
    ctb: PathBuf,
    out: PathBuf,
}

fn find(dir: &Path, ext: &str) -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext)))
        .collect();
    hits.sort();
    hits.into_iter().next()
}

fn owner(tag: &str) -> Option<Owner> {
    let dir = PathBuf::from(std::env::var_os("LIBVIPRS_OWNER_PLOT_DIR")?);
    find(&dir, "pc3")?;
    let out = std::env::var_os("LIBVIPRS_OWNER_PLOT_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("libviprs-1670-{}-{tag}", std::process::id()))
        })
        .join(tag);
    fs::create_dir_all(&out).unwrap();
    Some(Owner {
        dwg: find(&dir, "dwg")?,
        reference: find(&dir, "pdf")?,
        ctb: find(&dir, "ctb")?,
        out,
    })
}

/// The names of the optional content groups (the layers) a PDF lists.
fn pdf_layers(bytes: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(bytes);
    let mut names = Vec::new();
    for (at, _) in text
        .match_indices("/Type/OCG")
        .chain(text.match_indices("/Type /OCG"))
    {
        let from = text.floor_char_boundary(at.saturating_sub(200));
        let to = text.floor_char_boundary((at + 200).min(text.len()));
        let seg = &text[from..to];
        if let Some(i) = seg.find("/Name(").or_else(|| seg.find("/Name (")) {
            let rest = &seg[seg[i..].find('(').unwrap() + i + 1..];
            if let Some(j) = rest.find(')') {
                names.push(rest[..j].to_string());
            }
        }
    }
    names
}

/// Plots the pages with the print profile, and with `named_layers` also the no-plot layers the plotted
/// PDF names. Returns the PDF and the reference's page numbers it holds, in order.
fn plot(o: &Owner, named_layers: bool) -> (PathBuf, Vec<usize>) {
    let mut table = PlotStyleTable::from_bytes("owner.ctb", &fs::read(&o.ctb).unwrap()).unwrap();
    // `LIBVIPRS_OWNER_PLOT_ALLTHIN=1` puts every style at the list's first entry, to see what the
    // plotted PDF's widths are.
    if std::env::var_os("LIBVIPRS_OWNER_PLOT_ALLTHIN").is_some() {
        for e in &mut table.aci_entries {
            e.lineweight = 1;
        }
    }
    let provider = AcadrustProvider::open(CadSource::Path(&o.dwg)).unwrap();
    let paper: Vec<_> = provider
        .layouts()
        .unwrap()
        .into_iter()
        .filter(|l| l.kind == LayoutKind::Paper)
        .collect();
    // The owner plotted the first layouts in tab order and left the last ones out, so the plot takes as
    // many as the reference has pages.
    let want = libviprs::pdf_info(&o.reference).unwrap();
    let mut pages: Vec<usize> = match std::env::var("LIBVIPRS_OWNER_PLOT_PAGES") {
        Ok(list) => list
            .split(',')
            .filter_map(|n| n.trim().parse().ok())
            .collect(),
        Err(_) => (1..=want.page_count).collect(),
    };
    pages.sort_unstable();
    pages.dedup();
    let started = std::time::Instant::now();
    let dest = o.out.join("plotted.pdf");
    let mut profile = ViewProfile::print_profile(&paper[0].name);
    let no_plot: Vec<String> = provider
        .layers()
        .unwrap()
        .into_iter()
        .filter(|l| !l.plottable)
        .map(|l| l.name)
        .collect();
    if named_layers {
        // A layer the drawing flags no-plot but the plotted PDF names was plottable when the PDF was
        // made: it is drawn, and the other no-plot layers are not.
        let plotted = pdf_layers(&fs::read(&o.reference).unwrap());
        let named = no_plot.iter().filter(|l| plotted.contains(l)).count();
        eprintln!(
            "no-plot layers {}, named by the plotted PDF {named}",
            no_plot.len()
        );
        if named > 0 {
            profile.layers.no_plot = LayerStateRule::Ignore;
            profile.layers.no_plot_text = LayerStateRule::Ignore;
            for l in no_plot.iter().filter(|l| !plotted.contains(l)) {
                profile
                    .layers
                    .overrides
                    .push((l.clone(), LayerOverride::Hidden));
            }
        }
    }
    // `LIBVIPRS_OWNER_PLOT_DRAW=no_plot,no_plot_geometry,frozen,off` draws the layers in those states, and
    // `LIBVIPRS_OWNER_PLOT_LAYER_INDEX=3` only the fourth no-plot layer, to see what the plotted PDF shows
    // that the print profile leaves out.
    if let Some(i) = std::env::var("LIBVIPRS_OWNER_PLOT_LAYER_INDEX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        profile.layers.no_plot = LayerStateRule::Ignore;
        profile.layers.no_plot_text = LayerStateRule::Ignore;
        for (_, name) in no_plot.iter().enumerate().filter(|(n, _)| *n != i) {
            profile
                .layers
                .overrides
                .push((name.clone(), LayerOverride::Hidden));
        }
    }
    if let Ok(list) = std::env::var("LIBVIPRS_OWNER_PLOT_DRAW") {
        for state in list.split(',') {
            match state.trim() {
                "no_plot" => {
                    profile.layers.no_plot = LayerStateRule::Ignore;
                    profile.layers.no_plot_text = LayerStateRule::Ignore;
                }
                "no_plot_geometry" => profile.layers.no_plot = LayerStateRule::Ignore,
                "frozen" => profile.layers.frozen = LayerStateRule::Ignore,
                "off" => profile.layers.off = LayerStateRule::Ignore,
                _ => {}
            }
        }
    }
    let plotted = PdfPlot::new(CadSource::Path(&o.dwg), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(profile)
        .plot_style_table(table)
        .layouts(pages.iter().map(|n| paper[n - 1].name.clone()))
        .texts(true)
        .fills(true)
        // The plotted PDF has only two widths, 0 and 5 pt, though its table gives ACI 7 0.10 mm: every
        // line under 0.2 mm is a hairline.
        .hairline_below(
            std::env::var("LIBVIPRS_OWNER_PLOT_HAIRLINE_MM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.2),
        )
        .observer(Arc::new(SheetClock(started)))
        .convert_with::<AcadrustProvider>()
        .unwrap();
    eprintln!(
        "all layouts: {:.1} s, notes {:?}",
        started.elapsed().as_secs_f64(),
        plotted.notes
    );
    (dest, pages)
}

/// The median ink recall of `dest` against the reference at 150 dpi and 3 px, from
/// `tools/plot_recall.py`; `None` (with a line on stderr) when poppler or python3 is missing.
fn median_recall(o: &Owner, dest: &Path, pages: &[usize]) -> Option<f64> {
    let list: Vec<String> = pages.iter().map(usize::to_string).collect();
    let run = Command::new("python3")
        .arg("-I")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/plot_recall.py"))
        .arg(&o.reference)
        .arg(dest)
        .arg(list.join(","))
        .args(["150", "3"])
        .output();
    let out = match run {
        Ok(out) if out.status.success() => out,
        Ok(out) => {
            eprintln!(
                "the recall tool did not run ({}), recall not asserted",
                out.status
            );
            return None;
        }
        Err(e) => {
            eprintln!("python3 is not available ({e}), recall not asserted");
            return None;
        }
    };
    let text = String::from_utf8_lossy(&out.stdout);
    eprintln!("recall {text}");
    let at = text.find("\"median_recall\":")? + "\"median_recall\":".len();
    text[at..]
        .trim_start()
        .split([',', '}'])
        .next()?
        .trim()
        .parse()
        .ok()
}

fn check_sizes(o: &Owner, dest: &Path, pages: &[usize]) {
    let want = libviprs::pdf_info(&o.reference).unwrap();
    let got = libviprs::pdf_info(dest).unwrap();
    let mut size_mismatch = 0;
    for (g, n) in got.pages.iter().zip(pages) {
        let w = &want.pages[n - 1];
        let ok = (g.width_pts - w.width_pts).abs() <= SIZE_TOLERANCE_PT
            && (g.height_pts - w.height_pts).abs() <= SIZE_TOLERANCE_PT;
        size_mismatch += usize::from(!ok);
        eprintln!(
            "page {:>3} got {:>8.2} x {:>8.2} want {:>8.2} x {:>8.2} {}",
            n,
            g.width_pts,
            g.height_pts,
            w.width_pts,
            w.height_pts,
            if ok { "ok" } else { "SIZE" }
        );
    }
    assert_eq!(got.page_count, pages.len(), "page count");
    assert_eq!(size_mismatch, 0, "page sizes");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_owner_layouts_match_the_plotted_pdf_with_the_owner_knob() {
    let Some(o) = owner("knob") else {
        eprintln!(
            "issue 1670 owner test: LIBVIPRS_OWNER_PLOT_DIR is not set or incomplete, skipped"
        );
        return;
    };
    let (dest, pages) = plot(&o, true);
    check_sizes(&o, &dest, &pages);
    if let Some(median) = median_recall(&o, &dest, &pages) {
        assert!(
            median >= 0.95,
            "median ink recall {median} is under 95 percent"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_owner_layouts_match_the_plotted_pdf_with_the_default_rules() {
    let Some(o) = owner("default") else {
        eprintln!(
            "issue 1670 owner test: LIBVIPRS_OWNER_PLOT_DIR is not set or incomplete, skipped"
        );
        return;
    };
    let (dest, pages) = plot(&o, false);
    check_sizes(&o, &dest, &pages);
    if let Some(median) = median_recall(&o, &dest, &pages) {
        assert!(
            median >= 0.93,
            "median ink recall {median} is under 93 percent"
        );
    }
}

/// Prints when each sheet starts and finishes, in seconds since the plot began.
struct SheetClock(std::time::Instant);

impl EngineObserver for SheetClock {
    fn on_event(&self, event: EngineEvent) {
        let at = self.0.elapsed().as_secs_f64();
        match event {
            EngineEvent::ConversionSheetStarted { index, layout, .. } => {
                eprintln!("sheet {index:>3} start {at:>8.1} s {layout}");
            }
            EngineEvent::ConversionSheetFinished { index, status, .. } => {
                eprintln!("sheet {index:>3} done  {at:>8.1} s {status}");
            }
            _ => {}
        }
    }
}
