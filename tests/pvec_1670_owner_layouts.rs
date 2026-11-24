//! Issue #1670, the owner's own files: every Print Layout of the master floor plan plotted to one PDF
//! with the plot style table and plotter configuration that sit beside it, then set against the PDF the
//! owner plotted from the same drawing.
//!
//! The files are private and are never committed. Point `LIBVIPRS_OWNER_PLOT_DIR` at the directory that
//! holds the drawing (`*.dwg`), the plotted reference (`*.pdf`), one `*.ctb` and one `*.pc3`. With the
//! variable unset, or the directory missing any of the four, the test says so and passes, so it is inert
//! on every machine but the owner's. `LIBVIPRS_OWNER_PLOT_OUT` names a directory to keep the PDF in
//! (default: a scratch directory that is removed). Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use acadrust::plot_style::PlotStyleTable;
use libviprs::cad::CadSource;
use libviprs::cad::plot_files::Pc3;
use libviprs::observe::{EngineEvent, EngineObserver};
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{DwgProvider, LayoutKind, SourceFact, SourceUnit, ViewProfile};
use libviprs::pvec_convert::PdfPlot;

/// Pages whose size is within this many points of the reference page's.
const SIZE_TOLERANCE_PT: f64 = 1.0;

fn find(dir: &Path, ext: &str) -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext)))
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// The plotted reference is the PDF that is not the one this test writes.
fn files() -> Option<[PathBuf; 4]> {
    let dir = PathBuf::from(std::env::var_os("LIBVIPRS_OWNER_PLOT_DIR")?);
    Some([
        find(&dir, "dwg")?,
        find(&dir, "pdf")?,
        find(&dir, "ctb")?,
        find(&dir, "pc3")?,
    ])
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_owner_layouts_match_the_plotted_pdf() {
    let Some([dwg, reference, ctb, pc3]) = files() else {
        eprintln!(
            "issue 1670 owner test: LIBVIPRS_OWNER_PLOT_DIR is not set or incomplete, skipped"
        );
        return;
    };
    let mut table = PlotStyleTable::from_bytes("owner.ctb", &fs::read(&ctb).unwrap()).unwrap();
    // `LIBVIPRS_OWNER_PLOT_ALLTHIN=1` puts every style at the list's first entry, to see what the
    // plotted PDF's widths are.
    if std::env::var_os("LIBVIPRS_OWNER_PLOT_ALLTHIN").is_some() {
        for e in &mut table.aci_entries {
            e.lineweight = 1;
        }
    }
    let pc3 = Pc3::parse(&fs::read(&pc3).unwrap()).unwrap();
    let media = pc3.media().unwrap();
    eprintln!("pc3 media {:?}", media);

    let provider = AcadrustProvider::open(CadSource::Path(&dwg)).unwrap();
    let paper: Vec<_> = provider
        .layouts()
        .unwrap()
        .into_iter()
        .filter(|l| l.kind == LayoutKind::Paper)
        .collect();
    for l in &paper {
        if let Ok(SourceFact::Present(p)) = provider.layout_plot(l.id) {
            eprintln!(
                "layout {:>3} {:<24} {:>8.2} x {:>8.2} mm rot {} scale {}/{} margins {:?} origin {:?} style {:?} pc3 {:?}",
                l.id.0,
                l.name,
                p.paper_width,
                p.paper_height,
                p.rotation,
                p.scale_numerator,
                p.scale_denominator,
                p.margins,
                p.origin,
                p.style_sheet,
                provider.layout_plot_config(l.id)
            );
        } else {
            eprintln!("layout {:>3} {:<24} no plot settings", l.id.0, l.name);
        }
    }

    // The owner plotted the first layouts in tab order and left the last ones out, so the plot
    // takes as many as the reference has pages. `LIBVIPRS_OWNER_PLOT_PAGES=2,4,21` plots only
    // those pages (1-based, as the reference numbers them) for a quicker look.
    let want = libviprs::pdf_info(&reference).unwrap();
    let pages: Vec<usize> = match std::env::var("LIBVIPRS_OWNER_PLOT_PAGES") {
        Ok(list) => list
            .split(',')
            .filter_map(|n| n.trim().parse().ok())
            .collect(),
        Err(_) => (1..=want.page_count).collect(),
    };
    let out_dir = std::env::var_os("LIBVIPRS_OWNER_PLOT_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("libviprs-1670-{}", std::process::id()))
        });
    fs::create_dir_all(&out_dir).unwrap();
    let started = std::time::Instant::now();
    let dest = out_dir.join("plotted.pdf");
    let observer = Arc::new(SheetClock(started));
    // `LIBVIPRS_OWNER_PLOT_DRAW=no_plot,frozen,off` draws the layers in those states, to see what the
    // plotted PDF shows that the print profile leaves out.
    let mut profile = ViewProfile::print_profile(&paper[0].name);
    if let Ok(list) = std::env::var("LIBVIPRS_OWNER_PLOT_DRAW") {
        use libviprs::pvec::LayerStateRule::Ignore;
        for state in list.split(',') {
            match state.trim() {
                "no_plot" => {
                    profile.layers.no_plot = Ignore;
                    profile.layers.no_plot_text = Ignore;
                }
                "no_plot_geometry" => profile.layers.no_plot = Ignore,
                "frozen" => profile.layers.frozen = Ignore,
                "off" => profile.layers.off = Ignore,
                _ => {}
            }
        }
    }
    let plotted = PdfPlot::new(CadSource::Path(&dwg), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(profile)
        .plot_style_table(table)
        .layouts(pages.iter().map(|n| paper[n - 1].name.clone()))
        .texts(true)
        .fills(true)
        // The plotted PDF has no width between 0 and 5 pt: every line under 0.2 mm is a hairline.
        .hairline_below(
            std::env::var("LIBVIPRS_OWNER_PLOT_HAIRLINE_MM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.2),
        )
        .observer(observer)
        .convert_with::<AcadrustProvider>()
        .unwrap();
    eprintln!(
        "all layouts: {:.1} s, notes {:?}",
        started.elapsed().as_secs_f64(),
        plotted.notes
    );

    let got = libviprs::pdf_info(&dest).unwrap();
    let mut size_mismatch = 0;
    for (g, n) in got.pages.iter().zip(&pages) {
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
    eprintln!(
        "pages got {} want {}, size mismatches {size_mismatch}",
        got.page_count,
        pages.len()
    );
    assert_eq!(got.page_count, pages.len(), "page count");
    assert_eq!(size_mismatch, 0, "page sizes");
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
