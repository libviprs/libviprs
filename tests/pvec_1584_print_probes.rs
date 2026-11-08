//! The print profile against the gold print-profile probes (#1584, part of #1557).
//!
//! pvec-gold-fixtures#101 wrote seven known-truth drawings for what a sheet does that the multi-viewport
//! probe can't show: a 30 degree twist, a layer frozen in one of two viewports, a non-plan view, a
//! hide_plot viewport beside one on a no-plot layer, a no-plot layer in model and paper space, and plot
//! settings in millimetres and in inches. Each is converted through `print_profile("Layout1")` and held
//! to its authored record. The twisted and non-plan viewports stay refused (508) until a plotted
//! reference print settles them, so those cells pin the refusal. The drawings are read from disk at
//! run time, so a pin without them fails here rather than at compile time. Needs
//! `--features pvec-convert`.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

use std::fs;
use std::path::{Path, PathBuf};

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::report::{ConversionReport, EntityRow, SheetSummary};
use libviprs::pvec::{DwgProvider, ReasonCode, SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PvecConversion, PvecConvertError};
use pvec_facade::unstable::Provenance;

fn drawing(id: &str) -> Vec<u8> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/fixtures/pvec-gold/fixtures/{id}/source.dwg"));
    fs::read(&p).unwrap_or_else(|e| panic!("{id} is not in the vendored corpus: {e}"))
}

fn scratch(id: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1584-{id}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Converts `id` as `Layout1`; the drawings are in metres.
fn sheet(id: &str) -> (Result<ConversionReport, PvecConvertError>, PathBuf) {
    let bytes = drawing(id);
    let dir = scratch(id);
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dir.join("sheet.pvec"))
        .unit(SourceUnit::Meters)
        .view_profile(ViewProfile::print_profile("Layout1"))
        .convert()
        .map(|o| o.report);
    (out, dir)
}

fn summary(report: &ConversionReport) -> &SheetSummary {
    report.sheet.as_ref().expect("a sheet section")
}

/// The viewports' statuses in the reader's order.
fn statuses(report: &ConversionReport) -> Vec<(&str, Option<&str>, u64)> {
    summary(report)
        .viewports
        .iter()
        .map(|v| (v.status.as_str(), v.reason.as_deref(), v.entities))
        .collect()
}

/// The rows drawn or dropped through `viewport` (`None`: the sheet's own entities), LINEs only.
fn lines(report: &ConversionReport, viewport: Option<u64>) -> Vec<&EntityRow> {
    report
        .entities
        .iter()
        .filter(|r| r.source_class == "LINE" && r.viewport == viewport)
        .collect()
}

/// A drawing whose only viewport the profile refuses: nothing to write, and the probe says why.
fn refused_alone(id: &str) {
    let (out, dir) = sheet(id);
    assert!(
        matches!(
            out,
            Err(PvecConvertError::Write(ref e)) if format!("{e:?}") == "NoGeometry"
        ),
        "{id}: {out:?}"
    );
    let bytes = drawing(id);
    let probed = PvecConversion::new(CadSource::Bytes(&bytes), Path::new("x.pvec"))
        .unit(SourceUnit::Meters)
        .view_profile(ViewProfile::print_profile("Layout1"))
        .probe()
        .expect("Layout1 is walked");
    let reasons: Vec<_> = probed.reasons().collect();
    assert_eq!(
        reasons,
        [(ReasonCode::ViewNotEnabled, "VIEWPORT", 1)],
        "{id}"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1584_the_twisted_viewport_is_read_as_authored_and_stays_refused() {
    let provider =
        AcadrustProvider::open(CadSource::Bytes(&drawing("viewport-twisted-001"))).unwrap();
    let vps = provider.viewports().unwrap();
    assert_eq!(vps.len(), 1);
    // pi/6 rounded once, as the record has it, in a plan view with its target at the origin: the
    // only thing that refuses it is the twist, whose sign no reference print has measured yet.
    assert_eq!(vps[0].twist.to_bits(), 0x3fe0c152382d7366);
    assert_eq!(vps[0].view_direction, [0.0, 0.0, 1.0]);
    assert_eq!(vps[0].view_target, [0.0, 0.0, 0.0]);
    refused_alone("viewport-twisted-001");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1584_the_non_plan_viewport_is_read_as_authored_and_stays_refused() {
    let provider =
        AcadrustProvider::open(CadSource::Bytes(&drawing("viewport-non-plan-001"))).unwrap();
    let vps = provider.viewports().unwrap();
    assert_eq!(vps.len(), 1);
    assert_eq!(vps[0].view_direction, [-1.0, -1.0, 1.0]);
    assert_eq!(vps[0].twist, 0.0);
    refused_alone("viewport-non-plan-001");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1584_a_layer_frozen_in_one_viewport_is_dropped_there_and_drawn_in_the_other() {
    let (out, dir) = sheet("viewport-frozen-layer-001");
    let report = out.expect("the sheet converts");
    assert_eq!(
        statuses(&report),
        [("drawn", None, 1), ("drawn", None, 2)],
        "DOORS is frozen in the first viewport only"
    );
    let first = summary(&report).viewports[0].handle;
    let second = summary(&report).viewports[1].handle;
    let through_first = lines(&report, Some(first));
    assert_eq!(through_first.len(), 2);
    assert_eq!(
        through_first
            .iter()
            .filter(|r| r.policy_field.as_deref() == Some("layers.frozen_in_viewport"))
            .count(),
        1
    );
    assert!(
        lines(&report, Some(second))
            .iter()
            .all(|r| r.semantic_type.is_some())
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1584_a_hide_plot_viewport_draws_nothing_and_one_on_a_no_plot_layer_hides_only_its_border()
{
    let (out, dir) = sheet("viewport-hide-plot-001");
    let report = out.expect("the sheet converts");
    // Since #1577 a viewport on a no-plot layer still shows model space; only its border is not
    // printed. hide_plot keeps the whole viewport off the print.
    assert_eq!(
        statuses(&report),
        [
            ("hide-plot", None, 0),
            ("drawn", None, 1),
            ("drawn", None, 1)
        ]
    );
    let borders: Vec<_> = summary(&report)
        .viewports
        .iter()
        .map(|v| v.border.as_deref())
        .collect();
    assert_eq!(borders, [None, Some("no-plot-layer"), None]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1584_a_no_plot_layer_is_left_out_in_model_and_in_paper_space() {
    let (out, dir) = sheet("no-plot-layer-001");
    let report = out.expect("the sheet converts");
    let vp = summary(&report).viewports[0].handle;
    // The NOPLOT line of each space is excluded once, before any viewport sees it, and the line on
    // layer 0 of each space is drawn: the paper one on the sheet, the model one through the viewport.
    let dropped: Vec<&EntityRow> = report
        .entities
        .iter()
        .filter(|r| r.source_class == "LINE" && r.policy_field.as_deref() == Some("layers.no_plot"))
        .collect();
    assert_eq!(dropped.len(), 2, "{:#?}", report.entities);
    assert!(
        dropped
            .iter()
            .all(|r| r.semantic_type.is_none() && r.viewport.is_none())
    );
    let drawn = |viewport: Option<u64>| {
        lines(&report, viewport)
            .iter()
            .filter(|r| r.semantic_type.is_some())
            .count()
    };
    assert_eq!(
        (drawn(None), drawn(Some(vp))),
        (1, 1),
        "{:#?}",
        report.entities
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1584_plot_settings_in_millimetres_frame_and_calibrate_the_sheet() {
    let (out, dir) = sheet("layout-plot-settings-mm-001");
    let report = out.expect("the sheet converts");
    let s = summary(&report);
    assert_eq!(
        (s.frame.as_str(), s.unit.as_str()),
        ("paper", "Millimeters")
    );
    assert_eq!(s.unit_source, "plot-paper-units");
    // Where the paper sits from the margins and the origin is still unmeasured.
    assert_eq!(s.position.as_deref(), Some("unmeasured"));
    let plot = s.plot.as_ref().expect("the plot settings");
    assert_eq!(
        (
            plot.paper_width.as_str(),
            plot.paper_height.as_str(),
            plot.rotation
        ),
        ("420", "297", 1)
    );
    assert_eq!(plot.style_sheet, "monochrome.ctb");
    assert!(!plot.print_lineweights);
    assert!(s.widths_dropped, "lineweights are not printed");
    assert!(
        s.notes.iter().any(|n| n == "rotation-not-applied"),
        "{:?}",
        s.notes
    );
    assert_eq!(
        (s.scale.as_deref(), s.calibration.as_str()),
        (Some("1:50"), "plot-scale")
    );
    let claim = open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .calibration()
        .unwrap()
        .and_then(|c| c.scale)
        .expect("the plot scale");
    assert_eq!(
        claim.provenance,
        Provenance::PlanScale {
            numerator: 1,
            denominator: 50,
            paper_metres_per_unit: 0.001,
        }
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1584_plot_settings_in_inches_frame_and_calibrate_the_sheet() {
    let (out, dir) = sheet("layout-plot-settings-inch-001");
    let report = out.expect("the sheet converts");
    let s = summary(&report);
    assert_eq!((s.frame.as_str(), s.unit.as_str()), ("paper", "Inches"));
    let plot = s.plot.as_ref().expect("the plot settings");
    assert_eq!(plot.style_sheet, "acad.ctb");
    assert!(plot.print_lineweights);
    assert!(!s.widths_dropped);
    assert!(
        !s.notes.iter().any(|n| n == "rotation-not-applied"),
        "{:?}",
        s.notes
    );
    assert_eq!(s.scale.as_deref(), Some("1:96"));
    let claim = open_published(&dir.join("sheet.pvec"))
        .unwrap()
        .calibration()
        .unwrap()
        .and_then(|c| c.scale)
        .expect("the plot scale");
    assert_eq!(
        claim.provenance,
        Provenance::PlanScale {
            numerator: 1,
            denominator: 96,
            paper_metres_per_unit: 0.0254,
        }
    );
    let _ = fs::remove_dir_all(dir);
}
