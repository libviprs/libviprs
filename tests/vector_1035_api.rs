//! The public Rust surface of the vector path (issue #1035): list views, convert
//! one, report without writing, inspect and validate an archive.
//!
//! Every drawing here is built out of a loop, so none of this needs the native
//! provider. The one test that does says so and skips without it, unless
//! `LIBVIPRS_REQUIRE_ACADSHARP_NATIVE=1` makes that a failure.

use std::path::{Path, PathBuf};
use std::time::Duration;

use libviprs::CancelToken;
use libviprs::cad::{
    CadDrawing, CadError, CadView, DecodeReport, Line, PrimitiveKind, PrimitiveSink,
};
use libviprs::vector::convert::{
    ConvertError, ConvertOptions, ViewSelector, check, convert, inspect, list_views, report,
    select_view,
};
use libviprs::vector::tiler::{TilerConfig, TilerError, build_archive_from_view};

/// A drawing with a model space view, a layout, and `rungs` lines in the model.
struct Ladder {
    rungs: u32,
    /// How long `decode` waits before emitting, to give a deadline something to pass.
    stall: Duration,
}

impl Ladder {
    fn new(rungs: u32) -> Self {
        Self {
            rungs,
            stall: Duration::ZERO,
        }
    }
}

impl CadDrawing for Ladder {
    fn views(&self) -> Result<Vec<CadView>, CadError> {
        Ok(vec![
            CadView::new(0, 0, [0.0, 0.0, 100.0, 100.0], u64::from(self.rungs), "Model"),
            CadView::new(1, 1, [0.0, 0.0, 10.0, 10.0], 0, "Sheet 1"),
        ])
    }

    fn decode(
        &self,
        view: u32,
        sink: &mut dyn PrimitiveSink,
        report: &mut DecodeReport,
    ) -> Result<(), CadError> {
        if view > 1 {
            return Err(CadError::NoSuchView { index: view, views: 2 });
        }
        std::thread::sleep(self.stall);
        for rung in 0..self.rungs {
            let y = f64::from(rung) * 100.0 / f64::from(self.rungs.max(1));
            sink.primitive(Line::new([0.0, y, 0.0], [100.0, y + 1.0, 0.0])?.into())?;
            report.record(PrimitiveKind::Line);
        }
        report.mark_complete();
        sink.finish()
    }
}

/// A drawing whose only view is a layout.
struct PaperOnly;

impl CadDrawing for PaperOnly {
    fn views(&self) -> Result<Vec<CadView>, CadError> {
        Ok(vec![CadView::new(0, 1, [0.0, 0.0, 10.0, 10.0], 0, "Sheet 1")])
    }

    fn decode(&self, _: u32, _: &mut dyn PrimitiveSink, _: &mut DecodeReport) -> Result<(), CadError> {
        Ok(())
    }
}

/// A drawing whose decoder fails part way through.
struct Broken;

impl CadDrawing for Broken {
    fn views(&self) -> Result<Vec<CadView>, CadError> {
        Ok(vec![CadView::new(0, 0, [0.0, 0.0, 10.0, 10.0], 1, "Model")])
    }

    fn decode(
        &self,
        _: u32,
        sink: &mut dyn PrimitiveSink,
        report: &mut DecodeReport,
    ) -> Result<(), CadError> {
        sink.primitive(Line::new([0.0; 3], [10.0, 10.0, 0.0])?.into())?;
        report.record(PrimitiveKind::Line);
        Err(CadError::provider("test", "the reader fell over"))
    }
}

fn opts(zoom: u8) -> ConvertOptions {
    ConvertOptions::new().with_tiler(TilerConfig::default().with_max_zoom(zoom))
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

fn converted(dir: &Path, rungs: u32) -> PathBuf {
    let dest = dir.join("out.pmtiles");
    convert(&Ladder::new(rungs), &ViewSelector::Model, &dest, &opts(4)).unwrap();
    dest
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

#[test]
fn views_are_listed_in_the_drawings_order() {
    let views = list_views(&Ladder::new(3)).unwrap();
    assert_eq!(views.len(), 2);
    assert_eq!(views[0].name(), "Model");
    assert_eq!(views[1].name(), "Sheet 1");
}

#[test]
fn a_selector_resolves_by_default_index_and_name() {
    let d = Ladder::new(3);
    assert_eq!(select_view(&d, &ViewSelector::Model).unwrap().index(), 0);
    assert_eq!(select_view(&d, &ViewSelector::Index(1)).unwrap().name(), "Sheet 1");
    assert_eq!(
        select_view(&d, &ViewSelector::Name("Sheet 1".into())).unwrap().index(),
        1
    );
}

#[test]
fn a_selector_that_matches_nothing_says_what_the_drawing_has() {
    let d = Ladder::new(3);
    let by_name = select_view(&d, &ViewSelector::Name("Elevation".into())).unwrap_err();
    assert!(matches!(by_name, ConvertError::NoSuchViewName { .. }), "{by_name:?}");
    let text = by_name.to_string();
    assert!(text.contains("Elevation") && text.contains("Model") && text.contains("Sheet 1"), "{text}");
    let by_index = select_view(&d, &ViewSelector::Index(9)).unwrap_err();
    assert!(matches!(by_index, ConvertError::Cad(CadError::NoSuchView { index: 9, .. })), "{by_index:?}");
    assert!(matches!(
        select_view(&PaperOnly, &ViewSelector::Model).unwrap_err(),
        ConvertError::NoModelSpace
    ));
}

#[test]
fn a_selector_reads_the_way_a_command_line_spells_it() {
    assert_eq!(ViewSelector::parse(""), ViewSelector::Model);
    assert_eq!(ViewSelector::parse("Model"), ViewSelector::Model);
    assert_eq!(ViewSelector::parse("2"), ViewSelector::Index(2));
    assert_eq!(ViewSelector::parse("Sheet 1"), ViewSelector::Name("Sheet 1".into()));
    assert_eq!("model".parse::<ViewSelector>().unwrap(), ViewSelector::Model);
    assert_eq!(ViewSelector::Index(4).to_string(), "4");
}

#[test]
fn a_layout_is_refused_with_the_tilers_reason() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("sheet.pmtiles");
    let err = convert(&Ladder::new(3), &ViewSelector::Index(1), &dest, &opts(3)).unwrap_err();
    assert!(
        matches!(err, ConvertError::Tiler(TilerError::ViewNotEnabled { index: 1, .. })),
        "{err:?}"
    );
    assert!(!dest.exists());
}

// ---------------------------------------------------------------------------
// Convert, and the engine it shares
// ---------------------------------------------------------------------------

#[test]
fn a_conversion_is_a_valid_archive_that_inspect_describes() {
    let dir = tempfile::tempdir().unwrap();
    let dest = converted(dir.path(), 20);

    let verdict = check(&dest).unwrap();
    assert!(verdict.is_valid(), "{:?}", verdict.findings);
    assert!(verdict.tile_entries > 0);

    let info = inspect(&dest).unwrap();
    assert_eq!(info.name.as_deref(), Some("Model"), "the view's name is the default");
    assert_eq!(info.tile_type, "mvt");
    assert_eq!(info.tile_compression, "gzip");
    assert_eq!(info.min_zoom, 0);
    assert_eq!(info.max_zoom, 4);
    assert_eq!(info.tile_entries, verdict.tile_entries);
    assert!(info.file_bytes.is_some_and(|b| b > 0));
    assert!(info.layers.iter().any(|l| l.id == "lines"));
    assert!(info.layers.iter().any(|l| l.id == "text"));
    let [w, s, e, n] = info.bounds;
    assert!(w < e && s < n, "{:?}", info.bounds);
    let tiler = info.tiler.as_ref().expect("the tiler's own metadata");
    assert_eq!(tiler["max_zoom"], 4);
}

#[test]
fn the_name_option_overrides_the_views_name() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("named.pmtiles");
    convert(&Ladder::new(3), &ViewSelector::Model, &dest, &opts(2).with_name("Level 2 plan")).unwrap();
    assert_eq!(inspect(&dest).unwrap().name.as_deref(), Some("Level 2 plan"));
}

#[test]
fn the_conversion_says_what_it_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("a.pmtiles");
    let done = convert(&Ladder::new(10), &ViewSelector::Model, &dest, &opts(3)).unwrap();
    assert_eq!(done.destination.as_deref(), Some(dest.as_path()));
    assert_eq!(done.bytes, Some(std::fs::metadata(&dest).unwrap().len()));
    assert_eq!(done.view.name(), "Model");
    assert_eq!(done.tiler.primitives, 10);
    assert!(done.decode.is_complete());
    assert_eq!(done.decode.counts().get(PrimitiveKind::Line), 10);
}

#[test]
fn the_library_call_and_the_function_the_command_line_drives_are_one_engine() {
    // The CLI calls `convert`. This pins `convert` to the tiler's own entry
    // point byte for byte, so the two cannot drift into separate behaviours.
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.pmtiles");
    let b = dir.path().join("b.pmtiles");
    let d = Ladder::new(40);
    convert(&d, &ViewSelector::Model, &a, &opts(5)).unwrap();
    build_archive_from_view(&d, 0, &TilerConfig::default().with_max_zoom(5), &b, "Model").unwrap();
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
}

// ---------------------------------------------------------------------------
// Atomic output: temp, fsync, rename, and nothing left behind
// ---------------------------------------------------------------------------

#[test]
fn a_finished_conversion_leaves_only_the_archive() {
    let dir = tempfile::tempdir().unwrap();
    converted(dir.path(), 10);
    assert_eq!(names(dir.path()), vec!["out.pmtiles"]);
}

#[test]
fn a_cancelled_conversion_leaves_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("never.pmtiles");
    let token = CancelToken::new();
    token.cancel();
    let options = ConvertOptions::new().with_tiler(TilerConfig::default().with_max_zoom(6).with_cancel(token));
    let err = convert(&Ladder::new(50), &ViewSelector::Model, &dest, &options).unwrap_err();
    assert!(err.is_cancelled() && !err.is_deadline(), "{err:?}");
    assert!(!dest.exists());
    assert_eq!(names(dir.path()), Vec::<String>::new(), "no staging files either");
}

#[test]
fn a_conversion_cancelled_while_the_decoder_runs_leaves_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("never.pmtiles");
    let token = CancelToken::new();
    let options = ConvertOptions::new()
        .with_tiler(TilerConfig::default().with_max_zoom(6).with_cancel(token.clone()));
    let canceller = {
        let token = token.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            token.cancel();
        })
    };
    let slow = Ladder {
        rungs: 200,
        stall: Duration::from_millis(300),
    };
    let err = convert(&slow, &ViewSelector::Model, &dest, &options).unwrap_err();
    canceller.join().unwrap();
    assert!(err.is_cancelled(), "{err:?}");
    assert_eq!(names(dir.path()), Vec::<String>::new());
}

#[test]
fn a_run_past_its_deadline_leaves_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("never.pmtiles");
    let options = ConvertOptions::new()
        .with_tiler(TilerConfig::default().with_max_zoom(6).with_deadline(Duration::from_millis(1)));
    let slow = Ladder {
        rungs: 100,
        stall: Duration::from_millis(40),
    };
    let err = convert(&slow, &ViewSelector::Model, &dest, &options).unwrap_err();
    assert!(err.is_deadline() && !err.is_cancelled(), "{err:?}");
    assert_eq!(names(dir.path()), Vec::<String>::new());
}

#[test]
fn a_failed_decode_leaves_nothing_behind_and_keeps_the_partial_report() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("never.pmtiles");
    let err = convert(&Broken, &ViewSelector::Model, &dest, &opts(3)).unwrap_err();
    let ConvertError::Tiler(TilerError::DecodeFailed { report, .. }) = err else {
        panic!("expected a decode failure");
    };
    assert_eq!(report.counts().get(PrimitiveKind::Line), 1);
    assert_eq!(names(dir.path()), Vec::<String>::new());
}

#[test]
fn a_failed_conversion_leaves_an_existing_destination_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let dest = converted(dir.path(), 10);
    let before = std::fs::read(&dest).unwrap();
    let token = CancelToken::new();
    token.cancel();
    let options = ConvertOptions::new().with_tiler(TilerConfig::default().with_cancel(token));
    assert!(convert(&Ladder::new(10), &ViewSelector::Model, &dest, &options).is_err());
    assert_eq!(std::fs::read(&dest).unwrap(), before, "the old archive survives a failed rebuild");
    assert_eq!(names(dir.path()), vec!["out.pmtiles"]);
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[test]
fn a_report_writes_nothing_and_agrees_with_a_conversion() {
    let dir = tempfile::tempdir().unwrap();
    let d = Ladder::new(30);
    let dry = report(&d, &ViewSelector::Model, &opts(4)).unwrap();
    assert!(dry.destination.is_none() && dry.bytes.is_none());
    assert_eq!(names(dir.path()), Vec::<String>::new());

    let dest = dir.path().join("a.pmtiles");
    let real = convert(&d, &ViewSelector::Model, &dest, &opts(4)).unwrap();
    assert_eq!(dry.tiler.tiles, real.tiler.tiles);
    assert_eq!(dry.tiler.points, real.tiler.points);
    assert_eq!(dry.tiler.largest_tile_bytes, real.tiler.largest_tile_bytes);
    assert_eq!(dry.tiler.tiles_per_zoom, real.tiler.tiles_per_zoom);
    assert_eq!(dry.decode.counts().get(PrimitiveKind::Line), 30);
}

#[test]
fn a_report_honours_the_cancel_token() {
    let token = CancelToken::new();
    token.cancel();
    let options = ConvertOptions::new().with_tiler(TilerConfig::default().with_cancel(token));
    let err = report(&Ladder::new(10), &ViewSelector::Model, &options).unwrap_err();
    assert!(err.is_cancelled(), "{err:?}");
}

#[test]
fn a_conversion_serialises_for_a_machine_reader() {
    let d = Ladder::new(12);
    let done = report(&d, &ViewSelector::Model, &opts(3)).unwrap();
    let j = done.to_json();
    assert_eq!(j["view"]["name"], "Model");
    assert_eq!(j["view"]["kind"], "model");
    assert_eq!(j["view"]["enabled"], true);
    assert_eq!(j["decode"]["complete"], true);
    assert_eq!(j["decode"]["primitives"]["line"], 12);
    assert!(j["tiles"]["written"].as_u64().unwrap() > 0);
    assert!(j["fidelity"]["undrawable"].is_number());
    assert!(j["diagnostics"].is_array());
    assert!(j["destination"].is_null());
}

// ---------------------------------------------------------------------------
// Validate and inspect on files that are not what they should be
// ---------------------------------------------------------------------------

#[test]
fn a_file_that_is_not_an_archive_is_a_finding_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("junk.pmtiles");
    std::fs::write(&p, vec![7u8; 4096]).unwrap();
    let verdict = check(&p).unwrap();
    assert!(!verdict.is_valid());
    assert!(!verdict.findings.is_empty());
    assert_eq!(verdict.to_json()["valid"], false);
    assert!(matches!(inspect(&p).unwrap_err(), ConvertError::PmTiles(_) | ConvertError::Archive(_)));
}

#[test]
fn a_truncated_archive_is_invalid() {
    let dir = tempfile::tempdir().unwrap();
    let dest = converted(dir.path(), 30);
    let bytes = std::fs::read(&dest).unwrap();
    let cut = dir.path().join("cut.pmtiles");
    std::fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
    assert!(!check(&cut).unwrap().is_valid());
}

#[test]
fn a_missing_file_is_an_error_for_every_reader() {
    let p = Path::new("/nonexistent/viprs/none.pmtiles");
    assert!(check(p).is_err());
    assert!(inspect(p).is_err());
}

#[test]
fn a_raster_archive_is_not_a_vector_archive() {
    use libviprs::pmtiles::TileType;
    use libviprs::pmtiles::writer::{Writer, WriterOptions};
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("raster.pmtiles");
    let mut w = Writer::create(&p, WriterOptions::default().with_tile_type(TileType::Png)).unwrap();
    w.add_tile(0, 0, 0, b"\x89PNG not really", [1u8; 32]).unwrap();
    w.finish().unwrap();
    assert!(matches!(inspect(&p).unwrap_err(), ConvertError::Archive(_)));
    assert!(!check(&p).unwrap().is_valid(), "MVT is what makes it a vector archive");
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

#[cfg(not(feature = "dwg-acadsharp"))]
#[test]
fn without_a_provider_the_default_decoder_says_how_to_get_one() {
    let err = libviprs::vector::convert::default_decoder(false).err().expect("no provider built");
    assert!(matches!(err, ConvertError::ProviderNotBuilt));
    assert!(err.to_string().contains("dwg-acadsharp"), "{err}");
}

#[cfg(feature = "dwg-acadsharp")]
#[test]
fn a_real_dwg_converts_through_the_default_decoder() {
    use libviprs::cad::CadSource;
    let decoder = match libviprs::vector::convert::default_decoder(true) {
        Ok(d) => d,
        Err(e) => {
            assert!(
                std::env::var("LIBVIPRS_REQUIRE_ACADSHARP_NATIVE").as_deref() != Ok("1"),
                "native lane required: {e}"
            );
            eprintln!("skipped: {e}");
            return;
        }
    };
    let dwg = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/cad/sample_AC1018.dwg")).unwrap();
    let drawing = decoder.open(CadSource::Bytes(&dwg)).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("sample.pmtiles");
    let done = convert(drawing.as_ref(), &ViewSelector::Model, &dest, &opts(6)).unwrap();
    assert!(done.tiler.tiles > 5 && done.decode.is_complete());
    assert!(check(&dest).unwrap().is_valid());
    assert_eq!(names(dir.path()), vec!["sample.pmtiles"]);
}
