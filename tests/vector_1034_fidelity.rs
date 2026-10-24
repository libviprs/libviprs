//! Text, blocks, dimensions and hatches, and every gap reported (issue #1034).
//!
//! Two kinds of test. The ones with a fake drawing need no native library and
//! always run: they pin the policies (Model Space only, a bound on what a
//! decode may hold, text read the way a person reads it). The ones that open
//! DWGs from `tests/fixtures/cad/g13/` need the NativeAOT archive and skip
//! without it, unless `LIBVIPRS_REQUIRE_ACADSHARP_NATIVE=1`.

use std::collections::BTreeSet;
use std::path::Path;

use libviprs::cad::{
    CadDrawing, CadError, CadView, DecodeReport, Diagnostic, DiagnosticCode, Line, Primitive,
    PrimitiveSink, Text,
};
use libviprs::vector::mvt::Value;
use libviprs::vector::style::{SERVED_GLYPH_RANGES, glyph_available};
use libviprs::vector::tiler::{
    CollectTiles, TilerConfig, TilerError, build_archive_from_view, tile_primitives,
};

fn text(at: [f64; 2], h: f64, rot: f64, s: &str) -> Primitive {
    Text::new([at[0], at[1], 0.0], h, rot, s).unwrap().into()
}

fn frame() -> Primitive {
    Line::new([0.0, 0.0, 0.0], [1000.0, 600.0, 0.0])
        .unwrap()
        .into()
}

fn tile_texts(tiles: &CollectTiles) -> Vec<(u8, String, f64)> {
    let mut out = Vec::new();
    for (&(z, _, _), t) in &tiles.tiles {
        for l in t.layers.iter().filter(|l| l.name == "text") {
            for f in &l.features {
                let get = |k: &str| {
                    f.properties
                        .iter()
                        .find(|(n, _)| n == k)
                        .map(|(_, v)| v.clone())
                };
                if let (Some(Value::String(s)), Some(Value::Double(h))) =
                    (get("text"), get("height"))
                {
                    out.push((z, s, h));
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The glyph ranges the style serves are the files that are there
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn the_declared_glyph_ranges_are_exactly_the_font_files_shipped() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("web/vector-viewer/public/fonts/Noto Sans Regular");
    let mut ranges: Vec<(u32, u32)> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            let stem = name.strip_suffix(".pbf")?.to_owned();
            let (a, b) = stem.split_once('-')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .collect();
    ranges.sort_unstable();
    // Merge touching files into spans.
    let mut spans: Vec<(u32, u32)> = Vec::new();
    for (lo, hi) in ranges {
        match spans.last_mut() {
            Some(last) if last.1 + 1 == lo => last.1 = hi,
            _ => spans.push((lo, hi)),
        }
    }
    assert_eq!(spans, SERVED_GLYPH_RANGES.to_vec());
    assert!(glyph_available('A') && glyph_available('\u{2205}') && glyph_available('\u{b0}'));
    assert!(!glyph_available('\u{4e2d}') && !glyph_available('\u{3000}'));
}

// ---------------------------------------------------------------------------
// Text, read as a person reads it
// ---------------------------------------------------------------------------

#[test]
fn mtext_formatting_is_read_and_its_losses_are_reported() {
    let prims = vec![
        frame(),
        text([100.0, 500.0], 60.0, 0.0, r#"4'-0{\H1x;\S1#4;}""#),
        text(
            [100.0, 300.0],
            60.0,
            0.0,
            r"Line one\PLine {\fArial|b1;two}",
        ),
        text([100.0, 200.0], 60.0, 0.0, r"\A1;7.0711"),
        text([100.0, 100.0], 60.0, 0.0, r"\A1;"),
        text([500.0, 500.0], 60.0, 0.0, "\u{4e2d}\u{6587} ok"),
    ];
    let mut out = CollectTiles::default();
    let (report, _) =
        tile_primitives(&prims, &TilerConfig::default().with_max_zoom(5), &mut out).unwrap();

    let texts: BTreeSet<String> = tile_texts(&out).into_iter().map(|t| t.1).collect();
    assert!(texts.contains("4'-01/4\""), "{texts:?}");
    assert!(texts.contains("Line one\nLine two"), "{texts:?}");
    assert!(texts.contains("7.0711"));
    assert!(
        texts.iter().all(|t| !t.contains('\\') && !t.contains('{')),
        "{texts:?}"
    );

    // What was let go is counted and said.
    assert_eq!(
        report.culled_empty_text, 1,
        "the string that was only formatting"
    );
    assert_eq!(report.fonts_substituted.get("Arial"), Some(&1));
    assert_eq!(report.text_codes_ignored.get(&'H'), Some(&1));
    assert_eq!(report.text_codes_ignored.get(&'A'), Some(&2));
    assert_eq!(report.glyphs_unavailable.get(&'\u{4e2d}'), Some(&1));
    assert_eq!(report.glyphs_unavailable.get(&'\u{6587}'), Some(&1));
    let codes: BTreeSet<DiagnosticCode> = report.diagnostics.iter().map(Diagnostic::code).collect();
    assert!(codes.contains(&DiagnosticCode::FONT_SUBSTITUTED));
    assert!(codes.contains(&DiagnosticCode::GLYPH_UNAVAILABLE));
    assert!(codes.contains(&DiagnosticCode::TEXT_FORMAT_IGNORED));
    // The font diagnostic names the font.
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.message().contains("Arial"))
    );
}

#[test]
fn multi_line_text_is_one_label_with_line_breaks() {
    let prims = vec![
        frame(),
        text(
            [100.0, 300.0],
            80.0,
            0.0,
            "this is a Mtext\nwith multiple lines in it",
        ),
    ];
    let mut out = CollectTiles::default();
    tile_primitives(&prims, &TilerConfig::default().with_max_zoom(4), &mut out).unwrap();
    let t = tile_texts(&out);
    assert!(
        t.iter()
            .all(|(_, s, _)| s == "this is a Mtext\nwith multiple lines in it")
    );
    assert!(!t.is_empty());
}

#[test]
fn diagnostics_are_bounded_however_many_distinct_gaps_there_are() {
    let mut prims = vec![frame()];
    for i in 0..600u32 {
        let c = char::from_u32(0x4e00 + i).unwrap();
        prims.push(text(
            [
                10.0 + f64::from(i % 50) * 18.0,
                50.0 + f64::from(i / 50) * 40.0,
            ],
            12.0,
            0.0,
            &c.to_string(),
        ));
    }
    let mut out = CollectTiles::default();
    let (report, _) =
        tile_primitives(&prims, &TilerConfig::default().with_max_zoom(2), &mut out).unwrap();
    assert_eq!(
        report.glyphs_unavailable.len(),
        600,
        "every character is counted"
    );
    assert_eq!(
        report.diagnostics.len(),
        libviprs::vector::tiler::TilerReport::DIAGNOSTIC_LIMIT
    );
    assert_eq!(
        report.dropped_diagnostics,
        600 - libviprs::vector::tiler::TilerReport::DIAGNOSTIC_LIMIT as u64
    );
}

// ---------------------------------------------------------------------------
// Policies, against a fake drawing
// ---------------------------------------------------------------------------

struct Fake {
    views: Vec<CadView>,
    primitives: usize,
    emitted: std::cell::Cell<usize>,
    fail_after: Option<usize>,
}

impl CadDrawing for Fake {
    fn views(&self) -> Result<Vec<CadView>, CadError> {
        Ok(self.views.clone())
    }

    fn decode(
        &self,
        _view: u32,
        sink: &mut dyn PrimitiveSink,
        report: &mut DecodeReport,
    ) -> Result<(), CadError> {
        for i in 0..self.primitives {
            if self.fail_after == Some(i) {
                report.push(Diagnostic::from_code(
                    DiagnosticCode::UNSUPPORTED_ENTITY,
                    "a thing before the failure",
                ));
                return Err(CadError::provider("fake", "gave up"));
            }
            let x = i as f64;
            sink.primitive(Line::new([x, 0.0, 0.0], [x, 10.0, 0.0])?.into())?;
            self.emitted.set(self.emitted.get() + 1);
            report.record(libviprs::cad::PrimitiveKind::Line);
        }
        report.mark_complete();
        sink.finish()
    }
}

fn model_and_layout() -> Vec<CadView> {
    vec![
        CadView::new(0, 0, [0.0, 0.0, 100.0, 100.0], 0, "Model"),
        CadView::new(1, 1, [0.0, 0.0, 420.0, 297.0], 0, "Layout1"),
    ]
}

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("fid1034-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn paper_space_is_refused_until_it_is_enabled() {
    let d = Fake {
        views: model_and_layout(),
        primitives: 5,
        emitted: 0.into(),
        fail_after: None,
    };
    let dir = scratch("layout");
    let path = dir.join("l.pmtiles");
    let r = build_archive_from_view(&d, 1, &TilerConfig::default().with_max_zoom(2), &path, "x");
    assert!(
        matches!(r, Err(TilerError::ViewNotEnabled { index: 1, ref name }) if name == "Layout1"),
        "{r:?}"
    );
    assert_eq!(
        d.emitted.get(),
        0,
        "nothing was decoded for a view that is not enabled"
    );
    assert!(!path.exists());
    // Model Space is.
    assert!(
        build_archive_from_view(&d, 0, &TilerConfig::default().with_max_zoom(2), &path, "x")
            .is_ok()
    );
    assert!(matches!(
        build_archive_from_view(&d, 7, &TilerConfig::default(), &path, "x"),
        Err(TilerError::Decode(CadError::NoSuchView { index: 7, .. }))
    ));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_repeated_block_is_stopped_at_the_primitive_budget_not_at_out_of_memory() {
    let d = Fake {
        views: model_and_layout(),
        primitives: 1_000_000,
        emitted: 0.into(),
        fail_after: None,
    };
    let dir = scratch("budget");
    let cfg = TilerConfig {
        max_primitives: 1_000,
        ..TilerConfig::default().with_max_zoom(2)
    };
    let r = build_archive_from_view(&d, 0, &cfg, dir.join("b.pmtiles"), "x");
    assert!(
        matches!(r, Err(TilerError::TooManyPrimitives { limit: 1_000 })),
        "{r:?}"
    );
    // The decode was stopped as soon as the budget was passed, not run to the
    // end and checked afterwards.
    assert_eq!(d.emitted.get(), 1_000);
    assert!(!dir.join("b.pmtiles").exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_decode_that_fails_keeps_what_it_reported() {
    let d = Fake {
        views: model_and_layout(),
        primitives: 100,
        emitted: 0.into(),
        fail_after: Some(10),
    };
    let dir = scratch("fail");
    let r = build_archive_from_view(
        &d,
        0,
        &TilerConfig::default().with_max_zoom(2),
        dir.join("f.pmtiles"),
        "x",
    );
    match r {
        Err(TilerError::DecodeFailed { report, .. }) => {
            assert_eq!(report.counts().total(), 10);
            assert!(
                report
                    .diagnostics()
                    .iter()
                    .any(|x| x.message().contains("before the failure"))
            );
        }
        other => panic!("{other:?}"),
    }
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// Real drawings
// ---------------------------------------------------------------------------

#[cfg(feature = "dwg-acadsharp")]
mod native {
    use super::*;
    use libviprs::cad::acadsharp::AcadSharpDecoder;
    use libviprs::cad::{CadDecoder, CadSource, CollectSink};

    fn decoder() -> Option<AcadSharpDecoder> {
        match AcadSharpDecoder::new() {
            Ok(d) => Some(d),
            Err(e) => {
                assert!(
                    std::env::var("LIBVIPRS_REQUIRE_ACADSHARP_NATIVE").as_deref() != Ok("1"),
                    "native lane required: {e:?}"
                );
                eprintln!("skipped: {e}");
                None
            }
        }
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/cad/g13")
                .join(format!("{name}.dwg")),
        )
        .unwrap()
    }

    fn decode(d: &AcadSharpDecoder, name: &str) -> (Vec<Primitive>, DecodeReport) {
        let bytes = fixture(name);
        let drawing = d.open(CadSource::Bytes(&bytes)).unwrap();
        let mut sink = CollectSink::new();
        let mut report = DecodeReport::new();
        let _ = drawing.decode(0, &mut sink, &mut report);
        (sink.into_collected(), report)
    }

    fn codes(r: &DecodeReport) -> BTreeSet<u32> {
        r.diagnostics().iter().map(|d| d.code().get()).collect()
    }

    fn convert(
        d: &AcadSharpDecoder,
        name: &str,
        view: u32,
        cfg: &TilerConfig,
    ) -> Result<(libviprs::vector::tiler::TilerReport, DecodeReport), TilerError> {
        let bytes = fixture(name);
        let drawing = d.open(CadSource::Bytes(&bytes)).unwrap();
        let dir = scratch(name);
        let r = build_archive_from_view(drawing.as_ref(), view, cfg, dir.join("o.pmtiles"), name);
        let _ = std::fs::remove_dir_all(dir);
        r
    }

    fn tiles_of(d: &AcadSharpDecoder, name: &str, max_zoom: u8) -> CollectTiles {
        let (prims, _) = decode(d, name);
        let mut out = CollectTiles::default();
        tile_primitives(
            &prims,
            &TilerConfig::default().with_max_zoom(max_zoom),
            &mut out,
        )
        .unwrap();
        out
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn text_and_mtext_are_visible_in_the_tiles() {
        let Some(d) = decoder() else { return };
        let out = tiles_of(&d, "g13_text", 12);
        let found: BTreeSet<String> = tile_texts(&out).into_iter().map(|t| t.1).collect();
        assert!(found.contains("VIPRS-G13-TEXT"), "{found:?}");
        assert!(found.contains("VIPRS-G13-MTEXT"), "{found:?}");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn a_very_long_string_is_one_label_not_a_crash() {
        let Some(d) = decoder() else { return };
        let (prims, _) = decode(&d, "g13_long_text");
        assert_eq!(prims.len(), 1);
        let Primitive::Text(t) = &prims[0] else {
            panic!()
        };
        let len = t.content().chars().count();
        let mut out = CollectTiles::default();
        let (rep, _) =
            tile_primitives(&prims, &TilerConfig::default().with_max_zoom(10), &mut out).unwrap();
        eprintln!(
            "MEASURED g13_long_text: {len} chars, culled_text {}, text written {}",
            rep.culled_text, rep.text
        );
        // The label is a few hundred times wider than it is tall, so it is
        // small in a world sized to its width. It is drawn from the zoom its
        // height reaches the minimum, and counted as culled below that.
        assert!(rep.text >= 1, "never drawn: {rep:?}");
        assert!(rep.culled_text >= 1, "and the lower zooms are counted");
        assert!(tile_texts(&out).iter().all(|t| t.1.chars().count() == len));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn nested_blocks_are_visible() {
        let Some(d) = decoder() else { return };
        for name in [
            "g13_insert",
            "g13_deep_blocks",
            "g13_ocs_rotated",
            "g13_slot_block",
        ] {
            let (prims, report) = decode(&d, name);
            let expanded = prims
                .iter()
                .filter(|p| match p {
                    Primitive::Line(x) => x.origin().is_expanded_insert(),
                    Primitive::Polyline(x) => x.origin().is_expanded_insert(),
                    Primitive::Arc(x) => x.origin().is_expanded_insert(),
                    Primitive::Circle(x) => x.origin().is_expanded_insert(),
                    _ => false,
                })
                .count();
            assert!(expanded > 0, "{name}: nothing came from an expanded insert");
            assert!(report.is_complete(), "{name}");
            // And it reaches the tiles.
            let mut out = CollectTiles::default();
            let (rep, _) =
                tile_primitives(&prims, &TilerConfig::default().with_max_zoom(4), &mut out)
                    .unwrap();
            assert!(rep.points > 0 && rep.tiles > 0, "{name}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn a_block_inserted_thirty_thousand_times_stays_inside_its_budgets() {
        let Some(d) = decoder() else { return };
        let (prims, report) = decode(&d, "g13_many_inserts");
        assert_eq!(prims.len(), 30_000);
        assert!(report.is_complete());
        let mut out = CollectTiles::default();
        let cfg = TilerConfig::default().with_max_zoom(8);
        let t = std::time::Instant::now();
        let (rep, _) = tile_primitives(&prims, &cfg, &mut out).unwrap();
        eprintln!(
            "MEASURED g13_many_inserts: 30000 primitives -> {} tiles, {} points, largest tile {} bytes, {:.2?}",
            rep.tiles,
            rep.points,
            rep.largest_tile_bytes,
            t.elapsed()
        );
        assert_eq!(rep.oversized_count, 0);
        assert!(rep.largest_tile_bytes < cfg.max_tile_bytes);
        assert!(t.elapsed().as_secs() < 120);

        // The memory bound for repeats is the primitive budget.
        let tight = TilerConfig {
            max_primitives: 5_000,
            ..cfg
        };
        assert!(matches!(
            convert(&d, "g13_many_inserts", 0, &tight),
            Err(TilerError::TooManyPrimitives { limit: 5_000 })
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn dimensions_are_readable() {
        let Some(d) = decoder() else { return };
        let (prims, _) = decode(&d, "g13_dimension");
        assert_eq!(
            prims
                .iter()
                .filter(|p| matches!(p, Primitive::Text(_)))
                .count(),
            1
        );
        assert_eq!(
            prims
                .iter()
                .filter(|p| matches!(p, Primitive::Line(_)))
                .count(),
            3
        );
        let out = tiles_of(&d, "g13_dimension", 14);
        let t = tile_texts(&out);
        assert!(
            t.iter()
                .any(|(_, s, h)| s == "10" && (*h - 0.18).abs() < 1e-9),
            "{t:?}"
        );
        // The extension and dimension lines are in the tiles with it.
        let lines: usize = out
            .tiles
            .values()
            .flat_map(|t| t.layers.iter())
            .filter(|l| l.name == "lines")
            .count();
        assert!(lines > 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn pathological_dimension_nesting_fails_bounded_and_says_so() {
        let Some(d) = decoder() else { return };
        let t = std::time::Instant::now();
        let r = convert(
            &d,
            "g13_dimension_deep",
            0,
            &TilerConfig::default().with_max_zoom(2),
        );
        assert!(t.elapsed().as_secs() < 60, "took {:?}", t.elapsed());
        match r {
            Err(TilerError::DecodeFailed { report, .. }) => {
                assert!(
                    report
                        .errors()
                        .any(|e| e.code() == DiagnosticCode::DECODE_FAILED)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn hatches_are_drawn_as_outlines_and_every_gap_is_reported() {
        let Some(d) = decoder() else { return };
        let (prims, report) = decode(&d, "g13_hatch");
        assert_eq!(
            prims
                .iter()
                .filter(|p| matches!(p, Primitive::Polygon(_)))
                .count(),
            3
        );
        let c = codes(&report);
        // Pattern-only hatch, a loop that is not a polygon, and a spline
        // refused for a zero weight: all three said.
        for expected in [103, 104, 502] {
            assert!(c.contains(&expected), "{c:?}");
        }
        let mut out = CollectTiles::default();
        let (rep, _) =
            tile_primitives(&prims, &TilerConfig::default().with_max_zoom(5), &mut out).unwrap();
        assert!(rep.points > 0);
        assert!(
            out.tiles
                .values()
                .flat_map(|t| &t.layers)
                .all(|l| l.name != "fills" || l.features.is_empty())
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn entities_that_are_not_drawn_are_always_in_the_report() {
        let Some(d) = decoder() else { return };
        for name in [
            "g13_solid",
            "g13_point",
            "g13_mline",
            "g13_leader",
            "g13_wipeout",
            "g13_mesh",
            "g13_face3d",
            "g13_tolerance",
            "g13_unsupported",
        ] {
            let (prims, report) = decode(&d, name);
            assert!(prims.is_empty(), "{name} draws {} primitives", prims.len());
            let c = codes(&report);
            assert!(
                c.contains(&100),
                "{name}: nothing was drawn and UNSUPPORTED_ENTITY is not in {c:?}"
            );
            assert!(
                c.contains(&108),
                "{name}: and the view is not marked empty: {c:?}"
            );
        }
        for name in ["g13_xref", "g13_xref_long"] {
            let (_, report) = decode(&d, name);
            assert!(codes(&report).contains(&105), "{name}: UNRESOLVED_BLOCK");
        }
        let (_, report) = decode(&d, "g13_nonuniform");
        assert!(codes(&report).contains(&106));
        let (_, report) = decode(&d, "g13_nan_bulge");
        assert!(codes(&report).contains(&107));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn a_view_with_nothing_drawable_is_an_error_not_an_empty_archive() {
        let Some(d) = decoder() else { return };
        let r = convert(
            &d,
            "g13_unsupported",
            0,
            &TilerConfig::default().with_max_zoom(2),
        );
        assert!(matches!(r, Err(TilerError::NoGeometry)), "{r:?}");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn layouts_are_not_converted_even_when_the_drawing_has_them() {
        let Some(d) = decoder() else { return };
        let r = convert(&d, "g13_text", 1, &TilerConfig::default().with_max_zoom(2));
        assert!(
            matches!(r, Err(TilerError::ViewNotEnabled { index: 1, .. })),
            "{r:?}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
    fn the_real_drawings_formatting_codes_are_read_and_counted() {
        let Some(d) = decoder() else { return };
        let bytes = include_bytes!("fixtures/cad/sample_AC1018.dwg");
        let drawing = d.open(CadSource::Bytes(bytes)).unwrap();
        let dir = scratch("sample");
        let (rep, _) = build_archive_from_view(
            drawing.as_ref(),
            0,
            &TilerConfig::default().with_max_zoom(9),
            dir.join("s.pmtiles"),
            "s",
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(dir);
        // `4'-0{\H1x;\S1#4;}"` and `\A1;7.0711` are in this drawing.
        assert!(
            rep.text_codes_ignored.get(&'H').copied().unwrap_or(0) >= 1,
            "{:?}",
            rep.text_codes_ignored
        );
        assert!(rep.text_codes_ignored.get(&'A').copied().unwrap_or(0) >= 1);
        assert!(
            rep.diagnostics
                .iter()
                .any(|d| d.code() == DiagnosticCode::TEXT_FORMAT_IGNORED)
        );
        eprintln!(
            "MEASURED sample_AC1018 text: ignored codes {:?}, fonts {:?}, glyphs {:?}",
            rep.text_codes_ignored, rep.fonts_substituted, rep.glyphs_unavailable
        );
    }
}
