//! A real DWG through the whole pipeline: decoder, tiler, vector PMTiles
//! (issues #1030, #1031 and #1032 together).
//!
//! Needs the native acadsharp archive; without one it says so and returns,
//! unless `LIBVIPRS_REQUIRE_ACADSHARP_NATIVE=1` makes that a failure.
#![cfg(feature = "dwg-acadsharp")]

use libviprs::cad::acadsharp::AcadSharpDecoder;
use libviprs::cad::{CadDecoder, CadSource};
use libviprs::pmtiles::validate::{ValidationLimits, validate_bytes};
use libviprs::vector::VectorArchive;
use libviprs::vector::tiler::{TilerConfig, build_archive_from_view};

const SAMPLE: &[u8] = include_bytes!("fixtures/cad/sample_AC1018.dwg");

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_real_dwg_becomes_a_valid_black_line_archive() {
    let decoder = match AcadSharpDecoder::new() {
        Ok(d) => d,
        Err(e) => {
            assert!(
                std::env::var("LIBVIPRS_REQUIRE_ACADSHARP_NATIVE").as_deref() != Ok("1"),
                "native lane required: {e:?}"
            );
            eprintln!("skipped: {e}");
            return;
        }
    };
    let drawing = decoder.open(CadSource::Bytes(SAMPLE)).unwrap();
    let dir = std::env::temp_dir().join(format!("dwg1032-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sample.pmtiles");
    let cfg = TilerConfig::default().with_max_zoom(8);
    let (report, decode) =
        build_archive_from_view(drawing.as_ref(), 0, &cfg, &path, "sample_AC1018").unwrap();

    assert!(decode.is_complete());
    assert!(report.tiles > 20, "{report:?}");
    assert_eq!(report.oversized_count, 0);
    assert_eq!(report.curves_capped, 0);
    assert_eq!(report.undrawable, 0, "every decoded primitive was drawable");
    eprintln!(
        "MEASURED sample_AC1018 view 0 at max_zoom 8: {} tiles, {} points, {} text, largest tile {} bytes, culled {} / text {}, precision {:?}",
        report.tiles,
        report.points,
        report.text,
        report.largest_tile_bytes,
        report.culled_subpixel,
        report.culled_text,
        report.precision_at_max_zoom
    );

    let bytes = std::fs::read(&path).unwrap();
    let v = validate_bytes(&bytes, &ValidationLimits::default()).unwrap();
    assert!(v.is_valid(), "{:?}", v.findings);
    let archive = VectorArchive::try_open(&path).unwrap();
    let z0 = archive.tile(0, 0, 0).unwrap().expect("a zoom 0 tile");
    assert!(z0.layers.iter().any(|l| l.name == "lines"));
    assert!(report.text > 0, "the drawing's text is in the tiles");

    if let Some(out) = std::env::var_os("LIBVIPRS_VECTOR_GOLDEN_DIR") {
        std::fs::create_dir_all(&out).unwrap();
        std::fs::copy(&path, std::path::Path::new(&out).join("dwg.pmtiles")).unwrap();
    }
    let _ = std::fs::remove_dir_all(dir);
}
