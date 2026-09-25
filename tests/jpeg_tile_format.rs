//! JPEG as a tile format: the alpha it refuses (issue #1133) and the two knobs
//! it never reaches (issue #1132).
//!
//! # Why the two issues share a file
//!
//! They share one function, `encode_jpeg` in `src/sink.rs`, and they are not
//! independent. #1132 is about what a JPEG tile costs, and the input whose
//! cost anybody actually cares about is a rendered vector PDF, which #1133
//! says cannot be tiled as JPEG at all. So the refusal goes first and the cost
//! cells then have something real to stand on.
//!
//! # #1133, the refusal
//!
//! `render_page_pdfium` hands the sink a [`PixelFormat::Rgba8`] raster, on
//! both the full-page and the strip path, and `image`'s JPEG encoder has no
//! RGBA colour type. So `--render --format jpeg` dies with `encoding tile to
//! jpeg failed: The encoder or decoder for Jpeg does not support the color
//! type Rgba8` before one tile lands, while `--format png` and `--format webp`
//! tile the same input.
//!
//! The fixtures here are 4-byte-per-pixel rasters rather than a real render,
//! and [`the_render_path_still_hands_the_sink_rgba`] is what keeps that
//! honest: it reads the two return sites in `src/pdf.rs` instead of trusting
//! my memory of them. A cell that drives pdfium belongs in libviprs-tests,
//! where the PDF fixtures and the libpdfium runtime live. Nothing in this
//! repository's CI loads pdfium at all (`ci.yml` lints and type-checks the
//! feature, twice, and never runs it), so a render cell here would be a skip
//! wearing the colour of a pass.
//!
//! # #1132, the two knobs
//!
//! The tile path calls `encode_jpeg(raster, quality)` with no subsampling
//! argument, so it takes whatever the encoder does by default: 4:4:4 chroma
//! with the standard Annex K Huffman tables. Both of those are written into
//! the file, which is what every cell below reads. The SOF0 header carries
//! the sampling factors and the DHT segments carry the tables, so no
//! assertion here has to take the encoder's word for what it did.
//!
//! The measurements in the issue are the reason those two knobs are worth
//! moving: a blank 256x256 JPEG tile costs 2419 bytes against lossless WebP's
//! 44 to 90, because 4:4:4 RGB is 3072 blocks and every one of them pays a DC
//! code and an end-of-block even on blank paper.

use std::path::{Path, PathBuf};

use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner, TileCoord};
use libviprs::sink::{Tile, TileFormat, TileSink};
use libviprs::source::decode_bytes;
use libviprs::{EngineBuilder, EngineKind, FsSink, JpegSubsample, PixelFormat, Raster};

// ---------------------------------------------------------------------------
// Reading a JPEG's own account of itself
// ---------------------------------------------------------------------------

/// Start of frame, baseline sequential.
const SOF0: u8 = 0xC0;
/// Define Huffman table(s).
const DHT: u8 = 0xC4;
/// Start of scan.
const SOS: u8 = 0xDA;

/// The marker segments of a JPEG, as `(marker, payload)`, up to and including
/// the scan header.
///
/// Everything this crate emits is a run of length-prefixed segments between
/// SOI and SOS, so the walk asserts that shape rather than tolerating the
/// standalone markers a general parser would have to handle. A stream that
/// does not have it fails here, loudly, instead of silently handing back a
/// short list that every assertion below would then read as "the segment is
/// absent".
fn segments(jpeg: &[u8]) -> Vec<(u8, Vec<u8>)> {
    assert!(jpeg.len() > 4, "a JPEG is longer than four bytes");
    assert_eq!(&jpeg[..2], [0xFF, 0xD8], "a JPEG opens with SOI");
    let mut out = Vec::new();
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        assert_eq!(jpeg[i], 0xFF, "expected a marker byte at offset {i}");
        let marker = jpeg[i + 1];
        let len = usize::from(u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]));
        assert!(
            len >= 2,
            "marker {marker:#04x} declares a {len}-byte segment"
        );
        assert!(
            i + 2 + len <= jpeg.len(),
            "marker {marker:#04x} declares {len} bytes and the file has \
             {} left",
            jpeg.len() - i - 2
        );
        out.push((marker, jpeg[i + 4..i + 2 + len].to_vec()));
        i += 2 + len;
        if marker == SOS {
            break;
        }
    }
    assert!(
        out.iter().any(|(m, _)| *m == SOS),
        "the walk never reached a scan header"
    );
    out
}

/// The `(component id, horizontal, vertical)` sampling factors SOF0 declares.
///
/// This is the whole of what "4:2:0" means on the wire: luma 2x2 against
/// chroma 1x1. Reading it back is the only way to tell a subsample mode that
/// reached the encoder from one that was accepted and dropped, which is
/// exactly the shape #1132 is about.
fn sampling_factors(jpeg: &[u8]) -> Vec<(u8, u8, u8)> {
    let segs = segments(jpeg);
    let (_, sof) = segs
        .iter()
        .find(|(m, _)| *m == SOF0)
        .expect("a baseline JPEG carries an SOF0 header");
    // precision, height, width, component count, then three bytes each.
    let count = usize::from(sof[5]);
    assert_eq!(
        sof.len(),
        6 + 3 * count,
        "SOF0 declares {count} components and is {} bytes",
        sof.len()
    );
    (0..count)
        .map(|c| {
            let at = 6 + 3 * c;
            (sof[at], sof[at + 1] >> 4, sof[at + 1] & 0x0F)
        })
        .collect()
}

/// The pixel dimensions SOF0 declares, which is the frame the decoder will
/// reconstruct rather than the raster we handed the encoder.
fn frame_size(jpeg: &[u8]) -> (u32, u32) {
    let segs = segments(jpeg);
    let (_, sof) = segs
        .iter()
        .find(|(m, _)| *m == SOF0)
        .expect("a baseline JPEG carries an SOF0 header");
    (
        u32::from(u16::from_be_bytes([sof[3], sof[4]])),
        u32::from(u16::from_be_bytes([sof[1], sof[2]])),
    )
}

/// Every DHT payload, concatenated in the order they appear.
///
/// Concatenated on purpose: the question these cells ask is whether the tables
/// came from this image or out of a constant, and that is a property of the
/// whole set rather than of any one table.
fn huffman_tables(jpeg: &[u8]) -> Vec<u8> {
    segments(jpeg)
        .into_iter()
        .filter(|(m, _)| *m == DHT)
        .flat_map(|(_, payload)| payload)
        .collect()
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// What `render_page_pdfium` hands the sink: four bytes per pixel with alpha
/// 255 everywhere.
///
/// The issue measured exactly this on `blueprint.pdf` at 150 dpi: the alpha
/// channel was opaque in every pixel, so nothing was riding on it and the
/// refusal is purely about the colour type. The content is dark ink on pale
/// paper because that is what a drawing looks like and because a uniform
/// raster would let a blank-tile strategy write a one-byte marker instead of
/// an encoded tile.
///
/// **The ink is blue**, which used to be incidental and is not any more: since
/// issue #1134 `JpegSubsample::Auto` asks the content as well as the quality,
/// so this fixture is the *coloured* case and [`mono_rendered_rgba`] is the
/// one that still subsamples. A cell about subsampling wants the second; a
/// cell about alpha or about tile geometry can have either.
fn rendered_rgba(w: u32, h: u32) -> Raster {
    ink_on_paper(w, h, [24, 24, 90])
}

/// The same drawing in grey ink, which is what a CAD sheet actually is and
/// what the corpus behind #1134 is made of.
fn mono_rendered_rgba(w: u32, h: u32) -> Raster {
    ink_on_paper(w, h, [24, 24, 24])
}

fn ink_on_paper(w: u32, h: u32, ink_rgb: [u8; 3]) -> Raster {
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let off = ((y * w + x) * 4) as usize;
            let ink = x % 32 == 0 || y % 32 == 0 || (x + y) % 71 == 0;
            let px = if ink { ink_rgb } else { [236, 236, 236] };
            data[off..off + 3].copy_from_slice(&px);
            data[off + 3] = 255;
        }
    }
    Raster::new(w, h, PixelFormat::Rgba8, data).expect("an rgba raster is well formed")
}

/// The same shape with a genuinely transparent square in the middle, which is
/// the case the opaque corpus never exercised.
fn rgba_with_a_hole(w: u32, h: u32) -> Raster {
    let mut raster = rendered_rgba(w, h);
    let data = raster.data_mut();
    for y in h / 4..h * 3 / 4 {
        for x in w / 4..w * 3 / 4 {
            let off = ((y * w + x) * 4) as usize;
            // Opaque black underneath, so a fix that merely drops the alpha
            // byte reads as black here and a fix that composites reads as the
            // configured background. The two are not the same answer.
            data[off] = 0;
            data[off + 1] = 0;
            data[off + 2] = 0;
            data[off + 3] = 0;
        }
    }
    raster
}

/// Three-band ink on paper, no alpha: the control that says these cells fail
/// on the alpha rather than on anything else about the fixture.
fn rgb_drawing(w: u32, h: u32) -> Raster {
    drop_alpha(rendered_rgba(w, h), w, h)
}

/// [`mono_rendered_rgba`] with the alpha dropped: the fixture for every cell
/// about subsampling, because it is the one `Auto` still subsamples.
fn mono_rgb_drawing(w: u32, h: u32) -> Raster {
    drop_alpha(mono_rendered_rgba(w, h), w, h)
}

fn drop_alpha(rgba: Raster, w: u32, h: u32) -> Raster {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for px in rgba.data().as_chunks::<4>().0 {
        data.extend_from_slice(&px[..3]);
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).expect("an rgb raster is well formed")
}

/// Luma RMSE and worst single-pixel luma error between two RGB buffers.
///
/// Luma rather than per-band because 4:2:0 throws away three quarters of the
/// chroma samples on purpose, so a per-band bound would either have to be
/// loose enough to hide a real defect or fail on the subsampling itself. The
/// worst-pixel figure is the one that can see a handful of corrupt blocks,
/// because an average over a whole tile cannot.
fn luma_error(want: &[u8], got: &[u8]) -> (f64, f64) {
    let luma =
        |px: &[u8]| 0.299 * f64::from(px[0]) + 0.587 * f64::from(px[1]) + 0.114 * f64::from(px[2]);
    let mut worst = 0f64;
    let mut sum = 0f64;
    let mut n = 0f64;
    for (a, b) in want.as_chunks::<3>().0.iter().zip(got.as_chunks::<3>().0) {
        let d = (luma(a) - luma(b)).abs();
        worst = worst.max(d);
        sum += d * d;
        n += 1.0;
    }
    ((sum / n).sqrt(), worst)
}

fn plan_for(w: u32, h: u32, tile: u32) -> PyramidPlan {
    PyramidPlanner::new(w, h, tile, 0, Layout::DeepZoom)
        .expect("a square plan is valid")
        .plan()
}

/// The full-resolution level, which is the last one in a Deep Zoom plan.
fn full_res(plan: &PyramidPlan) -> u32 {
    u32::try_from(plan.levels.len() - 1).expect("a level index fits in u32")
}

/// Write one tile through an `FsSink` with no engine attached and hand back
/// the bytes that landed on disk.
fn tile_bytes(dir: &Path, plan: &PyramidPlan, raster: Raster, quality: u8) -> Vec<u8> {
    let root = dir.join(format!("q{quality}-{}", raster.width()));
    let sink = FsSink::new(&root, plan.clone()).with_format(TileFormat::Jpeg { quality });
    let coord = TileCoord::new(full_res(plan), 0, 0);
    sink.write_tile(&Tile {
        coord,
        raster,
        blank: false,
    })
    .expect("a jpeg tile encodes");
    sink.finish().expect("the sink finishes");
    let rel = plan
        .tile_path(coord, "jpeg")
        .expect("the coord is inside the plan");
    std::fs::read(root.join(rel)).expect("the tile is on disk")
}

/// Run a source through the engine into a JPEG tree and hand back the root.
fn jpeg_tree(dir: &Path, plan: &PyramidPlan, src: &Raster, background: [u8; 3]) -> PathBuf {
    let root = dir.join("jpeg-tiles");
    let sink = FsSink::new(&root, plan.clone()).with_format(TileFormat::Jpeg { quality: 85 });
    EngineBuilder::new(src, plan.clone(), sink)
        .with_background_rgb(background)
        .run()
        .expect("a jpeg pyramid generates");
    root
}

// ---------------------------------------------------------------------------
// #1133: the rendered-PDF refusal
// ---------------------------------------------------------------------------

/// The reproduction. An RGBA source tiles as JPEG, the way it already tiles as
/// PNG and WebP.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn an_rgba_source_tiles_as_jpeg() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);
    let root = jpeg_tree(
        &dir.path().join("rgba"),
        &plan,
        &rendered_rgba(256, 256),
        [255, 255, 255],
    );

    let rel = plan
        .tile_path(TileCoord::new(full_res(&plan), 0, 0), "jpeg")
        .expect("the coord is inside the plan");
    let bytes = std::fs::read(root.join(rel)).expect("the full-resolution tile is on disk");
    assert_eq!(
        frame_size(&bytes),
        (256, 256),
        "the tile the run wrote is the tile the plan asked for"
    );
    assert_eq!(
        sampling_factors(&bytes).len(),
        3,
        "an RGBA source narrows to three JPEG components, not four: the format \
         has no alpha and never will"
    );
}

/// The alpha does not vanish, it lands on the background the engine was
/// configured with.
///
/// The fixture's transparent square is opaque black underneath, so this
/// separates the two fixes that both make the cell above pass: dropping the
/// fourth byte reads back black here, compositing reads back the background.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn transparent_pixels_take_the_engine_background() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);
    let background = [206, 17, 38];
    let root = jpeg_tree(
        &dir.path().join("hole"),
        &plan,
        &rgba_with_a_hole(256, 256),
        background,
    );

    let rel = plan
        .tile_path(TileCoord::new(full_res(&plan), 0, 0), "jpeg")
        .expect("the coord is inside the plan");
    let bytes = std::fs::read(root.join(rel)).expect("the full-resolution tile is on disk");
    let decoded = decode_bytes(&bytes).expect("the tile decodes");
    assert_eq!(decoded.format(), PixelFormat::Rgb8);

    // Well inside the hole, so 4:2:0 chroma averaging at its edge cannot
    // reach the sample, and away from the quantization ringing that follows
    // the transition.
    let (x, y) = (128u32, 128u32);
    let off = ((y * 256 + x) * 3) as usize;
    let px = [
        decoded.data()[off],
        decoded.data()[off + 1],
        decoded.data()[off + 2],
    ];
    for band in 0..3 {
        let delta = i32::from(px[band]) - i32::from(background[band]);
        assert!(
            delta.abs() <= 8,
            "the transparent square reads {px:?} and the configured background \
             is {background:?}; band {band} is {delta} off, which is more than \
             lossy coding of a flat colour explains"
        );
    }
}

/// The background reaches the flattening on **every** engine, not just the one
/// a small in-memory fixture happens to select.
///
/// `transparent_pixels_take_the_engine_background` above drives a 256x256
/// `Raster` with no memory budget, so `EngineKind::Auto` resolves to
/// `Monolithic` and the cell can only ever exercise that engine. That is the
/// shape of a probe set that lands on fixed points: it passes, and it is
/// incapable of failing for the reason anybody cares about.
///
/// The reason anybody cares is that `--render` on a large PDF is a strip
/// source, `resolve_engine_kind` sends every strip source to `Streaming`, and
/// the CLI picks Streaming or MapReduce whenever a memory budget is set. That
/// is precisely the input issue #1133 was filed about, so the headline case
/// was the one running down the path this cell exists to cover.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn every_engine_flattens_onto_the_configured_background() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);
    let background = [206, 17, 38];

    for kind in [
        EngineKind::Monolithic,
        EngineKind::Streaming,
        EngineKind::MapReduce,
    ] {
        let root = dir.path().join(format!("{kind:?}"));
        let sink = FsSink::new(&root, plan.clone()).with_format(TileFormat::Jpeg { quality: 85 });
        let src = rgba_with_a_hole(256, 256);
        EngineBuilder::new(&src, plan.clone(), sink)
            .with_engine(kind)
            .with_background_rgb(background)
            .run()
            .unwrap_or_else(|e| panic!("{kind:?} generates a jpeg pyramid: {e}"));

        let rel = plan
            .tile_path(TileCoord::new(full_res(&plan), 0, 0), "jpeg")
            .expect("the coord is inside the plan");
        let bytes = std::fs::read(root.join(rel))
            .unwrap_or_else(|e| panic!("{kind:?} wrote the full-resolution tile: {e}"));
        let decoded = decode_bytes(&bytes).expect("the tile decodes");
        let off = ((128 * 256 + 128) * 3) as usize;
        let px = [
            decoded.data()[off],
            decoded.data()[off + 1],
            decoded.data()[off + 2],
        ];
        for band in 0..3 {
            let delta = i32::from(px[band]) - i32::from(background[band]);
            assert!(
                delta.abs() <= 8,
                "{kind:?}: the transparent square reads {px:?} against a \
                 configured background of {background:?}. White here means the \
                 engine never handed the sink its config, so the flattening \
                 used the standalone default while the padding in the same \
                 tile used the real one"
            );
        }
    }
}

/// The control for the fixtures above: the render path really does still hand
/// the sink RGBA.
///
/// Both cells here stand on a raster I wrote by hand, so if `render_page_pdfium`
/// ever starts answering `Rgb8` they keep passing while testing nothing anybody
/// asked about. Reading the two return sites is cheap and it makes that
/// divergence fail here rather than go unnoticed.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn the_render_path_still_hands_the_sink_rgba() {
    let pdf = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("pdf.rs"),
    )
    .expect("src/pdf.rs is readable");
    // Both render sites, by name. `src/pdf.rs` also builds an `Rgb8` raster
    // for the embedded-image extraction path, so counting mentions of either
    // format across the file would answer a different question than the one
    // this cell asks.
    for site in ["fn bitmap_to_raster", "fn render_page_strip_with_page"] {
        let at = pdf
            .find(site)
            .unwrap_or_else(|| panic!("`{site}` moved or was renamed in src/pdf.rs"));
        let body: String = pdf[at..].chars().take(2500).collect();
        assert!(
            body.contains("PixelFormat::Rgba8"),
            "`{site}` no longer builds an Rgba8 raster. That is the pixel \
             format #1133 is about and the fixtures in this file stand on it, \
             so they need re-reading rather than re-running"
        );
    }
}

// ---------------------------------------------------------------------------
// #1132: subsampling and the entropy tables
// ---------------------------------------------------------------------------

/// The tile path reaches a subsampling decision instead of taking the
/// encoder's default.
///
/// `JpegSubsample::Auto` started as libvips' `VIPS_FOREIGN_SUBSAMPLE_AUTO`,
/// 4:2:0 below quality 90 and 4:4:4 at or above it, and since issue #1134 it
/// asks the content too. The tile default is quality 85 and a CAD sheet is
/// black ink on white paper, so that default still selects 4:2:0 here, which
/// is the whole of #1132's size argument.
///
/// The fixture is the **monochrome** one on purpose. It used to be the blue
/// one, and with the content gate in place that cell would have been asserting
/// 4:2:0 on the one input the gate exists to keep away from it.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_tile_at_the_default_quality_subsamples_chroma() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);
    let bytes = tile_bytes(dir.path(), &plan, mono_rgb_drawing(256, 256), 85);

    assert_eq!(
        sampling_factors(&bytes),
        vec![(1, 2, 2), (2, 1, 1), (3, 1, 1)],
        "a quality-85 tile of black ink on white paper is 4:2:0: luma 2x2 \
         against chroma 1x1"
    );
}

/// And a coloured tile at the same quality keeps its chroma, through the sink.
///
/// The encoder's own cells cover the decision; this one covers the path, since
/// `FsSink::encode_tile` is what passes `Auto` down and a gate the tile path
/// never reaches is the exact shape of the bug #1132 was filed about.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_coloured_tile_at_the_default_quality_keeps_full_chroma() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);
    let bytes = tile_bytes(dir.path(), &plan, rgb_drawing(256, 256), 85);

    assert_eq!(
        sampling_factors(&bytes),
        vec![(1, 1, 1), (2, 1, 1), (3, 1, 1)],
        "blue linework at quality 85 should keep full chroma (issue #1134)"
    );
}

/// And the mode is a decision rather than a constant: at quality 90 the same
/// content keeps full chroma.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_tile_at_quality_90_keeps_full_chroma() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);
    // Monochrome, so the only thing that can be keeping the chroma here is the
    // quality. On the coloured fixture this cell would pass for two reasons
    // and tell them apart for neither.
    let src = mono_rgb_drawing(256, 256);
    let bytes = tile_bytes(dir.path(), &plan, src.clone(), 90);

    assert_eq!(
        sampling_factors(&bytes),
        vec![(1, 1, 1), (2, 1, 1), (3, 1, 1)],
        "quality 90 is where Auto stops subsampling, so this one is 4:4:4"
    );

    // The 4:4:4 path had no pixel check at all, only a sampling factor and a
    // decode, which between them are true of any decodable file that declares
    // the right header. Full chroma at quality 90 should be *closer* to the
    // source than the 4:2:0 default is, so the bounds here are tighter than
    // the ones the subsampled cells carry rather than the same.
    let decoded = decode_bytes(&bytes).expect("the tile decodes");
    let (rmse, worst) = luma_error(src.data(), decoded.data());
    assert!(rmse < 4.0, "4:4:4 luma RMSE is {rmse:.2} at quality 90");
    assert!(
        worst < 60.0,
        "4:4:4 moved one pixel by {worst} at quality 90"
    );
}

/// The explicit modes reach the sampling factors too, which is what
/// `Raster::encode_jpeg_options` has been promising in its signature and
/// dropping in its body.
#[test]
fn the_explicit_subsample_modes_reach_the_encoder() {
    let src = rgb_drawing(64, 64);

    let off = src
        .encode_jpeg_options(80, JpegSubsample::Off)
        .expect("4:4:4 encodes");
    assert_eq!(
        sampling_factors(&off),
        vec![(1, 1, 1), (2, 1, 1), (3, 1, 1)],
        "Off is 4:4:4 whatever the quality says"
    );

    let on = src
        .encode_jpeg_options(95, JpegSubsample::On)
        .expect("4:2:0 encodes");
    assert_eq!(
        sampling_factors(&on),
        vec![(1, 2, 2), (2, 1, 1), (3, 1, 1)],
        "On is 4:2:0 whatever the quality says"
    );
}

/// The empty-tile floor falls.
///
/// 2419 bytes is what the issue measured for a blank 256x256 tile as shipped:
/// 4:4:4 means 3072 blocks and the standard tables charge about six bits for
/// each one's DC-plus-end-of-block even though there is nothing in it. 4:2:0
/// halves the block count and tables built from the tile itself cut what each
/// one costs, so the bound here is well under half. Lossless WebP's floor on
/// the same tile is 44 to 90 bytes, which is the number that made this worth
/// filing; this cell is not claiming JPEG gets there.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn the_blank_tile_floor_falls() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);
    let blank = Raster::new(256, 256, PixelFormat::Rgb8, vec![255u8; 256 * 256 * 3])
        .expect("a blank raster is well formed");
    let bytes = tile_bytes(dir.path(), &plan, blank, 85);

    assert!(
        bytes.len() < 1200,
        "a blank 256x256 JPEG tile is {} bytes; it was 2419 as shipped and the \
         point of #1132 is that most of that was the 4:4:4 block count and the \
         standard tables",
        bytes.len()
    );
}

/// The Huffman tables come from the tile rather than out of Annex K.
///
/// Two tiles with nothing in common produce different tables when the tables
/// are built from the content and byte-identical ones when they are a
/// constant, so this asks the one question that separates the two without
/// needing a copy of the standard tables to compare against.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn the_huffman_tables_are_built_from_the_tile() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);

    let drawing = tile_bytes(dir.path(), &plan, rgb_drawing(256, 256), 85);
    let mut noise = vec![0u8; 256 * 256 * 3];
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for byte in &mut noise {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *byte = (seed >> 24) as u8;
    }
    let noise = tile_bytes(
        &dir.path().join("noise"),
        &plan,
        Raster::new(256, 256, PixelFormat::Rgb8, noise).expect("a noise raster is well formed"),
        85,
    );

    assert_ne!(
        huffman_tables(&drawing),
        huffman_tables(&noise),
        "line art and noise got the same Huffman tables, so the tables are a \
         constant and every tile is paying for symbols it never emits"
    );
}

// ---------------------------------------------------------------------------
// Controls: what must not move
// ---------------------------------------------------------------------------

/// Every emitted tile is the shape and the picture its lossless sibling is.
///
/// The reference is a PNG tree generated from the same source and the same
/// plan, not the JPEG file's own SOF0 header. Comparing a decoded size against
/// the header it was read out of is an identity that holds for any file that
/// decodes at all, which is what this cell asserted before: it said "a JPEG is
/// self-consistent" under a name that claimed it matched the plan.
///
/// `plan.tile_rect(coord)` is not the referent either, and I checked rather
/// than assumed. It reports the content window, 1x1 at the top of this
/// pyramid, while the engine pads every tile out to `tile_size` for every
/// format: the PNG sibling is 128x128 at that same coord. So the honest
/// comparison is against the format that was already right.
///
/// The fixture is 300x220 at a tile size of 128, which leaves a right-hand
/// column 44 px of content wide and a bottom row 92 px tall. Those edge tiles
/// are the only exercise `Pixels::fill_block`'s edge clamping gets, and
/// because PNG is lossless this cell compares their **pixels** and not just
/// their dimensions. A short scan, a mis-clamped edge or a wrong MCU layout
/// shows up here as ink in the wrong place rather than as a decode error.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn every_tile_matches_its_lossless_sibling() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(300, 220, 128);
    let src = rgb_drawing(300, 220);

    let jpeg_root = jpeg_tree(&dir.path().join("grid"), &plan, &src, [255, 255, 255]);
    let png_root = dir.path().join("png");
    let png_sink = FsSink::new(&png_root, plan.clone()).with_format(TileFormat::Png);
    EngineBuilder::new(&src, plan.clone(), png_sink)
        .run()
        .expect("a png pyramid generates");

    let mut seen = 0;
    let mut worst_seen = 0f64;
    let mut worst_rmse = 0f64;
    for (level, lp) in plan.levels.iter().enumerate() {
        for row in 0..lp.rows {
            for col in 0..lp.cols {
                let coord = TileCoord::new(
                    u32::try_from(level).expect("a level index fits in u32"),
                    col,
                    row,
                );
                let rel_jpeg = plan
                    .tile_path(coord, "jpeg")
                    .expect("the coord is inside the plan");
                let rel_png = plan
                    .tile_path(coord, "png")
                    .expect("the coord is inside the plan");
                let bytes = std::fs::read(jpeg_root.join(&rel_jpeg)).expect("the tile is on disk");
                let got = decode_bytes(&bytes)
                    .unwrap_or_else(|e| panic!("{rel_jpeg} does not decode: {e}"));
                let want = decode_bytes(&std::fs::read(png_root.join(&rel_png)).expect("png tile"))
                    .expect("the png sibling decodes");

                assert_eq!(
                    (got.width(), got.height()),
                    (want.width(), want.height()),
                    "{rel_jpeg} is {}x{} and its lossless sibling is {}x{}",
                    got.width(),
                    got.height(),
                    want.width(),
                    want.height()
                );

                let (rmse, worst) = luma_error(want.data(), got.data());
                worst_rmse = worst_rmse.max(rmse);
                worst_seen = worst_seen.max(worst);
                assert!(
                    rmse < 6.0,
                    "{rel_jpeg}: luma RMSE {rmse:.2} against the lossless tile"
                );
                assert!(
                    worst < 70.0,
                    "{rel_jpeg}: one pixel is {worst} off the lossless tile, which \
                     is a block reconstructed from something other than what was \
                     encoded rather than a quantization error"
                );
                seen += 1;
            }
        }
    }
    assert!(seen > 10, "the walk only found {seen} tiles");
    println!("{seen} tiles, worst luma RMSE {worst_rmse:.2}, worst pixel {worst_seen}");
}

/// Only the Ultra HDR lane still builds an `image` JPEG encoder.
///
/// Four call sites used to construct one: `FsSink`'s, the packfile sink's own
/// copy, the object-store sink's own copy, and the raster route in
/// `src/encode.rs`. They are one function now and it is this crate's own
/// encoder, which is what makes the subsampling and the tables reachable at
/// all.
///
/// A fifth copy is not hypothetical. The packfile one drifted while nobody was
/// looking: it reported its failures as `png: {e}`, and the doc comment above
/// it still argued for a duplication that the comment inside it explained had
/// already been undone. This is the guard that makes the next one fail here
/// rather than in a year.
///
/// Two files keep one, and the scan names both rather than filtering by what
/// looks like test code. `src/uhdr.rs` is the real exception: its base image
/// and gain map are two JPEGs inside an ISO container with hand-computed MPF
/// offsets, checked against a libuhdr capture, so moving them is a
/// measurement against that oracle rather than a call-site change.
/// `src/source.rs` builds one in a `#[cfg(test)]` helper to make a JPEG for
/// its own decode cells, which is a fixture and not a route.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn only_the_ultra_hdr_lane_builds_an_image_jpeg_encoder() {
    fn walk(dir: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).expect("src/ is readable") {
            let path = entry.expect("a readable entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let body = std::fs::read_to_string(&path).expect("a source file is readable");
                if body.contains("jpeg::JpegEncoder") {
                    out.push(
                        path.file_name()
                            .expect("a file has a name")
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            }
        }
    }

    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found = Vec::new();
    walk(&src, &mut found);
    found.sort();
    assert_eq!(
        found,
        ["source.rs", "uhdr.rs"],
        "these files construct an `image` JPEG encoder. Every tile and raster \
         route goes through `crate::encode_jpeg` instead, which is the only \
         encoder here that can be told what to subsample and which tables to \
         use (issues #1132, #1133). A third name on this list is either a new \
         fixture helper, which is fine and belongs in the doc above, or a \
         fifth copy of the encoder, which is the thing this cell exists to \
         stop"
    );
}

/// A greyscale tile still encodes, as one component with no chroma to
/// subsample.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_greyscale_tile_is_one_component() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);
    let mut data = vec![0u8; 256 * 256];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i % 251) as u8;
    }
    let source = data.clone();
    let grey = Raster::new(256, 256, PixelFormat::Gray8, data).expect("a grey raster");
    let bytes = tile_bytes(dir.path(), &plan, grey, 85);

    assert_eq!(
        sampling_factors(&bytes),
        vec![(1, 1, 1)],
        "there is no chroma in a greyscale JPEG to subsample"
    );
    let decoded = decode_bytes(&bytes).expect("a greyscale tile decodes");
    assert_eq!((decoded.width(), decoded.height()), (256, 256));
    assert_eq!(decoded.format(), PixelFormat::Gray8);

    // And the pixels, because a component count and a successful decode are
    // both true of a tile full of noise. One component means no chroma to
    // subsample, so the only thing between this and the source is the
    // quantizer, and the bound is tight enough to say so.
    let mut worst = 0f64;
    let mut sum = 0f64;
    for (a, b) in source.iter().zip(decoded.data()) {
        let d = (f64::from(*a) - f64::from(*b)).abs();
        worst = worst.max(d);
        sum += d * d;
    }
    let rmse = (sum / (256.0 * 256.0)).sqrt();
    assert!(rmse < 6.0, "greyscale RMSE is {rmse:.2}");
    assert!(worst < 70.0, "one greyscale pixel moved by {worst}");
}

/// The subsampled tile is still the picture it was handed.
///
/// 4:2:0 throws away three quarters of the chroma samples, so the guard that
/// matters is luma: the ink stays where it was and the paper stays pale. A
/// fix that moved bytes by degrading the tile would land here.
///
/// **Both bounds are tight on purpose.** The measured figures on this fixture
/// are an RMSE of 2.9 and a worst pixel of 19, and the first version of this
/// cell allowed 12 and 160. That slack is what let a real defect through: a
/// bit-writer bug dropped the last symbol of the scan, so the final one or two
/// blocks of every tile decoded as noise, and two bad blocks in 1536 move the
/// RMSE by less than a tenth of the old bound. The worst-pixel bound is the
/// one that can see a handful of corrupt blocks at all, because it does not
/// average them away.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn subsampling_does_not_move_the_ink() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let plan = plan_for(256, 256, 256);
    // Monochrome, because since #1134 the coloured fixture is not subsampled
    // and a cell with this name would have stopped covering what it says.
    let src = mono_rgb_drawing(256, 256);
    let bytes = tile_bytes(dir.path(), &plan, src.clone(), 85);
    let decoded = decode_bytes(&bytes).expect("the tile decodes");

    let luma = |px: &[u8]| {
        (0.299 * f64::from(px[0]) + 0.587 * f64::from(px[1]) + 0.114 * f64::from(px[2])).round()
    };
    let mut worst = 0f64;
    let mut sum = 0f64;
    for (want, got) in src
        .data()
        .as_chunks::<3>()
        .0
        .iter()
        .zip(decoded.data().as_chunks::<3>().0)
    {
        let d = (luma(want) - luma(got)).abs();
        worst = worst.max(d);
        sum += d * d;
    }
    let rmse = (sum / (256.0 * 256.0)).sqrt();
    assert!(
        rmse < 5.0,
        "luma RMSE is {rmse:.2} over the whole tile, which is not lossy coding \
         of line art at quality 85 any more"
    );
    assert!(
        worst < 40.0,
        "one pixel moved by {worst}, so a block is being reconstructed from \
         something other than what was encoded rather than merely quantized"
    );
}
