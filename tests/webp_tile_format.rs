//! WebP as a tile format, and the five sites a new variant does not break
//! (issue #1123).
//!
//! `Raster::encode_webp` has existed since the webp lane landed, `TileType`
//! has answered `"webp"` for its extension since the PMTiles header went in,
//! and the two were never joined up: `TileFormat` had no `Webp` variant, so
//! the mapping in `TileType::try_from_tile_format` was one-way and a
//! `--format webp` was a documentation claim. This file is the acceptance
//! suite for joining them.
//!
//! # Why most of this file is about verify and not about encoding
//!
//! Adding a variant to `TileFormat` breaks seven exhaustive matches, and
//! those are the easy ones: the build stops and somebody has to answer them.
//! Five more sites carry a catch-all and keep compiling while doing the wrong
//! thing, and every one of them is a probe list of tile extensions spelled out
//! by hand:
//!
//! * `stream_verify::CANDIDATE_EXTS`, feeding three call sites;
//! * the same four strings inline in `engine::raster_verify`;
//! * the same four again in `engine::coord_for_manifest_rel`;
//! * and its twin in `stream_verify`.
//!
//! A sink that does not pin its format (any transparent wrapper: the default
//! `TileSink::content_format` returns `None`) falls back to that list, so a
//! WebP tree verified through one finds **no tiles at all** and reports the
//! pyramid missing. That is the failure this file exists to catch, because a
//! naive implementation that answers the seven forced matches passes
//! everything else here.
//!
//! The coordinate cells are the same bug wearing a different coat. #139 was
//! filed because a manifest key resolved to a fabricated `TileCoord(0, 0, 0)`
//! or to a col/row-transposed one, and a `.webp` key that the plan scan cannot
//! match re-opens it: the structural fallback parses `2/1/3.webp` as
//! `{level}/{col}/{row}` and a Google-layout tree stores `{level}/{row}/{col}`.
//! So those assertions check *which* coordinate comes back, not that one does.
//! Those two functions are private, so their cells live as unit tests beside
//! them in `src/engine.rs` and `src/stream_verify.rs`.

use std::path::{Path, PathBuf};

use libviprs::engine::{EngineConfig, raster_verify};
use libviprs::manifest::GenerationSettings;
use libviprs::observe::NoopObserver;
use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner};
use libviprs::pmtiles::writer::content_hash;
use libviprs::pmtiles::{Compression, LibviprsMetadata, Metadata, TileType, Writer, WriterOptions};
use libviprs::pyramid_reader::{PmTilesPyramidReader, PyramidReadError, PyramidReader};
use libviprs::sink::{SinkError, Tile, TileFormat, TileSink};
use libviprs::sink_pmtiles::PmTilesSink;
use libviprs::streaming::RasterStripSource;
use libviprs::stream_verify::verify_from_strip_source;
use libviprs::{EngineBuilder, FsSink, PixelFormat, Raster};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A raster where no two tiles can come out the same.
///
/// Uniform tiles would let a deduplicating archive collapse to one payload and
/// would let a blank-tile strategy write a one-byte placeholder, and both
/// would make the pixel-exact comparison below compare something other than an
/// encoded tile.
fn gradient(w: u32, h: u32) -> Raster {
    let mut data = vec![0u8; w as usize * h as usize * 3];
    for y in 0..h {
        for x in 0..w {
            let off = (y as usize * w as usize + x as usize) * 3;
            data[off] = (x % 251) as u8;
            data[off + 1] = (y % 241) as u8;
            data[off + 2] = ((x * 7 + y * 13) % 239) as u8;
        }
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).expect("a gradient raster is well formed")
}

fn plan(w: u32, h: u32, tile: u32, layout: Layout) -> PyramidPlan {
    PyramidPlanner::new(w, h, tile, 0, layout)
        .expect("a square plan is valid")
        .plan()
}

/// A sink that writes through to an [`FsSink`] and refuses to say what format
/// it writes.
///
/// This is not a contrivance: it is what every transparent wrapper in the wild
/// looks like, because `TileSink::content_format` defaults to `None` and a
/// tee / retry / recording wrapper has no reason to override it. Verify then
/// falls back to probing every extension it knows, and the whole point of this
/// file is that "every extension it knows" is a list that used to be written
/// out by hand in four places.
struct FormatBlind(FsSink);

impl TileSink for FormatBlind {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        self.0.write_tile(tile)
    }

    fn finish(&self) -> Result<(), SinkError> {
        self.0.finish()
    }

    fn checkpoint_root(&self) -> Option<&Path> {
        self.0.checkpoint_root()
    }

    // `content_format` is deliberately NOT overridden. The default is `None`,
    // which is the case under test.
}

/// Run `src` into a WebP directory tree and hand back the root.
fn webp_tree(dir: &Path, plan_: &PyramidPlan, src: &Raster) -> PathBuf {
    let root = dir.join("webp-tiles");
    let sink = FsSink::new(&root, plan_.clone()).with_format(TileFormat::Webp);
    EngineBuilder::new(src, plan_.clone(), sink)
        .run()
        .expect("a webp pyramid generates");
    root
}

// ---------------------------------------------------------------------------
// The variant itself
// ---------------------------------------------------------------------------

/// The extension, and the probe set that is derived from the enum rather than
/// typed out a fifth time.
#[test]
fn the_probe_set_is_derived_from_the_enum() {
    assert_eq!(TileFormat::Webp.extension(), "webp");

    // Every variant contributes its own spellings, and JPEG contributes two
    // because a tile written by another tool can be `.jpg`.
    assert_eq!(TileFormat::Webp.extensions(), &["webp"]);
    assert_eq!(TileFormat::Jpeg { quality: 80 }.extensions(), &["jpeg", "jpg"]);

    let probe = TileFormat::candidate_extensions();
    assert_eq!(
        probe,
        vec!["raw", "png", "jpeg", "jpg", "webp"],
        "the fallback probe set is the union of every variant's spellings, in \
         the order the hand-written copies used, with webp appended"
    );

    // The property, rather than the pinned list: nothing a variant can be
    // stored under is missing from the blind probe.
    for fmt in TileFormat::ALL {
        for ext in fmt.extensions() {
            assert!(
                probe.contains(ext),
                "{fmt:?} can be stored as .{ext} and a blind probe would never look for it"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The silent sites
// ---------------------------------------------------------------------------

/// A WebP tree verified through a format-blind sink must find its tiles.
///
/// This is the cell a naive implementation fails. Answer the seven exhaustive
/// matches and nothing else, and `stream_verify`'s `CANDIDATE_EXTS` still says
/// `["raw", "png", "jpeg", "jpg"]`, so `find_tile_on_disk` misses every
/// `.webp` file and the run fails with "missing tile" on the first coordinate
/// it looks at. The pyramid is entirely present; the probe list is what is
/// missing.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_webp_tree_verifies_through_a_format_blind_sink_streaming() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let p = plan(512, 512, 256, Layout::Xyz);
    let src = gradient(p.image_width, p.image_height);
    let root = webp_tree(dir.path(), &p, &src);

    // Positive control: the tree really is there, as `.webp` files, before
    // anything asserts that a verify found it. A verify over an empty tree
    // would agree with a working one.
    let written = std::fs::read_dir(&root).expect("the tree exists").count();
    assert!(written > 0, "the webp run wrote nothing to {root:?}");

    let blind = FormatBlind(FsSink::new(&root, p.clone()).with_format(TileFormat::Webp));
    assert_eq!(
        blind.content_format(),
        None,
        "this cell is only about the None branch; a wrapper that pinned the \
         format would take the other one"
    );

    let strip = RasterStripSource::new(&src);
    let res = verify_from_strip_source(
        &strip,
        &p,
        &blind,
        &EngineConfig::default(),
        &NoopObserver,
    )
    .expect("a webp tree verifies through a format-blind sink");
    assert_eq!(res.tiles_produced, 0, "verify must not write tiles");
    assert_eq!(res.levels_processed, p.levels.len() as u32);
}

/// The same tree, the same wrapper, the monolithic path.
///
/// `engine::raster_verify` carries its own inline copy of the four strings, so
/// the two paths share the bug and not a line. Fixing one and leaving the
/// other is exactly the shape this pair exists to catch.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_webp_tree_verifies_through_a_format_blind_sink_monolithic() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let p = plan(512, 512, 256, Layout::Xyz);
    let src = gradient(p.image_width, p.image_height);
    let root = webp_tree(dir.path(), &p, &src);

    let blind = FormatBlind(FsSink::new(&root, p.clone()).with_format(TileFormat::Webp));
    let res = raster_verify(&src, &p, &blind, &EngineConfig::default(), &NoopObserver)
        .expect("a webp tree verifies through raster_verify too");
    assert_eq!(res.tiles_produced, 0, "verify must not write tiles");
    assert_eq!(res.levels_processed, p.levels.len() as u32);
}

/// The negative control for the pair above.
///
/// Both cells would also pass if verify had quietly stopped checking anything,
/// so one of them has to fail when a tile really is gone. Delete one `.webp`
/// file and the same call must refuse.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_webp_tree_with_a_hole_still_fails_through_a_format_blind_sink() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let p = plan(512, 512, 256, Layout::Xyz);
    let src = gradient(p.image_width, p.image_height);
    let root = webp_tree(dir.path(), &p, &src);

    let victim = p.tile_coords().next().expect("the plan has a tile");
    let rel = p
        .tile_path(victim, "webp")
        .expect("the victim is inside the plan");
    std::fs::remove_file(root.join(&rel)).expect("the victim tile was on disk as .webp");

    let blind = FormatBlind(FsSink::new(&root, p.clone()).with_format(TileFormat::Webp));
    let strip = RasterStripSource::new(&src);
    verify_from_strip_source(
        &strip,
        &p,
        &blind,
        &EngineConfig::default(),
        &NoopObserver,
    )
    .expect_err("a missing webp tile must not verify");
}

// ---------------------------------------------------------------------------
// PMTiles, end to end
// ---------------------------------------------------------------------------

/// Write a WebP archive, reopen it, and check the bytes are the pixels.
///
/// Three claims, and the third is the one with teeth:
///
/// * the header records `TileType::Webp`, which is the wire byte `0x04`;
/// * `describe()` answers `Some(TileFormat::Webp)`;
/// * the stored bytes decode back to the pixels that went in, **exactly**.
///
/// Exact rather than tolerant on purpose. `webp::Compression` has one variant
/// and it is lossless, so an exact comparison is the assertion that would go
/// red if a `quality` knob were ever added and thrown away by the encoder. A
/// tolerant comparison passes for every quality value, which is precisely why
/// `TileFormat::Webp` carries no quality field.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_pmtiles_webp_archive_round_trips_pixel_exact() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let p = plan(512, 512, 256, Layout::Xyz);
    let src = gradient(p.image_width, p.image_height);

    let archive = dir.path().join("pyramid.pmtiles");
    let sink = PmTilesSink::builder(&archive)
        .plan(p.clone())
        .tile_format(TileFormat::Webp)
        .build()
        .expect("a webp archive sink builds");
    EngineBuilder::new(&src, p.clone(), sink)
        .run()
        .expect("the webp archive run succeeds");

    // The same run into a `Raw` tree, which is the pixel oracle: those files
    // are the exact bytes the engine handed the encoder, so comparing the
    // decoded archive tile against them isolates the encoder from every
    // resampling decision upstream of it.
    let raw_root = dir.path().join("raw-tiles");
    EngineBuilder::new(
        &src,
        p.clone(),
        FsSink::new(&raw_root, p.clone()).with_format(TileFormat::Raw),
    )
    .run()
    .expect("the raw run succeeds");

    let reader = PmTilesPyramidReader::try_open(&archive).expect("the archive reopens");
    assert_eq!(
        reader.reader().tile_format(),
        TileType::Webp,
        "the header must record webp"
    );
    assert_eq!(
        reader.reader().tile_format().to_byte(),
        0x04,
        "webp is wire byte 0x04"
    );
    assert_eq!(
        reader
            .describe()
            .expect("the archive describes itself")
            .format,
        Some(TileFormat::Webp)
    );

    let mut compared = 0usize;
    for coord in p.tile_coords() {
        let stored = reader
            .tile(coord)
            .expect("reading a planned tile is not an error")
            .unwrap_or_else(|| panic!("the archive is missing {coord:?}"));
        let decoded = libviprs::decode_webp(&stored).expect("the stored bytes are webp");

        let rel = p
            .tile_path(coord, "raw")
            .expect("the coord is inside the plan");
        let want = std::fs::read(raw_root.join(&rel)).expect("the raw sibling tile is on disk");

        assert_eq!(
            decoded.data(),
            want.as_slice(),
            "{coord:?} did not survive the webp round trip byte for byte"
        );
        compared += 1;
    }
    // A comparison over an empty set agrees perfectly.
    assert_eq!(
        compared,
        p.tile_coords().count(),
        "every planned coordinate must have been compared"
    );
    assert!(compared > 0, "the plan addressed no tiles");
}

/// A foreign WebP archive says what it is.
///
/// `TileType::Webp` carries no parameters, so unlike JPEG (whose quality lives
/// only in the `vnd.libviprs` namespace) there is nothing to be dishonest
/// about: an archive written by another tool with tile type `0x04` and no
/// libviprs namespace at all is a WebP pyramid and `describe()` can say so.
/// Before this change it answered `format: None`, indistinguishable from "I
/// have no idea what this is".
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_foreign_webp_archive_reports_its_format() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let archive = dir.path().join("foreign.pmtiles");
    write_foreign_archive(&archive, TileType::Webp, Metadata::default());

    let reader = PmTilesPyramidReader::try_open(&archive).expect("the archive reopens");
    let meta = reader
        .reader()
        .metadata()
        .expect("a default Metadata parses");
    assert!(
        meta.vnd_libviprs.is_none(),
        "this cell is about an archive with no libviprs namespace"
    );
    assert_eq!(
        reader
            .describe()
            .expect("a foreign archive still describes itself")
            .format,
        Some(TileFormat::Webp)
    );
}

// ---------------------------------------------------------------------------
// Forward compatibility
// ---------------------------------------------------------------------------

/// An archive from a newer libviprs is named, not silently blanked.
///
/// This is the decision #1123 makes and it is worth restating, because the
/// behaviour it replaces looks harmless. `Metadata` parses the whole object or
/// none of it, so one unknown `format` variant fails the parse; `generation()`
/// swallowed that with `.ok()?`, and `describe()` then answered `tile_size:
/// None, layout: None, format: None` for an archive that records all three.
/// Worse, `format: None` is *already* the right answer for a foreign
/// go-pmtiles archive, so a user could not tell "made by another tool" from
/// "made by libviprs and I am too old for it", and those want opposite
/// reactions.
///
/// The fixture uses `avif`, a variant this build will not know either, so the
/// cell runs today and goes red the day somebody relaxes `.ok()?` without
/// deciding what should happen instead.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn an_archive_from_a_newer_libviprs_is_named_rather_than_blanked() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let archive = dir.path().join("from-the-future.pmtiles");

    // What a libviprs that knows AVIF would write. `generation` goes through
    // `extra` because this build cannot spell `{"kind":"avif"}` in the typed
    // field: the named field is skipped when `None`, and the flattened map
    // supplies the key, so the bytes on disk are exactly the newer build's.
    let mut vnd = LibviprsMetadata::default();
    vnd.libviprs_version = "9.9.9".to_string();
    vnd.extra.insert(
        "generation".to_string(),
        serde_json::json!({
            "tile_size": 256,
            "overlap": 0,
            "layout": "xyz",
            "format": {"kind": "avif"},
            "concurrency": 4,
            "background_rgb": [0, 0, 0],
            "blank_strategy": {"kind": "emit"},
        }),
    );
    let mut meta = Metadata::default();
    meta.name = Some("written by a newer libviprs".to_string());
    meta.vnd_libviprs = Some(vnd);

    write_foreign_archive(&archive, TileType::Avif, meta);

    // The premise: this build genuinely cannot parse it.
    let reader = PmTilesPyramidReader::try_open(&archive).expect("the archive reopens");
    reader
        .reader()
        .metadata()
        .expect_err("this build must not be able to parse an avif format variant");

    let err = reader
        .describe()
        .expect_err("an unreadable libviprs namespace is an answer, not a blank");
    match &err {
        PyramidReadError::MetadataFromANewerLibviprs {
            libviprs_version, ..
        } => assert_eq!(
            libviprs_version, "9.9.9",
            "the error has to name the version that wrote the archive, \
             because that is the whole actionable content of it"
        ),
        other => panic!("expected MetadataFromANewerLibviprs, got {other:?}"),
    }
    assert!(
        err.to_string().contains("9.9.9"),
        "the rendered message names the writing version too, got: {err}"
    );
}

/// The control: no `vnd.libviprs` key means `None` stays `None`.
///
/// Without this the change above would be a regression for every foreign
/// archive, because an unparseable metadata object is not evidence that
/// libviprs wrote it. The fixture is a `type` field holding a number, which
/// the v3 spec says is a string, so the parse fails for a reason that has
/// nothing to do with libviprs at all.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_foreign_archive_with_unparseable_metadata_still_describes() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let archive = dir.path().join("foreign-broken.pmtiles");

    let mut meta = Metadata::default();
    meta.extra
        .insert("type".to_string(), serde_json::json!(5));
    write_foreign_archive(&archive, TileType::Png, meta);

    let reader = PmTilesPyramidReader::try_open(&archive).expect("the archive reopens");
    reader
        .reader()
        .metadata()
        .expect_err("a numeric `type` is not parseable as the spec's string");

    let described = reader
        .describe()
        .expect("a foreign archive that libviprs never touched still describes itself");
    assert_eq!(
        described.format,
        Some(TileFormat::Png),
        "the header's tile type is still readable and still the answer"
    );
    assert_eq!(described.tile_size, None);
    assert_eq!(described.layout, None);
}

// ---------------------------------------------------------------------------
// The manifest wire shape
// ---------------------------------------------------------------------------

/// `{"kind":"webp"}` on the wire, read back from outside the crate.
///
/// The pin inside `src/pmtiles/metadata.rs` uses a `Jpeg` fixture, so adding a
/// variant does not move it: it stays green whatever `Webp` serialises as, or
/// whether it serialises at all. That file gains its own `Webp` pin for the
/// serialise direction (a `GenerationSettings` literal needs to be inside the
/// crate, the struct being `#[non_exhaustive]`); this is the parse direction,
/// which is the one an *older* archive exercises and the one a mis-spelled
/// `Repr` arm would break without the build noticing.
#[test]
fn the_manifest_wire_shape_for_webp_parses_back() {
    let wire = r#"{"tile_size":256,"overlap":0,"layout":"xyz","format":{"kind":"webp"},"#.to_string()
        + r#""concurrency":4,"background_rgb":[0,0,0],"blank_strategy":{"kind":"emit"}}"#;
    let settings: GenerationSettings =
        serde_json::from_str(&wire).expect("the webp wire shape parses");
    assert_eq!(settings.format, TileFormat::Webp);

    // And it is `webp` with nothing else in it. A `quality` field would have
    // to be spelled here, and `TileFormat::Webp` deliberately has none: the
    // encoder is lossless and has no knob, so a quality an encoder throws away
    // would invert the contract and be a semver time bomb the day a lossy
    // encoder lands.
    let back = serde_json::to_string(&settings).expect("it serialises back");
    assert!(
        back.contains(r#""format":{"kind":"webp"}"#),
        "expected a bare webp tag, got: {back}"
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write a one-tile archive with a tile type and metadata of our choosing.
///
/// `PmTilesSink` always stamps the `vnd.libviprs` namespace, so the archives
/// this file needs (one with no namespace, one with a namespace from a build
/// that does not exist) have to be written through the raw `Writer`.
fn write_foreign_archive(path: &Path, tile_type: TileType, metadata: Metadata) {
    let scratch = path.parent().expect("the archive has a parent directory");
    let options = WriterOptions::default()
        .with_tile_type(tile_type)
        .with_tile_compression(Compression::None)
        .with_metadata(metadata);
    let payload = b"not really an image".to_vec();
    let mut writer = Writer::try_new(
        std::fs::File::create(path).expect("the archive file is creatable"),
        scratch,
        options,
    )
    .expect("the writer opens its scratch files");
    writer
        .add_tile(0, 0, 0, &payload, content_hash(&payload))
        .expect("one tile goes in");
    writer.finish().expect("the archive finalises");
}
