//! `PmTilesSink` driven by the real engine, and `DirectoryPyramidReader`
//! (issue #990).
//!
//! # Nothing here reads the archive with our reader
//!
//! F1.2 (#988) owns the indexed reader and it is a sibling branch, so every
//! archive this file produces is opened with the **format module** instead:
//! [`Header::try_decode`], [`Compression::decompress`] and
//! [`deserialize_entries`], which are #987's primitives and the same ones the
//! writer's own suite uses. That is deliberate on two counts. It keeps this
//! file green on a branch that does not carry the reader, and it keeps the
//! sink from being proved correct by a reader that could share its mistake.
//! The cross-backend equivalence through both [`PyramidReader`] impls lives in
//! `tests/pmtiles_pyramid_reader.rs`, which does need the reader.
//!
//! The one thing a self-consistent walk still cannot settle is whether
//! `TileCoord { level, col, row }` maps onto `(z, x, y)` the way the rest of
//! the world does. A sink, a walk and a reader that all transpose `col` and
//! `row` agree with each other perfectly. So the mapping is pinned against
//! go-pmtiles' own tile ids, loaded at run time from
//! `tests/fixtures/pmtiles/vectors/tileid.json`, by
//! [`the_coordinate_mapping_matches_the_oracle_tile_ids`].
//!
//! # The two traps this file exists to catch
//!
//! * **Archive dedupe has to be unconditional.** [`DedupeStrategy`] defaults
//!   to `None`, and under `None` `DedupeIndex::record` answers `WriteNew` for
//!   every call by design. A sink that keys the archive's payload table off
//!   that decision stores every duplicate on the default settings, and the
//!   "duplicates collapse into one payload" criterion would then only hold for
//!   a caller who had opted into `--dedupe-all`.
//!   [`duplicate_tiles_collapse_to_one_payload_on_the_default_strategy`] runs
//!   on the default and asserts the archive's own `tile_contents_count`.
//! * **`checkpoint_root()` must not name the archive's directory.** Whatever
//!   it returns is fed to `wipe_directory` on every `Overwrite`, whose
//!   ownership guard then refuses any directory holding an unrelated file.
//!   [`overwrite_leaves_the_unrelated_file_beside_the_archive_alone`] puts a
//!   file there and runs Overwrite.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use libviprs::engine::EngineError;
use libviprs::engine::{BlankTileStrategy, EngineConfig};
use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner, TileCoord};
use libviprs::pmtiles::directory::deserialize_entries;
use libviprs::pmtiles::header::HEADER_BYTES;
use libviprs::pmtiles::tileid::zxy_to_tileid;
use libviprs::pmtiles::{
    Compression, Entry, Header, Layout as ArchiveLayout, Metadata, TileType, WriterOptions,
};
use libviprs::pyramid_reader::{DirectoryPyramidReader, PyramidReader};
use libviprs::resume::{ResumeMode, ResumePolicy};
use libviprs::sink::{EmissionOrder, SinkError, Tile, TileFormat, TileSink};
use libviprs::sink_pmtiles::{PmTilesSink, tile_coord_to_zxy};
use libviprs::{EngineBuilder, EngineKind, FsSink, PixelFormat, Raster};

#[path = "common/pmtiles_oracle.rs"]
mod oracle;

/// A generous ceiling for the gunzips below. Nothing a unit-scale run produces
/// comes near it; the point of naming one at all is that PMTiles v3 stores no
/// uncompressed length anywhere, so the output has to be capped rather than
/// pre-sized.
const DECOMPRESS_CEILING: usize = 1 << 22;

// ---------------------------------------------------------------------------
// Sources and plans
// ---------------------------------------------------------------------------

/// A raster where no two tiles can come out the same.
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

/// A raster where every tile comes out the same, which is what a drawing
/// pyramid mostly is.
fn uniform(w: u32, h: u32) -> Raster {
    Raster::new(
        w,
        h,
        PixelFormat::Rgb8,
        vec![0xf0; w as usize * h as usize * 3],
    )
    .expect("a uniform raster is well formed")
}

fn plan_for(w: u32, h: u32, tile: u32, layout: Layout) -> PyramidPlan {
    PyramidPlanner::new(w, h, tile, 0, layout)
        .expect("a square power-of-two plan is valid")
        .plan()
}

// ---------------------------------------------------------------------------
// Opening an archive with nothing but the format module
// ---------------------------------------------------------------------------

/// Every addressed tile of an archive, with the header that described it.
struct Walked {
    header: Header,
    /// TileID to the stored payload, runs expanded and leaves followed.
    tiles: BTreeMap<u64, Vec<u8>>,
}

impl Walked {
    /// The payload stored for a plan coordinate.
    ///
    /// The mapping is spelled out here rather than routed through
    /// `tile_coord_to_zxy`, which is the code under test. A walk that asks the
    /// sink where it put a tile agrees with the sink whatever it answers, so a
    /// transposition of `col` and `row` survived every comparison in this file
    /// when it was written the other way. Measured: the mutation reddened only
    /// the oracle test until this changed.
    fn tile(&self, coord: TileCoord) -> Option<&Vec<u8>> {
        let z = u8::try_from(coord.level).ok()?;
        self.tiles
            .get(&zxy_to_tileid(z, coord.col, coord.row).ok()?)
    }
}

fn section(bytes: &[u8], header: &Header, offset: u64, length: u64) -> Vec<u8> {
    let start = usize::try_from(offset).expect("a unit-scale archive fits in a usize");
    let end = start + usize::try_from(length).expect("a unit-scale section fits in a usize");
    header
        .internal_compression
        .decompress(&bytes[start..end], DECOMPRESS_CEILING)
        .expect("the writer compresses its own sections with something it can read back")
}

fn push_run(tiles: &mut BTreeMap<u64, Vec<u8>>, bytes: &[u8], header: &Header, entry: Entry) {
    let start = usize::try_from(header.tile_data_offset + entry.offset)
        .expect("a unit-scale archive fits in a usize");
    let blob = bytes[start..start + entry.length as usize].to_vec();
    for i in 0..u64::from(entry.run_length) {
        let previous = tiles.insert(entry.tile_id + i, blob.clone());
        assert!(
            previous.is_none(),
            "tile id {} was addressed twice by one archive",
            entry.tile_id + i
        );
    }
}

/// Read an archive with the format primitives and nothing else.
fn walk(path: &Path) -> Walked {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("{} unreadable: {e}", path.display()));
    let header = Header::try_decode(&bytes[..HEADER_BYTES]).expect("the sink wrote a v3 header");

    let root = section(&bytes, &header, header.root_offset, header.root_length);
    let mut tiles = BTreeMap::new();
    for entry in deserialize_entries(&root).expect("the root directory parses") {
        if entry.is_leaf() {
            let leaf = section(
                &bytes,
                &header,
                header.leaf_directories_offset + entry.offset,
                u64::from(entry.length),
            );
            for inner in deserialize_entries(&leaf).expect("a leaf directory parses") {
                assert!(!inner.is_leaf(), "this writer emits no leaves under leaves");
                push_run(&mut tiles, &bytes, &header, inner);
            }
        } else {
            push_run(&mut tiles, &bytes, &header, entry);
        }
    }
    Walked { header, tiles }
}

/// The metadata object the archive stores, decompressed and parsed.
fn walk_metadata(path: &Path) -> Metadata {
    let bytes = std::fs::read(path).expect("the archive is readable");
    let header = Header::try_decode(&bytes[..HEADER_BYTES]).expect("the sink wrote a v3 header");
    let raw = section(
        &bytes,
        &header,
        header.metadata_offset,
        header.metadata_length,
    );
    Metadata::try_from_json(&raw).expect("the sink wrote parseable metadata")
}

// ---------------------------------------------------------------------------
// Running the engine into a sink
// ---------------------------------------------------------------------------

/// Run one pyramid into a fresh `.pmtiles` under `dir` and return its path.
fn run_into_archive(src: &Raster, plan: &PyramidPlan, dir: &Path) -> PathBuf {
    let out = dir.join("pyramid.pmtiles");
    let sink = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .build()
        .expect("a PNG XYZ sink builds");
    EngineBuilder::new(src, plan.clone(), sink)
        .run()
        .expect("a unit-scale run into a PMTiles archive succeeds");
    out
}

/// Run the same pyramid into a loose-file tree and return its root.
fn run_into_directory(src: &Raster, plan: &PyramidPlan, dir: &Path) -> PathBuf {
    let root = dir.join("tree");
    let sink = FsSink::new(&root, plan.clone());
    EngineBuilder::new(src, plan.clone(), sink)
        .run()
        .expect("a unit-scale run into a directory succeeds");
    root
}

// ---------------------------------------------------------------------------
// The headline criterion
// ---------------------------------------------------------------------------

/// An engine run into a `PmTilesSink` produces an archive that holds every
/// planned coordinate, with the same bytes the loose-file sink wrote.
///
/// The positive control is the part that matters. An equivalence assertion
/// over two backends that both answer nothing for every coordinate passes
/// while proving nothing, so the planned coordinate set is asserted non-empty
/// and the archive's addressed count is asserted to equal it before a single
/// payload is compared.
#[test]
#[cfg_attr(miri, ignore)]
fn an_engine_run_produces_an_archive_holding_every_planned_tile() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Not square, deliberately: a square grid maps onto itself under a
    // col/row transposition, so half the evidence would be missing.
    let plan = plan_for(1024, 768, 256, Layout::Xyz);
    let src = gradient(1024, 768);

    let archive = run_into_archive(&src, &plan, dir.path());
    let tree = run_into_directory(&src, &plan, dir.path());

    let coords: Vec<TileCoord> = plan.tile_coords().collect();
    assert!(
        coords.len() >= 4,
        "the positive control: a plan with fewer than four tiles cannot tell a \
         working sink from one that writes nothing, got {}",
        coords.len()
    );

    let walked = walk(&archive);
    assert_eq!(
        walked.tiles.len(),
        coords.len(),
        "the archive addresses a different number of tiles than the plan has"
    );
    assert_eq!(
        walked.header.addressed_tiles_count,
        coords.len() as u64,
        "the header's addressed count disagrees with the plan"
    );

    let fs = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the directory the run just filled opens");
    for coord in &coords {
        let from_archive = walked
            .tile(*coord)
            .unwrap_or_else(|| panic!("{coord:?} is missing from the archive"));
        let from_tree = fs
            .tile(*coord)
            .expect("the directory reader answers")
            .unwrap_or_else(|| panic!("{coord:?} is missing from the tree"));
        assert_eq!(
            from_archive, &from_tree,
            "{coord:?} differs between the archive and the loose-file tree"
        );
    }
}

/// `TileCoord { level, col, row }` addresses the `(z, x, y)` go-pmtiles
/// addresses, and the tile id that comes out is the one it computes.
///
/// This is the only assertion in the file that does not come from this crate.
/// Twelve rows from zoom 9 to 15, off every quadrant boundary, four of them
/// arranged as swapped pairs, so a mapping that transposes `col` and `row` has
/// nowhere to hide: `(13, 5107, 2884)` is 79053962 and `(13, 2884, 5107)` is
/// 53847284.
#[test]
#[cfg_attr(miri, ignore)]
fn the_coordinate_mapping_matches_the_oracle_tile_ids() {
    let vectors = oracle::tileid_vectors();
    let rows = oracle::tile_id_rows(&vectors, "convention_discriminators");
    assert_eq!(
        rows.len(),
        12,
        "the discriminating set is twelve rows; a parse that found fewer would \
         pass every loop below vacuously"
    );

    for row in &rows {
        let coord = TileCoord {
            level: u32::from(row.z),
            col: row.x,
            row: row.y,
        };
        let (z, x, y) = tile_coord_to_zxy(coord).expect("a discriminator is addressable");
        assert_eq!(
            (z, x, y),
            (row.z, row.x, row.y),
            "level/col/row must map straight onto z/x/y"
        );
        assert_eq!(
            zxy_to_tileid(z, x, y).expect("a discriminator has an id"),
            row.tile_id,
            "({z}, {x}, {y}) must be tile id {}",
            row.tile_id
        );
    }
}

// ---------------------------------------------------------------------------
// Dedupe, which has to be unconditional
// ---------------------------------------------------------------------------

/// Identical tiles at different coordinates collapse into **one** stored
/// payload, on the default dedupe strategy.
///
/// `DedupeStrategy::None` is the default and it makes `DedupeIndex::record`
/// answer `WriteNew` for every tile on purpose, so a sink that drives the
/// archive's payload table off that decision stores every duplicate here and
/// passes the issue's dedupe criterion only when a caller has turned dedupe
/// on. The assertion is the archive's own `tile_contents_count`, and the
/// control beside it is that more than one tile was addressed: one payload out
/// of one tile proves nothing.
///
/// The tiles are handed to the sink directly rather than produced by a run,
/// because a real pyramid's levels have different pixel geometries and so
/// cannot all be the same bytes. The engine-driven half is
/// [`an_engine_run_collapses_its_identical_tiles`].
#[test]
#[cfg_attr(miri, ignore)]
fn duplicate_tiles_collapse_to_one_payload_on_the_default_strategy() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(1024, 1024, 256, Layout::Xyz);
    let out = dir.path().join("dupes.pmtiles");

    assert_eq!(
        EngineConfig::default().dedupe_strategy,
        None,
        "this test is about the DEFAULT strategy; if the default changed the \
         trap it guards moved with it"
    );

    let sink = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .build()
        .expect("the sink builds");
    let top = plan
        .levels
        .iter()
        .map(|level| level.level)
        .max()
        .expect("a plan has levels");
    let mut addressed = 0u64;
    for coord in plan.tile_coords().filter(|c| c.level == top) {
        sink.write_tile(&Tile {
            coord,
            raster: uniform(256, 256),
            blank: false,
        })
        .expect("the sink takes a tile");
        addressed += 1;
    }
    sink.finish().expect("the archive publishes");

    assert!(
        addressed > 1,
        "the negative control: one payload out of one tile proves nothing, got \
         {addressed} tiles"
    );

    let walked = walk(&out);
    assert_eq!(walked.header.addressed_tiles_count, addressed);
    assert_eq!(
        walked.header.tile_contents_count, 1,
        "{addressed} identical tiles are one stored payload"
    );
    let distinct: BTreeSet<&Vec<u8>> = walked.tiles.values().collect();
    assert_eq!(
        distinct.len(),
        1,
        "and every id resolves to that one payload"
    );
}

/// A real run over a uniform source stores far fewer payloads than it
/// addresses, on the default strategy.
///
/// The levels of a pyramid have different pixel geometries, so not every tile
/// can be the same bytes; what must hold is that the ones that are the same
/// are stored once. The full-resolution level alone is sixteen identical
/// tiles, so the biggest group is the assertion, and `tile_contents_count`
/// agreeing with the set of distinct payloads is what ties the header's claim
/// to the bytes.
#[test]
#[cfg_attr(miri, ignore)]
fn an_engine_run_collapses_its_identical_tiles() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(1024, 1024, 256, Layout::Xyz);
    let archive = run_into_archive(&uniform(1024, 1024), &plan, dir.path());
    let walked = walk(&archive);

    let mut groups: BTreeMap<&Vec<u8>, usize> = BTreeMap::new();
    for blob in walked.tiles.values() {
        *groups.entry(blob).or_default() += 1;
    }
    assert_eq!(
        walked.header.tile_contents_count as usize,
        groups.len(),
        "the header's payload count must be the number of distinct payloads \
         actually stored"
    );
    assert!(
        groups.len() < walked.tiles.len(),
        "a uniform pyramid that stores one payload per tile has deduped nothing"
    );
    let biggest = groups.values().copied().max().unwrap_or_default();
    assert!(
        biggest >= 16,
        "the sixteen tiles of the full-resolution level are the same bytes and \
         must share one payload, biggest group was {biggest}"
    );
}

/// A blank tile is stored, not skipped, even under
/// `BlankTileStrategy::Placeholder`.
///
/// `PackfileSink::write_tile` returns early on `tile.blank`, and copying that
/// into a PMTiles sink leaves a hole: the format has no placeholder concept, a
/// missing tile id is a missing tile, and the whole point of keying the
/// payload table on content is that ten thousand blanks cost one payload. The
/// 1-byte `BLANK_TILE_MARKER` is not written either, because a one-byte blob
/// is not a decodable tile.
#[test]
#[cfg_attr(miri, ignore)]
fn blank_tiles_are_stored_as_real_tiles_rather_than_skipped_or_marked() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let src = uniform(512, 512);
    let out = dir.path().join("blank.pmtiles");

    let sink = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .build()
        .expect("the sink builds");
    EngineBuilder::new(&src, plan.clone(), sink)
        .with_blank_strategy(BlankTileStrategy::Placeholder)
        .run()
        .expect("a placeholder run into a PMTiles archive succeeds");

    let walked = walk(&out);
    let coords: Vec<TileCoord> = plan.tile_coords().collect();
    assert_eq!(
        walked.tiles.len(),
        coords.len(),
        "a blank tile is still a tile and still needs an id"
    );
    for coord in &coords {
        let blob = walked
            .tile(*coord)
            .unwrap_or_else(|| panic!("{coord:?} was skipped"));
        assert!(
            blob.starts_with(&[0x89, b'P', b'N', b'G']),
            "{coord:?} stored {} bytes that are not a PNG, so the placeholder \
             marker reached the archive",
            blob.len()
        );
    }
}

// ---------------------------------------------------------------------------
// The engine hooks
// ---------------------------------------------------------------------------

/// The sink reports the format it encodes, and the resume plan contract picks
/// it up.
///
/// A `None` here is not cosmetic: it changes the plan hash and makes
/// `raster_verify` probe every known extension instead of the one the sink
/// writes, which silently loosens both.
#[test]
#[cfg_attr(miri, ignore)]
fn content_format_is_the_configured_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(256, 256, 256, Layout::Xyz);

    let png = PmTilesSink::builder(dir.path().join("a.pmtiles"))
        .plan(plan.clone())
        .build()
        .expect("a PNG sink builds");
    assert_eq!(png.content_format(), Some(TileFormat::Png));

    let jpeg = PmTilesSink::builder(dir.path().join("b.pmtiles"))
        .plan(plan.clone())
        .tile_format(TileFormat::Jpeg { quality: 71 })
        .build()
        .expect("a JPEG sink builds");
    assert_eq!(
        jpeg.content_format(),
        Some(TileFormat::Jpeg { quality: 71 })
    );

    let config = EngineConfig::default();
    let contract = libviprs::resume::PlanContract::from_engine(&config, &jpeg);
    assert_eq!(
        contract.format,
        Some(TileFormat::Jpeg { quality: 71 }),
        "the resume contract reads the format through the sink"
    );
}

/// The run's engine settings reach the archive's `vnd.libviprs` namespace.
#[test]
#[cfg_attr(miri, ignore)]
fn the_engine_config_reaches_the_archive_metadata() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let archive = dir.path().join("meta.pmtiles");
    let sink = PmTilesSink::builder(&archive)
        .plan(plan.clone())
        .build()
        .expect("the sink builds");
    // A concurrency the fallback cannot produce. Without it, a
    // `record_engine_config` that dropped the config on the floor would leave
    // `concurrency: 0`, which is also what the default says, and this test
    // would pass while proving the hook is wired.
    EngineBuilder::new(&gradient(512, 512), plan.clone(), sink)
        .with_concurrency(3)
        .run()
        .expect("a run with three workers succeeds");

    let meta = walk_metadata(&archive);
    let vnd = meta
        .vnd_libviprs
        .expect("the sink stamps its own namespace into the metadata");
    assert_eq!(vnd.coordinate_convention, "zxy");
    let generation = vnd
        .generation
        .expect("record_engine_config gives the namespace its generation block");
    assert_eq!(generation.tile_size, 256);
    assert_eq!(generation.layout, Layout::Xyz);
    assert_eq!(generation.format, TileFormat::Png);
    assert_eq!(
        generation.concurrency, 3,
        "the run's worker count comes from the engine config the hook captured, \
         and 3 is a value no fallback produces"
    );

    let source = vnd.source.expect("the source block records the raster");
    assert_eq!((source.width, source.height), (512, 512));
}

/// The archive's zoom range is the plan's level range.
#[test]
#[cfg_attr(miri, ignore)]
fn the_archive_zoom_range_is_the_plan_level_range() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let archive = run_into_archive(&gradient(512, 512), &plan, dir.path());
    let walked = walk(&archive);

    let levels: Vec<u32> = plan.levels.iter().map(|l| l.level).collect();
    assert_eq!(
        u32::from(walked.header.min_zoom),
        *levels.iter().min().expect("a plan has levels")
    );
    assert_eq!(
        u32::from(walked.header.max_zoom),
        *levels.iter().max().expect("a plan has levels")
    );
    assert_eq!(walked.header.tile_type, TileType::Png);
    assert_eq!(walked.header.tile_compression, Compression::None);
}

/// `sync_pending` moves the staged bytes out of the writer's buffer and onto
/// the disk.
///
/// The control is the assertion before the barrier: the staging files must
/// still be **empty**, or the test is measuring a flush that had already
/// happened for unrelated reasons and would stay green with the barrier
/// removed. `arm_durability_tracking` is called first because the engine calls
/// it first; the barrier does not depend on it, and the sink's docs say so.
#[test]
#[cfg_attr(miri, ignore)]
fn sync_pending_pushes_the_staged_payloads_out_of_the_buffer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let out = dir.path().join("durable.pmtiles");
    let sink = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .build()
        .expect("the sink builds");
    sink.arm_durability_tracking();

    let mut written = 0usize;
    for coord in plan.tile_coords().take(6) {
        let tile = Tile {
            coord,
            raster: gradient(64, 64),
            blank: false,
        };
        sink.write_tile(&tile).expect("the sink takes a tile");
        written += 1;
    }
    assert!(
        written >= 4,
        "the positive control: fewer than four tiles is not a buffer"
    );

    let (files, before) = staging(dir.path(), &out);
    assert!(
        files >= 1,
        "the positive control: no staging file means this test is measuring \
         nothing at all"
    );
    assert_eq!(
        before, 0,
        "the control: the staged payloads must still be in the writer's buffer, \
         otherwise the barrier below is not what put them on disk"
    );

    sink.sync_pending().expect("the durability barrier runs");

    let (_, after) = staging(dir.path(), &out);
    assert!(
        after > 0,
        "sync_pending left {after} staged bytes on disk for {written} tiles"
    );
}

/// How many files the sink is staging under `dir`, and how many bytes of them
/// have reached the disk. Everything that is not the archive itself.
fn staging(dir: &Path, archive: &Path) -> (usize, u64) {
    let mut files = 0;
    let mut bytes = 0;
    for entry in std::fs::read_dir(dir).expect("the output directory is readable") {
        let entry = entry.expect("a directory entry");
        if entry.path() == archive || entry.path().is_dir() {
            continue;
        }
        files += 1;
        bytes += entry.metadata().expect("stat").len();
    }
    (files, bytes)
}

// ---------------------------------------------------------------------------
// checkpoint_root, the run lock, and the resume modes
// ---------------------------------------------------------------------------

/// An `Overwrite` run does not wipe the directory the archive sits in.
///
/// `prepare_resume_state` hands `sink.checkpoint_root()` to `wipe_directory`,
/// whose ownership guard refuses any directory that is non-empty and holds no
/// `.libviprs-job.json`. A sink returning the archive's own parent therefore
/// either deletes a user's files or refuses every Overwrite run the moment one
/// of their files is there. Both halves are asserted: the unrelated file
/// survives **and** the run succeeds.
#[test]
#[cfg_attr(miri, ignore)]
fn overwrite_leaves_the_unrelated_file_beside_the_archive_alone() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bystander = dir.path().join("notes.txt");
    std::fs::write(&bystander, b"someone else's file").expect("write the bystander");

    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let out = dir.path().join("over.pmtiles");
    let sink = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .build()
        .expect("the sink builds");

    EngineBuilder::new(&gradient(512, 512), plan, sink)
        .with_resume(ResumePolicy::overwrite())
        .run()
        .expect("an Overwrite run into a PMTiles archive is not refused");

    assert_eq!(
        std::fs::read(&bystander).expect("the bystander survives"),
        b"someone else's file",
        "Overwrite wiped a file it does not own"
    );
    assert!(out.is_file(), "the archive was still produced");
}

/// The sink exposes no checkpoint root at all.
#[test]
#[cfg_attr(miri, ignore)]
fn the_sink_exposes_no_checkpoint_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(256, 256, 256, Layout::Xyz);
    let sink = PmTilesSink::builder(dir.path().join("cp.pmtiles"))
        .plan(plan)
        .build()
        .expect("the sink builds");

    assert!(
        sink.checkpoint_root().is_none(),
        "a single-file archive has no directory of tiles to checkpoint against, \
         and anything it returned here would be wiped on Overwrite"
    );
}

/// Two sinks aimed at one archive cannot both exist.
///
/// They would share `<path>.tmp.data` and `<path>.tmp.idx` and corrupt each
/// other's staging, so the second one is refused rather than allowed to race.
#[test]
#[cfg_attr(miri, ignore)]
fn a_second_sink_on_one_archive_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(256, 256, 256, Layout::Xyz);
    let out = dir.path().join("contended.pmtiles");

    let first = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .build()
        .expect("the first sink takes the lock");
    assert!(
        first.job_dir().is_dir(),
        "the lock lives in a sidecar directory the sink owns"
    );

    let second = PmTilesSink::builder(&out).plan(plan.clone()).build();
    match second {
        Err(SinkError::RunLock(libviprs::ResumeError::Locked { .. })) => {}
        other => panic!("a second sink on one archive must be refused, got {other:?}"),
    }

    // And the refusal is not permanent: releasing the first one frees it.
    drop(first);
    PmTilesSink::builder(&out)
        .plan(plan)
        .build()
        .expect("the lock is released with the sink that held it");
}

/// `ResumeMode::Resume` is refused by name at build time.
#[test]
#[cfg_attr(miri, ignore)]
fn resume_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(256, 256, 256, Layout::Xyz);

    let built = PmTilesSink::builder(dir.path().join("r.pmtiles"))
        .plan(plan)
        .resume_mode(ResumeMode::Resume)
        .build();
    match built {
        Err(SinkError::UnsupportedResumeMode {
            mode: ResumeMode::Resume,
        }) => {}
        other => panic!("Resume must be refused by name, got {other:?}"),
    }
    assert!(
        !dir.path().join("r.pmtiles").exists(),
        "a refused build leaves nothing behind"
    );
}

/// `ResumeMode::Verify` builds and `ResumeMode::Resume` still does not
/// (issue #1122).
///
/// Both halves are asserted here rather than in two cells, because the gate is
/// one `matches!` over the mode and the failure mode worth guarding against is
/// somebody widening it. A verify-only cell stays green for a gate that lets
/// everything through, and the sibling Resume cell would be the only thing
/// that noticed, which is exactly the kind of coupling that gets lost in a
/// rebase. One test, both directions, one `matches!`.
#[test]
#[cfg_attr(miri, ignore)]
fn verify_builds_where_resume_does_not() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(256, 256, 256, Layout::Xyz);

    let verify = PmTilesSink::builder(dir.path().join("v.pmtiles"))
        .plan(plan.clone())
        .resume_mode(ResumeMode::Verify)
        .build()
        .expect("Verify reads the archive through PyramidReader, so it builds");
    assert!(
        verify.job_dir().is_dir(),
        "a Verify sink takes the advisory run lock like any other, so two jobs \
         aimed at one archive still cannot overlap"
    );
    drop(verify);

    let resumed = PmTilesSink::builder(dir.path().join("r2.pmtiles"))
        .plan(plan)
        .resume_mode(ResumeMode::Resume)
        .build();
    match resumed {
        Err(SinkError::UnsupportedResumeMode {
            mode: ResumeMode::Resume,
        }) => {}
        other => panic!(
            "Resume needs the writer's staging to be reconstructible from a \
             checkpoint and it is not, so it must still be refused by name, got \
             {other:?}"
        ),
    }
}

/// A resume that would actually drop a tile is refused at the tile, not
/// silently honoured.
///
/// `seed_completed_tile` is the one hook the engine calls for each coordinate
/// a resume skips. A sink that leaves it at the trait default answers `Ok` and
/// the archive comes out missing every pre-crash tile, which is the silent
/// corruption the issue asks to avoid.
#[test]
#[cfg_attr(miri, ignore)]
fn seeding_a_completed_tile_is_refused_rather_than_accepted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(256, 256, 256, Layout::Xyz);
    let sink = PmTilesSink::builder(dir.path().join("seed.pmtiles"))
        .plan(plan)
        .build()
        .expect("the sink builds");

    let tile = Tile {
        coord: TileCoord {
            level: 0,
            col: 0,
            row: 0,
        },
        raster: gradient(8, 8),
        blank: false,
    };
    match sink.seed_completed_tile(&tile) {
        Err(SinkError::UnsupportedResumeMode {
            mode: ResumeMode::Resume,
        }) => {}
        other => panic!("a skipped resume tile must be refused, got {other:?}"),
    }
}

/// A `Verify` run answers about the archive, and answers both ways
/// (issue #1122).
///
/// This cell used to assert `EngineError::VerifyRequiresOnDiskSink`, and the
/// reason it named the variant instead of calling `is_err()` still applies
/// with the refusal gone. A verify that is half-wired reports the first
/// coordinate missing on every archive, and `is_err()` is green for it. What
/// tells a real verify from that one is not the failure, it is the pairing: a
/// sound archive has to come back `Ok` and a short one has to come back with a
/// named coordinate, and no half-wired implementation does both. So both are
/// asserted here, in one cell, over archives that differ in exactly one way.
///
/// The fuller matrix (the archive that is wider than the plan, the damaged
/// one, the one generated at another tile size, the read-only guarantee and
/// the event transcript) lives in `tests/pmtiles_plan_aware_verify.rs`.
#[test]
#[cfg_attr(miri, ignore)]
fn a_verify_run_reports_on_the_archive_in_both_directions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let src = gradient(512, 512);

    let out = dir.path().join("verify.pmtiles");
    let sink = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .build()
        .expect("the sink builds");
    EngineBuilder::new(&src, plan.clone(), sink)
        .with_engine(libviprs::EngineKind::Monolithic)
        .run()
        .expect("the archive run succeeds");

    let verifier = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .resume_mode(ResumeMode::Verify)
        .build()
        .expect("a Verify-mode sink builds");
    let result = EngineBuilder::new(&src, plan.clone(), verifier)
        .with_engine(libviprs::EngineKind::Monolithic)
        .with_resume(ResumePolicy::verify())
        .run()
        .expect("a sound archive verifies against the plan that wrote it");
    assert_eq!(
        result.tiles_produced, 0,
        "Verify writes nothing, so it produced no tiles"
    );

    // The other half. The same plan, over an archive written from a source
    // that is half as tall, so the top level is short by a row of tiles.
    //
    // The plan the short archive is checked against carries the 512x512 grid
    // and the archive's own 512x256 source size, and it has to. Since #1130 a
    // verify compares the recorded source size before it sweeps, so the pair
    // as it stood was refused for being a pyramid of a different picture,
    // which is true and is the root cause, and left this cell unable to fail
    // for the reason it is named for. The two cannot be separated through the
    // planner: a level's grid comes from the source size, so any pair with
    // different grids has different sizes. The sibling helper in
    // `tests/pmtiles_plan_aware_verify.rs` spells the whole argument out.
    let short = dir.path().join("short.pmtiles");
    let narrow = plan_for(512, 256, 256, Layout::Xyz);
    let sink = PmTilesSink::builder(&short)
        .plan(narrow.clone())
        .build()
        .expect("the sink builds");
    EngineBuilder::new(&gradient(512, 256), narrow, sink)
        .with_engine(libviprs::EngineKind::Monolithic)
        .run()
        .expect("the short archive run succeeds");

    let mut relabelled = plan.clone();
    relabelled.image_width = 512;
    relabelled.image_height = 256;
    let verifier = PmTilesSink::builder(&short)
        .plan(relabelled.clone())
        .resume_mode(ResumeMode::Verify)
        .build()
        .expect("a Verify-mode sink builds");
    let err = EngineBuilder::new(&gradient(512, 256), relabelled, verifier)
        .with_engine(libviprs::EngineKind::Monolithic)
        .with_resume(ResumePolicy::verify())
        .run()
        .expect_err("an archive short of the plan cannot verify");
    let missing = TileCoord {
        level: 9,
        col: 0,
        row: 1,
    };
    assert!(
        err.to_string().contains(&format!("{missing:?}")),
        "the refusal must name the coordinate it could not resolve \
         ({missing:?}), got: {err}"
    );
}

// ---------------------------------------------------------------------------
// What the sink refuses
// ---------------------------------------------------------------------------

/// A layout whose level index is not a zoom is refused.
///
/// `Layout::Xyz` and `Layout::Google` address `(z, x, y)`. DeepZoom, Zoomify
/// and IIIF do not: their level index is a tier and their tile path is not a
/// `z/x/y` triple, so an archive built from one is addressable and renders
/// nonsense in anything that opens it.
#[test]
#[cfg_attr(miri, ignore)]
fn a_layout_that_is_not_addressed_by_zxy_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    for layout in [Layout::DeepZoom, Layout::Zoomify] {
        let plan = plan_for(512, 512, 256, layout);
        let built = PmTilesSink::builder(dir.path().join("layout.pmtiles"))
            .plan(plan)
            .build();
        match built {
            Err(SinkError::Unsupported(msg)) => {
                assert!(
                    msg.contains("layout"),
                    "the refusal must name the layout, got {msg}"
                );
            }
            other => panic!("{layout:?} must be refused, got {other:?}"),
        }
    }

    // The control: the two layouts that do map onto (z, x, y) still build.
    for layout in [Layout::Xyz, Layout::Google] {
        let plan = plan_for(512, 512, 256, layout);
        PmTilesSink::builder(dir.path().join(format!("{layout:?}.pmtiles")))
            .plan(plan)
            .build()
            .unwrap_or_else(|e| panic!("{layout:?} must still build, got {e}"));
    }
}

/// `TileFormat::Raw` is refused, because PMTiles has no tile type for it.
#[test]
#[cfg_attr(miri, ignore)]
fn raw_tiles_are_refused_because_the_format_cannot_describe_them() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(256, 256, 256, Layout::Xyz);
    let built = PmTilesSink::builder(dir.path().join("raw.pmtiles"))
        .plan(plan)
        .tile_format(TileFormat::Raw)
        .build();
    match built {
        Err(SinkError::PmTiles(libviprs::pmtiles::PmTilesError::UnsupportedTileFormat {
            format: TileFormat::Raw,
        })) => {}
        other => panic!("raw pixel bytes are not a PMTiles tile, got {other:?}"),
    }
}

/// The builder refuses to build without a plan.
#[test]
#[cfg_attr(miri, ignore)]
fn the_builder_refuses_to_build_without_a_plan() {
    let dir = tempfile::tempdir().expect("tempdir");
    match PmTilesSink::builder(dir.path().join("noplan.pmtiles")).build() {
        Err(SinkError::MissingField("PmTilesSinkBuilder::plan")) => {}
        other => panic!("a sink without a plan must not build, got {other:?}"),
    }
}

/// Finishing twice is a typed error rather than a second archive.
#[test]
#[cfg_attr(miri, ignore)]
fn finishing_twice_is_a_typed_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let out = dir.path().join("twice.pmtiles");
    let sink = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .build()
        .expect("the sink builds");

    for coord in plan.tile_coords() {
        let tile = Tile {
            coord,
            raster: gradient(32, 32),
            blank: false,
        };
        sink.write_tile(&tile).expect("the sink takes a tile");
    }
    sink.finish()
        .expect("the first finish publishes the archive");
    assert!(out.is_file(), "the archive is there after the first finish");

    match sink.finish() {
        Err(SinkError::Unsupported(msg)) => {
            assert!(
                msg.contains("finished"),
                "the second finish must say the sink is finished, got {msg}"
            );
        }
        other => panic!("a second finish must be a typed error, got {other:?}"),
    }
}

/// Writing after `finish` is refused rather than silently dropped.
#[test]
#[cfg_attr(miri, ignore)]
fn writing_after_finish_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(256, 256, 256, Layout::Xyz);
    let out = dir.path().join("after.pmtiles");
    let sink = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .build()
        .expect("the sink builds");

    for coord in plan.tile_coords() {
        sink.write_tile(&Tile {
            coord,
            raster: gradient(16, 16),
            blank: false,
        })
        .expect("the sink takes a tile");
    }
    sink.finish().expect("finish publishes");

    let late = sink.write_tile(&Tile {
        coord: TileCoord {
            level: 0,
            col: 0,
            row: 0,
        },
        raster: gradient(16, 16),
        blank: false,
    });
    assert!(
        late.is_err(),
        "a tile written after the archive was published has nowhere to go"
    );
}

// ---------------------------------------------------------------------------
// The one hashing call site
// ---------------------------------------------------------------------------

/// The tile path hashes each tile exactly once, through the crate's own
/// digest.
///
/// #990 says the engine's digest must be passed into the writer rather than
/// recomputed, and #989's `add_tile` takes the digest as a parameter precisely
/// so it never re-derives one. The only way that promise is broken is a second
/// hashing call site inside the sink, so this reads the module and asserts
/// there is exactly one, with a positive control that the read found the file
/// it thinks it did.
///
/// The call it looks for is `dedupe::content_digest_for`, which is where the
/// hashing lives since #1145 took the `DedupeIndex` mutex off the write path.
/// `DedupeIndex::content_digest` delegates to it, so this is the same one
/// place it always was under a different name.
#[test]
fn the_sink_hashes_a_tile_in_exactly_one_place() {
    let source = include_str!("../src/sink_pmtiles.rs");
    assert!(
        source.len() > 4_000,
        "the positive control: this assertion is worthless if the include \
         picked up an empty or truncated file ({} bytes)",
        source.len()
    );

    let code: Vec<&str> = source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect();
    let code = code.join("\n");

    let digests = code.matches("content_digest_for(").count();
    assert_eq!(
        digests, 1,
        "the sink derives its digest in exactly one place"
    );
    assert!(
        !code.contains("DedupeIndex"),
        "the sink holds the run's strategy, not an index, so the hash runs with \
         nothing locked"
    );
    for forbidden in ["blake3", "content_hash(", "hash_content"] {
        assert!(
            !code.contains(forbidden),
            "the sink must not hash a tile itself; found {forbidden:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// DirectoryPyramidReader
// ---------------------------------------------------------------------------

/// The directory reader returns the bytes `FsSink` wrote, and `None` for a
/// coordinate the pyramid does not have.
#[test]
#[cfg_attr(miri, ignore)]
fn the_directory_reader_returns_what_the_run_wrote_and_none_elsewhere() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let tree = run_into_directory(&gradient(512, 512), &plan, dir.path());

    let reader = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");

    let mut seen = 0usize;
    for coord in plan.tile_coords() {
        let bytes = reader
            .tile(coord)
            .expect("the reader answers")
            .unwrap_or_else(|| panic!("{coord:?} is missing"));
        let on_disk = std::fs::read(
            tree.join(
                plan.tile_path(coord, "png")
                    .expect("a planned coord has a path"),
            ),
        )
        .expect("the file is there");
        assert_eq!(bytes, on_disk);
        seen += 1;
    }
    assert!(
        seen >= 4,
        "the positive control: a reader that answered nothing would pass an \
         empty loop"
    );

    let off_the_end = TileCoord {
        level: 0,
        col: 4_000,
        row: 4_000,
    };
    assert_eq!(
        reader
            .tile(off_the_end)
            .expect("out of range is not an error"),
        None,
        "a coordinate the plan does not have is absent, not a failure"
    );
}

/// The directory reader resolves a tile through the plan's own tile path, not
/// through a hardcoded `{z}/{x}/{y}.png`.
///
/// This test exists because a mutation that replaced the `plan.tile_path` call
/// with exactly that literal reddened nothing: every other test in this file
/// uses `Layout::Xyz` and PNG, which is the one shape the literal reproduces.
/// Two things tell them apart and both are here, a layout whose tile path is
/// not a `z/x/y` triple and a tile encoding whose extension is not `png`.
#[test]
#[cfg_attr(miri, ignore)]
fn the_directory_reader_resolves_a_tile_through_the_plan_not_a_literal_path() {
    let dir = tempfile::tempdir().expect("tempdir");

    // DeepZoom writes `{level}/{col}_{row}.{ext}`, which is not a z/x/y triple.
    let deep = plan_for(512, 512, 256, Layout::DeepZoom);
    let deep_root = dir.path().join("deep");
    EngineBuilder::new(
        &gradient(512, 512),
        deep.clone(),
        FsSink::new(&deep_root, deep.clone()),
    )
    .run()
    .expect("a DeepZoom run into a directory succeeds");

    let reader = DirectoryPyramidReader::try_open(&deep_root, deep.clone(), TileFormat::Png)
        .expect("the DeepZoom tree opens");
    let mut seen = 0usize;
    for coord in deep.tile_coords() {
        assert!(
            reader.tile(coord).expect("the reader answers").is_some(),
            "{coord:?} is missing from a DeepZoom tree, so the reader is not              reading through the plan"
        );
        seen += 1;
    }
    assert!(
        seen >= 4,
        "the positive control: an empty loop proves nothing"
    );

    // And an encoding whose extension is not `png`.
    let jpeg = plan_for(512, 512, 256, Layout::Xyz);
    let jpeg_root = dir.path().join("jpeg");
    let quality = TileFormat::Jpeg { quality: 80 };
    EngineBuilder::new(
        &gradient(512, 512),
        jpeg.clone(),
        FsSink::new(&jpeg_root, jpeg.clone()).with_format(quality),
    )
    .run()
    .expect("a JPEG run into a directory succeeds");

    let reader = DirectoryPyramidReader::try_open(&jpeg_root, jpeg.clone(), quality)
        .expect("the JPEG tree opens");
    let mut seen = 0usize;
    for coord in jpeg.tile_coords() {
        let bytes = reader
            .tile(coord)
            .expect("the reader answers")
            .unwrap_or_else(|| panic!("{coord:?} is missing from a JPEG tree"));
        assert!(
            bytes.starts_with(&[0xff, 0xd8, 0xff]),
            "{coord:?} came back without a JPEG signature, so the extension the              reader used was not the one the sink wrote"
        );
        seen += 1;
    }
    assert!(
        seen >= 4,
        "the positive control: an empty loop proves nothing"
    );
}

/// The directory reader describes the pyramid it was opened over.
#[test]
#[cfg_attr(miri, ignore)]
fn the_directory_reader_describes_the_pyramid() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let tree = run_into_directory(&gradient(512, 512), &plan, dir.path());

    let reader = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");
    let described = reader.describe().expect("the plan describes itself");

    assert_eq!(described.tile_size, Some(256));
    assert_eq!(described.layout, Some(Layout::Xyz));
    assert_eq!(described.format, Some(TileFormat::Png));
    assert_eq!(
        described.max_level,
        plan.levels
            .iter()
            .map(|l| l.level)
            .max()
            .expect("a plan has levels")
    );
    assert_eq!(reader.tile_format(), Some(TileFormat::Png));
}

// ---------------------------------------------------------------------------
// Ordered emission (issue #1145)
// ---------------------------------------------------------------------------

/// A sink that records the coordinates it is handed, in the order it is
/// handed them, and asks the engine for a particular order.
///
/// It writes nothing. The whole point is the sequence, and a sink that also
/// produced an archive would let a passing cell be explained by the archive
/// instead of by the calls.
struct OrderProbe {
    order: EmissionOrder,
    seen: Mutex<Vec<TileCoord>>,
}

impl OrderProbe {
    fn new(order: EmissionOrder) -> Self {
        Self {
            order,
            seen: Mutex::new(Vec::new()),
        }
    }

    /// The tile ids of the coordinates it was handed, in call order.
    fn tile_ids(&self) -> Vec<u64> {
        self.seen
            .lock()
            .expect("the probe's mutex is never poisoned")
            .iter()
            .map(|c| {
                let z = u8::try_from(c.level).expect("a unit-scale plan stays under zoom 32");
                zxy_to_tileid(z, c.col, c.row).expect("a planned coordinate is addressable")
            })
            .collect()
    }
}

impl TileSink for OrderProbe {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        self.seen
            .lock()
            .expect("the probe's mutex is never poisoned")
            .push(tile.coord);
        Ok(())
    }

    fn emission_order(&self) -> EmissionOrder {
        self.order
    }
}

/// Every coordinate the plan holds, as a tile id.
fn planned_tile_ids(plan: &PyramidPlan) -> BTreeSet<u64> {
    let mut ids = BTreeSet::new();
    for level in &plan.levels {
        for row in 0..level.rows {
            for col in 0..level.cols {
                let z = u8::try_from(level.level).expect("a unit-scale plan stays under zoom 32");
                ids.insert(
                    zxy_to_tileid(z, col, row).expect("a planned coordinate is addressable"),
                );
            }
        }
    }
    ids
}

/// Run one pyramid into an archive with the layout and emission order named,
/// under a fixed file name so two runs differ in nothing but what is asked
/// for here.
///
/// The file **stem** is load-bearing and is why this helper exists rather than
/// two calls to `run_into_archive`. `PmTilesSink` fills the archive's
/// `metadata.name` from it when the caller supplies none, so two archives
/// written to differently named files carry different metadata and cannot be
/// compared byte for byte however identical their tiles are.
fn run_with_layout(
    src: &Raster,
    plan: &PyramidPlan,
    dir: &Path,
    layout: ArchiveLayout,
    ordered: bool,
    concurrency: usize,
) -> PathBuf {
    let out = dir.join("pyramid.pmtiles");
    let sink = PmTilesSink::builder(&out)
        .plan(plan.clone())
        .writer_options(WriterOptions::default().with_layout(layout))
        .ordered_emission(ordered)
        .build()
        .expect("a PNG XYZ sink builds");
    EngineBuilder::new(src, plan.clone(), sink)
        .with_concurrency(concurrency)
        .run()
        .expect("a unit-scale run into a PMTiles archive succeeds");
    out
}

/// A plan with a level whose grid is wide enough for Hilbert order and
/// row-major order to disagree.
///
/// 1024 at a 256 tile is a 4x4 top level, and the Hilbert curve over a 4x4
/// grid visits `(1, 0)` fifteenth and `(3, 0)` sixth. A 2x2 top level would
/// not do: the Hilbert order of a 2x2 grid is `(0,0) (0,1) (1,1) (1,0)`, and
/// the assertions below would still pass for a walk that merely reversed the
/// levels.
fn ordered_plan() -> PyramidPlan {
    plan_for(1024, 1024, 256, Layout::Xyz)
}

/// A sink that asks for tile id order is handed every planned tile exactly
/// once, in strictly ascending tile id order.
///
/// Ascending tile id is two claims at once and the fixture is chosen so that
/// both bite. The levels have to come out smallest-first, which is the
/// opposite of the order the cascade makes them in, and within a level the
/// tiles have to come out in Hilbert order, which is not the row-major order
/// the engine walks. A pyramid whose every level were a single tile would
/// pass the second claim by accident.
///
/// The control at the bottom is the same run with the default order, and it
/// is what says this cell can fail. It asserts the default is *not* ascending
/// rather than asserting some particular interleaving, because the default
/// order inside a level is whatever the workers finished in.
#[test]
#[cfg_attr(miri, ignore)]
fn ordered_emission_hands_the_sink_every_tile_in_ascending_tile_id_order() {
    let plan = ordered_plan();
    let src = gradient(1024, 1024);
    let expected = planned_tile_ids(&plan);
    assert!(
        expected.len() > 16,
        "the positive control: a fixture with one level per tile could not tell the orders apart, \
         got {} tiles",
        expected.len()
    );

    let probe = Arc::new(OrderProbe::new(EmissionOrder::TileId));
    EngineBuilder::new(&src, plan.clone(), Arc::clone(&probe))
        .with_concurrency(4)
        .run()
        .expect("a run into a recording sink succeeds");

    let seen = probe.tile_ids();
    assert_eq!(
        seen.len(),
        expected.len(),
        "an ordered run must still hand over every planned tile exactly once"
    );
    assert_eq!(
        seen.iter().copied().collect::<BTreeSet<_>>(),
        expected,
        "an ordered run must hand over the same set of tiles as the plan holds"
    );
    for pair in seen.windows(2) {
        assert!(
            pair[0] < pair[1],
            "tile ids must ascend, got {} then {} in {seen:?}",
            pair[0],
            pair[1]
        );
    }

    let control = Arc::new(OrderProbe::new(EmissionOrder::Cascade));
    EngineBuilder::new(&src, plan, Arc::clone(&control))
        .with_concurrency(4)
        .run()
        .expect("a run into a recording sink succeeds");
    let control_seen = control.tile_ids();
    assert!(
        control_seen.windows(2).any(|p| p[0] > p[1]),
        "the control: the default order is not ascending, so this cell can fail"
    );
}

/// Every entry of an archive, root first and leaves followed, in the order the
/// directories list them.
fn entries(path: &Path) -> Vec<Entry> {
    let bytes = std::fs::read(path).expect("the archive is readable");
    let header = Header::try_decode(&bytes[..HEADER_BYTES]).expect("the sink wrote a v3 header");
    let root = section(&bytes, &header, header.root_offset, header.root_length);
    let mut out = Vec::new();
    for entry in deserialize_entries(&root).expect("the root directory parses") {
        if entry.is_leaf() {
            let leaf = section(
                &bytes,
                &header,
                header.leaf_directories_offset + entry.offset,
                u64::from(entry.length),
            );
            out.extend(deserialize_entries(&leaf).expect("a leaf directory parses"));
        } else {
            out.push(entry);
        }
    }
    out
}

/// The raw tile data region of an archive.
fn tile_data(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).expect("the archive is readable");
    let header = Header::try_decode(&bytes[..HEADER_BYTES]).expect("the sink wrote a v3 header");
    let start = usize::try_from(header.tile_data_offset).expect("a unit-scale archive fits");
    let length = usize::try_from(header.tile_data_length).expect("a unit-scale archive fits");
    bytes[start..start + length].to_vec()
}

/// An ordered run in arrival layout stores the bytes the reordering pass would
/// have stored, in the order it would have stored them.
///
/// # The issue asked for whole-file identity and the two layouts cannot have it
///
/// #1145's "done when" is an archive byte-identical to the same tiles through
/// `Layout::TileId`, and that is unreachable by construction rather than by
/// anything ordered emission does or does not do. The two layouts put the
/// sections in different places, which is what `Layout` means:
///
/// * `TileId` writes header, root, metadata, leaves, tile data, so the tile
///   data starts wherever the three sections before it ended.
/// * `Arrival` reserves the first 16384 bytes for the header and the root
///   before the first payload lands, appends the tile data from there, and
///   writes the metadata and the leaves after it. `writer.rs` has the reason
///   the reservation is exactly 16384: go-pmtiles' `Verify` accepts two
///   archive sizes and the padded one is the only padded size it recognises.
///
/// Measured on this fixture before the ordered walk existed, and again after:
/// 2063063 bytes against 2046919, differing first at offset 16, which is the
/// header's `metadata_offset`. The 16144 the arrival archive is larger by is
/// exactly `16384 - 127 - root_length`, the hole between the root's end and
/// the reserved ceiling, and the assertion at the bottom of this cell pins
/// that arithmetic so the claim is a measurement rather than a paragraph.
///
/// So this cell asserts the strongest thing that is available and is the thing
/// the bar was after: every byte that is *content* is identical. The tile data
/// region byte for byte, every directory entry with its offset and run length,
/// the metadata, and every header field that is not one of the three section
/// offsets or the `clustered` flag. What is left over is the framing, and the
/// framing is the layout.
///
/// The dedupe window is what makes the comparison fair. Which payloads an
/// archive stores depends on how far apart two identical tiles *arrived*, so
/// two different arrival orders are only comparable while the window holds the
/// whole job. It does here by a wide margin, and `tile_contents_count` is
/// asserted equal first so a run where it did not is a failure about the
/// window rather than a failure about the layout.
#[test]
#[cfg_attr(miri, ignore)]
fn an_ordered_arrival_run_stores_what_the_tile_id_layout_would_have() {
    let plan = ordered_plan();
    let src = gradient(1024, 1024);

    let ordered_dir = tempfile::tempdir().expect("tempdir");
    let sorted_dir = tempfile::tempdir().expect("tempdir");
    let ordered = run_with_layout(
        &src,
        &plan,
        ordered_dir.path(),
        ArchiveLayout::Arrival,
        true,
        4,
    );
    let sorted = run_with_layout(
        &src,
        &plan,
        sorted_dir.path(),
        ArchiveLayout::TileId,
        false,
        4,
    );

    let a = walk(&ordered);
    let b = walk(&sorted);
    assert!(
        a.header.addressed_tiles_count > 16,
        "the positive control: two empty archives agree about everything"
    );
    assert_eq!(
        a.header.tile_contents_count, b.header.tile_contents_count,
        "the window held the whole job in both runs, or this comparison is about the window"
    );

    assert_eq!(
        tile_data(&ordered),
        tile_data(&sorted),
        "the data region an ordered arrival run writes must be the one the sort produces"
    );
    assert_eq!(
        entries(&ordered),
        entries(&sorted),
        "every entry must land at the same offset with the same run length"
    );
    assert_eq!(
        walk_metadata(&ordered),
        walk_metadata(&sorted),
        "the two archives must carry the same metadata object"
    );

    // Every header field but the three section offsets and `clustered`, which
    // is the whole of what the layout is allowed to move.
    let mut expected = b.header;
    expected.tile_data_offset = a.header.tile_data_offset;
    expected.metadata_offset = a.header.metadata_offset;
    expected.leaf_directories_offset = a.header.leaf_directories_offset;
    expected.clustered = a.header.clustered;
    assert_eq!(
        a.header, expected,
        "an ordered arrival archive must differ from the sorted one only in where its \
         sections sit"
    );

    // And the sizes differ by the reserved hole, exactly. This is the
    // arithmetic behind the paragraph above, pinned.
    let ordered_len = std::fs::metadata(&ordered)
        .expect("the archive exists")
        .len();
    let sorted_len = std::fs::metadata(&sorted)
        .expect("the archive exists")
        .len();
    assert_eq!(a.header.tile_data_offset, 16384, "the reserved ceiling");
    assert_eq!(
        ordered_len - sorted_len,
        16384 - HEADER_BYTES as u64 - a.header.root_length,
        "the arrival archive is larger by the hole between the root's end and the ceiling, \
         and by nothing else"
    );
}

/// An ordered arrival run earns `clustered`.
///
/// This is the one cell #1145 cannot turn green on its own. `clustered` is
/// read off the layout today, `true` for `Layout::TileId` and `false` for
/// `Layout::Arrival` whatever the tiles did, so an arrival archive whose data
/// region genuinely is in tile id order still reports `false`. #1144 makes the
/// flag measured at ingest instead of assumed from the layout, and this cell
/// is what says ordered emission was worth doing once it lands.
///
/// It asserts the decoded field and the byte, because the two are a pair: the
/// field is what a reader of this crate sees and byte 96 is what `pmtiles
/// extract` reads, and a header that decoded `true` from a byte that said
/// something else would be a bug nobody would look for.
#[test]
#[cfg_attr(miri, ignore)]
fn an_ordered_arrival_run_earns_the_clustered_flag() {
    let plan = ordered_plan();
    let src = gradient(1024, 1024);
    let dir = tempfile::tempdir().expect("tempdir");
    let archive = run_with_layout(&src, &plan, dir.path(), ArchiveLayout::Arrival, true, 4);

    let header = walk(&archive).header;
    assert!(
        header.addressed_tiles_count > 16,
        "the positive control: an empty archive has no order to be in"
    );
    assert!(
        header.clustered,
        "an ordered arrival run puts the data region in tile id order, so the flag is true"
    );

    // `header.rs` serialises `clustered` at offset 96. The raw byte is
    // asserted beside the decoded field because that byte is what the
    // reference tools read.
    let bytes = std::fs::read(&archive).expect("the archive is readable");
    assert_eq!(bytes[96], 1, "byte 96 is the clustered flag");
}

/// Two ordered runs over one source produce the same archive.
///
/// Arrival order puts the payloads in the file in the order they turn up, so
/// without an emission order this is a statement about the thread schedule and
/// is false. It is the half of #1145 that does not need the `clustered` flag,
/// and it is what "deterministic" means for a caller who wants to compare two
/// builds of the same drawing.
///
/// Both runs are at a concurrency of four against a level of sixteen tiles, so
/// the workers genuinely race; a single-threaded pair would be identical
/// whatever the emission order did.
#[test]
#[cfg_attr(miri, ignore)]
fn two_ordered_arrival_runs_produce_the_same_archive() {
    let plan = ordered_plan();
    let src = gradient(1024, 1024);

    let first_dir = tempfile::tempdir().expect("tempdir");
    let second_dir = tempfile::tempdir().expect("tempdir");
    let first = run_with_layout(
        &src,
        &plan,
        first_dir.path(),
        ArchiveLayout::Arrival,
        true,
        4,
    );
    let second = run_with_layout(
        &src,
        &plan,
        second_dir.path(),
        ArchiveLayout::Arrival,
        true,
        4,
    );

    let left = std::fs::read(&first).expect("the first archive is readable");
    let right = std::fs::read(&second).expect("the second archive is readable");
    assert!(
        left.len() > 16384,
        "the positive control: an archive of nothing would match itself"
    );
    assert_eq!(
        left.len(),
        right.len(),
        "two ordered runs must produce archives of one size"
    );
    let differing = (0..left.len()).filter(|&i| left[i] != right[i]).count();
    assert_eq!(
        differing, 0,
        "two ordered runs must produce one archive, {differing} bytes differ"
    );
}

/// A sink that asks for tile id order on an engine that cannot walk one is
/// refused by name.
///
/// Only the monolithic engine walks the plan. The streaming and MapReduce
/// engines render the source a strip at a time and emit whatever tiles a strip
/// completes, so tile id order is not something they could produce without
/// holding the whole pyramid, which is the thing they exist to avoid.
///
/// The interesting half is that this must be a refusal rather than a
/// downgrade. A sink asks for an order when its output depends on it, so an
/// archive in `Layout::Arrival` quietly given the cascade would be published
/// with bytes that depend on the thread schedule while its caller believed the
/// opposite. That is the one failure ordered emission exists to remove, and
/// getting it back through the engine selection would be worse than never
/// having the mode.
///
/// The control underneath is the same engine with the default order, which
/// runs. Without it this cell would pass for a build that refused the
/// streaming engine for any sink at all.
#[test]
fn an_ordered_sink_on_an_engine_that_cannot_walk_the_plan_is_refused() {
    let plan = plan_for(512, 512, 256, Layout::Xyz);
    let src = gradient(512, 512);

    let ordered = Arc::new(OrderProbe::new(EmissionOrder::TileId));
    let refused = EngineBuilder::new(&src, plan.clone(), Arc::clone(&ordered))
        .with_engine(EngineKind::Streaming)
        .run();
    match refused {
        Err(EngineError::UnsupportedEmissionOrder { kind, order }) => {
            assert_eq!(kind, EngineKind::Streaming);
            assert_eq!(order, EmissionOrder::TileId);
        }
        other => panic!(
            "an order no engine but the monolithic one can walk must be refused by name, got {other:?}"
        ),
    }
    assert!(
        ordered.tile_ids().is_empty(),
        "a refused run must not have written a tile first"
    );

    // The control: the same engine takes the same sink on the default order.
    let cascade = Arc::new(OrderProbe::new(EmissionOrder::Cascade));
    EngineBuilder::new(&src, plan, Arc::clone(&cascade))
        .with_engine(EngineKind::Streaming)
        .run()
        .expect("the streaming engine runs an ordinary sink");
    assert!(
        !cascade.tile_ids().is_empty(),
        "the control has to have emitted something to be a control"
    );
}

/// An ordered run holds no more tiles in flight than an unordered one.
///
/// This is a guard rather than a reproduction, and it is worth saying which.
/// It passes before the ordered walk exists, because an ordered run is an
/// ordinary run then. What it is here to catch is the obvious way to build
/// one: extract in parallel and reorder at the consumer with a buffer keyed on
/// position. A consumer that drains the channel into such a buffer has taken
/// the backpressure off the workers, so one slow tile lets every other worker
/// run to the end of the level and the buffer holds the level. The peak below
/// would then be the level's tile count rather than the queue's capacity.
///
/// The bound is `buffer_size` for the queue plus one tile in each worker's
/// hand. Rounding gets its own slack: a run that splits the queue's capacity
/// across the workers cannot always divide it evenly.
#[test]
#[cfg_attr(miri, ignore)]
fn ordered_emission_keeps_no_more_tiles_in_flight_than_the_queue_allows() {
    let plan = ordered_plan();
    let src = gradient(1024, 1024);
    let concurrency = 4;
    let buffer_size = 4;

    let probe = Arc::new(OrderProbe::new(EmissionOrder::TileId));
    let result = EngineBuilder::new(&src, plan, Arc::clone(&probe))
        .with_concurrency(concurrency)
        .with_buffer_size(buffer_size)
        .run()
        .expect("a run into a recording sink succeeds");

    let ceiling = buffer_size + 2 * concurrency;
    assert!(
        (result.queue_pressure_peak as usize) <= ceiling,
        "an ordered run must stay inside the queue's capacity, peaked at {} against {ceiling}",
        result.queue_pressure_peak
    );
    assert!(
        probe.tile_ids().len() > 16,
        "the positive control: a run that emitted nothing holds nothing"
    );
}
