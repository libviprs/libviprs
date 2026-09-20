//! Turning a `{z}/{x}/{y}` tree into a PMTiles archive, and the one mistake
//! that route invites (issue #1118).
//!
//! `DirectoryPyramidReader` reads a tree through [`PyramidPlan::tile_path`]
//! and `Writer` takes tiles in any order, so a migration is a walk from one
//! into the other and the code for it is short. The tests are not short, and
//! the reason is worth stating before any of them.
//!
//! # The comparison that cannot fail
//!
//! `DirectoryPyramidReader::try_open` refuses to infer a plan, because a
//! directory of tiles does not say what its level indices mean, how big a tile
//! is or which layout placed it. A migration tool is exactly the place someone
//! adds a `--guess`, and a guess has four independent ways to be wrong: the
//! layout, the level base, the tile size and the row axis. Every one of them
//! produces a **structurally perfect archive**. `pmtiles verify` passes it,
//! the header counts are right, the directory is sorted and clustered, and
//! every tile sits at the wrong tile id.
//!
//! A round trip through our own reader cannot see any of that. The migration
//! reads through `plan.tile_path` and writes through `tile_coord_to_zxy`, so
//! reading the result back with the same plan un-applies whatever permutation
//! the wrong plan applied and every tile compares equal. The comparison is
//! sitting on the identity element of the operation under test, which is the
//! shape `.epicF/oracle/discrimination/` recorded.
//!
//! It is also why `leaves-z0z7.pmtiles` appears nowhere below. Its 21845 tiles
//! alternate between two payloads by `(x + y) % 2`, and a transposition, a
//! y-flip and a level shift all preserve that parity at every zoom, so all of
//! its cells come back identical under four wrong conventions.
//! `distinct-z0z7.pmtiles` is used instead: 19843 distinct payloads, each a
//! self-describing record carrying its own zoom, x and y, so a tile at the
//! wrong id is a payload whose embedded coordinate disagrees with where it
//! sits.
//!
//! # So nothing here reads an archive back through a `PyramidReader`
//!
//! Every cell stands on one of three legs instead.
//!
//! 1. **Absolute tile ids from an independent source.**
//!    `tests/fixtures/pmtiles/vectors/tileid.json` is 162 `(z,x,y)` to tile id
//!    pairs, every one of them the return value of `pmtiles.ZxyToID` in
//!    go-pmtiles v1.31.2. Its rows cover all 85 coordinates of a full
//!    `z0..z3` pyramid, which is what
//!    [`every_migrated_tile_sits_at_the_tile_id_go_pmtiles_gives_its_coordinate`]
//!    pins, one assertion per tile.
//! 2. **Bytes from an archive we did not write.**
//!    [`the_distinct_golden_comes_back_at_the_ids_and_bytes_it_arrived_with`]
//!    unpacks `distinct-z0z7.pmtiles` into a tree using each payload's own
//!    embedded coordinate, migrates it, and requires every tile to land at the
//!    id go-pmtiles had it at, carrying the bytes go-pmtiles had there.
//! 3. **Discrimination on the plan itself.** A suite that only ever runs the
//!    right plan cannot tell whether the function reads the plan at all, so
//!    [`the_same_tree_migrates_differently_under_a_guessed_layout`] runs a
//!    wrong one and requires the output to move.
//!
//! And before any of that, [`a_wrong_plan_reads_a_real_tile_from_the_wrong_coordinate`]
//! puts the hazard itself in the repository. It passes against the code as it
//! was before this module existed, which is the point: the trap is a property
//! of the reader and the plan, not a bug the migration introduced.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner, TileCoord};
use libviprs::pmtiles::directory::deserialize_entries;
use libviprs::pmtiles::header::HEADER_BYTES;
use libviprs::pmtiles::writer::WriterOptions;
use libviprs::pmtiles::{Entry, Header, PmTilesError};
use libviprs::pyramid_migrate::{
    MigrateError, MigrateOptions, migrate_directory_to_pmtiles, migrate_to_pmtiles,
};
use libviprs::pyramid_reader::{
    DirectoryPyramidReader, PyramidDescription, PyramidReadError, PyramidReader,
};
use libviprs::resume::ResumeMode;
use libviprs::sink::{SinkError, TileFormat};
use libviprs::sink_pmtiles::PmTilesSink;
use libviprs::{EngineBuilder, FsSink, PixelFormat, Raster};

#[path = "common/pmtiles_oracle.rs"]
mod oracle;

// ---------------------------------------------------------------------------
// Trees to migrate
// ---------------------------------------------------------------------------

/// A source whose every 256-pixel tile differs from every other.
///
/// A flat fill would make the cells that use it vacuous in the quietest
/// possible way: identical tiles are identical under every permutation, so a
/// transposed, shifted or flipped archive compares equal to a correct one.
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

/// Run a real engine into a real `FsSink`, which is how every tree that exists
/// in the wild was made.
fn generate_tree(root: &Path, plan: &PyramidPlan) {
    let src = gradient(plan.image_width, plan.image_height);
    EngineBuilder::new(&src, plan.clone(), FsSink::new(root, plan.clone()))
        .run()
        .expect("the engine writes the tree");
}

/// Write a tree by hand, at the paths `plan` puts tiles at, from a function of
/// the coordinate.
///
/// Returning `None` leaves a hole, which is a thing a real pyramid has and
/// which the migration has to count rather than fail on. The assertion that
/// something was written is the positive control: an empty tree migrates to an
/// empty archive, and every comparison made against an empty archive passes.
fn write_tree(
    root: &Path,
    plan: &PyramidPlan,
    format: TileFormat,
    tile: impl Fn(TileCoord) -> Option<Vec<u8>>,
) -> u64 {
    let mut written = 0;
    for coord in plan.tile_coords() {
        let Some(bytes) = tile(coord) else { continue };
        let relative = plan
            .tile_path(coord, format.extension())
            .expect("tile_coords only yields coordinates inside their level's grid");
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().expect("a tile path has a parent"))
            .expect("the tile's directory is creatable");
        std::fs::write(&path, &bytes).expect("the tile is writable");
        written += 1;
    }
    assert!(
        written > 0,
        "an empty tree makes every assertion downstream of it vacuous"
    );
    written
}

/// A plan whose levels are the full `2^z` grids PMTiles addresses.
///
/// `Layout::Google` is what produces them. `Layout::Xyz` halves the image down
/// to one pixel, so a 2048-pixel image puts its 8x8 grid at level 11 and gives
/// the nine levels under it a 1x1 grid each. The oracle's vectors cover whole
/// grids at z2 and z3, and a cell that pins every tile of a pyramid against
/// them needs the pyramid's levels to *be* those grids.
fn quad_plan(levels: u32) -> PyramidPlan {
    let edge = 256u32 << (levels - 1);
    let plan = PyramidPlanner::new(edge, edge, 256, 0, Layout::Google)
        .expect("a square power-of-two image plans")
        .plan();
    assert_eq!(
        plan.levels.len() as u32,
        levels,
        "a {edge}-pixel Google pyramid should have {levels} levels"
    );
    for level in &plan.levels {
        assert_eq!(
            (level.cols, level.rows),
            (1 << level.level, 1 << level.level),
            "level {} is not the full 2^z grid",
            level.level
        );
    }
    plan
}

// ---------------------------------------------------------------------------
// Self-describing payloads
// ---------------------------------------------------------------------------

/// The record shape `distinct-z0z7.pmtiles`' payloads use: `T`, the zoom, then
/// x and y as big-endian `u16`s, then coordinate-derived padding.
///
/// Writing the same shape for this suite's own trees means one decoder serves
/// both, and it means a tile that ends up at the wrong id is caught by the
/// tile itself rather than by a bookkeeping map alongside it.
fn self_describing(z: u8, x: u32, y: u32) -> Vec<u8> {
    let mut out = vec![b'T', z];
    out.extend_from_slice(
        &u16::try_from(x)
            .expect("x fits in a u16 here")
            .to_be_bytes(),
    );
    out.extend_from_slice(
        &u16::try_from(y)
            .expect("y fits in a u16 here")
            .to_be_bytes(),
    );
    // Varying length as well as varying content, so an entry whose `length`
    // came from the wrong tile is wrong in the directory and not only in the
    // payload.
    let pad = 1 + (usize::from(z) * 7 + x as usize * 3 + y as usize) % 11;
    for i in 0..pad {
        out.push(
            (i as u8)
                .wrapping_mul(13)
                .wrapping_add((x as u8) ^ (y as u8)),
        );
    }
    out
}

/// What a self-describing payload says it is, or `None` for a payload that is
/// not one.
///
/// `distinct-z0z7.pmtiles` carries sixteen addressed tiles whose payload is
/// the literal `BLANK-TILE` rather than a record, which is how its generator
/// put holes and dedupe into the same fixture. They are the reason this
/// answers `Option` rather than asserting.
fn described_coord(bytes: &[u8]) -> Option<(u8, u32, u32)> {
    if bytes.first() != Some(&b'T') || bytes.len() < 6 {
        return None;
    }
    Some((
        bytes[1],
        u32::from(u16::from_be_bytes([bytes[2], bytes[3]])),
        u32::from(u16::from_be_bytes([bytes[4], bytes[5]])),
    ))
}

// ---------------------------------------------------------------------------
// The oracle's tile ids
// ---------------------------------------------------------------------------

/// Every `(z, x, y)` to tile id pair in `vectors/tileid.json`, as a map.
///
/// The out-of-range section is left out on purpose: `ZxyToID` masks an
/// out-of-range coordinate into a different valid tile instead of refusing, so
/// those rows are a record of a reference bug rather than a target. Every row
/// that is loaded is asserted to have round-tripped, which is the same
/// statement made from the file's own data.
fn oracle_tile_ids() -> BTreeMap<(u8, u32, u32), u64> {
    let vectors = oracle::tileid_vectors();
    let mut out = BTreeMap::new();
    for section in [
        "first_and_last_of_level",
        "orientation_boundaries",
        "off_grid",
        "hilbert_order_z2",
        "hilbert_order_z3",
        "convention_discriminators",
    ] {
        for row in oracle::tile_id_rows(&vectors, section) {
            assert!(
                row.roundtrip_ok,
                "{section} carries a masked row for z{} {},{}",
                row.z, row.x, row.y
            );
            if let Some(previous) = out.insert((row.z, row.x, row.y), row.tile_id) {
                assert_eq!(
                    previous, row.tile_id,
                    "two sections disagree about z{} {},{}",
                    row.z, row.x, row.y
                );
            }
        }
    }
    assert!(
        out.len() >= 85,
        "the oracle parse produced only {} pairs",
        out.len()
    );
    out
}

// ---------------------------------------------------------------------------
// Reading an archive without asking a PyramidReader
// ---------------------------------------------------------------------------

/// An archive, parsed with nothing but `flate2` and the format module.
struct Archive {
    header: Header,
    /// Root and leaf entries concatenated, leaf pointers resolved.
    entries: Vec<Entry>,
    bytes: Vec<u8>,
}

fn gunzip(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(bytes)
        .read_to_end(&mut out)
        .expect("an archive's gzip members decompress");
    out
}

fn slice(bytes: &[u8], offset: u64, length: u64) -> &[u8] {
    let start = usize::try_from(offset).expect("an archive under test fits in memory");
    let end = start + usize::try_from(length).expect("an archive under test fits in memory");
    &bytes[start..end]
}

/// Parse an archive and check the parse against the three counts the archive
/// states about itself.
///
/// The self-check is the positive control. Everything downstream reads
/// `entries`, so a parse that produced nothing would make all of it vacuous,
/// and requiring the parse to reproduce `addressed_tiles_count`,
/// `tile_entries_count` and `tile_contents_count` means a silent mis-parse
/// fails here instead of passing everywhere.
fn parse_archive(bytes: Vec<u8>, what: &str) -> Archive {
    let header = Header::try_decode(&bytes[..HEADER_BYTES])
        .unwrap_or_else(|e| panic!("{what} has a decodable header: {e}"));

    let mut entries = Vec::new();
    let root = gunzip(slice(&bytes, header.root_offset, header.root_length));
    for entry in deserialize_entries(&root).unwrap_or_else(|e| panic!("{what}'s root decodes: {e}"))
    {
        if entry.is_leaf() {
            let leaf = gunzip(slice(
                &bytes,
                header.leaf_directories_offset + entry.offset,
                u64::from(entry.length),
            ));
            entries.extend(
                deserialize_entries(&leaf).unwrap_or_else(|e| panic!("{what}'s leaf decodes: {e}")),
            );
        } else {
            entries.push(entry);
        }
    }

    let addressed: u64 = entries.iter().map(|e| u64::from(e.run_length)).sum();
    let contents: BTreeSet<(u64, u32)> = entries.iter().map(|e| (e.offset, e.length)).collect();
    assert_eq!(
        entries.len() as u64,
        header.tile_entries_count,
        "{what}: parsed a different number of entries than the header states"
    );
    assert_eq!(
        addressed, header.addressed_tiles_count,
        "{what}: the run lengths do not sum to the addressed tile count"
    );
    assert_eq!(
        contents.len() as u64,
        header.tile_contents_count,
        "{what}: parsed a different number of distinct blobs than the header states"
    );
    assert!(
        entries.windows(2).all(|w| w[0].tile_id < w[1].tile_id),
        "{what}: entries are not in ascending tile id order"
    );

    Archive {
        header,
        entries,
        bytes,
    }
}

impl Archive {
    /// One `(tile_id, payload)` per addressed tile, runs expanded, ascending.
    fn tiles(&self) -> Vec<(u64, &[u8])> {
        let mut out = Vec::new();
        for entry in &self.entries {
            let payload = slice(
                &self.bytes,
                self.header.tile_data_offset + entry.offset,
                u64::from(entry.length),
            );
            for step in 0..u64::from(entry.run_length) {
                out.push((entry.tile_id + step, payload));
            }
        }
        out
    }
}

fn read_archive(path: &Path, what: &str) -> Archive {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("{what} is on disk at {path:?}: {e}"));
    parse_archive(bytes, what)
}

// ---------------------------------------------------------------------------
// A reader that knows nothing about itself
// ---------------------------------------------------------------------------

/// The shape a foreign archive has: tiles, and no description worth the name.
///
/// It exists for two refusals. `describe()` fails outright, which is how a
/// backend that carries none of the plan behaves, so a migration that gated on
/// `describe().layout` would fail here with the wrong error or wave a DeepZoom
/// plan through on a `None`. And `tile_format` is configurable, so the
/// "nobody said what these bytes are" refusal has something to refuse.
struct ForeignReader {
    tiles: BTreeMap<(u32, u32, u32), Vec<u8>>,
    format: Option<TileFormat>,
}

impl PyramidReader for ForeignReader {
    fn describe(&self) -> Result<PyramidDescription, PyramidReadError> {
        Err(PyramidReadError::NoDescription(
            "this backend carries no plan".to_string(),
        ))
    }

    fn tile(&self, coord: TileCoord) -> Result<Option<Vec<u8>>, PyramidReadError> {
        Ok(self
            .tiles
            .get(&(coord.level, coord.col, coord.row))
            .cloned())
    }

    fn tile_format(&self) -> Option<TileFormat> {
        self.format
    }
}

fn foreign(plan: &PyramidPlan, format: Option<TileFormat>) -> ForeignReader {
    let tiles = plan
        .tile_coords()
        .map(|c| {
            (
                (c.level, c.col, c.row),
                self_describing(c.level as u8, c.col, c.row),
            )
        })
        .collect();
    ForeignReader { tiles, format }
}

fn scratch() -> tempfile::TempDir {
    tempfile::tempdir().expect("a scratch directory")
}

/// The `SinkError` behind a refusal, or a panic naming what came instead.
fn refusal(error: MigrateError) -> SinkError {
    match error {
        MigrateError::Refused(e) => e,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The hazard, before there is anything to migrate
// ---------------------------------------------------------------------------

/**
 * Tests that a directory reader opened with the wrong layout answers a real
 * tile from the wrong coordinate, with no error anywhere (issue #1118).
 *
 * This is the whole argument for the plan being a parameter rather than
 * something a migration infers, and it is the reason the cells below never
 * compare an archive against our own reader.
 *
 * `Layout::Xyz` writes `{z}/{x}/{y}.png` and `Layout::Google` writes
 * `{z}/{y}/{x}.png`. The two plans differ in that one field and in nothing
 * else here, so a reader handed the Google one walks an Xyz tree and finds a
 * file at every coordinate it asks for: not a missing tile, not a decode
 * failure, a valid PNG of the transposed tile. Anything downstream that treats
 * `Ok(Some(bytes))` as confirmation that the plan was right is confirming
 * nothing.
 *
 * The positive control is the assertion that the two tiles differ before the
 * swap is made. On a flat image they would not, and the test would pass
 * without the bug.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn a_wrong_plan_reads_a_real_tile_from_the_wrong_coordinate() {
    let dir = scratch();
    let tree = dir.path().join("tiles");

    let plan = PyramidPlanner::new(512, 512, 256, 0, Layout::Xyz)
        .expect("a 512x512 pyramid plans")
        .plan();
    generate_tree(&tree, &plan);

    // The top level is the only one with more than one tile, and it is square,
    // so a transposition stays inside the grid instead of falling off it.
    let top = plan.levels.last().expect("a plan has levels").clone();
    assert_eq!(
        (top.cols, top.rows),
        (2, 2),
        "the hazard needs a square grid wider than one tile"
    );
    let level = top.level;

    let at = |col: u32, row: u32| {
        std::fs::read(tree.join(format!("{level}/{col}/{row}.png")))
            .unwrap_or_else(|e| panic!("the tree has {level}/{col}/{row}.png: {e}"))
    };
    let upper_left = at(0, 1);
    let lower_right = at(1, 0);
    assert_ne!(
        upper_left, lower_right,
        "the two tiles this swaps are identical, so the swap would be invisible"
    );

    let mut guessed = plan.clone();
    guessed.layout = Layout::Google;
    let reader = DirectoryPyramidReader::try_open(&tree, guessed, TileFormat::Png)
        .expect("the tree opens under a plan that does not describe it");

    let coord = TileCoord {
        level,
        col: 0,
        row: 1,
    };
    let answered = reader
        .tile(coord)
        .expect("reading a tile through the wrong plan is not an error")
        .expect("and it is not a miss either");

    assert_eq!(
        answered, lower_right,
        "the wrong plan should have handed back the transposed tile"
    );
    assert_ne!(
        answered, upper_left,
        "and it should not have handed back the tile actually at {coord:?}"
    );
}

// ---------------------------------------------------------------------------
// Leg one: absolute tile ids from go-pmtiles
// ---------------------------------------------------------------------------

/**
 * Tests that every migrated tile lands at the tile id go-pmtiles gives its
 * coordinate (issue #1118).
 *
 * This is the only leg that can catch a level shift or a y-flip. Both of those
 * permute a full square grid onto itself, so the *set* of tile ids in the
 * archive is unchanged and an assertion over the set passes; what moves is
 * which payload sits at which id. So the assertion is per tile and it is on
 * the pair: the payload says which coordinate it is, `vectors/tileid.json`
 * says which id go-pmtiles puts that coordinate at, and the entry has to be
 * there.
 *
 * A full `z0..z3` pyramid is 85 coordinates and the vectors cover all 85 of
 * them, across four sections, so nothing here is a sample. The cell asserts
 * that coverage before it migrates anything, because a lookup that quietly
 * found nothing would skip the tile rather than fail it.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn every_migrated_tile_sits_at_the_tile_id_go_pmtiles_gives_its_coordinate() {
    let ids = oracle_tile_ids();
    let dir = scratch();
    let plan = quad_plan(4);
    let tree = dir.path().join("tiles");

    let coords: Vec<TileCoord> = plan.tile_coords().collect();
    assert_eq!(
        coords.len(),
        85,
        "a full z0..z3 pyramid is 85 tiles, and this plan is not one"
    );
    for coord in &coords {
        assert!(
            ids.contains_key(&(coord.level as u8, coord.col, coord.row)),
            "the oracle has no id for z{} {},{}, so that tile would go unchecked",
            coord.level,
            coord.col,
            coord.row
        );
    }

    write_tree(&tree, &plan, TileFormat::Png, |c| {
        Some(self_describing(c.level as u8, c.col, c.row))
    });

    let reader = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");
    let out = dir.path().join("quad.pmtiles");
    let report = migrate_directory_to_pmtiles(&reader, &out, MigrateOptions::default())
        .expect("a full pyramid migrates");

    assert_eq!(
        (
            report.coords_visited,
            report.tiles_written,
            report.tiles_absent,
            report.distinct_payloads
        ),
        (85, 85, 0, 85),
        "the migration did not do the work this cell then measures"
    );
    assert_eq!(report.tile_format, TileFormat::Png);
    assert_eq!(report.out_path, out);

    let archive = read_archive(&out, "the migrated quad pyramid");
    let tiles = archive.tiles();
    assert_eq!(tiles.len(), 85, "the archive does not hold 85 tiles");

    let mut checked = BTreeSet::new();
    for (tile_id, payload) in tiles {
        let (z, x, y) = described_coord(payload)
            .unwrap_or_else(|| panic!("the payload at id {tile_id} is not a record"));
        let want = *ids.get(&(z, x, y)).unwrap_or_else(|| {
            panic!("a tile claims to be z{z} {x},{y}, which is not a coordinate this plan has")
        });
        assert_eq!(
            tile_id, want,
            "the tile that says it is z{z} {x},{y} sits at id {tile_id}, and go-pmtiles \
             puts that coordinate at {want}"
        );
        assert!(
            checked.insert((z, x, y)),
            "z{z} {x},{y} appears twice in the archive"
        );
    }
    assert_eq!(
        checked.len(),
        85,
        "only {} tiles were pinned",
        checked.len()
    );
}

// ---------------------------------------------------------------------------
// Leg two: bytes from an archive we did not write
// ---------------------------------------------------------------------------

/**
 * Tests that `distinct-z0z7.pmtiles` unpacked into a tree and migrated back
 * comes out at the ids and with the bytes it arrived with (issue #1118).
 *
 * Nothing in this cell computes a tile id. The tree is written at each
 * payload's **own** embedded coordinate, the expected id for that coordinate
 * is the id go-pmtiles had the payload at, and the expected bytes are the
 * bytes go-pmtiles had there. A transposition, a level shift or a y-flip in
 * the migration moves a payload to an id the golden had a different payload at
 * and the comparison fails per tile.
 *
 * `distinct-z0z7` rather than `leaves-z0z7` for the reason in this file's
 * header: the latter's two alternating payloads are invariant under every
 * plausible mistake.
 *
 * The sixteen `BLANK-TILE` payloads are left out of the tree and counted, so
 * they cannot be quietly dropped by a classification bug and read as a pass.
 * What is left is 19842 tiles over 21845 planned coordinates, which is also
 * the only cell here with real holes in it.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn the_distinct_golden_comes_back_at_the_ids_and_bytes_it_arrived_with() {
    let golden = parse_archive(
        oracle::golden("distinct-z0z7.pmtiles", oracle::DISTINCT_GOLDEN_SHA256),
        "the distinct golden",
    );
    assert_eq!(golden.header.addressed_tiles_count, 19858);
    assert_eq!(golden.header.tile_contents_count, 19843);

    let mut want: BTreeMap<(u8, u32, u32), (u64, Vec<u8>)> = BTreeMap::new();
    let mut blanks = 0;
    for (tile_id, payload) in golden.tiles() {
        match described_coord(payload) {
            Some(key) => {
                assert!(
                    want.insert(key, (tile_id, payload.to_vec())).is_none(),
                    "two payloads claim z{} {},{}",
                    key.0,
                    key.1,
                    key.2
                );
            }
            None => blanks += 1,
        }
    }
    assert_eq!(
        blanks, 16,
        "the golden's blank payloads were not the ones this cell expects to skip"
    );
    assert_eq!(want.len(), 19842, "the golden's records did not all parse");

    let dir = scratch();
    let plan = quad_plan(8);
    let tree = dir.path().join("tiles");
    let placed = write_tree(&tree, &plan, TileFormat::Png, |c| {
        want.get(&(c.level as u8, c.col, c.row))
            .map(|(_, bytes)| bytes.clone())
    });
    assert_eq!(
        placed as usize,
        want.len(),
        "the plan does not reach every coordinate the golden describes"
    );

    let reader = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the unpacked tree opens");
    let out = dir.path().join("distinct.pmtiles");
    let report = migrate_directory_to_pmtiles(&reader, &out, MigrateOptions::default())
        .expect("a sparse tree migrates");

    assert_eq!(report.coords_visited, 21845);
    assert_eq!(report.tiles_written, 19842);
    assert_eq!(report.tiles_absent, 21845 - 19842);
    assert_eq!(
        report.distinct_payloads, 19842,
        "every record is its own payload, so nothing should have deduplicated"
    );

    let ours = read_archive(&out, "the re-migrated distinct archive");
    assert_eq!(ours.header.addressed_tiles_count, 19842);
    assert_eq!(ours.header.tile_contents_count, 19842);

    let mut compared = 0;
    for (tile_id, payload) in ours.tiles() {
        let key = described_coord(payload)
            .unwrap_or_else(|| panic!("we wrote a payload at id {tile_id} that is not a record"));
        let (golden_id, golden_bytes) = want
            .get(&key)
            .unwrap_or_else(|| panic!("we invented a tile at z{} {},{}", key.0, key.1, key.2));
        assert_eq!(
            tile_id, *golden_id,
            "z{} {},{} came back at id {tile_id}, and go-pmtiles had it at {golden_id}",
            key.0, key.1, key.2
        );
        assert_eq!(
            payload,
            golden_bytes.as_slice(),
            "z{} {},{} came back with different bytes",
            key.0,
            key.1,
            key.2
        );
        compared += 1;
    }
    assert_eq!(
        compared, 19842,
        "only {compared} tiles were compared against the golden"
    );
}

// ---------------------------------------------------------------------------
// Leg three: discrimination on the plan
// ---------------------------------------------------------------------------

/**
 * Tests that migrating one tree under two layouts produces two different
 * archives (issue #1118).
 *
 * Every other cell runs the right plan, and a function that ignored the plan
 * entirely, or read the layout off the reader's `describe()`, would pass all
 * of them. This is the one that says the plan is an input.
 *
 * The two plans differ in the layout field and in nothing else, so both
 * enumerate the same 93 coordinates and both find a file at every one of them:
 * every level of a square Xyz pyramid has a square grid, so a transposition
 * stays inside it. That equality is asserted, and it is what makes the
 * inequality of the two archives mean what it says. Two archives that differ
 * because one of them is empty would prove nothing.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn the_same_tree_migrates_differently_under_a_guessed_layout() {
    let dir = scratch();
    let plan = PyramidPlanner::new(2048, 2048, 256, 0, Layout::Xyz)
        .expect("a 2048x2048 pyramid plans")
        .plan();
    let tree = dir.path().join("tiles");
    let written = write_tree(&tree, &plan, TileFormat::Png, |c| {
        Some(self_describing(c.level as u8, c.col, c.row))
    });

    let migrate = |plan: PyramidPlan, name: &str| {
        let reader =
            DirectoryPyramidReader::try_open(&tree, plan, TileFormat::Png).expect("the tree opens");
        let out = dir.path().join(format!("{name}.pmtiles"));
        let report = migrate_directory_to_pmtiles(&reader, &out, MigrateOptions::default())
            .expect("the migration runs");
        (report, std::fs::read(&out).expect("the archive is on disk"))
    };

    let (right, right_bytes) = migrate(plan.clone(), "right");
    let mut guessed = plan.clone();
    guessed.layout = Layout::Google;
    let (wrong, wrong_bytes) = migrate(guessed, "guessed");

    assert_eq!(
        right.tiles_written, written,
        "the right plan did not find the whole tree"
    );
    assert_eq!(
        (
            wrong.coords_visited,
            wrong.tiles_written,
            wrong.tiles_absent
        ),
        (
            right.coords_visited,
            right.tiles_written,
            right.tiles_absent
        ),
        "the guessed plan found a different number of tiles, so the archives \
         would differ for a reason that is not the one under test"
    );
    assert_eq!(
        right.distinct_payloads, wrong.distinct_payloads,
        "both archives hold the same payloads; only their placement moves"
    );
    assert_ne!(
        right_bytes, wrong_bytes,
        "the two archives are identical, so the migration is not reading the \
         layout out of the plan at all"
    );
}

// ---------------------------------------------------------------------------
// Counts
// ---------------------------------------------------------------------------

/**
 * Tests that a tree of duplicated tiles migrates to the counts go-pmtiles
 * measured for the same tiles (issue #1118).
 *
 * `dupes-z0z3.pmtiles` is 85 addressed tiles over 63 distinct payloads, and
 * go-pmtiles collapsed them into 67 entries: all four z1 tiles are one green
 * PNG and all sixteen z2 tiles are one red PNG, which are consecutive in tile
 * id space and become runs.
 *
 * That 67 is the trap. A migration written as a walk over the source
 * archive's *entries* rather than over the plan's coordinates visits 67
 * coordinates and silently drops the 18 tiles hiding in run tails, and what it
 * produces is a perfectly valid archive with `addressed_tiles_count == 67`.
 * So both 85s are asserted: the one the migration reports and the one the
 * finished archive states.
 *
 * `tile_entries_count` is deliberately not asserted. Our writer re-derives
 * runs from the tiles it was given, so the entry count is a property of the
 * writer's run-forming and not of the migration, and pinning it here would
 * make this cell fail for a reason it is not about.
 *
 * The tree is laid out using go-pmtiles' own ids, inverted: no coordinate in
 * it was computed by this crate.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn a_tree_of_duplicated_tiles_keeps_every_tile_and_stores_each_payload_once() {
    let ids = oracle_tile_ids();
    let by_id: BTreeMap<u64, (u8, u32, u32)> = ids
        .iter()
        .filter(|((z, _, _), _)| *z <= 3)
        .map(|(coord, id)| (*id, *coord))
        .collect();
    assert_eq!(by_id.len(), 85, "the oracle does not cover all of z0..z3");

    let golden = parse_archive(
        oracle::golden("dupes-z0z3.pmtiles", oracle::DUPES_GOLDEN_SHA256),
        "the dupes golden",
    );
    assert_eq!(golden.header.addressed_tiles_count, 85);
    assert_eq!(golden.header.tile_entries_count, 67);
    assert_eq!(golden.header.tile_contents_count, 63);

    let mut payloads: BTreeMap<(u8, u32, u32), Vec<u8>> = BTreeMap::new();
    for (tile_id, payload) in golden.tiles() {
        let coord = *by_id
            .get(&tile_id)
            .unwrap_or_else(|| panic!("the oracle has no coordinate for id {tile_id}"));
        assert!(
            payloads.insert(coord, payload.to_vec()).is_none(),
            "id {tile_id} resolved to a coordinate already taken"
        );
    }
    assert_eq!(payloads.len(), 85);

    let dir = scratch();
    let plan = quad_plan(4);
    let tree = dir.path().join("tiles");
    let placed = write_tree(&tree, &plan, TileFormat::Png, |c| {
        payloads.get(&(c.level as u8, c.col, c.row)).cloned()
    });
    assert_eq!(placed, 85);

    let reader = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");
    let out = dir.path().join("dupes.pmtiles");
    let report = migrate_directory_to_pmtiles(&reader, &out, MigrateOptions::default())
        .expect("the tree migrates");

    assert_eq!(
        report.tiles_written, 85,
        "the migration visited the archive's entries rather than the plan's coordinates"
    );
    assert_eq!(report.coords_visited, 85);
    assert_eq!(report.tiles_absent, 0);
    assert_eq!(report.distinct_payloads, 63);

    let ours = read_archive(&out, "the migrated dupes archive");
    assert_eq!(
        ours.header.addressed_tiles_count, 85,
        "the archive addresses fewer tiles than the tree held"
    );
    assert_eq!(
        ours.header.tile_contents_count, 63,
        "the content-hash dedupe did not collapse the duplicates"
    );
}

/**
 * Tests that a tree of identical tiles costs one stored payload however many
 * tiles reference it (issue #1118).
 *
 * This is the case the whole "a mostly blank pyramid is nearly free as an
 * archive" claim rests on, and it is the case a count of stored bytes cannot
 * tell from a migration that dropped everything: both produce a small archive.
 * So the two counts are asserted against each other. One payload, and all 21
 * coordinates still addressed.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn a_tree_of_identical_tiles_is_one_payload_addressed_by_all_of_them() {
    let dir = scratch();
    let plan = quad_plan(3);
    let tree = dir.path().join("tiles");
    let blank = b"one blank tile, repeated".to_vec();
    let written = write_tree(&tree, &plan, TileFormat::Png, |_| Some(blank.clone()));
    assert_eq!(written, 21, "a full z0..z2 pyramid is 21 tiles");

    let reader = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");
    let out = dir.path().join("blank.pmtiles");
    let report = migrate_directory_to_pmtiles(&reader, &out, MigrateOptions::default())
        .expect("the tree migrates");

    assert_eq!(report.tiles_written, 21);
    assert_eq!(report.distinct_payloads, 1);

    let ours = read_archive(&out, "the migrated blank archive");
    assert_eq!(
        ours.header.addressed_tiles_count, 21,
        "the tiles were collapsed away rather than collapsed together"
    );
    assert_eq!(ours.header.tile_contents_count, 1);
}

/**
 * Tests that a small sort buffer really reaches the writer's external merge
 * (issue #1118).
 *
 * The migration claims bounded memory, and it inherits that from the writer,
 * which sorts index records in `sort_buffer_records` batches and spills each
 * batch to a log. A test that sets the buffer small and then only checks the
 * output cannot tell whether it exercised the merge or whether the whole index
 * fitted in one batch anyway, and those two runs produce the same archive. So
 * `MigrateReport::spilled_run_count` is asserted in both directions: above zero
 * with a small buffer, and zero with the default, which is what says the option
 * was read rather than defaulted.
 *
 * The two archives are also required to be byte identical, because the sort
 * buffer is an internal knob and not a format choice.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn a_small_sort_buffer_spills_runs_and_produces_the_same_archive() {
    let dir = scratch();
    let plan = quad_plan(4);
    let tree = dir.path().join("tiles");
    write_tree(&tree, &plan, TileFormat::Png, |c| {
        Some(self_describing(c.level as u8, c.col, c.row))
    });
    let reader = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");

    let bounded = dir.path().join("bounded.pmtiles");
    let bounded_report = migrate_directory_to_pmtiles(
        &reader,
        &bounded,
        MigrateOptions::default().with_writer(WriterOptions::default().with_sort_buffer_records(8)),
    )
    .expect("the bounded migration runs");

    let roomy = dir.path().join("roomy.pmtiles");
    let roomy_report = migrate_directory_to_pmtiles(&reader, &roomy, MigrateOptions::default())
        .expect("the default migration runs");

    assert!(
        bounded_report.spilled_run_count > 0,
        "an 85-tile migration with an 8-record sort buffer never spilled a run, \
         so it exercised the in-memory sort while claiming the external merge"
    );
    assert_eq!(
        roomy_report.spilled_run_count, 0,
        "the default buffer holds the whole index, so a spill here means the \
         option is not being read"
    );
    assert_eq!(bounded_report.tiles_written, roomy_report.tiles_written);
    assert_eq!(
        std::fs::read(&bounded).expect("the bounded archive is on disk"),
        std::fs::read(&roomy).expect("the roomy archive is on disk"),
        "the sort buffer changed the bytes of the archive, which it is not \
         supposed to reach"
    );
}

// ---------------------------------------------------------------------------
// The four refusals
// ---------------------------------------------------------------------------

/**
 * Tests that a layout PMTiles cannot address is refused by name, with the
 * sink's own sentence, whatever the reader thinks (issue #1118).
 *
 * DeepZoom, Zoomify and IIIF are not addressed by `(z, x, y)`, so there is no
 * tile id for one of their coordinates to go to and an archive built from one
 * would be addressable and render nonsense. `PmTilesSink::build` already says
 * so, and this refuses with the same error and the same words rather than a
 * second wording of the same fact.
 *
 * The gate is on `plan.layout`, and the second half of this cell is why.
 * `PyramidDescription::layout` is an `Option` and it is `None` for a backend
 * that carries no plan, so a gate reading the description would either wave a
 * DeepZoom plan through on a `None` or fail with the wrong error on a reader
 * whose `describe()` does not answer at all. `ForeignReader` is the second of
 * those, and it still has to be refused for its layout.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn a_layout_pmtiles_cannot_address_is_refused_by_name() {
    let dir = scratch();
    let plan = PyramidPlanner::new(1024, 1024, 256, 0, Layout::DeepZoom)
        .expect("a DeepZoom pyramid plans")
        .plan();
    let tree = dir.path().join("tiles");
    write_tree(&tree, &plan, TileFormat::Png, |c| {
        Some(self_describing(c.level as u8, c.col, c.row))
    });

    let reader = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("a DeepZoom tree opens; it is the destination that cannot hold it");
    let out = dir.path().join("deepzoom.pmtiles");
    let error = refusal(
        migrate_directory_to_pmtiles(&reader, &out, MigrateOptions::default())
            .expect_err("a DeepZoom tree has no PMTiles tile ids"),
    );
    match &error {
        SinkError::Unsupported(message) => assert_eq!(
            message,
            "DeepZoom layout is not addressed by (z, x, y), so it has no PMTiles \
             tile ids; use Layout::Xyz or Layout::Google",
            "the refusal does not carry the sink's own sentence"
        ),
        other => panic!("expected SinkError::Unsupported, got {other:?}"),
    }
    assert!(!out.exists(), "the refusal left an archive behind");

    // The sentence above is a literal, and a literal drifts. So it is also
    // compared against the one the sink raises for the same plan, which is the
    // statement that matters: two routes into one format saying two things
    // about the same refusal is how a caller ends up handling one and not the
    // other.
    let from_the_sink = PmTilesSink::builder(dir.path().join("sink.pmtiles"))
        .plan(plan.clone())
        .build()
        .expect_err("the sink refuses a DeepZoom plan too");
    assert_eq!(
        from_the_sink.to_string(),
        error.to_string(),
        "the sink and the migration refuse the same layout in different words"
    );

    // The same refusal for a reader that cannot describe itself at all, which
    // is what a foreign archive looks like.
    let foreign = foreign(&plan, Some(TileFormat::Png));
    assert!(
        foreign.describe().is_err(),
        "this reader is supposed to have no description to gate on"
    );
    let foreign_out = dir.path().join("foreign.pmtiles");
    assert!(matches!(
        refusal(
            migrate_to_pmtiles(&foreign, &plan, &foreign_out, MigrateOptions::default())
                .expect_err("the layout is still the layout")
        ),
        SinkError::Unsupported(_)
    ));
    assert!(!foreign_out.exists());
}

/**
 * Tests that raw tiles are refused because PMTiles has no type for them
 * (issue #1118).
 *
 * A PMTiles tile is a self-describing image blob and raw pixel bytes are not
 * one: nothing in the format records the width, the height or the pixel layout
 * that would make them decodable.
 *
 * The refusal is not a `match` on `Raw` here. It comes out of
 * `TileType::try_from_tile_format`, the same call `PmTilesSinkBuilder::build`
 * makes, so the day a `Webp` variant lands there is one place that decides
 * what a PMTiles archive can hold rather than two that can disagree. That is
 * why this asserts the error the conversion raises rather than a sentence of
 * its own.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn raw_tiles_are_refused_because_the_format_cannot_describe_them() {
    let dir = scratch();
    let plan = quad_plan(3);
    let tree = dir.path().join("tiles");
    write_tree(&tree, &plan, TileFormat::Raw, |c| {
        Some(self_describing(c.level as u8, c.col, c.row))
    });

    let reader = DirectoryPyramidReader::try_open(&tree, plan, TileFormat::Raw)
        .expect("a raw tree opens; it is the destination that cannot hold it");
    let out = dir.path().join("raw.pmtiles");
    let error = refusal(
        migrate_directory_to_pmtiles(&reader, &out, MigrateOptions::default())
            .expect_err("PMTiles has no tile type for raw pixels"),
    );
    match &error {
        SinkError::PmTiles(PmTilesError::UnsupportedTileFormat { format }) => {
            assert_eq!(*format, TileFormat::Raw);
        }
        other => panic!("expected the tile-type conversion's own refusal, got {other:?}"),
    }
    assert!(!out.exists(), "the refusal left an archive behind");
}

/**
 * Tests that resume and verify are refused by name, before a byte moves
 * (issue #1118).
 *
 * Neither has a meaning here. `Verify` reads a pyramid back by stat-ing one
 * file per coordinate, which a single-file archive has no seam for, and
 * `Resume` would need the writer's staging to be reconstructible from a
 * checkpoint, which it is not. An interrupted migration restarts from the
 * beginning instead, and that is safe rather than merely tolerable: the writer
 * stages elsewhere and publishes with a rename, so a half-done run never wears
 * the finished archive's name.
 *
 * "Before a byte moves" is the assertion that nothing exists at the
 * destination afterwards.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn resume_and_verify_are_refused_by_name() {
    let dir = scratch();
    let plan = quad_plan(3);
    let tree = dir.path().join("tiles");
    write_tree(&tree, &plan, TileFormat::Png, |c| {
        Some(self_describing(c.level as u8, c.col, c.row))
    });
    let reader =
        DirectoryPyramidReader::try_open(&tree, plan, TileFormat::Png).expect("the tree opens");

    for mode in [ResumeMode::Resume, ResumeMode::Verify] {
        let out = dir.path().join(format!("{mode:?}.pmtiles"));
        let error = refusal(
            migrate_directory_to_pmtiles(
                &reader,
                &out,
                MigrateOptions::default().with_resume_mode(mode),
            )
            .expect_err("only Overwrite runs"),
        );
        match &error {
            SinkError::UnsupportedResumeMode { mode: refused } => assert_eq!(*refused, mode),
            other => panic!("expected SinkError::UnsupportedResumeMode, got {other:?}"),
        }
        assert!(!out.exists(), "the {mode:?} refusal left an archive behind");
    }
}

/**
 * Tests that a reader which will not say what its tiles are is refused rather
 * than guessed at (issue #1118).
 *
 * `PyramidReader::tile_format` is an `Option` because the two backends know
 * different things, and a foreign archive may commit to nothing. Defaulting to
 * PNG there writes a `tile_type` byte nobody measured into an archive that
 * every viewer opening it will believe, and the bytes it describes might be
 * JPEG.
 *
 * So it is `SinkError::MissingField`, naming the option the caller should have
 * set, which is the same shape `PmTilesSinkBuilder::build` uses for a missing
 * plan. The positive control is the second half: the same reader with the same
 * tiles migrates once the caller says what they are, so the refusal is about
 * the missing answer and not about the reader.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn a_reader_that_will_not_say_what_its_tiles_are_is_refused_rather_than_guessed() {
    let dir = scratch();
    let plan = quad_plan(3);
    let silent = foreign(&plan, None);

    let out = dir.path().join("silent.pmtiles");
    let error = refusal(
        migrate_to_pmtiles(&silent, &plan, &out, MigrateOptions::default())
            .expect_err("nobody said what these bytes are"),
    );
    match &error {
        SinkError::MissingField(field) => assert_eq!(*field, "MigrateOptions::tile_format"),
        other => panic!("expected SinkError::MissingField, got {other:?}"),
    }
    assert!(!out.exists(), "the refusal left an archive behind");

    let told = dir.path().join("told.pmtiles");
    let report = migrate_to_pmtiles(
        &silent,
        &plan,
        &told,
        MigrateOptions::default().with_tile_format(TileFormat::Png),
    )
    .expect("the same reader migrates once the caller says what the bytes are");
    assert_eq!(report.tiles_written, 21);
    assert_eq!(report.tile_format, TileFormat::Png);

    // And the reader's own answer is taken when there is one, which is what
    // makes the override an override rather than the only way in.
    let speaking = foreign(&plan, Some(TileFormat::Png));
    let spoken = dir.path().join("spoken.pmtiles");
    let report = migrate_to_pmtiles(&speaking, &plan, &spoken, MigrateOptions::default())
        .expect("a reader that commits to a format needs no override");
    assert_eq!(report.tile_format, TileFormat::Png);
    assert_eq!(
        std::fs::read(&told).expect("the told archive is on disk"),
        std::fs::read(&spoken).expect("the spoken archive is on disk"),
        "the two routes to the same tile format produced different archives"
    );
}

/**
 * Tests that `migrate_directory_to_pmtiles` uses the plan the reader was
 * opened with (issue #1118).
 *
 * The convenience wrapper exists so a caller who has already supplied a plan
 * does not supply a second one that can differ from it. That is only true if
 * it really reads the reader's plan, so this runs both entry points over one
 * tree and requires the archives to be byte identical.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn the_directory_entry_point_migrates_through_the_plan_its_reader_holds() {
    let dir = scratch();
    let plan = quad_plan(3);
    let tree = dir.path().join("tiles");
    write_tree(&tree, &plan, TileFormat::Png, |c| {
        Some(self_describing(c.level as u8, c.col, c.row))
    });
    let reader = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");

    let wrapped = dir.path().join("wrapped.pmtiles");
    let explicit = dir.path().join("explicit.pmtiles");
    let a = migrate_directory_to_pmtiles(&reader, &wrapped, MigrateOptions::default())
        .expect("the wrapper runs");
    let b = migrate_to_pmtiles(&reader, &plan, &explicit, MigrateOptions::default())
        .expect("the explicit call runs");

    assert_eq!(a.tiles_written, b.tiles_written);
    assert_eq!(a.tiles_written, 21);
    assert_eq!(
        std::fs::read(&wrapped).expect("the wrapped archive is on disk"),
        std::fs::read(&explicit).expect("the explicit archive is on disk"),
        "the wrapper is not migrating through the reader's own plan"
    );

    // The report says where it put the archive, and the two calls disagree
    // about that and nothing else.
    assert_eq!(a.out_path, PathBuf::from(&wrapped));
    assert_eq!(b.out_path, PathBuf::from(&explicit));
}
