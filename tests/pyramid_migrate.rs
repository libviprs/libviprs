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
//! the wrong plan applied. That comparison is sitting on the identity element
//! of the operation under test, which is the shape `.epicF/oracle/discrimination/`
//! recorded and the reason `leaves-z0z7` is not used anywhere below.
//!
//! So nothing here reads the migrated archive through a `PyramidReader`. Every
//! cell stands on one of three legs instead:
//!
//! 1. **Absolute tile ids from an independent source.** `vectors/tileid.json`
//!    is 162 `(z,x,y)` to tile id pairs, every one of them the return value of
//!    `pmtiles.ZxyToID` in go-pmtiles v1.31.2. Its rows cover all 85
//!    coordinates of a full `z0..z3` pyramid, which is what
//!    [`every_migrated_tile_sits_at_the_tile_id_go_pmtiles_gives_its_coordinate`]
//!    pins.
//! 2. **Bytes from an archive we did not write.** `distinct-z0z7.pmtiles` has
//!    19843 distinct payloads and each one is a self-describing record that
//!    carries its own zoom, x and y, so a tile at the wrong id is a payload
//!    whose embedded coordinate disagrees with where it sits.
//! 3. **Discrimination on the plan itself.** A suite that only ever runs the
//!    right plan cannot tell whether the function reads the plan at all, so
//!    [`the_same_tree_migrates_differently_under_a_different_layout`] runs a
//!    wrong one and requires the output to move.
//!
//! And before any of that, [`a_wrong_plan_reads_a_real_tile_from_the_wrong_coordinate`]
//! puts the hazard itself in the repository. It passes against the code as it
//! was before this module existed, which is the point: the trap is a property
//! of the reader and the plan, not a bug the migration introduced.

use std::path::Path;

use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner, TileCoord};
use libviprs::pyramid_reader::{DirectoryPyramidReader, PyramidReader};
use libviprs::sink::TileFormat;
use libviprs::{EngineBuilder, FsSink, PixelFormat, Raster};

// ---------------------------------------------------------------------------
// Trees to migrate
// ---------------------------------------------------------------------------

/// A source whose every 256-pixel tile differs from every other.
///
/// A flat fill would make the whole suite vacuous in the quietest possible
/// way: identical tiles are identical under every permutation, so a
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
    let dir = tempfile::tempdir().expect("a scratch directory");
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
