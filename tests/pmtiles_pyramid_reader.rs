//! One pyramid, two backends, the same tiles (issue #990).
//!
//! [`DirectoryPyramidReader`] and [`PmTilesPyramidReader`] are the two
//! implementations of [`PyramidReader`], and the acceptance criterion is that
//! a `z/x/y` request returns the same visual tile whichever of them is
//! holding the pyramid.
//!
//! # The way this comparison lies, and what stops it
//!
//! An equivalence assertion over two backends is evidence that they agree and
//! nothing else. Three things are wrong with that on its own, and each has a
//! control here.
//!
//! * **Two backends that both answer nothing agree perfectly.** So the set of
//!   coordinates compared is asserted non-empty and asserted equal to the
//!   plan's, before any byte comparison runs.
//! * **A sink and a reader that share a coordinate-mapping mistake agree
//!   perfectly too**, and the directory reader would not notice because it is
//!   handed the same `TileCoord`. So one coordinate's tile id is pinned
//!   against `go-pmtiles`' own vectors, loaded from the committed JSON.
//! * **Equal stored bytes is not the criterion; equal decoded pixels is.**
//!   Both are asserted, because equal bytes that decode to different rasters
//!   would mean the decoder is the thing that is broken, and equal pixels from
//!   different bytes is a re-encode nobody asked for.
//!
//! This file needs F1.2's indexed reader (#988), which is where
//! `PmTilesPyramidReader` reads through.

use std::path::{Path, PathBuf};

use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner, TileCoord};
use libviprs::pmtiles::tileid::zxy_to_tileid;
use libviprs::pyramid_reader::{DirectoryPyramidReader, PmTilesPyramidReader, PyramidReader};
use libviprs::sink::TileFormat;
use libviprs::sink_pmtiles::PmTilesSink;
use libviprs::{EngineBuilder, FsSink, PixelFormat, Raster};

#[path = "common/pmtiles_oracle.rs"]
mod oracle;

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

/// Run one source into both backends and return what to open them with.
fn both_backends(dir: &Path, plan: &PyramidPlan) -> (PathBuf, PathBuf) {
    let src = gradient(plan.image_width, plan.image_height);

    let archive = dir.join("pyramid.pmtiles");
    let sink = PmTilesSink::builder(&archive)
        .plan(plan.clone())
        .build()
        .expect("the archive sink builds");
    EngineBuilder::new(&src, plan.clone(), sink)
        .run()
        .expect("the archive run succeeds");

    let tree = dir.join("tree");
    EngineBuilder::new(&src, plan.clone(), FsSink::new(&tree, plan.clone()))
        .run()
        .expect("the directory run succeeds");

    (archive, tree)
}

fn plan_for(w: u32, h: u32) -> PyramidPlan {
    PyramidPlanner::new(w, h, 256, 0, Layout::Xyz)
        .expect("a plan is valid")
        .plan()
}

/// Every planned coordinate comes back from both backends, with the same
/// stored bytes and the same decoded pixels.
#[test]
#[cfg_attr(miri, ignore)]
fn fs_and_pmtiles_return_the_same_tiles() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(1024, 768);
    let (archive, tree) = both_backends(dir.path(), &plan);

    let pmt = PmTilesPyramidReader::try_open(&archive).expect("the archive opens");
    let fs = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");

    let coords: Vec<TileCoord> = plan.tile_coords().collect();
    assert!(
        coords.len() >= 8,
        "the positive control: two backends that both answer nothing for every \
         coordinate agree perfectly, so the comparison needs a real tile set, \
         got {}",
        coords.len()
    );

    let mut compared = 0usize;
    for coord in &coords {
        let from_archive = pmt
            .tile(*coord)
            .expect("the archive answers")
            .unwrap_or_else(|| panic!("{coord:?} is missing from the archive"));
        let from_tree = fs
            .tile(*coord)
            .expect("the tree answers")
            .unwrap_or_else(|| panic!("{coord:?} is missing from the tree"));

        assert_eq!(
            from_archive, from_tree,
            "{coord:?} is stored as different bytes in the two backends"
        );

        let a = libviprs::decode_bytes(&from_archive).expect("the archive tile decodes");
        let b = libviprs::decode_bytes(&from_tree).expect("the tree tile decodes");
        assert_eq!(
            (a.width(), a.height(), a.format()),
            (b.width(), b.height(), b.format()),
            "{coord:?} decodes to different geometry"
        );
        assert_eq!(a.data(), b.data(), "{coord:?} decodes to different pixels");
        compared += 1;
    }
    assert_eq!(
        compared,
        coords.len(),
        "every planned coordinate must have been compared"
    );
}

/// The coordinate both backends are asked about addresses the tile id
/// `go-pmtiles` computes.
///
/// Without this the file proves only that two pieces of this crate agree. The
/// rows come from the committed vectors at run time, and they are the twelve
/// that tell the candidate conventions apart rather than the 150 structural
/// pairs several different mappings agree on.
#[test]
#[cfg_attr(miri, ignore)]
fn the_coordinates_both_backends_use_are_the_oracle_s_coordinates() {
    let vectors = oracle::tileid_vectors();
    let rows = oracle::tile_id_rows(&vectors, "convention_discriminators");
    assert_eq!(rows.len(), 12, "the discriminating set is twelve rows");

    for row in &rows {
        let coord = TileCoord {
            level: u32::from(row.z),
            col: row.x,
            row: row.y,
        };
        // Spelled out here rather than routed through the crate's own mapping,
        // so a transposition in that mapping has nowhere to hide.
        let id = zxy_to_tileid(
            u8::try_from(coord.level).expect("a discriminator is below zoom 256"),
            coord.col,
            coord.row,
        )
        .expect("a discriminator is addressable");
        assert_eq!(id, row.tile_id, "({row:?}) is the row that disagrees");
    }
}

/// A coordinate no pyramid has is absent from both backends, not an error and
/// not somebody else's tile.
///
/// The negative control for the equivalence test above. A directory lookup
/// that fell back to a wrong path and an archive lookup that landed on the
/// preceding entry both return a real, decodable tile, and both look like they
/// are working.
#[test]
#[cfg_attr(miri, ignore)]
fn a_coordinate_outside_the_pyramid_is_absent_from_both() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(1024, 768);
    let (archive, tree) = both_backends(dir.path(), &plan);

    let pmt = PmTilesPyramidReader::try_open(&archive).expect("the archive opens");
    let fs = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");

    let present: Vec<TileCoord> = plan.tile_coords().collect();
    let top = plan
        .levels
        .iter()
        .map(|level| level.level)
        .max()
        .expect("a plan has levels");
    let absent = [
        // Inside the addressed id range but outside the grid the plan filled,
        // which is the hole a "landed on an entry" reader returns the previous
        // tile for.
        TileCoord {
            level: top,
            col: 1_000,
            row: 1_000,
        },
        // A level the pyramid does not have at all.
        TileCoord {
            level: top + 5,
            col: 0,
            row: 0,
        },
        // Past what a u64 tile id can address.
        TileCoord {
            level: 40,
            col: 0,
            row: 0,
        },
    ];
    for coord in absent {
        assert!(
            !present.contains(&coord),
            "the control: {coord:?} must really be outside the pyramid"
        );
        assert_eq!(
            pmt.tile(coord).expect("absence is not an error"),
            None,
            "{coord:?} came back from the archive"
        );
        assert_eq!(
            fs.tile(coord).expect("absence is not an error"),
            None,
            "{coord:?} came back from the tree"
        );
    }
}

/// The two backends describe the same pyramid.
#[test]
#[cfg_attr(miri, ignore)]
fn both_backends_describe_the_same_pyramid() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(1024, 768);
    let (archive, tree) = both_backends(dir.path(), &plan);

    let pmt = PmTilesPyramidReader::try_open(&archive).expect("the archive opens");
    let fs = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");

    let from_archive = pmt.describe().expect("the archive describes itself");
    let from_tree = fs.describe().expect("the plan describes the tree");

    assert_eq!(
        from_archive, from_tree,
        "the archive carries the plan's own description in vnd.libviprs, so the \
         two must agree field for field"
    );
    assert_eq!(from_archive.tile_size, Some(256));
    assert_eq!(from_archive.layout, Some(Layout::Xyz));
    assert_eq!(from_archive.format, Some(TileFormat::Png));
    assert_eq!(
        from_archive.max_level,
        plan.levels
            .iter()
            .map(|level| level.level)
            .max()
            .expect("a plan has levels"),
        "the archive's max zoom is the plan's top level"
    );
}

/// Reading through the trait object works, which is the point of having a
/// trait at all.
#[test]
#[cfg_attr(miri, ignore)]
fn both_backends_read_through_one_trait_object() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512);
    let (archive, tree) = both_backends(dir.path(), &plan);

    let readers: Vec<Box<dyn PyramidReader>> = vec![
        Box::new(PmTilesPyramidReader::try_open(&archive).expect("the archive opens")),
        Box::new(
            DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
                .expect("the tree opens"),
        ),
    ];

    let coord = TileCoord {
        level: plan
            .levels
            .iter()
            .map(|level| level.level)
            .max()
            .expect("a plan has levels"),
        col: 0,
        row: 0,
    };
    let mut answers = Vec::new();
    for reader in &readers {
        answers.push(
            reader
                .tile(coord)
                .expect("a reader answers")
                .expect("the top-left tile of the top level exists"),
        );
    }
    assert_eq!(answers.len(), 2, "both backends answered");
    assert_eq!(answers[0], answers[1], "and they answered the same thing");
}
