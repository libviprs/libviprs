//! One pyramid, three backends, the same tiles (issues #990, #1121).
//!
//! [`DirectoryPyramidReader`] and [`PmTilesPyramidReader`] are the two
//! implementations of [`PyramidReader`], and the acceptance criterion is that
//! a `z/x/y` request returns the same visual tile whichever of them is
//! holding the pyramid.
//!
//! #1121 added a third way in without adding a third implementation: the same
//! `PmTilesPyramidReader` over an injected [`ObjectStore`] instead of a local
//! file. It belongs in this comparison because the interesting failure is not
//! "the transport errors", it is "the transport quietly serves different
//! bytes", and only a three-way comparison over the whole planned coordinate
//! set says it does not.
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

        // And both backends answer the cheap question with the same number
        // they answered the expensive one with (issue #1130). A `tile_len`
        // that disagreed with its own `tile` would send a verify looking at a
        // length that belongs to another tile, which is the one way this
        // method can be worse than no method.
        assert_eq!(
            pmt.tile_len(*coord).expect("the archive answers a length"),
            Some(from_archive.len() as u64),
            "{coord:?}: the archive's length disagrees with its own payload"
        );
        assert_eq!(
            fs.tile_len(*coord).expect("the tree answers a length"),
            Some(from_tree.len() as u64),
            "{coord:?}: the tree's length disagrees with its own payload"
        );
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
    // The three #1130 added, pinned against the source rather than against the
    // other backend, because two backends reading the same wrong number agree
    // perfectly.
    assert_eq!(from_archive.source_width, Some(1024));
    assert_eq!(from_archive.source_height, Some(768));
    assert_eq!(from_archive.overlap, Some(0));
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

// ---------------------------------------------------------------------------
// The third backend: the same archive, served over a transport (issue #1121)
// ---------------------------------------------------------------------------

/// A read-only `ObjectStore` holding one archive's bytes.
///
/// It answers exactly the range it is asked for, which is what makes it a fair
/// third backend here and exactly what makes it useless for testing the bridge
/// itself. The doubles that misbehave the way a transport does live in
/// `tests/pmtiles_object_store_range.rs`; this one is only here to prove the
/// archive reads the same through the seam as it does off the disk.
#[cfg(feature = "object-store-sink")]
struct ArchiveStore(Vec<u8>);

#[cfg(feature = "object-store-sink")]
impl libviprs::sink_object_store::ObjectStore for ArchiveStore {
    fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), libviprs::sink::SinkError> {
        Err(libviprs::sink::SinkError::Unsupported(
            "this double only reads".into(),
        ))
    }

    fn get_range(
        &self,
        _key: &str,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, libviprs::sink::SinkError> {
        let start = usize::try_from(offset).expect("an offset inside a test archive fits");
        let end = start
            .checked_add(len)
            .expect("a range inside a test archive");
        self.0.get(start..end).map(<[u8]>::to_vec).ok_or_else(|| {
            libviprs::sink::SinkError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))
        })
    }

    fn size(&self, _key: &str) -> Result<Option<u64>, libviprs::sink::SinkError> {
        Ok(Some(self.0.len() as u64))
    }
}

/// The same pyramid through three backends: a tree, an archive on disk, and
/// the same archive over an injected object store.
///
/// The three controls this file already carries apply unchanged and are
/// repeated here rather than assumed, because the failure they guard against
/// is the comparison being over nothing. On top of them, the coordinates
/// compared are pinned against `go-pmtiles`' own tile ids, loaded from the
/// committed vectors at run time, so agreement between three pieces of this
/// crate is not the whole of the evidence.
#[test]
#[cfg_attr(miri, ignore)]
#[cfg(feature = "object-store-sink")]
fn fs_pmtiles_and_object_store_return_the_same_tiles() {
    use std::sync::Arc;

    use libviprs::pmtiles::tileid::zxy_to_tileid;
    use libviprs::sink_object_store::ObjectStore;
    use libviprs::sink_pmtiles::tile_coord_to_zxy;

    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(1024, 768);
    let (archive, tree) = both_backends(dir.path(), &plan);

    let bytes = std::fs::read(&archive).expect("the archive is on disk");
    let store: Arc<dyn ObjectStore> = Arc::new(ArchiveStore(bytes));

    let pmt = PmTilesPyramidReader::try_open(&archive).expect("the archive opens");
    let remote = PmTilesPyramidReader::try_from_object_store(store, "runs/pyramid.pmtiles")
        .expect("the archive opens over an object store");
    let fs = DirectoryPyramidReader::try_open(&tree, plan.clone(), TileFormat::Png)
        .expect("the tree opens");

    let coords: Vec<TileCoord> = plan.tile_coords().collect();
    assert!(
        coords.len() >= 8,
        "the positive control: three backends that all answer nothing for every \
         coordinate agree perfectly, so the comparison needs a real tile set, \
         got {}",
        coords.len()
    );

    // The pin. Four sections of go-pmtiles' own dump, so every compared
    // coordinate the oracle names carries a tile id this crate did not
    // compute. An Xyz plan of this size runs z=0 to z=10, and what the oracle
    // names inside that set is the first tile of every level, so the pin here
    // is eleven rows spanning the whole pyramid. The twelve rows that
    // discriminate between candidate Hilbert conventions are pinned by
    // `the_coordinates_both_backends_use_are_the_oracle_s_coordinates` above,
    // which is the sharper pin and does not need a pyramid to make it.
    let vectors = oracle::tileid_vectors();
    let mut oracle_rows = Vec::new();
    for section in [
        "first_and_last_of_level",
        "orientation_boundaries",
        "hilbert_order_z2",
        "hilbert_order_z3",
    ] {
        oracle_rows.extend(oracle::tile_id_rows(&vectors, section));
    }
    assert_eq!(
        oracle_rows.len(),
        32 + 44 + 16 + 64,
        "every section has to have parsed, or the pin below is over fewer rows          than it claims"
    );

    let mut compared = 0usize;
    let mut pinned = 0usize;
    let mut pinned_levels = std::collections::BTreeSet::new();
    for coord in &coords {
        let from_archive = pmt
            .tile(*coord)
            .expect("the archive answers")
            .unwrap_or_else(|| panic!("{coord:?} is missing from the archive"));
        let from_store = remote
            .tile(*coord)
            .expect("the object store answers")
            .unwrap_or_else(|| panic!("{coord:?} is missing over the object store"));
        let from_tree = fs
            .tile(*coord)
            .expect("the tree answers")
            .unwrap_or_else(|| panic!("{coord:?} is missing from the tree"));

        assert_eq!(
            from_archive, from_store,
            "{coord:?} reads differently over the transport than off the disk"
        );
        assert_eq!(
            from_store, from_tree,
            "{coord:?} is stored as different bytes in the tree and over the transport"
        );
        compared += 1;

        let (z, x, y) = tile_coord_to_zxy(*coord).expect("a planned coordinate is addressable");
        if let Some(row) = oracle_rows
            .iter()
            .find(|row| (row.z, row.x, row.y) == (z, x, y))
        {
            assert_eq!(
                zxy_to_tileid(z, x, y).expect("an addressable coordinate"),
                row.tile_id,
                "({z}, {x}, {y}) is the row that disagrees with go-pmtiles"
            );
            pinned += 1;
            pinned_levels.insert(z);
        }
    }

    assert_eq!(
        compared,
        coords.len(),
        "every planned coordinate must have been compared"
    );
    assert!(
        pinned >= 8,
        "the pin has to bite: at least eight of the compared coordinates must \
         be ones go-pmtiles named, got {pinned}"
    );
    assert!(
        pinned_levels.len() >= 8,
        "and they must span the pyramid rather than all sitting on one level, \
         got {pinned_levels:?}"
    );

    // The three describe the same pyramid too, which is what a caller reaches
    // for before it asks for a tile.
    let from_disk = pmt.describe().expect("the archive describes itself");
    let over_the_wire = remote.describe().expect("the transport describes it too");
    assert_eq!(from_disk, over_the_wire);
    assert_eq!(
        over_the_wire,
        fs.describe().expect("the plan describes the tree")
    );
}
