//! What a verify over a PMTiles archive costs, counted off the transport
//! (issue #1130).
//!
//! `pyramid_verify` is correct and it is expensive in two ways that no
//! round-trip test can see, because both are invisible in the answer and only
//! show up in what was fetched.
//!
//! * **It reads every tile to learn every tile's length.** The sweep calls
//!   `reader.tile(coord)` for every planned coordinate and uses the result
//!   only for `bytes.is_empty()`, then drops the `Vec`. A PMTiles directory
//!   entry already carries the length, so verifying a 21851-tile pyramid pulls
//!   the whole archive off storage to learn 21851 numbers that cost no extra
//!   bytes. Over the transport seam #1121 opened that is one ranged GET per
//!   tile, serially.
//! * **It walks the directories twice.** `self_check` at the top and
//!   `addressed_tiles` at the bottom both run a full `validate::validate`, ten
//!   lines apart, under a run lock that guarantees the archive cannot change
//!   between them.
//!
//! So this file counts requests rather than checking answers. The archive is a
//! real one, written by the sink through the engine, and it is served back
//! through a [`RangeReader`] that remembers every range it was asked for.
//!
//! # The classifier needs a positive control
//!
//! "No bytes were fetched from the tile-data section" is satisfied by a run
//! that fetched nothing at all, and by a classifier that looks in the wrong
//! place. Both would be green for the wrong reason, so
//! [`payload_reads_are_visible_to_the_classifier`] fetches tiles deliberately
//! and asserts the same classifier counts them.

use std::io;
use std::path::Path;
use std::sync::Mutex;

use libviprs::observe::NoopObserver;
use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner};
use libviprs::pmtiles::{RangeReader, Reader};
use libviprs::pyramid_reader::{PmTilesPyramidReader, PyramidReader};
use libviprs::sink::TileFormat;
use libviprs::sink_pmtiles::PmTilesSink;
use libviprs::verify::pyramid_verify;
use libviprs::{EngineBuilder, EngineKind, PixelFormat, Raster};

// ---------------------------------------------------------------------------
// The counting transport
// ---------------------------------------------------------------------------

/// One range the reader asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Request {
    offset: u64,
    len: usize,
}

/// The archive's bytes, served whole and remembered range by range.
///
/// Serving from memory rather than from the file is deliberate: what is being
/// measured is what the reader *asked* for, and a request that a page cache
/// would have made free is still a request the transport in #1121 would have
/// paid a round trip for.
struct Counting {
    bytes: Vec<u8>,
    requests: Mutex<Vec<Request>>,
}

impl Counting {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Drop everything recorded so far, so a measurement starts at the seam it
    /// is about rather than at the reader's open.
    fn forget(&self) {
        self.requests
            .lock()
            .expect("the log is not poisoned")
            .clear();
    }

    fn requests(&self) -> Vec<Request> {
        self.requests
            .lock()
            .expect("the log is not poisoned")
            .clone()
    }
}

impl RangeReader for Counting {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        self.requests
            .lock()
            .expect("the log is not poisoned")
            .push(Request { offset, len });
        let start = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset past usize"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflows"))?;
        if end > self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the range reaches past the archive",
            ));
        }
        Ok(self.bytes[start..end].to_vec())
    }

    fn size(&self) -> io::Result<Option<u64>> {
        Ok(Some(self.bytes.len() as u64))
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A raster where no two tiles come out the same, so every entry has
/// `run_length == 1` and the archive holds one payload per coordinate.
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

fn plan_for(w: u32, h: u32, tile: u32) -> PyramidPlan {
    PyramidPlanner::new(w, h, tile, 0, Layout::Xyz)
        .expect("an Xyz plan over a positive source is valid")
        .plan()
}

/// Write `src` into a published archive at `path` and release the sink.
fn write_archive(path: &Path, plan: &PyramidPlan, src: &Raster) {
    let sink = PmTilesSink::builder(path)
        .plan(plan.clone())
        .build()
        .expect("the archive sink builds for an Overwrite run");
    EngineBuilder::new(src, plan.clone(), sink)
        .with_engine(EngineKind::Monolithic)
        .run()
        .expect("the archive run succeeds");
    assert!(path.is_file(), "the run published {}", path.display());
}

/// An archive written from a 512x512 gradient, served through a counting
/// transport with the open-time requests already forgotten.
fn served(dir: &Path) -> (PyramidPlan, PmTilesPyramidReader<Counting>) {
    let plan = plan_for(512, 512, 256);
    let archive = dir.join("counted.pmtiles");
    write_archive(&archive, &plan, &gradient(512, 512));

    let bytes = std::fs::read(&archive).expect("read the archive back");
    let reader = Reader::try_new(Counting::new(bytes)).expect("the archive opens");
    reader.source().forget();
    (plan, PmTilesPyramidReader::from_reader(reader))
}

/// Bytes fetched from inside the tile-data section, which is the only region
/// tile payloads live in.
fn payload_bytes(pyramid: &PmTilesPyramidReader<Counting>) -> u64 {
    let header = pyramid.reader().header();
    let start = header.tile_data_offset;
    let end = start + header.tile_data_length;
    pyramid
        .reader()
        .source()
        .requests()
        .iter()
        .filter(|request| request.offset >= start && request.offset < end)
        .map(|request| request.len as u64)
        .sum()
}

/// How many times the 127-byte header at offset 0 was fetched.
///
/// One structural walk begins with exactly one of these, so counting them
/// counts walks.
fn header_fetches(pyramid: &PmTilesPyramidReader<Counting>) -> usize {
    pyramid
        .reader()
        .source()
        .requests()
        .iter()
        .filter(|request| request.offset == 0)
        .count()
}

// ---------------------------------------------------------------------------
// The control on the classifier
// ---------------------------------------------------------------------------

/// Fetching tiles on purpose is visible to [`payload_bytes`].
///
/// Without this, "the verify fetched no payload bytes" is equally satisfied by
/// a classifier pointed at the wrong section, and the cell below it would be
/// green for an implementation that never changed.
#[test]
#[cfg_attr(miri, ignore)]
fn payload_reads_are_visible_to_the_classifier() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (plan, pyramid) = served(dir.path());

    let mut fetched = 0u64;
    for coord in plan.tile_coords() {
        let bytes = pyramid
            .tile(coord)
            .expect("reading a tile from a sound archive")
            .unwrap_or_else(|| panic!("{coord:?} is missing from the archive"));
        fetched += bytes.len() as u64;
    }

    assert!(fetched > 0, "the archive stores no bytes at all");
    assert_eq!(
        payload_bytes(&pyramid),
        fetched,
        "the classifier saw {} payload bytes and the tiles were {fetched} bytes, \
         so it is not looking at the tile-data section",
        payload_bytes(&pyramid)
    );
}

// ---------------------------------------------------------------------------
// Finding 1: a verify reads lengths, not payloads
// ---------------------------------------------------------------------------

/// A verify over an archive whose transport reports a size fetches no tile
/// payload at all.
///
/// The sweep needs two things per coordinate: that the archive holds one, and
/// that what it holds is not zero bytes. The directory entry carries both, and
/// `validate` has already bounds-checked every entry against the archive's
/// real size, so the payload read proves nothing the walk did not.
#[test]
#[cfg_attr(miri, ignore)]
fn a_verify_over_a_sized_transport_fetches_no_tile_payload() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (plan, pyramid) = served(dir.path());

    let result = pyramid_verify(&pyramid, &plan, Some(TileFormat::Png), &NoopObserver)
        .expect("a good archive verifies against its own plan");

    assert_eq!(
        payload_bytes(&pyramid),
        0,
        "the verify pulled {} bytes out of the tile-data section to learn {} \
         lengths the directory already carries",
        payload_bytes(&pyramid),
        plan.tile_coords().count()
    );
    assert_eq!(
        result.bytes_read, 0,
        "the run reports {} bytes read and it read none, which is the same \
         claim the fetch above disproves",
        result.bytes_read
    );
}

// ---------------------------------------------------------------------------
// Finding 2: one structural walk, not two
// ---------------------------------------------------------------------------

/// A verify walks the archive's directories once.
///
/// `self_check` and `addressed_tiles` are two questions about one walk, asked
/// ten lines apart under a run lock that guarantees the archive cannot change
/// between them. Free on a local file; over an injected transport it is two
/// full sets of round trips, and above roughly 262144 tiles the leaf cache
/// evicts between them so the second walk refetches what the first one read.
#[test]
#[cfg_attr(miri, ignore)]
fn a_verify_walks_the_archive_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (plan, pyramid) = served(dir.path());

    pyramid_verify(&pyramid, &plan, Some(TileFormat::Png), &NoopObserver)
        .expect("a good archive verifies against its own plan");

    let walks = header_fetches(&pyramid);
    assert!(
        walks > 0,
        "the verify never read the header, so it never walked the archive at \
         all and this cell is no longer about walking it twice"
    );
    assert_eq!(
        walks, 1,
        "the verify walked the archive {walks} times; one walk answers both \
         the structural check and the addressed count"
    );
}
