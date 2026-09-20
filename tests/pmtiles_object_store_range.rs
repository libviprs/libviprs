//! The read-side transport seam: `ObjectStore` ranges under a PMTiles reader
//! (issue #1121).
//!
//! `pmtiles::Reader` has always been generic over [`RangeReader`], and the
//! write side has always taken an injected [`ObjectStore`]. This suite covers
//! the piece that joins them, [`ObjectStoreRangeReader`], plus the two
//! `RangeReader` impls for `Box<R>` and `Arc<R>` that make a runtime-chosen
//! backend usable in a generic position at all.
//!
//! # Every test double in the tree returns exactly the bytes it was asked for
//!
//! That is the trap this file exists for. `Sized` in `src/pmtiles/reader.rs`
//! and `InMemory` in `src/pmtiles/range.rs` both slice the exact range out of
//! a `Vec`, so a bridge that forwarded blindly would pass every assertion
//! either of them can make. A real transport does none of that reliably:
//!
//! * a server that ignores `Range` answers **200 with the whole object**, and
//!   a bridge with no length check hands the reader a 40 GB body. The first
//!   127 bytes of it are a perfectly valid header, `Header::try_decode`
//!   accepts them, and the reader then inflates whatever happens to sit at the
//!   root offset. Nothing errors. [`WholeObjectStore`] is that server, and the
//!   positive control below decodes those 127 bytes on purpose to show the
//!   refusal is doing real work rather than tripping over a malformed header;
//! * a connection that drops mid-body returns **fewer bytes than the range**,
//!   which is a directory page that stops in the middle of a column.
//!   [`TruncatingStore`] is that connection;
//! * a backend that cannot cheaply answer `size` is legitimate, and `None` is
//!   the legal `RangeReader` answer for it. That is exactly why the default on
//!   `ObjectStore::size` has to be a **refusal** and the bridge may map only
//!   [`SinkError::Unsupported`] to `Ok(None)`. `None` disables the four
//!   `SectionOutOfBounds` checks in `Reader::try_new`, so a store whose `size`
//!   failed transiently would silently buy a weaker reader instead of an
//!   error. [`sizeless`] and [`BrokenSizeStore`] are the two halves of that,
//!   and [`what_a_missing_size_costs_on_an_archive_that_is_actually_broken`]
//!   prices it against archives that really are corrupt.
//!
//! # The request count is pinned because it is a number, not a property
//!
//! Opening an archive is one `size` and two `get_range`: the 127-byte header,
//! then the root at the header's own offset and length. Two round trips where
//! the spec's 16 KiB rule exists to make it one is a known cost that #1119
//! deliberately left for a separate change, so [`RecordingStore`] pins it.
//! That turns a prefetch into a deliberate number change rather than a silent
//! one, and it catches the off-by-one somebody writes thinking in HTTP's
//! inclusive `bytes=offset-(offset+len-1)`: the ranges are asserted against
//! the header's own `root_offset` and `root_length`, read out of the golden at
//! run time rather than transcribed.

#![cfg(feature = "object-store-sink")]

use std::error::Error;
use std::io;
use std::sync::Arc;
use std::sync::Mutex;

use libviprs::pmtiles::header::HEADER_BYTES;
use libviprs::pmtiles::{
    Header, ObjectStoreRangeReader, PmTilesError, RangeReader, Reader, TileType,
};
use libviprs::pyramid_reader::{PmTilesPyramidReader, PyramidReader};
use libviprs::sink::{SinkError, TileFormat};
use libviprs::sink_object_store::ObjectStore;

#[path = "common/pmtiles_oracle.rs"]
mod oracle;

/// The golden this file reads through every backend it builds.
///
/// One archive go-pmtiles wrote, sha256-checked on every load, so a range that
/// comes back wrong is wrong against bytes this crate did not produce.
const RASTER: &str = "raster-z0z2.pmtiles";

fn golden() -> Vec<u8> {
    oracle::golden(RASTER, oracle::RASTER_GOLDEN_SHA256)
}

/// The top-right tile of the top level, which is the **last** blob in the
/// file. Chosen so a test about truncation has something that really is past
/// the cut rather than something that happens to be.
const LAST_TILE: (u8, u32, u32) = (2, 3, 3);

// ---------------------------------------------------------------------------
// The doubles, all of which sit BELOW the bridge and misbehave like transports
// ---------------------------------------------------------------------------

/// The exact-range read every honest backend performs, and the refusal every
/// honest backend gives for a range past the end (an S3 416).
fn exact(bytes: &[u8], offset: u64, len: usize) -> Result<Vec<u8>, SinkError> {
    let start = usize::try_from(offset)
        .map_err(|_| SinkError::Other(format!("offset {offset} does not fit this machine")))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| SinkError::Other("range end overflows".to_string()))?;
    bytes
        .get(start..end)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| SinkError::Io(io::Error::from(io::ErrorKind::UnexpectedEof)))
}

/// A store that only writes, so it inherits every defaulted method.
///
/// This is the shape the defaults are written for, and the one that proves
/// `size` refuses rather than answering `Ok(None)`.
struct PutOnlyStore;

impl ObjectStore for PutOnlyStore {
    fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
        Ok(())
    }
}

/// A well-behaved read-only backend: exact ranges, and it knows its size.
struct ExactStore(Vec<u8>);

impl ObjectStore for ExactStore {
    fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
        Err(SinkError::Unsupported("this double only reads".into()))
    }

    fn get_range(&self, _key: &str, offset: u64, len: usize) -> Result<Vec<u8>, SinkError> {
        exact(&self.0, offset, len)
    }

    fn size(&self, _key: &str) -> Result<Option<u64>, SinkError> {
        Ok(Some(self.0.len() as u64))
    }
}

/// Exact ranges, and no `size` at all: it inherits the defaulted refusal.
///
/// A streaming HTTP backend that skips the HEAD request is precisely this, and
/// it is legitimate. What it costs is measured further down.
struct SizelessStore(Vec<u8>);

impl ObjectStore for SizelessStore {
    fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
        Err(SinkError::Unsupported("this double only reads".into()))
    }

    fn get_range(&self, _key: &str, offset: u64, len: usize) -> Result<Vec<u8>, SinkError> {
        exact(&self.0, offset, len)
    }
}

fn sizeless(bytes: Vec<u8>) -> ObjectStoreRangeReader {
    ObjectStoreRangeReader::new(Arc::new(SizelessStore(bytes)), "archive.pmtiles")
}

fn sized(bytes: Vec<u8>) -> ObjectStoreRangeReader {
    ObjectStoreRangeReader::new(Arc::new(ExactStore(bytes)), "archive.pmtiles")
}

/// A server that ignores `Range` and answers 200 with the whole object.
///
/// The ranges it is asked for are recorded so a test can prove the reader did
/// ask for a short one and got a long one, rather than the store simply never
/// having been called.
struct WholeObjectStore(Vec<u8>);

impl ObjectStore for WholeObjectStore {
    fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
        Err(SinkError::Unsupported("this double only reads".into()))
    }

    fn get_range(&self, _key: &str, _offset: u64, _len: usize) -> Result<Vec<u8>, SinkError> {
        Ok(self.0.clone())
    }

    fn size(&self, _key: &str) -> Result<Option<u64>, SinkError> {
        Ok(Some(self.0.len() as u64))
    }
}

/// A connection that drops half way through the body.
struct TruncatingStore(Vec<u8>);

impl ObjectStore for TruncatingStore {
    fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
        Err(SinkError::Unsupported("this double only reads".into()))
    }

    fn get_range(&self, _key: &str, offset: u64, len: usize) -> Result<Vec<u8>, SinkError> {
        let full = exact(&self.0, offset, len)?;
        let half = len / 2;
        Ok(full[..half].to_vec())
    }

    fn size(&self, _key: &str) -> Result<Option<u64>, SinkError> {
        Ok(Some(self.0.len() as u64))
    }
}

/// Exact ranges, but `size` fails the way a network call fails.
///
/// The sharpest double in the file. A bridge that wrote `.ok().flatten()`
/// passes every other test here and fails only this one, because
/// `.ok().flatten()` turns a transient failure into "unknown size", which is a
/// legal answer that silently disables four bounds checks.
struct BrokenSizeStore {
    bytes: Vec<u8>,
    /// The failure `size` reports. `Io` is the transport shape; `Other` is
    /// here so the test can show the mapping is not "the `Unsupported` variant
    /// happens to be the one I tried".
    failure: fn() -> SinkError,
}

impl ObjectStore for BrokenSizeStore {
    fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
        Err(SinkError::Unsupported("this double only reads".into()))
    }

    fn get_range(&self, _key: &str, offset: u64, len: usize) -> Result<Vec<u8>, SinkError> {
        exact(&self.bytes, offset, len)
    }

    fn size(&self, _key: &str) -> Result<Option<u64>, SinkError> {
        Err((self.failure)())
    }
}

/// Exact ranges, and it writes down every call it was asked to make.
struct RecordingStore {
    bytes: Vec<u8>,
    ranges: Mutex<Vec<(u64, usize)>>,
    sizes: Mutex<usize>,
    keys: Mutex<Vec<String>>,
}

impl RecordingStore {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            ranges: Mutex::new(Vec::new()),
            sizes: Mutex::new(0),
            keys: Mutex::new(Vec::new()),
        }
    }

    fn ranges(&self) -> Vec<(u64, usize)> {
        self.ranges
            .lock()
            .expect("the recorder is not poisoned")
            .clone()
    }

    fn size_calls(&self) -> usize {
        *self.sizes.lock().expect("the recorder is not poisoned")
    }

    fn keys(&self) -> Vec<String> {
        self.keys
            .lock()
            .expect("the recorder is not poisoned")
            .clone()
    }
}

impl ObjectStore for RecordingStore {
    fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
        Err(SinkError::Unsupported("this double only reads".into()))
    }

    fn get_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>, SinkError> {
        self.ranges
            .lock()
            .expect("the recorder is not poisoned")
            .push((offset, len));
        self.keys
            .lock()
            .expect("the recorder is not poisoned")
            .push(key.to_string());
        exact(&self.bytes, offset, len)
    }

    fn size(&self, key: &str) -> Result<Option<u64>, SinkError> {
        *self.sizes.lock().expect("the recorder is not poisoned") += 1;
        self.keys
            .lock()
            .expect("the recorder is not poisoned")
            .push(key.to_string());
        Ok(Some(self.bytes.len() as u64))
    }
}

/// A `RangeReader` that is not an `ObjectStore` at all, for the `Box` and
/// `Arc` impls.
struct InMemory(Vec<u8>);

impl RangeReader for InMemory {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let start = usize::try_from(offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "end"))?;
        self.0
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))
    }

    fn size(&self) -> io::Result<Option<u64>> {
        Ok(Some(self.0.len() as u64))
    }
}

/// Walk an error's `source()` chain looking for a typed [`SinkError`].
///
/// `tests/error_source_typing.rs` sets the rule this checks: a typed error must
/// not be laundered into a string on its way up. A backend failure crosses two
/// layers here (`SinkError` into `io::Error` into [`PmTilesError`]) and the
/// concrete variant has to survive both.
fn sink_error_in_chain<'a>(err: &'a (dyn Error + 'static)) -> Option<&'a SinkError> {
    let mut current = Some(err);
    while let Some(link) = current {
        if let Some(sink) = link.downcast_ref::<SinkError>() {
            return Some(sink);
        }
        current = link.source();
    }
    None
}

// ---------------------------------------------------------------------------
// The defaults on the trait
// ---------------------------------------------------------------------------

/// A write-only backend refuses both new methods by name, and `size` refuses
/// rather than answering `Ok(None)`.
///
/// `Ok(None)` is the single most tempting default here and it is the wrong
/// one, because it is a **legal** `RangeReader` answer. A store that inherited
/// it would hand every reader built over it an archive with the four
/// `SectionOutOfBounds` checks quietly switched off, and nothing anywhere
/// would report a problem. The refusal is what makes a backend say out loud
/// that it has not implemented a HEAD.
#[test]
fn the_defaulted_read_methods_refuse_and_size_does_not_answer_ok_none() {
    let store = PutOnlyStore;

    match store.get_range("k", 0, 127) {
        Err(SinkError::Unsupported(message)) => {
            assert!(
                message.contains("get_range"),
                "the refusal must name the operation, got: {message}"
            );
        }
        other => panic!("a write-only backend must refuse get_range, got {other:?}"),
    }

    match store.size("k") {
        Err(SinkError::Unsupported(message)) => {
            assert!(
                message.contains("size"),
                "the refusal must name the operation, got: {message}"
            );
        }
        Ok(None) => panic!(
            "the default `size` must REFUSE. `Ok(None)` is a legal RangeReader \
             answer, so a backend inheriting it would silently disable the four \
             SectionOutOfBounds checks in Reader::try_new"
        ),
        other => panic!("a write-only backend must refuse size, got {other:?}"),
    }

    // The positive control: the one method a write-only backend does
    // implement still works, so the refusals above are about the defaults and
    // not about the double being broken.
    assert!(store.put("k", b"x").is_ok());
}

// ---------------------------------------------------------------------------
// The two lines that carry the bridge
// ---------------------------------------------------------------------------

/// A store that returns more bytes than the range is refused, and the control
/// shows what accepting it would have meant.
///
/// The control is the whole point. Those 127 bytes decode to a valid header on
/// their own, so a bridge with no length check does not fail anywhere: it
/// reads a valid header out of a 40 GB body and goes on to inflate whatever
/// sits at the root offset.
#[test]
fn a_backend_that_ignores_the_range_is_refused_rather_than_read_from_the_front() {
    let bytes = golden();

    // The control, first, because it is what makes the refusal meaningful.
    assert!(
        Header::try_decode(&bytes[..HEADER_BYTES]).is_ok(),
        "the control: the first 127 bytes of a whole-object body ARE a valid \
         header, which is why a bridge with no length check cannot notice"
    );

    let reader =
        ObjectStoreRangeReader::new(Arc::new(WholeObjectStore(bytes.clone())), "archive.pmtiles");

    let err = reader
        .read_range(0, HEADER_BYTES)
        .expect_err("a body longer than the range must be refused");
    let text = err.to_string();
    assert!(
        text.contains(&bytes.len().to_string()) && text.contains(&HEADER_BYTES.to_string()),
        "the refusal must name what it asked for and what it got, got: {text}"
    );

    // And it stops the reader rather than producing a reader over the wrong
    // bytes.
    assert!(
        Reader::try_new(reader).is_err(),
        "an archive opened over a backend that ignores Range must not open"
    );
}

/// A store that returns fewer bytes than the range is refused too.
///
/// Same one line, other direction. A short body is a directory page that stops
/// in the middle of a column, and `Vec<u8>` carries no length the caller did
/// not ask for, so nothing downstream can tell.
#[test]
fn a_truncated_body_is_refused_rather_than_handed_back_short() {
    let bytes = golden();
    let reader =
        ObjectStoreRangeReader::new(Arc::new(TruncatingStore(bytes.clone())), "archive.pmtiles");

    let err = reader
        .read_range(0, HEADER_BYTES)
        .expect_err("a short body must be refused");
    let text = err.to_string();
    assert!(
        text.contains(&HEADER_BYTES.to_string()) && text.contains(&(HEADER_BYTES / 2).to_string()),
        "the refusal must name both lengths, got: {text}"
    );

    assert!(
        Reader::try_new(reader).is_err(),
        "an archive opened over a truncating backend must not open"
    );

    // The positive control: the same golden through an exact backend opens and
    // serves, so the refusals above are about the truncation.
    let good = Reader::try_new(sized(bytes)).expect("the golden opens over an exact backend");
    let (z, x, y) = LAST_TILE;
    assert!(good.get_tile(z, x, y).expect("a lookup").is_some());
}

/// A zero-length range is the empty answer and never a request.
///
/// A round trip for nothing is a round trip, and the reader asks for a
/// zero-length section whenever an archive has no leaves.
#[test]
fn a_zero_length_range_is_answered_without_touching_the_backend() {
    let store = Arc::new(RecordingStore::new(golden()));
    let reader = ObjectStoreRangeReader::new(store.clone(), "archive.pmtiles");

    assert!(
        reader
            .read_range(324, 0)
            .expect("empty is not an error")
            .is_empty()
    );
    assert!(
        store.ranges().is_empty(),
        "a zero-length range must not become a request, got {:?}",
        store.ranges()
    );
}

// ---------------------------------------------------------------------------
// size: only Unsupported means "unknown"
// ---------------------------------------------------------------------------

/// A backend that cannot say how big the object is still serves the archive.
#[test]
fn a_backend_with_no_size_opens_the_archive_and_serves_tiles() {
    let reader = Reader::try_new(sizeless(golden())).expect("an archive with no known size opens");

    assert_eq!(
        reader.archive_size(),
        None,
        "a refused size is reported as unknown, which is the legal RangeReader answer"
    );
    assert_eq!(reader.tile_format(), TileType::Png);

    let (z, x, y) = LAST_TILE;
    let tile = reader
        .get_tile(z, x, y)
        .expect("a lookup over a sizeless backend")
        .expect("the golden holds the top-right tile of its top level");
    assert!(!tile.is_empty());

    // And it is the same tile the sized backend serves, so "it opened" is not
    // standing in for "it read the right thing".
    let sized_reader = Reader::try_new(sized(golden())).expect("the sized backend opens");
    assert_eq!(
        Some(tile),
        sized_reader.get_tile(z, x, y).expect("a lookup"),
        "the two backends must serve the same bytes"
    );
}

/// A `size` that failed is a failure, not an unknown size.
///
/// This is the `.ok().flatten()` killer. That spelling passes every other test
/// in this file: the archive still opens, the tiles still serve, and the only
/// thing that changed is that four bounds checks are gone and nobody was told.
#[test]
fn a_transient_size_failure_stops_the_open_rather_than_reading_as_unknown() {
    for failure in [
        (|| SinkError::Io(io::Error::from(io::ErrorKind::ConnectionReset))) as fn() -> SinkError,
        || SinkError::Other("the bucket answered 503".to_string()),
    ] {
        let reader = ObjectStoreRangeReader::new(
            Arc::new(BrokenSizeStore {
                bytes: golden(),
                failure,
            }),
            "archive.pmtiles",
        );

        // The bridge's own answer first, so the assertion below is not resting
        // on whatever `Reader::try_new` happens to do with it.
        assert!(
            reader.size().is_err(),
            "a failed size must stay a failure; only SinkError::Unsupported means unknown"
        );

        let err = Reader::try_new(reader)
            .expect_err("a reader must not open over a backend whose size failed");
        assert!(
            matches!(err, PmTilesError::Io(_)),
            "a transport failure surfaces as I/O, got {err:?}"
        );
    }

    // The control, and it is the one that makes the loop above mean something:
    // the SAME store shape with the SAME exact ranges opens fine when `size`
    // refuses with `Unsupported` instead of failing.
    assert!(
        Reader::try_new(sizeless(golden())).is_ok(),
        "the control: only SinkError::Unsupported reads as an unknown size"
    );
}

/// What a missing size costs, priced against archives that really are broken.
///
/// Both corruptions are the ones `tests/pmtiles_reader.rs` already pins: an
/// archive truncated to 900 bytes, and the root offset of 999999 in an
/// 1878-byte file that makes `pmtiles verify` nil-dereference. Through a
/// size-answering backend each is refused by the bounds check. Through a
/// size-refusing one the first opens, because the check that would have caught
/// it needs a total, and only fails when a tile past the cut is fetched.
#[test]
fn what_a_missing_size_costs_on_an_archive_that_is_actually_broken() {
    // ---- truncated to 900 bytes -------------------------------------------
    let mut truncated = golden();
    truncated.truncate(900);

    let err = Reader::try_new(sized(truncated.clone()))
        .expect_err("a truncated archive is refused when the size is known");
    assert!(
        matches!(err, PmTilesError::SectionOutOfBounds { .. }),
        "the bounds check is what refuses it, got {err:?}"
    );

    let reader = Reader::try_new(sizeless(truncated))
        .expect("without a size there is no total to check the sections against");
    assert_eq!(reader.archive_size(), None);

    // The control: the tile this test then asks for really does live past the
    // cut, read off the archive's own root rather than assumed.
    let header_tile_data_offset = reader.header().tile_data_offset;
    let last = *reader
        .root_entries()
        .last()
        .expect("the golden's root addresses every tile");
    assert!(
        header_tile_data_offset + last.offset + u64::from(last.length) > 900,
        "the control: the last blob must really be past the truncation point"
    );

    let (z, x, y) = LAST_TILE;
    assert!(
        reader.get_tile(z, x, y).is_err(),
        "the failure is deferred to the fetch rather than prevented at open"
    );

    // ---- root offset past the end -----------------------------------------
    let mut moved_root = golden();
    assert_eq!(
        moved_root.len(),
        1878,
        "the golden is the one the pin names"
    );
    moved_root[8..16].copy_from_slice(&999_999u64.to_le_bytes());

    let err = Reader::try_new(sized(moved_root.clone()))
        .expect_err("a root past the end is refused when the size is known");
    assert!(
        matches!(err, PmTilesError::SectionOutOfBounds { .. }),
        "the bounds check is what refuses it, got {err:?}"
    );

    // Without a size the bounds check cannot fire, and what stops it is the
    // root's own 16 KiB budget: a second line of defence that happens to cover
    // this one shape and covers nothing about the other three sections.
    let err =
        Reader::try_new(sizeless(moved_root)).expect_err("the root budget still refuses this one");
    assert!(
        matches!(err, PmTilesError::RootDirectoryTooLarge { .. }),
        "a different, weaker refusal than the bounds check, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// What an open actually asks for
// ---------------------------------------------------------------------------

/// One `size` and exactly two `get_range` per open, at the header's own
/// offsets.
///
/// The second range is the one worth pinning. HTTP's `Range` header is
/// inclusive on both ends, so somebody translating `(offset, len)` into
/// `bytes=offset-(offset+len)` reads one byte too many, and somebody
/// translating it back reads one too few. Asserting the pair against
/// `root_offset` and `root_length` as the header spells them leaves that
/// nowhere to hide.
#[test]
fn opening_an_archive_is_one_size_and_two_ranges_at_the_header_s_own_offsets() {
    let bytes = golden();
    let header = Header::try_decode(&bytes[..HEADER_BYTES]).expect("the golden's header decodes");
    assert!(
        header.root_length > 0,
        "the control: an archive with an empty root would make the second \
         range a zero-length one and this pin vacuous"
    );

    let store = Arc::new(RecordingStore::new(bytes));
    let reader = Reader::try_new(ObjectStoreRangeReader::new(
        store.clone(),
        "tiles/drawing.pmtiles",
    ))
    .expect("the golden opens");

    assert_eq!(
        store.size_calls(),
        1,
        "the size is asked for once at open and never again"
    );
    assert_eq!(
        store.ranges(),
        vec![
            (0, HEADER_BYTES),
            (
                header.root_offset,
                usize::try_from(header.root_length).expect("a root fits a usize")
            ),
        ],
        "an open is the header and the root, at the header's own numbers"
    );

    // Every call carried the key the reader was built with, rather than a
    // prefix or a normalised copy of it.
    assert!(
        store.keys().iter().all(|k| k == "tiles/drawing.pmtiles"),
        "every call must carry the key verbatim, got {:?}",
        store.keys()
    );

    // And the archive is usable, so the counts above are the counts of a real
    // open rather than of one that failed early.
    let (z, x, y) = LAST_TILE;
    assert!(reader.get_tile(z, x, y).expect("a lookup").is_some());
    assert_eq!(
        store.size_calls(),
        1,
        "a tile fetch must not ask for the size again"
    );
    assert_eq!(
        store.ranges().len(),
        3,
        "a clustered archive with no leaves serves a tile in one more range"
    );
}

// ---------------------------------------------------------------------------
// The accessors, and the typed error chain
// ---------------------------------------------------------------------------

#[test]
fn the_bridge_reports_the_store_and_the_key_it_was_built_with() {
    let store: Arc<dyn ObjectStore> = Arc::new(ExactStore(golden()));
    let reader = ObjectStoreRangeReader::new(store.clone(), "tiles/drawing.pmtiles");

    assert_eq!(reader.key(), "tiles/drawing.pmtiles");
    assert!(
        Arc::ptr_eq(reader.store(), &store),
        "the bridge must hand back the backend it was given, not a copy"
    );
}

/// A backend failure keeps its typed `SinkError` all the way up.
///
/// The failure crosses two error types on its way to the caller
/// (`SinkError` into `io::Error` into `PmTilesError`), and each crossing is a
/// chance to flatten it into a sentence. `tests/error_source_typing.rs` is the
/// file that made that a rule; this is the same rule at the transport seam.
#[test]
fn an_object_store_failure_keeps_its_typed_sink_error_in_the_source_chain() {
    let reader = ObjectStoreRangeReader::new(
        Arc::new(BrokenSizeStore {
            bytes: golden(),
            failure: || SinkError::Io(io::Error::from(io::ErrorKind::ConnectionReset)),
        }),
        "archive.pmtiles",
    );

    let err = Reader::try_new(reader).expect_err("the size failed, so the open fails");
    let sink = sink_error_in_chain(&err).expect(
        "the concrete SinkError must survive into the source() chain rather than \
         being stringified",
    );
    assert!(
        matches!(sink, SinkError::Io(_)),
        "and it must be the variant the backend reported, got {sink:?}"
    );

    // The read path too, not only the size path.
    let reader =
        ObjectStoreRangeReader::new(Arc::new(SizelessStore(vec![0u8; 10])), "archive.pmtiles");
    let err = reader
        .read_range(0, HEADER_BYTES)
        .expect_err("ten bytes is not a header");
    let sink = sink_error_in_chain(&err).expect("a read failure carries its SinkError too");
    assert!(matches!(sink, SinkError::Io(_)), "got {sink:?}");
}

// ---------------------------------------------------------------------------
// Box and Arc in a generic position
// ---------------------------------------------------------------------------

/// `Box<dyn RangeReader>` and `Arc<dyn RangeReader>` are `RangeReader`.
///
/// This does not compile without the two impls, which is the point: the
/// existing coercion test in `src/pmtiles/range.rs` only ever calls
/// `read_range` **through** a box, and method auto-deref makes that work
/// whether or not `Box<dyn RangeReader>` implements the trait. A generic
/// position is what asks the real question, and a CLI picking a backend from a
/// URI scheme at runtime has nothing but a trait object to hand over.
#[test]
fn a_boxed_and_an_arced_range_reader_can_open_an_archive() {
    let boxed: Box<dyn RangeReader> = Box::new(InMemory(golden()));
    let from_box = Reader::try_new(boxed).expect("a boxed trait object opens an archive");

    let arced: Arc<dyn RangeReader> = Arc::new(InMemory(golden()));
    let from_arc = Reader::try_new(arced).expect("an arced trait object opens an archive");

    let (z, x, y) = LAST_TILE;
    let a = from_box
        .get_tile(z, x, y)
        .expect("a lookup")
        .expect("a tile");
    let b = from_arc
        .get_tile(z, x, y)
        .expect("a lookup")
        .expect("a tile");
    assert_eq!(a, b, "the two wrappers must read the same bytes");

    // A sized implementation behind a box too, so the impls are not secretly
    // specific to `dyn`.
    let sized_box = Reader::try_new(Box::new(InMemory(golden())))
        .expect("a box around a concrete reader opens too");
    assert_eq!(sized_box.archive_size(), Some(1878));
}

// ---------------------------------------------------------------------------
// PmTilesPyramidReader over a transport
// ---------------------------------------------------------------------------

/// The pyramid reader opens over an injected store, and it still opens over a
/// path with no type annotation anywhere.
///
/// The defaulted type parameter is what keeps the existing 19 call sites
/// compiling, so the `try_open` half of this test is the regression guard for
/// the change rather than a restatement of an old test.
#[test]
fn the_pyramid_reader_opens_over_an_injected_store() {
    let store: Arc<dyn ObjectStore> = Arc::new(ExactStore(golden()));
    let pyramid = PmTilesPyramidReader::try_from_object_store(store, "tiles/drawing.pmtiles")
        .expect("the golden opens through an object store");

    let described = pyramid.describe().expect("it describes itself");
    assert_eq!(described.min_level, 0);
    assert_eq!(described.max_level, 2);
    // A foreign archive carries no vnd.libviprs block, so the two fields that
    // only libviprs records are honestly unknown rather than invented.
    assert_eq!(described.tile_size, None);
    assert_eq!(described.layout, None);
    assert_eq!(described.format, Some(TileFormat::Png));

    let coord = libviprs::planner::TileCoord {
        level: 2,
        col: 3,
        row: 3,
    };
    assert!(
        pyramid.tile(coord).expect("a lookup").is_some(),
        "the pyramid reader must serve the tile the archive holds"
    );

    // And it reads through the trait object the trait exists for.
    let boxed: Box<dyn PyramidReader> = Box::new(
        PmTilesPyramidReader::try_from_object_store(
            Arc::new(ExactStore(golden())),
            "tiles/drawing.pmtiles",
        )
        .expect("the golden opens"),
    );
    assert!(boxed.tile(coord).expect("a lookup").is_some());
}

/// The reader underneath is reachable whichever backend it was built over.
#[test]
fn from_reader_and_reader_work_for_a_non_file_backend() {
    let inner = Reader::try_new(sized(golden())).expect("the golden opens");
    let pyramid = PmTilesPyramidReader::from_reader(inner);

    assert_eq!(pyramid.reader().archive_size(), Some(1878));
    assert_eq!(pyramid.reader().tile_format(), TileType::Png);
    assert_eq!(pyramid.reader().source().key(), "archive.pmtiles");

    // The boxed instantiation #1119 names as the escape hatch for anyone who
    // wants a runtime-chosen backend without a type parameter.
    let boxed: Reader<Box<dyn RangeReader>> =
        Reader::try_new(Box::new(InMemory(golden())) as Box<dyn RangeReader>)
            .expect("a boxed backend opens");
    let pyramid = PmTilesPyramidReader::from_reader(boxed);
    assert_eq!(
        pyramid.describe().expect("it describes itself").max_level,
        2
    );
}
