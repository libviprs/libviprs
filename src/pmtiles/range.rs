//! Fetching byte ranges, and the local-file implementation of it.
//!
//! Everything a PMTiles reader does is a ranged read: 127 bytes for the
//! header, a few hundred for the root directory, one more for a leaf, one for
//! the tile. It never scans and it never reads the whole file, which is what
//! makes a 40 GB archive on an object store behave like a local one.
//!
//! [`RangeReader`] is the single seam that makes that true for any transport.
//! Two implementations live here: the local one, [`FileRangeReader`], and
//! [`ObjectStoreRangeReader`], which bridges the trait onto the injected
//! [`ObjectStore`](crate::sink_object_store::ObjectStore) the write side
//! already takes. An HTTP or S3 backend is neither of those. It is
//! [`RangeReader::read_range`] written by somebody else, plus
//! [`RangeReader::size`] if their transport can cheaply answer it, and it
//! needs no change to the reader or the writer above it.
//!
//! # Object-safe, on purpose
//!
//! The trait takes no generic methods and returns no `Self`, so
//! `Box<dyn RangeReader>` works. That matters more than it looks: a CLI that
//! picks a backend from a URI scheme at runtime needs a trait object, and a
//! generic-only design pushes that decision into the type system where a
//! command line cannot reach it. There is a test whose only job is to coerce
//! one, because object safety is the kind of property that a single
//! innocent-looking method signature quietly removes.
//!
//! Object safety is only half of what a runtime-chosen backend needs, though,
//! and the other half was missing until #1121. `Reader<R>` takes `R` by value
//! under an `R: RangeReader` bound, so a trait object has to satisfy that
//! bound *through its wrapper*, and `Box<dyn RangeReader>` did not implement
//! the trait at all. Nothing noticed, because the coercion test only ever
//! called `read_range` **through** the box and method auto-deref resolves that
//! either way. So [`RangeReader`] is now implemented for `Box<R>` and
//! `Arc<R>`, for `R: ?Sized`, which is what makes `Reader<Box<dyn
//! RangeReader>>` a thing you can write.
//!
//! # Positional reads, not seek-then-read
//!
//! [`FileRangeReader`] uses `pread` on Unix rather than seeking. A seek plus a
//! read is two operations against one shared file cursor, so two threads
//! reading two tiles from one archive interleave and hand each other the wrong
//! bytes unless a lock serialises them. A positional read carries its offset
//! in the call and needs no shared state at all, which is why the trait takes
//! `&self` and requires `Send + Sync`: the reader above it is expected to be
//! shared across the engine's worker threads.
//!
//! Targets without `pread` fall back to a `Mutex` around seek-then-read, which
//! is correct and slower. Only the Unix path is exercised in CI.

use std::fs::File;
use std::io;
use std::path::Path;

use crate::pmtiles::PmTilesError;

/// Somewhere bytes can be fetched from by offset and length.
///
/// Implementations must be safe to call from several threads at once on one
/// instance, which is why `Send + Sync` is a supertrait rather than a bound
/// applied at the use site: a reader shared across the engine's workers should
/// not have to care which backend it was handed.
pub trait RangeReader: Send + Sync {
    /// Read exactly `len` bytes starting at `offset`.
    ///
    /// **A short read is an error, never a truncated buffer.** The return type
    /// carries no length the caller did not already ask for, so an
    /// implementation that returned fewer bytes would be handing back a
    /// directory page that silently stops in the middle of a column. A request
    /// that runs past the end of the underlying object must fail.
    ///
    /// The error is an `io::Error` rather than a [`PmTilesError`] so that a
    /// backend can implement this without depending on the format layer's
    /// error type. [`PmTilesError`] converts from it, so a `?` in the reader
    /// picks it up.
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>>;

    /// The total size of the object, when it is known cheaply.
    ///
    /// A reader uses it to bounds-check the header's offsets against the real
    /// archive before trusting them. The default is `None` rather than a
    /// required method, because a streaming HTTP backend may genuinely not
    /// know, and a backend that had to invent a number would be worse than one
    /// that says it does not know: a wrong size turns a bounds check into a
    /// false refusal.
    fn size(&self) -> io::Result<Option<u64>> {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Pointer wrappers
// ---------------------------------------------------------------------------

/// A boxed reader is a reader.
///
/// `?Sized` is the whole point: without it this covers `Box<FileRangeReader>`,
/// which nobody has ever wanted, and not `Box<dyn RangeReader>`, which is the
/// only shape a CLI choosing a backend from a URI scheme can produce.
///
/// This is a permanent coherence commitment. Nobody outside this crate can
/// ever write their own `impl RangeReader for Box<TheirType>`, because the
/// blanket impl here already covers it. That is the right trade for a trait
/// whose reason to exist is being usable behind a pointer, and it is the same
/// bargain `std` makes for `Read`, `Write` and `Iterator`.
impl<R: RangeReader + ?Sized> RangeReader for Box<R> {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        (**self).read_range(offset, len)
    }

    fn size(&self) -> io::Result<Option<u64>> {
        (**self).size()
    }
}

/// An `Arc`'d reader is a reader, for the same reasons as [`Box<R>`].
///
/// Worth having separately because the engine shares one reader across its
/// workers, and sharing is what `Arc` is for: an `Arc<dyn RangeReader>` can go
/// into a `Reader` and still be held elsewhere, where a `Box` has to be given
/// away.
impl<R: RangeReader + ?Sized> RangeReader for std::sync::Arc<R> {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        (**self).read_range(offset, len)
    }

    fn size(&self) -> io::Result<Option<u64>> {
        (**self).size()
    }
}

/// A [`RangeReader`] over a local file.
///
/// # Examples
///
/// ```no_run
/// use libviprs::pmtiles::{FileRangeReader, RangeReader};
///
/// let reader = FileRangeReader::try_open("drawing.pmtiles")?;
/// let header_bytes = reader.read_range(0, 127)?;
/// assert_eq!(&header_bytes[..7], b"PMTiles");
/// # Ok::<(), libviprs::pmtiles::PmTilesError>(())
/// ```
#[derive(Debug)]
pub struct FileRangeReader {
    file: File,
    len: u64,
    /// Only used where there is no positional read to call.
    #[cfg(not(unix))]
    cursor: std::sync::Mutex<()>,
}

impl FileRangeReader {
    /// Open a file for ranged reads.
    ///
    /// The length is taken once here rather than on every [`RangeReader::size`]
    /// call: an archive being read is not being written, and a `stat` per tile
    /// would be a syscall for a number that cannot change.
    pub fn try_open(path: impl AsRef<Path>) -> Result<Self, PmTilesError> {
        let file = File::open(path)?;
        Self::try_from_file(file)
    }

    /// Wrap a file that is already open.
    pub fn try_from_file(file: File) -> Result<Self, PmTilesError> {
        let len = file.metadata()?.len();
        Ok(Self {
            file,
            len,
            #[cfg(not(unix))]
            cursor: std::sync::Mutex::new(()),
        })
    }

    /// The file's length as measured when it was opened.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the file was empty when it was opened. An empty file is never a
    /// PMTiles archive, since the header alone is 127 bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl RangeReader for FileRangeReader {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }

        // Check against the length before allocating. `len` comes out of a
        // directory entry, so it is a number an attacker chooses, and
        // `Vec::with_capacity` on it would be an allocation of their size
        // before a single byte has been read. The bound is cheap and exact.
        let end = offset.checked_add(len as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "range end overflows u64")
        })?;
        if end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "range {offset}..{end} runs past the end of a {} byte file",
                    self.len
                ),
            ));
        }

        // Fallible allocation even after the bound above: the file really can
        // be larger than this process can allocate, and a pyramid generator
        // running near its memory budget should get an error rather than an
        // abort.
        let mut buf = Vec::new();
        buf.try_reserve_exact(len)
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        buf.resize(len, 0);

        self.read_exact_at_offset(&mut buf, offset)?;
        Ok(buf)
    }

    fn size(&self) -> io::Result<Option<u64>> {
        Ok(Some(self.len))
    }
}

#[cfg(unix)]
impl FileRangeReader {
    /// Positional read, no shared cursor, safe to call concurrently.
    fn read_exact_at_offset(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file.read_exact_at(buf, offset)
    }
}

#[cfg(not(unix))]
impl FileRangeReader {
    /// Seek then read, serialised, because there is no positional read here.
    /// The lock is what stops two concurrent reads from moving each other's
    /// file cursor between the seek and the read.
    fn read_exact_at_offset(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        use std::io::{Read, Seek, SeekFrom};
        let _guard = self
            .cursor
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut file = &self.file;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf)
    }
}

// ---------------------------------------------------------------------------
// ObjectStoreRangeReader — the transport seam (issue #1121)
// ---------------------------------------------------------------------------

/// A [`RangeReader`] over one object in an injected
/// [`ObjectStore`](crate::sink_object_store::ObjectStore).
///
/// The write side of this crate has taken an injected backend since #382, and
/// this is its counterpart: hand it a store and a key and a PMTiles archive
/// opens over whatever that store talks to. libviprs still ships no HTTP or S3
/// client and #1119 records that as a permanent decision, so the transport is
/// the caller's and the trait is the product.
///
/// # The two lines that carry this type
///
/// Everything else here is forwarding. These two are not, and both exist
/// because a real transport fails in ways no in-tree test double can:
///
/// * **[`read_range`](RangeReader::read_range) checks the length it got.** A
///   server that ignores `Range` answers 200 with the whole object, and
///   without this check the reader takes the first 127 bytes of a 40 GB body,
///   decodes a perfectly valid header out of them and goes on to inflate
///   whatever happens to sit at the root offset. Nothing errors. The same one
///   line catches the other direction, a connection that drops mid-body and
///   returns a directory page that stops in the middle of a column.
/// * **[`size`](RangeReader::size) maps only
///   [`SinkError::Unsupported`](crate::sink::SinkError::Unsupported) to
///   `Ok(None)`.** `None` means "this backend cannot cheaply say", which is
///   legal and which costs the reader the four `SectionOutOfBounds` checks in
///   [`Reader::try_new`](crate::pmtiles::Reader::try_new). A transient failure
///   is not that. Written as `.ok().flatten()` it would be, and the archive
///   would still open, still serve tiles, and quietly have lost its bounds
///   checks.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use libviprs::pmtiles::{ObjectStoreRangeReader, RangeReader};
/// use libviprs::sink::SinkError;
/// use libviprs::sink_object_store::ObjectStore;
///
/// /// A backend that happens to hold its bytes in memory. A real one puts
/// /// the same two methods on the wire.
/// struct InMemory(Vec<u8>);
///
/// impl ObjectStore for InMemory {
///     fn put(&self, _key: &str, _bytes: &[u8]) -> Result<(), SinkError> {
///         Err(SinkError::Unsupported("read-only".into()))
///     }
///     fn get_range(&self, _key: &str, offset: u64, len: usize) -> Result<Vec<u8>, SinkError> {
///         let start = offset as usize;
///         Ok(self.0[start..start + len].to_vec())
///     }
///     fn size(&self, _key: &str) -> Result<Option<u64>, SinkError> {
///         Ok(Some(self.0.len() as u64))
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let store = Arc::new(InMemory(b"PMTiles and then some".to_vec()));
/// let reader = ObjectStoreRangeReader::new(store, "tiles/drawing.pmtiles");
///
/// assert_eq!(reader.read_range(0, 7)?, b"PMTiles".to_vec());
/// assert_eq!(reader.size()?, Some(21));
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "object-store-sink")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store-sink")))]
pub struct ObjectStoreRangeReader {
    store: std::sync::Arc<dyn crate::sink_object_store::ObjectStore>,
    key: String,
}

#[cfg(feature = "object-store-sink")]
impl std::fmt::Debug for ObjectStoreRangeReader {
    /// Hand-written because `Arc<dyn ObjectStore>` is not `Debug` and the
    /// trait is not going to grow it: a backend is somebody else's type and
    /// requiring `Debug` of it would be this crate choosing their derives. The
    /// key is the half worth printing anyway, and it is what a
    /// `Reader { source: .. }` dump needs to be useful.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectStoreRangeReader")
            .field("key", &self.key)
            .field("store", &"<dyn ObjectStore>")
            .finish()
    }
}

#[cfg(feature = "object-store-sink")]
impl ObjectStoreRangeReader {
    /// Point a reader at one object in a store.
    ///
    /// Not `try_new`: nothing is contacted here. The first request is the
    /// header read the reader above it makes, which is also where a wrong key
    /// or an unreachable endpoint shows up.
    pub fn new(
        store: std::sync::Arc<dyn crate::sink_object_store::ObjectStore>,
        key: impl Into<String>,
    ) -> Self {
        Self {
            store: store.into(),
            key: key.into(),
        }
    }

    /// The backend underneath, for a caller that wants to reach it again.
    pub fn store(&self) -> &std::sync::Arc<dyn crate::sink_object_store::ObjectStore> {
        &self.store
    }

    /// The key every request carries, verbatim.
    pub fn key(&self) -> &str {
        &self.key
    }
}

/// A backend failure on its way through [`RangeReader`].
///
/// It exists so the concrete [`SinkError`](crate::sink::SinkError) survives
/// into the `source()` chain. `io::Error`'s own `source()` delegates to the
/// custom payload's source rather than handing back the payload, so wrapping
/// the `SinkError` directly would put it one layer out of reach of a
/// chain walk. `tests/error_source_typing.rs` is the file that made "a typed
/// error must not be laundered into a string" a rule here.
#[cfg(feature = "object-store-sink")]
#[derive(Debug, thiserror::Error)]
#[error("object store: {0}")]
struct ObjectStoreFailure(#[source] crate::sink::SinkError);

#[cfg(feature = "object-store-sink")]
impl ObjectStoreFailure {
    /// Turn a backend failure into the `io::Error` the trait returns, keeping
    /// the `ErrorKind` when the backend had one.
    ///
    /// The kind matters to a caller that retries: a `ConnectionReset` is worth
    /// another attempt and an `Unsupported` never is, and flattening both to
    /// `Other` would throw that away at the one boundary where it is known.
    fn into_io(error: crate::sink::SinkError) -> io::Error {
        let kind = match &error {
            crate::sink::SinkError::Io(inner) => inner.kind(),
            crate::sink::SinkError::Unsupported(_) => io::ErrorKind::Unsupported,
            _ => io::ErrorKind::Other,
        };
        io::Error::new(kind, Self(error))
    }
}

#[cfg(feature = "object-store-sink")]
impl RangeReader for ObjectStoreRangeReader {
    fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        // A zero-length range is the empty answer and never a request. The
        // reader asks for one whenever a section is empty, and a round trip
        // for nothing is still a round trip.
        if len == 0 {
            return Ok(Vec::new());
        }

        let got = self
            .store
            .get_range(&self.key, offset, len)
            .map_err(ObjectStoreFailure::into_io)?;

        // The line the whole type is for. Read the doc comment above before
        // deleting it: a body that is too long is a 200 where a 206 was asked
        // for, and the first 127 bytes of it decode to a valid header.
        if got.len() != len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "a ranged read of {len} bytes at offset {offset} from {} came back \
                     {} bytes long. A backend must answer exactly the range it was \
                     asked for: a longer body is a 200 where a 206 was asked for, and \
                     a shorter one is a truncated read",
                    self.key,
                    got.len()
                ),
            ));
        }

        Ok(got)
    }

    fn size(&self) -> io::Result<Option<u64>> {
        match self.store.size(&self.key) {
            Ok(size) => Ok(size),
            // The ONLY failure that reads as "unknown". A backend that has not
            // implemented a HEAD is a legitimate streaming transport and the
            // reader copes with it by skipping the section bounds checks.
            Err(crate::sink::SinkError::Unsupported(_)) => Ok(None),
            // Everything else is a failure and stays one. A 503 is not the
            // same statement as "I cannot tell you", and treating it as one
            // buys a weaker reader in exchange for an error nobody sees.
            Err(other) => Err(ObjectStoreFailure::into_io(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A file of 256 distinct bytes, so a range that comes back shifted by one
    /// is visible rather than being a repeat of the same byte.
    fn ramp_file() -> (tempfile::NamedTempFile, Vec<u8>) {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let payload: Vec<u8> = (0..=255u8).collect();
        file.write_all(&payload).expect("write");
        file.flush().expect("flush");
        (file, payload)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_range_comes_back_exactly_as_asked_for() {
        let (file, payload) = ramp_file();
        let reader = FileRangeReader::try_open(file.path()).unwrap();

        assert_eq!(reader.len(), 256);
        assert!(!reader.is_empty());
        assert_eq!(reader.size().unwrap(), Some(256));

        assert_eq!(reader.read_range(0, 4).unwrap(), payload[0..4]);
        assert_eq!(reader.read_range(127, 1).unwrap(), vec![127]);
        assert_eq!(reader.read_range(200, 56).unwrap(), payload[200..256]);
        assert_eq!(reader.read_range(255, 1).unwrap(), vec![255]);
        // A zero-length read is the empty answer, not an error: a directory
        // section can legitimately be empty.
        assert!(reader.read_range(0, 0).unwrap().is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_bound_is_on_offset_plus_length_not_on_length_alone() {
        // This is the exact shape of a live bug in the reference
        // implementation: `pmtiles verify` checks `length > fileSize` and
        // never `offset + length`, so a root directory offset of 999999 in an
        // 1878-byte archive walks straight past the check and segfaults in the
        // gzip reader. Checking the sum is what makes that a refusal here.
        let (file, _) = ramp_file();
        let reader = FileRangeReader::try_open(file.path()).unwrap();

        assert!(
            reader.read_range(999_999, 10).is_err(),
            "offset past the end"
        );
        assert!(
            reader.read_range(250, 10).is_err(),
            "the sum is past the end"
        );
        assert!(reader.read_range(256, 1).is_err(), "one byte past the end");

        // The positive control, and it is the one that matters: the same
        // length one byte earlier succeeds, so these refusals are about the
        // bound rather than about the length being large.
        assert_eq!(reader.read_range(246, 10).unwrap().len(), 10);

        // And the arithmetic cannot be made to wrap into a pass.
        assert!(reader.read_range(u64::MAX, 2).is_err());
        assert!(reader.read_range(u64::MAX - 1, usize::MAX).is_err());
        assert!(reader.read_range(0, usize::MAX).is_err());
    }

    /// The refusal has to come from the bound in [`FileRangeReader::read_range`],
    /// not from the positional read underneath it.
    ///
    /// **NO TEST REDDENED** this before I wrote it, and the mutation that
    /// exposed the hole is one line: bound on `len as u64 > self.len` instead
    /// of on `offset + len`, the exact mistake `pmtiles verify` makes. All 79
    /// tests stayed green, because the short read fails anyway and every
    /// assertion above is `is_err()`.
    ///
    /// So the two are not distinguishable by whether they error. They are
    /// distinguishable by **what the caller is told**, and by **when**: the
    /// bound runs before the `len`-sized allocation, and `len` came out of a
    /// directory entry, which makes it a number somebody else chose. Bounding
    /// on the length alone allocates their number first and finds out
    /// afterwards. `read_exact_at` answers the bare "failed to fill whole
    /// buffer"; this message names the range and the size of the file, which
    /// is what somebody debugging a corrupt archive can act on.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn the_out_of_range_refusal_names_the_range_and_the_file() {
        let (file, _) = ramp_file();
        let reader = FileRangeReader::try_open(file.path()).unwrap();

        let err = reader.read_range(250, 10).unwrap_err();
        let text = err.to_string();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        assert!(
            text.contains("250..260"),
            "the refusal must name the range it refused, got: {text}"
        );
        assert!(
            text.contains("256 byte file"),
            "the refusal must name the size it was measured against, got: {text}"
        );

        // A wildly out-of-bounds offset with an honest length, which is the
        // shape that walks past `pmtiles verify` and segfaults it.
        let err = reader.read_range(999_999, 10).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("999999..1000009") && text.contains("256 byte file"),
            "an offset past the end must be refused by the bound, got: {text}"
        );

        // An end that overflows `u64` is a different refusal, by a different
        // kind, so one over-broad arm is not carrying both.
        let err = reader.read_range(u64::MAX, 2).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("overflows"), "{err}");

        // The positive control: the same length one byte earlier reads, and
        // says nothing, so this is about the bound and not about the length.
        assert_eq!(reader.read_range(246, 10).unwrap().len(), 10);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_short_read_is_an_error_rather_than_a_shorter_buffer() {
        // The return type carries no length the caller did not ask for, so an
        // implementation that returned fewer bytes would hand back a directory
        // page that stops in the middle of a column and looks complete.
        let (file, _) = ramp_file();
        let reader = FileRangeReader::try_open(file.path()).unwrap();
        let err = reader.read_range(254, 4).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn an_empty_file_is_never_an_archive() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let reader = FileRangeReader::try_open(file.path()).unwrap();
        assert!(reader.is_empty());
        assert_eq!(reader.size().unwrap(), Some(0));
        // The header alone is 127 bytes, so this is the first thing a reader
        // asks for and the first thing it must not get a panic from.
        assert!(reader.read_range(0, 127).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn opening_something_that_is_not_there_is_a_typed_error() {
        let missing = std::path::Path::new("/nonexistent/f11/definitely-not-here.pmtiles");
        assert!(matches!(
            FileRangeReader::try_open(missing),
            Err(PmTilesError::Io(_))
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn one_reader_serves_several_threads_at_once() {
        // The reason for positional reads rather than seek-then-read. With a
        // shared cursor, two threads reading two ranges interleave and hand
        // each other the wrong bytes, and the failure is intermittent.
        let (file, payload) = ramp_file();
        let reader = FileRangeReader::try_open(file.path()).unwrap();

        std::thread::scope(|scope| {
            for start in 0..8u64 {
                let reader = &reader;
                let payload = &payload;
                scope.spawn(move || {
                    for _ in 0..64 {
                        let at = start * 32;
                        let got = reader.read_range(at, 32).unwrap();
                        assert_eq!(got, payload[at as usize..at as usize + 32]);
                    }
                });
            }
        });
    }

    /// A `RangeReader` that is not a file, proving the trait is implementable
    /// outside this module without touching anything else.
    struct InMemory(Vec<u8>);

    impl RangeReader for InMemory {
        fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
            let start = usize::try_from(offset)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset"))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "end"))?;
            if end > self.0.len() {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
            }
            Ok(self.0[start..end].to_vec())
        }
    }

    #[test]
    fn a_backend_that_does_not_know_its_size_says_so() {
        // The default `size()` is `None` rather than a required method,
        // because a streaming backend genuinely may not know and a number it
        // invented would turn a bounds check into a false refusal.
        let reader = InMemory(b"PMTiles".to_vec());
        assert_eq!(reader.size().unwrap(), None);
        assert_eq!(reader.read_range(0, 7).unwrap(), b"PMTiles".to_vec());

        // And it works behind a trait object, which is what a CLI picking a
        // backend from a URI scheme needs.
        let boxed: Box<dyn RangeReader> = Box::new(InMemory(b"PMTiles".to_vec()));
        assert_eq!(boxed.read_range(3, 4).unwrap(), b"iles".to_vec());
    }
}
