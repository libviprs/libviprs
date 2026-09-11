//! Fetching byte ranges, and the local-file implementation of it.
//!
//! Everything a PMTiles reader does is a ranged read: 127 bytes for the
//! header, a few hundred for the root directory, one more for a leaf, one for
//! the tile. It never scans and it never reads the whole file, which is what
//! makes a 40 GB archive on an object store behave like a local one.
//!
//! [`RangeReader`] is the single seam that makes that true for any transport.
//! This issue ships the local one, [`FileRangeReader`]; an HTTP or S3 backend
//! is another implementation of the same three-method trait and needs no
//! change to the reader or the writer above it.
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
        let end = offset
            .checked_add(len as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range end overflows u64"))?;
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
