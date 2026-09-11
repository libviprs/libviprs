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
