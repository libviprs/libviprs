//! Streaming IO connections: [`Source`] and [`Target`].
//!
//! This module ports libvips' `VipsSource` / `VipsTarget` streaming
//! abstraction onto the pure-Rust codec spine. A [`Source`] wraps any
//! [`Read`] (a file, a pipe, an in-memory buffer) and feeds it to the magic
//! byte sniffing decoder; a [`Target`] wraps any [`Write`] (a file, stdout,
//! an in-memory buffer) and receives encoded bytes.
//!
//! ## Load parity: stdin equals file
//!
//! [`decode_source`] reads a source to end and hands the bytes to
//! [`crate::source::decode_bytes`], the same entry point [`crate::decode_file`]
//! funnels through after reading a path. Format selection is by magic-byte
//! sniffing inside `decode_bytes`, so a JPEG or PNG loaded over a pipe decodes
//! to bit-identical pixels as the same bytes loaded from a file. This mirrors
//! the connection oracle's `stdin-load == file-load` parity records
//! (`max_abs_pixel_diff == 0.0`).
//!
//! ## Save capture: a memory target keeps its bytes
//!
//! [`Target::new_to_memory`] captures every encoded byte into an in-memory
//! buffer that [`Target::get_blob`] exposes. This is a deliberate, documented
//! improvement over the upstream CLI, which silently writes zero bytes when
//! streaming certain save formats (matrix/csv/ppm) to stdout. Here the memory
//! target always holds exactly what was encoded.
//!
//! ## Wired formats
//!
//! [`encode_to_target`] and [`Raster::encode_to_buffer`] dispatch `"jpeg"` /
//! `"jpg"` / `"png"` to the existing sink encoders and `"v"` / `"vips"` to the
//! native `.v` encoder. Every other format name returns
//! [`EncodeError::Unsupported`] so callers compile against the connection
//! surface without depending on the encoder lanes that wire those formats.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::codec::{DecodeError, EncodeError};
use crate::raster::Raster;

/// Default JPEG quality for the connection encoders, matching the libvips
/// `jpegsave` default and the extension-dispatched [`Raster::save`] path.
const DEFAULT_JPEG_QUALITY: u8 = 75;

/// A readable byte source for the decode surface.
///
/// Wraps any [`Read`] and, optionally, remembers the filename it was opened
/// from. Format detection is by magic-byte sniffing in
/// [`crate::source::decode_bytes`], so the wrapped reader needs no extension
/// or seek support: a pipe reads exactly like a file.
pub struct Source<R: Read> {
    reader: R,
    filename: Option<String>,
}

impl<R: Read> Source<R> {
    /// Wrap an arbitrary reader (a pipe, socket, or in-memory buffer).
    ///
    /// The source carries no filename; format is discovered from the bytes
    /// themselves when [`decode_source`] runs.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            filename: None,
        }
    }

    /// The filename this source was opened from, if any.
    ///
    /// [`Source::new`] leaves this `None`; [`Source::from_file`] records the
    /// path.
    pub fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }
}

impl Source<File> {
    /// Open a file as a source, recording its path as the filename.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`io::Error`] if the file cannot be opened.
    pub fn from_file(path: &Path) -> Result<Source<File>, io::Error> {
        let file = File::open(path)?;
        Ok(Source {
            reader: file,
            filename: path.to_str().map(str::to_owned),
        })
    }
}

/// A writable byte target for the encode surface.
///
/// Wraps any [`Write`] (a file, stdout, an in-memory buffer). A target built
/// with [`Target::new_to_memory`] captures the encoded bytes so
/// [`Target::get_blob`] can read them back; a target built with
/// [`Target::new`] or [`Target::new_to_file`] streams straight to the wrapped
/// writer and keeps no in-memory copy.
pub struct Target<W: Write> {
    writer: W,
    filename: Option<String>,
    blob: Vec<u8>,
    capture: bool,
}

impl<W: Write> Target<W> {
    /// Wrap an arbitrary writer. Encoded bytes stream to the writer and are
    /// not retained in memory, so [`Target::get_blob`] returns an empty slice.
    ///
    /// [`Target::get_blob`] is meaningful only for [`Target::new_to_memory`]
    /// targets. A writer-backed target keeps no copy of what it encodes, so a
    /// caller here should read the encoded bytes back from the writer they own,
    /// not from `get_blob` (which stays empty).
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            filename: None,
            blob: Vec::new(),
            capture: false,
        }
    }

    /// The filename this target writes to, if any.
    pub fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }

    /// The bytes captured by an in-memory target.
    ///
    /// This is meaningful only for a [`Target::new_to_memory`] target, where it
    /// returns every byte encoded into the target so far. For a writer-backed
    /// target ([`Target::new`] / [`Target::new_to_file`]) nothing is retained,
    /// so this stays empty and the caller should read the encoded bytes back
    /// from the writer they own instead.
    pub fn get_blob(&self) -> &[u8] {
        &self.blob
    }

    /// Route encoded bytes to their destination: the in-memory capture for a
    /// memory target, otherwise the wrapped writer.
    ///
    /// The writer-backed branch flushes after writing so a buffered writer
    /// surfaces any deferred error here, at the encode call, rather than
    /// swallowing it on drop. `emit` runs once per encode, so the flush cost is
    /// negligible.
    fn emit(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.capture {
            self.blob.extend_from_slice(bytes);
            Ok(())
        } else {
            self.writer.write_all(bytes)?;
            self.writer.flush()
        }
    }
}

impl Target<File> {
    /// Create a file target, recording its path as the filename.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`io::Error`] if the file cannot be created.
    pub fn new_to_file(path: &Path) -> Result<Target<File>, io::Error> {
        let file = File::create(path)?;
        Ok(Target {
            writer: file,
            filename: path.to_str().map(str::to_owned),
            blob: Vec::new(),
            capture: false,
        })
    }
}

impl Target<Vec<u8>> {
    /// Create an in-memory target that captures everything encoded into it.
    ///
    /// The captured bytes are read back with [`Target::get_blob`]. Unlike the
    /// upstream CLI's silent-empty stdout for some save formats, this target
    /// always holds exactly the bytes that were encoded.
    pub fn new_to_memory() -> Target<Vec<u8>> {
        Target {
            writer: Vec::new(),
            filename: None,
            blob: Vec::new(),
            capture: true,
        }
    }
}

/// Decode a source into a [`Raster`].
///
/// Reads the source to end, then hands the bytes to
/// [`crate::source::decode_bytes`], which sniffs the format from the leading
/// magic bytes. This is the streaming counterpart of [`crate::decode_file`]:
/// the same bytes decode identically whether they arrive from a file or a pipe.
///
/// # Errors
///
/// Returns [`DecodeError`] if the source cannot be read or the bytes are not a
/// recognised, well-formed image.
pub fn decode_source<R: Read>(source: &mut Source<R>) -> Result<Raster, DecodeError> {
    let mut bytes = Vec::new();
    source.reader.read_to_end(&mut bytes)?;
    crate::source::decode_bytes(&bytes)
}

/// Encode a raster into a target in the named format.
///
/// Dispatches `"jpeg"` / `"jpg"` / `"png"` to the sink encoders and
/// `"v"` / `"vips"` to the native `.v` encoder, then writes the encoded bytes
/// to the target. A leading `.` and letter case are ignored, so `"PNG"` and
/// `".png"` both select PNG.
///
/// # Errors
///
/// Returns [`EncodeError::Unsupported`] for a format this lane does not wire,
/// [`EncodeError::Encode`] if the encoder rejects the raster, or
/// [`EncodeError::Io`] if writing to the target fails.
pub fn encode_to_target<W: Write>(
    raster: &Raster,
    target: &mut Target<W>,
    format: &str,
) -> Result<(), EncodeError> {
    let bytes = encode_for_format(raster, format)?;
    target.emit(&bytes)?;
    Ok(())
}

impl Raster {
    /// Encode this raster into a freshly allocated buffer in the named format.
    ///
    /// Uses the same dispatch as [`encode_to_target`]: `"jpeg"` / `"jpg"`,
    /// `"png"`, `"gif"`, `"webp"`, `"jxl"`, `"fits"` / `"fit"` / `"fts"` and
    /// `"v"` / `"vips"` are wired; any other format returns
    /// [`EncodeError::Unsupported`]. `"webp"` encodes losslessly at
    /// [`crate::webp::SaveOptions::default`], keeping any attached metadata,
    /// `"gif"` at [`crate::gif::SaveOptions::default`], and `"jxl"` losslessly at
    /// [`crate::jxl::SaveOptions::default`], which carries no metadata
    /// because the encoder writes no box container;
    /// [`Raster::encode_webp`], [`Raster::encode_gif`] and
    /// [`Raster::encode_jxl`] take the options explicitly.
    ///
    /// `"jxl"` needs the non-default `jxl` feature to produce bytes. It
    /// stays a live row without it and reports
    /// [`EncodeError::Unsupported`] carrying `"jxl"`, which is the same
    /// variant an unrecognised format name gets, so the dispatch has one
    /// answer for "this build cannot write that" however the caller
    /// arrived at it.
    ///
    /// # Errors
    ///
    /// As [`encode_to_target`], minus the I/O case (there is no external
    /// writer).
    pub fn encode_to_buffer(&self, format: &str) -> Result<Vec<u8>, EncodeError> {
        encode_for_format(self, format)
    }
}

/// Shared format dispatch for the connection encoders.
///
/// Normalises the format name (trim, strip a leading `.`, lowercase) and
/// routes to the existing encoders. Formats without a wired encoder return
/// [`EncodeError::Unsupported`] carrying the caller's original spelling.
fn encode_for_format(raster: &Raster, format: &str) -> Result<Vec<u8>, EncodeError> {
    let key = format.trim().trim_start_matches('.').to_ascii_lowercase();
    match key.as_str() {
        "jpeg" | "jpg" => {
            crate::sink::encode_jpeg(raster, DEFAULT_JPEG_QUALITY).map_err(sink_err_to_encode)
        }
        "png" => crate::sink::encode_png(raster).map_err(sink_err_to_encode),
        "gif" => raster.encode_gif(crate::gif::SaveOptions::default()),
        "webp" => raster.encode_webp(crate::webp::SaveOptions::default()),
        "jxl" => raster.encode_jxl(crate::jxl::SaveOptions::default()),
        // The three suffixes vips registers for FITS (`vips__fits_suffs`,
        // `fits.c:125`). `fitssave` takes no options, so there is nothing
        // to default here.
        "fits" | "fit" | "fts" => raster.encode_fits(),
        "v" | "vips" => raster.encode_vips().map_err(save_err_to_encode),
        _ => Err(EncodeError::unsupported(format.to_owned())),
    }
}

/// Map a [`crate::sink::SinkError`] from the sink encoders onto the shared
/// [`EncodeError`] spine.
fn sink_err_to_encode(err: crate::sink::SinkError) -> EncodeError {
    use crate::sink::SinkError;
    match err {
        SinkError::Io(e) => EncodeError::Io(e),
        SinkError::Encode { format, source } => EncodeError::Encode(format!("{format}: {source}")),
        SinkError::EncodeMsg(msg) => EncodeError::Encode(msg),
        other => EncodeError::Encode(other.to_string()),
    }
}

/// Map a [`crate::imageio::SaveError`] from the native `.v` encoder onto the
/// shared [`EncodeError`] spine.
fn save_err_to_encode(err: crate::imageio::SaveError) -> EncodeError {
    use crate::imageio::SaveError;
    match err {
        SaveError::Io(e) => EncodeError::Io(e),
        SaveError::UnsupportedExtension { extension } => EncodeError::unsupported(extension),
        SaveError::Encode(sink) => sink_err_to_encode(sink),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;

    /// A small deterministic RGB gradient raster to encode and round-trip.
    fn sample_raster() -> Raster {
        let (w, h) = (32u32, 24u32);
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push((x * 8 % 256) as u8);
                data.push((y * 10 % 256) as u8);
                data.push(((x + y) * 4 % 256) as u8);
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// Mean absolute per-byte difference between two equal-length buffers.
    fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len(), "buffers must be the same length");
        let total: u64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| u64::from(x.abs_diff(*y)))
            .sum();
        total as f64 / a.len() as f64
    }

    /// `"webp"` is a live row in the shared format dispatch, and the
    /// bytes it returns are the same ones `Raster::encode_webp` writes
    /// at the default options, so the connection lane and the codec
    /// module cannot drift apart.
    #[test]
    fn encode_for_format_routes_webp_to_the_lossless_encoder() {
        let raster = sample_raster();
        let via_dispatch = raster.encode_to_buffer("webp").unwrap();
        let direct = raster
            .encode_webp(crate::webp::SaveOptions::default())
            .unwrap();
        assert_eq!(via_dispatch, direct);
        assert_eq!(&via_dispatch[..4], b"RIFF");
        assert_eq!(&via_dispatch[8..12], b"WEBP");
        let back = crate::decode_bytes(&via_dispatch).unwrap();
        assert_eq!(back.data(), raster.data());
    }

    /// `"jxl"` is a live row in the shared format dispatch, and the
    /// bytes it returns are the same ones `Raster::encode_jxl` writes at
    /// the default options, so the connection lane and the codec module
    /// cannot drift apart. The leading `FF 0A` is the bare-codestream
    /// magic, which is what the encoder writes and what
    /// `vips jxlsave --keep none` writes too.
    #[test]
    #[cfg(feature = "jxl")]
    fn encode_for_format_routes_jxl_to_the_lossless_encoder() {
        let raster = sample_raster();
        let via_dispatch = raster.encode_to_buffer("jxl").unwrap();
        let direct = raster
            .encode_jxl(crate::jxl::SaveOptions::default())
            .unwrap();
        assert_eq!(via_dispatch, direct);
        assert_eq!(&via_dispatch[..2], b"\xff\x0a");
        let back = crate::decode_bytes(&via_dispatch).unwrap();
        assert_eq!(back.data(), raster.data());
    }

    /// Without the `jxl` feature the row is still there and still typed:
    /// it reports `Unsupported` carrying the name the caller asked for,
    /// which is what the dispatch promises for any format this build has
    /// no encoder behind. Pinned so the row cannot quietly become a
    /// fall-through to the catch-all arm, which would lose the name.
    #[test]
    #[cfg(not(feature = "jxl"))]
    fn encode_for_format_refuses_jxl_by_name_without_the_feature() {
        let raster = sample_raster();
        let err = raster.encode_to_buffer("jxl").unwrap_err();
        assert!(
            matches!(err, EncodeError::Unsupported { ref format } if format == "jxl"),
            "{err:?}"
        );
    }

    /// Oracle `stdin-load-pixel-parity-jpeg`: the same JPEG bytes decode
    /// bit-identically whether read through a [`Source`] (stdin) or handed
    /// straight to `decode_bytes` (file). `max_abs_pixel_diff == 0.0`.
    #[test]
    fn source_load_equals_file_load_jpeg() {
        let bytes = sample_raster().encode_to_buffer("jpeg").unwrap();

        let from_file = crate::source::decode_bytes(&bytes).unwrap();

        let mut source = Source::new(&bytes[..]);
        let from_stdin = decode_source(&mut source).unwrap();

        assert_eq!(from_stdin.width(), from_file.width());
        assert_eq!(from_stdin.height(), from_file.height());
        assert_eq!(from_stdin.format(), from_file.format());
        assert_eq!(
            from_stdin.data(),
            from_file.data(),
            "stdin and file JPEG loads must be pixel-identical"
        );
    }

    /// Same parity check for PNG (oracle `stdin-load-pixel-parity-png`).
    #[test]
    fn source_load_equals_file_load_png() {
        let bytes = sample_raster().encode_to_buffer("png").unwrap();

        let from_file = crate::source::decode_bytes(&bytes).unwrap();

        let mut source = Source::new(&bytes[..]);
        let from_stdin = decode_source(&mut source).unwrap();

        assert_eq!(from_stdin.width(), from_file.width());
        assert_eq!(from_stdin.height(), from_file.height());
        assert_eq!(from_stdin.data(), from_file.data());
    }

    /// A `new_to_memory` target captures the encoded PNG, and decoding that
    /// capture reproduces the original raster losslessly (PNG round trip).
    #[test]
    fn memory_target_png_round_trip() {
        let raster = sample_raster();

        let mut target = Target::new_to_memory();
        encode_to_target(&raster, &mut target, "png").unwrap();

        assert!(
            !target.get_blob().is_empty(),
            "memory target must capture the encoded bytes"
        );

        let mut source = Source::new(target.get_blob());
        let decoded = decode_source(&mut source).unwrap();

        assert_eq!(decoded.width(), raster.width());
        assert_eq!(decoded.height(), raster.height());
        assert_eq!(decoded.format(), raster.format());
        assert_eq!(decoded.data(), raster.data(), "PNG round trip is lossless");
    }

    /// `encode_to_buffer("jpeg")` yields non-empty bytes that decode back to
    /// the original dimensions within JPEG's lossy tolerance.
    #[test]
    fn encode_to_buffer_jpeg_non_empty_and_decodes() {
        let raster = sample_raster();

        let bytes = raster.encode_to_buffer("jpeg").unwrap();
        assert!(!bytes.is_empty(), "JPEG buffer must be non-empty");

        let mut source = Source::new(&bytes[..]);
        let decoded = decode_source(&mut source).unwrap();

        assert_eq!(decoded.width(), raster.width());
        assert_eq!(decoded.height(), raster.height());
        assert_eq!(decoded.format(), raster.format());
        // Lossy, but a low-detail gradient stays well within tolerance.
        assert!(
            mean_abs_diff(decoded.data(), raster.data()) < 12.0,
            "JPEG round trip drifted beyond tolerance"
        );
    }

    /// An unwired format returns the typed [`EncodeError::Unsupported`], not a
    /// panic, through both the buffer and the target entry points.
    #[test]
    fn unsupported_format_returns_typed_error() {
        let raster = sample_raster();

        let buf_err = raster.encode_to_buffer("tiff").unwrap_err();
        assert!(
            matches!(buf_err, EncodeError::Unsupported { .. }),
            "expected Unsupported, got {buf_err:?}"
        );

        let mut target = Target::new_to_memory();
        let target_err = encode_to_target(&raster, &mut target, "heif").unwrap_err();
        assert!(matches!(target_err, EncodeError::Unsupported { .. }));
        assert!(
            target.get_blob().is_empty(),
            "nothing should be written for an unsupported format"
        );
    }

    /// The native `.v` format is wired and round-trips through a memory target.
    #[test]
    fn native_v_format_round_trips_through_memory_target() {
        let raster = sample_raster();

        let mut target = Target::new_to_memory();
        encode_to_target(&raster, &mut target, "v").unwrap();
        assert!(!target.get_blob().is_empty());

        let mut source = Source::new(target.get_blob());
        let decoded = decode_source(&mut source).unwrap();

        assert_eq!(decoded.width(), raster.width());
        assert_eq!(decoded.height(), raster.height());
        assert_eq!(decoded.data(), raster.data());
    }

    /// Format matching ignores case and a leading dot.
    #[test]
    fn format_matching_is_case_and_dot_insensitive() {
        let raster = sample_raster();
        assert!(raster.encode_to_buffer("PNG").is_ok());
        assert!(raster.encode_to_buffer(".jpg").is_ok());
        assert!(raster.encode_to_buffer("Vips").is_ok());
    }

    /// A writer-backed [`Target::new`] streams through the `emit` else-branch
    /// and the writer receives exactly the bytes [`Raster::encode_to_buffer`]
    /// produces (byte-identity). `get_blob` stays empty for this target.
    #[test]
    fn writer_target_emits_bytes_identical_to_buffer() {
        let raster = sample_raster();
        let expected = raster.encode_to_buffer("png").unwrap();

        let mut buf: Vec<u8> = Vec::new();
        {
            let mut target = Target::new(&mut buf);
            encode_to_target(&raster, &mut target, "png").unwrap();
            assert!(
                target.get_blob().is_empty(),
                "a writer-backed target retains nothing in get_blob"
            );
        }

        assert_eq!(
            buf, expected,
            "writer-backed emit must deliver the same bytes as encode_to_buffer"
        );
    }

    /// A [`Target::new_to_file`] target writes the encoded bytes to disk; reading
    /// the file back yields exactly [`Raster::encode_to_buffer`]'s bytes and
    /// decodes to the original raster.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn file_target_writes_and_reads_back_identical_bytes() {
        let raster = sample_raster();
        let expected = raster.encode_to_buffer("png").unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("connection-file-target.png");

        let mut target = Target::new_to_file(&path).unwrap();
        encode_to_target(&raster, &mut target, "png").unwrap();
        assert!(
            target.get_blob().is_empty(),
            "a file target retains nothing in get_blob"
        );
        drop(target);

        let written = std::fs::read(&path).unwrap();
        assert_eq!(
            written, expected,
            "bytes on disk must match encode_to_buffer"
        );

        let mut source = Source::from_file(&path).unwrap();
        let decoded = decode_source(&mut source).unwrap();
        assert_eq!(decoded.width(), raster.width());
        assert_eq!(decoded.height(), raster.height());
        assert_eq!(decoded.data(), raster.data());
    }

    /// `from_file` records the path; `new` leaves the filename unset.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn source_filename_reflects_construction() {
        let bytes = sample_raster().encode_to_buffer("png").unwrap();
        let source = Source::new(&bytes[..]);
        assert_eq!(source.filename(), None);

        let dir = std::env::temp_dir().join(format!("viprs-conn-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample.png");
        std::fs::write(&path, &bytes).unwrap();

        let file_source = Source::from_file(&path).unwrap();
        assert_eq!(file_source.filename(), path.to_str());

        std::fs::remove_file(&path).ok();
    }
}
