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

/// Encode a raster into a target in the named format, then write the encoded
/// bytes to it.
///
/// The dispatch is [`Raster::encode_to_buffer`]'s, and **that** doc carries the
/// list of format names, deliberately in one place. This one used to keep its
/// own copy, which named five of the seventeen spellings the dispatch had by
/// the time anyone measured it, because it was written when five was the whole
/// of it and nothing connected the two afterwards. So a caller reading here
/// concluded WebP was unsupported years after it was wired. The list is not
/// repeated below, and a check refuses to let it come back (issue #881).
///
/// A leading `.` and letter case are ignored, so `PNG` and `.png` both select
/// PNG.
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
    /// `"png"`, `"gif"`, `"webp"`, `"jxl"`,
    /// `"jp2k"` / `"jp2"` / `"j2k"` / `"jpt"` / `"j2c"` / `"jpc"`, `"uhdr"`,
    /// `"fits"` / `"fit"` / `"fts"` and
    /// `"v"` / `"vips"` are wired; any other format returns
    /// [`EncodeError::Unsupported`]. `"webp"` encodes losslessly at
    /// [`crate::webp::SaveOptions::default`], keeping any attached metadata,
    /// `"gif"` at [`crate::gif::SaveOptions::default`], and `"jxl"` losslessly at
    /// [`crate::jxl::SaveOptions::default`], which carries no metadata
    /// because the encoder writes no box container;
    /// [`Raster::encode_webp`], [`Raster::encode_gif`] and
    /// [`Raster::encode_jxl`] take the options explicitly.
    ///
    /// `"jxl"` needs the non-default `jxl` feature to produce bytes, and the
    /// six JPEG 2000 spellings need `jp2k`. They stay live rows without it and
    /// report [`EncodeError::Unsupported`] carrying `"jxl"` or `"jp2k"`, which
    /// is the same variant an unrecognised format name gets, so the dispatch
    /// has one answer for "this build cannot write that" however the caller
    /// arrived at it.
    ///
    /// All six JPEG 2000 spellings write the **same** JP2 container. That is
    /// not a shortcut: `jp2ksave` hard-codes `OPJ_CODEC_JP2` and, measured on
    /// 8.18.6, writes byte-identical files under all five suffixes it
    /// registers.
    ///
    /// `"uhdr"` is Ultra HDR (gain-map JPEG, libvips `uhdrsave`) at that
    /// saver's default quality of 75, and it is the **only** route to the
    /// writer that takes a format name: `uhdrsave` registers no file suffix
    /// at all, so [`Raster::save`] has no row for it. Unlike the rows above
    /// it has an input contract, a 3-band `f32` raster holding linear-light
    /// scRGB, and a raster that does not meet it is refused with
    /// [`EncodeError::InvalidParameter`] naming the raster rather than
    /// [`EncodeError::Unsupported`] naming the format: this build can write
    /// Ultra HDR, and what is wrong is the input.
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
        // Every spelling `jp2ksave` answers to, on one arm, because vips
        // writes the same JP2 container for all five suffixes: measured on
        // 8.18.6, `vips copy base.v out.EXT` over `jp2`, `j2k`, `jpt`, `j2c`
        // and `jpc` gives five files with one SHA-256 between them. `"jp2k"`
        // is here too because that is the saver's name and what a caller who
        // read `vips -l` would type; `jp2ksave` itself does not answer to it
        // as a suffix, and neither does anything else, so it costs nothing.
        //
        // Not gated, like `"jxl"` above and unlike the extension route:
        // without the feature `encode_jp2k` already reports
        // `EncodeError::Unsupported { format: "jp2k" }`, which is the same
        // variant an unrecognised name gets, so the row stays live and typed
        // rather than disappearing.
        "jp2k" | "jp2" | "j2k" | "jpt" | "j2c" | "jpc" => {
            raster.encode_jp2k(crate::jp2k::SaveOptions::default())
        }
        // `uhdrsave`'s nickname, and the only name it has: measured on 8.18.6,
        // `vips -l` gives `VipsForeignSaveUhdrFile (uhdrsave), save image in
        // UltraHDR format, nocache (), priority=0`, an **empty** suffix list,
        // and `vips copy base.v out.uhdr` is refused as an unknown format. So
        // one spelling here and no row at all in the extension table, which is
        // the reverse of the JPEG 2000 arm above.
        //
        // Not gated because there is nothing to gate: #508 wrote the container
        // out of the JPEG codec the crate already required, so Ultra HDR costs
        // no feature and no dependency.
        //
        // The default quality is `uhdrsave`'s own 75, through
        // `uhdr::SaveOptions::default`, the same way the rows above take their
        // codec's defaults. A caller who wants another quality or another
        // gain-map scale factor calls `Raster::encode_uhdr` or
        // `Raster::encode_uhdr_gainmap_scale`, which is where those knobs live.
        "uhdr" => raster.encode_uhdr(crate::uhdr::SaveOptions::default().quality),
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

    /// All six JPEG 2000 spellings are live rows in the shared format
    /// dispatch and every one produces the same JP2 container (issue #770).
    ///
    /// Measured on the pinned vips 8.18.6 rather than read out of the C:
    /// `vips copy base.v out.EXT` over `jp2`, `j2k`, `jpt`, `j2c` and `jpc`
    /// writes five files with one SHA-256 between them, and `out.jp2000` is
    /// refused as an unknown format. `"jp2k"` is here as well because that is
    /// the saver's name, which is what a caller reading `vips -l` would type,
    /// and it is the spelling #770 names.
    ///
    /// The normalisation the dispatch already does is exercised too, since a
    /// six-way arm is exactly where a leading dot or a capital would get lost.
    #[test]
    #[cfg(feature = "jp2k")]
    fn encode_for_format_routes_every_jpeg_2000_spelling_to_one_container() {
        let raster = Raster::new(
            8,
            6,
            PixelFormat::Rgb8,
            (0..8u32 * 6 * 3).map(|i| (i % 251) as u8).collect(),
        )
        .unwrap();
        let direct = raster
            .encode_jp2k(crate::jp2k::SaveOptions::default())
            .expect("the encoder takes an 8x6 RGB raster");

        for spelling in ["jp2k", "jp2", "j2k", "jpt", "j2c", "jpc", ".JP2", " Jp2k "] {
            let bytes = raster
                .encode_to_buffer(spelling)
                .unwrap_or_else(|e| panic!("{spelling:?} must be a live row, got {e}"));
            assert_eq!(
                bytes, direct,
                "{spelling:?} must write the same container as encode_jp2k"
            );
        }

        // The positive control for the sweep above: a name vips does not know
        // either still has to come back typed rather than routed anywhere.
        assert!(matches!(
            raster.encode_to_buffer("jp2000"),
            Err(EncodeError::Unsupported { .. })
        ));
    }

    /// Without the `jp2k` feature the six rows are still there and still
    /// typed: `Unsupported` carrying `"jp2k"`, which is the same variant an
    /// unrecognised name gets (issue #770).
    ///
    /// #770 says the row "must be `#[cfg(feature = "jp2k")]` on both sides".
    /// On this side that is wrong, and measurably so: `encode_jp2k` without
    /// the feature already returns exactly this, so gating the arm would take
    /// a live row out of the dispatch for no gain. The `.jxl` row above is
    /// ungated for the same reason. The extension route is the side that does
    /// need the cfg, because `saveable_extensions()` must not advertise an
    /// encoder this build has not got.
    #[test]
    #[cfg(not(feature = "jp2k"))]
    fn encode_for_format_refuses_every_jpeg_2000_spelling_by_name_without_the_feature() {
        let raster = Raster::new(8, 6, PixelFormat::Rgb8, vec![0u8; 8 * 6 * 3]).unwrap();
        for spelling in ["jp2k", "jp2", "j2k", "jpt", "j2c", "jpc"] {
            let err = raster.encode_to_buffer(spelling).unwrap_err();
            assert!(
                matches!(err, EncodeError::Unsupported { ref format } if format == "jp2k"),
                "{spelling:?} must report the codec name it has no encoder for, got {err}"
            );
        }
    }

    /// Every format spelling `encode_for_format` has an arm for, read out of
    /// this module's own source.
    ///
    /// Only arm *heads* are scanned, so a quoted name inside an arm body or a
    /// comment (there are several: `"jp2k"` appears in three of them) is not
    /// mistaken for a row.
    fn wired_format_arms(src: &str) -> Vec<&str> {
        let start = src
            .find("fn encode_for_format(")
            .expect("the dispatch lives in this file");
        let body = &src[start..];
        let end = body
            .find("\n}\n")
            .expect("the function closes at column zero");
        let mut names = Vec::new();
        for line in body[..end].lines() {
            let Some((head, _)) = line.split_once("=>") else {
                continue;
            };
            if !head.trim_start().starts_with('"') {
                continue;
            }
            for piece in head.split('|') {
                let piece = piece.trim();
                if let Some(inner) = piece.strip_prefix('"').and_then(|p| p.strip_suffix('"')) {
                    names.push(inner);
                }
            }
        }
        names
    }

    /// Every ``​`"name"`​`` in the doc block immediately above `marker`.
    ///
    /// The quoted-literal spelling is what it looks for, because that is how
    /// the list is written: those are the exact argument values a caller
    /// passes. Prose naming a format without quoting an argument (\"both
    /// select PNG\") is not a list and is not matched.
    ///
    /// Used twice: once on [`Raster::encode_to_buffer`], which is the one place
    /// the format list is written down for a caller, and once on
    /// `encode_to_target`, which must name **none**, because a second copy of
    /// the list is what drifted (issue #881).
    fn documented_format_names<'a>(src: &'a str, marker: &str) -> Vec<&'a str> {
        let at = src
            .find(marker)
            .unwrap_or_else(|| panic!("{marker} lives in this file"));
        let mut names = Vec::new();
        for line in src[..at].lines().rev() {
            let line = line.trim_start();
            if !line.starts_with("///") {
                break;
            }
            let mut rest = line;
            while let Some(i) = rest.find("`\"") {
                rest = &rest[i + 2..];
                let Some(j) = rest.find("\"`") else { break };
                names.push(&rest[..j]);
                rest = &rest[j + 2..];
            }
        }
        names
    }

    /**
     * Tests that the format list a caller is given and the arms the dispatch
     * actually has name exactly the same set (issue #881).
     *
     * The extension route has had this since the `.jxl` arm landed while the
     * refusal message still read "png, jpg/jpeg, gif, webp, and v/vips", so
     * `save("x.avif")` told the caller JPEG XL was unsupported at the moment it
     * became supported. `saveable_extensions()` is a function rather than a
     * literal for that reason and
     * `save_error_lists_exactly_the_wired_extensions` walks it back through
     * `Raster::save`.
     *
     * The format route had nothing, and it drifted the same way and further:
     * `encode_to_target`'s doc named five of the eighteen spellings the
     * dispatch had, having been written when five was the whole of it, and
     * three format lanes went past it without noticing. That is the reason
     * #770, #809 and #880 could each go unnoticed as long as they did, so the
     * fix is a check rather than an edit.
     *
     * Set equality, so it is red in both directions: an arm added without the
     * doc moving, and a doc naming something with no arm behind it. The
     * `encode_to_target` copy is gone and its doc now points here, so there is
     * one list.
     *
     * The length assertion is the positive control. Both halves are source
     * scans, and two scans that have stopped finding anything agree perfectly.
     */
    #[test]
    fn the_format_dispatch_and_the_list_a_caller_is_given_cannot_drift_apart() {
        const SRC: &str = include_str!("connection.rs");

        let mut wired = wired_format_arms(SRC);
        wired.sort_unstable();
        wired.dedup();
        let mut documented = documented_format_names(SRC, "    pub fn encode_to_buffer");
        documented.sort_unstable();
        documented.dedup();

        assert!(
            wired.len() >= 15,
            "the arm scan found only {wired:?}, so it has stopped reading the dispatch"
        );
        assert_eq!(
            wired, documented,
            "the dispatch has arms for {wired:?} and the doc on `encode_to_buffer` names \
             {documented:?}; a caller only ever sees the second"
        );

        // And there is exactly one list. `encode_to_target` kept a second copy
        // and that copy is what drifted: measured on `origin/main` before this
        // PR, the dispatch had 17 arms, `encode_to_buffer`'s doc named all 17,
        // and `encode_to_target`'s named five. Its doc points here now and
        // names none, so re-growing a list there is red rather than merely
        // unfortunate. Two checked copies would be worse than one, because the
        // check would then keep them agreeing rather than keeping there being
        // one.
        let second = documented_format_names(SRC, "pub fn encode_to_target<W: Write>");
        assert!(
            second.is_empty(),
            "`encode_to_target` has grown its own format list again, naming {second:?}; \
             it should defer to `Raster::encode_to_buffer` (issue #881)"
        );
    }

    /// A 3-band `f32` linear-light ramp reaching past the SDR ceiling, which
    /// is the input contract [`crate::uhdr::encode_uhdr`] computes a gain map
    /// from. Same shape as the fixture in `foreign_stubs.rs`.
    fn scrgb_ramp(w: u32, h: u32) -> Raster {
        let mut px: Vec<f32> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let t = f64::from(x) / f64::from(w - 1);
                let s = f64::from(y) / f64::from(h - 1);
                px.push((0.02 + 6.0 * t * s) as f32);
                px.push((0.5 * (1.0 - t) + 3.0 * s) as f32);
                px.push((1.5 * t + 0.25) as f32);
            }
        }
        Raster::new(
            w,
            h,
            PixelFormat::FloatF32(std::num::NonZeroU16::new(3).unwrap()),
            px.into_iter().flat_map(f32::to_ne_bytes).collect(),
        )
        .unwrap()
    }

    /// `"uhdr"` is a live row in the shared format dispatch and it reaches the
    /// Ultra HDR writer (issue #809).
    ///
    /// One spelling, not six like JPEG 2000, and that is measured rather than
    /// chosen: on the pinned vips 8.18.6, `vips -l` reports
    /// `VipsForeignSaveUhdrFile (uhdrsave), save image in UltraHDR format,
    /// nocache (), priority=0` with an **empty** suffix list, so `uhdrsave`'s
    /// nickname is the only name there is. `"ultrahdr"` is the nearest miss
    /// and is the positive control below: a dispatch that accepted every
    /// string would pass the first half of this on its own.
    ///
    /// The bytes are compared against [`Raster::encode_uhdr`] at the default
    /// quality and then put through the crate's own two-stage Ultra HDR gate,
    /// so "reaches the writer" means a real container and not merely "some
    /// bytes came back".
    #[test]
    fn encode_for_format_routes_uhdr_to_the_ultra_hdr_writer() {
        let raster = scrgb_ramp(16, 16);
        let direct = raster
            .encode_uhdr(crate::uhdr::SaveOptions::default().quality)
            .expect("a 3-band f32 raster encodes");

        for spelling in ["uhdr", "UHDR", ".uhdr", " Uhdr "] {
            let bytes = raster
                .encode_to_buffer(spelling)
                .unwrap_or_else(|e| panic!("{spelling:?} must be a live row, got {e}"));
            assert_eq!(
                bytes, direct,
                "{spelling:?} must write the same container as encode_uhdr at the default quality"
            );
            assert!(
                crate::uhdr::is_uhdr(&bytes),
                "{spelling:?} must produce something that satisfies the Ultra HDR gate"
            );
        }

        // vips has no `ultrahdr` anything, so neither has this.
        for miss in ["ultrahdr", "uhdr2", "gainmap"] {
            assert!(
                matches!(
                    raster.encode_to_buffer(miss),
                    Err(EncodeError::Unsupported { .. })
                ),
                "{miss:?} is not a name vips knows either"
            );
        }
    }

    /// The `"uhdr"` row answers a raster it cannot write with
    /// [`EncodeError::InvalidParameter`] naming the input, not with
    /// [`EncodeError::Unsupported`] naming the format (issue #809).
    ///
    /// The two are different answers and the difference is the whole value of
    /// the row being live. `Unsupported` says "this build cannot write Ultra
    /// HDR", which is false: it can, and #508 and #757 are why. What is wrong
    /// is the raster, and a caller can act on that by casting to 3-band `f32`
    /// scRGB.
    ///
    /// Contrast the `"jxl"` and JPEG 2000 rows above, where `Unsupported` is
    /// exactly right because the feature really is off.
    #[test]
    fn the_uhdr_row_refuses_the_wrong_raster_by_naming_the_raster() {
        let rgb = Raster::new(8, 6, PixelFormat::Rgb8, vec![128u8; 8 * 6 * 3]).unwrap();
        let err = rgb.encode_to_buffer("uhdr").unwrap_err();
        assert!(
            matches!(err, EncodeError::InvalidParameter(_)),
            "expected InvalidParameter naming the raster, got {err:?}"
        );
        assert!(
            err.to_string().contains("Rgb8"),
            "the refusal must name what it got, given {err}"
        );
        // Positive control: the same raster through a row this build has no
        // encoder for still reports `Unsupported`, so the two answers are
        // genuinely distinguishable here and not just one variant everywhere.
        assert!(matches!(
            rgb.encode_to_buffer("heic"),
            Err(EncodeError::Unsupported { .. })
        ));
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
