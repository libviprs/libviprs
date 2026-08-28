//! GIF: still-image load and save, and the `gifsave` option surface on
//! [`Raster`].
//!
//! This module is the single file the GIF lane (issue #499) owns, split out
//! from [`crate::encode`] ahead of the lane so four format PRs would not
//! collide inside one file (issue #563). It is a port of libvips'
//! `foreign/nsgifload.c` (load) and `foreign/cgifsave.c` (save), and it goes
//! straight to the `gif` crate rather than through the `image` facade,
//! because the facade cannot express what either half needs: measured on
//! `image` 0.25.10, `codecs/gif.rs` contains zero occurrences of "interlac"
//! and zero of "dither", `GifEncoder::encode` routes to
//! `Frame::from_rgb_speed` with no palette control, and `GifDecoder`
//! hard-codes `ColorType::Rgba8` where vips emits three bands for an opaque
//! GIF.
//!
//! # Operations
//!
//! | libviprs method          | libvips equivalent | result                                              |
//! |--------------------------|--------------------|-----------------------------------------------------|
//! | [`decode_gif`]           | `gifload`          | frame 0 as `Rgb8` or `Rgba8`, plus the GIF fields    |
//! | [`Raster::encode_gif`]   | `gifsave_buffer`   | one-frame GIF89a bytes                              |
//! | [`Raster::save_gif`]     | `gifsave`          | the same bytes written to a path                    |
//!
//! # Semantics
//!
//! * **Band count follows the whole file, not the loaded frame.**
//!   `nsgifload.c:271` sizes the image `has_transparency ? 4 : 3`, and
//!   `:431-432` sets `has_transparency` from *any* frame's graphic control
//!   extension, so a still that declares no transparent index loads as
//!   `Rgb8` and one that does loads as `Rgba8`. `:522-534` drops the alpha
//!   byte per pixel in the three-band case rather than compositing anything.
//! * **The canvas around frame 0 is transparent black, not the background
//!   colour.** Measured: an 8x8 GIF whose background index points at blue
//!   and whose only frame is a 4x4 inset reports `background: 0 0 255` in
//!   the header while `vips getpoint` prints `0 0 0` at the corner.
//! * **`bits-per-sample` is `ceil(log2(colours))`** over the colour table as
//!   it appears on the wire, padded to a power of two
//!   (`nsgifload.c:337-343`). `palette` is always 1.
//! * **`loop` is not the NETSCAPE count.** No application extension means
//!   `loop = 1`, a count of 0 means `loop = 0` (forever), and a count of `n`
//!   means `loop = n + 1`. Measured across the reference suite:
//!   `dispose-background.gif` carries 10 and reports 11, `garden.gif`
//!   carries 0 and reports 0, `cramps.gif` carries none and reports 1.
//! * **Alpha is thresholded at 128 on save**, and a pixel below it is zeroed
//!   to `(0, 0, 0, 0)` before quantisation, colour included
//!   (`cgifsave.c:538-548`). GIF has one transparent index and no partial
//!   alpha, so this is the whole of the alpha model.
//! * **The palette holds at most `min(255, 1 << bitdepth)` entries**
//!   (`cgifsave.c:795-796`), and the 255 is deliberate: vips keeps one index
//!   free for cgif's transparency optimisation. So `bitdepth: 8` can never
//!   produce a 256-colour GIF, and a 256-colour raster is quantised while a
//!   255-colour one round-trips exactly. Measured both sides of that edge.
//! * **A transparent entry is reserved at index 0 whenever the palette does
//!   not saturate.** Measured over twelve (bitdepth, distinct-colour)
//!   combinations: an opaque source with room to spare comes back from
//!   `vips gifsave` carrying an unused transparent index and reloads as four
//!   bands, and only a saturated palette does not. libviprs reproduces that,
//!   because the band count reaches operations downstream and a header-only
//!   divergence there would become an op-surface one.
//! * **Interlace is a storage order, not a pixel change.** The four passes
//!   are rows `0, 8, 16, ...`, then `4, 12, ...`, then `2, 6, ...`, then
//!   every odd row. Verified by LZW-decoding what `vips gifsave --interlace`
//!   wrote for an 8-row image and recovering the order `0, 4, 2, 6, 1, 3, 5,
//!   7`.
//!
//! # Where libviprs and vips diverge, and why it is not a bug
//!
//! LZW is exactly lossless and deterministic in both directions, so the
//! bitstream is not where the divergence lives. **Palette quantisation is.**
//! vips quantises with libimagequant; libviprs uses the median-cut quantiser
//! [`crate::encode::quantize_palette`] that already backs
//! [`Raster::encode_png_palette`]. Two different algorithms pick two
//! different palettes for the same image, so the bytes will never match and
//! chasing that would be chasing the wrong thing.
//!
//! What is pinned instead is structural, and all of it is checked here:
//! palette size, transparent-index handling, interlace row order, the alpha
//! threshold, and exact round-trip fidelity whenever the source already fits
//! the palette. On the quantised path the disagreement is bounded rather
//! than merely acknowledged: on the reference 32x24 768-colour source vips
//! reports `avg_abs_diff 3.37, max_abs_diff 23` against its own input, and
//! `gif_quantisation_error_is_within_the_measured_vips_band` requires
//! libviprs to stay in the same band.
//!
//! Two smaller, deliberate divergences:
//!
//! * `dither` is Floyd-Steinberg error diffusion scaled by the option, where
//!   vips passes the value to `liq_set_dithering_level`
//!   (`cgifsave.c:584`). Both are 0 for "nearest colour, no diffusion" and
//!   both diffuse more as the value rises, but libimagequant's variant is
//!   edge-aware and the dithered pixels differ. Measured, the vips knob is
//!   not even monotone in pixels-changed (70, 95, 88, 90 at 0.25, 0.5, 0.75,
//!   1.0 on a 768-pixel ramp), so only the `dither == 0` identity is worth
//!   pinning as an equality.
//! * The logical screen descriptor's colour-resolution bits are written by
//!   the `gif` crate as the table-size flag; cgif writes zero. Nothing reads
//!   the field.
//!
//! # Not handled here
//!
//! Animation. `decode_gif` loads **frame 0**, which is exactly what `vips
//! gifload` does by default (`page = 0`, `n = 1`), and attaches `n-pages` so
//! a caller can see the rest is there. Multi-page load and save are #572 and
//! #573, blocked on the page model (#564). For the same reason the array
//! fields `delay`, `background`, and `gif-palette` are read but not
//! attached: [`crate::imageio::MetadataValue`] has no array variant yet, and
//! adding one is #564's call. `gifsave`'s `effort`, `reuse`,
//! `interpalette-maxerror`, `interframe-maxerror` and `keep-duplicate-frames`
//! are cgif-specific palette-reuse and frame-coalescing machinery with no
//! pure-Rust equivalent and are not modelled.
//!
//! Every entry point here is fallible. Load failures arrive as
//! [`GifError`] through [`SourceError`], save failures as [`EncodeError`];
//! there is no panicking twin, matching the rest of the codec surface.

use crate::codec::EncodeError;
use crate::imageio::{MetadataValue, SaveError};
use crate::pixel::PixelFormat;
use crate::raster::{Raster, buffer_len};
use crate::source::{DecodeLimits, SourceError};
use std::io::Cursor;
use std::path::Path;
use thiserror::Error;

/// libvips `gifsave`'s `dither` default, measured on vips 8.18.4
/// (`dither`, `default: 1`, `min: 0`, `max: 1`).
const DEFAULT_DITHER: f64 = 1.0;

/// libvips `gifsave`'s `bitdepth` default, measured on vips 8.18.4
/// (`bitdepth`, `default: 8`, `min: 1`, `max: 8`).
const DEFAULT_BITDEPTH: u8 = 8;

/// The most palette entries `gifsave` will ever ask for, whatever the
/// bitdepth.
///
/// `cgifsave.c:795-796` is `vips__quantise_set_max_colors(attr, VIPS_MIN(255,
/// 1 << bitdepth))`, with the comment "Limit the number of colours to 255 so
/// there is always one index free for transparency optimization". So the
/// ceiling is 255 and not 256, and a 256-colour raster cannot be saved
/// losslessly by vips either.
const MAX_PALETTE_ENTRIES: usize = 255;

/// Bytes per entry in a GIF colour table.
const PALETTE_STRIDE: usize = 3;

/// The alpha level at or above which a pixel is opaque on save.
///
/// `cgifsave.c:538-548` promotes `p[3] >= 128` to 255 and zeroes everything
/// below it, colour included.
const ALPHA_THRESHOLD: u8 = 128;

/// The four passes of GIF interlacing as `(first_row, row_step)`.
///
/// Verified against `vips gifsave --interlace` by LZW-decoding the stored
/// rows of an 8-row image, which came back in the order `0, 4, 2, 6, 1, 3,
/// 5, 7`.
const INTERLACE_PASSES: [(u32, u32); 4] = [(0, 8), (4, 8), (2, 4), (1, 2)];

/// Errors from the GIF codec.
///
/// Decode failures are typed rather than stringly so a caller can tell a
/// truncated file from an unsupported one without matching on a message,
/// the same way [`crate::radiance::RadianceError`] does.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GifError {
    /// The bytes are not a GIF, or the GIF is malformed or truncated.
    ///
    /// Carries the `gif` crate's own message rather than its error type, so
    /// [`SourceError`] does not leak a dependency into its public shape.
    #[error("gif: {message}")]
    Decode {
        /// The underlying decoder failure, rendered through its `Display`.
        message: String,
    },
    /// The file parsed but holds no frames at all.
    ///
    /// vips raises the same case as `"no frames in GIF"`
    /// (`nsgifload.c:419-421`).
    #[error("gif: no frames in GIF")]
    NoFrames,
    /// The raster could not be built from the decoded pixels.
    #[error(transparent)]
    Raster(#[from] crate::raster::RasterError),
}

/// Options for [`Raster::encode_gif`] (libvips `gifsave` / `gifsave_buffer`).
///
/// Plain, `Default`, and module-scoped, so callers write
/// `gif::SaveOptions { dither: 0.0, ..Default::default() }` and later fields
/// can be added without a breaking change. Deliberately *not*
/// `#[non_exhaustive]`, which would block the struct literal downstream and
/// kill `..Default::default()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaveOptions {
    /// Write the frame interlaced (libvips `gifsave` `interlace`).
    /// Defaults to `false`, as vips does.
    pub interlaced: bool,
    /// Amount of dithering applied during palette quantisation, 0.0 to 1.0
    /// (libvips `gifsave` `dither`). Defaults to 1.0, as vips does. Values
    /// outside the range are clamped into it, matching the `min`/`max` vips
    /// declares on the argument.
    pub dither: f64,
    /// Bits per pixel, 1 to 8 (libvips `gifsave` `bitdepth`). Defaults to 8,
    /// as vips does. The palette holds at most `min(255, 1 << bitdepth)`
    /// colours; see [`MAX_PALETTE_ENTRIES`] for why 8 does not mean 256.
    pub bitdepth: u8,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            interlaced: false,
            dither: DEFAULT_DITHER,
            bitdepth: DEFAULT_BITDEPTH,
        }
    }
}

impl SaveOptions {
    /// The palette ceiling these options imply: `min(255, 1 << bitdepth)`,
    /// with the bitdepth clamped into the 1..=8 range vips declares.
    fn max_colours(self) -> usize {
        let bitdepth = self.bitdepth.clamp(1, 8);
        (1usize << bitdepth).min(MAX_PALETTE_ENTRIES)
    }
}

/// What one scan of every frame in a GIF says about the file as a whole.
///
/// libvips builds the same summary in `vips_foreign_load_nsgif_header`
/// (`nsgifload.c:424-435`) before it decodes a single pixel, because the
/// output band count depends on it.
struct FileScan {
    /// Total frames, which becomes `n-pages`.
    frames: u32,
    /// Whether *any* frame declares a transparent index.
    has_transparency: bool,
    /// Whether *any* frame is stored interlaced.
    interlaced: bool,
    /// The widest colour table in the file, in entries, padded to the power
    /// of two it occupies on the wire.
    colours: usize,
    /// The NETSCAPE loop count, translated to the vips `loop` convention.
    loop_count: i64,
}

/// Decode a GIF into a [`Raster`] (libvips `gifload`).
///
/// Loads **frame 0** at the logical screen size, which is what `vips
/// gifload` does by default (`page = 0`, `n = 1`). The palette is expanded
/// to `Rgb8`, or to `Rgba8` when any frame in the file declares a
/// transparent index; see the [module docs](crate::gif) for the full
/// semantics and for what is deferred to the animation lanes.
///
/// # Errors
///
/// * [`SourceError::Gif`] wrapping [`GifError::Decode`] for bytes that are
///   not a GIF or that are malformed, [`GifError::NoFrames`] for a file with
///   no frames, or [`SourceError::AllocLimitExceeded`] when the declared
///   geometry is over [`DecodeLimits::max_alloc_bytes`].
/// * [`SourceError::CoordLimitExceeded`] when either axis exceeds
///   [`DecodeLimits::max_coord`].
/// * [`SourceError::DimensionLimitExceeded`] when `width * height` exceeds
///   [`DecodeLimits::max_pixels`].
pub fn decode_gif(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let scan = scan_file(bytes)?;
    if scan.frames == 0 {
        return Err(GifError::NoFrames.into());
    }

    let bands = if scan.has_transparency { 4 } else { 3 };
    let mut decoder = open(bytes)?;
    let (width, height) = (u32::from(decoder.width()), u32::from(decoder.height()));
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    // The price and the comparison are the crate's, not this module's
    // (issue #632). This one used to be a plain `*`, safe only because a
    // GIF states its logical screen in `u16` and so cannot reach a product
    // a `u64` will not hold; nothing in the expression said so, and the
    // three codecs that copied the shape do not have that guarantee.
    // `decode_alloc_bytes` saturates, so the guarantee is no longer load
    // bearing. The price, the comparison and now the reporting are all the
    // crate's: this used to build a `GifError::AllocLimitExceeded` of its
    // own, one of five variants re-tagging the same refusal, which #686
    // collapsed onto `SourceError::AllocLimitExceeded`.
    //
    // `bands` is what the canvas actually costs, four with a transparent index
    // and three without, which is the count fifteen lines up and the count the
    // refusal reports.
    limits.check_image_alloc("GIF canvas", width, height, bands as u64, 1)?;

    // The canvas starts fully transparent and stays that way outside the
    // frame rectangle. libnsgif renders frame 0 over a cleared buffer and
    // never paints the background colour, which is why `vips getpoint`
    // reports `0 0 0` at a corner the header calls blue.
    // Through `buffer_len` rather than a plain `usize` product for the
    // same reason the price goes through `decode_alloc_bytes`: clearing
    // the budget says the byte count fits a `u64`, which on a 32-bit
    // target is not the same as fitting the address space, and a caller
    // can raise `max_alloc_bytes` past 4 GiB there.
    let mut data = vec![0u8; buffer_len(width, height, bands).map_err(GifError::Raster)?];
    let frame = decoder
        .next_frame_info()
        .map_err(decode_error)?
        .ok_or(GifError::NoFrames)?;
    let transparent = frame.transparent;
    let (left, top) = (u32::from(frame.left), u32::from(frame.top));
    let (fw, fh) = (u32::from(frame.width), u32::from(frame.height));
    let local = frame.palette.clone();
    let mut indices = vec![0u8; decoder.buffer_size()];
    // A truncated frame keeps the rows that did arrive. The buffer starts
    // zeroed and `read_into_buffer` fills it in order, so the tail is left
    // as index 0 exactly where libnsgif leaves it uncomposited -- which is
    // why `vips getpoint truncated.gif 574 799` prints `0 0 0 0` while the
    // top of the same file is real image data.
    let _truncated = decoder.read_into_buffer(&mut indices);
    let palette = local
        .or_else(|| decoder.global_palette().map(<[u8]>::to_vec))
        .unwrap_or_default();

    for y in 0..fh.min(height.saturating_sub(top)) {
        for x in 0..fw.min(width.saturating_sub(left)) {
            let Some(&index) = indices.get((y * fw + x) as usize) else {
                continue;
            };
            if transparent == Some(index) {
                continue;
            }
            let entry = usize::from(index) * PALETTE_STRIDE;
            // An index past the end of the colour table is left as the
            // cleared canvas rather than guessed at, which is what the `gif`
            // crate's own RGBA expansion does with it.
            let Some(rgb) = palette.get(entry..entry + PALETTE_STRIDE) else {
                continue;
            };
            let out = (((y + top) * width) + x + left) as usize * bands;
            data[out..out + PALETTE_STRIDE].copy_from_slice(rgb);
            if bands == 4 {
                data[out + 3] = u8::MAX;
            }
        }
    }

    let format = if bands == 4 {
        PixelFormat::Rgba8
    } else {
        PixelFormat::Rgb8
    };
    let mut raster = Raster::new(width, height, format, data).map_err(GifError::Raster)?;
    raster.meta.interpretation = Some(crate::conversion::Interpretation::Srgb);
    raster.set_n_pages(scan.frames);
    raster.set_field("loop", MetadataValue::Int(scan.loop_count));
    raster.set_field("palette", MetadataValue::Int(1));
    if scan.colours > 0 {
        // `nsgifload.c:337-343`: ceil(log2(colours)) over the table as it
        // sits on the wire. The table is always a power of two there, so the
        // logarithm is exact and `trailing_zeros` is the same number without
        // the float round trip.
        let bits = scan.colours.next_power_of_two().trailing_zeros();
        raster.set_field("bits-per-sample", MetadataValue::Int(i64::from(bits)));
    }
    if scan.interlaced {
        raster.set_field("interlaced", MetadataValue::Int(1));
    }
    Ok(raster)
}

/// Open a decoder over `bytes` producing palette indices rather than RGBA.
///
/// Indexed output is what makes the transparent index visible: the crate's
/// RGBA path resolves it internally, and libviprs needs it to decide between
/// three and four bands and to expand the palette itself.
fn open(bytes: &[u8]) -> Result<gif::Decoder<Cursor<&[u8]>>, GifError> {
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::Indexed);
    // Unrecognised blocks end the scan rather than being skipped, because
    // that is where libnsgif stops too: `garden.gif` in the reference suite
    // carries a stray `0x00` after its 35th frame, and vips reports
    // `n-pages: 35`, not the 48 a decoder that skips ahead and keeps
    // hunting finds in the 1.3 MB of tail. The blocks before the stray one
    // still load, because the scan keeps what it found (see [`scan_file`]).
    options.allow_unknown_blocks(false);
    options.check_lzw_end_code(false);
    options.read_info(Cursor::new(bytes)).map_err(decode_error)
}

/// Walk every frame's metadata without decoding any pixels.
///
/// This is `vips_foreign_load_nsgif_header`'s scan (`nsgifload.c:424-435`):
/// the band count, the page count and the palette depth are all properties
/// of the whole file, so they cannot be read off frame 0 alone.
fn scan_file(bytes: &[u8]) -> Result<FileScan, GifError> {
    let mut decoder = open(bytes)?;
    let global = decoder
        .global_palette()
        .map_or(0, |p| p.len() / PALETTE_STRIDE);
    let mut scan = FileScan {
        frames: 0,
        has_transparency: false,
        interlaced: false,
        colours: global,
        loop_count: 1,
    };
    // A scan that hits trouble stops where it is rather than failing:
    // vips' `fail-on` defaults to `VIPS_FAIL_ON_NONE`, so
    // `nsgifload.c:388-406` downgrades every complaint from
    // `nsgif_data_scan` to a `g_warning`, then calls `nsgif_data_complete`
    // so the frames that did parse are still readable.
    loop {
        match decoder.next_frame_info() {
            Ok(Some(frame)) => {
                scan.frames += 1;
                scan.has_transparency |= frame.transparent.is_some();
                scan.interlaced |= frame.interlaced;
                if let Some(local) = &frame.palette {
                    scan.colours = scan.colours.max(local.len() / PALETTE_STRIDE);
                }
            }
            Ok(None) => break,
            Err(err) => {
                // A frame whose pixels run off the end of the file counts as
                // transparent, because the rows that never arrived stay
                // uncomposited. That is not a guess: truncating six bytes
                // off an 8x8 opaque GIF flips `vipsheader` from `3 bands` to
                // `4 bands`, and it is why `truncated.gif` in the reference
                // suite loads as RGBA despite its graphic control extension
                // clearing the transparency flag.
                scan.has_transparency |= ran_out_of_data(&err);
                break;
            }
        }
    }
    // vips reports `loop` as libnsgif's `loop_max`, which is one more than
    // the NETSCAPE count except that a count of zero means forever and no
    // extension at all means play once. Measured across the reference
    // suite; `gif-loop`, the deprecated field, is the raw count instead.
    scan.loop_count = match decoder.repeat() {
        gif::Repeat::Infinite => 0,
        gif::Repeat::Finite(0) => 1,
        gif::Repeat::Finite(n) => i64::from(n) + 1,
    };
    Ok(scan)
}

/// Whether a decode failure is "the file stopped early" rather than "the
/// file says something impossible".
///
/// The two are treated differently on load: running out of data leaves part
/// of the canvas uncomposited and therefore transparent, where a malformed
/// block simply ends the scan.
fn ran_out_of_data(err: &gif::DecodingError) -> bool {
    match err {
        gif::DecodingError::UnexpectedEof => true,
        gif::DecodingError::Io(io) => io.kind() == std::io::ErrorKind::UnexpectedEof,
        _ => false,
    }
}

/// Wrap a `gif` crate failure as this module's typed decode error.
fn decode_error(err: gif::DecodingError) -> GifError {
    GifError::Decode {
        message: err.to_string(),
    }
}

impl Raster {
    /// Encode as GIF bytes (libvips `gifsave_buffer`).
    ///
    /// Writes a single-frame GIF89a: a global colour table quantised to
    /// `min(255, 1 << bitdepth)` entries, one image block, and the NETSCAPE
    /// looping extension cgif always emits. Requires an 8-bit raster
    /// (`Gray8` / `Rgb8` / `Rgba8`); see the [module docs](crate::gif) for
    /// the alpha threshold, the reserved transparent index, and where the
    /// output deliberately differs from vips.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] for a raster that is not 8-bit, or whose
    /// width or height exceeds the 65535-pixel GIF axis limit (vips rejects
    /// the same case as `"frame too large"`, `cgifsave.c:744-750`), or if
    /// the GIF writer itself fails.
    pub fn encode_gif(&self, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
        let (width, height) = (self.width(), self.height());
        let (Ok(gif_width), Ok(gif_height)) = (u16::try_from(width), u16::try_from(height)) else {
            return Err(EncodeError::encode(format!(
                "gif: frame too large; {width}x{height} exceeds the 65535-pixel GIF axis limit"
            )));
        };
        let pixels = self.gif_rgba()?;

        let max_colours = options.max_colours();
        let has_transparency = pixels.iter().any(|p| p[3] == 0);
        let opaque_budget = if has_transparency {
            max_colours - 1
        } else {
            max_colours
        };
        let mut palette = if has_transparency {
            let opaque: Vec<[u8; 4]> = pixels.iter().copied().filter(|p| p[3] != 0).collect();
            crate::encode::quantize_palette(&opaque, opaque_budget)
        } else {
            crate::encode::quantize_palette(&pixels, opaque_budget)
        };
        palette.sort_unstable();
        // Reserve index 0 for transparency whenever the palette does not
        // saturate. libimagequant does this unconditionally when it has room
        // (measured over twelve bitdepth/colour-count pairs), and cgif then
        // takes the free index for its own optimisation, so an opaque source
        // with palette headroom comes back from `vips gifsave` as four
        // bands. Matching that keeps the reload band count in parity.
        let reserved = has_transparency || palette.len() < max_colours;
        let offset = u8::from(reserved);

        let mut indices = remap(&pixels, &palette, offset, options.dither, width, height);
        if options.interlaced {
            indices = interlace(&indices, width, height);
        }

        let mut table = Vec::with_capacity((palette.len() + 1) * PALETTE_STRIDE);
        if reserved {
            table.extend_from_slice(&[0, 0, 0]);
        }
        for colour in &palette {
            table.extend_from_slice(&colour[..PALETTE_STRIDE]);
        }

        let mut out = Vec::new();
        {
            let mut encoder = gif::Encoder::new(&mut out, gif_width, gif_height, &table)
                .map_err(encode_error_from_gif)?;
            // cgif sets CGIF_ATTR_IS_ANIMATED with `numLoops = 0` for every
            // file it writes, single-frame ones included, so the NETSCAPE
            // block is there even on a still. Measured on `vips gifsave` of
            // a one-frame image: `NETSCAPE2.0` with a loop count of 0.
            encoder
                .set_repeat(gif::Repeat::Infinite)
                .map_err(encode_error_from_gif)?;
            let mut frame = gif::Frame::from_indexed_pixels(
                gif_width,
                gif_height,
                indices,
                reserved.then_some(0),
            );
            frame.interlaced = options.interlaced;
            // cgif writes disposal "keep" on every frame.
            frame.dispose = gif::DisposalMethod::Keep;
            encoder.write_frame(&frame).map_err(encode_error_from_gif)?;
        }
        Ok(out)
    }

    /// Save as a GIF file (libvips `gifsave`).
    ///
    /// [`Raster::encode_gif`] with the bytes written to `path`.
    ///
    /// # Errors
    ///
    /// [`SaveError::Encode`] for anything [`Raster::encode_gif`] rejects, or
    /// [`SaveError::Io`] if the write fails.
    pub fn save_gif(&self, path: &Path, options: SaveOptions) -> Result<(), SaveError> {
        let bytes = self.encode_gif(options).map_err(|e| match e {
            EncodeError::Io(io) => SaveError::Io(io),
            other => SaveError::Encode(crate::sink::SinkError::EncodeMsg(other.to_string())),
        })?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// This raster as thresholded RGBA quadruples, ready to quantise.
    ///
    /// `cgifsave.c:538-548` promotes any alpha at or above 128 to 255 and
    /// zeroes the whole pixel below it, which is both the alpha model GIF
    /// can express and, per the comment there, what "helps the quantiser
    /// generate a better palette".
    fn gif_rgba(&self) -> Result<Vec<[u8; 4]>, EncodeError> {
        let channels = match self.format() {
            PixelFormat::Gray8 => 1,
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 => 4,
            other => {
                return Err(EncodeError::encode(format!(
                    "gif encode requires an 8-bit raster (Gray8/Rgb8/Rgba8), got {other:?}"
                )));
            }
        };
        let data = self.data();
        let count = self.width() as usize * self.height() as usize;
        let mut pixels = Vec::with_capacity(count);
        for p in 0..count {
            let base = p * channels;
            pixels.push(match channels {
                1 => {
                    let g = data[base];
                    [g, g, g, u8::MAX]
                }
                3 => [data[base], data[base + 1], data[base + 2], u8::MAX],
                _ if data[base + 3] >= ALPHA_THRESHOLD => {
                    [data[base], data[base + 1], data[base + 2], u8::MAX]
                }
                _ => [0, 0, 0, 0],
            });
        }
        Ok(pixels)
    }
}

/// Map `pixels` onto `palette`, adding `offset` to every index so a reserved
/// transparent entry can sit at 0.
///
/// At `dither == 0` this is a plain nearest-colour lookup, cached by exact
/// colour. Above 0 it is Floyd-Steinberg error diffusion with the propagated
/// error scaled by the option, which gives the same two endpoints
/// `liq_set_dithering_level` has (`cgifsave.c:584`); the dithered pixels in
/// between differ, because libimagequant's variant is edge-aware and
/// serpentine where this one is a plain left-to-right sweep.
///
/// Transparent pixels take index 0 and neither absorb nor emit error, since
/// `cgifsave.c:541-547` has already zeroed them out of the quantiser's view.
fn remap(
    pixels: &[[u8; 4]],
    palette: &[[u8; 4]],
    offset: u8,
    dither: f64,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let dither = dither.clamp(0.0, 1.0);
    if palette.is_empty() {
        return vec![0u8; pixels.len()];
    }
    if dither == 0.0 {
        let mut cache: std::collections::HashMap<[u8; 4], u8> = std::collections::HashMap::new();
        return pixels
            .iter()
            .map(|&p| {
                if p[3] == 0 {
                    0
                } else {
                    *cache
                        .entry(p)
                        .or_insert_with(|| crate::encode::nearest(palette, p) + offset)
                }
            })
            .collect();
    }

    let stride = width as usize;
    let strength = dither as f32;
    let mut indices = vec![0u8; pixels.len()];
    // Two rows of carried error is all Floyd-Steinberg ever needs: the
    // kernel reaches one pixel right and one row down.
    let mut this_row = vec![[0f32; PALETTE_STRIDE]; stride];
    let mut next_row = vec![[0f32; PALETTE_STRIDE]; stride];
    for y in 0..height as usize {
        for x in 0..stride {
            let pixel = pixels[y * stride + x];
            if pixel[3] == 0 {
                this_row[x] = [0.0; PALETTE_STRIDE];
                continue;
            }
            let mut wanted = [0f32; PALETTE_STRIDE];
            let mut probe = [u8::MAX; 4];
            for c in 0..PALETTE_STRIDE {
                let v = f32::from(pixel[c]) + this_row[x][c];
                wanted[c] = v;
                probe[c] = v.clamp(0.0, 255.0).round() as u8;
            }
            let chosen = crate::encode::nearest(palette, probe);
            indices[y * stride + x] = chosen + offset;
            let entry = palette[chosen as usize];
            let mut error = [0f32; PALETTE_STRIDE];
            for c in 0..PALETTE_STRIDE {
                error[c] = (wanted[c] - f32::from(entry[c])) * strength;
            }
            // The classic kernel: 7/16 right, 3/16 down-left, 5/16 down,
            // 1/16 down-right.
            let diffuse = |row: &mut [[f32; PALETTE_STRIDE]], dx: isize, weight: f32| {
                let Some(nx) = x.checked_add_signed(dx).filter(|&nx| nx < stride) else {
                    return;
                };
                for c in 0..PALETTE_STRIDE {
                    row[nx][c] += error[c] * weight;
                }
            };
            diffuse(&mut this_row, 1, 7.0 / 16.0);
            diffuse(&mut next_row, -1, 3.0 / 16.0);
            diffuse(&mut next_row, 0, 5.0 / 16.0);
            diffuse(&mut next_row, 1, 1.0 / 16.0);
        }
        std::mem::swap(&mut this_row, &mut next_row);
        next_row.fill([0.0; PALETTE_STRIDE]);
    }
    indices
}

/// Reorder `indices` from progressive rows into GIF's four interlace passes.
///
/// The `gif` crate's encoder sets the descriptor bit but never touches the
/// buffer, so the row order has to be built here. cgif reorders too: an
/// interlaced and a progressive save of the same image differ from the
/// image-descriptor byte onwards, and both reload identically.
fn interlace(indices: &[u8], width: u32, height: u32) -> Vec<u8> {
    let stride = width as usize;
    let mut out = Vec::with_capacity(indices.len());
    for (first, step) in INTERLACE_PASSES {
        let mut row = first;
        while row < height {
            let start = row as usize * stride;
            out.extend_from_slice(&indices[start..start + stride]);
            row += step;
        }
    }
    out
}

/// Wrap a `gif` crate encode failure on the shared [`EncodeError`] spine.
fn encode_error_from_gif(err: gif::EncodingError) -> EncodeError {
    match err {
        gif::EncodingError::Io(io) => EncodeError::Io(io),
        other => EncodeError::encode(format!("gif: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::source::{DeclaredGeometry, DecodeLimits, decode_bytes};

    /// One frame of a hand-assembled GIF fixture.
    struct Frame {
        /// Frame origin inside the logical screen.
        left: u16,
        /// Frame origin inside the logical screen.
        top: u16,
        /// Frame extent.
        width: u16,
        /// Frame extent.
        height: u16,
        /// Palette indices, row-major, `width * height` of them, in
        /// progressive order even when `interlaced` is set.
        indices: Vec<u8>,
        /// The graphic control extension's transparent index, if any.
        transparent: Option<u8>,
        /// Whether the frame is stored in the four-pass interlaced order.
        interlaced: bool,
        /// Frame delay, in centiseconds, as it goes on the wire.
        delay_cs: u16,
    }

    impl Frame {
        /// A full-canvas opaque frame.
        fn full(width: u16, height: u16, indices: Vec<u8>) -> Self {
            Self {
                left: 0,
                top: 0,
                width,
                height,
                indices,
                transparent: None,
                interlaced: false,
                delay_cs: 0,
            }
        }
    }

    /// Assemble a GIF89a from a global palette and a frame list.
    ///
    /// Everything the decode tests turn on -- the transparent index, the
    /// interlace bit, the frame rectangle, the NETSCAPE loop count -- is
    /// written here explicitly, so the fixtures do not depend on
    /// [`Raster::encode_gif`] and cannot drift with it.
    fn fixture(
        screen: (u16, u16),
        palette: &[[u8; 3]],
        background: u8,
        loop_ext: Option<u16>,
        frames: &[Frame],
    ) -> Vec<u8> {
        let table_size = flag_size(palette.len());
        let entries = 2usize << table_size;
        let mut out = Vec::from(*b"GIF89a");
        out.extend_from_slice(&screen.0.to_le_bytes());
        out.extend_from_slice(&screen.1.to_le_bytes());
        out.push(0x80 | table_size);
        out.push(background);
        out.push(0);
        for i in 0..entries {
            out.extend_from_slice(palette.get(i).unwrap_or(&[0, 0, 0]));
        }
        if let Some(count) = loop_ext {
            out.extend_from_slice(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01");
            out.extend_from_slice(&count.to_le_bytes());
            out.push(0);
        }
        for frame in frames {
            let mut flags = 0u8;
            if frame.transparent.is_some() {
                flags |= 1;
            }
            flags |= 1 << 2; // disposal method "keep", as cgif writes
            out.extend_from_slice(&[0x21, 0xF9, 4, flags]);
            out.extend_from_slice(&frame.delay_cs.to_le_bytes());
            out.push(frame.transparent.unwrap_or(0));
            out.push(0);
            out.push(0x2C);
            out.extend_from_slice(&frame.left.to_le_bytes());
            out.extend_from_slice(&frame.top.to_le_bytes());
            out.extend_from_slice(&frame.width.to_le_bytes());
            out.extend_from_slice(&frame.height.to_le_bytes());
            out.push(if frame.interlaced { 0x40 } else { 0 });
            let min_code_size = (table_size + 1).max(2);
            out.push(min_code_size);
            let stored = if frame.interlaced {
                let mut rows = Vec::with_capacity(frame.indices.len());
                for row in interlace_rows(u32::from(frame.height)) {
                    let start = row as usize * frame.width as usize;
                    rows.extend_from_slice(&frame.indices[start..start + frame.width as usize]);
                }
                rows
            } else {
                frame.indices.clone()
            };
            for chunk in lzw_literal(&stored, min_code_size).chunks(255) {
                out.push(chunk.len() as u8);
                out.extend_from_slice(chunk);
            }
            out.push(0);
        }
        out.push(0x3B);
        out
    }

    /// The GIF colour-table size flag for `n` entries: the table on the wire
    /// holds `2 << flag` colours.
    fn flag_size(n: usize) -> u8 {
        (n.clamp(2, 256).next_power_of_two().trailing_zeros() - 1) as u8
    }

    /// LZW-encode `indices` as literals only: a clear code, one code per
    /// pixel, then end-of-information.
    ///
    /// No dictionary matches are emitted, but the code width still grows on
    /// the schedule a decoder tracks, so the stream is ordinary GIF LZW and
    /// any conformant decoder reads it.
    fn lzw_literal(indices: &[u8], min_code_size: u8) -> Vec<u8> {
        let clear = 1u16 << min_code_size;
        let mut width = u32::from(min_code_size) + 1;
        let mut next_code = clear + 2;
        let mut out = Vec::new();
        let mut acc = 0u32;
        let mut bits = 0u32;
        let mut emit = |code: u16, width: u32, out: &mut Vec<u8>| {
            acc |= u32::from(code) << bits;
            bits += width;
            while bits >= 8 {
                out.push((acc & 0xFF) as u8);
                acc >>= 8;
                bits -= 8;
            }
        };
        emit(clear, width, &mut out);
        for &index in indices {
            emit(u16::from(index), width, &mut out);
            next_code += 1;
            if u32::from(next_code) > (1u32 << width) && width < 12 {
                width += 1;
            }
        }
        emit(clear + 1, width, &mut out);
        if bits > 0 {
            out.push((acc & 0xFF) as u8);
        }
        out
    }

    /// The rows of a `height`-tall frame in GIF's four-pass interlaced
    /// order: rows 0, 8, 16, ...; then 4, 12, ...; then 2, 6, ...; then
    /// every odd row.
    fn interlace_rows(height: u32) -> Vec<u32> {
        let mut rows = Vec::with_capacity(height as usize);
        for (start, step) in [(0u32, 8u32), (4, 8), (2, 4), (1, 2)] {
            let mut row = start;
            while row < height {
                rows.push(row);
                row += step;
            }
        }
        rows
    }

    /// A four-colour palette and a 4x4 frame that uses all of it.
    fn opaque_fixture() -> Vec<u8> {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let indices: Vec<u8> = (0..16u8).map(|i| i % 4).collect();
        fixture((4, 4), &palette, 0, None, &[Frame::full(4, 4, indices)])
    }

    /**
     * Tests that a GIF with no transparent index anywhere loads as a
     * three-band raster, as `vips gifload` does. Works by decoding a
     * hand-assembled four-colour GIF through the shared sniff route and
     * reading the pixel format back. `nsgifload.c:271` sizes the image
     * `has_transparency ? 4 : 3`, and `:522-534` drops the alpha byte per
     * pixel when it is 3; measured on the reference `cramps.gif`, which
     * `vipsheader` reports as `3 bands`.
     * Input: a 4x4 GIF whose graphic control extension clears the
     * transparency flag -> Output: `PixelFormat::Rgb8` and the four palette
     * colours, byte for byte.
     */
    #[test]
    fn an_opaque_gif_loads_as_three_bands() {
        let raster = decode_bytes(&opaque_fixture()).expect("the fixture is a valid GIF");
        assert_eq!(raster.width(), 4);
        assert_eq!(raster.height(), 4);
        assert_eq!(
            raster.format(),
            PixelFormat::Rgb8,
            "vips gifload emits 3 bands when no frame declares transparency"
        );
        assert_eq!(
            &raster.data()[..12],
            &[0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 255, 0],
            "the palette must be expanded exactly; GIF's LZW is lossless"
        );
    }

    /**
     * Tests that a declared transparent index keeps the fourth band and
     * expands to transparent black, so a transparent GIF is not silently
     * flattened. Works by decoding a fixture whose graphic control
     * extension names index 0 as transparent and checking the first pixel.
     * `nsgifload.c:431-432` sets `has_transparency` from any frame's
     * transparency flag.
     * Input: a 4x4 GIF with transparent index 0 -> Output:
     * `PixelFormat::Rgba8`, pixel 0 fully transparent, pixel 1 opaque red.
     */
    #[test]
    fn a_transparent_index_keeps_four_bands() {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let indices: Vec<u8> = (0..16u8).map(|i| i % 4).collect();
        let mut frame = Frame::full(4, 4, indices);
        frame.transparent = Some(0);
        let bytes = fixture((4, 4), &palette, 0, None, &[frame]);
        let raster = decode_bytes(&bytes).expect("the fixture is a valid GIF");
        assert_eq!(raster.format(), PixelFormat::Rgba8);
        assert_eq!(&raster.data()[..8], &[0, 0, 0, 0, 255, 0, 0, 255]);
    }

    /**
     * Tests that the canvas around a frame smaller than the logical screen
     * comes back transparent black rather than the background colour, which
     * is what libnsgif renders and what `vips getpoint` reports. Works by
     * building an 8x8 screen with the background index pointing at blue and
     * a 4x4 red frame inset at (2,2), then reading a corner pixel.
     * Measured: `vips gifload` on exactly this file prints `0 0 0` at (0,0)
     * while `vipsheader` reports `background: 0 0 255`.
     * Input: an 8x8 GIF with an inset opaque frame -> Output: corner pixels
     * black, the inset red.
     */
    #[test]
    fn the_canvas_outside_frame_zero_is_transparent_black_not_the_background() {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let frame = Frame {
            left: 2,
            top: 2,
            width: 4,
            height: 4,
            indices: vec![1; 16],
            transparent: None,
            interlaced: false,
            delay_cs: 0,
        };
        let bytes = fixture((8, 8), &palette, 2, None, &[frame]);
        let raster = decode_bytes(&bytes).expect("the fixture is a valid GIF");
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(
            &raster.data()[..3],
            &[0, 0, 0],
            "the corner is not the background colour"
        );
        let inset = (2 * 8 + 2) * 3;
        assert_eq!(&raster.data()[inset..inset + 3], &[255, 0, 0]);
    }

    /**
     * Tests that an interlaced GIF is de-interlaced on load, so the row
     * order on the wire is invisible to callers. Works by writing the same
     * eight rows twice, once progressive and once in GIF's four-pass order
     * with the interlace bit set, and requiring the two decodes to be
     * byte-identical.
     * Input: two 8x8 GIFs differing only in row order and the interlace bit
     * -> Output: identical rasters.
     */
    #[test]
    fn an_interlaced_gif_decodes_to_the_same_rows_as_a_progressive_one() {
        let palette: Vec<[u8; 3]> = (0..8u8).map(|i| [i * 30, 255 - i * 30, i * 17]).collect();
        let indices: Vec<u8> = (0..64u8).map(|i| i / 8).collect();
        let progressive = fixture(
            (8, 8),
            &palette,
            0,
            None,
            &[Frame::full(8, 8, indices.clone())],
        );
        let mut frame = Frame::full(8, 8, indices);
        frame.interlaced = true;
        let interlaced = fixture((8, 8), &palette, 0, None, &[frame]);
        let a = decode_bytes(&progressive).expect("progressive fixture decodes");
        let b = decode_bytes(&interlaced).expect("interlaced fixture decodes");
        assert_eq!(a.data(), b.data(), "interlaced rows must be reassembled");
    }

    /**
     * Tests that an animated GIF loads its first frame and still reports how
     * many pages the file holds, so a caller can tell an animation from a
     * still without a page model. `vips gifload` defaults to `page=0, n=1`
     * and sets `n-pages` from the whole file
     * (`nsgifload.c:284-285`). Works by assembling a three-frame GIF and
     * reading the first frame's pixels and the attached field.
     * Input: a 4x4 GIF with three frames -> Output: frame zero's pixels and
     * `n-pages == 3`.
     */
    #[test]
    fn an_animation_loads_frame_zero_and_records_the_page_count() {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let frames: Vec<Frame> = (0..3u8)
            .map(|f| Frame::full(4, 4, vec![f + 1; 16]))
            .collect();
        let bytes = fixture((4, 4), &palette, 0, Some(0), &frames);
        let raster = decode_bytes(&bytes).expect("the fixture is a valid GIF");
        assert_eq!(&raster.data()[..3], &[255, 0, 0], "frame zero is index 1");
        assert_eq!(
            raster.get_int("n-pages"),
            Some(3),
            "n-pages counts every frame in the file, not the ones loaded"
        );
    }

    /**
     * Tests what the number under `n-pages` actually counts, which is the
     * question issue #635 was filed on: the frames in the file and nothing
     * else. The test above uses three frames, which is also the band count
     * a GIF with no transparency expands to, so a loader that attached the
     * wrong number would still read as right there. Works by assembling a
     * FIVE-frame GIF on a 4x4 screen with a five-entry palette, so the
     * expected count matches neither axis, nor either band count, nor the
     * one frame that was loaded.
     * Input: a 4x4 GIF with five frames -> Output: frame zero's pixels and
     * `get_n_pages() == 5`.
     */
    #[test]
    fn n_pages_counts_the_frames_in_the_file_and_nothing_else() {
        let palette = [
            [0, 0, 0],
            [255, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [255, 255, 0],
        ];
        let frames: Vec<Frame> = (0..5u8).map(|f| Frame::full(4, 4, vec![f; 16])).collect();
        let bytes = fixture((4, 4), &palette, 0, Some(0), &frames);

        let raster = decode_bytes(&bytes).expect("the fixture is a valid GIF");
        assert_eq!(&raster.data()[..3], &[0, 0, 0], "frame zero is index 0");
        assert_eq!((raster.width(), raster.height()), (4, 4));
        assert_eq!(
            raster.get_n_pages(),
            5,
            "n-pages is the frame count, not an axis, a band count, or the \
             one frame that was loaded"
        );
    }

    /**
     * Tests that the GIF encoder produces bytes the decoder reads back
     * unchanged when the raster already fits in the palette. GIF's LZW is
     * exactly lossless, so this is a byte-for-byte equality and not a
     * tolerance band; the `lossy_decoder: true` tag on `cogs.gif` in the
     * existing oracle capture is wrong and would let a real palette or
     * expansion bug through. Measured against vips: `gifsave` of a
     * sixteen-colour source reloads with `max_abs_diff 0`.
     * Input: a 16x16 Rgb8 raster with four distinct colours -> Output: a GIF
     * whose decode equals the input pixel for pixel.
     */
    #[test]
    fn encode_then_decode_is_exactly_lossless_within_the_palette() {
        let colours = [[0u8, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        let mut data = Vec::with_capacity(16 * 16 * 3);
        for i in 0..256usize {
            data.extend_from_slice(&colours[i % 4]);
        }
        let im = Raster::new(16, 16, PixelFormat::Rgb8, data.clone()).unwrap();
        let bytes = im
            .encode_gif(SaveOptions::default())
            .expect("a four-colour raster is well inside a GIF palette");
        let back = decode_bytes(&bytes).expect("libviprs must read back what it writes");
        assert_eq!(back.width(), 16);
        assert_eq!(back.height(), 16);
        let rgb: Vec<u8> = match back.format() {
            PixelFormat::Rgb8 => back.data().to_vec(),
            PixelFormat::Rgba8 => back
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| p[..3].to_vec())
                .collect(),
            other => panic!("gifload emits 3 or 4 bands, got {other:?}"),
        };
        assert_eq!(
            rgb, data,
            "LZW is lossless; a palette that fits must round-trip"
        );
    }

    /**
     * Tests that a raster whose alpha straddles 128 is thresholded the way
     * `cgifsave.c:538-548` does: `p[3] >= 128` becomes fully opaque with its
     * colour intact, anything below becomes fully transparent black.
     * Measured on vips 8.18.4 with a 32-wide alpha ramp: columns 0..15
     * (alpha 0..120) reload as `0 0 0 0` and columns 16..31 (alpha 128..248)
     * as `200 100 50 255`.
     * Input: a 32x1 Rgba8 ramp of alpha `x * 8` over a constant colour ->
     * Output: a hard cut at alpha 128 with no partial values anywhere.
     */
    #[test]
    fn alpha_is_thresholded_at_128_on_save() {
        let mut data = Vec::with_capacity(32 * 4);
        for x in 0..32u32 {
            data.extend_from_slice(&[200, 100, 50, (x * 8).min(255) as u8]);
        }
        let im = Raster::new(32, 1, PixelFormat::Rgba8, data).unwrap();
        let bytes = im.encode_gif(SaveOptions::default()).expect("encodes");
        let back = decode_bytes(&bytes).expect("decodes");
        assert_eq!(back.format(), PixelFormat::Rgba8);
        for (x, pixel) in back.data().as_chunks::<4>().0.iter().enumerate() {
            let expected: [u8; 4] = if x >= 16 {
                [200, 100, 50, 255]
            } else {
                [0, 0, 0, 0]
            };
            assert_eq!(*pixel, expected, "alpha {} at x={x}", x * 8);
        }
    }

    /**
     * Tests that the interlace option reaches the wire, so a caller asking
     * for a progressive GIF gets one. Works by encoding the same raster
     * twice and requiring the image descriptor's interlace bit to differ
     * while both decode to the same pixels. Measured: `vips gifsave
     * --interlace` sets bit 6 of the image descriptor's packed byte and
     * reorders the stored rows, and vips reloads both files identically.
     * Input: an 8x8 Rgb8 raster -> Output: two GIFs whose interlace bit
     * differs and whose decodes agree.
     */
    #[test]
    fn the_interlace_option_reaches_the_image_descriptor() {
        let data: Vec<u8> = (0..64u32)
            .flat_map(|i| [(i * 4) as u8, 255 - (i * 3) as u8, (i * 7) as u8])
            .collect();
        let im = Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap();
        let plain = im.encode_gif(SaveOptions::default()).expect("encodes");
        let woven = im
            .encode_gif(SaveOptions {
                interlaced: true,
                ..Default::default()
            })
            .expect("encodes");
        assert_eq!(wire(&plain).flags & 0x40, 0);
        assert_ne!(
            wire(&woven).flags & 0x40,
            0,
            "interlace must set bit 6 of the image descriptor"
        );
        let a = decode_bytes(&plain).expect("decodes");
        let b = decode_bytes(&woven).expect("decodes");
        assert_eq!(
            a.data(),
            b.data(),
            "interlacing changes storage, not pixels"
        );
    }

    /// The wire-level facts a GIF's first frame carries, read back off the
    /// bytes without decoding any pixels.
    #[derive(Debug, PartialEq)]
    struct Wire {
        /// Entries in the global colour table, padded to a power of two.
        colours: usize,
        /// The image descriptor's packed byte.
        flags: u8,
        /// Whether the graphic control extension declares a transparent
        /// index, and which one.
        transparent: Option<u8>,
        /// The LZW minimum code size that opens the image data.
        min_code_size: u8,
    }

    /// Parse the header, the global colour table and the first frame's
    /// graphic control extension and image descriptor out of `bytes`.
    fn wire(bytes: &[u8]) -> Wire {
        let mut p = 13usize;
        let colours = if bytes[10] & 0x80 != 0 {
            let n = 2usize << (bytes[10] & 7);
            p += PALETTE_STRIDE * n;
            n
        } else {
            0
        };
        let mut transparent = None;
        loop {
            match bytes[p] {
                0x21 => {
                    let label = bytes[p + 1];
                    p += 2;
                    let mut first = true;
                    while bytes[p] != 0 {
                        let len = bytes[p] as usize;
                        if label == 0xF9 && first && bytes[p + 1] & 1 != 0 {
                            transparent = Some(bytes[p + 4]);
                        }
                        first = false;
                        p += 1 + len;
                    }
                    p += 1;
                }
                0x2C => {
                    let flags = bytes[p + 9];
                    p += 10;
                    if flags & 0x80 != 0 {
                        p += PALETTE_STRIDE * (2usize << (flags & 7));
                    }
                    return Wire {
                        colours,
                        flags,
                        transparent,
                        min_code_size: bytes[p],
                    };
                }
                other => panic!("unexpected block {other:#x} at {p}"),
            }
        }
    }

    /// A `w * h` raster whose pixel `(x, y)` is `[x * 5, y * 8, (x + y) * 3]`.
    ///
    /// Every pixel is a distinct colour, and neighbours are close together,
    /// so the palette saturates and the quantiser's error is visible. This
    /// is the source the quantisation band below was measured on.
    fn gradient(w: u32, h: u32) -> Raster {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[(x * 5) as u8, (y * 8) as u8, ((x + y) * 3) as u8]);
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// A `16 * 16` raster cycling through `unique` distinct colours.
    fn cycling(unique: u32) -> Raster {
        let mut data = Vec::with_capacity(16 * 16 * 3);
        for i in 0..256u32 {
            let c = i % unique;
            data.extend_from_slice(&[
                (c * 7 % 256) as u8,
                (c * 13 % 256) as u8,
                (c * 29 % 256) as u8,
            ]);
        }
        Raster::new(16, 16, PixelFormat::Rgb8, data).unwrap()
    }

    /// A decoded raster's pixels as RGB triples, whichever band count
    /// `decode_gif` chose.
    fn as_rgb(raster: &Raster) -> Vec<u8> {
        match raster.format() {
            PixelFormat::Rgb8 => raster.data().to_vec(),
            PixelFormat::Rgba8 => raster
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| p[..PALETTE_STRIDE].to_vec())
                .collect(),
            other => panic!("gifload emits 3 or 4 bands, got {other:?}"),
        }
    }

    /// Mean and maximum absolute per-byte difference between two buffers.
    fn error(a: &[u8], b: &[u8]) -> (f64, u8) {
        assert_eq!(a.len(), b.len());
        let total: u64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| u64::from(x.abs_diff(*y)))
            .sum();
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        (total as f64 / a.len() as f64, worst)
    }

    /**
     * Tests that `bitdepth` sizes the colour table exactly as `vips gifsave
     * --bitdepth N` does, over the whole 1..=8 range vips accepts. Works by
     * encoding one 768-colour source at each depth and reading the global
     * colour table size and the LZW minimum code size back off the bytes.
     * Measured on vips 8.18.4: the table is `2^N` entries and the minimum
     * code size is `max(2, N)`, the floor coming from the GIF89a
     * requirement that it never drops below 2.
     * Input: a 16x16 raster of 256 distinct colours, encoded eight times ->
     * Output: table sizes 2, 4, 8, ..., 256 and code sizes 2, 2, 3, ..., 8.
     */
    #[test]
    fn bitdepth_sizes_the_colour_table_the_way_vips_does() {
        // (bitdepth, colour table entries, LZW minimum code size), every row
        // read off a `vips gifsave --bitdepth N` output.
        let measured = [
            (1u8, 2usize, 2u8),
            (2, 4, 2),
            (3, 8, 3),
            (4, 16, 4),
            (5, 32, 5),
            (6, 64, 6),
            (7, 128, 7),
            (8, 256, 8),
        ];
        let im = cycling(256);
        for (bitdepth, colours, min_code_size) in measured {
            let bytes = im
                .encode_gif(SaveOptions {
                    bitdepth,
                    ..Default::default()
                })
                .expect("every bitdepth encodes");
            let got = wire(&bytes);
            assert_eq!(got.colours, colours, "bitdepth {bitdepth} colour table");
            assert_eq!(
                got.min_code_size, min_code_size,
                "bitdepth {bitdepth} LZW minimum code size"
            );
        }
    }

    /**
     * Tests the rule that decides whether a saved GIF carries a transparent
     * index, which is what decides whether it reloads as three bands or
     * four. `cgifsave.c:795-796` caps the palette at `min(255, 1 <<
     * bitdepth)` precisely so one index stays free, and libimagequant takes
     * it whenever the colours do not fill the cap. Works by walking twelve
     * (bitdepth, distinct-colour) pairs whose vips answers were measured and
     * requiring the same table size and transparency flag from each.
     * Input: twelve encodes across bitdepths 1, 2, 4, 6 and 8 -> Output: the
     * colour table size and transparent-index presence vips produced.
     */
    #[test]
    fn the_palette_reserves_index_zero_unless_it_saturates() {
        // (bitdepth, distinct colours in the source, colour table entries,
        // transparent index present), every row measured on vips 8.18.4.
        let measured = [
            (8u8, 2u32, 4usize, true),
            (8, 16, 32, true),
            (8, 254, 256, true),
            (8, 255, 256, false),
            (8, 256, 256, false),
            (1, 2, 2, false),
            (2, 2, 4, true),
            (2, 8, 4, false),
            (4, 8, 16, true),
            (4, 100, 16, false),
            (6, 8, 16, true),
            (6, 100, 64, false),
        ];
        for (bitdepth, unique, colours, transparent) in measured {
            let bytes = cycling(unique)
                .encode_gif(SaveOptions {
                    bitdepth,
                    ..Default::default()
                })
                .expect("encodes");
            let got = wire(&bytes);
            assert_eq!(
                (got.colours, got.transparent.is_some()),
                (colours, transparent),
                "bitdepth {bitdepth} over {unique} distinct colours"
            );
            assert_eq!(
                got.transparent,
                transparent.then_some(0),
                "the reserved index is always 0, as cgif's transIndex is"
            );
            // The band count on reload is the whole reason to match this.
            let back = decode_bytes(&bytes).expect("decodes");
            let expected = if transparent {
                PixelFormat::Rgba8
            } else {
                PixelFormat::Rgb8
            };
            assert_eq!(back.format(), expected, "reload band count");
        }
    }

    /**
     * Tests the 255-colour ceiling, which is the one place the GIF save
     * surface is lossy for a reason that is not quantiser choice: vips caps
     * the palette at 255 rather than 256 so an index stays free for
     * transparency (`cgifsave.c:795-796`), so `bitdepth: 8` cannot describe
     * a 256-colour image and neither implementation is lossless there.
     * Measured on vips 8.18.4: a 255-colour source reloads with
     * `max_abs_diff 0` and a 256-colour one with `max_abs_diff 16`.
     * Input: 16x16 rasters of 255 then 256 distinct colours -> Output: an
     * exact round trip and then a lossy one.
     */
    #[test]
    fn two_hundred_and_fifty_five_colours_round_trip_and_two_hundred_and_fifty_six_do_not() {
        let exact = cycling(255);
        let bytes = exact.encode_gif(SaveOptions::default()).expect("encodes");
        let back = decode_bytes(&bytes).expect("decodes");
        assert_eq!(
            as_rgb(&back),
            exact.data(),
            "255 colours fit the palette exactly, so nothing may move"
        );

        let over = cycling(256);
        let bytes = over.encode_gif(SaveOptions::default()).expect("encodes");
        let back = decode_bytes(&bytes).expect("decodes");
        assert_ne!(
            as_rgb(&back),
            over.data(),
            "256 colours cannot fit in vips' 255-entry ceiling"
        );
    }

    /**
     * Tests that `dither: 0.0` is exactly nearest-colour mapping and that a
     * higher level diffuses error into neighbouring pixels, which are the
     * two endpoints `liq_set_dithering_level` has (`cgifsave.c:584`). Works
     * by encoding a smooth gradient at a bitdepth low enough to force
     * banding and comparing the index buffers through a decode. Measured on
     * vips: at `--dither 0` a 32-wide grey ramp comes back in four clean
     * bands, and at higher levels pixels near each boundary alternate.
     * Input: a 48x32 gradient at bitdepth 2 -> Output: `dither: 0.0`
     * reproduces the plain nearest mapping, `dither: 1.0` does not.
     */
    #[test]
    fn dither_zero_is_nearest_colour_and_a_higher_level_diffuses() {
        let im = gradient(48, 32);
        let low = SaveOptions {
            dither: 0.0,
            bitdepth: 2,
            ..Default::default()
        };
        let plain = im.encode_gif(low).expect("encodes");
        assert_eq!(
            plain,
            im.encode_gif(low).expect("encodes"),
            "the undithered path must be deterministic"
        );
        let dithered = im
            .encode_gif(SaveOptions {
                dither: 1.0,
                bitdepth: 2,
                ..low
            })
            .expect("encodes");
        let a = decode_bytes(&plain).expect("decodes");
        let b = decode_bytes(&dithered).expect("decodes");
        assert_ne!(
            a.data(),
            b.data(),
            "dithering must actually change which palette entry a pixel takes"
        );
        // Dithering trades local accuracy for a better average, so it may
        // not lower the worst-case error; what it must not do is wander off.
        let (_, worst) = error(&as_rgb(&b), im.data());
        assert!(worst <= 128, "dithered error ran away: max {worst}");
    }

    /**
     * Bounds how far libviprs' palette may sit from the one vips picks,
     * rather than leaving the divergence as a shrug. LZW is exactly
     * lossless, so the only place the two can disagree is quantisation:
     * vips uses libimagequant, libviprs uses the median-cut quantiser that
     * already backs `encode_png_palette`. Works by quantising a gradient
     * whose every pixel is a distinct colour and comparing the reload error
     * with what vips scored on the identical source.
     * Measured on vips 8.18.4, `vips gifsave` of this exact 48x32 gradient:
     * `avg_abs_diff 3.895399`, `max_abs_diff 22`. libviprs scores
     * `avg_abs_diff 3.457465`, `max_abs_diff 12` -- better on both here,
     * and on the reference `synth_rgb8` fixture worse on the mean (3.944 vs
     * 3.366) and better on the worst case (18 vs 23). So neither quantiser
     * dominates, and the honest contract is a band, not an ordering.
     * Input: a 48x32 gradient of 1536 distinct colours -> Output: mean error
     * within one level of vips' 3.895 and a worst case no higher than vips'
     * 22.
     */
    #[test]
    fn quantisation_error_stays_within_the_measured_vips_band() {
        /// `avg_abs_diff` vips scored on this source.
        const VIPS_MEAN: f64 = 3.895_399;
        /// `max_abs_diff` vips scored on this source.
        const VIPS_WORST: u8 = 22;

        let im = gradient(48, 32);
        let bytes = im.encode_gif(SaveOptions::default()).expect("encodes");
        let back = decode_bytes(&bytes).expect("decodes");
        let (mean, worst) = error(&as_rgb(&back), im.data());
        assert!(
            (mean - VIPS_MEAN).abs() <= 1.0,
            "mean absolute error {mean:.6} is more than one level from vips' {VIPS_MEAN:.6}"
        );
        assert!(
            worst <= VIPS_WORST,
            "worst-case error {worst} exceeds vips' {VIPS_WORST}"
        );
    }

    /**
     * Tests that encoding the same raster twice produces the same bytes.
     * The quantiser gathers distinct colours through a `HashMap`, whose
     * iteration order the default `RandomState` reseeds per process, so
     * without the sort in `quantize_palette` the palette came out in a
     * different order on every run and identical input hashed to different
     * output. That is the one thing the content-addressed side of this crate
     * cannot tolerate.
     * Input: one 48x32 gradient encoded twice, at the default options and
     * interlaced -> Output: byte-identical buffers both times.
     */
    #[test]
    fn encoding_is_byte_reproducible() {
        let im = gradient(48, 32);
        for options in [
            SaveOptions::default(),
            SaveOptions {
                interlaced: true,
                bitdepth: 4,
                ..Default::default()
            },
        ] {
            let first = im.encode_gif(options).expect("encodes");
            let second = im.encode_gif(options).expect("encodes");
            assert_eq!(first, second, "{options:?} did not reproduce");
        }
    }

    /**
     * Tests that a single-band raster saves through the grey ramp rather
     * than being refused, which is what vips does: `gifsave` of a `b-w`
     * image reloads as `8 8 8`, `30 30 30`, `60 60 60`, and so on, because
     * `cgifsave.c:741-745` adds an alpha band to whatever `save->ready`
     * handed it and the colourspace conversion has already tripled the grey.
     * Measured on an 8x8 mono ramp.
     * Input: an 8x8 Gray8 ramp -> Output: each grey level back as an equal
     * RGB triple.
     */
    #[test]
    fn a_gray8_raster_saves_through_the_grey_ramp() {
        let data: Vec<u8> = (0..64u32).map(|i| ((i % 8) * 30) as u8).collect();
        let im = Raster::new(8, 8, PixelFormat::Gray8, data).unwrap();
        let bytes = im.encode_gif(SaveOptions::default()).expect("encodes");
        let back = decode_bytes(&bytes).expect("decodes");
        for (x, pixel) in as_rgb(&back)[..24].as_chunks::<3>().0.iter().enumerate() {
            let g = (x as u8) * 30;
            assert_eq!(*pixel, [g, g, g], "grey {g} did not survive as a triple");
        }
    }

    /**
     * Tests that the encoder says plainly what it cannot take rather than
     * panicking or writing something a decoder will choke on. Works by
     * handing it a 16-bit raster, which has no GIF colour model at all, and
     * a raster wider than the 65535-pixel axis the GIF logical screen
     * descriptor can address (vips rejects the same case as "frame too
     * large", `cgifsave.c:744-750`).
     * Input: a `Rgb16` raster and a 65536-wide `Gray8` one -> Output: a
     * typed `EncodeError::Encode` from each, naming what went wrong.
     */
    #[test]
    fn encode_refuses_what_gif_cannot_express() {
        let deep = Raster::new(2, 2, PixelFormat::Rgb16, vec![0u8; 2 * 2 * 3 * 2]).unwrap();
        match deep.encode_gif(SaveOptions::default()) {
            Err(EncodeError::Encode(message)) => {
                assert!(message.contains("8-bit"), "unhelpful message: {message}");
            }
            other => panic!("expected a typed encode error, got {other:?}"),
        }

        let wide = Raster::new(65_536, 1, PixelFormat::Gray8, vec![0u8; 65_536]).unwrap();
        match wide.encode_gif(SaveOptions::default()) {
            Err(EncodeError::Encode(message)) => {
                assert!(
                    message.contains("frame too large"),
                    "unhelpful message: {message}"
                );
            }
            other => panic!("expected a typed encode error, got {other:?}"),
        }
    }

    /**
     * Tests that malformed input arrives as this codec's own typed variant
     * through `SourceError`, so a caller can match the failure instead of
     * parsing a message. Works by decoding a GIF header with nothing after
     * it and by truncating a valid fixture mid-frame.
     * Input: `GIF89a` alone, and the first 20 bytes of a valid GIF ->
     * Output: `SourceError::Gif(GifError::Decode { .. })` from both.
     */
    #[test]
    fn malformed_bytes_surface_a_typed_gif_error() {
        for truncated in [b"GIF89a".to_vec(), opaque_fixture()[..20].to_vec()] {
            match decode_gif(&truncated, DecodeLimits::default()) {
                Err(SourceError::Gif(GifError::Decode { .. } | GifError::NoFrames)) => {}
                other => panic!("expected a typed GifError, got {other:?}"),
            }
        }
    }

    /**
     * Tests that the decode budget is applied to the declared logical screen
     * before any pixel buffer is reserved. A GIF is LZW-compressed, so a
     * few hundred bytes can declare a very large canvas; the ceilings are
     * what stop that turning into an allocation.
     * Input: a 4x4 fixture decoded under a 1-pixel coordinate ceiling, then
     * under a 4-byte allocation budget -> Output:
     * `SourceError::CoordLimitExceeded` and
     * `SourceError::AllocLimitExceeded`, both before the raster is built.
     */
    #[test]
    fn the_decode_budget_is_checked_before_the_canvas_is_allocated() {
        let bytes = opaque_fixture();
        let tight = DecodeLimits::default().with_max_coord(1);
        match decode_gif(&bytes, tight) {
            Err(SourceError::CoordLimitExceeded { width, height, .. }) => {
                assert_eq!((width, height), (4, 4));
            }
            other => panic!("expected CoordLimitExceeded, got {other:?}"),
        }
        let starved = DecodeLimits::default().with_max_alloc_bytes(4);
        match decode_gif(&bytes, starved) {
            Err(SourceError::AllocLimitExceeded { needed_bytes, .. }) => {
                assert_eq!(needed_bytes, 4 * 4 * 3);
            }
            other => panic!("expected AllocLimitExceeded, got {other:?}"),
        }
    }

    /**
     * Tests that the allocation budget bites at exactly the byte the
     * declared canvas costs, and not one byte either side. The case above
     * refuses at a budget of four against a price of forty-eight, which a
     * price wrong by a factor would also refuse; this one cannot pass
     * unless the price is exact.
     * Input: the 4x4 opaque fixture at `max_alloc_bytes` 48 then 47 ->
     * Output: a clean 4x4 three-band decode, then `AllocLimitExceeded
     * { needed: 48 }`.
     */
    #[test]
    fn the_canvas_budget_bites_at_exactly_the_declared_price() {
        let bytes = opaque_fixture();
        let exact = DecodeLimits::default().with_max_alloc_bytes(48);
        let raster = decode_gif(&bytes, exact).expect("48 bytes is exactly a 4x4 RGB canvas");
        assert_eq!((raster.width(), raster.height()), (4, 4));
        assert_eq!(raster.format(), PixelFormat::Rgb8);

        let short = DecodeLimits::default().with_max_alloc_bytes(47);
        let err = decode_gif(&bytes, short).expect_err("47 bytes is one short of the canvas");
        assert!(
            matches!(
                err,
                SourceError::AllocLimitExceeded {
                    what: "GIF canvas",
                    geometry: Some(DeclaredGeometry {
                        width: 4,
                        height: 4,
                        bands: 3,
                    }),
                    needed_bytes: 48,
                    max_alloc_bytes: 47,
                }
            ),
            "{err:?}"
        );
    }

    /**
     * Tests the `loop` field, which is not the NETSCAPE repeat count and is
     * easy to be off by one about. libnsgif reports `loop_max`, and vips
     * copies it straight through (`nsgifload.c:286`): no application
     * extension at all means play once, a stored count of 0 means forever,
     * and a stored count of `n` means `n + 1` plays. Measured across the
     * reference suite: `cramps.gif` carries no extension and reports 1,
     * `garden.gif` carries 0 and reports 0, `dispose-background.gif`
     * carries 10 and reports 11.
     * Input: the same fixture with no extension, with a count of 0, and
     * with a count of 10 -> Output: `loop` 1, 0, and 11.
     */
    #[test]
    fn loop_follows_the_vips_convention_not_the_netscape_count() {
        let palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]];
        for (stored, expected) in [(None, 1), (Some(0), 0), (Some(10), 11)] {
            let indices: Vec<u8> = (0..16u8).map(|i| i % 4).collect();
            let bytes = fixture((4, 4), &palette, 0, stored, &[Frame::full(4, 4, indices)]);
            let raster = decode_bytes(&bytes).expect("decodes");
            assert_eq!(
                raster.get_int("loop"),
                Some(expected),
                "NETSCAPE count {stored:?} should report loop {expected}"
            );
        }
    }

    /**
     * Tests the fields `gifload` attaches beside the pixels, so a caller
     * sees the same header libvips would show. `bits-per-sample` is
     * `ceil(log2(colours))` over the colour table as padded on the wire
     * (`nsgifload.c:337-343`) -- 4 for `cogs.gif`'s 16 entries, 3 for
     * `truncated.gif`'s 8, 1 for `dispose-previous.gif`'s 2 -- `palette` is
     * always 1, and `interlaced` appears only when some frame is stored
     * interlaced.
     * Input: fixtures with 2-, 4- and 16-entry tables, one interlaced ->
     * Output: `bits-per-sample` 1, 2 and 4, `palette` 1 throughout, and
     * `interlaced` set only on the interlaced file.
     */
    #[test]
    fn the_attached_fields_match_what_gifload_sets() {
        for (entries, bits) in [(2usize, 1), (4, 2), (16, 4)] {
            let palette: Vec<[u8; 3]> = (0..entries).map(|i| [i as u8, 0, 0]).collect();
            let indices: Vec<u8> = (0..16u8).map(|i| i % entries as u8).collect();
            let bytes = fixture((4, 4), &palette, 0, None, &[Frame::full(4, 4, indices)]);
            let raster = decode_bytes(&bytes).expect("decodes");
            assert_eq!(
                raster.get_int("bits-per-sample"),
                Some(bits),
                "{entries} palette entries should report {bits} bits per sample"
            );
            assert_eq!(raster.get_int("palette"), Some(1));
            assert_eq!(raster.get_int("interlaced"), None);
            assert_eq!(
                raster.interpretation(),
                crate::conversion::Interpretation::Srgb
            );
        }

        let palette: Vec<[u8; 3]> = (0..8u8).map(|i| [i * 30, 255 - i * 30, i * 17]).collect();
        let mut frame = Frame::full(8, 8, (0..64u8).map(|i| i / 8).collect());
        frame.interlaced = true;
        let bytes = fixture((8, 8), &palette, 0, None, &[frame]);
        let raster = decode_bytes(&bytes).expect("decodes");
        assert_eq!(raster.get_int("interlaced"), Some(1));
    }

    /**
     * Tests that a frame whose pixel data runs off the end of the file
     * still loads, and that it loads as four bands. Rows that never arrived
     * stay uncomposited, so libnsgif reports the frame as transparent even
     * when its graphic control extension clears the flag. Measured: chopping
     * six bytes off an 8x8 opaque GIF flips `vipsheader` from `3 bands` to
     * `4 bands` and prints "Unexpected end of GIF source data" as a warning
     * rather than failing, because `fail-on` defaults to none. The reference
     * `truncated.gif` is the same case, and vips loads 575x800 four-band
     * from it.
     * Input: an 8x8 fixture with its last six bytes removed -> Output: a
     * `Rgba8` raster whose surviving rows still carry their palette colours.
     */
    #[test]
    fn a_truncated_frame_still_loads_and_counts_as_transparency() {
        let palette: Vec<[u8; 3]> = (0..8u8).map(|i| [i * 30, 255 - i * 30, i * 17]).collect();
        let indices: Vec<u8> = (0..64u8).map(|i| i / 8).collect();
        let whole = fixture((8, 8), &palette, 0, None, &[Frame::full(8, 8, indices)]);
        assert_eq!(
            decode_bytes(&whole).expect("decodes").format(),
            PixelFormat::Rgb8,
            "the intact file has no transparency anywhere"
        );

        let clipped = &whole[..whole.len() - 6];
        let raster = decode_bytes(clipped).expect("a truncated GIF still loads, as vips loads it");
        assert_eq!(raster.width(), 8);
        assert_eq!(raster.height(), 8);
        assert_eq!(
            raster.format(),
            PixelFormat::Rgba8,
            "the rows that never arrived are transparent, so the file is RGBA"
        );
        assert_eq!(
            &raster.data()[..4],
            &[0, 255, 0, 255],
            "the rows that did arrive keep their palette colours"
        );
    }

    /**
     * Tests that the three ways to write a GIF agree: the byte-producing
     * `encode_gif`, the file-writing `save_gif`, and the extension dispatch
     * `Raster::save` shares with the connection encoders. A format wired
     * into one and not the others is the failure mode this guards.
     * Input: an 8x8 gradient -> Output: identical bytes from `encode_gif`,
     * `save_gif`, `save(".gif")` and `encode_to_buffer("gif")`.
     */
    #[test]
    fn every_save_entry_point_writes_the_same_bytes() {
        let im = gradient(8, 8);
        let expected = im.encode_gif(SaveOptions::default()).expect("encodes");
        let dir = tempfile::tempdir().unwrap();

        let direct = dir.path().join("direct.gif");
        im.save_gif(&direct, SaveOptions::default()).expect("saves");
        assert_eq!(std::fs::read(&direct).unwrap(), expected);

        let dispatched = dir.path().join("dispatched.GIF");
        im.save(&dispatched)
            .expect("the extension dispatch is wired");
        assert_eq!(std::fs::read(&dispatched).unwrap(), expected);

        assert_eq!(im.encode_to_buffer("gif").expect("wired"), expected);
    }

    /**
     * Tests that the GIF encoder is reachable at every option combination
     * and reports its errors through the shared spine rather than panicking.
     * Works by encoding a small RGB raster at the default options,
     * interlaced, and with dithering off.
     * Input: 8x8 Rgb8 raster -> Output: a non-empty GIF89a buffer from every
     * call.
     */
    #[test]
    fn encode_gif_answers_every_option_combination() {
        let im = Raster::new(8, 8, PixelFormat::Rgb8, vec![7u8; 8 * 8 * 3]).unwrap();
        for options in [
            SaveOptions::default(),
            SaveOptions {
                interlaced: true,
                ..Default::default()
            },
            SaveOptions {
                dither: 0.0,
                ..Default::default()
            },
        ] {
            let bytes = im
                .encode_gif(options)
                .expect("every option combination encodes");
            assert_eq!(&bytes[..6], b"GIF89a", "{options:?} did not produce a GIF");
        }
    }

    /**
     * Pins the option defaults against the vips 8.18.4 `gifsave` operation
     * description, so the GIF lane inherits the right starting point rather
     * than rediscovering it. Works by reading the defaults straight off
     * `SaveOptions::default()`.
     * Input: none -> Output: `interlaced == false` and `dither == 1.0`,
     * matching `vips gifsave`'s reported defaults.
     */
    #[test]
    fn save_options_defaults_match_vips() {
        let d = SaveOptions::default();
        assert!(!d.interlaced, "vips gifsave interlace defaults to false");
        assert!(
            (d.dither - 1.0).abs() < f64::EPSILON,
            "vips gifsave dither defaults to 1, got {}",
            d.dither
        );
    }
}
