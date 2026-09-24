//! Baseline JPEG encoding, with the two knobs `image` 0.25 does not expose.
//!
//! # Why this is here and not another call into `image`
//!
//! `image` 0.25's `JpegEncoder` fixes all three components at 1x1 sampling in
//! `new_with_quality` and borrows the Annex K Huffman tables as constants.
//! Neither is reachable: there is no setter for either, and both fields are
//! private. So every JPEG this crate wrote was 4:4:4 with textbook tables, and
//! [`JpegSubsample`] could be threaded the whole way down to the call and
//! still change nothing. Issue #1132 found that half-built: the mode existed
//! in [`crate::codec`], the signature of `Raster::encode_jpeg_options` took
//! it, and the body dropped it on the floor with a comment saying so.
//!
//! The crate has been in this position before and answered it the same way.
//! `Raster::encode_png_interlaced` (Adam7) and `Raster::encode_png_palette`
//! are hand-rolled on `flate2` because `image`'s PNG encoder exposes neither
//! knob, and CONTRIBUTING.md's dependency rule is why that is cheaper than it
//! looks: a second JPEG crate is a permanent line in the dependency table and
//! a permanent obligation in `tests/dependency_policy.rs`, and this is about
//! four hundred lines of arithmetic that nothing else in the tree can break.
//!
//! # What it costs, measured on a 256x256 tile
//!
//! 4:2:0 halves the block count, and tables built from the tile stop charging
//! for symbols it never emits. Together they take a blank tile from 2419 bytes
//! to well under half of that, which is the floor issue #1132 was filed about,
//! and they leave the decoded pixels where they were: subsampling throws away
//! chroma resolution, and on the black-on-white drawings this crate tiles
//! there is almost none to throw.
//!
//! # What it emits
//!
//! Baseline sequential JPEG: SOI, a JFIF APP0, the quantization tables, an
//! SOF0 frame header, the Huffman tables, one interleaved scan, EOI. Eight-bit
//! samples, one or three components, no restart markers.
//!
//! It is deliberately not a general JPEG writer. No progressive mode, no
//! arithmetic coding, no restart intervals, no 12-bit samples, no CMYK, and no
//! metadata beyond the JFIF header, because nothing in this crate asks for any
//! of them and every one of them is a way for this file to grow a second
//! reason to exist. `Raster::jpegsave_buffer_restart` still answers
//! [`EncodeError::Unsupported`] for exactly that reason.
//!
//! [`crate::uhdr`] keeps its own call into `image`'s encoder rather than
//! coming here. Its base image and gain map are two JPEGs inside an ISO
//! container with hand-computed MPF offsets, checked against a libuhdr
//! capture, so moving them is a measurement against that oracle rather than a
//! call-site change, and it is not what either issue here asked for.
//!
//! Decoding is untouched and stays with `image` (`zune-jpeg`), which is also
//! what makes the round-trip cells in this file worth something: they are read
//! back by a decoder that shares no code with the encoder under test.
//!
//! Worth knowing what that decoder will not tell you, though. `zune-jpeg`
//! fills past the end of a truncated entropy segment and hands back an image;
//! libjpeg says `Corrupt JPEG data: premature end of data segment` about the
//! same file. A defect that cost this encoder its last symbol per scan was
//! green through every round trip here and was found by running the output
//! through `djpeg` while comparing sizes against the vips oracle. So the round
//! trips are worth something, and they are not the whole check: output from a
//! change to this file wants a pass through something stricter, and `vips` and
//! `djpeg` are both on the machine this crate is developed on.
//!
//! # Two passes, and what they cost
//!
//! Building the Huffman tables from the image means counting the symbols
//! before writing any of them, and this counts them by running the transform
//! twice rather than by keeping the coefficients. Keeping them costs
//! `size_of::<i32>()` per coefficient in the shape this encoder produces them,
//! which is **6 bytes a pixel at 4:2:0 and 12 at 4:4:4**: 600 MB and 1.2 GB
//! respectively on a 10000x10000 image, against 300 MB for the raster itself.
//! Three bytes a pixel is libjpeg's figure and not this one, because libjpeg
//! keeps a `JCOEF` as an `i16`; quoting its best case as the general one would
//! be understating the thing the decision turns on. Running the transform
//! again costs time that
//! scales with the same image and no memory at all. For a crate whose whole
//! shape is "do not hold the image twice", that is the right way round.
//!
//! One qualification, because the argument above is about this file and the
//! caller in front of it does not keep to it. `crate::sink::flatten_alpha`
//! builds a whole `Rgb8` raster while the `Rgba8` original is still live, so
//! an RGBA input peaks at seven bytes a pixel before a single coefficient
//! exists. On the tile path that is one 256x256 tile at a time and it never
//! bites; on `Raster::encode_jpeg` over a whole image it does, and on the
//! render path that produces `Rgba8` in the first place it is 488 MB against
//! 279 MB for a 9932x7020 page. It buys the crate's own port of
//! `vips_flatten` instead of a second copy of the compositing arithmetic,
//! which is why it is there. Compositing inside `Pixels::sample` would cost
//! one multiply-add per channel and no allocation, and that is the change to
//! make if the whole-image route ever matters; the two-pass trade below is
//! about this encoder's own footprint and not about that copy.
//!
//! The price is measured, not guessed. On a 256x256 line-art tile at quality
//! 85, release build, this machine: 0.93 ms against `image`'s 0.50 ms at
//! 4:2:0, and 1.29 ms at 4:4:4 where there are twice the blocks to transform.
//! Almost exactly half of that is the second pass, so a caller who wanted the
//! speed back would buffer the coefficients rather than look for a faster
//! transform. The tile is also 1.24x to 3.62x smaller for it, which is the
//! trade issue #1132 asked for.

use crate::codec::{EncodeError, JpegSubsample};

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// Zigzag order: `ZIGZAG[k]` is the natural (row-major) index of the
/// coefficient that comes `k`th in the scan.
///
/// Coefficients live in natural order everywhere in this file. The zigzag is
/// applied exactly twice, when writing a quantization table and when walking a
/// block's AC coefficients, because those are the two places the format asks
/// for it.
#[rustfmt::skip]
const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Annex K table K.1, the luminance quantization table.
#[rustfmt::skip]
const STD_LUMA_QTABLE: [u16; 64] = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
];

/// Annex K table K.2, the chrominance quantization table.
#[rustfmt::skip]
const STD_CHROMA_QTABLE: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// The prescale the AAN transform leaves in its output: `cos(k * PI / 16) *
/// sqrt(2)` for `k > 0`, and 1 for `k = 0`.
///
/// [`fdct`] is fast because it does not normalise; what comes out is the real
/// DCT multiplied by `8 * AAN[u] * AAN[v]`. Folding the inverse of that into
/// the quantization divisors makes it free, which is the whole trick and the
/// reason the divisors below are floats rather than the table itself.
const AAN: [f32; 8] = [
    1.0,
    1.387_039_8,
    1.306_563,
    1.175_875_6,
    1.0,
    0.785_695,
    0.541_196_1,
    0.275_899_4,
];

// Markers.
const SOI: u8 = 0xD8;
const EOI: u8 = 0xD9;
const SOF0: u8 = 0xC0;
const DHT: u8 = 0xC4;
const DQT: u8 = 0xDB;
const SOS: u8 = 0xDA;
const APP0: u8 = 0xE0;

/// The largest frame the two-byte SOF0 dimensions can describe.
const MAX_AXIS: u32 = 65535;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Whether a quality and a mode ask for 4:2:0.
///
/// [`JpegSubsample::Auto`] is libvips' `VIPS_FOREIGN_SUBSAMPLE_AUTO`: 4:2:0
/// below quality 90 and 4:4:4 at or above. The tile default is quality 85, so
/// `Auto` is what selects 4:2:0 for a tile without anybody having to pass a
/// mode down through `TileFormat`.
fn subsampled(quality: u8, subsample: JpegSubsample) -> bool {
    match subsample {
        JpegSubsample::Off => false,
        JpegSubsample::On => true,
        JpegSubsample::Auto => quality < 90,
    }
}

/// Encode 8-bit greyscale or colour samples as a baseline JPEG.
///
/// `color` comes from the caller's own `color_type_for_format`, so the
/// refusals for the compute-intermediate pixel formats keep their existing
/// wording and this function only answers for the colour types that reach it.
///
/// An alpha channel is the caller's to deal with before it gets here, through
/// `crate::sink::flatten_alpha`, which is `vips_flatten` and is what every
/// vips saver does for a format that cannot carry one (issue #1133).
///
/// # Errors
///
/// [`EncodeError::Encode`] for a colour type JPEG has no sample layout for
/// (anything 16-bit, and RGBA, which should have been flattened), for a zero
/// or over-large axis, or for a buffer whose length does not match the
/// dimensions.
pub(crate) fn encode(
    data: &[u8],
    width: u32,
    height: u32,
    color: image::ColorType,
    quality: u8,
    subsample: JpegSubsample,
) -> Result<Vec<u8>, EncodeError> {
    let channels = match color {
        image::ColorType::L8 => 1usize,
        image::ColorType::Rgb8 => 3,
        other => {
            return Err(EncodeError::encode(format!(
                "JPEG has no sample layout for {other:?}: the format is 8-bit \
                 and carries no alpha, so a 16-bit raster has to be cast and \
                 an RGBA one flattened first"
            )));
        }
    };
    if width == 0 || height == 0 {
        return Err(EncodeError::encode(format!(
            "a {width}x{height} raster has no pixels to encode as JPEG"
        )));
    }
    if width > MAX_AXIS || height > MAX_AXIS {
        return Err(EncodeError::encode(format!(
            "a {width}x{height} raster is past JPEG's {MAX_AXIS}-pixel axis \
             limit, which the frame header cannot describe"
        )));
    }
    let (w, h) = (width as usize, height as usize);
    let want = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| {
            EncodeError::encode(format!(
                "a {width}x{height} {color:?} raster does not fit this target's \
                 address space"
            ))
        })?;
    if data.len() != want {
        return Err(EncodeError::encode(format!(
            "a {width}x{height} {color:?} raster is {want} bytes and this one \
             is {}",
            data.len()
        )));
    }

    let plan = Plan::new(w, h, channels, quality, subsample);
    let px = Pixels {
        data,
        width: w,
        height: h,
        channels,
    };

    // Pass one: what does this image actually emit?
    let mut counts = [[0u32; 256]; 4];
    let mut prev_dc = [0i32; 3];
    plan.for_each_block(&px, |comp, block| {
        let base = plan.comps[comp].huff * 2;
        prev_dc[comp] = block_symbols(block, prev_dc[comp], |is_ac, symbol, _, _| {
            counts[base + usize::from(is_ac)][symbol as usize] += 1;
        });
    });

    // Pass two: the same walk, now that there are tables to write it with.
    let tables: Vec<HuffTable> = (0..plan.huff_tables())
        .map(|t| HuffTable::optimal(&counts[t]))
        .collect::<Result<_, _>>()?;

    let mut out = Vec::with_capacity(data.len() / 4 + 1024);
    plan.write_headers(&mut out, &tables);

    let mut bits = BitWriter::new(out);
    let mut prev_dc = [0i32; 3];
    plan.for_each_block(&px, |comp, block| {
        let base = plan.comps[comp].huff * 2;
        prev_dc[comp] = block_symbols(block, prev_dc[comp], |is_ac, symbol, extra, len| {
            let (size, code) = tables[base + usize::from(is_ac)].codes[symbol as usize];
            // Every symbol pass two writes was counted in pass one, so every
            // one of them has a code. A zero length here would mean the two
            // walks disagreed, and `BitWriter::write` would answer by writing
            // nothing at all, which is a stream that decodes into noise from
            // that point rather than an error. The check is here and not in
            // the writer because the second call below legitimately passes a
            // width of zero, for a ZRL or an end-of-block.
            debug_assert!(
                size > 0,
                "symbol {symbol:#04x} (ac={is_ac}) reached the scan with no code"
            );
            bits.write(code, size);
            bits.write(extra, len);
        });
    });
    let mut out = bits.finish();
    out.extend_from_slice(&[0xFF, EOI]);
    Ok(out)
}

// ---------------------------------------------------------------------------
// The frame
// ---------------------------------------------------------------------------

/// One frame component: where its samples come from and which tables it uses.
#[derive(Debug, Clone, Copy)]
struct Component {
    /// The component id the frame and scan headers carry. 1, 2, 3 is what
    /// JFIF expects for Y, Cb, Cr.
    id: u8,
    /// Horizontal sampling factor.
    h: usize,
    /// Vertical sampling factor.
    v: usize,
    /// Quantization table index: 0 luma, 1 chroma.
    quant: usize,
    /// Huffman table pair index: 0 luma, 1 chroma.
    huff: usize,
    /// Which of Y, Cb, Cr this component reads.
    channel: usize,
}

/// Everything decided before the first block is touched.
struct Plan {
    width: usize,
    height: usize,
    comps: Vec<Component>,
    hmax: usize,
    vmax: usize,
    mcus_x: usize,
    mcus_y: usize,
    /// Quantization tables in natural order, as they go into DQT.
    quant: Vec<[u16; 64]>,
    /// The same tables folded together with the AAN prescale, as the
    /// multipliers [`quantize`] applies.
    divisors: Vec<[f32; 64]>,
}

impl Plan {
    fn new(
        width: usize,
        height: usize,
        channels: usize,
        quality: u8,
        subsample: JpegSubsample,
    ) -> Self {
        let grey = channels == 1;
        let (h, v) = if !grey && subsampled(quality, subsample) {
            (2, 2)
        } else {
            (1, 1)
        };
        let comps = if grey {
            vec![Component {
                id: 1,
                h: 1,
                v: 1,
                quant: 0,
                huff: 0,
                channel: 0,
            }]
        } else {
            vec![
                Component {
                    id: 1,
                    h,
                    v,
                    quant: 0,
                    huff: 0,
                    channel: 0,
                },
                Component {
                    id: 2,
                    h: 1,
                    v: 1,
                    quant: 1,
                    huff: 1,
                    channel: 1,
                },
                Component {
                    id: 3,
                    h: 1,
                    v: 1,
                    quant: 1,
                    huff: 1,
                    channel: 2,
                },
            ]
        };
        let quant: Vec<[u16; 64]> = if grey {
            vec![scaled_qtable(&STD_LUMA_QTABLE, quality)]
        } else {
            vec![
                scaled_qtable(&STD_LUMA_QTABLE, quality),
                scaled_qtable(&STD_CHROMA_QTABLE, quality),
            ]
        };
        let divisors = quant.iter().map(divisors_for).collect();
        let (mcu_w, mcu_h) = (8 * h, 8 * v);
        Self {
            width,
            height,
            comps,
            hmax: h,
            vmax: v,
            mcus_x: width.div_ceil(mcu_w),
            mcus_y: height.div_ceil(mcu_h),
            quant,
            divisors,
        }
    }

    /// How many Huffman tables the scan needs: a DC and an AC per table pair.
    fn huff_tables(&self) -> usize {
        if self.comps.len() == 1 { 2 } else { 4 }
    }

    /// Walk every block of the scan in the order the interleaved scan writes
    /// them, handing each one to `f` already transformed and quantized.
    ///
    /// Both passes go through here, which is what makes the counts in the
    /// first describe the stream the second writes. Splitting them would let
    /// the two drift, and a table built for a stream that is not the one being
    /// written is not a corrupt file, it is a slightly larger one, which is
    /// the kind of bug that never gets found.
    fn for_each_block(&self, px: &Pixels<'_>, mut f: impl FnMut(usize, &[i32; 64])) {
        let mut samples = [0f32; 64];
        let mut coeffs = [0i32; 64];
        for my in 0..self.mcus_y {
            for mx in 0..self.mcus_x {
                for (ci, comp) in self.comps.iter().enumerate() {
                    for by in 0..comp.v {
                        for bx in 0..comp.h {
                            px.fill_block(
                                comp,
                                self.hmax / comp.h,
                                self.vmax / comp.v,
                                mx * comp.h + bx,
                                my * comp.v + by,
                                &mut samples,
                            );
                            fdct(&mut samples);
                            quantize(&samples, &self.divisors[comp.quant], &mut coeffs);
                            f(ci, &coeffs);
                        }
                    }
                }
            }
        }
    }

    /// Everything from SOI up to the first entropy-coded byte.
    fn write_headers(&self, out: &mut Vec<u8>, tables: &[HuffTable]) {
        out.extend_from_slice(&[0xFF, SOI]);

        // A JFIF header with a 1:1 pixel aspect ratio and no thumbnail. No
        // density: this crate's resolution metadata does not travel through
        // the tile encoders.
        let mut jfif = Vec::from(*b"JFIF\0");
        jfif.extend_from_slice(&[0x01, 0x02, 0x00, 0, 1, 0, 1, 0, 0]);
        segment(out, APP0, &jfif);

        for (i, table) in self.quant.iter().enumerate() {
            let mut dqt = Vec::with_capacity(65);
            // 8-bit precision, table i. Every entry fits a byte because
            // `scaled_qtable` clamps to 255 for exactly this reason.
            dqt.push(u8::try_from(i).expect("at most two quantization tables"));
            for &k in &ZIGZAG {
                dqt.push(u8::try_from(table[k]).expect("an 8-bit quantization entry"));
            }
            segment(out, DQT, &dqt);
        }

        let mut sof = vec![8];
        sof.extend_from_slice(
            &u16::try_from(self.height)
                .expect("the axis limit was checked")
                .to_be_bytes(),
        );
        sof.extend_from_slice(
            &u16::try_from(self.width)
                .expect("the axis limit was checked")
                .to_be_bytes(),
        );
        sof.push(u8::try_from(self.comps.len()).expect("one or three components"));
        for c in &self.comps {
            let hv = u8::try_from(c.h << 4 | c.v).expect("sampling factors are 1 or 2");
            sof.extend_from_slice(&[c.id, hv, u8::try_from(c.quant).expect("table 0 or 1")]);
        }
        segment(out, SOF0, &sof);

        for (i, table) in tables.iter().enumerate() {
            let mut dht = Vec::with_capacity(17 + table.values.len());
            // class in the high nibble (0 DC, 1 AC), destination in the low.
            let class = u8::try_from(i % 2).expect("0 or 1");
            let dest = u8::try_from(i / 2).expect("0 or 1");
            dht.push(class << 4 | dest);
            dht.extend_from_slice(&table.bits[1..]);
            dht.extend_from_slice(&table.values);
            segment(out, DHT, &dht);
        }

        let mut sos = vec![u8::try_from(self.comps.len()).expect("one or three components")];
        for c in &self.comps {
            // The DC table in the high nibble and the AC table in the low.
            // A component uses the same destination for both, which is what
            // the `huff * 2 + class` indexing above means on the wire.
            let pair = u8::try_from(c.huff << 4 | c.huff).expect("table 0 or 1");
            sos.extend_from_slice(&[c.id, pair]);
        }
        // Spectral selection 0 to 63 and no successive approximation, which is
        // the only shape a baseline scan has.
        sos.extend_from_slice(&[0, 63, 0]);
        segment(out, SOS, &sos);
    }
}

/// Write one length-prefixed marker segment.
fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    let len = u16::try_from(payload.len() + 2).expect("a marker segment is under 64 KiB");
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
}

// ---------------------------------------------------------------------------
// Pixels
// ---------------------------------------------------------------------------

/// The source raster and the one place colour is converted.
struct Pixels<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
    channels: usize,
}

impl Pixels<'_> {
    /// One component sample at a pixel, with coordinates past the edge clamped
    /// back onto it.
    ///
    /// The clamp is how a block that runs past the image is filled. Replicating
    /// the edge is what libjpeg does and it is the cheap choice for the right
    /// reason: the padding is inside a block the decoder will reconstruct and
    /// then crop, so anything with an edge in it (zeros, the background) costs
    /// high-frequency coefficients to code and shows up as ringing along the
    /// last real column.
    fn sample(&self, x: usize, y: usize, channel: usize) -> f32 {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let off = (y * self.width + x) * self.channels;
        if self.channels == 1 {
            return f32::from(self.data[off]);
        }
        let (r, g, b) = (
            f32::from(self.data[off]),
            f32::from(self.data[off + 1]),
            f32::from(self.data[off + 2]),
        );
        // JPEG's RGB to YCbCr, the BT.601 full-range matrix JFIF specifies.
        match channel {
            0 => 0.299 * r + 0.587 * g + 0.114 * b,
            1 => -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0,
            _ => 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0,
        }
    }

    /// Fill one 8x8 block of a component's sample grid, level-shifted.
    ///
    /// `sx` by `sy` source pixels are box-averaged into each sample, which is
    /// `1x1` for luma and `2x2` for chroma under 4:2:0. libjpeg calls the
    /// alternative "fancy" downsampling and it is a triangular filter; the box
    /// is what it does with `do_fancy_downsampling` off, and on the content
    /// this crate tiles the difference is invisible next to the quantizer.
    fn fill_block(
        &self,
        comp: &Component,
        sx: usize,
        sy: usize,
        bx: usize,
        by: usize,
        out: &mut [f32; 64],
    ) {
        let n = (sx * sy) as f32;
        for j in 0..8 {
            for i in 0..8 {
                let gx = (bx * 8 + i) * sx;
                let gy = (by * 8 + j) * sy;
                let mut acc = 0.0;
                for dy in 0..sy {
                    for dx in 0..sx {
                        acc += self.sample(gx + dx, gy + dy, comp.channel);
                    }
                }
                // The level shift: JPEG codes samples centred on zero.
                out[j * 8 + i] = acc / n - 128.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transform and quantization
// ---------------------------------------------------------------------------

/// The scaled quantization table for a quality, on libjpeg's curve.
///
/// This is the curve `image`, libjpeg and libvips all use, so quality 85 keeps
/// meaning what it meant before this file existed. The clamp to 255 is the
/// format's, not a choice: an 8-bit DQT cannot spell a larger divisor, and the
/// clamp to 1 keeps a quality-100 table from dividing by zero.
fn scaled_qtable(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let q = u32::from(quality.clamp(1, 100));
    let scale = if q < 50 { 5000 / q } else { 200 - q * 2 };
    let mut out = [0u16; 64];
    for (slot, &b) in out.iter_mut().zip(base.iter()) {
        let v = (u32::from(b) * scale + 50) / 100;
        *slot = u16::try_from(v.clamp(1, 255)).expect("clamped to 255");
    }
    out
}

/// Fold a quantization table together with the AAN prescale, so quantizing is
/// one multiply per coefficient.
fn divisors_for(table: &[u16; 64]) -> [f32; 64] {
    let mut out = [0f32; 64];
    for (i, slot) in out.iter_mut().enumerate() {
        // Natural order, so the row is the vertical frequency and the column
        // the horizontal one.
        let (u, v) = (i % 8, i / 8);
        *slot = 1.0 / (f32::from(table[i]) * AAN[u] * AAN[v] * 8.0);
    }
    out
}

/// The AAN float forward DCT, in place, in natural order.
///
/// This is libjpeg's `jpeg_fdct_float`: five multiplies per eight-point pass
/// instead of the sixty-four a direct evaluation needs. What it returns is not
/// the DCT, it is the DCT times `8 * AAN[u] * AAN[v]`, and [`divisors_for`]
/// takes that back out for free while it is dividing by the quantization table
/// anyway. `the_fast_transform_agrees_with_the_definition` is the cell that
/// holds this to the real thing, because an arithmetic slip here would show up
/// as slightly wrong pixels rather than as anything that fails.
fn fdct(block: &mut [f32; 64]) {
    for pass in 0..2 {
        // Rows first, then columns. The two passes differ only in how the
        // eight values are indexed, so the butterfly below is written once.
        for k in 0..8 {
            let (base, step) = if pass == 0 { (k * 8, 1) } else { (k, 8) };
            let at = |i: usize| base + i * step;

            let tmp0 = block[at(0)] + block[at(7)];
            let tmp7 = block[at(0)] - block[at(7)];
            let tmp1 = block[at(1)] + block[at(6)];
            let tmp6 = block[at(1)] - block[at(6)];
            let tmp2 = block[at(2)] + block[at(5)];
            let tmp5 = block[at(2)] - block[at(5)];
            let tmp3 = block[at(3)] + block[at(4)];
            let tmp4 = block[at(3)] - block[at(4)];

            // Even part.
            let tmp10 = tmp0 + tmp3;
            let tmp13 = tmp0 - tmp3;
            let tmp11 = tmp1 + tmp2;
            let tmp12 = tmp1 - tmp2;

            block[at(0)] = tmp10 + tmp11;
            block[at(4)] = tmp10 - tmp11;

            let z1 = (tmp12 + tmp13) * 0.707_106_77;
            block[at(2)] = tmp13 + z1;
            block[at(6)] = tmp13 - z1;

            // Odd part.
            let tmp10 = tmp4 + tmp5;
            let tmp11 = tmp5 + tmp6;
            let tmp12 = tmp6 + tmp7;

            let z5 = (tmp10 - tmp12) * 0.382_683_43;
            let z2 = 0.541_196_1 * tmp10 + z5;
            let z4 = 1.306_563 * tmp12 + z5;
            let z3 = tmp11 * 0.707_106_77;

            let z11 = tmp7 + z3;
            let z13 = tmp7 - z3;

            block[at(5)] = z13 + z2;
            block[at(3)] = z13 - z2;
            block[at(1)] = z11 + z4;
            block[at(7)] = z11 - z4;
        }
    }
}

/// Divide out the quantization table and round to integers.
fn quantize(block: &[f32; 64], divisors: &[f32; 64], out: &mut [i32; 64]) {
    for ((slot, &c), &d) in out.iter_mut().zip(block.iter()).zip(divisors.iter()) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the AAN output is the DCT times 8*AAN[u]*AAN[v] and reaches \
                      about +/-15800 on an 8-bit block, which the divisor brings \
                      back under +/-1024; either way the product is far inside i32"
        )]
        let rounded = (c * d).round() as i32;
        *slot = rounded;
    }
}

// ---------------------------------------------------------------------------
// Entropy coding
// ---------------------------------------------------------------------------

/// The magnitude category of a coefficient and the bits that follow its
/// symbol.
///
/// The bits are the value itself for a positive coefficient and its one's
/// complement for a negative one, which is how JPEG spells a signed value in
/// `category` bits without spending one of them on the sign.
fn magnitude(v: i32) -> (u8, u16) {
    let mut m = v.unsigned_abs();
    let mut size = 0u8;
    while m > 0 {
        m >>= 1;
        size += 1;
    }
    if size == 0 {
        return (0, 0);
    }
    // `size` is bounded here, before it is used as a shift distance, rather
    // than downstream where the `u16::try_from` below reads as if it were the
    // guard. It is not: the shift happens first, and a `size` past 31 would be
    // undefined behaviour in C and a panic here before anything checked it.
    //
    // It cannot happen, and that is measured rather than assumed. The 2D
    // transform is orthogonal as `divisors_for` normalises it, so no
    // coefficient of an 8-bit block exceeds 1024 in magnitude and the largest
    // DC difference is about 2040. At quality 100, where every divisor clamps
    // to 1, that is a category of 11. The assertion is here because the bound
    // is worth stating at the point it is relied on.
    debug_assert!(
        size <= 16,
        "a magnitude category of {size} would shift past the accumulator"
    );
    let mask = (1u32 << size) - 1;
    #[expect(
        clippy::cast_sign_loss,
        reason = "the mask keeps only the low `size` bits, which is the value's \
                  own magnitude for a positive coefficient and its complement \
                  for a negative one"
    )]
    let bits = if v < 0 {
        (v - 1) as u32 & mask
    } else {
        (v as u32) & mask
    };
    (size, u16::try_from(bits).expect("size is at most 16 bits"))
}

/// The entropy symbols one block contributes, in order, to `emit(is_ac,
/// symbol, extra bits, extra bit count)`. Returns the DC value to predict the
/// next block from.
fn block_symbols(block: &[i32; 64], prev_dc: i32, mut emit: impl FnMut(bool, u8, u16, u8)) -> i32 {
    let (size, bits) = magnitude(block[0] - prev_dc);
    emit(false, size, bits, size);

    let mut run = 0u8;
    for &k in &ZIGZAG[1..] {
        let v = block[k];
        if v == 0 {
            run += 1;
            continue;
        }
        // A run longer than fifteen zeros is spelled with as many ZRLs as it
        // takes, each standing for sixteen.
        while run > 15 {
            emit(true, 0xF0, 0, 0);
            run -= 16;
        }
        let (size, bits) = magnitude(v);
        emit(true, run << 4 | size, bits, size);
        run = 0;
    }
    // A trailing run of zeros is an end-of-block rather than sixty-three
    // symbols, which is where most of a blank tile's saving comes from.
    if run > 0 {
        emit(true, 0x00, 0, 0);
    }
    block[0]
}

/// A Huffman table, in the two shapes it is needed in.
struct HuffTable {
    /// `bits[l]` is how many codes have length `l`, for `l` in 1 to 16.
    /// `bits[0]` is unused and exists so the indices read as lengths.
    bits: [u8; 17],
    /// The symbols, ordered by code length.
    values: Vec<u8>,
    /// `(length, code)` per symbol, for writing the scan.
    codes: [(u8, u16); 256],
}

impl HuffTable {
    /// Build the table that codes this symbol distribution in the fewest bits.
    ///
    /// This is libjpeg's `jpeg_gen_optimal_table`, and it is a textbook Huffman
    /// build plus the two things JPEG adds. First, a sentinel symbol with
    /// frequency 1 that is never written: it takes the all-ones code, which a
    /// decoder is entitled to treat as the start of a marker, so no real
    /// symbol can be assigned it. Second, a fold that pulls any code longer
    /// than sixteen bits back under the limit, because the format has no way
    /// to spell one. The fold costs a fraction of a bit on the rarest symbols
    /// and cannot happen at all on an image this crate would tile.
    fn optimal(freq: &[u32; 256]) -> Result<Self, EncodeError> {
        let mut f = [0u32; 257];
        f[..256].copy_from_slice(freq);
        // A table with no symbols at all cannot be written, and a component
        // always emits at least one DC symbol, so this only guards the
        // impossible. It guards it by making the table valid rather than by
        // panicking, because an unwritable table is a corrupt file and a
        // one-symbol table is merely pointless.
        if freq.iter().all(|&c| c == 0) {
            f[0] = 1;
        }
        f[256] = 1;

        let mut codesize = [0u32; 257];
        let mut others = [usize::MAX; 257];
        loop {
            let (mut v1, mut c1) = (usize::MAX, u32::MAX);
            for (i, &c) in f.iter().enumerate() {
                if c > 0 && c <= c1 {
                    c1 = c;
                    v1 = i;
                }
            }
            let (mut v2, mut c2) = (usize::MAX, u32::MAX);
            for (i, &c) in f.iter().enumerate() {
                if c > 0 && c <= c2 && i != v1 {
                    c2 = c;
                    v2 = i;
                }
            }
            if v2 == usize::MAX {
                break;
            }

            f[v1] += f[v2];
            f[v2] = 0;
            codesize[v1] += 1;
            while others[v1] != usize::MAX {
                v1 = others[v1];
                codesize[v1] += 1;
            }
            others[v1] = v2;
            codesize[v2] += 1;
            while others[v2] != usize::MAX {
                v2 = others[v2];
                codesize[v2] += 1;
            }
        }

        // `counts` runs to 32 because that is as long as a code can get before
        // the fold below, which is libjpeg's `MAX_CLEN`.
        let mut counts = [0u32; 33];
        for &size in &codesize {
            if size > 0 {
                // libjpeg's `MAX_CLEN`, and it `ERREXIT`s here rather than
                // folding the length in. So does this: clamping would silently
                // build a table that codes some symbol at a length the
                // distribution did not ask for, and the file would still
                // decode, which is the shape of defect that does not get
                // found. It takes a Fibonacci-like 2.1 million symbols in one
                // table to reach 32 bits, so a tile cannot get here and a
                // whole-image encode would have to be adversarial.
                if size > 32 {
                    return Err(EncodeError::encode(format!(
                        "a Huffman code of {size} bits is past the 32 the \
                         format's length-limiting can fold back under 16"
                    )));
                }
                counts[size as usize] += 1;
            }
        }
        for len in (17..=32).rev() {
            while counts[len] > 0 {
                let mut j = len - 2;
                while counts[j] == 0 {
                    j -= 1;
                }
                counts[len] -= 2;
                counts[len - 1] += 1;
                counts[j + 1] += 2;
                counts[j] -= 1;
            }
        }
        // Take the sentinel's code back out of the longest length in use.
        let mut longest = 16;
        while counts[longest] == 0 {
            longest -= 1;
        }
        counts[longest] -= 1;

        // The symbols in order of their pre-fold length, which is the order
        // the fold preserves and the order the format wants.
        let mut values = Vec::new();
        for len in 1..=32u32 {
            for (sym, &size) in codesize[..256].iter().enumerate() {
                if size == len {
                    values.push(u8::try_from(sym).expect("a symbol is one byte"));
                }
            }
        }

        let mut bits = [0u8; 17];
        let mut codes = [(0u8, 0u16); 256];
        let mut code = 0u32;
        let mut next = 0usize;
        for len in 1..=16usize {
            bits[len] = u8::try_from(counts[len]).expect("at most 255 codes of one length");
            for _ in 0..counts[len] {
                let sym = values[next];
                codes[sym as usize] = (
                    u8::try_from(len).expect("a length is at most 16"),
                    u16::try_from(code).expect("a code fits its own length"),
                );
                code += 1;
                next += 1;
            }
            code <<= 1;
        }
        values.truncate(next);

        Ok(Self {
            bits,
            values,
            codes,
        })
    }
}

/// The entropy-coded segment's bit accumulator.
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u8,
}

impl BitWriter {
    fn new(out: Vec<u8>) -> Self {
        Self {
            out,
            acc: 0,
            nbits: 0,
        }
    }

    /// Append the low `size` bits of `bits`, most significant first.
    ///
    /// The mask is load bearing and was not there to begin with. Anything set
    /// above bit `size` lands in the accumulator *above* this field, which is
    /// where the bits already written live, so a caller passing a wider value
    /// than it declared silently rewrites the symbol before it. That is
    /// exactly what the pad in [`BitWriter::finish`] used to do: it passed
    /// `0x7F` with a width of one to six, and the leftover ones overwrote the
    /// last four bits of the scan. `the_pad_cannot_overwrite_the_last_symbol`
    /// is the cell that holds this.
    fn write(&mut self, bits: u16, size: u8) {
        if size == 0 {
            return;
        }
        let bits = u32::from(bits) & ((1u32 << size) - 1);
        self.nbits += size;
        self.acc |= bits << (32 - u32::from(self.nbits));
        while self.nbits >= 8 {
            // The shift leaves exactly the top byte, so the narrowing is the
            // whole point rather than a loss.
            let byte = (self.acc >> 24) as u8;
            self.out.push(byte);
            // Byte stuffing: an 0xFF in the entropy stream is followed by a
            // zero so it cannot be read as a marker.
            if byte == 0xFF {
                self.out.push(0x00);
            }
            self.nbits -= 8;
            self.acc <<= 8;
        }
    }

    /// Pad to a byte boundary with ones and hand the buffer back.
    ///
    /// Ones rather than zeros because a zero pad can complete a valid code and
    /// hand the decoder a sixty-fifth coefficient.
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let n = 8 - self.nbits;
            self.write(u16::MAX, n);
        }
        self.out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random block, so a failure is reproducible.
    fn noise(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        f32::from(u8::try_from(*seed >> 40 & 0xFF).expect("masked to a byte")) - 128.0
    }

    /// The DCT as its definition states it, which is what [`fdct`] has to
    /// agree with after its prescale is taken back out.
    fn reference_dct(block: &[f32; 64]) -> [f32; 64] {
        let mut out = [0f32; 64];
        for v in 0..8 {
            for u in 0..8 {
                let mut sum = 0f64;
                for y in 0..8 {
                    for x in 0..8 {
                        let cx = (f64::from(2 * x as i32 + 1)
                            * f64::from(u as i32)
                            * std::f64::consts::PI
                            / 16.0)
                            .cos();
                        let cy = (f64::from(2 * y as i32 + 1)
                            * f64::from(v as i32)
                            * std::f64::consts::PI
                            / 16.0)
                            .cos();
                        sum += f64::from(block[y * 8 + x]) * cx * cy;
                    }
                }
                let c = |k: usize| if k == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                out[v * 8 + u] = (0.25 * c(u) * c(v) * sum) as f32;
            }
        }
        out
    }

    /// The fast transform is the real one, times a prescale the divisors take
    /// back out.
    ///
    /// This is the cell that earns the butterfly. An arithmetic slip in
    /// [`fdct`] does not fail anything else here: it produces slightly wrong
    /// pixels in a file that decodes perfectly well, which is the kind of
    /// defect that ships.
    #[test]
    fn the_fast_transform_agrees_with_the_definition() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for _ in 0..8 {
            let mut block = [0f32; 64];
            for slot in &mut block {
                *slot = noise(&mut seed);
            }
            let want = reference_dct(&block);
            fdct(&mut block);
            for (i, &coeff) in block.iter().enumerate() {
                let (u, v) = (i % 8, i / 8);
                let unscaled = coeff / (8.0 * AAN[u] * AAN[v]);
                assert!(
                    (unscaled - want[i]).abs() < 0.05,
                    "coefficient ({u}, {v}) is {unscaled} and the definition \
                     says {}",
                    want[i]
                );
            }
        }
    }

    /// A flat block is all DC, which is the property the whole blank-tile
    /// saving rests on.
    #[test]
    fn a_flat_block_has_one_coefficient() {
        let mut block = [42.0f32; 64];
        fdct(&mut block);
        assert!((block[0] - 42.0 * 64.0).abs() < 0.01, "DC is {}", block[0]);
        for (i, &c) in block.iter().enumerate().skip(1) {
            assert!(c.abs() < 0.01, "coefficient {i} is {c} and should be zero");
        }
    }

    /// The generated tables are prefix codes a decoder can actually use.
    ///
    /// Three properties, and the third is the one that is easy to lose: the
    /// all-ones code of the longest length stays unassigned, because a decoder
    /// is allowed to read it as the start of a marker. That is what the
    /// frequency-1 sentinel buys, and dropping the sentinel would pass every
    /// other assertion in this file.
    #[test]
    fn an_optimal_table_is_a_usable_prefix_code() {
        let mut seed = 0xfeed_face_dead_beefu64;
        for case in 0..6 {
            let mut freq = [0u32; 256];
            for (i, slot) in freq.iter_mut().enumerate() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // Case 0 is one symbol, which is the degenerate table. The
                // rest are skewed distributions of increasing width.
                *slot = if case == 0 {
                    u32::from(i == 7)
                } else if i < case * 40 {
                    u32::try_from(seed % 1000).expect("under a thousand") + 1
                } else {
                    0
                };
            }
            let table = HuffTable::optimal(&freq).expect("a 256-symbol table is under MAX_CLEN");

            let total: u32 = table.bits[1..].iter().map(|&b| u32::from(b)).sum();
            assert_eq!(
                total as usize,
                table.values.len(),
                "case {case}: the length counts and the symbol list disagree"
            );

            let mut longest = 0;
            let mut largest_at_longest = 0u32;
            for &sym in &table.values {
                let (len, code) = table.codes[sym as usize];
                assert!((1..=16).contains(&len), "case {case}: a {len}-bit code");
                if usize::from(len) > longest {
                    longest = usize::from(len);
                    largest_at_longest = u32::from(code);
                } else if usize::from(len) == longest {
                    largest_at_longest = largest_at_longest.max(u32::from(code));
                }
            }
            // Kraft *equality*, which is the thing that says the code is
            // complete and leaves no dead prefixes. The assertion used to be
            // `kraft <= 65536` under a comment claiming equality, and an
            // under-full code passes that: a fold or a `values` ordering slip
            // that dropped symbols would leave prefixes nothing decodes to and
            // still read as fine. The exact figure is the whole space less the
            // one code the sentinel holds, and the sentinel's code is the
            // all-ones one at `longest`, so it is worth `1 << (16 - longest)`.
            let kraft: u64 = (1..=16).map(|l| u64::from(table.bits[l]) << (16 - l)).sum();
            let reserved = 1u64 << (16 - longest);
            assert_eq!(
                kraft,
                65536 - reserved,
                "case {case}: the code spans {kraft} of 65536 and a complete one \
                 less the sentinel's {longest}-bit code spans {}",
                65536 - reserved
            );

            let all_ones = (1u32 << longest) - 1;
            assert!(
                largest_at_longest < all_ones,
                "case {case}: the all-ones {longest}-bit code was assigned, so \
                 the sentinel is gone and a decoder may read a symbol as a \
                 marker"
            );

            // The frequent symbol is the cheap one, which is the entire point
            // of building the table from the image.
            if case > 0 {
                let (busiest, _) = freq
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &c)| c)
                    .expect("a non-empty table");
                let (rarest, _) = freq
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c > 0)
                    .min_by_key(|&(_, &c)| c)
                    .expect("a non-empty table");
                assert!(
                    table.codes[busiest].0 <= table.codes[rarest].0,
                    "case {case}: the most frequent symbol is not the shortest \
                     code"
                );
            }
        }
    }

    /// Ink on paper: three bands, a grid and a diagonal, which is what the
    /// tile encoders actually see.
    fn drawing(w: u32, h: u32) -> Vec<u8> {
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let off = ((y * w + x) * 3) as usize;
                let ink = x % 32 == 0 || y % 32 == 0 || (x + y) % 71 == 0;
                data[off] = if ink { 24 } else { 236 };
                data[off + 1] = if ink { 24 } else { 236 };
                data[off + 2] = if ink { 90 } else { 236 };
            }
        }
        data
    }

    /// Decode through `image`, which is a different implementation than the
    /// one under test, so a round trip says something.
    fn decode(bytes: &[u8]) -> image::RgbImage {
        image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg)
            .expect("the encoder emits a decodable JPEG")
            .to_rgb8()
    }

    /// Peak signal to noise over the three bands, in dB.
    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mse: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = f64::from(x) - f64::from(y);
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }

    #[test]
    fn a_colour_image_round_trips_in_both_subsample_modes() {
        let src = drawing(64, 48);
        for (mode, factors) in [(JpegSubsample::Off, (1, 1)), (JpegSubsample::On, (2, 2))] {
            let bytes =
                encode(&src, 64, 48, image::ColorType::Rgb8, 85, mode).expect("a drawing encodes");
            let decoded = decode(&bytes);
            assert_eq!(decoded.dimensions(), (64, 48), "{mode:?}");
            let quality = psnr(&src, decoded.as_raw());
            assert!(
                quality > 28.0,
                "{mode:?} round trips at {quality:.1} dB, which is not lossy \
                 coding any more"
            );
            // The sampling factors the header declares, read back off the
            // wire rather than off the plan that wrote it.
            let sof = bytes
                .windows(2)
                .position(|w| w == [0xFF, SOF0])
                .expect("an SOF0 header");
            // Past the marker, the length and the precision byte: height,
            // width, the component count, then the first component's id and
            // its packed sampling factors. The dimensions are checked too, so
            // a marker byte matched inside some other segment's payload fails
            // here rather than reading a sampling factor out of a quantization
            // table.
            assert_eq!(
                (
                    u16::from_be_bytes([bytes[sof + 7], bytes[sof + 8]]),
                    u16::from_be_bytes([bytes[sof + 5], bytes[sof + 6]]),
                ),
                (64, 48),
                "{mode:?}: that was not the frame header"
            );
            assert_eq!(
                (bytes[sof + 11] >> 4, bytes[sof + 11] & 0x0F),
                factors,
                "{mode:?} declared the wrong luma sampling factors"
            );
        }
    }

    #[test]
    fn greyscale_round_trips_as_one_component() {
        let mut src = vec![0u8; 40 * 24];
        for (i, v) in src.iter_mut().enumerate() {
            *v = u8::try_from(i % 251).expect("under 256");
        }
        let bytes = encode(&src, 40, 24, image::ColorType::L8, 90, JpegSubsample::Auto)
            .expect("a greyscale raster encodes");
        let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)
            .expect("the greyscale JPEG decodes")
            .to_luma8();
        assert_eq!(decoded.dimensions(), (40, 24));
        let quality = psnr(&src, decoded.as_raw());
        assert!(quality > 30.0, "greyscale round trips at {quality:.1} dB");
    }

    /// Sizes that are not multiples of the MCU, down to the degenerate one.
    ///
    /// Every one of these has a block that runs past the image, so this is the
    /// cell that exercises the edge replication in `Pixels::sample`. A 1x1
    /// image is entirely padding bar one pixel.
    #[test]
    fn awkward_sizes_encode_and_decode() {
        for (w, h) in [(1u32, 1u32), (8, 8), (17, 13), (9, 33), (255, 3)] {
            let src = drawing(w, h);
            let bytes = encode(&src, w, h, image::ColorType::Rgb8, 85, JpegSubsample::Auto)
                .unwrap_or_else(|e| panic!("{w}x{h} encodes: {e}"));
            let decoded = decode(&bytes);
            assert_eq!(decoded.dimensions(), (w, h), "{w}x{h} came back wrong");
        }
    }

    /// The floor issue #1132 is about, measured here rather than through the
    /// sink so the number is the encoder's own.
    #[test]
    fn a_blank_tile_costs_less_than_it_did_at_444() {
        let blank = vec![255u8; 256 * 256 * 3];
        let at_444 = encode(
            &blank,
            256,
            256,
            image::ColorType::Rgb8,
            85,
            JpegSubsample::Off,
        )
        .expect("a blank tile encodes");
        let at_420 = encode(
            &blank,
            256,
            256,
            image::ColorType::Rgb8,
            85,
            JpegSubsample::Auto,
        )
        .expect("a blank tile encodes");
        assert!(
            at_420.len() * 2 < 2419,
            "a blank 4:2:0 tile is {} bytes against the 2419 this shipped with",
            at_420.len()
        );
        assert!(
            at_420.len() < at_444.len(),
            "4:2:0 is {} bytes and 4:4:4 is {}, which is the wrong way round",
            at_420.len(),
            at_444.len()
        );
    }

    /// Quality still means what it meant: a lower number is a smaller file.
    #[test]
    fn quality_moves_the_size() {
        let src = drawing(96, 96);
        let sizes: Vec<usize> = [20u8, 50, 85, 95]
            .iter()
            .map(|&q| {
                encode(&src, 96, 96, image::ColorType::Rgb8, q, JpegSubsample::On)
                    .expect("encodes")
                    .len()
            })
            .collect();
        for pair in sizes.windows(2) {
            assert!(
                pair[0] < pair[1],
                "the sizes do not climb with quality: {sizes:?}"
            );
        }
    }

    /// The pad cannot overwrite the last symbol.
    ///
    /// This is a regression test for a defect libjpeg found and `image`'s
    /// decoder did not. [`BitWriter::finish`] pads to a byte boundary with
    /// ones and used to pass `0x7F` whatever width it needed. `write` ORs its
    /// argument in at the field its width describes, so the leftover ones in
    /// `0x7F` landed *above* that field, on the bits already written, and
    /// rewrote the last four bits of the scan. Every file long enough to end
    /// mid-byte lost its final symbol, and both libjpeg and a reference
    /// decoder read past the end of the data and reconstructed the last one or
    /// two blocks as noise.
    ///
    /// The cell is at this level rather than over a whole tile because that is
    /// where it is decidable: two bad blocks in 1536 moved a tile's PSNR by
    /// 0.75 dB, which no fidelity bound anybody would write is tight enough to
    /// catch.
    #[test]
    fn the_pad_cannot_overwrite_the_last_symbol() {
        for pending in 1..8u8 {
            let mut w = BitWriter::new(Vec::new());
            // `pending` bits ending in a one, so anything the pad writes over
            // them is visible in the byte.
            w.write(0, pending - 1);
            w.write(1, 1);
            let out = w.finish();
            let tail = 8 - u32::from(pending);
            let want = u8::try_from((1u32 << tail) | ((1u32 << tail) - 1))
                .expect("a byte's worth of ones");
            assert_eq!(
                out[0], want,
                "{pending} pending bits padded to {:#010b}, not {want:#010b}",
                out[0]
            );
            // One byte, unless the pad made it `0xFF`, which the stuffing
            // follows with a zero so no decoder reads it as a marker.
            let expected_len = 1 + usize::from(want == 0xFF);
            assert_eq!(out.len(), expected_len, "{pending} pending bits: {out:?}");
        }
    }

    /// A value wider than the width it is written at cannot reach the bits
    /// already in the accumulator.
    ///
    /// The mask in [`BitWriter::write`] is what the cell above needs, and this
    /// one says so directly rather than through the pad that found it.
    #[test]
    fn a_too_wide_value_cannot_reach_the_bits_already_written() {
        let mut clean = BitWriter::new(Vec::new());
        clean.write(0b101, 3);
        clean.write(0b1, 1);
        let mut wide = BitWriter::new(Vec::new());
        wide.write(0b101, 3);
        // Same one bit, spelled with every bit above it set as well.
        wide.write(u16::MAX, 1);
        assert_eq!(clean.finish(), wide.finish());
    }

    /// Both ends of the quality range, in every mode.
    ///
    /// Quality 1 clamps every quantization entry to the 255 an 8-bit DQT can
    /// spell, and quality 100 puts every one of them at 1, so the two ends
    /// exercise the clamps in [`scaled_qtable`] from opposite directions and
    /// produce the two extremes of coefficient magnitude. A 1x1 image is the
    /// degenerate case on top of that: one real pixel and 63 replicated ones.
    #[test]
    fn both_ends_of_the_quality_range_round_trip() {
        let src = drawing(64, 64);
        for q in [1u8, 2, 10, 50, 99, 100] {
            for mode in [JpegSubsample::Auto, JpegSubsample::Off, JpegSubsample::On] {
                let bytes = encode(&src, 64, 64, image::ColorType::Rgb8, q, mode)
                    .unwrap_or_else(|e| panic!("q{q} {mode:?} encodes: {e}"));
                assert_eq!(decode(&bytes).dimensions(), (64, 64), "q{q} {mode:?}");
            }
            let one = encode(
                &[7, 8, 9],
                1,
                1,
                image::ColorType::Rgb8,
                q,
                JpegSubsample::Auto,
            )
            .unwrap_or_else(|e| panic!("a 1x1 raster at q{q} encodes: {e}"));
            assert_eq!(decode(&one).dimensions(), (1, 1), "q{q} 1x1");
        }
    }

    /// The refusals: the colour types `image_color_type` maps that JPEG has no
    /// sample layout for, and the two shapes of malformed call.
    ///
    /// `Rgba8` is on the list because the flattening happens in
    /// `crate::sink::flatten_alpha` before anything reaches here, so an RGBA
    /// buffer arriving at this function means a call site skipped it rather
    /// than that the caller wants the alpha dropped.
    #[test]
    fn sixteen_bit_rgba_and_degenerate_rasters_are_refused() {
        let q = 85;
        let m = JpegSubsample::Auto;
        let err = encode(&[0; 8], 2, 2, image::ColorType::L16, q, m)
            .expect_err("16-bit has no JPEG sample type");
        assert!(format!("{err}").contains("L16"), "{err}");

        let err = encode(&[0; 16], 2, 2, image::ColorType::Rgba8, q, m)
            .expect_err("alpha is flattened before this point");
        assert!(format!("{err}").contains("Rgba8"), "{err}");

        let err = encode(&[], 0, 4, image::ColorType::Rgb8, q, m).expect_err("no pixels");
        assert!(format!("{err}").contains("0x4"), "{err}");

        let err = encode(&[0; 5], 2, 2, image::ColorType::Rgb8, q, m).expect_err("a short buffer");
        assert!(format!("{err}").contains("12 bytes"), "{err}");
    }
}
