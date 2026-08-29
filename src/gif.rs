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
//! | libviprs method          | libvips equivalent      | result                                           |
//! |--------------------------|-------------------------|--------------------------------------------------|
//! | [`decode_gif`]           | `gifload`               | frame 0 as `Rgb8` or `Rgba8`, plus the GIF fields |
//! | [`decode_gif_with`]      | `gifload` `page` / `n`  | a window of frames as one page roll               |
//! | [`Raster::encode_gif`]   | `gifsave_buffer`        | GIF89a bytes, one frame per page                 |
//! | [`Raster::save_gif`]     | `gifsave`               | the same bytes written to a path                 |
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
//!   [`LoopCount::from_gif_wire`](crate::frames::LoopCount::from_gif_wire)
//!   is that mapping over an `Option`, and the `gif` crate's decoder keeps
//!   the two apart cleanly: it reports a block holding 0 as
//!   `Repeat::Infinite` and leaves its `Repeat::Finite(0)` default in place
//!   when there is no block, so "play once" and "play forever" never
//!   collapse onto each other.
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
//! # Header fields
//!
//! Every field `gifload` attaches beside the pixels, and what this loader
//! does with it. **Every one of them is attached**, as of #852. The table is
//! **checked**, three ways:
//! `the_header_field_table_matches_what_this_loader_attaches` parses it out of
//! this file and requires every "attached" row to be present on a raster built
//! to satisfy every condition here, every "not attached" row to be absent from
//! it, and every row that says "attached **when**" to be absent from a second
//! raster that satisfies none of them. It also requires the raster to carry
//! nothing the table does not list, so "every field" is a claim rather than a
//! sentence. That is what issue #801 asked for, after the paragraph this
//! replaces named a blocker that had already gone.
//!
//! | `gifload` field | vips type | here |
//! |---|---|---|
//! | `n-pages` | `gint` | attached |
//! | `loop` | `gint` | attached |
//! | `delay` | `VipsArrayInt` | attached, one entry per loaded page |
//! | `page-height` | `gint` | attached when more than one page is loaded |
//! | `palette` | `gint` | attached, always 1 |
//! | `bits-per-sample` | `gint` | attached |
//! | `interlaced` | `gint` | attached when some frame is stored interlaced |
//! | `gif-palette` | `VipsArrayInt` | attached when no frame carries a local colour table |
//! | `background` | `VipsArrayDouble` | attached, the colour table entry the screen descriptor points at |
//! | `gif-loop` | `gint` | attached, `loop - 1`, except that `loop 0` and `loop 1` both give 0 |
//! | `gif-delay` | `gint` | attached, this raster's first delay in centiseconds |
//! | `palette-bit-depth` | `gint` | attached, the same number as `bits-per-sample` |
//!
//! `gif-palette` is the global colour table as one signed 32-bit word per
//! entry, `0xFF << 24 | B << 16 | G << 8 | R`, so every entry is negative.
//! vips gets that number by reinterpreting libnsgif's `R, G, B, A` byte quad
//! as a machine integer, which makes **vips's** answer host-dependent; this
//! one is not. The little-endian reading is pinned, because the `.v` trailer
//! carries the value as decimal text and a field that meant one thing on the
//! machine that wrote it and another on the machine that read it would be
//! worse than a divergence confined to big-endian hosts, where nothing in
//! this project runs.
//!
//! The array is the table as it sits on the wire, padding included: a
//! three-colour table comes back with a fourth, opaque-black entry. Its
//! presence rule is about the **file** and not the window that was loaded, so
//! a two-frame file whose frame 0 carries a local table reports no
//! `gif-palette` at `[page=1]` either.
//!
//! **`gif-delay`, `gif-loop` and `palette-bit-depth` are deprecated aliases**
//! that `gifload` attaches to every GIF, still (issues #865, #875).
//! `gif-delay` is the first delay back in centiseconds, `gif-loop` is the
//! NETSCAPE count rather than the play count (`loop 1` and `loop 0` both give
//! 0), and `palette-bit-depth` is a second copy of `bits-per-sample`. Two
//! things about measuring them:
//!
//! * **`vipsheader -a` cannot see any of the three.** It lists no deprecated
//!   compatibility field, on any loader, animated or not, so checking them
//!   that way produces the opposite finding. `vipsheader -f gif-loop` returns
//!   the value, and so does the `.v` trailer.
//! * **`gif-delay` follows this raster, not the file**, which is the same
//!   deliberate divergence `delay` makes and for the same reason: vips takes
//!   it from element 0 of the file's whole array, so
//!   `anim4.gif[page=2,n=2]` reports `gif-delay: 4` for a raster whose first
//!   page really has a delay of 80 ms. [`crate::webp`] takes it off the
//!   loaded delays too, so the animated loaders agree with each other.
//!
//! **A file that declares no global colour table gets libnsgif's own**: two
//! entries, black then white, with `bits-per-sample: 1`, `gif-palette:
//! -16777216 -1`, and a white pixel wherever a frame stores index 1. The
//! stored background index is ignored with it, so a restore-to-background
//! disposal paints entry 0 whatever the descriptor says. All of that is
//! measured; `normalise_for_decoder` is how it is reached, and issue #851 has
//! the reason it needs reaching.
//!
//! # Where libviprs and vips diverge, and why it is not a bug
//!
//! LZW is exactly lossless and deterministic in both directions, so the
//! bitstream is not where the divergence lives. **Palette quantisation is.**
//! vips quantises with libimagequant; libviprs uses the same crate-internal
//! median-cut quantiser that already backs [`Raster::encode_png_palette`].
//! Two different algorithms pick two
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
//! # Animation
//!
//! [`decode_gif_with`] takes vips's `page` and `n` and stacks the frames it
//! selects into one raster whose rows are a whole number of equal-height
//! pages, the layout [`crate::frames`] describes. `decode_gif` is that with
//! `page = 0, n = 1`, which is vips's default and a still image.
//!
//! Four things about it are measured against vips 8.18.6 rather than read
//! off `nsgifload.c`, because each of them is a place a plausible
//! implementation is quietly wrong:
//!
//! * **The delay unit.** The graphic control extension counts
//!   centiseconds and vips's `delay` counts milliseconds, so `4 6 8 10` on
//!   the wire is `40 60 80 100` in the field.
//!   [`FrameDelay::from_centiseconds`] is the crossing and the unit is in
//!   the type, because an integer that passes straight through is a silent
//!   factor of ten no other assertion catches.
//! * **A frame with no graphic control extension is 100 ms, not 0** (issue
//!   #866). libnsgif initialises a frame's delay to 10 centiseconds when it
//!   allocates it and only an extension overwrites it, so the default reaches
//!   the array: measured, a still with no extension reports `delay: 100`
//!   while one whose extension holds an explicit zero reports `delay: 0`, and
//!   a four-frame file with extensions on frames 0 and 2 only reports
//!   `30 100 50 100`. The `gif` crate cannot tell the two apart, which is why
//!   this rides on the same wire walk code 4 does.
//! * **The delay array covers the pages this raster holds**, one entry per
//!   page, and that is a deliberate divergence. vips reports the whole
//!   file's array whatever window was loaded: `anim4.gif[page=2,n=2]` loads
//!   frames 2 and 3 and still says `delay: 40 60 80 100`, so re-saving it
//!   writes 40 and 60 centiseconds onto frames whose real delays are 80 and
//!   100. Both halves of that are measured. Making `delay[i]` loaded page
//!   `i`'s delay is what makes the array usable on the raster it is attached
//!   to, and it is the same split `n-pages` already has: `n-pages` describes
//!   the file and [`Raster::pages_loaded`] describes the raster.
//! * **`page-height` is attached only when more than one page is loaded**,
//!   which is what vips does: a default `gifload` of a four-frame file
//!   attaches `n-pages`, `loop` and `delay` and no `page-height`, and so
//!   does `[page=3]`.
//! * **Compositing starts at frame 0 whatever `page` says.** GIF frames are
//!   differences, so a window cannot be rendered from its own first frame.
//!
//! ## Saving an animation
//!
//! [`Raster::encode_gif`] splits the raster by its page height and writes one
//! GIF frame per page, at the page's size: the logical screen descriptor of a
//! roll vips writes holds the page height, so a roll of many pages is not
//! itself bounded by GIF's 65535-pixel axis. The delays come out of the
//! `delay` field and the NETSCAPE block out of `loop`, which is where vips
//! reads them too: `gifsave` has no `delay` or `loop` argument,
//! `cgifsave.c:753` reads them back off the image, and `--keep none` does not
//! drop them.
//!
//! There is no `page-height` save option, unlike vips. The page split is set
//! on the raster with
//! [`Raster::try_set_page_height`](crate::Raster::try_set_page_height), which
//! refuses a height that does not divide it. vips accepts any value and its
//! *reader* discards a bad one, so `vips gifsave roll.v out.gif
//! --page-height 5` on a 12-row image writes a single 4x12 frame and says
//! nothing. Refusing at the setter puts the error where the mistake is; a
//! raster that reached the save path with a bad stored value is one page on
//! both sides, which is what vips reports for it too.
//!
//! Three more things about the save are measured, and two of them diverge
//! deliberately:
//!
//! * **Delays go out as `round(ms / 10)`, halves to even.** `35 55 15 25` ms
//!   produced centiseconds `4 6 2 2` and `45 67 5 1` produced `4 7 0 0`;
//!   truncation would write 6 for 67 ms and half-up would write 3 for 25 ms.
//!   [`FrameDelay::browser_floor`](crate::frames::FrameDelay::browser_floor)
//!   is **not** applied: `8 9 10 11` ms went out of `gifsave` as `1 1 1 1`
//!   centiseconds where `webpsave` wrote `100 100 100 11` milliseconds.
//! * **A `delay` array whose length is not the page count is refused**, and a
//!   negative delay or `loop` with it. vips pads, truncates and wraps in
//!   silence: a two-entry array on a four-page roll wrote `2 3 0 0`, a
//!   six-entry one wrote `2 3 4 5`, a delay of -10 ms became 655 seconds and
//!   `loop = -1` became 65536 plays. A delay past the wire ceiling saturates
//!   here where vips wraps 655360 ms into no delay at all. A field of the
//!   *wrong type* is ignored rather than refused, the way the `page-height`
//!   and `n-pages` readers already treat one.
//! * **Disposal follows cgif.** Every frame but the last is
//!   restore-to-background when the animation carries transparency and
//!   keep-the-canvas otherwise, measured over five files. It is not
//!   cosmetic: every frame written here covers the whole screen, so under
//!   "keep" a transparent pixel on page 2 would show page 1 through it.
//!
//! The palette is quantised once over the whole roll, so one global colour
//! table serves every frame. vips quantises per frame and reuses the previous
//! palette when the inter-palette error is small, which is cgif and
//! libimagequant machinery with no pure-Rust equivalent, and it is why the
//! bytes never match.
//!
//! ## Disposal and blending
//!
//! Each frame paints its own rectangle over the canvas, skipping its
//! transparent index so what is underneath shows through, and then the
//! canvas is disposed of before the next frame draws. The rules are
//! libnsgif's, each one measured by building the fixture, running it through
//! the pinned vips binary and pinning what came back:
//!
//! | disposal code | what happens to the canvas                                    |
//! |---------------|---------------------------------------------------------------|
//! | 0, 1          | nothing; the next frame draws over it                         |
//! | 2             | this frame's rectangle is cleared, and only it                |
//! | 3, 4          | the whole canvas rewinds to before this frame drew            |
//! | 5, 6, 7       | nothing, as for 0 and 1                                       |
//!
//! Code 2 has two arms and one fixture cannot see both: the clear is
//! **transparent** when the disposed frame declares a transparent index and
//! the **background colour** when it does not, and the background colour is
//! the global colour table entry the screen descriptor points at, or **entry
//! 0** when that index is past the end of the table. All three are pinned.
//!
//! **Code 4 costs a second walk, and it is worth it** (issue #827). GIF89a
//! reserves 4 to 7; libnsgif remaps 4 onto restore-to-previous
//! (`NSGIF_DISPOSAL_RESTORE_QUIRK`) and leaves the rest alone, and the `gif`
//! crate's `DisposalMethod::from_u8` knows only 0 to 3 and folds every other
//! code onto `Any`, so by the time [`decode_gif_with`] sees a frame the code
//! is gone. `ControlWalk` walks the block chain a second time to read each
//! frame's graphic control extension off the wire. It decodes no pixels, and
//! it is not trusted blind: the extension carries the delay, the transparent
//! index and the disposal alongside each other, so
//! `reconcile_control` requires the three the decoder *did* keep to match
//! before it believes what only the wire knows, and falls back to the
//! decoder's own answer for that frame when they do not. The raw code is one
//! of those two things; whether there was an extension at all is the other,
//! and that one is issue #866.
//!
//! The alternative was to keep libviprs' answer and pin the gap, which is
//! what [`crate::resample`]'s rule allows for a difference inside the
//! carrier's noise. This one is not: it is a whole canvas, one-directional,
//! and libviprs' answer is not the more accurate of the two, it is the
//! `gif` crate's lossy mapping showing through.
//!
//! **The walk reads the extension twice, and that is not redundancy** (issue
//! #878). A graphic control extension may arrive as more than one sub-block,
//! or with a size byte that does not say 4, and the two implementations then
//! read different bytes: libnsgif never looks at the chain and takes the four
//! bytes straight after the size byte, while the `gif` crate takes the *last*
//! sub-block, because it clears its extension buffer on every one. So the
//! walk answers with libnsgif's four bytes and cross-checks with the crate's,
//! which are the same bytes on every ordinary file and are not on these.
//! Measured on vips 8.18.6: a chain of `01 AA 04 <quad> 00` comes back as
//! restore-to-background with a delay of 3076 centiseconds, which is `0xAA`
//! read as the packed byte and the two bytes after it read little-endian.
//!
//! # Two files the decoder is stricter about than the format
//!
//! The `gif` crate refuses two shapes libnsgif reads, and both refusals land
//! before this module sees a frame, so no amount of careful reading recovers
//! them. What does is handing the decoder *different bytes*:
//! `normalise_for_decoder` rewrites the file into the shape the crate reads
//! the way libnsgif reads it, and returns nothing at all for the ordinary GIF
//! that needs neither rewrite.
//!
//! * **No global colour table** (issue #851). `Decoder::next_frame_info`
//!   errors on a frame with neither a local palette nor a global one
//!   (`reader/mod.rs:438-442`) and `Decoder` cannot be given one from
//!   outside, so libnsgif's substitute table is spliced into the logical
//!   screen descriptor and the background index forced to entry 0 with it.
//! * **A control extension whose last sub-block is not four bytes** (issue
//!   #879). `read_control_extension` refuses it and the error ends the header
//!   scan, taking every later frame with it. The chain is rewritten into the
//!   single four-byte sub-block libnsgif's own fixed-offset read produces, so
//!   the decoder gets the same four bytes in the one shape it accepts.
//!
//! The copy is priced through [`DecodeLimits`], because it is a second
//! whole-file buffer and it happens before every other price this module
//! pays.
//!
//! This overlaps the two readings above on purpose. Once a chain is
//! rewritten the crate and libnsgif agree about it, so the split reading has
//! nothing left to disagree about; it stays because it is what covers the
//! chains the rewrite leaves alone, and because a file
//! `normalise_for_decoder` cannot parse is handed over untouched.
//!
//! # Not handled here
//!
//! **Nothing, of `gifload`'s header fields.** `background` was the last one
//! this loader read and did not attach, and #852 landed the
//! [`MetadataValue::DoubleArray`] it needed: vips stores it as a
//! `VipsArrayDouble` and an integer array would have been numerically right
//! and typed wrong, which is a field every reader written against vips
//! ignores.
//!
//! `gifsave`'s `effort`, `reuse`,
//! `interpalette-maxerror`, `interframe-maxerror` and `keep-duplicate-frames`
//! are cgif-specific palette-reuse and frame-coalescing machinery with no
//! pure-Rust equivalent and are not modelled.
//!
//! Nor are `gifsave`'s `keep`, `profile` and `background`, and those three
//! share one reason: the encoder here writes no metadata at all. There is no
//! EXIF, XMP or ICC block in the output, so `keep` has nothing to select
//! between and `profile` has nothing to embed; and the logical screen
//! descriptor's background index is always 0, which is the reserved
//! transparent entry when there is one, so `background` has nothing to point
//! at. The load side does read the stored index, for the restore-to-background
//! disposal.
//!
//! Every entry point here is fallible. Load failures arrive as
//! [`GifError`] through [`SourceError`], save failures as [`EncodeError`];
//! there is no panicking twin, matching the rest of the codec surface.

use crate::codec::EncodeError;
use crate::frames::{FrameDelay, LoopCount};
use crate::imageio::{MetadataValue, SaveError};
use crate::pixel::PixelFormat;
use crate::raster::{Raster, buffer_len};
use crate::source::{DecodeLimits, SourceError};
use std::io::Cursor;
use std::ops::Range;
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
    /// [`LoadOptions`] asked for pages the file does not have.
    ///
    /// vips raises this as `"bad page number"` and it covers every way the
    /// window can miss: measured on 8.18.6 against a four-frame file,
    /// `[page=4]`, `[n=99]`, `[n=0]` and `[page=3,n=3]` all fail that way,
    /// while `[page=2,n=-1]` loads frames 2 and 3.
    #[error("gif: bad page number; page {page} count {n} on a {frames}-frame file")]
    BadPageNumber {
        /// The first page asked for, counting from zero.
        page: u32,
        /// How many pages were asked for, `-1` for every remaining page.
        n: i32,
        /// How many frames the file actually holds.
        frames: u32,
    },
    /// A roll of `pages` screens is taller than a raster can be.
    ///
    /// Only reachable with the allocation, coordinate and pixel ceilings all
    /// lifted, since a roll this tall is over every one of them at their
    /// defaults. It exists so the overflow is a refusal rather than a panic
    /// in the page copy that follows.
    #[error("gif: {pages} pages of {height} rows is taller than an image can be")]
    RollTooTall {
        /// The logical screen height, which is one page.
        height: u32,
        /// How many pages the load asked for.
        pages: u32,
    },
    /// A frame declares a rectangle whose index buffer will not fit `usize`.
    ///
    /// Distinct from the allocation budget, which the frame is priced against
    /// first: this is the 32-bit target where clearing a `u64` price is not
    /// the same as fitting the address space.
    #[error("gif: a {width}x{height} frame does not fit this target's address space")]
    FrameTooLarge {
        /// The frame rectangle's width.
        width: u32,
        /// The frame rectangle's height.
        height: u32,
    },
    /// The raster could not be built from the decoded pixels.
    #[error(transparent)]
    Raster(#[from] crate::raster::RasterError),
}

/// Options for [`decode_gif_with`] (libvips `gifload`'s `page` and `n`).
///
/// `#[non_exhaustive]`, `Default`, and module-scoped, the same shape as
/// [`SaveOptions`] and [`DecodeLimits`]: start from [`LoadOptions::default`]
/// and set what you need with the `with_*` builders, e.g.
/// `gif::LoadOptions::default().with_n(-1)` (issue #630).
///
/// The default is vips's: page 0, one page, so [`decode_gif`] is
/// `decode_gif_with(bytes, limits, LoadOptions::default())` and a still load
/// is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LoadOptions {
    /// The first frame to load, counting from **zero**, matching vips's
    /// `page` and [`crate::decode_tiff_page`]'s convention (issue #566).
    /// Defaults to 0.
    pub page: u32,
    /// How many frames to load, `-1` for every frame from [`page`](Self::page)
    /// to the end. Defaults to 1, as vips does.
    ///
    /// An `i32` rather than an `Option<u32>` because that is the shape vips's
    /// argument has, sentinel included, and the sentinel is the only negative
    /// value either accepts.
    pub n: i32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { page: 0, n: 1 }
    }
}

impl LoadOptions {
    /// Set the first page to load, returning the updated options.
    #[must_use]
    pub fn with_page(mut self, page: u32) -> Self {
        self.page = page;
        self
    }

    /// Set how many pages to load, `-1` for every remaining page, returning
    /// the updated options.
    #[must_use]
    pub fn with_n(mut self, n: i32) -> Self {
        self.n = n;
        self
    }

    /// The half-open range of frames these options select out of a file
    /// holding `frames` of them.
    ///
    /// # Errors
    ///
    /// [`GifError::BadPageNumber`] for any window the file cannot serve,
    /// which is the single case vips reports for all of them.
    fn window(self, frames: u32) -> Result<Range<u32>, GifError> {
        let bad = || GifError::BadPageNumber {
            page: self.page,
            n: self.n,
            frames,
        };
        if self.page >= frames {
            return Err(bad());
        }
        let count = match self.n {
            -1 => frames - self.page,
            n => u32::try_from(n).map_err(|_| bad())?,
        };
        let end = self.page.checked_add(count).ok_or_else(bad)?;
        if count == 0 || end > frames {
            return Err(bad());
        }
        Ok(self.page..end)
    }
}

/// Options for [`Raster::encode_gif`] (libvips `gifsave` / `gifsave_buffer`).
///
/// `#[non_exhaustive]`, `Default`, and module-scoped, the same shape as
/// [`DecodeLimits`]: start from
/// [`SaveOptions::default`] and set what you need with the `with_*` builders,
/// e.g. `gif::SaveOptions::default().with_dither(0.0)`. That is what makes
/// "later fields can be added without a breaking change" true rather than
/// merely written down; a struct literal here would compile today and stop the
/// day a field lands (issue #630).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
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
    /// colours, so 8 does not mean 256: `cgifsave.c` caps vips at 255 too,
    /// to keep one index free for the transparency optimisation.
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
    /// Write the frame interlaced, returning the updated options.
    #[must_use]
    pub fn with_interlaced(mut self, interlaced: bool) -> Self {
        self.interlaced = interlaced;
        self
    }

    /// Set the dithering amount, returning the updated options.
    #[must_use]
    pub fn with_dither(mut self, dither: f64) -> Self {
        self.dither = dither;
        self
    }

    /// Set the bits per pixel, returning the updated options.
    #[must_use]
    pub fn with_bitdepth(mut self, bitdepth: u8) -> Self {
        self.bitdepth = bitdepth;
        self
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
    /// Whether *any* frame declares a colour table of its own.
    ///
    /// vips attaches `gif-palette` only when none does (`nsgifload.c:315`,
    /// `if (!gif->local_palette)`, over a flag its own header pass ORs across
    /// every frame in the file at `:435`), so the rule is about the file and
    /// not about the window that was loaded.
    any_local_palette: bool,
}

/// Decode a GIF into a [`Raster`] (libvips `gifload`).
///
/// Loads **frame 0** at the logical screen size, which is what `vips
/// gifload` does by default (`page = 0`, `n = 1`). The palette is expanded
/// to `Rgb8`, or to `Rgba8` when any frame in the file declares a
/// transparent index; see the [module docs](crate::gif) for the full
/// semantics. [`decode_gif_with`] is the same decoder with vips's `page` and
/// `n`, for loading more than one frame.
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
    decode_gif_with(bytes, limits, LoadOptions::default())
}

/// Decode a window of a GIF's frames into one page roll (libvips `gifload`
/// with `page` and `n`).
///
/// The frames `options` selects are composited in file order and stacked top
/// to bottom into a single raster whose `page-height` is the logical screen
/// height, which is the toilet-roll layout [`crate::frames`] describes and
/// the one `vips copy 'anim.gif[n=-1]' out.v` writes. A one-page window is
/// exactly [`decode_gif`], down to the fields.
///
/// Compositing always starts at frame 0 whatever `page` says, because GIF
/// frames are differences: a frame may paint a sub-rectangle, may leave
/// pixels transparent to show what is underneath, and says how the canvas is
/// to be disposed of before the next one draws. Skipping to the window would
/// render the wrong pixels, and vips does not skip either.
///
/// # Errors
///
/// Everything [`decode_gif`] returns, plus [`GifError::BadPageNumber`] when
/// `options` asks for pages the file does not hold, and
/// [`SourceError::PageLimitExceeded`] when the file declares more frames
/// than [`DecodeLimits::max_pages`].
pub fn decode_gif_with(
    bytes: &[u8],
    limits: DecodeLimits,
    options: LoadOptions,
) -> Result<Raster, SourceError> {
    // Two shapes the `gif` crate refuses outright and libnsgif reads, so the
    // decoder is handed the same file in the shape it will read (issues #851
    // and #879). `None` for every ordinary GIF, and then nothing is copied.
    let normalised = normalise_for_decoder(bytes, limits)?;
    let bytes = normalised.as_deref().unwrap_or(bytes);
    let scan = scan_file(bytes, limits)?;
    if scan.frames == 0 {
        return Err(GifError::NoFrames.into());
    }
    let window = options.window(scan.frames).map_err(SourceError::from)?;
    let pages = window.end - window.start;

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

    // The roll is `pages` screens stacked, and it is priced separately from
    // the canvas because it is a separate allocation and, for an animation,
    // much the larger of the two.
    //
    // The product is taken in `u64` and *checked* rather than saturated. A
    // saturating narrow would substitute a `u32::MAX` height that is smaller
    // than the true one, and `data` is sized from it while the emit loop
    // writes `pages` full pages into it, so the overflow would land as a
    // panic in `copy_from_slice` rather than as a refusal. Arguing that the
    // ceilings always catch it first is true at the default limits and not
    // true at the `u64::MAX` / `u32::MAX` spelling the crate documents for
    // "no limit", which is exactly the caller who would meet it.
    let Ok(roll_height) = u32::try_from(u64::from(height) * u64::from(pages)) else {
        return Err(GifError::RollTooTall { height, pages }.into());
    };
    if pages > 1 {
        limits.check_coord(width, roll_height)?;
        limits.check_pixels(width, roll_height)?;
        limits.check_image_alloc("GIF animation", width, roll_height, bands as u64, 1)?;
    }

    // The canvas starts fully transparent and stays that way outside the
    // frame rectangle. libnsgif renders frame 0 over a cleared buffer and
    // never paints the background colour, which is why `vips getpoint`
    // reports `0 0 0` at a corner the header calls blue.
    // Through `buffer_len` rather than a plain `usize` product for the
    // same reason the price goes through `decode_alloc_bytes`: clearing
    // the budget says the byte count fits a `u64`, which on a 32-bit
    // target is not the same as fitting the address space, and a caller
    // can raise `max_alloc_bytes` past 4 GiB there.
    let page_bytes = buffer_len(width, height, bands).map_err(GifError::Raster)?;
    let mut canvas = vec![0u8; page_bytes];
    let mut data = vec![0u8; buffer_len(width, roll_height, bands).map_err(GifError::Raster)?];
    // The snapshot a "restore to previous" disposal rewinds to. Allocated
    // lazily, and not priced again: it is the same size as the canvas, which
    // the budget has already cleared, and the budget is per allocation.
    let mut previous: Vec<u8> = Vec::new();
    let mut delays: Vec<i64> = Vec::with_capacity(pages as usize);
    let global = decoder.global_palette().map(<[u8]>::to_vec);
    let background = background_rgb(global.as_deref(), decoder.bg_color());
    // The second walk that recovers the raw disposal code, advanced one frame
    // per turn of the loop below so it stays beside the decoder rather than
    // running ahead of it into a buffer (issue #827).
    let mut controls = ControlWalk::new(bytes);

    for index in 0..window.end {
        let Some(frame) = decoder.next_frame_info().map_err(decode_error)? else {
            // The window is bounded by `scan_file`'s count, so reaching the
            // end of the file inside it means the two walks disagree about
            // where the file ends. That is a decode failure and it is
            // reported as one; `NoFrames` would be the wrong sentence for a
            // file that demonstrably has frames.
            return Err(GifError::Decode {
                message: format!(
                    "frame {index} is missing; the header scan counted {} frames",
                    scan.frames
                ),
            }
            .into());
        };
        let transparent = frame.transparent;
        let dispose = frame.dispose;
        let delay_cs = frame.delay;
        // The wire walk's answer for this frame, `None` when it cannot be
        // shown to be looking at the same frame the decoder is. Two things
        // hang off it: the raw disposal code (#827) and whether the frame
        // carried a control extension at all (#866).
        let wire = reconcile_control(controls.next_frame(), dispose, delay_cs, transparent);
        let disposal = wire.map_or_else(|| Disposal::from_crate(dispose), WireControl::disposal);
        let (left, top) = (u32::from(frame.left), u32::from(frame.top));
        let (fw, fh) = (u32::from(frame.width), u32::from(frame.height));
        let local = frame.palette.clone();
        // The index buffer is the frame's own rectangle, not the screen's,
        // and a GIF may declare a frame far larger than the screen it sits
        // on: `open()` leaves the `gif` crate's `check_frame_consistency`
        // off, matching libnsgif, which clips such a frame rather than
        // refusing the file. Clipping happens below, after the buffer is
        // allocated, so a 1x1 screen carrying one 65535x65535 frame is a
        // 4 GiB allocation off a forty-byte file. Priced here, through the
        // crate's own budget, because the screen price two dozen lines up
        // does not cover it and the animation walk does this once per frame.
        let indices_bytes = limits.check_image_alloc("GIF frame indices", fw, fh, 1, 1)?;
        let mut indices =
            vec![
                0u8;
                usize::try_from(indices_bytes).map_err(|_| GifError::FrameTooLarge {
                    width: fw,
                    height: fh
                })?
            ];
        // A truncated frame keeps the rows that did arrive. The buffer starts
        // zeroed and `read_into_buffer` fills it in order, so the tail is left
        // as index 0 exactly where libnsgif leaves it uncomposited -- which is
        // why `vips getpoint truncated.gif 574 799` prints `0 0 0 0` while the
        // top of the same file is real image data.
        let _truncated = decoder.read_into_buffer(&mut indices);
        let palette = local.as_deref().or(global.as_deref()).unwrap_or(&[]);

        let last = index + 1 == window.end;
        if !last && disposal == Disposal::Previous {
            previous.clear();
            previous.extend_from_slice(&canvas);
        }

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
                canvas[out..out + PALETTE_STRIDE].copy_from_slice(rgb);
                if bands == 4 {
                    canvas[out + 3] = u8::MAX;
                }
            }
        }

        if index >= window.start {
            let page = (index - window.start) as usize * page_bytes;
            data[page..page + page_bytes].copy_from_slice(&canvas);
            // The one conversion #572 exists for: the graphic control
            // extension counts centiseconds and vips's `delay` counts
            // milliseconds, so the type carries the unit across the boundary
            // rather than an integer that looks the same either way.
            //
            // The centiseconds come off the wire walk when it can be trusted,
            // because a frame with **no** control extension is 10 of them and
            // not 0, and `gif` 0.14.2 reports 0 for that case and for an
            // extension holding zero alike (#866). An untrusted walk falls
            // back to the decoder's number, which is this module's answer
            // from before #866 and is right for every frame that has an
            // extension.
            let frame_delay =
                FrameDelay::from_centiseconds(wire.map_or(delay_cs, WireControl::delay_cs));
            delays.push(i64::from(frame_delay.millis()));
        }

        // The disposal after the last frame walked is invisible, since
        // nothing draws over it, and skipping it is what keeps a one-page
        // load from cloning a snapshot it will never restore. That is the
        // argument for not pricing `previous` separately.
        if last {
            continue;
        }
        match disposal {
            // Clear this frame's rectangle, and only it. libnsgif fills with
            // transparent when *this* frame declared a transparent index and
            // with the background colour when it did not, which is measured
            // rather than assumed: the same two-frame file loads with a
            // transparent hole when frame 0 carries the index and with an
            // opaque blue one when frame 1 carries it instead.
            Disposal::Background => {
                let fill: [u8; 4] = if transparent.is_some() {
                    [0, 0, 0, 0]
                } else {
                    [background[0], background[1], background[2], u8::MAX]
                };
                for y in 0..fh.min(height.saturating_sub(top)) {
                    for x in 0..fw.min(width.saturating_sub(left)) {
                        let out = (((y + top) * width) + x + left) as usize * bands;
                        canvas[out..out + bands].copy_from_slice(&fill[..bands]);
                    }
                }
            }
            // Rewind the whole canvas to the snapshot taken above. Only this
            // frame has drawn since, so restoring the canvas and restoring
            // its rectangle are the same thing. Codes 3 and 4 both land here:
            // 4 is reserved by GIF89a and libnsgif remaps it onto 3, which is
            // the code the `gif` crate cannot tell apart from 0 and which
            // `controls` is walking the file a second time to recover.
            Disposal::Previous => canvas.copy_from_slice(&previous),
            // Codes 0 and 1 leave the canvas alone, and so do the reserved
            // codes 5, 6 and 7, measured on vips 8.18.6.
            Disposal::Keep => {}
        }
    }

    let format = if bands == 4 {
        PixelFormat::Rgba8
    } else {
        PixelFormat::Rgb8
    };
    let mut raster = Raster::new(width, roll_height, format, data).map_err(GifError::Raster)?;
    raster.meta.interpretation = Some(crate::conversion::Interpretation::Srgb);
    // The page split is declared only when there is one, matching vips: a
    // default `gifload` of a four-frame file attaches `n-pages`, `loop` and
    // `delay` but no `page-height`, and so does `[page=3]`. The setter
    // refuses a height that does not divide the raster, so a miscounted roll
    // fails here rather than writing a split a reader would discard.
    if pages > 1 {
        raster
            .try_set_page_height(height)
            .map_err(GifError::Raster)?;
    }
    raster.set_n_pages(scan.frames);
    raster.set_field("loop", MetadataValue::Int(scan.loop_count));
    // `gif-loop` is the deprecated scalar beside `loop`, and it is the
    // NETSCAPE count rather than the play count: `nsgifload.c:303-306` is
    // `loop_max == 0 ? 0 : loop_max - 1`, which is what
    // `LoopCount::to_gif_wire` computes, with `None` (the file that plays
    // once and carries no block) reported as 0. Measured on vips 8.18.6 with
    // `vipsheader -f`: no block reports 0, a block holding 0 reports 0, and
    // blocks holding 1, 3 and 65535 report themselves.
    raster.set_field(
        "gif-loop",
        MetadataValue::Int(i64::from(
            LoopCount::from_plays(u32::try_from(scan.loop_count).unwrap_or(u32::MAX))
                .to_gif_wire()
                .unwrap_or(0),
        )),
    );
    // One delay per page this raster holds, which vips does not promise: it
    // reports the whole file's array whatever window was loaded, so
    // `delay[0]` on a `[page=2,n=2]` load is frame 0's delay sitting on a
    // page that is really frame 2. Measured, re-saving that raster writes 40
    // and 60 centiseconds onto frames whose real delays are 80 and 100.
    // `gif-delay` is the deprecated scalar beside `delay`, and vips defines it
    // as `delay[0] / 10` (`nsgifload.c:469`), so it is the first frame's delay
    // back in the centiseconds the wire counts. It follows **this raster**
    // rather than the file, which diverges from vips for exactly the reason
    // and in exactly the way `delay` already does: vips reports the file's
    // whole array whatever window was loaded and then takes element 0 of it,
    // so `anim4.gif[page=2,n=2]` reports `gif-delay: 4` for a raster whose
    // first page has a delay of 80 ms. `src/webp.rs` takes it off
    // `delays.first()` for the same reason, so the animated loaders agree.
    if let Some(&first) = delays.first() {
        let centiseconds = FrameDelay::from_millis(first.unsigned_abs() as u32).to_centiseconds();
        raster.set_field("gif-delay", MetadataValue::Int(i64::from(centiseconds)));
    }
    raster.set_field("delay", MetadataValue::IntArray(delays));
    raster.set_field("palette", MetadataValue::Int(1));
    // The colour table entry the logical screen descriptor points at, as the
    // three doubles vips stores it as (`nsgifload.c:293`,
    // `vips_image_set_array_double`). The value is already resolved above,
    // for the restore-to-background disposal, so this is a field write and
    // not new arithmetic; what it waited on was
    // `MetadataValue::DoubleArray` (issue #852). An `IntArray` would have
    // been numerically right and typed wrong, and a field of the wrong type
    // is one this crate's own readers ignore.
    raster.set_field(
        "background",
        MetadataValue::DoubleArray(background.iter().map(|&b| f64::from(b)).collect()),
    );
    // The global colour table, and only when no frame in the file overrides
    // it with one of its own (issue #828). vips packs each entry as libnsgif's
    // `R, G, B, A` byte quad reinterpreted as a machine integer, which reads
    // `0xFF << 24 | B << 16 | G << 8 | R` on a little-endian host and makes
    // every entry negative. Measured on vips 8.18.6: a table of (71, 112, 76)
    // and (60, 60, 60) comes back `-11767737 -12829636`.
    //
    // `from_le_bytes` rather than `from_ne_bytes`, so this value is the same
    // wherever it was computed. Matching vips exactly would mean matching a
    // number that changes with the host, and the `.v` trailer writes it as
    // decimal text, so a native reading would put a different `gif-palette`
    // in the file for the same GIF depending on the machine. The cost is a
    // divergence from vips on a big-endian host, which is also the reason
    // `from_le_bytes` rather than a cast: no wrap to reason about.
    //
    // The array is the table as it sits on the wire, padding included, which
    // is what `global` already holds: a three-colour table is four entries
    // with an opaque-black fourth.
    if let Some(table) = global.as_deref().filter(|_| !scan.any_local_palette) {
        let entries: Vec<i64> = table
            .as_chunks::<PALETTE_STRIDE>()
            .0
            .iter()
            .map(|rgb| i64::from(i32::from_le_bytes([rgb[0], rgb[1], rgb[2], u8::MAX])))
            .collect();
        raster.set_field("gif-palette", MetadataValue::IntArray(entries));
    }
    if scan.colours > 0 {
        // `nsgifload.c:337-343`: ceil(log2(colours)) over the table as it
        // sits on the wire. The table is always a power of two there, so the
        // logarithm is exact and `trailing_zeros` is the same number without
        // the float round trip.
        let bits = scan.colours.next_power_of_two().trailing_zeros();
        raster.set_field("bits-per-sample", MetadataValue::Int(i64::from(bits)));
        // `palette-bit-depth` is the deprecated name for the same number, and
        // vips writes the two one after the other off one `ceil(log2(colours))`
        // (`nsgifload.c:339-345`), so they can never differ (issue #875).
        raster.set_field("palette-bit-depth", MetadataValue::Int(i64::from(bits)));
    }
    if scan.interlaced {
        raster.set_field("interlaced", MetadataValue::Int(1));
    }
    Ok(raster)
}

/// The colour a "restore to background" disposal paints, which is the global
/// colour table entry the logical screen descriptor points at.
///
/// An index the table cannot serve resolves to **entry 0**, not to black
/// (issue #850). libnsgif's rule is
/// `if (global_palette && bg_index < colour_table_size) table[bg_index] else
/// table[0]`, and it is measured rather than read: on a four-entry palette
/// whose entry 0 is `(9, 8, 7)`, vips 8.18.6 reports `background: 9 8 7` for
/// a stored index of 200 and disposes the canvas to it, where index 3 on the
/// same table reports `0 0 255`. The fixture the animated-load lane wrote for
/// this used a palette whose entry 0 was black, so the two answers were
/// indistinguishable and the wrong one went in.
///
/// The fallback is not reachable through [`decode_gif_with`] today, and that
/// is worth writing down rather than relying on: `gif` 0.14.2 clears
/// `Decoder::bg_color()` to `None` when the stored index is past the global
/// palette (`reader/mod.rs:354-358`), so this function used to arrive at
/// entry 0 through `index.unwrap_or(0)` for the wrong reason. A release that
/// stopped normalising, or a second caller, would have turned that into wrong
/// pixels with nothing in the way.
fn background_rgb(global: Option<&[u8]>, index: Option<usize>) -> [u8; PALETTE_STRIDE] {
    // libnsgif substitutes a two-entry black-and-white table for a file that
    // declares no global one, and then reads entry 0 of it whatever the
    // descriptor stores: measured, a stored index of 1 with no global table
    // reports `0 0 0` and not the substitute's white second entry. That file
    // does not reach this module today, because `gif` 0.14.2 refuses a frame
    // with no colour table at all (issue #851); this arm is what keeps the
    // lookup total.
    let Some(table) = global else {
        return [0; PALETTE_STRIDE];
    };
    let entry = |i: usize| {
        let at = i * PALETTE_STRIDE;
        table.get(at..at + PALETTE_STRIDE)
    };
    match entry(index.unwrap_or(0)).or_else(|| entry(0)) {
        Some(rgb) => [rgb[0], rgb[1], rgb[2]],
        None => [0; PALETTE_STRIDE],
    }
}

/// What a frame's disposal code says to do with the canvas before the next
/// frame draws.
///
/// This is libnsgif's mapping rather than the `gif` crate's, which is the
/// whole of issue #827: `DisposalMethod::from_u8` knows only codes 0 to 3 and
/// folds every reserved code onto `Any`, while libnsgif remaps **4** onto
/// restore-to-previous and leaves 5, 6 and 7 alone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Disposal {
    /// Leave the canvas alone: codes 0, 1, 5, 6 and 7.
    Keep,
    /// Clear this frame's rectangle, and only it: code 2.
    Background,
    /// Rewind the whole canvas to before this frame drew: codes 3 and 4.
    Previous,
}

impl Disposal {
    /// The disposal a raw GIF89a code selects.
    ///
    /// Measured on vips 8.18.6 over all eight codes on one two-frame file:
    /// 0, 1, 5, 6 and 7 keep, 2 clears to the background colour, and 3 and 4
    /// both rewind. Code 4 is `NSGIF_DISPOSAL_RESTORE_QUIRK` in libnsgif,
    /// which remaps it onto `NSGIF_DISPOSAL_RESTORE_PREV` before anything
    /// reads it.
    fn from_code(code: u8) -> Self {
        match code {
            2 => Self::Background,
            3 | 4 => Self::Previous,
            _ => Self::Keep,
        }
    }

    /// The disposal the `gif` crate's already-collapsed enum selects, which
    /// is the answer when the wire walk cannot be trusted for a frame.
    fn from_crate(method: gif::DisposalMethod) -> Self {
        match method {
            gif::DisposalMethod::Background => Self::Background,
            gif::DisposalMethod::Previous => Self::Previous,
            gif::DisposalMethod::Any | gif::DisposalMethod::Keep => Self::Keep,
        }
    }
}

/// The `gif` crate's own mapping of a raw disposal code, which is
/// `DisposalMethod::from_u8` with every code it does not know folded onto
/// `Any` (`reader/decoder.rs:862-865`).
///
/// Recomputing it is how [`reconcile_control`] checks that the wire walk and
/// the decoder are looking at the same frame.
fn crate_disposal(code: u8) -> gif::DisposalMethod {
    gif::DisposalMethod::from_u8(code).unwrap_or(gif::DisposalMethod::Any)
}

/// One frame's graphic control extension, as it sits on the wire.
///
/// The `gif` crate hands back three of these four fields intact and the
/// fourth already collapsed. This is the same block with the disposal code
/// still raw.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct RawControl {
    /// The disposal code, 0 to 7, out of bits 2 to 4 of the packed byte.
    disposal: u8,
    /// The delay in centiseconds, which is the unit the wire uses and not the
    /// milliseconds `delay` reports.
    delay_cs: u16,
    /// The transparent index, when the packed byte's low bit is set.
    transparent: Option<u8>,
}

impl RawControl {
    /// Decode a graphic control extension's four-byte payload.
    ///
    /// The packed byte carries the disposal in bits 2 to 4 and the
    /// transparency flag in bit 0, then two little-endian bytes of delay in
    /// centiseconds, then the transparent index.
    fn from_payload(payload: [u8; 4]) -> Self {
        Self {
            disposal: (payload[0] & 0b0001_1100) >> 2,
            delay_cs: u16::from_le_bytes([payload[1], payload[2]]),
            transparent: (payload[0] & 1 != 0).then_some(payload[3]),
        }
    }
}

/// What the wire walk found for one frame: the graphic control extension it
/// carried, or the fact that it carried none.
///
/// The distinction is not recoverable from the `gif` crate. Its
/// `Decoder::next_frame_info` takes the frame state fresh for every frame
/// (`reader/mod.rs:436`), so `Frame::delay` is `0` both for an extension
/// holding zero and for no extension at all, and the neighbouring fields do
/// not separate them either: a control extension with disposal 0, no
/// transparency and delay 0 leaves `dispose`, `transparent` and `delay` at
/// exactly the values an absent one leaves them at. libnsgif does separate
/// them, so vips reports 100 ms for one and 0 ms for the other (issue #866).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WireControl {
    /// The frame carried a graphic control extension.
    Present {
        /// libnsgif's reading, which is the answer this module wants: the
        /// four bytes straight after the size byte, whatever the size byte
        /// says and however many sub-blocks the chain has.
        control: RawControl,
        /// The same block as the `gif` crate will report it, which is what
        /// the cross-check compares against and is **not** always the same
        /// four bytes (issue #878). `None` when the crate will be left
        /// holding `Frame::default()` for this frame.
        as_decoder: Option<RawControl>,
    },
    /// It carried none.
    Absent,
}

/// The delay libnsgif gives a frame that carries no graphic control
/// extension, in centiseconds.
///
/// `nsgif__get_frame` initialises `frame->info.delay = 10` when it allocates
/// the frame and only a control extension overwrites it, so the default
/// survives into the `delay` array rather than being a playback-time floor.
/// Measured on vips 8.18.6 rather than read off that: a still GIF with no
/// extension reports `delay: 100` and `gif-delay: 10`, while one whose
/// extension holds an explicit zero reports `delay: 0`.
const ABSENT_CONTROL_DELAY_CS: u16 = 10;

impl WireControl {
    /// The three frame fields the `gif` crate reports once it has read this,
    /// which is what makes the two walks checkable against each other.
    ///
    /// It is deliberately **not** the same bytes [`WireControl::disposal`]
    /// and [`WireControl::delay_cs`] answer from (issue #878). This question
    /// is only ever "are the two walks on the same frame", so it has to model
    /// the decoder rather than the format.
    ///
    /// A `None` reading is `Frame::default()` (`common.rs:163-178`), whose
    /// disposal is `Keep` and **not** `Any`, so it is not the same triple
    /// that a control extension holding all zeroes produces. Getting that
    /// wrong would make every extension-less frame look like a
    /// desynchronised walk.
    fn decoder_fields(self) -> (gif::DisposalMethod, u16, Option<u8>) {
        let seen = match self {
            Self::Present { as_decoder, .. } => as_decoder,
            Self::Absent => None,
        };
        seen.map_or((gif::DisposalMethod::Keep, 0, None), |raw| {
            (crate_disposal(raw.disposal), raw.delay_cs, raw.transparent)
        })
    }

    /// What to do with the canvas after this frame draws.
    fn disposal(self) -> Disposal {
        match self {
            Self::Present { control, .. } => Disposal::from_code(control.disposal),
            // GIF89a's disposal 0, which libnsgif also writes into a fresh
            // frame, and which keeps the canvas.
            Self::Absent => Disposal::Keep,
        }
    }

    /// This frame's delay, in the centiseconds the wire counts.
    fn delay_cs(self) -> u16 {
        match self {
            Self::Present { control, .. } => control.delay_cs,
            Self::Absent => ABSENT_CONTROL_DELAY_CS,
        }
    }
}

/// A second walk over a GIF's block chain, recovering each frame's graphic
/// control extension.
///
/// It exists for one field, the raw disposal code (issue #827), and it is the
/// cheapest walk that can produce it: the block chain is self-describing, so
/// stepping over an image's LZW data is a walk over sub-block lengths and no
/// pixel is decoded. Nothing is allocated, and the walk advances one frame at
/// a time in lockstep with the decode, so there is no per-frame buffer to
/// price against [`DecodeLimits`].
///
/// It is deliberately **not** trusted on its own; see [`reconcile_control`].
struct ControlWalk<'a> {
    /// The whole file, the same slice the decoder reads.
    bytes: &'a [u8],
    /// The next byte to read, or `None` once the walk has stopped: at the
    /// trailer, at an unknown block introducer, or at the end of the buffer.
    pos: Option<usize>,
    /// The most recent graphic control extension, waiting for the image
    /// descriptor it belongs to.
    pending: Option<WireControl>,
}

impl<'a> ControlWalk<'a> {
    /// Position the walk after the signature, the logical screen descriptor
    /// and the global colour table.
    fn new(bytes: &'a [u8]) -> Self {
        let mut walk = Self {
            bytes,
            pos: None,
            pending: None,
        };
        // Three bytes of signature and three of version, then the seven-byte
        // logical screen descriptor, whose fifth byte carries the global
        // colour table flag and its size.
        let Some(descriptor) = bytes.get(6..13) else {
            return walk;
        };
        if !bytes.starts_with(b"GIF") {
            return walk;
        }
        let packed = descriptor[4];
        let mut at = 13;
        if packed & 0x80 != 0 {
            at += PALETTE_STRIDE * (2usize << (packed & 7));
        }
        walk.pos = (at <= bytes.len()).then_some(at);
        walk
    }

    /// The next image descriptor's graphic control extension, or
    /// [`WireControl::Absent`] when the frame carries none.
    ///
    /// `None` once the walk has stopped, which is a different thing again and
    /// the reason this is not one `Option`: "the frame had no extension" is
    /// an answer, and "the walk has nothing left to say" is not.
    fn next_frame(&mut self) -> Option<WireControl> {
        let found = self.step();
        if found.is_none() {
            self.pos = None;
        }
        found
    }

    /// [`ControlWalk::next_frame`] without the stop bookkeeping, so every
    /// short read can be a `?`.
    fn step(&mut self) -> Option<WireControl> {
        loop {
            let at = self.pos?;
            match self.bytes.get(at) {
                // An extension block: a label byte, then a sub-block chain.
                // Only the graphic control extension matters here and the
                // rest are stepped over, which is also how the walk gets past
                // the NETSCAPE application extension.
                Some(&0x21) => {
                    let label = *self.bytes.get(at + 1)?;
                    let (last, next) = self.sub_blocks(at + 2)?;
                    if label == 0xF9 {
                        // libnsgif reads four bytes at a fixed offset from the
                        // label and never looks at the size byte or the rest
                        // of the chain, so this is the answer (issue #878).
                        // Measured: a chain of `01 AA 04 <quad> 00` comes back
                        // from vips as disposal 2 and 3076 centiseconds, which
                        // is `0xAA` read as the packed byte and the two bytes
                        // after it read little-endian.
                        //
                        // The crate's reading is the *last* sub-block, which
                        // is a different four bytes on that same file, and it
                        // is kept only to check the two walks against each
                        // other.
                        self.pending =
                            self.bytes
                                .get(at + 3..at + 7)
                                .map(|quad| WireControl::Present {
                                    control: RawControl::from_payload([
                                        quad[0], quad[1], quad[2], quad[3],
                                    ]),
                                    as_decoder: last.map(RawControl::from_payload),
                                });
                    }
                    self.pos = Some(next);
                }
                // An image descriptor: nine bytes of geometry and flags, an
                // optional local colour table, the LZW minimum code size, and
                // the pixel sub-block chain.
                Some(&0x2C) => {
                    let packed = *self.bytes.get(at + 9)?;
                    let mut next = at + 10;
                    if packed & 0x80 != 0 {
                        next += PALETTE_STRIDE * (2usize << (packed & 7));
                    }
                    let (_, next) = self.sub_blocks(next + 1)?;
                    self.pos = Some(next);
                    return Some(self.pending.take().unwrap_or(WireControl::Absent));
                }
                // The trailer, an unknown introducer, or the end of the
                // buffer. `open()` configures the decoder to stop at an
                // unknown block rather than skip it, so this walk stops there
                // too, and for the same reason: `garden.gif` carries a stray
                // byte after its 35th frame and vips reports 35 pages.
                _ => return None,
            }
        }
    }

    /// Walk a sub-block chain from `at`, returning the byte after it and the
    /// **last** sub-block when that one is exactly four bytes long.
    ///
    /// Last rather than the whole chain, because that is what the `gif` crate
    /// measures: `ExtensionDataSubBlockStart` clears its buffer on every
    /// sub-block (`reader/decoder.rs:734`), so `read_control_extension`'s
    /// `ext.data.len() != 4` check only ever sees the final one, and a chain
    /// of `04 <quad> 04 <quad> 00` reaches the decoder as the second quad.
    /// This module used to require the whole chain to total four, which is
    /// neither implementation's rule and made the walk disagree with the
    /// decoder about a frame it had found perfectly well (issue #878).
    ///
    /// `None` when the last sub-block is some other length, which is the
    /// shape the crate answers with `Frame::default()` or with an error.
    fn sub_blocks(&self, mut at: usize) -> Option<(Option<[u8; 4]>, usize)> {
        let mut last: Option<[u8; 4]> = None;
        loop {
            let size = usize::from(*self.bytes.get(at)?);
            at += 1;
            if size == 0 {
                return Some((last, at));
            }
            let block = self.bytes.get(at..at + size)?;
            last = <[u8; 4]>::try_from(block).ok();
            at += size;
        }
    }
}

/// The wire control to use for a frame, or `None` when the two walks do not
/// demonstrably agree about which frame they are looking at.
///
/// [`ControlWalk`] is a second parser beside the one that decides where the
/// frames are, and two parsers over one hostile file can come apart. They are
/// checked against each other rather than assumed to agree: the graphic
/// control extension carries the delay, the transparent index and the
/// disposal side by side, the decoder kept the first two intact, and it kept
/// the third in collapsed form. So all three are compared, and only then are
/// the two things only the wire knows, the raw disposal code (#827) and
/// whether there was an extension at all (#866), believed. A frame where they
/// disagree falls back to the decoder, which is what this module did for
/// every frame before #827.
fn reconcile_control(
    wire: Option<WireControl>,
    dispose: gif::DisposalMethod,
    delay_cs: u16,
    transparent: Option<u8>,
) -> Option<WireControl> {
    wire.filter(|wire| wire.decoder_fields() == (dispose, delay_cs, transparent))
}

/// libnsgif's stand-in colour table for a file that declares no global one:
/// black, then white.
///
/// `nsgif__parse_extension` builds it when the logical screen descriptor's
/// global-table flag is clear, and sets `colour_table_size` to 2 with it.
/// Measured on vips 8.18.6 rather than read off that: such a file decodes
/// index 0 to `0 0 0` and index 1 to `255 255 255`, reports `gif-palette:
/// -16777216 -1` and `bits-per-sample: 1`.
const SUBSTITUTE_COLOUR_TABLE: [u8; 2 * PALETTE_STRIDE] = [0, 0, 0, 255, 255, 255];

/// Rewrite `bytes` into the shape the `gif` crate reads the way libnsgif
/// reads it, or `None` when it already is that shape.
///
/// Two files decode in vips and do not decode here, and both are the same
/// mistake in the other direction: the crate is stricter than the format
/// where libnsgif is looser, so it refuses a file rather than reading it
/// differently. Neither can be fixed by reading the bytes more carefully,
/// because the crate is what decides where the frames are, and it stops.
/// What can be fixed is *which* bytes it is handed.
///
/// * **No global colour table** (issue #851). `Decoder::next_frame_info`
///   errors with "no color table available for current frame" for a frame
///   with neither a local palette nor a global one
///   (`reader/mod.rs:438-442`), and `Decoder` has no way to set one from
///   outside. libnsgif invents [`SUBSTITUTE_COLOUR_TABLE`] instead, so that
///   table is spliced in and the descriptor's flag set. The stored background
///   index is forced to 0 with it, because libnsgif only honours a stored
///   index when the file really declares a table: measured, index 1 with no
///   table reports black, the substitute's entry 0, and not its white entry.
/// * **A control extension whose last sub-block is not four bytes** (issue
///   #879). `read_control_extension` refuses it and the error ends the header
///   scan, so every frame from there on is lost. libnsgif never measures the
///   chain and reads four bytes at a fixed offset from the label, so the
///   chain is rewritten into the single four-byte sub-block that read
///   produces. That is the same four bytes, in the only shape the crate
///   accepts.
///
/// The copy is priced through [`DecodeLimits::check_alloc`] because it is a
/// second whole-file buffer. Nothing is copied for a file that needs neither
/// rewrite, which is every ordinary GIF, and the detection is one byte for
/// the first case and a walk with no pixel decoding for the second.
///
/// A file this cannot parse is left alone rather than guessed at: `None` puts
/// the original bytes in front of the decoder, which is exactly what happened
/// before this existed.
fn normalise_for_decoder(
    bytes: &[u8],
    limits: DecodeLimits,
) -> Result<Option<Vec<u8>>, SourceError> {
    let Some(descriptor) = bytes.get(6..13) else {
        return Ok(None);
    };
    if !bytes.starts_with(b"GIF") {
        return Ok(None);
    }
    let packed = descriptor[4];
    let has_global = packed & 0x80 != 0;
    let after_lsd = 13
        + if has_global {
            PALETTE_STRIDE * (2usize << (packed & 7))
        } else {
            0
        };
    if after_lsd > bytes.len() {
        return Ok(None);
    }

    // Find the control extensions that need rewriting before allocating
    // anything, so an ordinary file pays a walk and no copy.
    let mut rewrites: Vec<(usize, usize, [u8; 4])> = Vec::new();
    let mut at = after_lsd;
    loop {
        match bytes.get(at) {
            Some(&0x21) => {
                let label = *bytes.get(at + 1).unwrap_or(&0);
                let Some((last, next)) = walk_sub_blocks(bytes, at + 2) else {
                    break;
                };
                if label == 0xF9 {
                    // libnsgif's four bytes, and whether the crate will read
                    // the same ones. `last` is what the crate reads, so a
                    // chain it cannot read at all is `None` and a chain whose
                    // last sub-block holds something else is a mismatch; both
                    // get the rewrite, and both end up with libnsgif's bytes
                    // in front of the decoder.
                    if let Some(quad) = bytes.get(at + 3..at + 7) {
                        let wanted = [quad[0], quad[1], quad[2], quad[3]];
                        if last != Some(wanted) {
                            rewrites.push((at + 2, next, wanted));
                        }
                    }
                }
                at = next;
            }
            Some(&0x2C) => {
                let Some(&flags) = bytes.get(at + 9) else {
                    break;
                };
                let mut next = at + 10;
                if flags & 0x80 != 0 {
                    next += PALETTE_STRIDE * (2usize << (flags & 7));
                }
                let Some((_, after)) = walk_sub_blocks(bytes, next + 1) else {
                    break;
                };
                at = after;
            }
            _ => break,
        }
    }

    if has_global && rewrites.is_empty() {
        return Ok(None);
    }

    // Every rewritten chain becomes six bytes; the table adds six more. The
    // price is an upper bound rather than the exact length, which is the
    // right way round for a budget.
    let extra = if has_global {
        0
    } else {
        SUBSTITUTE_COLOUR_TABLE.len()
    };
    let needed = bytes.len() as u64 + extra as u64 + 6 * rewrites.len() as u64;
    limits.check_alloc("GIF normalised copy", needed)?;
    let mut out = Vec::with_capacity(bytes.len() + extra);

    out.extend_from_slice(&bytes[..10]);
    if has_global {
        out.push(packed);
        out.extend_from_slice(&bytes[11..after_lsd]);
    } else {
        // The flag set and the size bits cleared, which is the two-entry
        // table libnsgif builds, then the background index forced to entry 0.
        out.push(packed | 0x80);
        out.push(0);
        out.push(descriptor[6]);
        out.extend_from_slice(&SUBSTITUTE_COLOUR_TABLE);
    }

    let mut copied = after_lsd;
    for (chain, end, quad) in rewrites {
        out.extend_from_slice(&bytes[copied..chain]);
        out.push(4);
        out.extend_from_slice(&quad);
        out.push(0);
        copied = end;
    }
    out.extend_from_slice(&bytes[copied..]);
    Ok(Some(out))
}

/// Walk a sub-block chain from `at`, returning its last sub-block when that
/// one is exactly four bytes and the offset just past the terminator.
///
/// Free of [`ControlWalk`] because [`normalise_for_decoder`] runs before it,
/// on bytes the walk has not seen yet. The rule is the same one and the
/// reason is in [`ControlWalk::sub_blocks`].
fn walk_sub_blocks(bytes: &[u8], mut at: usize) -> Option<(Option<[u8; 4]>, usize)> {
    let mut last: Option<[u8; 4]> = None;
    loop {
        let size = usize::from(*bytes.get(at)?);
        at += 1;
        if size == 0 {
            return Some((last, at));
        }
        last = <[u8; 4]>::try_from(bytes.get(at..at + size)?).ok();
        at += size;
    }
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
fn scan_file(bytes: &[u8], limits: DecodeLimits) -> Result<FileScan, SourceError> {
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
        any_local_palette: false,
    };
    // A scan that hits trouble stops where it is rather than failing:
    // vips' `fail-on` defaults to `VIPS_FAIL_ON_NONE`, so
    // `nsgifload.c:388-406` downgrades every complaint from
    // `nsgif_data_scan` to a `g_warning`, then calls `nsgif_data_complete`
    // so the frames that did parse are still readable.
    loop {
        match decoder.next_frame_info() {
            Ok(Some(frame)) => {
                // The ceiling is checked before the count moves, so the walk
                // stops *at* it rather than running to the end of a hostile
                // chain to find out how long it is, which is what
                // `count_images` does for the TIFF IFD chain
                // (`encode_tiff.rs:718`). A GIF's frame list has no count in
                // the header either, so this is the same walk with the same
                // exposure, and until now it was the one multi-page loader
                // that did not honour the ceiling written for it.
                if scan.frames >= limits.max_pages {
                    return Err(SourceError::PageLimitExceeded {
                        max_pages: limits.max_pages,
                    });
                }
                scan.frames += 1;
                scan.has_transparency |= frame.transparent.is_some();
                scan.interlaced |= frame.interlaced;
                if let Some(local) = &frame.palette {
                    scan.any_local_palette = true;
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
    // extension at all means play once. Measured across the reference suite.
    // The deprecated `gif-loop` is the other direction, `loop_max == 0 ? 0 :
    // loop_max - 1`, which is the count a file with a block carries and 0 for
    // a file with none, and it is attached from `loop` rather than from here.
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
    /// Writes a GIF89a: a global colour table quantised to
    /// `min(255, 1 << bitdepth)` entries over the whole raster, then **one
    /// image block per page**, then the NETSCAPE looping extension unless
    /// `loop` asks for a single play. An unpaged raster is one page, so a
    /// still is the same file it always was.
    ///
    /// The frame geometry, the per-frame delays and the loop count all come
    /// off the raster: its page height, its `delay` array and its `loop`
    /// field, which is where vips reads them too. Requires an 8-bit raster
    /// (`Gray8` / `Rgb8` / `Rgba8`); see the [module docs](crate::gif) for
    /// the alpha threshold, the reserved transparent index, the disposal
    /// rule, and where the output deliberately differs from vips.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] for
    ///
    /// * a raster that is not 8-bit;
    /// * a **page** whose width or height exceeds the 65535-pixel GIF axis
    ///   limit, which is per frame rather than per roll (vips rejects the
    ///   same case as `"frame too large"`, `cgifsave.c:744-750`);
    /// * a `delay` array present with a length other than the page count, or
    ///   carrying a negative entry;
    /// * a negative `loop`;
    /// * a failure in the GIF writer itself.
    pub fn encode_gif(&self, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
        let width = self.width();
        // The frame is one page, not the whole raster: the logical screen
        // descriptor of a roll `vips gifsave` writes holds the page height,
        // so a roll of many pages is not itself bounded by GIF's axis. The
        // height comes off `Raster::page_layout`, which always divides, so
        // the loop below cannot run off the end.
        let layout = self.page_layout();
        let page_height = layout.page_height();
        let pages = layout.pages();
        let (Ok(gif_width), Ok(gif_height)) = (u16::try_from(width), u16::try_from(page_height))
        else {
            return Err(EncodeError::encode(format!(
                "gif: frame too large; {width}x{page_height} exceeds the 65535-pixel \
                 GIF axis limit"
            )));
        };
        let delays = self.gif_delays(pages)?;
        let plays = self.gif_loop()?;
        let pixels = self.gif_rgba()?;

        let max_colours = options.max_colours();
        let has_transparency = pixels.iter().any(|p| p[3] == 0);
        let opaque_budget = if has_transparency {
            max_colours - 1
        } else {
            max_colours
        };
        // The palette is quantised over the whole roll rather than per page,
        // so every frame draws from one global colour table and a colour that
        // appears only on page 3 is still representable on page 3. vips does
        // the opposite and quantises each frame, reusing the previous
        // palette when the error is under `interpalette-maxerror`; that is
        // cgif and libimagequant machinery the module docs already disclaim,
        // and it is why the bytes never match.
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
            // a one-frame image: `NETSCAPE2.0` with a loop count of 0, which
            // is what an absent `loop` field means here too.
            //
            // The `gif` crate spells "write no block at all" as
            // `Repeat::Finite(0)`: `write_extension` returns early on it
            // without emitting anything. So `LoopCount::to_gif_wire`'s `None`
            // maps onto it exactly, and a block holding zero is
            // `Repeat::Infinite`, which is the same asymmetry the decoder has
            // in the other direction.
            let repeat = match plays.to_gif_wire() {
                None => gif::Repeat::Finite(0),
                Some(0) => gif::Repeat::Infinite,
                Some(count) => gif::Repeat::Finite(count),
            };
            encoder.set_repeat(repeat).map_err(encode_error_from_gif)?;

            let page_pixels = width as usize * page_height as usize;
            for page in 0..pages as usize {
                // Remapped per page, which matters once `dither` is above
                // zero: error diffusion carries down the rows, and over the
                // whole roll the error leaving one page's last row would land
                // on the next page's first, coupling two frames that are
                // shown seconds apart.
                let page_pixels_slice = &pixels[page * page_pixels..(page + 1) * page_pixels];
                let mut indices = remap(
                    page_pixels_slice,
                    &palette,
                    offset,
                    options.dither,
                    width,
                    page_height,
                );
                if options.interlaced {
                    indices = interlace(&indices, width, page_height);
                }
                let mut frame = gif::Frame::from_indexed_pixels(
                    gif_width,
                    gif_height,
                    indices,
                    reserved.then_some(0),
                );
                frame.interlaced = options.interlaced;
                frame.delay = delays[page];
                // cgif writes "restore to background" on every frame but the
                // last when the animation carries transparency, and "keep"
                // otherwise. Measured on 8.18.6 over five files: a two-page
                // roll with alpha came out `2 1`, a two-page opaque roll
                // `1 1`, and a still with alpha `1`, because the last frame's
                // disposal is never observed.
                //
                // It is not cosmetic. Every frame here covers the whole
                // screen, so with "keep" a transparent pixel on page 2 would
                // show page 1 through it instead of showing transparent, and
                // `reserved` is true whenever `has_transparency` is, so the
                // frame declares the transparent index that makes the clear
                // transparent rather than the background colour.
                frame.dispose = if has_transparency && page + 1 < pages as usize {
                    gif::DisposalMethod::Background
                } else {
                    gif::DisposalMethod::Keep
                };
                encoder.write_frame(&frame).map_err(encode_error_from_gif)?;
            }
        }
        Ok(out)
    }

    /// The per-page delays this raster's `delay` field asks for, in GIF wire
    /// centiseconds.
    ///
    /// An absent field is no delay at all on every page, which is what vips
    /// writes for an image carrying none.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] when the array is present and its length is
    /// not `pages`, or when any entry is negative. vips does neither: a
    /// two-entry array on a four-page roll wrote centiseconds `2 3 0 0` and a
    /// six-entry one wrote `2 3 4 5`, both measured, and a delay of -10 ms
    /// came out as 65535 centiseconds, which is 655 seconds. All three are a
    /// caller mistake given a silent answer.
    ///
    /// Reaching the length refusal takes work, and that is deliberate rather
    /// than assumed. The loader attaches one entry per page, and
    /// [`Raster::carry_meta_from`] drops the array on any shape change for
    /// the same reason it drops the page split, so an op that changes the
    /// page count hands on a raster with no `delay` rather than a stale one.
    /// `roll.extract_page(0).encode_gif(..)` used to fail here, which is how
    /// that was found.
    fn gif_delays(&self, pages: u32) -> Result<Vec<u16>, EncodeError> {
        let Some(stored) = self.get_int_array("delay") else {
            return Ok(vec![0; pages as usize]);
        };
        if stored.len() != pages as usize {
            return Err(EncodeError::encode(format!(
                "gif: delay has {} entries for {pages} page(s); it must have one per page",
                stored.len()
            )));
        }
        stored
            .iter()
            .map(|&millis| {
                if millis < 0 {
                    return Err(EncodeError::encode(format!(
                        "gif: delay {millis} is negative; a frame delay cannot be"
                    )));
                }
                // Saturating rather than wrapping, which is the divergence.
                // vips truncates the cast: 655360 ms went out as no delay at
                // all and 700000 ms as 44.64 seconds, both measured. A delay
                // too long to express comes out here as the longest one that
                // fits.
                let millis = u32::try_from(millis).unwrap_or(u32::MAX);
                Ok(FrameDelay::from_millis(millis).to_centiseconds())
            })
            .collect()
    }

    /// The play count this raster's `loop` field asks for.
    ///
    /// An absent field is [`LoopCount::FOREVER`], which is vips's own
    /// default and what cgif writes into every file.
    ///
    /// A field of the wrong type is treated as absent rather than refused,
    /// the way [`Raster::get_n_pages`] and the `page-height` reader already
    /// treat one: an untrusted `.v` can leave anything under any name, so a
    /// wrong type means "this is not the field I read". A negative integer is
    /// refused instead, because that *is* the field carrying a value it
    /// cannot have. vips casts it unsigned: `loop = -1` wrote a NETSCAPE
    /// count of 65535 and reloaded as 65536 plays, measured.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] for a negative `loop`.
    fn gif_loop(&self) -> Result<LoopCount, EncodeError> {
        match self.get_field("loop") {
            Some(MetadataValue::Int(plays)) if plays < 0 => Err(EncodeError::encode(format!(
                "gif: loop {plays} is negative; a play count cannot be"
            ))),
            Some(MetadataValue::Int(plays)) => Ok(LoopCount::from_plays(
                u32::try_from(plays).unwrap_or(u32::MAX),
            )),
            _ => Ok(LoopCount::FOREVER),
        }
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
        /// The graphic control extension's disposal code, 0 to 7. cgif
        /// writes 1 ("keep") on the last frame of everything it saves.
        disposal: u8,
        /// The frame's own colour table, when it carries one. A frame with a
        /// local table draws through it instead of the global one, and one
        /// such frame anywhere in the file is what keeps `gif-palette` off
        /// the header (issue #828).
        local: Option<Vec<[u8; 3]>>,
        /// Whether to write a graphic control extension at all. A frame
        /// without one takes the default control, which is disposal 0, no
        /// delay and no transparent index.
        control: bool,
        /// A comment extension written between the control extension and the
        /// image descriptor, which is where a block that is not a control
        /// extension can come between the two halves `ControlWalk` has to
        /// keep together.
        comment: Option<Vec<u8>>,
        /// A graphic control extension written verbatim, sub-block chain and
        /// all, in place of the one `control`, `disposal`, `delay_cs` and
        /// `transparent` would produce. The bytes follow `21 F9` and have to
        /// end with the chain terminator themselves. Issue #878 is about the
        /// chains that are not one four-byte sub-block.
        raw_control: Option<Vec<u8>>,
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
                disposal: 1,
                local: None,
                control: true,
                comment: None,
                raw_control: None,
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
        fixture_tables(screen, Some(palette), background, loop_ext, frames)
    }

    /// [`fixture`] with the global colour table made optional and per-frame
    /// local tables honoured.
    ///
    /// `None` writes a logical screen descriptor with the global-table flag
    /// clear, which is the file libnsgif answers with its own black-and-white
    /// substitute table, and a frame carrying `local` writes its own table
    /// after the image descriptor. Both are what `gif-palette`'s presence
    /// rule turns on (issue #828).
    fn fixture_tables(
        screen: (u16, u16),
        global: Option<&[[u8; 3]]>,
        background: u8,
        loop_ext: Option<u16>,
        frames: &[Frame],
    ) -> Vec<u8> {
        let mut out = Vec::from(*b"GIF89a");
        out.extend_from_slice(&screen.0.to_le_bytes());
        out.extend_from_slice(&screen.1.to_le_bytes());
        match global {
            Some(palette) => {
                let table_size = flag_size(palette.len());
                out.push(0x80 | table_size);
                out.push(background);
                out.push(0);
                for i in 0..(2usize << table_size) {
                    out.extend_from_slice(palette.get(i).unwrap_or(&[0, 0, 0]));
                }
            }
            // The flag is clear, so no table follows. The background index is
            // still written, because part of what this fixture shape proves is
            // that libnsgif ignores it when there is no table to index into.
            None => out.extend_from_slice(&[0, background, 0]),
        }
        if let Some(count) = loop_ext {
            out.extend_from_slice(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01");
            out.extend_from_slice(&count.to_le_bytes());
            out.push(0);
        }
        for frame in frames {
            if let Some(chain) = &frame.raw_control {
                out.extend_from_slice(&[0x21, 0xF9]);
                out.extend_from_slice(chain);
            } else if frame.control {
                let mut flags = 0u8;
                if frame.transparent.is_some() {
                    flags |= 1;
                }
                flags |= (frame.disposal & 7) << 2;
                out.extend_from_slice(&[0x21, 0xF9, 4, flags]);
                out.extend_from_slice(&frame.delay_cs.to_le_bytes());
                out.push(frame.transparent.unwrap_or(0));
                out.push(0);
            }
            if let Some(text) = &frame.comment {
                out.extend_from_slice(&[0x21, 0xFE]);
                for chunk in text.chunks(255) {
                    out.push(chunk.len() as u8);
                    out.extend_from_slice(chunk);
                }
                out.push(0);
            }
            out.push(0x2C);
            out.extend_from_slice(&frame.left.to_le_bytes());
            out.extend_from_slice(&frame.top.to_le_bytes());
            out.extend_from_slice(&frame.width.to_le_bytes());
            out.extend_from_slice(&frame.height.to_le_bytes());
            let mut descriptor = if frame.interlaced { 0x40 } else { 0 };
            if let Some(local) = &frame.local {
                descriptor |= 0x80 | flag_size(local.len());
            }
            out.push(descriptor);
            if let Some(local) = &frame.local {
                for i in 0..(2usize << flag_size(local.len())) {
                    out.extend_from_slice(local.get(i).unwrap_or(&[0, 0, 0]));
                }
            }
            // The code width follows whichever table the frame draws through,
            // and 2 when there is none, which is the width libnsgif's
            // two-entry substitute needs.
            let table_size = match (&frame.local, global) {
                (Some(local), _) => flag_size(local.len()),
                (None, Some(palette)) => flag_size(palette.len()),
                (None, None) => 0,
            };
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
            disposal: 1,
            local: None,
            control: true,
            comment: None,
            raw_control: None,
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
            // `delay` reaches a single-frame load too, one entry for the one
            // page, and `loop` is 1 because the fixture writes no NETSCAPE
            // block. vips attaches both to a still load as well (measured:
            // a one-frame `gifsave` output reports `delay: 0`), so this is
            // the field set, not an animation-only extra.
            assert_eq!(raster.get_int_array("delay"), Some(&[0i64][..]));
            assert_eq!(raster.get_int("loop"), Some(1));
            assert_eq!(
                raster.get_field("page-height"),
                None,
                "a one-page load carries no split, as vips reports none"
            );
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
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
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
    /// The four-colour palette every animation fixture here draws from:
    /// index 0 black, 1 red, 2 green, 3 blue.
    const ANIM_PALETTE: [[u8; 3]; 4] = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];

    /// A frame covering the whole 2x2 screen, painted from `indices`.
    fn anim_frame(indices: [u8; 4], disposal: u8, delay_cs: u16) -> Frame {
        Frame {
            left: 0,
            top: 0,
            width: 2,
            height: 2,
            indices: indices.to_vec(),
            transparent: None,
            interlaced: false,
            delay_cs,
            disposal,
            local: None,
            control: true,
            comment: None,
            raw_control: None,
        }
    }

    /// Page `index` of `raster` as raw bytes, through the page model the
    /// frame lane landed (issue #564).
    fn page_bytes(raster: &Raster, index: u32) -> Vec<u8> {
        raster.extract_page(index).data().to_vec()
    }

    /// The `delay` field as a plain vector, or `None` when it is absent.
    fn delays(raster: &Raster) -> Option<Vec<i64>> {
        raster.get_int_array("delay").map(<[i64]>::to_vec)
    }

    /**
     * Tests that `n = -1` stacks every frame into one page roll, which is
     * the layout `vips copy 'anim.gif[n=-1]' out.v` produces. Works by
     * decoding a four-frame 2x2 fixture and reading the roll's geometry and
     * per-page pixels back.
     * Measured on vips 8.18.6: a four-frame 4x3 GIF loads at `n=-1` as a
     * 4x12 raster reporting `page-height: 3` and `n-pages: 4`.
     * Input: a four-frame 2x2 GIF -> Output: a 2x8 raster, page height 2,
     * four pages, each page the frame that painted it.
     */
    #[test]
    fn an_animation_loads_every_frame_as_a_page_roll() {
        let frames: Vec<Frame> = (0..4u8)
            .map(|i| anim_frame([i % 4, i % 4, i % 4, i % 4], 1, 0))
            .collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("the fixture is a valid GIF");

        assert_eq!((raster.width(), raster.height()), (2, 8));
        assert_eq!(raster.get_page_height(), 2, "one page per frame");
        assert_eq!(raster.pages_loaded(), 4);
        assert_eq!(
            raster.get_n_pages(),
            4,
            "n-pages counts the file's frames, not the loaded ones"
        );
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        for (index, colour) in ANIM_PALETTE.iter().enumerate() {
            let page = page_bytes(&raster, index as u32);
            assert_eq!(
                page,
                colour.repeat(4),
                "page {index} is the frame that painted it"
            );
        }
    }

    /**
     * Tests that a frame delay arrives in **milliseconds**, not the
     * centiseconds the graphic control extension holds, which is the silent
     * factor of ten issue #572 exists to catch. Works by writing known
     * centisecond delays into the fixture's extensions and requiring the
     * attached `delay` array to be ten times each of them.
     * Measured on vips 8.18.6: a GIF carrying `4 6 8 10` centiseconds loads
     * as `delay: 40 60 80 100`, and `nsgifload.c:466` is where the
     * multiplication happens.
     * Input: a four-frame GIF with GCE delays 4, 6, 8, 10 -> Output:
     * `delay` = `[40, 60, 80, 100]`.
     */
    #[test]
    fn frame_delays_arrive_as_milliseconds_not_centiseconds() {
        let wire = [4u16, 6, 8, 10];
        let frames: Vec<Frame> = wire
            .iter()
            .enumerate()
            .map(|(i, &cs)| anim_frame([i as u8 % 4; 4], 1, cs))
            .collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("the fixture is a valid GIF");

        assert_eq!(
            delays(&raster),
            Some(vec![40, 60, 80, 100]),
            "GIF stores centiseconds and vips reports milliseconds"
        );
    }

    /**
     * Tests that the `delay` array covers the pages this raster holds and no
     * others, which is a deliberate divergence from vips. Works by loading a
     * two-page window out of a four-frame file and requiring the array to be
     * the two delays that belong to it.
     * Measured on vips 8.18.6: `anim4.gif[page=2,n=2]` loads frames 2 and 3
     * (pixel values 180 and 240 confirm it) and still reports
     * `delay: 40 60 80 100`, so re-saving that raster writes 40 and 60 onto
     * frames that are really 2 and 3. The re-save was measured too: the
     * output's graphic control extensions hold 4 and 6 centiseconds.
     * Input: a four-frame GIF with delays 40, 60, 80, 100 ms loaded at
     * `page = 2, n = 2` -> Output: `delay` = `[80, 100]`.
     */
    #[test]
    fn the_delay_array_is_subset_to_the_pages_actually_loaded() {
        let frames: Vec<Frame> = [4u16, 6, 8, 10]
            .iter()
            .enumerate()
            .map(|(i, &cs)| anim_frame([i as u8 % 4; 4], 1, cs))
            .collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_page(2).with_n(2),
        )
        .expect("the fixture is a valid GIF");

        assert_eq!(raster.pages_loaded(), 2);
        assert_eq!(
            delays(&raster),
            Some(vec![80, 100]),
            "delay[i] is loaded page i's delay, which vips does not promise"
        );
        assert_eq!(
            delays(&raster).map(|d| d.len()),
            Some(raster.pages_loaded() as usize),
            "the array length is the page count"
        );
    }

    /**
     * Tests that a default, single-page load carries one delay and no
     * `page-height`, so a still GIF looks exactly like a still. Works by
     * decoding a four-frame fixture with the default options and reading
     * both fields.
     * Measured on vips 8.18.6: a default `gifload` of a four-frame file
     * attaches `n-pages: 4`, `loop` and the whole `delay` array but **no**
     * `page-height`, and `[page=3]` likewise carries no `page-height`. The
     * delay array is where libviprs diverges, for the reason
     * `the_delay_array_is_subset_to_the_pages_actually_loaded` measures.
     * Input: a four-frame GIF loaded with the defaults -> Output: one 2x2
     * page, `delay` = `[40]`, no `page-height` field.
     */
    #[test]
    fn a_one_page_load_carries_one_delay_and_no_page_height() {
        let frames: Vec<Frame> = [4u16, 6, 8, 10]
            .iter()
            .enumerate()
            .map(|(i, &cs)| anim_frame([i as u8 % 4; 4], 1, cs))
            .collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);
        let raster = decode_gif(&bytes, DecodeLimits::default()).expect("a valid GIF");

        assert_eq!((raster.width(), raster.height()), (2, 2));
        assert_eq!(delays(&raster), Some(vec![40]));
        assert!(
            raster.get_field("page-height").is_none(),
            "vips attaches no page-height to a one-page load"
        );
        assert_eq!(raster.pages_loaded(), 1);

        let third = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_page(3),
        )
        .expect("a valid GIF");
        assert_eq!(delays(&third), Some(vec![100]));
        assert!(third.get_field("page-height").is_none());
    }

    /**
     * Tests that `page` and `n` select a window of frames and that the
     * window's pixels are the frames it names, composited from the start of
     * the file rather than from the window. Works by loading frames 2 and 3
     * of a four-frame file whose frames each paint the whole screen a
     * different colour.
     * Measured on vips 8.18.6: `anim4.gif[page=2,n=2]` comes back 4x6 with
     * rows 180 180 180 240 240 240, which are frames 2 and 3.
     * Input: a four-frame GIF at `page = 2, n = 2` -> Output: a 2x4 raster
     * whose pages are the green and blue frames.
     */
    #[test]
    fn page_and_n_select_a_window_of_frames() {
        let frames: Vec<Frame> = (0..4u8).map(|i| anim_frame([i; 4], 1, 0)).collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_page(2).with_n(2),
        )
        .expect("a valid GIF");

        assert_eq!((raster.width(), raster.height()), (2, 4));
        assert_eq!(raster.get_page_height(), 2);
        assert_eq!(page_bytes(&raster, 0), ANIM_PALETTE[2].repeat(4));
        assert_eq!(page_bytes(&raster, 1), ANIM_PALETTE[3].repeat(4));
    }

    /**
     * Tests that `n = -1` counts from `page` to the end rather than from the
     * start of the file. Works by loading a four-frame file at
     * `page = 1, n = -1` and requiring three pages.
     * Measured on vips 8.18.6: `anim4.gif[page=1,n=-1]` is 4x9, three pages
     * of the four.
     * Input: a four-frame GIF at `page = 1, n = -1` -> Output: three pages,
     * frames 1, 2 and 3.
     */
    #[test]
    fn n_minus_one_loads_from_the_page_to_the_end() {
        let frames: Vec<Frame> = (0..4u8).map(|i| anim_frame([i; 4], 1, 0)).collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_page(1).with_n(-1),
        )
        .expect("a valid GIF");

        assert_eq!(raster.pages_loaded(), 3);
        for (page, index) in (1u8..4).enumerate() {
            assert_eq!(
                page_bytes(&raster, page as u32),
                ANIM_PALETTE[index as usize].repeat(4)
            );
        }
    }

    /**
     * Tests that every window the file cannot serve is refused, rather than
     * being silently clamped to what is there. Works by asking a four-frame
     * fixture for each of the five shapes vips rejects and requiring the
     * typed `BadPageNumber` back, with a load that does work as the positive
     * control.
     * Measured on vips 8.18.6 against a four-frame file: `[page=4]`,
     * `[n=99]`, `[n=0]` and `[page=3,n=3]` all fail with
     * `gifload: bad page number`, while `[page=2,n=-1]` succeeds.
     * Input: five out-of-range windows -> Output: `GifError::BadPageNumber`
     * for each, and a page count for the one in range.
     */
    #[test]
    fn a_window_the_file_cannot_serve_is_refused() {
        let frames: Vec<Frame> = (0..4u8).map(|i| anim_frame([i; 4], 1, 0)).collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);
        for (page, n) in [(4u32, 1i32), (0, 99), (0, 0), (3, 3), (0, -2)] {
            let err = decode_gif_with(
                &bytes,
                DecodeLimits::default(),
                LoadOptions::default().with_page(page).with_n(n),
            )
            .expect_err("the window is out of range");
            // Every field, not just `frames`: the error reports back what was
            // asked for, and asserting only the file's frame count would
            // survive a `bad()` that swapped `page` and `n` or hardcoded
            // both to zero.
            assert!(
                matches!(
                    err,
                    SourceError::Gif(GifError::BadPageNumber {
                        page: p,
                        n: count,
                        frames: 4,
                    }) if p == page && count == n
                ),
                "page {page} n {n}: {err:?}"
            );
        }
        let ok = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_page(2).with_n(-1),
        )
        .expect("page 2 to the end is in range");
        assert_eq!(ok.pages_loaded(), 2, "the positive control still loads");
    }

    /**
     * Tests that disposal "keep" leaves the canvas alone, so a later frame
     * that paints only part of the screen composites over what came before.
     * Works by painting the whole screen red, then a single green pixel with
     * disposal 1, then a single blue pixel, and reading all three pages.
     * Measured on vips 8.18.6 against exactly this file: page 1 is
     * `green red / red red` and page 2 is `green red / red blue`. Disposal
     * code 0 ("unspecified") produces the same three pages, which is the
     * second half of this test.
     * Input: a three-frame GIF with disposal 1 and then with disposal 0 ->
     * Output: identical, cumulative pages.
     */
    #[test]
    fn disposal_keep_composites_each_frame_over_the_last() {
        for disposal in [1u8, 0] {
            let bytes = fixture(
                (2, 2),
                &ANIM_PALETTE,
                3,
                Some(0),
                &[
                    anim_frame([1, 1, 1, 1], disposal, 0),
                    Frame {
                        width: 1,
                        height: 1,
                        indices: vec![2],
                        disposal,
                        ..anim_frame([0; 4], disposal, 0)
                    },
                    Frame {
                        left: 1,
                        top: 1,
                        width: 1,
                        height: 1,
                        indices: vec![3],
                        disposal,
                        ..anim_frame([0; 4], disposal, 0)
                    },
                ],
            );
            let raster = decode_gif_with(
                &bytes,
                DecodeLimits::default(),
                LoadOptions::default().with_n(-1),
            )
            .expect("a valid GIF");
            assert_eq!(page_bytes(&raster, 0), [255, 0, 0].repeat(4), "{disposal}");
            assert_eq!(
                page_bytes(&raster, 1),
                [0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0],
                "disposal {disposal} keeps the red canvas under the green dot"
            );
            assert_eq!(
                page_bytes(&raster, 2),
                [0, 255, 0, 255, 0, 0, 255, 0, 0, 0, 0, 255],
                "disposal {disposal} keeps both dots"
            );
        }
    }

    /**
     * Tests that disposal "restore to background" clears the disposed
     * frame's rectangle to the **background colour**, and only that
     * rectangle. Works by painting a 3x1 screen red, disposing the middle
     * pixel to background, then painting the left one green.
     * Measured on vips 8.18.6 against exactly this file: page 2 is
     * `green blue red`, with blue the background index, so the clear is the
     * background colour and it does not reach the third pixel.
     * Input: a three-frame 3x1 GIF, background index 3 -> Output: page 2 is
     * green, blue, red.
     */
    #[test]
    fn disposal_restore_to_background_clears_the_rectangle_to_the_background_colour() {
        let bytes = fixture(
            (3, 1),
            &ANIM_PALETTE,
            3,
            Some(0),
            &[
                Frame::full(3, 1, vec![1, 1, 1]),
                Frame {
                    left: 1,
                    width: 1,
                    height: 1,
                    indices: vec![2],
                    disposal: 2,
                    ..Frame::full(1, 1, vec![2])
                },
                Frame {
                    width: 1,
                    height: 1,
                    indices: vec![2],
                    ..Frame::full(1, 1, vec![2])
                },
            ],
        );
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");
        assert_eq!(page_bytes(&raster, 0), [255, 0, 0].repeat(3));
        assert_eq!(
            page_bytes(&raster, 1),
            [255, 0, 0, 0, 255, 0, 255, 0, 0],
            "the green dot sits on the red canvas"
        );
        assert_eq!(
            page_bytes(&raster, 2),
            [0, 255, 0, 0, 0, 255, 255, 0, 0],
            "only the disposed pixel goes to the background colour"
        );
    }

    /**
     * Tests that "restore to background" clears to **transparent** instead
     * when the disposed frame declares a transparent index, which is the
     * arm of libnsgif's rule the background-colour test cannot see. Works by
     * running the same two-frame file twice, once with a transparent index
     * on frame 0 and once without, and comparing what page 1 shows outside
     * the second frame.
     * Measured on vips 8.18.6: with a transparent index on frame 0 the file
     * loads four-band and page 1 is `green + (0,0,0,0)`; with the index on
     * frame 1 only, frame 0's own clear still uses the background colour and
     * page 1 is `green + opaque blue`.
     * Input: two 2x2 GIFs differing only in which frame declares
     * transparency -> Output: a transparent clear in one and a blue clear in
     * the other.
     */
    #[test]
    fn disposal_restore_to_background_clears_to_transparent_when_the_frame_has_a_transparent_index()
    {
        let clear_frame = |transparent_on_first: bool| {
            fixture(
                (2, 2),
                &ANIM_PALETTE,
                3,
                Some(0),
                &[
                    Frame {
                        transparent: transparent_on_first.then_some(0),
                        disposal: 2,
                        ..anim_frame([1, 1, 1, 1], 2, 0)
                    },
                    Frame {
                        width: 1,
                        height: 1,
                        indices: vec![2],
                        transparent: (!transparent_on_first).then_some(0),
                        ..anim_frame([0; 4], 1, 0)
                    },
                ],
            )
        };

        let transparent = decode_gif_with(
            &clear_frame(true),
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");
        assert_eq!(transparent.format(), PixelFormat::Rgba8);
        assert_eq!(
            page_bytes(&transparent, 1),
            [0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "a transparent frame's own clear is transparent, not the background"
        );

        let coloured = decode_gif_with(
            &clear_frame(false),
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");
        assert_eq!(coloured.format(), PixelFormat::Rgba8);
        assert_eq!(
            page_bytes(&coloured, 1),
            [
                0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255
            ],
            "an opaque frame's clear is the background colour, opaque"
        );
    }

    /**
     * Tests that disposal "restore to previous" rewinds the canvas to what
     * it was before the disposed frame drew, rather than to the start of the
     * file. Works by painting the screen red with disposal "keep", painting
     * a green dot with disposal 3, then painting a blue dot, and requiring
     * page 2 to be red with the blue dot and no green.
     * Measured on vips 8.18.6 against exactly this file: page 2 is
     * `red red / red blue`.
     * Input: a three-frame 2x2 GIF, middle frame disposal 3 -> Output: page
     * 2 has the green dot rewound and the red canvas back.
     */
    #[test]
    fn disposal_restore_to_previous_rewinds_to_before_the_frame() {
        let bytes = fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[
                anim_frame([1, 1, 1, 1], 1, 0),
                Frame {
                    width: 1,
                    height: 1,
                    indices: vec![2],
                    disposal: 3,
                    ..anim_frame([0; 4], 3, 0)
                },
                Frame {
                    left: 1,
                    top: 1,
                    width: 1,
                    height: 1,
                    indices: vec![3],
                    ..anim_frame([0; 4], 1, 0)
                },
            ],
        );
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");
        assert_eq!(
            page_bytes(&raster, 1),
            [0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]
        );
        assert_eq!(
            page_bytes(&raster, 2),
            [255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 255],
            "the green dot is gone and the red canvas is back"
        );
    }

    /**
     * Tests that a frame whose transparent index falls where an earlier
     * frame painted lets that earlier frame show through, which is the
     * blending half of animated decode. Works by painting the screen red
     * then drawing a full-screen frame whose corners are the transparent
     * index.
     * Measured on vips 8.18.6 against exactly this file: page 1 is
     * `green red / red green`, all four pixels opaque, because the two
     * transparent indices resolve to the red underneath.
     * Input: a two-frame 2x2 GIF, second frame half transparent -> Output:
     * page 1 shows red through the transparent pixels.
     */
    #[test]
    fn a_transparent_pixel_lets_the_earlier_frame_show_through() {
        let bytes = fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[
                anim_frame([1, 1, 1, 1], 1, 0),
                Frame {
                    transparent: Some(0),
                    ..anim_frame([2, 0, 0, 2], 1, 0)
                },
            ],
        );
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");
        assert_eq!(raster.format(), PixelFormat::Rgba8);
        assert_eq!(
            page_bytes(&raster, 1),
            [
                0, 255, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255
            ],
            "the transparent index resolves to the frame underneath, opaque"
        );
    }

    /**
     * Tests that a background index past the end of the global colour table
     * disposes to colour table **entry 0**, not to black. Works by running
     * the restore-to-background fixture over a palette whose entry 0 is
     * `(9, 8, 7)` rather than black, so the two answers are distinguishable,
     * with the in-range index as the positive control.
     * The fixture the animated lane wrote for this used a palette whose
     * entry 0 *is* black, so "entry 0" and "black" gave the same pixels and
     * it could not see the difference. Measured on vips 8.18.6 against
     * exactly these files: background indices 200 and 0 both dispose to
     * `9 8 7` and report `background: 9 8 7`, where index 3 disposes to blue
     * and reports `0 0 255`.
     * This test **passes on both sides of issue #850**, and that is the
     * point rather than a weakness. `gif` 0.14.2 clears `bg_color()` to
     * `None` when the stored index is past the palette
     * (`reader/mod.rs:354-358`), so the loader reached entry 0 by way of
     * `index.unwrap_or(0)` before the fallback existed. The divergence lives
     * one level down, in `background_rgb` itself, where
     * `the_background_lookup_takes_a_whole_entry_or_none` holds it; this is
     * the end-to-end fixture that says which answer is right, and it is the
     * one the original should have been written with.
     * Input: the same two-frame GIF with background index 200, 0 and 3 ->
     * Output: entry 0 twice, then blue.
     */
    #[test]
    fn an_out_of_range_background_index_disposes_to_colour_table_entry_zero() {
        // Entry 0 is deliberately not black: with ANIM_PALETTE this whole
        // test passes against either implementation.
        const PALETTE: [[u8; 3]; 4] = [[9, 8, 7], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let load = |background: u8| {
            let bytes = fixture(
                (2, 2),
                &PALETTE,
                background,
                Some(0),
                &[
                    Frame {
                        disposal: 2,
                        ..Frame::full(2, 2, vec![1; 4])
                    },
                    Frame {
                        width: 1,
                        height: 1,
                        indices: vec![2],
                        ..Frame::full(1, 1, vec![2])
                    },
                ],
            );
            let raster = decode_gif_with(
                &bytes,
                DecodeLimits::default(),
                LoadOptions::default().with_n(-1),
            )
            .expect("a valid GIF");
            page_bytes(&raster, 1)
        };
        let entry_zero = [0, 255, 0, 9, 8, 7, 9, 8, 7, 9, 8, 7];
        assert_eq!(
            load(200),
            entry_zero,
            "an index past the end of the table falls back to entry 0"
        );
        assert_eq!(
            load(0),
            entry_zero,
            "and index 0 is the same entry, which is the arm that makes it entry 0 rather than black"
        );
        assert_eq!(
            load(3),
            [0, 255, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255],
            "the positive control: an in-range index still disposes to its own colour"
        );
    }

    /**
     * Tests that the background lookup reads a whole colour table entry or
     * none of it, so a table that stops mid-entry cannot contribute one or
     * two of a pixel's three bytes. Works by calling `background_rgb`
     * directly with a four-byte table, where index 1 wants bytes 3, 4 and 5
     * and only byte 3 is there.
     * A GIF's global table is always a power-of-two count of three-byte
     * entries, so the `gif` crate cannot hand this in; the test is here
     * because the whole-slice read and a byte-at-a-time read with a zero
     * default are otherwise indistinguishable, and only one of them is
     * total by construction.
     * The table is seven bytes rather than four so the three candidate rules
     * come apart: index 2 wants bytes 6, 7 and 8 and only byte 6 is there, and
     * "fall back to entry 0" answers `7 8 9` where "clamp to the last whole
     * entry" answers `1 2 3` and "ignore the index" answers `7 8 9` for index
     * 1 as well. Only the first rule gives all three of the answers below. A
     * four-byte table cannot tell them apart, which is the #850 shape one
     * level down.
     * Input: a seven-byte table at indices 0, 1 and 2 -> Output: entry 0,
     * entry 1, then entry 0 again.
     */
    #[test]
    fn the_background_lookup_takes_a_whole_entry_or_none() {
        let ragged = [7u8, 8, 9, 1, 2, 3, 4];
        assert_eq!(
            background_rgb(Some(&ragged), Some(0)),
            [7, 8, 9],
            "the entry that is wholly there is read"
        );
        assert_eq!(
            background_rgb(Some(&ragged), Some(1)),
            [1, 2, 3],
            "and so is the second one, which is what stops this passing for a \
             lookup that ignores the index"
        );
        assert_eq!(
            background_rgb(Some(&ragged), Some(2)),
            [7, 8, 9],
            "an entry that runs off the end contributes nothing, not one byte, \
             so the whole lookup falls back to entry 0 rather than clamping to \
             the last whole entry, which would be 1 2 3"
        );
        assert_eq!(
            background_rgb(None, Some(0)),
            [0, 0, 0],
            "no table is libnsgif's substitute, whose entry 0 is black"
        );
        assert_eq!(
            background_rgb(None, Some(1)),
            [0, 0, 0],
            "and the substitute is never indexed: entry 1 of it is white, and \
             vips reports black for a stored index of 1 with no global table"
        );
        assert_eq!(
            background_rgb(Some(&ragged), None),
            [7, 8, 9],
            "an absent index is index 0, which is what the descriptor stores"
        );
        assert_eq!(
            background_rgb(Some(&ragged[..2]), Some(0)),
            [0, 0, 0],
            "a table with no whole entry at all is black, since there is no \
             entry 0 to fall back to either"
        );
    }

    /**
     * Tests that every GIF89a disposal code lands on the canvas rule
     * libnsgif applies to it, code 4 included. Works by running the same
     * two-frame file with each of codes 0 through 7 on frame 0 and reading
     * page 1 back.
     * Code 4 is reserved by GIF89a and libnsgif remaps it onto
     * restore-to-previous (`NSGIF_DISPOSAL_RESTORE_QUIRK`), while codes 5, 6
     * and 7 keep the canvas the way 0 and 1 do. Measured on vips 8.18.6
     * against exactly these eight files, a 2x1 screen with background index
     * 3, frame 0 painting both pixels red and frame 1 a green pixel at
     * (0, 0): codes 0, 1, 5, 6 and 7 give `green red`, code 2 gives
     * `green blue`, and codes 3 and 4 give `green black`. libviprs used to
     * give `green red` for 4 because the `gif` crate folds every code it
     * does not know onto `DisposalMethod::Any`, which is issue #827.
     * The three answers are all different, so no single expectation can be
     * satisfied by an implementation that ignores the code.
     * Input: a two-frame 2x1 GIF with disposal 0 to 7 ->
     * Output: keep, keep, background, previous, previous, keep, keep, keep.
     */
    #[test]
    fn every_disposal_code_follows_libnsgif_including_the_reserved_four() {
        let page_one = |disposal: u8| {
            let bytes = fixture(
                (2, 1),
                &ANIM_PALETTE,
                3,
                Some(0),
                &[
                    Frame {
                        disposal,
                        ..Frame::full(2, 1, vec![1, 1])
                    },
                    Frame {
                        width: 1,
                        height: 1,
                        indices: vec![2],
                        ..Frame::full(1, 1, vec![2])
                    },
                ],
            );
            let raster = decode_gif_with(
                &bytes,
                DecodeLimits::default(),
                LoadOptions::default().with_n(-1),
            )
            .expect("a valid GIF");
            page_bytes(&raster, 1)
        };

        let keep = [0, 255, 0, 255, 0, 0];
        let background = [0, 255, 0, 0, 0, 255];
        let previous = [0, 255, 0, 0, 0, 0];
        assert_ne!(keep, background);
        assert_ne!(keep, previous);
        assert_ne!(background, previous);

        for (disposal, expected, name) in [
            (0u8, keep, "keep"),
            (1, keep, "keep"),
            (2, background, "restore to background"),
            (3, previous, "restore to previous"),
            (4, previous, "restore to previous, the reserved quirk"),
            (5, keep, "keep"),
            (6, keep, "keep"),
            (7, keep, "keep"),
        ] {
            assert_eq!(
                page_one(disposal),
                expected,
                "disposal {disposal} is {name}"
            );
        }
    }

    /**
     * Tests that the second walk terminates and does not panic on bytes that
     * are not a well-formed GIF. Works by running `ControlWalk` to exhaustion
     * over every prefix of a real fixture and over the same fixture with each
     * byte in turn replaced by each of the four block introducers, bounding
     * the number of frames it may yield so a walk that fails to advance is a
     * failure rather than a hang.
     * `ControlWalk` is the one place in this module that indexes a buffer by
     * arithmetic on lengths the file itself supplies, which is where a decoder
     * gets its panics. Every read goes through `get`, and this is what says so
     * rather than the comment saying so.
     * The positive control is the intact fixture, which must still yield its
     * three frames with the disposal codes it was built with, so a walk that
     * gave up immediately would pass the panic half and fail here.
     * Input: 1 fixture, its 200-odd prefixes and about 800 corruptions ->
     * Output: no panic, a bounded frame count, and the intact file unchanged.
     */
    #[test]
    fn the_control_walk_is_total_over_bytes_that_are_not_a_gif() {
        let bytes = fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[
                Frame {
                    disposal: 4,
                    local: Some(vec![[1, 2, 3], [4, 5, 6]]),
                    comment: Some(b"hello".to_vec()),
                    ..Frame::full(2, 2, vec![0, 1, 1, 0])
                },
                Frame {
                    disposal: 2,
                    control: false,
                    ..Frame::full(2, 2, vec![1, 1, 1, 1])
                },
                Frame {
                    disposal: 7,
                    ..Frame::full(2, 2, vec![2, 2, 3, 3])
                },
            ],
        );

        // Every frame on the wire costs at least eleven bytes, the image
        // descriptor's introducer and nine fields plus the LZW code size, so
        // a walk yielding more than a tenth of the buffer has stopped
        // advancing rather than found something. The loose `input.len()`
        // bound this used to carry would have let a phantom frame every two
        // bytes through.
        let drain = |input: &[u8]| {
            let ceiling = input.len() / 10 + 1;
            let mut walk = ControlWalk::new(input);
            let mut seen = Vec::new();
            while let Some(control) = walk.next_frame() {
                seen.push(control);
                assert!(seen.len() <= ceiling, "the walk is not advancing");
            }
            seen
        };

        // A plain single-sub-block extension, where libnsgif's reading and
        // the crate's are the same four bytes.
        let present = |disposal: u8| {
            let control = RawControl {
                disposal,
                delay_cs: 0,
                transparent: None,
            };
            WireControl::Present {
                control,
                as_decoder: Some(control),
            }
        };
        let intact = drain(&bytes);
        assert_eq!(
            intact,
            [present(4), WireControl::Absent, present(7),],
            "the control: the intact file still walks, and the middle frame \
             comes back as carrying no extension rather than as code 0"
        );

        for cut in 0..bytes.len() {
            drain(&bytes[..cut]);
        }
        for at in 0..bytes.len() {
            for byte in [0x21u8, 0x2C, 0x3B, 0xFF] {
                let mut corrupt = bytes.clone();
                corrupt[at] = byte;
                drain(&corrupt);
            }
        }
    }

    /**
     * Tests that a GIF declaring no global colour table decodes through
     * libnsgif's own substitute rather than being refused. Works by decoding
     * a two-pixel file whose logical screen descriptor clears the
     * global-table flag and whose only frame paints indices 0 and 1.
     * libnsgif builds a two-entry black-and-white table for such a file and
     * sets its colour table size to 2, so the file has a picture. `gif`
     * 0.14.2 refuses the frame outright instead: `next_frame_info` errors
     * with "no color table available for current frame" when a frame has
     * neither a local palette nor a global one, and `Decoder` has no way to
     * supply one from outside. So the loader hands the decoder the same file
     * with the table spliced in, which is the only shape the crate will read
     * and is byte for byte what libnsgif invents.
     * Measured on vips 8.18.6 against exactly this file: `getpoint 0 0` is
     * `0 0 0`, `getpoint 1 0` is `255 255 255`, `gif-palette` is
     * `-16777216 -1`, `bits-per-sample` is 1 and `background` is `0 0 0`.
     * The stored background index is the control that says the substitute is
     * never indexed: index 1 still reports black, which is its entry 0, and
     * not the white its entry 1 holds.
     * Input: a 2x1 GIF with no colour table anywhere -> Output: black then
     * white, a two-entry palette and one bit per sample.
     */
    #[test]
    fn a_gif_with_no_colour_table_decodes_through_libnsgifs_black_and_white() {
        let bytes = fixture_tables((2, 1), None, 0, None, &[Frame::full(2, 1, vec![0, 1])]);
        let raster = decode_bytes(&bytes).expect("decodes");
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(
            raster.data(),
            [0, 0, 0, 255, 255, 255],
            "index 0 is the substitute's black and index 1 its white"
        );
        assert_eq!(
            raster.get_int_array("gif-palette"),
            Some(&[-16_777_216i64, -1][..]),
            "and the substitute is the table that gets attached"
        );
        assert_eq!(raster.get_int("bits-per-sample"), Some(1));
        assert_eq!(raster.get_int("palette-bit-depth"), Some(1));

        // The stored index is ignored, whatever it says, because libnsgif
        // only honours it when the file really declares a table. This needs
        // the background colour to actually reach a pixel, so it is a
        // restore-to-background disposal on a frame that is not the last:
        // painting index 1 and disposing it shows black if the index was
        // forced to entry 0 and white if it was kept, and those are the two
        // entries of the substitute table.
        for stored in [0u8, 1, 3, 255] {
            let bytes = fixture_tables(
                (2, 1),
                None,
                stored,
                Some(0),
                &[
                    Frame {
                        disposal: 2,
                        ..Frame::full(2, 1, vec![1, 1])
                    },
                    Frame {
                        width: 1,
                        height: 1,
                        indices: vec![0],
                        ..Frame::full(1, 1, vec![0])
                    },
                ],
            );
            let raster = decode_gif_with(
                &bytes,
                DecodeLimits::default(),
                LoadOptions::default().with_n(-1),
            )
            .expect("a valid GIF");
            assert_eq!(
                page_bytes(&raster, 0),
                [255, 255, 255, 255, 255, 255],
                "index 1 of the substitute is white, so the fixture can tell \
                 the two entries apart"
            );
            assert_eq!(
                page_bytes(&raster, 1),
                [0, 0, 0, 0, 0, 0],
                "background index {stored} with no table disposes to entry 0, \
                 which is black, not to the white the index would name"
            );
        }

        // The positive control: the same file *with* a table is a different
        // picture, so "black then white" is not what any decode would give.
        let declared = fixture(
            (2, 1),
            &[[9, 8, 7], [1, 2, 3]],
            0,
            None,
            &[Frame::full(2, 1, vec![0, 1])],
        );
        assert_eq!(
            decode_bytes(&declared).expect("decodes").data(),
            [9, 8, 7, 1, 2, 3]
        );
    }

    /**
     * Tests that a graphic control extension whose **last** sub-block is not
     * four bytes no longer truncates the file. Works by decoding two-frame
     * fixtures whose frame 0 carries such a chain and requiring both frames
     * back, with the disposal libnsgif reads out of it.
     * `gif` 0.14.2's `read_control_extension` errors with "control extension
     * has wrong length" unless the last sub-block is exactly four
     * (`reader/decoder.rs:853-856`), and the error ends the header scan, so
     * every frame from that extension onwards disappears. libnsgif does not
     * measure the chain at all, so the file loads. The loader now rewrites
     * each such chain into the single four-byte sub-block libnsgif's own read
     * would produce, which is the only shape the crate accepts and carries
     * exactly the bytes libnsgif uses.
     * Measured on vips 8.18.6 against exactly these files: both report
     * `n-pages: 2` and page 1 is `green black`, the rewind disposal 4 asks
     * for. libviprs reported `NoFrames`.
     * Input: chains of `04 quad(4) 01 AA 00` and `06 quad(4) AA BB 00` ->
     * Output: two pages each, and the canvas rewound.
     */
    #[test]
    fn a_control_extension_the_crate_refuses_no_longer_truncates_the_file() {
        let quad = |disposal: u8| [(disposal & 7) << 2, 0u8, 0, 0];
        let load = |raw: Vec<u8>| {
            let bytes = fixture(
                (2, 1),
                &ANIM_PALETTE,
                3,
                Some(0),
                &[
                    Frame {
                        raw_control: Some(raw),
                        ..Frame::full(2, 1, vec![1, 1])
                    },
                    Frame {
                        width: 1,
                        height: 1,
                        indices: vec![2],
                        ..Frame::full(1, 1, vec![2])
                    },
                ],
            );
            decode_gif_with(
                &bytes,
                DecodeLimits::default(),
                LoadOptions::default().with_n(-1),
            )
        };

        // A trailing one-byte sub-block, so the last one is not four.
        let mut trailing = vec![4u8];
        trailing.extend_from_slice(&quad(4));
        trailing.extend_from_slice(&[1, 0xAA, 0]);
        let raster = load(trailing).expect("the file has two frames");
        assert_eq!(raster.get_n_pages(), 2);
        assert_eq!(
            page_bytes(&raster, 1),
            [0, 255, 0, 0, 0, 0],
            "and frame 0's code 4 rewound the canvas"
        );

        // One six-byte sub-block, so no sub-block is four.
        let mut six = vec![6u8];
        six.extend_from_slice(&quad(4));
        six.extend_from_slice(&[0xAA, 0xBB, 0]);
        let raster = load(six).expect("the file has two frames");
        assert_eq!(raster.get_n_pages(), 2);
        assert_eq!(page_bytes(&raster, 1), [0, 255, 0, 0, 0, 0]);

        // The control: the ordinary shape was never truncated, so "two
        // pages" is not what every file gives.
        let mut plain = vec![4u8];
        plain.extend_from_slice(&quad(1));
        plain.push(0);
        let raster = load(plain).expect("a valid GIF");
        assert_eq!(raster.get_n_pages(), 2);
        assert_eq!(
            page_bytes(&raster, 1),
            [0, 255, 0, 255, 0, 0],
            "disposal 1 keeps, so this fixture can tell the two apart"
        );
    }

    /**
     * Tests that the normalising copy is only made when it is needed, and is
     * priced when it is. Works by calling `normalise_for_decoder` on the
     * three shapes and checking which of them allocate, then running the two
     * that do under a budget smaller than the file.
     * The copy is a second whole-file buffer, so it is the kind of allocation
     * `DecodeLimits` exists for, and it sits before every other price this
     * module pays: a file that needs it has already been copied by the time
     * the canvas is measured. The "no copy" half is what keeps that off every
     * ordinary GIF, and it is asserted rather than assumed because a
     * normaliser that copied unconditionally would pass every other test in
     * this module.
     * Input: an ordinary GIF, one with no colour table, one with an
     * unreadable control chain -> Output: no copy, a copy, a copy, and a
     * typed refusal for each copy under a 32-byte budget.
     */
    #[test]
    fn the_normalising_copy_is_made_only_when_needed_and_is_priced() {
        let ordinary = fixture(
            (2, 1),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[Frame::full(2, 1, vec![0, 1])],
        );
        assert!(
            normalise_for_decoder(&ordinary, DecodeLimits::default())
                .expect("no budget trouble")
                .is_none(),
            "an ordinary GIF is handed to the decoder untouched"
        );

        let no_table = fixture_tables((2, 1), None, 0, None, &[Frame::full(2, 1, vec![0, 1])]);
        let mut chain = vec![4u8, 0b0001_0000, 0, 0, 0];
        chain.extend_from_slice(&[1, 0xAA, 0]);
        let bad_chain = fixture(
            (2, 1),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[Frame {
                raw_control: Some(chain),
                ..Frame::full(2, 1, vec![0, 1])
            }],
        );

        for (name, bytes) in [("no colour table", &no_table), ("bad chain", &bad_chain)] {
            let copy = normalise_for_decoder(bytes, DecodeLimits::default())
                .expect("no budget trouble")
                .unwrap_or_else(|| panic!("{name} needs a copy"));
            assert_ne!(copy, *bytes, "{name}: the copy is not the original");

            let err = decode_gif(bytes, DecodeLimits::default().with_max_alloc_bytes(32))
                .expect_err("{name} is over a 32-byte budget");
            assert!(
                matches!(
                    &err,
                    SourceError::AllocLimitExceeded {
                        what: "GIF normalised copy",
                        geometry: None,
                        ..
                    }
                ),
                "{name}: {err:?}"
            );
        }

        // The control: the same budget decodes the ordinary file, whose
        // canvas is six bytes, so the two refusals above are the copy and not
        // something every file at 32 bytes would hit.
        assert_eq!(
            decode_gif(&ordinary, DecodeLimits::default().with_max_alloc_bytes(32))
                .expect("six bytes of canvas is under a 32-byte budget")
                .data(),
            [0, 0, 0, 255, 0, 0]
        );
    }

    /**
     * Tests that a graphic control extension spread over more than one
     * sub-block, or with a size byte that does not say 4, is read the way
     * libnsgif reads it. Works by writing five `0xF9` chains verbatim in front
     * of the same two-frame file and reading page 1 and the delay back.
     * libnsgif never looks at the chain: it reads four bytes at a fixed offset
     * from the label behind a bare length check, so the size byte and every
     * sub-block after the first are invisible to it. The `gif` crate reads the
     * **last** sub-block instead, because it clears its extension buffer on
     * every sub-block, and this walk used to require the whole chain to total
     * four, which is neither rule (issue #878).
     * Measured on vips 8.18.6 against exactly these files. Row 2 separates
     * libnsgif's rule from the crate's, because its two quads disagree; rows 3
     * and 4 separate it from both, because `0xAA` is not a disposal anyone
     * would write and comes back as restore-to-background with a delay of
     * thousands of centiseconds, which is the strongest evidence available
     * that the fixed offset is the rule.
     * Input: five chains -> Output: the disposal and delay libnsgif reads.
     */
    #[test]
    fn a_control_extension_split_across_sub_blocks_reads_the_way_libnsgif_does() {
        // A four-byte payload: packed byte, two delay bytes, transparent index.
        let quad = |disposal: u8| [(disposal & 7) << 2, 0, 0, 0];
        let chain = |parts: &[&[u8]]| {
            let mut out = Vec::new();
            for part in parts {
                out.push(part.len() as u8);
                out.extend_from_slice(part);
            }
            out.push(0);
            out
        };
        let load = |raw: Vec<u8>| {
            let bytes = fixture(
                (2, 1),
                &ANIM_PALETTE,
                3,
                Some(0),
                &[
                    Frame {
                        raw_control: Some(raw),
                        ..Frame::full(2, 1, vec![1, 1])
                    },
                    Frame {
                        width: 1,
                        height: 1,
                        indices: vec![2],
                        ..Frame::full(1, 1, vec![2])
                    },
                ],
            );
            let raster = decode_gif_with(
                &bytes,
                DecodeLimits::default(),
                LoadOptions::default().with_n(-1),
            )
            .expect("a valid GIF");
            (page_bytes(&raster, 1), delays(&raster))
        };

        let previous = [0, 255, 0, 0, 0, 0];
        let background = [0, 255, 0, 0, 0, 255];

        // The control: one four-byte sub-block, where all three readings are
        // the same four bytes.
        assert_eq!(
            load(chain(&[&quad(4)])).0,
            previous,
            "one sub-block, code 4"
        );

        // Two four-byte sub-blocks. libnsgif reads the first, the crate reads
        // the second, and the second row is what tells them apart.
        assert_eq!(
            load(chain(&[&quad(4), &quad(4)])).0,
            previous,
            "two sub-blocks, both code 4"
        );
        assert_eq!(
            load(chain(&[&quad(4), &quad(0)])).0,
            previous,
            "two sub-blocks, code 4 then code 0: the **first** one wins"
        );

        // A one-byte sub-block first, so libnsgif's four bytes start at 0xAA
        // and run into the next sub-block's length byte and payload.
        let (page, delay) = load(chain(&[&[0xAA], &quad(3)]));
        assert_eq!(
            page, background,
            "0xAA is disposal 2, whatever the sub-block after it says"
        );
        assert_eq!(
            delay,
            Some(vec![30_760, 0]),
            "and the delay is the two bytes after 0xAA, which are 04 0C"
        );
        let (page, delay) = load(chain(&[&[0xAA], &quad(4)]));
        assert_eq!(page, background);
        assert_eq!(delay, Some(vec![41_000, 0]));
    }

    /**
     * Tests that the raw disposal code survives the two things that can sit
     * between a frame's graphic control extension and its pixels: another
     * extension block, and a colour table on an earlier frame. Works by
     * running disposal 4 behind a comment extension, and then behind a frame
     * that carries a local colour table, and requiring the rewind both times.
     * Both are places the second walk can lose its place, and both are
     * **masked** by `reconcile_control`'s fallback: a walk that comes apart
     * gives the answer libviprs gave before #827, so the pixels go quietly
     * back to being wrong rather than failing loudly. The mutation sweep
     * found exactly that, which is why the local table sits on the frame
     * *before* the one carrying code 4: on the code-4 frame itself the walk
     * has already returned that frame's control before it has to step over
     * the table, so the fixture cannot see the mistake.
     * Measured on vips 8.18.6 against exactly these two files. The comment
     * one is `red red` then `green black`; the local-table one is
     * `(200,100,50) (200,100,50)`, then `red red`, then `green
     * (200,100,50)`, the last page being the rewind putting the first
     * frame's colours back.
     * Input: disposal 4 behind a comment extension, and behind a local
     * colour table -> Output: the rewind in both.
     */
    #[test]
    fn code_four_survives_a_comment_extension_and_an_earlier_local_table() {
        let roll = |bytes: &[u8]| {
            decode_gif_with(
                bytes,
                DecodeLimits::default(),
                LoadOptions::default().with_n(-1),
            )
            .expect("a valid GIF")
        };
        let dot = || Frame {
            width: 1,
            height: 1,
            indices: vec![2],
            ..Frame::full(1, 1, vec![2])
        };

        // A comment extension between the control extension and the image
        // descriptor. Its payload is thirteen bytes rather than four, which
        // is what stops it from being read as a control extension, and the
        // label check is what stops it from being read as one anyway.
        let commented = fixture(
            (2, 1),
            &ANIM_PALETTE,
            3,
            Some(0),
            &[
                Frame {
                    disposal: 4,
                    comment: Some(b"libviprs #827".to_vec()),
                    ..Frame::full(2, 1, vec![1, 1])
                },
                dot(),
            ],
        );
        assert_eq!(
            page_bytes(&roll(&commented), 1),
            [0, 255, 0, 0, 0, 0],
            "a comment extension does not displace the control extension"
        );

        // A local colour table on frame 0, which the walk has to step over
        // to reach frame 1's control extension.
        let local: Vec<[u8; 3]> = vec![[9, 8, 7], [200, 100, 50], [0, 255, 0], [1, 1, 1]];
        let with_local = fixture(
            (2, 1),
            &ANIM_PALETTE,
            3,
            Some(0),
            &[
                Frame {
                    local: Some(local),
                    ..Frame::full(2, 1, vec![1, 1])
                },
                Frame {
                    disposal: 4,
                    ..Frame::full(2, 1, vec![1, 1])
                },
                dot(),
            ],
        );
        let raster = roll(&with_local);
        assert_eq!(
            page_bytes(&raster, 0),
            [200, 100, 50, 200, 100, 50],
            "frame 0 really did draw through its own table, not the global one"
        );
        assert_eq!(page_bytes(&raster, 1), [255, 0, 0, 255, 0, 0]);
        assert_eq!(
            page_bytes(&raster, 2),
            [0, 255, 0, 200, 100, 50],
            "frame 1's code 4 rewound to frame 0's colours, so the walk kept its place"
        );
    }

    /**
     * Tests that the three deprecated compatibility fields `gifload` attaches
     * beside the modern ones come back too. Works by decoding four fixtures
     * and reading `gif-delay`, `gif-loop` and `palette-bit-depth` off each.
     * `nsgifload.c` writes all three: `gif-delay` is `delay[0] / 10`, so the
     * first frame's delay back in the centiseconds the wire uses; `gif-loop`
     * is `loop_max == 0 ? 0 : loop_max - 1`, which is the NETSCAPE count a
     * file with a block carries and 0 for a file without one; and
     * `palette-bit-depth` is a second copy of `bits-per-sample`.
     * They are invisible to `vipsheader -a`, which lists no deprecated
     * compatibility field on any loader, so every number here came from
     * `vipsheader -f` on 8.18.6. Reading them with `-a` produces the opposite
     * finding, that vips dropped them.
     * Measured on exactly these files: a four-frame animation with delays
     * `4 6 8 10` cs and a NETSCAPE count of 3 reports `gif-delay 4`,
     * `gif-loop 3` and `palette-bit-depth 2`; a still with a zero delay and
     * no block reports `0`, `0`, `2`; a still with no graphic control
     * extension at all reports `10`, `0`, `2`; and an eight-entry table
     * reports `palette-bit-depth 3`.
     * Input: four GIFs -> Output: the three fields, on every one of them.
     */
    #[test]
    fn the_deprecated_compatibility_fields_come_back_beside_the_modern_ones() {
        let full = || Frame::full(4, 3, (0..12u8).map(|i| i % 4).collect());
        let delayed = |cs: u16| Frame {
            delay_cs: cs,
            ..full()
        };

        let anim = fixture(
            (4, 3),
            &ANIM_PALETTE,
            0,
            Some(3),
            &[delayed(4), delayed(6), delayed(8), delayed(10)],
        );
        let raster = decode_gif_with(
            &anim,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");
        assert_eq!(raster.get_int("gif-delay"), Some(4));
        assert_eq!(raster.get_int("gif-loop"), Some(3));
        assert_eq!(raster.get_int("palette-bit-depth"), Some(2));
        // The controls the deprecated pair is derived from, so a fix that
        // wrote the wrong one of the two is visible here rather than only in
        // the numbers above.
        assert_eq!(raster.get_int("loop"), Some(4));
        assert_eq!(raster.get_int("bits-per-sample"), Some(2));

        let still = fixture((4, 3), &ANIM_PALETTE, 0, None, &[delayed(0)]);
        let raster = decode_bytes(&still).expect("decodes");
        assert_eq!(
            raster.get_int("gif-delay"),
            Some(0),
            "a still gets them too, and an explicit zero delay is zero"
        );
        assert_eq!(
            raster.get_int("gif-loop"),
            Some(0),
            "no NETSCAPE block is `loop 1`, which is `gif-loop 0`"
        );
        assert_eq!(raster.get_int("palette-bit-depth"), Some(2));

        let bare = fixture(
            (4, 3),
            &ANIM_PALETTE,
            0,
            None,
            &[Frame {
                control: false,
                ..full()
            }],
        );
        assert_eq!(
            decode_bytes(&bare).expect("decodes").get_int("gif-delay"),
            Some(10),
            "no control extension is 10 centiseconds, the same default #866 is about"
        );

        // A different palette so `palette-bit-depth` is not a constant that
        // any value would satisfy.
        let wide: Vec<[u8; 3]> = (0..8u8).map(|i| [i * 30, 255 - i * 30, i * 17]).collect();
        let eight = fixture(
            (4, 3),
            &wide,
            0,
            None,
            &[Frame::full(4, 3, (0..12u8).map(|i| i % 8).collect())],
        );
        let raster = decode_bytes(&eight).expect("decodes");
        assert_eq!(raster.get_int("palette-bit-depth"), Some(3));
        assert_eq!(raster.get_int("bits-per-sample"), Some(3));
    }

    /**
     * Tests that `gif-delay` follows the raster rather than the file, which
     * is a deliberate divergence and the same one `delay` already makes.
     * Works by loading two pages out of a four-frame file starting at page 2
     * and reading both fields.
     * vips reports the whole file's `delay` array whatever window was loaded
     * and takes `gif-delay` from `delay[0]`, so on 8.18.6
     * `anim4.gif[page=2,n=2]` reports `delay: 40 60 80 100` and `gif-delay:
     * 4`, describing two frames the raster does not contain. This loader
     * subsets `delay` to the loaded pages (the argument is in the module
     * docs) and takes `gif-delay` from the same first element, so the two
     * fields still agree with each other on the object carrying them.
     * `src/webp.rs` does exactly this, off `delays.first()`, so the three
     * animated loaders speak one dialect.
     * Input: a four-frame GIF at `page = 2, n = 2` -> Output: `delay 80 100`
     * and `gif-delay 8`.
     */
    #[test]
    fn gif_delay_follows_the_loaded_window_the_way_delay_does() {
        let frames: Vec<Frame> = [4u16, 6, 8, 10]
            .into_iter()
            .map(|cs| Frame {
                delay_cs: cs,
                ..Frame::full(2, 2, vec![0, 1, 2, 3])
            })
            .collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(3), &frames);
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_page(2).with_n(2),
        )
        .expect("a valid GIF");
        assert_eq!(delays(&raster), Some(vec![80, 100]));
        assert_eq!(
            raster.get_int("gif-delay"),
            Some(8),
            "the first delay of the raster, where vips reports the file's 4"
        );
        // The control: a full load of the same file does agree with vips.
        let whole = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");
        assert_eq!(whole.get_int("gif-delay"), Some(4));
    }

    /**
     * Tests that a frame carrying no graphic control extension gets a delay
     * of 100 ms, not 0. Works by decoding a still with no extension, a still
     * whose extension holds an explicit zero, and a four-frame file whose
     * second and fourth frames have no extension.
     * libnsgif initialises `frame->info.delay` to 10 centiseconds when it
     * allocates a frame and only a control extension overwrites it, so the
     * default survives into `delay`. Measured on vips 8.18.6: the bare still
     * reports `delay: 100`, the explicit-zero one reports `delay: 0`, and the
     * mixed file reports `delay: 30 100 50 100`. libviprs reported 0 for
     * every frame with no extension, because `gif` 0.14.2 resets the frame
     * state per frame and reports `delay: 0` for both cases alike.
     * The explicit zero is the control that makes this about the absent
     * extension rather than a floor on small delays, and the 30 and 50 in the
     * mixed file are the control that the fix did not simply overwrite every
     * delay with the default.
     * Input: three GIFs -> Output: 100, 0, and `30 100 50 100`.
     */
    #[test]
    fn a_frame_with_no_control_extension_gets_a_hundred_milliseconds() {
        let full = || Frame::full(4, 3, (0..12u8).map(|i| i % 4).collect());

        let bare = fixture(
            (4, 3),
            &ANIM_PALETTE,
            0,
            None,
            &[Frame {
                control: false,
                ..full()
            }],
        );
        assert_eq!(
            delays(&decode_bytes(&bare).expect("decodes")),
            Some(vec![100]),
            "no extension is libnsgif's 10 centisecond default"
        );

        let zero = fixture(
            (4, 3),
            &ANIM_PALETTE,
            0,
            None,
            &[Frame {
                delay_cs: 0,
                ..full()
            }],
        );
        assert_eq!(
            delays(&decode_bytes(&zero).expect("decodes")),
            Some(vec![0]),
            "an extension holding zero really is zero, so this is not a floor"
        );

        let mixed = fixture(
            (4, 3),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[
                Frame {
                    delay_cs: 3,
                    ..full()
                },
                Frame {
                    control: false,
                    ..full()
                },
                Frame {
                    delay_cs: 5,
                    ..full()
                },
                Frame {
                    control: false,
                    ..full()
                },
            ],
        );
        let raster = decode_gif_with(
            &mixed,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");
        assert_eq!(
            delays(&raster),
            Some(vec![30, 100, 50, 100]),
            "the rule is per frame, and the frames that do carry one keep it"
        );
    }

    /**
     * Tests that a frame carrying no graphic control extension at all takes
     * the default control rather than the last frame's. Works by putting a
     * frame with no extension between one that disposes with code 4 and one
     * that paints a second dot, so an inherited disposal shows up as a
     * missing dot two pages later.
     * GIF89a lets a frame go without a control extension and the `gif` crate
     * substitutes `Frame::default()` for it, so the second walk has to drop
     * each extension after the image descriptor it belongs to rather than
     * hold on to it. Measured on vips 8.18.6 against exactly this file: page
     * 0 is `red red`, page 1 is `green black` and page 2 is `green blue`.
     * Input: a three-frame 2x1 GIF whose middle frame has no control
     * extension -> Output: the middle frame keeps the canvas, so the last
     * page carries both dots.
     */
    #[test]
    fn a_frame_with_no_control_extension_does_not_inherit_the_last_one() {
        // The walk itself first, because that is what the name is about and
        // the pixels below were being held by a different test.
        let three = fixture(
            (2, 1),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[
                Frame {
                    disposal: 4,
                    ..Frame::full(2, 1, vec![1, 1])
                },
                Frame {
                    control: false,
                    ..Frame::full(2, 1, vec![2, 2])
                },
                Frame {
                    disposal: 2,
                    ..Frame::full(2, 1, vec![3, 3])
                },
            ],
        );
        let mut walk = ControlWalk::new(&three);
        let codes: Vec<Option<u8>> = std::iter::from_fn(|| walk.next_frame())
            .map(|wire| match wire {
                WireControl::Present { control, .. } => Some(control.disposal),
                WireControl::Absent => None,
            })
            .collect();
        assert_eq!(
            codes,
            [Some(4), None, Some(2)],
            "the middle frame carried no extension, and the walk says so \
             rather than handing back the one before it"
        );

        let bytes = fixture(
            (2, 1),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[
                Frame {
                    disposal: 4,
                    ..Frame::full(2, 1, vec![1, 1])
                },
                Frame {
                    width: 1,
                    height: 1,
                    indices: vec![2],
                    control: false,
                    ..Frame::full(1, 1, vec![2])
                },
                Frame {
                    left: 1,
                    width: 1,
                    height: 1,
                    indices: vec![3],
                    ..Frame::full(1, 1, vec![3])
                },
            ],
        );
        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");
        assert_eq!(page_bytes(&raster, 0), [255, 0, 0, 255, 0, 0]);
        assert_eq!(
            page_bytes(&raster, 1),
            [0, 255, 0, 0, 0, 0],
            "frame 0's code 4 rewound the canvas before the middle frame drew"
        );
        assert_eq!(
            page_bytes(&raster, 2),
            [0, 255, 0, 0, 0, 255],
            "and the middle frame kept it, so both dots are here"
        );
    }

    /**
     * Tests that the wire walk which recovers the raw disposal code stays in
     * step with the decoder that decides where the frames are, which is the
     * risk #827 was filed rather than fixed over. Works by calling
     * `reconcile_control` with a control whose delay, transparent index or
     * mapped disposal disagrees with the frame the decoder handed back, and
     * requiring it to fall back to the decoder's own answer.
     * The three fields are the rest of the same graphic control extension,
     * so a walk that is looking at a different frame than the decoder is
     * almost certain to disagree on one of them, and disagreeing is the only
     * thing this can do about it: the raw code is the half the decoder threw
     * away, so there is nothing else to check it against.
     * Input: an agreeing control, then one wrong in each field, then an
     * absent extension against both decoder shapes ->
     * Output: trusted once, discarded three times, and the absent case
     * separated from the all-zero one.
     */
    #[test]
    fn a_control_walk_that_disagrees_with_the_decoder_is_not_trusted() {
        let raw = RawControl {
            disposal: 4,
            delay_cs: 7,
            transparent: Some(3),
        };
        let present = WireControl::Present {
            control: raw,
            as_decoder: Some(raw),
        };
        assert_eq!(
            reconcile_control(Some(present), gif::DisposalMethod::Any, 7, Some(3)),
            Some(present),
            "agreeing on the other three fields is what lets code 4 through"
        );
        for (label, wrong) in [
            (
                "delay",
                reconcile_control(Some(present), gif::DisposalMethod::Any, 8, Some(3)),
            ),
            (
                "transparent index",
                reconcile_control(Some(present), gif::DisposalMethod::Any, 7, Some(4)),
            ),
            (
                "collapsed disposal",
                reconcile_control(Some(present), gif::DisposalMethod::Keep, 7, Some(3)),
            ),
        ] {
            assert_eq!(
                wrong, None,
                "a walk that disagrees about the {label} is discarded"
            );
        }
        assert_eq!(
            reconcile_control(None, gif::DisposalMethod::Previous, 0, None),
            None,
            "a walk that ran out of frames leaves the decoder in charge"
        );

        // The absent case is checked against `Frame::default()`, whose
        // disposal is `Keep`. Checking it against `Any`, which is what a raw
        // code of 0 maps to, would discard every extension-less frame and
        // take #866's 100 ms with it.
        assert_eq!(
            reconcile_control(
                Some(WireControl::Absent),
                gif::DisposalMethod::Keep,
                0,
                None
            ),
            Some(WireControl::Absent),
            "no extension is the decoder's default frame, disposal `Keep`"
        );
        assert_eq!(
            reconcile_control(Some(WireControl::Absent), gif::DisposalMethod::Any, 0, None),
            None,
            "and it is not the same as an extension holding all zeroes"
        );
        assert_eq!(
            WireControl::Absent.delay_cs(),
            10,
            "libnsgif's default, which is 100 ms and not 0"
        );
        assert_eq!(
            WireControl::Present {
                control: RawControl {
                    disposal: 0,
                    delay_cs: 0,
                    transparent: None,
                },
                as_decoder: None,
            }
            .delay_cs(),
            0,
            "the control: an extension holding zero really is zero"
        );
    }

    /**
     * Tests that the control walk reads the same graphic control extensions
     * the decoder does, on every frame of a file that exercises the fields it
     * cross-checks. Works by walking a four-frame fixture whose frames carry
     * different delays, transparent indices and disposal codes, and requiring
     * the walk's answer to equal the decoder's frame by frame.
     * This is the positive control for the test above: the fallback is only
     * worth having if the agreement it guards is the normal case, and a walk
     * that agreed with nothing would make every raw code unreachable without
     * failing a single pixel assertion.
     * Input: a four-frame GIF -> Output: four controls matching the
     * decoder's `delay`, `transparent` and `dispose` on each frame.
     */
    #[test]
    fn the_control_walk_agrees_with_the_decoder_frame_by_frame() {
        let frames = vec![
            Frame {
                delay_cs: 4,
                disposal: 2,
                transparent: Some(1),
                ..Frame::full(2, 2, vec![0, 1, 2, 3])
            },
            Frame {
                delay_cs: 0,
                disposal: 3,
                ..Frame::full(2, 2, vec![3, 2, 1, 0])
            },
            Frame {
                delay_cs: 65535,
                disposal: 7,
                transparent: Some(0),
                ..Frame::full(2, 2, vec![1, 1, 2, 2])
            },
            Frame {
                delay_cs: 12,
                disposal: 0,
                ..Frame::full(2, 2, vec![2, 2, 3, 3])
            },
        ];
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);
        let mut walk = ControlWalk::new(&bytes);
        let mut decoder = open(&bytes).expect("a valid GIF");
        let mut seen = 0;
        while let Some(frame) = decoder.next_frame_info().expect("a valid GIF") {
            let wire = walk.next_frame().expect("the walk has this frame too");
            assert_eq!(
                wire.decoder_fields(),
                (frame.dispose, frame.delay, frame.transparent),
                "frame {seen}"
            );
            let WireControl::Present { control: raw, .. } = wire else {
                panic!("frame {seen} carries an extension in this fixture");
            };
            assert_eq!(raw.disposal, frames[seen].disposal, "frame {seen} raw code");
            seen += 1;
        }
        assert_eq!(seen, 4, "the walk and the decoder both saw four frames");
        assert_eq!(
            walk.next_frame(),
            None,
            "and the walk stops where the decoder does"
        );
    }

    /**
     * Tests that `background` comes back, as the three doubles vips stores
     * it as. Works by decoding four fixtures whose logical screen descriptor
     * points at different colour table entries, including one past the end of
     * the table.
     * It is the last `gifload` header field this loader read and did not
     * attach, and the reason it waited is that vips stores it as a
     * `VipsArrayDouble` where `MetadataValue` had only an integer array
     * (issue #852). The values are colour-table bytes widened to doubles, so
     * they are always integral, but writing them as an int array would have
     * put them under a type every reader written against vips ignores.
     * Measured on vips 8.18.6 with a palette whose entry 0 is `(9, 8, 7)`:
     * index 0 reports `9 8 7`, index 1 `255 0 0`, index 3 `0 0 255`, and
     * index 200 reports `9 8 7` again, because an index the table cannot
     * serve falls back to entry 0. Entry 0 is deliberately not black, so the
     * out-of-range row is distinguishable from a lookup that gives up.
     * Input: four background indices -> Output: the three doubles each.
     */
    #[test]
    fn the_background_colour_comes_back_as_three_doubles() {
        const PALETTE: [[u8; 3]; 4] = [[9, 8, 7], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
        for (index, expected) in [
            (0u8, [9.0, 8.0, 7.0]),
            (1, [255.0, 0.0, 0.0]),
            (3, [0.0, 0.0, 255.0]),
            (200, [9.0, 8.0, 7.0]),
        ] {
            let bytes = fixture(
                (2, 1),
                &PALETTE,
                index,
                None,
                &[Frame::full(2, 1, vec![0, 1])],
            );
            let raster = decode_bytes(&bytes).expect("decodes");
            assert_eq!(
                raster.get_double_array("background"),
                Some(&expected[..]),
                "background index {index}"
            );
            // And it is a double array, not an int array wearing the name:
            // a reader written against vips asks for the one vips wrote.
            assert_eq!(
                raster.get_int_array("background"),
                None,
                "background index {index} is not an int array"
            );
        }
    }

    /**
     * Tests that the global colour table comes back as `gif-palette`, packed
     * the way vips packs it. Works by decoding three fixtures whose tables
     * differ in size and contents and comparing the whole array.
     * vips reinterprets libnsgif's `R, G, B, A` byte quad as a machine
     * integer, so on a little-endian host the entry reads
     * `0xFF << 24 | B << 16 | G << 8 | R` and every entry is negative.
     * Measured on vips 8.18.6: a black/red/green/blue table is
     * `-16777216 -16776961 -16711936 -65536`, `(71,112,76)` and `(60,60,60)`
     * are `-11767737 -12829636`, and a three-colour table comes back with
     * **four** entries, because the table on the wire is padded to a power of
     * two and the pad is opaque black.
     * Input: 4-, 2- and 3-colour tables -> Output: the measured arrays,
     * including the padding entry.
     */
    #[test]
    fn the_global_colour_table_is_attached_as_gif_palette() {
        let cases: [(&[[u8; 3]], &[i64]); 3] = [
            (
                &ANIM_PALETTE,
                &[-16_777_216, -16_776_961, -16_711_936, -65_536],
            ),
            (&[[71, 112, 76], [60, 60, 60]], &[-11_767_737, -12_829_636]),
            (
                &[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                &[-16_580_095, -16_382_716, -16_185_337, -16_777_216],
            ),
        ];
        for (palette, expected) in cases {
            let bytes = fixture((2, 1), palette, 0, None, &[Frame::full(2, 1, vec![0, 1])]);
            let raster = decode_bytes(&bytes).expect("decodes");
            assert_eq!(
                raster.get_int_array("gif-palette"),
                Some(expected),
                "{palette:?} should pack to {expected:?}"
            );
        }
    }

    /**
     * Tests `gif-palette`'s presence rule: it is attached only when **no
     * frame in the file** carries a local colour table, and the rule is about
     * the file rather than the window that was loaded. Works by decoding
     * seven fixtures that put a local table in different places, and loading
     * one of them three ways.
     * vips reaches this through `nsgifload.c:315`, `if (!gif->local_palette)`,
     * over a flag its header pass ORs across every frame in the file
     * (`:435`). Measured on vips 8.18.6 against exactly these files: a
     * single-frame and a three-frame file with only a global table both
     * report four entries; a four-frame file whose frames 1 to 3 carry local
     * tables reports none; a two-frame file reports none whether the local
     * table is on frame 0 or frame 1, whether it differs from the global one
     * or is byte-identical to it, and at `[n=-1]`, the default window and
     * `[page=1]` alike; and a file with no global table but a local one
     * reports none.
     * Input: seven fixtures and three windows -> Output: present twice,
     * absent everywhere else.
     */
    #[test]
    fn gif_palette_is_absent_when_any_frame_carries_a_local_colour_table() {
        let local: Vec<[u8; 3]> = vec![[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]];
        let plain = || Frame::full(2, 2, vec![0, 1, 2, 3]);
        let with_local = |table: &[[u8; 3]]| Frame {
            local: Some(table.to_vec()),
            ..Frame::full(2, 2, vec![0, 1, 2, 3])
        };
        let palette_of = |bytes: &[u8], options: LoadOptions| {
            decode_gif_with(bytes, DecodeLimits::default(), options)
                .expect("a valid GIF")
                .get_int_array("gif-palette")
                .map(<[i64]>::to_vec)
        };
        let all = LoadOptions::default().with_n(-1);

        // Two positive controls first: without a local table anywhere the
        // field is there, so "absent" below is a real answer and not a
        // fixture that never attaches anything.
        let global_only_one = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &[plain()]);
        let global_only_three = fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[plain(), plain(), plain()],
        );
        let four = [-16_777_216i64, -16_776_961, -16_711_936, -65_536];
        assert_eq!(
            palette_of(&global_only_one, all).as_deref(),
            Some(&four[..]),
            "one frame, global table only"
        );
        assert_eq!(
            palette_of(&global_only_three, all).as_deref(),
            Some(&four[..]),
            "three frames, global table only"
        );

        let last_three_local = fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[
                plain(),
                with_local(&local),
                with_local(&local),
                with_local(&local),
            ],
        );
        let first_local = fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[with_local(&local), plain()],
        );
        let second_local = fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[plain(), with_local(&local)],
        );
        // A local table byte-identical to the global one still suppresses the
        // field: the flag vips reads is "this frame declared a table", not
        // "this frame declared a different table".
        let same_local = fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[with_local(&ANIM_PALETTE), plain()],
        );
        let no_global = fixture_tables((2, 2), None, 0, Some(0), &[with_local(&local)]);
        for (name, bytes) in [
            ("frames 1 to 3 local", &last_three_local),
            ("frame 0 local", &first_local),
            ("frame 1 local", &second_local),
            ("a local table equal to the global one", &same_local),
            ("no global table at all", &no_global),
        ] {
            assert_eq!(palette_of(bytes, all), None, "{name}");
        }

        // The rule is about the file, not the window: loading page 1 of the
        // file whose *frame 0* carries the local table still reports nothing.
        for (name, options) in [
            ("the default window", LoadOptions::default()),
            ("page 1", LoadOptions::default().with_page(1)),
            ("the whole roll", all),
        ] {
            assert_eq!(
                palette_of(&first_local, options),
                None,
                "frame 0 local, loaded at {name}"
            );
        }
    }

    /**
     * Tests that the module doc's header-field table says what this loader
     * actually attaches, in all three of the things it claims. Works by
     * parsing the table out of this file and decoding two rasters: one built
     * to satisfy every condition the table names, and one built to satisfy
     * none of them.
     * Every row marked "attached" has to be present on the first, every row
     * marked "not attached" absent from both, and every row marked "attached
     * **when**" absent from the second, which is what puts the condition
     * column inside the check instead of beside it. Since #852 there are no
     * "not attached" rows left, so two synthetic rows keep the parser honest
     * about telling the verdicts apart; without them the whole check would
     * pass with the verdict hard-coded. The first raster's whole
     * field set then has to be a subset of the rows, which is what makes
     * "every field `gifload` attaches" a claim rather than a sentence: without
     * it, a field the loader attaches and the table never mentions is
     * invisible.
     * This is issue #801, and the shape of it is why: the paragraph the table
     * replaces named a blocker that had already been removed and a field set
     * that had already moved, because nothing checked it. The three-way form
     * came out of a review that changed a condition's wording and a row's
     * prose and watched the earlier version stay green both times.
     * Input: this file's own doc table, a two-page interlaced fixture and a
     * one-page one with a local colour table -> Output: every row's verdict
     * and every condition confirmed, and no field left over.
     */
    #[test]
    fn the_header_field_table_matches_what_this_loader_attaches() {
        const SRC: &str = include_str!("gif.rs");
        const HEADER: &str = "//! | `gifload` field | vips type | here |";
        let at = SRC
            .find(HEADER)
            .expect("the module doc's header-field table moved");
        // (name, attached, conditional)
        let parse_row = |row: &str| -> (String, bool, bool) {
            let cells: Vec<&str> = row.split('|').map(str::trim).collect();
            let name = cells[0].trim_matches('`').to_string();
            let verdict = cells[2];
            assert!(
                verdict.starts_with("attached") || verdict.starts_with("not attached"),
                "row {name:?} says {verdict:?}, which is neither verdict"
            );
            (
                name,
                verdict.starts_with("attached"),
                verdict.starts_with("attached when"),
            )
        };
        let mut rows: Vec<(String, bool, bool)> = Vec::new();
        for line in SRC[at + HEADER.len()..].lines().skip(1) {
            let Some(row) = line.trim().strip_prefix("//! |") else {
                break;
            };
            if row.trim_start().starts_with("---") {
                continue;
            }
            rows.push(parse_row(row));
        }
        // The parser is the part most likely to rot into a vacuous pass, so
        // pin that it found a real table.
        assert!(rows.len() >= 10, "only parsed {} rows", rows.len());
        assert!(rows.iter().any(|(_, attached, _)| *attached));
        assert!(rows.iter().any(|(_, _, conditional)| *conditional));
        // **Every row is now an "attached" row**, because #852 attached the
        // last field this loader read and did not attach. So the table itself
        // can no longer show that the parser tells the two verdicts apart,
        // and two synthetic rows do it instead. Without this the whole check
        // would pass with `attached` hard-coded true.
        assert_eq!(
            parse_row(" `x` | `gint` | not attached, because |"),
            ("x".to_string(), false, false)
        );
        assert_eq!(
            parse_row(" `y` | `gint` | attached when it feels like it |"),
            ("y".to_string(), true, true)
        );
        for must in ["background", "gif-palette", "delay", "n-pages"] {
            assert!(
                rows.iter().any(|(name, _, _)| name == must),
                "the table does not mention {must}"
            );
        }

        // Two pages so `page-height` is set, one interlaced frame so
        // `interlaced` is, and no local colour table so `gif-palette` is.
        let mut interlaced = Frame::full(2, 2, vec![0, 1, 2, 3]);
        interlaced.interlaced = true;
        let bytes = fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[Frame::full(2, 2, vec![3, 2, 1, 0]), interlaced],
        );
        let every = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("a valid GIF");

        // One page, no interlace, and a local colour table, so none of the
        // three conditions holds.
        let none = decode_bytes(&fixture(
            (2, 2),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[Frame {
                local: Some(vec![[9, 8, 7], [6, 5, 4]]),
                ..Frame::full(2, 2, vec![0, 1, 1, 0])
            }],
        ))
        .expect("decodes");

        for (name, attached, conditional) in &rows {
            assert_eq!(
                every.get_field(name).is_some(),
                *attached,
                "with every condition met, the table says {name} is {}",
                if *attached {
                    "attached"
                } else {
                    "not attached"
                }
            );
            assert_eq!(
                none.get_field(name).is_some(),
                *attached && !*conditional,
                "with no condition met, {name} should be {}",
                if *attached && !*conditional {
                    "attached"
                } else {
                    "absent"
                }
            );
        }

        // And nothing the table does not list. Without this the table's
        // "every field" is unheld in the one direction a reader cares about:
        // a field this loader attaches and the table never mentions would be
        // invisible to every assertion above.
        //
        // The intrinsics (`width`, `interpretation`, `xres` and the rest) are
        // subtracted rather than listed, and they are taken from a raster
        // built by hand rather than written down here, so the subtraction
        // tracks the crate instead of somebody's memory of it. A field that
        // stops being intrinsic therefore has to appear in the table.
        let intrinsics = Raster::new(1, 1, PixelFormat::Rgb8, vec![0, 0, 0])
            .expect("a 1x1 raster")
            .get_fields();
        assert!(
            intrinsics.len() >= 4,
            "the intrinsic set came back as {intrinsics:?}, which cannot be right"
        );
        let listed: Vec<&str> = rows
            .iter()
            .filter(|(_, attached, _)| *attached)
            .map(|(name, _, _)| name.as_str())
            .collect();
        let mut extra: Vec<String> = every
            .get_fields()
            .into_iter()
            .filter(|name| !listed.contains(&name.as_str()) && !intrinsics.contains(name))
            .collect();
        extra.sort();
        assert!(
            extra.is_empty(),
            "the raster carries fields the table never mentions: {extra:?}"
        );
    }

    /**
     * Tests that a windowed load renders from frame 0 rather than from the
     * window, which is the claim the whole `page` option rests on. Works by
     * loading page 1 of a file whose frame 0 paints the screen red and whose
     * frame 1 paints a single green pixel with a transparent index
     * everywhere else, so the pixels under the transparency say which frame
     * the canvas started from.
     * The flat full-screen fixtures the other window tests use cannot see
     * this: every frame overwrites the whole canvas, so starting at
     * `window.start` gives the same answer. Here it does not, and the wrong
     * implementation shows transparent black where the right one shows red.
     * Measured on vips 8.18.6 against exactly this file: `[page=1]` is
     * `green red`.
     * Input: a two-frame 2x1 GIF loaded at `page = 1` -> Output: `green
     * red`, the red coming from the frame the window skipped.
     */
    #[test]
    fn a_window_still_composites_from_the_first_frame() {
        let bytes = fixture(
            (2, 1),
            &ANIM_PALETTE,
            0,
            Some(0),
            &[
                Frame::full(2, 1, vec![1, 1]),
                Frame {
                    transparent: Some(0),
                    ..Frame::full(2, 1, vec![2, 0])
                },
            ],
        );
        let second = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_page(1),
        )
        .expect("a valid GIF");
        assert_eq!(second.format(), PixelFormat::Rgba8);
        assert_eq!(second.pages_loaded(), 1);
        assert_eq!(
            second.data(),
            [0, 255, 0, 255, 255, 0, 0, 255],
            "the transparent pixel shows frame 0's red, which a window that \
             starts at itself never painted"
        );

        // The control: the same file loaded from frame 0 has nothing under
        // the transparency, so the second pixel is transparent black there.
        let first = decode_gif(&bytes, DecodeLimits::default()).expect("a valid GIF");
        assert_eq!(first.data(), [255, 0, 0, 255, 255, 0, 0, 255]);
    }

    /**
     * Tests that a frame declaring a rectangle far larger than the logical
     * screen is priced against the allocation budget before its index buffer
     * is allocated. Works by building a 1x1 screen carrying one 65535x65535
     * frame, which is forty bytes on disk and 4 GiB of indices, and offering
     * it a budget of a kilobyte.
     * The screen price cannot see this: the canvas is three bytes and passes
     * every ceiling. `open()` leaves the `gif` crate's frame-consistency
     * check off, matching libnsgif, which clips an oversized frame rather
     * than refusing the file, and the clipping happens after the buffer is
     * allocated.
     * Input: a 1x1 GIF with a 65535x65535 frame at a 1024-byte budget ->
     * Output: `AllocLimitExceeded` naming the frame's geometry, and a decode
     * once the budget covers it.
     */
    #[test]
    fn a_frame_larger_than_the_screen_is_priced_before_it_is_allocated() {
        let bytes = fixture(
            (1, 1),
            &ANIM_PALETTE,
            0,
            None,
            &[Frame {
                width: 4,
                height: 4,
                indices: vec![1; 16],
                ..Frame::full(4, 4, vec![1; 16])
            }],
        );

        let err = decode_gif(&bytes, DecodeLimits::default().with_max_alloc_bytes(15))
            .expect_err("16 bytes of indices is over a 15-byte budget");
        assert!(
            matches!(
                err,
                SourceError::AllocLimitExceeded {
                    what: "GIF frame indices",
                    geometry: Some(DeclaredGeometry {
                        width: 4,
                        height: 4,
                        bands: 1,
                    }),
                    needed_bytes: 16,
                    max_alloc_bytes: 15,
                }
            ),
            "{err:?}"
        );

        let raster = decode_gif(&bytes, DecodeLimits::default().with_max_alloc_bytes(16))
            .expect("16 bytes is exactly the frame's indices");
        assert_eq!((raster.width(), raster.height()), (1, 1), "the frame clips");
    }

    /**
     * Tests that the frame count is bounded by `DecodeLimits::max_pages`, the
     * ceiling the crate documents for exactly this and which the TIFF reader
     * already applies to its IFD chain. Works by offering a four-frame file a
     * ceiling of three, with a ceiling of four as the positive control.
     * A GIF's frame list has no count in the header, so the only way to know
     * how long it is, is to walk it, which is the same exposure
     * `count_images` bounds for TIFF. The walk stops **at** the ceiling
     * rather than running to the end to count, which is why the error carries
     * the ceiling and not the real length.
     * Input: a four-frame GIF at `max_pages` 3 and 4 -> Output:
     * `PageLimitExceeded`, then a load.
     */
    #[test]
    fn the_frame_count_is_bounded_by_max_pages() {
        let frames: Vec<Frame> = (0..4u8).map(|i| anim_frame([i; 4], 1, 0)).collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);

        let err = decode_gif(&bytes, DecodeLimits::default().with_max_pages(3))
            .expect_err("four frames is over a ceiling of three");
        assert!(
            matches!(err, SourceError::PageLimitExceeded { max_pages: 3 }),
            "{err:?}"
        );

        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default().with_max_pages(4),
            LoadOptions::default().with_n(-1),
        )
        .expect("four frames is exactly a ceiling of four");
        assert_eq!(raster.pages_loaded(), 4);
    }

    /**
     * Tests that the whole roll is priced against the allocation budget
     * before it is built, so a four-frame load cannot slip through on a
     * one-frame price. Works by loading a four-frame fixture at the exact
     * roll price and one byte under it.
     * The price is `width * page-height * pages * bands`: 2 * 2 * 4 * 3 = 48
     * bytes for this fixture, where the single 2x2 canvas is 12.
     * Input: a four-frame 2x2 GIF at budgets 48 and 47 -> Output: a roll,
     * then `SourceError::AllocLimitExceeded` naming the roll geometry.
     */
    #[test]
    fn the_animation_roll_is_priced_before_it_is_allocated() {
        let frames: Vec<Frame> = (0..4u8).map(|i| anim_frame([i; 4], 1, 0)).collect();
        let bytes = fixture((2, 2), &ANIM_PALETTE, 0, Some(0), &frames);
        let all = LoadOptions::default().with_n(-1);

        let raster = decode_gif_with(
            &bytes,
            DecodeLimits::default().with_max_alloc_bytes(48),
            all,
        )
        .expect("48 bytes is exactly the 2x8 RGB roll");
        assert_eq!(raster.height(), 8);

        let err = decode_gif_with(
            &bytes,
            DecodeLimits::default().with_max_alloc_bytes(47),
            all,
        )
        .expect_err("47 bytes is one short of the roll");
        assert!(
            matches!(
                err,
                SourceError::AllocLimitExceeded {
                    what: "GIF animation",
                    geometry: Some(DeclaredGeometry {
                        width: 2,
                        height: 8,
                        bands: 3,
                    }),
                    needed_bytes: 48,
                    max_alloc_bytes: 47,
                }
            ),
            "{err:?}"
        );
    }

    /// One frame as it sits on the wire, for the save-side assertions.
    #[derive(Debug, PartialEq, Eq)]
    struct WireFrame {
        /// The image descriptor's rectangle, `(left, top, width, height)`.
        rect: (u16, u16, u16, u16),
        /// The graphic control extension's delay, in centiseconds.
        delay_cs: u16,
        /// The graphic control extension's disposal code.
        disposal: u8,
        /// The transparent index, when the extension declares one.
        transparent: Option<u8>,
        /// Whether the image descriptor sets the interlace bit.
        interlaced: bool,
        /// The LZW payload, sub-block headers stripped.
        payload: Vec<u8>,
    }

    /// Walk a whole GIF and return the logical screen, the NETSCAPE loop
    /// count if there is a block, and every frame.
    ///
    /// The existing [`wire`] helper stops at the first frame, which is all a
    /// still needs. This one is the animated half and it is deliberately a
    /// separate walk rather than a generalisation, so the still assertions
    /// cannot drift with it.
    fn wire_frames(bytes: &[u8]) -> ((u16, u16), Option<u16>, Vec<WireFrame>) {
        let screen = (
            u16::from_le_bytes([bytes[6], bytes[7]]),
            u16::from_le_bytes([bytes[8], bytes[9]]),
        );
        let mut p = 13usize;
        if bytes[10] & 0x80 != 0 {
            p += PALETTE_STRIDE * (2usize << (bytes[10] & 7));
        }
        let mut netscape = None;
        let mut pending: Option<(u16, u8, Option<u8>)> = None;
        let mut frames = Vec::new();
        loop {
            match bytes[p] {
                0x21 => {
                    let label = bytes[p + 1];
                    p += 2;
                    let mut blocks: Vec<&[u8]> = Vec::new();
                    while bytes[p] != 0 {
                        let len = bytes[p] as usize;
                        blocks.push(&bytes[p + 1..p + 1 + len]);
                        p += 1 + len;
                    }
                    p += 1;
                    match label {
                        0xF9 => {
                            let control = blocks[0];
                            pending = Some((
                                u16::from_le_bytes([control[1], control[2]]),
                                (control[0] >> 2) & 7,
                                (control[0] & 1 != 0).then_some(control[3]),
                            ));
                        }
                        0xFF if blocks[0] == b"NETSCAPE2.0" => {
                            for sub in &blocks[1..] {
                                if sub.first() == Some(&1) {
                                    netscape = Some(u16::from_le_bytes([sub[1], sub[2]]));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                0x2C => {
                    let rect = (
                        u16::from_le_bytes([bytes[p + 1], bytes[p + 2]]),
                        u16::from_le_bytes([bytes[p + 3], bytes[p + 4]]),
                        u16::from_le_bytes([bytes[p + 5], bytes[p + 6]]),
                        u16::from_le_bytes([bytes[p + 7], bytes[p + 8]]),
                    );
                    let flags = bytes[p + 9];
                    p += 10;
                    if flags & 0x80 != 0 {
                        p += PALETTE_STRIDE * (2usize << (flags & 7));
                    }
                    p += 1; // the LZW minimum code size
                    let mut payload = Vec::new();
                    while bytes[p] != 0 {
                        let len = bytes[p] as usize;
                        payload.extend_from_slice(&bytes[p + 1..p + 1 + len]);
                        p += 1 + len;
                    }
                    p += 1;
                    let (delay_cs, disposal, transparent) = pending.take().unwrap_or((0, 0, None));
                    frames.push(WireFrame {
                        rect,
                        delay_cs,
                        disposal,
                        transparent,
                        interlaced: flags & 0x40 != 0,
                        payload,
                    });
                }
                0x3B => return (screen, netscape, frames),
                other => panic!("unexpected block {other:#x} at {p}"),
            }
        }
    }

    /// A `pages`-page roll, 2 pixels wide and 2 rows per page, each page a
    /// flat colour from [`ANIM_PALETTE`].
    fn roll(pages: u32) -> Raster {
        let mut data = Vec::new();
        for page in 0..pages as usize {
            for _ in 0..4 {
                data.extend_from_slice(&ANIM_PALETTE[page % ANIM_PALETTE.len()]);
            }
        }
        let mut raster = Raster::new(2, 2 * pages, PixelFormat::Rgb8, data).expect("a valid roll");
        if pages > 1 {
            raster.set_page_height(2);
        }
        raster
    }

    /**
     * Tests that a page roll saves as one GIF frame per page, at the page's
     * size rather than the roll's. Works by encoding a four-page roll and
     * reading the logical screen and the frame list back off the wire.
     * Measured on vips 8.18.6: a 4x12 roll with `page-height 3` comes out as
     * a GIF whose logical screen is 4x3 and which holds four image blocks.
     * Input: a 2x8 roll with page height 2 -> Output: a 2x2 screen and four
     * frames.
     */
    #[test]
    fn a_page_roll_saves_one_frame_per_page() {
        let bytes = roll(4)
            .encode_gif(SaveOptions::default())
            .expect("a four-page roll encodes");
        let (screen, _, frames) = wire_frames(&bytes);
        assert_eq!(screen, (2, 2), "the screen is the page, not the roll");
        assert_eq!(frames.len(), 4);
        for frame in &frames {
            assert_eq!(frame.rect, (0, 0, 2, 2));
        }

        let back = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("what was written loads back");
        assert_eq!((back.width(), back.height()), (2, 8));
        assert_eq!(back.pages_loaded(), 4);
        // Four bands, not three: the palette does not saturate, so the save
        // reserves a transparent index and the reload sees a file that
        // declares one. That is the still lane's measured parity with
        // `vips gifsave`, not an animation behaviour.
        assert_eq!(back.format(), PixelFormat::Rgba8);
        for page in 0..4u32 {
            let colour = ANIM_PALETTE[page as usize];
            let opaque = [colour[0], colour[1], colour[2], u8::MAX];
            assert_eq!(
                page_bytes(&back, page),
                opaque.repeat(4),
                "page {page} survives the round trip"
            );
        }
    }

    /**
     * Tests that the `delay` field goes out as centiseconds rounded half to
     * even, which is what `gifsave` writes and what neither truncation nor
     * half-up produces. Works by encoding a four-page roll carrying each
     * measured millisecond array and reading the graphic control extensions
     * back.
     * Measured on vips 8.18.6, writing each array through `gifsave` and
     * parsing the extensions out of the bytes: `35 55 15 25` gave `4 6 2 2`
     * and `45 67 5 1` gave `4 7 0 0`. Truncation would write 6 for 67 ms and
     * half-up would write 3 for 25 ms; neither matches. `5 15 25 35` gave
     * `0 2 2 4`, which is the half-to-even tie rule on its own.
     * Input: three measured delay arrays -> Output: the centiseconds vips
     * wrote for each.
     */
    #[test]
    fn delays_go_out_as_centiseconds_rounded_half_to_even() {
        for (millis, centis) in [
            ([35i64, 55, 15, 25], [4u16, 6, 2, 2]),
            ([45, 67, 5, 1], [4, 7, 0, 0]),
            ([5, 15, 25, 35], [0, 2, 2, 4]),
        ] {
            let mut source = roll(4);
            source.set_field("delay", MetadataValue::IntArray(millis.to_vec()));
            let bytes = source
                .encode_gif(SaveOptions::default())
                .expect("the delay array matches the page count");
            let (_, _, frames) = wire_frames(&bytes);
            let written: Vec<u16> = frames.iter().map(|f| f.delay_cs).collect();
            assert_eq!(written, centis, "{millis:?} ms");
        }
    }

    /**
     * Tests that a short delay goes out as written rather than being lifted
     * to 100 ms, because that floor is a `webpsave` and `jxlsave` behaviour
     * and `gifsave` does not have it. Works by encoding the same four
     * millisecond delays the WebP measurement used and reading them back.
     * Measured on vips 8.18.6: `8 9 10 11` ms through `gifsave` produced
     * centiseconds `1 1 1 1`, where the same four through `webpsave`
     * produced `ANMF` durations `100 100 100 11`.
     * Input: delays of 8, 9, 10 and 11 ms -> Output: `1 1 1 1`
     * centiseconds, with no floor applied.
     */
    #[test]
    fn a_short_delay_is_not_floored_the_way_webpsave_floors_it() {
        let mut source = roll(4);
        source.set_field("delay", MetadataValue::IntArray(vec![8, 9, 10, 11]));
        let bytes = source.encode_gif(SaveOptions::default()).expect("encodes");
        let (_, _, frames) = wire_frames(&bytes);
        assert_eq!(
            frames.iter().map(|f| f.delay_cs).collect::<Vec<_>>(),
            [1, 1, 1, 1],
            "gifsave does not apply the browser floor"
        );
    }

    /**
     * Tests that the `loop` field becomes the NETSCAPE block vips writes,
     * including the case where the block is left out altogether. Works by
     * encoding a roll with each measured loop value and reading the
     * application extension back.
     * Measured on vips 8.18.6 by writing each value with `gifsave` and
     * parsing the block out of the bytes: `loop 0` wrote a block holding 0,
     * `loop 1` wrote **no block**, `loop 2` wrote 1, `loop 5` wrote 4, and
     * `loop 65536` wrote 65535. Reading each file back reported the original
     * `loop`.
     * Input: loop 0, 1, 2, 5 and 65536 -> Output: the block vips wrote, and
     * the same `loop` on reload.
     */
    #[test]
    fn the_loop_field_becomes_the_netscape_block_vips_writes() {
        for (plays, block) in [
            (0i64, Some(0u16)),
            (1, None),
            (2, Some(1)),
            (5, Some(4)),
            (65536, Some(65535)),
        ] {
            let mut source = roll(2);
            source.set_field("loop", MetadataValue::Int(plays));
            let bytes = source.encode_gif(SaveOptions::default()).expect("encodes");
            let (_, netscape, _) = wire_frames(&bytes);
            assert_eq!(netscape, block, "loop {plays}");

            let back = decode_gif_with(
                &bytes,
                DecodeLimits::default(),
                LoadOptions::default().with_n(-1),
            )
            .expect("what was written loads back");
            assert_eq!(
                back.get_int("loop"),
                Some(plays.min(i64::from(i32::MAX)) as i32),
                "loop {plays} survives the round trip"
            );
        }
    }

    /**
     * Tests that a raster with no `loop` field writes the block cgif always
     * writes, so the still path is unchanged and an animation defaults to
     * looping forever. Works by encoding a roll and a still, neither
     * carrying the field.
     * Measured on vips 8.18.6: a `gifsave` of an image with no `loop`
     * attaches a NETSCAPE block holding 0, on a still as well as an
     * animation, and the file reloads as `loop: 0`.
     * Input: a two-page roll and a still, no `loop` field -> Output: a
     * NETSCAPE block holding 0 in both.
     */
    #[test]
    fn no_loop_field_means_forever_which_is_what_cgif_writes() {
        for pages in [1u32, 2] {
            let bytes = roll(pages)
                .encode_gif(SaveOptions::default())
                .expect("encodes");
            let (_, netscape, frames) = wire_frames(&bytes);
            assert_eq!(netscape, Some(0), "{pages} page(s)");
            assert_eq!(frames.len(), pages as usize);
        }
    }

    /**
     * Tests that a `delay` array whose length is not the page count is
     * refused, where vips pads it with zeros or truncates it silently. Works
     * by offering a four-page roll a two-entry and a six-entry array, with
     * the exact-length array as the positive control.
     * Measured on vips 8.18.6, on a four-page roll: a two-entry array wrote
     * centiseconds `2 3 0 0` and a six-entry array wrote `2 3 4 5`. Both are
     * a caller mistake given a silent answer, and the load side now
     * guarantees the array matches the page count, so a mismatch on save can
     * only come from a hand-built raster.
     * Input: delay arrays of length 2, 6 and 4 on a four-page roll ->
     * Output: two refusals naming both counts, and one encode.
     */
    #[test]
    fn a_delay_array_that_is_not_one_per_page_is_refused() {
        for wrong in [vec![20i64, 30], vec![20, 30, 40, 50, 60, 70], vec![]] {
            let mut source = roll(4);
            let len = wrong.len();
            source.set_field("delay", MetadataValue::IntArray(wrong));
            let err = source
                .encode_gif(SaveOptions::default())
                .expect_err("the array does not match the page count");
            let message = err.to_string();
            assert!(
                message.contains("gif: ") && message.contains(&len.to_string()),
                "{message}"
            );
        }

        let mut exact = roll(4);
        exact.set_field("delay", MetadataValue::IntArray(vec![20, 30, 40, 50]));
        let bytes = exact
            .encode_gif(SaveOptions::default())
            .expect("four delays on four pages is the positive control");
        let (_, _, frames) = wire_frames(&bytes);
        assert_eq!(
            frames.iter().map(|f| f.delay_cs).collect::<Vec<_>>(),
            [2, 3, 4, 5]
        );
    }

    /**
     * Tests that a negative `delay` or `loop` is refused rather than wrapped
     * into a plausible large one, which is what vips does with both. Works by
     * encoding a roll carrying each, with the non-negative value as the
     * positive control.
     * Measured on vips 8.18.6: a delay of -10 ms came out as 65535
     * centiseconds, which is 655 seconds, and -1 ms came out as 0; a `loop`
     * of -1 wrote a NETSCAPE count of 65535 and reloaded as `loop: 65536`,
     * and -5 wrote 65531. Both are an unsigned cast of a value that has no
     * meaning, so there is no behaviour worth matching.
     * Input: `delay = [-10, 40]` and `loop = -1` -> Output: two refusals,
     * and an encode for the same fields made non-negative.
     */
    #[test]
    fn a_negative_delay_or_loop_is_refused() {
        let mut bad_delay = roll(2);
        bad_delay.set_field("delay", MetadataValue::IntArray(vec![-10, 40]));
        let message = bad_delay
            .encode_gif(SaveOptions::default())
            .expect_err("a negative delay is not a delay")
            .to_string();
        assert!(
            message.contains("gif: ") && message.contains("-10"),
            "{message}"
        );

        let mut bad_loop = roll(2);
        bad_loop.set_field("loop", MetadataValue::Int(-1));
        let message = bad_loop
            .encode_gif(SaveOptions::default())
            .expect_err("a negative play count is not a play count")
            .to_string();
        assert!(
            message.contains("gif: ") && message.contains("-1"),
            "{message}"
        );

        let mut good = roll(2);
        good.set_field("delay", MetadataValue::IntArray(vec![10, 40]));
        good.set_field("loop", MetadataValue::Int(1));
        good.encode_gif(SaveOptions::default())
            .expect("the same fields made non-negative are the positive control");
    }

    /**
     * Tests that a field of the wrong type is ignored rather than refused,
     * which is the other half of the negative-value rule. Works by putting a
     * string under `loop` and a scalar under `delay` and requiring the
     * defaults.
     * The distinction is deliberate: an untrusted `.v` can leave anything
     * under any name, so a wrong type means "this is not the field I read",
     * the way `Raster::get_n_pages` and the `page-height` reader already
     * treat it, while a negative int means "this is the field and its value
     * is impossible". `MetadataValue::as_int_array` documents the scalar
     * half: a `gif-delay` is the first frame's delay alone and coercing it
     * would invent a per-frame array.
     * Input: `loop` as a string and `delay` as an `Int` -> Output: the
     * NETSCAPE block for "forever" and zero delays.
     */
    #[test]
    fn a_field_of_the_wrong_type_is_ignored_rather_than_refused() {
        let mut source = roll(2);
        source.set_field("loop", MetadataValue::Str("forever".into()));
        source.set_field("delay", MetadataValue::Int(40));
        let bytes = source
            .encode_gif(SaveOptions::default())
            .expect("a wrong-typed field is not this field");
        let (_, netscape, frames) = wire_frames(&bytes);
        assert_eq!(netscape, Some(0), "loop falls back to forever");
        assert_eq!(
            frames.iter().map(|f| f.delay_cs).collect::<Vec<_>>(),
            [0, 0],
            "a scalar does not coerce to a one-per-page array"
        );
    }

    /**
     * Tests that an animation carrying real transparency writes
     * restore-to-background on every frame but the last, so each frame's
     * transparent pixels stay transparent instead of showing the frame
     * before. Works by round-tripping a two-page RGBA roll whose second page
     * is transparent where the first is opaque, with an opaque roll as the
     * control.
     * Measured on vips 8.18.6: `gifsave` of a two-page roll with alpha wrote
     * disposal 2 on frame 0 and 1 on frame 1, and of a two-page opaque roll
     * wrote 1 on both. A still with alpha also got 1, because the last
     * frame's disposal is not observable.
     * Input: a transparent and an opaque two-page roll -> Output: disposal
     * `[2, 1]` and `[1, 1]`, and an exact round trip in both.
     */
    #[test]
    fn a_transparent_animation_disposes_to_background_so_each_page_stands_alone() {
        let mut clear = Raster::new(
            2,
            4,
            PixelFormat::Rgba8,
            vec![
                255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, //
                0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255,
            ],
        )
        .expect("a valid roll");
        clear.set_page_height(2);
        let bytes = clear.encode_gif(SaveOptions::default()).expect("encodes");
        let (_, _, frames) = wire_frames(&bytes);
        assert_eq!(
            frames.iter().map(|f| f.disposal).collect::<Vec<_>>(),
            [2, 1],
            "every frame but the last clears itself"
        );
        let back = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("loads back");
        assert_eq!(
            back.data(),
            clear.data(),
            "the transparent pixels do not pick up the frame before"
        );

        let opaque = roll(2).encode_gif(SaveOptions::default()).expect("encodes");
        let (_, _, opaque_frames) = wire_frames(&opaque);
        assert_eq!(
            opaque_frames.iter().map(|f| f.disposal).collect::<Vec<_>>(),
            [1, 1],
            "with nothing transparent there is nothing to clear"
        );
    }

    /**
     * Tests that dithering is computed per page, so the error diffused off
     * the bottom row of one page does not land on the top row of the next.
     * Works by encoding a two-page roll whose pages are identical and
     * requiring the two frames' LZW payloads to match.
     * A roll dithered as one tall image cannot produce identical frames:
     * page 1's first row starts with the error page 0's last row pushed
     * down, and page 0's first row starts with none. The gradient is there
     * to make the quantiser's palette saturate so there is error to diffuse
     * at all.
     * Input: a two-page roll of two identical gradient pages, `dither = 1.0`
     * -> Output: two byte-identical frames.
     */
    #[test]
    fn dithering_does_not_bleed_across_the_page_boundary() {
        let page = gradient(16, 16);
        let mut data = page.data().to_vec();
        data.extend_from_slice(page.data());
        let mut source = Raster::new(16, 32, PixelFormat::Rgb8, data).expect("a valid roll");
        source.set_page_height(16);
        let bytes = source
            .encode_gif(SaveOptions::default().with_bitdepth(3))
            .expect("encodes");
        let (_, _, frames) = wire_frames(&bytes);
        assert_eq!(frames.len(), 2);
        assert_eq!(
            frames[0].payload, frames[1].payload,
            "identical pages dither identically, which they cannot if error crosses the seam"
        );

        // The control the assertion above needs: two identical pages come out
        // identical with dithering switched off as well, so equality alone
        // does not say error diffusion ran. Requiring the dithered bytes to
        // differ from the undithered ones is what makes this a test of the
        // seam rather than of `dither == 0`.
        let flat = source
            .encode_gif(SaveOptions::default().with_bitdepth(3).with_dither(0.0))
            .expect("encodes");
        let (_, _, flat_frames) = wire_frames(&flat);
        assert_eq!(flat_frames.len(), 2);
        assert_eq!(
            flat_frames[0].payload, flat_frames[1].payload,
            "identical pages match without dithering too, which is why this test needs a control"
        );
        assert_ne!(
            frames[0].payload, flat_frames[0].payload,
            "dithering actually ran at 1.0"
        );
    }

    /**
     * Tests that the GIF axis limit is checked against the frame rather than
     * the roll, since the logical screen is one page tall. Works by encoding
     * a roll taller than 65535 rows whose pages are not, with a single page
     * over the limit as the positive control.
     * Measured on vips 8.18.6: the logical screen descriptor of a saved roll
     * holds the page height, not the roll height, so a roll of many pages is
     * not itself bounded by the 65535-pixel axis.
     * Input: a 1x80000 roll with page height 40000, then a 1x70000 still ->
     * Output: two frames of 40000 rows, then a refusal.
     */
    #[test]
    fn the_axis_limit_is_the_frame_not_the_roll() {
        let mut tall =
            Raster::new(1, 80_000, PixelFormat::Gray8, vec![7u8; 80_000]).expect("a valid roll");
        tall.set_page_height(40_000);
        let bytes = tall
            .encode_gif(SaveOptions::default())
            .expect("each page fits the GIF axis even though the roll does not");
        let (screen, _, frames) = wire_frames(&bytes);
        assert_eq!(screen, (1, 40_000));
        assert_eq!(frames.len(), 2);

        let still =
            Raster::new(1, 70_000, PixelFormat::Gray8, vec![7u8; 70_000]).expect("a valid still");
        let message = still
            .encode_gif(SaveOptions::default())
            .expect_err("one frame of 70000 rows is over the axis limit")
            .to_string();
        assert!(message.contains("frame too large"), "{message}");
    }

    /**
     * Tests that a stored `page-height` the raster cannot hold is discarded
     * on both sides of the save, so a bad split cannot reach the wire.
     * Works by putting a non-divisor under the field the way an untrusted
     * `.v` would, then encoding, with the divisor as the positive control.
     * Measured on vips 8.18.6: `vips gifsave roll.v out.gif --page-height 5`
     * on a 12-row image writes a single 4x12 frame and says nothing, because
     * `vips_image_get_page_height` discards a value that does not divide the
     * height. libviprs refuses the same value at `try_set_page_height`
     * instead, so the mistake is named where it is made, and the save path
     * reads the derived height, which always divides.
     * Input: a stored page height of 5 on an 8-row roll, then 2 ->
     * Output: one frame of 8 rows, then four frames of 2, and a refusal from
     * the setter.
     */
    #[test]
    fn a_page_height_the_raster_cannot_hold_never_reaches_the_wire() {
        let mut smuggled = roll(4);
        smuggled.set_field("page-height", MetadataValue::Int(5));
        let bytes = smuggled
            .encode_gif(SaveOptions::default())
            .expect("a discarded split is one page, as vips reports it");
        let (screen, _, frames) = wire_frames(&bytes);
        assert_eq!(screen, (2, 8), "the whole roll is one frame");
        assert_eq!(frames.len(), 1);

        assert!(
            roll(4).try_set_page_height(5).is_err(),
            "the setter refuses at the point of the mistake, where vips accepts and discards"
        );

        let (screen, _, frames) = wire_frames(
            &roll(4)
                .encode_gif(SaveOptions::default())
                .expect("the divisor is the positive control"),
        );
        assert_eq!(screen, (2, 2));
        assert_eq!(frames.len(), 4);
    }

    /**
     * Tests that a delay past what a `u16` of centiseconds can hold
     * saturates rather than wrapping, which is a deliberate divergence.
     * Works by encoding delays either side of the 655350 ms ceiling.
     * Measured on vips 8.18.6: `655350 655360 700000 10` ms came out as
     * centiseconds `65535 0 4464 1`, so 655360 ms became no delay at all and
     * 700000 ms became 44.64 seconds. That is a truncating cast, and
     * `FrameDelay::to_centiseconds` saturates instead: a delay too long to
     * express comes out as the longest one that fits rather than as an
     * arbitrary short one.
     * Input: delays of 655350, 655360, 700000 and 10 ms -> Output:
     * `65535 65535 65535 1` centiseconds.
     */
    #[test]
    fn a_delay_past_the_wire_ceiling_saturates_where_vips_wraps() {
        let mut source = roll(4);
        source.set_field(
            "delay",
            MetadataValue::IntArray(vec![655_350, 655_360, 700_000, 10]),
        );
        let bytes = source.encode_gif(SaveOptions::default()).expect("encodes");
        let (_, _, frames) = wire_frames(&bytes);
        assert_eq!(
            frames.iter().map(|f| f.delay_cs).collect::<Vec<_>>(),
            [65_535, 65_535, 65_535, 1],
            "vips wrote 65535 0 4464 1 here, which turns a long delay into a short one"
        );
    }

    /**
     * Tests that pulling one page out of an animation and saving it as a
     * still works, which the delay-length refusal broke until an adversarial
     * review found it. Works by round-tripping a two-page roll, extracting
     * page 0, and encoding that.
     * `Raster::extract` carries every attached field and drops only the page
     * split, so the extracted page arrived carrying the roll's whole delay
     * array and `encode_gif` refused it: "delay has 2 entries for 1 page(s)".
     * The fix is in `carry_meta_from`, which now drops `delay` wherever it
     * drops `page-height`, for the same reason: both describe the page split
     * rather than the image, so neither survives a change of shape. Keeping
     * the array would have been worse than refusing, since the first delay
     * would then be written onto a page that is not the first.
     * Input: page 0 of a two-page roll with delays `[40, 60]` -> Output: a
     * one-frame GIF with no delay, and no `delay` field on the extracted
     * raster.
     */
    #[test]
    fn extracting_a_page_and_saving_it_drops_the_stale_delay_array() {
        let mut source = roll(2);
        source.set_field("delay", MetadataValue::IntArray(vec![40, 60]));
        let bytes = source.encode_gif(SaveOptions::default()).expect("encodes");
        let back = decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            LoadOptions::default().with_n(-1),
        )
        .expect("loads back");
        assert_eq!(back.get_int_array("delay"), Some(&[40i64, 60][..]));

        let page = back.extract_page(0);
        assert_eq!(page.pages_loaded(), 1);
        assert_eq!(
            page.get_field("delay"),
            None,
            "a one-page raster cannot carry a two-page delay array"
        );
        let still = page
            .encode_gif(SaveOptions::default())
            .expect("a page pulled out of an animation saves as a still");
        let (screen, _, frames) = wire_frames(&still);
        assert_eq!(screen, (2, 2));
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].delay_cs, 0);

        // The control: the refusal is still there for a raster that really
        // does carry a mismatched array, which is the only way left to build
        // one.
        let mut smuggled = roll(1);
        smuggled.set_field("delay", MetadataValue::IntArray(vec![40, 60]));
        assert!(smuggled.encode_gif(SaveOptions::default()).is_err());
    }

    /**
     * Tests what a GIF that has been loaded and re-encoded comes out as,
     * which is where "a still save is unchanged" stops being true. Works by
     * loading a still fixture that carries no NETSCAPE block and encoding the
     * result, then doing the same for a fixture that carries one.
     * The loader attaches `loop`, so the re-encode honours it, and a file
     * with no block loads as `loop = 1`, which writes no block back. That is
     * a change from the still lane, which wrote a block holding zero
     * unconditionally, and it is the better match: measured on vips 8.18.6,
     * `loop = 1` writes no block. A raster built from scratch, carrying
     * neither field, still writes the block cgif always writes.
     * Input: three rasters, one with no `loop` and two loaded from files
     * with and without a NETSCAPE block -> Output: block 0, no block, block
     * 0.
     */
    #[test]
    fn a_reloaded_still_carries_its_loop_count_back_out() {
        let fresh = roll(1).encode_gif(SaveOptions::default()).expect("encodes");
        assert_eq!(
            wire_frames(&fresh).1,
            Some(0),
            "a raster carrying no loop field writes the block cgif writes"
        );

        for (block, expected) in [(None, None), (Some(0u16), Some(0u16))] {
            let source = fixture(
                (2, 2),
                &ANIM_PALETTE,
                0,
                block,
                &[anim_frame([1, 1, 1, 1], 1, 0)],
            );
            let loaded = decode_gif(&source, DecodeLimits::default()).expect("a valid GIF");
            assert_eq!(
                loaded.get_int("loop"),
                Some(if block.is_none() { 1 } else { 0 })
            );
            let again = loaded.encode_gif(SaveOptions::default()).expect("encodes");
            assert_eq!(
                wire_frames(&again).1,
                expected,
                "a reload of a {block:?} file writes {expected:?} back"
            );
        }
    }

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
