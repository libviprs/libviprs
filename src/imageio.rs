//! Image IO, metadata fields, and library-level free functions ported
//! from libvips.
//!
//! This module is the seventh batch of the libvips operation surface
//! required by the ported integration tests (alongside
//! [`crate::composite`]): saving a [`Raster`] to a file by extension, the
//! named metadata field system (`vips_image_get` / `vips_image_set`), the
//! native `.v` container, and the small library-level helpers
//! ([`tokenize`], [`parse_thumbnail_geometry`]).
//!
//! # Saving
//!
//! [`Raster::save`] encodes by file extension and keeps attached metadata
//! where the container supports it; [`Raster::save_stripped`] writes
//! pixels only (libvips `strip`):
//!
//! | Extension | Encoder | Metadata kept by `save` |
//! |---|---|---|
//! | `.png` | [`crate::sink::encode_png`] | none yet (iCCP embedding is an open gap) |
//! | `.jpg` / `.jpeg` | the sink JPEG encoder at quality 75 | `icc-profile-data` (APP2), `exif-data` (APP1, raw blob) |
//! | `.gif` | [`Raster::encode_gif`] at the vips defaults | none: what a GIF carries (palette, `loop`, `delay`) is structural, not EXIF-class |
//! | `.webp` | [`Raster::encode_webp`], lossless | `icc-profile-data` (`ICCP`), `exif-data` (`EXIF`), `xmp-data` (`XMP `) |
//! | `.jxl` (needs the `jxl` feature) | [`Raster::encode_jxl`], lossless | none: the encoder writes a bare codestream with no box container |
//! | `.jp2` / `.j2k` / `.jpt` / `.j2c` / `.jpc` (needs the `jp2k` feature) | [`Raster::encode_jp2k`] at the `jp2ksave` defaults | none: `jp2ksave.c` has no code for ICC, EXIF or XMP |
//! | `.fits` / `.fit` / `.fts` | [`Raster::encode_fits`] | the `fits-` header records, minus the cards cfitsio regenerates |
//! | `.v` / `.vips` | [`Raster::encode_vips`] | header geometry plus every attached field |
//!
//! Formats libviprs cannot encode yet (TIFF-with-metadata, ...) return
//! [`SaveError::UnsupportedExtension`]; they arrive with the
//! foreign-format batch. `.jxl` and the five JPEG 2000 suffixes join them
//! when the crate is built without their non-default feature, and the
//! refusal follows the build: it names the extensions this binary actually
//! has an encoder behind rather than a fixed list, so no cfg can advertise
//! another's.
//!
//! The JPEG 2000 row is the one place in that table where the suffix does
//! **not** pick the codec. `jp2ksave` registers five and writes the same JP2
//! container for every one of them, measured on 8.18.6: `vips copy` over
//! `.jp2`, `.j2k`, `.jpt`, `.j2c` and `.jpc` produces five files with one
//! SHA-256 between them, and `.jp2000` is refused as an unknown format. So
//! all five are rows, and the suffix only decides whether the file is written
//! at all.
//!
//! **Ultra HDR is not in that table and is not missing from it.** `uhdrsave`
//! registers no file suffix at all (measured on 8.18.6: an empty suffix list
//! in `vips -l`, and `vips copy base.v out.uhdr` refused as an unknown
//! format), and the four suffixes `uhdrload` claims on the way in at priority
//! 100 all route to `jpegsave` on the way out. So there is no extension for
//! this route to key on. Ultra HDR is written by name, through
//! [`Raster::encode_to_buffer`] with `"uhdr"` or through
//! [`Raster::encode_uhdr`] (issue #809).
//!
//! Structured EXIF tag writing (`exif-ifd0-*` fields into the TIFF
//! directory of a JPEG APP1 segment) is also deferred to the foreign
//! batch: those fields round-trip through `.v` and travel on the raster,
//! but JPEG save only re-embeds the raw `exif-data` blob captured at
//! decode time.
//!
//! # The `.v` container
//!
//! `encode_vips` writes the libvips native header (64 bytes: magic,
//! geometry, band format, coding, interpretation, resolution, offsets)
//! followed by the raw pixel data in the file's byte order, then the
//! metadata trailer libvips reads: the same small XML document real vips
//! writes, `<root>` with a `<header>` and a `<meta>` block of
//! `<field type="..." name="...">value</field>` elements
//! (`libvips/iofuncs/vips.c:846-890` at `v8.18.0-95-gfe420cf3a`). The reader
//! accepts both byte orders (swapping 16-bit and float samples as needed),
//! rejects band formats other than uchar, ushort, and float, and enforces
//! the [`DecodeLimits::max_coord`] dimension ceiling on untrusted header
//! geometry.
//!
//! The trailer is written only when it would carry something: a raster with
//! the upright orientation and no attached fields writes the header and the
//! pixels and stops (issue #546). An empty trailer is not free, because the
//! slot is not optional to libvips once a byte is in it.
//!
//! The five [`MetadataValue`] variants map one to one onto five of the GTypes
//! vips round-trips through its `VIPS_TYPE_SAVE_STRING` transforms
//! (`libvips/iofuncs/type.c:424-800`):
//!
//! | [`MetadataValue`] | `type=` | text |
//! |---|---|---|
//! | [`Int`](MetadataValue::Int) | `gint` | decimal |
//! | [`Double`](MetadataValue::Double) | `gdouble` | shortest round-tripping decimal |
//! | [`Str`](MetadataValue::Str) | `VipsRefString` | the string |
//! | [`Blob`](MetadataValue::Blob) | `VipsBlob` | standard base64, padded, unwrapped |
//! | [`IntArray`](MetadataValue::IntArray) | `VipsArrayInt` | decimals, each followed by one space |
//!
//! The array spelling has a **trailing space**, which is not decoration.
//! Measured on the pinned vips 8.18.6, `vips copy 'anim3.webp[n=-1]' out.v`
//! writes `<field type="VipsArrayInt" name="delay">100 100 100 </field>`,
//! one space per element including the last, because `vips_array_int`'s save
//! transform appends a separator after every value rather than between them.
//! The reader is looser than the writer on purpose, and so is vips's: a
//! trailer carrying `40 60 80`, `40 60 80 ` or `  40   60   80  ` reads back
//! as the same three elements in both libraries, and an element list that is
//! empty is an empty array rather than a missing field.
//!
//! Two things about that carrier that are libviprs's own answer rather than
//! vips's, both measured on 8.18.6:
//!
//! * **an element that is not a number keeps the whole field opaque.** vips
//!   hands back an *empty* array for `40 x 80` (`vipsheader -f delay` prints
//!   nothing and a `vips copy` writes the field back out empty), silently
//!   losing the two elements that did parse. libviprs carries the text
//!   through untouched instead, which is the same rule every other GType in
//!   this table follows when its text will not parse;
//! * **an element outside `gint` does not survive a trip through vips.**
//!   [`MetadataValue::Int`] is an `i64` and so are these elements, but vips's
//!   `gint` is 32 bits and wraps: a trailer carrying `3000000000` reads back
//!   as `-1294967296`, and `9223372036854775807 -9223372036854775808` reads
//!   back as `-1 0`. libviprs round-trips all of them through its own reader.
//!   The element type is signed and 64-bit anyway, because the value has to
//!   survive a hostile `.v` rather than only a well-formed one.
//!
//! The orientation tag rides in `<meta>` as an ordinary
//! `<field type="gint" name="orientation">`, which is where libvips keeps it
//! and where the reader has always looked for it.
//!
//! Two places where this deliberately does *not* copy vips byte for byte:
//!
//! * **non-ASCII text survives.** `vips_target_write_amp`
//!   (`libvips/iofuncs/target.c:821`) tests `*p < 32` on a plain `char`,
//!   which is signed on this target, so every byte of a multi-byte UTF-8
//!   sequence is treated as a control character and replaced by the
//!   Unicode control picture at `0x2400 + *p`. Measured on vips 8.18.4:
//!   `vips copy` over a `.v` carrying `café ☃ 日本` rewrites it as
//!   `caf&#x23c3;&#x23a9; &#x23e2;&#x2398;&#x2383; …`, irreversibly. This
//!   writer escapes only what XML actually needs, so vips reads libviprs's
//!   UTF-8 back correctly (verified with `vipsheader -a`) even though vips
//!   cannot rewrite its own;
//! * **a field name containing `"` is escaped as `&quot;`,** where
//!   `target_write_quotes` (`vips.c:790-804`) writes a backslash and leaves
//!   the attribute unterminated for its own expat parser.
//!
//! A C0 control character inside a string field does *not* survive, in
//! either direction: XML 1.0 cannot represent one at all, so the only
//! spelling vips's parser accepts is the substitution above, and reading it
//! back yields the control picture rather than the control character.
//! Binary belongs in a [`Blob`](MetadataValue::Blob), which is base64 and
//! exact. A carriage return is the exception and is written as `&#x000d;`,
//! because a literal one would be folded into a newline by XML end-of-line
//! normalisation.
//!
//! ## Trailer compatibility across versions
//!
//! The trailer is read field by field rather than as one `serde` value, so
//! a `.v` written by a newer libviprs costs an older one only the fields it
//! genuinely cannot represent (issue #565). One unknown [`MetadataValue`]
//! variant used to fail the whole parse and take every other field on the
//! image with it, silently, which made a new variant a data-loss break that
//! `cargo semver-checks` could not see. The rules the format holds to:
//!
//! * a field whose `type` is not one of the five above is carried opaquely
//!   as that type name plus its character data exactly as it sat on disk:
//!   invisible to [`Raster::get_field`] and [`Raster::get_fields`], because
//!   this build cannot say what it means, but written back out byte for
//!   byte, so an old build that opens a new file and re-saves it does not
//!   strip what it could not read. Setting or removing a field of the same
//!   name supersedes it, so stripping still strips. vips itself keeps
//!   reading such a field, because the carrier *is* its own encoding: a
//!   `background` array goes out as `VipsArrayDouble` and `vipsheader -a`
//!   prints it, exactly as `delay` did until #787 gave it a variant;
//! * a `type` name libvips does not know is skipped by vips silently, with
//!   no warning and no error (measured on 8.18.4), which is what makes the
//!   carrier safe to write;
//! * an element this build expects and does not find falls back to the
//!   default (`orientation` to 1);
//! * a trailer that is not XML is treated as absent, as libvips treats a
//!   trailer that is not XML. libviprs is not the only writer of that slot,
//!   so silence there is the honest answer.
//!
//! ### The legacy JSON trailer
//!
//! libviprs 0.4.0 and earlier wrote their own JSON trailer,
//! `{"orientation":N,"fields":{"entries":[[name,value],...]}}` with values
//! in [`MetadataValue`]'s externally tagged form. The reader still accepts
//! it, entry by entry and with the same opaque carry, so every `.v` already
//! written keeps its metadata. A trailer that opens with `{` and is not
//! valid JSON claimed to be that format and is genuinely unrecoverable, so
//! the reader reports it rather than swallowing it.
//!
//! The writer no longer produces it, with one exception. A value carried
//! opaquely out of a JSON trailer has **no XML spelling**, and cannot get
//! one: the two formats encode the same value differently
//! (`{"DoubleArray":[1.5,2.5]}` against `type="VipsArrayDouble">1.5 2.5 `),
//! so translating between them means interpreting the value, which is the
//! one thing a carried value is defined not to allow. Rather than drop it, a
//! raster still carrying such a value keeps the JSON trailer. That costs
//! the vips warning on exactly the files that were already unreadable to
//! vips, and it loses nothing.
//!
//! Naming a variant releases the files that only needed *it*: a legacy
//! trailer whose one unnameable value was a `{"IntArray":[...]}` delay is
//! read as a value now rather than carried, so the rewrite comes back out as
//! the XML vips reads. That is the whole payoff of #787 on the disk side, and
//! it is why the JSON fallback is keyed on what is still carried rather than
//! on where the file came from.
//!
//! The consequence of writing XML, stated plainly: libviprs 0.4.0 reads a
//! `.v` written now for its pixels, its geometry and its orientation, and
//! not for its attached fields, because its reader takes the trailer as
//! metadata only when the first non-whitespace byte is `{`. No byte
//! sequence can be both that and the XML vips requires, so full interop
//! with vips and full field recovery on 0.4.0 cannot both hold. New reads
//! old completely; old reads new down to the orientation.
//!
//! The interpretation word holds libvips' own `VipsInterpretation` codes,
//! which since libvips 8.18 include `30` and `31` for OkLab and OkLch
//! (`libvips/include/vips/image.h:115-116`); a `.v` libviprs writes now
//! carries the same tag real vips writes. Files libviprs wrote before this
//! release carry the private codes `1000` / `1001` it used while libvips
//! had none, and still read back tagged, because the reader keeps those two
//! as permanently reserved read-only aliases. Nothing writes them any more,
//! so the incompatibility only runs the other way: a `.v` written now reads
//! as `Multiband` on libviprs 0.4.0 and earlier.
//!
//! # Metadata fields
//!
//! [`Raster::get_field`] / [`Raster::set_field`] expose the libvips
//! named-field system over a [`MetadataValue`] enum (int, double, string,
//! blob). The built-in header fields (`width`, `height`, `bands`,
//! `format`, `coding`, `interpretation`, `xoffset`, `yoffset`, `xres`,
//! `yres`, `orientation`, `filename`) read through to the raster header
//! and [`crate::conversion`] metadata block; every other name is an
//! attached field carried with the raster (cloned with it, written to
//! `.v`, and — for `icc-profile-data` / `exif-data` — embedded on JPEG
//! save). Geometry fields are read-only, matching what a fixed pixel
//! buffer can honour.
//!
//! # Free functions
//!
//! * [`tokenize`] — the libvips option-string tokeniser (whitespace
//!   splitting with `"..."` / `'...'` quoting and backslash escapes).
//! * [`parse_thumbnail_geometry`] — the vipsthumbnail size parser
//!   (`W`, `WxH`, `Wx`, `xH`, with trailing `<` / `>` / `!` modifiers
//!   tolerated).
//!
//! The single-axis coordinate ceiling (libvips `VIPS_MAX_COORD`) is not a
//! library-level free function: it lives on the per-decode
//! [`DecodeLimits::max_coord`] field (default [`DEFAULT_MAX_COORD`]),
//! enforced by *every* decoder. Build a [`DecodeLimits`] with
//! [`DecodeLimits::with_max_coord`] and pass it to a `*_with_limits` decode
//! call to tighten it; there is no process-global to set.
//!
//! # Decode limits
//!
//! Two levels govern how much work an untrusted input may provoke, and no
//! more: a per-decode [`DecodeLimits`] (single-axis `max_coord`, total
//! `max_pixels`, and decoder `max_alloc_bytes`, all applied before the
//! pixels are materialised) and a per-constructor allocation budget
//! enforced by the [`Raster`] constructors. `max_coord` is honoured
//! uniformly by both the native `.v` reader and the `image`-crate raster
//! path (PNG/JPEG/TIFF); see [`DecodeLimits`] for the full per-format
//! field mapping. There is no third process-global regime: the coordinate
//! ceiling lives solely on `DecodeLimits`.

use std::borrow::Cow;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::conversion::Interpretation;
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::sink::SinkError;
use crate::source::{DecodeLimits, SourceError};

// ---------------------------------------------------------------------------
// Metadata values and the attached-field store
// ---------------------------------------------------------------------------

/// A typed metadata field value (libvips stores these as boxed `GValue`s;
/// here the four kinds the ported suites read and write are first-class).
///
/// `From` conversions cover the literal shapes the ported tests use
/// (`"TestSoftware".into()`, `format!(...).into()`), and the `as_*`
/// accessors mirror pyvips-style coercing reads: they panic with a
/// descriptive message when the variant does not match, in line with the
/// panicking convenience layer of the operation modules.
///
/// # Growing
///
/// `#[non_exhaustive]`, because five is not the number of types a vips
/// metadata field can have (issue #609). `VipsArrayDouble` and `gboolean`
/// are both still live in a `.v` trailer this crate only forwards opaquely.
/// Marking the enum before the first of those variants landed cost
/// downstream a `_ =>` arm on an exhaustive `match`; marking it after would
/// have cost a major version instead, and `cargo semver-checks` would have
/// been right to demand one. [`IntArray`](MetadataValue::IntArray) is the
/// first variant to arrive through that door (issue #787).
///
/// Only matching is affected. Every variant stays constructible from
/// outside, so `MetadataValue::Int(3)` and the `From` impls are unchanged,
/// and the `as_*` accessors are the reading path that never needed a match
/// anyway.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MetadataValue {
    /// A signed integer field (`width`, `orientation`, counts, flags).
    Int(i64),
    /// A floating-point field (`xres`, `yres`).
    Double(f64),
    /// A text field (EXIF strings, `interpretation`, `filename`).
    Str(String),
    /// A binary field (`icc-profile-data`, `exif-data`).
    Blob(Vec<u8>),
    /// An ordered list of signed integers (vips `VipsArrayInt`): the
    /// per-frame `delay` of every animated format, and the shape every
    /// animated loader in this crate needs to attach one at all (issue
    /// #787).
    ///
    /// The elements are `i64` like [`Int`](MetadataValue::Int), not `u32`,
    /// so a hostile `.v` round-trips instead of being silently clamped.
    /// vips's own `gint` is 32 bits and wraps: see the
    /// [module docs](crate::imageio) for what that costs a value this crate
    /// hands back to vips.
    IntArray(Vec<i64>),
}

impl MetadataValue {
    /// The value as a string slice.
    ///
    /// # Panics
    ///
    /// Panics if the value is not [`MetadataValue::Str`].
    #[track_caller]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Str(s) => s,
            other => panic!("metadata value is {}, not a string", other.kind()),
        }
    }

    /// The value as a `u32`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not [`MetadataValue::Int`] or does not fit
    /// in a `u32`.
    #[track_caller]
    pub fn as_u32(&self) -> u32 {
        match self {
            Self::Int(i) => u32::try_from(*i)
                .unwrap_or_else(|_| panic!("metadata int {i} does not fit in a u32")),
            other => panic!("metadata value is {}, not an int", other.kind()),
        }
    }

    /// The value as an `i64`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not [`MetadataValue::Int`].
    #[track_caller]
    pub fn as_i64(&self) -> i64 {
        match self {
            Self::Int(i) => *i,
            other => panic!("metadata value is {}, not an int", other.kind()),
        }
    }

    /// The value as an `f64`; an [`MetadataValue::Int`] coerces.
    ///
    /// # Panics
    ///
    /// Panics if the value is not numeric.
    #[track_caller]
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Double(d) => *d,
            Self::Int(i) => *i as f64,
            other => panic!("metadata value is {}, not numeric", other.kind()),
        }
    }

    /// The value as a byte slice.
    ///
    /// # Panics
    ///
    /// Panics if the value is not [`MetadataValue::Blob`].
    #[track_caller]
    pub fn as_blob(&self) -> &[u8] {
        match self {
            Self::Blob(b) => b,
            other => panic!("metadata value is {}, not a blob", other.kind()),
        }
    }

    /// The value as a slice of ints.
    ///
    /// Borrowed rather than cloned, like [`MetadataValue::as_blob`], because
    /// a `delay` array is read once per frame and there is no reason to copy
    /// it each time.
    ///
    /// # Panics
    ///
    /// Panics if the value is not [`MetadataValue::IntArray`]. A scalar
    /// [`MetadataValue::Int`] does *not* coerce to a one-element array: vips
    /// writes `gint` and `VipsArrayInt` as different types and `gifsave`
    /// reads only the array, so treating one as the other would invent a
    /// per-frame delay out of a `gif-delay` that is the first frame's alone.
    #[track_caller]
    pub fn as_int_array(&self) -> &[i64] {
        match self {
            Self::IntArray(v) => v,
            other => panic!("metadata value is {}, not an int array", other.kind()),
        }
    }

    /// The type code returned by [`Raster::get_typeof`] for this value:
    /// 1 int, 2 double, 3 string, 4 blob, 5 int array. These are libviprs
    /// codes (the C library returns GObject `GType` numbers); the ported call
    /// sites only distinguish zero (absent) from non-zero (present).
    ///
    /// Zero is not a code here, it is [`Raster::get_typeof`]'s answer for a
    /// field that is not there, so every variant has to have one of its own.
    pub fn type_code(&self) -> u64 {
        match self {
            Self::Int(_) => 1,
            Self::Double(_) => 2,
            Self::Str(_) => 3,
            Self::Blob(_) => 4,
            Self::IntArray(_) => 5,
        }
    }

    /// The length of this value in its natural unit.
    ///
    /// * [`MetadataValue::Blob`]: the number of bytes (what
    ///   `image.get("icc-profile-data").len()` reports in the ported foreign
    ///   cell, for example the 564-byte ICC profile of `sample.jpg` read
    ///   through magick).
    /// * [`MetadataValue::Str`]: the number of UTF-8 bytes in the string.
    /// * [`MetadataValue::IntArray`]: the number of elements, which is the
    ///   frame count for a `delay`.
    /// * [`MetadataValue::Int`] / [`MetadataValue::Double`]: `1`, a scalar
    ///   is a single-element field.
    pub fn len(&self) -> usize {
        match self {
            Self::Blob(b) => b.len(),
            Self::Str(s) => s.len(),
            Self::IntArray(v) => v.len(),
            Self::Int(_) | Self::Double(_) => 1,
        }
    }

    /// Whether this value has zero length; see [`MetadataValue::len`]. A
    /// scalar [`MetadataValue::Int`] or [`MetadataValue::Double`] is never
    /// empty (its length is `1`); a [`MetadataValue::IntArray`] with no
    /// elements is, and that is a value vips writes rather than an error.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Human-readable kind name for panic messages.
    fn kind(&self) -> &'static str {
        match self {
            Self::Int(_) => "an int",
            Self::Double(_) => "a double",
            Self::Str(_) => "a string",
            Self::Blob(_) => "a blob",
            Self::IntArray(_) => "an int array",
        }
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        Self::Str(s.to_string())
    }
}
impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        Self::Str(s)
    }
}
impl From<i64> for MetadataValue {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}
impl From<i32> for MetadataValue {
    fn from(i: i32) -> Self {
        Self::Int(i64::from(i))
    }
}
impl From<u32> for MetadataValue {
    fn from(i: u32) -> Self {
        Self::Int(i64::from(i))
    }
}
impl From<f64> for MetadataValue {
    fn from(d: f64) -> Self {
        Self::Double(d)
    }
}
impl From<Vec<u8>> for MetadataValue {
    fn from(b: Vec<u8>) -> Self {
        Self::Blob(b)
    }
}
impl From<&[u8]> for MetadataValue {
    fn from(b: &[u8]) -> Self {
        Self::Blob(b.to_vec())
    }
}
impl From<Vec<i64>> for MetadataValue {
    fn from(v: Vec<i64>) -> Self {
        Self::IntArray(v)
    }
}
impl From<&[i64]> for MetadataValue {
    fn from(v: &[i64]) -> Self {
        Self::IntArray(v.to_vec())
    }
}

/// The attached (non-header) metadata fields carried by a [`Raster`]:
/// an insertion-ordered name/value list, so [`Raster::get_fields`]
/// reports attachments in the order they were set, like libvips.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct MetadataFields {
    entries: Vec<(String, MetadataValue)>,
    /// Fields read from a `.v` trailer that this build cannot interpret,
    /// kept in the trailer form they arrived in (issue #565).
    ///
    /// They stay out of the field API on purpose: this build can say that
    /// the field was there, not what it means, and a wrong answer is worse
    /// than none. Carrying them is what stops a rewrite by an older build
    /// from stripping what a newer one wrote.
    unknown: Vec<(String, CarriedValue)>,
}

/// A `.v` trailer value this build cannot name, kept exactly as it was
/// written so it can go back out unchanged (issue #565).
///
/// The form matters, because the two trailer formats spell the same value
/// differently and converting between them would mean interpreting it. See
/// the [module docs](crate::imageio) for what that costs.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
enum CarriedValue {
    /// From a libvips XML trailer: the `type` attribute and the element's
    /// character data as it sat on disk, escapes and all.
    ///
    /// The JSON spelling is only reachable on a raster that also carries a
    /// [`CarriedValue::Json`] value, which needs one trailer to hold both.
    /// No `MetadataValue` matches this shape, so any libviprs build carries
    /// it on rather than reading it as something it is not.
    Xml {
        /// The GType name from the `type` attribute, e.g. `VipsArrayInt`.
        #[serde(rename = "vips-xml-type")]
        gtype: String,
        /// The character data between the tags, still XML-escaped.
        #[serde(rename = "vips-xml-text")]
        text: String,
    },
    /// From the legacy libviprs JSON trailer: the value exactly as it parsed.
    Json(serde_json::Value),
}

impl MetadataFields {
    pub(crate) fn get(&self, name: &str) -> Option<&MetadataValue> {
        self.entries
            .iter()
            .find_map(|(n, v)| (n == name).then_some(v))
    }

    /// Upsert keeping first-set order. A value the caller can name
    /// supersedes an uninterpretable one carried under the same name.
    pub(crate) fn set(&mut self, name: &str, value: MetadataValue) {
        self.unknown.retain(|(n, _)| n != name);
        if let Some(slot) = self
            .entries
            .iter_mut()
            .find_map(|(n, v)| (n == name).then_some(v))
        {
            *slot = value;
        } else {
            self.entries.push((name.to_string(), value));
        }
    }

    /// Remove a field under either carrier, so removing a name that arrived
    /// uninterpretable really removes it rather than leaving it to reappear
    /// on the next save. The uninterpretable carrier has no
    /// [`MetadataValue`] to hand back, so removing one reads as absent.
    pub(crate) fn remove(&mut self, name: &str) -> Option<MetadataValue> {
        self.unknown.retain(|(n, _)| n != name);
        let idx = self.entries.iter().position(|(n, _)| n == name)?;
        Some(self.entries.remove(idx).1)
    }

    /// Whether a name is carried at all, under either carrier.
    fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|(n, _)| n == name) || self.unknown.iter().any(|(n, _)| n == name)
    }

    /// Take `other`'s fields for every name this map does not already carry,
    /// in `other`'s own order, leaving this map's values alone.
    ///
    /// This is the multi-input rule, measured on vips 8.18.6: `insert`,
    /// `join`, `arrayjoin` and `bandjoin` put the union of both inputs'
    /// attachments on the output and let the *first* input win a name they
    /// share, while the header block comes from the first input alone. A
    /// profile that only the second input carries reaches the output; a
    /// `lane-711` both carry keeps the first one's value (#718).
    ///
    /// Uninterpretable `.v` trailer values merge on the same terms, so a name
    /// this build cannot read still travels rather than being dropped because
    /// the reader could not name it (#565). A name held under one carrier here
    /// blocks the other carrier's copy from `other`, which is the same
    /// "one value per name" invariant [`MetadataFields::set`] keeps.
    pub(crate) fn merge_under(&mut self, other: &Self) {
        for (name, value) in &other.entries {
            if !self.contains(name) {
                self.entries.push((name.clone(), value.clone()));
            }
        }
        for (name, value) in &other.unknown {
            if !self.contains(name) {
                self.unknown.push((name.clone(), value.clone()));
            }
        }
    }

    /// Record a field this build cannot interpret; see
    /// [`MetadataFields::unknown`].
    fn set_unknown(&mut self, name: &str, value: CarriedValue) {
        self.entries.retain(|(n, _)| n != name);
        if let Some(slot) = self
            .unknown
            .iter_mut()
            .find_map(|(n, v)| (n == name).then_some(v))
        {
            *slot = value;
        } else {
            self.unknown.push((name.to_string(), value));
        }
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _)| n.as_str())
    }

    /// Whether the raster carries no attached field at all, interpretable
    /// or not. A `.v` trailer with nothing in it is worse than no trailer
    /// (issue #546), so the writer asks this before writing one.
    fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.unknown.is_empty()
    }

    /// The interpretable fields, in the order they were set.
    fn known(&self) -> impl Iterator<Item = (&str, &MetadataValue)> {
        self.entries.iter().map(|(n, v)| (n.as_str(), v))
    }

    /// The fields carried opaquely; see [`MetadataFields::unknown`].
    fn unknown_fields(&self) -> impl Iterator<Item = (&str, &CarriedValue)> {
        self.unknown.iter().map(|(n, v)| (n.as_str(), v))
    }

    /// Whether anything here can only be written as the legacy JSON
    /// trailer. See the [module docs](crate::imageio): a value carried out
    /// of a JSON trailer has no XML spelling, so the writer keeps the old
    /// format for that file rather than dropping it.
    fn needs_legacy_json_trailer(&self) -> bool {
        self.unknown
            .iter()
            .any(|(_, v)| matches!(v, CarriedValue::Json(_)))
    }
}

/// Errors from [`Raster::try_set_field`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetadataError {
    #[error("metadata field {name:?} is read-only (fixed by the pixel buffer)")]
    ReadOnlyField { name: String },
    #[error("metadata field {name:?} expects {expected}, got {got}")]
    WrongType {
        name: String,
        expected: &'static str,
        got: &'static str,
    },
    #[error("unknown interpretation nickname {value:?}")]
    UnknownInterpretation { value: String },
}

/// The built-in header field names, in the order libvips reports them
/// (header geometry first, `width` leading).
const BUILTIN_FIELDS: [&str; 12] = [
    "width",
    "height",
    "bands",
    "format",
    "coding",
    "interpretation",
    "xoffset",
    "yoffset",
    "xres",
    "yres",
    "orientation",
    "filename",
];

/// The libvips nickname for an [`Interpretation`].
fn interpretation_nickname(i: Interpretation) -> &'static str {
    match i {
        Interpretation::Multiband => "multiband",
        Interpretation::Bw => "b-w",
        Interpretation::Histogram => "histogram",
        Interpretation::Xyz => "xyz",
        Interpretation::Lab => "lab",
        Interpretation::Cmyk => "cmyk",
        Interpretation::Labq => "labq",
        Interpretation::Rgb => "rgb",
        Interpretation::Cmc => "cmc",
        Interpretation::Lch => "lch",
        Interpretation::Labs => "labs",
        Interpretation::Srgb => "srgb",
        Interpretation::Yxy => "yxy",
        Interpretation::Fourier => "fourier",
        Interpretation::Rgb16 => "rgb16",
        Interpretation::Grey16 => "grey16",
        Interpretation::Matrix => "matrix",
        Interpretation::ScRgb => "scrgb",
        Interpretation::Hsv => "hsv",
        Interpretation::OkLab => "oklab",
        Interpretation::OkLch => "oklch",
    }
}

/// Parse a libvips interpretation nickname.
fn interpretation_from_nickname(s: &str) -> Option<Interpretation> {
    Some(match s {
        "multiband" => Interpretation::Multiband,
        "b-w" => Interpretation::Bw,
        "histogram" => Interpretation::Histogram,
        "xyz" => Interpretation::Xyz,
        "lab" => Interpretation::Lab,
        "cmyk" => Interpretation::Cmyk,
        "labq" => Interpretation::Labq,
        "rgb" => Interpretation::Rgb,
        "cmc" => Interpretation::Cmc,
        "lch" => Interpretation::Lch,
        "labs" => Interpretation::Labs,
        "srgb" => Interpretation::Srgb,
        "yxy" => Interpretation::Yxy,
        "fourier" => Interpretation::Fourier,
        "rgb16" => Interpretation::Rgb16,
        "grey16" => Interpretation::Grey16,
        "matrix" => Interpretation::Matrix,
        "scrgb" => Interpretation::ScRgb,
        "hsv" => Interpretation::Hsv,
        "oklab" => Interpretation::OkLab,
        "oklch" => Interpretation::OkLch,
        _ => return None,
    })
}

/// The libvips `VipsInterpretation` enum value for the `.v` header
/// `Type` word, as declared in `libvips/include/vips/image.h:96-117`
/// (8.18.4). `OkLab` and `OkLch` are `VIPS_INTERPRETATION_OKLAB` = 30
/// and `VIPS_INTERPRETATION_OKLCH` = 31 (`image.h:115-116`), so a `.v`
/// libviprs writes carries the same tag real vips writes.
fn interpretation_code(i: Interpretation) -> i32 {
    match i {
        Interpretation::Multiband => 0,
        Interpretation::Bw => 1,
        Interpretation::Histogram => 10,
        Interpretation::Xyz => 12,
        Interpretation::Lab => 13,
        Interpretation::Cmyk => 15,
        Interpretation::Labq => 16,
        Interpretation::Rgb => 17,
        Interpretation::Cmc => 18,
        Interpretation::Lch => 19,
        Interpretation::Labs => 21,
        Interpretation::Srgb => 22,
        Interpretation::Yxy => 23,
        Interpretation::Fourier => 24,
        Interpretation::Rgb16 => 25,
        Interpretation::Grey16 => 26,
        Interpretation::Matrix => 27,
        Interpretation::ScRgb => 28,
        Interpretation::Hsv => 29,
        Interpretation::OkLab => 30,
        Interpretation::OkLch => 31,
    }
}

/// The `.v` header `Type` word back to an [`Interpretation`]: a *left*
/// inverse of [`interpretation_code`] (every code that function writes reads
/// back as the variant it came from), widened by the two read-only legacy
/// aliases `1000` / `1001`, which land on `OkLab` / `OkLch` alongside the
/// libvips codes `30` / `31`. So it is not a bijection, and it is not the
/// other direction of the round trip for those four codes. Unknown codes
/// read as `None` and the raster falls back to format inference, like an
/// untagged image.
fn interpretation_from_code(code: i32) -> Option<Interpretation> {
    Some(match code {
        0 => Interpretation::Multiband,
        1 => Interpretation::Bw,
        10 => Interpretation::Histogram,
        12 => Interpretation::Xyz,
        13 => Interpretation::Lab,
        15 => Interpretation::Cmyk,
        16 => Interpretation::Labq,
        17 => Interpretation::Rgb,
        18 => Interpretation::Cmc,
        19 => Interpretation::Lch,
        21 => Interpretation::Labs,
        22 => Interpretation::Srgb,
        23 => Interpretation::Yxy,
        24 => Interpretation::Fourier,
        25 => Interpretation::Rgb16,
        26 => Interpretation::Grey16,
        27 => Interpretation::Matrix,
        28 => Interpretation::ScRgb,
        29 => Interpretation::Hsv,
        30 => Interpretation::OkLab,
        31 => Interpretation::OkLch,
        // Read-only legacy aliases: before libvips 8.18 assigned 30 / 31,
        // libviprs wrote these private codes above the libvips range. Files
        // it already wrote keep loading; nothing emits them any more. They
        // are reserved permanently and must never be reused for anything
        // else: the only thing retiring them would achieve is to silently
        // re-break every `.v` libviprs has already written.
        1000 => Interpretation::OkLab,
        1001 => Interpretation::OkLch,
        _ => return None,
    })
}

/// How many samples a colour takes in the space an ICC profile describes, read
/// from bytes 16..20 of the profile header (issue #720).
///
/// `None` for a blob too short to hold the field or carrying a signature this
/// build does not know. Those are **kept** by the caller: dropping an
/// attachment because the parser could not reach a verdict is worse than
/// keeping one that may not apply, and it is the same call this module makes
/// for `.v` trailer values it cannot interpret (#565). It also stops the rule
/// silently eating a profile in a colour space a later libviprs learns about.
///
/// The signatures are the ICC.1 data colour spaces, grouped by channel count.
/// Only the count matters here, so the 2-channel and 5-to-15-channel `xCLR`
/// spaces resolve to their own counts rather than being listed one by one.
pub(crate) fn profile_space_bands(profile: &[u8]) -> Option<usize> {
    let sig: &[u8; 4] = profile.get(16..20)?.try_into().ok()?;
    Some(match sig {
        b"GRAY" => 1,
        b"CMY " | b"RGB " | b"XYZ " | b"Lab " | b"Luv " | b"YCbr" | b"Yxy " | b"HSV " | b"HLS "
        | b"3CLR" => 3,
        b"CMYK" | b"4CLR" => 4,
        b"2CLR" => 2,
        b"5CLR" => 5,
        b"6CLR" => 6,
        b"7CLR" => 7,
        b"8CLR" => 8,
        _ => return None,
    })
}

impl Raster {
    /// Read a metadata field by name (libvips `vips_image_get`): the
    /// built-in header fields (`width`, `height`, `bands`, `format`,
    /// `coding`, `interpretation`, `xoffset`, `yoffset`, `xres`, `yres`,
    /// `orientation`, `filename`) or any attached field previously set
    /// with [`Raster::set_field`] / [`Raster::set_icc_profile`] or
    /// captured at decode time (`icc-profile-data`, `exif-data`).
    ///
    /// Returns `None` for a name that is neither built-in nor attached.
    pub fn get_field(&self, name: &str) -> Option<MetadataValue> {
        Some(match name {
            "width" => MetadataValue::Int(i64::from(self.width())),
            "height" => MetadataValue::Int(i64::from(self.height())),
            "bands" => MetadataValue::Int(self.format().channels() as i64),
            "format" => MetadataValue::Str(
                match self.format().bytes_per_channel() {
                    1 => "uchar",
                    2 => "ushort",
                    _ => "float",
                }
                .to_string(),
            ),
            "coding" => MetadataValue::Str("none".to_string()),
            "interpretation" => {
                MetadataValue::Str(interpretation_nickname(self.interpretation()).to_string())
            }
            "xoffset" => MetadataValue::Int(i64::from(self.xoffset())),
            "yoffset" => MetadataValue::Int(i64::from(self.yoffset())),
            "xres" => MetadataValue::Double(self.xres()),
            "yres" => MetadataValue::Double(self.yres()),
            "orientation" => MetadataValue::Int(i64::from(self.orientation())),
            "filename" => self
                .fields
                .get("filename")
                .cloned()
                .unwrap_or_else(|| MetadataValue::Str(String::new())),
            other => return self.fields.get(other).cloned(),
        })
    }

    /// Read a metadata field as an `i32` (libvips `vips_image_get_int`).
    ///
    /// Answers the same names [`Raster::get_field`] does and returns the
    /// value when it is an integer that fits in an `i32` (for example the
    /// built-in `width`/`height`/`bands` header fields, or an attached field
    /// such as `bits-per-sample`, `tile-width`, or `page-height` set by a
    /// loader). Returns `None` for an absent field, a non-integer value, or
    /// an integer outside the `i32` range.
    ///
    /// It borrows the stored value rather than cloning one out, so reading an
    /// int costs nothing even when the name happens to hold a large
    /// [`MetadataValue::Blob`] (issue #635).
    pub fn get_int(&self, name: &str) -> Option<i32> {
        i32::try_from(self.field_i64(name)?).ok()
    }

    /// Fallible form of [`Raster::set_field`].
    ///
    /// # Errors
    ///
    /// [`MetadataError::ReadOnlyField`] for the geometry fields fixed by
    /// the pixel buffer (`width`, `height`, `bands`, `format`,
    /// `coding`), [`MetadataError::WrongType`] when a built-in field is
    /// given the wrong value kind or an out-of-range number, or
    /// [`MetadataError::UnknownInterpretation`] for an unrecognised
    /// interpretation nickname.
    pub fn try_set_field(&mut self, name: &str, value: MetadataValue) -> Result<(), MetadataError> {
        fn int_as<T: TryFrom<i64>>(
            name: &str,
            value: &MetadataValue,
            expected: &'static str,
        ) -> Result<T, MetadataError> {
            match value {
                MetadataValue::Int(i) => T::try_from(*i).map_err(|_| MetadataError::WrongType {
                    name: name.to_string(),
                    expected,
                    got: "an out-of-range int",
                }),
                other => Err(MetadataError::WrongType {
                    name: name.to_string(),
                    expected,
                    got: other.kind(),
                }),
            }
        }
        match name {
            "width" | "height" | "bands" | "format" | "coding" => {
                Err(MetadataError::ReadOnlyField {
                    name: name.to_string(),
                })
            }
            "xres" | "yres" => {
                let v = match &value {
                    MetadataValue::Double(d) => *d,
                    MetadataValue::Int(i) => *i as f64,
                    other => {
                        return Err(MetadataError::WrongType {
                            name: name.to_string(),
                            expected: "a double",
                            got: other.kind(),
                        });
                    }
                };
                if name == "xres" {
                    self.meta.xres = v;
                } else {
                    self.meta.yres = v;
                }
                Ok(())
            }
            "xoffset" => {
                self.meta.xoffset = int_as::<i32>(name, &value, "an int (i32 range)")?;
                Ok(())
            }
            "yoffset" => {
                self.meta.yoffset = int_as::<i32>(name, &value, "an int (i32 range)")?;
                Ok(())
            }
            "orientation" => {
                self.meta.orientation = int_as::<u8>(name, &value, "an int (u8 range)")?;
                Ok(())
            }
            "interpretation" => match &value {
                MetadataValue::Str(s) => match interpretation_from_nickname(s) {
                    Some(i) => {
                        self.meta.interpretation = Some(i);
                        Ok(())
                    }
                    None => Err(MetadataError::UnknownInterpretation { value: s.clone() }),
                },
                other => Err(MetadataError::WrongType {
                    name: name.to_string(),
                    expected: "a string",
                    got: other.kind(),
                }),
            },
            other => {
                self.fields.set(other, value);
                Ok(())
            }
        }
    }

    /// Set a metadata field by name (libvips `vips_image_set`): built-in
    /// header fields route into the raster metadata block, every other
    /// name becomes an attached field carried with the raster. Panicking
    /// form of [`Raster::try_set_field`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`MetadataError`]; see [`Raster::try_set_field`].
    #[track_caller]
    pub fn set_field(&mut self, name: &str, value: MetadataValue) {
        if let Err(e) = self.try_set_field(name, value) {
            panic!("set_field: {e}");
        }
    }

    /// Every readable field name: the built-in header fields (in libvips
    /// order, `width` first) followed by the attached fields in the
    /// order they were set (libvips `vips_image_get_fields`).
    pub fn get_fields(&self) -> Vec<String> {
        let mut out: Vec<String> = BUILTIN_FIELDS.iter().map(|s| s.to_string()).collect();
        for name in self.fields.names() {
            if !BUILTIN_FIELDS.contains(&name) {
                out.push(name.to_string());
            }
        }
        out
    }

    /// The type code of a field: 0 when the field does not exist,
    /// otherwise the [`MetadataValue::type_code`] of its value (libvips
    /// `vips_image_get_typeof`, which returns 0 or a `GType`).
    pub fn get_typeof(&self, name: &str) -> u64 {
        self.get_field(name).map_or(0, |v| v.type_code())
    }

    /// The EXIF orientation tag (libvips `orientation`, values 1-8),
    /// defaulting to `1` (upright) when the image carries no orientation.
    ///
    /// Reads the `orientation` header field; every raster has one, so this
    /// is total. The oracle reports `orientation = 1` for the reference
    /// images that are not pre-rotated.
    pub fn get_orientation(&self) -> i32 {
        i32::from(self.orientation())
    }

    /// The number of pages the **original file** holds (libvips `n-pages`),
    /// defaulting to `1`.
    ///
    /// A port of `vips_image_get_n_pages` (`iofuncs/header.c:917-928`),
    /// sanity check included. The key has one meaning across this crate and
    /// it is the one vips gives it: **how many pages the file this raster
    /// was decoded from contains**, where a page is something a loader's
    /// zero-based `page` argument can select. It is a count and not an
    /// index, so the sweep over every page is
    /// `for page in 0..raster.get_n_pages()` and the last page is
    /// `get_n_pages() - 1` (issue #566).
    ///
    /// It is *not* how many pages were loaded into this raster. Every loader
    /// here reads page 0, so a raster reporting `3` still holds one page of
    /// pixels, exactly as a default `vips` load does.
    ///
    /// # Which loaders attach it
    ///
    /// Four do, and each was measured against `vipsheader -a` on the same
    /// file under vips 8.18.6:
    ///
    /// | loader | what it counts | the vips writer it ports |
    /// |---|---|---|
    /// | [`crate::gif`] | frames in the GIF | `nsgifload.c:281` |
    /// | [`decode_tiff_page`](crate::decode_tiff_page) | IFDs in the chain | `tiff2vips.c:1879` |
    /// | [`crate::webp`] | frames in the original animation | `webp2vips.c:508` |
    /// | [`crate::jxl`] | frames in the original | `jxlload.c:747` |
    ///
    /// GIF and TIFF attach it to every load, a one-frame GIF and a
    /// single-page TIFF included, where it reads `1`. WebP attaches it only
    /// when the container is animated, and JPEG XL only when there is more
    /// than one frame, so a still of either carries no such field at all.
    /// That split is vips's rather than an inconsistency to tidy away:
    /// `vipsheader -a` reports `n-pages: 1` for a still GIF and a one-page
    /// TIFF, and no `n-pages` at all for a still WebP or a single-frame
    /// JPEG XL. Both shapes read back as `1` here, so a caller never has to
    /// know which one it is holding.
    ///
    /// # What stays off this key
    ///
    /// A count belongs here only if a page index can select it. Anything
    /// else gets a name that says what it is (issue #635):
    ///
    /// * [`crate::exr`] reports its multi-part count as `exr-parts`. An EXR
    ///   part is a layer, `openexrload` attaches no `n-pages` to an EXR, and
    ///   [`crate::decode_exr`] takes no part index, so `0..get_n_pages()`
    ///   would be a sweep over something unreachable (issue #626).
    /// * The PDF readers attach nothing. vips's `pdfload` does attach it
    ///   (measured: `n-pages: 3` for a three-page document, `1` for a
    ///   one-page one), but its `page` argument is zero-based where this
    ///   crate's PDF page numbers are one-based on purpose, so a caller
    ///   sweeping `0..get_n_pages()` would be off by one. The document's
    ///   count is [`crate::PdfInfo::page_count`].
    ///
    /// vips's own `jp2kload` is the case not to copy: it puts the JPEG 2000
    /// *resolution* count under this key (`jp2kload.c:586`), so `page` there
    /// picks a shrink level rather than a frame.
    ///
    /// # The sanity check
    ///
    /// vips reports a single page unless the field is an int strictly
    /// between 1 and 10000, and so does this. Measured against
    /// `vips_image_get_n_pages` on 8.18.6: `9999` comes back as `9999`,
    /// `10000` and `65536` come back as `1`, and a string-typed field comes
    /// back as `1` whatever it spells, because `vips_image_get_int` will not
    /// coerce one. The stored value is untouched either way and stays
    /// readable through [`Raster::get_field`], and a TIFF's real chain
    /// length is [`tiff_page_count`](crate::tiff_page_count) regardless of
    /// what this reports. Reading costs no allocation whatever type is
    /// sitting under the name.
    pub fn get_n_pages(&self) -> u32 {
        // `iofuncs/header.c:921-926`: the field has to be an int and it has
        // to sit strictly between 1 and this ceiling, or vips calls the
        // value crazy and reports a single page.
        const CEILING: i64 = 10_000;
        match self.field_i64("n-pages") {
            Some(n) if (2..CEILING).contains(&n) => u32::try_from(n).unwrap_or(1),
            _ => 1,
        }
    }

    /// Attach `n-pages`: the one place in the crate that names the key
    /// (issue #635).
    ///
    /// `count` is how many pages the **file** holds, where a page is
    /// something a loader's zero-based `page` argument can select. That is
    /// the whole of the contract [`Raster::get_n_pages`] documents, and
    /// routing every writer through one function is what stops a fifth
    /// meaning arriving under the same name: a count no page index can reach
    /// gets a key of its own instead, the way the OpenEXR part count became
    /// `exr-parts` (issue #626).
    ///
    /// `tests/n_pages_meaning.rs` asserts that the literal key appears in
    /// exactly one source file, so a new writer either comes through here or
    /// fails that guard.
    pub(crate) fn set_n_pages(&mut self, count: u32) {
        self.fields
            .set("n-pages", MetadataValue::Int(i64::from(count)));
    }

    /// The integer under `name`, borrowed rather than materialised.
    ///
    /// [`Raster::get_field`] hands back an **owned** [`MetadataValue`], so
    /// every reader that goes through it deep-copies whatever sits under the
    /// name before looking at it. For a [`MetadataValue::Blob`] that copy is
    /// bounded only by the file that wrote it: any name can hold any type
    /// ([`Raster::try_set_field`] stores what it is given outside the
    /// built-ins) and a `.v` trailer restores arbitrary named fields with
    /// arbitrary types from an untrusted file (issue #565). Reading a `u32`
    /// out of `n-pages` should not depend on what else got stored there, so
    /// the readers borrow through here instead (issue #635).
    ///
    /// The built-in header fields answer exactly as [`Raster::get_field`]
    /// would: the six int-valued ones report their value out of the header,
    /// the string and double ones report `None` because they are not ints
    /// there either, and `filename` goes to the field list because that is
    /// where `get_field` reads it from.
    fn field_i64(&self, name: &str) -> Option<i64> {
        match name {
            "width" => Some(i64::from(self.width())),
            "height" => Some(i64::from(self.height())),
            "bands" => Some(self.format().channels() as i64),
            "xoffset" => Some(i64::from(self.xoffset())),
            "yoffset" => Some(i64::from(self.yoffset())),
            "orientation" => Some(i64::from(self.orientation())),
            "format" | "coding" | "interpretation" | "xres" | "yres" => None,
            // `filename` is the one built-in [`Raster::get_field`] answers
            // out of the field list rather than the header, so it takes the
            // same route here and reports an int if that is what is stored.
            other => match self.fields.get(other) {
                Some(&MetadataValue::Int(n)) => Some(n),
                _ => None,
            },
        }
    }

    /// The int array under `name`, borrowed rather than materialised; the
    /// array twin of `field_i64`, which does the same job for a scalar int.
    ///
    /// Every built-in header field is a scalar or a string, so none of them
    /// answers here. `filename` still routes to the attached-field list,
    /// because that is where [`Raster::get_field`] reads it from and the two
    /// have to agree on every readable name.
    fn field_int_array(&self, name: &str) -> Option<&[i64]> {
        match name {
            "width" | "height" | "bands" | "format" | "coding" | "interpretation" | "xoffset"
            | "yoffset" | "xres" | "yres" | "orientation" => None,
            other => match self.fields.get(other) {
                Some(MetadataValue::IntArray(v)) => Some(v.as_slice()),
                _ => None,
            },
        }
    }

    /// Read a metadata field as a slice of ints (libvips
    /// `vips_image_get_array_int`).
    ///
    /// Answers the same names [`Raster::get_field`] does and returns the
    /// elements when the value is a [`MetadataValue::IntArray`] — the
    /// per-frame `delay` of an animated format, and nothing else in this
    /// crate today. Returns `None` for an absent field or a value of any
    /// other type, including a scalar [`MetadataValue::Int`], which vips
    /// does not treat as a one-element array either.
    ///
    /// It borrows the stored value rather than cloning one out, so reading a
    /// delay costs nothing even when the name happens to hold a large
    /// [`MetadataValue::Blob`] instead (issue #635). Any name can hold any
    /// type — [`Raster::try_set_field`] stores what it is given, and a `.v`
    /// trailer restores arbitrary named fields with arbitrary types from an
    /// untrusted file (#565) — so going through [`Raster::get_field`] would
    /// deep-copy that blob before discovering it is not an array.
    pub fn get_int_array(&self, name: &str) -> Option<&[i64]> {
        self.field_int_array(name)
    }

    /// Remove an attached field by setting its type to 0, the libvips
    /// removal idiom (`vips_image_set` with a zero `GType` /
    /// `vips_image_remove`). Removing an absent field is a no-op.
    ///
    /// # Panics
    ///
    /// Panics when `typeof_` is non-zero (retyping a field is not
    /// supported; set a new value instead) or when `name` is a built-in
    /// header field, which cannot be removed.
    #[track_caller]
    pub fn set_typeof(&mut self, name: &str, typeof_: u64) {
        if typeof_ != 0 {
            panic!("set_typeof: only 0 (remove) is supported, got {typeof_}");
        }
        if BUILTIN_FIELDS.contains(&name) {
            panic!("set_typeof: cannot remove built-in field {name:?}");
        }
        self.fields.remove(name);
    }

    /// Attach an ICC profile, stored as the `icc-profile-data` blob
    /// field (embedded on JPEG and `.v` save).
    pub fn set_icc_profile(&mut self, profile: &[u8]) {
        self.fields
            .set("icc-profile-data", MetadataValue::Blob(profile.to_vec()));
    }

    /// Drop the attached ICC profile, the way libvips'
    /// `vips_image_remove(VIPS_META_ICC_NAME)` does. Removing an absent
    /// profile is a no-op.
    ///
    /// Crate-private because a caller already has the same reach through
    /// `set_typeof("icc-profile-data", 0)`; this exists so the ops that must
    /// do it can say what they mean. The ops in question are the inverse
    /// Fourier transforms: measured on vips 8.18.6 they retag the output
    /// `b-w`, and a three-channel profile does not survive that retag (#717,
    /// and the general rule is #720).
    pub(crate) fn remove_icc_profile(&mut self) {
        let _ = self.fields.remove("icc-profile-data");
    }

    /// The attached ICC profile, if any: the `icc-profile-data` blob set
    /// by [`Raster::set_icc_profile`] or captured from a decoded JPEG.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        match self.fields.get("icc-profile-data") {
            Some(MetadataValue::Blob(b)) => Some(b),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

/// Errors from [`Raster::save`] / [`Raster::save_stripped`] /
/// [`Raster::encode_vips`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SaveError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error(
        "unsupported save extension {extension:?}; libviprs encodes {}",
        saveable_extensions()
    )]
    UnsupportedExtension { extension: String },
    #[error("encode error: {0}")]
    Encode(#[from] SinkError),
}

/// The extensions [`Raster::save`] has an encoder behind, in the order the
/// [module table](crate::imageio) lists them.
///
/// It is a function rather than a literal inside the `#[error]` string
/// because `.jxl` and the five JPEG 2000 suffixes are only live arms when
/// their non-default feature is on. A fixed list would either promise an
/// encoder to a build that has none, or hide one from a build that has it,
/// and the message is the only thing a caller who guessed an extension ever
/// sees. The `save_error_lists_exactly_the_wired_extensions` test walks this
/// string back through [`Raster::save`], so a new arm that forgets to update
/// it fails rather than drifting.
///
/// Two optional features means four builds, so the list is assembled rather
/// than written out four times: a fifth format would make it eight literals,
/// and the third of the four is the one nobody would ever run.
fn saveable_extensions() -> &'static str {
    match (cfg!(feature = "jxl"), cfg!(feature = "jp2k")) {
        (true, true) => {
            "png, jpg/jpeg, gif, webp, jxl, jp2/j2k/jpt/j2c/jpc, fits/fit/fts, and v/vips"
        }
        (true, false) => "png, jpg/jpeg, gif, webp, jxl, fits/fit/fts, and v/vips",
        (false, true) => "png, jpg/jpeg, gif, webp, jp2/j2k/jpt/j2c/jpc, fits/fit/fts, and v/vips",
        (false, false) => "png, jpg/jpeg, gif, webp, fits/fit/fts, and v/vips",
    }
}

/// JPEG quality used by extension-dispatched [`Raster::save`], matching
/// the libvips `jpegsave` default.
const SAVE_JPEG_QUALITY: u8 = 75;

impl Raster {
    /// Save to a file, choosing the encoder from the extension (libvips
    /// `vips_image_write_to_file`). Attached metadata is kept where the
    /// container supports it; see the [module docs](crate::imageio) for
    /// the per-format table.
    ///
    /// # Errors
    ///
    /// [`SaveError::UnsupportedExtension`] for extensions this build
    /// cannot encode, which is everything outside the
    /// [module table](crate::imageio) plus `.jxl` when the crate is built
    /// without the `jxl` feature; the message enumerates the extensions
    /// that are live in this build rather than a fixed set.
    /// [`SaveError::Encode`] if the encoder rejects the pixel format, or
    /// [`SaveError::Io`] on write failure.
    pub fn save(&self, path: &Path) -> Result<(), SaveError> {
        self.save_impl(path, true)
    }

    /// Save with all metadata stripped (libvips `strip`): pixels and
    /// geometry only, no ICC, EXIF, or attached fields.
    ///
    /// # Errors
    ///
    /// As [`Raster::save`].
    pub fn save_stripped(&self, path: &Path) -> Result<(), SaveError> {
        self.save_impl(path, false)
    }

    fn save_impl(&self, path: &Path, keep_metadata: bool) -> Result<(), SaveError> {
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let bytes = self.encode_for_extension(&extension, keep_metadata)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// The extension route itself: pick the encoder from an already
    /// lowercased extension and produce the bytes, without writing them.
    ///
    /// Split out of [`Raster::save_impl`] so the dispatch table can be
    /// asserted without touching the filesystem. That is not tidiness: every
    /// test that reaches this route through [`Raster::save`] has to carry
    /// `#[cfg_attr(miri, ignore)]` and a row in
    /// `tests/miri_fs_test_inventory.txt`, because Miri aborts the whole run
    /// on the first filesystem call it refuses (#652). A route test does not
    /// need a file on disk to say which extensions have an encoder behind
    /// them, so it should not create one.
    fn encode_for_extension(
        &self,
        extension: &str,
        keep_metadata: bool,
    ) -> Result<Vec<u8>, SaveError> {
        Ok(match extension {
            "png" => crate::sink::encode_png(self)?,
            "jpg" | "jpeg" => {
                let encoded = crate::sink::encode_jpeg(self, SAVE_JPEG_QUALITY)?;
                if keep_metadata {
                    let exif = match self.fields.get("exif-data") {
                        Some(MetadataValue::Blob(b)) => Some(b.as_slice()),
                        _ => None,
                    };
                    inject_jpeg_metadata(encoded, exif, self.icc_profile())
                } else {
                    encoded
                }
            }
            // GIF carries no EXIF or ICC through libviprs' encoder, so
            // `keep_metadata` has nothing to act on: the container holds a
            // palette, a loop count and per-frame delays, all of which are
            // structural rather than EXIF-class (`vips gifsave --keep none`
            // does not drop `delay` either). WebP is the other way round and
            // takes the flag.
            "gif" => self
                .encode_gif(crate::gif::SaveOptions::default())
                .map_err(|e| match e {
                    crate::codec::EncodeError::Io(io) => SaveError::Io(io),
                    other => SaveError::Encode(SinkError::EncodeMsg(other.to_string())),
                })?,
            "webp" => crate::webp::encode_webp_for_save(self, keep_metadata)?,
            // JPEG XL takes no `keep_metadata` because `zune-jpegxl` writes
            // a bare codestream with no box container, so there is nowhere
            // to put an ICC profile, an EXIF block or an XMP packet and
            // nothing for the flag to drop. `vips jxlsave --keep none`
            // writes the same form; `--keep all` has no encoder behind it
            // here. See `crate::jxl` for the whole argument.
            //
            // Gated, so that without the `jxl` feature `.jxl` falls through
            // to `UnsupportedExtension` like any other extension with no
            // encoder, and `saveable_extensions()` above stops naming it.
            #[cfg(feature = "jxl")]
            "jxl" => crate::jxl::encode_jxl_for_save(self)?,
            // All five suffixes `jp2ksave` registers, and they are one arm
            // rather than five because vips writes the **same bytes** for all
            // of them: measured on 8.18.6, `vips copy base.v out.EXT` over
            // `jp2`, `j2k`, `jpt`, `j2c` and `jpc` produces five files with
            // one SHA-256 between them. So unlike every other row here, the
            // suffix does not pick the codec, it only gets past the sniffing
            // chain; `jp2ksave.c` hard-codes `OPJ_CODEC_JP2`.
            //
            // `keep_metadata` has nothing to act on, like GIF and FITS above:
            // `jp2ksave.c` has no code for an ICC profile, an EXIF block or an
            // XMP packet, so a stripped save and a kept one write the same
            // bytes. Asserted, not assumed.
            //
            // Gated, so that without the `jp2k` feature these five fall
            // through to `UnsupportedExtension` like any other extension with
            // no encoder, and `saveable_extensions()` stops naming them.
            #[cfg(feature = "jp2k")]
            "jp2" | "j2k" | "jpt" | "j2c" | "jpc" => crate::jp2k::encode_jp2k_for_save(self)?,
            // All three suffixes vips registers (`vips__fits_suffs`,
            // `fits.c:125`). `keep_metadata` has nothing to act on: the
            // records a FITS header carries are the geometry cfitsio
            // regenerates anyway, and vips filters them out on the way
            // back (`fits.c:596-613`), so a stripped save and a kept one
            // write the same bytes.
            "fits" | "fit" | "fts" => self.encode_fits().map_err(|e| match e {
                crate::codec::EncodeError::Io(io) => SaveError::Io(io),
                other => SaveError::Encode(SinkError::EncodeMsg(other.to_string())),
            })?,
            "v" | "vips" => self.encode_vips_impl(keep_metadata),
            _ => {
                return Err(SaveError::UnsupportedExtension {
                    extension: extension.to_owned(),
                });
            }
        })
    }

    /// Encode as native `.v` bytes (libvips `vipssave_buffer`): the
    /// 64-byte libvips header, raw pixels, and a metadata trailer. See
    /// the [module docs](crate::imageio) for the container contract.
    ///
    /// # Errors
    ///
    /// Currently infallible for every representable [`PixelFormat`];
    /// the `Result` reserves room for unencodable future formats.
    pub fn encode_vips(&self) -> Result<Vec<u8>, SaveError> {
        Ok(self.encode_vips_impl(true))
    }

    fn encode_vips_impl(&self, keep_metadata: bool) -> Vec<u8> {
        let bpc = self.format().bytes_per_channel();
        let mut out = Vec::with_capacity(VIPS_HEADER_LEN + self.data().len());
        out.extend_from_slice(&VIPS_MAGIC_NATIVE);
        fn push_i32(out: &mut Vec<u8>, v: i32) {
            out.extend_from_slice(&v.to_ne_bytes());
        }
        push_i32(&mut out, self.width() as i32);
        push_i32(&mut out, self.height() as i32);
        push_i32(&mut out, self.format().channels() as i32);
        push_i32(&mut out, 8 * bpc as i32); // deprecated Bbits
        // BandFmt (VipsBandFormat codes): 0 = uchar, 2 = ushort, 6 = float.
        let band_fmt = match bpc {
            1 => 0,
            2 => 2,
            _ => 6,
        };
        push_i32(&mut out, band_fmt);
        push_i32(&mut out, 0); // Coding: none
        push_i32(&mut out, interpretation_code(self.interpretation()));
        out.extend_from_slice(&(self.xres() as f32).to_ne_bytes());
        out.extend_from_slice(&(self.yres() as f32).to_ne_bytes());
        push_i32(&mut out, 0); // deprecated Length
        out.extend_from_slice(&0i16.to_ne_bytes()); // deprecated Compression
        out.extend_from_slice(&0i16.to_ne_bytes()); // deprecated Level
        push_i32(&mut out, self.xoffset());
        push_i32(&mut out, self.yoffset());
        out.resize(VIPS_HEADER_LEN, 0); // reserved tail of the header
        out.extend_from_slice(self.data());
        if keep_metadata && self.has_v_trailer_content() {
            if self.fields.needs_legacy_json_trailer() {
                self.append_legacy_json_trailer(&mut out);
            } else {
                self.append_vips_xml_trailer(&mut out);
            }
        }
        out
    }

    /// Append the metadata trailer libvips reads: `<root>`, a `<header>`
    /// holding the (empty) history string, and a `<meta>` block of one
    /// `<field>` per attached value plus the orientation tag.
    ///
    /// Mirrors `build_xml` (`libvips/iofuncs/vips.c:846-890` at
    /// `v8.18.0-95-gfe420cf3a`) element for element, including the two-space
    /// indent and the trailing newline, so a `.v` libviprs writes and a `.v`
    /// vips writes for the same metadata differ only where this
    /// deliberately fixes vips's escaping. See the
    /// [module docs](crate::imageio) for the type mapping and for the two
    /// divergences.
    ///
    /// The `<header>` block has to be there even though nothing reads it
    /// back: vips's parser flips out of history mode on `<meta>`
    /// (`vips.c:606-610`), and a document with a `<header>` and no `<meta>`
    /// would leave every field looking like history.
    fn append_vips_xml_trailer(&self, out: &mut Vec<u8>) {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\"?>\n");
        xml.push_str("<root xmlns=\"");
        xml.push_str(VIPS_XML_NAMESPACE);
        xml.push_str("\">\n  <header>\n    <field type=\"");
        xml.push_str(GTYPE_STRING);
        xml.push_str("\" name=\"Hist\"></field>\n  </header>\n  <meta>\n");
        for (name, value) in self.fields.known() {
            // The orientation tag is written from the header below. An
            // attached field of that name can only come from a hand-made
            // trailer, is shadowed by the header everywhere it is read, and
            // would put two `orientation` elements in one `<meta>` block.
            if name == "orientation" {
                continue;
            }
            let (gtype, text) = xml_field_of(value);
            push_xml_field(&mut xml, gtype, name, &text, XmlText::Escape);
        }
        for (name, carried) in self.fields.unknown_fields() {
            // `needs_legacy_json_trailer` is false here, so every carried
            // value is an XML one and goes back exactly as it arrived.
            if let CarriedValue::Xml { gtype, text } = carried {
                push_xml_field(&mut xml, gtype, name, text, XmlText::Verbatim);
            }
        }
        push_xml_field(
            &mut xml,
            GTYPE_INT,
            "orientation",
            &self.orientation().to_string(),
            XmlText::Escape,
        );
        xml.push_str("  </meta>\n</root>\n");
        out.extend_from_slice(xml.as_bytes());
    }

    /// Append the trailer libviprs 0.4.0 wrote, byte for byte.
    ///
    /// Only reached for a raster carrying a value that has no XML spelling;
    /// see the [module docs](crate::imageio). Keeping the exact bytes is the
    /// point, because the whole reason to take this path is that some other
    /// build has to be able to read the value back.
    fn append_legacy_json_trailer(&self, out: &mut Vec<u8>) {
        let trailer = VTrailer {
            orientation: self.orientation(),
            fields: VFields {
                entries: self
                    .fields
                    .known()
                    .map(|(name, v)| (name, VFieldValue::Known(v)))
                    .chain(
                        self.fields
                            .unknown_fields()
                            .map(|(name, v)| (name, VFieldValue::Carried(v))),
                    )
                    .collect(),
            },
        };
        if let Ok(json) = serde_json::to_vec(&trailer) {
            out.extend_from_slice(&json);
        }
    }

    /// Whether a `.v` trailer would carry anything: an orientation other
    /// than the upright default, or at least one attached field (readable
    /// or carried opaquely).
    ///
    /// An empty trailer is not free. libvips parses that slot as XML, so the
    /// 41 bytes of `{"orientation":1,"fields":{"entries":[]}}` that every
    /// plain [`Raster::save`] used to append made `vipsheader -a` print
    /// `VIPS-WARNING **: error reading vips image metadata: VipsImage: XML
    /// parse error` and throw the whole metadata block away. A file with no
    /// trailer at all reads silently (measured on vips 8.18.4, issue #546),
    /// so writing nothing is strictly better than writing nothing useful.
    fn has_v_trailer_content(&self) -> bool {
        self.orientation() != 1 || !self.fields.is_empty()
    }
}

// ---------------------------------------------------------------------------
// The .v container
// ---------------------------------------------------------------------------

/// Size of the on-disk libvips image header.
const VIPS_HEADER_LEN: usize = 64;

/// First four file bytes of a big-endian (SPARC-order) `.v` file.
///
/// `pub(crate)` for the same reason [`crate::exr::MAGIC`],
/// [`crate::fits::MAGIC`] and [`crate::radiance::MAGIC`] are: the container's
/// signature belongs to the module that owns the container, and the route
/// table in [`crate::source`] reads it from here rather than keeping a
/// second copy.
pub(crate) const VIPS_MAGIC_BE: [u8; 4] = [0x08, 0xf2, 0xa6, 0xb6];
/// First four file bytes of a little-endian (Intel-order) `.v` file.
pub(crate) const VIPS_MAGIC_LE: [u8; 4] = [0xb6, 0xa6, 0xf2, 0x08];

/// The magic this build writes: native byte order, as libvips does.
#[cfg(target_endian = "little")]
const VIPS_MAGIC_NATIVE: [u8; 4] = VIPS_MAGIC_LE;
#[cfg(target_endian = "big")]
const VIPS_MAGIC_NATIVE: [u8; 4] = VIPS_MAGIC_BE;

// --- the libvips XML metadata trailer ---------------------------------------

/// The XML namespace real libvips stamps on a `.v` metadata trailer
/// (`NAMESPACE_URI` at `libvips/iofuncs/vips.c:124`, joined with the writing
/// version by `build_xml` at `vips.c:857-860`, `v8.18.0-95-gfe420cf3a`).
///
/// The version suffix names the release whose trailer layout this writes and
/// was measured against, not the writer: vips's parser only checks that the
/// namespace starts with `.../vips` and ignores the rest
/// (`parser_element_start_handler`, `vips.c:614-621`), so a wrong version
/// here would be silent rather than loud, and a right one is worth more as
/// documentation than as a gate.
const VIPS_XML_NAMESPACE: &str = "http://www.vips.ecs.soton.ac.uk/vips/8.18.4";

/// GType name for [`MetadataValue::Int`] (`g_type_name(G_TYPE_INT)`).
const GTYPE_INT: &str = "gint";
/// GType name for [`MetadataValue::Double`].
const GTYPE_DOUBLE: &str = "gdouble";
/// GType name for [`MetadataValue::Str`] (vips's refcounted string).
const GTYPE_STRING: &str = "VipsRefString";
/// GType name for [`MetadataValue::Blob`], carried as base64
/// (`transform_blob_save_string`, `libvips/iofuncs/type.c:745-758`).
const GTYPE_BLOB: &str = "VipsBlob";
/// GType name for [`MetadataValue::IntArray`], carried as space-separated
/// decimals with a trailing separator (`transform_array_int_save_string`,
/// `libvips/iofuncs/type.c`). Measured on the pinned 8.18.6: a three-frame
/// animation's `delay` goes out as `100 100 100 `.
const GTYPE_ARRAY_INT: &str = "VipsArrayInt";

/// The `type` attribute and character data for one [`MetadataValue`]; see
/// the type table in the [module docs](crate::imageio).
///
/// The double goes out in Rust's shortest round-tripping form rather than
/// vips's `g_ascii_dtostr` `%.17g` (`type.c:438-446`), because the two agree
/// on every value that matters and the short one is what survives a
/// `f64 -> text -> f64` trip here. `g_ascii_strtod` reads it either way:
/// measured on 8.18.4, a `gdouble` field written `1e300` reads back as
/// `1e+300`.
fn xml_field_of(value: &MetadataValue) -> (&'static str, Cow<'_, str>) {
    match value {
        MetadataValue::Int(i) => (GTYPE_INT, Cow::Owned(i.to_string())),
        MetadataValue::Double(d) => (GTYPE_DOUBLE, Cow::Owned(format!("{d:?}"))),
        MetadataValue::Str(s) => (GTYPE_STRING, Cow::Borrowed(s.as_str())),
        MetadataValue::Blob(b) => (GTYPE_BLOB, Cow::Owned(base64_encode(b))),
        MetadataValue::IntArray(v) => (GTYPE_ARRAY_INT, Cow::Owned(int_array_text(v))),
    }
}

/// The character data vips writes for a `VipsArrayInt`: every element
/// followed by one space, the last one included.
///
/// The trailing separator is not a stray. Measured on the pinned 8.18.6,
/// `vips copy 'anim3.webp[n=-1]' out.v` writes
/// `<field type="VipsArrayInt" name="delay">100 100 100 </field>`, because
/// vips's save transform appends after each value rather than joining
/// between them. Nothing in this crate's own round trip can see the
/// difference, since [`parse_int_array_text`] ignores the trailing
/// whitespace either way, which is exactly why it is pinned as bytes.
fn int_array_text(values: &[i64]) -> String {
    let mut out = String::new();
    for v in values {
        out.push_str(&v.to_string());
        out.push(' ');
    }
    out
}

/// Parse the character data of a `VipsArrayInt` field.
///
/// Whitespace-separated decimals, and **all or nothing**: an element that
/// will not parse as an `i64` gives `None`, so the caller carries the whole
/// field opaquely rather than handing back the elements that happened to
/// work. That is the same rule `gint`, `gdouble` and `VipsBlob` already
/// follow in [`read_vips_xml_trailer`], and it is a deliberate divergence
/// from vips, which hands back an **empty** array for `40 x 80` and loses
/// the two elements that parsed (measured on 8.18.6: `vipsheader -f delay`
/// prints nothing and `vips copy` writes the field back out empty).
///
/// An empty element list is an empty array, not a refusal: vips writes and
/// reads that, so a `.v` carrying one has to survive a rewrite here.
///
/// The elements are `i64` where vips's `gint` is 32 bits. vips wraps rather
/// than refusing (measured: `3000000000` reads back as `-1294967296`), so a
/// narrower carrier here would lose data on a file libviprs did not write
/// and could not warn about.
fn parse_int_array_text(text: &str) -> Option<Vec<i64>> {
    text.split_whitespace()
        .map(|t| t.parse::<i64>().ok())
        .collect()
}

/// Whether [`push_xml_field`] escapes the text it is given or writes it out
/// as-is.
#[derive(Clone, Copy)]
enum XmlText {
    /// A value this build produced: escape it.
    Escape,
    /// Character data read off disk and carried opaquely: it is already
    /// escaped, and re-escaping it would double every `&`.
    Verbatim,
}

/// Append one `    <field type="..." name="...">text</field>` line, indented
/// and newline-terminated exactly as `build_xml_meta` writes it
/// (`libvips/iofuncs/vips.c:803-844`).
fn push_xml_field(out: &mut String, gtype: &str, name: &str, text: &str, mode: XmlText) {
    out.push_str("    <field type=\"");
    push_xml_attr(out, gtype);
    out.push_str("\" name=\"");
    push_xml_attr(out, name);
    out.push_str("\">");
    match mode {
        XmlText::Escape => push_xml_text(out, text),
        XmlText::Verbatim => out.push_str(text),
    }
    out.push_str("</field>\n");
}

/// Escape `s` as XML character data.
///
/// `&`, `<` and `>` become entities. A C0 control character other than tab
/// and newline is replaced by its Unicode control picture at `0x2400 + c`,
/// which is what vips does (`vips_target_write_amp`,
/// `libvips/iofuncs/target.c:821-845`) and the only thing an XML 1.0 parser
/// will take, since a numeric reference to a control character is not a
/// legal `Char`. That substitution does not reverse, so a control character
/// in a string field does not survive the round trip in either library.
///
/// A carriage return goes out as `&#x000d;` rather than literally, because a
/// literal one is folded into a newline by end-of-line normalisation
/// (XML 1.0 section 2.11) while the reference is not.
///
/// Unlike vips, bytes above 0x7f are left alone. vips tests `*p < 32` on a
/// signed `char`, so it mangles every multi-byte UTF-8 sequence it writes;
/// see the [module docs](crate::imageio).
fn push_xml_text(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\r' => out.push_str("&#x000d;"),
            '\n' | '\t' => out.push(c),
            c if (c as u32) < 0x20 => push_control_picture(out, c),
            c => out.push(c),
        }
    }
}

/// Escape `s` as an XML attribute value: [`push_xml_text`] plus the quote,
/// and with the three whitespace characters written as references because an
/// XML parser normalises literal ones to spaces (XML 1.0 section 3.3.3).
fn push_xml_attr(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\n' => out.push_str("&#x000a;"),
            '\r' => out.push_str("&#x000d;"),
            '\t' => out.push_str("&#x0009;"),
            c if (c as u32) < 0x20 => push_control_picture(out, c),
            c => out.push(c),
        }
    }
}

/// The `&#x24xx;` substitution for a C0 control character; see
/// [`push_xml_text`].
fn push_control_picture(out: &mut String, c: char) {
    use std::fmt::Write as _;
    // Infallible: writing into a String cannot fail.
    let _ = write!(out, "&#x{:04x};", 0x2400 + c as u32);
}

/// Resolve XML entity and character references in `s`.
///
/// Anything it does not recognise is left standing, which keeps a stray `&`
/// in a foreign trailer from eating the rest of the value.
fn unescape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        let Some(semi) = tail.find(';') else {
            out.push_str(tail);
            return out;
        };
        match decode_xml_entity(&tail[1..semi]) {
            Some(c) => out.push(c),
            None => out.push_str(&tail[..=semi]),
        }
        rest = &tail[semi + 1..];
    }
    out.push_str(rest);
    out
}

/// One entity body (between `&` and `;`) as a character, or `None` when it
/// is not one of the five predefined entities or a numeric reference.
fn decode_xml_entity(body: &str) -> Option<char> {
    match body {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        _ => {
            let digits = body.strip_prefix('#')?;
            let code = match digits.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => digits.parse::<u32>().ok()?,
            };
            char::from_u32(code)
        }
    }
}

/// The slice of a `.v` XML trailer that holds the metadata fields.
///
/// Normally the inside of `<meta>`, which is where `build_xml` puts them.
/// A trailer with no `<meta>` element falls back to everything after
/// `</header>`, so the history block cannot be mistaken for metadata, and a
/// bare fragment with neither is scanned whole.
fn vips_xml_meta_region(text: &str) -> &str {
    if let Some(open) = text.find("<meta>").or_else(|| text.find("<meta ")) {
        let after = &text[open..];
        let Some(gt) = after.find('>') else {
            return "";
        };
        let inner = &after[gt + 1..];
        let end = inner.find("</meta>").unwrap_or(inner.len());
        &inner[..end]
    } else if let Some(idx) = text.find("</header>") {
        &text[idx + "</header>".len()..]
    } else {
        text
    }
}

/// One `<field>` element of a `.v` XML trailer.
struct VipsXmlField<'a> {
    /// The `type` attribute: a GType name such as `gint`.
    gtype: &'a str,
    /// The `name` attribute, still escaped.
    name: &'a str,
    /// The character data between the tags, still escaped.
    text: &'a str,
}

/// Scanner over the `<field>` elements of a `.v` XML trailer.
///
/// Deliberately not a general XML parser. The trailer is one small document
/// with a shape libvips writes deterministically, this build has to read it
/// without taking on an XML dependency, and a scanner that skips what it
/// does not understand degrades the way the rest of the trailer path does:
/// a field it cannot make sense of costs that field and nothing else.
///
/// A field missing either attribute is skipped rather than guessed at: vips
/// writes both on every element, and a field with no type cannot be
/// interpreted *or* carried faithfully.
struct VipsXmlFields<'a> {
    rest: &'a str,
}

impl<'a> VipsXmlFields<'a> {
    fn new(text: &'a str) -> Self {
        Self { rest: text }
    }
}

impl<'a> Iterator for VipsXmlFields<'a> {
    type Item = VipsXmlField<'a>;

    fn next(&mut self) -> Option<VipsXmlField<'a>> {
        loop {
            let start = self.rest.find("<field")?;
            let after = &self.rest[start + "<field".len()..];
            // `<fieldset>` is not a `<field>`.
            if !after.starts_with([' ', '\t', '\r', '\n', '>', '/']) {
                self.rest = after;
                continue;
            }
            let Some(close) = after.find('>') else {
                self.rest = "";
                return None;
            };
            let attrs = &after[..close];
            let body = &after[close + 1..];
            let (text, rest) = if attrs.ends_with('/') {
                ("", body)
            } else if let Some(i) = body.find("</field>") {
                (&body[..i], &body[i + "</field>".len()..])
            } else {
                self.rest = "";
                return None;
            };
            self.rest = rest;
            let attrs = attrs.strip_suffix('/').unwrap_or(attrs);
            let mut gtype = None;
            let mut name = None;
            let pairs = XmlAttrs { rest: attrs };
            for (key, value) in pairs {
                match key {
                    "type" => gtype = Some(value),
                    "name" => name = Some(value),
                    _ => {}
                }
            }
            let (Some(gtype), Some(name)) = (gtype, name) else {
                continue;
            };
            return Some(VipsXmlField { gtype, name, text });
        }
    }
}

/// Scanner over the `name="value"` pairs of one start tag.
///
/// Pair by pair rather than by searching for `name=`, so an attribute value
/// that happens to contain another attribute's name cannot be picked up as
/// that attribute.
struct XmlAttrs<'a> {
    rest: &'a str,
}

impl<'a> Iterator for XmlAttrs<'a> {
    type Item = (&'a str, &'a str);

    fn next(&mut self) -> Option<(&'a str, &'a str)> {
        let rest = self.rest.trim_start();
        let eq = rest.find('=')?;
        let key = rest[..eq].trim_end();
        let after = rest[eq + 1..].trim_start();
        let quote = after.chars().next()?;
        if quote != '"' && quote != '\'' {
            self.rest = "";
            return None;
        }
        let body = &after[quote.len_utf8()..];
        let end = body.find(quote)?;
        self.rest = &body[end + quote.len_utf8()..];
        Some((key, &body[..end]))
    }
}

/// The standard base64 alphabet (RFC 4648), which is what `g_base64_encode`
/// uses for a `VipsBlob` save string.
const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode `data` as padded, unwrapped base64, matching `g_base64_encode`
/// (which never breaks lines) as used by `transform_blob_save_string`.
fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b1 = u32::from(chunk[0]);
        let b2 = u32::from(chunk.get(1).copied().unwrap_or(0));
        let b3 = u32::from(chunk.get(2).copied().unwrap_or(0));
        let n = (b1 << 16) | (b2 << 8) | b3;
        out.push(BASE64_ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(BASE64_ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            BASE64_ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            BASE64_ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Decode padded base64, tolerating whitespace, or `None` when `text` is not
/// base64 at all.
///
/// `None` is not a failure the caller reports: a `VipsBlob` field whose text
/// will not decode is carried opaquely instead, so a value this build cannot
/// read is still a value it does not destroy.
fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    let mut padding = 0usize;
    for byte in text.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if byte == b'=' {
            padding += 1;
            continue;
        }
        if padding > 0 {
            return None; // data after the padding
        }
        acc = (acc << 6) | base64_value(byte)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    // A well-formed group leaves 0, 2 or 4 spare bits, and they are zero.
    if padding > 2 || bits >= 6 || acc & ((1 << bits) - 1) != 0 {
        return None;
    }
    Some(out)
}

/// One base64 digit's value, or `None` when the byte is not one.
fn base64_value(byte: u8) -> Option<u32> {
    Some(match byte {
        b'A'..=b'Z' => u32::from(byte - b'A'),
        b'a'..=b'z' => u32::from(byte - b'a') + 26,
        b'0'..=b'9' => u32::from(byte - b'0') + 52,
        b'+' => 62,
        b'/' => 63,
        _ => return None,
    })
}

/// The trailer libviprs 0.4.0 and earlier wrote after the pixel data: the
/// orientation tag plus the attached fields, as JSON.
///
/// The writer produces this only for a raster carrying a value that has no
/// XML spelling; everything else gets the libvips XML trailer now (issue
/// #546). The shape stays frozen at what 0.4.0 wrote,
/// `{"orientation":N,"fields":{"entries":[[name,value],...]}}` with values
/// in [`MetadataValue`]'s externally tagged form, because the only reason to
/// take this path is that some other build has to read the value back.
///
/// Write-only. Reading goes through [`read_json_trailer`], which walks the
/// JSON entry by entry rather than deserialising into this shape, because
/// one `serde` value for the whole trailer is exactly what made a single
/// unreadable field cost the image every other one (issue #565).
#[derive(Serialize)]
struct VTrailer<'a> {
    orientation: u8,
    fields: VFields<'a>,
}

/// The attached fields as they sit in a [`VTrailer`].
#[derive(Serialize)]
struct VFields<'a> {
    entries: Vec<(&'a str, VFieldValue<'a>)>,
}

/// One trailer value on its way out to disk.
///
/// `untagged` means each arm serialises as its own contents and adds
/// nothing, so a [`MetadataValue`] keeps its externally tagged form and an
/// opaque value goes back exactly as it arrived.
#[derive(Serialize)]
#[serde(untagged)]
enum VFieldValue<'a> {
    /// A value this build understands, e.g. `{"Int":3}`.
    Known(&'a MetadataValue),
    /// A value a newer build wrote, echoed back as it arrived.
    Carried(&'a CarriedValue),
}

/// Decode a native `.v` file (both byte orders). Enforces all three of the
/// caller's [`DecodeLimits`] geometry ceilings on the untrusted header before
/// anything is allocated: the [`max_coord`](DecodeLimits::max_coord)
/// single-axis ceiling, the [`max_pixels`](DecodeLimits::max_pixels) count,
/// and the [`max_alloc_bytes`](DecodeLimits::max_alloc_bytes) budget on the
/// pixel body.
///
/// The third arrived last, as issue #710. `.v` was never a decompression-bomb
/// vector, because the body has to be physically present before it is copied,
/// so the allocation was already bounded by the input length. What was missing
/// was the contract: a caller who set `max_alloc_bytes` did not get it here,
/// and the two decode entry points disagreed about the same run of bytes,
/// since [`crate::source::decode_file_with_limits`] spends the budget on the
/// bounded whole-file read and [`crate::source::decode_bytes_with_limits`] has
/// no file to spend it on.
pub(crate) fn decode_vips_bytes(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    if bytes.len() < VIPS_HEADER_LEN {
        return Err(SourceError::VipsFormat(format!(
            "file too short for a .v header: {} bytes",
            bytes.len()
        )));
    }
    let swapped = if bytes[..4] == VIPS_MAGIC_LE {
        cfg!(target_endian = "big")
    } else if bytes[..4] == VIPS_MAGIC_BE {
        cfg!(target_endian = "little")
    } else {
        return Err(SourceError::VipsFormat("bad .v magic".to_string()));
    };
    let read_i32 = |off: usize| -> i32 {
        let raw: [u8; 4] = bytes[off..off + 4].try_into().expect("length checked");
        let v = i32::from_ne_bytes(raw);
        if swapped { v.swap_bytes() } else { v }
    };
    let read_f32 = |off: usize| -> f32 { f32::from_bits(read_i32(off) as u32) };

    let width = read_i32(4);
    let height = read_i32(8);
    let bands = read_i32(12);
    let band_fmt = read_i32(20);
    let coding = read_i32(24);
    let type_code = read_i32(28);
    let xres = read_f32(32);
    let yres = read_f32(36);
    let xoffset = read_i32(48);
    let yoffset = read_i32(52);

    if coding != 0 {
        return Err(SourceError::VipsFormat(format!(
            "unsupported .v coding {coding}; only uncoded images are supported"
        )));
    }
    let bpc = match band_fmt {
        0 => 1, // uchar
        2 => 2, // ushort
        6 => 4, // float
        other => {
            return Err(SourceError::VipsFormat(format!(
                "unsupported .v band format {other}; only uchar, ushort, and float \
                 are supported"
            )));
        }
    };
    if width <= 0 || height <= 0 || bands <= 0 {
        return Err(SourceError::VipsFormat(format!(
            "bad .v geometry {width}x{height} with {bands} bands"
        )));
    }
    let (width, height) = (width as u32, height as u32);
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    let format = PixelFormat::with_channels(bands as usize, bpc)
        .ok_or_else(|| SourceError::VipsFormat(format!("unrepresentable .v band count {bands}")))?;

    // And the allocation budget, which neither ceiling above implies: a pixel
    // count sees neither the band count nor the sample depth, so the default
    // 1-gigapixel `max_pixels` still waves a 4 GiB `Rgba8` body through. This
    // reader consulted it nowhere at all until issue #710, so it was the one
    // container out of ten where setting `max_alloc_bytes` bought nothing.
    //
    // It sits after `with_channels` rather than before, for two reasons. A
    // band count with no `PixelFormat` keeps coming back as the format error
    // it always was rather than as an allocation refusal. And the price is
    // then provably the product the copy below is sized from: `with_channels`
    // returns a format whose `channels` and `bytes_per_channel` are exactly
    // the arguments, and `bytes_per_pixel` is their product, so `bands * bpc`
    // is `format.bytes_per_pixel()` for every representable `.v`. That is why
    // there is one spelling of the product here now and not two.
    let data_len =
        limits.check_image_alloc(".v pixel buffer", width, height, bands as u64, bpc as u64)?;
    let end = usize::try_from(data_len)
        .ok()
        .and_then(|len| VIPS_HEADER_LEN.checked_add(len))
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| {
            SourceError::VipsFormat(format!(
                "truncated .v pixel data: header promises {data_len} bytes, file has {}",
                bytes.len().saturating_sub(VIPS_HEADER_LEN)
            ))
        })?;
    let mut data = bytes[VIPS_HEADER_LEN..end].to_vec();
    if swapped && bpc == 2 {
        for pair in data.as_chunks_mut::<2>().0 {
            pair.swap(0, 1);
        }
    }
    if swapped && bpc == 4 {
        for quad in data.as_chunks_mut::<4>().0 {
            quad.reverse();
        }
    }

    let mut raster = Raster::new(width, height, format, data)?;
    raster.meta.xres = f64::from(xres);
    raster.meta.yres = f64::from(yres);
    raster.meta.xoffset = xoffset;
    raster.meta.yoffset = yoffset;
    raster.meta.interpretation = interpretation_from_code(type_code);
    // Trailer: the XML block libvips writes and libviprs now writes too,
    // carrying the orientation tag and every attached field. A `.v` from
    // libviprs 0.4.0 or earlier carries the old JSON trailer there instead
    // and is still read. Anything else is treated as absent, as libvips
    // treats a trailer that is not XML.
    if end < bytes.len() {
        let trailer = &bytes[end..];
        if is_json_trailer(trailer) {
            let json: serde_json::Value = serde_json::from_slice(trailer).map_err(|err| {
                SourceError::VipsFormat(format!("corrupt .v metadata trailer: {err}"))
            })?;
            read_json_trailer(&json, &mut raster);
        } else {
            read_vips_xml_trailer(trailer, &mut raster);
        }
    }
    Ok(raster)
}

/// Whether `trailer` claims to be the legacy libviprs JSON trailer: its
/// first non-whitespace byte is `{`.
///
/// This is what picks the reader, and it is also what lets a broken JSON
/// trailer be reported where a broken XML one cannot be. Only libviprs ever
/// wrote JSON into that slot, so a `{` that will not parse is corruption
/// with a known author (issue #565). The XML slot is shared with libvips and
/// with anything else that writes a `.v`, so silence is the honest answer
/// there.
///
/// It is also the reason libviprs 0.4.0 cannot read the fields out of a `.v`
/// written now: no byte sequence starts with `{` and is the XML libvips
/// requires. See the [module docs](crate::imageio).
fn is_json_trailer(trailer: &[u8]) -> bool {
    trailer
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|&b| b == b'{')
}

/// Apply a legacy libviprs JSON trailer to `raster`, one entry at a time.
///
/// Total on purpose. Every part of the trailer is optional and every entry
/// is read on its own, so a `.v` written by a newer libviprs costs this
/// build only the entries it genuinely cannot represent and never the ones
/// it can (issue #565). An entry whose value matches no [`MetadataValue`]
/// variant is carried opaquely rather than dropped, so re-saving the image
/// here does not strip what the newer build wrote.
fn read_json_trailer(json: &serde_json::Value, raster: &mut Raster) {
    if let Some(orientation) = json
        .get("orientation")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| u8::try_from(n).ok())
    {
        raster.meta.orientation = orientation;
    }
    let Some(entries) = json
        .get("fields")
        .and_then(|fields| fields.get("entries"))
        .and_then(serde_json::Value::as_array)
    else {
        return;
    };
    for entry in entries {
        let Some([name, value]) = entry.as_array().map(Vec::as_slice) else {
            continue;
        };
        let Some(name) = name.as_str() else {
            continue;
        };
        match serde_json::from_value::<MetadataValue>(value.clone()) {
            Ok(known) => raster.fields.set(name, known),
            Err(_) => raster
                .fields
                .set_unknown(name, CarriedValue::Json(value.clone())),
        }
    }
}

/// Apply a libvips XML metadata trailer to `raster`, one `<field>` at a time.
///
/// Total on purpose, like [`read_json_trailer`]: every element is read on its
/// own, so a field this build cannot represent costs that field and nothing
/// else. A field whose `type` is not one of the four GTypes
/// [`MetadataValue`] covers, or whose text will not parse as the type it
/// claims, is carried opaquely and written back byte for byte rather than
/// dropped (issue #565).
///
/// `orientation` is a header value here, not an attached field, so it is
/// taken out of the stream: writing it back is the writer's job, and letting
/// it through as well would put two of it in the next file. Out of the 1-8
/// range it falls back to the upright default, which is how `vips autorot`
/// treats a tag it cannot use.
fn read_vips_xml_trailer(trailer: &[u8], raster: &mut Raster) {
    let Ok(text) = std::str::from_utf8(trailer) else {
        return;
    };
    for field in VipsXmlFields::new(vips_xml_meta_region(text)) {
        let name = unescape_xml(field.name);
        let gtype = unescape_xml(field.gtype);
        let value = unescape_xml(field.text);
        if name == "orientation" {
            if gtype == GTYPE_INT
                && let Ok(tag) = value.trim().parse::<u16>()
                && (1..=8).contains(&tag)
            {
                raster.meta.orientation = tag as u8;
            }
            continue;
        }
        let known = match gtype.as_str() {
            GTYPE_INT => value.trim().parse::<i64>().ok().map(MetadataValue::Int),
            GTYPE_DOUBLE => value.trim().parse::<f64>().ok().map(MetadataValue::Double),
            GTYPE_STRING => Some(MetadataValue::Str(value)),
            GTYPE_BLOB => base64_decode(value.trim()).map(MetadataValue::Blob),
            GTYPE_ARRAY_INT => parse_int_array_text(&value).map(MetadataValue::IntArray),
            _ => None,
        };
        match known {
            Some(known) => raster.fields.set(&name, known),
            None => raster.fields.set_unknown(
                &name,
                CarriedValue::Xml {
                    gtype,
                    text: field.text.to_string(),
                },
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// JPEG metadata segments
// ---------------------------------------------------------------------------

/// APP1 EXIF identifier.
const EXIF_HEADER: &[u8] = b"Exif\0\0";
/// APP2 ICC identifier.
const ICC_HEADER: &[u8] = b"ICC_PROFILE\0";
/// Maximum ICC payload bytes per APP2 segment: 65535 total segment
/// length, minus 2 length bytes, 12 identifier bytes, and 2 chunk-index
/// bytes.
const ICC_CHUNK_MAX: usize = 65535 - 2 - 12 - 2;

/// Scan a JPEG byte stream and attach the `exif-data` and
/// `icc-profile-data` fields found in its APP1/APP2 segments, as libvips
/// loaders populate the image header. Non-JPEG or malformed input
/// attaches nothing.
pub(crate) fn attach_jpeg_metadata(raster: &mut Raster, bytes: &[u8]) {
    let (exif, icc) = extract_jpeg_metadata(bytes);
    if let Some(exif) = exif {
        raster.fields.set("exif-data", MetadataValue::Blob(exif));
    }
    if let Some(icc) = icc {
        raster
            .fields
            .set("icc-profile-data", MetadataValue::Blob(icc));
    }
}

/// Extract the EXIF payload (after `Exif\0\0`) and the reassembled ICC
/// profile from a JPEG stream's application segments.
fn extract_jpeg_metadata(bytes: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut exif = None;
    let mut icc_chunks: Vec<(u8, &[u8])> = Vec::new();
    for (marker, data) in JpegSegments::new(bytes) {
        match marker {
            0xE1 if exif.is_none() && data.starts_with(EXIF_HEADER) => {
                exif = Some(data[EXIF_HEADER.len()..].to_vec());
            }
            0xE2 if data.starts_with(ICC_HEADER) && data.len() >= ICC_HEADER.len() + 2 => {
                let seq = data[ICC_HEADER.len()];
                icc_chunks.push((seq, &data[ICC_HEADER.len() + 2..]));
            }
            _ => {}
        }
    }
    let icc = if icc_chunks.is_empty() {
        None
    } else {
        icc_chunks.sort_by_key(|(seq, _)| *seq);
        let mut profile = Vec::with_capacity(icc_chunks.iter().map(|(_, d)| d.len()).sum());
        for (_, d) in &icc_chunks {
            profile.extend_from_slice(d);
        }
        Some(profile)
    };
    (exif, icc)
}

/// Iterator over JPEG marker segments: yields `(marker, segment data)`
/// for every length-carrying segment before the scan data.
struct JpegSegments<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JpegSegments<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        // Require SOI; otherwise iterate nothing.
        let pos = if bytes.starts_with(&[0xFF, 0xD8]) {
            2
        } else {
            bytes.len()
        };
        Self { bytes, pos }
    }
}

impl<'a> Iterator for JpegSegments<'a> {
    type Item = (u8, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Marker: 0xFF then a non-fill byte.
            while self.pos < self.bytes.len() && self.bytes[self.pos] == 0xFF {
                self.pos += 1;
            }
            if self.pos == 0 || self.pos >= self.bytes.len() || self.bytes[self.pos - 1] != 0xFF {
                return None;
            }
            let marker = self.bytes[self.pos];
            self.pos += 1;
            match marker {
                // EOI or SOS: nothing structured follows.
                0xD9 | 0xDA => return None,
                // Standalone markers with no length word.
                0x01 | 0xD0..=0xD7 => continue,
                _ => {
                    if self.pos + 2 > self.bytes.len() {
                        return None;
                    }
                    let len = u16::from_be_bytes([self.bytes[self.pos], self.bytes[self.pos + 1]])
                        as usize;
                    if len < 2 || self.pos + len > self.bytes.len() {
                        return None;
                    }
                    let data = &self.bytes[self.pos + 2..self.pos + len];
                    self.pos += len;
                    return Some((marker, data));
                }
            }
        }
    }
}

/// Insert APP1 (EXIF) and APP2 (ICC) segments into an encoded JPEG,
/// after the SOI and any APP0 (JFIF) segment the encoder wrote. Payloads
/// too large for their segment framing are skipped rather than
/// corrupting the stream (EXIF cannot be split; ICC splits into up to
/// 255 chunks).
fn inject_jpeg_metadata(jpeg: Vec<u8>, exif: Option<&[u8]>, icc: Option<&[u8]>) -> Vec<u8> {
    if !jpeg.starts_with(&[0xFF, 0xD8]) {
        return jpeg;
    }
    // Find the insertion point: after SOI and consecutive APP0 segments.
    let mut insert_at = 2;
    while insert_at + 4 <= jpeg.len() && jpeg[insert_at] == 0xFF && jpeg[insert_at + 1] == 0xE0 {
        let len = u16::from_be_bytes([jpeg[insert_at + 2], jpeg[insert_at + 3]]) as usize;
        if len < 2 || insert_at + 2 + len > jpeg.len() {
            break;
        }
        insert_at += 2 + len;
    }

    let mut segments = Vec::new();
    if let Some(exif) = exif {
        let payload_len = EXIF_HEADER.len() + exif.len();
        if payload_len + 2 <= u16::MAX as usize {
            segments.extend_from_slice(&[0xFF, 0xE1]);
            segments.extend_from_slice(&((payload_len + 2) as u16).to_be_bytes());
            segments.extend_from_slice(EXIF_HEADER);
            segments.extend_from_slice(exif);
        }
    }
    if let Some(icc) = icc {
        let chunks: Vec<&[u8]> = icc.chunks(ICC_CHUNK_MAX).collect();
        if !chunks.is_empty() && chunks.len() <= 255 {
            let total = chunks.len() as u8;
            for (i, chunk) in chunks.iter().enumerate() {
                let payload_len = ICC_HEADER.len() + 2 + chunk.len();
                segments.extend_from_slice(&[0xFF, 0xE2]);
                segments.extend_from_slice(&((payload_len + 2) as u16).to_be_bytes());
                segments.extend_from_slice(ICC_HEADER);
                segments.push(i as u8 + 1);
                segments.push(total);
                segments.extend_from_slice(chunk);
            }
        }
    }
    if segments.is_empty() {
        return jpeg;
    }
    let mut out = Vec::with_capacity(jpeg.len() + segments.len());
    out.extend_from_slice(&jpeg[..insert_at]);
    out.extend_from_slice(&segments);
    out.extend_from_slice(&jpeg[insert_at..]);
    out
}

// ---------------------------------------------------------------------------
// Free functions: tokeniser, thumbnail geometry, max_coord, env init
// ---------------------------------------------------------------------------

/// Split an option string into tokens the way the libvips tokeniser
/// (`vips__token_get`) does: whitespace separates tokens, double or
/// single quotes group a token, and a backslash escapes the next
/// character both inside and outside quotes.
///
/// ```
/// assert_eq!(libviprs::tokenize("a \"b c\" d"), vec!["a", "b c", "d"]);
/// ```
pub fn tokenize(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = input.chars().peekable();
    loop {
        while chars.next_if(|c| c.is_whitespace()).is_some() {}
        let Some(&first) = chars.peek() else { break };
        let mut token = String::new();
        if first == '"' || first == '\'' {
            let quote = first;
            chars.next();
            while let Some(c) = chars.next() {
                match c {
                    c if c == quote => break,
                    '\\' => {
                        if let Some(escaped) = chars.next() {
                            token.push(escaped);
                        }
                    }
                    c => token.push(c),
                }
            }
        } else {
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                chars.next();
                if c == '\\' {
                    if let Some(escaped) = chars.next() {
                        token.push(escaped);
                    }
                } else {
                    token.push(c);
                }
            }
        }
        out.push(token);
    }
    out
}

/// A parsed thumbnail size specification; see
/// [`parse_thumbnail_geometry`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThumbnailGeometry {
    /// Requested width bound, if the spec named one.
    pub width: Option<u32>,
    /// Requested height bound, if the spec named one.
    pub height: Option<u32>,
}

/// Parse a vipsthumbnail-style geometry string: `"200"` (width only),
/// `"200x150"`, `"200x"` (width only), `"x150"` (height only), with the
/// trailing vipsthumbnail modifiers `<` (only enlarge), `>` (only
/// shrink), and `!` (force) tolerated and stripped. Unparseable parts
/// read as `None`, so garbage yields an empty geometry rather than an
/// error, matching the forgiving CLI parser.
pub fn parse_thumbnail_geometry(spec: &str) -> ThumbnailGeometry {
    let s = spec.trim().trim_end_matches(['<', '>', '!']).trim();
    let (w, h) = match s.split_once(['x', 'X']) {
        Some((w, h)) => (w, Some(h)),
        None => (s, None),
    };
    ThumbnailGeometry {
        width: w.trim().parse().ok(),
        height: h.and_then(|h| h.trim().parse().ok()),
    }
}

/// The default [`DecodeLimits::max_coord`] ceiling: 10,000,000 pixels per
/// axis, the libvips `VIPS_MAX_COORD` value.
pub const DEFAULT_MAX_COORD: u32 = 10_000_000;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{decode_bytes, decode_file};

    #[test]
    fn metadata_value_len_reports_blob_and_string_bytes() {
        // The foreign magickload cell asserts `icc.len() == 564` on the ICC
        // blob; `len` on a blob is its byte count.
        assert_eq!(MetadataValue::Blob(vec![0u8; 564]).len(), 564);
        assert_eq!(MetadataValue::Str("abc".to_string()).len(), 3);
        // A scalar field is a single element.
        assert_eq!(MetadataValue::Int(7).len(), 1);
        assert_eq!(MetadataValue::Double(1.5).len(), 1);
    }

    #[test]
    fn metadata_value_is_empty_tracks_len() {
        assert!(MetadataValue::Blob(Vec::new()).is_empty());
        assert!(MetadataValue::Str(String::new()).is_empty());
        assert!(!MetadataValue::Blob(vec![1]).is_empty());
        assert!(!MetadataValue::Int(0).is_empty());
    }

    #[test]
    fn get_orientation_defaults_to_one() {
        // Matches the oracle: `orientation = 1` for the un-rotated fixtures.
        assert_eq!(Raster::black(4, 4).get_orientation(), 1);
    }

    #[test]
    fn get_orientation_reads_the_exif_field() {
        let mut im = Raster::black(4, 4);
        im.set_field("orientation", 6i32.into());
        assert_eq!(im.get_orientation(), 6);
    }

    #[test]
    fn get_n_pages_defaults_to_one() {
        // Matches the oracle: `n-pages = 1` for single-page images.
        assert_eq!(Raster::black(4, 4).get_n_pages(), 1);
    }

    #[test]
    fn get_n_pages_reads_the_attached_field() {
        let mut im = Raster::black(4, 4);
        im.set_field("n-pages", MetadataValue::Int(5));
        assert_eq!(im.get_n_pages(), 5);

        // A non-positive count falls back to the single-page default.
        let mut im0 = Raster::black(4, 4);
        im0.set_field("n-pages", MetadataValue::Int(0));
        assert_eq!(im0.get_n_pages(), 1);
    }

    #[test]
    fn get_n_pages_ports_the_whole_vips_sanity_check() {
        // `vips_image_get_n_pages` (`iofuncs/header.c:917-928`) reports a
        // single page unless the field is an int strictly between 1 and
        // 10000. Measured rather than transcribed: a C program linking
        // libvips 8.18.6 set each of these on a fresh image and printed what
        // the accessor gave back (issue #635).
        for (stored, expected) in [
            (-5i64, 1u32),
            (0, 1),
            (1, 1),
            (2, 2),
            (9_999, 9_999),
            (10_000, 1),
            (10_001, 1),
            (65_536, 1),
            (2_000_000_000, 1),
        ] {
            let mut im = Raster::black(1, 1);
            im.set_field("n-pages", MetadataValue::Int(stored));
            assert_eq!(im.get_n_pages(), expected, "stored n-pages = {stored}");
        }

        // vips reads the field with `vips_image_get_int`, which will not
        // coerce a string, so a `gchararray` "3" reports 1 there too. This
        // crate's own `get_int` refuses the same way, and the raw value
        // stays readable through `get_field`.
        let mut str_field = Raster::black(1, 1);
        str_field.set_field("n-pages", MetadataValue::Str("3".to_string()));
        assert_eq!(str_field.get_n_pages(), 1);
        assert_eq!(str_field.get_int("n-pages"), None);
        assert_eq!(str_field.get_field("n-pages").unwrap().as_str(), "3");
    }

    /// An 8x6 RGB raster, the smallest shape the JPEG 2000 tests in
    /// [`crate::jp2k`] use. `rgb_2x2` is too small for the encoder's
    /// resolution-count rule (`floor(log2(min(w, h))) - 5`), so the save-route
    /// tests take their own.
    #[allow(dead_code)]
    fn jp2k_sized() -> Raster {
        Raster::new(
            8,
            6,
            PixelFormat::Rgb8,
            (0..8u32 * 6 * 3).map(|i| (i % 251) as u8).collect(),
        )
        .unwrap()
    }

    fn rgb_2x2() -> Raster {
        Raster::new(
            2,
            2,
            PixelFormat::Rgb8,
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        )
        .unwrap()
    }

    // -- metadata fields ---------------------------------------------------

    /**
     * Tests the built-in header fields read through get_field with the
     * libvips names and value kinds.
     * Input: 2x2 Rgb8 with default metadata.
     * Output: width/height/bands ints, format "uchar", coding "none",
     * interpretation "srgb", xres/yres doubles, orientation 1.
     */
    #[test]
    fn builtin_fields_read_through() {
        let im = rgb_2x2();
        assert_eq!(im.get_field("width"), Some(MetadataValue::Int(2)));
        assert_eq!(im.get_field("height"), Some(MetadataValue::Int(2)));
        assert_eq!(im.get_field("bands"), Some(MetadataValue::Int(3)));
        assert_eq!(im.get_field("format").unwrap().as_str(), "uchar");
        assert_eq!(im.get_field("coding").unwrap().as_str(), "none");
        assert_eq!(im.get_field("interpretation").unwrap().as_str(), "srgb");
        assert_eq!(im.get_field("xres"), Some(MetadataValue::Double(1.0)));
        assert_eq!(im.get_field("yres"), Some(MetadataValue::Double(1.0)));
        assert_eq!(im.get_field("xoffset"), Some(MetadataValue::Int(0)));
        assert_eq!(im.get_field("yoffset"), Some(MetadataValue::Int(0)));
        assert_eq!(im.get_field("orientation"), Some(MetadataValue::Int(1)));
        assert_eq!(im.get_field("filename").unwrap().as_str(), "");
        assert_eq!(im.get_field("no-such-field"), None);

        let im16 = Raster::zeroed(1, 1, PixelFormat::Gray16).unwrap();
        assert_eq!(im16.get_field("format").unwrap().as_str(), "ushort");
        assert_eq!(im16.get_field("interpretation").unwrap().as_str(), "grey16");
    }

    /**
     * Tests get_fields: more than 10 names, "width" first, attached
     * fields listed after the built-ins in insertion order.
     */
    #[test]
    fn get_fields_order_and_count() {
        let mut im = rgb_2x2();
        let fields = im.get_fields();
        assert!(fields.len() > 10, "got {}", fields.len());
        assert_eq!(fields[0], "width");

        im.set_field("zz-custom", 7i32.into());
        im.set_field("aa-custom", "x".into());
        let fields = im.get_fields();
        let zz = fields.iter().position(|f| f == "zz-custom").unwrap();
        let aa = fields.iter().position(|f| f == "aa-custom").unwrap();
        assert!(zz < aa, "insertion order, not name order: {fields:?}");
    }

    /**
     * Tests set_field routing: header fields update the metadata block,
     * arbitrary string/int/double fields round-trip, and the geometry
     * fields reject writes with a typed error.
     */
    #[test]
    fn set_field_routes_builtins_and_attachments() {
        let mut im = rgb_2x2();
        im.set_field("xres", 2.5f64.into());
        im.set_field("yres", MetadataValue::Int(3));
        im.set_field("xoffset", 7i32.into());
        im.set_field("yoffset", (-4i32).into());
        im.set_field("orientation", 6i32.into());
        im.set_field("interpretation", "b-w".into());
        assert_eq!(im.xres(), 2.5);
        assert_eq!(im.yres(), 3.0);
        assert_eq!(im.xoffset(), 7);
        assert_eq!(im.yoffset(), -4);
        assert_eq!(im.orientation(), 6);
        assert_eq!(im.interpretation(), Interpretation::Bw);

        im.set_field("exif-ifd0-Software", "TestSoftware".into());
        im.set_field("page-count", 3i32.into());
        im.set_field("gamma", 2.2f64.into());
        assert_eq!(
            im.get_field("exif-ifd0-Software").unwrap().as_str(),
            "TestSoftware"
        );
        assert_eq!(im.get_field("page-count").unwrap().as_i64(), 3);
        assert_eq!(im.get_field("gamma").unwrap().as_f64(), 2.2);

        assert!(matches!(
            im.try_set_field("width", MetadataValue::Int(99)),
            Err(MetadataError::ReadOnlyField { .. })
        ));
        assert!(matches!(
            im.try_set_field("orientation", "sideways".into()),
            Err(MetadataError::WrongType { .. })
        ));
        assert!(matches!(
            im.try_set_field("orientation", MetadataValue::Int(999)),
            Err(MetadataError::WrongType { .. })
        ));
        assert!(matches!(
            im.try_set_field("interpretation", "plaid".into()),
            Err(MetadataError::UnknownInterpretation { .. })
        ));
    }

    /**
     * Tests get_typeof/set_typeof: 0 for absent fields, the value kind
     * code for present ones, and set_typeof(_, 0) removes an attachment
     * (the libvips removal idiom the ported EXIF test uses).
     */
    #[test]
    fn typeof_reports_and_removes() {
        let mut im = rgb_2x2();
        assert_eq!(im.get_typeof("exif-ifd0-Software"), 0);
        im.set_field("exif-ifd0-Software", "TestSoftware".into());
        assert_ne!(im.get_typeof("exif-ifd0-Software"), 0);
        assert_eq!(im.get_typeof("width"), MetadataValue::Int(0).type_code());
        assert_eq!(
            im.get_typeof("xres"),
            MetadataValue::Double(0.0).type_code()
        );

        im.set_typeof("exif-ifd0-Software", 0);
        assert_eq!(im.get_typeof("exif-ifd0-Software"), 0);
        assert_eq!(im.get_field("exif-ifd0-Software"), None);
        // Removing an absent field is a no-op.
        im.set_typeof("never-set", 0);
    }

    /**
     * Tests set_typeof rejects non-zero codes and built-in removal.
     */
    #[test]
    #[should_panic(expected = "only 0 (remove) is supported")]
    fn set_typeof_rejects_retype() {
        rgb_2x2().set_typeof("x", 3);
    }

    #[test]
    #[should_panic(expected = "cannot remove built-in field")]
    fn set_typeof_rejects_builtin_removal() {
        rgb_2x2().set_typeof("width", 0);
    }

    /**
     * Tests the ICC profile accessors and that the profile travels with
     * clones (the copy builder path every op uses for metadata).
     */
    #[test]
    fn icc_profile_roundtrip_and_clone() {
        let mut im = rgb_2x2();
        assert_eq!(im.icc_profile(), None);
        im.set_icc_profile(&[1, 2, 3, 4]);
        assert_eq!(im.icc_profile(), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(
            im.get_field("icc-profile-data"),
            Some(MetadataValue::Blob(vec![1, 2, 3, 4]))
        );
        let copy = im.copy().xres(9.0).build();
        assert_eq!(copy.icc_profile(), Some(&[1u8, 2, 3, 4][..]));
    }

    /**
     * Tests MetadataValue accessors panic descriptively on kind
     * mismatches and coerce where documented (Int -> f64).
     */
    #[test]
    fn metadata_value_accessors() {
        assert_eq!(MetadataValue::from("x").as_str(), "x");
        assert_eq!(MetadataValue::from(7u32).as_u32(), 7);
        assert_eq!(MetadataValue::from(-7i64).as_i64(), -7);
        assert_eq!(MetadataValue::from(2i32).as_f64(), 2.0);
        assert_eq!(MetadataValue::from(vec![9u8]).as_blob(), &[9]);
        let r = std::panic::catch_unwind(|| MetadataValue::Int(1).as_str());
        assert!(r.is_err());
        let r = std::panic::catch_unwind(|| MetadataValue::Int(-1).as_u32());
        assert!(r.is_err());
    }

    /**
     * Tests `get_int` coerces integer fields (built-in `bands` and an
     * attached loader field) to `i32`, and returns `None` for an absent
     * field, a non-integer value, and an out-of-`i32`-range integer.
     */
    #[test]
    fn get_int_reads_integer_fields() {
        let mut im = rgb_2x2();
        // Built-in header field resolves through get_field as an Int.
        assert_eq!(im.get_int("bands"), Some(3));

        // An attached loader-style field round-trips through get_int.
        im.set_field("bits-per-sample", MetadataValue::Int(8));
        assert_eq!(im.get_int("bits-per-sample"), Some(8));

        // Absent, non-integer, and out-of-range cases all yield None.
        assert_eq!(im.get_int("no-such-field"), None);
        im.set_field("note", "hello".into());
        assert_eq!(im.get_int("note"), None);
        im.set_field("huge", MetadataValue::Int(i64::from(i32::MAX) + 1));
        assert_eq!(im.get_int("huge"), None);
    }

    /**
     * Tests that `get_int` answers exactly what resolving through
     * `get_field` answers, on every readable name. It used to *be* that
     * resolution and now borrows through `field_i64` instead, so this is
     * what pins the refactor: the built-in header fields, the ones that are
     * strings or doubles there, `filename` (the one built-in `get_field`
     * reads out of the field list rather than the header), attached ints,
     * an attached blob, an attached string and a name that is not set at
     * all. Works by asking both accessors for every name `get_fields`
     * reports and comparing them (issue #635).
     */
    #[test]
    fn get_int_agrees_with_get_field_on_every_readable_name() {
        let mut im = rgb_2x2();
        im.set_field("orientation", MetadataValue::Int(6));
        im.set_field("xoffset", MetadataValue::Int(-4));
        im.set_field("yres", MetadataValue::Double(1.5));
        im.set_field("filename", MetadataValue::Int(11));
        im.set_field("bits-per-sample", MetadataValue::Int(8));
        im.set_field("icc-profile-data", MetadataValue::Blob(vec![1, 2, 3]));
        im.set_field("note", MetadataValue::Str("hello".to_string()));
        im.set_field("huge", MetadataValue::Int(i64::from(i32::MAX) + 1));

        let names = im.get_fields();
        assert!(names.len() > 12, "the sweep has to reach the attachments");
        for name in names {
            let through_get_field = match im.get_field(&name) {
                Some(MetadataValue::Int(v)) => i32::try_from(v).ok(),
                _ => None,
            };
            assert_eq!(
                im.get_int(&name),
                through_get_field,
                "get_int and get_field disagree on {name}"
            );
        }
        assert_eq!(im.get_int("no-such-field"), None);
    }

    // -- save / .v ----------------------------------------------------------

    /**
     * Tests the .v container round-trip: pixels, header metadata
     * (interpretation, resolution, offsets), orientation, and attached
     * fields (string and blob) all survive encode_vips + decode.
     */
    #[test]
    fn vips_roundtrip_preserves_pixels_and_metadata() {
        let mut im = rgb_2x2()
            .copy()
            .interpretation(Interpretation::Bw)
            .xres(4.5)
            .yres(2.25)
            .xoffset(-3)
            .yoffset(9)
            .orientation(6)
            .build();
        im.set_field("exif-data", MetadataValue::Blob(vec![9, 8, 7]));
        im.set_field("note", "hello".into());
        im.set_icc_profile(&[5, 5, 5]);

        let bytes = im.encode_vips().unwrap();
        assert_eq!(bytes[..4], VIPS_MAGIC_NATIVE);
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.width(), 2);
        assert_eq!(back.height(), 2);
        assert_eq!(back.format(), PixelFormat::Rgb8);
        assert_eq!(back.data(), im.data());
        assert_eq!(back.interpretation(), Interpretation::Bw);
        assert_eq!(back.xres(), 4.5);
        assert_eq!(back.yres(), 2.25);
        assert_eq!(back.xoffset(), -3);
        assert_eq!(back.yoffset(), 9);
        assert_eq!(back.orientation(), 6);
        assert_eq!(back.get_field("exif-data"), im.get_field("exif-data"));
        assert_eq!(back.get_field("note").unwrap().as_str(), "hello");
        assert_eq!(back.icc_profile(), Some(&[5u8, 5, 5][..]));
    }

    /**
     * Tests 16-bit .v round-trip including the byte-swap path: a file
     * written in the foreign byte order decodes to the same samples.
     */
    #[test]
    fn vips_16bit_and_foreign_endian() {
        let im = Raster::new(
            2,
            1,
            PixelFormat::Gray16,
            [513u16, 65534]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        )
        .unwrap();
        let bytes = im.encode_vips().unwrap();
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.data(), im.data());

        // Flip the file to the foreign byte order by hand: swap the
        // magic, every header word, and the 16-bit samples.
        let mut foreign = bytes.clone();
        foreign[..4].reverse();
        for off in (4..VIPS_HEADER_LEN).step_by(4) {
            // The header is i32 words except the two i16 words at 44/46,
            // which are zero here, so a 4-byte reverse is safe for this
            // all-zero region too... except it would swap their order.
            // Keep it exact: reverse i32 words, then fix the i16 pair.
            foreign[off..off + 4].reverse();
        }
        // Both i16 words are zero, so no fix-up is needed after the
        // 4-byte reverse of bytes 44..48.
        for off in (VIPS_HEADER_LEN..VIPS_HEADER_LEN + 4).step_by(2) {
            foreign[off..off + 2].reverse();
        }
        // Drop the trailer: its JSON does not participate in byte order,
        // and hand-flipping only covers header + pixels.
        foreign.truncate(VIPS_HEADER_LEN + 4);
        let swapped = decode_bytes(&foreign).unwrap();
        assert_eq!(swapped.data(), im.data());
        assert_eq!(swapped.format(), PixelFormat::Gray16);
    }

    /**
     * Tests the float .v round-trip: a FloatF32 raster encodes with
     * Bbits 32 and BandFmt 6 (FLOAT) and decodes back byte-identical,
     * an RgbaF32 raster canonicalizes back to RgbaF32, and a file in
     * the foreign byte order has its 4-byte float samples swapped.
     */
    #[test]
    fn vips_float_roundtrip_and_foreign_endian() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let im = Raster::from_f32_samples(2, 1, f1, &[0.5, -3.25]).unwrap();
        let bytes = im.encode_vips().unwrap();
        // Header words: Bbits (offset 16) is 32, BandFmt (offset 20) is 6.
        assert_eq!(i32::from_ne_bytes(bytes[16..20].try_into().unwrap()), 32);
        assert_eq!(i32::from_ne_bytes(bytes[20..24].try_into().unwrap()), 6);
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.format(), f1);
        assert_eq!(back.data(), im.data());
        assert_eq!(back.f32_samples().unwrap(), vec![0.5, -3.25]);
        assert_eq!(back.get_field("format").unwrap().as_str(), "float");

        // Four-band float canonicalizes back to the named RgbaF32.
        let rgba =
            Raster::from_f32_samples(1, 1, PixelFormat::RgbaF32, &[1.0, 2.0, 3.0, 4.5]).unwrap();
        let rgba_back = decode_bytes(&rgba.encode_vips().unwrap()).unwrap();
        assert_eq!(rgba_back.format(), PixelFormat::RgbaF32);
        assert_eq!(rgba_back.f32_samples().unwrap(), vec![1.0, 2.0, 3.0, 4.5]);

        // Flip the file to the foreign byte order by hand: reverse the
        // magic, every i32 header word (the two i16 words at 44/46 are
        // zero, so the 4-byte reverse is exact here), and each 4-byte
        // float sample. The decoder must swap the samples back.
        let mut foreign = bytes.clone();
        foreign[..4].reverse();
        for off in (4..VIPS_HEADER_LEN).step_by(4) {
            foreign[off..off + 4].reverse();
        }
        for off in (VIPS_HEADER_LEN..VIPS_HEADER_LEN + 8).step_by(4) {
            foreign[off..off + 4].reverse();
        }
        // Drop the trailer; hand-flipping only covers header + pixels.
        foreign.truncate(VIPS_HEADER_LEN + 8);
        let swapped = decode_bytes(&foreign).unwrap();
        assert_eq!(swapped.format(), f1);
        assert_eq!(swapped.f32_samples().unwrap(), vec![0.5, -3.25]);
    }

    /**
     * Tests save/save_stripped through the extension dispatcher for .v:
     * save keeps attached fields, save_stripped drops them but keeps
     * pixels and header geometry.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn save_v_and_stripped() {
        let dir = tempfile::tempdir().unwrap();
        let mut im = rgb_2x2();
        im.set_icc_profile(&[1, 2, 3]);
        im.set_field("note", "keep me".into());

        let kept = dir.path().join("kept.v");
        im.save(&kept).unwrap();
        let back = decode_file(&kept).unwrap();
        assert_eq!(back.icc_profile(), Some(&[1u8, 2, 3][..]));
        assert_eq!(back.get_field("note").unwrap().as_str(), "keep me");

        let stripped = dir.path().join("stripped.v");
        im.save_stripped(&stripped).unwrap();
        let back = decode_file(&stripped).unwrap();
        assert_eq!(back.data(), im.data());
        assert_eq!(back.icc_profile(), None);
        assert_eq!(back.get_field("note"), None);
        assert_eq!(back.get_field("exif-data"), None);
    }

    /**
     * Tests that `.webp` is a live row in the extension route and that
     * the lossless encoder behind it round-trips: the file written by
     * `save` decodes back to the same pixels, and `save_stripped` drops
     * the metadata chunks the plain `save` embeds. Works by attaching an
     * ICC blob, saving both ways, and reading each file back.
     * Input: 2x2 Rgb8 with `icc-profile-data` -> Output: identical
     * pixels from both files, the profile present after `save` and
     * absent after `save_stripped`.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn save_webp_round_trips_losslessly_and_honours_strip() {
        let dir = tempfile::tempdir().unwrap();
        let mut im = rgb_2x2();
        im.fields
            .set("icc-profile-data", MetadataValue::Blob(vec![1, 2, 3, 4]));

        let kept = dir.path().join("kept.webp");
        im.save(&kept).unwrap();
        let back = decode_file(&kept).unwrap();
        assert_eq!(back.data(), im.data(), "the WebP encoder is lossless");
        assert_eq!(back.icc_profile(), Some(&[1u8, 2, 3, 4][..]));

        let stripped = dir.path().join("stripped.webp");
        im.save_stripped(&stripped).unwrap();
        let bare = decode_file(&stripped).unwrap();
        assert_eq!(bare.data(), im.data());
        assert_eq!(bare.icc_profile(), None);
    }

    /**
     * Tests that `.jxl` is a live row in the extension route, that the
     * lossless encoder behind it round-trips, and that `save_stripped`
     * makes no difference here: the encoder writes a bare codestream with
     * no box container, so there is nothing for the strip flag to drop and
     * both files are byte-identical. That is the one place the `.jxl` row
     * differs from the `.webp` one above, and it is worth pinning rather
     * than leaving as an accident. Works by attaching an ICC blob, saving
     * both ways, and reading each file back.
     * Input: 2x2 Rgb8 with `icc-profile-data` -> Output: identical pixels
     * from both files, identical bytes on disk, and the profile absent
     * from both because nothing carried it.
     */
    #[test]
    #[cfg(feature = "jxl")]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn save_jxl_round_trips_losslessly_and_carries_no_metadata_either_way() {
        let dir = tempfile::tempdir().unwrap();
        let mut im = rgb_2x2();
        im.fields
            .set("icc-profile-data", MetadataValue::Blob(vec![1, 2, 3, 4]));

        let kept = dir.path().join("kept.jxl");
        im.save(&kept).unwrap();
        let back = decode_file(&kept).unwrap();
        assert_eq!(back.data(), im.data(), "the JPEG XL encoder is lossless");

        let stripped = dir.path().join("stripped.jxl");
        im.save_stripped(&stripped).unwrap();
        assert_eq!(
            std::fs::read(&kept).unwrap(),
            std::fs::read(&stripped).unwrap(),
            "there is no box container to strip, so both writes are the same bytes"
        );
        // The profile the raster carried is not in either file; what comes
        // back is the one `jxlload` synthesises for the colour encoding,
        // which is never the four bytes attached above.
        assert_ne!(back.icc_profile(), Some(&[1u8, 2, 3, 4][..]));
    }

    /// Every suffix `jp2ksave` registers is a live row in the extension route,
    /// and all of them write the **same** JP2 container (issue #770).
    ///
    /// This is the one row in the table where the suffix does not pick the
    /// codec, and that is measured rather than read out of `jp2ksave.c`. On
    /// the pinned vips 8.18.6:
    ///
    /// ```text
    /// vips black base.v 8 6 --bands 3
    /// for ext in jp2 j2k jpt j2c jpc; do vips copy base.v out.$ext; done
    /// ```
    ///
    /// writes five files with one SHA-256 between them
    /// (`fbe9f8f7fbe8d044...`), while `out.jp2000` and `out.xyz` are refused
    /// with "is not a known file format". So five rows, one encoder, and the
    /// refusal still has to work: the negative half is the positive control
    /// for the positive half, since a route that accepted everything would
    /// pass the first assertion on its own.
    ///
    /// Goes through `encode_for_extension` rather than [`Raster::save`] so it
    /// needs no tempdir, no `#[cfg_attr(miri, ignore)]` and no row in
    /// `tests/miri_fs_test_inventory.txt`.
    #[test]
    #[cfg(feature = "jp2k")]
    fn every_suffix_jp2ksave_registers_is_a_row_and_they_all_write_one_container() {
        let im = jp2k_sized();
        let direct = im
            .encode_jp2k(crate::jp2k::SaveOptions::default())
            .expect("the encoder takes an 8x6 RGB raster");

        for extension in ["jp2", "j2k", "jpt", "j2c", "jpc"] {
            let bytes = im
                .encode_for_extension(extension, true)
                .unwrap_or_else(|e| panic!(".{extension} must be a live row, got {e}"));
            assert_eq!(
                bytes, direct,
                ".{extension} must write the same JP2 container as encode_jp2k"
            );
        }

        // vips refuses these two, so the route has to as well, or "every
        // suffix is a row" would just mean "every string is a row".
        for extension in ["jp2000", "xyz"] {
            assert!(
                matches!(
                    im.encode_for_extension(extension, true),
                    Err(SaveError::UnsupportedExtension { .. })
                ),
                ".{extension} is not a format vips knows either"
            );
        }
    }

    /// The JPEG 2000 row takes no `keep_metadata`, because there is nothing
    /// for it to drop (issue #770).
    ///
    /// `jp2ksave.c` has no code for an ICC profile, an EXIF block or an XMP
    /// packet, so a stripped save and a kept one write the same bytes. Same
    /// shape as the `.jxl` row, and worth pinning rather than leaving as an
    /// accident: the day the encoder learns to embed a profile, this says so.
    ///
    /// The control is `.webp` in the same assertion, which is the row that
    /// genuinely does carry metadata and genuinely does differ under the flag.
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_jp2k_row_has_nothing_for_the_strip_flag_to_drop() {
        let mut im = jp2k_sized();
        im.set_icc_profile(&[1, 2, 3, 4]);
        im.fields
            .set("exif-data", MetadataValue::Blob(vec![9, 8, 7]));

        let kept = im.encode_for_extension("jp2", true).unwrap();
        assert_eq!(
            kept,
            im.encode_for_extension("jp2", false).unwrap(),
            "jp2ksave writes no metadata, so the strip flag cannot change the bytes"
        );
        assert_ne!(
            im.encode_for_extension("webp", true).unwrap(),
            im.encode_for_extension("webp", false).unwrap(),
            "positive control: the WebP row does carry metadata and does differ"
        );

        // Identical bytes on their own would also be what a row that stripped
        // *both* ways produces, so the second half says which way it went: the
        // profile is not in the file at all, under either flag. Read back
        // through the decoder rather than by scanning for the four bytes,
        // because a JP2 could legitimately contain them by accident.
        let back = crate::jp2k::decode_jp2k(&kept, DecodeLimits::default())
            .expect("the container this row writes must read back");
        assert_eq!(
            back.icc_profile(),
            None,
            "jp2ksave writes no colr box for an attached profile, so nothing \
             carries it and there is nothing for the flag to drop"
        );
        assert_eq!(back.get_field("exif-data"), None);
        // And the pixels did survive, so "no metadata" is not "no file".
        assert_eq!(back.width(), 8);
        assert_eq!(back.height(), 6);
    }

    /// Without the `jp2k` feature the five suffixes fall through to
    /// `UnsupportedExtension` like any other extension with no encoder, and
    /// the refusal message stops naming them (issue #770).
    ///
    /// The failure this guards against is not "the save fails": it is the save
    /// failing with a *JPEG 2000* error, which would tell a caller the format
    /// is broken when the truth is that this build has no encoder for it.
    #[test]
    #[cfg(not(feature = "jp2k"))]
    fn without_the_jp2k_feature_the_five_suffixes_are_plain_unsupported_extensions() {
        let im = jp2k_sized();
        for extension in ["jp2", "j2k", "jpt", "j2c", "jpc"] {
            let err = im.encode_for_extension(extension, true).unwrap_err();
            assert!(
                matches!(&err, SaveError::UnsupportedExtension { extension: e } if e == extension),
                ".{extension} must read as an unsupported extension, got {err}"
            );
            assert!(
                !saveable_extensions().contains(extension),
                "the refusal must not advertise .{extension}: {}",
                saveable_extensions()
            );
        }
        // Positive control: a row that *is* live in every build.
        assert!(im.encode_for_extension("png", true).is_ok());
    }

    /// A 3-band `f32` linear-light ramp reaching past the SDR ceiling: the
    /// input contract [`crate::uhdr::encode_uhdr`] computes a gain map from,
    /// and the one raster on which "the extension route did not write Ultra
    /// HDR" is a claim with anything behind it.
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

    /**
     * Tests that **no** extension selects the Ultra HDR writer, which is the
     * measured answer rather than a gap (issue #809).
     *
     * Every other row in this table exists because vips registers the suffix.
     * `uhdrsave` registers none: on the pinned 8.18.6, `vips -l` reports
     * `VipsForeignSaveUhdrFile (uhdrsave), save image in UltraHDR format,
     * nocache (), priority=0` with an empty suffix list, and
     * `vips copy base.v out.uhdr` is refused with `"out.uhdr" is not a known
     * file format`. So this table has nothing to add, and the format is
     * reached by name through `Raster::encode_to_buffer("uhdr")` and
     * [`Raster::encode_uhdr`] instead.
     *
     * The half that is worth a check is the collision. `uhdrload` **does**
     * claim `.jpg`, `.jpeg`, `.jpe` and `.jfif` on the way in, at priority
     * 100 against `jpegload`'s 50, so the obvious way to "fix" this issue
     * later is to make `.jpg` route to Ultra HDR when the raster happens to
     * suit it. vips does not: `vips copy base.v out.jpg` writes 803 bytes
     * that `vips uhdrload` then refuses with `not an UltraHDR image`. This
     * pins the same answer here, on the one raster where the two routes could
     * possibly disagree.
     *
     * `is_uhdr` on the container `encode_uhdr` writes is the positive control.
     * Without it, every assertion below is "this predicate said no", which a
     * predicate that always says no also satisfies.
     */
    #[test]
    fn no_extension_selects_the_ultra_hdr_writer_because_vips_registers_none() {
        let hdr = scrgb_ramp(16, 16);

        // The control: this raster genuinely does have an Ultra HDR encoding,
        // and the gate genuinely does recognise it.
        let container = hdr
            .encode_uhdr(crate::uhdr::SaveOptions::default().quality)
            .expect("a 3-band f32 raster encodes");
        assert!(
            crate::uhdr::is_uhdr(&container),
            "positive control: the gate has to say yes to something"
        );

        // `.uhdr` is not a row, and the refusal does not advertise one.
        let err = hdr.encode_for_extension("uhdr", true).unwrap_err();
        assert!(
            matches!(&err, SaveError::UnsupportedExtension { extension } if extension == "uhdr"),
            "vips refuses the same suffix, so this must too, got {err}"
        );
        assert!(
            !saveable_extensions().contains("uhdr"),
            "the refusal must not advertise .uhdr: {}",
            saveable_extensions()
        );

        // And the four suffixes `uhdrload` claims on the way in stay plain
        // JPEG on the way out: either this build has no row for them, or the
        // row writes something the Ultra HDR gate does not recognise.
        for extension in ["jpg", "jpeg", "jpe", "jfif"] {
            match hdr.encode_for_extension(extension, true) {
                Ok(bytes) => assert!(
                    !crate::uhdr::is_uhdr(&bytes),
                    ".{extension} must not select the Ultra HDR writer: vips routes it to \
                     jpegsave and `vips uhdrload` refuses the result"
                ),
                Err(SaveError::UnsupportedExtension { .. } | SaveError::Encode(_)) => {}
                Err(other) => panic!(".{extension} answered with {other}"),
            }
        }
    }

    /**
     * Tests that the `UnsupportedExtension` message names exactly the
     * extensions this build has an encoder behind, so the string and the
     * match arms cannot drift apart. They already did once: the `.jxl` arm
     * landed while the message still read "libviprs encodes png, jpg/jpeg,
     * gif, webp, and v/vips", so `save("x.avif")` told the caller JPEG XL
     * was unsupported at the moment it became supported. The same thing was
     * about to happen to the five JPEG 2000 suffixes (#770), which is why
     * they are swept here by feature rather than named in one build.
     * `jp2000` sits in the unlisted set because vips refuses that suffix too,
     * measured, so it is the nearest miss to a live row. Works by parsing
     * the extension list back out of a rendered message, saving under every
     * name it holds, and then checking a name it does not hold is refused.
     * Input: the message from `save("out.avif")` -> Output: every listed
     * extension writes a file, `jxl` is listed exactly when the feature is
     * on, and the unlisted extensions come back as
     * `SaveError::UnsupportedExtension`.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn save_error_lists_exactly_the_wired_extensions() {
        let dir = tempfile::tempdir().unwrap();
        let im = rgb_2x2();

        let message = im
            .save(&dir.path().join("out.avif"))
            .unwrap_err()
            .to_string();
        let listed = message
            .split_once("libviprs encodes ")
            .expect("the refusal names the encodable set")
            .1;
        // "png, jpg/jpeg, gif, webp, and v/vips" -> the individual
        // extensions, with the prose comma-and and the `/` alternatives
        // taken apart.
        let extensions: Vec<&str> = listed
            .split(", ")
            .map(|part| part.trim_start_matches("and "))
            .flat_map(|part| part.split('/'))
            .collect();
        assert_eq!(
            extensions.contains(&"jxl"),
            cfg!(feature = "jxl"),
            "the message follows the feature: {message}"
        );
        for suffix in ["jp2", "j2k", "jpt", "j2c", "jpc"] {
            assert_eq!(
                extensions.contains(&suffix),
                cfg!(feature = "jp2k"),
                "the message follows the feature for .{suffix}: {message}"
            );
        }

        for extension in &extensions {
            let path = dir.path().join(format!("listed.{extension}"));
            im.save(&path)
                .unwrap_or_else(|e| panic!("save(.{extension}) is a live arm, got {e}"));
            assert!(path.exists(), ".{extension} wrote a file");
        }

        // The other direction, so the list cannot go stale by growing
        // either: an extension it does not name has no arm behind it.
        let mut unlisted = vec!["avif", "heic", "tif", "jp2000"];
        if !cfg!(feature = "jxl") {
            unlisted.push("jxl");
        }
        if !cfg!(feature = "jp2k") {
            unlisted.extend(["jp2", "j2k", "jpt", "j2c", "jpc"]);
        }
        for extension in unlisted {
            assert!(
                !extensions.contains(&extension),
                ".{extension} is not in {listed:?}"
            );
            let err = im
                .save(&dir.path().join(format!("unlisted.{extension}")))
                .unwrap_err();
            assert!(
                matches!(err, SaveError::UnsupportedExtension { .. }),
                "{err}"
            );
        }
    }

    /**
     * Tests save dispatch to PNG: the file decodes back to the same
     * pixels (PNG is lossless), and unknown extensions error.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn save_png_and_unknown_extension() {
        let dir = tempfile::tempdir().unwrap();
        let im = rgb_2x2();
        let png = dir.path().join("out.png");
        im.save(&png).unwrap();
        let back = decode_file(&png).unwrap();
        assert_eq!(back.width(), 2);
        assert_eq!(back.data(), im.data());

        let err = im.save(&dir.path().join("noextension")).unwrap_err();
        assert!(
            matches!(err, SaveError::UnsupportedExtension { .. }),
            "{err}"
        );
    }

    /**
     * Tests JPEG save keeps the ICC profile and EXIF blob while
     * save_stripped drops them: the keep_icc/keep_none contract of the
     * ported infrastructure suite, end to end through the decoder's
     * segment scan.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn save_jpeg_keeps_and_strips_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let mut im = Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap();
        let icc: Vec<u8> = (0u8..=255).cycle().take(400).collect();
        im.set_icc_profile(&icc);
        im.set_field("exif-data", MetadataValue::Blob(vec![0x4D, 0x4D, 0, 42]));

        let kept = dir.path().join("kept.jpg");
        im.save(&kept).unwrap();
        let back = decode_file(&kept).unwrap();
        assert_eq!(back.icc_profile(), Some(icc.as_slice()));
        assert_eq!(
            back.get_field("exif-data"),
            Some(MetadataValue::Blob(vec![0x4D, 0x4D, 0, 42]))
        );

        let stripped = dir.path().join("stripped.jpg");
        im.save_stripped(&stripped).unwrap();
        let back = decode_file(&stripped).unwrap();
        assert_eq!(back.icc_profile(), None);
        assert_eq!(back.get_field("icc-profile-data"), None);
        assert_eq!(back.get_field("exif-data"), None);
    }

    /**
     * Tests decode_file attaches the filename field, like the libvips
     * header's filename slot.
     */
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn decode_file_sets_filename() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("named.v");
        rgb_2x2().save(&path).unwrap();
        let back = decode_file(&path).unwrap();
        assert_eq!(
            back.get_field("filename").unwrap().as_str(),
            path.display().to_string()
        );
    }

    /**
     * Tests multi-chunk ICC embedding: a profile larger than one APP2
     * segment splits into ordered chunks and reassembles on decode.
     */
    #[test]
    fn jpeg_icc_multichunk_roundtrip() {
        let icc: Vec<u8> = (0..(ICC_CHUNK_MAX + 1234))
            .map(|i| (i % 251) as u8)
            .collect();
        let im = Raster::zeroed(4, 4, PixelFormat::Gray8).unwrap();
        let jpeg = crate::sink::encode_jpeg(&im, 75).unwrap();
        let with_icc = inject_jpeg_metadata(jpeg, None, Some(&icc));
        let (exif, got) = extract_jpeg_metadata(&with_icc);
        assert_eq!(exif, None);
        assert_eq!(got.as_deref(), Some(icc.as_slice()));
        // The stream still decodes as a JPEG.
        assert!(decode_bytes(&with_icc).is_ok());
    }

    /**
     * Tests the .v reader's typed rejections: bad magic, truncated
     * header, truncated pixels, unsupported band format, and coding.
     */
    #[test]
    fn vips_decode_rejects_malformed() {
        let im = rgb_2x2();
        let good = im.encode_vips().unwrap();

        assert!(matches!(
            decode_vips_bytes(&good[..10], DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        let mut bad_magic = good.clone();
        bad_magic[0] = 0;
        assert!(matches!(
            decode_vips_bytes(&bad_magic, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        let mut truncated = good.clone();
        truncated.truncate(VIPS_HEADER_LEN + 2);
        assert!(matches!(
            decode_vips_bytes(&truncated, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        let mut double_fmt = good.clone();
        double_fmt[20..24].copy_from_slice(&8i32.to_ne_bytes()); // DOUBLE
        assert!(matches!(
            decode_vips_bytes(&double_fmt, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        // FLOAT (6) is a supported band format now, but retagging the
        // 8-bit fixture as float quadruples the promised pixel bytes.
        // Use the trailer-free encoding (the JSON metadata trailer would
        // otherwise be long enough to be eaten as pixel data): the
        // shortfall must surface as a truncation error, not a misread.
        let mut float_fmt = im.encode_vips_impl(false);
        float_fmt[20..24].copy_from_slice(&6i32.to_ne_bytes()); // FLOAT
        assert!(matches!(
            decode_vips_bytes(&float_fmt, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        let mut coded = good.clone();
        coded[24..28].copy_from_slice(&1i32.to_ne_bytes()); // LABQ coding
        assert!(matches!(
            decode_vips_bytes(&coded, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));
    }

    /**
     * Tests a real-libvips-style trailer (XML, not JSON) is tolerated:
     * pixels and header decode, attached fields read as absent.
     */
    #[test]
    fn vips_decode_ignores_foreign_trailer() {
        let im = rgb_2x2();
        let mut bytes = im.encode_vips_impl(false);
        bytes.extend_from_slice(b"<root xmlns=\"http://www.vips.ecs.soton.ac.uk//dsim\"/>");
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.data(), im.data());
        assert_eq!(back.get_field("note"), None);
    }

    /// A `.v` file written by real libvips stores metadata as an XML block
    /// (not our JSON), so its orientation was previously lost and `autorot`
    /// had no cross-oracle. The decoder now recovers the orientation tag from
    /// the `<field type="gint" name="orientation">N</field>` element that vips
    /// writes, matching what `vipsheader -a` reports for the same file.
    #[test]
    fn vips_decode_reads_orientation_from_xml_trailer() {
        let im = rgb_2x2();
        let mut bytes = im.encode_vips_impl(false);
        bytes.extend_from_slice(
            b"<?xml version=\"1.0\"?>\n<root xmlns=\"http://www.vips.ecs.soton.ac.uk/vips/8.18.4\">\n\
              <meta>\n    <field type=\"VipsRefString\" name=\"exif-ifd0-Orientation\">\
              1 (Top-left, Short, 1 components, 2 bytes)</field>\n\
              <field type=\"gint\" name=\"orientation\">6</field>\n  </meta>\n</root>\n",
        );
        let back = decode_bytes(&bytes).unwrap();
        // Pixels and geometry are untouched; only the orientation is recovered.
        assert_eq!(back.data(), im.data());
        assert_eq!(back.orientation(), 6);
        // And autorot of the orientation-6 image rotates 2x2 → 2x2 (rot90) and
        // clears the tag, as `vips autorot` does.
        let rot = back.autorot();
        assert_eq!(rot.orientation(), 1);
    }

    /// Unit-covers the XML trailer reader on the shapes that used to be the
    /// orientation extractor's whole job: it finds the lowercase
    /// `orientation` gint field, ignores the distinct
    /// `exif-ifd0-Orientation` string field, and leaves the upright default
    /// standing for a value that is out of range, not a number, or on a
    /// field with no type at all.
    #[test]
    fn vips_xml_trailer_orientation_cases() {
        fn orientation_of(fragment: &[u8]) -> u8 {
            let mut im = rgb_2x2();
            read_vips_xml_trailer(fragment, &mut im);
            im.orientation()
        }

        assert_eq!(
            orientation_of(b"<field type=\"gint\" name=\"orientation\">8</field>"),
            8
        );
        for fragment in [
            // The exif string field alone must not be mistaken for the tag.
            &b"<field type=\"VipsRefString\" name=\"exif-ifd0-Orientation\">1 (Top-left)</field>"[..],
            // Out of range, not a number, no type to read it as, and a
            // longer name that merely starts with "orientation".
            b"<field type=\"gint\" name=\"orientation\">9</field>",
            b"<field type=\"gint\" name=\"orientation\">x</field>",
            b"<field name=\"orientation\">6</field>",
            b"<field type=\"gint\" name=\"orientation-foo\">6</field>",
            b"no field here",
        ] {
            assert_eq!(
                orientation_of(fragment),
                1,
                "fragment: {}",
                String::from_utf8_lossy(fragment)
            );
        }
    }

    // -- .v trailer: nothing to say, nothing written (issue #546) ------------

    /// A raster with no attached fields and the upright orientation has
    /// nothing to put in the trailer, so nothing is written there.
    ///
    /// The 41 bytes it used to write, `{"orientation":1,"fields":{"entries":
    /// []}}`, are the whole of issue #546 for the common case: libvips parses
    /// that slot as XML, so every plain `save()` made `vipsheader -a` print
    /// `VIPS-WARNING **: error reading vips image metadata: VipsImage: XML
    /// parse error` and drop the metadata block. Measured on vips 8.18.4: the
    /// same file with the trailer truncated off reads silently.
    #[test]
    fn v_trailer_is_absent_when_there_is_nothing_to_say() {
        let im = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        let bytes = im.encode_vips().unwrap();
        assert_eq!(
            bytes.len(),
            VIPS_HEADER_LEN + im.data().len(),
            "a raster with no metadata must write header + pixels and stop, got \
             {} trailer bytes: {:?}",
            bytes.len() - VIPS_HEADER_LEN - im.data().len(),
            String::from_utf8_lossy(&bytes[VIPS_HEADER_LEN + im.data().len()..])
        );
        // And it still reads back as the same image.
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.data(), im.data());
        assert_eq!(back.orientation(), 1);
        assert_eq!(
            back.get_fields(),
            im.get_fields(),
            "no attached field may appear out of a trailer that was never written"
        );
    }

    /// The other side of the same rule: anything the trailer would actually
    /// carry brings it back. A non-default orientation, an attached field,
    /// and a value carried opaquely from a newer build each count on their
    /// own, because each of them is metadata that would otherwise be lost.
    #[test]
    fn v_trailer_is_written_whenever_it_carries_something() {
        let plain = Raster::zeroed(2, 2, PixelFormat::Rgb8).unwrap();
        let body_len = VIPS_HEADER_LEN + plain.data().len();

        let rotated = plain.copy().orientation(6).build();
        let bytes = rotated.encode_vips().unwrap();
        assert!(
            bytes.len() > body_len,
            "a non-default orientation must still be written"
        );
        assert_eq!(decode_bytes(&bytes).unwrap().orientation(), 6);

        let mut noted = plain.clone();
        noted.set_field("note", "hello".into());
        let bytes = noted.encode_vips().unwrap();
        assert!(
            bytes.len() > body_len,
            "an attached field must still be written"
        );
        assert_eq!(
            decode_bytes(&bytes)
                .unwrap()
                .get_field("note")
                .unwrap()
                .as_str(),
            "hello"
        );

        // A value this build cannot interpret is metadata too: it is the only
        // thing left on this raster once the four readable fields are removed
        // and the orientation is back to upright, and it must still be
        // written (issue #565).
        let mut carried = decode_bytes(&file_from_a_newer_build()).unwrap();
        for name in ["note", "n-pages", "delay", "xres-hint", "icc-profile-data"] {
            carried.set_typeof(name, 0);
        }
        let carried = carried.copy().orientation(1).build();
        assert!(
            carried.encode_vips().unwrap().len() > VIPS_HEADER_LEN + carried.data().len(),
            "a value carried opaquely from a newer build must still be written"
        );
    }

    // -- .v trailer forward compatibility (issue #565) -----------------------

    /// A `MetadataValue` as a *newer* libviprs writes it: the five variants
    /// this build has, plus one it does not. The whole point of the #565 lane
    /// is that adding a variant must not cost an older reader the rest of its
    /// metadata, so the future writer is modelled here instead of shipped.
    ///
    /// `IntArray` used to be the unknown one. #787 shipped it, so it is the
    /// **positive control** now: the same file exercises a variant this build
    /// reads and one it does not, and the reader has to tell them apart. The
    /// unknown one is `DoubleArray`, which is not invented either —
    /// `VipsArrayDouble` is live in a `.v` trailer today (vips writes
    /// `background` as one) and is the next variant in the queue.
    ///
    /// The derive carries no serde attributes, exactly like [`MetadataValue`],
    /// so the bytes it produces are the bytes a future build would produce.
    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    enum FutureMetadataValue {
        Int(i64),
        Double(f64),
        Str(String),
        Blob(Vec<u8>),
        IntArray(Vec<i64>),
        /// The variant this build has never heard of.
        DoubleArray(Vec<f64>),
    }

    /// The attached-field list as a newer libviprs writes it.
    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct FutureFields {
        entries: Vec<(String, FutureMetadataValue)>,
    }

    /// The `.v` trailer as a newer libviprs writes it.
    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct FutureTrailer {
        orientation: u8,
        fields: FutureFields,
    }

    /// The trailer reader as libviprs shipped it *before* this change:
    /// `serde_json::from_slice` straight onto a struct holding the plain
    /// externally tagged [`MetadataValue`]. Keeping it verbatim makes "an
    /// already-released build can still read what we write" an executed
    /// assertion rather than a claim about bytes.
    #[derive(Debug, PartialEq, Deserialize)]
    struct ReleasedTrailer {
        orientation: u8,
        fields: ReleasedFields,
    }

    /// The released reader's field list; see [`ReleasedTrailer`].
    #[derive(Debug, PartialEq, Deserialize)]
    struct ReleasedFields {
        entries: Vec<(String, MetadataValue)>,
    }

    /// The trailer libviprs 0.4 wrote, captured byte for byte from the
    /// released encoder for a 2x2 RGB raster carrying one field of each
    /// variant and orientation 6. Frozen here so the reader is pinned
    /// against the shipped format and not against whatever the current
    /// writer happens to emit.
    const RELEASED_TRAILER: &[u8] = br#"{"orientation":6,"fields":{"entries":[["note",{"Str":"hello"}],["n-pages",{"Int":3}],["xres-hint",{"Double":1.5}],["icc-profile-data",{"Blob":[5,5,5]}]]}}"#;

    /// Header and pixels for a 2x2 RGB raster with no trailer, so a test can
    /// staple an arbitrary trailer onto a real `.v` body.
    fn v_body() -> Vec<u8> {
        Raster::new(2, 2, PixelFormat::Rgb8, vec![7u8; 12])
            .unwrap()
            .encode_vips_impl(false)
    }

    /// libviprs 0.4.0's trailer reader, verbatim: try the JSON trailer as one
    /// `serde` value, and on failure fall back to scratching the orientation
    /// tag out of an XML block. Reproduced rather than described so
    /// "what an already-released build gets out of a file written now" is an
    /// executed assertion.
    fn released_reader(trailer: &[u8]) -> (Option<u8>, Vec<(String, MetadataValue)>) {
        match serde_json::from_slice::<ReleasedTrailer>(trailer) {
            Ok(parsed) => (Some(parsed.orientation), parsed.fields.entries),
            Err(_) => (released_xml_orientation(trailer), Vec::new()),
        }
    }

    /// libviprs 0.4.0's `parse_vips_xml_orientation`, verbatim; see
    /// [`released_reader`].
    fn released_xml_orientation(trailer: &[u8]) -> Option<u8> {
        let text = std::str::from_utf8(trailer).ok()?;
        let anchor = text.find(r#"name="orientation">"#)?;
        let after = &text[anchor..];
        let open = after.find('>')?;
        let rest = &after[open + 1..];
        let close = rest.find('<')?;
        let value: u16 = rest[..close].trim().parse().ok()?;
        (1..=8).contains(&value).then_some(value as u8)
    }

    /// The `(name, type, text)` of every `<field>` in the trailer of a `.v`
    /// encoded from a 2x2 RGB raster, in file order.
    fn trailer_fields(bytes: &[u8]) -> Vec<(String, String, String)> {
        let trailer =
            std::str::from_utf8(&bytes[v_body().len()..]).expect("the trailer must be UTF-8");
        VipsXmlFields::new(vips_xml_meta_region(trailer))
            .map(|f| {
                (
                    unescape_xml(f.name),
                    unescape_xml(f.gtype),
                    f.text.to_string(),
                )
            })
            .collect()
    }

    /// The metadata trailer of a `.v` written by the real thing, captured
    /// byte for byte from vips 8.18.4 (`/opt/homebrew/bin/vips`) with
    ///
    /// ```text
    /// vips black plain.v 2 2 --bands 1
    /// ```
    ///
    /// The EXIF blob is vips's own doing: `vips_foreign_save_build` runs
    /// `vips__exif_update` on every save, so a file that started with no
    /// metadata still ends up with one, which makes this a fair sample of
    /// what the reader meets in the wild. Frozen here so the reader is pinned
    /// against a file libviprs did not write.
    const VIPS_8184_TRAILER: &str = r#"<?xml version="1.0"?>
<root xmlns="http://www.vips.ecs.soton.ac.uk/vips/8.18.4">
  <header>
    <field type="VipsRefString" name="Hist"></field>
  </header>
  <meta>
    <field type="VipsBlob" name="exif-data">RXhpZgAASUkqAAgAAAAGABIBAwABAAAAAQAAABoBBQABAAAAVgAAABsBBQABAAAAXgAAACgBAwABAAAAAgAAABMCAwABAAAAAQAAAGmHBAABAAAAZgAAAAAAAAA4YwAA6AMAADhjAADoAwAABgAAkAcABAAAADAyMTABkQcABAAAAAECAwAAoAcABAAAADAxMDABoAMAAQAAAP//AAACoAQAAQAAAAIAAAADoAQAAQAAAAIAAAAAAAAA</field>
    <field type="VipsRefString" name="resolution-unit">in</field>
    <field type="VipsRefString" name="exif-ifd0-Orientation">1 (Top-left, Short, 1 components, 2 bytes)</field>
    <field type="VipsRefString" name="exif-ifd0-XResolution">25400/1000 (25.400, Rational, 1 components, 8 bytes)</field>
    <field type="VipsRefString" name="exif-ifd0-YResolution">25400/1000 (25.400, Rational, 1 components, 8 bytes)</field>
    <field type="VipsRefString" name="exif-ifd0-ResolutionUnit">2 (Inch, Short, 1 components, 2 bytes)</field>
    <field type="VipsRefString" name="exif-ifd0-YCbCrPositioning">1 (Centred, Short, 1 components, 2 bytes)</field>
    <field type="VipsRefString" name="exif-ifd2-ExifVersion">Exif Version 2.1 (Exif Version 2.1, Undefined, 4 components, 4 bytes)</field>
    <field type="VipsRefString" name="exif-ifd2-ComponentsConfiguration">Y Cb Cr - (Y Cb Cr -, Undefined, 4 components, 4 bytes)</field>
    <field type="VipsRefString" name="exif-ifd2-FlashpixVersion">FlashPix Version 1.0 (FlashPix Version 1.0, Undefined, 4 components, 4 bytes)</field>
    <field type="VipsRefString" name="exif-ifd2-ColorSpace">65535 (Uncalibrated, Short, 1 components, 2 bytes)</field>
    <field type="VipsRefString" name="exif-ifd2-PixelXDimension">2 (2, Long, 1 components, 4 bytes)</field>
    <field type="VipsRefString" name="exif-ifd2-PixelYDimension">2 (2, Long, 1 components, 4 bytes)</field>
    <field type="gint" name="orientation">1</field>
  </meta>
</root>
"#;

    /// A `.v` file as a newer libviprs would write it: a real header and real
    /// pixels, followed by a trailer carrying one field of every variant this
    /// build knows plus a `background` array it does not.
    fn file_from_a_newer_build() -> Vec<u8> {
        let mut bytes = v_body();
        let trailer = FutureTrailer {
            orientation: 6,
            fields: FutureFields {
                entries: vec![
                    ("note".to_string(), FutureMetadataValue::Str("hello".into())),
                    ("n-pages".to_string(), FutureMetadataValue::Int(3)),
                    (
                        "delay".to_string(),
                        FutureMetadataValue::IntArray(vec![40, 40, 90]),
                    ),
                    (
                        "background".to_string(),
                        FutureMetadataValue::DoubleArray(vec![1.5, 2.5]),
                    ),
                    ("xres-hint".to_string(), FutureMetadataValue::Double(1.5)),
                    (
                        "icc-profile-data".to_string(),
                        FutureMetadataValue::Blob(vec![5, 5, 5]),
                    ),
                ],
            },
        };
        bytes.extend_from_slice(&serde_json::to_vec(&trailer).unwrap());
        bytes
    }

    /// A `.v` written by a build that has a `MetadataValue` variant this one
    /// does not must still hand back every field this build *does* understand
    /// (issue #565). The old behaviour was that one unknown variant failed the
    /// whole `serde_json::from_slice`, so the ICC profile, the EXIF blob and
    /// the orientation tag all vanished with no error at all.
    ///
    /// The unknown field itself reads as absent: it is carried, not
    /// interpretable, so it stays out of the field API. The `delay` array
    /// beside it is the positive control, because #787 gave that one a
    /// variant: the two travel in the same trailer, in the same encoding
    /// family, and the reader has to hand one back and carry the other.
    #[test]
    fn v_trailer_from_a_newer_build_keeps_the_fields_this_build_understands() {
        let back = decode_bytes(&file_from_a_newer_build()).unwrap();

        assert_eq!(back.orientation(), 6, "orientation must survive");
        assert_eq!(back.get_field("note").unwrap().as_str(), "hello");
        assert_eq!(back.get_field("n-pages"), Some(MetadataValue::Int(3)));
        assert_eq!(back.get_n_pages(), 3);
        assert_eq!(
            back.get_field("xres-hint"),
            Some(MetadataValue::Double(1.5))
        );
        assert_eq!(back.icc_profile(), Some(&[5u8, 5, 5][..]));
        // Pixels and geometry are untouched by any of this.
        assert_eq!(back.data(), &[7u8; 12]);

        // The array variant that landed reads back as a value (issue #787).
        assert_eq!(
            back.get_field("delay"),
            Some(MetadataValue::IntArray(vec![40, 40, 90]))
        );
        assert_eq!(back.get_int_array("delay"), Some(&[40i64, 40, 90][..]));
        assert_eq!(back.get_typeof("delay"), 5);
        assert!(back.get_fields().iter().any(|n| n == "delay"));

        // The variant this build cannot represent reads as absent rather than
        // as a wrong value.
        assert_eq!(back.get_field("background"), None);
        assert_eq!(back.get_typeof("background"), 0);
        assert!(
            !back.get_fields().iter().any(|n| n == "background"),
            "an uninterpretable field must not be advertised as readable"
        );
    }

    /// Preserving an unknown field only matters if it survives being written
    /// back out: an old build that opens a new file and re-saves it must not
    /// quietly strip the field it could not read, which would be the same
    /// data loss one step later.
    #[test]
    fn v_trailer_unknown_field_survives_a_rewrite_by_this_build() {
        let back = decode_bytes(&file_from_a_newer_build()).unwrap();
        let rewritten = back.encode_vips().unwrap();

        // Read the rewritten file with the *newer* build's reader.
        let trailer: FutureTrailer =
            serde_json::from_slice(&rewritten[v_body().len()..]).expect("newer build can re-read");
        assert_eq!(trailer.orientation, 6);
        let background = trailer
            .fields
            .entries
            .iter()
            .find(|(n, _)| n == "background")
            .map(|(_, v)| v);
        assert_eq!(
            background,
            Some(&FutureMetadataValue::DoubleArray(vec![1.5, 2.5])),
            "the unknown field must round-trip untouched"
        );
        // And the fields this build does understand are still there too,
        // including the array it now reads rather than carries (#787).
        assert_eq!(trailer.fields.entries.len(), 6);
        assert!(trailer.fields.entries.iter().any(
            |(n, v)| n == "icc-profile-data" && *v == FutureMetadataValue::Blob(vec![5, 5, 5])
        ));
        assert!(
            trailer
                .fields
                .entries
                .iter()
                .any(|(n, v)| n == "delay" && *v == FutureMetadataValue::IntArray(vec![40, 40, 90])),
            "the named array must survive the rewrite as a value, not as a carrier"
        );
    }

    /// Issue #718. The multi-input field union covers the **uninterpretable**
    /// carrier as well as the named one, in both directions.
    ///
    /// The named half is pinned in `tests/metadata_carry.rs`. This half is here
    /// because [`MetadataFields::unknown`] is only reachable from inside this
    /// module, through a trailer written by a build this one cannot fully read
    /// (#565), and without it `merge_under` could drop the opaque carrier and
    /// stay green: a field a newer build wrote would survive a load and a
    /// re-save, and then vanish the moment someone inserted the raster into
    /// another one.
    #[test]
    fn the_multi_input_field_merge_carries_an_uninterpretable_field_too() {
        let sub = decode_bytes(&file_from_a_newer_build()).unwrap();
        let mut main = Raster::new(2, 2, PixelFormat::Rgb8, vec![1u8; 12]).unwrap();
        main.set_field("main-only", MetadataValue::Str("from-main".into()));
        main.set_field("note", MetadataValue::Str("from-main".into()));

        let out = main.try_insert(&sub, 0, 0, true, None).unwrap();

        // The header block is `main`'s alone, so `sub`'s orientation 6 does
        // not reach the output.
        assert_eq!(out.orientation(), 1, "the header block is main's");
        // The named half of the union: main wins the shared name, both sides'
        // own fields arrive.
        assert_eq!(out.get_field("note").unwrap().as_str(), "from-main");
        assert_eq!(out.get_field("main-only").unwrap().as_str(), "from-main");
        assert_eq!(out.get_field("n-pages"), Some(MetadataValue::Int(3)));
        assert_eq!(out.icc_profile(), Some(&[5u8, 5, 5][..]));

        // And the opaque one, which reads as absent through the field API and
        // is only visible on the way back out.
        assert_eq!(
            out.get_field("background"),
            None,
            "an uninterpretable field stays out of the field API"
        );
        let rewritten = out.encode_vips().unwrap();
        let trailer: FutureTrailer = serde_json::from_slice(&rewritten[v_body().len()..])
            .expect("a carried opaque value keeps the file on the JSON trailer");
        let background = trailer
            .fields
            .entries
            .iter()
            .find(|(n, _)| n == "background")
            .map(|(_, v)| v);
        assert_eq!(
            background,
            Some(&FutureMetadataValue::DoubleArray(vec![1.5, 2.5])),
            "sub's uninterpretable field must reach the output"
        );

        // The other direction: a name `main` holds *only* under the opaque
        // carrier still blocks `sub`'s interpretable value, so the output
        // never ends up holding one name under both carriers.
        let main = decode_bytes(&file_from_a_newer_build()).unwrap();
        let mut sub = Raster::new(2, 2, PixelFormat::Rgb8, vec![1u8; 12]).unwrap();
        sub.set_field("background", MetadataValue::Int(9));
        let out = main.try_insert(&sub, 0, 0, true, None).unwrap();
        assert_eq!(
            out.get_field("background"),
            None,
            "main's opaque value wins the shared name"
        );
    }

    /// A field the caller sets by hand supersedes an unknown field of the same
    /// name, and removing it removes it for good. Otherwise a stripped raster
    /// would leak the old opaque value back into the file.
    ///
    /// Naming the value is also what releases the file from the legacy JSON
    /// trailer: with nothing left that only JSON can hold, the rewrite comes
    /// back out as the XML vips reads.
    #[test]
    fn setting_or_removing_a_field_supersedes_the_unknown_one() {
        let mut back = decode_bytes(&file_from_a_newer_build()).unwrap();
        back.set_field("background", MetadataValue::Int(4));
        let fields = trailer_fields(&back.encode_vips().unwrap());
        let hits: Vec<_> = fields
            .iter()
            .filter(|(n, _, _)| n == "background")
            .collect();
        assert_eq!(hits.len(), 1, "the field must not be written twice");
        assert_eq!(hits[0].1, GTYPE_INT);
        assert_eq!(hits[0].2, "4");
        assert_eq!(
            fields.len(),
            7,
            "overwriting one field must not disturb the other five, or the \
             orientation tag: {fields:?}"
        );

        let mut back = decode_bytes(&file_from_a_newer_build()).unwrap();
        back.set_typeof("background", 0);
        let fields = trailer_fields(&back.encode_vips().unwrap());
        assert!(
            !fields.iter().any(|(n, _, _)| n == "background"),
            "a removed field must not come back from the opaque carrier"
        );
        assert_eq!(
            fields.len(),
            6,
            "removing one field must not remove the other five: {fields:?}"
        );
    }

    /// The other direction: what this build writes still has to reach a build
    /// that shipped before this change. It cannot reach it whole, and that is
    /// the price of the fix rather than an accident, so it is pinned here.
    ///
    /// The released reader is reproduced verbatim as [`released_reader`], so
    /// this runs it rather than asserting about it. It takes the pixels, the
    /// geometry and the orientation off a `.v` written now, and not the
    /// attached fields, because it only reads a trailer whose first
    /// non-whitespace byte is `{` and no byte sequence is both that and the
    /// XML libvips requires. Nothing errors, which is the part that matters:
    /// an old build opening a new file is not a failure, it is a partial read.
    #[test]
    fn v_trailer_written_now_reads_as_far_as_the_released_build_can_read_it() {
        let mut im = Raster::new(2, 2, PixelFormat::Rgb8, vec![7u8; 12])
            .unwrap()
            .copy()
            .orientation(6)
            .build();
        im.set_field("note", "hello".into());
        im.set_field("n-pages", MetadataValue::Int(3));
        im.set_field("xres-hint", MetadataValue::Double(1.5));
        im.set_icc_profile(&[5, 5, 5]);

        let bytes = im.encode_vips().unwrap();
        let (orientation, fields) = released_reader(&bytes[v_body().len()..]);
        assert_eq!(
            orientation,
            Some(6),
            "the released build must still get the orientation tag"
        );
        assert!(
            fields.is_empty(),
            "the released build reads no attached fields out of an XML trailer, \
             and pretending otherwise would hide the cost: {fields:?}"
        );

        // The same file read here is whole, so the loss runs one way only.
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.orientation(), 6);
        assert_eq!(back.get_field("note").unwrap().as_str(), "hello");
        assert_eq!(back.get_field("n-pages"), Some(MetadataValue::Int(3)));
        assert_eq!(
            back.get_field("xres-hint"),
            Some(MetadataValue::Double(1.5))
        );
        assert_eq!(back.icc_profile(), Some(&[5u8, 5, 5][..]));
    }

    /// And a file the released build wrote still reads here, checked against
    /// the frozen bytes rather than against a fresh encode.
    #[test]
    fn v_trailer_written_by_the_released_build_still_reads() {
        let mut bytes = v_body();
        bytes.extend_from_slice(RELEASED_TRAILER);
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.orientation(), 6);
        assert_eq!(back.get_field("note").unwrap().as_str(), "hello");
        assert_eq!(back.get_field("n-pages"), Some(MetadataValue::Int(3)));
        assert_eq!(
            back.get_field("xres-hint"),
            Some(MetadataValue::Double(1.5))
        );
        assert_eq!(back.icc_profile(), Some(&[5u8, 5, 5][..]));
    }

    /// The reader must survive more than a new variant: a newer trailer may
    /// carry keys this build has never seen, drop keys it expects, or hold an
    /// entry shaped in some way it cannot use. None of that may cost the
    /// entries that *are* readable.
    #[test]
    fn v_trailer_tolerates_shapes_a_newer_build_might_write() {
        let mut bytes = v_body();
        bytes.extend_from_slice(
            br#"{"fields":{"entries":[["note",{"Str":"hi"}],["broken"],[7,{"Int":1}],
                 ["n-pages",{"Int":2}]],"grouping":"page"},"trailer-version":9}"#,
        );
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.get_field("note").unwrap().as_str(), "hi");
        assert_eq!(back.get_field("n-pages"), Some(MetadataValue::Int(2)));
        // No orientation key: the upright default, not a dropped trailer.
        assert_eq!(back.orientation(), 1);
    }

    /// A trailer that opens with `{` claims to be a libviprs JSON trailer, so
    /// one that is not valid JSON is corruption rather than a foreign format,
    /// and the metadata is genuinely unrecoverable. That is reported instead
    /// of swallowed. Nothing else changes: a trailer that never claimed to be
    /// ours is still ignored, pixels and header intact.
    #[test]
    fn v_trailer_that_claims_to_be_ours_but_is_corrupt_is_reported() {
        let mut bytes = v_body();
        bytes.extend_from_slice(br#"{"orientation":6,"fields":{"entr"#);
        let err = decode_bytes(&bytes).unwrap_err();
        assert!(
            matches!(&err, SourceError::VipsFormat(m) if m.contains("trailer")),
            "expected a reported trailer failure, got {err}"
        );

        // A foreign or junk trailer is not ours to complain about.
        for foreign in [
            &b"<?xml version=\"1.0\"?><root/>"[..],
            &[0u8; 8][..],
            b"junk",
        ] {
            let mut bytes = v_body();
            bytes.extend_from_slice(foreign);
            let back = decode_bytes(&bytes).unwrap();
            assert_eq!(back.data(), &[7u8; 12]);
            assert_eq!(back.get_field("note"), None);
        }
    }

    // -- .v trailer: the XML libvips reads (issue #546) -----------------------

    /// The trailer written for a raster carrying one field of every variant,
    /// frozen byte for byte.
    ///
    /// This is the wire format, so it is pinned against a literal rather than
    /// against the reader, which would agree with the writer no matter what
    /// either of them did. The shape is `build_xml`'s
    /// (`libvips/iofuncs/vips.c:846-890` at `v8.18.0-95-gfe420cf3a`) down to
    /// the four-space field indent, and the fields come out in the order they
    /// were set with the orientation tag last.
    #[test]
    fn v_trailer_is_the_xml_libvips_reads() {
        let mut im = Raster::new(2, 2, PixelFormat::Rgb8, vec![7u8; 12])
            .unwrap()
            .copy()
            .orientation(6)
            .build();
        im.set_field("note", "hello".into());
        im.set_field("n-pages", MetadataValue::Int(3));
        im.set_field("xres-hint", MetadataValue::Double(1.5));
        im.set_icc_profile(&[5, 5, 5]);

        let bytes = im.encode_vips().unwrap();
        let trailer = std::str::from_utf8(&bytes[v_body().len()..]).unwrap();
        assert_eq!(
            trailer,
            "<?xml version=\"1.0\"?>\n\
             <root xmlns=\"http://www.vips.ecs.soton.ac.uk/vips/8.18.4\">\n\
             \x20 <header>\n\
             \x20   <field type=\"VipsRefString\" name=\"Hist\"></field>\n\
             \x20 </header>\n\
             \x20 <meta>\n\
             \x20   <field type=\"VipsRefString\" name=\"note\">hello</field>\n\
             \x20   <field type=\"gint\" name=\"n-pages\">3</field>\n\
             \x20   <field type=\"gdouble\" name=\"xres-hint\">1.5</field>\n\
             \x20   <field type=\"VipsBlob\" name=\"icc-profile-data\">BQUF</field>\n\
             \x20   <field type=\"gint\" name=\"orientation\">6</field>\n\
             \x20 </meta>\n\
             </root>\n"
        );
    }

    /// Every [`MetadataValue`] variant survives the XML round trip, including
    /// the awkward numbers: a negative and a 64-bit integer, a double that
    /// has no short decimal, one that needs an exponent, a negative zero, a
    /// blob holding all 256 byte values, and an int array with a negative
    /// element, a 64-bit one and no elements at all.
    ///
    /// The doubles are compared exactly on purpose. A tolerance would pass on
    /// a writer that threw away digits, and throwing away digits is the whole
    /// failure mode a text encoding of a float has.
    ///
    /// The variant list here is hand-maintained, and it is the one place that
    /// matters. `xml_field_of`, `type_code` and `len` all match on the enum,
    /// so the compiler makes a sixth variant impossible to forget in those
    /// three; the **reader** has a `_ => None` fallthrough and would carry a
    /// new variant opaquely for ever without a word. This test is what
    /// notices.
    #[test]
    fn v_trailer_xml_round_trips_every_metadata_variant() {
        let mut im = rgb_2x2().copy().orientation(8).build();
        im.set_field("neg", MetadataValue::Int(-42));
        im.set_field("big", MetadataValue::Int(i64::from(i32::MAX) + 1));
        im.set_field("tenth", MetadataValue::Double(0.1));
        im.set_field("huge", MetadataValue::Double(1e300));
        im.set_field("negzero", MetadataValue::Double(-0.0));
        im.set_field("text", MetadataValue::Str("café ☃ 日本".to_string()));
        im.set_field(
            "bytes",
            MetadataValue::Blob((0..=255u8).collect::<Vec<_>>()),
        );
        im.set_field("delay", MetadataValue::IntArray(vec![40, -5, i64::MAX]));
        im.set_field("no-delay", MetadataValue::IntArray(Vec::new()));

        let back = decode_bytes(&im.encode_vips().unwrap()).unwrap();
        assert_eq!(back.orientation(), 8);
        assert_eq!(back.get_field("neg"), Some(MetadataValue::Int(-42)));
        assert_eq!(
            back.get_field("big"),
            Some(MetadataValue::Int(i64::from(i32::MAX) + 1))
        );
        assert_eq!(back.get_field("tenth"), Some(MetadataValue::Double(0.1)));
        assert_eq!(back.get_field("huge"), Some(MetadataValue::Double(1e300)));
        assert!(
            back.get_field("negzero")
                .is_some_and(|v| v.as_f64().is_sign_negative() && v.as_f64() == 0.0),
            "a negative zero must not come back positive"
        );
        assert_eq!(back.get_field("text").unwrap().as_str(), "café ☃ 日本");
        assert_eq!(
            back.get_field("bytes").unwrap().as_blob(),
            (0..=255u8).collect::<Vec<_>>()
        );
        assert_eq!(
            back.get_field("delay"),
            Some(MetadataValue::IntArray(vec![40, -5, i64::MAX]))
        );
        assert_eq!(
            back.get_field("no-delay"),
            Some(MetadataValue::IntArray(Vec::new())),
            "an empty array is a value, not a dropped field"
        );
        // The blob went out as base64 rather than as anything binary.
        let bytes = im.encode_vips().unwrap();
        let trailer = std::str::from_utf8(&bytes[v_body().len()..]).unwrap();
        assert!(
            trailer.contains("<field type=\"VipsBlob\" name=\"bytes\">AAECAwQFBgcICQ"),
            "got: {trailer}"
        );
    }

    /// A `.v` real vips wrote reads whole now, not just down to its
    /// orientation tag. [`VIPS_8184_TRAILER`] is the capture; the body under
    /// it is this build's, because the body is not what is being tested.
    ///
    /// The `Hist` field in the `<header>` block must *not* arrive as an
    /// attached field: it is vips's command history, it lives outside
    /// `<meta>`, and letting it through would invent a field on every vips
    /// file libviprs opens.
    #[test]
    fn v_trailer_reads_a_file_real_vips_wrote() {
        let body = Raster::zeroed(2, 2, PixelFormat::Gray8).unwrap();
        let mut bytes = body.encode_vips().unwrap();
        assert_eq!(
            bytes.len(),
            VIPS_HEADER_LEN + 4,
            "the body carries no trailer"
        );
        bytes.extend_from_slice(VIPS_8184_TRAILER.as_bytes());

        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.orientation(), 1);
        assert_eq!(back.get_field("resolution-unit").unwrap().as_str(), "in");
        assert_eq!(
            back.get_field("exif-ifd0-XResolution").unwrap().as_str(),
            "25400/1000 (25.400, Rational, 1 components, 8 bytes)"
        );
        let exif = back.get_field("exif-data").unwrap();
        assert_eq!(exif.len(), 186, "the base64 blob must decode to its bytes");
        assert_eq!(&exif.as_blob()[..6], b"Exif\0\0");
        assert_eq!(
            back.get_typeof("Hist"),
            0,
            "the history block is not metadata"
        );
        assert_eq!(
            back.get_fields().len() - BUILTIN_FIELDS.len(),
            13,
            "every <meta> field but the orientation tag, and nothing else: {:?}",
            back.get_fields()
        );

        // And it survives a rewrite here: this is the round trip the `.v`
        // container exists for.
        let again = decode_bytes(&back.encode_vips().unwrap()).unwrap();
        assert_eq!(again.get_field("exif-data"), back.get_field("exif-data"));
        assert_eq!(again.get_fields(), back.get_fields());
    }

    /// XML metacharacters in a field name and in a value survive, and
    /// non-ASCII text is written as itself rather than mangled.
    ///
    /// vips gets the second one wrong in its own writer: `*p < 32` on a
    /// signed `char` (`libvips/iofuncs/target.c:821`) catches every
    /// continuation byte, so `vips copy` over a `.v` carrying `café ☃ 日本`
    /// rewrites it as `caf&#x23c3;&#x23a9; …` (measured on 8.18.4). Reading
    /// it is fine, which is why writing real UTF-8 is the right call.
    #[test]
    fn v_trailer_xml_escaping_round_trips() {
        let mut im = rgb_2x2();
        im.set_field("a\"b&c<d>e", "x < y & z > w \"q\" 'r'".into());
        im.set_field("tabbed", "one\ttwo\nthree".into());
        im.set_field("unicode", "café ☃ 日本".into());

        let bytes = im.encode_vips().unwrap();
        let trailer = std::str::from_utf8(&bytes[v_body().len()..]).unwrap();
        assert!(
            trailer.contains("name=\"a&quot;b&amp;c&lt;d&gt;e\""),
            "the name must be escaped for an attribute: {trailer}"
        );
        assert!(
            trailer.contains(">x &lt; y &amp; z &gt; w \"q\" 'r'<"),
            "character data escapes the three that matter and nothing else: {trailer}"
        );
        assert!(
            trailer.contains(">café ☃ 日本<"),
            "UTF-8 goes out as itself: {trailer}"
        );

        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(
            back.get_field("a\"b&c<d>e").unwrap().as_str(),
            "x < y & z > w \"q\" 'r'"
        );
        assert_eq!(
            back.get_field("tabbed").unwrap().as_str(),
            "one\ttwo\nthree"
        );
        assert_eq!(back.get_field("unicode").unwrap().as_str(), "café ☃ 日本");
    }

    /// The XML form of a carried unknown value: a `type` this build does not
    /// know keeps its type name and its character data, byte for byte, and
    /// goes back out that way (issue #565).
    ///
    /// This is where the XML trailer beats the JSON one it replaces. The
    /// carrier *is* vips's own encoding, so the value is not merely preserved
    /// for the build that wrote it: vips reads it too. Measured on 8.18.6,
    /// `vipsheader -f background` on a file with this exact element prints
    /// `1.5 2.5`, and a `type` name libvips does not know is skipped
    /// with no warning at all, which is what makes the carrier safe to write.
    ///
    /// `delay` sits in the same trailer as the positive control: #787 gave
    /// `VipsArrayInt` a variant, so that one has to come back as a *value*
    /// while the other two are still carried. Without it the test would pass
    /// on a reader that carried everything, which is what it did before.
    #[test]
    fn v_trailer_unknown_xml_type_is_carried_verbatim() {
        let body = rgb_2x2();
        let mut bytes = body.encode_vips_impl(false);
        bytes.extend_from_slice(
            b"<?xml version=\"1.0\"?>\n\
              <root xmlns=\"http://www.vips.ecs.soton.ac.uk/vips/8.18.4\">\n  <meta>\n\
              \x20   <field type=\"VipsRefString\" name=\"note\">hi</field>\n\
              \x20   <field type=\"VipsArrayInt\" name=\"delay\">40 40 90</field>\n\
              \x20   <field type=\"VipsArrayDouble\" name=\"background\">1.5 2.5 </field>\n\
              \x20   <field type=\"nosuchtype\" name=\"mystery\">a &amp; b</field>\n\
              \x20   <field type=\"gint\" name=\"orientation\">6</field>\n\
              \x20 </meta>\n</root>\n",
        );

        let back = decode_bytes(&bytes).unwrap();
        // Readable things stay readable.
        assert_eq!(back.get_field("note").unwrap().as_str(), "hi");
        assert_eq!(back.orientation(), 6);
        // The array this build now names comes back as a value.
        assert_eq!(back.get_int_array("delay"), Some(&[40i64, 40, 90][..]));
        // The two it still cannot name read as absent rather than as a
        // wrong value.
        for name in ["background", "mystery"] {
            assert_eq!(back.get_field(name), None);
            assert_eq!(back.get_typeof(name), 0);
            assert!(!back.get_fields().iter().any(|n| n == name));
        }

        // The carried ones go back out unchanged, escapes included; the named
        // one goes back out in this writer's spelling, which is vips's own,
        // trailing separator and all.
        let rewritten = back.encode_vips().unwrap();
        let trailer = std::str::from_utf8(&rewritten[v_body().len()..]).unwrap();
        assert!(
            trailer.contains("<field type=\"VipsArrayInt\" name=\"delay\">40 40 90 </field>"),
            "got: {trailer}"
        );
        assert!(
            trailer
                .contains("<field type=\"VipsArrayDouble\" name=\"background\">1.5 2.5 </field>"),
            "got: {trailer}"
        );
        assert!(
            trailer.contains("<field type=\"nosuchtype\" name=\"mystery\">a &amp; b</field>"),
            "the character data must not be re-escaped into `a &amp;amp; b`: {trailer}"
        );
        // Still XML, so vips still reads the rest of the file.
        assert!(!is_json_trailer(&rewritten[v_body().len()..]));
    }

    // -- the VipsArrayInt variant (issue #787) -------------------------------

    /// A 2x2 RGB `.v` carrying `meta` as its whole `<meta>` block, so a test
    /// can hand the reader an arbitrary field element.
    fn v_with_meta(meta: &str) -> Vec<u8> {
        let mut bytes = rgb_2x2().encode_vips_impl(false);
        bytes.extend_from_slice(
            format!(
                "<?xml version=\"1.0\"?>\n<root xmlns=\"{VIPS_XML_NAMESPACE}\">\n  <meta>\n\
                 {meta}\n  </meta>\n</root>\n"
            )
            .as_bytes(),
        );
        bytes
    }

    /// The `<field>` element the writer produces for an int array is the one
    /// vips produces, **trailing separator included**.
    ///
    /// Measured on the pinned vips 8.18.6:
    ///
    /// ```text
    /// vips copy 'oracle-captures/foreign-webp/fixtures/anim3.webp[n=-1]' out.v
    /// ```
    ///
    /// puts `<field type="VipsArrayInt" name="delay">100 100 100 </field>` in
    /// the trailer: one space after every element, not between them. Getting
    /// that wrong is invisible to any round trip through this crate's own
    /// reader, which is exactly why it is pinned as bytes here.
    #[test]
    fn an_int_array_goes_out_in_the_spelling_vips_writes() {
        let mut im = rgb_2x2();
        im.set_field("delay", MetadataValue::IntArray(vec![40, 60, 80, 100]));
        let bytes = im.encode_vips().unwrap();
        let trailer = std::str::from_utf8(&bytes[v_body().len()..]).unwrap();
        assert!(
            trailer.contains("<field type=\"VipsArrayInt\" name=\"delay\">40 60 80 100 </field>"),
            "got: {trailer}"
        );

        // An empty array is a legal value with no elements, and vips writes
        // exactly that: measured, a trailer holding `<field ...></field>`
        // round-trips through `vips copy` unchanged.
        let mut im = rgb_2x2();
        im.set_field("delay", MetadataValue::IntArray(Vec::new()));
        let bytes = im.encode_vips().unwrap();
        let trailer = std::str::from_utf8(&bytes[v_body().len()..]).unwrap();
        assert!(
            trailer.contains("<field type=\"VipsArrayInt\" name=\"delay\"></field>"),
            "got: {trailer}"
        );

        // A single element still gets its separator, which is the case a
        // "join with spaces" implementation gets wrong and a four-element
        // check cannot see. `vipsheader -f delay` on a one-frame array prints
        // `40 ` on 8.18.6.
        let mut im = rgb_2x2();
        im.set_field("delay", MetadataValue::IntArray(vec![40]));
        let bytes = im.encode_vips().unwrap();
        let trailer = std::str::from_utf8(&bytes[v_body().len()..]).unwrap();
        assert!(
            trailer.contains("<field type=\"VipsArrayInt\" name=\"delay\">40 </field>"),
            "got: {trailer}"
        );
    }

    /// An int array survives `encode_vips` and `decode_bytes` as a value,
    /// reads back through both accessors, and reports type code 5.
    ///
    /// The negative and out-of-`gint` elements are the reason the element type
    /// is `i64`. vips's own `gint` is 32 bits and wraps rather than refusing:
    /// measured on 8.18.6, a trailer carrying `3000000000` reads back through
    /// `vipsheader -f delay` as `-1294967296`, and
    /// `9223372036854775807 -9223372036854775808` as `-1 0`. A `u32` element
    /// would make libviprs lose the sign as well, on a file it did not write.
    #[test]
    fn an_int_array_round_trips_through_the_v_trailer() {
        let mut im = rgb_2x2();
        let delays = vec![40i64, -5, 3_000_000_000, i64::MIN, i64::MAX];
        im.set_field("delay", MetadataValue::IntArray(delays.clone()));
        let back = decode_bytes(&im.encode_vips().unwrap()).unwrap();

        assert_eq!(
            back.get_field("delay"),
            Some(MetadataValue::IntArray(delays.clone()))
        );
        assert_eq!(back.get_int_array("delay"), Some(delays.as_slice()));
        assert_eq!(back.get_typeof("delay"), 5);
        assert!(back.get_fields().iter().any(|n| n == "delay"));
        // The trailer is the XML vips reads, not the legacy JSON fallback:
        // the value has a spelling now, so nothing forces the old format.
        assert!(!is_json_trailer(
            &im.encode_vips().unwrap()[v_body().len()..]
        ));
    }

    /// The reader takes every spelling vips takes.
    ///
    /// Measured on 8.18.6 by editing the `delay` element of a real `.v` and
    /// reading it back with `vipsheader -f delay`: `40 60 80`, `40 60 80 ` and
    /// `  40   60   80  ` all print `40 60 80 `, and an element list that is
    /// empty prints nothing rather than dropping the field. Writing one
    /// spelling and accepting only that one would refuse files vips wrote.
    #[test]
    fn the_int_array_reader_takes_every_spelling_vips_takes() {
        for text in ["40 60 80", "40 60 80 ", "  40   60   80  ", "40\t60\n80"] {
            let bytes = v_with_meta(&format!(
                "    <field type=\"VipsArrayInt\" name=\"delay\">{text}</field>"
            ));
            let back = decode_bytes(&bytes).unwrap();
            assert_eq!(
                back.get_int_array("delay"),
                Some(&[40i64, 60, 80][..]),
                "spelling {text:?} must read as three elements"
            );
        }

        let bytes = v_with_meta("    <field type=\"VipsArrayInt\" name=\"delay\"></field>");
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(
            back.get_int_array("delay"),
            Some(&[][..]),
            "an empty element list is an empty array, not a missing field"
        );
        assert_eq!(back.get_typeof("delay"), 5);
    }

    /// An element that is not a number keeps the **whole** field opaque, the
    /// same rule every other GType in the trailer follows when its text will
    /// not parse.
    ///
    /// This is a deliberate divergence and it is measured. vips hands back an
    /// *empty* array for `40 x 80`: on 8.18.6 `vipsheader -f delay` prints
    /// nothing and `vips copy` writes the field back out as
    /// `<field type="VipsArrayInt" name="delay"></field>`, so the two
    /// elements that did parse are gone from the file. Carrying the text
    /// through loses nothing and lets a build that understands it read it.
    ///
    /// The positive control is the same trailer with the element fixed: it
    /// has to come back as a value, or "carried" would just mean "the reader
    /// never worked".
    #[test]
    fn an_int_array_element_that_is_not_a_number_keeps_the_whole_field_opaque() {
        for text in ["40 x 80", "40 60 80.5", "40 99999999999999999999", "40 -"] {
            let bytes = v_with_meta(&format!(
                "    <field type=\"VipsArrayInt\" name=\"delay\">{text}</field>"
            ));
            let back = decode_bytes(&bytes).unwrap();
            assert_eq!(
                back.get_field("delay"),
                None,
                "{text:?} must not be read as a partial array"
            );
            assert_eq!(back.get_typeof("delay"), 0);
            // And it goes back out byte for byte, so a build that can read it
            // still can.
            let rewritten = back.encode_vips().unwrap();
            let trailer = std::str::from_utf8(&rewritten[v_body().len()..]).unwrap();
            assert!(
                trailer.contains(&format!("name=\"delay\">{text}</field>")),
                "got: {trailer}"
            );
        }

        // Positive control: fix the element and the same route reads a value.
        let bytes = v_with_meta("    <field type=\"VipsArrayInt\" name=\"delay\">40 60 80</field>");
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.get_int_array("delay"), Some(&[40i64, 60, 80][..]));
    }

    /// The array variant answers [`MetadataValue::type_code`],
    /// [`MetadataValue::len`] and the panic-message kind for itself, rather
    /// than borrowing a scalar's answers.
    #[test]
    fn the_array_variant_reports_its_own_type_code_and_element_count() {
        let three = MetadataValue::IntArray(vec![40, 60, 80]);
        assert_eq!(three.type_code(), 5, "a fifth type needs a fifth code");
        assert_eq!(three.len(), 3, "len is the element count, not 1");
        assert!(!three.is_empty());
        assert_eq!(three.as_int_array(), &[40, 60, 80]);

        let empty = MetadataValue::IntArray(Vec::new());
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty(), "an empty array is empty, unlike a scalar");
        assert_eq!(empty.type_code(), 5);

        // No code collides: five variants, five codes.
        let codes: Vec<u64> = [
            MetadataValue::Int(1),
            MetadataValue::Double(1.0),
            MetadataValue::Str(String::new()),
            MetadataValue::Blob(Vec::new()),
            MetadataValue::IntArray(Vec::new()),
        ]
        .iter()
        .map(MetadataValue::type_code)
        .collect();
        assert_eq!(codes, vec![1, 2, 3, 4, 5]);

        // The kind name reaches the panic messages and the WrongType error.
        assert_eq!(
            std::panic::catch_unwind(|| MetadataValue::IntArray(vec![1]).as_i64())
                .unwrap_err()
                .downcast_ref::<String>()
                .cloned(),
            Some("metadata value is an int array, not an int".to_string())
        );
        let err = rgb_2x2()
            .try_set_field("xoffset", MetadataValue::IntArray(vec![1]))
            .unwrap_err();
        assert!(
            err.to_string().contains("an int array"),
            "the error has to name the kind it got: {err}"
        );

        // And a scalar is not a one-element array, in either direction.
        assert!(
            std::panic::catch_unwind(|| MetadataValue::Int(40).as_int_array()).is_err(),
            "an Int must not coerce to a one-element array"
        );
    }

    /// `get_int_array` answers exactly what resolving through `get_field`
    /// answers, on every readable name, and borrows rather than cloning
    /// (issue #635).
    ///
    /// Same shape as `get_int_agrees_with_get_field_on_every_readable_name`,
    /// and for the same reason: any name can hold any type, so the accessor
    /// that skips the clone has to give the same answer on the built-in
    /// header fields, on `filename` (the one built-in read out of the field
    /// list), on an attached blob sitting under an array's name, and on a
    /// name that is not set at all.
    #[test]
    fn get_int_array_agrees_with_get_field_on_every_readable_name() {
        let mut im = rgb_2x2();
        im.set_field("orientation", MetadataValue::Int(6));
        im.set_field("yres", MetadataValue::Double(1.5));
        im.set_field("filename", MetadataValue::IntArray(vec![11, 12]));
        im.set_field("delay", MetadataValue::IntArray(vec![40, 60, 80]));
        im.set_field("empty-delay", MetadataValue::IntArray(Vec::new()));
        im.set_field("icc-profile-data", MetadataValue::Blob(vec![1, 2, 3]));
        im.set_field("note", MetadataValue::Str("hello".to_string()));
        im.set_field("bits-per-sample", MetadataValue::Int(8));

        let names = im.get_fields();
        assert!(names.len() > 12, "the sweep has to reach the attachments");
        for name in names {
            let through_get_field = match im.get_field(&name) {
                Some(MetadataValue::IntArray(v)) => Some(v),
                _ => None,
            };
            assert_eq!(
                im.get_int_array(&name).map(<[i64]>::to_vec),
                through_get_field,
                "get_int_array and get_field disagree on {name}"
            );
        }
        assert_eq!(im.get_int_array("no-such-field"), None);
        // The #635 case: a blob under an array's name reads as absent here
        // rather than being deep-copied out first.
        assert_eq!(im.get_int_array("icc-profile-data"), None);
        // And a scalar int under the name is not an array.
        assert_eq!(im.get_int_array("bits-per-sample"), None);

        // The borrow really is a borrow: the slice points into the raster.
        let slice = im.get_int_array("delay").unwrap();
        let stored = match im.get_field("delay") {
            Some(MetadataValue::IntArray(v)) => v,
            other => panic!("expected an array, got {other:?}"),
        };
        assert_eq!(slice, stored.as_slice());
    }

    /// A legacy JSON trailer whose only unnameable value was an array is read
    /// as a value now, and the rewrite comes back out as the XML vips reads.
    ///
    /// This is the disk-side payoff of #787 and the thing the JSON fallback
    /// was always meant to do: it is keyed on what is *still* carried, not on
    /// where the file came from, so naming a variant releases every file that
    /// only needed that one. The positive control is the same trailer with a
    /// variant this build still cannot name, which must keep the old format.
    #[test]
    fn a_legacy_json_array_is_read_and_releases_the_file_from_the_json_trailer() {
        let mut bytes = v_body();
        bytes.extend_from_slice(
            br#"{"orientation":6,"fields":{"entries":[["note",{"Str":"hi"}],["delay",{"IntArray":[40,40,90]}]]}}"#,
        );
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.get_int_array("delay"), Some(&[40i64, 40, 90][..]));
        assert_eq!(back.get_field("note").unwrap().as_str(), "hi");

        let rewritten = back.encode_vips().unwrap();
        assert!(
            !is_json_trailer(&rewritten[v_body().len()..]),
            "with the array named, nothing holds the file on the legacy format"
        );
        let trailer = std::str::from_utf8(&rewritten[v_body().len()..]).unwrap();
        assert!(
            trailer.contains("<field type=\"VipsArrayInt\" name=\"delay\">40 40 90 </field>"),
            "got: {trailer}"
        );

        // Positive control: an array this build still cannot name keeps it.
        let mut bytes = v_body();
        bytes.extend_from_slice(
            br#"{"orientation":6,"fields":{"entries":[["background",{"DoubleArray":[1.5]}]]}}"#,
        );
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.get_field("background"), None);
        assert!(
            is_json_trailer(&back.encode_vips().unwrap()[v_body().len()..]),
            "a JSON-only carried value still keeps the JSON trailer"
        );
    }

    /// A value carried out of a *legacy JSON* trailer has no XML spelling, so
    /// the file keeps the JSON trailer rather than losing it.
    ///
    /// Translating it would mean interpreting it: the two formats encode the
    /// same value differently (`{"DoubleArray":[1.5,2.5]}` against
    /// `type="VipsArrayDouble">1.5 2.5 `), and a build that cannot name the
    /// variant cannot convert between the encodings. Dropping it would be the
    /// #565 data loss one step later, so the writer stays on the old format
    /// for exactly the files that need it, and flips to XML the moment
    /// nothing needs it any more.
    #[test]
    fn v_trailer_keeps_the_json_form_for_a_value_with_no_xml_spelling() {
        let carried = decode_bytes(&file_from_a_newer_build()).unwrap();
        let rewritten = carried.encode_vips().unwrap();
        assert!(
            is_json_trailer(&rewritten[v_body().len()..]),
            "a JSON-only carried value keeps the JSON trailer"
        );
        // Which is the only format that can still hold it: everything comes
        // back, twice over, and the released reader gets nothing at all,
        // because an unknown value fails its whole-trailer parse. That is the
        // #565 bug it shipped with, not something this change added.
        let twice =
            decode_bytes(&decode_bytes(&rewritten).unwrap().encode_vips().unwrap()).unwrap();
        assert_eq!(twice.get_field("note").unwrap().as_str(), "hello");
        assert_eq!(twice.icc_profile(), Some(&[5u8, 5, 5][..]));
        assert_eq!(
            released_reader(&rewritten[v_body().len()..]),
            (None, vec![])
        );

        // Drop the value that forced it and the format flips.
        let mut named = decode_bytes(&file_from_a_newer_build()).unwrap();
        named.set_typeof("background", 0);
        let rewritten = named.encode_vips().unwrap();
        assert!(
            !is_json_trailer(&rewritten[v_body().len()..]),
            "with nothing JSON-only left, the trailer must be the XML vips reads"
        );
        assert_eq!(
            decode_bytes(&rewritten).unwrap().icc_profile(),
            Some(&[5u8, 5, 5][..])
        );
    }

    /// The base64 codec used for a `VipsBlob` field: the RFC 4648 section 10
    /// vectors, which is the specification `g_base64_encode` implements, plus
    /// an all-bytes round trip and the rejections that send a malformed field
    /// to the opaque carrier instead of to a wrong value.
    #[test]
    fn base64_matches_the_reference_vectors() {
        for (plain, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_encode(plain.as_bytes()), encoded);
            assert_eq!(
                base64_decode(encoded).as_deref(),
                Some(plain.as_bytes()),
                "decoding {encoded:?}"
            );
        }

        let all: Vec<u8> = (0..=255u8).collect();
        assert_eq!(base64_decode(&base64_encode(&all)), Some(all));
        // Whitespace and a missing pad are tolerated, the way
        // `g_base64_decode` tolerates them; junk is not.
        assert_eq!(base64_decode(" Zm9v \n"), Some(b"foo".to_vec()));
        assert_eq!(base64_decode("Zg"), Some(b"f".to_vec()));
        for bad in [
            "Zm9*",       // not in the alphabet
            "Zm==9v",     // data after the padding
            "Zm9vYg====", // more padding than a group can hold
            "Zh==",       // trailing bits that are not zero
        ] {
            assert_eq!(base64_decode(bad), None, "must reject {bad:?}");
        }
    }

    /// Pins every [`Interpretation`] to its `VipsInterpretation` code and
    /// nickname, so the mapping cannot drift from
    /// `libvips/include/vips/image.h:96-117` (8.18.4) when a variant lands.
    ///
    /// The table here is the `vips_tag` match, which is exhaustive on
    /// purpose: a new variant does not compile until it is handed both a code
    /// and a nickname right there, and every assertion below is driven off
    /// what it returns rather than off a second copy. `VARIANTS` only supplies
    /// the iteration order; the sweep at the end pins its contents from the
    /// other side, so a variant left out of it is caught as soon as the reader
    /// learns the code. The real guarantee that no variant ships untagged is
    /// still that the production `interpretation_code` and
    /// `interpretation_nickname` have no `_` arm.
    #[test]
    fn interpretation_code_table_matches_vips() {
        // The `VipsInterpretation` code and libvips nickname every variant
        // must carry, straight off `image.h`. Exhaustive: adding a variant
        // to `Interpretation` breaks this match until both are filled in.
        fn vips_tag(i: Interpretation) -> (i32, &'static str) {
            match i {
                Interpretation::Multiband => (0, "multiband"),
                Interpretation::Bw => (1, "b-w"),
                Interpretation::Histogram => (10, "histogram"),
                Interpretation::Xyz => (12, "xyz"),
                Interpretation::Lab => (13, "lab"),
                Interpretation::Cmyk => (15, "cmyk"),
                Interpretation::Labq => (16, "labq"),
                Interpretation::Rgb => (17, "rgb"),
                Interpretation::Cmc => (18, "cmc"),
                Interpretation::Lch => (19, "lch"),
                Interpretation::Labs => (21, "labs"),
                Interpretation::Srgb => (22, "srgb"),
                Interpretation::Yxy => (23, "yxy"),
                Interpretation::Fourier => (24, "fourier"),
                Interpretation::Rgb16 => (25, "rgb16"),
                Interpretation::Grey16 => (26, "grey16"),
                Interpretation::Matrix => (27, "matrix"),
                Interpretation::ScRgb => (28, "scrgb"),
                Interpretation::Hsv => (29, "hsv"),
                Interpretation::OkLab => (30, "oklab"),
                Interpretation::OkLch => (31, "oklch"),
            }
        }

        const VARIANTS: [Interpretation; 21] = [
            Interpretation::Multiband,
            Interpretation::Bw,
            Interpretation::Histogram,
            Interpretation::Xyz,
            Interpretation::Lab,
            Interpretation::Cmyk,
            Interpretation::Labq,
            Interpretation::Rgb,
            Interpretation::Cmc,
            Interpretation::Lch,
            Interpretation::Labs,
            Interpretation::Srgb,
            Interpretation::Yxy,
            Interpretation::Fourier,
            Interpretation::Rgb16,
            Interpretation::Grey16,
            Interpretation::Matrix,
            Interpretation::ScRgb,
            Interpretation::Hsv,
            Interpretation::OkLab,
            Interpretation::OkLch,
        ];

        for interp in VARIANTS {
            let (code, nickname) = vips_tag(interp);
            assert_eq!(
                interpretation_code(interp),
                code,
                "{nickname} must write VipsInterpretation {code}"
            );
            assert_eq!(
                interpretation_from_code(code),
                Some(interp),
                "VipsInterpretation {code} must read back as {nickname}"
            );
            assert_eq!(interpretation_nickname(interp), nickname);
            assert_eq!(interpretation_from_nickname(nickname), Some(interp));
        }

        // Nothing in the language binds `VARIANTS`' length to the enum, so
        // pin it against the reader instead: every code the reader accepts
        // has to be one this test drives, and every code it rejects has to
        // be one this test does not. That covers VIPS_INTERPRETATION_ERROR
        // (-1), which vips does emit, the codes libvips leaves unassigned
        // inside its range (2, 11, 14, 20), and everything past
        // VIPS_INTERPRETATION_LAST (32). The read-only legacy aliases 1000 /
        // 1001 sit outside this range on purpose and are pinned by
        // `legacy_private_oklab_codes_read_but_are_not_written`.
        for code in -1..=64 {
            let driven = VARIANTS.iter().any(|&v| vips_tag(v).0 == code);
            assert_eq!(
                interpretation_from_code(code).is_some(),
                driven,
                "VipsInterpretation {code}: the reader and this table disagree"
            );
        }
    }

    /// A `.v` written by real vips tags OkLab/OkLch with `Type = 30` / `31`
    /// (`VIPS_INTERPRETATION_OKLAB` / `_OKLCH`,
    /// `libvips/include/vips/image.h:115-116`), so libviprs has to read those
    /// files back as [`Interpretation::OkLab`] / [`Interpretation::OkLch`]
    /// instead of falling through to format inference and reporting
    /// `Multiband`.
    ///
    /// The fixture is the byte-for-byte 64-byte header vips 8.18.4 wrote for
    ///
    /// ```text
    /// vips black t.v 4 4 --bands 3
    /// vips colourspace t.v ok.v oklab      # and again for oklch
    /// ```
    ///
    /// followed by that file's 4x4x3 float pixels (all zero, it is black), so
    /// nothing binary is checked in. vips writes the header in the machine's
    /// own byte order; these are the little-endian bytes, which on a
    /// big-endian host also exercises the decoder's swap path.
    #[test]
    fn vips_written_oklab_and_oklch_read_back_tagged() {
        #[rustfmt::skip]
        const VIPS_OKLAB_HEADER: [u8; VIPS_HEADER_LEN] = [
            0xb6, 0xa6, 0xf2, 0x08, // magic (little-endian)
            0x04, 0x00, 0x00, 0x00, // Xsize 4
            0x04, 0x00, 0x00, 0x00, // Ysize 4
            0x03, 0x00, 0x00, 0x00, // Bands 3
            0x20, 0x00, 0x00, 0x00, // Bbits 32
            0x06, 0x00, 0x00, 0x00, // BandFmt 6 (float)
            0x00, 0x00, 0x00, 0x00, // Coding 0 (none)
            0x1e, 0x00, 0x00, 0x00, // Type 30 (VIPS_INTERPRETATION_OKLAB)
            0x00, 0x00, 0x80, 0x3f, // Xres 1.0
            0x00, 0x00, 0x80, 0x3f, // Yres 1.0
            0x00, 0x00, 0x00, 0x00, // Length (deprecated)
            0x00, 0x00, 0x00, 0x00, // Compression + Level (deprecated)
            0x00, 0x00, 0x00, 0x00, // Xoffset 0
            0x00, 0x00, 0x00, 0x00, // Yoffset 0
            0x00, 0x00, 0x00, 0x00, // reserved
            0x00, 0x00, 0x00, 0x00, // reserved
        ];
        // The oklch file vips wrote differs from the oklab one in exactly one
        // byte: the Type word at offset 28, 30 -> 31.
        const TYPE_OFFSET: usize = 28;

        for (code, expected) in [(30u8, Interpretation::OkLab), (31, Interpretation::OkLch)] {
            let mut bytes = VIPS_OKLAB_HEADER.to_vec();
            bytes[TYPE_OFFSET] = code;
            // 4x4 pixels, 3 float bands, all zero: `vips black` output.
            bytes.resize(VIPS_HEADER_LEN + 4 * 4 * 3 * 4, 0);

            let back = decode_bytes(&bytes).unwrap();
            assert_eq!(
                back.interpretation(),
                expected,
                "Type {code} must tag {expected:?}"
            );
            assert_eq!(
                back.get_field("interpretation").unwrap().as_str(),
                interpretation_nickname(expected)
            );
            assert_eq!((back.width(), back.height()), (4, 4));
            assert_eq!(back.format(), PixelFormat::with_channels(3, 4).unwrap());
        }
    }

    /// libviprs used to write private codes 1000 / 1001 for OkLab / OkLch,
    /// before libvips 8.18 assigned them 30 / 31. The private codes stay
    /// readable so `.v` files libviprs already wrote keep loading, but the
    /// encoder must never emit them again.
    #[test]
    fn legacy_private_oklab_codes_read_but_are_not_written() {
        // The header `Type` word, the same offset the vips fixture patches.
        const TYPE_OFFSET: usize = 28;

        assert_eq!(interpretation_from_code(1000), Some(Interpretation::OkLab));
        assert_eq!(interpretation_from_code(1001), Some(Interpretation::OkLch));

        for (interp, code) in [(Interpretation::OkLab, 30i32), (Interpretation::OkLch, 31)] {
            let im = Raster::black(2, 2).copy().interpretation(interp).build();
            let bytes = im.encode_vips().unwrap();
            let written =
                i32::from_ne_bytes(bytes[TYPE_OFFSET..TYPE_OFFSET + 4].try_into().unwrap());
            assert_eq!(written, code, "encode_vips must write the libvips code");
            // And it round-trips through our own reader.
            assert_eq!(decode_bytes(&bytes).unwrap().interpretation(), interp);
        }

        // A legacy libviprs file (private code in the Type word) still loads.
        let mut legacy = Raster::black(2, 2)
            .copy()
            .interpretation(Interpretation::OkLab)
            .build()
            .encode_vips()
            .unwrap();
        legacy[TYPE_OFFSET..TYPE_OFFSET + 4].copy_from_slice(&1000i32.to_ne_bytes());
        assert_eq!(
            decode_bytes(&legacy).unwrap().interpretation(),
            Interpretation::OkLab
        );

        // The one above is written in the host's byte order, which is the
        // easy half. A migrating file is a fixed byte pattern on somebody
        // else's disk, and 1000 / 1001 have a low byte of 0xe8 / 0xe9, so
        // unlike the codes 30 / 31 they do not survive a byte-order swap by
        // accident. This is the explicit little-endian header, laid out like
        // the vips fixture above, which puts the decoder's swap path under
        // test on a big-endian host.
        #[rustfmt::skip]
        const LEGACY_LE_HEADER: [u8; VIPS_HEADER_LEN] = [
            0xb6, 0xa6, 0xf2, 0x08, // magic (little-endian)
            0x02, 0x00, 0x00, 0x00, // Xsize 2
            0x02, 0x00, 0x00, 0x00, // Ysize 2
            0x03, 0x00, 0x00, 0x00, // Bands 3
            0x20, 0x00, 0x00, 0x00, // Bbits 32
            0x06, 0x00, 0x00, 0x00, // BandFmt 6 (float)
            0x00, 0x00, 0x00, 0x00, // Coding 0 (none)
            0xe8, 0x03, 0x00, 0x00, // Type 1000 (legacy libviprs OkLab)
            0x00, 0x00, 0x80, 0x3f, // Xres 1.0
            0x00, 0x00, 0x80, 0x3f, // Yres 1.0
            0x00, 0x00, 0x00, 0x00, // Length (deprecated)
            0x00, 0x00, 0x00, 0x00, // Compression + Level (deprecated)
            0x00, 0x00, 0x00, 0x00, // Xoffset 0
            0x00, 0x00, 0x00, 0x00, // Yoffset 0
            0x00, 0x00, 0x00, 0x00, // reserved
            0x00, 0x00, 0x00, 0x00, // reserved
        ];

        for (legacy_code, expected) in [
            (1000i32, Interpretation::OkLab),
            (1001, Interpretation::OkLch),
        ] {
            let mut bytes = LEGACY_LE_HEADER.to_vec();
            bytes[TYPE_OFFSET..TYPE_OFFSET + 4].copy_from_slice(&legacy_code.to_le_bytes());
            // 2x2 pixels, 3 float bands, all zero.
            bytes.resize(VIPS_HEADER_LEN + 2 * 2 * 3 * 4, 0);

            assert_eq!(
                decode_bytes(&bytes).unwrap().interpretation(),
                expected,
                "little-endian legacy Type {legacy_code} must still read as {expected:?}"
            );
        }
    }

    // -- free functions -----------------------------------------------------

    /**
     * Tests the libvips tokeniser cases from the ported infrastructure
     * suite plus escape handling.
     */
    #[test]
    fn tokenize_quoting_and_escapes() {
        assert_eq!(tokenize("hello world"), vec!["hello", "world"]);
        assert_eq!(tokenize("\"hello world\""), vec!["hello world"]);
        assert_eq!(tokenize("a \"b c\" d"), vec!["a", "b c", "d"]);
        assert_eq!(tokenize("hello\\ world"), vec!["hello world"]);
        assert_eq!(tokenize("'single quoted'"), vec!["single quoted"]);
        assert_eq!(tokenize("\"esc \\\" quote\""), vec!["esc \" quote"]);
        assert_eq!(tokenize("   "), Vec::<String>::new());
        assert_eq!(tokenize(""), Vec::<String>::new());
        // Unterminated quote: the rest of the string is one token.
        assert_eq!(tokenize("\"open ended"), vec!["open ended"]);
    }

    /**
     * Tests the thumbnail geometry parser forms: W, WxH, Wx, xH,
     * modifiers, and garbage.
     */
    #[test]
    fn thumbnail_geometry_forms() {
        assert_eq!(
            parse_thumbnail_geometry("200"),
            ThumbnailGeometry {
                width: Some(200),
                height: None
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("200x150"),
            ThumbnailGeometry {
                width: Some(200),
                height: Some(150)
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("200x"),
            ThumbnailGeometry {
                width: Some(200),
                height: None
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("x150"),
            ThumbnailGeometry {
                width: None,
                height: Some(150)
            }
        );
        assert_eq!(
            parse_thumbnail_geometry(" 200x150> "),
            ThumbnailGeometry {
                width: Some(200),
                height: Some(150)
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("128X128!"),
            ThumbnailGeometry {
                width: Some(128),
                height: Some(128)
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("banana"),
            ThumbnailGeometry::default()
        );
    }

    /// The `.v` reader enforces the single-axis ceiling from the per-decode
    /// [`DecodeLimits::max_coord`] field: an over-ceiling dimension returns
    /// the dedicated typed [`SourceError::CoordLimitExceeded`] (never a
    /// panic or a wrapping cast/overflow), while an in-bounds decode is
    /// unaffected. This replaces the former process-global enforcement
    /// (#293).
    #[test]
    fn max_coord_enforced_per_decode() {
        let im = Raster::zeroed(2000, 1, PixelFormat::Gray8).unwrap();
        let bytes = im.encode_vips().unwrap();

        // (a) Over the per-decode ceiling → dedicated typed error carrying
        // the offending dimensions and ceiling, not a panic or an opaque
        // substring-matched string. Built through the `#[non_exhaustive]`
        // builder setter, the supported external construction path.
        let tight = DecodeLimits::default().with_max_coord(1000);
        let err = decode_vips_bytes(&bytes, tight).unwrap_err();
        assert!(
            matches!(
                err,
                SourceError::CoordLimitExceeded {
                    width: 2000,
                    height: 1,
                    max_coord: 1000,
                }
            ),
            "expected a typed CoordLimitExceeded error, got {err}"
        );

        // Boundary: exactly at the ceiling is accepted; one below rejects.
        let at = DecodeLimits::default().with_max_coord(2000);
        assert!(decode_vips_bytes(&bytes, at).is_ok());
        let below = DecodeLimits::default().with_max_coord(1999);
        assert!(matches!(
            decode_vips_bytes(&bytes, below),
            Err(SourceError::CoordLimitExceeded { .. })
        ));

        // (b) The default in-bounds decode is unaffected and round-trips.
        let decoded = decode_vips_bytes(&bytes, DecodeLimits::default()).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (2000, 1));

        // (c) Security-critical ordering (#422): a malicious header claiming
        // a width far above the ceiling with a body far shorter than the
        // promised pixel data must reject on the coordinate ceiling *before*
        // the truncation/allocation check — proving the ceiling is enforced
        // ahead of reading or allocating pixels, not after. A 4x4 Gray8 `.v`
        // whose width field is overwritten with 20,000,000 (past the default
        // 10,000,000 ceiling) and whose body is truncated to a few bytes must
        // surface CoordLimitExceeded, never the truncated-pixel-data
        // VipsFormat error and never an oversized allocation or wrapping cast.
        let small = Raster::zeroed(4, 4, PixelFormat::Gray8).unwrap();
        let mut malicious = small.encode_vips().unwrap();
        // Width lives at header offset 4 (native-endian i32); see
        // `encode_vips_impl`. A 20,000,000-px width promises 80 MB of body.
        let huge: i32 = 20_000_000;
        malicious[4..8].copy_from_slice(&huge.to_ne_bytes());
        // Truncate so only a handful of the promised body bytes are present.
        malicious.truncate(VIPS_HEADER_LEN + 8);
        let err = decode_vips_bytes(&malicious, DecodeLimits::default()).unwrap_err();
        assert!(
            matches!(
                err,
                SourceError::CoordLimitExceeded {
                    width: 20_000_000,
                    height: 4,
                    max_coord: DEFAULT_MAX_COORD,
                }
            ),
            "expected CoordLimitExceeded before the truncation check, got {err}"
        );
    }
}
