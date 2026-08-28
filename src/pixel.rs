use core::num::NonZeroU16;

/// Canonical pixel formats used throughout the pipeline.
///
/// Source images are normalized into one of these formats at decode time.
/// This keeps format-specific complexity out of the planner, execution engine,
/// and [`Raster`](crate::raster::Raster) buffer management. Every named format
/// is defined by two axes -- channel count (1, 3, or 4) and sample type
/// (unsigned 8-bit, unsigned 16-bit, or 32-bit float). The band operations in
/// [`crate::bands`] can produce intermediate images with any other band count
/// (for example 2 bands from `extract_bands`, or 100 bands from `bandfold`);
/// those are carried by the `Multi8` / `Multi16` / `FloatF32` variants.
///
/// Float samples are stored as native-endian `f32` values in the raster's
/// byte buffer, matching the native-order convention the 16-bit formats
/// already use. `RgbaF32` is the named four-band float format the ported
/// compositing tests cast to; every other float band count is carried by
/// `FloatF32(n)`.
///
/// # Canonical spelling
///
/// A layout with a named variant has two constructible spellings, because
/// the tuple variants are public: `FloatF32(4)` names what `RgbaF32` names,
/// and `Multi8(3)` names what `Rgb8` names. The named one is canonical.
/// [`PixelFormat::with_channels`] produces it, [`PixelFormat::canonical`]
/// converts to it, and everything this crate hands you is already in it: a
/// [`Raster`](crate::raster::Raster) canonicalises the format it is built
/// with, and the manifest wire format writes the canonical tag. `PartialEq`
/// and `Hash` are derived and so distinguish the two spellings, which is why
/// nothing here produces the non-canonical one (issue #531).
///
/// # Variants
///
/// | Variant      | Channels | Bits/channel | Bytes/pixel |
/// |--------------|----------|--------------|-------------|
/// | `Gray8`      | 1        | 8            | 1           |
/// | `Gray16`     | 1        | 16           | 2           |
/// | `Rgb8`       | 3        | 8            | 3           |
/// | `Rgba8`      | 4        | 8            | 4           |
/// | `Rgb16`      | 3        | 16           | 6           |
/// | `Rgba16`     | 4        | 16           | 8           |
/// | `RgbaF32`    | 4        | 32 (float)   | 16          |
/// | `Multi8(n)`  | n        | 8            | n           |
/// | `Multi16(n)` | n        | 16           | 2n          |
/// | `FloatF32(n)`| n        | 32 (float)   | 4n          |
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
/// * [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-format)
/// (pyramid overview at [`#pyramid`](https://libviprs.org/cli/#pyramid))
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelFormat {
    /// Single-channel 8-bit grayscale.
    Gray8,
    /// Single-channel 16-bit grayscale.
    Gray16,
    /// Three-channel 8-bit RGB colour.
    Rgb8,
    /// Four-channel 8-bit RGBA colour with alpha.
    Rgba8,
    /// Three-channel 16-bit RGB colour.
    Rgb16,
    /// Four-channel 16-bit RGBA colour with alpha.
    Rgba16,
    /// Four-channel 32-bit float RGBA colour with alpha, stored as
    /// native-endian `f32` samples. This is the float format the ported
    /// compositing tests cast to (`cast(PixelFormat::RgbaF32)`). Float
    /// rasters are compute intermediates: the tile encoding sinks reject
    /// them with a typed error, and the `.v` container is the only
    /// encode/decode path that carries them.
    RgbaF32,
    /// N-channel 8-bit multiband image, produced by the band operations in
    /// [`crate::bands`] when the band count is not 1, 3, or 4. Multiband
    /// rasters are compute intermediates: the decode, resize, and tile
    /// encoding paths do not accept them.
    Multi8(NonZeroU16),
    /// N-channel 16-bit multiband image; see [`PixelFormat::Multi8`].
    Multi16(NonZeroU16),
    /// N-channel 32-bit float image, stored as native-endian `f32` samples.
    /// This is the carrier for every float band count other than 4 (which
    /// canonicalizes to [`PixelFormat::RgbaF32`]): single-band float ramps
    /// and maths results use `FloatF32(1)`, float colour intermediates
    /// `FloatF32(3)`, and so on. Like the `Multi` variants, float rasters
    /// are compute intermediates; see [`PixelFormat::RgbaF32`].
    FloatF32(NonZeroU16),
}

/// What the bytes at one channel sample *are*: the sample's type, as
/// distinct from how many bytes of it there are.
///
/// The crate used to answer that question with
/// [`PixelFormat::bytes_per_channel`], and byte width is not a sample kind.
/// One byte means `u8` or `i8`, two means `u16` or `i16`, and four means
/// `u32`, `i32` or `f32`. Every `match` keyed on the width therefore
/// carries a trailing `_` arm that reads whichever of those the caller did
/// not have in mind, silently, with nothing in the compiler to say so.
/// Dispatching on this enum instead makes each of those sites a decision
/// the compiler forces when a carrier is added (issue #607).
///
/// Both accessors stay, and they answer different questions: use
/// [`PixelFormat::bytes_per_channel`] for a stride or a buffer size, and
/// [`PixelFormat::kind`] to decide how to interpret the bytes.
///
/// # Variants
///
/// | Variant | Rust type | Bytes | Range |
/// |---|---|---|---|
/// | `U8`  | `u8`  | 1 | `0..=255` |
/// | `I8`  | `i8`  | 1 | `-128..=127` |
/// | `U16` | `u16` | 2 | `0..=65535` |
/// | `I16` | `i16` | 2 | `-32768..=32767` |
/// | `U32` | `u32` | 4 | `0..=4294967295` |
/// | `I32` | `i32` | 4 | `-2147483648..=2147483647` |
/// | `F32` | `f32` | 4 | none |
///
/// Multi-byte samples are stored in **native** byte order throughout the
/// crate.
///
/// # Kinds without a carrier
///
/// [`PixelFormat`] carries `U8`, `U16` and `F32`. `I8`, `I16`, `I32` and
/// `U32` are the sample kinds issues #516 and #517 ask for, and no
/// `PixelFormat` variant produces them yet, so
/// [`PixelFormat::kind`] never returns one and
/// [`PixelFormat::with_kind`] answers `None` for them.
///
/// They are named here rather than when the carriers land because the
/// answers this enum gives are the part of those issues that has to be
/// measured rather than reasoned about, and measuring them costs nothing
/// while the carriers cost a crate-wide refactor. [`SampleKind::promote`]
/// is the case in point: it is the libvips `vips__formatalike` order, and
/// four of its pairs are ones "the wider kind wins" gets wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleKind {
    /// Unsigned 8-bit samples (`u8`).
    U8,
    /// Unsigned 16-bit samples (`u16`), native byte order.
    U16,
    /// 32-bit IEEE-754 float samples (`f32`), native byte order.
    F32,
    /// Signed 8-bit samples (`i8`), the libvips `VIPS_FORMAT_CHAR` carrier
    /// (issue #516). No [`PixelFormat`] produces this kind yet.
    I8,
    /// Signed 16-bit samples (`i16`), native byte order, the libvips
    /// `VIPS_FORMAT_SHORT` carrier (issue #516). No [`PixelFormat`]
    /// produces this kind yet.
    I16,
    /// Unsigned 32-bit samples (`u32`), native byte order, the libvips
    /// `VIPS_FORMAT_UINT` carrier (issue #517) and the one the counting
    /// ops of issue #532 need. No [`PixelFormat`] produces this kind yet.
    U32,
    /// Signed 32-bit samples (`i32`), native byte order, the libvips
    /// `VIPS_FORMAT_INT` carrier (issue #516) and the one the `profile`
    /// op would need. No [`PixelFormat`] produces this kind yet.
    I32,
}

impl SampleKind {
    /// Bytes one sample of this kind occupies: 1, 2, or 4.
    ///
    /// This is the same number [`PixelFormat::bytes_per_channel`] returns
    /// for any format of this kind, and
    /// [`PixelFormat::kind`]`().bytes()` equals it for every variant. The
    /// direction that does *not* hold is the reason this enum exists: a
    /// byte width does not name a kind.
    pub fn bytes(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
        }
    }

    /// Whether samples of this kind are floating point.
    ///
    /// The same answer [`PixelFormat::is_float`] gives for any format of
    /// this kind.
    pub fn is_float(self) -> bool {
        match self {
            Self::U8 | Self::U16 | Self::I8 | Self::I16 | Self::U32 | Self::I32 => false,
            Self::F32 => true,
        }
    }

    /// Whether a sample of this kind can hold a negative value.
    ///
    /// True for the signed integer kinds and for `F32`. This is the
    /// assumption issue #607 names as the expensive half of #516: the
    /// crate's integer paths clamp into `0..=max`, and that floor is a
    /// property of the kind rather than of arithmetic. Where the floor
    /// itself is wanted, read it off [`SampleKind::range`].
    pub fn is_signed(self) -> bool {
        match self {
            Self::U8 | Self::U16 | Self::U32 => false,
            Self::I8 | Self::I16 | Self::I32 | Self::F32 => true,
        }
    }

    /// The inclusive value range an integer sample of this kind can hold,
    /// or `None` for a float kind.
    ///
    /// The signed counterpart of [`SampleKind::max_value`], and the one to
    /// reach for when the floor matters: a saturating write into a signed
    /// carrier clamps at the low end too, and `0` is only the right floor
    /// for three of the six integer kinds.
    ///
    /// `None` on `F32` says the same thing [`SampleKind::max_value`]'s
    /// `None` says: a float carrier's meaningful range comes from its
    /// [`Interpretation`](crate::conversion::Interpretation), not from the
    /// sample kind.
    pub fn range(self) -> Option<(i64, i64)> {
        let range = match self {
            Self::U8 => (0, 0xFF),
            Self::I8 => (i64::from(i8::MIN), i64::from(i8::MAX)),
            Self::U16 => (0, 0xFFFF),
            Self::I16 => (i64::from(i16::MIN), i64::from(i16::MAX)),
            Self::U32 => (0, 0xFFFF_FFFF),
            Self::I32 => (i64::from(i32::MIN), i64::from(i32::MAX)),
            Self::F32 => return None,
        };
        Some(range)
    }

    /// The largest value an integer sample of this kind can hold, or `None`
    /// for a float kind.
    ///
    /// `None` is not "no ceiling exists in practice", it is "the sample kind
    /// does not imply one". A float raster's meaningful range comes from its
    /// [`Interpretation`](crate::conversion::Interpretation) instead, which
    /// is why [`crate::arithmetic`]'s premultiply bracket takes its
    /// `max_alpha` from the tag rather than from the depth.
    pub fn max_value(self) -> Option<u32> {
        // Every integer kind's ceiling fits a `u32`, including
        // `I32`'s 2147483647, so this stays the convenient spelling for the
        // unsigned paths. `range` is the total answer and this is derived
        // from it, so the two cannot drift.
        self.range()
            .map(|(_, hi)| u32::try_from(hi).expect("every sample kind's ceiling fits a u32"))
    }

    /// The kind that carries both `self` and `other` losslessly: the
    /// promotion libvips calls `vips__formatalike`.
    ///
    /// `F32` wins over every integer kind, and among the integers the
    /// answer is the narrowest kind that holds both operands' whole ranges.
    /// That is what a two-image arithmetic op needs when its inputs
    /// disagree.
    ///
    /// The match is over the *pair*, deliberately, and not over
    /// [`SampleKind::bytes`]. Byte width cannot order the kinds, and there
    /// are two independent reasons rather than one: a four-byte integer and
    /// a four-byte float are the same width and promote in opposite
    /// directions, and a signed kind paired with an unsigned one of the
    /// same or greater width promotes to something **wider than either**.
    /// Adding a kind leaves pairs uncovered here and so fails to compile,
    /// which is the point (issue #607).
    ///
    /// # The measured table
    ///
    /// Swept on `/opt/homebrew/bin/vips` 8.18.6 with
    /// `vips boolean <a> <b> out and`, whose format table maps every
    /// integer format to itself, so the output format *is* the formatalike
    /// result. The float column was taken with `vips multiply`
    /// (`FLOAT -> FLOAT`), and the two probes agree everywhere they
    /// overlap.
    ///
    /// |  | `U8` | `I8` | `U16` | `I16` | `U32` | `I32` |
    /// |---|---|---|---|---|---|---|
    /// | `U8`  | `U8`  | `I16` | `U16` | `I16` | `U32` | `I32` |
    /// | `I8`  | `I16` | `I8`  | `I32` | `I16` | `I32` | `I32` |
    /// | `U16` | `U16` | `I32` | `U16` | `I32` | `U32` | `I32` |
    /// | `I16` | `I16` | `I16` | `I32` | `I16` | `I32` | `I32` |
    /// | `U32` | `U32` | `I32` | `U32` | `I32` | `U32` | `I32` |
    /// | `I32` | `I32` | `I32` | `I32` | `I32` | `I32` | `I32` |
    ///
    /// Four of those cells are the ones "the wider kind wins" misses:
    /// `(U8, I8)` is two one-byte kinds promoting to a two-byte one,
    /// `(I8, U16)` and `(U16, I16)` are two-byte-or-less pairs promoting to
    /// four bytes, and `(U32, I8)` takes its sign from the one-byte
    /// operand.
    pub fn promote(self, other: Self) -> Self {
        use SampleKind::{F32, I8, I16, I32, U8, U16, U32};
        #[rustfmt::skip]
        let promoted = match (self, other) {
            (F32, _) | (_, F32) => F32,
            (U8, U8) => U8,
            (I8, I8) => I8,
            (U8, U16) | (U16, U8) | (U16, U16) => U16,
            (U8, I8) | (U8, I16) | (I8, U8) | (I8, I16)
            | (I16, U8) | (I16, I8) | (I16, I16) => I16,
            (U8, U32) | (U16, U32) | (U32, U8) | (U32, U16) | (U32, U32) => U32,
            (U8, I32) | (I8, U16) | (I8, U32) | (I8, I32)
            | (U16, I8) | (U16, I16) | (U16, I32)
            | (I16, U16) | (I16, U32) | (I16, I32)
            | (U32, I8) | (U32, I16) | (U32, I32)
            | (I32, U8) | (I32, I8) | (I32, U16) | (I32, I16)
            | (I32, U32) | (I32, I32) => I32,
        };
        promoted
    }

    /// The number of histogram bins that covers every value of this kind
    /// exactly once, or `None` where a value-indexed table is not the right
    /// shape.
    ///
    /// 256 for the one-byte kinds and 65536 for the two-byte ones, matching
    /// the bin counts [`crate::histogram`] uses and the ones libvips picks
    /// in `hist_find.c`.
    ///
    /// `F32` is `None` because a float histogram needs a range and a bin
    /// width, not a value-indexed table. `U32` and `I32` are `None` for the
    /// sibling reason: 2^32 bins is not a table either, and libvips does
    /// not build one. It casts a 32-bit input down first, which is
    /// observable from the outside: on 8.18.6 a `uint` image whose largest
    /// sample is 70000 gives a **65536**-wide histogram, so the sample was
    /// saturated into `ushort` before it was counted.
    ///
    /// The signed one-byte and two-byte kinds report the same counts their
    /// unsigned siblings do, and the bins are indexed by the *unsigned*
    /// value: libvips casts a signed input to the unsigned kind of the same
    /// width first, so every negative sample lands in bin zero. Measured on
    /// 8.18.6, a `char` image holding `[-128, -1, 0, 127]` gives a 256-wide
    /// histogram with `bin 0 = 3` and `bin 127 = 1`.
    pub fn hist_bins(self) -> Option<usize> {
        match self {
            Self::U8 | Self::I8 => Some(256),
            Self::U16 | Self::I16 => Some(65536),
            Self::U32 | Self::I32 | Self::F32 => None,
        }
    }
}

impl PixelFormat {
    /// Bytes per pixel for this format.
    ///
    /// Equal to `channels() * bytes_per_channel()`. Used by [`Raster`](crate::raster::Raster)
    /// to compute buffer sizes and strides.
    pub fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_channel()
    }

    /// Number of channels (1 for grayscale, 3 for RGB, 4 for RGBA, `n` for
    /// the multiband variants).
    pub fn channels(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16 => 1,
            Self::Rgb8 | Self::Rgb16 => 3,
            Self::Rgba8 | Self::Rgba16 | Self::RgbaF32 => 4,
            Self::Multi8(n) | Self::Multi16(n) | Self::FloatF32(n) => n.get() as usize,
        }
    }

    /// The canonical format for a channel count and byte depth.
    ///
    /// For the 8- and 16-bit depths, counts 1, 3, and 4 map to the named
    /// `Gray` / `Rgb` / `Rgba` variants and every other count maps to
    /// `Multi8` / `Multi16`. For the 4-byte float depth, count 4 maps to
    /// the named `RgbaF32` and every other count to `FloatF32`. The band
    /// operations use this so a 3-band multiband result compares equal to
    /// `Rgb8` rather than living as a `Multi8(3)` alias.
    ///
    /// Returns `None` when `channels` is 0 or above `u16::MAX`, or when
    /// `bytes_per_channel` is not 1, 2, or 4.
    pub fn with_channels(channels: usize, bytes_per_channel: usize) -> Option<Self> {
        let fmt = match (channels, bytes_per_channel) {
            (1, 1) => Self::Gray8,
            (1, 2) => Self::Gray16,
            (3, 1) => Self::Rgb8,
            (3, 2) => Self::Rgb16,
            (4, 1) => Self::Rgba8,
            (4, 2) => Self::Rgba16,
            (4, 4) => Self::RgbaF32,
            (n, 1) => Self::Multi8(NonZeroU16::new(u16::try_from(n).ok()?)?),
            (n, 2) => Self::Multi16(NonZeroU16::new(u16::try_from(n).ok()?)?),
            (n, 4) => Self::FloatF32(NonZeroU16::new(u16::try_from(n).ok()?)?),
            _ => return None,
        };
        Some(fmt)
    }

    /// The canonical spelling of this format's pixel layout.
    ///
    /// The tuple variants are public, so a layout that has a named variant
    /// has two constructible spellings: `FloatF32(4)` names what `RgbaF32`
    /// names, `Multi8(3)` names what `Rgb8` names, and so on for the seven
    /// rows below. [`PixelFormat::with_channels`] produces the right-hand
    /// column; direct construction can produce the left-hand one. This maps
    /// one to the other, and is the identity on everything else.
    ///
    /// | non-canonical | canonical |
    /// |---|---|
    /// | `Multi8(1)`   | `Gray8`   |
    /// | `Multi8(3)`   | `Rgb8`    |
    /// | `Multi8(4)`   | `Rgba8`   |
    /// | `Multi16(1)`  | `Gray16`  |
    /// | `Multi16(3)`  | `Rgb16`   |
    /// | `Multi16(4)`  | `Rgba16`  |
    /// | `FloatF32(4)` | `RgbaF32` |
    ///
    /// Note that `FloatF32(1)` and `FloatF32(3)` are *already* canonical:
    /// four is the only float band count with a named variant.
    ///
    /// You rarely need to call this. Nothing this crate produces is
    /// non-canonical: every raster canonicalises its format at construction
    /// (see [`Raster::new`](crate::raster::Raster::new)), and the manifest
    /// wire format writes the canonical tag. It is here for the case where
    /// you built a format yourself and want to compare it against one of
    /// ours, since `PartialEq` is derived and so distinguishes the two
    /// spellings.
    pub fn canonical(self) -> Self {
        // Spelled out per carrier rather than as
        // `with_channels(self.channels(), self.bytes_per_channel())`, which
        // would give the same answers today and be shorter. Two reasons, and
        // the second is the load-bearing one.
        //
        // `with_channels` is keyed on a byte depth, and a byte depth does
        // not identify a carrier: an unsigned-32 carrier (issue #517) would
        // share `bytes_per_channel() == 4` with the float one, so the short
        // form would quietly canonicalise `Uint32(3)` to `FloatF32(3)` --
        // the same class of silent retag this method exists to remove.
        //
        // And there is no wildcard arm here, so adding a carrier variant to
        // this `#[non_exhaustive]` enum is a compile error at this match
        // rather than a default that happens to be wrong. The decision gets
        // forced instead of inherited.
        match self {
            Self::Gray8
            | Self::Gray16
            | Self::Rgb8
            | Self::Rgba8
            | Self::Rgb16
            | Self::Rgba16
            | Self::RgbaF32 => self,
            Self::Multi8(n) => match n.get() {
                1 => Self::Gray8,
                3 => Self::Rgb8,
                4 => Self::Rgba8,
                _ => self,
            },
            Self::Multi16(n) => match n.get() {
                1 => Self::Gray16,
                3 => Self::Rgb16,
                4 => Self::Rgba16,
                _ => self,
            },
            // Four is the only float band count with a named variant, so
            // `FloatF32(1)` and `FloatF32(3)` are already canonical.
            Self::FloatF32(n) => match n.get() {
                4 => Self::RgbaF32,
                _ => self,
            },
        }
    }

    /// Whether this format is the canonical spelling of its pixel layout.
    ///
    /// Equivalent to `self.canonical() == self`. See
    /// [`PixelFormat::canonical`] for the seven layouts where it is `false`.
    pub fn is_canonical(self) -> bool {
        self.canonical() == self
    }

    /// Whether this format includes an alpha (transparency) channel.
    ///
    /// Returns `true` for the four-band layouts: `Rgba8`, `Rgba16`,
    /// `RgbaF32`, and the tuple spellings of those same layouts
    /// (`Multi8(4)`, `Multi16(4)`, `FloatF32(4)`). The question is about the
    /// pixel layout, not about which of its two spellings you are holding,
    /// so it is answered on [`PixelFormat::canonical`] (issue #531).
    pub fn has_alpha(self) -> bool {
        matches!(self.canonical(), Self::Rgba8 | Self::Rgba16 | Self::RgbaF32)
    }

    /// Whether this format stores 32-bit float samples (`RgbaF32` or
    /// `FloatF32`).
    ///
    /// Float samples are raw `f32` values in native byte order; the unsigned
    /// formats store `u8` / `u16` samples. Code that interprets raw sample
    /// bytes must dispatch on this (or on [`PixelFormat::bytes_per_channel`])
    /// rather than assuming "not 8-bit means 16-bit".
    pub fn is_float(self) -> bool {
        matches!(self, Self::RgbaF32 | Self::FloatF32(_))
    }

    /// Bytes per channel sample (1 for 8-bit formats, 2 for 16-bit formats,
    /// 4 for float formats).
    ///
    /// Useful when converting between bit depths or when working with raw
    /// sample values that need to be read as `u8` vs `u16` vs `f32`.
    pub fn bytes_per_channel(self) -> usize {
        match self {
            Self::Gray8 | Self::Rgb8 | Self::Rgba8 | Self::Multi8(_) => 1,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 | Self::Multi16(_) => 2,
            Self::RgbaF32 | Self::FloatF32(_) => 4,
        }
    }

    /// The sample kind this format carries.
    ///
    /// This is the crate's single answer to "what are these bytes", and the
    /// match below is the chokepoint that makes it one: it has no wildcard
    /// arm, so a new carrier variant on this `#[non_exhaustive]` enum is a
    /// compile error here rather than a width that some other module reads
    /// as the wrong type (issue #607).
    ///
    /// Prefer this over [`PixelFormat::bytes_per_channel`] whenever the
    /// question is how to interpret a sample; the width is still the right
    /// call for a stride or a buffer size.
    pub fn kind(self) -> SampleKind {
        match self {
            Self::Gray8 | Self::Rgb8 | Self::Rgba8 | Self::Multi8(_) => SampleKind::U8,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 | Self::Multi16(_) => SampleKind::U16,
            Self::RgbaF32 | Self::FloatF32(_) => SampleKind::F32,
        }
    }

    /// The canonical format for a channel count and a sample kind.
    ///
    /// The kind-keyed form of [`PixelFormat::with_channels`], and the one to
    /// reach for: `with_channels(bands, 4)` has to decide what four bytes
    /// means, and today it answers "float" for every caller including one
    /// that wanted an integer. This cannot be asked ambiguously.
    ///
    /// Returns `None` for two different reasons, and a caller that needs to
    /// tell them apart has to check the kind itself:
    ///
    /// * the channel counts `with_channels` rejects, zero and anything
    ///   above `u16::MAX`;
    /// * a [`SampleKind`] no `PixelFormat` carries. `I8`, `I16`, `I32` and
    ///   `U32` are named by issues #516 and #517 and have no variant yet, so
    ///   there is no format to return. Falling through to
    ///   `with_channels(channels, kind.bytes())` would answer `Rgb16` for
    ///   three bands of `I16` and `FloatF32(3)` for three bands of `U32`,
    ///   which is the silent retag [`PixelFormat::canonical`]'s comment
    ///   warns about, arriving through the very constructor built to make
    ///   the question unambiguous.
    ///
    /// The match is total, so a carrier variant added to `PixelFormat` has
    /// to move its kind out of the refusing arm here.
    pub fn with_kind(channels: usize, kind: SampleKind) -> Option<Self> {
        match kind {
            SampleKind::U8 | SampleKind::U16 | SampleKind::F32 => {
                Self::with_channels(channels, kind.bytes())
            }
            SampleKind::I8 | SampleKind::I16 | SampleKind::U32 | SampleKind::I32 => None,
        }
    }

    /// Return the variant of this format that includes an alpha channel.
    ///
    /// `Gray8` and `Gray16` promote to `Rgba8` / `Rgba16` respectively (not
    /// `GrayAlpha`), because the pipeline does not use a gray+alpha format.
    /// One- and three-band float images promote to `RgbaF32`, since
    /// `FloatF32(1)` / `FloatF32(3)` are the canonical float gray and RGB
    /// carriers. If the format already has alpha, returns `self` unchanged.
    /// A band count with no named variant has no alpha concept, so
    /// `Multi8(2)` and `FloatF32(7)` come back as themselves.
    ///
    /// The answer is the same for both spellings of a layout: `Multi8(1)`
    /// promotes to `Rgba8` exactly as `Gray8` does, and the returned format
    /// is always canonical (issue #531).
    pub fn with_alpha(self) -> Self {
        // On the canonical spelling, so `Multi8(1)` promotes the way `Gray8`
        // does rather than falling through unchanged (issue #531).
        match self.canonical() {
            Self::Gray8 => Self::Rgba8,
            Self::Gray16 => Self::Rgba16,
            Self::Rgb8 => Self::Rgba8,
            Self::Rgb16 => Self::Rgba16,
            Self::FloatF32(n) if matches!(n.get(), 1 | 3) => Self::RgbaF32,
            other => other,
        }
    }

    /// Return the variant of this format with the alpha channel removed.
    ///
    /// `Rgba8` demotes to `Rgb8`, `Rgba16` to `Rgb16`, and `RgbaF32` to
    /// `FloatF32(3)` (the canonical three-band float carrier). A format
    /// without alpha keeps its layout, and comes back in that layout's
    /// canonical spelling: `Multi8(3)` demotes to `Rgb8`, which is the same
    /// pixel layout under the name `with_channels` gives it (issue #531).
    pub fn without_alpha(self) -> Self {
        // On the canonical spelling, for the reason `with_alpha` is.
        match self.canonical() {
            Self::Rgba8 => Self::Rgb8,
            Self::Rgba16 => Self::Rgb16,
            // Expect: 3 is non-zero, so the constructor cannot fail.
            Self::RgbaF32 => Self::FloatF32(NonZeroU16::new(3).expect("3 is non-zero")),
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /**
     * Tests that the sample-kind spine lives in this module, so the rest of
     * the crate has one shared answer to "what are these bytes" rather than
     * a hand-rolled depth enum per module (`colour.rs`'s private `SpaceDepth`
     * is exactly that duplicate, issue #607).
     * Works by scanning this module's own source, compiled in with
     * `include_str!`, for the type declaration; the needle is spelled in two
     * halves so this assertion is not itself a hit.
     * Input: `src/pixel.rs` -> Output: the declaration is present.
     */
    #[test]
    fn sample_kind_spine_is_declared_here() {
        const SRC: &str = include_str!("pixel.rs");
        // Positive control: the same scan finds a declaration that is
        // present, so a miss below is a real miss and not an empty read.
        assert!(
            SRC.contains(concat!("pub enum Pixel", "Format")),
            "positive control failed: the scan cannot see this module's source"
        );
        assert!(
            SRC.contains(concat!("pub enum Sample", "Kind")),
            "the SampleKind spine must be declared in src/pixel.rs, not \
             hand-rolled per module"
        );
    }

    /**
     * Tests that every format's sample kind agrees with its byte width, so
     * the two accessors cannot drift apart while both are in the crate.
     * Works by walking every variant, including a multiband spelling of
     * each carrier, and asserting `kind().bytes() == bytes_per_channel()`
     * and `kind().is_float() == is_float()`.
     * Input: all 10 variants -> Output: both identities hold for each.
     */
    #[test]
    fn kind_agrees_with_width_and_floatness() {
        let n = |v: u16| NonZeroU16::new(v).unwrap();
        for fmt in [
            PixelFormat::Gray8,
            PixelFormat::Gray16,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
            PixelFormat::RgbaF32,
            PixelFormat::Multi8(n(7)),
            PixelFormat::Multi16(n(7)),
            PixelFormat::FloatF32(n(7)),
        ] {
            assert_eq!(
                fmt.kind().bytes(),
                fmt.bytes_per_channel(),
                "kind width disagrees for {fmt:?}"
            );
            assert_eq!(
                fmt.kind().is_float(),
                fmt.is_float(),
                "kind floatness disagrees for {fmt:?}"
            );
        }
    }

    /**
     * Tests the sample kind each carrier reports, pinned per variant rather
     * than derived, so a variant that gets remapped is caught even if the
     * width identity above still holds.
     * Works by asserting the exact kind for one named and one multiband
     * spelling of all three carriers.
     * Input: Gray8 -> U8, Rgb16 -> U16, RgbaF32 -> F32, and the tuple
     * spellings alongside.
     */
    #[test]
    fn kind_per_carrier() {
        let n = |v: u16| NonZeroU16::new(v).unwrap();
        assert_eq!(PixelFormat::Gray8.kind(), SampleKind::U8);
        assert_eq!(PixelFormat::Rgba8.kind(), SampleKind::U8);
        assert_eq!(PixelFormat::Multi8(n(5)).kind(), SampleKind::U8);
        assert_eq!(PixelFormat::Gray16.kind(), SampleKind::U16);
        assert_eq!(PixelFormat::Rgb16.kind(), SampleKind::U16);
        assert_eq!(PixelFormat::Multi16(n(5)).kind(), SampleKind::U16);
        assert_eq!(PixelFormat::RgbaF32.kind(), SampleKind::F32);
        assert_eq!(PixelFormat::FloatF32(n(5)).kind(), SampleKind::F32);
    }

    /**
     * Tests the per-kind constants the sample modules read: byte width, the
     * integer ceiling, and the histogram bin count.
     * Works by asserting each accessor for all three kinds, including that
     * the float kind reports `None` for both quantities a depth would
     * imply, since a float carrier has neither a value ceiling nor a
     * value-indexed bin table.
     * Input: U8/U16/F32 -> Output: (1, 255, 256), (2, 65535, 65536),
     * (4, None, None).
     */
    #[test]
    fn sample_kind_constants() {
        assert_eq!(SampleKind::U8.bytes(), 1);
        assert_eq!(SampleKind::I8.bytes(), 1);
        assert_eq!(SampleKind::U16.bytes(), 2);
        assert_eq!(SampleKind::I16.bytes(), 2);
        assert_eq!(SampleKind::U32.bytes(), 4);
        assert_eq!(SampleKind::I32.bytes(), 4);
        assert_eq!(SampleKind::F32.bytes(), 4);

        assert_eq!(SampleKind::U8.max_value(), Some(255));
        assert_eq!(SampleKind::I8.max_value(), Some(127));
        assert_eq!(SampleKind::U16.max_value(), Some(65535));
        assert_eq!(SampleKind::I16.max_value(), Some(32767));
        assert_eq!(SampleKind::U32.max_value(), Some(4_294_967_295));
        assert_eq!(SampleKind::I32.max_value(), Some(2_147_483_647));
        assert_eq!(SampleKind::F32.max_value(), None);

        assert_eq!(SampleKind::U8.hist_bins(), Some(256));
        assert_eq!(SampleKind::I8.hist_bins(), Some(256));
        assert_eq!(SampleKind::U16.hist_bins(), Some(65536));
        assert_eq!(SampleKind::I16.hist_bins(), Some(65536));
        assert_eq!(SampleKind::U32.hist_bins(), None);
        assert_eq!(SampleKind::I32.hist_bins(), None);
        assert_eq!(SampleKind::F32.hist_bins(), None);

        for kind in ALL_KINDS {
            assert_eq!(
                kind.is_float(),
                kind == SampleKind::F32,
                "{kind:?} disagrees about being a float kind"
            );
        }
    }

    /// Every [`SampleKind`], so a kind added without a test here fails to
    /// compile at the array length rather than being quietly skipped.
    const ALL_KINDS: [SampleKind; 7] = [
        SampleKind::U8,
        SampleKind::I8,
        SampleKind::U16,
        SampleKind::I16,
        SampleKind::U32,
        SampleKind::I32,
        SampleKind::F32,
    ];

    /**
     * Tests the signedness and the inclusive value range of every sample
     * kind, which is what a saturating write into a signed carrier needs
     * and what the crate's `clamp(0, max)` floor currently assumes away
     * (issues #516, #607).
     * Works by asserting `is_signed` and `range` per kind, and then
     * cross-checking that `max_value` is exactly `range`'s upper bound, so
     * the two accessors cannot drift apart.
     * Input: U8 -> (0, 255) unsigned, I8 -> (-128, 127) signed, U32 ->
     * (0, 4294967295) unsigned, I32 -> (i32::MIN, i32::MAX) signed, F32 ->
     * None.
     */
    #[test]
    fn signed_kinds_report_their_range() {
        assert!(!SampleKind::U8.is_signed());
        assert!(!SampleKind::U16.is_signed());
        assert!(!SampleKind::U32.is_signed());
        assert!(SampleKind::I8.is_signed());
        assert!(SampleKind::I16.is_signed());
        assert!(SampleKind::I32.is_signed());
        assert!(SampleKind::F32.is_signed());

        assert_eq!(SampleKind::U8.range(), Some((0, 255)));
        assert_eq!(SampleKind::I8.range(), Some((-128, 127)));
        assert_eq!(SampleKind::U16.range(), Some((0, 65535)));
        assert_eq!(SampleKind::I16.range(), Some((-32768, 32767)));
        assert_eq!(SampleKind::U32.range(), Some((0, 4_294_967_295)));
        assert_eq!(
            SampleKind::I32.range(),
            Some((i64::from(i32::MIN), i64::from(i32::MAX)))
        );
        assert_eq!(SampleKind::F32.range(), None);

        for kind in ALL_KINDS {
            assert_eq!(
                kind.range().map(|(_, hi)| u32::try_from(hi).unwrap()),
                kind.max_value(),
                "{kind:?} range and max_value disagree"
            );
            // A signed kind is exactly one whose range reaches below zero,
            // and the float kind has no range to read it from.
            assert_eq!(
                kind.is_signed(),
                kind.range().is_none_or(|(lo, _)| lo < 0),
                "{kind:?} signedness and range disagree"
            );
        }
    }

    /**
     * Tests that `promote` is the `vips__formatalike` order and that it is
     * symmetric, which is what the two-image arithmetic ops rely on when
     * their inputs disagree.
     * Works by asserting all nine ordered pairs, so an arm that is right in
     * one direction and wrong in the other cannot pass.
     * Input: (U8,U16) -> U16, (U16,F32) -> F32, (U8,F32) -> F32, and each
     * kind with itself.
     */
    #[test]
    fn promote_is_the_formatalike_order() {
        use SampleKind::{F32, I8, I16, I32, U8, U16, U32};
        // The integer block, transcribed from the oracle sweep described
        // above `SampleKind::promote`. Every ordered pair is listed, so an
        // arm that is right in one direction and wrong in the other cannot
        // pass, and the six rows below are the six integer kinds in the
        // order the doc table lists them.
        #[rustfmt::skip]
        let table: [(SampleKind, SampleKind, SampleKind); 36] = [
            (U8, U8, U8),    (U8, I8, I16),  (U8, U16, U16),  (U8, I16, I16),  (U8, U32, U32),  (U8, I32, I32),
            (I8, U8, I16),   (I8, I8, I8),   (I8, U16, I32),  (I8, I16, I16),  (I8, U32, I32),  (I8, I32, I32),
            (U16, U8, U16),  (U16, I8, I32), (U16, U16, U16), (U16, I16, I32), (U16, U32, U32), (U16, I32, I32),
            (I16, U8, I16),  (I16, I8, I16), (I16, U16, I32), (I16, I16, I16), (I16, U32, I32), (I16, I32, I32),
            (U32, U8, U32),  (U32, I8, I32), (U32, U16, U32), (U32, I16, I32), (U32, U32, U32), (U32, I32, I32),
            (I32, U8, I32),  (I32, I8, I32), (I32, U16, I32), (I32, I16, I32), (I32, U32, I32), (I32, I32, I32),
        ];
        for (a, b, want) in table {
            assert_eq!(a.promote(b), want, "promote({a:?}, {b:?})");
        }

        // The float kind absorbs every integer kind and itself, in both
        // directions.
        for kind in ALL_KINDS {
            assert_eq!(kind.promote(F32), F32, "promote({kind:?}, F32)");
            assert_eq!(F32.promote(kind), F32, "promote(F32, {kind:?})");
        }

        // Symmetry over every ordered pair, which the table above only
        // covers for the integer block.
        for a in ALL_KINDS {
            for b in ALL_KINDS {
                assert_eq!(
                    a.promote(b),
                    b.promote(a),
                    "promote is not symmetric on ({a:?}, {b:?})"
                );
            }
        }

        // The four pairs "the wider kind wins" gets wrong, called out so
        // the reason the match is over the pair does not have to be
        // rediscovered: two of them promote *past* both inputs' widths, and
        // two land on a different kind at the same width.
        assert_eq!(U8.promote(I8), I16, "one-byte pair promotes to two bytes");
        assert_eq!(I8.promote(U16), I32, "a two-byte input promotes to four");
        assert_eq!(U16.promote(I16), I32, "a two-byte pair promotes to four");
        assert_eq!(U32.promote(I8), I32, "the narrower input decides the sign");
    }

    /**
     * Tests that `with_kind` names the same formats `with_channels` does at
     * each kind's width, and that it rejects the same channel counts.
     * Works by comparing the two constructors across the band counts with
     * named variants and one without, plus the zero and over-`u16::MAX`
     * rejections.
     * Input: (3, U8) -> Rgb8, (4, F32) -> RgbaF32, (0, U8) -> None,
     * (65536, U8) -> None.
     */
    #[test]
    fn with_kind_matches_with_channels() {
        for kind in [SampleKind::U8, SampleKind::U16, SampleKind::F32] {
            for channels in [1usize, 2, 3, 4, 7] {
                assert_eq!(
                    PixelFormat::with_kind(channels, kind),
                    PixelFormat::with_channels(channels, kind.bytes()),
                    "with_kind disagrees at {channels} channels of {kind:?}"
                );
            }
            assert_eq!(PixelFormat::with_kind(0, kind), None);
            assert_eq!(PixelFormat::with_kind(65_536, kind), None);
        }
        assert_eq!(
            PixelFormat::with_kind(3, SampleKind::U8),
            Some(PixelFormat::Rgb8)
        );
        assert_eq!(
            PixelFormat::with_kind(4, SampleKind::F32),
            Some(PixelFormat::RgbaF32)
        );
    }

    /**
     * Tests that `with_kind` refuses the sample kinds no `PixelFormat`
     * carries, rather than handing back a format of the same byte width
     * (issues #516, #517).
     * Works by asserting `None` for the four carrierless kinds at band
     * counts that are themselves perfectly legal, with a positive control
     * that `with_channels` at the same width *does* answer `Some` there, so
     * the `None` cannot be passing for a band-count rejection. Also pins
     * which format the width-keyed fall-through would have produced, since
     * that silent retag is the whole reason for the refusal.
     * Input: (3, I16) -> None while (3, 2 bytes) -> Some(Rgb16); (3, U32)
     * -> None while (3, 4 bytes) -> Some(FloatF32(3)).
     */
    #[test]
    fn with_kind_refuses_a_kind_no_format_carries() {
        let n = |v: u16| NonZeroU16::new(v).unwrap();
        for kind in [
            SampleKind::I8,
            SampleKind::I16,
            SampleKind::U32,
            SampleKind::I32,
        ] {
            for channels in [1usize, 2, 3, 4, 7] {
                assert_eq!(
                    PixelFormat::with_kind(channels, kind),
                    None,
                    "with_kind answered a format for the carrierless {kind:?}"
                );
                // Positive control: the band count is fine and the width
                // does name a format, so the `None` above is about the
                // kind and nothing else.
                assert!(
                    PixelFormat::with_channels(channels, kind.bytes()).is_some(),
                    "the control failed: {channels} channels at {} bytes names no format",
                    kind.bytes()
                );
            }
        }
        // What the width-keyed fall-through would have answered, which is
        // an unsigned format for a signed kind and a float one for an
        // integer kind.
        assert_eq!(
            PixelFormat::with_channels(3, SampleKind::I16.bytes()),
            Some(PixelFormat::Rgb16)
        );
        assert_eq!(
            PixelFormat::with_channels(3, SampleKind::U32.bytes()),
            Some(PixelFormat::FloatF32(n(3)))
        );
    }

    /**
     * Tests that no `PixelFormat` reports a sample kind the crate has no
     * carrier for, which is the claim every carrierless arm added in this
     * change rests on.
     * Works by sweeping every variant, including both spellings of the
     * tuple carriers, and asserting the kind is one of the three carried
     * ones, with a control that the sweep would have caught a fourth.
     * Input: every `PixelFormat` -> U8, U16 or F32.
     */
    #[test]
    fn no_format_reports_a_carrierless_kind() {
        let n = |v: u16| NonZeroU16::new(v).unwrap();
        let carried = [SampleKind::U8, SampleKind::U16, SampleKind::F32];
        let formats = [
            PixelFormat::Gray8,
            PixelFormat::Gray16,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
            PixelFormat::RgbaF32,
            PixelFormat::Multi8(n(1)),
            PixelFormat::Multi8(n(7)),
            PixelFormat::Multi16(n(1)),
            PixelFormat::Multi16(n(7)),
            PixelFormat::FloatF32(n(4)),
            PixelFormat::FloatF32(n(7)),
        ];
        for fmt in formats {
            assert!(
                carried.contains(&fmt.kind()),
                "{fmt:?} reports the carrierless {:?}",
                fmt.kind()
            );
        }
        // Control: the membership test does discriminate, so the sweep
        // above is not passing because `carried` accepts everything.
        for kind in [
            SampleKind::I8,
            SampleKind::I16,
            SampleKind::U32,
            SampleKind::I32,
        ] {
            assert!(!carried.contains(&kind));
        }
    }

    /**
     * Tests that bytes_per_pixel equals channels * bytes_per_channel for every format.
     * Works by iterating all PixelFormat variants and checking the arithmetic identity,
     * catching mismatches if one method is updated without the others.
     * Input: all 6 variants → Output: identity holds for each (e.g. Rgb8: 3 == 3*1).
     */
    #[test]
    fn bytes_per_pixel_matches_channels_times_depth() {
        for fmt in [
            PixelFormat::Gray8,
            PixelFormat::Gray16,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
        ] {
            assert_eq!(
                fmt.bytes_per_pixel(),
                fmt.channels() * fmt.bytes_per_channel(),
                "Mismatch for {fmt:?}"
            );
        }
    }

    /**
     * Tests that with_alpha and without_alpha are inverses of each other.
     * Works by converting non-alpha formats to alpha and back, verifying the
     * original format is recovered.
     * Input: Rgb8 → with_alpha → Rgba8 → without_alpha → Rgb8.
     */
    #[test]
    fn alpha_round_trip() {
        assert_eq!(PixelFormat::Rgb8.with_alpha(), PixelFormat::Rgba8);
        assert_eq!(PixelFormat::Rgba8.without_alpha(), PixelFormat::Rgb8);
        assert_eq!(PixelFormat::Rgb16.with_alpha(), PixelFormat::Rgba16);
        assert_eq!(PixelFormat::Rgba16.without_alpha(), PixelFormat::Rgb16);
    }

    /**
     * Tests that calling with_alpha on formats that already have alpha is a no-op.
     * Works by applying with_alpha to Rgba8/Rgba16 and asserting the result is unchanged.
     * Input: Rgba8.with_alpha() → Output: Rgba8.
     */
    #[test]
    fn with_alpha_is_idempotent() {
        assert_eq!(PixelFormat::Rgba8.with_alpha(), PixelFormat::Rgba8);
        assert_eq!(PixelFormat::Rgba16.with_alpha(), PixelFormat::Rgba16);
    }

    /**
     * Tests that calling without_alpha on formats without alpha is a no-op.
     * Works by applying without_alpha to Rgb8/Gray8 and asserting the result is unchanged.
     * Input: Rgb8.without_alpha() → Output: Rgb8.
     */
    #[test]
    fn without_alpha_is_idempotent() {
        assert_eq!(PixelFormat::Rgb8.without_alpha(), PixelFormat::Rgb8);
        assert_eq!(PixelFormat::Gray8.without_alpha(), PixelFormat::Gray8);
    }

    /**
     * Tests that has_alpha returns true only for Rgba8 and Rgba16.
     * Works by checking every variant and asserting the expected boolean.
     * Input: Gray8→false, Rgb8→false, Rgba8→true, Rgb16→false, Rgba16→true.
     */
    #[test]
    fn has_alpha_correctness() {
        assert!(!PixelFormat::Gray8.has_alpha());
        assert!(!PixelFormat::Rgb8.has_alpha());
        assert!(PixelFormat::Rgba8.has_alpha());
        assert!(!PixelFormat::Gray16.has_alpha());
        assert!(!PixelFormat::Rgb16.has_alpha());
        assert!(PixelFormat::Rgba16.has_alpha());
    }

    /**
     * Tests that with_channels canonicalizes 1/3/4-channel requests to the
     * named variants and everything else to Multi8/Multi16.
     * Works by mapping each (channels, depth) pair and asserting the variant.
     * Input: (3,1)→Rgb8, (2,1)→Multi8(2), (5,2)→Multi16(5), (0,1)→None,
     * (1,3)→None.
     */
    #[test]
    fn with_channels_canonicalizes() {
        assert_eq!(PixelFormat::with_channels(1, 1), Some(PixelFormat::Gray8));
        assert_eq!(PixelFormat::with_channels(1, 2), Some(PixelFormat::Gray16));
        assert_eq!(PixelFormat::with_channels(3, 1), Some(PixelFormat::Rgb8));
        assert_eq!(PixelFormat::with_channels(3, 2), Some(PixelFormat::Rgb16));
        assert_eq!(PixelFormat::with_channels(4, 1), Some(PixelFormat::Rgba8));
        assert_eq!(PixelFormat::with_channels(4, 2), Some(PixelFormat::Rgba16));

        let two = PixelFormat::with_channels(2, 1).unwrap();
        assert_eq!(two, PixelFormat::Multi8(NonZeroU16::new(2).unwrap()));
        let five16 = PixelFormat::with_channels(5, 2).unwrap();
        assert_eq!(five16, PixelFormat::Multi16(NonZeroU16::new(5).unwrap()));

        assert_eq!(PixelFormat::with_channels(0, 1), None);
        assert_eq!(PixelFormat::with_channels(1, 3), None);
        assert_eq!(
            PixelFormat::with_channels(usize::from(u16::MAX) + 1, 1),
            None
        );
    }

    /**
     * Tests that the Multi variants report consistent geometry.
     * Works by constructing Multi8(5) and Multi16(5) and checking channels,
     * bytes_per_channel, and bytes_per_pixel; also checks the alpha helpers
     * treat Multi as alpha-free and identity.
     * Input: Multi8(5) → channels 5, bpp 5; Multi16(5) → bpp 10.
     */
    #[test]
    fn multi_variant_geometry() {
        let m8 = PixelFormat::with_channels(5, 1).unwrap();
        assert_eq!(m8.channels(), 5);
        assert_eq!(m8.bytes_per_channel(), 1);
        assert_eq!(m8.bytes_per_pixel(), 5);
        assert!(!m8.has_alpha());
        assert_eq!(m8.with_alpha(), m8);
        assert_eq!(m8.without_alpha(), m8);

        let m16 = PixelFormat::with_channels(5, 2).unwrap();
        assert_eq!(m16.channels(), 5);
        assert_eq!(m16.bytes_per_channel(), 2);
        assert_eq!(m16.bytes_per_pixel(), 10);
    }

    /**
     * Tests the geometry of the float variants: RgbaF32 is 4 channels at
     * 4 bytes each (16 bytes/pixel) and FloatF32(n) is n channels at
     * 4 bytes each. Works by checking channels, bytes_per_channel, and
     * bytes_per_pixel for RgbaF32, FloatF32(1), and FloatF32(3).
     * Input: RgbaF32 → (4, 4, 16); FloatF32(3) → (3, 4, 12).
     */
    #[test]
    fn float_variant_geometry() {
        let rgba = PixelFormat::RgbaF32;
        assert_eq!(rgba.channels(), 4);
        assert_eq!(rgba.bytes_per_channel(), 4);
        assert_eq!(rgba.bytes_per_pixel(), 16);

        let gray = PixelFormat::FloatF32(NonZeroU16::new(1).unwrap());
        assert_eq!(gray.channels(), 1);
        assert_eq!(gray.bytes_per_channel(), 4);
        assert_eq!(gray.bytes_per_pixel(), 4);

        let rgb = PixelFormat::FloatF32(NonZeroU16::new(3).unwrap());
        assert_eq!(rgb.channels(), 3);
        assert_eq!(rgb.bytes_per_channel(), 4);
        assert_eq!(rgb.bytes_per_pixel(), 12);
    }

    /**
     * Tests that with_channels canonicalizes the 4-byte float depth:
     * 4 bands map to the named RgbaF32, every other count to FloatF32(n),
     * and 0 bands or an unknown depth stay None.
     * Input: (4,4)→RgbaF32, (1,4)→FloatF32(1), (7,4)→FloatF32(7),
     * (0,4)→None, (1,8)→None.
     */
    #[test]
    fn with_channels_canonicalizes_float() {
        assert_eq!(PixelFormat::with_channels(4, 4), Some(PixelFormat::RgbaF32));
        assert_eq!(
            PixelFormat::with_channels(1, 4),
            Some(PixelFormat::FloatF32(NonZeroU16::new(1).unwrap()))
        );
        assert_eq!(
            PixelFormat::with_channels(7, 4),
            Some(PixelFormat::FloatF32(NonZeroU16::new(7).unwrap()))
        );
        assert_eq!(PixelFormat::with_channels(0, 4), None);
        assert_eq!(PixelFormat::with_channels(1, 8), None);
    }

    /**
     * Tests is_float: true for RgbaF32 and FloatF32(n), false for every
     * unsigned variant. Works by checking each variant directly.
     * Input: RgbaF32→true, FloatF32(2)→true, Gray8/Rgba16/Multi16(5)→false.
     */
    #[test]
    fn is_float_correctness() {
        assert!(PixelFormat::RgbaF32.is_float());
        assert!(PixelFormat::FloatF32(NonZeroU16::new(2).unwrap()).is_float());
        assert!(!PixelFormat::Gray8.is_float());
        assert!(!PixelFormat::Gray16.is_float());
        assert!(!PixelFormat::Rgb8.is_float());
        assert!(!PixelFormat::Rgba8.is_float());
        assert!(!PixelFormat::Rgb16.is_float());
        assert!(!PixelFormat::Rgba16.is_float());
        assert!(!PixelFormat::with_channels(5, 1).unwrap().is_float());
        assert!(!PixelFormat::with_channels(5, 2).unwrap().is_float());
    }

    /**
     * Tests the float alpha helpers: RgbaF32 has alpha and demotes to
     * FloatF32(3); FloatF32(1) and FloatF32(3) promote to RgbaF32; other
     * float band counts are alpha-free and unchanged (like Multi).
     * Input: RgbaF32.without_alpha()→FloatF32(3); FloatF32(3).with_alpha()
     * →RgbaF32; FloatF32(2).with_alpha()→FloatF32(2).
     */
    #[test]
    fn float_alpha_helpers() {
        let f1 = PixelFormat::FloatF32(NonZeroU16::new(1).unwrap());
        let f2 = PixelFormat::FloatF32(NonZeroU16::new(2).unwrap());
        let f3 = PixelFormat::FloatF32(NonZeroU16::new(3).unwrap());

        assert!(PixelFormat::RgbaF32.has_alpha());
        assert!(!f1.has_alpha());
        assert!(!f3.has_alpha());

        assert_eq!(f1.with_alpha(), PixelFormat::RgbaF32);
        assert_eq!(f3.with_alpha(), PixelFormat::RgbaF32);
        assert_eq!(f2.with_alpha(), f2);
        assert_eq!(PixelFormat::RgbaF32.with_alpha(), PixelFormat::RgbaF32);

        assert_eq!(PixelFormat::RgbaF32.without_alpha(), f3);
        assert_eq!(f3.without_alpha(), f3);
        assert_eq!(f2.without_alpha(), f2);
    }

    /// Every `(band count, byte depth)` pair that has both a tuple spelling
    /// and a named one, paired with the named variant `with_channels`
    /// produces for it. Direct construction of the left-hand column is what
    /// issue #531 is about: the tuple variants are public, so both spellings
    /// of one pixel layout are constructible.
    fn alias_table() -> [(PixelFormat, PixelFormat); 7] {
        let nz = |n: u16| NonZeroU16::new(n).expect("the table holds no zeroes");
        [
            (PixelFormat::Multi8(nz(1)), PixelFormat::Gray8),
            (PixelFormat::Multi8(nz(3)), PixelFormat::Rgb8),
            (PixelFormat::Multi8(nz(4)), PixelFormat::Rgba8),
            (PixelFormat::Multi16(nz(1)), PixelFormat::Gray16),
            (PixelFormat::Multi16(nz(3)), PixelFormat::Rgb16),
            (PixelFormat::Multi16(nz(4)), PixelFormat::Rgba16),
            (PixelFormat::FloatF32(nz(4)), PixelFormat::RgbaF32),
        ]
    }

    /**
     * Tests the exact disagreement issue #531 reproduces: FloatF32(4) is a
     * constructible second spelling of the layout RgbaF32 names, and the two
     * must not answer differently about that layout.
     * Works by asking both spellings every question PixelFormat can be
     * asked and comparing the answers pairwise, so a future accessor that
     * forgets the alias fails here rather than at a call site.
     * Input: FloatF32(4) vs RgbaF32 -> Output: 4 channels, 4 bytes each,
     * 16 bytes per pixel, float, has_alpha true, with_alpha RgbaF32,
     * without_alpha FloatF32(3), for both.
     */
    #[test]
    fn floatf32_4_and_rgbaf32_answer_alike() {
        let alias = PixelFormat::FloatF32(NonZeroU16::new(4).expect("4 is non-zero"));
        let named = PixelFormat::RgbaF32;

        assert_eq!(alias.channels(), named.channels(), "channels disagree");
        assert_eq!(
            alias.bytes_per_channel(),
            named.bytes_per_channel(),
            "bytes_per_channel disagree"
        );
        assert_eq!(
            alias.bytes_per_pixel(),
            named.bytes_per_pixel(),
            "bytes_per_pixel disagree"
        );
        assert_eq!(alias.is_float(), named.is_float(), "is_float disagrees");
        assert_eq!(
            alias.has_alpha(),
            named.has_alpha(),
            "has_alpha disagrees: FloatF32(4) says {} and RgbaF32 says {}",
            alias.has_alpha(),
            named.has_alpha()
        );
        assert_eq!(
            alias.with_alpha(),
            named.with_alpha(),
            "with_alpha disagrees"
        );
        assert_eq!(
            alias.without_alpha(),
            named.without_alpha(),
            "without_alpha disagrees"
        );
    }

    /**
     * Tests that no tuple spelling of a layout behaves differently from the
     * named variant with_channels canonicalizes it to, for any band count
     * a named variant exists for.
     * Works by sweeping band counts 1 to 8 across all three byte depths,
     * building the tuple variant directly and the canonical one through
     * with_channels, and comparing every accessor. Counts with no named
     * variant (2, 5, 6, 7, 8, and 1/3 at the float depth) compare a value
     * against itself and hold trivially, which is what makes the sweep safe
     * to state over the whole range.
     * Input: Multi8(4) vs Rgba8, Multi16(1) vs Gray16, FloatF32(4) vs
     * RgbaF32, ... -> Output: identical answers in every row.
     */
    #[test]
    fn every_tuple_spelling_behaves_like_its_canonical_form() {
        for n in 1..=8u16 {
            let nz = NonZeroU16::new(n).expect("n starts at 1");
            for (tuple, depth) in [
                (PixelFormat::Multi8(nz), 1usize),
                (PixelFormat::Multi16(nz), 2),
                (PixelFormat::FloatF32(nz), 4),
            ] {
                let named = PixelFormat::with_channels(usize::from(n), depth)
                    .expect("1..=8 bands at depth 1/2/4 is a valid format");
                assert_eq!(
                    tuple.channels(),
                    named.channels(),
                    "{tuple:?} and {named:?} disagree on channels"
                );
                assert_eq!(
                    tuple.bytes_per_channel(),
                    named.bytes_per_channel(),
                    "{tuple:?} and {named:?} disagree on bytes_per_channel"
                );
                assert_eq!(
                    tuple.bytes_per_pixel(),
                    named.bytes_per_pixel(),
                    "{tuple:?} and {named:?} disagree on bytes_per_pixel"
                );
                assert_eq!(
                    tuple.is_float(),
                    named.is_float(),
                    "{tuple:?} and {named:?} disagree on is_float"
                );
                assert_eq!(
                    tuple.has_alpha(),
                    named.has_alpha(),
                    "{tuple:?} and {named:?} disagree on has_alpha"
                );
                assert_eq!(
                    tuple.with_alpha(),
                    named.with_alpha(),
                    "{tuple:?} and {named:?} disagree on with_alpha"
                );
                assert_eq!(
                    tuple.without_alpha(),
                    named.without_alpha(),
                    "{tuple:?} and {named:?} disagree on without_alpha"
                );
            }
        }
    }

    /**
     * Tests that the alpha helpers land on the named variant rather than on
     * the tuple spelling of the same layout, so a promotion or demotion
     * cannot introduce an alias that was not there before.
     * Works by promoting and demoting every row of the alias table and
     * asserting the result is the named variant's answer, by value.
     * Input: Multi8(1).with_alpha() -> Rgba8; Multi8(4).without_alpha() ->
     * Rgb8; FloatF32(4).without_alpha() -> FloatF32(3).
     */
    #[test]
    fn alpha_helpers_land_on_the_named_variant() {
        for (alias, named) in alias_table() {
            assert_eq!(
                alias.with_alpha(),
                named.with_alpha(),
                "{alias:?}.with_alpha() must match {named:?}.with_alpha()"
            );
            assert_eq!(
                alias.without_alpha(),
                named.without_alpha(),
                "{alias:?}.without_alpha() must match {named:?}.without_alpha()"
            );
        }
    }

    /**
     * Tests that canonical maps every non-canonical spelling to its named
     * variant and leaves everything else alone, which is the table issue
     * #531 enumerates.
     * Works by asserting the seven rows by value, then asserting canonical
     * is the identity on all seven named variants and on the tuple
     * spellings that have no named twin, and that is_canonical agrees with
     * it everywhere.
     * Input: FloatF32(4) -> RgbaF32, Multi8(3) -> Rgb8, ...; FloatF32(3),
     * Multi8(2), Rgb8 -> unchanged.
     */
    #[test]
    fn canonical_maps_the_alias_table_and_nothing_else() {
        let nz = |n: u16| NonZeroU16::new(n).expect("the table holds no zeroes");

        for (alias, named) in alias_table() {
            assert_eq!(
                alias.canonical(),
                named,
                "{alias:?} must canonicalize to {named:?}"
            );
            assert!(
                !alias.is_canonical(),
                "{alias:?} is not the canonical spelling of its layout"
            );
            assert!(named.is_canonical(), "{named:?} is canonical");
        }

        // The named variants and the tuple spellings with no named twin are
        // fixed points. FloatF32(1) and FloatF32(3) are in this list on
        // purpose: four is the only float band count with a named variant,
        // so the float row of the table has one entry where the 8- and
        // 16-bit rows have three.
        for fmt in [
            PixelFormat::Gray8,
            PixelFormat::Gray16,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
            PixelFormat::RgbaF32,
            PixelFormat::Multi8(nz(2)),
            PixelFormat::Multi8(nz(7)),
            PixelFormat::Multi16(nz(2)),
            PixelFormat::Multi16(nz(5)),
            PixelFormat::FloatF32(nz(1)),
            PixelFormat::FloatF32(nz(3)),
            PixelFormat::FloatF32(nz(7)),
        ] {
            assert_eq!(
                fmt.canonical(),
                fmt,
                "{fmt:?} is already canonical and must be left alone"
            );
            assert!(fmt.is_canonical(), "{fmt:?} must report as canonical");
        }
    }

    /**
     * Tests exactly which band counts have a named variant, which is what
     * decides the shape of the alias table: a count with a named variant is
     * one where a tuple spelling of the same layout also exists.
     * Works by sweeping band counts 1 to 300 across all three byte depths
     * and asserting with_channels lands on a named variant for precisely
     * 1, 3 and 4 at the 8- and 16-bit depths and precisely 4 at the float
     * depth, and on a tuple variant everywhere else. Asserting
     * `is_canonical()` here instead would pin nothing: `canonical` is
     * defined as `with_channels`, so it holds for any implementation.
     * Input: (1..=300) x {1, 2, 4} -> Output: named at (1|3|4, 1|2) and
     * (4, 4), tuple elsewhere.
     */
    #[test]
    fn with_channels_uses_a_named_variant_for_exactly_the_aliased_counts() {
        for n in 1..=300usize {
            for depth in [1usize, 2, 4] {
                let fmt = PixelFormat::with_channels(n, depth)
                    .expect("1..=300 bands at depth 1/2/4 is a valid format");
                let is_named = matches!(
                    fmt,
                    PixelFormat::Gray8
                        | PixelFormat::Gray16
                        | PixelFormat::Rgb8
                        | PixelFormat::Rgba8
                        | PixelFormat::Rgb16
                        | PixelFormat::Rgba16
                        | PixelFormat::RgbaF32
                );
                let wants_named = if depth == 4 {
                    n == 4
                } else {
                    n == 1 || n == 3 || n == 4
                };
                assert_eq!(
                    is_named,
                    wants_named,
                    "with_channels({n}, {depth}) produced {fmt:?}, which is \
                     {}a named variant",
                    if is_named { "" } else { "not " }
                );
                assert!(
                    fmt.is_canonical(),
                    "with_channels({n}, {depth}) produced the non-canonical {fmt:?}"
                );
            }
        }
    }
}
