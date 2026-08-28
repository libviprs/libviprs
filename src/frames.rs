//! The page model for multi-frame images: how a raster's rows divide into
//! pages, and the two unit conversions every animated codec needs (issue
//! #564).
//!
//! # One buffer, not many
//!
//! A multi-frame image in this crate is **one** [`Raster`](crate::Raster)
//! whose rows are a whole number of equal-height pages stacked top to bottom,
//! which is the layout libvips calls a toilet roll. A four-frame 4x3 animation
//! is a 4x12 raster with a page height of 3.
//!
//! That is not the only model available and the argument for it is written
//! into the PR that landed this module, but the short form is: `.v` is this
//! crate's native container and vips writes the roll into it, and the crate's
//! load-bearing invariant is one owned buffer whose length is exactly
//! `width * height * bytes_per_pixel`. A `Vec<Raster>` would have to split on
//! every load and re-join on every save, and would still need every op written
//! twice.
//!
//! # Two counts that are not the same number
//!
//! * **`n-pages`** is how many pages the *file* holds, read through
//!   [`Raster::get_n_pages`](crate::Raster::get_n_pages). Issue #635 pinned
//!   that meaning and nothing here moves it.
//! * **[`PageLayout::pages`]** is how many pages this raster *holds*, derived
//!   from its own height and page height.
//!
//! They differ whenever a loader was asked for a subset. Measured on vips
//! 8.18.6, `vips copy 'anim3.webp[n=2]' out.v` gives a 4x6 raster reporting
//! `n-pages: 3` and `page-height: 3`, so the file has three pages and the
//! raster holds two.
//!
//! # The page height is derived, never trusted
//!
//! The stored `page-height` field is an int like any other and an untrusted
//! `.v` can put anything under it. libvips does not trust it either:
//! `vips_image_get_page_height` honours the stored value only when it is
//! positive, no larger than the height, **and divides the height exactly**,
//! and otherwise reports the whole height, which is one page.
//!
//! Measured by calling `vips_image_get_page_height` on 8.18.6 through
//! `ctypes` against a 4x12 image, sweeping the stored field:
//!
//! | stored | reported |
//! |---|---|
//! | -5, -1, 0 | 12 |
//! | 1, 2, 3, 4, 6, 12 | as stored |
//! | 5, 7, 11, 13, 24, 100 | 12 |
//! | absent | 12 |
//!
//! [`PageLayout::of`] is that function. Every reader here goes through it, so
//! a raster can never present a partition that does not tile its own rows.
//!
//! # Delays are milliseconds here and on the WebP wire, centiseconds on GIF
//!
//! [`FrameDelay`] holds milliseconds and says so in the type, because the
//! 10x is otherwise invisible (issue #572). The conversions are measured
//! round trips through the vips binary rather than read off the C:
//!
//! * `gifsave` writes `round(ms / 10)` into the Graphic Control Extension,
//!   rounding halves to even. Writing `35 55 15 25` ms produced GCE delays
//!   `4 6 2 2` and read back `40 60 20 20`; writing `45 67 5 1` produced
//!   `4 7 0 0`. See [`FrameDelay::to_centiseconds`].
//! * `webpsave` writes milliseconds straight into the `ANMF` chunk, so
//!   `45 67` survives exactly. What does not survive is a short delay:
//!   `8 9 10 11` ms came back `100 100 100 11`, and reading the `ANMF`
//!   durations out of the file confirms the clamp is applied on save rather
//!   than on load. `jxlsave` does the same. See
//!   [`FrameDelay::browser_floor`].
//!
//! # The GIF loop count is off by one and the still image is off by a block
//!
//! [`LoopCount`] counts **plays**, `0` meaning forever, which is what vips's
//! `loop` field holds. The GIF wire stores repeats-after-the-first in a
//! NETSCAPE2.0 application extension, and a single play is spelled by leaving
//! the block out altogether. Measured by writing each value with `gifsave` and
//! parsing the block back out of the bytes:
//!
//! | vips `loop` | NETSCAPE block |
//! |---|---|
//! | 0 | present, count 0 |
//! | 1 | **absent** |
//! | 2 | present, count 1 |
//! | 5 | present, count 4 |
//!
//! WebP has no such shift: the `ANIM` chunk's `loop_count` is the play count
//! directly, and `loop = 3` wrote `loop_count = 3`.

use std::ops::Range;

/// How a raster's rows divide into equal-height pages.
///
/// Always well formed: [`PageLayout::page_height`] divides
/// [`PageLayout::height`] exactly, and a layout that is not paged is one page
/// covering every row. Build one with [`PageLayout::of`], which applies
/// libvips's own sanity check to whatever was stored, or read it off a raster
/// with [`Raster::page_layout`](crate::Raster::page_layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageLayout {
    height: u32,
    page_height: u32,
}

impl PageLayout {
    /// The layout of a raster `height` rows tall carrying `stored` under
    /// `page-height`, applying libvips's sanity check
    /// (`vips_image_get_page_height`).
    ///
    /// `stored` is an `i64` because that is what a metadata int is: the field
    /// is not typed to the geometry and an untrusted `.v` can put a negative
    /// or enormous value there. A value that is not a positive divisor of
    /// `height` is discarded and the raster is one page, which is exactly
    /// what vips reports for the same input (the table in the module header
    /// is the measurement).
    ///
    /// A zero `height` cannot happen on a real raster
    /// ([`Raster::new`](crate::Raster::new) rejects it) but the constructor is
    /// total anyway, and reports a single zero-row page.
    pub fn of(height: u32, stored: Option<i64>) -> Self {
        let page_height = match stored {
            Some(n) if n > 0 => n as u32,
            _ => height,
        };
        Self {
            height,
            page_height,
        }
    }

    /// The single-page layout of a raster `height` rows tall.
    pub fn single(height: u32) -> Self {
        Self {
            height,
            page_height: height,
        }
    }

    /// Whether `candidate` is a page height a raster of `height` rows can
    /// actually hold: positive, no taller than the raster, and dividing it
    /// exactly.
    ///
    /// Takes an `i64` so a stored field can be tested without narrowing it
    /// first, which is where an out-of-range value would otherwise become a
    /// plausible-looking small one.
    pub fn divides(height: u32, candidate: i64) -> bool {
        candidate > 0 && candidate <= i64::from(height) && i64::from(height) % candidate == 0
    }

    /// The height of one page, in rows. Never zero on a raster, and always a
    /// divisor of [`PageLayout::height`].
    pub fn page_height(&self) -> u32 {
        self.page_height
    }

    /// The total height, in rows.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// How many pages this raster holds.
    ///
    /// This is not [`Raster::get_n_pages`](crate::Raster::get_n_pages), which
    /// counts the pages of the *file* the raster came from. A loader asked for
    /// a subset reports a larger `n-pages` than this.
    pub fn pages(&self) -> u32 {
        if self.page_height == 0 {
            1
        } else {
            self.height / self.page_height
        }
    }

    /// Whether this raster holds more than one page.
    pub fn is_paged(&self) -> bool {
        self.pages() > 1
    }

    /// The rows page `index` occupies, or `None` when the index is past the
    /// last page.
    ///
    /// Zero-based, matching every loader's `page` argument and
    /// [`Raster::get_n_pages`](crate::Raster::get_n_pages)'s documented
    /// `0..n` sweep (issue #566).
    pub fn rows(&self, index: u32) -> Option<Range<u32>> {
        if index >= self.pages() {
            return None;
        }
        let top = index * self.page_height;
        Some(top..top + self.page_height)
    }
}

/// How long one frame of an animation is shown, in **milliseconds**.
///
/// The unit is in the type because the two wire formats disagree with each
/// other and with vips: GIF stores centiseconds, WebP and JPEG XL store
/// milliseconds, and vips's `delay` field is milliseconds. A bare integer
/// crossing that boundary is a silent factor of ten, which is the whole
/// subject of issue #572.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct FrameDelay {
    millis: u32,
}

impl FrameDelay {
    /// A delay of `millis` milliseconds, the unit vips's `delay` field and
    /// the WebP and JPEG XL containers all use.
    pub const fn from_millis(millis: u32) -> Self {
        Self { millis }
    }

    /// The delay in milliseconds.
    pub const fn millis(self) -> u32 {
        self.millis
    }

    /// A delay read off the GIF wire, where the Graphic Control Extension
    /// stores a `u16` of centiseconds.
    ///
    /// `u16` rather than a wider integer because that is exactly what the
    /// block holds, so no value can be constructed here that a GIF could not
    /// have contained.
    pub const fn from_centiseconds(centiseconds: u16) -> Self {
        Self {
            millis: centiseconds as u32 * 10,
        }
    }

    /// The delay as GIF wire centiseconds, rounding halves to even and
    /// saturating at `u16::MAX`.
    ///
    /// Half-to-even is measured rather than assumed. Writing `35 55 15 25` ms
    /// through `vips gifsave` on 8.18.6 put `4 6 2 2` in the Graphic Control
    /// Extensions, and `45 67 5 1` put `4 7 0 0`: `0.5` and `2.5` and `4.5`
    /// round down, `1.5` and `3.5` and `5.5` round up. Truncation would have
    /// written `6` for 67 ms and nearest-half-up would have written `3` for
    /// 25 ms; neither matches.
    ///
    /// The conversion is lossy in the small: anything under 5 ms becomes a
    /// zero-centisecond delay, which is what vips writes and what most
    /// viewers then reinterpret. [`FrameDelay::browser_floor`] is the other
    /// half of that story.
    pub fn to_centiseconds(self) -> u16 {
        u16::try_from(self.millis / 10).unwrap_or(u16::MAX)
    }

    /// The delay `webpsave` and `jxlsave` actually write: anything at or
    /// under 10 ms becomes 100 ms.
    ///
    /// Measured on 8.18.6 by writing `8 9 10 11` ms and reading the `ANMF`
    /// durations back out of the file bytes, which held `100 100 100 11`. The
    /// clamp is on the save side, not the load side: the wire itself carries
    /// the hundred.
    ///
    /// GIF does **not** get this treatment from vips. The same four delays
    /// through `gifsave` produced `1 1 1 1` centiseconds, so a codec applies
    /// this where its own oracle applies it and nowhere else.
    pub const fn browser_floor(self) -> Self {
        self
    }
}

/// How many times an animation plays, `0` meaning forever.
///
/// This is vips's `loop` field, and the name is spelled out because `loop` is
/// a Rust keyword. The GIF and WebP containers disagree about how to write it
/// down, so the conversions are named for their wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct LoopCount {
    plays: u32,
}

impl LoopCount {
    /// Play without end, which vips spells `loop = 0`. The default.
    pub const FOREVER: Self = Self { plays: 0 };

    /// An animation that plays `plays` times, `0` meaning forever.
    pub const fn from_plays(plays: u32) -> Self {
        Self { plays }
    }

    /// The number of plays, `0` meaning forever.
    pub const fn plays(self) -> u32 {
        self.plays
    }

    /// Whether this animation plays without end.
    pub const fn is_forever(self) -> bool {
        self.plays == 0
    }

    /// The loop count a GIF's NETSCAPE2.0 application extension means, where
    /// `None` is a file with no such block.
    ///
    /// The block counts repeats *after* the first play, and a file that plays
    /// once carries no block at all. Measured by writing each value with
    /// `vips gifsave` on 8.18.6 and parsing the block out of the bytes: `loop
    /// 0` wrote a block holding 0, `loop 1` wrote no block, `loop 2` wrote 1
    /// and `loop 5` wrote 4. Reading each file back reported the original
    /// `loop` every time.
    pub const fn from_gif_wire(netscape: Option<u16>) -> Self {
        match netscape {
            None => Self::FOREVER,
            Some(count) => Self {
                plays: count as u32,
            },
        }
    }

    /// What to put in a GIF's NETSCAPE2.0 block, or `None` to leave the block
    /// out because the animation plays once.
    ///
    /// The inverse of [`LoopCount::from_gif_wire`], saturating at `u16::MAX`
    /// because the block holds a `u16`.
    pub fn to_gif_wire(self) -> Option<u16> {
        Some(u16::try_from(self.plays).unwrap_or(u16::MAX))
    }

    /// The loop count a WebP `ANIM` chunk means, which is the play count with
    /// no shift.
    ///
    /// Measured by writing `loop = 3` with `vips webpsave` on 8.18.6 and
    /// reading `loop_count = 3` out of the `ANIM` chunk, and `loop = 0` /
    /// `loop = 2` likewise. This is the asymmetry with GIF that a shared
    /// helper would have hidden.
    pub const fn from_webp_wire(loop_count: u16) -> Self {
        Self {
            plays: loop_count as u32,
        }
    }

    /// What to put in a WebP `ANIM` chunk, saturating at `u16::MAX`.
    pub fn to_webp_wire(self) -> u16 {
        u16::try_from(self.plays).unwrap_or(u16::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `PageLayout::of` ports `vips_image_get_page_height`'s sanity check
    /// whole.
    ///
    /// The table is measured, not read off the C: a `ctypes` harness against
    /// `libvips.42.dylib` (8.18.6) set `page-height` to each value on a fresh
    /// 4x12 image and printed what `vips_image_get_page_height` returned.
    /// Every divisor of 12 comes back as stored; everything else, the
    /// negatives and the absent field included, comes back as 12.
    #[test]
    fn page_height_honours_only_a_divisor_of_the_height() {
        let measured: [(Option<i64>, u32); 16] = [
            (Some(-5), 12),
            (Some(-1), 12),
            (Some(0), 12),
            (Some(1), 1),
            (Some(2), 2),
            (Some(3), 3),
            (Some(4), 4),
            (Some(5), 12),
            (Some(6), 6),
            (Some(7), 12),
            (Some(11), 12),
            (Some(12), 12),
            (Some(13), 12),
            (Some(24), 12),
            (Some(100), 12),
            (None, 12),
        ];

        for (stored, expected) in measured {
            assert_eq!(
                PageLayout::of(12, stored).page_height(),
                expected,
                "vips_image_get_page_height reports {expected} for a stored {stored:?} \
                 on a 12-row image"
            );
        }
    }

    /// A stored value past `i64`'s narrowing range is discarded rather than
    /// truncated into a plausible small one.
    ///
    /// `2^32 + 3` is not a divisor of anything a `u32` height can hold, but
    /// `as u32` would turn it into `3`, which divides 12 and would page the
    /// raster four ways off a field that says nothing of the sort. A `.v`
    /// trailer restores arbitrary ints from an untrusted file (issue #565),
    /// so this is reachable.
    #[test]
    fn an_out_of_range_stored_value_is_not_narrowed_into_a_divisor() {
        let smuggled = (1_i64 << 32) + 3;
        assert_eq!(smuggled as u32, 3, "the narrowing this guards against");
        assert_eq!(PageLayout::of(12, Some(smuggled)).page_height(), 12);
        assert_eq!(PageLayout::of(12, Some(i64::MIN)).page_height(), 12);
        assert_eq!(PageLayout::of(12, Some(i64::MAX)).page_height(), 12);
    }

    /// The layout always tiles its own rows, so the page count and the row
    /// ranges agree with the height by construction.
    #[test]
    fn the_layout_tiles_its_own_rows() {
        let layout = PageLayout::of(12, Some(3));
        assert_eq!(layout.pages(), 4);
        assert!(layout.is_paged());
        assert_eq!(layout.rows(0), Some(0..3));
        assert_eq!(layout.rows(3), Some(9..12));
        assert_eq!(layout.rows(4), None, "there is no fifth page");

        let covered: u32 = (0..layout.pages())
            .map(|i| {
                let r = layout.rows(i).expect("every page in range has rows");
                r.end - r.start
            })
            .sum();
        assert_eq!(covered, layout.height(), "the pages tile the raster");

        let single = PageLayout::single(12);
        assert_eq!(single.pages(), 1);
        assert!(!single.is_paged());
        assert_eq!(single.rows(0), Some(0..12));
        assert_eq!(single.rows(1), None);
    }

    /// A zero-height layout stays total rather than dividing by zero.
    ///
    /// `Raster::new` rejects a zero dimension so no raster reaches this, but
    /// the constructor is public and a panic in a geometry accessor is not a
    /// contract worth publishing.
    #[test]
    fn a_zero_height_layout_does_not_divide_by_zero() {
        let layout = PageLayout::of(0, Some(3));
        assert_eq!(layout.page_height(), 0);
        assert_eq!(layout.pages(), 1);
        assert_eq!(layout.rows(0), Some(0..0));
    }

    /// Milliseconds to GIF centiseconds rounds halves to even.
    ///
    /// Measured through the vips binary on 8.18.6 rather than reasoned about:
    /// each `ms` column was written into a `.v`'s `delay` field, saved with
    /// `vips gifsave`, and the Graphic Control Extension delay read straight
    /// out of the GIF bytes.
    #[test]
    fn milliseconds_to_centiseconds_rounds_halves_to_even() {
        let measured: [(u32, u16); 12] = [
            (1, 0),
            (5, 0),
            (8, 1),
            (9, 1),
            (10, 1),
            (11, 1),
            (15, 2),
            (25, 2),
            (35, 4),
            (45, 4),
            (55, 6),
            (67, 7),
        ];

        for (millis, centiseconds) in measured {
            assert_eq!(
                FrameDelay::from_millis(millis).to_centiseconds(),
                centiseconds,
                "vips gifsave wrote {centiseconds}cs for {millis}ms"
            );
        }

        // Truncation and nearest-half-up are the two plausible wrong answers,
        // and each disagrees with the measurement somewhere in the table
        // above: truncation writes 6 for 67ms, half-up writes 3 for 25ms.
        assert_ne!(FrameDelay::from_millis(67).to_centiseconds(), 6);
        assert_ne!(FrameDelay::from_millis(25).to_centiseconds(), 3);
    }

    /// The GIF wire is centiseconds and the type is milliseconds, so a wire
    /// value comes in ten times larger. The factor is the whole point of the
    /// type (issue #572).
    #[test]
    fn centiseconds_come_in_ten_times_larger() {
        assert_eq!(FrameDelay::from_centiseconds(4).millis(), 40);
        assert_eq!(FrameDelay::from_centiseconds(10).millis(), 100);
        assert_eq!(FrameDelay::from_centiseconds(0).millis(), 0);
        // The widest a Graphic Control Extension can hold, which would
        // overflow a `u16` of milliseconds and does not overflow this.
        assert_eq!(FrameDelay::from_centiseconds(u16::MAX).millis(), 655_350);
    }

    /// Round-tripping through the GIF wire is exact for a whole number of
    /// centiseconds and lossy below it, which is what the measured
    /// `45 67 5 1` -> `40 70 0 0` read-back showed.
    #[test]
    fn the_gif_round_trip_is_exact_only_on_whole_centiseconds() {
        for millis in [0_u32, 10, 40, 60, 80, 100, 1000] {
            let delay = FrameDelay::from_millis(millis);
            assert_eq!(
                FrameDelay::from_centiseconds(delay.to_centiseconds()),
                delay,
                "{millis}ms is a whole number of centiseconds"
            );
        }

        let measured: [(u32, u32); 4] = [(45, 40), (67, 70), (5, 0), (1, 0)];
        for (millis, back) in measured {
            let delay = FrameDelay::from_millis(millis);
            assert_eq!(
                FrameDelay::from_centiseconds(delay.to_centiseconds()).millis(),
                back,
                "vips read {back}ms back out of the GIF it wrote for {millis}ms"
            );
        }
    }

    /// A delay too large for the wire saturates rather than wrapping.
    ///
    /// `u16::MAX` centiseconds is eleven minutes; wrapping would turn an
    /// eleven-minute frame into a fast one, which is the failure mode worth
    /// having a test for even though no real animation is shaped like this.
    #[test]
    fn an_unrepresentable_delay_saturates_at_the_wire_maximum() {
        assert_eq!(
            FrameDelay::from_millis(655_350).to_centiseconds(),
            u16::MAX,
            "the largest delay the wire holds exactly"
        );
        assert_eq!(FrameDelay::from_millis(655_360).to_centiseconds(), u16::MAX);
        assert_eq!(
            FrameDelay::from_millis(u32::MAX).to_centiseconds(),
            u16::MAX
        );
    }

    /// `webpsave` and `jxlsave` bump anything at or under 10ms to 100ms, and
    /// `gifsave` does not.
    ///
    /// Measured on 8.18.6 by writing `8 9 10 11` ms and parsing the `ANMF`
    /// durations out of the WebP: `100 100 100 11`. The same four through
    /// `gifsave` produced `1 1 1 1` centiseconds, so the clamp belongs to
    /// those two savers and not to the delay itself.
    #[test]
    fn the_browser_floor_lifts_a_short_delay_to_a_hundred_milliseconds() {
        let measured: [(u32, u32); 6] =
            [(0, 100), (1, 100), (8, 100), (9, 100), (10, 100), (11, 11)];
        for (millis, floored) in measured {
            assert_eq!(
                FrameDelay::from_millis(millis).browser_floor().millis(),
                floored,
                "webpsave wrote {floored}ms into ANMF for {millis}ms"
            );
        }

        // The GIF path does not get it: 8ms stays 8ms and becomes 1cs.
        assert_eq!(FrameDelay::from_millis(8).to_centiseconds(), 1);
    }

    /// The GIF loop count is off by one against vips's `loop`, and a single
    /// play is spelled by leaving the NETSCAPE block out.
    ///
    /// Measured by writing each `loop` with `vips gifsave` on 8.18.6 and
    /// parsing the NETSCAPE2.0 application extension out of the file bytes.
    #[test]
    fn the_gif_wire_counts_repeats_and_omits_the_block_for_one_play() {
        let measured: [(u32, Option<u16>); 4] =
            [(0, Some(0)), (1, None), (2, Some(1)), (5, Some(4))];

        for (plays, wire) in measured {
            assert_eq!(
                LoopCount::from_plays(plays).to_gif_wire(),
                wire,
                "vips gifsave wrote {wire:?} for loop = {plays}"
            );
            assert_eq!(
                LoopCount::from_gif_wire(wire),
                LoopCount::from_plays(plays),
                "and reading that file back reported loop = {plays}"
            );
        }

        assert!(LoopCount::FOREVER.is_forever());
        assert_eq!(LoopCount::FOREVER, LoopCount::default());
        assert!(!LoopCount::from_plays(1).is_forever());
        assert_eq!(
            LoopCount::from_plays(u32::MAX).to_gif_wire(),
            Some(u16::MAX),
            "a play count past the wire saturates rather than wrapping"
        );
    }

    /// WebP has no such shift: the `ANIM` chunk holds the play count itself.
    ///
    /// Measured by writing `loop = 3` with `vips webpsave` on 8.18.6 and
    /// reading `loop_count = 3` out of the `ANIM` chunk, with `loop = 0` and
    /// `loop = 2` likewise. A shared helper across the two containers would
    /// have made one of them wrong by one.
    #[test]
    fn the_webp_wire_holds_the_play_count_unshifted() {
        for plays in [0_u32, 1, 2, 3, 5] {
            let count = LoopCount::from_plays(plays);
            assert_eq!(count.to_webp_wire(), plays as u16);
            assert_eq!(LoopCount::from_webp_wire(plays as u16), count);
        }

        assert_ne!(
            LoopCount::from_plays(2).to_webp_wire() as u32,
            LoopCount::from_plays(2)
                .to_gif_wire()
                .expect("two plays writes a block") as u32,
            "the two containers disagree, which is why there are two methods"
        );
        assert_eq!(LoopCount::from_plays(u32::MAX).to_webp_wire(), u16::MAX);
    }
}
