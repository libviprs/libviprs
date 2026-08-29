//! One dialect for every multi-frame container (issue #564).
//!
//! The page model exists so that a multi-frame image has **one** shape in this
//! crate whatever container it came out of. Three loaders reach it today, and
//! each was measured against its own vips loader on its own fixtures inside its
//! own module. That is how the animated WebP and JPEG XL work landed, and it is
//! a suite of three separate agreements rather than one agreement: nothing in
//! the tree loads the same animation through all three and asks whether the
//! answers match.
//!
//! They already do not, and this file is what found it. The two deprecated
//! compatibility names diverge three ways: the GIF loader attaches neither of
//! them where `gifload` attaches both (issue #865), the WebP loader attaches
//! both, and the JPEG XL loader attaches one, the two of them matching their
//! oracles exactly. So the model's own consumers had drifted apart on the field
//! set, and nothing was going to notice.
//!
//! # The fixture
//!
//! One four-frame 4x3 animation, three containers. The GIF is written by hand:
//! four solid frames of black, red, green and blue, graphic control extensions
//! carrying `4 6 8 10` centiseconds, and a NETSCAPE2.0 block holding a repeat
//! count of 3. The WebP and the JPEG XL are `vips copy 'anim4.gif[n=-1]'` on
//! 8.18.6 with `strip` and `lossless`, so all three hold the same four frames
//! and the same timings and nothing but the container changes.
//!
//! Embedded as bytes rather than read from `oracle-captures/`, so no test here
//! touches the filesystem, none needs a Miri annotation, and none moves the
//! pinned filesystem-test count.
//!
//! # A measurement trap worth keeping
//!
//! **`vipsheader -a` is a broken probe for `gif-delay` and `gif-loop`.** It
//! lists neither on any file, on any loader, animated or not, while
//! `vipsheader -f gif-loop` returns the value. Reading the absence in `-a` as
//! "vips does not attach it" produces the exact inverse of the truth, which is
//! what nearly happened here. Every number in the tables below came from `-f`.
//!
//! # What holds a fourth loader
//!
//! This file pins the three that exist. What catches a fourth arriving with a
//! fourth dialect is `only_the_three_animated_loaders_declare_a_page_split` in
//! `tests/page_model.rs`, which scans `src/` for callers of
//! `Raster::try_set_page_height` and fails on one that is not listed here.

use libviprs::source::DecodeLimits;
use libviprs::{Raster, decode_gif_with, decode_webp_with, gif, webp};

#[cfg(feature = "jxl")]
use libviprs::{decode_jxl_with, jxl};

// ---------------------------------------------------------------------------
// The one animation, in three containers
// ---------------------------------------------------------------------------

const ANIM4_GIF: [u8; 197] = [
    0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x04, 0x00, 0x03, 0x00, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0xff, 0x0b, 0x4e, 0x45, 0x54, 0x53,
    0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30, 0x03, 0x01, 0x03, 0x00, 0x00, 0x21, 0xf9, 0x04, 0x04,
    0x04, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x03, 0x00, 0x00, 0x08, 0x08,
    0x00, 0x01, 0x08, 0x1c, 0x48, 0x50, 0x60, 0x40, 0x00, 0x21, 0xf9, 0x04, 0x05, 0x06, 0x00, 0x01,
    0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x03, 0x00, 0x81, 0xff, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x08, 0x00, 0x01, 0x08, 0x1c, 0x48, 0x50, 0x60,
    0x40, 0x00, 0x21, 0xf9, 0x04, 0x05, 0x08, 0x00, 0x01, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x04,
    0x00, 0x03, 0x00, 0x81, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x08, 0x08, 0x00, 0x01, 0x08, 0x1c, 0x48, 0x50, 0x60, 0x40, 0x00, 0x21, 0xf9, 0x04, 0x05, 0x0a,
    0x00, 0x01, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x03, 0x00, 0x81, 0x00, 0x00, 0xff,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x08, 0x00, 0x01, 0x08, 0x1c, 0x48,
    0x50, 0x60, 0x40, 0x00, 0x3b,
];

const ANIM4_WEBP: [u8; 234] = [
    0x52, 0x49, 0x46, 0x46, 0xe2, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x58,
    0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x41, 0x4e,
    0x49, 0x4d, 0x06, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x04, 0x00, 0x41, 0x4e, 0x4d, 0x46,
    0x26, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00,
    0x28, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c, 0x0e, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00,
    0x00, 0x07, 0x10, 0x11, 0xfd, 0x0f, 0x44, 0x44, 0xff, 0x03, 0x41, 0x4e, 0x4d, 0x46, 0x28, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x3c, 0x00,
    0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x0f, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x07,
    0x10, 0xfd, 0x8f, 0xfe, 0x07, 0x22, 0xa2, 0xff, 0x01, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x28, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x50, 0x00,
    0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x0f, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x07,
    0xd0, 0xff, 0x88, 0xfe, 0x07, 0x22, 0xa2, 0xff, 0x01, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x28, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x64, 0x00,
    0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x0f, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x07,
    0x10, 0xd1, 0xff, 0xfe, 0x07, 0x22, 0xa2, 0xff, 0x01, 0x00,
];

#[cfg(feature = "jxl")]
const ANIM4_JXL: [u8; 103] = [
    0xff, 0x0a, 0x10, 0x30, 0xc1, 0x88, 0xb0, 0x26, 0x08, 0x00, 0x20, 0x0a, 0x00, 0x00, 0x3c, 0x00,
    0x4b, 0x18, 0x8b, 0x15, 0xc2, 0x49, 0x41, 0x0e, 0x00, 0x00, 0x00, 0x00, 0xf4, 0x07, 0x00, 0x08,
    0x00, 0x20, 0x0f, 0x00, 0x00, 0x40, 0x00, 0x4b, 0x18, 0x8b, 0x15, 0xc2, 0x49, 0x41, 0x4e, 0x7f,
    0x00, 0x00, 0x00, 0x00, 0xf4, 0x07, 0x00, 0x08, 0x00, 0x20, 0x14, 0x00, 0x00, 0x40, 0x00, 0x4b,
    0x18, 0x8b, 0x15, 0xc2, 0x49, 0x41, 0x0e, 0x00, 0xf4, 0x07, 0x00, 0x00, 0xf4, 0x07, 0x00, 0x08,
    0x00, 0x20, 0x59, 0x00, 0x00, 0x40, 0x00, 0x4b, 0x18, 0x8b, 0x15, 0xc2, 0x49, 0x41, 0x0e, 0x00,
    0x00, 0x40, 0x7f, 0x00, 0xf4, 0x07, 0x00,
];

/// The delays the fixture carries, in milliseconds, one per file page.
///
/// The GIF wire holds `4 6 8 10` centiseconds and vips reads them back as
/// these, which is the factor of ten `crate::frames::FrameDelay` exists to keep
/// visible. The WebP and JPEG XL copies carry the milliseconds directly and
/// `vipsheader -f delay` reports the same four for all three files.
const FILE_DELAYS: [i64; 4] = [40, 60, 80, 100];

/// How many times the animation plays. `vipsheader -f loop` reports 4 on all
/// three files, off a NETSCAPE repeat count of 3.
const PLAYS: i32 = 4;

/// The colour every pixel of file page `i` holds, RGB.
///
/// The same four in the same order in all three containers, which is the thing
/// a roll can silently get wrong: a loader that stacks its frames backwards, or
/// that leaves a page uncomposited, produces a raster with the right geometry
/// and the wrong contents.
const PAGE_COLOURS: [[u8; 3]; 4] = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];

/// Decode one window of the fixture out of one container.
///
/// A function per container rather than a trait, because the three loaders take
/// three unrelated `LoadOptions` types on purpose: the page model is what they
/// share and the option structs are not part of it.
fn load(container: &str, page: u32, n: i32) -> Raster {
    let limits = DecodeLimits::default();
    match container {
        "gif" => {
            let options = gif::LoadOptions::default().with_page(page).with_n(n);
            decode_gif_with(&ANIM4_GIF, limits, options).expect("the GIF fixture decodes")
        }
        "webp" => {
            let options = webp::LoadOptions::default().with_page(page).with_n(n);
            decode_webp_with(&ANIM4_WEBP, limits, options).expect("the WebP fixture decodes")
        }
        #[cfg(feature = "jxl")]
        "jxl" => {
            let options = jxl::LoadOptions::default().with_page(page).with_n(n);
            decode_jxl_with(&ANIM4_JXL, limits, options).expect("the JPEG XL fixture decodes")
        }
        other => panic!("no loader for {other}"),
    }
}

/// Every container this file holds to the shared dialect.
///
/// JPEG XL is behind a non-default feature, so the list is shorter in a build
/// without it and every test below still runs on the containers that are there.
/// `only_the_three_animated_loaders_declare_a_page_split` in
/// `tests/page_model.rs` is what notices a loader missing from this list
/// altogether, and it scans the source rather than this array, so turning the
/// feature off cannot hide one.
const CONTAINERS: &[&str] = &[
    "gif",
    "webp",
    #[cfg(feature = "jxl")]
    "jxl",
];

/// The windows every container is asked for, and the roll each one must give
/// back.
///
/// `(page, n, height, pages_loaded, page-height)`. Measured on 8.18.6 with
/// `vipsheader -f height` and `-f page-height` against `anim4.gif`, the WebP
/// and the JPEG XL in turn, and the three files agree cell for cell.
const WINDOWS: [(u32, i32, u32, u32, Option<i32>); 4] = [
    // The default load, which is `page = 0, n = 1` in all three loaders.
    (0, 1, 3, 1, None),
    // Every frame.
    (0, -1, 12, 4, Some(3)),
    // A window that starts past the first frame, which is where a delay array
    // that was not subset stops lining up with its raster.
    (1, 2, 6, 2, Some(3)),
    // The last frame alone, the far edge of the range.
    (3, 1, 3, 1, None),
];

// ---------------------------------------------------------------------------
// The geometry
// ---------------------------------------------------------------------------

/// Every container stacks the same window into the same roll, and declares the
/// split the same way.
///
/// The page geometry is the whole of what "one shape" means, so this is the
/// test the other three lean on. `page-height` is attached only when more than
/// one page was loaded, which is not an arbitrary choice: measured on 8.18.6,
/// `vipsheader -f page-height` fails with `field "page-height" not found` on a
/// default load and on `[page=3,n=1]` of all three files, and reports 3 on
/// `[n=-1]` and `[page=1,n=2]` of all three.
///
/// `Raster::get_page_height` still answers on an unpaged raster, because it
/// ports `vips_image_get_page_height`, which reports the whole height when
/// there is no split. So the presence of the *field* is what separates the
/// rows, not the accessor.
///
/// Input: each container at each of the four windows.
/// Output: the height, `pages_loaded`, and the stored `page-height` in the
/// table.
#[test]
fn every_container_stacks_the_same_window_into_the_same_roll() {
    for container in CONTAINERS {
        for (page, n, height, pages_loaded, page_height) in WINDOWS {
            let raster = load(container, page, n);
            let where_ = format!("{container} at page={page}, n={n}");

            assert_eq!(raster.width(), 4, "{where_}: the width is one frame's");
            assert_eq!(
                raster.height(),
                height,
                "{where_}: the roll is one frame per loaded page"
            );
            assert_eq!(
                raster.pages_loaded(),
                pages_loaded,
                "{where_}: the derived page count"
            );
            assert_eq!(
                raster.get_int("page-height"),
                page_height,
                "{where_}: the split is declared only when there is one, which \
                 is what all three vips loaders do"
            );
            assert_eq!(
                raster.get_page_height(),
                3,
                "{where_}: one page is three rows whether or not the field is \
                 there, because the reader falls back to the whole height and \
                 an unpaged raster here is exactly one page tall"
            );
        }
    }
}

/// `n-pages` counts the file's frames in every container, and never the loaded
/// ones.
///
/// The two numbers are different questions and issue #635 pinned which is
/// which. This is the cross-container half of that: all three loaders report 4
/// on a fixture holding four frames, at every window, including the three
/// windows where the raster holds fewer than four pages.
///
/// Input: each container at each of the four windows.
/// Output: `n-pages` 4 throughout, and `pages_loaded` no larger than it.
#[test]
fn n_pages_counts_the_file_in_every_container() {
    for container in CONTAINERS {
        for (page, n, _, pages_loaded, _) in WINDOWS {
            let raster = load(container, page, n);
            let where_ = format!("{container} at page={page}, n={n}");

            assert_eq!(
                raster.get_n_pages(),
                4,
                "{where_}: the fixture has four frames whatever was loaded"
            );
            assert_eq!(
                raster.get_int("n-pages"),
                Some(4),
                "{where_}: and the \
                 field says so too, so a `.v` round trip carries it"
            );
            assert!(
                raster.pages_loaded() <= raster.get_n_pages(),
                "{where_}: {pages_loaded} pages loaded out of a four-page file"
            );
        }
    }
}

/// The pages come back in file order, with the same pixels, out of all three
/// containers.
///
/// Geometry alone does not prove a roll: a loader that stacks its frames
/// backwards, or that hands the same frame back four times, produces a raster
/// that passes every assertion above. This walks the roll with
/// `Raster::try_extract_page`, which is the model's own accessor, and checks
/// the colour of each page against the frame the file actually holds.
///
/// The alpha band is not compared. GIF and JPEG XL come back four-band here and
/// WebP three, which is each container's own business and not the page model's,
/// so only the RGB triple is asserted.
///
/// Input: every frame of each container, and the `[page=1,n=2]` window.
/// Output: pages 0 to 3 uniformly black, red, green and blue, and the window's
/// two pages being file pages 1 and 2.
#[test]
fn the_pages_come_back_in_file_order_with_the_same_pixels() {
    for container in CONTAINERS {
        let roll = load(container, 0, -1);
        let channels = roll.format().channels();
        assert_eq!(roll.pages_loaded(), 4, "{container}: four pages");

        for (index, colour) in PAGE_COLOURS.iter().enumerate() {
            let page = roll
                .try_extract_page(index as u32)
                .expect("every page of a four-page roll extracts");
            assert_eq!(
                page.height(),
                3,
                "{container}: an extracted page is one frame tall"
            );
            assert!(
                page.data().chunks(channels).all(|px| &px[..3] == colour),
                "{container}: page {index} is uniformly {colour:?}, and it is \
                 not: the first pixel is {:?}",
                &page.data()[..3]
            );
        }

        // And a window starting past the first frame really starts there,
        // which is the other way a roll goes wrong: the geometry is right and
        // the offset is not.
        let window = load(container, 1, 2);
        for (slot, index) in [1usize, 2].into_iter().enumerate() {
            let page = window
                .try_extract_page(slot as u32)
                .expect("both pages of a two-page window extract");
            assert!(
                page.data()
                    .chunks(channels)
                    .all(|px| px[..3] == PAGE_COLOURS[index]),
                "{container}: page {slot} of a `page=1, n=2` load is file page \
                 {index}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The timings
// ---------------------------------------------------------------------------

/// The delay array indexes the pages the raster holds, in every container.
///
/// This is the invariant that makes the array usable, and it is the one
/// deliberate divergence from vips in the whole model. vips attaches the
/// file's **whole** array whatever window was loaded: measured on 8.18.6,
/// `vipsheader -f delay` prints `40 60 80 100` on `anim4.gif[page=1,n=2]`, on
/// the WebP and on the JPEG XL alike, onto rasters holding two pages. Nothing
/// on such a raster records which file page its page 0 was, so the array
/// cannot be lined up with the pixels it is attached to, and a saver reading
/// it writes the first two delays onto frames that are really the second and
/// third.
///
/// Here `delay[i]` is loaded page `i`'s delay, so `delay.len() ==
/// pages_loaded()` holds everywhere. `Raster::encode_gif` refuses an array
/// whose length is not the page count, which is what turns this from a
/// preference into a requirement: under vips's rule a two-page load of a
/// four-frame animation could never be saved as an animated GIF at all.
///
/// Input: each container at each of the four windows.
/// Output: the loaded slice of `FILE_DELAYS`, and a length equal to
/// `pages_loaded`.
#[test]
fn the_delay_array_indexes_the_pages_the_raster_holds() {
    for container in CONTAINERS {
        for (page, n, _, pages_loaded, _) in WINDOWS {
            let raster = load(container, page, n);
            let where_ = format!("{container} at page={page}, n={n}");
            let delays = raster
                .get_int_array("delay")
                .unwrap_or_else(|| panic!("{where_}: an animation carries delays"));

            let expected = &FILE_DELAYS[page as usize..page as usize + pages_loaded as usize];
            assert_eq!(
                delays, expected,
                "{where_}: delay[i] is loaded page i's delay, where vips would \
                 report all of {FILE_DELAYS:?}"
            );
            assert_eq!(
                delays.len() as u32,
                raster.pages_loaded(),
                "{where_}: one delay per page, which is what encode_gif needs"
            );
        }
    }
}

/// A delay is milliseconds in every container, and the ten-times conversion
/// happens on the GIF wire and nowhere else.
///
/// The three containers disagree about the unit they store: the GIF graphic
/// control extension counts centiseconds, the WebP `ANMF` duration and the
/// JPEG XL frame duration count milliseconds. The field is milliseconds on all
/// three, which means the GIF loader is the only one that converts, and it is
/// the one place a silent factor of ten can live (issue #572).
///
/// The fixture is built so the two answers are far apart: `4 6 8 10` on the
/// wire is `40 60 80 100` here, and a loader that forgot the conversion would
/// report the wire numbers, which are still four plausible small delays.
///
/// Input: every frame of each container.
/// Output: `40 60 80 100`, and not `4 6 8 10`.
#[test]
fn a_delay_is_milliseconds_in_every_container() {
    for container in CONTAINERS {
        let roll = load(container, 0, -1);
        let delays = roll.get_int_array("delay").expect("delays are attached");

        assert_eq!(
            delays, &FILE_DELAYS,
            "{container}: the field is milliseconds whatever the wire held"
        );
        // The wire numbers are the failure this is shaped to catch, and they
        // are plausible enough to pass an eyeball.
        assert_ne!(
            delays,
            &[4i64, 6, 8, 10],
            "{container}: these are the GIF wire's centiseconds, not the field"
        );
    }
}

/// The loop count is plays in every container, `0` meaning forever.
///
/// The two wires disagree by one and in opposite directions, which is why
/// `crate::frames::LoopCount` has a conversion named for each: the GIF
/// NETSCAPE block counts repeats *after* the first play, so a count of 3 is 4
/// plays, and the WebP `ANIM` chunk holds the play count unshifted. The
/// fixture goes through both, since the WebP and the JPEG XL were written from
/// the GIF by vips, and all three report 4.
///
/// A `loop` that came out as 3 would be the shift applied twice or not at all,
/// and it is the answer a shared helper across the two containers would have
/// produced for one of them.
///
/// Input: each container at each window.
/// Output: `loop` = 4 throughout.
#[test]
fn the_loop_count_is_plays_in_every_container() {
    for container in CONTAINERS {
        for (page, n, _, _, _) in WINDOWS {
            let raster = load(container, page, n);
            assert_eq!(
                raster.get_int("loop"),
                Some(PLAYS),
                "{container} at page={page}, n={n}: four plays, off a NETSCAPE \
                 repeat count of 3"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The two deprecated compatibility names, which is where the dialect drifted
// ---------------------------------------------------------------------------

/// What each container's vips loader attaches under `gif-delay` and
/// `gif-loop`, and whether this crate's loader attaches them yet.
///
/// Measured on 8.18.6 with `vipsheader -f`, on the three fixture files, at the
/// default window:
///
/// | loader | `gif-delay` | `gif-loop` |
/// |---|---|---|
/// | `gifload` | 4 | 3 |
/// | `webpload` | 4 | 3 |
/// | `jxlload` | 4 | **absent** |
///
/// `gif-delay` is the first frame's delay in centiseconds, so 40 ms gives 4
/// under the round-half-to-even `FrameDelay::to_centiseconds` measures.
/// `gif-loop` is the NETSCAPE repeat count, so four plays gives 3, which is
/// `LoopCount::to_gif_wire().unwrap_or(0)`. That `jxlload` attaches one and not
/// the other is not an omission in the measurement: it is what vips does, and
/// it is why this table has a column per name rather than a single flag.
///
/// The last field is the one row that is not parity yet. **Issue #865**: the
/// GIF loader attaches neither, where `gifload` attaches both. The GIF lane
/// owns `src/gif.rs` and the fix; this table records the gap so it cannot be
/// forgotten, and turns green by flipping `false` to `true`.
const COMPAT_PAIR: [(&str, Option<i32>, Option<i32>, bool); 3] = [
    ("gif", Some(4), Some(3), false),
    ("webp", Some(4), Some(3), true),
    ("jxl", Some(4), None, true),
];

/// Each loader attaches the compatibility pair its own vips loader attaches,
/// with the one filed exception.
///
/// This is the test the divergence was found by, and the shape of it is the
/// point: three loaders, one table, one place to look. Before it existed each
/// loader was measured against its own oracle in its own module and the three
/// answers were never put side by side, so two loaders attaching a field that
/// the third does not was invisible.
///
/// Input: each container at the default window.
/// Output: the pair from `COMPAT_PAIR`, or both absent on the row still
/// carrying #865.
#[test]
fn each_loader_attaches_the_compatibility_pair_its_oracle_attaches() {
    for container in CONTAINERS {
        let (_, vips_delay, vips_loop, attached) = COMPAT_PAIR
            .iter()
            .find(|(name, ..)| name == container)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "{container} has no row in COMPAT_PAIR. A new animated \
                     loader has to say what its own vips loader attaches under \
                     `gif-delay` and `gif-loop`, measured with `vipsheader -f` \
                     and never with `-a`, which lists neither on any file"
                )
            });
        let raster = load(container, 0, 1);

        if attached {
            assert_eq!(
                raster.get_int("gif-delay"),
                vips_delay,
                "{container}: `gif-delay` is the first delay in centiseconds"
            );
            assert_eq!(
                raster.get_int("gif-loop"),
                vips_loop,
                "{container}: `gif-loop` is the NETSCAPE repeat count, which is \
                 one less than the play count, and `jxlload` attaches none at \
                 all"
            );
        } else {
            assert_eq!(
                raster.get_int("gif-delay"),
                None,
                "{container}: issue #865. `gifload` attaches `gif-delay` \
                 {vips_delay:?} here and this loader does not. If you have just \
                 fixed that, flip this row's last field to `true` in \
                 COMPAT_PAIR"
            );
            assert_eq!(
                raster.get_int("gif-loop"),
                None,
                "{container}: issue #865, same row, same fix"
            );
        }
    }
}

/// The list of loaders still short of their oracle is exactly the one that is
/// filed.
///
/// The exception above is only safe while it is an exception. Without this, a
/// second loader could be added to `COMPAT_PAIR` with `false` and the suite
/// would stay green over two gaps instead of one.
///
/// Input: `COMPAT_PAIR`.
/// Output: exactly `["gif"]` short, and it is #865.
#[test]
fn only_one_loader_is_short_of_its_oracle_and_it_is_filed() {
    let short: Vec<&str> = COMPAT_PAIR
        .iter()
        .filter(|(.., attached)| !attached)
        .map(|(name, ..)| *name)
        .collect();

    assert_eq!(
        short,
        ["gif"],
        "the GIF loader not attaching `gif-delay` and `gif-loop` is issue \
         #865 and it is the only known gap in the dialect. A second `false` \
         row needs its own issue and its number written next to it, not a \
         quiet seat on this one"
    );
}

/// `gif-delay` follows the loaded window in every container that attaches it.
///
/// It is the scalar half of the delay array, so it moves with the array: on a
/// `page = 1` load the first *loaded* page is file page 1, and `gif-delay` is
/// 6 centiseconds rather than the 4 the file's frame 0 would give. vips reports
/// 4 at every window, for the same reason it reports the whole `delay` array at
/// every window.
///
/// What this pins is that the two loaders that attach it agree with each other.
/// A loader that took `gif-delay` off the file while subsetting the array would
/// be internally inconsistent, and that is a state neither module's own suite
/// would notice.
///
/// Input: each attaching container at `page = 1, n = 2` and `page = 3, n = 1`.
/// Output: 6 and 10 centiseconds, the loaded first page's delay.
#[test]
fn the_compatibility_delay_follows_the_loaded_window_too() {
    for container in CONTAINERS {
        let attaches = COMPAT_PAIR
            .iter()
            .any(|(name, .., attached)| name == container && *attached);
        if !attaches {
            continue;
        }

        for (page, n, centiseconds) in [(1u32, 2i32, 6i32), (3, 1, 10)] {
            let raster = load(container, page, n);
            assert_eq!(
                raster.get_int("gif-delay"),
                Some(centiseconds),
                "{container} at page={page}, n={n}: `gif-delay` is loaded page \
                 0's delay, the same window the `delay` array follows"
            );
        }
    }
}
