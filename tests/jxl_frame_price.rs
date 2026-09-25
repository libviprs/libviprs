//! `decode_jxl` prices the frame buffer twice, and the two prices are
//! different checks rather than one written down twice (issue #901).
//!
//! # Why the module's own budget tests could not see this
//!
//! The first price runs on the declared frame before any frame data is fed
//! in, and the second on the stacked roll after the keyframe count is known.
//! For a **single-page** file those are the same product, because
//! `roll_height` is `height * loaded` and `loaded` is 1. So a mutation that
//! under-charges the first check leaves the second refusing the identical
//! file at the identical threshold with the identical error, and every budget
//! fixture in `src/jxl.rs` is single-page.
//!
//! The #748 mutation sweep found exactly that: charging one byte per sample
//! instead of the sample width in the first check left the whole suite green.
//! The code was correct; the coverage was not.
//!
//! # What makes them distinguishable
//!
//! A **multi-page 16-bit** file. Sixteen bits so a byte-per-sample mutation
//! changes the product at all, and multi-page so the two products differ.
//! Then the two checks refuse at different budgets *and* report different
//! geometry, because the first names the frame and the second names the roll,
//! so the observation is a typed field rather than a timing.
//!
//! Measured on the fixture below, at `n = -1`:
//!
//! | budget | refused by | geometry reported |
//! |---|---|---|
//! | 131071 | the frame pre-check | 256x256 |
//! | 131072 | the roll check | 256x512 |
//!
//! One byte apart, and that byte is the boundary between the two.
//!
//! The fixture arrives through `include_bytes!`, so nothing here touches the
//! filesystem and no row belongs in `tests/miri_fs_test_inventory.txt`.

#![cfg(feature = "jxl")]

use libviprs::source::{DeclaredGeometry, DecodeLimits, SourceError};
use libviprs::{decode_jxl_with, jxl};

/// `vips jxlsave --lossless --strip` on a 256x512 `grey16` toilet roll with
/// `page-height 256`: two 256x256 16-bit pages.
///
/// Sixteen-bit through `--interpretation grey16`, which is what makes vips
/// write a 16-bit codestream rather than narrowing to 8; the pixels are a
/// smooth ramp over the whole range, so the file is 2 KiB and still needs
/// every one of the sixteen bits. `vips getpoint` reads 819 at (3, 3) and
/// 5868 at (3, 300), which is `x * 256 + y * 17` in both cases.
const ROLL16: &[u8] = include_bytes!("fixtures/roll16_2page.jxl");

/// One page of the fixture, in bytes, at its real sample width.
const FRAME_PRICE: u64 = 256 * 256 * 2;

/// Both pages.
const ROLL_PRICE: u64 = 256 * 512 * 2;

/// The refusal `decode_jxl_with` gives for the whole roll under `budget`.
fn refusal_at(budget: u64) -> SourceError {
    decode_jxl_with(
        ROLL16,
        DecodeLimits::default().with_max_alloc_bytes(budget),
        jxl::LoadOptions::default().with_n(-1),
    )
    .expect_err("every budget here is under one of the two prices")
}

/// Tests that the frame pre-check is a check of its own, by making it refuse
/// a file the roll check would have let through at that budget and reporting
/// the frame's geometry rather than the roll's.
///
/// The mutation this exists for charges one byte per sample in the first
/// check. Under it the pre-check passes at this budget and the roll check
/// refuses instead, which is visible here as the reported height moving from
/// 256 to 512.
///
/// Input: the two-page 16-bit roll at a budget between half the frame price
/// and the frame price -> Output: a refusal naming 256x256 and the frame's
/// own 131072 bytes.
#[test]
fn the_frame_pre_check_refuses_before_the_roll_check_and_names_the_frame() {
    // Between the mutated frame price (65536) and the real one (131072), so
    // a correct pre-check refuses and a byte-per-sample one does not.
    let err = refusal_at(100_000);
    assert!(
        matches!(
            err,
            SourceError::AllocLimitExceeded {
                what: "JPEG XL frame buffer",
                geometry: Some(DeclaredGeometry {
                    width: 256,
                    height: 256,
                    bands: 1,
                    ..
                }),
                needed_bytes: FRAME_PRICE,
                max_alloc_bytes: 100_000,
            }
        ),
        "the frame pre-check has to be the one that answers, and it has to \
         name the frame's own 256 rows: {err:?}"
    );
}

/// Tests that the roll check is a check of its own too, at a budget the frame
/// pre-check accepts, and that it names the roll rather than the frame.
///
/// Without this the test above passes on a build where the roll check does
/// not exist, since one refusal would look like the other.
///
/// Input: the same roll at a budget between the frame price and the roll
/// price -> Output: a refusal naming 256x512 and the roll's 262144 bytes.
#[test]
fn the_roll_check_refuses_where_the_frame_fits_and_names_the_roll() {
    let err = refusal_at(200_000);
    assert!(
        matches!(
            err,
            SourceError::AllocLimitExceeded {
                what: "JPEG XL frame buffer",
                geometry: Some(DeclaredGeometry {
                    width: 256,
                    height: 512,
                    bands: 1,
                    ..
                }),
                needed_bytes: ROLL_PRICE,
                max_alloc_bytes: 200_000,
            }
        ),
        "the roll check has to answer here, naming both pages: {err:?}"
    );
}

/// Tests that the boundary between the two checks is exactly the frame price,
/// which is what proves they are two checks and not one comparison written
/// twice. One byte below it the frame answers; at it the roll does.
///
/// Input: the roll at `FRAME_PRICE - 1` and at `FRAME_PRICE` -> Output: the
/// frame's geometry and then the roll's, from adjacent budgets.
#[test]
fn one_byte_separates_the_two_prices() {
    let below = refusal_at(FRAME_PRICE - 1);
    let at = refusal_at(FRAME_PRICE);

    let height_of = |err: &SourceError| match err {
        SourceError::AllocLimitExceeded {
            geometry: Some(g), ..
        } => g.height,
        other => panic!("expected an allocation refusal, got {other:?}"),
    };

    assert_eq!(
        height_of(&below),
        256,
        "one byte under the frame price the frame pre-check answers"
    );
    assert_eq!(
        height_of(&at),
        512,
        "at the frame price it does not, and the roll check answers instead"
    );
}

/// Tests that a budget covering both prices really does load the file, so the
/// three refusals above are the budget biting rather than the fixture being
/// undecodable.
///
/// This is the positive control the other three need: without it they pass on
/// a build that refuses every JPEG XL for an unrelated reason.
///
/// Input: the roll at 8 MiB -> Output: a 256x512 two-page raster.
#[test]
fn a_budget_over_both_prices_loads_the_roll() {
    let roll = decode_jxl_with(
        ROLL16,
        DecodeLimits::default().with_max_alloc_bytes(8 << 20),
        jxl::LoadOptions::default().with_n(-1),
    )
    .expect("8 MiB covers both prices");
    assert_eq!((roll.width(), roll.height()), (256, 512));
    assert_eq!(roll.pages_loaded(), 2);
}
