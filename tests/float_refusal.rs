//! Catching "this operation refuses float rasters" is a question across four
//! enums, and this file pins that they all spell it the same way and that each
//! operation raises the one its own signature promises (issue #730).
//!
//! **The population moved in issue #945.** `join`, `arrayjoin`, `insert` and
//! `embed` were the four operations this file was written around, and all four
//! carry a float raster now: vips runs every one of them and answers FLOAT, so
//! refusing was posture 1, a parity regression. The refusals were not wrong
//! when they were written, and they are replaced here by value assertions
//! rather than deleted, the way #909's were.
//!
//! What #730's claim is asserted over now is the operations that still refuse,
//! one per enum, and they refuse for a reason that is not going away: each of
//! them indexes a table by the sample value, or writes an integer result, and
//! a float sample does neither.
//!
//! Four enums is the consequence of every module owning its error type, which
//! is the right shape and not what #730 changed. What it changed is that
//! `ConversionError` said `FloatFormatUnsupported` where the other three said
//! `FloatUnsupported`, so a caller writing the question had to know which one
//! was the odd one out.
//!
//! # Why not a predicate
//!
//! `SourceError::is_alloc_limit()` (#686) is the shape this could have taken,
//! and it does not fit: that one collapses *five variants of one enum* onto a
//! question, where this is one variant of each of four enums. A predicate would
//! be four impls that still cannot be called through a single type without a
//! trait, and a trait carrying one method is a lot of public surface for
//! something a `matches!` answers once the names agree.

use libviprs::{
    ArithmeticError, ConversionError, ExtractError, Interpretation, JoinDirection, OpError,
    PixelFormat, Raster, RasterError,
};

fn float_rgb() -> Raster {
    Raster::new(4, 4, PixelFormat::RgbaF32, vec![0u8; 4 * 4 * 4 * 4]).unwrap()
}

fn uchar_rgb() -> Raster {
    Raster::new(4, 4, PixelFormat::Rgb8, vec![7u8; 4 * 4 * 3]).unwrap()
}

/// The four-band unsigned twin of [`float_rgb`], for the mixed-carrier cases.
///
/// Four bands rather than three, deliberately: `join` and `arrayjoin` both
/// answer `BandCountMismatch` before they read a sample when the counts
/// differ and neither is 1, so a three-band fixture never reaches the
/// promotion the mixed cases are about.
fn uchar_rgba() -> Raster {
    Raster::new(4, 4, PixelFormat::Rgba8, vec![3u8; 4 * 4 * 4]).unwrap()
}

/// Issue #730. One `matches!` shape reaches every float refusal in the crate.
///
/// This is the whole point of the rename: before it, the `ConversionError` arm
/// had to be spelled differently from the other three, which is exactly the
/// sort of thing a caller gets wrong once and never notices.
#[test]
fn every_float_refusal_matches_one_shape() {
    fn is_float_refusal(e: &OpError) -> bool {
        matches!(
            e,
            OpError::Raster(RasterError::FloatUnsupported { .. })
                | OpError::Arithmetic(ArithmeticError::FloatUnsupported { .. })
                | OpError::Extract(ExtractError::FloatUnsupported { .. })
                | OpError::Conversion(ConversionError::FloatUnsupported { .. })
        )
    }

    let float = float_rgb();
    // One operation per enum, so the shape is asserted over all four and not
    // over four spellings of one. The population is #945's rather than #730's:
    // the four ops this used to read all carry a float raster now.
    let refusals: Vec<(&str, OpError)> = vec![
        (
            "downscale_half",
            libviprs::resize::downscale_half(&float).unwrap_err().into(),
        ),
        ("add_const", float.try_add_const(5.0).unwrap_err().into()),
        (
            "smartcrop",
            float
                .try_smartcrop(2, 2, libviprs::SmartcropInteresting::Entropy, false)
                .unwrap_err()
                .into(),
        ),
        ("gamma", float.try_gamma(None).unwrap_err().into()),
    ];

    for (name, e) in &refusals {
        assert!(is_float_refusal(e), "{name} must match the one shape: {e}");
    }

    // And the shape is not vacuous: something that is not a float refusal must
    // not match it.
    let not_a_refusal: OpError = uchar_rgb()
        .try_insert(&uchar_rgb(), 0, 0, false, Some(&[1.0, 2.0]))
        .unwrap_err()
        .into();
    assert!(
        !is_float_refusal(&not_a_refusal),
        "a background-length mismatch is not a float refusal: {not_a_refusal}"
    );
}

/// Issues #730 and #945. `try_join` carries a float raster on **either** side,
/// and the result takes the promotion of the two carriers.
///
/// This replaces `join_refuses_in_its_own_type_rather_than_delegating`, which
/// asserted that `join` kept the refusal in the type its own signature
/// promises rather than surfacing `try_insert`'s. That was the right shape for
/// a refusal and the wrong thing against vips, which answers FLOAT for all
/// three of these. Measured on `/opt/homebrew/bin/vips` 8.18.6:
/// `vips join rf.v qu.v out.v horizontal` over a 2x2 `float` and a 3x2 `uchar`
/// answers a 5x2 FLOAT image with both operands' samples intact, and so does
/// the same pair the other way round.
///
/// The three cases are kept because they are the ones that separate "reads
/// both inputs" from "reads the first": an implementation promoting off `self`
/// alone passes the all-float case and fails the second mixed one.
#[test]
fn join_carries_a_float_raster_on_either_side() {
    let float = float_rgb();
    let both = float
        .try_join(&float, JoinDirection::Horizontal, true, None, None, None)
        .expect("join carries a float raster since #945");
    assert!(both.format().is_float(), "got {:?}", both.format());

    for (a, b) in [(uchar_rgba(), float_rgb()), (float_rgb(), uchar_rgba())] {
        let out = a
            .try_join(&b, JoinDirection::Vertical, true, None, None, None)
            .expect("a mixed pair promotes rather than being refused");
        assert!(
            out.format().is_float(),
            "the promotion has to reach both operand orders, got {:?}",
            out.format()
        );
    }

    // The control that `join` still refuses what it should: a background
    // vector of the wrong length is not a carrier question, and it still
    // comes back typed.
    let bad = uchar_rgb()
        .try_join(
            &uchar_rgb(),
            JoinDirection::Horizontal,
            true,
            None,
            Some(&[1.0, 2.0]),
            None,
        )
        .unwrap_err();
    assert!(
        matches!(bad, ConversionError::Extract(_)),
        "a bad background is still refused: {bad:?}"
    );
}

/// Issues #730 and #945. `arrayjoin` carries a float cell with its **own**
/// copy, not through a delegate.
///
/// This replaces `arrayjoin_refuses_float_with_nothing_underneath_it`. That
/// test's fixture shape is kept and so is the reason for it: `arrayjoin` blits
/// the cells itself rather than delegating to `try_insert`, so it is the only
/// place the copy can be got wrong, and the band counts have to **match** to
/// reach it at all. With a 3-band and a 4-band input the band check answers
/// first, which is what nearly let the old claim ship unverified. The
/// four-band pair is the one that sits on the copy.
#[test]
fn arrayjoin_carries_a_float_cell_through_its_own_copy() {
    let float = float_rgb();
    let uchar4 = uchar_rgba();
    assert_eq!(
        float.format().channels(),
        uchar4.format().channels(),
        "the band counts must match or the band check answers first"
    );

    for list in [[&uchar4, &float], [&float, &uchar4], [&float, &float_rgb()]] {
        let out = Raster::try_arrayjoin(&list, Some(2), None)
            .expect("arrayjoin carries a float cell since #945");
        assert!(out.format().is_float(), "got {:?}", out.format());
        assert_eq!((out.width(), out.height()), (8, 4));
    }

    // And the band check still answers first for a mismatched pair, which is
    // the pre-emption the fixture above is shaped to avoid.
    let err = Raster::try_arrayjoin(&[&uchar_rgb(), &float], Some(2), None).unwrap_err();
    assert!(
        matches!(err, ConversionError::BandCountMismatch { .. }),
        "3 bands against 4 is refused before any sample is read: {err:?}"
    );
}

/// Issue #730. The colour ops are the reason a float raster reaches these at
/// all, which is what makes the refusal worth a typed error rather than a
/// debug assertion.
///
/// `colourspace` to any of the float-carried spaces hands back an `RgbaF32`,
/// so `im.colourspace(Lab).join(..)` is an ordinary thing to write. It used to
/// be refused; since #945 it works, which is the point of the whole change.
#[test]
fn a_colourspace_result_is_the_float_raster_that_reaches_these() {
    let lab = uchar_rgb().try_colourspace(Interpretation::Lab).unwrap();
    assert!(lab.format().is_float(), "colourspace to Lab is float");
    let joined = lab
        .try_join(&lab, JoinDirection::Horizontal, true, None, None, None)
        .expect("joining a Lab result is what #945 makes work");
    assert!(joined.format().is_float(), "got {:?}", joined.format());
    assert_eq!(joined.width(), lab.width() * 2);

    // The op that still refuses it, so this file keeps a live example of the
    // refusal the shape above is written for.
    assert!(
        matches!(
            lab.try_gamma(None),
            Err(ConversionError::FloatUnsupported { op: "gamma" })
        ),
        "gamma indexes a 256-entry table by the sample, and still refuses"
    );
}
