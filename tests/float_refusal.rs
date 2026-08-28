//! Catching "this operation refuses float rasters" is a question across four
//! enums, and this file pins that they all spell it the same way and that each
//! operation raises the one its own signature promises (issue #730).
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
    let refusals: Vec<(&str, OpError)> = vec![
        (
            "join",
            float
                .try_join(&float, JoinDirection::Horizontal, true, None, None, None)
                .unwrap_err()
                .into(),
        ),
        (
            "arrayjoin",
            Raster::try_arrayjoin(&[&float, &float], Some(2), None)
                .unwrap_err()
                .into(),
        ),
        (
            "insert",
            float
                .try_insert(&float, 0, 0, true, None)
                .unwrap_err()
                .into(),
        ),
        (
            "embed",
            float
                .try_embed(0, 0, 8, 8, libviprs::Extend::Black, None)
                .unwrap_err()
                .into(),
        ),
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

/// Issue #730. `try_join` refuses float in **its own** error type rather than
/// letting the delegated `try_insert` refusal surface.
///
/// The guard used to be what stopped a panic; #694 moved that into
/// `try_insert`, so what it does now is keep the refusal in the type
/// `try_join`'s signature promises. It runs before the delegation, so it is the
/// one that fires and `try_insert`'s is unreachable from here. Removing it does
/// not restore a panic, it changes this assertion to
/// `Extract(FloatUnsupported { op: "insert" })`, which names an operation the
/// caller never called.
#[test]
fn join_refuses_in_its_own_type_rather_than_delegating() {
    let float = float_rgb();
    let err = float
        .try_join(&float, JoinDirection::Horizontal, true, None, None, None)
        .unwrap_err();
    assert!(
        matches!(err, ConversionError::FloatUnsupported { op: "join" }),
        "join names itself: {err:?}"
    );

    // A float on either side alone is enough, so the guard reads both inputs.
    let mixed = uchar_rgb()
        .try_join(&float, JoinDirection::Vertical, true, None, None, None)
        .unwrap_err();
    assert!(
        matches!(mixed, ConversionError::FloatUnsupported { op: "join" }),
        "a float second input is refused the same way: {mixed:?}"
    );
    let mixed = float
        .try_join(&uchar_rgb(), JoinDirection::Vertical, true, None, None, None)
        .unwrap_err();
    assert!(
        matches!(mixed, ConversionError::FloatUnsupported { op: "join" }),
        "a float first input is refused the same way: {mixed:?}"
    );
}

/// Issue #730. `arrayjoin`'s guard is a different animal from `join`'s: it is
/// still the thing that stops a panic, because `arrayjoin` blits the cells with
/// `read_flat` / `write_flat` itself rather than delegating to `try_insert`.
///
/// So the two guards look identical and are not, and the comments now say
/// which is which.
#[test]
fn arrayjoin_refuses_float_with_nothing_underneath_it() {
    let float = float_rgb();
    let err = Raster::try_arrayjoin(&[&uchar_rgb(), &float], Some(2), None).unwrap_err();
    assert!(
        matches!(err, ConversionError::FloatUnsupported { op: "arrayjoin" }),
        "arrayjoin names itself: {err:?}"
    );
}

/// Issue #730. The colour ops are the reason a float raster reaches these at
/// all, which is what makes the refusal worth a typed error rather than a
/// debug assertion.
///
/// `colourspace` to any of the float-carried spaces hands back an `RgbaF32`,
/// so `im.colourspace(Lab).join(..)` is an ordinary thing to write.
#[test]
fn a_colourspace_result_is_the_float_raster_that_reaches_these() {
    let lab = uchar_rgb().try_colourspace(Interpretation::Lab).unwrap();
    assert!(lab.format().is_float(), "colourspace to Lab is float");
    assert!(
        matches!(
            lab.try_join(&lab, JoinDirection::Horizontal, true, None, None, None),
            Err(ConversionError::FloatUnsupported { op: "join" })
        ),
        "and joining it is refused rather than panicking"
    );
}
