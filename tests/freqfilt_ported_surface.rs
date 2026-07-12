//! Pins the frequency-filter call surface required by the libviprs-tests
//! ported suite (libviprs-tests issue #55, `tests/ported_create.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves. The ported suite's only frequency-domain call
//! site today is `fwfft` (`test_fwfft_small_image` in
//! `ported_create.rs`); its body is reproduced literally below.
//! Behaviour depth for the rest of the family (`invfft`, `invfft_real`,
//! `freqmult`, `spectrum`, `phasecor`) is covered by the unit tests in
//! `src/freqfilt.rs`; the smoke checks here pin their signatures from
//! the external-caller position.

use libviprs::{FreqfiltError, PixelFormat, Raster};

/// The ported `test_fwfft_small_image` body.
#[test]
fn ported_fwfft_small_image() {
    let im = Raster::black(2, 1);
    let _fft = im.fwfft(); // Should not panic
}

/// The rest of the freqfilt family compiles and runs from the external
/// position: panicking forms and `try_` twins.
#[test]
fn freqfilt_family_surface() {
    let im = Raster::black(8, 8);

    let fourier: Raster = im.fwfft();
    let _complex: Raster = fourier.invfft();
    let real: Raster = fourier.invfft_real();
    assert_eq!(real.format().channels(), 1);

    let mask = Raster::mask_ideal(8, 8, 0.5, false);
    let _filtered: Raster = im.freqmult(&mask);
    let _display: Raster = im.spectrum();
    let _corr: Raster = im.phasecor(&im);

    let _: Result<Raster, FreqfiltError> = im.try_fwfft();
    let _: Result<Raster, FreqfiltError> = fourier.try_invfft();
    let _: Result<Raster, FreqfiltError> = fourier.try_invfft_real();
    let _: Result<Raster, FreqfiltError> = im.try_freqmult(&mask);
    let _: Result<Raster, FreqfiltError> = im.try_spectrum();
    let _: Result<Raster, FreqfiltError> = im.try_phasecor(&im);
}

/// fwfft output is a complex (re, im) float raster: two bands for a
/// one-band input, in the interleaved convention the arithmetic
/// module's complex family uses.
#[test]
fn fwfft_output_is_complex_pairs() {
    let im = Raster::black(4, 4);
    let f = im.fwfft();
    assert_eq!(f.format().channels(), 2);
    assert!(f.format().is_float());
    assert_eq!(f.format(), PixelFormat::with_channels(2, 4).unwrap());
}
