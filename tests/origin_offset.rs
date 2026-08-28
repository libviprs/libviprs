//! The origin offset (`Raster::xoffset` / `Raster::yoffset`) says where the
//! input's origin sits inside the result. Some operations reposition or
//! resample and **stamp** a new one; the rest pass the input's through. This
//! file is the guard on which do which (issue #721).
//!
//! Before #721 every operation carried, because carrying is what
//! `Raster::carry_meta_from` does, so eleven of them reported the input's
//! offsets where vips reports a stamp. #706 found the first instance of this
//! split, in the other direction: `extract_area` and `crop` stamp `-left` /
//! `-top` where the six other extract ops carry.
//!
//! # What the oracle says
//!
//! Measured against the pinned `vips-8.18.6` (`oracle-captures/ORACLE_PIN.json`),
//! read off the binary at `/opt/homebrew/bin/vips`. Every source is tagged
//! `--xoffset 11 --yoffset 13`, and every op was run at three image shapes so a
//! rule is a rule rather than one cell.
//!
//! ```text
//! op                     5x6      7x3      8x8      rule
//! flip horizontal        5 / 0    7 / 0    8 / 0    (width, 0)
//! flip vertical          0 / 6    0 / 3    0 / 8    (0, height)
//! rot d90                6 / 0    3 / 0    8 / 0    (out width, 0)
//! rot d180               5 / 6    7 / 3    8 / 8    (width, height)
//! rot d270               0 / 5    0 / 7    0 / 8    (0, out height)
//! wrap (default)         3 / 3    4 / 2    4 / 4    (w - w/2, h - h/2)
//! conv 3x3              -1 /-1   -1 /-1   -1 /-1    (-(mask w / 2), -(mask h / 2))
//! conv 5x5              -2 /-2   -2 /-2   -2 /-2
//! conv 3 wide x 5 tall  -1 /-2   -1 /-2   -1 /-2
//! conv 7 wide x 3 tall  -3 /-1   -3 /-1   -3 /-1
//! convsep 3x1            0 /-1    0 /-1    0 /-1    the *rotated* mask's rule
//! convsep 1x3           -1 / 0   -1 / 0   -1 / 0
//! compass 3x3           -1 /-1   -1 /-1   -1 /-1    the convolution's rule
//! gaussblur sigma 1      0 /-1    0 /-1    0 /-1    the generated mask's rule
//! gaussblur sigma 2      0 /-3    0 /-3    0 /-3
//! gaussblur sigma 3      0 /-5    0 /-5    0 /-5
//! sobel/scharr/prewitt  -1 /-1   -1 /-1   -1 /-1    the 3x3 gradient mask
//! canny (any sigma)     -1 /-1   -1 /-1   -1 /-1    the 2x2 gradient mask
//! ```
//!
//! Not one of them depends on the input's offsets: `11 / 13` never appears in
//! that table, and re-running the whole sweep from a source left at `0 / 0`
//! gives byte-identical numbers.
//!
//! **`convsep` is the cell that says what the rule really is.** A 3-wide,
//! 1-tall mask gives `0 / -1`, not the `-1 / 0` the mask itself would imply.
//! `convsep` convolves with the mask and then with its 90-degree rotation, and
//! the offset is the *last* pass's: rotating 3x1 gives 1x3, whose rule is
//! `(-(1/2), -(3/2))` = `(0, -1)`. So there is one rule, `conv`'s, and
//! `convsep`, `compass` and `gaussblur` inherit it by composition rather than
//! needing one each.
//!
//! # The positive control
//!
//! "The offsets changed" on its own could be the `.v` writer rather than the
//! op, so the ops that *carry* are asserted here too. `rot45` at all seven
//! angles and three odd square sizes, `grid` at three shapes, `cast`, `gamma`,
//! `join`, `arrayjoin`, `fwfft`, `colourspace`, `composite2`, `spcor`,
//! `fastcor` and every op in `src/bands.rs` all hand `11 / 13` back through
//! the same writer.

use libviprs::{Angle, Angle45, Combine, Interpretation, Kernel, PixelFormat, Precision, Raster};

/// A raster of the given shape with a non-default origin, so a carried offset
/// and a stamped one can never look the same.
fn at_11_13(w: u32, h: u32) -> Raster {
    let data: Vec<u8> = (0..(w as usize * h as usize * 3))
        .map(|i| (i * 7 % 251) as u8)
        .collect();
    Raster::new(w, h, PixelFormat::Rgb8, data)
        .unwrap()
        .copy()
        .xoffset(11)
        .yoffset(13)
        .build()
}

/// An `n`-wide, `m`-tall box mask.
fn box_mask(n: usize, m: usize) -> Kernel {
    Kernel {
        data: vec![vec![1.0; n]; m],
        scale: (n * m) as f64,
    }
}

/// Issue #721. The repositioning ops stamp, and what they stamp is a function
/// of the geometry rather than of the input's offsets.
#[test]
fn flip_rot_and_wrap_stamp_the_origin_offset() {
    for (w, h) in [(5u32, 6u32), (7, 3), (8, 8)] {
        let im = at_11_13(w, h);
        let (wi, hi) = (w as i32, h as i32);
        for (name, want, out) in [
            ("flip horizontal", (wi, 0), im.try_fliphor().unwrap()),
            ("flip vertical", (0, hi), im.try_flipver().unwrap()),
            ("rot d90", (hi, 0), im.try_rot(Angle::D90).unwrap()),
            ("rot d180", (wi, hi), im.try_rot(Angle::D180).unwrap()),
            ("rot d270", (0, wi), im.try_rot(Angle::D270).unwrap()),
            ("wrap", (wi - wi / 2, hi - hi / 2), im.try_wrap().unwrap()),
        ] {
            assert_eq!((out.xoffset(), out.yoffset()), want, "{name} on {w}x{h}");
        }
    }
}

/// Issue #721. The stamp does not read the input's offsets at all, so an
/// input already at the origin gives the same numbers. Without this, "the
/// offsets moved" could be an arithmetic on `11 / 13` rather than a stamp.
#[test]
fn the_stamp_ignores_the_input_offsets() {
    let data = vec![9u8; 5 * 6 * 3];
    let plain = Raster::new(5, 6, PixelFormat::Rgb8, data).unwrap();
    assert_eq!((plain.xoffset(), plain.yoffset()), (0, 0), "control source");
    let tagged = at_11_13(5, 6);

    for (name, a, b) in [
        (
            "flip horizontal",
            plain.try_fliphor().unwrap(),
            tagged.try_fliphor().unwrap(),
        ),
        (
            "rot d90",
            plain.try_rot(Angle::D90).unwrap(),
            tagged.try_rot(Angle::D90).unwrap(),
        ),
        (
            "wrap",
            plain.try_wrap().unwrap(),
            tagged.try_wrap().unwrap(),
        ),
        (
            "conv 3x3",
            plain.try_conv(&box_mask(3, 3), Precision::Float).unwrap(),
            tagged.try_conv(&box_mask(3, 3), Precision::Float).unwrap(),
        ),
        (
            "sobel",
            plain.try_sobel().unwrap(),
            tagged.try_sobel().unwrap(),
        ),
    ] {
        assert_eq!(
            (a.xoffset(), a.yoffset()),
            (b.xoffset(), b.yoffset()),
            "{name} must stamp the same from either source"
        );
    }
}

/// Issue #721. `conv` stamps `(-(mask width / 2), -(mask height / 2))`, and
/// `convsep`, `compass` and `gaussblur` inherit that rule through the
/// convolutions they run rather than having one each.
#[test]
fn the_convolving_ops_stamp_a_mask_relative_origin() {
    for (w, h) in [(5u32, 6u32), (7, 3), (8, 8)] {
        let im = at_11_13(w, h);

        for (mw, mh) in [(3usize, 3usize), (5, 5), (7, 7), (3, 5), (7, 3), (5, 7)] {
            let out = im.try_conv(&box_mask(mw, mh), Precision::Float).unwrap();
            assert_eq!(
                (out.xoffset(), out.yoffset()),
                (-((mw / 2) as i32), -((mh / 2) as i32)),
                "conv {mw}x{mh} on {w}x{h}"
            );
        }

        // The rotated-mask rule: a wide mask stamps on y, a tall one on x.
        for (mw, mh, want) in [(3usize, 1usize, (0, -1)), (1, 3, (-1, 0)), (5, 1, (0, -2))] {
            let out = im.try_convsep(&box_mask(mw, mh), Precision::Float).unwrap();
            assert_eq!(
                (out.xoffset(), out.yoffset()),
                want,
                "convsep {mw}x{mh} on {w}x{h}"
            );
        }

        for (ms, want) in [(3usize, -1i32), (5, -2)] {
            let out = im
                .try_compass(
                    &box_mask(ms, ms),
                    2,
                    Angle45::D45,
                    Combine::Max,
                    Precision::Float,
                )
                .unwrap();
            assert_eq!(
                (out.xoffset(), out.yoffset()),
                (want, want),
                "compass {ms}x{ms} on {w}x{h}"
            );
        }

        // The generated Gaussian mask decides these, so they also pin that
        // libviprs builds the same mask widths vips does: 3, 7 and 11 taps.
        for (sigma, want_y) in [(1.0f64, -1i32), (2.0, -3), (3.0, -5)] {
            let out = im.try_gaussblur(sigma, 0.2, Precision::Float).unwrap();
            assert_eq!(
                (out.xoffset(), out.yoffset()),
                (0, want_y),
                "gaussblur sigma {sigma} on {w}x{h}"
            );
        }
    }
}

/// Issue #721. The edge detectors stamp their gradient mask's rule: `-1 / -1`
/// from a 3x3 for `sobel`, `scharr` and `prewitt`, and `-1 / -1` from canny's
/// 2x2 whatever its blur sigma is.
///
/// Canny is the interesting one: its offset does **not** follow the blur. At
/// sigma 3 the generated blur mask is 11 taps and `gaussblur` alone would
/// stamp `0 / -5`, but `canny` reports `-1 / -1` at sigma 1, 1.4 and 3, so the
/// 2x2 gradient it finishes on is what decides.
#[test]
fn the_edge_detectors_stamp_their_gradient_mask() {
    for (w, h) in [(5u32, 6u32), (7, 3), (8, 8)] {
        let im = at_11_13(w, h);
        for (name, out) in [
            ("sobel", im.try_sobel().unwrap()),
            ("scharr", im.try_scharr().unwrap()),
            ("prewitt", im.try_prewitt().unwrap()),
        ] {
            assert_eq!(
                (out.xoffset(), out.yoffset()),
                (-1, -1),
                "{name} on {w}x{h}"
            );
        }
        for sigma in [1.0f64, 1.4, 3.0] {
            let out = im.try_canny(sigma, Precision::Float).unwrap();
            assert_eq!(
                (out.xoffset(), out.yoffset()),
                (-1, -1),
                "canny sigma {sigma} on {w}x{h}"
            );
        }
    }
}

/// Issue #721. `autorot` composes the primitives, so its offset is whatever
/// the transform it finishes on stamped.
///
/// Orientation 4 is the cell that does not fall out of composition.
/// `vips_autorot` reaches it as a 180-degree rotation followed by a horizontal
/// flip and stamps the flip's `(width, 0)`, where libviprs does it in one
/// vertical flip whose own rule is `(0, height)`. Same pixels, different last
/// transform, so the offset is stamped explicitly there rather than paying for
/// a second pass over the image to make the composition match.
///
/// ```text
/// ori   5x6      7x3      9x4      what vips finishes on
/// 1     11 / 13  11 / 13  11 / 13  nothing, so it carries
/// 2      5 / 0    7 / 0    9 / 0   flip horizontal
/// 3      5 / 6    7 / 3    9 / 4   rot 180
/// 4      5 / 0    7 / 0    9 / 0   flip horizontal, after a rot 180
/// 5      6 / 0    3 / 0    4 / 0   transpose
/// 6      6 / 0    3 / 0    4 / 0   rot 90
/// 7      6 / 0    3 / 0    4 / 0   transverse
/// 8      0 / 5    0 / 7    0 / 9   rot 270
/// ```
#[test]
fn autorot_stamps_the_offset_of_the_transform_it_finishes_on() {
    for (w, h) in [(5u32, 6u32), (7, 3), (9, 4)] {
        let (wi, hi) = (w as i32, h as i32);
        for (ori, want) in [
            (1u8, (11, 13)),
            (2, (wi, 0)),
            (3, (wi, hi)),
            (4, (wi, 0)),
            (5, (hi, 0)),
            (6, (hi, 0)),
            (7, (hi, 0)),
            (8, (0, wi)),
        ] {
            let im = at_11_13(w, h).copy().orientation(ori).build();
            let out = im.try_autorot().unwrap();
            assert_eq!(
                (out.xoffset(), out.yoffset()),
                want,
                "autorot orientation {ori} on {w}x{h}"
            );
            assert_eq!(out.orientation(), 1, "autorot clears the tag");
        }
    }
}

/// Issue #721's positive control, and the reason the table above is a
/// measurement rather than an artefact: everything else hands `11 / 13`
/// straight back through the same code path.
///
/// Without this, a fix that stamped the offset *everywhere* would pass every
/// other test in this file.
#[test]
fn the_ops_that_carry_the_offset_still_carry_it() {
    let im = at_11_13(8, 8);
    let odd = at_11_13(7, 7);
    let tall = at_11_13(4, 12);

    let mut carried: Vec<(String, Raster)> = vec![
        ("cast".into(), im.try_cast(PixelFormat::Rgb16).unwrap()),
        ("gamma".into(), im.try_gamma(None).unwrap()),
        ("falsecolour".into(), im.try_falsecolour().unwrap()),
        ("addalpha".into(), im.try_addalpha().unwrap()),
        ("fwfft".into(), im.try_fwfft().unwrap()),
        (
            "colourspace".into(),
            im.try_colourspace(Interpretation::Lab).unwrap(),
        ),
        ("grid".into(), tall.try_grid(4, 3, 1).unwrap()),
        ("bandjoin".into(), im.try_bandjoin(&im).unwrap()),
        ("bandmean".into(), im.try_bandmean().unwrap()),
        ("bandfold".into(), im.try_bandfold(None).unwrap()),
        ("bandunfold".into(), im.try_bandunfold(None).unwrap()),
        ("bandrank".into(), im.try_bandrank(&[&im], None).unwrap()),
        ("extract_band".into(), im.try_extract_band(1).unwrap()),
        (
            "spcor".into(),
            im.try_spcor(&Raster::new(3, 3, PixelFormat::Rgb8, vec![9u8; 27]).unwrap())
                .unwrap(),
        ),
        (
            "fastcor".into(),
            im.try_fastcor(&Raster::new(3, 3, PixelFormat::Rgb8, vec![9u8; 27]).unwrap())
                .unwrap(),
        ),
    ];

    for angle in [
        Angle45::D45,
        Angle45::D90,
        Angle45::D135,
        Angle45::D180,
        Angle45::D225,
        Angle45::D270,
        Angle45::D315,
    ] {
        carried.push((format!("rot45 {angle:?}"), odd.try_rot45(angle).unwrap()));
    }

    for (name, out) in carried {
        assert_eq!(
            (out.xoffset(), out.yoffset()),
            (11, 13),
            "{name} must carry the offset, not stamp one"
        );
    }
}
