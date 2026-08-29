//! Guards for what this crate says about vips, where it says it (issue #952).
//!
//! Every claim here was true when it was written or was never measured, and
//! each one will mislead the next reader. That is the failure mode the epic
//! kept hitting, and it is not a behaviour bug, so nothing in the ordinary
//! suite is looking at any of it.
//!
//! Two halves per claim wherever both exist, on the shape
//! `the_image_shape_doc_names_exactly_the_containers_that_report_it` uses in
//! `tests/decode_alloc_refusal_shape.rs`: the doc says it, **and** the
//! behaviour it describes is asserted beside it. A doc guard on its own goes
//! stale in the other direction, and a behaviour test on its own is what let
//! four of these six sit unnoticed.
//!
//! `include_str!` rather than a read at runtime, deliberately: it costs
//! nothing at run time, and a test that opened a path would need a
//! `#[cfg_attr(miri, ignore)]` and a row in `tests/miri_fs_test_inventory.txt`,
//! which is a shared count two lanes can move at once.

use std::collections::BTreeSet;

use libviprs::{PixelFormat, Raster};
use serde_json::Value;

const HISTOGRAM_RS: &str = include_str!("../src/histogram.rs");
const EXTRACT_RS: &str = include_str!("../src/extract.rs");
const RESAMPLE_RS: &str = include_str!("../src/resample.rs");
const ARITHMETIC_RS: &str = include_str!("../src/arithmetic.rs");
const ENCODE_RS: &str = include_str!("../src/encode.rs");
const ORACLE_PIN_JSON: &str = include_str!("../oracle-captures/ORACLE_PIN.json");

/// The doc block immediately above `needle` in `src`.
///
/// Anchored on the **start** of the doc block rather than on the signature,
/// because the signature line has the doc above it: a scan that starts at
/// `fn <name>` reads the *next* item's doc and reports on the wrong thing.
/// Walks back over `///` lines from the anchor and returns them in order.
fn doc_above(src: &str, needle: &str) -> String {
    let at = src
        .find(needle)
        .unwrap_or_else(|| panic!("{needle:?} is not in this file"));
    let head = &src[..at];
    let mut lines: Vec<&str> = Vec::new();
    for line in head.lines().rev() {
        let t = line.trim_start();
        if t.starts_with("///") {
            lines.push(t);
        } else if t.starts_with("#[") || t.starts_with("//") || t.is_empty() {
            // An attribute or a plain comment between the doc and the item.
            if t.is_empty() && !lines.is_empty() {
                break;
            }
        } else {
            break;
        }
    }
    lines.reverse();
    lines.join("\n")
}

/// A one-band `Uint32` histogram from counts.
fn uint_hist(counts: &[u32]) -> Raster {
    let n = core::num::NonZeroU16::new(1).expect("one band");
    let data: Vec<u8> = counts.iter().flat_map(|c| c.to_ne_bytes()).collect();
    Raster::new(counts.len() as u32, 1, PixelFormat::Uint32(n), data).expect("histogram fixture")
}

/// A 2x1 `Gray8` raster.
fn gray8(vals: &[u8]) -> Raster {
    Raster::new(vals.len() as u32, 1, PixelFormat::Gray8, vals.to_vec()).expect("gray8 fixture")
}

// ---------------------------------------------------------------------------
// 1. hist_ismonotonic says "like vips" about an oracle it contradicts
// ---------------------------------------------------------------------------

/// Issue #952. The `hist_ismonotonic` divergence is recorded where a **user**
/// looks, not only in a test's doc block that rustdoc never renders.
///
/// The public method said "like `vips_hist_ismonotonic`" and the module
/// parity table mapped it with no marker, while the measured table lived in
/// `a_count_comparison_sees_counts_above_65535`'s doc, inside `mod tests`.
/// That is the "divergence recorded where a user cannot see it" failure one
/// door over from a divergence recorded in a PR body.
///
/// Measured on `/opt/homebrew/bin/vips` 8.18.6: `hist_ismonotonic` answers
/// **TRUE** for the strictly decreasing `uint` histogram `[70000, 65000]` and
/// for `[400000, 300000, 200000, 100000]`, and FALSE for
/// `[100000, 100000, 100000, 99999]`. So it calls two strictly decreasing
/// sequences monotonic while catching a decrease of one. This crate answers
/// correctly, which is posture 4: matching an oracle that contradicts itself
/// is not parity.
#[test]
fn the_hist_ismonotonic_divergence_is_in_the_public_doc() {
    let doc = doc_above(HISTOGRAM_RS, "pub fn try_hist_ismonotonic");
    for needle in ["70000", "65000", "8.18.6", "Divergence"] {
        assert!(
            doc.contains(needle),
            "try_hist_ismonotonic's public doc must carry the measured \
             divergence and name {needle:?}; the numbers live in a test doc \
             block rustdoc never renders (issue #952). Doc was:\n{doc}"
        );
    }

    // The module parity table maps the op to `vips_hist_ismonotonic` with no
    // marker, which is the row a reader scans before they open the method.
    let row = HISTOGRAM_RS
        .lines()
        .find(|l| l.contains("Raster::hist_ismonotonic"))
        .expect("the module table has a hist_ismonotonic row");
    assert!(
        row.contains("diverges") || row.contains("#952"),
        "the module parity table row must mark the divergence rather than \
         mapping it as a plain equivalent: {row:?}"
    );

    // The executable half, so the doc is not the only thing saying it. The
    // crate answers `false` where vips answers TRUE.
    assert!(
        !uint_hist(&[70_000, 65_000])
            .try_hist_ismonotonic()
            .expect("a 2x1 uint raster is histogram-shaped"),
        "a strictly decreasing histogram is not monotonic, whatever vips says"
    );
    assert!(
        !uint_hist(&[400_000, 300_000, 200_000, 100_000])
            .try_hist_ismonotonic()
            .expect("histogram-shaped"),
        "the second sequence vips calls monotonic is decreasing too"
    );
    // The positive control: an increasing histogram is monotonic, so the two
    // assertions above are not passing because the answer is always false.
    assert!(
        uint_hist(&[65_000, 70_000])
            .try_hist_ismonotonic()
            .expect("histogram-shaped")
    );
    // And the row vips gets right, so the divergence is three cells rather
    // than a blanket disagreement.
    assert!(
        !uint_hist(&[100_000, 100_000, 100_000, 99_999])
            .try_hist_ismonotonic()
            .expect("histogram-shaped")
    );
}

// ---------------------------------------------------------------------------
// 2. two accounts of the #692 white-ink mechanism, and the older one is refuted
// ---------------------------------------------------------------------------

/// Issue #952. `src/extract.rs` and `src/resample.rs` describe the same vips
/// mechanism, so they have to describe it the same way.
///
/// `extract.rs` said vips premultiplies into float **before** it paints the
/// affine border, so `FILL_LINE(float, ...)` runs and the memset never
/// happens. #692's own closing measurement refuted that: `vips_affine_build`
/// embeds before it premultiplies on every path (`affine.c:529` then `:551`),
/// so the ink is memset into the raster's own domain either way, and what
/// moves it is the non-cancelling clipped-alpha round trip. #745 corrected
/// the resample side and left the extract side stating the refuted story.
///
/// Both docs are `pub(crate)`, so no user reads either. A maintainer
/// following `extract.rs` re-derives a wrong model of vips, which is worse
/// than no model.
#[test]
fn the_white_ink_mechanism_reads_the_same_in_extract_and_in_resample() {
    // The refuted mechanism, in the words it was written in.
    assert!(
        !EXTRACT_RS.contains("the memset above never happens"),
        "src/extract.rs still claims the affine premultiply skips the memset, \
         which #692's closing measurement refuted (issue #952)"
    );

    // #692 is closed, so nothing tracks anything.
    for site in EXTRACT_RS.match_indices("#692") {
        let tail = &EXTRACT_RS[site.0..(site.0 + 60).min(EXTRACT_RS.len())];
        assert!(
            !tail.contains("tracks"),
            "issue #692 is closed and tracks nothing; src/extract.rs still \
             points at it: {tail:?}"
        );
    }

    // And the corrected account has to be the one both files carry. The
    // phrase is the **arithmetic**, `clip(E, 0, M)`, and not the English
    // around it: the first draft of this asserted "does not cancel", which
    // the `**not**` markup in both files breaks into three tokens, so the
    // assertion was passing on the fallback word `clipped` alone and would
    // have passed on a file that said nothing about the mechanism.
    const MECHANISM: &str = "clip(E, 0, M)";
    assert!(
        RESAMPLE_RS.contains(MECHANISM),
        "the corrected account lives in src/resample.rs; if this fails the \
         scan is looking for the wrong phrase, not the docs disagreeing"
    );
    assert!(
        EXTRACT_RS.contains(MECHANISM),
        "src/extract.rs must carry the corrected account too, and the \
         arithmetic is what says it (issue #952)"
    );
}

// ---------------------------------------------------------------------------
// 3. try_add_const cross-references an op it measurably is not
// ---------------------------------------------------------------------------

/// Issue #952. `add_const` is not `vips linear` with `a = 1`, and its doc
/// points float-seeking callers at [`Raster::linear`], which is.
///
/// Measured on `/opt/homebrew/bin/vips` 8.18.6 over a 2x1 `uchar` raster
/// `[200, 100]`:
///
/// | call | crate | `vips linear` |
/// |---|---|---|
/// | `+ 5` | `Gray16 [205, 105]` | `FLOAT [205, 105]` |
/// | `+ 0.5` | `Gray16 [201, 101]` | `FLOAT [200.5, 100.5]` |
/// | `- 300` | `Gray8 [0, 0]` | `FLOAT [-100, -200]` |
///
/// So the integer dialect is deliberate and documented, and the cross
/// reference is the thing that is wrong: a caller who wants what
/// `vips linear` gives wants [`Raster::linear`], which this crate has and
/// which was verified matching.
#[test]
fn add_const_is_the_integer_dialect_and_linear_is_the_vips_twin() {
    let im = gray8(&[200, 100]);

    let plus5 = im.add_const(5.0);
    assert_eq!(plus5.format(), PixelFormat::Gray16);
    assert_eq!(plus5.getpoint(0, 0), vec![205.0]);
    assert_eq!(plus5.getpoint(1, 0), vec![105.0]);

    // Rounded, where vips answers 200.5.
    let plus_half = im.add_const(0.5);
    assert_eq!(plus_half.getpoint(0, 0), vec![201.0]);
    assert_eq!(plus_half.getpoint(1, 0), vec![101.0]);

    // Saturated at zero, where vips answers -100 and -200.
    let minus = im.sub_const(300.0);
    assert_eq!(minus.format(), PixelFormat::Gray8);
    assert_eq!(minus.getpoint(0, 0), vec![0.0]);
    assert_eq!(minus.getpoint(1, 0), vec![0.0]);

    // The twin that does answer what vips answers, on all three rows.
    let lin = im.linear(1.0, 5.0);
    assert!(lin.format().is_float(), "got {:?}", lin.format());
    assert_eq!(lin.getpoint(0, 0), vec![205.0]);
    assert_eq!(lin.getpoint(1, 0), vec![105.0]);
    assert_eq!(im.linear(1.0, 0.5).getpoint(0, 0), vec![200.5]);
    assert_eq!(im.linear(1.0, -300.0).getpoint(0, 0), vec![-100.0]);
    assert_eq!(im.linear(1.0, -300.0).getpoint(1, 0), vec![-200.0]);

    // And the doc half: `add_const` must send a float-seeking caller to the
    // op that is the twin rather than claim to be it.
    let doc = doc_above(ARITHMETIC_RS, "pub fn try_add_const");
    assert!(
        doc.contains("Raster::linear"),
        "try_add_const's doc must point at Raster::linear, which is the op \
         that answers what `vips linear` answers (issue #952). Doc was:\n{doc}"
    );
}

// ---------------------------------------------------------------------------
// 4. the .hdr save refusal does not mention vips, where its siblings do
// ---------------------------------------------------------------------------

/// Issue #952. The `.hdr` save refusal names the divergence the way its
/// `.ppm` / `.pgm` siblings do.
///
/// Both are posture 1 held as policy: vips converts to suit the container and
/// libviprs does not. The Netpbm refusal says so in the error itself; the
/// Radiance one said only what libviprs wants, which leaves a caller with no
/// way to know that `vips radsave` would have taken the same image.
///
/// Measured on `/opt/homebrew/bin/vips` 8.18.6: `vips radsave` on a 3x1
/// `uchar` `srgb` image writes a 238-byte `.hdr` that loads back as
/// `3x1 rad, 4 bands`, with the samples RGBE-quantised (10 -> 9.96875,
/// 200 -> 199.5, 30 -> 29.9375). So it accepts the image this refuses.
#[test]
fn the_hdr_save_refusal_names_vips_like_its_netpbm_siblings() {
    let rgb = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 200, 30]).expect("rgb8 fixture");
    let err = rgb
        .encode_radiance(libviprs::radiance::SaveOptions::default())
        .expect_err("radiance carries three float bands");
    let msg = err.to_string();
    assert!(
        msg.contains("vips"),
        "the .hdr refusal must say what vips does with the same image, the way \
         the Netpbm one does (issue #952): {msg:?}"
    );

    // The sibling it is being aligned with, as the control that this is a
    // house convention and not a one-off: the Netpbm refusal already says it.
    let mono = Raster::new(1, 1, PixelFormat::Gray8, vec![7]).expect("gray8 fixture");
    let netpbm = mono
        .encode_to_buffer("ppm")
        .expect_err(".ppm is the P6 container and this raster has one band");
    assert!(
        netpbm.to_string().contains("vips"),
        "the .ppm refusal is the sibling this is matching: {netpbm}"
    );

    // The positive control that the refusal is about the carrier and not
    // about everything: the format the encoder does carry goes through.
    let n = core::num::NonZeroU16::new(3).expect("three bands");
    let float3 = Raster::new(
        1,
        1,
        PixelFormat::FloatF32(n),
        [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect(),
    )
    .expect("float3 fixture");
    assert!(
        float3
            .encode_radiance(libviprs::radiance::SaveOptions::default())
            .is_ok()
    );
}

// ---------------------------------------------------------------------------
// 5. the PNG integer refusals are argued from the dependency, not the oracle
// ---------------------------------------------------------------------------

/// Issue #952. The refusal of `uint`, float and the signed carriers for an
/// `image`-crate encode carries the measured oracle, not only the argument
/// from the dependency.
///
/// "The `image` crate's widest integer colour type is 16-bit" is true and it
/// is the weaker half of the case. Measured on `/opt/homebrew/bin/vips`
/// 8.18.6 over a 2x1 raster, the oracle does not agree with itself:
///
/// | route | `uint [3000000000, 100]` | `float [1.5, -0.25]` |
/// |---|---|---|
/// | `pngsave`, `b-w` tag | `[0, 100]` | `[1, 0]` |
/// | `pngsave`, `multiband` tag | `[0, 0]` | `[0, 0]` |
/// | `cast` to `uchar` | `[0, 100]` | `[1, 0]` |
/// | `dzsave` full-resolution tile | `0` | `1` |
/// | `dzsave` overview tile | `255` | `0` |
///
/// Three routes, and the interpretation tag moves the answer inside one of
/// them. No route answers the data. That makes the refusal *more* faithful
/// than the dependency argument makes it, and none of the numbers were in
/// the tree.
#[test]
fn the_png_integer_refusals_carry_the_oracle_not_only_the_dependency() {
    let doc = doc_above(ENCODE_RS, "fn image_color_type");
    for needle in ["8.18.6", "3000000000", "dzsave"] {
        assert!(
            doc.contains(needle),
            "image_color_type's doc must carry the measured oracle and name \
             {needle:?} (issue #952). Doc was:\n{doc}"
        );
    }

    // The executable half: the carriers the doc is about are refused, and
    // the ones it is not about go through, so this is a refusal of a kind
    // rather than of anything unusual.
    let n = |v: u16| core::num::NonZeroU16::new(v).expect("non-zero");
    let uint = Raster::new(
        1,
        1,
        PixelFormat::Uint32(n(1)),
        3_000_000_000u32.to_ne_bytes().to_vec(),
    )
    .expect("uint fixture");
    assert!(uint.encode_to_buffer("png").is_err());
    let float = Raster::new(
        1,
        1,
        PixelFormat::FloatF32(n(1)),
        1.5f32.to_ne_bytes().to_vec(),
    )
    .expect("float fixture");
    assert!(float.encode_to_buffer("png").is_err());
    // Positive control.
    assert!(gray8(&[7, 8]).encode_to_buffer("png").is_ok());
    assert!(
        Raster::new(1, 1, PixelFormat::Gray16, 300u16.to_ne_bytes().to_vec())
            .expect("gray16 fixture")
            .encode_to_buffer("png")
            .is_ok()
    );
}

// ---------------------------------------------------------------------------
// 6. oracle pin staleness is permanent rather than caught
// ---------------------------------------------------------------------------

/// Issue #952. An area whose capture is not on the pin says **why**, so the
/// pin file stops reading as though the reconciliation is pending.
///
/// Six areas record 8.18.4 with a state of `pre_pin`, and 8.18.4 cannot be
/// installed from the current tap. #650, which tracked the reconciliation, is
/// closed. So the guard is green because the pin file declares 8.18.4 for
/// those areas, which makes the staleness permanent rather than caught: the
/// one state word says "not yet moved" and means "will not move".
///
/// The check is that every off-pin area carries a non-empty `note`. That
/// cannot be satisfied by a state word alone, and it is the sentence a reader
/// needs: what a re-measure found, or what stops one.
#[test]
fn every_off_pin_capture_area_says_why_it_is_off_the_pin() {
    let pin: Value = serde_json::from_str(ORACLE_PIN_JSON).expect("ORACLE_PIN.json parses");
    let pinned = pin["pinned_vips_version"]
        .as_str()
        .expect("a string pinned_vips_version");
    let areas = pin["areas"].as_object().expect("an areas object");

    let mut off_pin = BTreeSet::new();
    for (area, entry) in areas {
        let version = entry["vips_version"]
            .as_str()
            .unwrap_or_else(|| panic!("area {area} needs a string vips_version"));
        let state = entry["state"]
            .as_str()
            .unwrap_or_else(|| panic!("area {area} needs a string state"));
        if version == pinned {
            assert_eq!(
                state, "on_pin",
                "area {area} records the pinned build, so its state is on_pin"
            );
            continue;
        }
        off_pin.insert(area.clone());
        assert!(
            matches!(state, "pre_pin" | "frozen"),
            "area {area} is off the pin with state {state:?}; use pre_pin \
             (a move is pending) or frozen (it is not)"
        );
        let note = entry["note"].as_str().unwrap_or("");
        assert!(
            note.len() > 40,
            "area {area} is off the pin at {version} and says nothing about \
             why. 8.18.4 cannot be installed, so a state word alone leaves the \
             staleness permanent rather than caught (issue #952)"
        );
    }

    // The predicate has no passing-on-nothing case: the six areas that are
    // off the pin are named here, so an area silently leaving the set (or
    // arriving in it) reddens rather than passing vacuously.
    let want: BTreeSet<String> = [
        "foreign-exr",
        "foreign-fits",
        "foreign-gif",
        "foreign-jxl",
        "foreign-radiance",
        "foreign-webp",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    assert_eq!(
        off_pin, want,
        "the set of areas off the pin changed; update this list in the same \
         commit that moves one, so 'nothing violates the rule' cannot be \
         confused with 'nothing was examined'"
    );
}
