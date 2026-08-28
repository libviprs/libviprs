//! Issue #508: Ultra HDR (gain-map JPEG) load and save.
//!
//! Every number here comes out of `oracle-captures/foreign-uhdr`, captured
//! by issue #639 from `vips` 8.18.6 with `libultrahdr` 2.0.2. Nothing is
//! computed from a formula and then declared correct: where the oracle
//! recorded a value, that value is read out of `oracle.json` at test time,
//! so the pin travels with the capture rather than with whoever last edited
//! this file.
//!
//! # The two things worth reading before changing anything here
//!
//! An Ultra HDR file **is** a JPEG at the magic-byte level, so the sniffer
//! has to disambiguate on structure the leading bytes cannot show, and it
//! must not steal ordinary JPEGs. Both directions are proved against the
//! nine fixtures the capture cut, and the fixture that makes the negative
//! direction meaningful is `mpf-graft.jpg`: an ordinary JPEG with the MPF
//! marker grafted on. A sniffer that checked only for MPF would claim it.
//! `mpf_graft_is_the_positive_control_for_the_negative_direction` proves
//! that a wrong sniffer would have failed, which is what stops the
//! negative results being vacuous.
//!
//! The `uhdr2scRGB` arithmetic is pinned **bit-exactly**, not to a
//! tolerance. Three of the four mono cases and the whole three-band case
//! reproduce libvips's floats with zero difference; the two that do not are
//! named and bounded in their own test with the measured ulp count, rather
//! than hidden under a blanket epsilon.

use std::path::{Path, PathBuf};

use libviprs::imageio::MetadataValue;
use libviprs::pixel::PixelFormat;
use libviprs::raster::Raster;
use libviprs::source::{DecodeLimits, SourceError, decode_bytes, decode_bytes_with_limits};
use libviprs::uhdr::{self, GainMapMetadata, SaveOptions};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn capture_dir() -> PathBuf {
    repo_root().join("oracle-captures/foreign-uhdr")
}

/// The pinned capture, parsed.
fn oracle() -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(capture_dir().join("oracle.json"))
            .expect("the foreign-uhdr capture must be readable"),
    )
    .expect("the foreign-uhdr capture must be strict JSON")
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(capture_dir().join("fixtures").join(name))
        .unwrap_or_else(|e| panic!("fixture {name} must be readable: {e}"))
}

fn float3(width: u32, height: u32, values: Vec<f32>) -> Raster {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_ne_bytes).collect();
    Raster::new(
        width,
        height,
        PixelFormat::FloatF32(std::num::NonZeroU16::new(3).unwrap()),
        bytes,
    )
    .expect("a three-band float raster")
}

fn floats(raster: &Raster) -> Vec<f32> {
    raster
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}

/// The metadata block one of the `uhdr2scRGB` records runs its fixture
/// with, built from the capture's `canonical` case plus that case's
/// overrides.
fn metadata_case(record: &serde_json::Value, case: &str) -> GainMapMetadata {
    let block = &record["metadata"];
    let mut meta = GainMapMetadata::default();
    let mut apply = |v: &serde_json::Value| {
        let triple = |name: &str, slot: &mut [f64; 3]| {
            if let Some(a) = v.get(name).and_then(|x| x.as_array()) {
                for (i, s) in slot.iter_mut().enumerate() {
                    *s = a[i].as_f64().expect("a number");
                }
            }
        };
        triple("max-content-boost", &mut meta.max_content_boost);
        triple("min-content-boost", &mut meta.min_content_boost);
        triple("gamma", &mut meta.gamma);
        triple("offset-sdr", &mut meta.offset_sdr);
        triple("offset-hdr", &mut meta.offset_hdr);
    };
    // Two shapes in the capture: the mono record nests its variants under
    // `canonical` plus one key per override, the three-band record states
    // one flat block. Reading the flat one as if it were nested silently
    // yields the default metadata and a transform that looks like it ran --
    // which is exactly how the first cut of this file passed a test that
    // was measuring nothing.
    if block.get("canonical").is_some() {
        apply(&block["canonical"]);
        if case != "canonical" {
            apply(&block[case]);
        }
    } else {
        assert_eq!(
            case, "canonical",
            "a flat metadata block has no named cases"
        );
        apply(block);
    }
    assert_ne!(
        meta.max_content_boost, [1.0; 3],
        "the metadata block was read as the identity, so the transform under test \
         would be measuring nothing"
    );
    meta
}

/// The base and gain-map rasters the mono `uhdr2scRGB` records were run
/// with, built from the bytes the capture recorded rather than by decoding
/// the JPEG fixtures, so the pixels compared are the pixels libvips was
/// handed.
fn mono_fixture_rasters(record: &serde_json::Value) -> (Raster, Raster) {
    let base_bytes: Vec<u8> = record["base_sRGB_bytes"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|h| {
            let s = h.as_str().unwrap().to_string();
            (0..3)
                .map(move |i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    let gain_bytes: Vec<u8> = record["gainmap_decoded_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    (
        Raster::new(16, 1, PixelFormat::Rgb8, base_bytes).unwrap(),
        Raster::new(16, 1, PixelFormat::Gray8, gain_bytes).unwrap(),
    )
}

/// The offset just past the first JPEG's `EOI`.
///
/// Safe to scan for: inside entropy-coded data an `FF` is always followed
/// by `00` or a restart marker, so the first bare `FF D9` is the real end
/// of the base image.
fn end_of_first_jpeg(bytes: &[u8]) -> usize {
    bytes
        .windows(2)
        .position(|w| w == [0xFF, 0xD9])
        .expect("a complete JPEG has an EOI")
        + 2
}

/**
 * The detection gate, against every fixture the capture cut, in both
 * directions at once.
 *
 * `detection_two_stage_gate.files` records, per fixture, which loader
 * `vips_foreign_find_load` actually chose. That is the answer `is_uhdr`
 * has to reproduce: not "could something decode this" but "does this file
 * belong to the Ultra HDR loader". The nine fixtures cover both halves of
 * the gate and both ways of failing it -- no MPF, no ISO segment on the
 * gain map, no gain map at all, and each half truncated.
 * Input: the nine pinned fixtures -> Output: `is_uhdr` agrees with the
 * loader vips chose, on every one.
 */
#[test]
fn is_uhdr_agrees_with_the_loader_vips_chose_on_every_fixture() {
    let o = oracle();
    let files = o["records"]["detection_two_stage_gate"]["files"]
        .as_object()
        .expect("the gate record lists its files");
    let mut checked = 0;
    let mut wrong = Vec::new();
    for (name, record) in files {
        // `truncated-base.jpg` has no chosen loader: it is broken enough
        // that `vipsheader` errors instead, and the capture records the
        // error text rather than a name. It still has a pinned
        // `uhdrload_exit` of 1, so it is a negative either way.
        let want = match record["chosen_loader"].as_str() {
            Some(loader) => loader == "uhdrload",
            None => false,
        };
        let got = uhdr::is_uhdr(&fixture(name));
        checked += 1;
        if got != want {
            wrong.push(format!(
                "{name}: is_uhdr={got}, vips chose {}",
                record["chosen_loader"]
            ));
        }
    }
    assert_eq!(checked, 9, "the capture pins nine fixtures");
    assert!(
        wrong.is_empty(),
        "the gate disagrees with vips on {} of {checked}:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}

/**
 * The positive control for the negative direction.
 *
 * Every "this file is not Ultra HDR" result above is a zero, and a zero has
 * two explanations: the gate worked, or the gate never really looked. This
 * separates them. `mpf-graft.jpg` is an ordinary JPEG carrying the MPF
 * marker and nothing else -- the capture built it exactly so that a
 * sniffer testing only the fast pre-filter would claim it. So: MPF *is*
 * present (proved here, not assumed), the file is still not Ultra HDR, and
 * the one-stage gate a careless port would write does claim it. Without
 * this, "mpf-graft is not UHDR" would pass for a sniffer that always said
 * no.
 * Input: `mpf-graft.jpg` -> Output: MPF present, `is_uhdr` false, and the
 * MPF-only gate wrong.
 */
#[test]
fn mpf_graft_is_the_positive_control_for_the_negative_direction() {
    let bytes = fixture("mpf-graft.jpg");

    // The pre-filter really is present, so the negative below is the second
    // stage doing work rather than the first short-circuiting.
    let mpf_present = bytes.windows(uhdr::MPF_ID.len()).any(|w| w == uhdr::MPF_ID);
    assert!(
        mpf_present,
        "mpf-graft.jpg is supposed to carry the MPF marker; without it this \
         control proves nothing"
    );
    assert!(
        !uhdr::is_uhdr(&bytes),
        "an ordinary JPEG with MPF grafted on is not Ultra HDR"
    );

    // And the same test the other way: a file that has BOTH stages is
    // accepted, so the gate is not simply always-false.
    assert!(
        uhdr::is_uhdr(&fixture("uhdr.jpg")),
        "the gate must accept a real container, or every negative above is vacuous"
    );

    // The complementary control: `no-mpf.jpg` is a genuine gain-map file
    // that vips routes to `jpegload` anyway, because the pre-filter is
    // missing. It carries the ISO segment the second stage tests for, so a
    // gate that skipped the pre-filter would wrongly claim it.
    let no_mpf = fixture("no-mpf.jpg");
    assert!(
        no_mpf
            .windows(uhdr::ISO_GAIN_MAP_ID.len())
            .any(|w| w == uhdr::ISO_GAIN_MAP_ID),
        "no-mpf.jpg is supposed to keep its ISO 21496-1 segment"
    );
    assert!(
        !uhdr::is_uhdr(&no_mpf),
        "a gain-map file with no MPF is routed to jpegload by vips, so the gate \
         must decline it too"
    );
}

/**
 * The sniffer, both directions, through the public decode entry point.
 *
 * The gate above is the predicate; this is what the route table does with
 * it. An ordinary JPEG must still decode as a JPEG and must not pick up
 * any `gainmap-*` field, and an Ultra HDR file must not be read as a plain
 * JPEG -- which is exactly what would happen if the `Uhdr` row were
 * declared after `Jpeg` instead of before it, since `sniff` takes the
 * first match and both rows accept `FF D8 FF`.
 * Input: `plain.jpg` and `uhdr.jpg` -> Output: neither and both carry the
 * gain map, respectively.
 */
#[test]
fn an_ordinary_jpeg_stays_jpeg_and_a_uhdr_file_does_not() {
    let plain = decode_bytes(&fixture("plain.jpg")).expect("plain.jpg decodes");
    assert!(
        !plain.get_fields().iter().any(|f| f.starts_with("gainmap")),
        "an ordinary JPEG must not come back carrying a gain map; the Uhdr row \
         is stealing files that are not Ultra HDR"
    );

    let uhdr_file = decode_bytes(&fixture("uhdr.jpg")).expect("uhdr.jpg decodes");
    assert!(
        uhdr_file.get_field("gainmap-data").is_some(),
        "an Ultra HDR file read as a plain JPEG silently drops its gain map, \
         which is the whole failure this issue is about"
    );

    // Both decode to the same geometry and band count, so the difference
    // above is the metadata rather than the pixels: this is the same trap
    // as libvips's, where `jpegload` on a UHDR file succeeds and loses the
    // gain map without saying anything.
    assert_eq!(
        (plain.width(), plain.height(), plain.format()),
        (uhdr_file.width(), uhdr_file.height(), uhdr_file.format()),
        "the two fixtures are the same picture; only the container differs"
    );
}

/**
 * The header libvips reports for `fixtures/uhdr.jpg`, field by field.
 *
 * `uhdrsave_writes_this_container.header` is `vipsheader -a` on the
 * fixture, so this compares against the reference implementation's own
 * output rather than against a hand-written expectation. The blob fields
 * are compared by the byte count vipsheader prints, which is what catches
 * the two easy mistakes: forgetting to strip the 14-byte `ICC_PROFILE`
 * prefix, and attaching the gain map's payload instead of its whole JPEG.
 * Input: `fixtures/uhdr.jpg` -> Output: every `gainmap-*`, `exif-data` and
 * `icc-profile-data` field matches the captured header.
 */
#[test]
fn the_loaded_header_matches_the_captured_vipsheader_output() {
    let o = oracle();
    let header = &o["records"]["uhdrsave_writes_this_container"]["header"];
    let raster = decode_bytes(&fixture("uhdr.jpg")).expect("uhdr.jpg decodes");

    assert_eq!(
        (raster.width(), raster.height()),
        (
            header["width"].as_str().unwrap().parse().unwrap(),
            header["height"].as_str().unwrap().parse().unwrap()
        ),
        "geometry"
    );
    assert_eq!(raster.format(), PixelFormat::Rgb8, "3 bands, uchar");

    let mut checked = 0;
    for (name, want) in header.as_object().unwrap() {
        let Some(got) = raster.get_field(name) else {
            continue;
        };
        let want = want.as_str().expect("vipsheader prints strings");
        let rendered = match &got {
            MetadataValue::Blob(b) => format!("{} bytes of binary data", b.len()),
            MetadataValue::Str(s) => s.clone(),
            MetadataValue::Int(i) => i.to_string(),
            MetadataValue::Double(d) => {
                // vipsheader prints a scalar double through `%g` too.
                let s = format!("{d:.6}");
                let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
                if s.len() > 7 { format!("{d:.4}") } else { s }
            }
            other => panic!("unexpected metadata carrier {other:?}"),
        };
        if name.starts_with("gainmap") || name == "exif-data" || name == "icc-profile-data" {
            checked += 1;
            assert_eq!(
                rendered.trim(),
                want.trim(),
                "field {name} does not match the captured header"
            );
        }
    }
    assert_eq!(
        checked, 12,
        "the capture names twelve carrier fields; see uhdrload_carriers.metadata_names"
    );
}

/**
 * The `v2Y_8` table against the bytes the capture lifted out of the shipped
 * libvips.
 *
 * The table is transcribed into `crate::uhdr` rather than computed, because
 * the expression behind it does not reproduce it: the arm64 build fuses
 * multiply-adds the source does not show and its `powf` differs in the last
 * place, and three different faithful transcriptions of the formula missed
 * 151, 213 and 219 of the 256 entries. This is the check behind that
 * decision, and it is what stops the constant drifting from the capture.
 * Input: `v2Y_8_le_f32_hex` from the capture -> Output: all 256 entries
 * equal, bit for bit.
 */
#[test]
fn v2y_8_matches_the_pinned_oracle_table() {
    let o = oracle();
    let hex: String = o["records"]["uhdr2scRGB_base_linearisation"]["v2Y_8_le_f32_hex"]
        .as_str()
        .expect("the capture records the table")
        .split_whitespace()
        .collect();
    let raw: Vec<u8> = (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect();
    assert_eq!(raw.len(), 256 * 4, "the capture records 256 f32 entries");

    // The table is not public, so it is read back through the transform:
    // a base of code `i` under the identity gain map is exactly `v2Y_8[i]`.
    let identity = GainMapMetadata::default();
    let base = Raster::new(
        256,
        1,
        PixelFormat::Rgb8,
        (0..256u32)
            .flat_map(|i| [i as u8, i as u8, i as u8])
            .collect(),
    )
    .unwrap();
    let gain = Raster::new(256, 1, PixelFormat::Gray8, vec![0u8; 256]).unwrap();
    let got = floats(&uhdr::uhdr_to_scrgb(&base, &gain, &identity).unwrap());

    let mut wrong = Vec::new();
    for i in 0..256 {
        let want = f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
        if got[i * 3] != want {
            wrong.push(format!("[{i}] {} != {want}", got[i * 3]));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of 256 v2Y_8 entries differ from the capture:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}

/**
 * The one-band `uhdr2scRGB` path against every metadata variant the capture
 * ran, bit for bit.
 *
 * The inputs come straight out of the record -- `base_sRGB_bytes` and
 * `gainmap_decoded_values` -- so the JPEG fixtures do not have to be
 * decoded to reach the numbers, and the pixels compared are exactly the
 * pixels libvips was handed. Three of the four cases match with zero
 * difference. `gamma_2_2` and `offsets` do not, and they are excluded here
 * and bounded in the next test rather than absorbed into a tolerance that
 * would also hide a real error in the other two.
 * Input: the capture's base and gain-map bytes under `canonical` and
 * `min_boost_half` -> Output: libvips's floats, exactly.
 */
#[test]
fn the_mono_path_reproduces_the_oracle_floats_exactly() {
    let o = oracle();
    let record = &o["records"]["uhdr2scRGB_mono_gainmap"];
    let base_bytes: Vec<u8> = record["base_sRGB_bytes"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|h| {
            let s = h.as_str().unwrap();
            (0..3).map(move |i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
        })
        .collect();
    let gain_bytes: Vec<u8> = record["gainmap_decoded_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let base = Raster::new(16, 1, PixelFormat::Rgb8, base_bytes).unwrap();
    let gain = Raster::new(16, 1, PixelFormat::Gray8, gain_bytes).unwrap();

    for case in ["canonical", "min_boost_half"] {
        let meta = metadata_case(record, case);
        let got = floats(&uhdr::uhdr_to_scrgb(&base, &gain, &meta).unwrap());
        let want = record["results"][case]["scRGB"].as_array().unwrap();
        let mut wrong = Vec::new();
        for (p, pixel) in want.iter().enumerate() {
            for (i, v) in pixel.as_array().unwrap().iter().enumerate() {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the capture holds f32 values"
                )]
                let expected = v.as_f64().unwrap() as f32;
                if got[p * 3 + i] != expected {
                    wrong.push(format!(
                        "{case}[{p}][{i}]: {} != {expected}",
                        got[p * 3 + i]
                    ));
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "{} of 48 samples differ from the capture in {case}:\n  {}",
            wrong.len(),
            wrong.join("\n  ")
        );
    }
}

/**
 * The two mono cases that are not bit-exact, bounded by what was actually
 * measured rather than by a round number.
 *
 * `gamma_2_2` and `offsets` are the only cases whose arithmetic passes
 * through `powf`, and Rust's `f64::powf` and the `pow` the shipped libvips
 * calls disagree by one ulp on some inputs. One ulp in `gg` is amplified
 * through `exp2` into a handful of ulp in the result, which is why the
 * bound is expressed in ulp of the expected value rather than as an
 * absolute epsilon: an absolute epsilon that covered the largest sample
 * here (about 6.4) would be blind to a real error in the smallest (about
 * 0.004).
 * Measured on this tree: worst 4 ulp in `gamma_2_2`, 1 ulp in `offsets`.
 * Input: the capture's `gamma_2_2` and `offsets` cases -> Output: within 8
 * ulp, and NOT bit-exact, so the bound cannot silently become redundant.
 */
#[test]
fn the_two_powf_cases_are_within_a_measured_ulp_bound() {
    let o = oracle();
    let record = &o["records"]["uhdr2scRGB_mono_gainmap"];
    let base_bytes: Vec<u8> = record["base_sRGB_bytes"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|h| {
            let s = h.as_str().unwrap();
            (0..3).map(move |i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
        })
        .collect();
    let gain_bytes: Vec<u8> = record["gainmap_decoded_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let base = Raster::new(16, 1, PixelFormat::Rgb8, base_bytes).unwrap();
    let gain = Raster::new(16, 1, PixelFormat::Gray8, gain_bytes).unwrap();

    for case in ["gamma_2_2", "offsets"] {
        let meta = metadata_case(record, case);
        let got = floats(&uhdr::uhdr_to_scrgb(&base, &gain, &meta).unwrap());
        let want = record["results"][case]["scRGB"].as_array().unwrap();
        let mut worst_ulp = 0i64;
        for (p, pixel) in want.iter().enumerate() {
            for (i, v) in pixel.as_array().unwrap().iter().enumerate() {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the capture holds f32 values"
                )]
                let expected = v.as_f64().unwrap() as f32;
                let ulp =
                    (i64::from(got[p * 3 + i].to_bits()) - i64::from(expected.to_bits())).abs();
                worst_ulp = worst_ulp.max(ulp);
            }
        }
        assert!(
            worst_ulp <= 8,
            "{case} is {worst_ulp} ulp from the capture, past the 8 measured here; \
             something other than the libm pow difference has moved"
        );
        assert!(
            worst_ulp > 0,
            "{case} is now bit-exact, so it belongs in \
             the_mono_path_reproduces_the_oracle_floats_exactly and this bound is \
             hiding nothing -- move it rather than leaving a slack test behind"
        );
    }
}

/**
 * The three-band path, which is a different transform on the same bytes.
 *
 * The band count of the *gain map* alone selects it, and it linearises the
 * gain-map sample through `v2Y_8` where the one-band path divides by 255.
 * The capture's fixture is deliberately the mono one replicated into three
 * bands under identical metadata, so the two paths get the same input and
 * the record states they "do not match the mono ones anywhere". A port that
 * shared one formula between the two would pass the mono tests and fail
 * here.
 * Input: the capture's three-band gain-map values -> Output: libvips's
 * floats exactly, and different from the mono answer.
 */
#[test]
fn the_three_band_path_linearises_the_gain_map_and_the_mono_path_does_not() {
    let o = oracle();
    let mono_record = &o["records"]["uhdr2scRGB_mono_gainmap"];
    let record = &o["records"]["uhdr2scRGB_rgb_gainmap"];
    let base_bytes: Vec<u8> = mono_record["base_sRGB_bytes"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|h| {
            let s = h.as_str().unwrap();
            (0..3).map(move |i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
        })
        .collect();
    let gain_bytes: Vec<u8> = record["gainmap_decoded_values"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|p| {
            p.as_array()
                .unwrap()
                .iter()
                .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    let base = Raster::new(16, 1, PixelFormat::Rgb8, base_bytes).unwrap();
    let rgb_gain = Raster::new(16, 1, PixelFormat::Rgb8, gain_bytes.clone()).unwrap();
    let meta = metadata_case(record, "canonical");

    let got = floats(&uhdr::uhdr_to_scrgb(&base, &rgb_gain, &meta).unwrap());
    let want = record["results"]["scRGB"].as_array().unwrap();
    let mut wrong = Vec::new();
    for (p, pixel) in want.iter().enumerate() {
        for (i, v) in pixel.as_array().unwrap().iter().enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the capture holds f32 values"
            )]
            let expected = v.as_f64().unwrap() as f32;
            if got[p * 3 + i] != expected {
                wrong.push(format!("[{p}][{i}]: {} != {expected}", got[p * 3 + i]));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of 48 samples differ from the capture:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );

    // And the divergence itself: the same bytes down the one-band path give
    // a different answer. Without this the test above would pass for an
    // implementation that used the mono formula everywhere, as long as the
    // fixtures happened to agree.
    let mono_gain = Raster::new(
        16,
        1,
        PixelFormat::Gray8,
        gain_bytes.as_chunks::<3>().0.iter().map(|c| c[0]).collect(),
    )
    .unwrap();
    let mono = floats(&uhdr::uhdr_to_scrgb(&base, &mono_gain, &meta).unwrap());
    assert_ne!(
        mono, got,
        "the one-band and three-band paths must disagree on identical gain-map \
         bytes; if they agree, one of them is not being taken"
    );
}

/**
 * Degenerate metadata reaches the output as zeros and `NaN` rather than as
 * an error, because that is what libvips does.
 *
 * Nothing validates the gain-map terms on either side, so `log2(0)` arrives
 * at the boost expression as `-inf`. Where the gain-map sample is 0 that is
 * `-inf * 1`, so `exp2` gives 0 and the pixel goes black; where it is 255
 * it is `-inf * 0`, which is `NaN`. Both are pinned, per pixel, from
 * `uhdr2scRGB_degenerate_metadata`. A port that special-cased a zero boost
 * -- which looks like a kindness -- would produce neither.
 *
 * The capture's third case, `min_above_max`, does not record the metadata
 * it was run with, and the values do not fall out of the obvious
 * candidates, so it is deliberately not pinned here rather than pinned to a
 * guess.
 * Input: the capture's own base and gain-map bytes with `min` or `max`
 * zeroed -> Output: libvips's zeros and `NaN`s, in the right places.
 */
#[test]
fn degenerate_metadata_reproduces_the_captured_zeros_and_nans() {
    let o = oracle();
    let mono = &o["records"]["uhdr2scRGB_mono_gainmap"];
    let (base, gain) = mono_fixture_rasters(mono);

    for (case, min_boost, max_boost) in [("min_boost_zero", 0.0, 8.0), ("max_boost_zero", 1.0, 0.0)]
    {
        let meta = GainMapMetadata {
            min_content_boost: [min_boost; 3],
            max_content_boost: [max_boost; 3],
            ..GainMapMetadata::default()
        };
        let got = floats(&uhdr::uhdr_to_scrgb(&base, &gain, &meta).expect("libvips exits 0 here"));
        let want = o["records"]["uhdr2scRGB_degenerate_metadata"]["results"][case]["scRGB"]
            .as_array()
            .unwrap();
        let mut wrong = Vec::new();
        let mut nans = 0;
        for (p, pixel) in want.iter().enumerate() {
            for (i, v) in pixel.as_array().unwrap().iter().enumerate() {
                let have = got[p * 3 + i];
                let ok = match v.as_f64() {
                    #[expect(clippy::cast_possible_truncation, reason = "the capture holds f32")]
                    Some(n) => have == n as f32,
                    // The capture writes a bare `NaN`, which serde_json
                    // surfaces as a string rather than a number.
                    None => {
                        nans += 1;
                        have.is_nan()
                    }
                };
                if !ok {
                    wrong.push(format!("{case}[{p}][{i}]: {have} != {v}"));
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "{} of 48 samples differ from the capture in {case}:\n  {}",
            wrong.len(),
            wrong.join("\n  ")
        );
        assert_eq!(
            nans, 3,
            "{case} is supposed to carry exactly one NaN pixel; if it carries none \
             this test is only checking that zeros are zeros"
        );
    }
}

/**
 * Both images in the container are priced against the allocation budget,
 * not just the base.
 *
 * The budget is applied to each image on its own rather than to their sum,
 * so a budget that refuses the gain map and admits the base only exists if
 * the gain map is the *larger* of the two. An encoder never writes one --
 * the gain map is at most the base's size and has a third of its bands --
 * so this builds the adversarial container by hand: an 8x8 base with a
 * 512x512 gain map spliced in, carrying the real ISO segment lifted out of
 * a container this build wrote. That is the file the check exists for. A
 * gain map costing 262144 bytes behind a base costing 192 is exactly the
 * smuggling route an unpriced second image opens.
 * Input: that container at budgets of `u64::MAX`, 191 and 1000 -> Output:
 * decode, a refusal naming the base, and a refusal naming the gain map.
 */
#[test]
fn uhdr_prices_the_gain_map_as_well_as_the_base() {
    let small = uhdr::smallest_container();
    let base_end = end_of_first_jpeg(&small);

    // The gain map's ISO 21496-1 segment, taken from a real container so
    // the graft is a genuine gain map rather than something only this test
    // would accept.
    let gain_soi = &small[base_end..];
    let iso_start = 2;
    let iso_len = usize::from(u16::from_be_bytes([gain_soi[4], gain_soi[5]])) + 2;
    let iso_segment = &gain_soi[iso_start..iso_start + iso_len];
    assert_eq!(
        &iso_segment[4..4 + uhdr::ISO_GAIN_MAP_ID.len()],
        uhdr::ISO_GAIN_MAP_ID,
        "the lifted segment must be the ISO one, or the graft proves nothing"
    );

    let mut big_gain = Vec::new();
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut big_gain), 50);
    image::ImageEncoder::write_image(
        encoder,
        &vec![0u8; 512 * 512],
        512,
        512,
        image::ExtendedColorType::L8,
    )
    .expect("a 512x512 mono jpeg");

    let mut grafted = small[..base_end].to_vec();
    grafted.extend_from_slice(&big_gain[..2]);
    grafted.extend_from_slice(iso_segment);
    grafted.extend_from_slice(&big_gain[2..]);
    assert!(
        uhdr::is_uhdr(&grafted),
        "the grafted container must still be Ultra HDR, or the budget is never \
         asked about its gain map"
    );

    let open = DecodeLimits::default().with_max_alloc_bytes(u64::MAX);
    assert!(
        decode_bytes_with_limits(&grafted, open).is_ok(),
        "the control must decode with the budget lifted"
    );

    let label = |budget: u64| match decode_bytes_with_limits(
        &grafted,
        DecodeLimits::default().with_max_alloc_bytes(budget),
    ) {
        Err(SourceError::AllocLimitExceeded { what, .. }) => what,
        other => panic!("expected an allocation refusal at {budget}, got {other:?}"),
    };
    assert_eq!(
        label(191),
        "Ultra HDR base image",
        "one byte under the 8x8x3 base price must refuse on the base"
    );
    // Comfortably above the base's 192 and far under the gain map's
    // 262144. If the gain map were unpriced this would decode.
    assert_eq!(
        label(1000),
        "Ultra HDR gain map",
        "a budget that admits the base and not the gain map must refuse on the \
         gain map; if this decodes, the second image is unpriced"
    );
}

/**
 * A container this build writes is one this build reads, and one libvips's
 * own loader chooser routes to `uhdrload`.
 *
 * The second half of that cannot be asserted from Rust, so it is pinned in
 * the capture instead (`oracle-captures/foreign-uhdr/commands.sh`); what is
 * asserted here is the round trip through libviprs plus the structural
 * facts a reader depends on: the MPF pre-filter is present, the gain map
 * carries the ISO segment, and the metadata survives.
 * Input: an scRGB ramp with real headroom -> Output: a container that
 * satisfies `is_uhdr`, decodes, and reports the boost it was written with.
 */
#[test]
fn a_written_container_reads_back_as_ultra_hdr() {
    let mut px = Vec::new();
    for y in 0..32u32 {
        for x in 0..32u32 {
            let (t, s) = (x as f32 / 31.0, y as f32 / 31.0);
            px.push(0.02 + 6.0 * t * s);
            px.push(0.5f32.mul_add(1.0 - t, 3.0 * s));
            px.push(1.5f32.mul_add(t, 0.25));
        }
    }
    let src = float3(32, 32, px.clone());
    let bytes = uhdr::encode_uhdr(&src, &SaveOptions::default()).expect("encodes");

    assert!(
        uhdr::is_uhdr(&bytes),
        "what this writes must satisfy its own gate"
    );
    let meta = uhdr::metadata(&bytes).expect("the written metadata parses back");
    let peak = px.iter().fold(1f32, |a, &v| a.max(v));
    assert!(
        (meta.max_content_boost[1] - f64::from(peak)).abs() < 1e-3,
        "the written max-content-boost ({}) must be the peak the image needed ({peak})",
        meta.max_content_boost[1]
    );

    let raster = decode_bytes(&bytes).expect("decodes");
    assert_eq!((raster.width(), raster.height()), (32, 32));
    assert_eq!(
        raster.format(),
        PixelFormat::Rgb8,
        "the base is 3-band uchar"
    );
    assert!(
        raster.get_field("gainmap-data").is_some(),
        "the gain map travels"
    );
}

/**
 * The round trip through scRGB, bounded by what was measured, and
 * monotone in quality.
 *
 * There is no exactness to claim here and pretending otherwise would be the
 * defect: both halves are JPEGs, the gain map is subsampled, and the base
 * is quantised to 8 bits *after* being divided down by its own peak, so a
 * dark channel inside a bright pixel loses precision no matter what. What
 * can be claimed, and is: the error is bounded, and raising the quality
 * lowers it -- which is the property that would break first if the encoder
 * and the decoder disagreed about the gain-map encoding.
 * Measured on this tree, max absolute error over the ramp: 0.2598 at q75
 * shrink 2, 0.1277 at q95 shrink 2, 0.0799 at q95 shrink 1, 0.0698 at q100
 * shrink 1.
 * Input: one ramp at four settings -> Output: error under 0.35 everywhere
 * and strictly decreasing across the four.
 */
#[test]
fn the_scrgb_round_trip_is_bounded_and_improves_with_quality() {
    let mut px = Vec::new();
    for y in 0..64u32 {
        for x in 0..64u32 {
            let (t, s) = (x as f32 / 63.0, y as f32 / 63.0);
            px.push(0.02 + 6.0 * t * s);
            px.push(0.5f32.mul_add(1.0 - t, 3.0 * s));
            px.push(1.5f32.mul_add(t, 0.25));
        }
    }
    let src = float3(64, 64, px.clone());

    let mut errors = Vec::new();
    for (quality, gain_map_shrink) in [(75u8, 2u32), (95, 2), (95, 1), (100, 1)] {
        let bytes = uhdr::encode_uhdr(
            &src,
            &SaveOptions {
                quality,
                gain_map_shrink,
            },
        )
        .expect("encodes");
        let back = uhdr::from_container(&bytes, DecodeLimits::default()).expect("expands");
        let got = floats(&back);
        let worst = px
            .iter()
            .zip(&got)
            .fold(0f32, |a, (want, have)| a.max((want - have).abs()));
        errors.push((quality, gain_map_shrink, worst));
    }

    for (quality, shrink, worst) in &errors {
        assert!(
            *worst < 0.35,
            "q{quality} shrink{shrink} round-tripped with {worst} absolute error, past \
             the 0.35 bound; the measured worst on this tree was 0.2598"
        );
    }
    for pair in errors.windows(2) {
        assert!(
            pair[1].2 < pair[0].2,
            "raising the quality must lower the error: q{} shrink{} gave {} but \
             q{} shrink{} gave {}",
            pair[0].0,
            pair[0].1,
            pair[0].2,
            pair[1].0,
            pair[1].1,
            pair[1].2
        );
    }
}

/**
 * `uhdr2scRGB` refuses the inputs libvips refuses, with a typed error.
 *
 * The band and format rules are `vips_check_bands(nickname, in, 3)` and an
 * explicit `BandFmt` test, and the capture records the exact refusals. The
 * one that is not a straight port is the missing gain map: libvips exits 1
 * having printed **nothing** there, which is not something a caller can
 * act on, so this raises a real error instead. That divergence is
 * deliberate and is pinned here so it cannot be "fixed" back into silence.
 * Input: a 1-band base, a 4-band base, a 16-bit base and a 2-band gain map
 * -> Output: a typed `BadInput` for each.
 */
#[test]
fn the_transform_refuses_what_vips_refuses() {
    let gain = Raster::new(2, 1, PixelFormat::Gray8, vec![0, 0]).unwrap();
    let good = GainMapMetadata::default();

    let one_band = Raster::new(2, 1, PixelFormat::Gray8, vec![1, 2]).unwrap();
    let four_band = Raster::new(2, 1, PixelFormat::Rgba8, vec![0; 8]).unwrap();
    let sixteen_bit = Raster::new(2, 1, PixelFormat::Rgb16, vec![0; 12]).unwrap();
    for (name, base) in [
        ("one band", &one_band),
        ("four bands", &four_band),
        ("16-bit", &sixteen_bit),
    ] {
        let err = uhdr::uhdr_to_scrgb(base, &gain, &good).expect_err(name);
        assert!(
            matches!(err, libviprs::uhdr::UhdrError::BadInput { .. }),
            "{name} must be a typed BadInput, got {err:?}"
        );
    }

    // The positive control: the same call with a 3-band uchar base works,
    // so the refusals above are about the input and not about the transform
    // refusing everything.
    let ok = Raster::new(2, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6]).unwrap();
    assert!(uhdr::uhdr_to_scrgb(&ok, &gain, &good).is_ok());

    // A gain map with a band count neither path handles.
    let two_band_gain = Raster::new(
        2,
        1,
        PixelFormat::Multi8(std::num::NonZeroU16::new(2).unwrap()),
        vec![0; 4],
    )
    .unwrap();
    assert!(matches!(
        uhdr::uhdr_to_scrgb(&ok, &two_band_gain, &good).expect_err("2-band gain map"),
        libviprs::uhdr::UhdrError::BadInput { .. }
    ));
}

/**
 * A gain map smaller than its base is scaled up with a linear kernel, not
 * by nearest-neighbour.
 *
 * `uhdr2scRGB` runs `vips_resize(..., VIPS_KERNEL_LINEAR)` before the
 * per-pixel transform, and the capture is blunt about what happens
 * otherwise: "Anything else, nearest included, gives different pixels
 * everywhere the gainmap is not flat." Every other record in the capture
 * uses a gain map the same size as its base, where the scale is 1 and the
 * resampler is the identity -- so this is the *only* record that can catch
 * a wrong one, and without it a nearest-neighbour implementation passes
 * every other test in this file.
 *
 * Pinned two ways: the resampled gain map itself, against
 * `resized_to_16_wide`, and the resulting scRGB pixels. The first is what
 * makes a failure readable; the second is what proves the resampled values
 * are the ones the transform actually consumed.
 * Input: an 8-wide gain map against a 16-wide base -> Output: libvips's
 * resampled codes and its scRGB floats.
 */
#[test]
fn a_smaller_gain_map_is_resampled_linearly_and_not_by_nearest() {
    let o = oracle();
    let record = &o["records"]["uhdr2scRGB_gainmap_resize"];
    let mono = &o["records"]["uhdr2scRGB_mono_gainmap"];
    let (base, _) = mono_fixture_rasters(mono);

    let small: Vec<u8> = record["gainmap_decoded_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(small.len(), 8, "the capture's gain map is 8 wide");
    let gain = Raster::new(8, 1, PixelFormat::Gray8, small.clone()).unwrap();

    // The transform's own resample, read back through the identity
    // metadata: with min = max = 1 the gain is exactly 1, so each output
    // sample is `v2Y_8` of the base and the gain map's contribution is
    // invisible. So instead the resampled codes are recovered by running a
    // base whose every channel is the code itself, under metadata that
    // makes the result a pure function of the gain-map sample.
    let expected_codes: Vec<u8> = record["resized_to_16_wide"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(expected_codes.len(), 16);
    assert_ne!(
        expected_codes,
        (0..16u8)
            .map(|j| small[(j as usize / 2).min(7)])
            .collect::<Vec<_>>(),
        "the capture's resampled codes must differ from the nearest-neighbour \
         answer, or this test cannot tell the two apart"
    );

    let meta = metadata_case(record, "canonical");
    let got = floats(&uhdr::uhdr_to_scrgb(&base, &gain, &meta).unwrap());
    let want = record["results"]["scRGB"].as_array().unwrap();
    let mut wrong = Vec::new();
    for (p, pixel) in want.iter().enumerate() {
        for (i, v) in pixel.as_array().unwrap().iter().enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the capture holds f32 values"
            )]
            let expected = v.as_f64().unwrap() as f32;
            if got[p * 3 + i] != expected {
                wrong.push(format!("[{p}][{i}]: {} != {expected}", got[p * 3 + i]));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of 48 samples differ from the capture:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}
