//! Every public options struct is `#[non_exhaustive]` and reachable from
//! outside the crate through `default()` plus a `with_*` setter per field
//! (issue #630).
//!
//! # Why this exists
//!
//! Five of these structs carried a doc line promising that later fields could
//! be added without a breaking change, and two of them went further and said
//! they were *deliberately* not `#[non_exhaustive]` so a downstream
//! `..Default::default()` would keep working. Be precise about what was false
//! there, because half of it was true. A `..Default::default()` literal does
//! survive a new field; an exhaustive one does not, and stops compiling with
//! `E0063`. The promise was made about the type, and the docs advertised the
//! exhaustive spelling, so it held only for callers who happened to pick the
//! other one and nothing made them.
//!
//! The crate proved that against itself. Four integration tests were called
//! `save_options_are_constructible_downstream` and each constructed its options
//! exhaustively as well as with the update form. Integration tests compile as an
//! external crate, so those four were exactly the downstream callers the promise
//! was made to, and measured on this tree, adding one field to the five save
//! options structs breaks eight of their twelve construction sites with `E0063`.
//! `#[non_exhaustive]` refuses all twelve up front instead, which is the trade:
//! one break now, at a time somebody picks, against a break later at a time
//! nobody does.
//!
//! `DecodeLimits` in `src/source.rs` already had the shape that delivers the
//! promise instead of stating it: `#[non_exhaustive]`, `Default`, and a `with_*`
//! setter per field. This holds the other ten to it.
//!
//! # What it asserts
//!
//! * **The attribute is there.** [`OPTIONS_STRUCTS`] pairs each module source
//!   with the options structs it declares, and [`public_structs`]
//!   pulls the attribute block above each `pub struct` out of the text. A
//!   struct that loses `#[non_exhaustive]` fails here, and so does a new
//!   options struct added to one of these files without it.
//! * **The builders reach every field.** One test per struct constructs it from
//!   `default()` through every `with_*` setter and reads the fields back. These
//!   run as an external crate, so they are the migration path a downstream
//!   caller actually has, and if a setter is missing the test crate does not
//!   compile.
//!
//! # What it cannot see
//!
//! A tenth options struct in a *new* module file. Covering the whole tree needs
//! either a runtime directory walk, which would make this a filesystem test and
//! put a line in `tests/miri_fs_test_inventory.txt`, or a hand-written list of
//! every file under `src/`. Both turn this into a mandatory-edit hotspot in a
//! repository whose batching rule is that no two pull requests touch the same
//! file, which is the trap `merge-gate.yml`'s test count already fell into
//! (see `tests/miri_invocation_parity.rs`). The ten files here are the ones
//! that have an options struct, and
//! [`the_options_struct_inventory_is_complete_for_the_files_it_covers`] catches
//! one more added to any of them.
//!
//! That limit is not hypothetical: `src/jp2k.rs` arrived from #783 while this
//! was being written, carrying the same promise word for word, and it is in the
//! list because the composition check found it, not because anything here
//! would have.
//!
//! Every file arrives through `include_str!` at compile time, so there is no
//! filesystem access to isolate and no `#[cfg_attr(miri, ignore)]` is needed.

use libviprs::{
    AffineOptions, Extend, MagickLoadOptions, ReduceKernel, ResizeOptions, SvgOptions, gif, jp2k,
    jxl, radiance, uhdr, webp,
};

/// Each module source, and the options structs it declares.
///
/// The pairing is the assertion: a struct named here that the file does not
/// declare fails, and a `pub struct *Options` the file declares that is not
/// named here fails too.
const OPTIONS_STRUCTS: [(&str, &str, &[&str]); 10] = [
    (
        "src/gif.rs",
        include_str!("../src/gif.rs"),
        &["LoadOptions", "SaveOptions"],
    ),
    (
        "src/jxl.rs",
        include_str!("../src/jxl.rs"),
        &["LoadOptions", "SaveOptions"],
    ),
    (
        "src/jp2k.rs",
        include_str!("../src/jp2k.rs"),
        &["SaveOptions"],
    ),
    (
        "src/radiance.rs",
        include_str!("../src/radiance.rs"),
        &["SaveOptions"],
    ),
    (
        "src/uhdr.rs",
        include_str!("../src/uhdr.rs"),
        &["SaveOptions"],
    ),
    (
        "src/webp.rs",
        include_str!("../src/webp.rs"),
        &["LoadOptions", "SaveOptions"],
    ),
    ("src/svg.rs", include_str!("../src/svg.rs"), &["SvgOptions"]),
    (
        "src/resample.rs",
        include_str!("../src/resample.rs"),
        &["AffineOptions", "ResizeOptions"],
    ),
    (
        "src/foreign_stubs.rs",
        include_str!("../src/foreign_stubs.rs"),
        &["MagickLoadOptions"],
    ),
    (
        "src/source.rs",
        include_str!("../src/source.rs"),
        &["DecodeLimits"],
    ),
];

/// Every `pub struct` in `source`, paired with the attributes declared directly
/// above it.
///
/// The attribute block is the run of `#[...]` lines immediately above the
/// declaration, walking back past doc comments, so a `#[non_exhaustive]` that
/// belongs to some other item cannot be mistaken for this one's.
fn public_structs(source: &str) -> Vec<(String, Vec<String>)> {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix("pub struct ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let mut attrs = Vec::new();
        for prev in lines[..i].iter().rev() {
            let t = prev.trim();
            if t.starts_with("#[") {
                attrs.push(t.to_string());
            } else if t.starts_with("///") || t.starts_with("//") {
                continue;
            } else {
                break;
            }
        }
        out.push((name, attrs));
    }
    out
}

#[test]
fn every_public_options_struct_is_non_exhaustive() {
    for (path, source, expected) in OPTIONS_STRUCTS {
        let found = public_structs(source);
        let options: Vec<&(String, Vec<String>)> = found
            .iter()
            .filter(|(n, _)| expected.contains(&n.as_str()))
            .collect();
        assert_eq!(
            options.len(),
            expected.len(),
            "{path} declares {:?}, expected to find {expected:?}",
            found.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        );
        for (name, attrs) in options {
            assert!(
                attrs.iter().any(|a| a == "#[non_exhaustive]"),
                "{path}: `pub struct {name}` is not `#[non_exhaustive]`, so a \
                 downstream struct literal compiles today and stops compiling \
                 the day a field lands. Its attributes are {attrs:?}",
            );
        }
    }
}

/// The pinned inventory: a `pub struct` whose name ends in `Options` and that
/// [`OPTIONS_STRUCTS`] does not name fails here, so a tenth one added to any of
/// these files cannot slip past the attribute check by not being listed.
#[test]
fn the_options_struct_inventory_is_complete_for_the_files_it_covers() {
    for (path, source, expected) in OPTIONS_STRUCTS {
        for (name, _) in public_structs(source) {
            if name.ends_with("Options") {
                assert!(
                    expected.contains(&name.as_str()),
                    "{path} declares `pub struct {name}`, which \
                     OPTIONS_STRUCTS does not list; add it there so the \
                     `#[non_exhaustive]` check covers it",
                );
            }
        }
    }
}

#[test]
fn gif_save_options_build_through_setters() {
    let o = gif::SaveOptions::default()
        .with_interlaced(true)
        .with_dither(0.5)
        .with_bitdepth(4);
    assert!(o.interlaced);
    assert!((o.dither - 0.5).abs() < f64::EPSILON);
    assert_eq!(o.bitdepth, 4);

    let d = gif::SaveOptions::default();
    assert!(!d.interlaced);
    assert!((d.dither - 1.0).abs() < f64::EPSILON);
    assert_eq!(d.bitdepth, 8);
}

#[test]
fn gif_load_options_build_through_setters() {
    let o = gif::LoadOptions::default().with_page(2).with_n(-1);
    assert_eq!(o.page, 2);
    assert_eq!(o.n, -1);

    let d = gif::LoadOptions::default();
    assert_eq!(d.page, 0);
    assert_eq!(d.n, 1, "vips's gifload loads one page by default");
}

/// The animated WebP loader's options, added by issue #569 and spelled the
/// same way `gif::LoadOptions` is: same fields, same sentinel, same
/// defaults. Two sibling loaders answering one libvips argument differently
/// is the thing this asserts against.
#[test]
fn webp_load_options_build_through_setters() {
    let o = webp::LoadOptions::default().with_page(2).with_n(-1);
    assert_eq!(o.page, 2);
    assert_eq!(o.n, -1);

    let d = webp::LoadOptions::default();
    assert_eq!(d.page, 0);
    assert_eq!(d.n, 1, "vips's webpload loads one page by default");
    assert_eq!(
        (d.page, d.n),
        (
            gif::LoadOptions::default().page,
            gif::LoadOptions::default().n
        ),
        "the three animated loaders share one default"
    );
}

/// The animated JPEG XL loader's options, added by issue #621. Same shape
/// again, and the default is asserted against the other two rather than
/// restated, so a drift in any one of them fails here.
#[test]
fn jxl_load_options_build_through_setters() {
    let o = jxl::LoadOptions::default().with_page(2).with_n(-1);
    assert_eq!(o.page, 2);
    assert_eq!(o.n, -1);

    let d = jxl::LoadOptions::default();
    assert_eq!(d.page, 0);
    assert_eq!(d.n, 1, "vips's jxlload loads one page by default");
    assert_eq!(
        (d.page, d.n),
        (
            webp::LoadOptions::default().page,
            webp::LoadOptions::default().n
        ),
        "the three animated loaders share one default"
    );
}

#[test]
fn jxl_save_options_build_through_setters() {
    let o = jxl::SaveOptions::default().with_compression(jxl::Compression::Lossless);
    assert_eq!(o.compression, jxl::Compression::Lossless);
    assert_eq!(jxl::SaveOptions::default(), o);
}

#[test]
fn jp2k_save_options_build_through_setters() {
    let o = jp2k::SaveOptions::default().with_compression(jp2k::Compression::Lossless);
    assert_eq!(o.compression, jp2k::Compression::Lossless);
    assert_eq!(jp2k::SaveOptions::default(), o);
}

#[test]
fn radiance_save_options_build_through_setters() {
    let o = radiance::SaveOptions::default()
        .with_exposure(Some(2.0))
        .with_aspect(Some(1.5));
    assert_eq!(o.exposure, Some(2.0));
    assert_eq!(o.aspect, Some(1.5));

    let partial = radiance::SaveOptions::default().with_exposure(Some(2.0));
    assert_eq!(o.exposure, partial.exposure);
    assert_eq!(partial.aspect, None);

    let d = radiance::SaveOptions::default();
    assert_eq!((d.exposure, d.aspect), (None, None));
}

#[test]
fn uhdr_save_options_build_through_setters() {
    let o = uhdr::SaveOptions::default()
        .with_quality(95)
        .with_gain_map_shrink(1);
    assert_eq!(o.quality, 95);
    assert_eq!(o.gain_map_shrink, 1);

    let d = uhdr::SaveOptions::default();
    assert_eq!((d.quality, d.gain_map_shrink), (75, 2));
}

#[test]
fn webp_save_options_build_through_setters() {
    let o = webp::SaveOptions::default()
        .with_compression(webp::Compression::Lossless)
        .with_keep(webp::Keep::None);
    assert_eq!(o.compression, webp::Compression::Lossless);
    assert_eq!(o.keep, webp::Keep::None);

    let partial = webp::SaveOptions::default().with_keep(webp::Keep::None);
    assert_eq!(o, partial);

    let d = webp::SaveOptions::default();
    assert_eq!(d.compression, webp::Compression::Lossless);
    assert_eq!(d.keep, webp::Keep::All);
}

#[test]
fn svg_options_build_through_setters() {
    let o = SvgOptions::default()
        .with_dpi(144.0)
        .with_scale(2.0)
        .with_unlimited(true);
    assert!((o.dpi - 144.0).abs() < f64::EPSILON);
    assert!((o.scale - 2.0).abs() < f64::EPSILON);
    assert!(o.unlimited);

    let d = SvgOptions::default();
    assert!((d.dpi - 72.0).abs() < f64::EPSILON);
    assert!((d.scale - 1.0).abs() < f64::EPSILON);
    assert!(!d.unlimited);
}

#[test]
fn affine_options_build_through_setters() {
    let o = AffineOptions::default()
        .with_odx(1.0)
        .with_ody(2.0)
        .with_idx(3.0)
        .with_idy(4.0)
        .with_oarea(Some([0, 0, 8, 8]))
        .with_extend(Extend::White)
        .with_background(255.0)
        .with_premultiplied(true);
    assert!((o.odx - 1.0).abs() < f64::EPSILON);
    assert!((o.ody - 2.0).abs() < f64::EPSILON);
    assert!((o.idx - 3.0).abs() < f64::EPSILON);
    assert!((o.idy - 4.0).abs() < f64::EPSILON);
    assert_eq!(o.oarea, Some([0, 0, 8, 8]));
    assert_eq!(o.extend, Extend::White);
    assert!((o.background - 255.0).abs() < f64::EPSILON);
    assert!(o.premultiplied);

    let d = AffineOptions::default();
    assert_eq!(d.oarea, None);
    assert_eq!(d.extend, Extend::Background);
    assert!(!d.premultiplied);
}

#[test]
fn resize_options_build_through_setters() {
    let o = ResizeOptions::default()
        .with_vscale(Some(0.5))
        .with_kernel(ReduceKernel::Nearest)
        .with_gap(1.0);
    assert_eq!(o.vscale, Some(0.5));
    assert_eq!(o.kernel, ReduceKernel::Nearest);
    assert!((o.gap - 1.0).abs() < f64::EPSILON);

    let d = ResizeOptions::default();
    assert_eq!(d.vscale, None);
    assert_eq!(d.kernel, ReduceKernel::Lanczos3);
    assert!((d.gap - 2.0).abs() < f64::EPSILON);
}

#[test]
fn magick_load_options_build_through_setters() {
    let o = MagickLoadOptions::default()
        .with_density(Some("200"))
        .with_page(Some(1))
        .with_n(Some(-1));
    assert_eq!(o.density, Some("200"));
    assert_eq!(o.page, Some(1));
    assert_eq!(o.n, Some(-1));

    assert_eq!(MagickLoadOptions::default(), MagickLoadOptions::default());
}
