//! Every public writer in this crate either reaches a save route or is on the
//! record as not reaching one (issue #948).
//!
//! Four times now a format lane has landed an encoder and left it wired to
//! nothing: #770 (jp2k), #809 (uhdr), #880 (radiance) and #882 (netpbm). #881
//! put a check behind the *doc*, so the list a caller reads and the arms the
//! dispatch has cannot drift apart, and that check has since caught a live
//! drift and two mutations. It could not catch TIFF, because a format that was
//! never wired at all is absent from both halves it compares, and two source
//! scans that agree on nothing agree perfectly.
//!
//! So this is the other question: not "do the two lists match" but **"can a
//! caller get to this writer"**. Every `pub fn save_*`, `pub fn encode_*`,
//! `pub fn *_save` and `pub fn *save_buffer*` under `src/` has to appear in
//! [`WRITERS`] below with a [`Reach`], and the set is checked for equality, so
//! a new writer with no row is red on the commit that adds it rather than four
//! releases later.
//!
//! # The vacuous state, and why there is not one
//!
//! A "no offenders" assertion passes exactly when it has stopped working: a
//! scan that finds nothing offends nobody. This one is stated the way #939
//! restated the `sha2` floor instead, as an expectation the scan has to
//! **carry**. [`WRITERS`] is a fixed non-empty list, `found == expected` is an
//! equality rather than a subset, and every `Reach::Wired` row names the arms
//! it is reached by, which are read back out of the two dispatches. If the
//! writer scan breaks, `found` is empty and the equality fails. If either arm
//! scan breaks, every `Wired` row fails. There is no state where nothing is
//! examined and the test still passes.
//!
//! # What a `Deferred` row buys
//!
//! A writer that always refuses has nothing to route to, so it is recorded
//! rather than routed, and the test **calls it** and requires the refusal.
//! That is the pin: the day somebody implements HEIF encoding or BigTIFF, this
//! goes red and hands them the reason, instead of the new encoder sitting
//! unreachable for four releases the way TIFF's did.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use libviprs::{EncodeError, PixelFormat, Raster, SaveError, TiffCompression};

/// How a caller reaches the container a writer produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reach {
    /// The route itself, not something routed to.
    Route,
    /// Reachable through these `Raster::save` extensions and these
    /// `Raster::encode_to_buffer` format names. Either list may be empty (Ultra
    /// HDR has no suffix at all, measured), but not both.
    Wired {
        extensions: &'static [&'static str],
        formats: &'static [&'static str],
    },
    /// Always refuses, so there is no container to route to. The test calls it
    /// and requires the refusal, so implementing it turns this red.
    Deferred,
    /// Produces real bytes and has no route, on purpose, with the reason.
    Unrouted { why: &'static str },
}

/// Every public writer under `src/`, with the route that reaches it.
///
/// Keyed by `<file>::<fn>` because two modules define an `encode_png` and two
/// define an `encode_uhdr`, and a bare function name cannot tell them apart.
const WRITERS: &[(&str, Reach)] = &[
    // The routes.
    ("connection.rs::encode_to_buffer", Reach::Route),
    ("connection.rs::encode_to_target", Reach::Route),
    ("imageio.rs::save_stripped", Reach::Route),
    // JPEG.
    (
        "encode.rs::encode_jpeg",
        Reach::Wired {
            extensions: &["jpg", "jpeg"],
            formats: &["jpeg", "jpg"],
        },
    ),
    (
        "encode.rs::encode_jpeg_options",
        Reach::Wired {
            extensions: &["jpg", "jpeg"],
            formats: &["jpeg", "jpg"],
        },
    ),
    (
        "encode.rs::jpegsave_buffer",
        Reach::Wired {
            extensions: &["jpg", "jpeg"],
            formats: &["jpeg", "jpg"],
        },
    ),
    (
        "encode.rs::save_jpeg",
        Reach::Wired {
            extensions: &["jpg", "jpeg"],
            formats: &["jpeg", "jpg"],
        },
    ),
    ("encode.rs::jpegsave_buffer_restart", Reach::Deferred),
    // PNG.
    (
        "encode.rs::encode_png",
        Reach::Wired {
            extensions: &["png"],
            formats: &["png"],
        },
    ),
    (
        "encode.rs::encode_png_interlaced",
        Reach::Wired {
            extensions: &["png"],
            formats: &["png"],
        },
    ),
    (
        "encode.rs::encode_png_palette",
        Reach::Wired {
            extensions: &["png"],
            formats: &["png"],
        },
    ),
    (
        "encode.rs::save_png",
        Reach::Wired {
            extensions: &["png"],
            formats: &["png"],
        },
    ),
    (
        "sink.rs::encode_png",
        Reach::Wired {
            extensions: &["png"],
            formats: &["png"],
        },
    ),
    // TIFF, wired by #948. Both suffixes and no more: measured on the pinned
    // vips 8.18.6, `tiffsave`'s `vips -l` line reads `nocache (.tif, .tiff)`
    // and `.btf`, `.tf8`, `.bigtiff` and `.tfx` are each refused.
    (
        "encode_tiff.rs::save_tiff",
        Reach::Wired {
            extensions: &["tif", "tiff"],
            formats: &["tif", "tiff"],
        },
    ),
    (
        "encode_tiff.rs::tiff_save",
        Reach::Wired {
            extensions: &["tif", "tiff"],
            formats: &["tif", "tiff"],
        },
    ),
    ("encode_tiff.rs::save_bigtiff", Reach::Deferred),
    ("encode_tiff.rs::save_tiff_tiled", Reach::Deferred),
    // GIF.
    (
        "gif.rs::encode_gif",
        Reach::Wired {
            extensions: &["gif"],
            formats: &["gif"],
        },
    ),
    (
        "gif.rs::save_gif",
        Reach::Wired {
            extensions: &["gif"],
            formats: &["gif"],
        },
    ),
    // WebP.
    (
        "webp.rs::encode_webp",
        Reach::Wired {
            extensions: &["webp"],
            formats: &["webp"],
        },
    ),
    (
        "webp.rs::save_webp",
        Reach::Wired {
            extensions: &["webp"],
            formats: &["webp"],
        },
    ),
    // JPEG XL, whose extension arm is behind the `jxl` feature and whose
    // format arm is not (#770's argument, applied on the side that needs it).
    (
        "jxl.rs::encode_jxl",
        Reach::Wired {
            extensions: &["jxl"],
            formats: &["jxl"],
        },
    ),
    (
        "jxl.rs::save_jxl",
        Reach::Wired {
            extensions: &["jxl"],
            formats: &["jxl"],
        },
    ),
    // JPEG 2000: five suffixes, six format names, one container.
    (
        "jp2k.rs::encode_jp2k",
        Reach::Wired {
            extensions: &["jp2", "j2k", "jpt", "j2c", "jpc"],
            formats: &["jp2k", "jp2", "j2k", "jpt", "j2c", "jpc"],
        },
    ),
    (
        "jp2k.rs::save_jp2k",
        Reach::Wired {
            extensions: &["jp2", "j2k", "jpt", "j2c", "jpc"],
            formats: &["jp2k", "jp2", "j2k", "jpt", "j2c", "jpc"],
        },
    ),
    // Ultra HDR: the one writer with no extension anywhere, and that is
    // measured rather than an omission. `uhdrsave` registers an empty suffix
    // list in `vips -l` and `vips copy base.v out.uhdr` is refused (#809).
    (
        "foreign_stubs.rs::encode_uhdr",
        Reach::Wired {
            extensions: &[],
            formats: &["uhdr"],
        },
    ),
    (
        "foreign_stubs.rs::encode_uhdr_gainmap_scale",
        Reach::Wired {
            extensions: &[],
            formats: &["uhdr"],
        },
    ),
    (
        "uhdr.rs::encode_uhdr",
        Reach::Wired {
            extensions: &[],
            formats: &["uhdr"],
        },
    ),
    // Radiance.
    (
        "radiance.rs::encode_radiance",
        Reach::Wired {
            extensions: &["hdr"],
            formats: &["hdr"],
        },
    ),
    (
        "radiance.rs::save_radiance",
        Reach::Wired {
            extensions: &["hdr"],
            formats: &["hdr"],
        },
    ),
    // Netpbm.
    (
        "textio.rs::encode_ppm",
        Reach::Wired {
            extensions: &["ppm", "pgm"],
            formats: &["ppm", "pgm"],
        },
    ),
    (
        "textio.rs::ppm_save",
        Reach::Wired {
            extensions: &["ppm", "pgm"],
            formats: &["ppm", "pgm"],
        },
    ),
    // FITS.
    (
        "fits.rs::encode_fits",
        Reach::Wired {
            extensions: &["fits", "fit", "fts"],
            formats: &["fits", "fit", "fts"],
        },
    ),
    (
        "fits.rs::save_fits",
        Reach::Wired {
            extensions: &["fits", "fit", "fts"],
            formats: &["fits", "fit", "fts"],
        },
    ),
    // The native container.
    (
        "imageio.rs::encode_vips",
        Reach::Wired {
            extensions: &["v", "vips"],
            formats: &["v", "vips"],
        },
    ),
    // Deferred: external delegates this build does not link.
    ("foreign_stubs.rs::encode_heif", Reach::Deferred),
    ("foreign_stubs.rs::encode_heif_chroma", Reach::Deferred),
    ("foreign_stubs.rs::encode_heif_lossless", Reach::Deferred),
    (
        "foreign_stubs.rs::encode_heif_lossless_bitdepth",
        Reach::Deferred,
    ),
    ("foreign_stubs.rs::encode_heif_tune", Reach::Deferred),
    ("foreign_stubs.rs::magicksave_buffer", Reach::Deferred),
    // The three the #948 sweep turned up, all of them writers that produce
    // real bytes with nothing routing to them. Filed rather than wired,
    // because each needs an oracle answer first and none of them is a
    // mechanical two-line arm the way TIFF was (issue #958).
    (
        "textio.rs::csv_save",
        Reach::Unrouted {
            why: "vips `csvsave` registers `.csv`, but it writes TAB separators \
                  and converts to mono first; this writes commas and band 0. \
                  Measured on 8.18.6 (issue #958)",
        },
    ),
    (
        "textio.rs::matrix_save",
        Reach::Unrouted {
            why: "vips `matrixsave` registers `.mat`, and on a 3-band ramp it \
                  writes the luminance (102 103 105) where this writes band 0 \
                  (1 11 21). `.mat` is also MATLAB's suffix on the way in. \
                  Measured on 8.18.6 (issue #958)",
        },
    ),
    (
        "foreign_stubs.rs::dzsave_buffer",
        Reach::Unrouted {
            why: "vips `dzsave` registers `.dz` and `.szi` and writes a tile \
                  pyramid; this writes a one-tile STORE zip with `Format=\"raw\"` \
                  that nothing in the crate reads back (issue #958)",
        },
    ),
];

/// `src/`.
fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `src/**/*.rs` file, named relative to `src/`.
///
/// Recursive, because `read_dir` alone stops at the top level and a codec that
/// moved into a subdirectory would leave this reading nothing about it. There
/// are no subdirectories under `src/` today, which is exactly why a
/// non-recursive walk would look correct.
fn rust_sources() -> Vec<(String, String)> {
    fn walk(dir: &Path, prefix: &str, out: &mut Vec<(String, String)>) {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()))
            .map(|entry| entry.expect("a readable directory entry").path())
            .collect();
        paths.sort();
        for path in paths {
            let name = path
                .file_name()
                .expect("a directory entry has a file name")
                .to_string_lossy()
                .into_owned();
            if path.is_dir() {
                walk(&path, &format!("{prefix}{name}/"), out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let body = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
                out.push((format!("{prefix}{name}"), body));
            }
        }
    }

    let mut out = Vec::new();
    walk(&src_dir(), "", &mut out);
    assert!(
        out.len() > 50,
        "the walk found only {} source files, so it is looking in the wrong \
         place and everything below would be reading an empty tree",
        out.len()
    );
    out
}

/// True for a function name that produces encoded output.
///
/// The four shapes the crate actually uses: `encode_jpeg`, `save_png`,
/// `tiff_save` and `jpegsave_buffer`. A bare `save` is not one of them, which
/// keeps `Raster::save` (the route) and `JobCheckpoint::save` (not a writer at
/// all) out without either needing a row.
fn is_writer_name(name: &str) -> bool {
    name.starts_with("save_")
        || name.starts_with("encode_")
        || name.ends_with("_save")
        || name.contains("save_buffer")
}

/// Every public writer under `src/`, as `<file>::<fn>`.
fn public_writers() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for (file, body) in rust_sources() {
        for line in body.lines() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix("pub fn ") else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            // `pub fn foo(` or `pub fn foo<T>(`, and nothing else.
            if name.is_empty() || !rest[name.len()..].starts_with(['(', '<']) {
                continue;
            }
            if is_writer_name(&name) {
                found.insert(format!("{file}::{name}"));
            }
        }
    }
    found
}

/// The quoted arm heads of the `match` inside `fn <needle>`, in `src/<file>`.
///
/// Only arm *heads* count, so a quoted name in an arm body or a comment is not
/// mistaken for a row. This is `wired_format_arms`' scan from
/// `src/connection.rs`, lifted so it can read both dispatches.
fn dispatch_arms(file: &str, needle: &str) -> BTreeSet<String> {
    let path = src_dir().join(file);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
    let start = src
        .find(needle)
        .unwrap_or_else(|| panic!("`{needle}` lives in src/{file}"));
    let body = &src[start..];
    // Both dispatches are the last thing in their item, and no inner block in
    // either closes at four-space indentation.
    let end = body
        .find("\n    }\n")
        .unwrap_or_else(|| panic!("`{needle}` closes at four-space indentation"));
    let mut arms = BTreeSet::new();
    for line in body[..end].lines() {
        let Some((head, _)) = line.split_once("=>") else {
            continue;
        };
        if !head.trim_start().starts_with('"') {
            continue;
        }
        for piece in head.split('|') {
            let piece = piece.trim();
            if let Some(inner) = piece.strip_prefix('"').and_then(|p| p.strip_suffix('"')) {
                arms.insert(inner.to_owned());
            }
        }
    }
    arms
}

/// The extensions `Raster::save` dispatches on.
fn extension_arms() -> BTreeSet<String> {
    dispatch_arms("imageio.rs", "fn encode_for_extension(")
}

/// The format names `Raster::encode_to_buffer` dispatches on.
fn format_arms() -> BTreeSet<String> {
    dispatch_arms("connection.rs", "fn encode_for_format(")
}

/// A small RGB raster every routed encoder here accepts.
fn rgb() -> Raster {
    Raster::new(4, 3, PixelFormat::Rgb8, vec![7u8; 4 * 3 * 3]).expect("a 4x3 RGB raster")
}

/**
 * Tests that every public writer under `src/` is either reachable through a
 * save route or recorded as unreachable, with the reason (issue #948).
 *
 * This is the check that would have caught TIFF. #881's guard compares the
 * documented format list against the dispatch arms, which keeps two lists
 * honest about each other and says nothing about a writer neither of them has
 * heard of. `Raster::save_tiff` shipped with round-trip tests and stayed
 * unreachable through four releases because no check asked this question.
 *
 * Set equality, not containment: a writer with no row is red, and a row whose
 * function was renamed or deleted is red too. `WRITERS` is non-empty, so a
 * scan that has stopped finding anything fails rather than passing vacuously.
 */
#[test]
#[cfg_attr(miri, ignore)] // reads src/ from disk, blocked by Miri isolation
fn every_public_writer_is_reachable_or_recorded() {
    let found = public_writers();
    let expected: BTreeSet<String> = WRITERS.iter().map(|(k, _)| (*k).to_owned()).collect();

    assert_eq!(
        WRITERS.len(),
        expected.len(),
        "`WRITERS` has a duplicate key"
    );
    assert!(
        found.len() >= 40,
        "the scan found only {} writers ({found:?}), so it has stopped reading \
         the tree and every assertion below would be about nothing",
        found.len()
    );

    let missing: Vec<&String> = expected.difference(&found).collect();
    let unrecorded: Vec<&String> = found.difference(&expected).collect();
    assert!(
        missing.is_empty() && unrecorded.is_empty(),
        "public writers with no row in `WRITERS`: {unrecorded:?}; rows whose \
         function is gone: {missing:?}. A new writer needs a `Reach`, which is \
         how #770, #809, #880, #882 and #948 stop happening a sixth time"
    );
}

/**
 * Tests that every `Reach::Wired` row names arms the two dispatches really
 * have (issue #948).
 *
 * The row is the claim "a caller can get here"; this reads the claim back out
 * of `encode_for_extension` and `encode_for_format`. Both scans carry a
 * positive control, because two source scans that have stopped finding
 * anything agree perfectly, and a `Wired` row naming nothing at all is
 * refused: Ultra HDR legitimately has no extension, and nothing legitimately
 * has neither.
 */
#[test]
#[cfg_attr(miri, ignore)] // reads src/ from disk, blocked by Miri isolation
fn every_wired_writer_names_arms_the_dispatches_have() {
    let extensions = extension_arms();
    let formats = format_arms();

    assert!(
        extensions.len() >= 18,
        "the extension-arm scan found only {extensions:?}"
    );
    assert!(
        formats.len() >= 20,
        "the format-arm scan found only {formats:?}"
    );

    let mut wired_rows = 0usize;
    for (key, reach) in WRITERS {
        let Reach::Wired {
            extensions: want_ext,
            formats: want_fmt,
        } = reach
        else {
            continue;
        };
        wired_rows += 1;
        assert!(
            !want_ext.is_empty() || !want_fmt.is_empty(),
            "{key} claims to be wired and names no arm at all"
        );
        for ext in *want_ext {
            assert!(
                extensions.contains(*ext),
                "{key} says `save(\"x.{ext}\")` reaches it, and \
                 `encode_for_extension` has no {ext:?} arm; it has {extensions:?}"
            );
        }
        for fmt in *want_fmt {
            assert!(
                formats.contains(*fmt),
                "{key} says `encode_to_buffer({fmt:?})` reaches it, and \
                 `encode_for_format` has no such arm; it has {formats:?}"
            );
        }
    }
    assert!(
        wired_rows >= 25,
        "only {wired_rows} wired rows, so most of `WRITERS` has stopped being checked"
    );
}

/**
 * Tests the other direction: every arm in either dispatch is claimed by some
 * writer row (issue #948).
 *
 * Without this a route could be added with nothing recorded about what it
 * reaches, and the table would go quietly out of date in the direction the
 * first check cannot see. Together the two make `WRITERS` a total map between
 * the writers and the routes rather than a list somebody remembers to update.
 */
#[test]
#[cfg_attr(miri, ignore)] // reads src/ from disk, blocked by Miri isolation
fn every_dispatch_arm_is_claimed_by_a_writer() {
    let mut claimed_ext: BTreeMap<&str, usize> = BTreeMap::new();
    let mut claimed_fmt: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, reach) in WRITERS {
        if let Reach::Wired {
            extensions,
            formats,
        } = reach
        {
            for e in *extensions {
                *claimed_ext.entry(e).or_default() += 1;
            }
            for f in *formats {
                *claimed_fmt.entry(f).or_default() += 1;
            }
        }
    }

    let extensions = extension_arms();
    let formats = format_arms();
    assert!(
        extensions.len() >= 18 && formats.len() >= 20,
        "scans are alive"
    );

    let orphan_ext: Vec<&String> = extensions
        .iter()
        .filter(|e| !claimed_ext.contains_key(e.as_str()))
        .collect();
    assert!(
        orphan_ext.is_empty(),
        "`encode_for_extension` has arms no writer row claims: {orphan_ext:?}"
    );

    let orphan_fmt: Vec<&String> = formats
        .iter()
        .filter(|f| !claimed_fmt.contains_key(f.as_str()))
        .collect();
    assert!(
        orphan_fmt.is_empty(),
        "`encode_for_format` has arms no writer row claims: {orphan_fmt:?}"
    );
}

/**
 * Tests that every `Reach::Deferred` writer really does refuse (issue #948).
 *
 * A row saying "there is nothing to route to" is only honest while that is
 * true, and a comment cannot stay honest on its own. So each one is called on
 * a raster the routed encoders all accept, and has to come back refused. The
 * day HEIF or BigTIFF gets an implementation this cell goes red and tells
 * whoever wrote it to give the new container a route, which is the step that
 * was skipped four times before #948 and once in it.
 *
 * `rgb()` is the positive control built in: the same raster goes through
 * `encode_to_buffer("png")` at the end, so "everything refuses" cannot pass
 * this.
 */
#[test]
fn every_deferred_writer_still_refuses() {
    let im = rgb();
    let path = Path::new("unused-by-a-refusal.tif");

    let encode_refusals: [(&str, Result<Vec<u8>, EncodeError>); 7] = [
        (
            "encode.rs::jpegsave_buffer_restart",
            im.jpegsave_buffer_restart(2),
        ),
        ("foreign_stubs.rs::encode_heif", im.encode_heif(75, "av1")),
        (
            "foreign_stubs.rs::encode_heif_lossless",
            im.encode_heif_lossless("av1"),
        ),
        (
            "foreign_stubs.rs::encode_heif_chroma",
            im.encode_heif_chroma(75, "av1", true),
        ),
        (
            "foreign_stubs.rs::encode_heif_lossless_bitdepth",
            im.encode_heif_lossless_bitdepth("av1", 12),
        ),
        (
            "foreign_stubs.rs::encode_heif_tune",
            im.encode_heif_tune(75, "av1", "ssim"),
        ),
        (
            "foreign_stubs.rs::magicksave_buffer",
            im.magicksave_buffer(".png"),
        ),
    ];
    for (key, result) in encode_refusals {
        assert!(
            matches!(result, Err(EncodeError::Unsupported { .. })),
            "{key} is recorded as `Deferred`; it produced bytes. Give the \
             container a save route and move its row to `Reach::Wired`"
        );
    }

    let save_refusals: [(&str, Result<(), SaveError>); 2] = [
        (
            "encode_tiff.rs::save_bigtiff",
            im.save_bigtiff(path, TiffCompression::None),
        ),
        (
            "encode_tiff.rs::save_tiff_tiled",
            im.save_tiff_tiled(path, TiffCompression::None, 256, 256, false, false),
        ),
    ];
    for (key, result) in save_refusals {
        let err = result.expect_err(key);
        assert!(
            matches!(err, SaveError::Encode(_)),
            "{key} must refuse with a typed encode error, got {err}"
        );
    }
    // The control: the same raster does encode through a live row, so
    // "everything refuses" is not what made the assertions above pass.
    assert!(
        im.encode_to_buffer("png").is_ok(),
        "positive control: a routed encoder accepts this raster"
    );
}

/**
 * Tests that every `Reach::Unrouted` row carries a reason and stays unrouted
 * (issue #948).
 *
 * These three write real bytes and vips registers a suffix for two of them, so
 * the honest state is "not yet", not "no". The reason has to name the measured
 * divergence and the issue, and the extension has to stay refused. Wiring one
 * turns this red, which is the handover: whoever wires it moves the row to
 * `Wired` and gets the sweep's other three checks for free.
 */
#[test]
fn every_unrouted_writer_is_still_unrouted_and_says_why() {
    let im = rgb();
    let unrouted: Vec<(&str, &str)> = WRITERS
        .iter()
        .filter_map(|(key, reach)| match reach {
            Reach::Unrouted { why } => Some((*key, *why)),
            _ => None,
        })
        .collect();
    assert_eq!(
        unrouted.len(),
        3,
        "the sweep found three unrouted writers; this is {unrouted:?}"
    );

    for (key, why) in &unrouted {
        assert!(
            why.contains("issue #"),
            "{key}'s reason must name the issue carrying it, got {why:?}"
        );
    }

    // The suffixes vips registers for the two text formats, still refused here.
    for extension in ["csv", "mat", "dz", "szi"] {
        let err = im
            .encode_to_buffer(extension)
            .expect_err("still unrouted (issue #958)");
        assert!(
            matches!(err, EncodeError::Unsupported { .. }),
            "{extension:?} must stay a typed refusal, got {err}"
        );
    }

    // The control: the writers themselves are alive and do produce bytes, so
    // this is a routing gap and not a dead function.
    assert!(!im.csv_save().is_empty());
    assert!(!im.matrix_save().is_empty());
    assert!(!im.dzsave_buffer().is_empty());
}
