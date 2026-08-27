//! Pins what `n-pages` means, crate-wide, from outside the crate (issue #635).
//!
//! The batch review that filed #635 reported the key as carrying four
//! different meanings behind one accessor. Re-measured on the branch point,
//! after #626 moved the OpenEXR multi-part count out to `exr-parts`, there is
//! **one** meaning left and four loaders that honour it: `n-pages` is how many
//! pages the *file* holds, where a page is something a zero-based `page`
//! argument can select.
//!
//! | writer | what it counts | vips writer it ports |
//! |---|---|---|
//! | `src/gif.rs` | frames in the GIF | `nsgifload.c:281` |
//! | `src/encode_tiff.rs` | IFDs in the chain | `tiff2vips.c:1879` |
//! | `src/webp.rs` | frames in the original animation | `webp2vips.c:508` |
//! | `src/jxl.rs` | frames in the original | `jxlload.c:747` |
//!
//! Measured against `vipsheader -a` on 8.18.6, each of the four agrees with
//! its vips counterpart on the value *and* on when the field is attached at
//! all: a still GIF and a one-page TIFF carry `n-pages: 1`, a still WebP and a
//! single-frame JPEG XL carry no such field.
//!
//! So the answer to #635 is a single documented meaning rather than per-format
//! keys, and these tests are what stops a fifth meaning from arriving under the
//! same name. The value each loader attaches is pinned in that module's own
//! unit tests, against a fixture whose page count is distinct from every other
//! number in play; what lives here is the crate-wide contract a caller sees:
//! which files may write the key at all, and what `Raster::get_n_pages` does
//! with what it finds.

use std::path::{Path, PathBuf};

use libviprs::{MetadataValue, Raster};

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `src/*.rs` file paired with its non-test body.
///
/// The `#[cfg(test)]` module at the foot of a source file is cut off first,
/// because a unit test attaches whatever fields it likes to a raster it built
/// itself and that says nothing about what a loader writes. The cut is at the
/// first `#[cfg(test)]` in the file, which in this crate is always the test
/// module's own attribute.
fn non_test_bodies() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(src_dir()).expect("src/ must be readable") {
        let path = entry.expect("a readable directory entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let name = path
            .file_name()
            .expect("a file path has a file name")
            .to_string_lossy()
            .into_owned();
        let text = std::fs::read_to_string(&path).expect("a readable source file");
        let body = match text.find("#[cfg(test)]") {
            Some(cut) => text[..cut].to_string(),
            None => text,
        };
        out.push((name, body));
    }
    out.sort();
    out
}

/// Whether this body attaches `field` to a raster, as opposed to merely
/// reading or mentioning it. Both spellings the crate uses are covered:
/// `raster.set_field(name, ...)` and `raster.fields.set(name, ...)`.
///
/// Whitespace is squeezed out first so the match survives rustfmt breaking the
/// call across lines, which is the shape that would otherwise let a writer
/// through unseen. A false positive here fails loudly and gets read; a false
/// negative is the one that would quietly defeat the guard.
fn attaches(body: &str, field: &str) -> bool {
    let squashed: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    squashed.contains(&format!("set(\"{field}\""))
        || squashed.contains(&format!("set_field(\"{field}\""))
}

fn body_of<'a>(bodies: &'a [(String, String)], file: &str) -> &'a str {
    bodies
        .iter()
        .find(|(name, _)| name == file)
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| panic!("src/{file} must exist"))
}

/// Exactly four loaders may attach `n-pages`, and all four count pages.
///
/// This is the guard the issue asks for. A fifth writer is not forbidden, but
/// it cannot arrive silently: whoever adds one has to come back here, say what
/// their count is, and confirm it is a page count that a zero-based `page`
/// argument can index. A count that is not that gets its own key, the way
/// `exr-parts` did.
#[test]
fn only_page_counts_reach_the_shared_key() {
    let writers: Vec<String> = non_test_bodies()
        .into_iter()
        .filter(|(_, body)| attaches(body, "n-pages"))
        .map(|(name, _)| name)
        .collect();

    assert_eq!(
        writers,
        ["encode_tiff.rs", "gif.rs", "jxl.rs", "webp.rs"],
        "`n-pages` means one thing (pages in the file, indexable by a \
         zero-based `page`) and only these loaders attach it. If you are \
         adding a writer, check your count is a page count and extend the \
         table on `Raster::get_n_pages`; if it is a layer, a part, a \
         resolution level or anything else a caller cannot ask for by page \
         index, give it its own key instead (issue #635)"
    );
}

/// A count that no page index can select stays off the shared key.
///
/// OpenEXR is the worked example and the reason #635 exists: an EXR part is a
/// layer, `decode_exr` takes no part index, and `vipsheader -a` attaches no
/// `n-pages` to an EXR, so the part count travels as `exr-parts` (#626).
///
/// The PDF readers are the second case. vips's `pdfload` *does* attach
/// `n-pages` (measured: 3 for a three-page document, 1 for a one-page one),
/// but its `page` argument is zero-based where this crate's PDF page numbers
/// are deliberately one-based, so `0..get_n_pages()` would be off by one for a
/// caller who trusted it. The document's page count is `PdfInfo::page_count`
/// instead.
#[test]
fn a_count_that_is_not_a_page_count_gets_its_own_key() {
    let bodies = non_test_bodies();

    let exr = body_of(&bodies, "exr.rs");
    assert!(
        attaches(exr, "exr-parts"),
        "the EXR multi-part count is `exr-parts` (issue #626)"
    );
    assert!(
        !attaches(exr, "n-pages"),
        "an EXR part is a layer, not a page, and vips attaches no `n-pages` \
         to an EXR at all"
    );

    let pdf = body_of(&bodies, "pdf.rs");
    assert!(
        !attaches(pdf, "n-pages"),
        "this crate's PDF page numbers are one-based, so a zero-based \
         `0..get_n_pages()` sweep would be off by one; the count is \
         `PdfInfo::page_count`"
    );
}

/// The value is a count and the loaders' `page` arguments are indices, so the
/// sweep a caller writes is `0..get_n_pages()` and the last page is one less
/// than the count (issue #566).
#[test]
fn get_n_pages_is_a_count_and_page_is_an_index() {
    let mut raster = Raster::black(4, 4);
    raster.set_field("n-pages", MetadataValue::Int(3));

    assert_eq!(raster.get_n_pages(), 3);
    assert_eq!((0..raster.get_n_pages()).count(), 3);
    assert_eq!(
        (0..raster.get_n_pages()).last(),
        Some(2),
        "the last page of a three-page file is index 2"
    );
}

/// `get_n_pages` ports `vips_image_get_n_pages`'s sanity check whole, ceiling
/// included: vips reports a single page unless the field is an int strictly
/// between 1 and 10000 (`iofuncs/header.c:917-928`).
///
/// The table is measured, not read off the source: a C program linking
/// libvips 8.18.6 set the field to each value on a fresh image and printed
/// what `vips_image_get_n_pages` returned. 9999 comes back as 9999; 10000 and
/// everything above it comes back as 1.
#[test]
fn get_n_pages_applies_the_vips_sanity_ceiling() {
    let measured: [(i64, u32); 11] = [
        (-5, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 3),
        (9_998, 9_998),
        (9_999, 9_999),
        (10_000, 1),
        (10_001, 1),
        (65_536, 1),
    ];

    for (stored, expected) in measured {
        let mut raster = Raster::black(1, 1);
        raster.set_field("n-pages", MetadataValue::Int(stored));
        assert_eq!(
            raster.get_n_pages(),
            expected,
            "vips_image_get_n_pages reports {expected} for a stored {stored}"
        );
    }

    // A raster that carries no field at all is the single-page default, which
    // is where every loader that does not attach the key lands.
    assert_eq!(Raster::black(1, 1).get_n_pages(), 1);
}

/// A field of the wrong type is not coerced. vips reads `n-pages` with
/// `vips_image_get_int`, which fails on a string-typed field and leaves the
/// caller with the default (measured: a `gchararray` `"3"` reports 1). This
/// crate's own `get_int` refuses the same way, so the accessor now agrees with
/// both. The raw value is still readable through `get_field`.
#[test]
fn get_n_pages_ignores_a_field_that_is_not_an_int() {
    let mut raster = Raster::black(1, 1);
    raster.set_field("n-pages", MetadataValue::Str("3".to_string()));

    assert_eq!(raster.get_n_pages(), 1);
    assert_eq!(raster.get_int("n-pages"), None);
    assert_eq!(
        raster
            .get_field("n-pages")
            .expect("the field is set")
            .as_str(),
        "3",
        "the sanity check is on the accessor, not on the stored value"
    );
}
