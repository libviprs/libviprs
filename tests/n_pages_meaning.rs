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
//! # How the guard is built, and why it is built this way
//!
//! The first version of this file asked a textual question of every source
//! file: does it contain `set_field("n-pages"` or `fields.set("n-pages"`. That
//! shape has to enumerate spellings, and it missed any it had not thought of:
//! a `const` holding the key, a helper taking it as a parameter. So the crate
//! was inverted instead. `Raster::set_n_pages` is now the only function that
//! names the key, and the guard's question is structural: **the literal
//! `"n-pages"` appears in exactly one source file**. A new writer either goes
//! through the one function, where the contract is documented, or spells the
//! key and fails here.
//!
//! The window that question is asked over is the second half of the fix. Each
//! file is cut at its test module, because a unit test attaches whatever
//! fields it likes to a raster it built itself and that says nothing about
//! what a loader writes. The cut used to be at the *first* `#[cfg(test)]` in
//! the file, described as "always the test module's own attribute". It is not:
//! `src/engine.rs` opens with a `#[cfg(test)] use` on line 8, so the guard
//! read 7 of its 4,374 lines, and a reviewer inserted a real `n-pages` writer
//! immediately above the test module and watched it pass. Across `src/` the
//! old cut scanned 50,883 of 101,484 lines; this one scans 58,440, and
//! `no_source_file_has_real_code_past_its_cut` is what stops that window
//! silently shrinking again.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use libviprs::{MetadataValue, Raster};

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

thread_local! {
    /// Allocation calls made on this thread. `const`-initialised and holding
    /// a `Copy` value with no destructor, so the accessor is a plain
    /// thread-local read and cannot itself allocate, which would recurse.
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// The system allocator with a per-thread call counter in front of it.
///
/// Per-thread rather than global because the test harness runs these in
/// parallel and a shared counter would see every sibling's traffic.
struct Counting;

// SAFETY: every method forwards to `System` with the layout it was handed and
// returns exactly what `System` returned, so the allocator contract is
// whatever `System`'s is. The counter is a thread-local `Cell<u64>` read and
// written without allocating.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Run `f` and report how many allocation calls it made on this thread.
fn allocations_during<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = ALLOCATIONS.with(Cell::get);
    let value = f();
    let after = ALLOCATIONS.with(Cell::get);
    (value, after - before)
}

// ---------------------------------------------------------------------------
// Source scanning
// ---------------------------------------------------------------------------

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// The item keywords a Rust file starts a top-level declaration with.
///
/// Used to decide whether anything past a file's cut is real code rather than
/// the gated test modules the cut is meant to drop.
const ITEM_KEYWORDS: [&str; 13] = [
    // Bare `pub`, not `pub `: `pub(crate) fn` is the spelling most of this
    // crate's internal items use and it slipped past a `"pub "` prefix.
    "pub",
    "fn ",
    "impl",
    "use ",
    "const ",
    "static ",
    "struct ",
    "enum ",
    "trait ",
    "type ",
    "unsafe ",
    "async ",
    "macro_rules!",
];

/// Byte offset of the first `#[cfg(test)]` that gates a `mod`, or `None` when
/// the file has no test module.
///
/// The *first* one and not the last, because several files here end in a run
/// of test modules with names of their own (`mod proptests`,
/// `mod retry_wiring_tests`, `mod single_level_tests`), and cutting at the
/// last would fold the earlier ones back into the body. What makes cutting at
/// the first safe is that nothing may follow it except more gated modules,
/// which `no_source_file_has_real_code_past_its_cut` checks on every file.
fn test_module_cut(text: &str) -> Option<usize> {
    const ATTR: &str = "#[cfg(test)]";
    let mut from = 0;
    while let Some(rel) = text[from..].find(ATTR) {
        let at = from + rel;
        if text[at + ATTR.len()..].trim_start().starts_with("mod ") {
            return Some(at);
        }
        from = at + ATTR.len();
    }
    None
}

/// Every `src/**/*.rs` file, named relative to `src/`, in sorted order.
///
/// Recursive, because `read_dir` alone stops at the top level and a loader
/// that moved into a subdirectory would leave the guard reading nothing at all
/// about it. There are no subdirectories under `src/` today, which is exactly
/// why the old non-recursive walk looked correct.
fn rust_sources() -> Vec<(String, PathBuf)> {
    fn walk(dir: &Path, prefix: &str, out: &mut Vec<(String, PathBuf)>) {
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
                out.push((format!("{prefix}{name}"), path));
            }
        }
    }

    let mut out = Vec::new();
    walk(&src_dir(), "", &mut out);
    assert!(
        out.len() > 50,
        "the scanner found only {} source files; it is looking in the wrong \
         place and every guard below would pass vacuously",
        out.len()
    );
    out
}

/// Every `src/**/*.rs` file paired with its non-test body.
fn non_test_bodies() -> Vec<(String, String)> {
    rust_sources()
        .into_iter()
        .map(|(name, path)| {
            let text = std::fs::read_to_string(&path).expect("a readable source file");
            let body = match test_module_cut(&text) {
                Some(cut) => text[..cut].to_string(),
                None => text,
            };
            (name, body)
        })
        .collect()
}

/// Whether `body` names `field` in code, as opposed to in prose.
///
/// Doc and line comments go first, because the key is discussed at length in
/// several module headers and none of those mentions is a writer. What is
/// left is the Rust string literal, quotes included, which is the one spelling
/// a writer cannot avoid: a `const` holding it, a helper taking it as an
/// argument and a macro building the call all still have to write it down
/// somewhere.
fn names_field(body: &str, field: &str) -> bool {
    let quoted = format!("\"{field}\"");
    body.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .any(|line| line.contains(&quoted))
}

/// Whether `body` calls `name`, tolerating rustfmt breaking the call across
/// lines by squeezing whitespace out first.
fn calls(body: &str, name: &str) -> bool {
    fn squash(s: &str) -> String {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }
    squash(body).contains(&squash(&format!("{name}(")))
}

fn body_of<'a>(bodies: &'a [(String, String)], file: &str) -> &'a str {
    bodies
        .iter()
        .find(|(name, _)| name == file)
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| panic!("src/{file} must exist"))
}

// ---------------------------------------------------------------------------
// The guard's own machinery
// ---------------------------------------------------------------------------

/// Nothing but gated test modules may follow a file's cut.
///
/// This is what keeps the window honest. Cutting at the first test module is
/// only safe while the rest of the file is more test modules; the moment real
/// code appears past it, the guards below would stop seeing it, and a writer
/// living there would pass unnoticed. That is the exact defect this file was
/// held on, so it fails loudly here instead of shrinking in silence.
#[test]
fn no_source_file_has_real_code_past_its_cut() {
    let mut offenders = Vec::new();
    for (name, path) in rust_sources() {
        let text = std::fs::read_to_string(&path).expect("a readable source file");
        let Some(cut) = test_module_cut(&text) else {
            continue;
        };
        let first_tail_line = text[..cut].lines().count() + 1;
        let mut gated = false;
        for (offset, line) in text[cut..].lines().enumerate() {
            if line.is_empty() || line.starts_with(char::is_whitespace) {
                continue;
            }
            let trimmed = line.trim();
            if trimmed.starts_with("//") {
                continue;
            }
            if trimmed.starts_with('#') {
                gated |= trimmed == "#[cfg(test)]";
                continue;
            }
            // A `mod` is allowed past the cut only while it is the one the
            // `#[cfg(test)]` just above it gates. An ungated one is real code.
            let is_gated_module = trimmed.starts_with("mod ") && gated;
            gated = false;
            if is_gated_module {
                continue;
            }
            if ITEM_KEYWORDS.iter().any(|kw| trimmed.starts_with(kw)) {
                offenders.push(format!("src/{name}:{} {trimmed}", first_tail_line + offset));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these lines are real code sitting past the first `#[cfg(test)] mod` \
         in their file, so every guard in this file stops reading before \
         them. Move the item above the test modules, or teach \
         `test_module_cut` the new shape. Do not leave it: the window \
         shrinking quietly is what #635's guard was held on.\n  {}",
        offenders.join("\n  ")
    );
}

/// The cut lands on the test module, not on the first `#[cfg(test)]`.
///
/// Pinned against a synthetic file in exactly `src/engine.rs`'s shape, which
/// is the one the old guard got wrong: a `#[cfg(test)] use` at the top, real
/// code after it, and the test module at the foot. The old rule cut at line 8
/// of 4,374 and a writer inserted just above `mod tests` was invisible.
#[test]
fn the_cut_is_the_test_module_not_the_first_cfg_test() {
    let engine_shaped = "\
//! A module.
#[cfg(test)]
use crate::observe::NoopObserver;

pub fn a_loader(raster: &mut Raster) {
    raster.set_field(\"n-pages\", MetadataValue::Int(7));
}

#[cfg(test)]
mod tests {
    #[test]
    fn t() {}
}
";
    let cut = test_module_cut(engine_shaped).expect("the file has a test module");
    let body = &engine_shaped[..cut];
    assert!(
        names_field(body, "n-pages"),
        "the writer sits between the two `#[cfg(test)]` attributes and has to \
         be inside the body; cutting at the first one hides it"
    );
    assert!(
        !body.contains("mod tests"),
        "the test module itself stays out of the body"
    );

    // A file with no test module is read whole rather than dropped.
    assert_eq!(test_module_cut("pub fn f() {}\n"), None);
}

// ---------------------------------------------------------------------------
// Who may write the key
// ---------------------------------------------------------------------------

/// One file names the key, and it is the one that documents what it means.
///
/// This is the guard the issue asks for, in its structural form. Every writer
/// goes through `Raster::set_n_pages`, so the literal `"n-pages"` appears in
/// `src/imageio.rs` and nowhere else outside a test module. A fifth writer is
/// not forbidden, but it cannot arrive silently: whoever adds one either calls
/// the one function, whose docs say what the count has to be, or writes the
/// key down and lands here.
#[test]
fn exactly_one_source_file_names_the_shared_key() {
    let namers: Vec<String> = non_test_bodies()
        .into_iter()
        .filter(|(_, body)| names_field(body, "n-pages"))
        .map(|(name, _)| name)
        .collect();

    assert_eq!(
        namers,
        ["imageio.rs"],
        "`n-pages` means one thing (pages in the file, indexable by a \
         zero-based `page`) and `Raster::set_n_pages` is the only place that \
         names it. If you are adding a writer, call that instead and check \
         your count is a page count; if it is a layer, a part, a resolution \
         level or anything else a caller cannot ask for by page index, give \
         it a key of its own (issue #635)"
    );
}

/// Exactly four loaders attach it, and all four count pages.
///
/// The companion to the test above: that one says the key is named once, this
/// one says who reaches it. Both have to hold, because a writer could route
/// through `set_n_pages` correctly and still be counting the wrong thing.
#[test]
fn only_the_four_page_counting_loaders_attach_it() {
    let bodies = non_test_bodies();

    let writers: Vec<&str> = bodies
        .iter()
        .filter(|(_, body)| calls(body, ".set_n_pages"))
        .map(|(name, _)| name.as_str())
        .collect();

    assert_eq!(
        writers,
        ["encode_tiff.rs", "gif.rs", "jxl.rs", "webp.rs"],
        "these four loaders attach `n-pages`, and each counts pages in the \
         file it decoded: IFDs, GIF frames, WebP animation frames, JPEG XL \
         frames. Adding a fifth means coming back here and saying what your \
         count is (issue #635)"
    );

    assert!(
        calls(body_of(&bodies, "imageio.rs"), "fn set_n_pages"),
        "`set_n_pages` is defined in imageio, next to the accessor whose \
         contract it writes for"
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
        names_field(exr, "exr-parts"),
        "the EXR multi-part count is `exr-parts` (issue #626)"
    );
    assert!(
        !calls(exr, ".set_n_pages"),
        "an EXR part is a layer, not a page, and vips attaches no `n-pages` \
         to an EXR at all"
    );

    assert!(
        !calls(body_of(&bodies, "pdf.rs"), ".set_n_pages"),
        "this crate's PDF page numbers are one-based, so a zero-based \
         `0..get_n_pages()` sweep would be off by one; the count is \
         `PdfInfo::page_count`"
    );
}

// ---------------------------------------------------------------------------
// What the accessor does with what it finds
// ---------------------------------------------------------------------------

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

/// Reading the count allocates nothing, whatever is stored under the name.
///
/// `n-pages` is not a built-in, so `try_set_field` stores whatever type it is
/// handed, and a `.v` trailer restores arbitrary named fields with arbitrary
/// types out of an untrusted file (issue #565). `get_n_pages` used to resolve
/// through `get_field`, which hands back an **owned** `MetadataValue` cloned
/// out of the field list, so a `Blob` sitting under the key was deep-copied
/// and dropped on every call — measured at 1.5 ms and a 64 MiB allocation per
/// call against 29 ns without, on a `&self` accessor that returns `1`, inside
/// the `for page in 0..raster.get_n_pages()` sweep its own docs recommend.
///
/// A timing assertion would be flaky on a machine running several test suites
/// at once, so this asserts the mechanism instead: zero allocation calls on
/// this thread across the accessor. That is size-independent, which is why the
/// blob here is small enough not to matter.
#[test]
fn get_n_pages_does_not_allocate_over_a_blob_field() {
    let mut raster = Raster::black(1, 1);
    raster.set_field("n-pages", MetadataValue::Blob(vec![0xAB; 8 << 20]));

    let (pages, allocations) = allocations_during(|| black_box(&raster).get_n_pages());

    assert_eq!(
        pages, 1,
        "a blob is not an int, so the vips sanity check reports a single page"
    );
    assert_eq!(
        allocations, 0,
        "`get_n_pages` must borrow the stored value, not clone it out: an \
         8 MiB copy per call on an accessor that returns 1 is what this \
         guards (issue #635)"
    );
}

/// `get_int` is the same accessor shape and had the same defect, so it gets
/// the same guard: it resolved through `get_field` too, and it is public and
/// takes any name a caller likes.
#[test]
fn get_int_does_not_allocate_over_a_blob_field() {
    let mut raster = Raster::black(1, 1);
    raster.set_field("thumbnail", MetadataValue::Blob(vec![0xCD; 8 << 20]));
    raster.set_field("bits-per-sample", MetadataValue::Int(8));

    let (blob, blob_allocations) = allocations_during(|| black_box(&raster).get_int("thumbnail"));
    assert_eq!(blob, None, "a blob is not an int");
    assert_eq!(
        blob_allocations, 0,
        "`get_int` must borrow the stored value rather than clone it out"
    );

    let (bits, bits_allocations) =
        allocations_during(|| black_box(&raster).get_int("bits-per-sample"));
    assert_eq!(bits, Some(8));
    assert_eq!(bits_allocations, 0);

    // The built-in header fields still answer, and still answer from the
    // header rather than the field list.
    let (width, width_allocations) = allocations_during(|| black_box(&raster).get_int("width"));
    assert_eq!(width, Some(1));
    assert_eq!(width_allocations, 0);
    assert_eq!(
        raster.get_int("interpretation"),
        None,
        "a string-valued built-in is not an int"
    );
    assert_eq!(raster.get_int("xres"), None, "nor is a double-valued one");
}
