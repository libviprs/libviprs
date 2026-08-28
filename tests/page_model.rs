//! Pins the page model from outside the crate (issue #564).
//!
//! Three things are held here that a unit test cannot hold:
//!
//! * **one file names `page-height`**, the same structural discipline #635
//!   put on `n-pages`. A second writer either goes through
//!   `Raster::try_set_page_height`, where the divisor rule and the contract
//!   live, or spells the key and fails here;
//! * **reading the geometry allocates nothing**, whatever type an untrusted
//!   `.v` left under the name. `get_n_pages` had exactly this defect (#635):
//!   it resolved through `get_field`, which hands back an owned value, so a
//!   blob under the key was deep-copied on every call of an accessor that
//!   returns a small integer. `page_layout` is on the same shape and gets the
//!   same guard;
//! * **the split survives a `.v` round trip**, which is the interop half of
//!   the whole design. The roll layout is chosen partly because vips writes
//!   `page-height` into the container this crate calls native, so a `.v` that
//!   loses it would take the argument with it.
//!
//! The source scanner is #635's, kept deliberately identical so the two
//! guards cannot drift apart: same recursive walk, same cut at the first
//! `#[cfg(test)] mod`, same "quoted literal in non-comment code" question.
//!
//! # A note on Miri, which is no longer an omission
//!
//! This module doc used to say that three tests here reach the filesystem to
//! read `src/` and that none carried `#[cfg_attr(miri, ignore)]`, deferring
//! them to #712's sweep. #781 measured it: it is **one**,
//! [`exactly_one_source_file_names_the_page_height_key`], which reaches
//! `std::fs` through `non_test_bodies`. It carries the annotation now and has a
//! row in `tests/miri_fs_test_inventory.txt`.
//!
//! The other two the note counted do not touch a file.
//! `a_page_split_survives_a_v_round_trip` goes through `encode_vips` and
//! `decode_bytes`, which are both in memory, and
//! `the_scanner_sees_a_writer_when_there_is_one` scans string literals it
//! declares inline. So the detector was right about them and the note was not.
//!
//! What was true is the shape: the scanner read test *bodies* and could not see
//! a filesystem call reached through a helper. It follows one into test
//! scaffolding now, one file deep and to a fixed point, which is how this test
//! was found rather than by a re-read.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use libviprs::{MetadataValue, PageLayout, PixelFormat, Raster, decode_bytes};

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
// Source scanning (#635's, unchanged)
// ---------------------------------------------------------------------------

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Byte offset of the first `#[cfg(test)]` that gates a `mod`, or `None`.
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

/// Every `src/**/*.rs` file paired with its non-test body.
fn non_test_bodies() -> Vec<(String, String)> {
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

    let mut files = Vec::new();
    walk(&src_dir(), "", &mut files);
    assert!(
        files.len() > 50,
        "the scanner found only {} source files; it is looking in the wrong \
         place and the guard below would pass vacuously",
        files.len()
    );

    files
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
fn names_field(body: &str, field: &str) -> bool {
    let quoted = format!("\"{field}\"");
    body.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .any(|line| line.contains(&quoted))
}

// ---------------------------------------------------------------------------
// Who may write the key
// ---------------------------------------------------------------------------

/// One file names `page-height`, and it is the one that owns the geometry.
///
/// `src/raster.rs` rather than `src/imageio.rs`, where `n-pages` lives,
/// because a page height is a statement about how the pixel buffer's rows
/// divide and `Raster` is what owns that buffer. The two keys answer
/// different questions and sit next to the thing each is about.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn exactly_one_source_file_names_the_page_height_key() {
    let namers: Vec<String> = non_test_bodies()
        .into_iter()
        .filter(|(_, body)| names_field(body, "page-height"))
        .map(|(name, _)| name)
        .collect();

    assert_eq!(
        namers,
        ["raster.rs"],
        "`page-height` has one meaning (how many rows one page of this raster \
         occupies) and `Raster::try_set_page_height` is the only place that \
         names it. If you are adding a writer, call that instead: it refuses a \
         value that does not divide the height, which is the check that stops \
         a raster claiming a split its own rows cannot hold (issue #564)"
    );
}

/// The scanner would have caught a second writer, so the empty result above
/// is a measurement rather than a coincidence.
///
/// Without this the guard passes just as happily when `names_field` is broken,
/// which is the failure mode a zero result always has.
#[test]
fn the_scanner_sees_a_writer_when_there_is_one() {
    let planted = "\
//! A module that mentions `page-height` in prose, which is not a writer.
// and in a line comment, \"page-height\", which is not one either.

pub fn a_loader(raster: &mut Raster) {
    raster.set_field(\"page-height\", MetadataValue::Int(3));
}

#[cfg(test)]
mod tests {}
";
    let cut = test_module_cut(planted).expect("the file has a test module");
    assert!(
        names_field(&planted[..cut], "page-height"),
        "the scanner has to see a real writer, or the guard above is vacuous"
    );

    let prose_only = "//! All about `page-height`.\n// \"page-height\" again.\npub fn f() {}\n";
    assert!(
        !names_field(prose_only, "page-height"),
        "and it has to ignore the prose, which is where the key is discussed"
    );
}

// ---------------------------------------------------------------------------
// What the accessors cost
// ---------------------------------------------------------------------------

/// Reading the page geometry allocates nothing, whatever is stored under the
/// name.
///
/// `page-height` is not a built-in header field, so `set_field` stores
/// whatever type it is handed and a `.v` trailer restores arbitrary named
/// fields with arbitrary types out of an untrusted file (issue #565). An
/// accessor that resolved through `get_field` would deep-copy an 8 MiB blob
/// on every call and return `12`, which is exactly the defect #635 found on
/// `get_n_pages`.
///
/// A timing assertion would be flaky on a loaded machine, so this asserts the
/// mechanism: zero allocation calls on this thread across each accessor.
#[test]
fn reading_the_page_geometry_does_not_allocate_over_a_blob_field() {
    let mut raster = Raster::new(4, 12, PixelFormat::Gray8, vec![0u8; 48]).unwrap();
    raster.set_field("page-height", MetadataValue::Blob(vec![0xAB; 8 << 20]));

    let (layout, allocations) = allocations_during(|| black_box(&raster).page_layout());
    assert_eq!(
        layout.page_height(),
        12,
        "a blob is not an int, so the sanity check reports a single page"
    );
    assert_eq!(
        allocations, 0,
        "`page_layout` must borrow the stored value, not clone it out"
    );

    let (height, allocations) = allocations_during(|| black_box(&raster).get_page_height());
    assert_eq!(height, 12);
    assert_eq!(allocations, 0);

    let (pages, allocations) = allocations_during(|| black_box(&raster).pages_loaded());
    assert_eq!(pages, 1);
    assert_eq!(allocations, 0);
}

// ---------------------------------------------------------------------------
// Interop
// ---------------------------------------------------------------------------

/// The page split survives a `.v` round trip, which is half the reason the
/// roll layout won.
///
/// vips writes `page-height` into the `.v` container as a plain `gint`, so
/// nothing special is needed to carry it, and this asserts that rather than
/// assuming it. The bytes go through the public encoder and the public
/// decoder, so this is the same path a caller takes.
#[test]
fn a_page_split_survives_a_v_round_trip() {
    let mut roll = Raster::new(
        4,
        12,
        PixelFormat::Gray8,
        (0..48).map(|i| i as u8).collect(),
    )
    .expect("a 4x12 grey raster");
    roll.set_page_height(3);
    roll.set_field("n-pages", MetadataValue::Int(4));

    let back = decode_bytes(&roll.encode_vips().expect("encode")).expect("decode");

    assert_eq!(back.get_page_height(), 3, "the split came back");
    assert_eq!(back.pages_loaded(), 4);
    assert_eq!(back.get_n_pages(), 4, "and so did the file's page count");
    assert_eq!(back.data(), roll.data(), "with the pixels untouched");

    // Positive control: an unpaged raster round-trips as unpaged rather than
    // acquiring a split from somewhere, so the assertion above is about the
    // field travelling and not about the default.
    let still = Raster::new(4, 12, PixelFormat::Gray8, vec![0u8; 48]).unwrap();
    let still_back = decode_bytes(&still.encode_vips().expect("encode")).expect("decode");
    assert_eq!(still_back.pages_loaded(), 1);
    assert_eq!(still_back.get_field("page-height"), None);
}

/// The layout type is reachable from outside the crate and answers the same
/// way the accessor does, so a caller can reason about a page split without
/// holding a raster.
#[test]
fn the_layout_type_is_public_and_agrees_with_the_raster() {
    let mut roll = Raster::new(4, 12, PixelFormat::Gray8, vec![0u8; 48]).unwrap();
    roll.set_page_height(3);

    assert_eq!(roll.page_layout(), PageLayout::of(12, Some(3)));
    assert_eq!(PageLayout::of(12, Some(5)), PageLayout::single(12));
    assert!(PageLayout::divides(12, 3));
    assert!(!PageLayout::divides(12, 5));
}
