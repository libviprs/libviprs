//! `Raster::extract` carries its input's metadata, like every other operation
//! in the crate, and the engine's region entry point carries it through to the
//! tiles (issue #740).
//!
//! `Raster::extract` is the crate's physical crop: `src/engine.rs` and
//! `src/streaming.rs` reach for it on every tile and every strip, and
//! `Raster::extract_area` is built on it. It returned a raster with the
//! interpretation, the resolution, the orientation and every attached field
//! gone, where `extract_area` has carried all of them since #690. Two public
//! methods, the same geometry, opposite behaviour, and neither doc said so.
//!
//! # It changes pixels, and only on the float carriers
//!
//! #664 makes the premultiply bracket take its alpha ceiling from the
//! **interpretation** on float carriers and from the storage depth on the
//! unsigned ones. So a float raster that lost its tag brackets against 255
//! where it should bracket against 1.0, and an unsigned one does not care.
//!
//! Measured here, on a 32x32 `RgbaF32` tagged `ScRgb` with a 3144-byte profile,
//! cropped 16x16 both ways and then `resize(0.5)`:
//!
//! ```text
//! FLOAT extract icc=false extract_area icc=true
//! FLOAT extract interp=Srgb  extract_area interp=ScRgb
//! FLOAT pixels identical going in: true
//! FLOAT resize(0.5):                98 of 1024 bytes differ
//! FLOAT control (explicit srgb vs scrgb): 98 of 1024 bytes differ
//!
//! U8    resize(0.5):                 0 of 256 bytes differ
//! U8    control:                      0 of 256 bytes differ
//! ```
//!
//! Two things about that table. The **`Rgba8` row is the trap**: measuring this
//! on the obvious 8-bit fixture reports no effect and the conclusion would be
//! wrong. And the **control is why the float row means something**: an
//! explicitly `Srgb`-retagged copy differs from the `ScRgb` one by exactly the
//! same bytes, so losing the tag through `extract` is precisely equivalent to
//! retagging, and the comparison is live rather than blind.
//!
//! The exact count depends on the fixture's pixel values; what is asserted is
//! that the float difference is non-zero and equals the control, and that the
//! unsigned one is zero.

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::cell::Cell;

use libviprs::{
    EngineConfig, Interpretation, Layout, MemorySink, PixelFormat, PyramidPlanner, Raster,
    generate_pyramid_region,
};

/// The attached profile's size, chosen to match a real sRGB profile so the cost
/// measurement below is about the allocation a real caller pays for.
const PROFILE_LEN: usize = 3144;

// ---------------------------------------------------------------------------
// A global allocator that counts allocations of exactly one size, so the cost
// of `fields.clone()` per crop can be measured rather than guessed.
// ---------------------------------------------------------------------------

struct CountingProfileClones;

thread_local! {
    /// Whether *this thread* is inside a [`watch`] call.
    static WATCHING: Cell<bool> = const { Cell::new(false) };
    /// Profile-sized allocations made by this thread while watching.
    static PROFILE_SIZED_ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingProfileClones {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        if layout.size() == PROFILE_LEN {
            // `try_with`, because a thread tearing down has already dropped its
            // locals and must not panic inside the allocator. `const`-initialised
            // `Cell`s, so reading one cannot itself allocate and recurse.
            let _ = WATCHING.try_with(|w| {
                if w.get() {
                    let _ = PROFILE_SIZED_ALLOCS.try_with(|n| n.set(n.get() + 1));
                }
            });
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingProfileClones = CountingProfileClones;

/// Count the profile-sized allocations `f` makes **on this thread**.
///
/// Thread-scoped rather than global on purpose: the test harness runs these
/// tests concurrently, and a global counter picks up every other test's
/// allocations of the same size. The first version of this was global and
/// reported 74 copies for a run that makes 29.
fn watch<R>(f: impl FnOnce() -> R) -> (R, usize) {
    PROFILE_SIZED_ALLOCS.with(|n| n.set(0));
    WATCHING.with(|w| w.set(true));
    let out = f();
    WATCHING.with(|w| w.set(false));
    (out, PROFILE_SIZED_ALLOCS.with(Cell::get))
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn tagged_float(n: u32) -> Raster {
    let mut data = Vec::new();
    for i in 0..(n as usize * n as usize * 4) {
        data.extend_from_slice(&(((i % 97) as f32) / 96.0).to_ne_bytes());
    }
    let mut im = Raster::new(n, n, PixelFormat::RgbaF32, data)
        .unwrap()
        .copy()
        .interpretation(Interpretation::ScRgb)
        .xres(5.0)
        .yres(7.0)
        .orientation(6)
        .build();
    im.set_icc_profile(&vec![9u8; PROFILE_LEN]);
    im
}

fn tagged_u8(n: u32) -> Raster {
    let data: Vec<u8> = (0..(n as usize * n as usize * 4))
        .map(|i| (i % 251) as u8)
        .collect();
    Raster::new(n, n, PixelFormat::Rgba8, data)
        .unwrap()
        .copy()
        .interpretation(Interpretation::ScRgb)
        .build()
}

fn differing_bytes(a: &Raster, b: &Raster) -> usize {
    a.data()
        .iter()
        .zip(b.data())
        .filter(|(p, q)| p != q)
        .count()
}

// ---------------------------------------------------------------------------

/// Issue #740. The two crops agree on metadata, because they are one crop.
#[test]
fn extract_carries_what_extract_area_carries() {
    let src = tagged_float(32);
    let e = src.extract(4, 6, 16, 16).unwrap();
    let ea = src.extract_area(4, 6, 16, 16);

    assert_eq!(
        e.data(),
        ea.data(),
        "the same pixels, so only metadata is at issue"
    );
    assert_eq!(
        e.interpretation(),
        Interpretation::ScRgb,
        "extract keeps the tag"
    );
    assert_eq!(e.xres(), 5.0, "extract keeps the resolution");
    assert_eq!(e.yres(), 7.0);
    assert_eq!(e.orientation(), 6, "extract keeps the orientation");
    assert_eq!(
        e.icc_profile().map(<[u8]>::len),
        Some(PROFILE_LEN),
        "extract keeps the profile"
    );
    assert_eq!(e.interpretation(), ea.interpretation());
    assert_eq!(e.icc_profile(), ea.icc_profile());
}

/// Issue #740. The origin offset is the one field the two do **not** share, and
/// that is a decision rather than an oversight.
///
/// `extract_area` stamps `(-left, -top)` because `vips_extract_area` does, and
/// #690 measured it. `Raster::extract` is not that operation: it is the crate's
/// physical crop, the thing `engine.rs` and `streaming.rs` call per tile and per
/// strip, and vips has no method it corresponds to, so there is no oracle
/// saying what a tile's origin should be. It carries, like every other
/// non-`extract_area` op in the crate.
///
/// The alternative was to stamp there too and let `extract_area` inherit it.
/// I did not take it: it would put a non-zero origin into every pyramid tile's
/// header on the strength of an analogy rather than a measurement, and a tile
/// is not a crop of a larger image in the sense `Xoffset` means.
#[test]
fn extract_carries_the_offset_where_extract_area_stamps_it() {
    let src = tagged_float(32).copy().xoffset(11).yoffset(13).build();

    let e = src.extract(4, 6, 16, 16).unwrap();
    assert_eq!(
        (e.xoffset(), e.yoffset()),
        (11, 13),
        "extract carries the source's origin"
    );

    let ea = src.extract_area(4, 6, 16, 16);
    assert_eq!(
        (ea.xoffset(), ea.yoffset()),
        (-4, -6),
        "extract_area still stamps (-left, -top), as #690 measured"
    );
}

/// Issue #740. The tag reaches resampled pixels, so dropping it was not
/// cosmetic.
///
/// The unsigned row is in the same test on purpose: it is the fixture that says
/// "no effect", and leaving it out is how this gets measured wrong.
#[test]
fn losing_the_tag_changes_resampled_pixels_on_float_carriers_only() {
    let float = tagged_float(32);
    let untagged = float
        .extract(0, 0, 16, 16)
        .unwrap()
        .copy()
        .interpretation(Interpretation::Srgb)
        .build();
    let tagged = float.extract_area(0, 0, 16, 16);
    assert_eq!(untagged.data(), tagged.data(), "identical going in");

    let d = differing_bytes(&untagged.resize(0.5), &tagged.resize(0.5));
    assert!(
        d > 0,
        "the float tag must reach the resampled pixels, got {d}"
    );

    // The control: an explicit retag differs by exactly the same bytes, so the
    // difference is the tag and nothing else.
    let control = tagged
        .copy()
        .interpretation(Interpretation::Srgb)
        .build()
        .resize(0.5);
    assert_eq!(
        differing_bytes(&control, &tagged.resize(0.5)),
        d,
        "an explicit srgb retag must differ by exactly the same bytes"
    );

    // And the trap, stated as an assertion: the 8-bit carrier shows nothing.
    let u8src = tagged_u8(32);
    let u8_untagged = u8src
        .extract(0, 0, 16, 16)
        .unwrap()
        .copy()
        .interpretation(Interpretation::Srgb)
        .build();
    let u8_tagged = u8src.extract_area(0, 0, 16, 16);
    assert_eq!(
        differing_bytes(&u8_untagged.resize(0.5), &u8_tagged.resize(0.5)),
        0,
        "the unsigned carrier takes its ceiling from the depth, so the tag \
         cannot bite; measuring this on an Rgba8 fixture reports no effect"
    );
}

/// Issue #740's real point: the guard on the **engine** path, because what the
/// engine does with the crop is why the drop mattered.
///
/// `generate_pyramid_region` crops with `Raster::extract` and hands the result
/// straight to the pyramid generator, so before this change every tile of a
/// cropped-region run came off a raster that had lost its interpretation, its
/// resolution, its orientation and its profile. A tile encoded to disk carried
/// none of them.
///
/// # A correction to the issue, measured rather than assumed
///
/// The issue says a pyramid of a **float** scRGB source through the region
/// entry point would not match a whole-image one. That is not reachable: the
/// engine refuses a float source outright, before any of this.
///
/// ```text
/// generate_pyramid_region(RgbaF32 source, ..)
///   -> Err(Raster(FloatUnsupported { op: "downscale_half" }))
/// ```
///
/// So the pixel-level consequence measured above is a **public-API** one, a
/// caller doing `extract` then `resize`, and not a pyramid one: the engine only
/// takes unsigned carriers, and on those #664's depth rule means the tag cannot
/// change a resampled byte. What the engine path loses is the metadata itself,
/// which is what this test pins.
#[test]
fn the_engine_region_path_carries_the_metadata_into_the_tiles() {
    let data: Vec<u8> = (0..(32usize * 32 * 3)).map(|i| (i % 251) as u8).collect();
    let mut src = Raster::new(32, 32, PixelFormat::Rgb8, data)
        .unwrap()
        .copy()
        .interpretation(Interpretation::ScRgb)
        .xres(5.0)
        .yres(7.0)
        .orientation(6)
        .build();
    src.set_icc_profile(&vec![9u8; PROFILE_LEN]);

    let plan = PyramidPlanner::new(32, 32, 16, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let sink = MemorySink::new();
    generate_pyramid_region(&src, &plan, &sink, &EngineConfig::default(), 0, 0, 32, 32).unwrap();

    check_tiles(&sink, "DeepZoom");

    // The Google layout takes a different padding path in `extract_tile`, with
    // three branches: a tile entirely outside the raster, a partial tile, and
    // the whole-tile fast path. A DeepZoom-only test reaches none of the first
    // two, and the mutation sweep said so twice: removing either carry survived
    // until this run existed.
    //
    // The **size** matters as much as the layout, which the sweep also had to
    // tell me. At 32x32 the Google canvas lands on tile boundaries and only the
    // fast path runs, so both mutations still survived. A non-power-of-two size
    // is what pushes tiles past the raster edge: with the carries removed, 40x40
    // leaves 16 of its 21 tiles untagged where 32x32 and 64x64 leave none.
    let gplan = PyramidPlanner::new(40, 40, 16, 0, Layout::Google)
        .unwrap()
        .plan();
    let gdata: Vec<u8> = (0..(40usize * 40 * 3)).map(|i| (i % 251) as u8).collect();
    let mut gsrc = Raster::new(40, 40, PixelFormat::Rgb8, gdata)
        .unwrap()
        .copy()
        .interpretation(Interpretation::ScRgb)
        .xres(5.0)
        .yres(7.0)
        .orientation(6)
        .build();
    gsrc.set_icc_profile(&vec![9u8; PROFILE_LEN]);
    let gsink = MemorySink::new();
    generate_pyramid_region(
        &gsrc,
        &gplan,
        &gsink,
        &EngineConfig::default(),
        0,
        0,
        40,
        40,
    )
    .unwrap();
    check_tiles(&gsink, "Google 40x40");
}

/// The per-tile assertions, shared by the two layouts.
fn check_tiles(sink: &MemorySink, layout: &str) {
    let tiles = sink.tiles();
    assert!(!tiles.is_empty(), "{layout} produced tiles at all");
    for t in &tiles {
        let c = t.coord;
        assert_eq!(
            t.raster.interpretation(),
            Interpretation::ScRgb,
            "{layout} tile L{} c{} r{} keeps the interpretation",
            c.level,
            c.col,
            c.row
        );
        assert_eq!(t.raster.xres(), 5.0, "{layout} tile keeps the resolution");
        assert_eq!(
            t.raster.orientation(),
            6,
            "{layout} tile keeps the orientation"
        );
        assert_eq!(
            t.raster.icc_profile().map(<[u8]>::len),
            Some(PROFILE_LEN),
            "{layout} tile keeps the ICC profile"
        );
    }
}

/// The float refusal above, as an assertion rather than a remark, so the
/// correction to the issue cannot quietly stop being true.
#[test]
fn the_engine_refuses_a_float_source_before_any_of_this_matters() {
    let f = Raster::new(32, 32, PixelFormat::RgbaF32, vec![0u8; 32 * 32 * 4 * 4]).unwrap();
    let plan = PyramidPlanner::new(32, 32, 16, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let r = generate_pyramid_region(
        &f,
        &plan,
        &MemorySink::new(),
        &EngineConfig::default(),
        0,
        0,
        32,
        32,
    );
    assert!(
        r.is_err(),
        "the pyramid path takes unsigned carriers only, so the float pixel \
         divergence is a public-API consequence and not an engine one"
    );
}

/// Issue #740, decision 1. What the carry costs on the tiling paths, measured
/// rather than assumed, because the issue asked for that before choosing.
///
/// `Raster::extract` runs once per tile and once per strip, the pyramid
/// downscale runs once per level, and the padded-tile path builds a tile from a
/// fresh buffer. Each of those now carries, and for an attached ICC profile
/// each carry is a real allocation that was not there before.
///
/// Counted with a global allocator watching allocations of exactly the
/// profile's size, over a whole `generate_pyramid_region` run:
///
/// ```text
/// image      tile  tiles  profile copies  per tile   pixels
/// 32x32      16        9              29      3.22     1024
/// 64x64      16       25              62      2.48     4096
/// 128x128    16       89             191      2.15    16384
/// 256x256    16      345             704      2.04    65536
/// ```
///
/// **The cost is O(tiles), not O(pixels)**: pixels grow 64x down that table and
/// the per-tile figure converges on two. That is the property worth asserting,
/// and it is why this is acceptable rather than something to engineer around.
///
/// At a realistic tile size the absolute figure is small: a 1024x1024 run at
/// 256px tiles makes 78 copies of a 3144-byte profile, 245 KB across the whole
/// run against 196 KB for a *single* tile buffer, so about 4% of one tile.
///
/// I considered making the clone cheap instead, by holding blob fields behind
/// an `Arc<[u8]>` in `MetadataFields`. Not taken: it changes a type every
/// module touches, for an overhead that is a few percent of one tile and
/// bounded by a count nobody scales, and there is no profile showing it
/// matters. If one ever does, the measurement to beat is in this table.
///
/// The ceiling is a rate with headroom over the worst row rather than an
/// equality, so an unrelated allocation of the same size cannot make it flaky
/// while a per-pixel regression still breaks it.
#[test]
fn the_carry_costs_a_bounded_number_of_profile_copies_per_tile() {
    for (n, tiles_at_least) in [(32u32, 9usize), (64, 25)] {
        let data: Vec<u8> = (0..(n as usize * n as usize * 3))
            .map(|i| (i % 251) as u8)
            .collect();
        let mut src = Raster::new(n, n, PixelFormat::Rgb8, data).unwrap();
        src.set_icc_profile(&vec![9u8; PROFILE_LEN]);
        let plan = PyramidPlanner::new(n, n, 16, 0, Layout::DeepZoom)
            .unwrap()
            .plan();

        // The sink is read *outside* the watch: `MemorySink::tiles()` clones
        // every stored raster, which is the harness's cost and not the
        // engine's, and counting it would triple the figure for no reason.
        let sink = MemorySink::new();
        let (_, copies) = watch(|| {
            generate_pyramid_region(&src, &plan, &sink, &EngineConfig::default(), 0, 0, n, n)
                .unwrap();
        });
        let tiles = sink.tiles().len();

        assert!(tiles >= tiles_at_least, "{n}x{n} produced {tiles} tiles");
        assert!(
            copies <= 4 * tiles,
            "{n}x{n}: the profile copy must stay bounded by the tile count, not \
             the pixel count: {copies} copies for {tiles} tiles"
        );
    }

    // And a single crop pays for exactly one, which is the unit the table is
    // built from.
    let mut one_src = Raster::new(32, 32, PixelFormat::Rgb8, vec![7u8; 32 * 32 * 3]).unwrap();
    one_src.set_icc_profile(&vec![9u8; PROFILE_LEN]);
    let (_, one) = watch(|| one_src.extract(0, 0, 16, 16).unwrap());
    assert_eq!(one, 1, "one crop, one profile copy");
}
