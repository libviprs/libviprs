//! Pins what the ICC **LUT** routes ask the CMS for, and how big it can get
//! (issue #693).
//!
//! # The property, stated as narrowly as it holds
//!
//! `src/colour.rs` reserves every image-sized buffer it owns through
//! `Vec::try_reserve_exact`, so a host that cannot serve one of those gets an
//! `Err`. That claim stops at `xf.transform`. moxcms 0.8.1 allocates the
//! katana engine's intermediates with a plain `vec![0f32; n]` sized from the
//! slice it is handed (`conversions/katana/md3x3.rs:176`, `md4x3.rs:164`,
//! `md_nx3.rs:160` and `md_pipeline.rs:90`), and a plain `vec!` that cannot be
//! served reaches `handle_alloc_error`, which ends the process. Every other
//! input-sized allocation in that crate already goes through its own
//! `try_vec!` macro; those four stages are the exceptions.
//!
//! Nothing in this repository can make that allocation fallible. What it can
//! do is stop it being image-sized, by handing the transform bounded slices,
//! and that is the property these checks pin: **whatever the image, the
//! largest buffer moxcms allocates on our behalf is the same fixed size.** The
//! abort is not gone. It is out of reach of the input, which is a different
//! and weaker statement, and
//! [`refusing_what_moxcms_still_allocates_itself_ends_the_process`] exists to
//! keep it stated honestly rather than implied away.
//!
//! # How the ceiling tells the two kinds of allocation apart
//!
//! Every image-sized buffer libviprs reserves goes through
//! `Vec::try_reserve_exact`, which reaches the allocator as a plain `alloc`.
//! So does moxcms's own `try_vec!`, which reserves and then resizes. The four
//! infallible katana sites are `vec![0f32; n]`, and std's zero specialisation
//! lowers that to `alloc_zeroed`. So **zeroed** picks out exactly the
//! allocations this file is about, and nothing else on either route.
//!
//! That distinction is what keeps these checks off the rock the last four
//! allocation lanes hit: a ceiling that answers before the call it means to
//! test. It cannot answer early here, because none of the crate's own buffers
//! is zeroed. What keeps that true rather than assumed is the pair of
//! `..._hands_moxcms_nothing_that_scales_with_the_image` probes: they take the
//! largest *zeroed* request at two geometries and insist the two are the same
//! number, so a zeroed buffer of libviprs's own that follows the image makes
//! them differ. Measured: putting the import fallback's PCS plane back to
//! `vec![0f32; n]` takes them to 786432 and 3145728 and reddens both. Issue
//! #460 would give the module such a buffer for real, if `alloc_colour_output`
//! ever takes the `calloc`-preserving shape its doc describes, and it would go
//! red here before the starvation checks could start refusing the wrong
//! buffer.
//!
//! # Why a child process
//!
//! `handle_alloc_error` aborts. There is no unwinding to catch and no thread
//! to join, so the only place the outcome can be observed from is another
//! process. Each check spawns this same test binary at the `child_run` entry
//! point below, hands it a route, a geometry and a ceiling mode through the
//! environment, and reads the report off its stdout.
//!
//! That is also why every check here carries `#[cfg_attr(miri, ignore)]` and
//! turns up in `tests/miri_fs_test_inventory.txt`. In this tree that annotation
//! means "reaches the real filesystem", and `spawn` does: it resolves
//! `std::env::current_exe()` and execs the binary at that path. The detector in
//! `tests/miri_ignore_convention.rs` cannot see it, because it is a syntactic
//! scan of one function body and the access is two calls away in a helper, so
//! these land as `not-detected` rows the same way `stream_verify`'s
//! malformed-strip checks do. `child_run` needs no annotation: it is `#[ignore]`
//! unconditionally, so no runner ever reaches it on its own.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write as _;
use std::process::{Command, Output};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use libviprs::colour::{Intent, Pcs};
use libviprs::pixel::PixelFormat;
use libviprs::raster::Raster;
use moxcms::{
    ColorProfile, DataColorSpace, LutMultidimensionalType, LutStore, LutWarehouse, Matrix3d,
    ProfileClass, ToneReprCurve, Vector3d,
};

// ---------------------------------------------------------------------------
// Geometry and ceilings
// ---------------------------------------------------------------------------

/// Side of the square fixture the single-geometry checks use.
///
/// Small enough that a debug-build LUT interpolation over it is a fraction of
/// a second, large enough that one image-sized plane is far above anything
/// else on the path: the encoded profile is about 30 kB and the katana engine
/// builds no grid at all.
const DIM: u32 = 256;

/// A second geometry, four times the area, for the two checks that compare one
/// against the other. The comparison is the whole point: a bound that holds at
/// one size and not at four times it is not a bound.
const WIDE_DIM: u32 = 512;

/// Bytes in one three-`f32` plane over a `dim`-square image.
///
/// This is the size of the crate's device plane, the size of the PCS plane on
/// both fallbacks, and the size of the katana intermediate moxcms used to
/// allocate between them, because all three are three floats a pixel.
fn plane_bytes(dim: u32) -> usize {
    (dim as usize) * (dim as usize) * 3 * size_of::<f32>()
}

/// What counts as a large **plain** allocation, for logging and for refusal.
///
/// Only the plain side carries a floor that means "image-sized", because only
/// the plain side is indexed: `REFUSE_PLAIN_AT` counts requests in order and
/// `refusing_the_crate_s_own_device_plane_reports_an_error_rather_than_aborting`
/// refuses index 0 expecting the device plane. Anything smaller admitted here
/// would shift that index onto a buffer the check is not about. 64 kB is above
/// the profile blob (about 30 kB), the CLUT and the tone curves, and below
/// every image-sized plane either route reserves.
const PLAIN_LOG_FLOOR: usize = 64 * 1024;

/// What counts as a large **zeroed** allocation.
///
/// Two orders of magnitude below the plain floor, and that asymmetry is the
/// point rather than an oversight. Zeroed requests are not indexed, so nothing
/// breaks by admitting small ones, and admitting them is what stops this file
/// silently depending on the chunk `src/colour.rs` happens to pick: the katana
/// intermediate is twelve bytes a pixel, so a 64 kB floor would stop seeing it
/// below a 5462-pixel chunk and every check here would fail claiming moxcms had
/// been fixed upstream. At 512 bytes the floor stops mattering until the chunk
/// drops under 43 pixels.
///
/// It costs nothing, measured rather than assumed: with the floor at 512 the
/// only zeroed requests in the measured window on either route are the four
/// katana intermediates. Nothing this crate reserves is zeroed, because
/// `Vec::try_reserve_exact` reaches the allocator as a plain `alloc`, so there
/// is no noise down here to exclude.
const ZEROED_LOG_FLOOR: usize = 512;

/// The floor a request of this kind has to clear to be logged at all.
fn log_floor(zeroed: bool) -> usize {
    if zeroed {
        ZEROED_LOG_FLOOR
    } else {
        PLAIN_LOG_FLOOR
    }
}

/// The bound `src/colour.rs` promises for a single moxcms allocation.
///
/// Deliberately looser than the chunk that module actually picks
/// (`ICC_TRANSFORM_CHUNK_PIXELS` at three `f32` a pixel), so retuning the
/// chunk for cache behaviour does not break these checks, while going back to
/// handing moxcms the whole image does: `plane_bytes(DIM)` is 768 kB and
/// `plane_bytes(WIDE_DIM)` is 3 MB, both well past this.
///
/// This is the *upper* end of the chunk window these checks accept, at 43690
/// pixels. [`ZEROED_LOG_FLOOR`] is the lower end, at 43. See the constant's own
/// doc in `src/colour.rs` for the window stated from that side.
const CMS_CEILING_BYTES: usize = 512 * 1024;

// ---------------------------------------------------------------------------
// The ceiling
// ---------------------------------------------------------------------------

/// How many large requests the ordered log below can hold.
///
/// Diagnostic only, and deliberately not load-bearing. At the current chunk
/// the routes make about twenty-two of these at `WIDE_DIM`, but the count goes
/// up as the chunk comes down, so a cap can always be reached by a legitimate
/// retune. Overflow is reported as `TRUNCATED` rather than asserted, and the
/// figure every check actually asserts on comes from [`MAX_ZEROED`], which is
/// counted outside this array and cannot truncate.
const LOG_CAP: usize = 256;

/// Large requests seen since [`WATCHING`] went up, in order.
static LOG_LEN: AtomicUsize = AtomicUsize::new(0);
static LOG_SIZE: [AtomicUsize; LOG_CAP] = [const { AtomicUsize::new(0) }; LOG_CAP];
static LOG_ZEROED: [AtomicBool; LOG_CAP] = [const { AtomicBool::new(false) }; LOG_CAP];

/// Whether the ceiling is recording. Down outside the measured window, so the
/// harness's own startup and the fixture construction stay out of the log.
static WATCHING: AtomicBool = AtomicBool::new(false);

/// Refuse any *zeroed* large request at or above this many bytes. `usize::MAX`
/// means refuse none, which is the watching mode.
static REFUSE_ZEROED_FROM: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Refuse the large *plain* request at this index, counting from zero within
/// the measured window. `usize::MAX` means refuse none.
static REFUSE_PLAIN_AT: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Large plain requests seen so far, which is what [`REFUSE_PLAIN_AT`] indexes.
static PLAIN_SEEN: AtomicUsize = AtomicUsize::new(0);

/// The largest zeroed request of the run, and how many there were.
///
/// Kept outside the ordered log because they must be exact whatever the chunk
/// is: at a small chunk a `WIDE_DIM` run makes more chunk-sized requests than
/// [`LOG_CAP`] holds, and a truncated log would quietly under-report the very
/// number the bound is asserted on. The ordered log stays for diagnostics and
/// for the plain index.
static MAX_ZEROED: AtomicUsize = AtomicUsize::new(0);
/// Companion to [`MAX_ZEROED`].
static ZEROED_SEEN: AtomicUsize = AtomicUsize::new(0);

/// How many requests the ceiling actually refused, which is what tells a run
/// that completed because nothing was refused from one that completed despite
/// a refusal.
static REFUSALS: AtomicUsize = AtomicUsize::new(0);

/// Record a large request and report whether the ceiling refuses it.
///
/// Allocation-free on purpose: it writes into fixed statics rather than
/// pushing to a `Vec`, because a `Vec` growing inside `alloc` re-enters this
/// function.
fn note(size: usize, zeroed: bool) -> bool {
    if size < log_floor(zeroed) || !WATCHING.load(Ordering::Relaxed) {
        return false;
    }
    if zeroed {
        MAX_ZEROED.fetch_max(size, Ordering::Relaxed);
        ZEROED_SEEN.fetch_add(1, Ordering::Relaxed);
    }
    let i = LOG_LEN.fetch_add(1, Ordering::Relaxed);
    if i < LOG_CAP {
        LOG_SIZE[i].store(size, Ordering::Relaxed);
        LOG_ZEROED[i].store(zeroed, Ordering::Relaxed);
    }
    let refuse = if zeroed {
        size >= REFUSE_ZEROED_FROM.load(Ordering::Relaxed)
    } else {
        PLAIN_SEEN.fetch_add(1, Ordering::Relaxed) == REFUSE_PLAIN_AT.load(Ordering::Relaxed)
    };
    if refuse {
        REFUSALS.fetch_add(1, Ordering::Relaxed);
    }
    refuse
}

/// A [`System`] allocator that can refuse the large requests [`note`] picks.
struct Ceiling;

unsafe impl GlobalAlloc for Ceiling {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if note(layout.size(), false) {
            return ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if note(layout.size(), true) {
            return ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Recorded but never refused. A grow through here on the ICC path
        // would mean a buffer reserved short and filled past its reservation,
        // which is the failure `debug_assert_plane_geometry` guards in
        // `colour.rs`; logging it makes the probes red rather than letting it
        // pass as noise.
        note(new_size, false);
        unsafe { System.realloc(p, layout, new_size) }
    }
}

#[global_allocator]
static CEILING: Ceiling = Ceiling;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A tone curve moxcms classifies as degenerate: a long run of duplicated
/// leading entries, which `is_curve_degenerated` reports at more than twenty.
///
/// It is what puts the transform on the katana engine at all. moxcms fuses a
/// LUT profile into a single interpolated grid whenever the curve analysis in
/// `lut_hint.rs` says it may, and a fused transform allocates its grid once at
/// *creation* and nothing per call, so a well-behaved LUT profile would drive
/// this whole file past the buffer it exists to bound.
fn degenerate_curve() -> ToneReprCurve {
    let mut v = vec![0u16; 40];
    v.extend((0..216u16).map(|i| i * 300));
    ToneReprCurve::Lut(v)
}

/// An identity colour lookup table over a `grid`-per-side cube, 16-bit.
fn identity_clut(grid: usize) -> LutStore {
    let mut out = Vec::with_capacity(grid * grid * grid * 3);
    let code = |x: usize| ((x as f32 / (grid - 1) as f32) * 65535.0) as u16;
    for r in 0..grid {
        for g in 0..grid {
            for b in 0..grid {
                out.push(code(r));
                out.push(code(g));
                out.push(code(b));
            }
        }
    }
    LutStore::Store16(out)
}

/// An RGB **LUT** profile: no TRCs and no colorant shaper, an A2B0 and a B2A0
/// built from [`degenerate_curve`], and a Lab PCS.
///
/// The three properties matter and the child asserts all of them rather than
/// trusting this function, because each one silently drops the run onto a
/// different engine if it stops holding:
///
/// * not a matrix shaper, so `icc_device_to_lab` and `icc_lab_to_device`
///   dispatch past their exact arms into the CMS fallbacks;
/// * an A2B0 whose curves are degenerate, so the import side takes the katana
///   engine rather than a fused grid;
/// * a Lab PCS with a degenerate B2A0, so the export side does too. moxcms
///   short-circuits the destination analysis when `dest.pcs` is XYZ, so an XYZ
///   PCS here would fuse the export and leave half this file testing nothing.
fn lut_profile() -> ColorProfile {
    let grid = 9usize;
    let mab = LutWarehouse::Multidimensional(LutMultidimensionalType {
        num_input_channels: 3,
        num_output_channels: 3,
        grid_points: [
            grid as u8, grid as u8, grid as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        clut: Some(identity_clut(grid)),
        a_curves: vec![degenerate_curve(), degenerate_curve(), degenerate_curve()],
        b_curves: vec![degenerate_curve(), degenerate_curve(), degenerate_curve()],
        m_curves: vec![],
        matrix: Matrix3d::IDENTITY,
        bias: Vector3d::default(),
    });
    // Built on top of `new_srgb` because `ColorProfile`'s version field is
    // private, so the struct cannot be written out literally from here, and
    // because it supplies a white point the encoder will accept.
    let mut p = ColorProfile::new_srgb();
    p.profile_class = ProfileClass::DisplayDevice;
    p.color_space = DataColorSpace::Rgb;
    p.pcs = DataColorSpace::Lab;
    p.red_trc = None;
    p.green_trc = None;
    p.blue_trc = None;
    p.lut_a_to_b_perceptual = Some(mab.clone());
    p.lut_a_to_b_colorimetric = Some(mab.clone());
    p.lut_b_to_a_perceptual = Some(mab.clone());
    p.lut_b_to_a_colorimetric = Some(mab);
    p
}

/// A `dim`-square 8-bit RGB raster with `profile` attached.
fn profiled_fixture(dim: u32, profile: &[u8]) -> Raster {
    let pixels = (dim as usize) * (dim as usize);
    let mut data = Vec::with_capacity(pixels * 3);
    for i in 0..pixels {
        data.push((i % 251) as u8);
        data.push((i % 241) as u8);
        data.push((i % 239) as u8);
    }
    Raster::new(dim, dim, PixelFormat::Rgb8, data)
        .map(|mut im| {
            im.set_icc_profile(profile);
            im
        })
        .expect("fixture raster")
}

// ---------------------------------------------------------------------------
// The child
// ---------------------------------------------------------------------------

/// Set by the parent on every child it spawns. A `child_run` without it is
/// being run directly, which cannot work: some of its modes end the process on
/// purpose.
const CHILD_MARKER: &str = "L693_CHILD";

/// Which ICC direction a child drives: `import` or `export`.
const ENV_ROUTE: &str = "L693_ROUTE";

/// The side of the child's square fixture.
const ENV_DIM: &str = "L693_DIM";

/// `watch`, `refuse-zeroed` or `refuse-plain`.
const ENV_MODE: &str = "L693_MODE";

/// The byte bound for `refuse-zeroed`, the index for `refuse-plain`.
const ENV_ARG: &str = "L693_ARG";

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|e| panic!("{key}: {e}"))
}

/// Run one ICC direction with the ceiling up, then report to stdout.
///
/// The report is a handful of line kinds, and the parent needs all of them to
/// tell a mis-aimed run from a real one: the fixture's routing properties,
/// `ARMED <dim> <plane bytes>`, `ALLOC <bytes> <zeroed>` per large request in
/// order up to [`LOG_CAP`], `MAXZEROED <bytes>` and `ZEROEDSEEN <n>` counted
/// exactly outside that cap, `TRUNCATED <bool>`, `REFUSALS <n>`, and
/// `RESULT ok` or `RESULT err <e>`.
#[test]
#[ignore = "spawned as a child process by the checks below; some modes end the process on purpose"]
fn child_run() {
    assert!(
        std::env::var_os(CHILD_MARKER).is_some(),
        "child_run is spawned by the checks in this file and starves \
         allocations that can end the process; run those instead"
    );
    let dim: u32 = env(ENV_DIM).parse().expect("L693_DIM");
    let route = env(ENV_ROUTE);
    let mode = env(ENV_MODE);
    let arg: usize = env(ENV_ARG).parse().expect("L693_ARG");

    let profile = lut_profile();
    let bytes = profile.encode().expect("encode the fixture profile");
    let reparsed = ColorProfile::new_from_slice(&bytes).expect("reparse the fixture profile");
    // The fixture's routing properties, checked on the profile as the crate
    // will actually see it (parsed back from the encoded blob), not on the
    // struct this file built. An encoder that dropped the A2B0 would otherwise
    // leave every check below passing against a matrix shaper.
    println!("SHAPER {}", reparsed.is_matrix_shaper());
    println!("PCS {:?}", reparsed.pcs);
    println!("A2B {}", reparsed.lut_a_to_b_perceptual.is_some());
    println!("B2A {}", reparsed.lut_b_to_a_perceptual.is_some());

    let im = profiled_fixture(dim, &bytes);
    // The export direction needs a Lab input, so its import runs first and
    // outside the measured window.
    let source = match route.as_str() {
        "import" => im,
        "export" => im
            .try_icc_import_with(Intent::Perceptual, None, Some(Pcs::Lab))
            .expect("the unstarved import must succeed"),
        other => panic!("unknown route {other}"),
    };

    println!("ARMED {dim} {}", plane_bytes(dim));
    std::io::stdout().flush().expect("flush before the ceiling");

    match mode.as_str() {
        "watch" => {}
        "refuse-zeroed" => REFUSE_ZEROED_FROM.store(arg, Ordering::SeqCst),
        "refuse-plain" => REFUSE_PLAIN_AT.store(arg, Ordering::SeqCst),
        other => panic!("unknown mode {other}"),
    }
    WATCHING.store(true, Ordering::SeqCst);
    let result = match route.as_str() {
        "import" => source
            .try_icc_import_with(Intent::Perceptual, None, Some(Pcs::Lab))
            .map(|_| ()),
        _ => source
            .try_icc_export_with(8, Intent::Perceptual, None)
            .map(|_| ()),
    };
    WATCHING.store(false, Ordering::SeqCst);
    REFUSE_ZEROED_FROM.store(usize::MAX, Ordering::SeqCst);
    REFUSE_PLAIN_AT.store(usize::MAX, Ordering::SeqCst);

    for i in 0..LOG_LEN.load(Ordering::SeqCst).min(LOG_CAP) {
        println!(
            "ALLOC {} {}",
            LOG_SIZE[i].load(Ordering::SeqCst),
            LOG_ZEROED[i].load(Ordering::SeqCst)
        );
    }
    println!("MAXZEROED {}", MAX_ZEROED.load(Ordering::SeqCst));
    println!("ZEROEDSEEN {}", ZEROED_SEEN.load(Ordering::SeqCst));
    println!("TRUNCATED {}", LOG_LEN.load(Ordering::SeqCst) > LOG_CAP);
    println!("REFUSALS {}", REFUSALS.load(Ordering::SeqCst));
    match result {
        Ok(()) => println!("RESULT ok"),
        Err(e) => println!("RESULT err {e}"),
    }
    std::io::stdout().flush().expect("flush the report");
}

// ---------------------------------------------------------------------------
// The parent side
// ---------------------------------------------------------------------------

/// Run `child_run` in a fresh process under one route, geometry and mode.
///
/// `--test-threads=1` so the process-wide ceiling has exactly one test to
/// answer, and `--ignored --exact` so it has exactly that one.
fn spawn(route: &str, dim: u32, mode: &str, arg: usize) -> Output {
    Command::new(std::env::current_exe().expect("this test binary's own path"))
        .args([
            "--exact",
            "child_run",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_MARKER, "1")
        .env(ENV_ROUTE, route)
        .env(ENV_DIM, dim.to_string())
        .env(ENV_MODE, mode)
        .env(ENV_ARG, arg.to_string())
        .output()
        .unwrap_or_else(|e| panic!("spawning the {route} child at {dim}x{dim}: {e}"))
}

/// One large request as the child reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Alloc {
    bytes: usize,
    zeroed: bool,
}

/// The child's report.
struct Report {
    allocs: Vec<Alloc>,
    /// Exact, counted outside the ordered log so a small chunk cannot truncate
    /// it: the largest zeroed request, and how many there were.
    max_zeroed: usize,
    zeroed_seen: usize,
    /// Whether the ordered log dropped entries past [`LOG_CAP`].
    truncated: bool,
    refusals: usize,
    result: String,
    stdout: String,
    stderr: String,
}

impl Report {
    /// The largest zeroed request on the run, which on both routes is the
    /// moxcms katana intermediate and nothing else.
    ///
    /// `None` means the run made no zeroed request at all, which is a real
    /// finding rather than a zero: see the message on the scaling checks for
    /// the three things it can mean.
    fn largest_zeroed(&self) -> Option<usize> {
        (self.zeroed_seen > 0).then_some(self.max_zeroed)
    }

    fn context(&self) -> String {
        format!(
            "--- stdout ---\n{}\n--- stderr ---\n{}",
            self.stdout, self.stderr
        )
    }
}

/// Parse a child's report, insisting first that it finished cleanly and drove
/// the profile this file thinks it drove.
fn read_report(what: &str, out: &Output) -> Report {
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let context = format!("{what}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
    assert!(
        out.status.success(),
        "{what} was expected to finish cleanly, got {:?}\n{context}",
        out.status,
    );
    // The routing properties are asserted here rather than in the child so a
    // wrong one reads as a failed check rather than as a panicking child.
    // `ends_with` rather than equality: libtest prints its own
    // "test child_run ... " prefix on the line the first `println!` lands on,
    // so the opening report line is never alone on it.
    for want in ["SHAPER false", "PCS Lab", "A2B true", "B2A true"] {
        assert!(
            stdout.lines().any(|l| l.ends_with(want)),
            "{what} must drive a LUT profile, expected {want:?} in\n{context}"
        );
    }
    let mut allocs = Vec::new();
    let mut refusals = None;
    let mut result = None;
    let mut max_zeroed = None;
    let mut zeroed_seen = None;
    let mut truncated = None;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("ALLOC ") {
            let (bytes, zeroed) = rest.split_once(' ').expect("ALLOC line shape");
            allocs.push(Alloc {
                bytes: bytes.parse().expect("ALLOC size"),
                zeroed: zeroed.parse().expect("ALLOC zeroed flag"),
            });
        } else if let Some(rest) = line.strip_prefix("REFUSALS ") {
            refusals = Some(rest.parse().expect("REFUSALS count"));
        } else if let Some(rest) = line.strip_prefix("MAXZEROED ") {
            max_zeroed = Some(rest.parse().expect("MAXZEROED size"));
        } else if let Some(rest) = line.strip_prefix("ZEROEDSEEN ") {
            zeroed_seen = Some(rest.parse().expect("ZEROEDSEEN count"));
        } else if let Some(rest) = line.strip_prefix("TRUNCATED ") {
            truncated = Some(rest.parse().expect("TRUNCATED flag"));
        } else if let Some(rest) = line.strip_prefix("RESULT ") {
            result = Some(rest.to_string());
        }
    }
    Report {
        refusals: refusals.unwrap_or_else(|| panic!("no REFUSALS line\n{context}")),
        result: result.unwrap_or_else(|| panic!("no RESULT line\n{context}")),
        max_zeroed: max_zeroed.unwrap_or_else(|| panic!("no MAXZEROED line\n{context}")),
        zeroed_seen: zeroed_seen.unwrap_or_else(|| panic!("no ZEROEDSEEN line\n{context}")),
        truncated: truncated.unwrap_or_else(|| panic!("no TRUNCATED line\n{context}")),
        allocs,
        stdout,
        stderr,
    }
}

/// The shared body of the two scaling checks: watch one route at both
/// geometries and insist the moxcms buffer is the same size in each.
fn assert_cms_buffer_does_not_scale(route: &str) {
    let small = read_report(
        &format!("the LUT {route} at {DIM}x{DIM}"),
        &spawn(route, DIM, "watch", 0),
    );
    let large = read_report(
        &format!("the LUT {route} at {WIDE_DIM}x{WIDE_DIM}"),
        &spawn(route, WIDE_DIM, "watch", 0),
    );
    for (what, r, dim) in [("small", &small, DIM), ("large", &large, WIDE_DIM)] {
        assert_eq!(
            r.refusals, 0,
            "the {what} run was only watched, not starved"
        );
        assert_eq!(
            r.result,
            "ok",
            "the {what} run must succeed when nothing is refused\n{}",
            r.context()
        );
        assert!(
            r.allocs.iter().any(|a| a.bytes >= plane_bytes(dim)),
            "the {what} run never reserved an image-sized plane of its own, so \
             it is not running the route this file is about: {:?}",
            r.allocs
        );
    }
    let (a, b) = (small.largest_zeroed(), large.largest_zeroed());
    assert!(
        a.is_some() && b.is_some(),
        "the LUT {route} logged no zeroed allocation at one of the two sizes. \
         Three things do that and they are in the order you should check them, \
         because the cheapest to cause is listed first and it is entirely \
         local:\n\
         \x20 1. `ICC_TRANSFORM_CHUNK_PIXELS` in src/colour.rs dropped below \
         {} pixels, so the katana intermediate at twelve bytes a pixel no \
         longer clears this file's {ZEROED_LOG_FLOOR}-byte zeroed floor. \
         Tightening the chunk is an improvement and should not read as a \
         failure: lower `ZEROED_LOG_FLOOR` to suit.\n\
         \x20 2. the fixture profile stopped being degenerate enough to defeat \
         moxcms's LUT fusion, so the transform never enters the katana engine.\n\
         \x20 3. moxcms adopted its own `try_vec!` in the `md*` stages, which is \
         issue #693 landing upstream: bump the pin in Cargo.toml, widen the \
         `# Allocation` claim in src/colour.rs, and turn these checks into ones \
         that assert the `Err`.\n\
         small={:?} large={:?}",
        ZEROED_LOG_FLOOR.div_ceil(12),
        small.allocs,
        large.allocs
    );
    assert_eq!(
        a, b,
        "the largest buffer moxcms allocated grew with the image, so it is \
         still being handed the whole plane: {DIM}x{DIM} asked for {a:?} and \
         {WIDE_DIM}x{WIDE_DIM} asked for {b:?}"
    );
    assert!(
        a.unwrap() <= CMS_CEILING_BYTES,
        "the largest buffer moxcms allocated on the LUT {route} is {a:?}, past \
         the {CMS_CEILING_BYTES}-byte bound src/colour.rs promises"
    );
}

/**
 * Tests that the LUT import hands moxcms nothing whose size follows the image.
 *
 * This is the headline claim of #693's fix. moxcms allocates the katana
 * intermediates with a plain `vec![0f32; n]` sized from the slice it is given,
 * and nothing here can make that fallible, so the only lever is `n`. Driving
 * the transform in bounded pixel chunks takes the largest request it makes off
 * the image size and onto a fixed one, which is what this measures: the same
 * route at 256x256 and at 512x512, and the largest *zeroed* request has to be
 * the same number in both.
 * Input: a Rgb8 raster with a degenerate-curve RGB LUT profile through
 * `try_icc_import_with` at both geometries -> equal largest zeroed request,
 * at or below 524288 bytes.
 * Mutation: handing `xf.transform` the whole plane again makes the two 786432
 * and 3145728 and reddens both halves of the assertion.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn the_lut_import_hands_moxcms_nothing_that_scales_with_the_image() {
    assert_cms_buffer_does_not_scale("import");
}

/**
 * The export twin of the check above, over `try_icc_export_with`.
 * Input: the same fixture imported to Lab and exported at both geometries ->
 * equal largest zeroed request, at or below 524288 bytes.
 * Mutation: as above.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn the_lut_export_hands_moxcms_nothing_that_scales_with_the_image() {
    assert_cms_buffer_does_not_scale("export");
}

/// The shared body of the two survival checks: refuse every zeroed request at
/// or above the promised bound and insist the route completes anyway.
fn assert_survives_the_cms_ceiling(route: &str) {
    let what = format!("the LUT {route} under a {CMS_CEILING_BYTES}-byte CMS ceiling");
    let out = spawn(route, DIM, "refuse-zeroed", CMS_CEILING_BYTES);
    let report = read_report(&what, &out);
    assert_eq!(
        report.refusals, 0,
        "{what} offered the ceiling a zeroed request at or above the bound, so \
         moxcms is still being handed a buffer that scales with the image: {:?}",
        report.allocs
    );
    assert_eq!(
        report.result,
        "ok",
        "{what} must run to completion\n{}",
        report.context()
    );
}

/**
 * Tests the same bound from the other side: with a ceiling that refuses every
 * zeroed request at or above what `src/colour.rs` promises, a LUT import runs
 * to completion instead of ending the process.
 *
 * The scaling check above measures the request. This one lets the allocator
 * answer it, which is the difference between "the number looks right" and "the
 * process survives". The ceiling is a real `GlobalAlloc` refusing a real
 * request, not a `cfg(test)` hook, so it cannot short-circuit ahead of the
 * call it is aimed at: it refuses only *zeroed* requests, and every buffer
 * this crate reserves arrives as a plain `alloc`.
 * Input: the 256x256 LUT fixture through `try_icc_import_with` with every
 * zeroed request at or above 524288 bytes refused -> exit 0, zero refusals,
 * `RESULT ok`.
 * Mutation: handing `xf.transform` the whole plane again makes moxcms ask for
 * 786432 zeroed bytes, the ceiling refuses it, and the child dies in
 * `handle_alloc_error` with SIGABRT.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn a_lut_import_survives_a_ceiling_at_the_promised_cms_bound() {
    assert_survives_the_cms_ceiling("import");
}

/**
 * The export twin of the check above.
 * Input: the 256x256 LUT fixture imported to Lab and exported with every
 * zeroed request at or above 524288 bytes refused -> exit 0, zero refusals,
 * `RESULT ok`.
 * Mutation: as above.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn a_lut_export_survives_a_ceiling_at_the_promised_cms_bound() {
    assert_survives_the_cms_ceiling("export");
}

/**
 * Tests that the crate's own image-sized reservations report an `Err` through
 * the *real* allocator, rather than through the `cfg(test)` ceiling the unit
 * tests use.
 *
 * This is the cell #689's fourteen did not have. Every one of those drives
 * `refuse_over_colour_cap`, a `cfg(test)` size check that answers *before*
 * `Vec::try_reserve_exact` is ever called, so all fourteen stay green with the
 * reservation reverted to the infallible `reserve_exact`. Here the allocator
 * itself hands back null on the first large plain request of the run, which is
 * the device plane `read_device_normalised` reserves, so the fallible spelling
 * is the only thing between that null and `handle_alloc_error`.
 * Input: the 256x256 LUT fixture through `try_icc_import_with` with the first
 * large plain request refused -> exit 0, one refusal, `RESULT err` naming an
 * allocation failure.
 * Mutation: `alloc_colour_plane`'s `try_reserve_exact` put back to
 * `reserve_exact` reddens this and nothing else in the file.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn refusing_the_crate_s_own_device_plane_reports_an_error_rather_than_aborting() {
    let what = "the LUT import with its first large plain request refused";
    let out = spawn("import", DIM, "refuse-plain", 0);
    let report = read_report(what, &out);
    assert_eq!(
        report.refusals, 1,
        "{what} should have been refused exactly once, got {:?}",
        report.allocs
    );
    assert_eq!(
        report.allocs.first().map(|a| a.bytes),
        Some(plane_bytes(DIM)),
        "the first large plain request should be the three-f32 device plane"
    );
    assert!(
        report.result.starts_with("err "),
        "{what} must come back as an Err rather than running to completion or \
         dying\n{}",
        report.context()
    );
    assert!(
        report.result.contains("allocat"),
        "the Err should name the allocation failure, got {:?}",
        report.result
    );
}

/**
 * Tests the residue, so the docs cannot quietly widen past it: moxcms still
 * allocates a buffer of its own infallibly, and refusing *that* still ends the
 * process.
 *
 * Bounding the request is not the same as making it fallible, and this file
 * would otherwise read as if it were. It is also the positive control the rest
 * of the file needs: without it, four green checks are equally consistent with
 * a ceiling that never refuses anything at all.
 * Input: the 256x256 LUT fixture through `try_icc_import_with` with every
 * zeroed request at or above the 512-byte zeroed floor refused -> SIGABRT, and a
 * `handle_alloc_error` message naming a size at or below the 524288-byte bound.
 * Mutation: handing `xf.transform` the whole plane again keeps the abort but
 * moves the size in that message to 786432, past the bound, which reddens the
 * second half of this check.
 */
#[test]
#[cfg_attr(miri, ignore)]
fn refusing_what_moxcms_still_allocates_itself_ends_the_process() {
    let out = spawn("import", DIM, "refuse-zeroed", ZEROED_LOG_FLOOR);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let context = format!("--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
    assert!(
        stdout.contains(&format!("ARMED {DIM} ")),
        "the child died before it reached the measured window\n{context}"
    );
    assert!(
        !out.status.success(),
        "refusing the buffer moxcms allocates for itself returned instead of \
         aborting. Check the local cause first: if \
         `ICC_TRANSFORM_CHUNK_PIXELS` dropped below {} pixels, the katana \
         intermediate stopped clearing this file's {ZEROED_LOG_FLOOR}-byte \
         zeroed floor, so the ceiling was never offered it and the child had \
         nothing to die on. Only once that is ruled out does this mean moxcms \
         adopted its own `try_vec!` in the katana `md*` stages, which is issue \
         #693 landing upstream: bump the pin in Cargo.toml, widen the \
         `# Allocation` claim in src/colour.rs, and turn this check into one \
         that asserts the Err.\n{context}",
        ZEROED_LOG_FLOOR.div_ceil(12)
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        assert_eq!(
            out.status.signal(),
            Some(6),
            "handle_alloc_error ends the process with SIGABRT\n{context}"
        );
    }
    // The message names the size, which is what makes this a check on the
    // residue's *bound* and not only on its existence.
    let marker = "memory allocation of ";
    let size: usize = stderr
        .split_once(marker)
        .and_then(|(_, rest)| rest.split_once(' '))
        .map(|(n, _)| n.parse().expect("the aborted allocation's size"))
        .unwrap_or_else(|| panic!("no handle_alloc_error message\n{context}"));
    assert!(
        size <= CMS_CEILING_BYTES,
        "moxcms aborted over a {size}-byte buffer, past the \
         {CMS_CEILING_BYTES}-byte bound src/colour.rs promises for it"
    );
}
