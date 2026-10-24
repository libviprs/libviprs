//! Memory stays bounded while the tiler works on hostile input (issue #1036).
//!
//! The binary installs a counting allocator, so a bound here is a measured
//! peak of live heap and not an estimate. The counter is global, so the tests
//! take a lock and run one at a time.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use libviprs::cad::{Line, Primitive};
use libviprs::vector::mvt::Tile;
use libviprs::vector::tiler::{TileOutput, TilerConfig, TilerError, tile_primitives};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, Relaxed) + by;
    PEAK.fetch_max(now, Relaxed);
}

// SAFETY: every method forwards to `System` unchanged and only adds counting.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is passed straight through.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: the caller's contract is passed straight through.
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        // SAFETY: the caller's contract is passed straight through.
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            if n >= l.size() {
                grew(n - l.size());
            } else {
                LIVE.fetch_sub(l.size() - n, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

static SERIAL: Mutex<()> = Mutex::new(());

/// Throws tiles away, so the measurement is the tiler's and not a collector's.
#[derive(Default)]
struct Discard {
    tiles: u64,
}

impl TileOutput for Discard {
    fn put(&mut self, _z: u8, _x: u32, _y: u32, t: &Tile) -> Result<Option<usize>, TilerError> {
        if t.is_empty() {
            return Ok(None);
        }
        self.tiles += 1;
        Ok(Some(t.encode().unwrap().len()))
    }
}

fn line(a: [f64; 2], b: [f64; 2]) -> Primitive {
    Line::new([a[0], a[1], 0.0], [b[0], b[1], 0.0])
        .unwrap()
        .into()
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 33) as f64) / ((1u64 << 31) as f64)
}

/// Runs `f` and returns what it returned with the most live heap it added.
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let r = f();
    (r, PEAK.load(Relaxed).saturating_sub(base))
}

const MIB: usize = 1 << 20;

#[test]
fn a_level_is_held_to_its_byte_budget_not_only_its_point_budget() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // 1500 lines from the top of the drawing to the bottom. At zoom 10 each
    // one splits over about a thousand tiles: 1.5 million two-point runs,
    // which is 3 million points and well under the point budget, and which
    // costs a hundred megabytes of bookkeeping.
    let mut s = 7;
    let prims: Vec<Primitive> = (0..1500)
        .map(|_| line([lcg(&mut s) * 1.0e5, 0.0], [lcg(&mut s) * 1.0e5, 1.0e5]))
        .collect();
    let budget = 16 * MIB;
    let cfg = TilerConfig::default()
        .with_max_zoom(10)
        .with_max_level_bytes(budget);
    let (result, peak) = peak_of(|| {
        let mut out = Discard::default();
        tile_primitives(&prims, &cfg, &mut out).map(|_| ())
    });
    assert!(
        matches!(result, Err(TilerError::LevelBudgetExceeded { .. })),
        "got {result:?}"
    );
    assert!(
        peak <= 2 * budget,
        "peak live heap was {} MiB against a {} MiB level budget",
        peak / MIB,
        budget / MIB
    );
}

#[test]
fn a_deep_zoom_tile_explosion_stops_inside_its_budget() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let prims = vec![line([0.0, 0.0], [1.0e5, 1.0e5])];
    let budget = 32 * MIB;
    let cfg = TilerConfig::default()
        .with_max_zoom(24)
        .with_max_level_bytes(budget);
    let t = std::time::Instant::now();
    let (result, peak) = peak_of(|| {
        let mut out = Discard::default();
        tile_primitives(&prims, &cfg, &mut out).map(|_| ())
    });
    assert!(
        matches!(result, Err(TilerError::LevelBudgetExceeded { .. })),
        "got {result:?}"
    );
    assert!(peak <= 2 * budget, "peak {} MiB", peak / MIB);
    assert!(t.elapsed().as_secs() < 60, "took {:?}", t.elapsed());
}

#[test]
fn a_million_primitives_stay_inside_the_default_budget() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut s = 1;
    let prims: Vec<Primitive> = (0..1_000_000)
        .map(|_| {
            let (x, y) = (lcg(&mut s) * 1.0e5, lcg(&mut s) * 1.0e5);
            line([x, y], [x + lcg(&mut s) * 50.0, y + lcg(&mut s) * 50.0])
        })
        .collect();
    let cfg = TilerConfig::default().with_max_zoom(6);
    let (result, peak) = peak_of(|| {
        let mut out = Discard::default();
        let r = tile_primitives(&prims, &cfg, &mut out);
        (r.map(|(rep, _)| rep.primitives), out.tiles)
    });
    let (primitives, tiles) = (result.0.unwrap(), result.1);
    assert_eq!(primitives, 1_000_000);
    assert!(tiles > 0);
    // The held primitives are the caller's and already counted by the time
    // the run starts, so this is the tiler's own working set.
    assert!(
        peak <= cfg.max_level_bytes,
        "a million primitives took {} MiB of working memory, past the {} MiB budget",
        peak / MIB,
        cfg.max_level_bytes / MIB
    );
    eprintln!("MEASURED 1,000,000 primitives, zoom 0..=6: {tiles} tiles, peak working heap {} MiB", peak / MIB);
}
