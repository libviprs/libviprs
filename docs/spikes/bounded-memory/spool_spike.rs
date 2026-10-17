// N1.7 spike: three ways to produce a spatially ordered index for a large
// stream of entities. Std only, build with `rustc -O spool_spike.rs`.
//
// This is a measurement harness. The "entity" is a synthetic record with a
// centre point and an opaque payload, and the "index" is a stand-in for any
// output structure that is ordered by a key quantised over the *global*
// extents of the drawing. It is not the PVec format and it does not call PVec.
//
// usage: spool_spike <direct|twopass|spool> <entities> <dir> [run_records]
//
// One strategy per process, so VmHWM (peak resident set) is that strategy's own.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

// ---- synthetic source ------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct Entity {
    id: u32,
    cx: f64,
    cy: f64,
    payload_len: u32,
}

/// Deterministic entity `i`. Position is a mixture: 70% inside one of 64
/// clusters, the rest uniform, so the global extents are not known from the
/// first few entities and the spatial order is not the arrival order.
fn entity(i: u32, buf: &mut Vec<u8>) -> Entity {
    let mut r = Rng(0xC0FFEE ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let (cx, cy) = if r.next() % 10 < 7 {
        let c = r.next() % 64;
        let mut cr = Rng(0xABCD ^ c);
        let (ox, oy) = (cr.unit() * 1.0e6, cr.unit() * 1.0e6);
        (ox + (r.unit() - 0.5) * 2000.0, oy + (r.unit() - 0.5) * 2000.0)
    } else {
        (r.unit() * 1.0e6, r.unit() * 1.0e6)
    };
    // 48 to 160 bytes of payload, standing in for a polyline or spline body.
    let payload_len = 48 + 16 * (r.next() % 8) as u32;
    buf.clear();
    for _ in 0..payload_len / 8 {
        buf.extend_from_slice(&r.next().to_le_bytes());
    }
    Entity { id: i, cx, cy, payload_len }
}

// ---- key -------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Extents {
    minx: f64,
    miny: f64,
    maxx: f64,
    maxy: f64,
}
impl Extents {
    fn empty() -> Self {
        Extents { minx: f64::INFINITY, miny: f64::INFINITY, maxx: f64::NEG_INFINITY, maxy: f64::NEG_INFINITY }
    }
    fn add(&mut self, x: f64, y: f64) {
        self.minx = self.minx.min(x);
        self.miny = self.miny.min(y);
        self.maxx = self.maxx.max(x);
        self.maxy = self.maxy.max(y);
    }
}

fn spread(v: u32) -> u64 {
    let mut x = v as u64;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Morton key of the centre, quantised to 32 bits per axis over `e`. This is
/// the part that needs the global extents.
fn key(e: &Extents, x: f64, y: f64) -> u64 {
    let q = |v: f64, lo: f64, hi: f64| -> u32 {
        let span = hi - lo;
        if span <= 0.0 { 0 } else { (((v - lo) / span) * 4294967295.0) as u32 }
    };
    spread(q(x, e.minx, e.maxx)) | (spread(q(y, e.miny, e.maxy)) << 1)
}

// ---- records ---------------------------------------------------------------

/// Final index entry. 24 bytes in memory and on disk.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Entry {
    key: u64,
    id: u32,
    len: u32,
    off: u64,
}
const ENTRY_BYTES: usize = 24;
impl Entry {
    fn put(&self, w: &mut impl Write) {
        w.write_all(&self.key.to_le_bytes()).unwrap();
        w.write_all(&self.id.to_le_bytes()).unwrap();
        w.write_all(&self.len.to_le_bytes()).unwrap();
        w.write_all(&self.off.to_le_bytes()).unwrap();
    }
    fn get(r: &mut impl Read) -> Option<Entry> {
        let mut b = [0u8; ENTRY_BYTES];
        match r.read_exact(&mut b) {
            Ok(()) => Some(Entry {
                key: u64::from_le_bytes(b[0..8].try_into().unwrap()),
                id: u32::from_le_bytes(b[8..12].try_into().unwrap()),
                len: u32::from_le_bytes(b[12..16].try_into().unwrap()),
                off: u64::from_le_bytes(b[16..24].try_into().unwrap()),
            }),
            Err(_) => None,
        }
    }
}

/// Arrival-order record, before the key exists. 32 bytes.
#[derive(Clone, Copy)]
struct Raw {
    cx: f64,
    cy: f64,
    off: u64,
    len: u32,
    id: u32,
}
const RAW_BYTES: usize = 32;

/// What direct streaming has to hold: a raw record plus the key computed at
/// finish. 40 bytes.
#[derive(Clone, Copy)]
struct Held {
    raw: Raw,
    key: u64,
}

// ---- measurement -----------------------------------------------------------

fn vm_kb(field: &str) -> u64 {
    let s = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for l in s.lines() {
        if let Some(rest) = l.strip_prefix(field) {
            return rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
        }
    }
    0
}

/// User plus system CPU time in ms, from /proc/self/stat (100 ticks a second).
/// Unlike wall time it does not move when the host is loaded.
fn cpu_ms() -> u64 {
    let s = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let tail = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = tail.split_whitespace().collect();
    // After the ')' the fields restart at index 0 = state (field 3), so
    // utime (field 14) is index 11 and stime (field 15) is index 12.
    let u: u64 = f.get(11).and_then(|v| v.parse().ok()).unwrap_or(0);
    let k: u64 = f.get(12).and_then(|v| v.parse().ok()).unwrap_or(0);
    (u + k) * 10
}

fn fnv(path: &str) -> u64 {
    let mut f = BufReader::with_capacity(1 << 20, File::open(path).unwrap());
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        for &b in &buf[..n] {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

fn size(path: &str) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

// ---- strategies ------------------------------------------------------------

struct Stats {
    source_bytes: u64,
    reread_bytes: u64,
    pass1_ms: u128,
    pass1_rss_kb: u64,
    spool_bytes: u64,
    run_bytes_written: u64,
    runs: u64,
    merge_passes: u64,
    sort_buffer_bytes: u64,
}

fn open_w(p: &str) -> BufWriter<File> {
    BufWriter::with_capacity(1 << 20, File::create(p).unwrap())
}

fn direct(n: u32, dir: &str, st: &mut Stats) {
    let mut payload = open_w(&format!("{dir}/payload.bin"));
    let mut held: Vec<Held> = Vec::new();
    let mut ext = Extents::empty();
    let mut buf = Vec::new();
    let mut off = 0u64;
    for i in 0..n {
        let e = entity(i, &mut buf);
        payload.write_all(&buf).unwrap();
        st.source_bytes += e.payload_len as u64 + 16;
        ext.add(e.cx, e.cy);
        held.push(Held { raw: Raw { cx: e.cx, cy: e.cy, off, len: e.payload_len, id: e.id }, key: 0 });
        off += e.payload_len as u64;
    }
    payload.flush().unwrap();
    // The extents are only complete now. Keys cannot be computed earlier.
    for h in held.iter_mut() {
        h.key = key(&ext, h.raw.cx, h.raw.cy);
    }
    held.sort_unstable_by_key(|h| (h.key, h.raw.id));
    let mut idx = open_w(&format!("{dir}/index.bin"));
    for h in &held {
        Entry { key: h.key, id: h.raw.id, len: h.raw.len, off: h.raw.off }.put(&mut idx);
    }
    idx.flush().unwrap();
}

fn twopass(n: u32, dir: &str, st: &mut Stats) {
    let t = Instant::now();
    let mut ext = Extents::empty();
    let mut buf = Vec::new();
    for i in 0..n {
        let e = entity(i, &mut buf);
        ext.add(e.cx, e.cy);
        st.reread_bytes += e.payload_len as u64 + 16;
    }
    st.pass1_ms = t.elapsed().as_millis();
    st.pass1_rss_kb = vm_kb("VmHWM:");
    let mut payload = open_w(&format!("{dir}/payload.bin"));
    let mut entries: Vec<Entry> = Vec::new();
    let mut off = 0u64;
    for i in 0..n {
        let e = entity(i, &mut buf);
        payload.write_all(&buf).unwrap();
        st.source_bytes += e.payload_len as u64 + 16;
        entries.push(Entry { key: key(&ext, e.cx, e.cy), id: e.id, len: e.payload_len, off });
        off += e.payload_len as u64;
    }
    payload.flush().unwrap();
    entries.sort_unstable();
    let mut idx = open_w(&format!("{dir}/index.bin"));
    for e in &entries {
        e.put(&mut idx);
    }
    idx.flush().unwrap();
}

const MAX_FANIN: usize = 128;

fn spool(n: u32, dir: &str, run_records: usize, st: &mut Stats) {
    // Pass A: one traversal. Payload goes out in arrival order, a raw record
    // goes to the spool, extents accumulate in O(1).
    let mut payload = open_w(&format!("{dir}/payload.bin"));
    let mut sp = open_w(&format!("{dir}/spool.bin"));
    let mut ext = Extents::empty();
    let mut buf = Vec::new();
    let mut off = 0u64;
    for i in 0..n {
        let e = entity(i, &mut buf);
        payload.write_all(&buf).unwrap();
        st.source_bytes += e.payload_len as u64 + 16;
        ext.add(e.cx, e.cy);
        sp.write_all(&e.cx.to_le_bytes()).unwrap();
        sp.write_all(&e.cy.to_le_bytes()).unwrap();
        sp.write_all(&off.to_le_bytes()).unwrap();
        sp.write_all(&e.payload_len.to_le_bytes()).unwrap();
        sp.write_all(&e.id.to_le_bytes()).unwrap();
        off += e.payload_len as u64;
    }
    payload.flush().unwrap();
    sp.flush().unwrap();
    st.spool_bytes = size(&format!("{dir}/spool.bin"));
    st.pass1_rss_kb = vm_kb("VmHWM:");

    // Pass B: read the spool back in bounded chunks, key each record against
    // the now-known extents, sort the chunk, write it as a run.
    st.sort_buffer_bytes = (run_records * ENTRY_BYTES) as u64;
    let mut rd = BufReader::with_capacity(1 << 20, File::open(format!("{dir}/spool.bin")).unwrap());
    let mut runs: Vec<String> = Vec::new();
    let mut chunk: Vec<Entry> = Vec::with_capacity(run_records);
    let mut b = [0u8; RAW_BYTES];
    let mut done = false;
    while !done {
        chunk.clear();
        while chunk.len() < run_records {
            if rd.read_exact(&mut b).is_err() {
                done = true;
                break;
            }
            let cx = f64::from_le_bytes(b[0..8].try_into().unwrap());
            let cy = f64::from_le_bytes(b[8..16].try_into().unwrap());
            chunk.push(Entry {
                key: key(&ext, cx, cy),
                off: u64::from_le_bytes(b[16..24].try_into().unwrap()),
                len: u32::from_le_bytes(b[24..28].try_into().unwrap()),
                id: u32::from_le_bytes(b[28..32].try_into().unwrap()),
            });
        }
        if chunk.is_empty() {
            break;
        }
        chunk.sort_unstable();
        let p = format!("{dir}/run-0-{}.bin", runs.len());
        let mut w = open_w(&p);
        for e in &chunk {
            e.put(&mut w);
        }
        w.flush().unwrap();
        st.run_bytes_written += (chunk.len() * ENTRY_BYTES) as u64;
        runs.push(p);
    }
    st.runs = runs.len() as u64;
    drop(chunk);

    // Merge with a capped fan-in. More than MAX_FANIN runs costs another pass,
    // not more file handles.
    let mut level = 0;
    while runs.len() > 1 {
        let last = runs.len() <= MAX_FANIN;
        let mut next: Vec<String> = Vec::new();
        for (gi, group) in runs.chunks(MAX_FANIN).enumerate() {
            let out = if last { format!("{dir}/index.bin") } else { format!("{dir}/run-{}-{}.bin", level + 1, gi) };
            merge(group, &out);
            if !last {
                st.run_bytes_written += size(&out);
            }
            for g in group {
                fs::remove_file(g).unwrap();
            }
            next.push(out);
        }
        st.merge_passes += 1;
        runs = next;
        level += 1;
    }
    if runs.len() == 1 && !fs::metadata(format!("{dir}/index.bin")).is_ok() {
        fs::rename(&runs[0], format!("{dir}/index.bin")).unwrap();
    }
}

fn merge(inputs: &[String], out: &str) {
    let mut readers: Vec<BufReader<File>> =
        inputs.iter().map(|p| BufReader::with_capacity(64 << 10, File::open(p).unwrap())).collect();
    let mut heap: BinaryHeap<Reverse<(Entry, usize)>> = BinaryHeap::new();
    for (i, r) in readers.iter_mut().enumerate() {
        if let Some(e) = Entry::get(r) {
            heap.push(Reverse((e, i)));
        }
    }
    let mut w = open_w(out);
    while let Some(Reverse((e, i))) = heap.pop() {
        e.put(&mut w);
        if let Some(nx) = Entry::get(&mut readers[i]) {
            heap.push(Reverse((nx, i)));
        }
    }
    w.flush().unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: spool_spike <direct|twopass|spool> <entities> <dir> [run_records]");
        std::process::exit(2);
    }
    let n: u32 = a[2].parse().unwrap();
    let dir = &a[3];
    let run_records: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(1 << 20);
    fs::create_dir_all(dir).unwrap();
    let mut st = Stats {
        source_bytes: 0,
        reread_bytes: 0,
        pass1_ms: 0,
        pass1_rss_kb: 0,
        spool_bytes: 0,
        run_bytes_written: 0,
        runs: 0,
        merge_passes: 0,
        sort_buffer_bytes: 0,
    };
    let t = Instant::now();
    match a[1].as_str() {
        "direct" => direct(n, dir, &mut st),
        "twopass" => twopass(n, dir, &mut st),
        "spool" => spool(n, dir, run_records, &mut st),
        _ => panic!("strategy"),
    }
    let wall = t.elapsed().as_millis();
    let peak = vm_kb("VmHWM:");
    let cpu = cpu_ms();
    let hash = fnv(&format!("{dir}/index.bin"));
    println!(
        "strategy={} entities={} wall_ms={} cpu_ms={} peak_rss_kb={} pass1_ms={} pass1_rss_kb={} source_bytes={} reread_bytes={} payload_bytes={} spool_bytes={} run_bytes_written={} runs={} merge_passes={} sort_buffer_bytes={} index_bytes={} index_fnv={:016x}",
        a[1], n, wall, cpu, peak, st.pass1_ms, st.pass1_rss_kb, st.source_bytes, st.reread_bytes,
        size(&format!("{dir}/payload.bin")), st.spool_bytes, st.run_bytes_written, st.runs,
        st.merge_passes, st.sort_buffer_bytes, size(&format!("{dir}/index.bin")), hash
    );
}
