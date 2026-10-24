//! A hostile-DWG corpus through the real decoder and the tiler (issue #1036).
//!
//! The native decoder is code libviprs links and cannot stop, so a crash in it
//! is a crash of whatever process called it. These tests therefore run every
//! mutated drawing in a child process: the parent asserts that no child died
//! of a signal or a panic, which is what "no panic or UB from the corpus"
//! means for a parser that is not Rust.
//!
//! Needs the native archive; without one the lane says so, unless
//! `LIBVIPRS_REQUIRE_ACADSHARP_NATIVE=1`, as in the other decoder tests.
#![cfg(all(feature = "dwg-acadsharp", target_os = "linux"))]

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use libviprs::cad::acadsharp::{AcadSharpDecoder, ProviderLimits};
use libviprs::cad::{CadDecoder, CadSource, ViewKind};
use libviprs::vector::tiler::{TilerConfig, build_archive_from_view};

const FILE: &str = "VIPRS_L23_FILE";
const MUTATIONS: u64 = 40;
/// A drawing that opens and decodes is a pass, and so is one that is refused.
const EXIT_PROCESSED: i32 = 0;

fn fixtures() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cad");
    let mut v: Vec<PathBuf> = std::fs::read_dir(root.join("g13"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "dwg"))
        // The deep-dimension drawing is the pathological case and is tested
        // by name below; a mutation of a megabyte is not a quick one.
        .filter(|p| p.metadata().unwrap().len() < 64 << 10)
        .collect();
    v.push(root.join("sample_AC1018.dwg"));
    v.sort();
    v
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *s >> 33
}

/// The `n`th deterministic mutation of `base`: byte flips, a zeroed run, a
/// truncation, or a run copied over another.
fn mutate(base: &[u8], n: u64) -> Vec<u8> {
    let mut s = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut b = base.to_vec();
    if b.len() < 32 {
        return b;
    }
    let len = b.len() as u64;
    match n % 4 {
        0 => {
            for _ in 0..1 + lcg(&mut s) % 8 {
                // Past the version stamp, so the decoder is reached.
                let at = 6 + (lcg(&mut s) % (len - 6)) as usize;
                b[at] ^= 1 << (lcg(&mut s) % 8);
            }
        }
        1 => {
            let at = 6 + (lcg(&mut s) % (len - 6)) as usize;
            let run = (lcg(&mut s) % 256) as usize;
            for x in b.iter_mut().skip(at).take(run) {
                *x = 0;
            }
        }
        2 => b.truncate(6 + (lcg(&mut s) % (len - 6)) as usize),
        _ => {
            let a = 6 + (lcg(&mut s) % (len - 6)) as usize;
            let c = 6 + (lcg(&mut s) % (len - 6)) as usize;
            let run = (lcg(&mut s) % 512) as usize;
            let chunk: Vec<u8> = b.iter().skip(a).take(run).copied().collect();
            for (i, x) in chunk.into_iter().enumerate() {
                if let Some(d) = b.get_mut(c + i) {
                    *d = x;
                }
            }
        }
    }
    b
}

fn convert(bytes: &[u8], dir: &Path) {
    let Ok(decoder) = AcadSharpDecoder::new() else {
        return;
    };
    let decoder = decoder.with_limits(ProviderLimits::hardened());
    let Ok(drawing) = decoder.open(CadSource::Bytes(bytes)) else {
        return;
    };
    let Ok(views) = drawing.views() else { return };
    let Some(model) = views.iter().find(|v| v.kind() == ViewKind::Model) else {
        return;
    };
    let cfg = TilerConfig {
        max_primitives: 200_000,
        ..TilerConfig::default().with_max_zoom(6)
    }
    .with_deadline(Duration::from_secs(20));
    let _ = build_archive_from_view(&*drawing, model.index(), &cfg, dir.join("o.pmtiles"), "x");
}

/// The child's side: every mutation of one file, in this process.
#[test]
fn child_entry() {
    let Ok(file) = std::env::var(FILE) else {
        return;
    };
    let base = std::fs::read(&file).unwrap();
    let dir = tempfile::tempdir().unwrap();
    for n in 0..MUTATIONS {
        println!("mutation {n}");
        convert(&mutate(&base, n), dir.path());
    }
    std::process::exit(EXIT_PROCESSED);
}

fn native_present() -> bool {
    match AcadSharpDecoder::new() {
        Ok(_) => true,
        Err(e) => {
            assert!(
                std::env::var("LIBVIPRS_REQUIRE_ACADSHARP_NATIVE").as_deref() != Ok("1"),
                "the native lane was required and the decoder did not start: {e:?}"
            );
            eprintln!("skipped: no native acadsharp library linked ({e})");
            false
        }
    }
}

#[test]
fn mutated_drawings_never_take_the_process_down() {
    if !native_present() {
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let mut decoded = 0usize;
    for file in fixtures() {
        let t = Instant::now();
        let out = Command::new(&exe)
            .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
            .env(FILE, &file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        let last = String::from_utf8_lossy(&out.stdout)
            .lines()
            .rev()
            .find(|l| l.starts_with("mutation"))
            .unwrap_or("none")
            .to_owned();
        assert!(
            out.status.success(),
            "{} died on {last} (signal {:?}, code {:?}) after {:?}\n{}",
            file.display(),
            out.status.signal(),
            out.status.code(),
            t.elapsed(),
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .join("\n")
        );
        decoded += 1;
    }
    eprintln!("MEASURED {decoded} drawings x {MUTATIONS} mutations, no child died");
    assert!(decoded >= 20, "the corpus shrank to {decoded} drawings");
}

#[test]
fn the_hardened_limits_set_every_bound() {
    let l = ProviderLimits::hardened();
    for (name, v) in [
        ("max_input_bytes", l.max_input_bytes),
        ("max_entities", l.max_entities),
        ("max_string_bytes", l.max_string_bytes),
        ("max_polyline_points", l.max_polyline_points),
        ("max_block_depth", l.max_block_depth),
        ("max_output_bytes", l.max_output_bytes),
    ] {
        assert!(v.is_some(), "{name} is unbounded in the hardened set");
    }
}

#[test]
fn the_pathological_deep_dimension_drawing_ends_inside_a_bound() {
    if !native_present() {
        return;
    }
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cad/g13/g13_dimension_deep.dwg"),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let t = Instant::now();
    convert(&bytes, dir.path());
    eprintln!(
        "MEASURED g13_dimension_deep under hardened limits: {:?}",
        t.elapsed()
    );
    assert!(
        t.elapsed() < Duration::from_secs(30),
        "took {:?}",
        t.elapsed()
    );
}

#[test]
fn the_provider_limits_actually_bite() {
    if !native_present() {
        return;
    }
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cad/g13/g13_many_inserts.dwg"),
    )
    .unwrap();
    let decode = |limits: ProviderLimits| {
        let decoder = AcadSharpDecoder::new().unwrap().with_limits(limits);
        let drawing = decoder.open(CadSource::Bytes(&bytes)).unwrap();
        let model = drawing
            .views()
            .unwrap()
            .into_iter()
            .find(|v| v.kind() == ViewKind::Model)
            .unwrap();
        let mut sink = libviprs::cad::CollectSink::new();
        let mut report = libviprs::cad::DecodeReport::new();
        let r = drawing.decode(model.index(), &mut sink, &mut report);
        (r.is_ok(), sink.len())
    };
    let (ok, free) = decode(ProviderLimits::default());
    assert!(ok && free >= 30_000, "the unbounded decode gave {free}");
    // The same drawing with the entity bound a hundred times under what it
    // holds: the decode may end in an error or in a short, reported result,
    // and must not deliver the whole drawing.
    let tight = ProviderLimits {
        max_entities: Some(300),
        ..ProviderLimits::default()
    };
    let (_, held) = decode(tight);
    assert!(
        held < free / 10,
        "max_entities=300 still delivered {held} of {free}"
    );
}
