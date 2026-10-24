//! A SaaS worker cannot be monopolised by an uploaded drawing (issue #1036).
//!
//! A static library is packaging, not a sandbox: the limits that hold when
//! the code inside will not stop are the process's. So these tests run the
//! conversion in a child process, the way a worker pool does, under a memory
//! ceiling and a wall-clock supervisor, and check what the parent sees.
//!
//! The child is this same test binary, re-run on one test with a mode in the
//! environment, so nothing here needs a helper executable.
#![cfg(target_os = "linux")]

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use libviprs::cad::{Line, Primitive};
use libviprs::vector::tiler::{TilerConfig, TilerError, build_archive};

const MODE: &str = "VIPRS_L23_MODE";
const OUT: &str = "VIPRS_L23_OUT";

/// Exit code the child uses when the library stopped it on its own deadline.
const EXIT_DEADLINE: i32 = 3;

fn explosive() -> Vec<Primitive> {
    let l = |a: [f64; 3], b: [f64; 3]| -> Primitive { Line::new(a, b).unwrap().into() };
    vec![
        l([0.0, 0.0, 0.0], [1.0e5, 1.0e5, 0.0]),
        l([0.0, 1.0e5, 0.0], [1.0e5, 0.0, 0.0]),
    ]
}

/// The child's side. A plain run of the suite does nothing here.
#[test]
fn child_entry() {
    let Ok(mode) = std::env::var(MODE) else { return };
    let out = PathBuf::from(std::env::var(OUT).unwrap());
    match mode.as_str() {
        // What a naive worker does: every budget switched off. The memory
        // ceiling is what has to stop it.
        "unbudgeted" => {
            let mut cfg = TilerConfig::default().with_max_zoom(24);
            cfg.max_level_points = usize::MAX;
            cfg.max_level_bytes = usize::MAX;
            let r = build_archive(&explosive(), &cfg, &out, "x");
            std::process::exit(i32::from(r.is_err()));
        }
        // Code that never looks at a token or a clock, which is what a native
        // decoder is from here.
        "spin" => loop {
            std::hint::black_box(0u64);
        },
        // The library's own deadline, with the budgets off, inside a ceiling.
        "deadline" => {
            let mut cfg = TilerConfig::default()
                .with_max_zoom(24)
                .with_deadline(Duration::from_millis(500));
            cfg.max_level_points = usize::MAX;
            cfg.max_level_bytes = usize::MAX;
            match build_archive(&explosive(), &cfg, &out, "x") {
                Err(TilerError::DeadlineExceeded { .. }) => std::process::exit(EXIT_DEADLINE),
                other => panic!("expected the deadline, got {other:?}"),
            }
        }
        other => panic!("unknown mode {other}"),
    }
}

struct Outcome {
    status: Option<ExitStatus>,
    killed_by_supervisor: bool,
    elapsed: Duration,
}

/// Runs the child under `ulimit -v` and kills it if it outlives `wall`.
fn supervise(mode: &str, out: &Path, address_space_kib: u64, wall: Duration) -> Outcome {
    let exe = std::env::current_exe().unwrap();
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(r#"ulimit -v "$1"; shift; exec "$@""#)
        .arg("sh")
        .arg(address_space_kib.to_string())
        .arg(exe)
        .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
        .env(MODE, mode)
        .env(OUT, out)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return Outcome {
                status: Some(status),
                killed_by_supervisor: false,
                elapsed: start.elapsed(),
            };
        }
        if start.elapsed() >= wall {
            child.kill().unwrap();
            let status = child.wait().ok();
            return Outcome {
                status,
                killed_by_supervisor: true,
                elapsed: start.elapsed(),
            };
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn the_parent_is_still_serviceable(dir: &Path) {
    let path = dir.join("after.pmtiles");
    let one: Vec<Primitive> = vec![Line::new([0.0; 3], [10.0, 5.0, 0.0]).unwrap().into()];
    build_archive(&one, &TilerConfig::default().with_max_zoom(3), &path, "ok").unwrap();
    assert!(path.exists());
}

const GIB_KIB: u64 = 1 << 20;

#[test]
fn a_memory_ceiling_stops_a_run_with_every_budget_off() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("a.pmtiles");
    let o = supervise("unbudgeted", &out, GIB_KIB, Duration::from_secs(120));
    let status = o.status.unwrap();
    assert!(!o.killed_by_supervisor, "the ceiling did not bite in time");
    assert!(
        !status.success(),
        "a run with no budget finished inside a 1 GiB ceiling: {status:?}"
    );
    assert!(!out.exists(), "a run that died published an archive");
    the_parent_is_still_serviceable(dir.path());
}

#[test]
fn a_supervisor_stops_code_that_never_looks_at_a_clock() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("a.pmtiles");
    let o = supervise("spin", &out, 4 * GIB_KIB, Duration::from_millis(1500));
    assert!(o.killed_by_supervisor);
    assert_eq!(o.status.unwrap().signal(), Some(9));
    assert!(
        o.elapsed < Duration::from_secs(10),
        "took {:?} to stop",
        o.elapsed
    );
    assert!(!out.exists());
    the_parent_is_still_serviceable(dir.path());
}

#[test]
fn the_library_deadline_ends_a_run_cleanly_inside_the_ceiling() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("a.pmtiles");
    let o = supervise("deadline", &out, 4 * GIB_KIB, Duration::from_secs(60));
    assert!(!o.killed_by_supervisor, "the deadline did not stop the run");
    assert_eq!(
        o.status.unwrap().code(),
        Some(EXIT_DEADLINE),
        "it should leave by its own deadline, not by the ceiling"
    );
    assert!(o.elapsed < Duration::from_secs(20), "{:?}", o.elapsed);
    assert!(!out.exists());
    let left: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert!(left.is_empty(), "a clean stop left {left:?}");
    the_parent_is_still_serviceable(dir.path());
}
