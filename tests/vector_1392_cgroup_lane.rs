//! The cgroup form of the worker test is a lane, not a manual run (issue #1392, part of #1256).
//!
//! The suite proves the `ulimit -v` form of the memory ceiling. What a real
//! deployer gets is a cgroup limit, where the child is OOM-killed (exit 137)
//! and leaves only its staging pair. A test binary cannot make its own cgroup,
//! so the lane is a script that runs the same child under `docker run
//! --memory`, called by the MARS scheduler after the native cells. This file
//! holds the script, its wiring and the report together.

use std::path::Path;

fn read(rel: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn the_lane_runs_the_unbudgeted_child_under_a_memory_cgroup_and_checks_the_result() {
    let lane = read("tools/cgroup-worker-lane.sh");
    for needle in [
        "--memory=256m",
        "--memory-swap=256m",
        "VIPRS_L23_MODE=unbudgeted",
        "vector_1036_worker",
        "OOMKilled",
        "137",
        "a.pmtiles",
    ] {
        assert!(
            lane.contains(needle),
            "the lane script must contain `{needle}`"
        );
    }
    for l in lane.lines().map(str::trim_start) {
        assert!(
            !(l.starts_with("cargo ") || l.starts_with("rustup ") || l.starts_with("curl ")),
            "the lane runs on a shared host and may only call docker: {l}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn the_child_modes_the_lane_drives_still_exist() {
    let worker = read("tests/vector_1036_worker.rs");
    for needle in [
        "VIPRS_L23_MODE",
        "VIPRS_L23_OUT",
        "\"unbudgeted\"",
        "fn child_entry",
    ] {
        assert!(
            worker.contains(needle),
            "tests/vector_1036_worker.rs lost `{needle}`"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn the_scheduler_runs_the_lane_and_the_report_calls_it_one() {
    assert!(
        read("tools/scheduled-native-dwg.sh").contains("tools/cgroup-worker-lane.sh"),
        "the MARS scheduler has to run the cgroup lane"
    );
    let report = read("docs/vector-performance-report.md");
    assert!(
        !report.contains("That cgroup run is a manual measurement"),
        "the report still calls the cgroup run manual"
    );
    assert!(report.contains("tools/cgroup-worker-lane.sh"));
    assert!(read("CONTRIBUTING.md").contains("tools/cgroup-worker-lane.sh"));
}
