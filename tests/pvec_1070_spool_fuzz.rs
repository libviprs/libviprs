//! N7.6 (#1070): the `fuzz_pvec_spool` target. The spool exists now (the writer's spill, #1314 and
//! #1417), so the fourth target of #1437 has an input: a drawing, a spill budget and a sink that may
//! fail. Its body is shared with the target by path, like the other three. Needs `--features
//! pvec-write`.

#![cfg(feature = "pvec-write")]

#[path = "../fuzz/fuzz_targets/pvec_bodies/spool.rs"]
mod spool;

use std::path::PathBuf;

fn seeds() -> Vec<(String, Vec<u8>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/corpus/fuzz_pvec_spool");
    let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| {
            let e = e.unwrap();
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect();
    v.sort();
    v
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1070_the_spool_seed_corpus_runs_clean() {
    let seeds = seeds();
    assert!(seeds.len() >= 8, "only {} seeds are committed", seeds.len());
    for (name, bytes) in &seeds {
        let r = std::panic::catch_unwind(|| spool::run(bytes));
        assert!(r.is_ok(), "the seed {name} panics");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1070_the_seeds_reach_both_a_finished_write_and_a_failing_sink() {
    let outcomes: Vec<spool::Outcome> = seeds().iter().map(|(_, b)| spool::run(b)).collect();
    assert!(
        outcomes
            .iter()
            .any(|o| matches!(o, spool::Outcome::Written { .. })),
        "no seed finishes a write, so the byte-equality check never runs"
    );
    assert!(
        outcomes
            .iter()
            .any(|o| matches!(o, spool::Outcome::SinkFailed)),
        "no seed fails its sink, so the cleanup check never runs"
    );
    assert!(
        outcomes
            .iter()
            .any(|o| matches!(o, spool::Outcome::Written { spilled: true, .. })),
        "no seed spills, so the spool is never exercised"
    );
}

#[test]
fn issue_1070_the_target_is_registered_with_the_fuzz_crate() {
    let manifest =
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/Cargo.toml"))
            .unwrap();
    assert!(manifest.contains("name = \"fuzz_pvec_spool\""));
    assert!(
        manifest.contains("\"pvec-write\""),
        "the fuzz crate does not enable pvec-write"
    );
    assert!(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fuzz/fuzz_targets/fuzz_pvec_spool.rs")
            .is_file()
    );
}
