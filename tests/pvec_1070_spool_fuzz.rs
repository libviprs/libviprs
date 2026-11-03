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
#[cfg_attr(miri, ignore)]
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

/// One drawing, one budget, and the sink's failure point swept across the whole artifact. The
/// failure point is two bytes of the input (the `0..=2000` field), so the sweep walks the sink from
/// "fails on the first byte" to "has room for all of it" and has to flip from `SinkFailed` to
/// `Written` exactly once, at the artifact's length: a sink that fails on the very last byte still
/// fails the write and leaves nothing behind.
/// Input bytes that are 5 mod 16, so the coordinate ratio never picks an arbitrary float, and that
/// differ from one another, so the polyline is not degenerate.
const FILL: [u8; 64] = {
    let mut b = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        b[i] = (i as u8).wrapping_mul(16).wrapping_add(5);
        i += 1;
    }
    b
};

fn sweep(indexed: bool, budget_index: u8) -> usize {
    let run = |limit: usize| {
        let mut data = vec![
            u8::from(indexed),
            budget_index,
            0,
            (limit >> 8) as u8,
            limit as u8,
            0,
        ];
        data.extend(FILL);
        spool::run(&data)
    };
    let flip = (0..=2000usize)
        .find(|&l| !matches!(run(l), spool::Outcome::SinkFailed))
        .expect("the sink has room for the artifact somewhere under 2000 bytes");
    assert!(flip > 0, "an artifact with no bytes");
    assert!(
        matches!(run(flip), spool::Outcome::Written { .. }),
        "room for exactly the artifact's {flip} bytes must finish the write"
    );
    assert!(
        matches!(run(flip - 1), spool::Outcome::SinkFailed),
        "one byte short of the artifact must fail the write"
    );
    for l in 0..flip {
        assert!(
            matches!(run(l), spool::Outcome::SinkFailed),
            "limit {l} of {flip}"
        );
    }
    flip
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1070_a_sink_that_fails_on_the_last_byte_fails_the_write_plain_and_indexed() {
    for indexed in [false, true] {
        for budget_index in [0u8, 4] {
            sweep(indexed, budget_index);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1070_a_budget_of_zero_still_streams_the_same_bytes() {
    for indexed in [false, true] {
        let mut data = vec![u8::from(indexed), 0, 1, 0];
        data.extend(FILL);
        assert!(
            matches!(spool::run(&data), spool::Outcome::Written { spilled: true }),
            "budget 0 must spill and still match the in-memory bytes (indexed {indexed})"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1070_the_crafted_seeds_sit_on_the_edges_they_were_made_for() {
    let by_name = |n: &str| seeds().into_iter().find(|(s, _)| s == n).unwrap().1;
    for n in ["seed16", "seed17"] {
        assert!(
            matches!(
                spool::run(&by_name(n)),
                spool::Outcome::Written { spilled: true }
            ),
            "{n}: budget 0 must spill"
        );
    }
    assert!(
        matches!(spool::run(&by_name("seed18")), spool::Outcome::SinkFailed),
        "seed18: a sink one byte short of the indexed artifact must fail"
    );
    assert!(
        matches!(
            spool::run(&by_name("seed19")),
            spool::Outcome::Written { .. }
        ),
        "seed19: a sink with room for exactly the indexed artifact must finish"
    );
}
