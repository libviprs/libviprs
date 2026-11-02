//! N7.6 (#1437, slice of #1070): the fuzz targets' bodies run on their committed seed corpus in
//! the ordinary suite, and a recursion bomb is refused by N3.6's limits under a direct test.
//!
//! The bodies are the files under `fuzz/fuzz_targets/pvec_bodies/`, included here by path, so what
//! `cargo fuzz` runs and what this runs are the same code. Needs `--features
//! "pvec-normalized pvec-spline"`.

#![cfg(all(feature = "pvec-normalized", feature = "pvec-spline"))]

#[path = "../fuzz/fuzz_targets/pvec_bodies/clip.rs"]
mod clip;
#[path = "../fuzz/fuzz_targets/pvec_bodies/provider.rs"]
mod provider;
#[path = "../fuzz/fuzz_targets/pvec_bodies/spline.rs"]
mod spline;

use std::path::PathBuf;

fn seeds(target: &str) -> Vec<(String, Vec<u8>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz/corpus")
        .join(target);
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

fn run_all(target: &str, body: fn(&[u8])) {
    let seeds = seeds(target);
    assert!(
        seeds.len() >= 8,
        "{target}: only {} seeds are committed",
        seeds.len()
    );
    for (name, bytes) in &seeds {
        let r = std::panic::catch_unwind(|| body(bytes));
        assert!(r.is_ok(), "{target}: the seed {name} panics");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_provider_seed_corpus_runs_clean() {
    run_all("fuzz_pvec_provider", provider::run);
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_spline_seed_corpus_runs_clean() {
    run_all("fuzz_pvec_spline", spline::run);
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_clip_seed_corpus_runs_clean() {
    run_all("fuzz_pvec_clip", clip::run);
}

#[test]
fn a_recursion_bomb_is_refused_by_the_limits_and_does_not_exhaust_the_stack() {
    for levels in [1u8, 2, 5, 200] {
        let outcome = provider::run_mock(&provider::recursion_bomb(levels))
            .expect("the drawing has a model space");
        let why = outcome.expect_err(&format!(
            "{levels} levels of mutual recursion were accepted"
        ));
        assert!(
            why.contains("recursion") || why.contains("cycle") || why.contains("max_block_depth"),
            "{levels} levels: refused, but not by the recursion rule: {why}"
        );
    }
}
