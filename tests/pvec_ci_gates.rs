//! N2.8 (#1060): the four PVec CI gates are four separate jobs of `.github/workflows/ci.yml`, so
//! the local gate (`tools/local-ci.py`) runs them by name and a failure says which one it was.
//!
//! This file holds the shape of that: each gate is its own job with its own name, runs only its own
//! suites, publishes its own artifact, and the baselines two of them compare against are checked
//! in. A suite listed under two gates, or a gate that quietly stopped running one of its suites,
//! fails here.

const CI_YML: &str = include_str!("../.github/workflows/ci.yml");

/// One gate: its job key, its display name, the test targets it runs and where its artifacts go.
/// The four fields vary together per gate.
struct Gate {
    job: &'static str,
    name: &'static str,
    suites: &'static [&'static str],
    out: &'static str,
}

const GATES: &[Gate] = &[
    Gate {
        job: "pvec-gate-fidelity",
        name: "PVec gate: mathematical fidelity",
        suites: &["pvec_known_truth", "pvec_structural", "pvec_geometric"],
        out: "fidelity",
    },
    Gate {
        job: "pvec-gate-view",
        name: "PVec gate: visibility and view correctness",
        suites: &["pvec_view_transform", "pvec_gate_view"],
        out: "view",
    },
    Gate {
        job: "pvec-gate-classification",
        name: "PVec gate: unsupported-classification stability",
        suites: &["pvec_gate_classification", "pvec_dual_oracle"],
        out: "classification",
    },
    Gate {
        job: "pvec-gate-determinism",
        name: "PVec gate: deterministic output",
        suites: &["pvec_gate_determinism", "pvec_end_to_end"],
        out: "determinism",
    },
];

/// The text of one job: from its key at two spaces to the next key at two spaces.
fn job_block(job: &str) -> &str {
    let head = format!("\n  {job}:\n");
    let start = CI_YML
        .find(&head)
        .unwrap_or_else(|| panic!("ci.yml has no job `{job}`"))
        + 1;
    let rest = &CI_YML[start + head.len() - 1..];
    let end = rest
        .lines()
        .skip(1)
        .scan(0, |at, l| {
            let here = *at;
            *at += l.len() + 1;
            Some((here, l))
        })
        .find(|(_, l)| {
            l.starts_with("  ")
                && !l.starts_with("   ")
                && !l.trim().is_empty()
                && !l.trim_start().starts_with('#')
        })
        .map_or(rest.len(), |(at, _)| at);
    &rest[..end]
}

#[test]
fn n2_8_1060_there_are_four_gates_and_each_is_its_own_named_job() {
    assert_eq!(GATES.len(), 4);
    for g in GATES {
        let block = job_block(g.job);
        assert!(
            block.contains(&format!("    name: {}\n", g.name)),
            "{}: the job must be named `{}`",
            g.job,
            g.name
        );
    }
    let names: std::collections::BTreeSet<_> = GATES.iter().map(|g| g.name).collect();
    assert_eq!(names.len(), 4, "four distinct names");
}

#[test]
fn n2_8_1060_each_gate_runs_its_own_suites_and_no_other_gates() {
    for g in GATES {
        let block = job_block(g.job);
        for suite in g.suites {
            assert!(
                block.contains(&format!("--test {suite}")),
                "{}: does not run `--test {suite}`",
                g.name
            );
        }
        for other in GATES.iter().filter(|o| o.job != g.job) {
            for suite in other.suites.iter().filter(|s| !g.suites.contains(s)) {
                assert!(
                    !block.contains(&format!("--test {suite} "))
                        && !block.contains(&format!("--test {suite}\n")),
                    "{}: runs `{suite}`, which belongs to the `{}` gate",
                    g.name,
                    other.name
                );
            }
        }
    }
    // A suite is listed under one gate only, except the end-to-end path, which the determinism
    // gate runs filtered to its byte comparison.
    let mut seen = std::collections::BTreeMap::new();
    for g in GATES {
        for s in g.suites {
            assert!(
                seen.insert(*s, g.name).is_none(),
                "`{s}` is listed under two gates"
            );
        }
    }
}

#[test]
fn n2_8_1060_the_determinism_gate_runs_the_end_to_end_path_only_for_its_byte_comparison() {
    let block = job_block("pvec-gate-determinism");
    assert!(
        block.contains("--test pvec_end_to_end -- byte_identical determinism"),
        "the end-to-end suite is filtered to the byte comparison and the ordering perturbation"
    );
}

#[test]
fn n2_8_1060_each_gate_publishes_its_own_artifact_directory() {
    for g in GATES {
        let block = job_block(g.job);
        let dir = format!("target/pvec-gates/{}", g.out);
        assert!(
            block.contains(&format!("PVEC_GATE_OUT: {dir}\n")),
            "{}: PVEC_GATE_OUT must be {dir}",
            g.name
        );
        assert!(
            block.contains("actions/upload-artifact@"),
            "{}: no artifact upload",
            g.name
        );
        assert!(
            block.contains(&format!("path: {dir}\n"))
                && block.contains(&format!("name: {}\n", g.job)),
            "{}: the upload must be named `{}` and take `{dir}`",
            g.name,
            g.job
        );
        assert!(
            block.contains("if: always()"),
            "{}: the artifact is uploaded when the gate is red too",
            g.name
        );
    }
}

#[test]
fn n2_8_1060_no_gate_starts_a_container_or_needs_a_converter() {
    for g in GATES {
        let block = job_block(g.job).to_lowercase();
        for needle in ["docker", "cad exchanger", "oda "] {
            assert!(!block.contains(needle), "{}: mentions {needle:?}", g.name);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the test sources from disk
fn n2_8_1060_only_the_fidelity_gate_reads_the_fixture_expectations() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let read = |s: &str| std::fs::read_to_string(root.join(format!("{s}.rs"))).unwrap();
    let needles = [
        ["expected", ".math.json"].concat(),
        [".tru", "th()"].concat(),
    ];
    for g in GATES.iter().filter(|g| g.out != "fidelity") {
        for s in g.suites {
            // The end-to-end path compares with the expectations in tests the determinism gate
            // filters out, so it is the one suite exempt here.
            if *s == "pvec_end_to_end" {
                continue;
            }
            let text = read(s);
            for n in &needles {
                assert!(
                    !text.contains(n.as_str()),
                    "{}: `{s}` reads {n:?}, so a mutated expectation would redden this gate too",
                    g.name
                );
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the baselines from disk
fn n2_8_1060_the_baselines_are_checked_in_sorted_and_not_empty() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-gate");
    for name in ["classification", "view-decisions", "determinism"] {
        let text = std::fs::read_to_string(dir.join(format!("{name}.baseline")))
            .unwrap_or_else(|e| panic!("{name}.baseline is not checked in: {e}"));
        let ids: Vec<&str> = text.lines().map(|l| l.split(' ').next().unwrap()).collect();
        assert!(ids.len() >= 50, "{name}: only {} lines", ids.len());
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(ids, sorted, "{name}: one sorted line per fixture");
    }
}
