//! Checked-in baselines for the PVec CI gates (N2.8, #1060).
//!
//! A gate that compares against a baseline keeps the baseline in `tests/fixtures/pvec-gate/`, one
//! line per fixture, sorted, so a change is a reviewable diff in the pull request and never a log
//! line. [`check`] compares the freshly computed text with the file and, when `PVEC_GATE_OUT` is
//! set, leaves the gate's artifacts there: `<name>.actual` (what this run computed) and
//! `<name>.diff` (empty when the baseline holds). `PVEC_GATE_BLESS=1` rewrites the baseline, which
//! is how a deliberate change is recorded.
//!
//! A baseline is the x86_64 one. A line that comes out differently on another architecture is kept
//! in `<name>.<arch>.baseline` next to it (only the differing lines), and that file replaces the
//! line of the same fixture when the suite runs there. Blessing on x86_64 writes the base file and
//! blessing elsewhere writes the override, so a divergence between architectures is a checked-in,
//! reviewable row and not a flaky gate.

#![allow(dead_code)] // each gate suite uses its own part

use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub fn baseline_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec-gate")
        .join(format!("{name}.baseline"))
}

/// The override file for this architecture, `None` on x86_64 where the base file is the truth.
pub fn arch_override_path(name: &str) -> Option<PathBuf> {
    (std::env::consts::ARCH != "x86_64").then(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/pvec-gate")
            .join(format!("{name}.{}.baseline", std::env::consts::ARCH))
    })
}

fn first_word(l: &str) -> &str {
    l.split_whitespace().next().unwrap_or("")
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The lines that differ between two baselines of one-line-per-fixture records, as `- old` and
/// `+ new` lines in fixture order. Empty when they agree.
pub fn diff(baseline: &str, actual: &str) -> Vec<String> {
    let key = |l: &str| l.split_whitespace().next().unwrap_or("").to_owned();
    let old: std::collections::BTreeMap<String, &str> =
        baseline.lines().map(|l| (key(l), l)).collect();
    let new: std::collections::BTreeMap<String, &str> =
        actual.lines().map(|l| (key(l), l)).collect();
    let mut keys: Vec<&String> = old.keys().chain(new.keys()).collect();
    keys.sort();
    keys.dedup();
    let mut out = Vec::new();
    for k in keys {
        match (old.get(k), new.get(k)) {
            (Some(a), Some(b)) if a == b => {}
            (a, b) => {
                if let Some(a) = a {
                    out.push(format!("- {a}"));
                }
                if let Some(b) = b {
                    out.push(format!("+ {b}"));
                }
            }
        }
    }
    out
}

/// Compares `actual` with the checked-in baseline `name`, panicking with the gate named and the
/// differing lines when they disagree.
pub fn check(gate: &str, name: &str, actual: &str) {
    let mut lines: Vec<&str> = actual.lines().collect();
    lines.sort_unstable();
    let actual = &(lines.join("\n") + "\n");
    let path = baseline_path(name);
    let base_text = || {
        fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{gate} gate: the baseline {} is missing ({e}); PVEC_GATE_BLESS=1 writes it",
                path.display()
            )
        })
    };
    if std::env::var_os("PVEC_GATE_BLESS").is_some() {
        match arch_override_path(name) {
            None => fs::write(&path, actual).unwrap(),
            Some(over) => {
                let base = base_text();
                let changed: Vec<&str> = actual
                    .lines()
                    .filter(|l| !base.lines().any(|b| b == *l))
                    .collect();
                if changed.is_empty() {
                    let _ = fs::remove_file(&over);
                } else {
                    fs::write(&over, changed.join("\n") + "\n").unwrap();
                }
            }
        }
        return;
    }
    let mut baseline = base_text();
    if let Some(over) = arch_override_path(name).filter(|p| p.exists()) {
        let rows = fs::read_to_string(over).unwrap();
        let keep: Vec<&str> = baseline
            .lines()
            .filter(|l| !rows.lines().any(|r| first_word(r) == first_word(l)))
            .chain(rows.lines())
            .collect();
        baseline = keep.join("\n") + "\n";
    }
    let changes = diff(&baseline, actual);
    if let Some(out) = std::env::var_os("PVEC_GATE_OUT") {
        let out = PathBuf::from(out);
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join(format!("{name}.actual")), actual).unwrap();
        fs::write(out.join(format!("{name}.diff")), changes.join("\n")).unwrap();
    }
    assert!(
        changes.is_empty(),
        "{gate} gate: {} line(s) differ from tests/fixtures/pvec-gate/{name}.baseline:\n{}",
        changes.len(),
        changes.join("\n")
    );
}
