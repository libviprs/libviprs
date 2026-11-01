//! The rows of the corpus regression report and the diff that gates them (N7.8, #1444).
//!
//! Every non-timing column here is deterministic: the same fixture converts to the same classes,
//! the same view decisions and the same bytes on every host, so the diff runs in the ordinary
//! suite. Wall time and RSS are in the ignored report only.

#![allow(dead_code)] // each suite uses its own part

use serde_json::{Map, Value, json};

/// The conformance of a converted fixture, from best to worst. EPIC P's meanings: `SupportedExact`
/// is the structure preserved, `SupportedEquivalent` the same locus under another construct. A
/// display-fidelity result is never either.
pub const CONFORMANCE: [&str; 3] = ["SupportedExact", "SupportedEquivalent", "NoGeometry"];

fn rank(c: &str) -> usize {
    CONFORMANCE
        .iter()
        .position(|x| *x == c)
        .unwrap_or(CONFORMANCE.len())
}

/// The conformance of a report's class counts: the worst class any written entity landed in.
pub fn conformance(by_fidelity: &Map<String, Value>) -> &'static str {
    let n = |k: &str| by_fidelity.get(k).and_then(Value::as_u64).unwrap_or(0);
    if n("SupportedExact") + n("SupportedEquivalent") == 0 {
        "NoGeometry"
    } else if n("SupportedEquivalent") > 0 {
        "SupportedEquivalent"
    } else {
        "SupportedExact"
    }
}

/// Every way `cur` regresses against `base`. Both are `{"rows": {fixture id: row}}`. A fixture
/// whose evidence class is `QUARANTINED` is in the report and never gates.
pub fn regressions(base: &Value, cur: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let (b, c) = (
        base["rows"].as_object().cloned().unwrap_or_default(),
        cur["rows"].as_object().cloned().unwrap_or_default(),
    );
    let gated = |r: &Value| r["evidence"].as_str() != Some("QUARANTINED");
    for id in c.keys().filter(|id| !b.contains_key(*id)) {
        out.push(format!(
            "{id}: not in the baseline, regenerate it and review the row"
        ));
    }
    let (mut total_base, mut total_cur) = (0.0, 0.0);
    for (id, was) in &b {
        let Some(now) = c.get(id) else {
            out.push(format!("{id}: in the baseline and missing from the run"));
            continue;
        };
        if !gated(was) || !gated(now) {
            continue;
        }
        let (cw, cn) = (
            was["mathematical"]["conformance"]
                .as_str()
                .unwrap_or("NoGeometry"),
            now["mathematical"]["conformance"]
                .as_str()
                .unwrap_or("NoGeometry"),
        );
        if rank(cn) > rank(cw) {
            out.push(format!("{id}: moved from {cw} to {cn}"));
        }
        let exact = |r: &Value| {
            r["mathematical"]["by_fidelity"]["SupportedExact"]
                .as_u64()
                .unwrap_or(0)
        };
        if exact(now) < exact(was) {
            out.push(format!(
                "{id}: {} entities fell out of SupportedExact",
                exact(was) - exact(now)
            ));
        }
        if was["outcome"] == "converted" && now["outcome"] != "converted" {
            out.push(format!("{id}: no longer converts ({})", now["outcome"]));
        }
        let bytes = |r: &Value| r["size"]["artifact_bytes"].as_f64().unwrap_or(0.0);
        total_base += bytes(was);
        total_cur += bytes(now);
        if bytes(was) > 0.0 && bytes(now) > bytes(was) * 1.05 {
            out.push(format!(
                "{id}: the artifact grew from {} to {} bytes, more than 5%",
                bytes(was),
                bytes(now)
            ));
        }
    }
    if total_base > 0.0 && total_cur > total_base * 1.02 {
        out.push(format!(
            "total .pvec bytes grew from {total_base} to {total_cur}, more than 2%"
        ));
    }
    out
}

/// A document for a set of rows.
pub fn document(revision: &str, rows: Vec<(String, Value)>) -> Value {
    json!({"schema": 1, "corpus_revision": revision, "rows": rows.into_iter().collect::<Map<String, Value>>()})
}
