//! N8.4b to N8.4d (#1518, #1519, #1520, slices of #1481): classify holds each entity once, the
//! geometry goes as soon as the writer has it, the report is written without being built as one
//! string, and the output does not move.
//!
//! The budget test converts `large_industrial` at a twentieth of the manifest's size and reads the
//! counting allocator: the tracked peak of the whole conversion, minus what the opened provider
//! holds on its own, per entity encountered. The pin test converts the cells and compares the
//! artifact and the report with the hashes taken from the code before the change.
//!
//! One process, one allocator: the cells take a lock so another test's allocations are not in the
//! peak.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_alloc.rs"]
mod pvec_alloc;
#[path = "common/pvec_bench.rs"]
mod pvec_bench;
#[path = "common/pvec_stages.rs"]
mod pvec_stages;

use std::sync::{Arc, Mutex};

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::publish::report_path;
use libviprs::pvec_convert::{PvecConversion, PvecConvertError};
use libviprs::{CancelToken, EngineEvent, EngineObserver};
use pvec_bench::{CELLS, cells, generate, manifest, sha256_hex};
use serde_json::{Value, json};

#[global_allocator]
static ALLOC: pvec_alloc::Counting = pvec_alloc::Counting;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

const PIN: &str = "tests/fixtures/pvec_1481_output_pin.json";

/// Tracked bytes per entity encountered, above the opened provider, that a conversion of
/// `large_industrial` may peak at. The code before #1518 measured 1,895 here; 1,329 after it, which
/// is what #1519 tightens to 1,200 (1,119 after).
const BUDGET_BYTES_PER_ENTITY: u64 = 1_200;

/// The generator parameters of `cell` as the manifest has them, `large_industrial` at 1/20.
fn params(cell: &str) -> Value {
    let m = manifest();
    let mut p = cells(&m)[cell]["params"].clone();
    if cell == "large_industrial" {
        p["entities"] = json!(p["entities"].as_u64().unwrap() / 20);
    }
    p
}

fn drawing(cell: &str) -> Vec<u8> {
    let m = manifest();
    let g = cells(&m)[cell]["generator"].as_str().unwrap().to_owned();
    generate(&g, &params(cell))
}

/// The artifact's and the report's sha256 after converting `bytes`.
fn hashes(bytes: &[u8]) -> (String, String) {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("cell.pvec");
    PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .expect("converts");
    (
        sha256_hex(&std::fs::read(&dest).unwrap()),
        sha256_hex(&std::fs::read(report_path(&dest)).unwrap()),
    )
}

#[test]
#[cfg_attr(miri, ignore)]
fn classify_peak_above_the_opened_provider_stays_under_budget_1518() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = drawing("large_industrial");
    let (opened, _) = pvec_stages::open_only(&bytes);
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pvec");
    let mark = pvec_alloc::mark();
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .expect("converts");
    let peak = pvec_alloc::since(mark).0;
    let entities = out.report.totals.encountered;
    let per_entity = peak.saturating_sub(opened) / entities;
    println!("1518 peak {peak} opened {opened} entities {entities} per_entity {per_entity}");
    assert!(
        per_entity <= BUDGET_BYTES_PER_ENTITY,
        "{per_entity} bytes an entity above the opened provider, budget {BUDGET_BYTES_PER_ENTITY}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn classify_is_byte_identical_on_the_cells_1518() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut now = serde_json::Map::new();
    for cell in CELLS.iter().filter(|c| **c != "malformed") {
        let (artifact, report) = hashes(&drawing(cell));
        now.insert(
            (*cell).to_owned(),
            json!({"artifact": artifact, "report": report}),
        );
    }
    if let Ok(out) = std::env::var("LIBVIPRS_1518_PIN_OUT") {
        std::fs::write(
            out,
            serde_json::to_string_pretty(&Value::Object(now)).unwrap(),
        )
        .unwrap();
        return;
    }
    let text = std::fs::read_to_string(pvec_bench::root().join(PIN))
        .unwrap_or_else(|e| panic!("{PIN}: {e}"));
    let pinned: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(Value::Object(now), pinned, "the output moved against {PIN}");
}

/// Bytes an entity that the report stage may add to the heap it started from (the rows are already
/// held then), measured from the live heap when the artifact was written to the stage's peak. The code before
/// #1520 measured 593 here, 156 after.
const REPORT_BYTES_PER_ENTITY: u64 = 250;

#[test]
#[cfg_attr(miri, ignore)]
fn the_report_stage_adds_no_whole_report_string_1520() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = drawing("large_industrial");
    let dir = tempfile::tempdir().unwrap();
    let v = pvec_stages::measure(&bytes, &dir.path().join("out.pvec"), SourceUnit::Meters);
    let entities = v["entities"].as_u64().unwrap();
    let added = v["stages"]["report"]["peak"].as_u64().unwrap()
        - v["stages"]["write"]["held"].as_u64().unwrap();
    println!(
        "1520 added {added} entities {entities} per_entity {}",
        added / entities
    );
    assert!(
        added / entities <= REPORT_BYTES_PER_ENTITY,
        "the report stage adds {} bytes an entity, budget {REPORT_BYTES_PER_ENTITY}",
        added / entities
    );
}

/// Cancels on the first chunk the writer reports.
struct CancelOnChunk(CancelToken);

impl EngineObserver for CancelOnChunk {
    fn on_event(&self, event: EngineEvent) {
        if event.name() == "conversion_chunk_written" {
            self.0.cancel();
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_cancel_mid_write_leaves_the_previous_pair_and_no_partial_1519() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = drawing("simple");
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("cell.pvec");
    let convert = |token: CancelToken| {
        PvecConversion::new(CadSource::Bytes(&bytes), &dest)
            .unit(SourceUnit::Meters)
            .cancel(token.clone())
            .observer(Arc::new(CancelOnChunk(token)))
            .convert()
    };
    let listing = || {
        let mut v: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        v.sort();
        v
    };
    // The first run is not cancelled in time to matter: a token nobody fires.
    let first = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .expect("converts");
    drop(first);
    let (artifact, report, names) = (
        std::fs::read(&dest).unwrap(),
        std::fs::read(report_path(&dest)).unwrap(),
        listing(),
    );
    let err = convert(CancelToken::new()).expect_err("the cancel stops the write");
    assert!(matches!(err, PvecConvertError::Cancelled), "{err:?}");
    assert_eq!(std::fs::read(&dest).unwrap(), artifact);
    assert_eq!(std::fs::read(report_path(&dest)).unwrap(), report);
    assert_eq!(listing(), names, "no partial file is left");
}
