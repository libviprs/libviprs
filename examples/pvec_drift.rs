//! Checks a `.pvec` against the fixture it was generated from.
//!
//! ```text
//! cargo run --features pvec-write --example pvec_drift -- FIXTURE_DIR FILE.pvec [--json] [--accepted FILE] [--dump]
//! ```
//!
//! `FIXTURE_DIR` is one fixture of the pinned corpus (`tests/fixtures/pvec-gold/fixtures/<id>`).
//! The human report is printed by default and the machine one with `--json`; `--dump` prints the
//! canonical dump instead, which is what the committed accepted output is made of. The accepted
//! output is read from `tests/fixtures/pvec-drift/accepted.dump` unless `--accepted` names another.
//!
//! Exit status: 0 no drift (or drift on a fixture that does not gate), 1 drift, 2 the check could
//! not be made. See `docs/pvec-drift-check.md`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use libviprs::pvec::drift;

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args.remove(i))
            .is_some()
    };
    let json = flag("--json");
    let dump = flag("--dump");
    let accepted = args
        .iter()
        .position(|a| a == "--accepted")
        .map(|i| {
            args.remove(i);
            args.remove(i)
        })
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-drift/accepted.dump")
        });
    let [fixture, pvec] = args.as_slice() else {
        eprintln!("usage: pvec_drift FIXTURE_DIR FILE.pvec [--json] [--accepted FILE] [--dump]");
        return ExitCode::from(2);
    };
    let outcome = (|| {
        if dump {
            let bytes = std::fs::read(pvec).map_err(|e| drift::DriftError::Io(e.to_string()))?;
            let id = Path::new(fixture)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("fixture");
            print!("{}", drift::canonical_dump(id, &bytes)?);
            return Ok(None);
        }
        let id = Path::new(fixture)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let file = std::fs::read_to_string(&accepted).ok();
        let section = file.as_deref().and_then(|f| drift::accepted_section(f, id));
        // The report the conversion published beside the artifact says how many entities PVec's
        // writer refused; without one the artifact is read as holding everything.
        let refused = std::fs::read(format!("{pvec}.report.json"))
            .ok()
            .and_then(|b| libviprs::pvec::report::ConversionReport::from_json_slice(&b).ok())
            .map_or(0, |r| {
                r.entities
                    .iter()
                    .filter(|e| e.reason.as_deref() == Some("pvec-cannot-store"))
                    .count()
            });
        drift::check_fixture(Path::new(fixture), Path::new(pvec), section, refused).map(Some)
    })();
    match outcome {
        Ok(None) => ExitCode::SUCCESS,
        Ok(Some(report)) => {
            print!(
                "{}",
                if json {
                    report.to_json()
                } else {
                    report.to_text()
                }
            );
            ExitCode::from(u8::from(report.fails()))
        }
        Err(e) => {
            eprintln!("pvec_drift: {e}");
            ExitCode::from(2)
        }
    }
}
