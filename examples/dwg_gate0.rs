//! Gate 0 for the acadrust provider (N4.1): open one drawing through
//! `AcadrustProvider` and print one TSV row about what it exposed.
//!
//! ```text
//! dwg_gate0 <tier> <source> <path>
//! ```
//!
//! `docs/spikes/acadrust-gate0/run.sh` runs it once per file, in its own process, so a
//! crash is one `OPEN_FAILED` row and not the end of the run. The columns are the
//! ones `tests/spike_acadrust_gate0.rs` reads.

use std::fmt::Write as _;
use std::path::Path;

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{DwgProvider, EntityScope, InstancePath, LayoutKind, RawGeometry};
use sha2::{Digest, Sha256};

struct Totals {
    model: usize,
    all: usize,
    unmapped: usize,
    transcript: String,
}

fn traverse(p: &AcadrustProvider) -> Result<Totals, String> {
    let mut t = Totals {
        model: 0,
        all: 0,
        unmapped: 0,
        transcript: String::new(),
    };
    let layouts = p.layouts().map_err(|e| e.to_string())?;
    let mut scopes: Vec<(EntityScope, bool)> = layouts
        .iter()
        .map(|l| (EntityScope::Layout(l.id), l.kind == LayoutKind::Model))
        .collect();
    for name in p.block_names() {
        if !layouts.iter().any(|l| l.block_name == name) {
            scopes.push((
                EntityScope::BlockDefinition {
                    name,
                    reached_via: InstancePath::root(),
                },
                false,
            ));
        }
    }
    for (scope, is_model) in scopes {
        let items = p
            .entities(&scope)
            .and_then(|mut c| c.collect_all())
            .map_err(|e| e.to_string())?;
        t.all += items.len();
        if is_model {
            t.model += items.len();
        }
        for e in &items {
            if matches!(e.geometry, RawGeometry::Unmapped { .. }) {
                t.unmapped += 1;
            }
            let _ = writeln!(t.transcript, "{scope:?} {e:?}");
        }
    }
    Ok(t)
}

fn clean(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c == '\t' || c == '\n' || c == '\r' {
                ' '
            } else {
                c
            }
        })
        .take(300)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [tier, source, path] = &args[..] else {
        eprintln!("usage: dwg_gate0 <tier> <source> <path>");
        std::process::exit(2);
    };
    let path = Path::new(path);
    let name = path
        .file_name()
        .map_or(String::new(), |n| n.to_string_lossy().into_owned());
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("{}: {e}", path.display());
        std::process::exit(2);
    });
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let version = String::from_utf8_lossy(&bytes[..bytes.len().min(6)]).into_owned();

    let fail = |e: String| {
        println!(
            "{name}\t{tier}\t{source}\t{sha}\t{version}\tOPEN_FAILED\t-\t-\t-\t-\t-\t-\t-\t{}",
            clean(&e)
        );
    };
    let first = match AcadrustProvider::open(CadSource::Path(path)) {
        Ok(p) => p,
        Err(e) => return fail(e.to_string()),
    };
    let layers = match first.layers() {
        Ok(l) => l.len(),
        Err(e) => return fail(e.to_string()),
    };
    let layouts = first.layouts().map(|l| l.len()).unwrap_or(0);
    let blocks = first.block_names().len();
    let a = match traverse(&first) {
        Ok(t) => t,
        Err(e) => return fail(e),
    };
    // A second pass on the same provider, and a pass on a fresh open, must agree.
    let b = traverse(&first);
    let fresh = AcadrustProvider::open(CadSource::Path(path)).map(|p| traverse(&p));
    let same = matches!(&b, Ok(b) if b.transcript == a.transcript)
        && matches!(&fresh, Ok(Ok(c)) if c.transcript == a.transcript);
    println!(
        "{name}\t{tier}\t{source}\t{sha}\t{version}\tOPEN_OK\t{layers}\t{layouts}\t{blocks}\t{}\t{}\t{}\t{}\t-",
        a.model,
        a.all,
        a.unmapped,
        if same { "yes" } else { "no" }
    );
}
