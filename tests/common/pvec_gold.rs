//! Loader for the pinned slice of `libviprs/pvec-gold-fixtures` that the PVec conformance suites
//! read (N2.1, #1050). The suites `#[path]`-include this file: `pvec_known_truth` (N2.1),
//! `pvec_dual_oracle` (N2.2), and the later structural, geometric and view suites.
//!
//! The corpus is vendored under `tests/fixtures/pvec-gold/` by `tools/pvec-gold-vendor.py`, which
//! also writes `PIN.json` (corpus revision, gold commit, upstream manifest hash) and `index.json`
//! (the trimmed manifest). [`Corpus::open`] refuses a tree that does not match its own index: a
//! file whose sha256 moved, a file the index does not list, or a listed file that is missing. So a
//! fixture cannot drift from the revision the pin names without a test going red.
//!
//! Nothing here writes anything: files are read into memory, so a run that fails halfway leaves no
//! temporary state behind, and nothing needs a converter, a container or a network.

#![allow(dead_code)] // each suite uses its own part

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

/// How strong the corpus says a fixture's expectation is (pvec-gold-fixtures ADR 0009).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceClass {
    /// Independently authored expectation. The only class that establishes source truth.
    KnownTruthExact,
    /// Two converters agree. Never source truth.
    DualOracleEquivalent,
    /// A real drawing with no expectation. Regression only.
    RealWorldDifferential,
    /// Runs and reports, never gates.
    Quarantined,
}

impl EvidenceClass {
    /// The name the corpus manifest uses.
    pub fn tag(self) -> &'static str {
        match self {
            Self::KnownTruthExact => "KNOWN_TRUTH_EXACT",
            Self::DualOracleEquivalent => "DUAL_ORACLE_EQUIVALENT",
            Self::RealWorldDifferential => "REAL_WORLD_DIFFERENTIAL",
            Self::Quarantined => "QUARANTINED",
        }
    }

    fn parse(tag: &str) -> Self {
        [
            Self::KnownTruthExact,
            Self::DualOracleEquivalent,
            Self::RealWorldDifferential,
            Self::Quarantined,
        ]
        .into_iter()
        .find(|c| c.tag() == tag)
        .unwrap_or_else(|| panic!("the index names an unknown evidence class {tag:?}"))
    }
}

/// What a fixture's class was established by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceScope {
    /// Both open converters' STEP output was read.
    StepOracle,
    /// Only the readers' records were compared.
    ReaderMappingOnly,
    /// Only a census of what the readers saw.
    ReaderCensusOnly,
    /// The expectation is a refusal.
    RejectionExpected,
}

impl EvidenceScope {
    fn parse(tag: &str) -> Self {
        match tag {
            "STEP_ORACLE" => Self::StepOracle,
            "READER_MAPPING_ONLY" => Self::ReaderMappingOnly,
            "READER_CENSUS_ONLY" => Self::ReaderCensusOnly,
            "REJECTION_EXPECTED" => Self::RejectionExpected,
            other => panic!("the index names an unknown evidence scope {other:?}"),
        }
    }
}

/// A fixture used on a path that needs another evidence class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassMismatch {
    pub fixture: String,
    pub found: EvidenceClass,
    pub needed: EvidenceClass,
}

impl fmt::Display for ClassMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "evidence class mismatch: fixture {} is {} and this path only takes {}; using it as \
             source truth is a test defect, not a pass",
            self.fixture,
            self.found.tag(),
            self.needed.tag()
        )
    }
}

/// One corpus entry.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub id: String,
    pub group: String,
    pub class: EvidenceClass,
    pub scope: EvidenceScope,
    pub expects_rejection: bool,
    /// What the STEP evaluator concluded, which can differ from `class` after a quarantine.
    pub step_verdict: Option<String>,
    files: BTreeMap<String, String>,
    dir: PathBuf,
}

impl Fixture {
    /// The guard of the known-truth path: `Ok` only for `KNOWN_TRUTH_EXACT`.
    pub fn require_known_truth(&self) -> Result<(), ClassMismatch> {
        if self.class == EvidenceClass::KnownTruthExact {
            Ok(())
        } else {
            Err(ClassMismatch {
                fixture: self.id.clone(),
                found: self.class,
                needed: EvidenceClass::KnownTruthExact,
            })
        }
    }

    /// Whether the vendored slice holds `name` for this fixture.
    pub fn has(&self, name: &str) -> bool {
        self.files.contains_key(name)
    }

    /// The bytes of a vendored file.
    pub fn bytes(&self, name: &str) -> Vec<u8> {
        fs::read(self.dir.join(name))
            .unwrap_or_else(|e| panic!("{}: cannot read {name}: {e}", self.id))
    }

    /// The source drawing.
    pub fn dwg(&self) -> Vec<u8> {
        self.bytes("source.dwg")
    }

    /// A vendored JSON file.
    pub fn json(&self, name: &str) -> Value {
        serde_json::from_slice(&self.bytes(name))
            .unwrap_or_else(|e| panic!("{}: {name} is not JSON: {e}", self.id))
    }

    /// The independently authored expectation (`expected.math.json`). Only a known-truth fixture
    /// has one, so a fixture of another class is refused here too.
    pub fn truth(&self) -> Result<Value, ClassMismatch> {
        self.require_known_truth()?;
        Ok(self.json("expected.math.json"))
    }
}

/// The pinned corpus.
#[derive(Debug)]
pub struct Corpus {
    pub revision: String,
    pub gold_commit: String,
    pub upstream_manifest_sha256: String,
    fixtures: Vec<Fixture>,
}

fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-gold")
}

impl Corpus {
    /// Opens the vendored slice and checks it against its own index.
    pub fn open() -> Corpus {
        Self::open_at(&root())
    }

    /// As [`open`](Self::open), from another root (the tests point it at a tampered copy).
    pub fn open_at(root: &Path) -> Corpus {
        let read = |name: &str| -> Value {
            let bytes = fs::read(root.join(name))
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", root.join(name).display()));
            serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{name} is not JSON: {e}"))
        };
        let pin = read("PIN.json");
        let index = read("index.json");
        let text = |v: &Value, k: &str| -> String {
            v[k].as_str()
                .unwrap_or_else(|| panic!("{k} is missing"))
                .to_string()
        };
        let revision = text(&pin, "corpus_revision");
        assert_eq!(
            text(&index, "corpus_revision"),
            revision,
            "PIN.json and index.json name different corpus revisions"
        );
        let mut fixtures = Vec::new();
        let mut listed = std::collections::BTreeSet::new();
        for f in index["fixtures"].as_array().expect("fixtures") {
            let id = text(f, "id");
            let dir = root.join("fixtures").join(&id);
            let mut files = BTreeMap::new();
            for (name, meta) in f["files"].as_object().expect("files") {
                let want = text(meta, "sha256");
                let path = dir.join(name);
                let data = fs::read(&path).unwrap_or_else(|e| {
                    panic!("{id}: the index lists {name} but {}: {e}", path.display())
                });
                let got = sha256_hex(&data);
                assert_eq!(
                    got, want,
                    "{id}/{name} no longer matches the pinned corpus revision {revision}"
                );
                listed.insert(path);
                files.insert(name.clone(), want);
            }
            fixtures.push(Fixture {
                id,
                group: text(f, "group"),
                class: EvidenceClass::parse(&text(f, "evidence_class")),
                scope: EvidenceScope::parse(&text(f, "evidence_scope")),
                expects_rejection: f["expects_rejection"].as_bool().unwrap_or(false),
                step_verdict: f["step_verdict"].as_str().map(str::to_string),
                files,
                dir,
            });
        }
        let mut on_disk = Vec::new();
        collect(&root.join("fixtures"), &mut on_disk);
        for p in on_disk {
            assert!(
                listed.contains(&p),
                "{} is in the vendored tree but not in index.json",
                p.display()
            );
        }
        Corpus {
            revision,
            gold_commit: text(&pin, "gold_commit"),
            upstream_manifest_sha256: text(&pin, "upstream_manifest_sha256"),
            fixtures,
        }
    }

    /// Every fixture, in manifest order.
    pub fn all(&self) -> &[Fixture] {
        &self.fixtures
    }

    /// The fixtures of one class.
    pub fn of_class(&self, class: EvidenceClass) -> Vec<&Fixture> {
        self.fixtures.iter().filter(|f| f.class == class).collect()
    }

    /// One fixture by id.
    pub fn get(&self, id: &str) -> &Fixture {
        self.fixtures
            .iter()
            .find(|f| f.id == id)
            .unwrap_or_else(|| panic!("no fixture {id} in corpus {}", self.revision))
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect(&p, out);
        } else {
            out.push(p);
        }
    }
}

// ---------------------------------------------------------------------------------- truth values

/// A truth number: `{"decimal": .., "f64_hex": ..}`. The hex is the value; the decimal is for
/// people, and the two must agree to the printed digits.
pub fn num(v: &Value) -> f64 {
    let hex = v["f64_hex"]
        .as_str()
        .unwrap_or_else(|| panic!("not a truth number: {v}"));
    f64::from_bits(
        u64::from_str_radix(hex.trim_start_matches("0x"), 16)
            .unwrap_or_else(|e| panic!("bad f64_hex {hex}: {e}")),
    )
}

/// A list of truth numbers.
pub fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("not a list: {v}"))
        .iter()
        .map(num)
        .collect()
}

/// A list of truth points.
pub fn points(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap_or_else(|| panic!("not a list: {v}"))
        .iter()
        .map(nums)
        .collect()
}
