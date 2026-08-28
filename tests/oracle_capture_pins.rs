//! Guards for the oracle captures under `oracle-captures/` (issues #649, #650).
//!
//! These captures are the reference answers the parity work in this crate is
//! argued against, so what they record about their own provenance has to be
//! checked by something rather than remembered by someone. It was not: a
//! `brew upgrade` nobody ran deliberately replaced vips 8.18.4 with 8.18.6
//! mid-session and deleted the old keg, and because every area wrote the
//! version into its own meta key and nothing ever compared those strings, the
//! oracle for twenty-odd issues of work moved silently. The only reason it
//! surfaced at all is that a lane happened to run the same command twice.
//!
//! `oracle-captures/ORACLE_PIN.json` is now the single place the target build
//! is written down. Each area's `capture.py` refuses to run against a binary
//! that disagrees with it, which stops a bad capture being taken, and the
//! tests here fail if a committed capture records a version that file does
//! not declare for it, which stops a bad capture being kept.
//!
//! The rest is about what a pin is allowed to be made of. A pin that cannot
//! survive a re-run of the same binary on the same input is worse than no
//! pin, because it trains everyone to ignore a red diff, and the convolution
//! area had 932 of those out of 13,794. Two mechanisms did it, and the two
//! shape tests below are what stops either coming back.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// Repo root (the directory containing the root `Cargo.toml`).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn captures_dir() -> PathBuf {
    repo_root().join("oracle-captures")
}

/// Captures that Python wrote and only Python can read back.
///
/// `json.dump` emits a bare `NaN` or `Infinity` for a non-finite float, and
/// neither is JSON: RFC 8259 has no such literal, so `serde_json`, `jq`,
/// `JSON.parse` and every other non-Python reader refuse the whole file
/// rather than the one token. `foreign-uhdr` writes eight bare `NaN`s in one
/// sample grid and `foreign-radiance` one bare `Infinity`, and they only got
/// away with it because nothing outside Python had ever read them.
///
/// `foreign-nifti` shows the fix: it records a non-finite sample as the
/// string `"NaN"`, which says the same thing and parses everywhere. Making
/// the other two `capture.py` scripts do that is the area owner's call
/// rather than this test's, so the list is allowed to shrink and nothing
/// else.
///
/// It has now shrunk to nothing. #677 quoted the non-finite floats in both
/// captures, so every committed capture parses strictly and there is no
/// longer an exception to record. This branch predates that merge, which is
/// why the list arrived here already stale.
///
/// The empty list is still doing work: the companion assertion fails when a
/// capture that needs repairing is *not* named here, so a new script writing
/// a bare `NaN` reddens the test rather than quietly joining an exception
/// list. What it no longer does is grant anyone an exemption. #682 is the
/// follow-up that stops the scripts being able to write one in the first
/// place, which is the difference between the data being clean today and it
/// staying clean.
const PYTHON_ONLY_JSON: &[&str] = &[];

/// Quote bare `NaN` / `Infinity` / `-Infinity` so a Python-written capture
/// parses. Returns the repaired text and whether anything needed repairing.
fn quote_non_finite(text: &str) -> (String, bool) {
    let mut out = String::with_capacity(text.len() + 32);
    let mut chars = text.char_indices().peekable();
    let mut in_string = false;
    let mut repaired = false;
    while let Some((i, c)) = chars.next() {
        if in_string {
            out.push(c);
            if c == '\\' {
                if let Some((_, escaped)) = chars.next() {
                    out.push(escaped);
                }
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            continue;
        }
        let token = ["-Infinity", "Infinity", "NaN"]
            .into_iter()
            .find(|t| text[i..].starts_with(t));
        match token {
            Some(token) => {
                out.push('"');
                out.push_str(token);
                out.push('"');
                for _ in 1..token.len() {
                    chars.next();
                }
                repaired = true;
            }
            None => out.push(c),
        }
    }
    (out, repaired)
}

/// Parse a capture, repairing Python's non-JSON floats if it has to.
fn read_json_repaired(path: &Path) -> (Value, bool) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    match serde_json::from_str(&text) {
        Ok(value) => (value, false),
        Err(strict) => {
            let (repaired, changed) = quote_non_finite(&text);
            let value = serde_json::from_str(&repaired).unwrap_or_else(|e| {
                panic!(
                    "{} is not valid JSON ({strict}) and quoting non-finite \
                     floats did not help ({e})",
                    path.display()
                )
            });
            assert!(
                changed,
                "{} failed to parse for some other reason: {strict}",
                path.display()
            );
            (value, true)
        }
    }
}

fn read_json(path: &Path) -> Value {
    read_json_repaired(path).0
}

fn pin_file() -> Value {
    read_json(&captures_dir().join("ORACLE_PIN.json"))
}

/// Every area on disk, as `("convolution", <parsed oracle.json>)`.
///
/// An area is any directory under `oracle-captures/` holding an
/// `oracle.json`, one level deep or two: `convolution/canny` is its own
/// capture with its own oracle and has to be declared like any other.
fn areas_on_disk() -> BTreeMap<String, Value> {
    areas_with_repair_flags()
        .into_iter()
        .map(|(k, (v, _))| (k, v))
        .collect()
}

fn areas_with_repair_flags() -> BTreeMap<String, (Value, bool)> {
    let root = captures_dir();
    let mut out = BTreeMap::new();
    let mut stack = vec![(String::new(), root.clone())];
    while let Some((name, dir)) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display()));
        for entry in entries {
            let entry = entry.expect("cannot stat an oracle-captures entry");
            if !entry.file_type().expect("cannot type an entry").is_dir() {
                continue;
            }
            let child = entry.file_name().to_string_lossy().into_owned();
            let area = if name.is_empty() {
                child
            } else {
                format!("{name}/{child}")
            };
            let path = entry.path();
            let json = path.join("oracle.json");
            if json.is_file() {
                out.insert(area.clone(), read_json_repaired(&json));
            }
            stack.push((area, path));
        }
    }
    assert!(
        out.len() >= 10,
        "expected the oracle-captures areas to be found, got {:?}",
        out.keys().collect::<Vec<_>>()
    );
    out
}

/// Pull every `vips-<major>.<minor>.<micro>` out of one string.
///
/// That spelling, with the dash, is exactly what `vips --version` prints, so
/// it separates a recorded measurement from prose that happens to mention a
/// release number.
fn vips_versions_in(text: &str, out: &mut BTreeSet<String>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while let Some(hit) = text[i..].find("vips-") {
        let start = i + hit;
        let mut j = start + "vips-".len();
        let mut dots = 0;
        let mut digits = 0;
        while j < bytes.len() {
            match bytes[j] {
                b'0'..=b'9' => digits += 1,
                b'.' if digits > 0 => dots += 1,
                _ => break,
            }
            j += 1;
        }
        if dots == 2 && digits >= 3 && bytes[j - 1].is_ascii_digit() {
            out.insert(text[start..j].to_string());
        }
        i = start + "vips-".len();
    }
}

/// Walk a capture, collecting version strings two ways.
///
/// `recorded` is what the capture claims about itself: values under a key
/// naming a vips version, minus `pinned_vips_version`, which is the pin it
/// was checked against rather than a measurement. `anywhere` is every such
/// string in the document, which also catches a version baked into a path or
/// left behind in a note.
fn collect_versions(
    value: &Value,
    key: Option<&str>,
    recorded: &mut BTreeSet<String>,
    anywhere: &mut BTreeSet<String>,
) {
    match value {
        Value::String(s) => {
            vips_versions_in(s, anywhere);
            let is_version_key =
                key.is_some_and(|k| k.contains("vips_version") && k != "pinned_vips_version");
            if is_version_key {
                vips_versions_in(s, recorded);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_versions(item, key, recorded, anywhere);
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                collect_versions(v, Some(k), recorded, anywhere);
            }
        }
        _ => {}
    }
}

/// Call `f` on every object in the document, with the key it hung off.
fn walk_objects(value: &Value, key: Option<&str>, f: &mut impl FnMut(Option<&str>, &Value)) {
    match value {
        Value::Object(map) => {
            f(key, value);
            for (k, v) in map {
                walk_objects(v, Some(k), f);
            }
        }
        Value::Array(items) => {
            for item in items {
                walk_objects(item, key, f);
            }
        }
        _ => {}
    }
}

/// Every capture names the libvips build it was measured against, and that
/// name matches what `ORACLE_PIN.json` declares for it.
///
/// Both directions of the area list are checked too. A new capture area that
/// forgets to declare its oracle fails here, which is the point: the cost of
/// the #650 drift was not that a version changed, it was that nothing was
/// looking at the version at all.
#[test]
fn every_capture_declares_the_oracle_it_was_measured_against() {
    let pin = pin_file();
    let pinned = pin["pinned_vips_version"]
        .as_str()
        .expect("ORACLE_PIN.json must carry a string pinned_vips_version")
        .to_string();
    let declared = pin["areas"]
        .as_object()
        .expect("ORACLE_PIN.json must carry an areas object");
    let on_disk = areas_on_disk();

    let disk_names: BTreeSet<&str> = on_disk.keys().map(String::as_str).collect();
    let pin_names: BTreeSet<&str> = declared.keys().map(String::as_str).collect();
    assert_eq!(
        disk_names, pin_names,
        "oracle-captures/ORACLE_PIN.json must declare exactly the areas that \
         carry an oracle.json; add the new area to the pin file rather than \
         leaving its oracle version unchecked"
    );

    for (area, capture) in &on_disk {
        let entry = &declared[area.as_str()];
        let want = entry["vips_version"]
            .as_str()
            .unwrap_or_else(|| panic!("area {area} needs a string vips_version in the pin file"));
        let state = entry["state"]
            .as_str()
            .unwrap_or_else(|| panic!("area {area} needs a string state in the pin file"));

        let mut recorded = BTreeSet::new();
        let mut anywhere = BTreeSet::new();
        collect_versions(capture, None, &mut recorded, &mut anywhere);

        assert_eq!(
            recorded,
            BTreeSet::from([want.to_string()]),
            "{area}/oracle.json records {recorded:?} as the vips it was \
             captured on, but ORACLE_PIN.json declares {want}. Re-capture the \
             area or move its entry in the pin file; do not leave the two \
             disagreeing."
        );

        let allowed = BTreeSet::from([want.to_string(), pinned.clone()]);
        assert!(
            anywhere.is_subset(&allowed),
            "{area}/oracle.json mentions {anywhere:?} somewhere in the \
             document but is declared at {want}. A version string surviving \
             in a path or a note after a re-capture is how a stale answer \
             gets read as a current one."
        );

        match state {
            "on_pin" => assert_eq!(
                want, pinned,
                "area {area} is marked on_pin but records {want}, not the \
                 pinned {pinned}"
            ),
            "pre_pin" => assert_ne!(
                want, pinned,
                "area {area} is marked pre_pin but already records the pinned \
                 {pinned}; mark it on_pin"
            ),
            other => panic!("area {area} has unknown state {other:?}; use on_pin or pre_pin"),
        }
    }
}

/// The convolution capture hashes pixels, never the `.v` container.
///
/// `capture.py` used to sha256 the whole output file, and a `.v` is a 64-byte
/// header, the pixels and an XML trailer whose namespace carries
/// `VIPS_MICRO_VERSION` (`libvips/iofuncs/vips.c:857-859`). So a vips patch
/// release moved all 452 hashes at once with no pixel moving: every one of
/// them is reproduced exactly by flipping that single byte back. A detector
/// that is blind to what it guards and loud about what it does not is worse
/// than none, so the pin is now `raw_sha256`, over `vips rawsave` output.
#[test]
fn convolution_pins_pixels_not_the_v_container() {
    let capture = read_json(&captures_dir().join("convolution").join("oracle.json"));
    let mut with_raw = 0usize;
    let mut container_hashes = Vec::new();
    walk_objects(&capture, None, &mut |_key, value| {
        let map = value.as_object().expect("walk_objects yields objects");
        if map.contains_key("raw_sha256") {
            with_raw += 1;
        }
        if map.contains_key("sha256") {
            container_hashes.push(
                map.get("path")
                    .and_then(Value::as_str)
                    .unwrap_or("<no path>")
                    .to_string(),
            );
        }
    });
    let shown: Vec<&String> = container_hashes.iter().take(3).collect();
    assert!(
        container_hashes.is_empty(),
        "convolution/oracle.json still hashes whole files, {} of them, e.g. \
         {shown:?}; hash `vips rawsave` output as raw_sha256 instead (#649)",
        container_hashes.len()
    );
    assert!(
        with_raw >= 400,
        "expected every convolution output block to carry raw_sha256, found {with_raw}"
    );
}

/// The convolution capture's min/max positions survive a re-run.
///
/// `vips min --x --y` reports where a worker thread found the extreme, and
/// 614 of the 904 extremes in that capture sit on a tie, so the coordinate is
/// a race: two runs of the identical binary on the identical fixtures used to
/// differ at 478 leaves, every one an `x` or a `y`. The value still comes
/// from the binary; the position is recomputed as the first occurrence in
/// raster order, and `ties` says how many positions hold it, so a reader can
/// tell an unambiguous pin from an arbitrary-but-reproducible one.
#[test]
fn convolution_min_max_positions_are_deterministic() {
    let capture = read_json(&captures_dir().join("convolution").join("oracle.json"));
    let mut blocks = 0usize;
    let mut unique = 0usize;
    walk_objects(&capture, None, &mut |key, value| {
        if key != Some("min") && key != Some("max") {
            return;
        }
        let map = value.as_object().expect("walk_objects yields objects");
        // `known_divergent` and the meta prose also hang things off keys
        // called min/max; only a real extreme block carries a value.
        if !map.contains_key("value") {
            return;
        }
        blocks += 1;
        for field in ["value", "x", "y", "band", "ties"] {
            assert!(
                map.contains_key(field),
                "an extreme block is missing {field}: {value}"
            );
        }
        let ties = map["ties"].as_u64().expect("ties must be a count");
        assert!(ties >= 1, "an extreme must be held by at least one pixel");
        if ties == 1 {
            unique += 1;
        }
    });
    assert!(
        blocks >= 900,
        "expected the convolution capture's min/max blocks, found {blocks}"
    );
    assert!(
        unique > 0 && unique < blocks,
        "expected both unique and tied extremes in the convolution capture, \
         got {unique} unique of {blocks}"
    );
}

/// Every capture parses as JSON, and the list of ones that need Python to do
/// it does not grow.
///
/// `json.dump` writes a bare `NaN` for a non-finite float and that is not
/// JSON, so `serde_json` and every other non-Python reader refuse the entire
/// file rather than the one token. Three captures do it today. The point of
/// pinning the list is that the next capture area cannot join them quietly:
/// these files are meant to be machine-readable evidence, and evidence only
/// one language can open is not much of a corpus.
#[test]
fn captures_parse_as_json_and_the_python_only_list_does_not_grow() {
    let areas = areas_with_repair_flags();
    let mut needed_repair: Vec<&str> = areas
        .iter()
        .filter(|(_, (_, repaired))| *repaired)
        .map(|(area, _)| area.as_str())
        .collect();
    needed_repair.sort_unstable();
    let mut known: Vec<&str> = PYTHON_ONLY_JSON.to_vec();
    known.sort_unstable();
    let grew: Vec<&&str> = needed_repair
        .iter()
        .filter(|a| !known.contains(a))
        .collect();
    assert!(
        grew.is_empty(),
        "{grew:?} emit bare NaN or Infinity, which is not JSON. Record a \
         non-finite sample as the string \"NaN\" instead, which parses \
         everywhere and says the same thing."
    );
    let fixed: Vec<&&str> = known
        .iter()
        .filter(|a| !needed_repair.contains(a))
        .collect();
    assert!(
        fixed.is_empty(),
        "{fixed:?} now parse as strict JSON; drop them from PYTHON_ONLY_JSON \
         so the list keeps meaning what it says"
    );
}

/// Every path git tracks under `oracle-captures/`, relative to the repo root.
///
/// This has to ask git rather than walk the directory, because the question is
/// what is IN THE INDEX and the filesystem cannot answer it either way round: a
/// `git rm --cached` leaves the file on disk, and a fresh clone does not have
/// it whether or not anyone ran that command. A walk would be green in CI for a
/// reason unrelated to the fix.
///
/// The listing is the part that can come back empty and take the guard down
/// with it, so it is anchored on files that are certainly tracked before any
/// caller reads a conclusion into an absence. `tests/workspace_layout.rs`
/// shells out to `cargo metadata` the same way.
fn tracked_under_oracle_captures() -> Vec<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo_root())
        .args(["ls-files", "-z", "--full-name", "--", "oracle-captures"])
        .output()
        .expect("failed to spawn git ls-files");
    assert!(
        out.status.success(),
        "git ls-files failed, so this guard has nothing to check:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let files: Vec<String> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).replace('\\', "/"))
        .collect();

    for anchor in [
        "oracle-captures/convolution/capture.py",
        "oracle-captures/foreign-analyze/capture.py",
        "oracle-captures/foreign-mat/capture.py",
    ] {
        assert!(
            files.iter().any(|f| f == anchor),
            "git tracks {} paths under oracle-captures and {anchor} is not one \
             of them, so this is not the listing the guard means to read",
            files.len()
        );
    }
    files
}

/// No compiled Python is tracked under `oracle-captures/` (issue #681).
///
/// `oracle-captures/foreign-analyze/__pycache__/capture.cpython-314.pyc` and
/// the matching `foreign-mat` one were in the index: build artefacts of the
/// capture scripts sitting next to them, tied to one CPython, read by nothing.
///
/// The reason this is a test and not just a `git rm --cached` is the ordering
/// that made the issue worth writing down. `oracle-captures/.gitignore` ignores
/// `__pycache__/`, and an ignore rule does nothing to a path that is already
/// tracked. So the ignore on its own leaves both files exactly where they were
/// AND stops `git status` mentioning them, which is worse than either half
/// alone. Nothing about the ignore rule can notice that; this can.
///
/// It asks git, which means it spawns a process, so it cannot run under Miri.
/// Miri supports process spawning on no target and under no flag, so the first
/// one it reaches ends the whole session with an unsupported-operation abort on
/// `fork` rather than failing this one test (issue #714).
///
/// The spawn is one call down, in [`tracked_under_oracle_captures`], not here,
/// so the body scan that classifies the filesystem rows reads this test as
/// pure. The call-following process detector in
/// `tests/miri_ignore_convention.rs` is what sees it.
#[test]
#[cfg_attr(miri, ignore)] // spawns a process, which Miri supports on no target (#714)
fn no_compiled_python_is_tracked_under_oracle_captures() {
    let tracked = tracked_under_oracle_captures();
    let artefacts: Vec<&String> = tracked
        .iter()
        .filter(|p| {
            p.ends_with(".pyc")
                || p.ends_with(".pyo")
                || p.split('/').any(|component| component == "__pycache__")
        })
        .collect();
    assert!(
        artefacts.is_empty(),
        "these are tracked under oracle-captures and are compiled Python, not \
         evidence: {artefacts:?}. Untrack them with `git rm --cached <path>`, \
         which keeps the working copy. oracle-captures/.gitignore only stops \
         the NEXT one being added, it cannot untrack these."
    );
}

// ---------------------------------------------------------------------------
// Recorded fixture hashes describe the files in the tree (issue #779).
// ---------------------------------------------------------------------------

/// Keys a capture hangs "the file this record is about" off, when the hash key
/// is the bare `sha256`.
///
/// Three spellings because three areas settled on three words for the same
/// thing before anything read across them: `convolution/canny` writes `path`,
/// `foreign-exr` writes `file`, and everything else writes `fixture`.
const FILE_KEYS: &[&str] = &["path", "file", "fixture"];

/// Recorded hashes that name a path this guard cannot reach, with the reason.
///
/// One entry, and it earns it: `foreign-uhdr` anchors its whole area on
/// libvips' own `ultra-hdr.jpg` from the reference test suite, which is not in
/// this repository and is named by absolute path. The hash is still worth
/// recording, because it says which byte-for-byte input the rest of that area
/// was measured from, but nothing here can check it.
///
/// The list is exact rather than a floor, so a record quietly acquiring an
/// absolute path (or a `..`) fails here instead of dropping out of the checked
/// set. That is the failure mode this whole test exists for: a hash nobody
/// compares to anything.
const HASHES_NAMING_A_PATH_THIS_TREE_DOES_NOT_HOLD: &[&str] =
    &["foreign-uhdr .records.reference_image sha256"];

/// How many recorded hashes name a committed file, across every area.
///
/// Pinned rather than floored because the steady state of this test is a
/// zero: with every hash agreeing there is nothing left for it to report, and
/// a selector that quietly stopped selecting would look exactly the same. The
/// number is what says the guard still has 95 things to be right about.
///
/// Moving it is fine and expected: a new capture area, or a new record with a
/// fixture behind it, moves it up. Re-run the suite and put the number it
/// prints here.
const EXPECTED_PINNED_FIXTURE_HASHES: usize = 95;

/// One recorded hash that names a file, lifted out of one capture.
#[derive(Debug)]
struct NamedFileHash {
    /// The capture area, as `areas_on_disk` keys it.
    area: String,
    /// Dotted path to the record inside that area's `oracle.json`.
    at: String,
    /// The key the hash was recorded under.
    hash_key: String,
    /// The key the file name was recorded under.
    path_key: String,
    /// The file name, as written, relative to the area directory.
    named: String,
    /// The recorded digest.
    recorded: String,
    /// The recorded byte count, when the record carries one.
    bytes: Option<u64>,
}

impl NamedFileHash {
    /// How the test names this row in a message and in the exemption list.
    fn label(&self) -> String {
        format!("{} {} {}", self.area, self.at, self.hash_key)
    }
}

/// Is this a 64-character lowercase hex digest?
fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Pull every recorded hash that names a file out of one capture.
///
/// The selector is deliberately narrow, because most of the 954 hashes in
/// these captures are not file hashes at all: `raw_sha256` is over a
/// `vips rawsave` dump, `vips_payload_sha256` and `payload_sha256` are over a
/// decoded payload, `frame_zero_pixel_sha256` is over pixels. Hashing the
/// file those records sit next to and calling it a mismatch would be a guard
/// that is wrong 26 times out of 121.
///
/// So a hash is taken to name a file only when the record says which one:
///
///   * the bare `sha256`, paired with a `path`, `file` or `fixture` sibling,
///   * `<name>_sha256`, paired with a sibling literally called `<name>`,
///     which is how `foreign-avif` records `fixture_444_sha256` against its
///     `fixture_444`.
///
/// Everything else is left alone. That is a real blind spot and it is the
/// right one: a hash with no file name next to it is not a claim about a file
/// in the tree, so there is nothing here to check it against.
fn collect_named_file_hashes(area: &str, at: &str, value: &Value, out: &mut Vec<NamedFileHash>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let Some(recorded) = child.as_str() else {
                    continue;
                };
                if !key.ends_with("sha256") || !is_sha256_hex(recorded) {
                    continue;
                }
                let path_key = match key.strip_suffix("_sha256") {
                    Some(prefix) => map
                        .get(prefix)
                        .and_then(Value::as_str)
                        .map(|_| prefix.to_string()),
                    None if key == "sha256" => FILE_KEYS
                        .iter()
                        .find(|k| map.get(**k).and_then(Value::as_str).is_some())
                        .map(|k| (*k).to_string()),
                    None => None,
                };
                let Some(path_key) = path_key else {
                    continue;
                };
                out.push(NamedFileHash {
                    area: area.to_string(),
                    at: at.to_string(),
                    hash_key: key.clone(),
                    named: map[&path_key]
                        .as_str()
                        .expect("path_key was chosen because it is a string")
                        .to_string(),
                    path_key,
                    recorded: recorded.to_string(),
                    bytes: map.get("bytes").and_then(Value::as_u64),
                });
            }
            for (key, child) in map {
                collect_named_file_hashes(area, &format!("{at}.{key}"), child, out);
            }
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                collect_named_file_hashes(area, &format!("{at}[{i}]"), item, out);
            }
        }
        _ => {}
    }
}

/// Every recorded file hash in every committed capture.
fn named_file_hashes() -> Vec<NamedFileHash> {
    let mut out = Vec::new();
    for (area, capture) in areas_on_disk() {
        collect_named_file_hashes(&area, "", &capture, &mut out);
    }
    out
}

/// The sha256 and size of a file, or `None` if it is not there.
///
/// The read lives here rather than in the test body on purpose: it is the
/// same shape as [`areas_on_disk`], and it keeps the filesystem call out of
/// the body scan in `tests/miri_ignore_convention.rs`, which classifies a
/// test by what its own body mentions.
fn digest_of(path: &Path) -> Option<(String, u64)> {
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    sha2::Digest::update(&mut hasher, &bytes);
    let digest = sha2::Digest::finalize(hasher);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    Some((hex, bytes.len() as u64))
}

/// Every recorded hash that names a committed fixture matches that fixture
/// (issue #779).
///
/// `oracle-captures/foreign-avif`'s bit-depth carrier recorded
/// `d5a55b1a…` / 323 bytes for `fixtures/rgb8.avif` and the file in the tree
/// was `c1f34aad…` / 355 bytes, because two records wrote different images to
/// the same name and the later write won. Nothing noticed for as long as the
/// capture existed. The version guard above checks that a capture says which
/// vips it was measured on; this one checks that it was measured on the bytes
/// anybody else can read.
///
/// Three populations come out of the walk and they are treated differently:
///
///   * a name under an `outputs/` directory is scratch the area's own
///     `.gitignore` excludes, so it is checked when it happens to be on disk
///     and skipped when it is not,
///   * a name that leaves the captures (absolute, or with a `..`) is
///     unreachable and has to be listed in
///     [`HASHES_NAMING_A_PATH_THIS_TREE_DOES_NOT_HOLD`],
///   * everything else is a committed fixture and has to be present and have
///     to match, with [`EXPECTED_PINNED_FIXTURE_HASHES`] pinning how many of
///     those there are.
#[test]
fn every_recorded_fixture_hash_matches_the_committed_file() {
    let root = captures_dir();
    let mut pinned = 0usize;
    let mut scratch_checked = 0usize;
    let mut unreachable = Vec::new();
    let mut absent = Vec::new();
    let mut wrong = Vec::new();

    for record in named_file_hashes() {
        let named = record.named.replace('\\', "/");
        let leaves_the_captures =
            named.starts_with('/') || named.split('/').any(|part| part == "..");
        if leaves_the_captures {
            unreachable.push(record.label());
            continue;
        }
        let is_scratch = named.split('/').any(|part| part == "outputs");
        let path = root.join(&record.area).join(&named);
        if !is_scratch {
            pinned += 1;
        }
        match digest_of(&path) {
            None if is_scratch => {}
            None => absent.push(format!(
                "{} names {} under `{}`, and no such file is committed",
                record.label(),
                record.named,
                record.path_key
            )),
            Some((digest, size)) => {
                if is_scratch {
                    scratch_checked += 1;
                }
                let size_wrong = record.bytes.is_some_and(|b| b != size);
                if digest != record.recorded || size_wrong {
                    wrong.push(format!(
                        "{} records {}/{} bytes for {}, which is really {}/{} bytes",
                        record.label(),
                        record.recorded,
                        record
                            .bytes
                            .map_or_else(|| "?".to_string(), |b| b.to_string()),
                        record.named,
                        digest,
                        size
                    ));
                }
            }
        }
    }

    assert!(
        absent.is_empty(),
        "a capture records a hash for a file that is not in the tree, so \
         nothing it says about that file can be checked:\n  {}\n\
         Either commit the file or move the record onto a name under \
         `outputs/`, which every area's .gitignore already treats as scratch.",
        absent.join("\n  ")
    );
    assert!(
        wrong.is_empty(),
        "{} recorded hash(es) do not describe the committed file they name:\n  \
         {}\nRe-run that area's capture.py against the pinned binary so the \
         record and the artefact come out of the same run; do not hand-edit \
         oracle.json to agree.",
        wrong.len(),
        wrong.join("\n  ")
    );

    let mut unreachable_sorted = unreachable;
    unreachable_sorted.sort();
    let mut allowed: Vec<String> = HASHES_NAMING_A_PATH_THIS_TREE_DOES_NOT_HOLD
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    allowed.sort();
    assert_eq!(
        unreachable_sorted, allowed,
        "the set of recorded hashes naming a path outside oracle-captures has \
         moved. A hash nothing can be compared against is exactly the shape \
         #779 was about, so each one is listed with its reason rather than \
         silently skipped."
    );

    assert_eq!(
        pinned, EXPECTED_PINNED_FIXTURE_HASHES,
        "this guard checked {pinned} recorded hashes against committed files, \
         not {EXPECTED_PINNED_FIXTURE_HASHES}. Adding a record or an area is \
         fine, move the number; a DROP with no records deleted means the \
         selector stopped selecting, and since every hash agreeing looks \
         identical to nothing being read, this count is the only thing saying \
         the test still has work to do. ({scratch_checked} further hashes \
         named a file under outputs/ that happened to be on disk and were \
         checked too.)"
    );
}
