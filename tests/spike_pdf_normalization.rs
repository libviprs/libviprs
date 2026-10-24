//! Executable acceptance checks for `docs/spikes/pdf-normalization.md` (issue #1045, N1.6).
//!
//! The spike is a document plus a measurement, so these tests read both as text and hold them to
//! the acceptance criteria of the issue, one group per criterion. They are std-only on purpose:
//! the claim they guard is "libviprs reads operators and PVec decides what they mean", and a test
//! that pulled in a PDF or curve crate would be the wrong shape.
//!
//! Measurements come from `docs/spikes/pdf-normalization/harness`, which reads real PDFs with
//! `lopdf`, maps path operators into segments and pushes every cubic and line through PVec's own
//! constructors. `run.sh` writes `results-x64.tsv` (native x86_64) and `results-arm64.tsv`.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let p = root().join(rel);
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

fn doc() -> String {
    read("docs/spikes/pdf-normalization.md")
}

/// (fixture, key) -> value, from a results file. Comment lines start with `#`.
fn results(file: &str) -> BTreeMap<(String, String), String> {
    let text = read(&format!("docs/spikes/pdf-normalization/{file}"));
    let mut out = BTreeMap::new();
    for l in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = l.splitn(3, '\t').collect();
        assert_eq!(f.len(), 3, "{file}: malformed row `{l}`");
        if f[0] == "fixture" {
            continue; // header
        }
        out.insert((f[0].to_string(), f[1].to_string()), f[2].to_string());
    }
    out
}

fn fixtures(res: &BTreeMap<(String, String), String>) -> Vec<String> {
    let mut v: Vec<String> = res
        .keys()
        .map(|(f, _)| f.clone())
        .filter(|f| f != "synthetic")
        .collect();
    v.sort();
    v.dedup();
    v
}

fn num(res: &BTreeMap<(String, String), String>, fx: &str, key: &str) -> u64 {
    res.get(&(fx.to_string(), key.to_string()))
        .unwrap_or_else(|| panic!("results have no `{key}` for `{fx}`"))
        .parse()
        .unwrap_or_else(|_| panic!("`{key}` for `{fx}` is not a number"))
}

fn section(doc: &str, heading: &str) -> String {
    let level = heading.chars().take_while(|c| *c == '#').count();
    let start = doc
        .find(&format!("\n{heading}"))
        .map(|i| i + 1)
        .or_else(|| doc.starts_with(heading).then_some(0))
        .unwrap_or_else(|| panic!("section `{heading}` is missing from the spike"));
    let body_from = doc[start..].find('\n').map_or(doc.len(), |n| start + n + 1);
    let rest = &doc[body_from..];
    let mut end = rest.len();
    let mut at = 0;
    for line in rest.split_inclusive('\n') {
        let hashes = line.chars().take_while(|c| *c == '#').count();
        if hashes > 0 && hashes <= level && line[hashes..].starts_with(' ') {
            end = at;
            break;
        }
        at += line.len();
    }
    rest[..end].to_string()
}

fn table_rows(sec: &str) -> Vec<Vec<String>> {
    sec.lines()
        .filter(|l| l.starts_with('|'))
        .map(|l| {
            l.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect::<Vec<_>>()
        })
        .filter(|r| {
            !r.iter()
                .all(|c| c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' '))
        })
        .collect()
}

fn ticked(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(a) = rest.find('`') {
        let Some(b) = rest[a + 1..].find('`') else {
            break;
        };
        out.push(rest[a + 1..a + 1 + b].to_string());
        rest = &rest[a + b + 2..];
    }
    out
}

// ---------------------------------------------------------------- AC 1

#[test]
fn ac1_states_pdf_rs_does_no_vector_path_extraction_at_the_pinned_revision() {
    let d = doc();
    assert!(d.contains("adcf1ff2"), "the revision adcf1ff2 is not named");
    assert!(d.contains("src/pdf.rs"), "src/pdf.rs is not named");
    let low = d.to_lowercase();
    assert!(
        low.contains("no vector path extraction"),
        "the document must say in so many words that there is no vector path extraction"
    );
    assert!(
        low.contains("new capability"),
        "the document must say this is new capability, not a refactor"
    );
    for entry in [
        "extract_page_image",
        "extract_page_image_dpi",
        "extract_page_image_with_background",
        "extract_page_image_with_password",
        "render_page_pdfium",
        "render_page_pdfium_budgeted",
        "pdf_info",
    ] {
        assert!(d.contains(entry), "entry point `{entry}` is not named");
    }
}

// ---------------------------------------------------------------- AC 2

const OPERATORS: &[&str] = &[
    "m", "l", "c", "v", "y", "re", "h", "cm", "q", "Q", "W", "W*",
];
const VERDICTS: &[&str] = &["exact", "equivalent", "unsupported for v1"];

fn operator_rows(d: &str) -> Vec<Vec<String>> {
    let rows = table_rows(&section(d, "## Operator mapping"));
    assert!(
        rows.len() > 1,
        "the Operator mapping section holds no table"
    );
    assert_eq!(
        rows[0].iter().map(|c| c.to_lowercase()).collect::<Vec<_>>(),
        ["operator", "verdict", "maps to", "evidence"],
        "operator table columns"
    );
    rows[1..].to_vec()
}

#[test]
fn ac2_every_operator_has_a_row_with_a_verdict() {
    let d = doc();
    let rows = operator_rows(&d);
    for op in OPERATORS {
        let r = rows
            .iter()
            .find(|r| ticked(&r[0]).first().map(String::as_str) == Some(op))
            .unwrap_or_else(|| panic!("no row for operator `{op}`"));
        let verdict = ticked(&r[1]).into_iter().next().unwrap_or_default();
        assert!(
            VERDICTS.contains(&verdict.as_str()),
            "operator `{op}` has verdict `{verdict}`"
        );
        assert!(
            !r[2].trim().is_empty(),
            "operator `{op}` says nothing about what it maps to"
        );
    }
}

#[test]
fn ac2_cubics_are_never_flattened() {
    let d = doc();
    let rows = operator_rows(&d);
    let c = rows
        .iter()
        .find(|r| ticked(&r[0]).first().map(String::as_str) == Some("c"))
        .expect("row c");
    assert_eq!(
        ticked(&c[1])[0],
        "exact",
        "`c` is a cubic Bezier and must map exactly"
    );
    for r in &rows {
        assert!(
            !r[2].to_lowercase().contains("flatten")
                || r[2].to_lowercase().contains("never flatten"),
            "row `{}` mentions flattening without saying it never happens",
            r[0]
        );
    }
}

#[test]
fn ac2_every_row_cites_a_passing_synthetic_check() {
    let d = doc();
    let res = results("results-x64.tsv");
    for r in operator_rows(&d) {
        let ids: Vec<String> = r[3]
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        assert!(!ids.is_empty(), "operator `{}` cites no check", r[0]);
        for id in ids {
            let v = res
                .get(&("synthetic".to_string(), id.clone()))
                .unwrap_or_else(|| {
                    panic!(
                        "operator `{}` cites {id}, which is not in results-x64.tsv",
                        r[0]
                    )
                });
            assert!(v.starts_with("pass"), "{id} does not pass: {v}");
        }
    }
}

#[test]
fn ac2_the_mapper_keeps_every_cubic_on_real_pdfs() {
    let res = results("results-x64.tsv");
    let fx = fixtures(&res);
    assert!(fx.len() >= 3, "fewer than three real PDFs were measured");
    let mut total = 0;
    for f in &fx {
        let ops = num(&res, f, "ops.c") + num(&res, f, "ops.v") + num(&res, f, "ops.y");
        assert_eq!(
            ops,
            num(&res, f, "segs.cubic"),
            "{f}: cubic operators and mapped cubic segments differ"
        );
        total += ops;
    }
    assert!(
        total > 0,
        "no real PDF carried a single cubic, so the claim was not exercised"
    );
}

#[test]
fn ac2_measured_table_agrees_with_the_results() {
    let d = doc();
    let res = results("results-x64.tsv");
    let rows = table_rows(&section(&d, "## Measured on real PDFs"));
    assert!(rows.len() > 1, "the Measured section holds no table");
    let header: Vec<String> = rows[0].iter().map(|c| c.to_lowercase()).collect();
    let key_of = |h: &str| match h {
        "paths" => "paths",
        "subpaths" => "subpaths",
        "lines" => "segs.line",
        "cubics" => "segs.cubic",
        "rects" => "segs.rect",
        "accepted by pvec" => "pvec.ok",
        "refused by pvec" => "pvec.refused",
        other => panic!("unexpected column `{other}`"),
    };
    let mut named = Vec::new();
    for r in &rows[1..] {
        let fx = ticked(&r[0])
            .into_iter()
            .next()
            .unwrap_or_else(|| r[0].clone());
        for (i, h) in header.iter().enumerate().skip(1) {
            let want = num(&res, &fx, key_of(h));
            let got: u64 = r[i]
                .replace(',', "")
                .parse()
                .unwrap_or_else(|_| panic!("{fx}: `{}` is not a number", r[i]));
            assert_eq!(got, want, "{fx}: column `{h}` differs from the measurement");
        }
        named.push(fx);
    }
    for f in fixtures(&res) {
        assert!(
            named.contains(&f),
            "{f} was measured but is missing from the table"
        );
    }
}

#[test]
fn ac2_fixtures_are_pinned_by_sha256() {
    let d = doc();
    let res = results("results-x64.tsv");
    for f in fixtures(&res) {
        let h = res
            .get(&(f.clone(), "sha256".to_string()))
            .unwrap_or_else(|| panic!("{f}: no sha256 in results"));
        assert_eq!(h.len(), 64, "{f}: sha256 is not 64 hex digits");
        assert!(
            d.contains(h.as_str()),
            "{f}: the document does not quote its sha256"
        );
    }
}

#[test]
fn ac2_clipping_is_split_into_exact_and_bounded_only() {
    let d = doc();
    let res = results("results-x64.tsv");
    let low = d.to_lowercase();
    assert!(
        low.contains("bounded-only") && low.contains("exact"),
        "clipping must be split exact / bounded-only"
    );
    let mut seen = 0;
    for f in fixtures(&res) {
        let e = num(&res, &f, "clip.exact");
        let b = num(&res, &f, "clip.bounded");
        assert_eq!(
            e + b,
            num(&res, &f, "ops.W") + num(&res, &f, "ops.W*"),
            "{f}: clip split does not add up to W + W*"
        );
        seen += e + b;
    }
    assert!(seen > 0, "no clip was measured on any real PDF");
}

#[test]
fn ac2_cm_is_folded_into_the_n15_classes_through_pvec() {
    let d = doc();
    let res = results("results-x64.tsv");
    assert!(
        d.contains("transform-matrix.md"),
        "the document must point at the N1.5 matrix"
    );
    let classes = [
        "translation",
        "rotation",
        "uniform scale",
        "non-uniform scale",
        "reflection",
        "shear / general affine",
        "singular",
    ];
    for f in fixtures(&res) {
        let sum: u64 = classes
            .iter()
            .map(|c| num(&res, &f, &format!("cm.{c}")))
            .sum::<u64>()
            + num(&res, &f, "cm.identity");
        assert_eq!(
            sum,
            num(&res, &f, "ops.cm"),
            "{f}: cm classes do not add up to the cm count"
        );
    }
}

#[test]
fn ac2_winding_and_compound_paths_are_measured() {
    let res = results("results-x64.tsv");
    let mut compound = 0;
    for f in fixtures(&res) {
        compound += num(&res, &f, "paths.compound");
        let _ = num(&res, &f, "fill.evenodd");
        let _ = num(&res, &f, "fill.nonzero");
        let _ = num(&res, &f, "compound.mixed_winding");
    }
    assert!(
        compound > 0,
        "no compound path was found on any real PDF, so subpath winding was not exercised"
    );
}

// ---------------------------------------------------------------- AC 3

#[test]
fn ac3_states_precisely_whether_a_dwg_pdf_pair_was_measured() {
    let d = doc();
    let sec = section(&d, "## Per-class geometry-count difference");
    let low = sec.to_lowercase();
    assert!(
        sec.contains("pvec-gold-fixtures#1"),
        "the section must name the Epic O tracker"
    );
    // #1240 closes with this PR, so the pair has to exist and the section has to say plainly that
    // the criterion is met, for which kind of pair, and on which files.
    let pair = results("results-pair.tsv");
    assert!(
        sec.trim_start().starts_with(&format!("Met, for an {LABEL}")),
        "the section must open by saying the criterion is met, and for which kind of pair"
    );
    assert!(
        !low.contains("not met"),
        "the section says both met and not met"
    );
    assert!(
        low.contains("searched"),
        "the section must record where I looked for an AutoCAD-class pair"
    );
    for f in ["`pair.dwg`", "`pair.pdf`", "results-pair.tsv"] {
        assert!(sec.contains(f), "the section does not name {f}");
    }
    // The provenance sentence states the entity and layer counts of the DWG; both must be the
    // read-back counts in results-pair.tsv, not the generator's intent.
    let entities: i64 = PAIR_CLASSES
        .iter()
        .map(|c| {
            pair.get(&("pair.dwg".to_string(), format!("dwg_{c}")))
                .unwrap_or_else(|| panic!("results-pair.tsv has no dwg_{c}"))
                .parse::<i64>()
                .unwrap()
        })
        .sum();
    let flat = sec.replace('\n', " ");
    let stated: i64 = flat
        .split(" entities")
        .next()
        .and_then(|s| s.split_whitespace().last())
        .and_then(|n| n.parse().ok())
        .expect("the section does not state the DWG's entity count as `<n> entities`");
    assert_eq!(
        stated, entities,
        "the section says the DWG has {stated} entities, the read-back in results-pair.tsv counts {entities}"
    );
    let layers: Vec<&str> = pair
        .keys()
        .filter(|(f, _)| f == "pair.dwg")
        .filter_map(|(_, k)| k.strip_prefix("dwg_layer_"))
        .collect();
    assert!(!layers.is_empty(), "results-pair.tsv records no DWG layers");
    for l in &layers {
        assert!(
            sec.contains(&format!("`{l}`")),
            "layer {l} from results-pair.tsv is not named in the section"
        );
    }
    let words = ["zero", "one", "two", "three", "four", "five", "six"];
    let n = words
        .get(layers.len())
        .expect("more layers than this check spells out");
    assert!(
        flat.contains(&format!("{n} layers")),
        "the section does not say the DWG has {n} layers, which is what results-pair.tsv records"
    );
}

#[test]
fn ac3_a_dwg_and_its_exported_pdf_were_measured_per_class() {
    let p = root().join("docs/spikes/pdf-normalization/results-pair.tsv");
    assert!(
        p.exists(),
        "results-pair.tsv does not exist: no DWG/PDF pair has been measured"
    );
    let text = fs::read_to_string(p).unwrap();
    assert!(text.lines().any(|l| l.contains("dwg")) && text.lines().any(|l| l.contains("pdf")));
    // The pair is an open-exporter pair and says so, and records who plotted it.
    assert!(
        text.contains("open-exporter pair, not AutoCAD-class"),
        "the pair is not labelled as an open-exporter pair"
    );
    assert!(
        text.contains("librecad "),
        "exporter name and version are not recorded"
    );
    // The DWG side is counted per class, and the PDF side is measured per class.
    for c in [
        "line", "polyline", "arc", "circle", "ellipse", "spline", "text", "hatch", "other",
    ] {
        assert!(
            text.lines()
                .any(|l| l.starts_with(&format!("pair.dwg\tdwg_{c}\t"))),
            "no DWG count for class `{c}`"
        );
    }
    for c in [
        "line", "polyline", "arc", "circle", "ellipse", "spline", "text", "hatch",
    ] {
        assert!(
            text.lines()
                .any(|l| l.starts_with(&format!("class_{c}.pdf\tsegs.line\t"))),
            "no PDF measurement for class `{c}`"
        );
    }
    // The document names the pair and fills a row per class.
    let d = doc();
    let sec = section(&d, "## Per-class geometry-count difference");
    assert!(
        sec.contains("open-exporter pair, not AutoCAD-class"),
        "the spike section does not carry the label"
    );
    let rows = table_rows(&sec);
    for c in [
        "line", "polyline", "arc", "circle", "ellipse", "spline", "text", "hatch", "other",
    ] {
        assert!(
            rows.iter().any(|r| r[0].to_lowercase().starts_with(c)),
            "no per-class row for `{c}`"
        );
    }
}

// ---------------------------------------------------------------- AC 4

const LOSS_ROWS: &[&str] = &[
    "layer identity and state",
    "block structure and insert transforms",
    "entity handles",
    "model/paper distinction",
    "true arc and ellipse parameters",
    "spline degree and knots",
    "drawing units",
];

#[test]
fn ac4_loss_table_has_every_row_and_the_detectable_column_filled() {
    let d = doc();
    let rows = table_rows(&section(&d, "## The loss table"));
    assert!(rows.len() > 1, "the loss table is missing");
    assert_eq!(
        rows[0].iter().map(|c| c.to_lowercase()).collect::<Vec<_>>(),
        [
            "lost in the export",
            "what the pdf keeps",
            "detectable from the pdf alone",
            "evidence"
        ],
        "loss table columns"
    );
    for want in LOSS_ROWS {
        let r = rows[1..]
            .iter()
            .find(|r| r[0].to_lowercase().starts_with(want))
            .unwrap_or_else(|| panic!("no loss row for `{want}`"));
        let first = r[2]
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(|c: char| !c.is_alphabetic())
            .to_lowercase();
        assert!(
            ["yes", "no", "partly"].contains(&first.as_str()),
            "row `{want}`: detectable column must start with yes, no or partly, got `{}`",
            r[2]
        );
        assert!(
            r[2].split_whitespace().count() >= 3,
            "row `{want}`: the detectable verdict gives no reason"
        );
        assert!(
            !r[3].trim().is_empty() && r[3].trim() != "-",
            "row `{want}` cites no evidence"
        );
    }
}

#[test]
fn ac4_loss_evidence_is_in_the_results() {
    let d = doc();
    let res = results("results-x64.tsv");
    let pair = results("results-pair.tsv");
    let rows = table_rows(&section(&d, "## The loss table"));
    for r in &rows[1..] {
        for id in r[3]
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
        {
            if let Some(rest) = id.strip_prefix("P:") {
                // Measured on the open-exporter pair: P:<pair pdf>:<key>.
                let (f, key) = rest
                    .split_once(':')
                    .unwrap_or_else(|| panic!("bad pair evidence id {id}"));
                assert!(
                    pair.contains_key(&(f.to_string(), key.to_string())),
                    "loss row `{}` cites {id}, which results-pair.tsv does not hold",
                    r[0]
                );
            } else if let Some(key) = id.strip_prefix("R:") {
                // Document-level fact measured on every real PDF.
                for f in fixtures(&res) {
                    assert!(
                        res.contains_key(&(f.clone(), key.to_string())),
                        "{f}: results have no `{key}` for loss row `{}`",
                        r[0]
                    );
                }
            } else {
                assert!(
                    res.contains_key(&("synthetic".to_string(), id.to_string())),
                    "loss row `{}` cites unknown {id}",
                    r[0]
                );
            }
        }
    }
}

// ---------------------------------------------------------------- AC 5

#[test]
fn ac5_never_structurally_exact_and_names_the_class_it_may_claim() {
    let d = doc();
    let sec = section(&d, "## Labelling");
    let low = sec.to_lowercase();
    assert!(
        low.contains("never labelled structurally exact") || low.contains("never labelled \"structurally exact\""),
        "the document must say a PDF-derived result is never labelled structurally exact against the originating DWG"
    );
    assert!(
        low.contains("originating dwg"),
        "the statement must be about the originating DWG"
    );
    assert!(
        sec.contains("EPIC P") || sec.contains("Epic P"),
        "the class must be attributed to EPIC P"
    );
    assert!(
        low.contains("display fidelity") || low.contains("geometric exactness"),
        "the document must name the EPIC P conformance class a PDF result may claim"
    );
}

// ---------------------------------------------------------------- scope and harness

#[test]
fn x64_and_arm64_agree_and_say_what_they_ran_on() {
    let x = read("docs/spikes/pdf-normalization/results-x64.tsv");
    let a = read("docs/spikes/pdf-normalization/results-arm64.tsv");
    assert!(
        x.contains("x86_64"),
        "results-x64.tsv does not record x86_64"
    );
    assert!(
        a.contains("aarch64"),
        "results-arm64.tsv does not record aarch64"
    );
    let body = |s: &str| {
        s.lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(
        body(&x),
        body(&a),
        "the two architectures measured different things"
    );
}

#[test]
fn the_harness_defines_no_tolerance_and_no_evaluator_of_its_own() {
    let src = read("docs/spikes/pdf-normalization/harness/src/main.rs");
    for banned in [
        "1e-",
        "EPSILON",
        "epsilon",
        "f64::EPS",
        "fn flatten",
        "fn tessellate",
        "fn eval",
    ] {
        assert!(
            !src.contains(banned),
            "the harness contains `{banned}`: tolerances and evaluators belong to PVec"
        );
    }
    assert!(
        src.contains("pvec_numeric::Policy") && src.contains("pvec_geometry"),
        "the harness must go through PVec"
    );
}

#[test]
fn the_spike_leaves_the_raster_path_and_the_manifest_alone() {
    let cargo = read("Cargo.toml");
    assert!(
        !cargo.contains("pdf-normalization"),
        "the spike must not enter the crate manifest"
    );
    let pdf = read("src/pdf.rs");
    assert!(
        !pdf.contains("content stream path"),
        "src/pdf.rs must not grow vector extraction in a spike"
    );
}

// ------------------------------------------------ review of #1238: guards on the pair

const PAIR_CLASSES: [&str; 9] = [
    "line", "polyline", "arc", "circle", "ellipse", "spline", "text", "hatch", "other",
];

fn first_number(cell: &str) -> i64 {
    let tok = cell.split_whitespace().next().unwrap_or("");
    tok.replace(',', "")
        .parse()
        .unwrap_or_else(|_| panic!("cell `{cell}` does not start with a number"))
}

#[test]
fn pair_table_agrees_with_results_pair_cell_by_cell() {
    let pair = results("results-pair.tsv");
    let get = |f: &str, k: &str| -> i64 {
        pair.get(&(f.to_string(), k.to_string()))
            .unwrap_or_else(|| panic!("results-pair.tsv has no {f} {k}"))
            .parse()
            .unwrap()
    };
    let d = doc();
    let sec = section(&d, "## Per-class geometry-count difference");
    let rows = table_rows(&sec);
    // Every LibreCAD PDF carries one rectangular clip: 1 path and 4 line segments (`W*`).
    for c in PAIR_CLASSES {
        let r = rows
            .iter()
            .find(|r| r[0].to_lowercase().starts_with(c))
            .unwrap_or_else(|| panic!("no per-class row for `{c}`"));
        let want: [i64; 5] = if c == "other" {
            [get("pair.dwg", "dwg_other"), 0, 0, 0, 0]
        } else {
            let f = format!("class_{c}.pdf");
            [
                get("pair.dwg", &format!("dwg_{c}")),
                get(&f, "paths") - 1,
                get(&f, "segs.line") - 4,
                get(&f, "segs.cubic"),
                get(&f, "fill.evenodd") + get(&f, "fill.nonzero"),
            ]
        };
        for (i, w) in want.iter().enumerate() {
            assert_eq!(
                first_number(&r[i + 1]),
                *w,
                "per-class row `{c}`, column {}: doc says `{}`, results-pair.tsv says {w}",
                i + 1,
                r[i + 1]
            );
        }
    }
    // The whole-drawing sentence.
    for (key, want) in [
        ("paths", get("pair.pdf", "paths") - 1),
        ("segs.line", get("pair.pdf", "segs.line") - 4),
        ("segs.cubic", get("pair.pdf", "segs.cubic")),
    ] {
        let s = format!("{}", want).chars().rev().collect::<Vec<_>>();
        let mut with_commas = String::new();
        for (i, ch) in s.iter().enumerate() {
            if i > 0 && i % 3 == 0 {
                with_commas.push(',');
            }
            with_commas.push(*ch);
        }
        let with_commas: String = with_commas.chars().rev().collect();
        assert!(
            sec.contains(&with_commas),
            "the whole-drawing {key} count {with_commas} is not in the section"
        );
    }
}

fn exporter_version(text: &str) -> String {
    let l = text
        .lines()
        .find(|l| l.contains("plotted by librecad "))
        .expect("results-pair.tsv does not record the LibreCAD version");
    l.split("plotted by librecad ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

#[test]
fn librecad_version_agrees_in_tsv_doc_and_driver() {
    let tsv = read("docs/spikes/pdf-normalization/results-pair.tsv");
    let ver = exporter_version(&tsv);
    assert!(
        ver.chars().next().map_or(false, |c| c.is_ascii_digit()),
        "version `{ver}` is not a version"
    );
    let sec = section(&doc(), "## Per-class geometry-count difference");
    assert!(
        sec.contains(&format!("LibreCAD {ver} ")),
        "the doc does not say LibreCAD {ver}, which is what results-pair.tsv records"
    );
    let sh = read("docs/spikes/pdf-normalization/pair.sh");
    assert!(
        sh.contains(&format!("librecad_ver={ver}\n")) && sh.contains("librecad='$librecad_ver'"),
        "pair.sh does not pin the apt package to librecad_ver={ver}"
    );
    assert!(
        sh.contains("acadrust_rev=") && sh.contains("git -C \"$acad\" archive"),
        "pair.sh must pin acadrust by rev with git archive"
    );
    let rev = tsv
        .lines()
        .find_map(|l| l.split("written by acadrust ").nth(1))
        .and_then(|s| s.split_whitespace().next())
        .expect("results-pair.tsv does not record the acadrust rev");
    assert!(
        sh.contains(&format!("acadrust_rev={rev}")),
        "pair.sh pins a different acadrust rev than the results record ({rev})"
    );
}

const LABEL: &str = "open-exporter pair, not AutoCAD-class";

#[test]
fn the_open_exporter_label_is_in_the_intro_the_section_and_the_tsv() {
    let d = doc();
    let intro = &d[..d.find("\n## ").expect("no sections")];
    assert!(intro.contains(LABEL), "the intro does not carry the label");
    assert!(
        section(&d, "## Per-class geometry-count difference").contains(LABEL),
        "the pair section does not carry the label"
    );
    assert!(
        read("docs/spikes/pdf-normalization/results-pair.tsv").contains(LABEL),
        "results-pair.tsv does not carry the label"
    );
}

#[test]
fn structurally_exact_is_only_ever_negated_outside_labelling() {
    let d = doc();
    let lab = section(&d, "## Labelling");
    let rest = d.replacen(&lab, "", 1);
    let flat = rest.replace('\n', " ");
    for sentence in flat.split(". ") {
        let low = sentence.to_lowercase();
        if low.contains("structurally exact") {
            assert!(
                ["never", "not ", "cannot", "no "]
                    .iter()
                    .any(|n| low.contains(n)),
                "`structurally exact` outside Labelling without a negation: {sentence}"
            );
        }
    }
}

#[test]
fn the_dxf_step_is_named_as_part_of_the_measured_loss() {
    let d = doc();
    let d = d.replace('\n', " ");
    let intro = &d[..d.find(" ## What is there today").expect("no first section")];
    let sec = section(&doc(), "## Per-class geometry-count difference").replace('\n', " ");
    for (name, text) in [("intro", intro.to_string()), ("pair section", sec)] {
        assert!(
            text.contains("DXF writer"),
            "the {name} does not say the DXF goes through acadrust's DXF writer"
        );
        assert!(
            text.contains("not separately measured"),
            "the {name} does not say the DXF step is not separately measured"
        );
        assert!(
            !text.contains("measures is LibreCAD's") && !text.contains("below is LibreCAD's"),
            "the {name} still attributes the whole loss to LibreCAD"
        );
    }
}

#[test]
fn page_rotate_is_in_the_entry_point_list() {
    let sec = section(&doc(), "## What is there today");
    assert!(
        sec.contains("`page_rotate`"),
        "page_rotate is an entry point and is not listed"
    );
}

#[test]
fn the_x86_64_harness_pass_over_the_pair_is_recorded_and_arm64_only_is_gone() {
    let sec = section(&doc(), "## Per-class geometry-count difference");
    assert!(
        !sec.contains("Measured on arm64 only"),
        "stale `arm64 only` wording"
    );
    assert!(
        sec.contains("585"),
        "the 585-row x86_64 harness result is not recorded"
    );
    assert!(sec.contains("x86_64"), "the x86_64 pass is not named");
    assert!(
        sec.contains("LibreCAD itself ran on arm64 only"),
        "the section must say LibreCAD itself ran on arm64 only"
    );
    assert!(
        sec.contains("timestamp"),
        "the section must say the PDF embeds a timestamp so its sha256 does not reproduce"
    );
}
