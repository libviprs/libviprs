//! N8.8 (#1451, slice of #1070): the integration documentation, held to the code where it states a
//! table that also exists in code, and held together as a set.
//!
//! Four tables are rendered from the code and must appear in their page exactly: the mapping table,
//! the transform promotion matrix, the v1 view profile and the resource limits. `DOCS_BLESS=1`
//! rewrites the marked regions of the pages from the code, so a code change is one command and a
//! reviewed diff. The set is held by two checks: the thirteen pages exist and `README.md` links
//! every one, and no relative link in `README.md` or `docs/` is dead.
//!
//! Needs `--features pvec-normalized`.

#![cfg(feature = "pvec-normalized")]

use std::path::{Path, PathBuf};

use libviprs::pvec::SourceLimits;
use libviprs::pvec::ViewProfile;
use libviprs::pvec::mapping::NORMALIZED_TO_PVEC;
use libviprs::pvec::transform::{Cell, PromotionMatrix, SourceRow};

const PAGES: [&str; 13] = [
    "docs/pvec-architecture.md",
    "docs/pvec-provider-api.md",
    "docs/pvec-iso-mapping.md",
    "docs/pvec-view-semantics.md",
    "docs/pvec-transform-matrix.md",
    "docs/pvec-dwg-support.md",
    "docs/pvec-pdf-loss.md",
    "docs/pvec-components.md",
    "docs/pvec-georeference.md",
    "docs/pvec-conversion-api.md",
    "docs/pvec-limits.md",
    "docs/pvec-provider-replacement.md",
    "docs/pvec-benchmarks.md",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn mapping() -> String {
    let mut s =
        String::from("| kind | PVec type | capability | conformance |\n|---|---|---|---|\n");
    for r in NORMALIZED_TO_PVEC {
        s += &format!(
            "| {:?} | `{}` | {} ({}) | {} |\n",
            r.kind,
            r.pvec_type,
            r.capability.name,
            r.capability.id,
            r.conformance.tag()
        );
    }
    s
}

fn matrix() -> String {
    let mut s = String::from("| source shape | PVec returns |\n|---|---|\n");
    for row in [
        SourceRow::Line,
        SourceRow::Polyline,
        SourceRow::PolylineWithBulge,
        SourceRow::CircularArc,
        SourceRow::Circle,
        SourceRow::Ellipse,
        SourceRow::EllipticalArc,
        SourceRow::BSpline,
        SourceRow::Nurbs,
    ] {
        let cell = match PromotionMatrix::FROZEN.cell(row) {
            Cell::Shapes(shapes) => shapes
                .iter()
                .map(|x| format!("{x:?}"))
                .collect::<Vec<_>>()
                .join(" or "),
            Cell::RequiresPVecCapability => "RequiresPVecCapability".to_string(),
        };
        s += &format!("| {row:?} | {cell} |\n");
    }
    s
}

fn view() -> String {
    let mut s = String::from("| field | value | N1.4 row |\n|---|---|---|\n");
    for k in ViewProfile::default_profile().knobs() {
        s += &format!("| `{}` | {} | {} |\n", k.field, k.value, k.row);
    }
    s
}

fn limits() -> String {
    let l = SourceLimits::default();
    let mut s = String::from("| limit | default |\n|---|---|\n");
    for (name, value) in [
        ("max_block_depth", l.max_block_depth.to_string()),
        ("max_entities", l.max_entities.to_string()),
        ("max_polyline_vertices", l.max_polyline_vertices.to_string()),
        (
            "max_spline_control_points",
            l.max_spline_control_points.to_string(),
        ),
        ("max_spline_knots", l.max_spline_knots.to_string()),
        ("max_insert_grid", l.max_insert_grid.to_string()),
        (
            "max_expanded_instances",
            l.max_expanded_instances.to_string(),
        ),
        ("max_allocation_bytes", l.max_allocation_bytes.to_string()),
        ("max_name_bytes", l.max_name_bytes.to_string()),
    ] {
        s += &format!("| `{name}` | {value} |\n");
    }
    s
}

/// The page, the marker name and what the code renders for it.
fn tables() -> Vec<(&'static str, &'static str, String)> {
    vec![
        ("docs/pvec-iso-mapping.md", "mapping", mapping()),
        ("docs/pvec-transform-matrix.md", "matrix", matrix()),
        ("docs/pvec-view-semantics.md", "view", view()),
        ("docs/pvec-limits.md", "limits", limits()),
    ]
}

fn region(doc: &str, name: &str) -> (usize, usize) {
    let begin = format!("<!-- {name}:begin -->\n");
    let end = format!("<!-- {name}:end -->");
    let b = doc
        .find(&begin)
        .unwrap_or_else(|| panic!("no {begin:?} marker"))
        + begin.len();
    let e = doc
        .find(&end)
        .unwrap_or_else(|| panic!("no {end:?} marker"));
    (b, e)
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_code_backed_tables_equal_the_code() {
    let bless = std::env::var_os("DOCS_BLESS").is_some();
    let mut bad = Vec::new();
    for (page, name, rendered) in tables() {
        let path = root().join(page);
        let doc = read(&path);
        let (b, e) = region(&doc, name);
        if &doc[b..e] != rendered.as_str() {
            if bless {
                std::fs::write(&path, format!("{}{rendered}{}", &doc[..b], &doc[e..])).unwrap();
            } else {
                bad.push(format!(
                    "{page}: the {name} table is not what the code renders:\n{rendered}"
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{}\n(DOCS_BLESS=1 rewrites them)",
        bad.join("\n")
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn changing_one_code_row_without_the_document_turns_it_red() {
    for (page, name, rendered) in tables() {
        let doc = read(&root().join(page));
        let (b, e) = region(&doc, name);
        let first_row = rendered.lines().nth(2).unwrap();
        let moved = doc[b..e].replacen(first_row, "| a changed row | in the code |", 1);
        assert_ne!(
            moved, rendered,
            "{name}: the edit must make the table differ from the code"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_thirteen_pages_exist_and_the_readme_links_each() {
    let readme = read(&root().join("README.md"));
    for p in PAGES {
        assert!(root().join(p).is_file(), "{p} is missing");
        assert!(readme.contains(p), "README.md does not link {p}");
    }
}

/// The relative link targets of a markdown file inside this repository: `](target)` without a
/// scheme, a fragment or a `../` that leaves it (the sibling repositories are not ours to check).
fn links(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("](") {
        rest = &rest[i + 2..];
        let Some(j) = rest.find(')') else { break };
        let target = rest[..j].split('#').next().unwrap_or("").trim();
        if !target.is_empty()
            && !target.contains("://")
            && !target.starts_with("mailto:")
            && !target.starts_with("../")
            && !target.contains(' ')
        {
            out.push(target.to_string());
        }
        rest = &rest[j..];
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn no_relative_link_in_the_readme_or_the_docs_is_dead() {
    let mut files = vec![root().join("README.md")];
    let mut stack = vec![root().join("docs")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "md") {
                files.push(p);
            }
        }
    }
    let mut dead = Vec::new();
    for f in &files {
        let dir = f.parent().unwrap();
        for t in links(&read(f)) {
            if !dir.join(&t).exists() {
                dead.push(format!(
                    "{}: {t}",
                    f.strip_prefix(root()).unwrap().display()
                ));
            }
        }
    }
    assert!(dead.is_empty(), "dead relative links:\n{}", dead.join("\n"));
}

#[test]
fn the_link_extractor_finds_relative_targets_only() {
    let got =
        links("[a](x.md) [b](https://e.org/x) [c](y.md#h) [d](#top) [e](dir/z.md) [f](../sibling)");
    assert_eq!(got, vec!["x.md", "y.md", "dir/z.md"]);
}
