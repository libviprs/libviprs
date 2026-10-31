//! The fence around what libviprs may name from PVec (N6.1, #1088).
//!
//! libviprs maps sources into PVec's public semantic API and owns no curve maths and no
//! container format. The coupling that would break that is not a copied algorithm, it is a
//! convenient reach into PVec's physical layout (a block header here, an index node there).
//! This file makes that reach a test failure.
//!
//! * Every `pvec*::` path under `src/` has a crate root on [`ALLOWED_ROOTS`], and no segment
//!   of it names a container, block, index or byte-layout module ([`PHYSICAL_SEGMENTS`]).
//! * The scanner is proved able to fail: it is fed an offending line first, and a physical
//!   import is flagged.
//! * `src/pvec/` implements none of the responsibilities NG-10 of
//!   `docs/adr/pvec-semantic-boundary.md` (via `pvec-normalized-geometry.md`) forbids: curve
//!   evaluation, knot insertion, tessellation.
//!
//! Not here yet: the writer funnel `src/pvec/writer.rs` and its allowlist of PVec writer
//! calls. They wait on the `pvec` facade crate (libviprs/pvec#52, P7.1), which does not
//! exist yet. When it lands, `pvec` goes on [`ALLOWED_ROOTS`] and the funnel's own paths get
//! a second, narrower list.
//!
//! Nothing here needs a feature, a network or a PVec checkout.

use std::fs;
use std::path::{Path, PathBuf};

/// Crate roots libviprs may name. `pvec-geometry` and `pvec-numeric` are PVec's semantic
/// crates (the curve model and the numeric policy under it).
const ALLOWED_ROOTS: &[&str] = &[
    "pvec_entity",
    "pvec_facade",
    "pvec_geometry",
    "pvec_numeric",
];

/// Path segments that belong to PVec's physical layout. None may appear in a PVec path.
const PHYSICAL_SEGMENTS: &[&str] = &[
    "container",
    "stream",
    "block",
    "blocks",
    "index",
    "directory",
    "header",
    "layout",
    "layouts",
    "codec",
    "hot",
    "advanced",
    "aligned",
    "arena",
    "spec",
    "manifest",
];

/// Crate roots that are physical by name.
const PHYSICAL_ROOTS: &[&str] = &[
    "pvec_container",
    "pvec_stream",
    "pvec_codec",
    "pvec_corpus",
    "pvec_sources",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display())) {
        let p = e.unwrap().path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Drops `//`, `///` and `//!` comments. Block comments are not used under `src/pvec/`, and
/// a doc line that names a path is not a dependency on it.
fn strip_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Every path whose first segment is a PVec crate root (`pvec_` and a name), as its
/// segments. `pvec::x`, `crate::pvec::x`, `self::pvec` and `super::pvec` are libviprs's own
/// module and are skipped, because the root is not the first segment there. Brace groups
/// (`use pvec_geometry::{a::b, c}`) are flattened to one path per leaf.
fn pvec_paths(src: &str) -> Vec<Vec<String>> {
    let text = strip_comments(src);
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let at_start = i == 0 || !(is_ident(chars[i - 1]) || chars[i - 1] == ':');
        if !at_start || !(chars[i] == 'p') {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && is_ident(chars[j]) {
            j += 1;
        }
        let word: String = chars[i..j].iter().collect();
        // A bare `pvec::` is libviprs's own `pvec` module (`pub use pvec::X` in lib.rs), so
        // it is not a dependency path. The facade crate (libviprs/pvec#52) will have to be
        // imported under a different name for that reason, and that name goes on
        // `ALLOWED_ROOTS` in the change that adds it.
        let is_root = word.starts_with("pvec_");
        let followed = chars[j..].starts_with(&[':', ':']);
        if !(is_root && followed) {
            i = j.max(i + 1);
            continue;
        }
        // Take the rest of the path, up to a delimiter that ends it.
        let mut k = j;
        let mut depth = 0i32;
        while k < chars.len() {
            match chars[k] {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth < 0 {
                        break;
                    }
                }
                ';' | '(' | ')' | '<' | '>' | '=' | '&' | '"' | '[' | ']' => break,
                ',' if depth == 0 => break,
                _ => {}
            }
            k += 1;
        }
        let rest: String = chars[j + 2..k].iter().collect();
        for leaf in flatten_group(&rest) {
            let mut seg = vec![word.clone()];
            seg.extend(leaf);
            out.push(seg);
        }
        i = k.max(i + 1);
    }
    out
}

/// Path leaves, one segment list each.
type Leaves = Vec<Vec<String>>;

/// `{a::b, c, d::{e, f}}` -> [[a,b],[c],[d,e],[d,f]]; a plain `a::b` -> [[a,b]].
fn flatten_group(s: &str) -> Vec<Vec<String>> {
    let s: String = s.split_whitespace().collect();
    let mut leaves: Leaves = vec![vec![]];
    let mut stack: Vec<(Leaves, Leaves)> = Vec::new();
    let mut cur = String::new();
    let push_seg = |leaves: &mut Vec<Vec<String>>, cur: &mut String| {
        if !cur.is_empty() {
            for l in leaves.iter_mut() {
                l.push(cur.clone());
            }
            cur.clear();
        }
    };
    let mut done: Vec<Vec<String>> = Vec::new();
    for c in s.chars() {
        match c {
            ':' => push_seg(&mut leaves, &mut cur),
            '{' => {
                push_seg(&mut leaves, &mut cur);
                // The prefix and what the enclosing group has finished sit on the stack
                // while `leaves` and `done` collect this group.
                stack.push((
                    std::mem::replace(&mut leaves, vec![vec![]]),
                    std::mem::take(&mut done),
                ));
            }
            ',' => {
                push_seg(&mut leaves, &mut cur);
                done.extend(std::mem::replace(&mut leaves, vec![vec![]]));
            }
            '}' => {
                push_seg(&mut leaves, &mut cur);
                done.extend(std::mem::take(&mut leaves));
                let (prefix, outer_done) = stack.pop().unwrap_or_default();
                let prefix = if prefix.is_empty() {
                    vec![vec![]]
                } else {
                    prefix
                };
                let mut merged = Vec::new();
                for p in &prefix {
                    for d in &done {
                        let mut m = p.clone();
                        m.extend(d.clone());
                        merged.push(m);
                    }
                }
                leaves = merged;
                done = outer_done;
            }
            c => cur.push(c),
        }
    }
    push_seg(&mut leaves, &mut cur);
    leaves.extend(done);
    leaves.retain(|l| !l.is_empty());
    leaves
}

/// The reason a path is outside the allowed surface, or `None` when it is inside.
fn violation(path: &[String]) -> Option<String> {
    let root = path[0].as_str();
    if PHYSICAL_ROOTS.contains(&root) {
        return Some(format!("`{root}` is a physical-layout crate"));
    }
    if !ALLOWED_ROOTS.contains(&root) {
        return Some(format!("`{root}` is not on the allowed root list"));
    }
    for seg in &path[1..] {
        if PHYSICAL_SEGMENTS.contains(&seg.to_ascii_lowercase().as_str()) {
            return Some(format!("segment `{seg}` names PVec's physical layout"));
        }
    }
    None
}

fn scan(src: &str) -> Vec<String> {
    pvec_paths(src)
        .iter()
        .filter_map(|p| violation(p).map(|why| format!("{}: {why}", p.join("::"))))
        .collect()
}

fn src_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    rust_files(&root().join("src"), &mut files);
    assert!(!files.is_empty(), "found no source files under src/");
    files
}

#[test]
fn the_scanner_flags_a_physical_import() {
    // The mutation check: adding a physical-layout import has to turn this red.
    for bad in [
        "use pvec_container::blocks::hot::HotBlock;",
        "use pvec_container::Header;",
        "use pvec_stream::Directory;",
        "use pvec_geometry::index::Node;",
        "use pvec_geometry::{Curve, layout::BlockHeader};",
        "let h = pvec_geometry::header::read(b);",
    ] {
        assert!(!scan(bad).is_empty(), "scanner let `{bad}` through");
    }
}

#[test]
fn the_scanner_accepts_the_semantic_surface() {
    for good in [
        "use pvec_geometry::{Curve, Point2};",
        "use pvec_geometry::{Bezier, Curve, Point2, Polyline};",
        "use pvec_numeric::{MAX_CONTROL_POINTS, MAX_DEGREE, NumericError, Policy};",
        "let c = pvec_geometry::Curve::line(a, b);",
        // libviprs's own module is not PVec.
        "use crate::pvec::provider::DwgProvider;",
        "use super::pvec::limits::Limit;",
        "// use pvec_container::Header; is only a comment",
    ] {
        assert_eq!(scan(good), Vec::<String>::new(), "scanner flagged `{good}`");
    }
}

#[test]
fn the_scanner_sees_through_brace_groups() {
    let paths = pvec_paths("use pvec_geometry::{Curve, nested::{A, B}, Point2};");
    let joined: Vec<String> = paths.iter().map(|p| p.join("::")).collect();
    for want in [
        "pvec_geometry::Curve",
        "pvec_geometry::nested::A",
        "pvec_geometry::nested::B",
        "pvec_geometry::Point2",
    ] {
        assert!(
            joined.iter().any(|j| j == want),
            "missing {want} in {joined:?}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn no_pvec_path_under_src_is_outside_the_allowed_surface() {
    let mut bad = Vec::new();
    let mut seen = 0usize;
    for f in src_files() {
        let src = fs::read_to_string(&f).unwrap();
        seen += pvec_paths(&src).len();
        for v in scan(&src) {
            bad.push(format!(
                "{}: {v}",
                f.strip_prefix(root()).unwrap().display()
            ));
        }
    }
    assert!(
        seen > 0,
        "the scan found no PVec path at all, so it is not looking"
    );
    assert!(
        bad.is_empty(),
        "PVec paths outside the allowed semantic surface:\n{}",
        bad.join("\n")
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn only_the_semantic_crates_are_named_today() {
    let mut roots = std::collections::BTreeSet::new();
    for f in src_files() {
        for p in pvec_paths(&fs::read_to_string(&f).unwrap()) {
            roots.insert(p[0].clone());
        }
    }
    let roots: Vec<_> = roots.into_iter().collect();
    assert_eq!(
        roots,
        [
            "pvec_entity",
            "pvec_facade",
            "pvec_geometry",
            "pvec_numeric"
        ]
    );
}

/// The PVec facade (`pvec_facade`, the renamed `pvec` crate; issue #1328) is the writer funnel
/// for the raster stream, and one file is allowed to name it: `src/raster_pvec/container.rs`.
/// Everything else in libviprs goes through that file, so the set of PVec writer calls
/// libviprs makes is the set in that file.
#[test]
#[cfg_attr(miri, ignore)]
fn the_facade_is_named_by_one_file_only() {
    let mut named_in = std::collections::BTreeSet::new();
    for f in src_files() {
        let src = fs::read_to_string(&f).unwrap();
        if pvec_paths(&src).iter().any(|p| p[0] == "pvec_facade") {
            named_in.insert(f.strip_prefix(root()).unwrap().to_path_buf());
        }
    }
    let want: std::collections::BTreeSet<PathBuf> =
        [PathBuf::from("src/raster_pvec/container.rs")].into();
    assert_eq!(named_in, want);
}

/// NG-10's prohibited constructs, as the names a function implementing them would carry.
/// A function named like one is a second implementation of something PVec owns.
const FORBIDDEN_FN_NAMES: &[&str] = &[
    // NG-B1 curve evaluator
    "evaluate",
    "eval_curve",
    "point_at",
    "tangent_at",
    "derivative",
    "curve_length",
    "arc_length",
    "de_boor",
    "basis_functions",
    // NG-B2 tessellator
    "tessellate",
    "tessellation",
    "curve_to_polyline",
    "to_polyline",
    "flatten_curve",
    "flatten_spline",
    "subdivide",
    // NG-B3 knot insertion and re-parametrisation
    "insert_knot",
    "remove_knot",
    "refine_knots",
    "elevate_degree",
    "reparametrize",
    "reparametrise",
    // NG-B4 promotion
    "arc_to_nurbs",
    "arc_to_spline",
    "circle_to_nurbs",
];

fn fn_names(src: &str) -> Vec<String> {
    let text = strip_comments(src);
    let mut out = Vec::new();
    let mut rest = text.as_str();
    while let Some(i) = rest.find("fn ") {
        let before_ok = i == 0 || !is_ident(rest[..i].chars().last().unwrap());
        let after = &rest[i + 3..];
        if before_ok {
            let name: String = after.chars().take_while(|c| is_ident(*c)).collect();
            if !name.is_empty() {
                out.push(name);
            }
        }
        rest = after;
    }
    out
}

#[test]
fn the_forbidden_responsibility_check_can_fail() {
    for bad in [
        "pub fn tessellate(c: &Curve) -> Vec<Point2> { todo!() }",
        "fn insert_knot(k: &mut Vec<f64>, u: f64) {}",
        "pub(crate) fn point_at(c: &Curve, t: f64) {}",
    ] {
        let hit = fn_names(bad)
            .iter()
            .any(|n| FORBIDDEN_FN_NAMES.contains(&n.as_str()));
        assert!(hit, "`{bad}` was not flagged");
    }
    let ok = fn_names("fn map_line(a: P, b: P) -> Curve { Curve::line(a, b) }");
    assert!(!ok.iter().any(|n| FORBIDDEN_FN_NAMES.contains(&n.as_str())));
}

#[test]
#[cfg_attr(miri, ignore)]
fn src_pvec_implements_no_curve_evaluation_knot_insertion_or_tessellation() {
    let mut bad = Vec::new();
    let mut files = Vec::new();
    rust_files(&root().join("src/pvec"), &mut files);
    assert!(!files.is_empty());
    for f in files {
        for name in fn_names(&fs::read_to_string(&f).unwrap()) {
            if FORBIDDEN_FN_NAMES.contains(&name.as_str()) {
                bad.push(format!(
                    "{}: fn {name}",
                    f.strip_prefix(root()).unwrap().display()
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "NG-10 forbids these responsibilities in src/pvec/ (call PVec instead):\n{}",
        bad.join("\n")
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_pvec_dependencies_stay_semantic_and_optional() {
    let manifest = fs::read_to_string(root().join("Cargo.toml")).unwrap();
    for physical in ["pvec-container", "pvec-stream", "pvec-codec"] {
        assert!(
            !manifest
                .lines()
                .any(|l| l.trim_start().starts_with(&format!("{physical} "))
                    || l.trim_start().starts_with(&format!("{physical}="))),
            "Cargo.toml depends on `{physical}`, a physical-layout crate"
        );
    }
}

/// P5.1 to P5.3 build the directory, in PVec. libviprs hands over boxes and builds none of it
/// (issue #1091).
#[test]
#[cfg_attr(miri, ignore)]
fn no_index_directory_or_straddler_type_is_defined_under_src_pvec() {
    let mut bad = Vec::new();
    for f in src_files() {
        if !f.starts_with(root().join("src/pvec")) {
            continue;
        }
        let src = strip_comments(&fs::read_to_string(&f).unwrap());
        for line in src.lines() {
            let t = line.trim_start();
            let t = t
                .strip_prefix("pub(crate) ")
                .or_else(|| t.strip_prefix("pub "))
                .unwrap_or(t);
            for kw in ["struct ", "enum ", "type ", "trait "] {
                if let Some(rest) = t.strip_prefix(kw) {
                    let name: String = rest.chars().take_while(|c| is_ident(*c)).collect();
                    let l = name.to_lowercase();
                    if ["index", "directory", "straddl", "quadtree", "rtree"]
                        .iter()
                        .any(|w| l.contains(w))
                    {
                        bad.push(format!("{}: {kw}{name}", f.display()));
                    }
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "spatial-structure types under src/pvec:\n{}",
        bad.join("\n")
    );
}
