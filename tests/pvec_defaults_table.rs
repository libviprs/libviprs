//! N8.6 (#1443, slice of #1064): every tunable constant in the conversion code is in the defaults
//! table of `docs/pvec-benchmarks.md` with its value, and every constant the table names is in the
//! code with that value.
//!
//! The constants are read from the source text: the top-level `const NAME: usize = VALUE;` lines of
//! the files that hold the conversion's tunables. A row that names a constant that is not there, a
//! value that moved, and a constant with no row, each fail.

use std::path::PathBuf;

const FILES: [&str; 2] = ["src/pvec_convert.rs", "src/pvec/writer.rs"];
const DOC: &str = "docs/pvec-benchmarks.md";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &str) -> String {
    std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// `(file, NAME, value)` for every top-level `const NAME: usize = VALUE;` in `text`.
fn constants(file: &str, text: &str) -> Vec<(String, String, String)> {
    text.lines()
        .filter_map(|l| {
            let rest = l.strip_prefix("const ")?;
            let (name, rest) = rest.split_once(": usize = ")?;
            let value = rest.strip_suffix(';')?;
            Some((file.to_string(), name.to_string(), value.replace('_', "")))
        })
        .collect()
}

/// `(constant, file, value)` from the rows of the table that name a constant.
fn table(doc: &str) -> Vec<(String, String, String)> {
    let start = doc
        .find("<!-- defaults:begin -->")
        .expect("the defaults table markers");
    let end = doc
        .find("<!-- defaults:end -->")
        .expect("the defaults table markers");
    doc[start..end]
        .lines()
        .filter(|l| l.starts_with('|') && !l.contains("---"))
        .filter_map(|l| {
            let cells: Vec<&str> = l.trim_matches('|').split('|').map(str::trim).collect();
            let name = cells.get(1)?.strip_prefix('`')?.strip_suffix('`')?;
            Some((
                name.to_string(),
                cells.get(2)?.trim_matches('`').to_string(),
                cells.get(3)?.replace([',', '_'], ""),
            ))
        })
        .collect()
}

fn problems(code: &[(String, String, String)], rows: &[(String, String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    for (file, name, value) in code {
        match rows.iter().find(|r| &r.0 == name) {
            None => out.push(format!(
                "{file}: the constant {name} has no row in the defaults table"
            )),
            Some(r) if &r.2 != value => out.push(format!(
                "{name}: the table says {} and {file} says {value}",
                r.2
            )),
            Some(r) if &r.1 != file => out.push(format!(
                "{name}: the table names {} and it is in {file}",
                r.1
            )),
            Some(_) => {}
        }
    }
    for (name, file, _) in rows {
        if !code.iter().any(|c| &c.1 == name) {
            out.push(format!(
                "the table names {name} in {file}, which is not a constant there"
            ));
        }
    }
    out
}

fn code() -> Vec<(String, String, String)> {
    FILES.iter().flat_map(|f| constants(f, &read(f))).collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_defaults_table_and_the_code_name_the_same_constants_with_the_same_values() {
    let bad = problems(&code(), &table(&read(DOC)));
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_constant_with_no_row_a_row_with_no_constant_and_a_moved_value_all_fail() {
    let doc = read(DOC);
    let rows = table(&doc);
    let mut code = code();
    // A constant the table does not have.
    code.push((
        "src/pvec_convert.rs".into(),
        "A_NEW_KNOB".into(),
        "7".into(),
    ));
    assert!(
        problems(&code, &rows)
            .iter()
            .any(|p| p.contains("A_NEW_KNOB") && p.contains("no row"))
    );
    // A row the code does not have.
    let mut rows2 = rows.clone();
    rows2.push(("GONE".into(), "src/pvec_convert.rs".into(), "1".into()));
    assert!(
        problems(&self::code(), &rows2)
            .iter()
            .any(|p| p.contains("GONE"))
    );
    // A value that moved in the code.
    let mut moved = self::code();
    moved[0].2 = "999999".into();
    assert!(
        problems(&moved, &rows)
            .iter()
            .any(|p| p.contains("the table says"))
    );
}

#[test]
fn the_constant_reader_finds_the_lines_it_is_meant_to() {
    let got = constants(
        "f.rs",
        "const A: usize = 64;\n    const NOT: usize = 1;\npub const B: usize = 1;\nconst C: usize = 1_024;\nconst D: u8 = 3;\n",
    );
    assert_eq!(
        got,
        vec![
            ("f.rs".to_string(), "A".to_string(), "64".to_string()),
            ("f.rs".to_string(), "C".to_string(), "1024".to_string())
        ]
    );
}
