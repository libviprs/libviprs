//! The supported-entity matrix is true (issue #1038).
//!
//! `docs/vector-entities.md` says what each kind of DWG entity becomes on the
//! vector path and what is reported when it is not fully drawn. A table that
//! nobody checks is how a "supported" claim outlives the code, so this test
//! decodes every committed g13 drawing with the real native decoder and holds
//! the table to it, both ways: a row that no longer matches fails, and so does
//! a drawing with no row.
#![cfg(feature = "dwg-acadsharp")]

use std::collections::BTreeSet;
use std::path::Path;

use libviprs::cad::acadsharp::AcadSharpDecoder;
use libviprs::cad::{CadDecoder, CadSource, CollectSink, DecodeReport};

struct Row {
    fixture: String,
    drawn_as: String,
    codes: String,
}

fn rows(doc: &str) -> Vec<Row> {
    doc.lines()
        .filter(|l| l.starts_with("| `g1"))
        .map(|l| {
            let cells: Vec<&str> = l.trim_matches('|').split('|').map(str::trim).collect();
            assert_eq!(cells.len(), 5, "a matrix row has five cells: {l}");
            Row {
                fixture: cells[0].trim_matches('`').to_owned(),
                drawn_as: cells[3].to_owned(),
                codes: cells[4].to_owned(),
            }
        })
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)] // reads files and loads a native library
fn the_entity_matrix_matches_what_the_decoder_does() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let doc = std::fs::read_to_string(root.join("docs/vector-entities.md"))
        .expect("docs/vector-entities.md is the published matrix");
    let rows = rows(&doc);
    assert!(rows.len() >= 40, "the matrix parsed to {} rows", rows.len());

    let dir = root.join("tests/fixtures/cad/g13");
    let on_disk: BTreeSet<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".dwg"))
        .collect();
    let listed: BTreeSet<String> = rows.iter().map(|r| r.fixture.clone()).collect();
    assert_eq!(
        on_disk, listed,
        "every committed drawing has a row, and every row has a drawing"
    );

    let decoder = match AcadSharpDecoder::new() {
        Ok(d) => d,
        Err(e) => {
            assert!(
                std::env::var("LIBVIPRS_REQUIRE_ACADSHARP_NATIVE").as_deref() != Ok("1"),
                "the native lane was required and the decoder did not start: {e:?}"
            );
            eprintln!("skipped: no native acadsharp library linked ({e})");
            return;
        }
    };
    for row in rows {
        let bytes = std::fs::read(dir.join(&row.fixture)).unwrap();
        let Ok(drawing) = decoder.open(CadSource::Bytes(&bytes)) else {
            assert_eq!(
                row.drawn_as, "refused at open",
                "{} was refused at open and the matrix says otherwise",
                row.fixture
            );
            continue;
        };
        let mut sink = CollectSink::new();
        let mut report = DecodeReport::new();
        let _ = drawing.decode(0, &mut sink, &mut report);
        let kinds: BTreeSet<String> = sink
            .into_collected()
            .iter()
            .map(|p| format!("{:?}", p.kind()))
            .collect();
        let drawn = if kinds.is_empty() {
            "nothing".to_owned()
        } else {
            kinds.into_iter().collect::<Vec<_>>().join(", ")
        };
        // 101 is a reader notification and is on nearly every drawing.
        let codes: BTreeSet<u32> = report
            .diagnostics()
            .iter()
            .map(|d| d.code().get())
            .filter(|c| *c >= 100 && *c != 101)
            .collect();
        let codes = codes
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(drawn, row.drawn_as, "{}: what it is drawn as", row.fixture);
        assert_eq!(codes, row.codes, "{}: what is reported", row.fixture);
    }
}

/// Anything the matrix says is not drawn must be reported, never silent.
#[test]
#[cfg_attr(miri, ignore)] // reads a file
fn no_row_is_both_empty_and_unreported() {
    let doc = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/vector-entities.md"),
    )
    .expect("docs/vector-entities.md is the published matrix");
    for row in rows(&doc) {
        if row.drawn_as == "nothing" || row.drawn_as == "refused at open" {
            // A drawing with a view that has nothing in it is the one honest
            // exception, and it carries code 108 to say so.
            assert!(
                !row.codes.is_empty() || row.fixture.contains("ac1009"),
                "{} draws nothing and reports nothing",
                row.fixture
            );
        }
    }
}
