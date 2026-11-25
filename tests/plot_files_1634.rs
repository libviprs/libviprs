//! PC3 and CTB round trips (issue #1634): the two real files import and
//! export to themselves, and an edit that is put back changes nothing.

use libviprs::cad::plot_files::{MAX_PC3_DEPTH, Pc3, Pc3Error, Pc3File, Pc3Media};

const PC3: &[u8] = include_bytes!("fixtures/cad/langara/Langara_2026_Master_Plan.pc3");

fn a4() -> Pc3Media {
    Pc3Media {
        paper_name: "ISO_A4_(210.00_x_297.00_MM)".into(),
        size_mm: [210.0, 297.0],
        printable_mm: [5.0, 17.0, 205.0, 280.1],
        landscape: true,
        orientation: 2,
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[test]
fn pc3_langara_rewrite_is_byte_identical() {
    let pc3 = Pc3::parse(PC3).unwrap();
    assert!(pc3.to_bytes() == PC3, "the rewrite differs from the file");
    let Pc3File::Json(pc3) = Pc3File::parse(PC3).unwrap() else {
        panic!("a JSON PC3 is read as one");
    };
    assert!(pc3.to_bytes() == PC3);
}

#[test]
fn pc3_unedited_values_keep_their_lexemes() {
    let mut pc3 = Pc3::parse(PC3).unwrap();
    let bounds = ["data", "media", "actual_printable_bounds_urx"];
    assert_eq!(pc3.scalar(&bounds), Some("274.39999389648438"));
    assert_eq!(
        pc3.scalar(&["data", "res_color_mem", "resolution", "addr_resolution_x"]),
        Some("1.0")
    );
    assert_eq!(pc3.scalar(&["version"]), Some("0.01"));
    let mut media = pc3.media().unwrap();
    media.paper_name = "ISO_A4_(210.00_x_297.00_MM)".into();
    pc3.set_media(&media).unwrap();
    // only the name moved: every other lexeme is the file's own
    assert_eq!(pc3.scalar(&bounds), Some("274.39999389648438"));
    let (before, after) = (text(PC3), text(&pc3.to_bytes()));
    let changed: Vec<_> = before
        .lines()
        .zip(after.lines())
        .filter(|(a, b)| a != b)
        .collect();
    assert_eq!(before.lines().count(), after.lines().count());
    assert_eq!(changed.len(), 1, "{changed:?}");
    assert!(changed[0].1.contains("ISO_A4"));
}

#[test]
fn pc3_edit_paper_then_revert_is_byte_identical() {
    let mut pc3 = Pc3::parse(PC3).unwrap();
    let original = pc3.media().unwrap();
    assert_eq!(original.paper_name, "ANSI_B_(11.00_x_17.00_Inches)");
    assert_eq!(
        original.size_mm,
        [f64::from(279.4_f32), f64::from(431.8_f32)]
    );
    pc3.set_media(&a4()).unwrap();
    assert!(pc3.to_bytes() != PC3);
    assert_eq!(pc3.media().unwrap(), a4());
    pc3.set_media(&original).unwrap();
    assert!(pc3.to_bytes() == PC3, "reverted file differs");
}

#[test]
fn pc3_new_values_format_like_jsoncpp() {
    let mut pc3 = Pc3::parse(PC3).unwrap();
    pc3.set_media(&a4()).unwrap();
    assert_eq!(
        pc3.scalar(&[
            "data",
            "media",
            "size",
            "media_description",
            "media_bounds",
            "urx"
        ]),
        Some("210.0")
    );
    assert_eq!(
        pc3.scalar(&["data", "media", "actual_printable_bounds_llx"]),
        Some("5.0")
    );
    // 280.1 does not fit in a double: 17 significant digits say so
    assert_eq!(
        pc3.scalar(&["data", "media", "actual_printable_bounds_ury"]),
        Some("280.10000000000002")
    );
    assert_eq!(pc3.scalar(&["data", "media", "dm_orientation"]), Some("2"));
    assert_eq!(
        pc3.scalar(&["data", "media", "size", "landscape_mode"]),
        Some("true")
    );
    // layout: sorted keys, one space per level, ` : `, an object on the next line
    let doc = b"PIAFILEVERSION_3.0,json\r\n{\"b\":[1,2],\"a\":{\"y\":0.1,\"x\":{}},\"c\":\"s\"}";
    let out = text(&Pc3::parse(doc).unwrap().to_bytes());
    assert_eq!(
        out,
        "PIAFILEVERSION_3.0,json\r\n{\n \"a\" : \n {\n  \"x\" : {},\n  \"y\" : 0.1\n },\n \"b\" : [ 1, 2 ],\n \"c\" : \"s\"\n}"
    );
}

#[test]
fn pc3_pc3ver1_binary_is_stored_verbatim_and_refused_for_editing() {
    for head in [&b"PC3VER1"[..], &b"PIAFILEVERSION_1.0"[..]] {
        let mut bytes = head.to_vec();
        bytes.extend_from_slice(&[0, 1, 2, 0xff, 0x80, b'\r', b'\n', 7]);
        let mut file = Pc3File::parse(&bytes).unwrap();
        assert!(matches!(file, Pc3File::Binary(_)));
        assert!(file.to_bytes() == bytes);
        let Err(Pc3Error::BinaryNotEditable(reason)) = file.editable() else {
            panic!("a binary PC3 is refused for editing");
        };
        assert!(reason.contains("binary"), "{reason}");
    }
}

#[test]
fn pc3_hostile_input_is_refused() {
    let deep = format!(
        "PIAFILEVERSION_3.0,json\r\n{}{}",
        "[".repeat(MAX_PC3_DEPTH + 1),
        "]".repeat(MAX_PC3_DEPTH + 1)
    );
    assert_eq!(Pc3::parse(deep.as_bytes()).unwrap_err(), Pc3Error::TooDeep);
    let mut big = b"PIAFILEVERSION_3.0,json\r\n{\"a\":\"".to_vec();
    big.resize(1024 * 1024 + 1, b'x');
    assert!(matches!(Pc3::parse(&big), Err(Pc3Error::TooLarge(_))));
    assert!(Pc3::parse(b"PIAFILEVERSION_3.0,json\r\n{\"a\":").is_err());
    assert!(Pc3::parse(b"not a pc3").is_err());
}

#[cfg(feature = "dwg-acadrust")]
#[test]
fn ctb_announced_length_over_cap_is_refused_before_inflating() {
    use libviprs::cad::plot_files::PlotStyleTable;
    let mut file = include_bytes!("fixtures/cad/langara/Langara_2024-LW0.ctb").to_vec();
    file[52..56].copy_from_slice(&(5u32 * 1024 * 1024).to_le_bytes());
    file.truncate(70); // the stream is damaged: the refusal has to come from the header
    let error = PlotStyleTable::from_bytes("x.ctb", &file)
        .unwrap_err()
        .to_string();
    // The fork's reader (since the pin of #1705) inflates a damaged stream under a hard output budget
    // and then says it is incomplete, where it used to refuse on the header alone. Either way the
    // table is refused and nothing past the budget is allocated.
    assert!(
        error.contains("announces") || error.contains("incomplete"),
        "{error}"
    );
}

#[test]
fn issue_1634_a_non_finite_paper_value_is_refused_not_a_panic() {
    // a file that spells a printable bound 1e999 reads back as infinity...
    let hostile = text(PC3).replace("274.39999389648438", "1e999");
    let mut pc3 = Pc3::parse(hostile.as_bytes()).unwrap();
    let mut media = pc3.media().unwrap();
    assert!(media.printable_mm[2].is_infinite());
    // ...and an edit of another bound recomputes the area from it
    media.printable_mm[0] += 1.0;
    assert_eq!(
        pc3.set_media(&media),
        Err(Pc3Error::NotFinite("printable bound"))
    );
    // the same through the caller: NaN, infinity, and finite values whose area overflows
    let mut pc3 = Pc3::parse(PC3).unwrap();
    for bad in [
        Pc3Media {
            size_mm: [f64::NAN, 297.0],
            ..a4()
        },
        Pc3Media {
            printable_mm: [0.0, 0.0, f64::INFINITY, 1.0],
            ..a4()
        },
        Pc3Media {
            printable_mm: [-1e200, -1e200, 1e200, 1e200],
            ..a4()
        },
    ] {
        assert!(
            matches!(pc3.set_media(&bad), Err(Pc3Error::NotFinite(_))),
            "{bad:?}"
        );
    }
    // a refused edit leaves the file as it was
    assert!(pc3.to_bytes() == PC3);
}
