//! Issue #1635 (C4.2 of iasbuilt/webapp #1434): the manifest carries the drawing's fingerprint GUID
//! and version GUID (`file.fingerprintGuid`, `file.versionGuid`) and, for each layout, the plot
//! configuration the page setup names (`layouts[].plot.configName`, the PC3) next to the plot
//! style table (`styleSheet`). The server uses the fingerprint to recognise a later revision of
//! the same drawing, and the wizard shows the PC3 and CTB the drawing itself names.
//!
//! The small drawings are written with acadrust so each value is known; the big one is the
//! 66-layout drawing named by `IASBUILT_DWG_LAYOUTS_FIXTURE`, which says so when unset. Needs
//! `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};

use acadrust::objects::ObjectType;
use acadrust::{CadDocument, DwgWriter, DxfWriter};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec_convert::{CadManifest, cad_manifest};

const FINGERPRINT: &str = "{3F1E9C52-7A0B-4D1E-9C55-0A6B1D2E3F40}";
const VERSION_1: &str = "{8B2D4A10-5C6E-4F77-A1B2-C3D4E5F60718}";
const VERSION_2: &str = "{C0FFEE00-1234-4ABC-8DEF-0123456789AB}";

/// A drawing with the given GUIDs and one layout per `(name, printer, style table)`.
fn drawing(guids: Option<(&str, &str)>, layouts: &[(&str, &str, &str)]) -> CadDocument {
    let mut doc = CadDocument::new();
    if let Some((fingerprint, version)) = guids {
        doc.header.fingerprint_guid = fingerprint.into();
        doc.header.version_guid = version.into();
    }
    for (name, printer, style) in layouts {
        doc.add_layout(name).expect("the layout is added");
        // An empty layout is not a sheet and carries no plot block, so give each one a line.
        let line = acadrust::entities::Line::from_coords(0.0, 0.0, 0.0, 10.0, 10.0, 0.0);
        doc.add_entity_to_layout(acadrust::EntityType::Line(line), name)
            .expect("the line is added");
        for object in doc.objects.values_mut() {
            if let ObjectType::Layout(l) = object {
                if l.name == *name {
                    l.plot_printer_name = (*printer).into();
                    l.plot_style_sheet = (*style).into();
                }
            }
        }
    }
    doc
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1635-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn list(dir: &Path, file: &str, bytes: &[u8]) -> CadManifest {
    let path = dir.join(file);
    fs::write(&path, bytes).unwrap();
    cad_manifest::<AcadrustProvider>(CadSource::Path(&path)).expect("the drawing lists")
}

fn config_of<'a>(m: &'a CadManifest, layout: &str) -> Option<&'a str> {
    m.layouts
        .iter()
        .find(|l| l.name == layout)
        .and_then(|l| l.plot.as_ref())
        .and_then(|p| p.config_name.as_deref())
}

fn style_of<'a>(m: &'a CadManifest, layout: &str) -> Option<&'a str> {
    m.layouts
        .iter()
        .find(|l| l.name == layout)
        .and_then(|l| l.plot.as_ref())
        .map(|p| p.style_sheet.as_str())
}

// The named test of the issue, on two DXF drawings: one that names everything, one that names
// nothing.
#[test]
#[cfg_attr(miri, ignore)]
fn manifest_carries_fingerprint_version_and_plot_config_name() {
    let dir = scratch("dxf");
    let named = drawing(
        Some((FINGERPRINT, VERSION_1)),
        &[
            (
                "SHEET_001",
                "Langara_2026_Master_Plan.pc3",
                "Langara_2024-LW0.ctb",
            ),
            ("SHEET_002", "DWG To PDF.pc3", "monochrome.ctb"),
            ("SHEET_003", "", "acad.ctb"),
        ],
    );
    let bytes = DxfWriter::new(&named).write_to_vec().unwrap();
    let m = list(&dir, "named.dxf", &bytes);
    assert_eq!(m.file.fingerprint_guid.as_deref(), Some(FINGERPRINT));
    assert_eq!(m.file.version_guid.as_deref(), Some(VERSION_1));
    assert_eq!(
        config_of(&m, "SHEET_001"),
        Some("Langara_2026_Master_Plan.pc3")
    );
    assert_eq!(style_of(&m, "SHEET_001"), Some("Langara_2024-LW0.ctb"));
    assert_eq!(config_of(&m, "SHEET_002"), Some("DWG To PDF.pc3"));
    assert_eq!(style_of(&m, "SHEET_002"), Some("monochrome.ctb"));
    assert_eq!(
        config_of(&m, "SHEET_003"),
        None,
        "a page setup that names no PC3 has no configName, not an empty one"
    );
    assert_eq!(style_of(&m, "SHEET_003"), Some("acad.ctb"));

    let bare = drawing(None, &[("SHEET_001", "", "")]);
    let bytes = DxfWriter::new(&bare).write_to_vec().unwrap();
    let m = list(&dir, "bare.dxf", &bytes);
    assert_eq!(m.file.fingerprint_guid, None, "a header without the GUIDs");
    assert_eq!(m.file.version_guid, None);
    assert_eq!(config_of(&m, "SHEET_001"), None);
    let _ = fs::remove_dir_all(dir);
}

// The same values through the DWG header and the layout's plot settings.
#[test]
#[cfg_attr(miri, ignore)]
fn the_dwg_manifest_carries_the_same_facts() {
    let dir = scratch("dwg");
    let doc = drawing(
        Some((FINGERPRINT, VERSION_1)),
        &[(
            "SHEET_001",
            "Langara_2026_Master_Plan.pc3",
            "Langara_2024-LW0.ctb",
        )],
    );
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let m = list(&dir, "named.dwg", &bytes);
    assert_eq!(m.file.fingerprint_guid.as_deref(), Some(FINGERPRINT));
    assert_eq!(m.file.version_guid.as_deref(), Some(VERSION_1));
    assert_eq!(
        config_of(&m, "SHEET_001"),
        Some("Langara_2026_Master_Plan.pc3")
    );
    assert_eq!(style_of(&m, "SHEET_001"), Some("Langara_2024-LW0.ctb"));
    let _ = fs::remove_dir_all(dir);
}

// A later save of the same drawing is the same fingerprint and a new version, which is the key the
// server matches revisions on (addendum A4).
#[test]
#[cfg_attr(miri, ignore)]
fn a_resaved_drawing_keeps_its_fingerprint_and_changes_its_version() {
    let dir = scratch("resave");
    let layouts = [("SHEET_001", "Langara_2026_Master_Plan.pc3", "")];
    let first = list(
        &dir,
        "rev0.dxf",
        &DxfWriter::new(&drawing(Some((FINGERPRINT, VERSION_1)), &layouts))
            .write_to_vec()
            .unwrap(),
    );
    let second = list(
        &dir,
        "rev1.dxf",
        &DxfWriter::new(&drawing(Some((FINGERPRINT, VERSION_2)), &layouts))
            .write_to_vec()
            .unwrap(),
    );
    assert!(first.file.fingerprint_guid.is_some());
    assert_eq!(first.file.fingerprint_guid, second.file.fingerprint_guid);
    assert_ne!(first.file.version_guid, second.file.version_guid);
    assert_eq!(second.file.version_guid.as_deref(), Some(VERSION_2));
    let _ = fs::remove_dir_all(dir);
}

// The manifest's wire shape: `file.fingerprintGuid`, `file.versionGuid`, `layouts[].plot.configName`
// next to `styleSheet`, and an absent GUID is null.
#[test]
fn the_manifest_serialises_the_new_fields_under_the_spec_names() {
    let mut m = CadManifest::default();
    m.file.fingerprint_guid = Some(FINGERPRINT.into());
    m.layouts.push(Default::default());
    m.layouts[0].plot = Some(libviprs::pvec_convert::ManifestPlot {
        style_sheet: "Langara_2024-LW0.ctb".into(),
        config_name: Some("Langara_2026_Master_Plan.pc3".into()),
        ..Default::default()
    });
    let v = serde_json::to_value(&m).unwrap();
    assert_eq!(v["file"]["fingerprintGuid"], FINGERPRINT);
    assert!(v["file"]["versionGuid"].is_null());
    assert_eq!(
        v["layouts"][0]["plot"]["configName"],
        "Langara_2026_Master_Plan.pc3"
    );
    assert_eq!(
        v["layouts"][0]["plot"]["styleSheet"],
        "Langara_2024-LW0.ctb"
    );
    let back: CadManifest = serde_json::from_value(v).unwrap();
    assert_eq!(back, m);
}

// The 66-layout drawing: every sheet's page setup is read, and the fingerprint is there.
#[test]
#[cfg_attr(miri, ignore)]
fn the_66_layout_drawing_carries_its_fingerprint_and_plot_config_names() {
    let Some(path) = std::env::var_os("IASBUILT_DWG_LAYOUTS_FIXTURE") else {
        eprintln!("skipped: IASBUILT_DWG_LAYOUTS_FIXTURE is not set (the 66-layout drawing)");
        return;
    };
    let m = cad_manifest::<AcadrustProvider>(CadSource::Path(Path::new(&path)))
        .expect("the drawing lists");
    assert_eq!(m.layouts.len(), 66);
    let fingerprint = m.file.fingerprint_guid.as_deref().expect("a fingerprint");
    assert!(
        fingerprint.starts_with('{') && fingerprint.ends_with('}'),
        "{fingerprint}"
    );
    assert!(m.file.version_guid.is_some(), "a version GUID");
    let sheets: Vec<_> = m.layouts.iter().filter(|l| l.sheet).collect();
    assert!(!sheets.is_empty());
    let mut named = 0;
    for l in &sheets {
        let plot = l
            .plot
            .as_ref()
            .unwrap_or_else(|| panic!("{} has a plot block", l.name));
        named += usize::from(plot.config_name.as_deref().is_some_and(|c| !c.is_empty()));
    }
    assert!(named > 0, "at least one sheet names a plot configuration");
}
