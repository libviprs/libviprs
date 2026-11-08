//! Issue #1564 (slice 5 of #1557): one `.pvec` per paper layout from one call. `PvecLayoutSet` opens the
//! drawing once, walks the paper layouts in tab order and writes `<stem>.<NN>-<slug>.pvec` per sheet, with
//! its report beside it and `<stem>.sheets.json` written last. An empty layout is skipped and listed, a
//! drawing with no paper layout writes nothing, and one sheet that fails does not lose the others.
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};

use libviprs::CancelToken;
use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::publish::{open_published, report_path};
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutState, ProvenanceStub, ProviderCaps,
    ProviderError, RawEntity, RawGeometry, SourceUnit, ViewportEntity,
};
use libviprs::pvec_convert::{LayoutSetError, PvecConvertError, PvecLayoutSet, SheetSkip};

// ---- a mock provider: the bytes it is opened with pick the drawing ----

struct Mock {
    layouts: Vec<LayoutState>,
    caps: ProviderCaps,
}

/// Hands out `items`, then fails when `fail` is set: a sheet that breaks partway through.
struct Cursor {
    items: std::vec::IntoIter<RawEntity>,
    fail: bool,
}

impl EntityCursor for Cursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        match self.items.next() {
            Some(e) => Ok(Some(e)),
            None if self.fail => Err(ProviderError::NoSuchScope("the sheet broke".into())),
            None => Ok(None),
        }
    }
}

fn entity(handle: u64, class: &str, layer: &str, geometry: RawGeometry) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry,
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(handle)),
            class_name: class.into(),
            block_path: InstancePath::root(),
            layer: layer.into(),
        },
    }
}

fn line(handle: u64, layer: &str) -> RawEntity {
    let x = handle as f64;
    entity(
        handle,
        "LINE",
        layer,
        RawGeometry::Line {
            start: [x, 0.0, 0.0],
            end: [x + 10.0, 5.0, 0.0],
        },
    )
}

/// The paper layout names of each scene, in tab order. Model space always comes first.
fn scene(bytes: &[u8]) -> Vec<String> {
    let names: Vec<String> = match bytes {
        b"MOCK-model" => Vec::new(),
        b"MOCK-names" => vec![
            "A-101 Plan".into(),
            "a/101 plan".into(),
            "Fa\u{e7}ade".into(),
            "..".into(),
            "Detail ".to_owned() + &"x".repeat(80),
        ],
        b"MOCK-mixed" => [
            "Cover",
            "Empty",
            "Viewport only",
            "Hidden",
            "Broken",
            "Plan",
        ]
        .map(String::from)
        .to_vec(),
        b"MOCK-blank" => ["Empty", "Viewport only"].map(String::from).to_vec(),
        b"MOCK-many" => (1..=100).map(|n| format!("Sheet {n}")).collect(),
        _ => vec!["Layout1".into()],
    };
    names
}

impl DwgProvider for Mock {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(matches!(source, CadSource::Bytes(b) if b.starts_with(b"MOCK")))
    }

    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let bytes = match source {
            CadSource::Bytes(b) => b.to_vec(),
            CadSource::Path(p) => fs::read(p).map_err(|e| ProviderError::Open {
                provider: "mock",
                detail: e.to_string(),
            })?,
            _ => Vec::new(),
        };
        if !bytes.starts_with(b"MOCK") {
            return Err(ProviderError::Unrecognised {
                provider: "mock",
                detail: "not a mock drawing".into(),
            });
        }
        let mut layouts = vec![LayoutState {
            id: LayoutId(0),
            name: "Model".into(),
            kind: LayoutKind::Model,
            block_name: "*Model_Space".into(),
        }];
        for (i, name) in scene(&bytes).into_iter().enumerate() {
            layouts.push(LayoutState {
                id: LayoutId(i as u32 + 1),
                name,
                kind: LayoutKind::Paper,
                block_name: format!("*Paper_Space{i}"),
            });
        }
        let caps = Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
            c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
        });
        Ok(Mock { layouts, caps })
    }

    fn name(&self) -> &'static str {
        "mock"
    }

    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }

    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(self.layouts.clone())
    }

    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        let layer = |name: &str, off| LayerState {
            name: name.into(),
            handle: None,
            off,
            frozen: false,
            locked: false,
            plottable: true,
        };
        Ok(vec![layer("0", false), layer("OFF", true)])
    }

    /// No viewport of any layout shows model space, so "Viewport only" holds nothing but its own
    /// paper view (the id 1 viewport the provider never lists).
    fn viewports(&self) -> Result<Vec<ViewportEntity>, ProviderError> {
        Ok(Vec::new())
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let EntityScope::Layout(id) = scope else {
            return Err(ProviderError::NoSuchScope("no blocks here".into()));
        };
        let layout = self
            .layouts
            .iter()
            .find(|l| l.id == *id)
            .ok_or_else(|| ProviderError::NoSuchScope("no such layout".into()))?;
        let h = u64::from(id.0) * 100;
        let (items, fail) = match layout.name.as_str() {
            "Empty" => (Vec::new(), false),
            "Viewport only" => (
                vec![entity(
                    h + 1,
                    "VIEWPORT",
                    "0",
                    RawGeometry::Unmapped {
                        reason: "viewport".into(),
                    },
                )],
                false,
            ),
            "Hidden" => (vec![line(h + 1, "OFF"), line(h + 2, "OFF")], false),
            "Broken" => (vec![line(h + 1, "0")], true),
            _ => (vec![line(h + 1, "0"), line(h + 2, "0")], false),
        };
        Ok(Box::new(Cursor {
            items: items.into_iter(),
            fail,
        }))
    }
}

// ---- helpers ----

fn scratch(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("libviprs-1564-{tag}-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Every file name in `dir`, sorted the way a directory listing sorts them.
fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn set<'a>(bytes: &'a [u8], dir: &'a Path) -> PvecLayoutSet<'a> {
    PvecLayoutSet::new(CadSource::Bytes(bytes), dir)
        .stem("plan")
        .sheets(|c| c.unit(SourceUnit::Millimeters))
}

fn index(dir: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(dir.join("plan.sheets.json")).unwrap()).unwrap()
}

// ---- naming ----

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_each_sheet_is_named_by_its_tab_position_and_a_slug_of_its_name() {
    let dir = scratch("names");
    let out = set(b"MOCK-names", &dir).convert_with::<Mock>().unwrap();

    let long = "Detail-".to_owned() + &"x".repeat(57);
    assert_eq!(long.len(), 64, "the slug stops at 64 bytes");
    let expected = [
        "plan.01-A-101-Plan.pvec".to_owned(),
        // A path separator is a byte like any other outside the safe set.
        "plan.02-a-101-plan.pvec".to_owned(),
        // Each byte of a non-ASCII character becomes '-', and the run collapses.
        "plan.03-Fa-ade.pvec".to_owned(),
        // Nothing left after trimming, so the slug is `layout`.
        "plan.04-layout.pvec".to_owned(),
        format!("plan.05-{long}.pvec"),
    ];
    let files: Vec<String> = out
        .sheets
        .iter()
        .map(|s| s.file.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(files, expected);
    let names: Vec<&str> = out.sheets.iter().map(|s| s.layout.as_str()).collect();
    assert_eq!(names[0], "A-101 Plan");
    assert_eq!(
        out.sheets.iter().map(|s| s.tab_order).collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5]
    );

    // Every sheet was written, with its report beside it, and the listing is the sheet order.
    for s in &out.sheets {
        let written = s.outcome.as_ref().expect("every sheet has content");
        assert_eq!(written.artifact, s.file);
        assert_eq!(open_published(&s.file).unwrap().entity_count().unwrap(), 2);
        assert!(report_path(&s.file).is_file());
    }
    let mut want: Vec<String> = expected
        .iter()
        .flat_map(|f| [f.clone(), format!("{f}.report.json")])
        .collect();
    want.push("plan.sheets.json".into());
    want.sort();
    assert_eq!(listing(&dir), want, "no partial file is left behind");
    assert_eq!(
        out.index.as_deref(),
        Some(dir.join("plan.sheets.json").as_path())
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_the_number_is_as_wide_as_the_count_and_counts_every_paper_layout() {
    let dir = scratch("many");
    // A hundred layouts take three digits, and a selection keeps the number the layout has in the
    // whole set, so a re-run of one sheet replaces the same file.
    let out = set(b"MOCK-many", &dir)
        .layouts(["sheet 7", "Sheet 100"])
        .convert_with::<Mock>()
        .unwrap();
    let files: Vec<String> = out
        .sheets
        .iter()
        .map(|s| s.file.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(files, ["plan.007-Sheet-7.pvec", "plan.100-Sheet-100.pvec"]);
    assert_eq!(out.sheets[0].tab_order, 7);
    assert_eq!(out.written().count(), 2);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_the_stem_comes_from_the_file_name_and_bytes_need_one() {
    let dir = scratch("stem");
    let drawing = dir.join("site plan.dwg");
    fs::write(&drawing, b"MOCK-plain").unwrap();
    let out_dir = dir.join("out");
    fs::create_dir(&out_dir).unwrap();
    let out = PvecLayoutSet::new(CadSource::Path(&drawing), &out_dir)
        .sheets(|c| c.unit(SourceUnit::Millimeters))
        .convert_with::<Mock>()
        .unwrap();
    assert_eq!(
        out.sheets[0].file,
        out_dir.join("site plan.01-Layout1.pvec")
    );
    assert!(out_dir.join("site plan.sheets.json").is_file());

    let bytes_dir = scratch("stemless");
    let err = PvecLayoutSet::new(CadSource::Bytes(b"MOCK-plain"), &bytes_dir)
        .sheets(|c| c.unit(SourceUnit::Millimeters))
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, LayoutSetError::StemRequired), "{err:?}");
    assert!(listing(&bytes_dir).is_empty());
    fs::remove_dir_all(&dir).unwrap();
    fs::remove_dir_all(&bytes_dir).unwrap();
}

// ---- empty and zero layouts ----

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_an_empty_layout_is_skipped_and_listed_and_writes_no_file() {
    let dir = scratch("empty");
    let out = set(b"MOCK-mixed", &dir).convert_with::<Mock>().unwrap();
    let by = |name: &str| out.sheets.iter().find(|s| s.layout == name).unwrap();

    // No entity at all, and only the paper view itself: both are not sheets.
    for name in ["Empty", "Viewport only"] {
        let s = by(name);
        assert!(
            matches!(s.outcome, Err(SheetSkip::Empty)),
            "{name}: {:?}",
            s.outcome
        );
        assert!(!s.file.exists(), "{name} wrote a file");
        assert!(!report_path(&s.file).exists(), "{name} wrote a report");
    }
    // Content on a layer that is off: a sheet, but nothing of it is drawn.
    let hidden = by("Hidden");
    assert!(
        matches!(hidden.outcome, Err(SheetSkip::EmptyAfterView)),
        "{:?}",
        hidden.outcome
    );
    assert!(!hidden.file.exists());

    // Empty sheets are not failures.
    let failed: Vec<&str> = out.failed().map(|s| s.layout.as_str()).collect();
    assert_eq!(failed, ["Broken"]);

    let json = index(&dir);
    let status = |name: &str| {
        json["sheets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["layout"] == name)
            .unwrap()["status"]
            .clone()
    };
    assert_eq!(status("Empty"), "empty");
    assert_eq!(status("Viewport only"), "empty");
    assert_eq!(status("Hidden"), "empty-after-view");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_a_drawing_with_no_paper_layout_writes_nothing_and_does_not_fall_back_to_model_space()
{
    let dir = scratch("model");
    let out = set(b"MOCK-model", &dir).convert_with::<Mock>().unwrap();
    assert!(out.sheets.is_empty());
    assert_eq!(out.written().count(), 0);
    assert_eq!(out.failed().count(), 0);
    assert_eq!(out.index, None);
    assert!(listing(&dir).is_empty(), "{:?}", listing(&dir));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_a_drawing_whose_layouts_are_all_empty_still_writes_the_index() {
    // No paper layout writes nothing; paper layouts that are all empty are still a finished run,
    // so the index is written and lists each one as skipped.
    let dir = scratch("blank");
    let out = set(b"MOCK-blank", &dir).convert_with::<Mock>().unwrap();
    assert_eq!(out.sheets.len(), 2);
    assert!(
        out.sheets
            .iter()
            .all(|s| matches!(s.outcome, Err(SheetSkip::Empty)))
    );
    assert_eq!(out.written().count(), 0);
    assert_eq!(out.failed().count(), 0);
    assert_eq!(
        out.index.as_deref(),
        Some(dir.join("plan.sheets.json").as_path())
    );
    assert_eq!(listing(&dir), ["plan.sheets.json"]);
    let json = index(&dir);
    let rows = json["sheets"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r["status"] == "empty"), "{json}");
    fs::remove_dir_all(&dir).unwrap();
}

// ---- one failing sheet among many ----

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_one_failing_sheet_does_not_lose_the_others() {
    let dir = scratch("broken");
    let out = set(b"MOCK-mixed", &dir).convert_with::<Mock>().unwrap();

    let order: Vec<&str> = out.sheets.iter().map(|s| s.layout.as_str()).collect();
    assert_eq!(
        order,
        [
            "Cover",
            "Empty",
            "Viewport only",
            "Hidden",
            "Broken",
            "Plan"
        ]
    );
    let broken = &out.sheets[4];
    assert!(
        matches!(broken.outcome, Err(SheetSkip::Failed(_))),
        "{:?}",
        broken.outcome
    );
    assert!(
        !broken.file.exists(),
        "a failed sheet leaves nothing at its path"
    );
    assert!(!report_path(&broken.file).exists());

    // The sheets on either side of it were written.
    let written: Vec<&str> = out.written().map(|s| s.layout.as_str()).collect();
    assert_eq!(written, ["Cover", "Plan"]);
    for s in out.written() {
        assert!(open_published(&s.file).is_ok());
    }
    assert_eq!(
        out.failed().map(|s| s.file.clone()).collect::<Vec<_>>(),
        [dir.join("plan.05-Broken.pvec")]
    );

    // The index is the record of the run: every layout, its file, status and the reason it failed.
    let json = index(&dir);
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["stem"], "plan");
    let rows = json["sheets"].as_array().unwrap();
    assert_eq!(rows.len(), 6);
    assert_eq!(rows[0]["layout"], "Cover");
    assert_eq!(rows[0]["tab_order"], 1);
    assert_eq!(rows[0]["nn"], "01");
    assert_eq!(rows[0]["file"], "plan.01-Cover.pvec");
    assert_eq!(rows[0]["status"], "written");
    assert_eq!(rows[4]["status"], "failed");
    assert!(
        rows[4]["reason"].as_str().is_some_and(|r| !r.is_empty()),
        "{}",
        rows[4]
    );
    assert!(!listing(&dir).iter().any(|f| f.ends_with(".partial")));
    fs::remove_dir_all(&dir).unwrap();
}

// ---- what fails the whole set ----

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_a_destination_that_is_not_a_directory_is_refused_before_anything_is_written() {
    let dir = scratch("notadir");
    let file = dir.join("taken");
    fs::write(&file, b"x").unwrap();
    let err = PvecLayoutSet::new(CadSource::Bytes(b"MOCK-mixed"), &file)
        .stem("plan")
        .sheets(|c| c.unit(SourceUnit::Millimeters))
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(
        matches!(err, LayoutSetError::NotADirectory(ref p) if p == &file),
        "{err:?}"
    );
    assert_eq!(listing(&dir), ["taken"]);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_a_drawing_that_does_not_open_is_an_error() {
    let dir = scratch("unopened");
    let err = PvecLayoutSet::new(CadSource::Bytes(b"not a drawing"), &dir)
        .stem("plan")
        .sheets(|c| c.unit(SourceUnit::Millimeters))
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, LayoutSetError::Open(_)), "{err:?}");
    assert!(listing(&dir).is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_a_selected_layout_that_does_not_exist_is_an_error() {
    let dir = scratch("unknown");
    let err = set(b"MOCK-mixed", &dir)
        .layouts(["Cover", "Nope"])
        .convert_with::<Mock>()
        .unwrap_err();
    match err {
        LayoutSetError::LayoutNotFound {
            requested,
            available,
        } => {
            assert_eq!(requested, "Nope");
            assert_eq!(available.len(), 6);
        }
        other => panic!("{other:?}"),
    }
    assert!(listing(&dir).is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1564_a_cancelled_run_is_an_error_and_writes_no_index() {
    let dir = scratch("cancel");
    let cancel = CancelToken::new();
    cancel.cancel();
    let err = PvecLayoutSet::new(CadSource::Bytes(b"MOCK-mixed"), &dir)
        .stem("plan")
        .sheets(|c| c.unit(SourceUnit::Millimeters).cancel(cancel.clone()))
        .convert_with::<Mock>()
        .unwrap_err();
    assert!(matches!(err, LayoutSetError::Cancelled), "{err:?}");
    assert!(!dir.join("plan.sheets.json").exists());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn issue_1564_the_failure_keeps_the_conversion_error() {
    // `SheetSkip::Failed` carries the sheet's own error, so a caller can tell a provider fault from
    // a writer refusal without parsing the index.
    fn takes(skip: &SheetSkip) -> Option<&PvecConvertError> {
        match skip {
            SheetSkip::Failed(e) => Some(e),
            _ => None,
        }
    }
    assert!(takes(&SheetSkip::Empty).is_none());
    assert!(takes(&SheetSkip::EmptyAfterView).is_none());
    assert!(takes(&SheetSkip::Failed(PvecConvertError::UnitRequired)).is_some());
}
