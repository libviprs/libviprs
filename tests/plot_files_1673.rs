//! Issue #1673: a PC3 is applied under the layout's page setup, and the plot files a caller hands
//! over are bounded and refused with a code that stays the same.
//!
//! Every file here is built in the test: a CTB from an identity table, a PC3 from a few lines of
//! JSON, a drawing from a hand-built provider. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::cell::RefCell;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use libviprs::cad::plot_files::{
    MAX_PLOT_FILE_BYTES, MAX_PLOT_STYLE_INFLATED_BYTES, Pc3, PlotDevice, PlotFileError,
    PlotStyleTable, load_pc3, load_plot_style,
};
use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::report::SheetSummary;
use libviprs::pvec::{
    Availability, DwgProvider, EntityCursor, EntityScope, EntityStyle, Facet, FacetSource,
    InstancePath, LayerState, LayoutId, LayoutKind, LayoutPlot, LayoutState, ProvenanceStub,
    ProviderCaps, ProviderError, RawEntity, RawGeometry, SourceFact, SourceUnit, ViewProfile,
};
use libviprs::pvec_convert::{PlotSettings, PvecConversion};

// ---- plot files -----------------------------------------------------------------------------

/// The text a CTB's header gives before its payload: 48 bytes, then checksum, text length and
/// compressed length, then the zlib payload.
const CTB_PREFIX: &[u8] = b"PIAFILEVERSION_2.0,CTBVER1,compress\r\npmzlibcodec";

fn ctb() -> Vec<u8> {
    PlotStyleTable::identity("synthetic.ctb")
        .to_bytes()
        .unwrap()
}

/// A CTB whose header announces `announced` bytes of text and whose payload inflates to
/// `inflated` spaces. The checksum is left zero: a reader tolerates a bad one.
fn ctb_of(announced: u32, inflated: usize) -> Vec<u8> {
    let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
    z.write_all(&vec![b' '; inflated]).unwrap();
    let payload = z.finish().unwrap();
    let mut file = CTB_PREFIX.to_vec();
    file.extend_from_slice(&0u32.to_le_bytes());
    file.extend_from_slice(&announced.to_le_bytes());
    file.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    file.extend_from_slice(&payload);
    file
}

/// A JSON PC3 of `paper` millimetres with `printable` (llx, lly, urx, ury) and the named custom
/// settings, laid out the way the PDF driver writes them.
fn pc3_json(paper: [f64; 2], printable: [f64; 4], custom: &[(&str, &str)]) -> Vec<u8> {
    let custom = custom
        .iter()
        .enumerate()
        .map(|(i, (name, value))| {
            format!("\"{i}\" : {{ \"name\" : \"{name}\", \"value\" : {value} }}")
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "PIAFILEVERSION_3.0,json\n{{\n\"data\" : {{\n\"custom\" : {{\n{custom}\n}},\n\"media\" : {{\n\
         \"actual_printable_bounds_llx\" : {},\n\"actual_printable_bounds_lly\" : {},\n\
         \"actual_printable_bounds_urx\" : {},\n\"actual_printable_bounds_ury\" : {},\n\
         \"dm_orientation\" : 1,\n\"size\" : {{\n\"landscape_mode\" : false,\n\
         \"media_description\" : {{ \"media_bounds\" : {{ \"urx\" : {}, \"ury\" : {} }} }},\n\
         \"name\" : \"SYNTHETIC_SHEET\"\n}}\n}}\n}}\n}}\n",
        printable[0], printable[1], printable[2], printable[3], paper[0], paper[1]
    )
    .into_bytes()
}

fn pc3() -> Pc3 {
    let bytes = pc3_json(
        [200.0, 100.0],
        [5.0, 10.0, 195.0, 90.0],
        &[("Raster_Limit", "400"), ("Font_Capture", "true")],
    );
    Pc3::parse(&bytes).unwrap()
}

/// The code a load was refused with, or `loaded` when it was not refused.
fn code<T>(result: Result<T, PlotFileError>) -> &'static str {
    result.err().map_or("loaded", |e| e.code())
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_ctb_over_the_cap_is_refused_before_it_is_read() {
    let mut file = ctb();
    file.resize(MAX_PLOT_FILE_BYTES + 1, 0);
    assert_eq!(
        code(load_plot_style("big.ctb", &file)),
        "plot-file-too-large"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_ctb_exactly_at_the_cap_still_loads() {
    let mut file = ctb();
    file.resize(MAX_PLOT_FILE_BYTES, 0);
    assert!(load_plot_style("edge.ctb", &file).is_ok());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_ctb_that_inflates_past_the_bound_is_refused_whatever_it_announces() {
    // The header says 100 bytes of text, the payload is 9 MiB: a small file that blows up.
    let bomb = ctb_of(100, MAX_PLOT_STYLE_INFLATED_BYTES + 1024 * 1024);
    assert!(bomb.len() < 64 * 1024, "the file itself is small");
    assert_eq!(
        code(load_plot_style("bomb.ctb", &bomb)),
        "plot-file-inflated-too-large"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_ctb_that_announces_past_the_bound_is_refused_without_inflating() {
    let announced = (MAX_PLOT_STYLE_INFLATED_BYTES + 1) as u32;
    let file = ctb_of(announced, 16);
    assert_eq!(
        code(load_plot_style("lie.ctb", &file)),
        "plot-file-inflated-too-large"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_ctb_inflating_to_exactly_the_bound_is_not_refused_for_its_size() {
    let file = ctb_of(100, MAX_PLOT_STYLE_INFLATED_BYTES);
    // Spaces are no table, but the bound is not what refuses them.
    let error = load_plot_style("edge.ctb", &file).err().map(|e| e.code());
    assert_ne!(error, Some("plot-file-inflated-too-large"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_ctb_round_trips_through_the_loader() {
    let table = load_plot_style("round.ctb", &ctb()).unwrap();
    assert_eq!(table.aci_entries.len(), 256);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_bytes_that_are_no_plot_file_are_unrecognised() {
    assert_eq!(
        code(load_plot_style("x.ctb", b"not a plot style table")),
        "plot-file-unrecognised"
    );
    assert_eq!(
        code(load_pc3(b"not a plot configuration")),
        "plot-file-unrecognised"
    );
    assert_eq!(
        code(load_plot_style("x.ctb", b"")),
        "plot-file-unrecognised"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_cut_ctb_is_unreadable_not_a_panic() {
    let file = ctb();
    let cut = &file[..file.len() / 2];
    assert_eq!(
        code(load_plot_style("cut.ctb", cut)),
        "plot-file-unreadable"
    );
    assert_eq!(
        code(load_plot_style("cut.ctb", &file[..50])),
        "plot-file-unreadable"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_pc3_of_either_kind_over_the_cap_is_refused() {
    let json = pc3_json([200.0, 100.0], [0.0, 0.0, 200.0, 100.0], &[]);
    let mut big = json.clone();
    big.resize(MAX_PLOT_FILE_BYTES + 1, b' ');
    assert_eq!(code(load_pc3(&big)), "plot-file-too-large");
    // The older binary kind is kept verbatim, so it is the kind that needs the cap most.
    let mut binary = b"PC3VER1".to_vec();
    binary.resize(MAX_PLOT_FILE_BYTES + 1, 0);
    assert_eq!(code(load_pc3(&binary)), "plot-file-too-large");
    assert!(load_pc3(&json).is_ok());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_pc3_nested_too_deep_is_refused() {
    let mut file = b"PIAFILEVERSION_3.0,json\n".to_vec();
    file.extend(std::iter::repeat_n(b'[', 200));
    assert_eq!(code(load_pc3(&file)), "plot-file-too-deep");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_every_refusal_has_its_own_stable_code() {
    let all = [
        PlotFileError::TooLarge(1),
        PlotFileError::InflatedTooLarge,
        PlotFileError::TooDeep,
        PlotFileError::Unrecognised,
        PlotFileError::Unreadable("x".into()),
    ];
    let codes: Vec<_> = all.iter().map(PlotFileError::code).collect();
    assert_eq!(
        codes,
        [
            "plot-file-too-large",
            "plot-file-inflated-too-large",
            "plot-file-too-deep",
            "plot-file-unrecognised",
            "plot-file-unreadable"
        ]
    );
}

// ---- the device facts ------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_the_device_facts_are_paper_printable_area_raster_limit_and_text_mode() {
    let device = PlotDevice::from_pc3(&pc3()).unwrap();
    assert_eq!(device.paper_mm, [200.0, 100.0]);
    assert_eq!(device.printable_mm, [5.0, 10.0, 195.0, 90.0]);
    assert_eq!(device.raster_limit_dpi, Some(400.0));
    assert_eq!(device.text_as_text, Some(true));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_pc3_that_does_not_say_leaves_raster_limit_and_text_mode_unset() {
    let bytes = pc3_json(
        [200.0, 100.0],
        [0.0, 0.0, 200.0, 100.0],
        &[("Font_Capture", "false")],
    );
    let device = PlotDevice::from_pc3(&Pc3::parse(&bytes).unwrap()).unwrap();
    assert_eq!(device.raster_limit_dpi, None);
    assert_eq!(device.text_as_text, Some(false));
}

// ---- the device under the layout's page setup --------------------------------------------------

#[derive(Clone)]
struct Mock(LayoutPlot);

thread_local! {
    static NEXT: RefCell<Option<Mock>> = const { RefCell::new(None) };
}

struct VecCursor(std::vec::IntoIter<RawEntity>);

impl EntityCursor for VecCursor {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        Ok(self.0.next())
    }
}

impl DwgProvider for Mock {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }
    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        NEXT.with(|n| n.borrow_mut().take())
            .ok_or_else(|| ProviderError::Open {
                provider: "mock-1673",
                detail: "no drawing was handed to this thread".into(),
            })
    }
    fn name(&self) -> &'static str {
        "mock-1673"
    }
    fn caps(&self) -> &ProviderCaps {
        static CAPS: std::sync::OnceLock<ProviderCaps> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| {
            Facet::ALL.iter().fold(ProviderCaps::new(), |c, f| {
                c.with(*f, Availability::Reported, FacetSource::DecoderAdapter)
            })
        })
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        let layout = |id, name: &str, kind, block: &str| LayoutState {
            id: LayoutId(id),
            name: name.into(),
            kind,
            block_name: block.into(),
        };
        Ok(vec![
            layout(0, "Model", LayoutKind::Model, "*Model_Space"),
            layout(1, "Sheet", LayoutKind::Paper, "*Paper_Space"),
        ])
    }
    fn layout_plot(&self, _: LayoutId) -> Result<SourceFact<LayoutPlot>, ProviderError> {
        Ok(SourceFact::Present(self.0.clone()))
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(vec![LayerState {
            name: "0".into(),
            plottable: true,
            ..LayerState::default()
        }])
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let list = match scope {
            EntityScope::Layout(LayoutId(1)) => vec![RawEntity {
                style: EntityStyle::absent(),
                geometry: RawGeometry::Line {
                    start: [10.0, 10.0, 0.0],
                    end: [40.0, 20.0, 0.0],
                },
                provenance: ProvenanceStub {
                    handle: Some(ItemHandle::new(1)),
                    class_name: "LINE".into(),
                    block_path: InstancePath::root(),
                    layer: "0".into(),
                },
            }],
            _ => Vec::new(),
        };
        Ok(Box::new(VecCursor(list.into_iter())))
    }
}

/// The layout's plot settings: `paper` millimetres, `margins` left, bottom, right, top.
fn layout_plot(paper: [f64; 2], margins: [f64; 4]) -> LayoutPlot {
    LayoutPlot {
        paper_size: "TEST_SHEET".into(),
        paper_width: paper[0],
        paper_height: paper[1],
        paper_units: 1,
        plot_type: 5,
        window: [0.0; 4],
        scale_numerator: 1.0,
        scale_denominator: 1.0,
        rotation: 0,
        margins,
        origin: [0.0; 2],
        paper_image_origin: [0.0; 2],
        limits: [0.0; 4],
        style_sheet: "monochrome.ctb".into(),
        print_lineweights: true,
        scale_lineweights: false,
        plot_plot_styles: true,
        plot_viewport_borders: true,
        draw_viewports_first: false,
    }
}

/// The sheet's report summary after converting a layout of `plot` with `device`.
fn sheet_of(tag: &str, plot: LayoutPlot, device: Option<PlotDevice>) -> SheetSummary {
    let dir: PathBuf =
        std::env::temp_dir().join(format!("libviprs-1673-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    NEXT.with(|n| *n.borrow_mut() = Some(Mock(plot)));
    let out = PvecConversion::new(CadSource::Bytes(b"mock"), &dir.join("sheet.pvec"))
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("Sheet"))
        .plot_settings(PlotSettings {
            device,
            ..PlotSettings::default()
        })
        .convert_with::<Mock>()
        .unwrap();
    let _ = fs::remove_dir_all(&dir);
    out.report.sheet.unwrap()
}

fn frame(s: &SheetSummary) -> [f64; 4] {
    s.sheet_box
        .clone()
        .unwrap_or_else(|| panic!("a paper frame: {s:#?}"))
        .map(|v| v.parse::<f64>().unwrap())
}

fn device(printable: [f64; 4]) -> PlotDevice {
    PlotDevice {
        paper_mm: [200.0, 100.0],
        printable_mm: printable,
        raster_limit_dpi: Some(400.0),
        text_as_text: Some(true),
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_layout_with_no_paper_takes_the_devices() {
    let plot = layout_plot([0.0, 0.0], [0.0; 4]);
    let without = sheet_of("nopaper-without", plot.clone(), None);
    assert_eq!(without.frame_reason.as_deref(), Some("paper-size-zero"));
    let s = sheet_of("nopaper", plot, Some(device([0.0, 0.0, 200.0, 100.0])));
    assert_eq!(s.frame, "paper", "{s:#?}");
    assert_eq!(frame(&s), [0.0, 0.0, 200.0, 100.0]);
    assert!(
        s.notes.contains(&"plot-device-paper".to_string()),
        "{:?}",
        s.notes
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_layout_with_no_margins_takes_the_devices_printable_area() {
    let plot = layout_plot([200.0, 100.0], [0.0; 4]);
    let without = sheet_of("nomargin-without", plot.clone(), None);
    assert_eq!(frame(&without), [0.0, 0.0, 200.0, 100.0]);
    let s = sheet_of("nomargin", plot, Some(device([5.0, 10.0, 195.0, 90.0])));
    // The paper's lower left sits 5 and 10 millimetres out from the printable corner.
    assert_eq!(frame(&s), [-5.0, -10.0, 195.0, 90.0]);
    assert!(
        s.notes.contains(&"plot-device-margins".to_string()),
        "{:?}",
        s.notes
    );
    assert!(
        !s.notes.contains(&"plot-device-paper".to_string()),
        "{:?}",
        s.notes
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_the_layouts_own_page_setup_wins_over_the_device() {
    let plot = layout_plot([100.0, 50.0], [10.0, 10.0, 10.0, 10.0]);
    let without = sheet_of("own-without", plot.clone(), None);
    let s = sheet_of("own", plot, Some(device([5.0, 10.0, 195.0, 90.0])));
    assert_eq!(frame(&s), frame(&without));
    assert_eq!(frame(&s), [-10.0, -10.0, 90.0, 40.0]);
    assert!(
        s.notes.iter().all(|n| !n.starts_with("plot-device")),
        "{:?}",
        s.notes
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1673_a_device_with_nothing_to_fill_changes_nothing() {
    let plot = layout_plot([100.0, 50.0], [0.0; 4]);
    let without = sheet_of("flush-without", plot.clone(), None);
    let s = sheet_of("flush", plot, Some(device([0.0, 0.0, 200.0, 100.0])));
    assert_eq!(frame(&s), frame(&without));
    assert_eq!(s.notes, without.notes);
}
