//! Plot to PDF (#1636): print layouts written as the pages of one PDF.
//!
//! Each layout goes through the same conversion a PVec sheet does (print profile, plot settings,
//! paper frame, rotation), into a scratch artifact. The page is that artifact's frame, which is
//! the paper, and its curves are drawn as stroked polylines, black, at the drawing's default
//! lineweight. No text is written (it waits for #1621) and no font is put on the page.

use std::fs;
use std::path::{Path, PathBuf};

use acadrust::plot_style::PlotStyleTable;
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use pvec_facade::Curve;
use pvec_facade::components::{LineStyleStream, StrokePaintStream, StrokeWidth};
use pvec_geometry::Curve2d;

use super::{PlotSettings, PvecConversion, PvecConvertError};
use crate::cad::CadSource;
use crate::pvec::components::StreamSelection;
use crate::pvec::publish::open_published;
use crate::pvec::report::SheetSummary;
use crate::pvec::view_profile::{SpaceSelection, ViewProfileError};
use crate::pvec::writer::metres_per_unit;
use crate::pvec::{DwgProvider, LayoutKind, SourceUnit, ViewProfile};

/// What a PDF plot wrote.
#[derive(Debug)]
pub struct PdfPlotOutput {
    /// The PDF file.
    pub file: PathBuf,
    /// Its page count.
    pub pages: u32,
    /// What was not applied or was replaced, in the vocabulary of the sheet report's `notes`
    /// (`plot-paper-overridden`, ...). The PDF plot has no report section of its own.
    pub notes: Vec<String>,
}

/// Print layouts, plotted to a PDF.
pub struct PdfPlot<'a> {
    source: CadSource<'a>,
    dest: &'a Path,
    profile: ViewProfile,
    unit: Option<SourceUnit>,
    settings: PlotSettings,
    table: Option<PlotStyleTable>,
    layouts: Option<Vec<String>>,
}

const POINTS_PER_METRE: f64 = 72.0 / 0.0254;

impl<'a> PdfPlot<'a> {
    /// A plot of `source` into the PDF `dest`.
    pub fn new(source: CadSource<'a>, dest: &'a Path) -> Self {
        Self {
            source,
            dest,
            profile: ViewProfile::default_profile(),
            unit: None,
            settings: PlotSettings::default(),
            table: None,
            layouts: None,
        }
    }

    /// The layout to plot, as the print profile names it.
    #[must_use]
    pub fn view_profile(mut self, profile: ViewProfile) -> Self {
        self.profile = profile;
        self
    }

    /// The drawing's unit.
    #[must_use]
    pub fn unit(mut self, unit: SourceUnit) -> Self {
        self.unit = Some(unit);
        self
    }

    /// The plot settings, the same type a PVec plot takes.
    #[must_use]
    pub fn plot_settings(mut self, settings: PlotSettings) -> Self {
        self.settings = settings;
        self
    }

    /// The plot style table: stroke colour (screening included) and width per AutoCAD colour
    /// index, as a PVec conversion applies it (#1630). An STB is reported in the notes as
    /// `plot-style-stb-unsupported` and applies nothing. Dash patterns are not drawn yet.
    #[must_use]
    pub fn plot_style_table(mut self, table: PlotStyleTable) -> Self {
        self.table = Some(table);
        self
    }

    /// Only these paper layouts, one page each in tab order, matched the way
    /// [`PvecLayoutSet::layouts`](super::PvecLayoutSet::layouts) matches them. Without this the
    /// page is the layout the view profile names.
    #[must_use]
    pub fn layouts<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        self.layouts = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// The cancel token, polled before each page (#1674).
    #[must_use]
    pub fn cancel(self, _cancel: crate::CancelToken) -> Self {
        self
    }

    /// Receives the sheet events and each failure's diagnostic (#1674).
    #[must_use]
    pub fn observer(self, _observer: std::sync::Arc<dyn crate::observe::EngineObserver>) -> Self {
        self
    }

    /// A wall-clock bound on the whole plot (#1674).
    #[must_use]
    pub fn deadline(self, _within: std::time::Duration) -> Self {
        self
    }

    /// Writes the PDF with provider `P`.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError`] as a PVec conversion: an unknown layout is
    /// [`PvecConvertError::View`], and a layout that has nothing to draw fails as its conversion
    /// does.
    pub fn convert_with<P: DwgProvider>(self) -> Result<PdfPlotOutput, PvecConvertError> {
        let unit = self.unit.ok_or(PvecConvertError::UnitRequired)?;
        let provider = P::open(self.source).map_err(PvecConvertError::Open)?;
        let paper: Vec<String> = provider
            .layouts()
            .map_err(|e| PvecConvertError::View(e.into()))?
            .into_iter()
            .filter(|l| l.kind == LayoutKind::Paper)
            .map(|l| l.name)
            .collect();
        let not_found = |requested: &str| {
            PvecConvertError::View(ViewProfileError::LayoutNotFound {
                requested: requested.to_owned(),
                available: paper.clone(),
            })
        };
        let wanted: Vec<&String> = match (&self.layouts, &self.profile.space) {
            (Some(names), _) => {
                let mut at = Vec::new();
                for name in names {
                    let i = paper
                        .iter()
                        .position(|p| p.eq_ignore_ascii_case(name))
                        .ok_or_else(|| not_found(name))?;
                    at.push(i);
                }
                at.sort_unstable();
                at.dedup();
                at.into_iter().map(|i| &paper[i]).collect()
            }
            (None, SpaceSelection::PaperLayout(name)) => {
                let i = paper
                    .iter()
                    .position(|p| p.eq_ignore_ascii_case(name))
                    .ok_or_else(|| not_found(name))?;
                vec![&paper[i]]
            }
            (None, _) => return Err(not_found("")),
        };

        let work = self.dest.with_extension("plot-work");
        fs::create_dir_all(&work)
            .map_err(|e| PvecConvertError::Publish(crate::pvec::publish::PublishError::Io(e)))?;
        let pages = wanted
            .iter()
            .enumerate()
            .map(|(n, name)| {
                let mut profile = self.profile.clone();
                profile.space = SpaceSelection::PaperLayout((*name).clone());
                let file = work.join(format!("{n}.pvec"));
                let mut conversion = PvecConversion::new(self.source, &file)
                    .view_profile(profile)
                    .plot_settings(self.settings.clone());
                conversion.empty_sheet_ok = true;
                if let Some(table) = &self.table {
                    conversion = conversion
                        .plot_style_table(table.clone())
                        .components(StreamSelection::ALL);
                }
                let out = conversion.run(&provider, unit)?;
                let sheet = out.report.sheet;
                let page = if out.written == 0 {
                    blank_page(sheet.as_ref(), unit)
                } else {
                    read_page(&file).map_err(PvecConvertError::Publish)?
                };
                Ok((page, sheet.map(|s| s.notes).unwrap_or_default()))
            })
            .collect::<Result<Vec<_>, PvecConvertError>>();
        // The scratch artifacts are this call's own, so they go whether or not it worked.
        let _ = fs::remove_dir_all(&work);
        let pages = pages?;

        let mut notes: Vec<String> = Vec::new();
        for n in pages.iter().flat_map(|(_, notes)| notes) {
            if !notes.contains(n) {
                notes.push(n.clone());
            }
        }
        let count = pages.len() as u32;
        write_pdf(self.dest, pages.into_iter().map(|(p, _)| p))
            .map_err(|e| PvecConvertError::Publish(crate::pvec::publish::PublishError::Io(e)))?;
        Ok(PdfPlotOutput {
            file: self.dest.to_path_buf(),
            pages: count,
            notes,
        })
    }
}

/// One page: its size in points and the content stream that draws it.
struct Page {
    size: [f64; 2],
    content: String,
}

/// Reads a scratch artifact into a page: the frame is the paper, the curves are the strokes.
fn read_page(artifact: &Path) -> Result<Page, crate::pvec::publish::PublishError> {
    // A stroke is 0.25 mm wide, AutoCAD's default plot lineweight.
    const DEFAULT_WIDTH_MM: f64 = 0.25;
    let other = |e: &dyn std::fmt::Display| {
        crate::pvec::publish::PublishError::Io(std::io::Error::other(e.to_string()))
    };
    let reader = open_published(artifact).map_err(|e| other(&e))?;
    let manifest = reader.manifest();
    let (lo, hi) = (manifest.bounds.min, manifest.bounds.max);
    let k = manifest.metres_per_unit * POINTS_PER_METRE;
    let size = [(hi[0] - lo[0]) * k, (hi[1] - lo[1]) * k];
    let to_pt = |x: f64, y: f64| ((x - lo[0]) * k, (y - lo[1]) * k);
    let (streams, _) = reader.streams().map_err(|e| other(&e))?;
    let payload = |capability| streams.iter().find(|s| s.capability == capability);
    let paint = payload(StrokePaintStream::CAPABILITY)
        .and_then(|s| StrokePaintStream::decode(&s.payload).ok());
    let style =
        payload(LineStyleStream::CAPABILITY).and_then(|s| LineStyleStream::decode(&s.payload).ok());
    let width_pt = |mm: f64| mm / 1000.0 * POINTS_PER_METRE;
    let mut content = String::new();
    for (id, curve) in reader.entities().map_err(|e| other(&e))? {
        // Stroke widths are in millimetres, as a lineweight is.
        let mm = style
            .as_ref()
            .and_then(|s| {
                let at = s.assignments.iter().find(|(e, _)| *e == id)?.1;
                match s.styles.get(at as usize)?.width? {
                    StrokeWidth::Width(w) => Some(w),
                    StrokeWidth::Thinnest => Some(0.0),
                }
            })
            .unwrap_or(DEFAULT_WIDTH_MM);
        let [r, g, b, _] = paint
            .as_ref()
            .and_then(|p| p.colour_of(id))
            .unwrap_or(0x0000_00ff)
            .to_be_bytes();
        content.push_str(&format!(
            "{:.4} w\n{:.4} {:.4} {:.4} RG\n",
            width_pt(mm),
            f64::from(r) / 255.0,
            f64::from(g) / 255.0,
            f64::from(b) / 255.0
        ));
        for run in polylines(&curve) {
            for (i, (x, y)) in run.into_iter().enumerate() {
                let (x, y) = to_pt(x, y);
                content.push_str(&format!(
                    "{x:.4} {y:.4} {}\n",
                    if i == 0 { "m" } else { "l" }
                ));
            }
            content.push_str("S\n");
        }
    }
    Ok(Page { size, content })
}

/// A paper with nothing on it: the sheet's box in its unit, which is inches, millimetres or the
/// drawing's own.
fn blank_page(sheet: Option<&SheetSummary>, drawing: SourceUnit) -> Page {
    let side = sheet
        .and_then(|s| {
            let b = s.sheet_box.as_ref()?;
            let unit = [SourceUnit::Inches, SourceUnit::Millimeters, drawing]
                .into_iter()
                .find(|u| u.name() == s.unit)?;
            let k = metres_per_unit(unit).ok()? * POINTS_PER_METRE;
            let v = |i: usize| b[i].parse::<f64>().ok();
            Some([(v(2)? - v(0)?) * k, (v(3)? - v(1)?) * k])
        })
        .unwrap_or([0.0; 2]);
    Page {
        size: side,
        content: String::new(),
    }
}

/// A curve as polylines. A bare line has no ends (a trimmed one does), so it has nothing to draw.
fn polylines(curve: &Curve) -> Vec<Vec<(f64, f64)>> {
    // Segments a curved piece is drawn with.
    const CURVED: usize = 96;
    let along = |c: &dyn Curve2d, n: usize| {
        let d = c.domain();
        (0..=n)
            .filter_map(|i| {
                let t = d.lo() + d.width() * (i as f64) / (n as f64);
                c.eval(t.min(d.hi())).ok().map(|p| (p.x, p.y))
            })
            .collect::<Vec<_>>()
    };
    match curve {
        Curve::Line(_) => Vec::new(),
        Curve::Trimmed(t) if matches!(t.basis(), Curve::Line(_)) => vec![along(t, 1)],
        Curve::Polyline(p) => vec![p.vertices().iter().map(|v| (v.x, v.y)).collect()],
        Curve::Composite(c) => c.segments().iter().flat_map(polylines).collect(),
        other => vec![along(other, CURVED)],
    }
}

/// Writes the pages as one PDF. The Producer names libviprs.
fn write_pdf(dest: &Path, pages: impl Iterator<Item = Page>) -> std::io::Result<()> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let mut kids = Vec::new();
    for page in pages {
        let content = doc.add_object(Stream::new(Dictionary::new(), page.content.into_bytes()));
        let id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), Object::Real(page.size[0] as f32), Object::Real(page.size[1] as f32)],
            "Rotate" => 0,
            "Resources" => Dictionary::new(),
            "Contents" => content,
        });
        kids.push(Object::Reference(id));
    }
    let count = kids.len() as i64;
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => count,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let info = doc.add_object(dictionary! {
        "Producer" => Object::string_literal(concat!("libviprs ", env!("CARGO_PKG_VERSION"))),
    });
    doc.trailer.set("Info", info);
    doc.save(dest).map(|_| ())
}
