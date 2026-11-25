//! Plot to PDF (#1636): print layouts written as the pages of one PDF.
//!
//! Each layout goes through the same conversion a PVec sheet does (print profile, plot settings,
//! paper frame, rotation), into a scratch artifact. The page is that artifact's frame, which is
//! the paper, and its curves are drawn as stroked polylines, black, at the drawing's default
//! lineweight. No text is written (it waits for #1621) and no font is put on the page.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::PlacedPatternLine;
use super::pdf_hatch;
use acadrust::plot_style::PlotStyleTable;
use flate2::{Compression, write::ZlibEncoder};
use pvec_components::{Band, Fill, HatchStream, TextStream};
use pvec_facade::Curve;
use pvec_facade::components::{LineStyleStream, StrokePaintStream, StrokeWidth};
use pvec_facade::unstable::RasterId;
use pvec_geometry::{Curve2d, Side};
use std::collections::BTreeMap;
use std::io::{self, Write};

use super::{MODEL_LEAF_BYTES, ModelPass, PlotSettings, PvecConversion, PvecConvertError};
use crate::CancelToken;
use crate::cad::CadSource;
use crate::observe::{EngineEvent, EngineObserver};
use crate::pvec::SourceLimits;
use crate::pvec::components::StreamSelection;
use crate::pvec::publish::{open_published, report_path};
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
    cancel: CancelToken,
    observer: Option<Arc<dyn EngineObserver>>,
    deadline: Option<Duration>,
    limits: Option<SourceLimits>,
    texts: bool,
    fills: bool,
    images: Option<PathBuf>,
    embedded_font: bool,
    hairline_below_mm: f64,
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
            cancel: CancelToken::new(),
            observer: None,
            deadline: None,
            limits: None,
            texts: false,
            fills: false,
            images: None,
            embedded_font: true,
            hairline_below_mm: 0.0,
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
    /// index, as a PVec conversion applies it (#1630). Without one the plot is monochrome: every
    /// stroke, text and solid fill is black and every stroke is 0.25 mm wide, a true colour object
    /// included, and the notes say `plot-style-default-monochrome`. [`PlotStyleTable::identity`]
    /// is the way back to the objects' own colours and widths. An STB is reported in the notes as
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

    /// The cancel token, polled before each page and inside each page's conversion (#1674). A plot
    /// that is cancelled writes no PDF and leaves no scratch directory.
    #[must_use]
    pub fn cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Receives each page's `ConversionSheetStarted` and `ConversionSheetFinished`, with the events
    /// and diagnostics of the page's own conversion (#1674).
    #[must_use]
    pub fn observer(mut self, observer: Arc<dyn EngineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// A wall-clock bound on the whole plot, counted from `convert_with` (#1674). Past it the plot
    /// fails with [`PvecConvertError::Deadline`], writes no PDF and leaves no scratch directory, and the
    /// cancel token stays as it was.
    #[must_use]
    pub fn deadline(mut self, within: Duration) -> Self {
        self.deadline = Some(within);
        self
    }

    /// The reader's limits, as a PVec conversion takes them.
    #[must_use]
    pub fn limits(mut self, limits: SourceLimits) -> Self {
        self.limits = Some(limits);
        self
    }

    /// Draw text as text.
    #[must_use]
    pub fn texts(mut self, on: bool) -> Self {
        self.texts = on;
        self
    }

    /// Fill solids.
    #[must_use]
    pub fn fills(mut self, on: bool) -> Self {
        self.fills = on;
        self
    }

    /// Place DWG IMAGEs as rasters, looking each up by file name in `dir` as
    /// [`PvecConversion::images`] does (#1667).
    #[must_use]
    pub fn images(mut self, dir: impl Into<PathBuf>) -> Self {
        self.images = Some(dir.into());
        self
    }

    /// Draw a stroke thinner than `mm` as the thinnest line (`0 w`), as the owner's plotted PDF does
    /// for every line under some width. Off (0) by default.
    #[must_use]
    pub fn hairline_below(mut self, mm: f64) -> Self {
        self.hairline_below_mm = mm;
        self
    }

    /// Set text in the embedded Arial-metric face (the default) or in the standard Helvetica.
    #[must_use]
    pub fn embedded_font(mut self, on: bool) -> Self {
        self.embedded_font = on;
        self
    }

    /// Writes the PDF, reading the unit from the drawing when none was given.
    ///
    /// # Errors
    ///
    /// As [`convert_with`](Self::convert_with).
    pub fn convert(self) -> Result<PdfPlotOutput, PvecConvertError> {
        self.run::<crate::pvec::dwg_acadrust::AcadrustProvider>(|p| Some(p.source_unit()))
    }

    /// Writes the PDF with provider `P`.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError`] as a PVec conversion: an unknown layout is
    /// [`PvecConvertError::View`], and a layout that has nothing to draw fails as its conversion
    /// does.
    pub fn convert_with<P: DwgProvider>(self) -> Result<PdfPlotOutput, PvecConvertError> {
        self.run::<P>(|_| None)
    }

    fn run<P: DwgProvider>(
        self,
        file_unit: impl FnOnce(&P) -> Option<SourceUnit>,
    ) -> Result<PdfPlotOutput, PvecConvertError> {
        // The deadline counts from here, and rides on this plot's own copy of the cancel token.
        let cancel = match self.deadline.and_then(|d| Instant::now().checked_add(d)) {
            Some(at) => self.cancel.with_deadline(at),
            None => self.cancel.clone(),
        };
        let stopped = || {
            if cancel.deadline_passed() {
                PvecConvertError::Deadline
            } else {
                PvecConvertError::Cancelled
            }
        };
        if cancel.is_cancelled() {
            return Err(stopped());
        }
        let provider = P::open(self.source).map_err(PvecConvertError::Open)?;
        let unit = self
            .unit
            .or_else(|| file_unit(&provider))
            .ok_or(PvecConvertError::UnitRequired)?;
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

        // The plot writes only to names it makes itself, beside the destination: a scratch directory
        // that did not exist before and a temporary file the PDF is renamed from.
        let io = |e| PvecConvertError::Publish(crate::pvec::publish::PublishError::Io(e));
        let dest_dir = match self.dest.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let stem = self
            .dest
            .file_stem()
            .map_or_else(|| "plot".to_owned(), |s| s.to_string_lossy().into_owned());
        fs::create_dir_all(dest_dir).map_err(io)?;
        let (work, tmp) = {
            let mut attempt = 0u32;
            loop {
                let id = format!("{}-{attempt}", std::process::id());
                let work = dest_dir.join(format!(".{stem}.work-{id}"));
                match fs::create_dir(&work) {
                    Ok(()) => break (work, dest_dir.join(format!(".{stem}.pdf-{id}.tmp"))),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && attempt < 1000 => {
                        attempt += 1;
                    }
                    Err(e) => return Err(io(e)),
                }
            }
        };
        // The file is made now and written as each sheet finishes. It goes away on every way out but
        // the rename, and so does the scratch directory.
        let temp = Temp::new(work, tmp);
        let total = wanted.len() as u32;
        let tell = |event: EngineEvent| {
            if let Some(o) = &self.observer {
                o.on_event(event);
            }
        };
        let mut out = PdfOut::new(&temp.pdf, self.embedded_font).map_err(io)?;
        // The model walked once for the plot, and kept for the sheets that read it the same way.
        let model_cache: Arc<Mutex<Option<Arc<ModelPass>>>> = Arc::new(Mutex::new(None));
        let ceiling = self
            .limits
            .unwrap_or(self.profile.limits)
            .max_allocation_bytes;
        let mut kids: Vec<u32> = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        for (n, name) in wanted.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(stopped());
            }
            let index = n as u32 + 1;
            tell(EngineEvent::ConversionSheetStarted {
                index,
                total,
                layout: (*name).clone(),
            });
            let mut profile = self.profile.clone();
            profile.space = SpaceSelection::PaperLayout((*name).clone());
            let file = temp.work.join(format!("{n}.pvec"));
            let mut conversion = PvecConversion::new(self.source, &file)
                .view_profile(profile)
                .plot_settings(self.settings.clone())
                .project_plan(true)
                .texts(self.texts)
                .fills(self.fills)
                .components(StreamSelection::ALL)
                .cancel(cancel.clone());
            if let Some(o) = &self.observer {
                conversion = conversion.observer(o.clone());
            }
            conversion.empty_sheet_ok = true;
            conversion.arc_sides = true;
            // Nothing reads the per entity report of a plot sheet, so none is built or kept.
            conversion.plot_mode = true;
            conversion.model_cache = Some(model_cache.clone());
            if let Some(dir) = &self.images {
                conversion = conversion.images(dir.clone());
            }
            if let Some(limits) = self.limits {
                conversion = conversion.limits(limits);
            }
            if let Some(table) = &self.table {
                conversion = conversion.plot_style_table(table.clone());
            }
            let mono = self.table.is_none();
            let page = conversion.run(&provider, unit).and_then(|done| {
                let sheet = done.report.sheet;
                let rotation = sheet
                    .as_ref()
                    .filter(|s| s.frame == "paper")
                    .and_then(|s| s.plot.as_ref())
                    .map_or(0, |p| p.rotation);
                let page = if done.written == 0 {
                    let size = blank_size(sheet.as_ref(), unit);
                    check_size(size)?;
                    let content = out.stream("", |_| Ok(()))?;
                    PageRef {
                        size,
                        rotate: 0,
                        content,
                        text: false,
                        images: Vec::new(),
                        notes: Vec::new(),
                    }
                } else {
                    let budget = Budget {
                        ceiling,
                        carried: model_cache
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .as_ref()
                            .map_or(0, |p| p.leaves.saturating_mul(MODEL_LEAF_BYTES)),
                        layout: name,
                    };
                    budget.check(
                        "artifact read",
                        // The curve, its lookups and its style: about 256 bytes an entity while the page is read.
                        done.written.saturating_mul(256),
                    )?;
                    read_page(
                        &budget,
                        &file,
                        rotation,
                        paper_frame(sheet.as_ref(), unit),
                        &done.hatch_patterns,
                        self.hairline_below_mm,
                        &cancel,
                        mono,
                        &mut out,
                    )?
                };
                kids.push(out.page(&page).map_err(io)?);
                let mut sheet_notes = sheet.map(|s| s.notes).unwrap_or_default();
                sheet_notes.extend(page.notes);
                if mono {
                    sheet_notes.push("plot-style-default-monochrome".into());
                }
                for note in sheet_notes {
                    if !notes.contains(&note) {
                        notes.push(note);
                    }
                }
                Ok(())
            });
            // The sheet's scratch artifact and report have been read, so they go before the next
            // sheet starts: the scratch never holds more than one.
            let _ = fs::remove_file(&file);
            let _ = fs::remove_file(report_path(&file));
            let status = match &page {
                Ok(()) => ("written", None),
                Err(PvecConvertError::Cancelled) => ("cancelled", Some("CANCELLED")),
                Err(PvecConvertError::Deadline) => ("deadline", Some("DEADLINE")),
                Err(e) => ("failed", Some(e.code())),
            };
            tell(EngineEvent::ConversionSheetFinished {
                index,
                total,
                layout: (*name).clone(),
                status: status.0.to_owned(),
                code: status.1.map(str::to_owned),
            });
            page.map_err(|e| e.settled(&cancel))?;
        }
        if cancel.is_cancelled() {
            return Err(stopped());
        }
        let count = kids.len() as u32;
        // The PDF goes to a name of the plot's own and is renamed over the destination only once it is
        // whole and on disk, so a failed plot leaves what was there alone.
        out.finish(&kids)
            .and_then(|file| file.sync_all())
            .and_then(|()| fs::rename(&temp.pdf, self.dest))
            .map_err(io)?;
        temp.disarm();
        Ok(PdfPlotOutput {
            file: self.dest.to_path_buf(),
            pages: count,
            notes,
        })
    }
}

/// The plot's own scratch names: the directory the sheets are converted in and the file the PDF is
/// written to. They are removed when this goes, unless the PDF was renamed into place.
struct Temp {
    work: PathBuf,
    pdf: PathBuf,
    armed: bool,
}

impl Temp {
    fn new(work: PathBuf, pdf: PathBuf) -> Self {
        Self {
            work,
            pdf,
            armed: true,
        }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.work);
        if self.armed {
            let _ = fs::remove_file(&self.pdf);
        }
    }
}

/// What a plot may hold at a stage: the ceiling, what the plot already carries between sheets and
/// the layout being read, so a refusal names where it was.
struct Budget<'a> {
    ceiling: u64,
    carried: u64,
    layout: &'a str,
}

impl Budget<'_> {
    fn check(&self, stage: &'static str, bytes: u64) -> Result<(), PvecConvertError> {
        let observed = self.carried.saturating_add(bytes);
        if observed > self.ceiling {
            return Err(PvecConvertError::AllocationBudget {
                stage,
                layout: self.layout.to_owned(),
                observed,
                ceiling: self.ceiling,
            });
        }
        Ok(())
    }
}

/// One page written to the file: its box in points, the objects it draws with, and what it could
/// not draw as asked.
struct PageRef {
    /// The size the page is displayed at, in points: the sheet's frame.
    size: [f64; 2],
    /// `/Rotate`: 0, 90 or 270. A layout rotated a quarter turn is displayed turned.
    rotate: i64,
    /// The object number of the content stream.
    content: u32,
    text: bool,
    /// The object numbers of the rasters the content draws as `/Im<n>`.
    images: Vec<u32>,
    /// What the page could not draw as asked, in the vocabulary of the sheet report's notes.
    notes: Vec<String>,
}

/// A path segment in page points.
enum Seg {
    Move(f64, f64),
    Line(f64, f64),
    Curve([f64; 6]),
}

/// A stroke or a fill's outline as segments, each run starting with a [`Seg::Move`].
type Runs = Vec<Seg>;

/// Content operators on their way into a stream. A write that fails is kept and ends the page at
/// `finish`, so the drawing code reads as it did when it built a string.
struct Content<'a> {
    out: &'a mut dyn Write,
    failed: Option<io::Error>,
}

impl Content<'_> {
    fn push_str(&mut self, s: &str) {
        if self.failed.is_none()
            && let Err(e) = self.out.write_all(s.as_bytes())
        {
            self.failed = Some(e);
        }
    }

    fn finish(self) -> Result<(), PvecConvertError> {
        self.failed.map_or(Ok(()), |e| Err(publish_io(e)))
    }
}

fn publish_io(e: io::Error) -> PvecConvertError {
    PvecConvertError::Publish(crate::pvec::publish::PublishError::Io(e))
}

/// Reads a scratch artifact into a page written to `pdf`: the frame is the paper, the curves are
/// the strokes. The content goes through Flate into the file as it is made, and each raster is
/// written and dropped before the next is decoded, so the page never exists in memory as a whole.
#[allow(clippy::too_many_arguments)]
fn read_page(
    budget: &Budget<'_>,
    artifact: &Path,
    rotation: i16,
    frame: Option<([f64; 2], [f64; 2])>,
    patterns: &BTreeMap<u32, Vec<PlacedPatternLine>>,
    hairline_below_mm: f64,
    cancel: &CancelToken,
    monochrome: bool,
    pdf: &mut PdfOut,
) -> Result<PageRef, PvecConvertError> {
    // A stroke is 0.25 mm wide, AutoCAD's default plot lineweight.
    const DEFAULT_WIDTH_MM: f64 = 0.25;
    let other = |e: &dyn std::fmt::Display| publish_io(io::Error::other(e.to_string()));
    // Strokes drawn by every pattern hatch of the page together.
    let mut stroke_budget = pdf_hatch::MAX_PAGE_STROKES;
    let reader = open_published(artifact).map_err(|e| other(&e))?;
    let manifest = reader.manifest();
    // The page is the paper, even when a text or fill carrier reaches past its edge and widens what
    // the artifact holds.
    let (lo, hi) = match frame {
        Some((lo, hi)) => (
            lo.map(|v| v / manifest.metres_per_unit),
            hi.map(|v| v / manifest.metres_per_unit),
        ),
        None => (manifest.bounds.min, manifest.bounds.max),
    };
    let k = manifest.metres_per_unit * POINTS_PER_METRE;
    let size = [(hi[0] - lo[0]) * k, (hi[1] - lo[1]) * k];
    check_size(size)?;
    let to_pt = |x: f64, y: f64| ((x - lo[0]) * k, (y - lo[1]) * k);
    let (streams, _) = reader.streams().map_err(|e| other(&e))?;
    let payload = |capability| streams.iter().find(|s| s.capability == capability);
    let paint = payload(StrokePaintStream::CAPABILITY)
        .and_then(|s| StrokePaintStream::decode(&s.payload).ok());
    let style =
        payload(LineStyleStream::CAPABILITY).and_then(|s| LineStyleStream::decode(&s.payload).ok());
    let texts = payload(TextStream::CAPABILITY).and_then(|s| TextStream::decode(&s.payload).ok());
    let hatches =
        payload(HatchStream::CAPABILITY).and_then(|s| HatchStream::decode(&s.payload).ok());
    let width_pt = |mm: f64| mm / 1000.0 * POINTS_PER_METRE;
    // A pattern's lines and a solid fill's edge are the thinnest line, or 0.25 mm in a monochrome plot.
    let edge = if monochrome {
        format!("{:.4}", width_pt(DEFAULT_WIDTH_MM))
    } else {
        "0".to_owned()
    };
    let rgb = |word: u32| {
        let [r, g, b, _] = word.to_be_bytes();
        format!(
            "{:.4} {:.4} {:.4}",
            f64::from(r) / 255.0,
            f64::from(g) / 255.0,
            f64::from(b) / 255.0
        )
    };
    let entities = reader.entities().map_err(|e| other(&e))?;
    let boundary: std::collections::HashSet<_> = hatches
        .iter()
        .flat_map(|h| h.records.iter().flat_map(|r| r.boundary.iter().copied()))
        .collect();

    // Rasters placed under the vectors, then over them (#1667). Each is written as an object of
    // its own and only its draw operators stay.
    let mut images: Vec<u32> = Vec::new();
    let mut underlay = String::new();
    let mut overlay = String::new();
    for (n, placement) in reader
        .raster_placements()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let Ok(id) = RasterId::new(n as u32) else {
            continue;
        };
        // The picture is decoded and converted to RGBA, so it is charged before it is read.
        budget.check(
            "image",
            u64::from(placement.width)
                .saturating_mul(u64::from(placement.height))
                .saturating_mul(4),
        )?;
        let top = u32::from(placement.levels).saturating_sub(1);
        let Ok(Some(bytes)) = reader.raster_tile(id, top, 0, 0) else {
            continue;
        };
        let Ok(decoded) = image::load_from_memory(&bytes) else {
            continue;
        };
        // The tile is padded to the tile edge: the level's own pixels are its top-left corner.
        let level_side = |n: u32| n.div_ceil(1u32 << top.min(31)).max(1);
        let rgba = image::imageops::crop_imm(
            &decoded.to_rgba8(),
            0,
            0,
            level_side(placement.width),
            level_side(placement.height),
        )
        .to_image();
        drop(decoded);
        let (w, h) = (f64::from(placement.width), f64::from(placement.height));
        let [a, b, c, d, tx, ty] = placement.pixel_to_source;
        let [x0, y0, x1, y1] = placement.visible_box.map(f64::from);
        // The unit square, bottom left at (0, 0): pixel (px, py) is (u * w, (1 - v) * h).
        let cm = [
            a * w * k,
            c * w * k,
            -b * h * k,
            -d * h * k,
            (tx + b * h - lo[0]) * k,
            (ty + d * h - lo[1]) * k,
        ];
        let mut draw = format!(
            "q\n{:.6} {:.6} {:.6} {:.6} {:.4} {:.4} cm\n",
            cm[0], cm[1], cm[2], cm[3], cm[4], cm[5]
        );
        if placement.visible_box != [0, 0, placement.width, placement.height] {
            draw.push_str(&format!(
                "{:.6} {:.6} {:.6} {:.6} re W n\n",
                x0 / w,
                1.0 - y1 / h,
                (x1 - x0) / w,
                (y1 - y0) / h
            ));
        }
        draw.push_str(&format!("/Im{} Do\nQ\n", images.len()));
        let opaque = rgba.pixels().all(|p| p.0[3] == 255);
        let dims = format!(
            "/Type /XObject /Subtype /Image /Width {} /Height {} /BitsPerComponent 8",
            rgba.width(),
            rgba.height()
        );
        let mask = if opaque {
            None
        } else {
            Some(
                pdf.stream(&format!("{dims} /ColorSpace /DeviceGray"), |out| {
                    write_pixels(out, &rgba, |p, v| v.push(p.0[3]))
                })?,
            )
        };
        let smask = mask.map_or(String::new(), |m| format!(" /SMask {m} 0 R"));
        let object = pdf.stream(&format!("{dims} /ColorSpace /DeviceRGB{smask}"), |out| {
            write_pixels(out, &rgba, |p, v| v.extend_from_slice(&p.0[..3]))
        })?;
        images.push(object);
        match placement.band {
            Band::Underlay => underlay.push_str(&draw),
            _ => overlay.push_str(&draw),
        }
    }

    // Lookups by entity, so a page of hundreds of thousands of curves is not a quadratic scan.
    let curve_at: std::collections::HashMap<_, _> = entities
        .iter()
        .enumerate()
        .map(|(n, (id, _))| (*id, n))
        .collect();
    let style_at: std::collections::HashMap<_, _> = style
        .iter()
        .flat_map(|s| s.assignments.iter().map(|(e, n)| (*e, *n)))
        .collect();

    // A layout rotated a quarter turn is the paper turned: the box is the paper as it feeds (the
    // frame with its sides swapped) and `/Rotate` shows it the way the layout reads, as a
    // plotter's own PDF does. A portrait Tabloid layout at rotation 1 is a portrait box rotated
    // 270 and displays as the landscape frame. The content is drawn in the frame, so it is
    // turned back into the box.
    let rotate = match rotation {
        1 => 270,
        3 => 90,
        _ => 0,
    };
    let mut notes: Vec<String> = Vec::new();
    let mut text = false;
    let content = pdf.stream("", |sink| {
        let mut content = Content {
            out: sink,
            failed: None,
        };
        if rotate != 0 {
            let (displayed_w, displayed_h) = (size[0], size[1]);
            let cm = if rotate == 270 {
                format!("0 -1 1 0 0 {displayed_w:.4} cm\n")
            } else {
                format!("0 1 -1 0 {displayed_h:.4} 0 cm\n")
            };
            content.push_str(&cm);
        }
        content.push_str(&underlay);

        // Fills first, under the strokes.
        for record in hatches.iter().flat_map(|h| &h.records) {
            if cancel.is_cancelled() {
                return Err(PvecConvertError::Cancelled);
            }
            let (rgba, solid_fill) = match &record.fill {
                Fill::Solid { rgba } => (rgba, true),
                Fill::Pattern { rgba, .. } => (rgba, false),
            };
            let mut runs: Runs = Vec::new();
            for id in &record.boundary {
                if let Some(n) = curve_at.get(id) {
                    runs.extend(segments(&entities[*n].1, &to_pt));
                }
            }
            // The hatch's own colour, not its boundary's stroke colour, thinned toward the white paper
            // by its transparency.
            let [r, g, b, a] = rgba.to_be_bytes();
            let [r, g, b] = if monochrome { [0; 3] } else { [r, g, b] };
            let thin =
                |c: u8| (255.0 - (255.0 - f64::from(c)) * f64::from(a) / 255.0).round() as u8;
            let word = u32::from_be_bytes([thin(r), thin(g), thin(b), 255]);
            let mut solid = solid_fill;
            if !solid_fill {
                // A pattern is the lines of its line families, cut to the boundary; a hatch whose
                // pattern is too fine to space is filled, and the plot says so.
                let Some(families) = record
                    .boundary
                    .first()
                    .and_then(|id| patterns.get(&id.index()))
                else {
                    continue;
                };
                let placed: Vec<PlacedPatternLine> = families
                    .iter()
                    .map(|f| PlacedPatternLine {
                        origin: {
                            let (x, y) = to_pt(f.origin[0], f.origin[1]);
                            [x, y]
                        },
                        direction: f.direction,
                        step: [f.step[0] * k, f.step[1] * k],
                        dashes: f.dashes.iter().map(|d| d * k).collect(),
                    })
                    .collect();
                match pdf_hatch::expand(
                    &rings_of(&runs),
                    &placed,
                    &|| cancel.is_cancelled(),
                    &mut stroke_budget,
                ) {
                    Ok(strokes) => {
                        content.push_str(&format!(
                            "{edge} w\n{} RG\n[] 0 d\n1 J 1 j\n",
                            rgb(word)
                        ));
                        for s in strokes {
                            content.push_str(&format!(
                                "{:.4} {:.4} m\n{:.4} {:.4} l\nS\n",
                                s[0], s[1], s[2], s[3]
                            ));
                        }
                    }
                    Err(pdf_hatch::Refused::Stopped) => return Err(PvecConvertError::Cancelled),
                    Err(pdf_hatch::Refused::TooDense) => {
                        if !notes.iter().any(|n| n == "hatch-too-dense") {
                            notes.push("hatch-too-dense".into());
                        }
                        solid = true;
                    }
                }
            }
            if solid {
                content.push_str(&format!("{} rg\n", rgb(word)));
                push_path(&mut content, &runs, true);
                content.push_str("f*\n");
                // A solid fill is filled and stroked: its edge, in its own colour, at the thinnest line.
                if solid_fill {
                    content.push_str(&format!("{edge} w\n{} RG\n[] 0 d\n1 J 1 j\n", rgb(word)));
                    push_path(&mut content, &runs, true);
                    content.push_str("S\n");
                }
            }
        }

        for (n, (id, curve)) in entities.iter().enumerate() {
            if n % 4096 == 0 && cancel.is_cancelled() {
                return Err(PvecConvertError::Cancelled);
            }
            if content.failed.is_some() {
                break;
            }
            if boundary.contains(id) {
                continue;
            }
            let def = style.as_ref().and_then(|s| {
                let at = *style_at.get(id)?;
                s.styles.get(at as usize)
            });
            // A style that is not visible is a text's or fill's carrier: the record draws, not the curve.
            if def.is_some_and(|d| !d.visible) {
                continue;
            }
            // Stroke widths are in millimetres, as a lineweight is.
            let mm = match def.and_then(|d| d.width) {
                _ if monochrome => DEFAULT_WIDTH_MM,
                Some(StrokeWidth::Width(w)) => w,
                Some(StrokeWidth::Thinnest) => 0.0,
                None => DEFAULT_WIDTH_MM,
            };
            // Under the hairline limit a stroke is the thinnest line.
            let mm = if mm < hairline_below_mm { 0.0 } else { mm };
            let word = paint
                .as_ref()
                .and_then(|p| p.colour_of(*id))
                .filter(|_| !monochrome)
                .unwrap_or(0x0000_00ff);
            content.push_str(&format!(
                "{:.4} w\n{} RG\n1 J 1 j\n",
                width_pt(mm),
                rgb(word)
            ));
            // Dashes are in the sheet's unit, as the coordinates are.
            let dashes = def.map(|d| d.dashes.as_slice()).unwrap_or_default();
            if dashes.is_empty() {
                content.push_str("[] 0 d\n");
            } else {
                let list: Vec<String> = dashes.iter().map(|d| format!("{:.4}", d * k)).collect();
                content.push_str(&format!(
                    "[{}] {:.4} d\n",
                    list.join(" "),
                    def.map_or(0.0, |d| d.phase) * k
                ));
            }
            let runs = segments(curve, &to_pt);
            push_path(&mut content, &runs, false);
            content.push_str("S\n");
        }

        content.push_str(&overlay);
        for t in texts.iter().flat_map(|t| &t.records) {
            let word = paint
                .as_ref()
                .and_then(|p| p.colour_of(t.entity))
                .filter(|_| !monochrome)
                .unwrap_or(0x0000_00ff);
            let (x, y) = to_pt(t.x, t.y);
            let (s, c) = t.rotation.sin_cos();
            content.push_str(&format!(
                "BT\n/F1 {:.4} Tf\n{} rg\n{c:.6} {s:.6} {:.6} {c:.6} {x:.4} {y:.4} Tm\n({}) Tj\nET\n",
                // A text's height is the height of a capital, and Helvetica's stand 0.716 em high.
                t.height * k / 0.716,
                rgb(word),
                -s,
                pdf_string(&t.text)
            ));
            text = true;
        }
        content.finish()
    })?;
    Ok(PageRef {
        size,
        rotate,
        content,
        text,
        images,
        notes,
    })
}

/// Writes the pixels of `image` as `bytes` gives them, a few thousand at a time.
fn write_pixels(
    out: &mut dyn Write,
    image: &image::RgbaImage,
    push: impl Fn(&image::Rgba<u8>, &mut Vec<u8>),
) -> Result<(), PvecConvertError> {
    let mut chunk: Vec<u8> = Vec::with_capacity(1 << 16);
    for p in image.pixels() {
        push(p, &mut chunk);
        if chunk.len() >= 1 << 16 {
            out.write_all(&chunk).map_err(publish_io)?;
            chunk.clear();
        }
    }
    out.write_all(&chunk).map_err(publish_io)
}

/// The closed rings of a path's runs, each run one ring, curves sampled.
fn rings_of(runs: &Runs) -> Vec<Vec<[f64; 2]>> {
    let mut rings: Vec<Vec<[f64; 2]>> = Vec::new();
    for seg in runs {
        match seg {
            Seg::Move(x, y) => rings.push(vec![[*x, *y]]),
            Seg::Line(x, y) => {
                if let Some(r) = rings.last_mut() {
                    r.push([*x, *y]);
                }
            }
            Seg::Curve(p) => {
                if let Some(r) = rings.last_mut() {
                    let a = r.last().copied().unwrap_or([p[4], p[5]]);
                    for i in 1..=8 {
                        let t = f64::from(i) / 8.0;
                        let u = 1.0 - t;
                        let at = |v0: f64, v1: f64, v2: f64, v3: f64| {
                            u * u * u * v0
                                + 3.0 * u * u * t * v1
                                + 3.0 * u * t * t * v2
                                + t * t * t * v3
                        };
                        r.push([at(a[0], p[0], p[2], p[4]), at(a[1], p[1], p[3], p[5])]);
                    }
                }
            }
        }
    }
    rings.retain(|r| r.len() >= 3);
    rings
}

/// A page's content as a PDF literal string: Latin-1, with anything else as `?`.
fn pdf_string(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '(' | ')' | '\\' => format!("\\{c}"),
            ' '..='~' => c.to_string(),
            c if (c as u32) < 256 => format!("\\{:03o}", c as u32),
            // The WinAnsi codes in 0x80 to 0x9F that the face has glyphs for.
            '\u{20AC}' => "\\200".into(),
            '\u{2026}' => "\\205".into(),
            '\u{2018}' => "\\221".into(),
            '\u{2019}' => "\\222".into(),
            '\u{201C}' => "\\223".into(),
            '\u{201D}' => "\\224".into(),
            '\u{2022}' => "\\225".into(),
            '\u{2013}' => "\\226".into(),
            '\u{2014}' => "\\227".into(),
            '\u{2122}' => "\\231".into(),
            _ => "?".to_owned(),
        })
        .collect()
}

/// A PDF page is at most this many points on a side.
const MAX_PAGE_POINTS: f64 = 14_400.0;

/// Refuses a page that is not a PDF page: a side that is not finite, is negative, or is over the
/// 14400 point limit.
fn check_size([w, h]: [f64; 2]) -> Result<(), PvecConvertError> {
    let bad = |v: f64| !v.is_finite() || !(0.0..=MAX_PAGE_POINTS).contains(&v);
    if bad(w) || bad(h) {
        return Err(PvecConvertError::PageRefused {
            reason: "a side is not a finite length of at most 14400 points",
        });
    }
    Ok(())
}

fn push_path(content: &mut Content<'_>, runs: &Runs, close: bool) {
    let mut open = false;
    for seg in runs {
        // A coordinate that is not finite is left out, as a number a PDF cannot hold.
        let finite = match seg {
            Seg::Move(x, y) | Seg::Line(x, y) => x.is_finite() && y.is_finite(),
            Seg::Curve(p) => p.iter().all(|v| v.is_finite()),
        };
        if !finite {
            continue;
        }
        match seg {
            Seg::Move(x, y) => {
                if close && open {
                    content.push_str("h\n");
                }
                open = true;
                content.push_str(&format!("{x:.4} {y:.4} m\n"));
            }
            Seg::Line(x, y) => content.push_str(&format!("{x:.4} {y:.4} l\n")),
            Seg::Curve(p) => content.push_str(&format!(
                "{:.4} {:.4} {:.4} {:.4} {:.4} {:.4} c\n",
                p[0], p[1], p[2], p[3], p[4], p[5]
            )),
        }
    }
    if close {
        content.push_str("h\n");
    }
}

/// The paper frame of a sheet whose frame is the paper, as [min, max] corners in metres.
fn paper_frame(sheet: Option<&SheetSummary>, drawing: SourceUnit) -> Option<([f64; 2], [f64; 2])> {
    let s = sheet.filter(|s| s.frame == "paper")?;
    let b = s.sheet_box.as_ref()?;
    let unit = [SourceUnit::Inches, SourceUnit::Millimeters, drawing]
        .into_iter()
        .find(|u| u.name() == s.unit)?;
    let k = metres_per_unit(unit).ok()?;
    let v = |i: usize| b[i].parse::<f64>().ok().map(|x| x * k);
    Some(([v(0)?, v(1)?], [v(2)?, v(3)?]))
}

/// The size of a paper with nothing on it: the sheet's box in its unit, which is inches,
/// millimetres or the drawing's own.
fn blank_size(sheet: Option<&SheetSummary>, drawing: SourceUnit) -> [f64; 2] {
    sheet
        .and_then(|s| {
            let b = s.sheet_box.as_ref()?;
            let unit = [SourceUnit::Inches, SourceUnit::Millimeters, drawing]
                .into_iter()
                .find(|u| u.name() == s.unit)?;
            let k = metres_per_unit(unit).ok()? * POINTS_PER_METRE;
            let v = |i: usize| b[i].parse::<f64>().ok();
            Some([(v(2)? - v(0)?) * k, (v(3)? - v(1)?) * k])
        })
        .unwrap_or([0.0; 2])
}

/// A curve as segments in page points. A bare line has no ends (a trimmed one does), so it has
/// nothing to draw. Circles, ellipses and trimmed pieces of them are cubic Beziers, a quarter turn
/// each; splines and Beziers are cubic Hermite spans of their own derivative.
fn segments(curve: &Curve, to_pt: &dyn Fn(f64, f64) -> (f64, f64)) -> Runs {
    let pt = |c: &dyn Curve2d, t: f64| {
        let d = c.domain();
        c.eval(t.clamp(d.lo(), d.hi()))
            .ok()
            .map(|p| to_pt(p.x, p.y))
    };
    // Cubic spans over the curve's domain; `angular` curves follow the circle rule for the
    // handle length (4/3 tan(a/4) of the speed), the others the Hermite one (a third of it).
    let spans = |c: &dyn Curve2d, n: usize, angular: bool| -> Runs {
        let d = c.domain();
        let h = d.width() / n as f64;
        let handle = if angular {
            4.0 / 3.0 * (h / 4.0).tan()
        } else {
            h / 3.0
        };
        let mut out = Vec::new();
        for i in 0..n {
            let (t0, t1) = (d.lo() + h * i as f64, d.lo() + h * (i + 1) as f64);
            let (Some(p0), Some(p1)) = (pt(c, t0), pt(c, t1)) else {
                continue;
            };
            if i == 0 {
                out.push(Seg::Move(p0.0, p0.1));
            }
            let slope = |t: f64, sign: f64| {
                let dd = c.derivative(t.clamp(d.lo(), d.hi()), 1, Side::Right).ok()?;
                // the derivative is in drawing units; the page is the same map scaled
                let (ox, oy) = to_pt(0.0, 0.0);
                let (ex, ey) = to_pt(dd.x * sign, dd.y * sign);
                Some((ex - ox, ey - oy))
            };
            let (Some(d0), Some(d1)) = (slope(t0, 1.0), slope(t1, -1.0)) else {
                out.push(Seg::Line(p1.0, p1.1));
                continue;
            };
            out.push(Seg::Curve([
                p0.0 + d0.0 * handle,
                p0.1 + d0.1 * handle,
                p1.0 + d1.0 * handle,
                p1.1 + d1.1 * handle,
                p1.0,
                p1.1,
            ]));
        }
        out
    };
    let quarter_turns = |c: &dyn Curve2d| {
        (c.domain().width() / std::f64::consts::FRAC_PI_2)
            .ceil()
            .max(1.0) as usize
    };
    match curve {
        Curve::Line(_) => Vec::new(),
        Curve::Trimmed(t) => match t.basis() {
            Curve::Line(_) => spans_line([pt(t, t.domain().lo()), pt(t, t.domain().hi())]),
            Curve::Circle(_) | Curve::Ellipse(_) => spans(t, quarter_turns(t), true),
            _ => spans(t, 16, false),
        },
        Curve::Polyline(p) => {
            let mut out = Vec::new();
            for (i, v) in p.vertices().iter().enumerate() {
                let (x, y) = to_pt(v.x, v.y);
                out.push(if i == 0 {
                    Seg::Move(x, y)
                } else {
                    Seg::Line(x, y)
                });
            }
            out
        }
        Curve::Composite(c) => {
            let mut out: Runs = Vec::new();
            for s in c.segments() {
                out.extend(segments(s, to_pt));
            }
            out
        }
        Curve::Circle(_) | Curve::Ellipse(_) => spans(curve, 4, true),
        other => spans(other, 16, false),
    }
}

fn spans_line(ends: [Option<(f64, f64)>; 2]) -> Runs {
    match ends {
        [Some(a), Some(b)] => vec![Seg::Move(a.0, a.1), Seg::Line(b.0, b.1)],
        _ => Vec::new(),
    }
}

/// A PDF written as it is made: objects go to the file the moment they are complete, their offsets
/// are kept (eight bytes each), and the page tree, catalog and cross reference table come last. The
/// output is a function of what was written and in what order: object numbers follow the order of
/// writing, and there is no date and no document id.
struct PdfOut {
    file: io::BufWriter<fs::File>,
    pos: u64,
    /// Where object `n + 1` starts.
    offsets: Vec<u64>,
    font: u32,
}

/// The object number of the page tree, reserved first so every page can name its parent.
const PAGES: u32 = 1;

impl PdfOut {
    /// Creates `path` and writes the header and the font.
    fn new(path: &Path, embedded: bool) -> io::Result<Self> {
        /// A subset of Liberation Sans Regular (SIL OFL 1.1, renamed, see `fonts/LIBERATION-OFL.txt`):
        /// Latin glyphs, with Arial's advance widths.
        const FACE: &[u8] = include_bytes!("../../fonts/LibviprsLatin-Regular.ttf");
        /// The face's advance widths in 1/1000 em for the WinAnsi codes 32 to 255 (0 where it has no glyph).
        const WIDTHS: [i64; 224] = [
            278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556,
            556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667,
            667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722,
            667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556, 333, 556, 556, 500,
            556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278,
            556, 500, 722, 500, 500, 500, 334, 260, 334, 584, 0, 556, 0, 0, 0, 0, 1000, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 222, 222, 333, 333, 350, 556, 1000, 0, 1000, 0, 0, 0, 0, 0, 0,
            278, 333, 556, 556, 556, 556, 260, 556, 333, 737, 370, 556, 584, 333, 737, 552, 400,
            549, 333, 333, 333, 576, 537, 333, 333, 333, 365, 556, 834, 834, 834, 611, 667, 667,
            667, 667, 667, 667, 1000, 722, 667, 667, 667, 667, 278, 278, 278, 278, 722, 722, 778,
            778, 778, 778, 778, 584, 778, 722, 722, 722, 722, 667, 667, 611, 556, 556, 556, 556,
            556, 556, 889, 500, 556, 556, 556, 556, 278, 278, 278, 278, 556, 556, 556, 556, 556,
            556, 556, 549, 611, 556, 556, 556, 556, 500, 556, 500,
        ];
        let mut out = Self {
            file: io::BufWriter::new(fs::File::create(path)?),
            pos: 0,
            offsets: vec![0],
            font: 0,
        };
        out.put(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n")?;
        out.font = if embedded {
            let program = out
                .stream(&format!("/Length1 {}", FACE.len()), |w| {
                    w.write_all(FACE).map_err(publish_io)
                })
                .map_err(|e| io::Error::other(e.to_string()))?;
            let descriptor = out.object(&format!(
                "<< /Type /FontDescriptor /FontName /AAAAAA+LibviprsLatin-Regular /Flags 32 \
                 /FontBBox [-544 -303 1302 980] /ItalicAngle 0 /Ascent 905 /Descent -212 \
                 /CapHeight 688 /StemV 80 /FontFile2 {program} 0 R >>"
            ))?;
            let widths: Vec<String> = WIDTHS.iter().map(i64::to_string).collect();
            out.object(&format!(
                "<< /Type /Font /Subtype /TrueType /BaseFont /AAAAAA+LibviprsLatin-Regular \
                 /FirstChar 32 /LastChar 255 /Widths [{}] /FontDescriptor {descriptor} 0 R \
                 /Encoding /WinAnsiEncoding >>",
                widths.join(" ")
            ))?
        } else {
            out.object(
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            )?
        };
        Ok(out)
    }

    fn put(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.file.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    /// The next object number, with its start recorded as the current position.
    fn begin(&mut self) -> io::Result<u32> {
        self.offsets.push(self.pos);
        let id = self.offsets.len() as u32;
        self.put(format!("{id} 0 obj\n").as_bytes())?;
        Ok(id)
    }

    /// A whole object whose body is `body`.
    fn object(&mut self, body: &str) -> io::Result<u32> {
        let id = self.begin()?;
        self.put(format!("{body}\nendobj\n").as_bytes())?;
        Ok(id)
    }

    /// A Flate stream whose bytes `fill` writes straight into the file. Its length is the
    /// indirect object written right after it, which is how the stream is not held to be counted.
    fn stream(
        &mut self,
        entries: &str,
        fill: impl FnOnce(&mut dyn Write) -> Result<(), PvecConvertError>,
    ) -> Result<u32, PvecConvertError> {
        let id = self.begin().map_err(publish_io)?;
        // The length object is the next number, and is written once the stream is.
        let length = id + 1;
        self.put(
            format!("<< {entries} /Filter /FlateDecode /Length {length} 0 R >>\nstream\n")
                .as_bytes(),
        )
        .map_err(publish_io)?;
        let start = self.pos;
        let mut counted = Counted {
            out: &mut self.file,
            written: 0,
        };
        let mut z = ZlibEncoder::new(&mut counted, Compression::best());
        fill(&mut z)?;
        z.finish().map_err(publish_io)?;
        let len = counted.written;
        self.pos = start + len;
        self.put(b"\nendstream\nendobj\n").map_err(publish_io)?;
        let made = self.object(&len.to_string()).map_err(publish_io)?;
        debug_assert_eq!(made, length);
        Ok(id)
    }

    /// The page object of `page`, which names objects already written.
    fn page(&mut self, page: &PageRef) -> io::Result<u32> {
        // The box is the paper as it feeds: the displayed frame, turned back when the page rotates.
        let [w, h] = if page.rotate == 0 {
            page.size
        } else {
            [page.size[1], page.size[0]]
        };
        let font = if page.text {
            format!("/Font << /F1 {} 0 R >> ", self.font)
        } else {
            String::new()
        };
        let xobjects = if page.images.is_empty() {
            String::new()
        } else {
            let list: Vec<String> = page
                .images
                .iter()
                .enumerate()
                .map(|(n, id)| format!("/Im{n} {id} 0 R"))
                .collect();
            format!("/XObject << {} >> ", list.join(" "))
        };
        self.object(&format!(
            "<< /Type /Page /Parent {PAGES} 0 R /MediaBox [0 0 {} {}] /Rotate {} \
             /Resources << {font}{xobjects}>> /Contents {} 0 R >>",
            w as f32, h as f32, page.rotate, page.content
        ))
    }

    /// Writes the page tree, the catalog, the information dictionary, the cross reference table and
    /// the trailer, and hands the file back to be synced.
    fn finish(mut self, kids: &[u32]) -> io::Result<fs::File> {
        let list: Vec<String> = kids.iter().map(|k| format!("{k} 0 R")).collect();
        // The page tree is object 1: its slot was kept for it, and it is written last.
        self.offsets[0] = self.pos;
        self.put(
            format!(
                "{PAGES} 0 obj\n<< /Type /Pages /Kids [{}] /Count {} >>\nendobj\n",
                list.join(" "),
                kids.len()
            )
            .as_bytes(),
        )?;
        let catalog = self.object(&format!("<< /Type /Catalog /Pages {PAGES} 0 R >>"))?;
        let info = self.object(&format!(
            "<< /Producer (libviprs {}) >>",
            env!("CARGO_PKG_VERSION")
        ))?;
        let start = self.pos;
        let size = self.offsets.len() + 1;
        let mut table = format!("xref\n0 {size}\n0000000000 65535 f \n");
        for at in &self.offsets {
            table.push_str(&xref_entry(*at)?);
        }
        table.push_str(&format!(
            "trailer\n<< /Size {size} /Root {catalog} 0 R /Info {info} 0 R >>\nstartxref\n{start}\n%%EOF\n"
        ));
        self.put(table.as_bytes())?;
        self.file
            .into_inner()
            .map_err(io::IntoInnerError::into_error)
    }
}

/// One in-use entry of a classic cross reference table: ten digits of offset, so a file whose
/// objects start at 10 GB or more cannot be addressed and is refused rather than written wrong.
fn xref_entry(offset: u64) -> io::Result<String> {
    if offset >= 10_000_000_000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "an object starts at 10 GB or more, past what a classic xref table can address",
        ));
    }
    Ok(format!("{offset:010} 00000 n \n"))
}

/// Counts what goes through to the file, so a stream's compressed length is known without holding it.
struct Counted<'a> {
    out: &'a mut io::BufWriter<fs::File>,
    written: u64,
}

impl Write for Counted<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.out.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::xref_entry;

    #[test]
    fn an_xref_entry_is_twenty_bytes_and_refuses_what_ten_digits_cannot_hold() {
        assert_eq!(xref_entry(9_999_999_999).unwrap().len(), 20);
        assert_eq!(xref_entry(17).unwrap(), "0000000017 00000 n \n");
        let e = xref_entry(10_000_000_000).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
    }
}
