//! Plot to PDF (#1636): print layouts written as the pages of one PDF.
//!
//! Each layout goes through the same conversion a PVec sheet does (print profile, plot settings,
//! paper frame, rotation), into a scratch artifact. The page is that artifact's frame, which is
//! the paper, and its curves are drawn as stroked polylines, black, at the drawing's default
//! lineweight. No text is written (it waits for #1621) and no font is put on the page.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use acadrust::plot_style::PlotStyleTable;
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use pvec_components::{Fill, HatchStream, TextStream};
use pvec_facade::Curve;
use pvec_facade::components::{LineStyleStream, StrokePaintStream, StrokeWidth};
use pvec_geometry::{Curve2d, Side};

use super::{PlotSettings, PvecConversion, PvecConvertError};
use crate::CancelToken;
use crate::cad::CadSource;
use crate::observe::{EngineEvent, EngineObserver};
use crate::pvec::SourceLimits;
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
    cancel: CancelToken,
    observer: Option<Arc<dyn EngineObserver>>,
    deadline: Option<Duration>,
    limits: Option<SourceLimits>,
    texts: bool,
    fills: bool,
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

        let work = self.dest.with_extension("plot-work");
        fs::create_dir_all(&work)
            .map_err(|e| PvecConvertError::Publish(crate::pvec::publish::PublishError::Io(e)))?;
        let total = wanted.len() as u32;
        let tell = |event: EngineEvent| {
            if let Some(o) = &self.observer {
                o.on_event(event);
            }
        };
        let pages = wanted
            .iter()
            .enumerate()
            .map(|(n, name)| {
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
                let file = work.join(format!("{n}.pvec"));
                let mut conversion = PvecConversion::new(self.source, &file)
                    .view_profile(profile)
                    .plot_settings(self.settings.clone())
                    .texts(self.texts)
                    .fills(self.fills)
                    .components(StreamSelection::ALL)
                    .cancel(cancel.clone());
                if let Some(o) = &self.observer {
                    conversion = conversion.observer(o.clone());
                }
                conversion.empty_sheet_ok = true;
                if let Some(limits) = self.limits {
                    conversion = conversion.limits(limits);
                }
                if let Some(table) = &self.table {
                    conversion = conversion.plot_style_table(table.clone());
                }
                let page = conversion.run(&provider, unit).and_then(|out| {
                    let sheet = out.report.sheet;
                    let rotation = sheet
                        .as_ref()
                        .filter(|s| s.frame == "paper")
                        .and_then(|s| s.plot.as_ref())
                        .map_or(0, |p| p.rotation);
                    let page = if out.written == 0 {
                        blank_page(sheet.as_ref(), unit)
                    } else {
                        read_page(&file, rotation).map_err(PvecConvertError::Publish)?
                    };
                    Ok((page, sheet.map(|s| s.notes).unwrap_or_default()))
                });
                let status = match &page {
                    Ok(_) => ("written", None),
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
                page
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

/// One page: its box in points, the content stream that draws it, and the fonts it uses.
struct Page {
    /// The size the page is displayed at, in points: the sheet's frame.
    size: [f64; 2],
    /// `/Rotate`: 0, 90 or 270. A layout rotated a quarter turn is displayed turned.
    rotate: i64,
    content: String,
    text: bool,
}

/// A path segment in page points.
enum Seg {
    Move(f64, f64),
    Line(f64, f64),
    Curve([f64; 6]),
}

/// A stroke or a fill's outline as segments, each run starting with a [`Seg::Move`].
type Runs = Vec<Seg>;

/// Reads a scratch artifact into a page: the frame is the paper, the curves are the strokes.
fn read_page(artifact: &Path, rotation: i16) -> Result<Page, crate::pvec::publish::PublishError> {
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
    let texts = payload(TextStream::CAPABILITY).and_then(|s| TextStream::decode(&s.payload).ok());
    let hatches =
        payload(HatchStream::CAPABILITY).and_then(|s| HatchStream::decode(&s.payload).ok());
    let width_pt = |mm: f64| mm / 1000.0 * POINTS_PER_METRE;
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
    let mut content = String::new();

    // Fills first, under the strokes.
    for record in hatches.iter().flat_map(|h| &h.records) {
        let Fill::Solid { rgba } = &record.fill else {
            continue;
        };
        let mut runs: Runs = Vec::new();
        for id in &record.boundary {
            if let Some((_, curve)) = entities.iter().find(|(e, _)| e == id) {
                runs.extend(segments(curve, &to_pt));
            }
        }
        let word = record
            .boundary
            .first()
            .and_then(|id| paint.as_ref()?.colour_of(*id))
            .unwrap_or(*rgba);
        content.push_str(&format!("{} rg\n", rgb(word)));
        push_path(&mut content, &runs, true);
        content.push_str("f*\n");
    }

    for (id, curve) in &entities {
        if boundary.contains(id) {
            continue;
        }
        let def = style.as_ref().and_then(|s| {
            let at = s.assignments.iter().find(|(e, _)| e == id)?.1;
            s.styles.get(at as usize)
        });
        // Stroke widths are in millimetres, as a lineweight is.
        let mm = match def.and_then(|d| d.width) {
            Some(StrokeWidth::Width(w)) => w,
            Some(StrokeWidth::Thinnest) => 0.0,
            None => DEFAULT_WIDTH_MM,
        };
        let word = paint
            .as_ref()
            .and_then(|p| p.colour_of(*id))
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

    let mut text = false;
    for t in texts.iter().flat_map(|t| &t.records) {
        let word = paint
            .as_ref()
            .and_then(|p| p.colour_of(t.entity))
            .unwrap_or(0x0000_00ff);
        let (x, y) = to_pt(t.x, t.y);
        let (s, c) = t.rotation.sin_cos();
        content.push_str(&format!(
            "BT\n/F1 {:.4} Tf\n{} rg\n{c:.6} {s:.6} {:.6} {c:.6} {x:.4} {y:.4} Tm\n({}) Tj\nET\n",
            t.height * k,
            rgb(word),
            -s,
            pdf_string(&t.text)
        ));
        text = true;
    }

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
    if rotate != 0 {
        let (displayed_w, displayed_h) = (size[0], size[1]);
        let cm = if rotate == 270 {
            format!("0 -1 1 0 0 {displayed_w:.4} cm\n")
        } else {
            format!("0 1 -1 0 {displayed_h:.4} 0 cm\n")
        };
        content.insert_str(0, &cm);
    }
    Ok(Page {
        size,
        rotate,
        content,
        text,
    })
}

/// A page's content as a PDF literal string: Latin-1, with anything else as `?`.
fn pdf_string(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '(' | ')' | '\\' => format!("\\{c}"),
            ' '..='~' => c.to_string(),
            c if (c as u32) < 256 => format!("\\{:03o}", c as u32),
            _ => "?".to_owned(),
        })
        .collect()
}

fn push_path(content: &mut String, runs: &Runs, close: bool) {
    let mut open = false;
    for seg in runs {
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
        rotate: 0,
        content: String::new(),
        text: false,
    }
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

/// Writes the pages as one PDF. The Producer names libviprs.
fn write_pdf(dest: &Path, pages: impl Iterator<Item = Page>) -> std::io::Result<()> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let font = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
        "Encoding" => "WinAnsiEncoding",
    });
    let mut kids = Vec::new();
    for page in pages {
        let content = doc.add_object(Stream::new(Dictionary::new(), page.content.into_bytes()));
        // The box is the paper as it feeds: the displayed frame, turned back when the page rotates.
        let [w, h] = if page.rotate == 0 {
            page.size
        } else {
            [page.size[1], page.size[0]]
        };
        let resources = if page.text {
            dictionary! { "Font" => dictionary! { "F1" => font } }
        } else {
            Dictionary::new()
        };
        let id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), Object::Real(w as f32), Object::Real(h as f32)],
            "Rotate" => page.rotate,
            "Resources" => resources,
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
    // Flate on every stream, as a plotter's PDF has it.
    doc.compress();
    doc.save(dest).map(|_| ())
}
