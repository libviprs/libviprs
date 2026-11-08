//! One `.pvec` per paper layout (issue #1564, slice 5 of #1557).
//!
//! [`PvecLayoutSet`] opens the drawing once, walks its paper layouts in tab order and converts each
//! one under [`ViewProfile::print_profile`] with the same `run` a [`PvecConversion`] uses, so a sheet
//! of the set and a sheet converted on its own are the same bytes. Each sheet is published atomically
//! at `<stem>.<NN>-<slug>.pvec` with its report beside it; the set is not atomic, and
//! `<stem>.sheets.json` is written last, so its presence says the run finished.
//!
//! `NN` is the layout's 1-based position among all the drawing's paper layouts, zero padded to the
//! width of their count and at least two digits, so a directory listing is the sheet order and two
//! files of one set never collide. The slug is the layout name with every byte outside
//! `[A-Za-z0-9._-]` turned into `-`, runs of `-` collapsed, leading and trailing `-` and `.` trimmed,
//! at most 64 bytes, and `layout` when nothing is left.
//!
//! A layout with no paper-space entity other than its viewports, and no viewport that is on, plotted
//! and has a view, is [`SheetSkip::Empty`]: listed, nothing written. One whose
//! kept content writes nothing is [`SheetSkip::EmptyAfterView`]; until the print profile says which
//! viewport is the paper view itself, a layout holding only that one comes out this way. Either way a
//! file an earlier run left at that sheet's path is removed. Any other failure of one sheet is
//! [`SheetSkip::Failed`], its path is left as it was, and the others go on. A drawing with no paper
//! layout writes nothing at all and does not fall back to model space: [`PvecConversion::convert`] is
//! the call for that. An index an earlier run left is removed before the first sheet, so a run that
//! stops part way never leaves one that says it finished.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use thiserror::Error;

use super::{PvecConversion, PvecConversionOutput, PvecConvertError, report_failure};
use crate::cad::CadSource;
use crate::pvec::dwg_acadrust::AcadrustProvider;
use crate::pvec::publish::{PublishError, report_path};
use crate::pvec::report::SheetSummary;
use crate::pvec::writer::WriteError;
use crate::pvec::{
    DwgProvider, EntityScope, LayoutKind, LayoutState, ProviderError, SourceUnit, SpaceSelection,
    ViewProfile,
};

/// Why a whole set failed. One sheet failing is not this: it is a [`SheetSkip::Failed`] row.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LayoutSetError {
    /// The destination is not an existing directory.
    #[error("{0} is not a directory")]
    NotADirectory(PathBuf),
    /// The source is bytes and no [`stem`](PvecLayoutSet::stem) was given, or the stem is empty or
    /// not a plain file name.
    #[error("the sheets need a file stem: give one, as a plain file name, with `stem`")]
    StemRequired,
    /// The drawing could not be opened.
    #[error("the source could not be opened")]
    Open(#[source] ProviderError),
    /// The drawing's layouts could not be listed.
    #[error("the drawing's layouts could not be listed")]
    Layouts(#[source] ProviderError),
    /// A layout named in [`layouts`](PvecLayoutSet::layouts) is not one of the drawing's paper layouts.
    #[error("paper layout {requested:?} not found, the paper layouts that exist are {available:?}")]
    LayoutNotFound {
        /// Asked for.
        requested: String,
        /// The paper layouts that exist, in tab order.
        available: Vec<String>,
    },
    /// The set index could not be written. The sheets before it were published.
    #[error("the set index could not be written")]
    Index(#[source] io::Error),
    /// The cancel token fired. Sheets published before it stay; no index is written.
    #[error("cancelled")]
    Cancelled,
}

/// Why a sheet has no file.
#[derive(Debug)]
#[non_exhaustive]
pub enum SheetSkip {
    /// The layout has nothing on it to print.
    Empty,
    /// The layout has content, and none of it is drawn under the print profile.
    EmptyAfterView,
    /// The sheet's conversion failed.
    Failed(PvecConvertError),
}

/// One paper layout of the set.
#[derive(Debug)]
pub struct SheetOutcome {
    /// The layout's own name.
    pub layout: String,
    /// Its 1-based position among the drawing's paper layouts, in tab order.
    pub tab_order: u32,
    /// Where its artifact is, or would have been.
    pub file: PathBuf,
    /// The conversion, or why there is no file.
    pub outcome: Result<PvecConversionOutput, SheetSkip>,
}

/// What a set produced.
#[derive(Debug)]
pub struct LayoutSetOutput {
    /// One row per layout converted or skipped, in tab order.
    pub sheets: Vec<SheetOutcome>,
    /// The set index, or `None` when the drawing has no paper layout and nothing was written.
    pub index: Option<PathBuf>,
}

impl LayoutSetOutput {
    /// The sheets that were written.
    pub fn written(&self) -> impl Iterator<Item = &SheetOutcome> {
        self.sheets.iter().filter(|s| s.outcome.is_ok())
    }

    /// The sheets whose conversion failed. An empty sheet is not a failure.
    pub fn failed(&self) -> impl Iterator<Item = &SheetOutcome> {
        self.sheets
            .iter()
            .filter(|s| matches!(s.outcome, Err(SheetSkip::Failed(_))))
    }
}

/// Every paper layout of a drawing, one `.pvec` each.
pub struct PvecLayoutSet<'a> {
    source: CadSource<'a>,
    dest_dir: &'a Path,
    stem: Option<String>,
    select: Option<Vec<String>>,
    #[allow(clippy::type_complexity)]
    sheet: Box<dyn for<'s> Fn(PvecConversion<'s>) -> PvecConversion<'s> + 'a>,
}

impl<'a> PvecLayoutSet<'a> {
    /// The paper layouts of `source`, into the existing directory `dest_dir`.
    #[must_use]
    pub fn new(source: CadSource<'a>, dest_dir: &'a Path) -> Self {
        Self {
            source,
            dest_dir,
            stem: None,
            select: None,
            sheet: Box::new(|c| c),
        }
    }

    /// Only these layouts, matched the way a [`SpaceSelection::PaperLayout`] is (ASCII case
    /// ignored). They keep the number they have in the whole set and come out in tab order.
    #[must_use]
    pub fn layouts<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        self.select = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// The file stem of every sheet. Required for [`CadSource::Bytes`]; a path's own stem otherwise.
    #[must_use]
    pub fn stem(mut self, stem: impl Into<String>) -> Self {
        self.stem = Some(stem.into());
        self
    }

    /// The options of each sheet's conversion, set with [`PvecConversion`]'s own builder: unit,
    /// limits, components, calibration, cancel token, observer and the rest. It runs once per
    /// sheet on a conversion under that layout's print profile; the profile's space is the layout
    /// whatever it sets.
    #[must_use]
    pub fn sheets(
        mut self,
        options: impl for<'s> Fn(PvecConversion<'s>) -> PvecConversion<'s> + 'a,
    ) -> Self {
        self.sheet = Box::new(options);
        self
    }

    /// Converts with the default provider, acadrust. Each sheet's unit is the header's unless the
    /// sheet options set one.
    ///
    /// # Errors
    ///
    /// [`LayoutSetError`]: only for what stops the whole set.
    pub fn convert(self) -> Result<LayoutSetOutput, LayoutSetError> {
        self.check().and_then(|stem| {
            let provider = AcadrustProvider::open(self.source).map_err(LayoutSetError::Open)?;
            let unit = provider.source_unit();
            self.run(&provider, Some(unit), &stem)
        })
    }

    /// Converts with provider `P`. The trait carries no unit, so the sheet options must set one,
    /// or every sheet fails with [`PvecConvertError::UnitRequired`].
    ///
    /// # Errors
    ///
    /// As [`convert`](Self::convert).
    pub fn convert_with<P: DwgProvider>(self) -> Result<LayoutSetOutput, LayoutSetError> {
        self.check().and_then(|stem| {
            let provider = P::open(self.source).map_err(LayoutSetError::Open)?;
            self.run(&provider, None, &stem)
        })
    }

    /// The destination and the stem, both before the drawing is opened.
    fn check(&self) -> Result<String, LayoutSetError> {
        if !self.dest_dir.is_dir() {
            return Err(LayoutSetError::NotADirectory(self.dest_dir.to_path_buf()));
        }
        let stem = self.stem.clone().or_else(|| match self.source {
            CadSource::Path(p) => p.file_stem().map(|s| s.to_string_lossy().into_owned()),
            CadSource::Bytes(_) => None,
        });
        stem.filter(|s| !s.is_empty() && !s.contains(['/', '\\']) && s != "." && s != "..")
            .ok_or(LayoutSetError::StemRequired)
    }

    fn run<P: DwgProvider>(
        self,
        provider: &P,
        header_unit: Option<SourceUnit>,
        stem: &str,
    ) -> Result<LayoutSetOutput, LayoutSetError> {
        /// An error and its sources, for the index.
        fn chain(e: &dyn std::error::Error) -> String {
            let mut out = e.to_string();
            let mut at = e.source();
            while let Some(s) = at {
                out.push_str(": ");
                out.push_str(&s.to_string());
                at = s.source();
            }
            out
        }
        /// Removes `path`; one that is not there is already gone.
        fn remove(path: &Path) -> io::Result<()> {
            match fs::remove_file(path) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
                _ => Ok(()),
            }
        }

        let paper: Vec<LayoutState> = provider
            .layouts()
            .map_err(LayoutSetError::Layouts)?
            .into_iter()
            .filter(|l| l.kind == LayoutKind::Paper)
            .collect();
        let mut wanted = vec![self.select.is_none(); paper.len()];
        for name in self.select.iter().flatten() {
            let at = paper
                .iter()
                .position(|l| l.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| LayoutSetError::LayoutNotFound {
                    requested: name.clone(),
                    available: paper.iter().map(|l| l.name.clone()).collect(),
                })?;
            wanted[at] = true;
        }
        // An index from an earlier run would say this one finished, so it goes before any sheet.
        let index = self.dest_dir.join(format!("{stem}.sheets.json"));
        remove(&index).map_err(LayoutSetError::Index)?;
        if paper.is_empty() {
            return Ok(LayoutSetOutput {
                sheets: Vec::new(),
                index: None,
            });
        }
        let width = paper.len().to_string().len().max(2);

        // A layout is a sheet (#1557, item 6) when it has an entity of its own other than a
        // viewport, or a viewport that is on, not hidden when plotting and has a view height. One
        // on a no-plot layer counts: it prints what it shows and only its border is hidden (P-3,
        // #1577). Which viewport is the paper view itself is the print profile's call (#1560) and
        // not made yet, so a layout holding only its paper view is a sheet that comes out
        // `EmptyAfterView`, never one that is called empty when it shows the model. A provider
        // that cannot list viewports has none to show.
        let viewports = provider.viewports().or_else(|e| match e {
            ProviderError::CannotReport(_) => Ok(Vec::new()),
            e => Err(e),
        });
        let has_content =
            |layout: &LayoutState| -> Result<bool, ProviderError> {
                let mut cursor = provider.entities(&EntityScope::Layout(layout.id))?;
                while let Some(e) = cursor.next_entity()? {
                    if e.provenance.class_name != "VIEWPORT" {
                        return Ok(true);
                    }
                }
                Ok(viewports.as_ref().map_err(Clone::clone)?.iter().any(|v| {
                    v.layout == layout.id && v.visible && !v.hide_plot && v.view_height > 0.0
                }))
            };
        // The layout name as a file name part: every byte outside `[A-Za-z0-9._-]` becomes `-`, runs
        // of `-` collapse, `-` and `.` are trimmed from both ends, at most 64 bytes, `layout` when
        // nothing is left.
        let slug = |name: &str| -> String {
            let mut s = String::with_capacity(name.len());
            for b in name.bytes() {
                let b = if b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_') {
                    b
                } else {
                    b'-'
                };
                if !(b == b'-' && s.ends_with('-')) {
                    s.push(char::from(b));
                }
            }
            let trim = |s: &str| s.trim_matches(['-', '.']).to_owned();
            let mut s = trim(&s);
            s.truncate(64);
            let s = trim(&s);
            if s.is_empty() { "layout".to_owned() } else { s }
        };

        let mut sheets = Vec::new();
        let mut rows = Vec::new();
        for (n, layout) in paper.iter().enumerate().filter(|(n, _)| wanted[*n]) {
            let nn = format!("{:0width$}", n + 1);
            let name = format!("{stem}.{nn}-{}.pvec", slug(&layout.name));
            let file = self.dest_dir.join(&name);
            let mut c = (self.sheet)(
                PvecConversion::new(self.source, &file)
                    .view_profile(ViewProfile::print_profile(layout.name.as_str())),
            );
            c.view_profile.space = SpaceSelection::PaperLayout(layout.name.clone());
            if c.cancel.is_cancelled() {
                return Err(LayoutSetError::Cancelled);
            }
            let observer = c.observer.clone();
            let result = match has_content(layout) {
                Ok(false) => Ok(None),
                Err(e) => Err(PvecConvertError::View(e.into())),
                Ok(true) => match c.unit.or(header_unit) {
                    Some(unit) => c.run(provider, unit).map(Some),
                    None => Err(PvecConvertError::UnitRequired),
                },
            };
            let mut outcome = match report_failure(observer.as_deref(), result) {
                Ok(Some(out)) => Ok(out),
                Ok(None) => Err(SheetSkip::Empty),
                Err(PvecConvertError::Cancelled) => return Err(LayoutSetError::Cancelled),
                Err(
                    PvecConvertError::NothingStorable { .. }
                    | PvecConvertError::Write(WriteError::NoGeometry),
                ) => Err(SheetSkip::EmptyAfterView),
                Err(e) => Err(SheetSkip::Failed(e)),
            };
            // A sheet with nothing to print has no file, so one an earlier run left there goes. A
            // failed sheet leaves its path as it was, the way a failed `PvecConversion` does.
            if matches!(outcome, Err(SheetSkip::Empty | SheetSkip::EmptyAfterView))
                && let Err(e) = remove(&file).and_then(|()| remove(&report_path(&file)))
            {
                outcome = Err(SheetSkip::Failed(PvecConvertError::Publish(
                    PublishError::Io(e),
                )));
            }
            let (status, reason, written) = match &outcome {
                Ok(o) => ("written", None, Some(o.written)),
                Err(SheetSkip::Empty) => ("empty", None, None),
                Err(SheetSkip::EmptyAfterView) => ("empty-after-view", None, None),
                Err(SheetSkip::Failed(e)) => ("failed", Some(chain(e)), None),
            };
            let sheet = outcome.as_ref().ok().and_then(|o| o.report.sheet.clone());
            rows.push(IndexRow {
                layout: layout.name.clone(),
                tab_order: n as u32 + 1,
                nn,
                file: name,
                status,
                reason,
                written,
                sheet,
            });
            sheets.push(SheetOutcome {
                layout: layout.name.clone(),
                tab_order: n as u32 + 1,
                file,
                outcome,
            });
        }

        let partial = self.dest_dir.join(format!("{stem}.sheets.json.partial"));
        let body = Index {
            schema_version: 1,
            source: match self.source {
                CadSource::Path(p) => p.display().to_string(),
                CadSource::Bytes(_) => "(bytes)".to_owned(),
            },
            stem,
            paper_layouts: paper.len() as u32,
            sheets: rows,
        };
        let written = File::create(&partial)
            .and_then(|mut f| {
                serde_json::to_writer_pretty(&mut f, &body).map_err(io::Error::other)?;
                f.sync_all()
            })
            .and_then(|()| fs::rename(&partial, &index));
        if let Err(e) = written {
            let _ = fs::remove_file(&partial);
            return Err(LayoutSetError::Index(e));
        }
        Ok(LayoutSetOutput {
            sheets,
            index: Some(index),
        })
    }
}

/// `<stem>.sheets.json`.
#[derive(Serialize)]
struct Index<'a> {
    schema_version: u32,
    source: String,
    stem: &'a str,
    paper_layouts: u32,
    sheets: Vec<IndexRow>,
}

#[derive(Serialize)]
struct IndexRow {
    layout: String,
    tab_order: u32,
    nn: String,
    file: String,
    status: &'static str,
    reason: Option<String>,
    written: Option<u64>,
    /// The report's `sheet` section: paper, unit, rotation, plot scale, CTB name. Absent for a
    /// sheet with no file, which has no report to take it from.
    sheet: Option<SheetSummary>,
}
