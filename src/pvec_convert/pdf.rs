//! A PDF page into one `.pvec` with its vectors and its rasters (issue #1491).
//!
//! [`PdfConversion`] is what [`PvecConversion`](super::PvecConversion) is for a DWG: one call, the
//! artifact, the report and nothing half written. It runs the four public steps a caller used to
//! wire by hand:
//!
//! ```text
//! extract_page_paths_from_bytes -> normalize_page -> open_geometry      the vector geometry
//!                               -> bake_page_images -> rasters, finish   the raster blocks
//!                               -> report -> atomic publish              both files or neither
//! ```
//!
//! Coordinates are the page's own, in points, unscaled. PVec cannot declare points as a unit, so the
//! manifest says `metres_per_unit` 1.0 and a reader that wants metres multiplies by 0.0254 / 72. The
//! manifest box is the displayed page (after `/Rotate`) grown to take in any path that leaves it.
//!
//! The vectors go through the geometry funnel (`pvec::writer::open_geometry`),
//! which is the one writer for entities, and the rasters are added to the open artifact before it is
//! sealed. Entities get ids from 0 in content-stream order, the ones with a curve first. A writer
//! refusal is reported as `pvec-cannot-store` and moves the later PVec ids down by one. The artifact
//! needs one entity (PVec's WRT-01), so a page with no path at all is refused with
//! [`PdfConvertError::NoGeometry`].
//!
//! Needs the `pvec-convert` and `pvec-raster` features.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use crate::CancelToken;
use crate::cad::CadSource;
use crate::observe::{EngineEvent, EngineObserver};
use crate::pvec::fidelity::{Classification, FidelityReport, ReasonCode, Tally, TallyError};
use crate::pvec::fill_policy::FillReport;
use crate::pvec::mapping::{kind_of, row_for};
use crate::pvec::ordering::EntityId;
use crate::pvec::pdf_bezier::{PdfNormalizedEntity, normalize_page};
use crate::pvec::pdf_image::{PdfImageNote, PdfImageOptions, PdfImageReport, bake_page_images};
use crate::pvec::pdf_vector::{PdfVectorError, PdfVectorLimits, extract_page_paths_from_bytes};
use crate::pvec::publish::{PublishError, publish};
use crate::pvec::report::{
    BoundsSplit, ConversionReport, EntityRow, PdfSummary, ReportBuilder, ReportError, ViewSummary,
    XrefSummary,
};
use crate::pvec::units::SourceUnit;
use crate::pvec::writer::{WriteConfig, WriteError, open_geometry};
use crate::raster_pvec::container::RasterWriteError;

/// Why a PDF conversion failed. Every variant that wraps a lower error keeps it as `source()`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PdfConvertError {
    /// The PDF file could not be read.
    #[error("the PDF could not be read")]
    Read(#[source] std::io::Error),
    /// The page could not be extracted.
    #[error("the page could not be extracted")]
    Open(#[source] PdfVectorError),
    /// The page has no path, and PVec's artifact needs one entity.
    #[error("the page has no vector path to write, and a PVec artifact needs one entity")]
    NoGeometry,
    /// The geometry funnel refused the vector geometry or the artifact.
    #[error("the artifact could not be written")]
    Write(#[source] WriteError),
    /// The raster blocks could not be written.
    #[error("the raster blocks could not be written")]
    Rasters(#[source] RasterWriteError),
    /// The class counts disagree with the entities met.
    #[error("the class counts do not add up")]
    Tally(#[source] TallyError),
    /// The report could not be built.
    #[error("the report could not be built")]
    Report(#[source] ReportError),
    /// The artifact and report could not be published.
    #[error("the artifact could not be published")]
    Publish(#[source] PublishError),
    /// The cancel token fired. Nothing was published.
    #[error("cancelled")]
    Cancelled,
}

/// What a PDF conversion produced.
#[derive(Debug)]
pub struct PdfConversionOutput {
    /// The published artifact, readable with [`open_published`](crate::pvec::publish::open_published).
    pub artifact: PathBuf,
    /// The conversion report, also published beside the artifact. Its `pdf` section counts the images,
    /// the fills and the text.
    pub report: ConversionReport,
    /// Vector entities per fidelity class, counted independently of the report rows.
    pub fidelity: FidelityReport,
    /// Vector entities written into the artifact.
    pub written: u64,
    /// Raster blocks written into the artifact.
    pub rasters: u64,
    /// What the image bake did.
    pub images: PdfImageReport,
    /// The fill, shading, pattern and image constructs counted on the page.
    pub fills: FillReport,
}

/// One PDF page conversion, configured the way [`PvecConversion`](super::PvecConversion) is.
pub struct PdfConversion<'a> {
    source: CadSource<'a>,
    dest: &'a Path,
    page: usize,
    limits: PdfVectorLimits,
    images: PdfImageOptions,
    cancel: CancelToken,
    observer: Option<Arc<dyn EngineObserver>>,
}

impl<'a> PdfConversion<'a> {
    /// A conversion of page 1 of `source` into `dest`, under the default limits.
    #[must_use]
    pub fn new(source: CadSource<'a>, dest: &'a Path) -> Self {
        Self {
            source,
            dest,
            page: 1,
            limits: PdfVectorLimits::default(),
            images: PdfImageOptions::default(),
            cancel: CancelToken::new(),
            observer: None,
        }
    }

    /// The 1-based page to convert.
    #[must_use]
    pub fn page(mut self, page: usize) -> Self {
        self.page = page;
        self
    }

    /// The resource ceilings of the content-stream walk.
    #[must_use]
    pub fn limits(mut self, limits: PdfVectorLimits) -> Self {
        self.limits = limits;
        self
    }

    /// How the images are baked: tile format and edge, the page-to-source map and the pixel ceiling.
    #[must_use]
    pub fn image_options(mut self, options: PdfImageOptions) -> Self {
        self.images = options;
        self
    }

    /// The cancel token, polled before the page is read, between the steps and before anything is
    /// published. It is not polled inside the image bake, which runs to the end of the page once it
    /// starts, so a cancel during a heavy page is seen when the bake returns.
    #[must_use]
    pub fn cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Receives the run's events: the source opened, the artifact written and the publish.
    #[must_use]
    pub fn observer(mut self, observer: Arc<dyn EngineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Converts the page.
    ///
    /// # Errors
    ///
    /// [`PdfConvertError`]. After any error the destination and its report are as they were.
    pub fn convert(self) -> Result<PdfConversionOutput, PdfConvertError> {
        let tell = |event: EngineEvent| {
            if let Some(o) = &self.observer {
                o.on_event(event);
            }
        };
        let checked = || {
            if self.cancel.is_cancelled() {
                Err(PdfConvertError::Cancelled)
            } else {
                Ok(())
            }
        };
        checked()?;
        let bytes: Cow<'_, [u8]> = match self.source {
            CadSource::Path(p) => Cow::Owned(std::fs::read(p).map_err(PdfConvertError::Read)?),
            CadSource::Bytes(b) => Cow::Borrowed(b),
        };
        let page = extract_page_paths_from_bytes(&bytes, self.page, &self.limits)
            .map_err(PdfConvertError::Open)?;
        drop(bytes);
        tell(EngineEvent::ConversionSourceOpened {
            provider: "pdf".to_owned(),
        });
        checked()?;
        let bake = bake_page_images(&page, &self.images);
        checked()?;

        // Vector side: every entity gets the next id in content-stream order, those with a curve first
        // in the funnel's order and the rest after them, so a row's id is a PVec id only when the funnel
        // wrote it (and a writer refusal moves the ones after it down).
        let normalized = normalize_page(&page);
        let (mut stored, mut other) = (Vec::new(), Vec::new());
        for e in normalized.iter().flat_map(|p| &p.entities) {
            if e.geometry.geometry().is_some() {
                stored.push(e);
            } else {
                other.push(e);
            }
        }
        let config = WriteConfig {
            unit: SourceUnit::Meters,
            leaves: None,
        };
        let items = stored
            .iter()
            .enumerate()
            .map(|(n, e)| (EntityId::at(n as u64), &e.geometry));
        let area = [0.0, 0.0, page.width, page.height];
        let mut open = open_geometry(
            &config,
            items,
            Vec::new(),
            Some(area),
            &self.cancel,
            &mut |_, _| {},
        )
        .map_err(|e| match e {
            WriteError::NoGeometry | WriteError::AllRefused(_) => PdfConvertError::NoGeometry,
            WriteError::Cancelled => PdfConvertError::Cancelled,
            other => PdfConvertError::Write(other),
        })?;
        checked()?;
        let rasters = open
            .rasters(bake.written.iter().map(|b| &b.pyramid))
            .map_err(PdfConvertError::Rasters)? as u64;
        let artifact = open.finish().map_err(PdfConvertError::Write)?;
        tell(EngineEvent::ConversionArtifactWritten {
            entities: artifact.written,
            bytes: artifact.bytes.len() as u64,
        });

        let (mut rows, mut tally) = (ReportBuilder::new(), Tally::new());
        let row = |id: u64, e: &PdfNormalizedEntity| EntityRow {
            id,
            source_class: e.classification.source_class().to_owned(),
            source_handle: None,
            semantic_type: kind_of(&e.geometry).map(|k| row_for(k).pvec_type.to_owned()),
            fidelity: e.classification.class().tag().to_owned(),
            reason: e.classification.reason().map(|r| r.code().to_owned()),
            policy_field: e.classification.policy_field().map(str::to_owned),
            warnings: Vec::new(),
        };
        for (n, e) in stored.iter().chain(&other).enumerate() {
            tally.record(&e.classification);
            rows.push(row(n as u64, e));
        }
        // An entity the classifier called supported can still be one PVec's container refuses, and it is
        // counted as that.
        for (id, rule) in &artifact.refused {
            let from = &stored[id.get() as usize].classification;
            let to = Classification::refused(ReasonCode::PvecCannotStore, from.source_class());
            tally.reclassify(from, &to);
            rows.refuse(*id, &to, format!("PVec's writer refused it ({rule})"));
        }
        let written = artifact.written;
        let encountered = (stored.len() + other.len()) as u64;
        let fidelity = tally.finish(encountered).map_err(PdfConvertError::Tally)?;
        let mut fills = FillReport::new();
        fills.count_pdf_page(&page);
        let mut report = rows
            .finish(
                encountered,
                ViewSummary {
                    space: format!("page {}", self.page),
                    included: written,
                    ..ViewSummary::default()
                },
                XrefSummary::default(),
                BoundsSplit {
                    exact: artifact.bounds.exact_count() as u64,
                    hull_upper_bound: artifact.bounds.hull_count() as u64,
                },
            )
            .map_err(PdfConvertError::Report)?;
        let counted = |pairs: &[(PdfImageNote, u64)]| -> BTreeMap<String, u64> {
            pairs
                .iter()
                .filter(|(_, n)| *n > 0)
                .map(|(k, n)| (k.as_str().to_owned(), *n))
                .collect()
        };
        let r = &bake.report;
        report.pdf = Some(PdfSummary {
            page: self.page as u64,
            images: r.images,
            rasters,
            images_refused: r.refused.clone(),
            image_notes: counted(&[
                (PdfImageNote::ColourSpaceApproximated, r.colour_approximated),
                (PdfImageNote::BitDepthReduced, r.bit_depth_reduced),
                (PdfImageNote::ReEncodedLossy, r.re_encoded_lossy),
                (PdfImageNote::Interleaved, r.interleaved),
                (PdfImageNote::Clipped, r.clipped),
                (PdfImageNote::MaskIgnored, r.mask_ignored),
                (PdfImageNote::Bilevel, r.bilevel),
            ]),
            fills: fills
                .rows()
                .map(|f| (f.policy.reason.to_owned(), f.count))
                .collect(),
            text_omitted: page.text.omitted(),
        });
        let json = report.to_json_string().map_err(PdfConvertError::Report)?;
        publish(
            self.dest,
            &artifact.bytes,
            json.as_bytes(),
            &self.cancel,
            &mut |_| {},
        )
        .map_err(|e| match e {
            PublishError::Cancelled => PdfConvertError::Cancelled,
            other => PdfConvertError::Publish(other),
        })?;
        tell(EngineEvent::ConversionPublished);
        Ok(PdfConversionOutput {
            artifact: self.dest.to_path_buf(),
            report,
            fidelity,
            written,
            rasters,
            images: bake.report,
            fills,
        })
    }
}
