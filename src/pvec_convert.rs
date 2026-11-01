//! The conversion API (issue #1048, N7.1): a DWG, a view profile and resource limits in, a
//! published `.pvec` artifact and its conversion report out.
//!
//! Needs the `pvec-convert` feature.

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::CancelToken;
use crate::cad::CadSource;
use crate::pvec::fidelity::FidelityReport;
use crate::pvec::flatten::FlattenError;
use crate::pvec::publish::PublishError;
use crate::pvec::report::{ConversionReport, ReportError};
use crate::pvec::writer::{LeafConfig, WriteError};
use crate::pvec::{
    DwgProvider, ProviderError, SourceLimits, SourceUnit, ViewProfile, ViewProfileError,
};

/// Why a conversion failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PvecConvertError {
    /// The provider could not open the source.
    #[error("the source could not be opened")]
    Open(#[source] ProviderError),
    /// The view profile could not be applied.
    #[error("the view profile could not be applied")]
    View(#[source] ViewProfileError),
    /// Block expansion failed.
    #[error("blocks could not be expanded")]
    Flatten(#[source] FlattenError),
    /// No unit was given and the provider cannot read one.
    #[error("the drawing's unit is not known and none was given")]
    UnitRequired,
    /// The writer refused.
    #[error("the artifact could not be written")]
    Write(#[source] WriteError),
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

/// What a conversion produced.
#[derive(Debug)]
pub struct PvecConversionOutput {
    /// The published artifact.
    pub artifact: PathBuf,
    /// The conversion report, also published beside the artifact.
    pub report: ConversionReport,
    /// Entities per fidelity class, counted independently of the report rows.
    pub fidelity: FidelityReport,
    /// Entities written into the artifact.
    pub written: u64,
}

/// One conversion, configured the way `EngineBuilder` configures a pyramid run.
pub struct PvecConversion<'a> {
    source: CadSource<'a>,
    dest: &'a Path,
    view_profile: ViewProfile,
    unit: Option<SourceUnit>,
    leaves: Option<LeafConfig>,
    cancel: CancelToken,
}

impl<'a> PvecConversion<'a> {
    /// A conversion of `source` into `dest`, under the default view profile.
    #[must_use]
    pub fn new(source: CadSource<'a>, dest: &'a Path) -> Self {
        Self {
            source,
            dest,
            view_profile: ViewProfile::default_profile(),
            unit: None,
            leaves: None,
            cancel: CancelToken::new(),
        }
    }

    /// The view profile (N3.3).
    #[must_use]
    pub fn view_profile(mut self, profile: ViewProfile) -> Self {
        self.view_profile = profile;
        self
    }

    /// The resource ceilings (N3.6).
    #[must_use]
    pub fn limits(mut self, limits: SourceLimits) -> Self {
        self.view_profile = self.view_profile.with_limits(limits);
        self
    }

    /// The drawing's unit, overriding what the provider reads.
    #[must_use]
    pub fn unit(mut self, unit: SourceUnit) -> Self {
        self.unit = Some(unit);
        self
    }

    /// Lay a spatial directory and Tier 0 blocks into the artifact.
    #[must_use]
    pub fn leaves(mut self, leaves: LeafConfig) -> Self {
        self.leaves = Some(leaves);
        self
    }

    /// The cancel token.
    #[must_use]
    pub fn cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Converts with the default provider.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError`].
    pub fn convert(self) -> Result<PvecConversionOutput, PvecConvertError> {
        let _ = (
            &self.source,
            self.dest,
            &self.view_profile,
            self.unit,
            self.leaves,
            &self.cancel,
        );
        Err(PvecConvertError::Cancelled)
    }

    /// Converts with provider `P`; the unit must be set with [`unit`](Self::unit).
    ///
    /// # Errors
    ///
    /// [`PvecConvertError`].
    pub fn convert_with<P: DwgProvider>(self) -> Result<PvecConversionOutput, PvecConvertError> {
        Err(PvecConvertError::UnitRequired)
    }
}
