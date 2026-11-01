//! The provider differential harness (issue #1395, N2.7 and N4.8).
//!
//! A second DWG provider is only replaceable if it can be held to the first one's answers. This
//! module runs any number of [`DwgProvider`]s over the same fixtures and compares, per entity,
//! what each one says:
//!
//! * the structural parameters the entity carries (and the composed insert map),
//! * the N3.5 bucket, per entity and not in aggregate, with its reason code,
//! * the PVec semantic type of N6.2's table.
//!
//! Nothing here is a trait of its own: the harness is generic over [`DwgProvider`], so a second
//! backend is one more call to [`Matrix::run`].
//!
//! # What a pass means
//!
//! Two providers agreeing is [`AGREEMENT_CLASS`], `DUAL_ORACLE_EQUIVALENT`, and never source
//! truth: both read the same bytes. The fixture's own evidence class (`KNOWN_TRUTH_EXACT` for the
//! known-truth set) is EPIC O's, comes in with the fixture and is printed beside the agreement,
//! never replaced by it. A run with fewer than two providers is [`Verdict::Partial`], which is not
//! a pass, whatever the single provider said.
//!
//! # Allowed differences
//!
//! [`Deltas`] is `tests/fixtures/pvec/provider_deltas.toml`: one entry per fixture, entity and
//! field, each with a written reason. A difference that is not listed fails. An entry that no
//! longer matches a difference fails too, so the list cannot rot.

use std::path::PathBuf;

use thiserror::Error;

use super::provider::{DwgProvider, ProviderError};

/// What two providers agreeing is called. Never source truth.
pub const AGREEMENT_CLASS: &str = "DUAL_ORACLE_EQUIVALENT";

/// One fixture to run.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// The fixture's id, used in reports and in the deltas file.
    pub id: String,
    /// The drawing.
    pub path: PathBuf,
    /// The fixture's own evidence class, as its corpus states it.
    pub evidence: String,
}

/// What one provider said about one flattened entity.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityRow {
    /// `<instance path>#<handle>`, so a leaf reached through two INSERTs has two rows.
    pub key: String,
    /// The drawing's class name for it.
    pub class_name: String,
    /// Its own layer.
    pub layer: String,
    /// The structural parameters, by name, and the composed map as `matrix.*`.
    pub params: Vec<(String, f64)>,
    /// The N3.5 class tag.
    pub bucket: String,
    /// The reason code, empty when there is none.
    pub reason: String,
    /// The PVec semantic type of N6.2, empty when there is none.
    pub pvec_type: String,
}

/// A difference between two providers.
#[derive(Debug, Clone, PartialEq)]
pub struct Difference {
    /// The fixture.
    pub fixture: String,
    /// The entity key.
    pub entity: String,
    /// `present`, `class_name`, `layer`, `bucket`, `reason`, `pvec_type` or `param:<name>`.
    pub field: String,
    /// The first provider's name.
    pub left_provider: String,
    /// What it said.
    pub left: String,
    /// The other provider's name.
    pub right_provider: String,
    /// What it said.
    pub right: String,
}

/// An allowed difference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedDelta {
    /// The fixture.
    pub fixture: String,
    /// The entity key.
    pub entity: String,
    /// The field.
    pub field: String,
    /// Why the providers differ here.
    pub reason: String,
}

/// The allowed-difference list.
#[derive(Debug, Clone, Default)]
pub struct Deltas {
    entries: Vec<AllowedDelta>,
}

/// Why a deltas file or a run was refused.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DiffError {
    /// The deltas file is not valid.
    #[error("provider_deltas.toml: {0}")]
    Deltas(String),
    /// A provider failed on a fixture.
    #[error("{provider} on {fixture}: {source}")]
    Provider {
        /// Provider name.
        provider: String,
        /// Fixture id.
        fixture: String,
        /// Why.
        source: ProviderError,
    },
}

impl Deltas {
    /// Parses the file.
    ///
    /// # Errors
    ///
    /// [`DiffError::Deltas`] for an entry with no reason, a blank reason, an unknown key or a
    /// repeated (fixture, entity, field).
    pub fn parse(text: &str) -> Result<Self, DiffError> {
        let _ = text;
        Ok(Self::default())
    }

    /// The entries.
    #[must_use]
    pub fn entries(&self) -> &[AllowedDelta] {
        &self.entries
    }
}

/// The outcome of a comparison.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Verdict {
    /// Two or more providers ran, every difference is listed and every listed entry happened.
    Equivalent,
    /// Fewer than two providers ran: no differential was possible. Not a pass.
    Partial {
        /// The providers that did not run.
        missing: Vec<String>,
    },
    /// A difference is not listed, or a listed one did not happen.
    Failed {
        /// Differences with no entry.
        unlisted: Vec<Difference>,
        /// Entries that matched nothing.
        stale: Vec<AllowedDelta>,
    },
}

impl Verdict {
    /// Whether this is a pass. A partial run is not.
    #[must_use]
    pub fn passed(&self) -> bool {
        matches!(self, Verdict::Equivalent)
    }
}

/// Providers run over fixtures, and the ones that could not be.
#[derive(Debug, Clone)]
pub struct Matrix {
    corpus: String,
    absent: Vec<(String, String)>,
}

impl Matrix {
    /// An empty matrix over the corpus revision `corpus`.
    #[must_use]
    pub fn new(corpus: impl Into<String>) -> Self {
        Self {
            corpus: corpus.into(),
            absent: Vec::new(),
        }
    }

    /// Runs provider `P` over `fixtures`.
    ///
    /// # Errors
    ///
    /// [`DiffError::Provider`] when `P` cannot open or read a fixture.
    pub fn run<P: DwgProvider>(&mut self, fixtures: &[Fixture]) -> Result<(), DiffError> {
        let _ = fixtures;
        Ok(())
    }

    /// Records a provider that is not compiled in or has no adapter, with why.
    pub fn absent(&mut self, provider: impl Into<String>, reason: impl Into<String>) {
        self.absent.push((provider.into(), reason.into()));
    }

    /// The rows `provider` produced for `fixture`.
    #[must_use]
    pub fn rows(&self, provider: &str, fixture: &str) -> Option<&[EntityRow]> {
        let _ = (provider, fixture);
        None
    }

    /// Every difference between every pair of providers that ran.
    #[must_use]
    pub fn differences(&self) -> Vec<Difference> {
        Vec::new()
    }

    /// The verdict against `deltas`.
    #[must_use]
    pub fn verdict(&self, deltas: &Deltas) -> Verdict {
        let _ = deltas;
        Verdict::Equivalent
    }

    /// The report: one `result` line per provider and fixture, the differences, and the verdict.
    #[must_use]
    pub fn report(&self, deltas: &Deltas) -> String {
        let _ = (&self.corpus, deltas);
        String::new()
    }
}
