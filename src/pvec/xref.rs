//! The xref and external-reference policy (issue #1076, N4.5).
//!
//! STUB: red tests first, bodies follow.

use std::path::{Path, PathBuf};

use thiserror::Error;

use super::limits::{LimitError, SourceGuard};
use super::provider::{ArrayGrid, SourceTransform};

/// What to do with an attached drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum XrefPolicy {
    /// Open the attached drawing, counting its depth against the shared budget.
    Resolve,
    /// Fail the conversion with a typed error.
    Reject,
    /// Omit the geometry and record the reference.
    #[default]
    ExternalDependency,
}

/// An xref attachment, as the host drawing states it.
#[derive(Debug, Clone)]
pub struct XrefRef {
    /// The xref's name.
    pub name: String,
    /// The attachment path, verbatim.
    pub path: String,
    /// The insert transform.
    pub transform: SourceTransform,
    /// The layer the attachment sits on.
    pub layer: String,
    /// The grid the attachment is placed on.
    pub grid: ArrayGrid,
}

/// Why an attachment path was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PathRefusal {
    /// Empty, or contains a NUL.
    Malformed,
    /// Absolute path.
    Absolute,
    /// UNC path.
    Unc,
    /// Windows drive letter.
    DriveLetter,
    /// Climbs out of the root.
    EscapesRoot,
}

/// A refused xref.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum XrefError {
    /// Policy is `Reject`.
    #[error("xref {name} rejected by policy (resolved path {})", .path.display())]
    Rejected {
        /// Xref name.
        name: String,
        /// Resolved path.
        path: PathBuf,
    },
    /// The attachment path was refused.
    #[error("xref {name}: attachment path {path:?} refused: {reason:?}")]
    PathRefused {
        /// Xref name.
        name: String,
        /// The path as attached.
        path: String,
        /// Why.
        reason: PathRefusal,
    },
    /// Two or more files reach each other.
    #[error("xref cycle: {}", .chain.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" -> "))]
    Cycle {
        /// First occurrence to the repeat, inclusive.
        chain: Vec<PathBuf>,
    },
    /// The resolved file does not exist.
    #[error("xref {name}: no file at {}", .path.display())]
    Missing {
        /// Xref name.
        name: String,
        /// Resolved path.
        path: PathBuf,
    },
    /// A shared budget was passed.
    #[error(transparent)]
    Limit(#[from] LimitError),
}

/// A recorded external dependency.
#[derive(Debug, Clone)]
pub struct UnresolvedXref {
    /// Xref name.
    pub name: String,
    /// The path as attached.
    pub path: String,
    /// The path under the root, when the attachment was acceptable.
    pub resolved_path: Option<PathBuf>,
    /// Why the path was not acceptable, when it was not.
    pub path_refusal: Option<PathRefusal>,
    /// Insert transform.
    pub transform: SourceTransform,
    /// Layer.
    pub layer: String,
}

/// An xref cleared to be opened by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedXref {
    /// Xref name.
    pub name: String,
    /// The path to open.
    pub path: PathBuf,
    /// Xref nesting depth, 1 for a direct attachment.
    pub depth: u32,
}

/// What `admit` decided.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum XrefOutcome {
    /// Omitted and recorded at this index of the report.
    Omitted(usize),
    /// Open it.
    Resolve(ResolvedXref),
}

/// The xref rows of a conversion.
#[derive(Debug, Clone, Default)]
pub struct XrefReport {
    /// Omitted references.
    pub unresolved: Vec<UnresolvedXref>,
    /// Names of the xrefs that were resolved.
    pub resolved: Vec<String>,
}

/// Applies a policy across one conversion.
#[derive(Debug)]
pub struct XrefSession {
    _root: PathBuf,
    _policy: XrefPolicy,
    report: XrefReport,
}

/// Resolve `attachment` under `root` without touching the filesystem.
///
/// # Errors
///
/// A [`PathRefusal`].
pub fn confine(_root: &Path, _attachment: &str) -> Result<PathBuf, PathRefusal> {
    unimplemented!("N4.5 stub")
}

impl XrefSession {
    /// A session for the host drawing at `host` (relative to `root`).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, host: impl AsRef<Path>, policy: XrefPolicy) -> Self {
        let _ = host;
        Self {
            _root: root.into(),
            _policy: policy,
            report: XrefReport::default(),
        }
    }

    /// Decide one attachment found in the drawing currently open.
    ///
    /// # Errors
    ///
    /// [`XrefError`].
    pub fn admit(
        &mut self,
        _guard: &mut SourceGuard,
        _xref: &XrefRef,
    ) -> Result<XrefOutcome, XrefError> {
        unimplemented!("N4.5 stub")
    }

    /// Leave the nested drawing entered last.
    pub fn release(&mut self, _guard: &mut SourceGuard) {
        unimplemented!("N4.5 stub")
    }

    /// The rows so far.
    #[must_use]
    pub fn report(&self) -> &XrefReport {
        &self.report
    }

    /// The rows, consuming the session.
    #[must_use]
    pub fn into_report(self) -> XrefReport {
        self.report
    }
}
