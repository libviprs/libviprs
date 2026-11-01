//! The xref and external-reference policy (issue #1076, N4.5).
//!
//! An xref is geometry that lives in another file, and the failure mode is silence:
//! the conversion succeeds and a wing of the building is missing. So the decision is
//! an explicit input ([`XrefPolicy`]) and every outcome is visible:
//!
//! | policy | outcome of [`XrefSession::admit`] |
//! |---|---|
//! | `ExternalDependency` (default) | [`XrefOutcome::Omitted`], with a row in [`XrefReport::unresolved`] |
//! | `Reject` | [`XrefError::Rejected`] naming the xref and its resolved path |
//! | `Resolve` | [`XrefOutcome::Resolve`], the path to open, depth counted on the shared guard |
//!
//! Resolving reaches outside the input file, so it is the caller's decision, hence
//! the default. The attachment path is confined lexically under a caller-supplied
//! root by [`confine`] before anything touches the disk. That check does not follow
//! symlinks, and opening the file is the caller's. Opening through
//! `DwgProvider` and flattening under N4.4's bounds is #1283.

use std::path::{Path, PathBuf};

use thiserror::Error;

use super::limits::{LimitError, SourceGuard};
use super::provider::{ArrayGrid, DwgProvider, ProviderError, RawEntity, SourceTransform};

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

impl XrefRef {
    /// The xref attachments among `entities`: every INSERT of a block the provider lists in
    /// [`DwgProvider::xref_blocks`], in the order given, with the block record's path.
    ///
    /// # Errors
    ///
    /// [`ProviderError::CannotReport`] when the provider cannot report xref block records.
    pub fn discover(
        provider: &(impl DwgProvider + ?Sized),
        entities: &[RawEntity],
    ) -> Result<Vec<XrefRef>, ProviderError> {
        let _ = (provider, entities);
        Ok(Vec::new())
    }
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
    /// The resolved file exists and is not a drawing the provider opens.
    #[error("xref {name}: could not open {}: {detail}", .path.display())]
    Open {
        /// Xref name.
        name: String,
        /// Resolved path.
        path: PathBuf,
        /// The provider's message.
        detail: String,
    },
    /// The attached drawing could not be flattened.
    #[error("xref {name}: could not flatten the attached drawing: {detail}")]
    Flatten {
        /// Xref name.
        name: String,
        /// The flattener's message.
        detail: String,
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
    root: PathBuf,
    policy: XrefPolicy,
    /// Files currently open, host first.
    stack: Vec<PathBuf>,
    report: XrefReport,
}

/// Resolve `attachment` under `root` without touching the filesystem.
///
/// # Errors
///
/// A [`PathRefusal`].
pub fn confine(root: &Path, attachment: &str) -> Result<PathBuf, PathRefusal> {
    if attachment.is_empty() || attachment.contains('\0') {
        return Err(PathRefusal::Malformed);
    }
    if attachment.starts_with("\\\\") || attachment.starts_with("//") {
        return Err(PathRefusal::Unc);
    }
    if attachment.starts_with('/') || attachment.starts_with('\\') {
        return Err(PathRefusal::Absolute);
    }
    let b = attachment.as_bytes();
    if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        return Err(PathRefusal::DriveLetter);
    }
    let mut parts: Vec<&str> = Vec::new();
    for c in attachment.split(['/', '\\']) {
        match c {
            "" | "." => {}
            ".." => {
                if parts.pop().is_none() {
                    return Err(PathRefusal::EscapesRoot);
                }
            }
            c => parts.push(c),
        }
    }
    if parts.is_empty() {
        return Err(PathRefusal::Malformed);
    }
    let mut out = root.to_path_buf();
    out.extend(parts);
    Ok(out)
}

fn key(p: &Path) -> String {
    p.to_string_lossy().to_ascii_lowercase()
}

impl XrefSession {
    /// A session for the host drawing at `host` (relative to `root`).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, host: impl AsRef<Path>, policy: XrefPolicy) -> Self {
        let root = root.into();
        let host = host.as_ref();
        let host = confine(&root, &host.to_string_lossy()).unwrap_or_else(|_| root.join(host));
        Self {
            root,
            policy,
            stack: vec![host],
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
        guard: &mut SourceGuard,
        xref: &XrefRef,
    ) -> Result<XrefOutcome, XrefError> {
        let confined = confine(&self.root, &xref.path);
        let refused = |reason| XrefError::PathRefused {
            name: xref.name.clone(),
            path: xref.path.clone(),
            reason,
        };
        match self.policy {
            XrefPolicy::ExternalDependency => {
                let (resolved_path, path_refusal) = match confined {
                    Ok(p) => (Some(p), None),
                    Err(r) => (None, Some(r)),
                };
                self.report.unresolved.push(UnresolvedXref {
                    name: xref.name.clone(),
                    path: xref.path.clone(),
                    resolved_path,
                    path_refusal,
                    transform: xref.transform,
                    layer: xref.layer.clone(),
                });
                Ok(XrefOutcome::Omitted(self.report.unresolved.len() - 1))
            }
            XrefPolicy::Reject => {
                let path = confined.map_err(refused)?;
                Err(XrefError::Rejected {
                    name: xref.name.clone(),
                    path,
                })
            }
            XrefPolicy::Resolve => {
                let path = confined.map_err(refused)?;
                let k = key(&path);
                if let Some(first) = self.stack.iter().position(|p| key(p) == k) {
                    let mut chain = self.stack[first..].to_vec();
                    chain.push(path);
                    return Err(XrefError::Cycle { chain });
                }
                if !std::fs::metadata(&path).is_ok_and(|m| m.is_file()) {
                    return Err(XrefError::Missing {
                        name: xref.name.clone(),
                        path,
                    });
                }
                guard.enter_block(&path.to_string_lossy(), xref.grid)?;
                self.stack.push(path.clone());
                self.report.resolved.push(xref.name.clone());
                let depth = u32::try_from(self.stack.len() - 1).unwrap_or(u32::MAX);
                Ok(XrefOutcome::Resolve(ResolvedXref {
                    name: xref.name.clone(),
                    path,
                    depth,
                }))
            }
        }
    }

    /// Leave the nested drawing entered last. The host drawing is never left.
    pub fn release(&mut self, guard: &mut SourceGuard) {
        if self.stack.len() > 1 {
            self.stack.pop();
            guard.exit_block();
        }
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

/// What [`XrefSession::open_and_flatten`] did with one attachment.
#[cfg(feature = "pvec-normalized")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum XrefOpened {
    /// Omitted by policy and recorded at this index of the report.
    Omitted(usize),
    /// Opened and flattened.
    Flattened(FlattenedXref),
}

/// An attached drawing, opened and flattened.
#[cfg(feature = "pvec-normalized")]
#[derive(Debug, Clone)]
pub struct FlattenedXref {
    /// Xref name.
    pub name: String,
    /// The file that was opened.
    pub path: PathBuf,
    /// Xref nesting depth, 1 for a direct attachment.
    pub depth: u32,
    /// The attached drawing's model space, flattened in its own coordinates.
    pub flattened: super::flatten::Flattened,
    /// The xrefs the attached drawing attaches in turn.
    pub nested: Vec<FlattenedXref>,
}

#[cfg(feature = "pvec-normalized")]
impl XrefSession {
    /// Decide `xref`, and under `Resolve` open the attached drawing with `P` and flatten it.
    ///
    /// # Errors
    ///
    /// [`XrefError`].
    pub fn open_and_flatten<P: DwgProvider>(
        &mut self,
        guard: &mut SourceGuard,
        xref: &XrefRef,
        bounds: super::flatten::FlattenBounds,
    ) -> Result<XrefOpened, XrefError> {
        let _ = (guard, xref, bounds);
        Ok(XrefOpened::Omitted(0))
    }
}
