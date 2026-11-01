//! Atomic publication of a PVec artifact and its report (issue #1314, N6.8).
//!
//! A conversion that dies two thirds of the way through must not leave something a downstream job
//! can mistake for a finished drawing. [`publish`] writes the artifact and its report next to the
//! destination as `<dest>.pvec.partial` (so `plan.pvec` stages as `plan.pvec.pvec.partial`) and `<report>.partial`, syncs both, and only then renames
//! them into place. Nothing at the destination changes before that last step.
//!
//! # What proves an artifact complete
//!
//! The artifact does, not its name. [`publish`] opens the bytes with PVec's reader before it
//! writes anything, so a truncated file, or one with no valid digest, is refused with
//! [`PublishError::NotAnArtifact`] and never becomes a destination file. [`open_published`] is the
//! reader path: it refuses a name ending in `.pvec.partial` outright, before any open or parse,
//! and hands everything else to PVec's reader, which rejects a committed artifact that was cut
//! short.
//!
//! # Failure paths
//!
//! A returned error, a panic and a cancellation each remove the partial files and leave the
//! previous artifact and report byte-identical (or absent, when there was none). A kill leaves the
//! partial files behind, which is unavoidable and harmless: they carry the refused suffix, they
//! are never opened, and the next publish to the same destination reuses their names, so no stale
//! partial outlives a re-run. Two publishers to the same destination at once are not supported.
//!
//! # The commit
//!
//! The report is renamed first, then the artifact, so a kill between the two leaves a report with
//! no artifact (reaped by the next publish) and never an artifact without its report. The two
//! renames are the commit and are not interruptible by [`CancelToken`]: once [`Stage::Committing`]
//! has been reported the publish runs to the end. Replacing an existing pair is two renames, not
//! one, so a failure of the second rename after the first leaves the new report beside the old
//! artifact. The partials were synced beforehand and both renames are in one directory, so that
//! failure needs the filesystem to refuse a rename it has just accepted.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use pvec_facade::PvecReader;
use thiserror::Error;

use crate::cancel::CancelToken;

/// The suffix of a file that is not yet an artifact.
const PARTIAL_SUFFIX: &str = ".pvec.partial";

/// A point in the publish, reported before the step it names runs. Cancellation is checked at
/// every stage up to and including [`Stage::Committing`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Before the artifact's partial file exists.
    ArtifactWriting,
    /// The artifact's partial is written, before it is synced.
    ArtifactSyncing,
    /// Before the report's partial file exists.
    ReportWriting,
    /// The report's partial is written, before it is synced.
    ReportSyncing,
    /// Both partials are synced, before the renames.
    Committing,
    /// Both renames are done.
    Committed,
}

/// Why a publish failed. Nothing is left at the destination except the previous pair.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PublishError {
    /// The cancel token fired before the commit.
    #[error("cancelled before the artifact was committed")]
    Cancelled,
    /// The bytes are not a complete artifact: PVec's reader refused them.
    #[error("the bytes are not a complete PVec artifact")]
    NotAnArtifact(#[source] pvec_facade::PvecError),
    /// Writing, syncing or renaming failed.
    #[error("publishing the artifact failed")]
    Io(#[source] io::Error),
}

/// Why a published artifact could not be opened.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OpenError {
    /// The name ends in `.pvec.partial`, so the file is not an artifact. It was not opened.
    #[error("{} is a partial file, not an artifact", .0.display())]
    Partial(PathBuf),
    /// The file could not be read.
    #[error("reading the artifact failed")]
    Io(#[source] io::Error),
    /// PVec's reader refused the bytes.
    #[error("PVec's reader refused the artifact")]
    Pvec(#[source] pvec_facade::PvecError),
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// Where the report for the artifact at `dest` is published: `dest` with `.report.json` appended.
#[must_use]
pub fn report_path(dest: &Path) -> PathBuf {
    with_suffix(dest, ".report.json")
}

/// Removes the partial files unless the commit has started.
struct Partials<'a> {
    paths: [&'a Path; 2],
    armed: bool,
}

impl Drop for Partials<'_> {
    fn drop(&mut self) {
        if self.armed {
            for p in self.paths {
                let _ = fs::remove_file(p);
            }
        }
    }
}

fn write_synced(path: &Path, bytes: &[u8]) -> io::Result<File> {
    let mut f = File::create(path)?;
    f.write_all(bytes)?;
    Ok(f)
}

/// Publish `artifact` at `dest` and `report` at [`report_path`]`(dest)`, both or neither.
///
/// `at` is told each [`Stage`] before the step runs. It is how a caller shows progress, and how the
/// tests stop the publish at every step.
///
/// # Errors
///
/// [`PublishError`]. After any error `dest` and its report are as they were before the call.
pub fn publish(
    dest: &Path,
    artifact: &[u8],
    report: &[u8],
    cancel: &CancelToken,
    at: &mut dyn FnMut(Stage),
) -> Result<(), PublishError> {
    PvecReader::open(artifact.to_vec()).map_err(PublishError::NotAnArtifact)?;
    let report_dest = report_path(dest);
    let artifact_partial = with_suffix(dest, PARTIAL_SUFFIX);
    let report_partial = with_suffix(&report_dest, ".partial");
    let mut partials = Partials {
        paths: [&artifact_partial, &report_partial],
        armed: true,
    };
    let mut step = |stage: Stage| -> Result<(), PublishError> {
        at(stage);
        if cancel.is_cancelled() {
            Err(PublishError::Cancelled)
        } else {
            Ok(())
        }
    };

    step(Stage::ArtifactWriting)?;
    let artifact_file = write_synced(&artifact_partial, artifact).map_err(PublishError::Io)?;
    step(Stage::ArtifactSyncing)?;
    artifact_file.sync_all().map_err(PublishError::Io)?;
    drop(artifact_file);

    step(Stage::ReportWriting)?;
    let report_file = write_synced(&report_partial, report).map_err(PublishError::Io)?;
    step(Stage::ReportSyncing)?;
    report_file.sync_all().map_err(PublishError::Io)?;
    drop(report_file);

    step(Stage::Committing)?;
    partials.armed = false;
    let renamed = fs::rename(&report_partial, &report_dest)
        .and_then(|()| fs::rename(&artifact_partial, dest));
    if let Err(e) = renamed {
        let _ = fs::remove_file(&artifact_partial);
        let _ = fs::remove_file(&report_partial);
        return Err(PublishError::Io(e));
    }
    at(Stage::Committed);
    Ok(())
}

/// Open the artifact at `path` with PVec's reader.
///
/// A path whose name ends in `.pvec.partial` is refused before it is opened.
///
/// # Errors
///
/// [`OpenError`]: a partial name, an unreadable file, or PVec's refusal (a truncated artifact
/// fails here, since its digest cannot verify).
pub fn open_published(path: &Path) -> Result<PvecReader, OpenError> {
    if path
        .file_name()
        .is_some_and(|n| n.to_string_lossy().ends_with(PARTIAL_SUFFIX))
    {
        return Err(OpenError::Partial(path.to_path_buf()));
    }
    let bytes = fs::read(path).map_err(OpenError::Io)?;
    PvecReader::open(bytes).map_err(OpenError::Pvec)
}

/// Why a streamed publish failed: the publish itself, or the producer of the artifact.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StreamedPublishError<E: std::error::Error + 'static> {
    /// The publish failed.
    #[error(transparent)]
    Publish(PublishError),
    /// The producer failed or was cancelled.
    #[error("the producer of the artifact failed")]
    Produce(#[source] E),
}

/// Stub: red tests first.
///
/// # Errors
///
/// None yet.
pub fn publish_streamed<E: std::error::Error + 'static>(
    _dest: &Path,
    _report: &[u8],
    _cancel: &CancelToken,
    _at: &mut dyn FnMut(Stage),
    _produce: impl FnOnce(&mut dyn io::Write) -> Result<(), E>,
) -> Result<(), StreamedPublishError<E>> {
    unimplemented!("pvec#235 sink")
}
