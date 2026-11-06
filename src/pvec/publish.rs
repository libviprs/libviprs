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

/// Publish the artifact `produce` writes at `dest`, and the report it returns at [`report_path`]`(dest)`,
/// both or neither. `produce` is handed the `.pvec.partial` file to write the artifact into, so the artifact
/// never has to exist in memory, and returns the report bytes once it knows them (the report counts what went
/// into the artifact). The file is synced, then read back and opened with PVec's reader (a truncated file or
/// one whose digest does not verify is refused before anything is committed), and then the pair is renamed
/// into place.
///
/// `at` is told each [`Stage`] before the step runs, and cancellation is checked at every stage up to
/// [`Stage::Committing`] and again when `produce` returns. If `produce` fails, or the token fired while it
/// ran, both partials are removed and `dest` and its report are as they were.
///
/// # Errors
///
/// [`StreamedPublishError::Produce`] with the producer's error, or [`StreamedPublishError::Publish`].
pub fn publish_streamed<E: std::error::Error + 'static>(
    dest: &Path,
    cancel: &CancelToken,
    at: &mut dyn FnMut(Stage),
    produce: impl FnOnce(&mut dyn Write) -> Result<Vec<u8>, E>,
) -> Result<(), StreamedPublishError<E>> {
    publish_streamed_with(dest, cancel, at, produce, |report, out| {
        out.write_all(report)
    })
    .map(|_| ())
}

/// As [`publish_streamed`], but the report is written from what `produce` returns instead of being
/// returned as bytes, so a large report never has to exist as one buffer. `write_report` is handed the
/// value `produce` returned and the `.report.json.partial` file (buffered), and the value comes back once
/// the pair is committed.
///
/// # Errors
///
/// As [`publish_streamed`]; a failing `write_report` is [`PublishError::Io`].
pub fn publish_streamed_with<T, E: std::error::Error + 'static>(
    dest: &Path,
    cancel: &CancelToken,
    at: &mut dyn FnMut(Stage),
    produce: impl FnOnce(&mut dyn Write) -> Result<T, E>,
    write_report: impl FnOnce(&T, &mut dyn Write) -> io::Result<()>,
) -> Result<T, StreamedPublishError<E>> {
    use StreamedPublishError::{Produce, Publish};
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

    step(Stage::ArtifactWriting).map_err(Publish)?;
    let mut artifact_file =
        File::create(&artifact_partial).map_err(|e| Publish(PublishError::Io(e)))?;
    let sealed = produce(&mut artifact_file).map_err(Produce)?;
    step(Stage::ArtifactSyncing).map_err(Publish)?;
    artifact_file
        .sync_all()
        .map_err(|e| Publish(PublishError::Io(e)))?;
    drop(artifact_file);
    // Completeness is proven by the artifact: read it back and let PVec's reader say.
    let written = fs::read(&artifact_partial).map_err(|e| Publish(PublishError::Io(e)))?;
    PvecReader::open(written).map_err(|e| Publish(PublishError::NotAnArtifact(e)))?;

    step(Stage::ReportWriting).map_err(Publish)?;
    let report_file = File::create(&report_partial)
        .and_then(|f| {
            let mut out = io::BufWriter::new(f);
            write_report(&sealed, &mut out)?;
            out.into_inner().map_err(io::IntoInnerError::into_error)
        })
        .map_err(|e| Publish(PublishError::Io(e)))?;
    step(Stage::ReportSyncing).map_err(Publish)?;
    report_file
        .sync_all()
        .map_err(|e| Publish(PublishError::Io(e)))?;
    drop(report_file);

    step(Stage::Committing).map_err(Publish)?;
    partials.armed = false;
    let renamed = fs::rename(&report_partial, &report_dest)
        .and_then(|()| fs::rename(&artifact_partial, dest));
    if let Err(e) = renamed {
        let _ = fs::remove_file(&artifact_partial);
        let _ = fs::remove_file(&report_partial);
        return Err(Publish(PublishError::Io(e)));
    }
    at(Stage::Committed);
    Ok(sealed)
}

/// Publish `artifact` at `dest` and `report` at [`report_path`]`(dest)`, both or neither: the artifact in
/// memory, through [`publish_streamed`].
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
    publish_streamed(dest, cancel, at, |out| {
        out.write_all(artifact)?;
        Ok(report.to_vec())
    })
    .map_err(|e| match e {
        StreamedPublishError::Publish(p) => p,
        StreamedPublishError::Produce(io) => PublishError::Io(io),
    })
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
