//! The user's default plot pair, kept as copies in a per-user config directory (issue #1670).
//!
//! Nothing here runs unless a caller asks. [`store`], [`load`] and [`clear`] take the directory as an
//! argument, [`config_dir`] works out the usual place from the environment it is handed, and
//! `PdfPlot` reads none of it: the caller (the CLI, a server) walks
//! its own precedence and passes the result to the plot.
//!
//! The two files are the whole state: `plot/default.ctb` and `plot/default.pc3` under the directory,
//! stored as given, and a user can drop them in by hand. A file is validated going in and again every
//! time it is read, so a stored default that no longer loads is an error with the code
//! `plot-default-unreadable`, never a quiet fall back to the built-in.

use std::fmt;
use std::path::{Path, PathBuf};

use super::plot_files::PlotFileError;

/// Which of the two default files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotDefaultFile {
    /// The plot style table, a CTB.
    Ctb,
    /// The plot configuration, a JSON PC3.
    Pc3,
}

/// Why a default was refused or could not be read or written.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlotDefaultsError {
    /// The bytes are refused by the bounded plot file loaders.
    File(PlotFileError),
    /// A STB, which applies nothing today.
    Stb,
    /// A PC3 that is not the JSON kind or has no readable device facts.
    NoDevice,
    /// A stored default that no longer loads, and why.
    Unreadable(String),
    /// A symlink or a directory where a default file belongs, or a symlinked `plot` directory.
    Unsafe(String),
    /// The directory or file could not be read or written.
    Io(String),
}

impl PlotDefaultsError {
    /// A short code that stays the same from release to release, for a caller to match on.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            PlotDefaultsError::File(e) => e.code(),
            PlotDefaultsError::Stb => "plot-default-stb",
            PlotDefaultsError::NoDevice => "plot-default-no-device",
            PlotDefaultsError::Unreadable(_) => "plot-default-unreadable",
            PlotDefaultsError::Unsafe(_) => "plot-default-unsafe",
            PlotDefaultsError::Io(_) => "plot-default-io",
        }
    }
}

impl fmt::Display for PlotDefaultsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlotDefaultsError::File(e) => write!(f, "{e}"),
            PlotDefaultsError::Stb => write!(f, "a STB applies nothing, so it cannot be a default"),
            PlotDefaultsError::NoDevice => write!(f, "the PC3 names no device the plot can read"),
            PlotDefaultsError::Unreadable(why) => {
                write!(f, "the stored default cannot be read: {why}")
            }
            PlotDefaultsError::Unsafe(why) => write!(f, "refused to go through {why}"),
            PlotDefaultsError::Io(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for PlotDefaultsError {}

/// The directory the defaults live in: `LIBVIPRS_CONFIG_DIR` itself when it is absolute, else
/// `$XDG_CONFIG_HOME/libviprs` or `~/.config/libviprs` (Linux and anything else),
/// `~/Library/Application Support/libviprs` (`"macos"`) or `%APPDATA%\libviprs` (`"windows"`).
/// A relative value is ignored. `os` is [`std::env::consts::OS`] and `env` reads a variable, so a
/// caller passes the real ones and a test passes its own.
#[must_use]
pub fn config_dir(os: &str, env: impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    let _ = (os, env);
    todo!("issue #1670")
}

/// Stores a copy of `bytes` as the default `which` under `dir`, once it passes the same checks a load makes.
///
/// # Errors
///
/// [`PlotDefaultsError`]; nothing is written when the bytes are refused.
pub fn store(dir: &Path, which: PlotDefaultFile, bytes: &[u8]) -> Result<(), PlotDefaultsError> {
    let _ = (dir, which, bytes);
    todo!("issue #1670")
}

/// The stored default `which` under `dir`, validated again, or `None` when there is none.
///
/// # Errors
///
/// [`PlotDefaultsError::Unreadable` (`plot-default-unreadable`) when the file is there and does not
/// load, [`PlotDefaultsError::Unsafe` for a symlink.
pub fn load(dir: &Path, which: PlotDefaultFile) -> Result<Option<Vec<u8>>, PlotDefaultsError> {
    let _ = (dir, which);
    todo!("issue #1670")
}

/// Removes the stored default `which`; `true` when there was one.
///
/// # Errors
///
/// [`PlotDefaultsError::Unsafe` for a directory, [`PlotDefaultsError::Io` when it cannot be removed.
pub fn clear(dir: &Path, which: PlotDefaultFile) -> Result<bool, PlotDefaultsError> {
    let _ = (dir, which);
    todo!("issue #1670")
}
