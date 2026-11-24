//! The user's default plot pair, kept as copies in a per-user config directory (issue #1670).
//!
//! Nothing here runs unless a caller asks. [`store`], [`load`] and [`clear`] take the directory as an
//! argument, [`config_dir`] works out the usual place from the environment it is handed, and
//! [`PdfPlot`](crate::pvec_convert::PdfPlot) reads none of it: the caller (the CLI, a server) walks
//! its own precedence and passes the result to the plot.
//!
//! The two files are the whole state: `plot/default.ctb` and `plot/default.pc3` under the directory,
//! stored as given, and a user can drop them in by hand. A file is validated going in and again every
//! time it is read, so a stored default that no longer loads is an error with the code
//! `plot-default-unreadable`, never a quiet fall back to the built-in.

use std::fmt;
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use super::plot_files::{
    MAX_PLOT_FILE_BYTES, Pc3File, PlotDevice, PlotFileError, load_pc3, load_plot_style,
};

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
    let absolute = |name: &str| env(name).map(PathBuf::from).filter(|p| p.is_absolute());
    if let Some(dir) = absolute("LIBVIPRS_CONFIG_DIR") {
        return Some(dir);
    }
    match os {
        "macos" => absolute("HOME").map(|h| h.join("Library/Application Support/libviprs")),
        "windows" => absolute("APPDATA").map(|a| a.join("libviprs")),
        _ => absolute("XDG_CONFIG_HOME")
            .map(|x| x.join("libviprs"))
            .or_else(|| absolute("HOME").map(|h| h.join(".config/libviprs"))),
    }
}

fn file_name(which: PlotDefaultFile) -> &'static str {
    match which {
        PlotDefaultFile::Ctb => "default.ctb",
        PlotDefaultFile::Pc3 => "default.pc3",
    }
}

/// What a stored or about to be stored file has to be: bounded, of its kind, and something a plot
/// applies. An STB applies nothing, and a PC3 with no device facts has nothing to give.
fn validate(which: PlotDefaultFile, bytes: &[u8]) -> Result<(), PlotDefaultsError> {
    match which {
        PlotDefaultFile::Ctb => {
            let table = load_plot_style("default.ctb", bytes).map_err(PlotDefaultsError::File)?;
            let no_aci_table = table
                .to_text()
                .lines()
                .any(|l| l.trim() == "aci_table_available=FALSE");
            if table.is_stb || no_aci_table {
                return Err(PlotDefaultsError::Stb);
            }
        }
        PlotDefaultFile::Pc3 => match load_pc3(bytes).map_err(PlotDefaultsError::File)? {
            Pc3File::Json(pc3) => {
                PlotDevice::from_pc3(&pc3).map_err(|_| PlotDefaultsError::NoDevice)?;
            }
            Pc3File::Binary(_) => return Err(PlotDefaultsError::NoDevice),
        },
    }
    Ok(())
}

fn io(path: &Path, e: &std::io::Error) -> PlotDefaultsError {
    PlotDefaultsError::Io(format!("{}: {e}", path.display()))
}

/// Refuses a symlink or anything that is not a regular file, and says whether there is a file.
fn regular_file(path: &Path) -> Result<Option<fs::Metadata>, PlotDefaultsError> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_file() => Ok(Some(meta)),
        Ok(_) => Err(PlotDefaultsError::Unsafe(format!(
            "{}, which is not a regular file",
            path.display()
        ))),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io(path, &e)),
    }
}

/// The `plot` directory under `dir`, refusing a symlink there. `create` makes it, private, when it is not there.
fn plot_dir(dir: &Path, create: bool) -> Result<Option<PathBuf>, PlotDefaultsError> {
    let plot = dir.join("plot");
    match fs::symlink_metadata(&plot) {
        Ok(meta) if meta.file_type().is_dir() => Ok(Some(plot)),
        Ok(_) => Err(PlotDefaultsError::Unsafe(format!(
            "{}, which is not a directory",
            plot.display()
        ))),
        Err(e) if e.kind() == ErrorKind::NotFound && create => {
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
            builder.create(&plot).map_err(|e| io(&plot, &e))?;
            Ok(Some(plot))
        }
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io(&plot, &e)),
    }
}

/// Stores a copy of `bytes` as the default `which` under `dir`, once it passes the same checks a load makes.
/// The copy is written beside its place and renamed over it, so a reader sees the old file or the
/// new one. The directory is 0700 and the file 0600 on unix.
///
/// # Errors
///
/// [`PlotDefaultsError`]; nothing is written when the bytes are refused.
pub fn store(dir: &Path, which: PlotDefaultFile, bytes: &[u8]) -> Result<(), PlotDefaultsError> {
    validate(which, bytes)?;
    let plot =
        plot_dir(dir, true)?.ok_or_else(|| PlotDefaultsError::Io("no plot directory".into()))?;
    let target = plot.join(file_name(which));
    regular_file(&target)?;
    let scratch = plot.join(format!(".{}.{}.tmp", file_name(which), std::process::id()));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let written = options
        .open(&scratch)
        .and_then(|mut f| f.write_all(bytes).and_then(|()| f.sync_all()))
        .and_then(|()| fs::rename(&scratch, &target));
    if let Err(e) = written {
        let _ = fs::remove_file(&scratch);
        return Err(io(&target, &e));
    }
    Ok(())
}

/// The stored default `which` under `dir`, validated again, or `None` when there is none. The size is
/// checked before the file is read.
///
/// # Errors
///
/// [`PlotDefaultsError::Unreadable` (`plot-default-unreadable`) when the file is there and does not
/// load, [`PlotDefaultsError::Unsafe` for a symlink.
pub fn load(dir: &Path, which: PlotDefaultFile) -> Result<Option<Vec<u8>>, PlotDefaultsError> {
    let Some(plot) = plot_dir(dir, false)? else {
        return Ok(None);
    };
    let path = plot.join(file_name(which));
    let Some(meta) = regular_file(&path)? else {
        return Ok(None);
    };
    let unreadable =
        |why: String| PlotDefaultsError::Unreadable(format!("{}: {why}", path.display()));
    if meta.len() > MAX_PLOT_FILE_BYTES as u64 {
        return Err(unreadable(format!(
            "{} bytes is over the limit",
            meta.len()
        )));
    }
    let mut bytes = Vec::new();
    fs::File::open(&path)
        .and_then(|f| {
            f.take(MAX_PLOT_FILE_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
        })
        .map_err(|e| unreadable(e.to_string()))?;
    validate(which, &bytes).map_err(|e| unreadable(e.to_string()))?;
    Ok(Some(bytes))
}

/// Removes the stored default `which`; `true` when there was one.
///
/// # Errors
///
/// [`PlotDefaultsError::Unsafe` for a directory, [`PlotDefaultsError::Io` when it cannot be removed.
pub fn clear(dir: &Path, which: PlotDefaultFile) -> Result<bool, PlotDefaultsError> {
    let Some(plot) = plot_dir(dir, false)? else {
        return Ok(false);
    };
    let path = plot.join(file_name(which));
    match fs::symlink_metadata(&path) {
        Ok(meta) if meta.file_type().is_dir() => Err(PlotDefaultsError::Unsafe(format!(
            "{}, which is a directory",
            path.display()
        ))),
        Ok(_) => fs::remove_file(&path)
            .map(|()| true)
            .map_err(|e| io(&path, &e)),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(io(&path, &e)),
    }
}
