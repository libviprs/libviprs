//! Finds and runs qpdf for the tests (#1715).
//!
//! qpdf is a check the tests make on the PDFs the plot writes, not something the crate links or
//! needs. `LIBVIPRS_QPDF` names the binary and wins; otherwise the first `qpdf` on `PATH` is used.
//! With neither, the check is skipped and says so on stderr, so a machine without qpdf still
//! passes. A pinned build for Linux and macOS is published by libviprs-dep (`qpdf-<platform>-<cpu>`).
#![allow(dead_code)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The qpdf to use: the file `var` (the value of `LIBVIPRS_QPDF`) names, or the first `qpdf` in
/// the directories of `path` (the value of `PATH`). A variable that names no file finds nothing,
/// rather than falling through to some other qpdf the caller did not ask for.
pub fn locate(var: Option<OsString>, path: Option<OsString>) -> Option<PathBuf> {
    if let Some(var) = var {
        let named = PathBuf::from(var);
        return named.is_file().then_some(named);
    }
    std::env::split_paths(&path?)
        .map(|dir| dir.join("qpdf"))
        .find(|candidate| candidate.is_file())
}

/// Runs `qpdf --check` on `pdf` with `qpdf`. `Ok(true)` is a clean check, `Ok(false)` is a skip
/// because there is no qpdf, and `Err` carries qpdf's own output.
pub fn check_with(qpdf: Option<PathBuf>, pdf: &Path) -> Result<bool, String> {
    let Some(qpdf) = qpdf else {
        eprintln!(
            "skipping qpdf --check on {}: no qpdf found (set LIBVIPRS_QPDF or put qpdf on PATH)",
            pdf.display()
        );
        return Ok(false);
    };
    let out = Command::new(&qpdf)
        .arg("--check")
        .arg(pdf)
        .output()
        .map_err(|e| format!("could not run {}: {e}", qpdf.display()))?;
    if out.status.success() {
        return Ok(true);
    }
    Err(format!(
        "qpdf --check rejected {} ({}):\n{}{}",
        pdf.display(),
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ))
}

/// `check_with` with the qpdf the environment names.
pub fn check_pdf(pdf: &Path) -> Result<bool, String> {
    let var = std::env::var_os("LIBVIPRS_QPDF");
    if var.is_some() && locate(var.clone(), None).is_none() {
        eprintln!("LIBVIPRS_QPDF is set but names no file, so qpdf --check is skipped");
    }
    check_with(locate(var, std::env::var_os("PATH")), pdf)
}
