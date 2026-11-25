//! Finds and runs qpdf for the tests. Stub: the real helper comes with the fix (#1715).
#![allow(dead_code)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub fn locate(_var: Option<OsString>, _path: Option<OsString>) -> Option<PathBuf> {
    None
}

pub fn check_with(_qpdf: Option<PathBuf>, _pdf: &Path) -> Result<bool, String> {
    Ok(false)
}

pub fn check_pdf(_pdf: &Path) -> Result<bool, String> {
    Ok(false)
}
