//! Issue #1715: the tests run `qpdf --check` on a PDF the plot writes, and skip that check, with a
//! line on stderr, when no qpdf can be found. `LIBVIPRS_QPDF` names the binary and wins; otherwise
//! the first `qpdf` on `PATH` is used. This is test scaffolding only: the crate does not depend on
//! qpdf.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "common/qpdf.rs"]
mod qpdf;

#[cfg(feature = "pvec-convert")]
#[path = "pvec_1704_support/mod.rs"]
mod support;

fn scratch(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("libviprs-1715-{tag}-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn file(dir: &Path, name: &str) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, b"x").unwrap();
    p
}

/// A stand in for qpdf: a shell script that prints `text` on stderr and exits with `code`.
#[cfg(unix)]
fn fake_qpdf(dir: &Path, code: i32, text: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join("fake-qpdf");
    fs::write(&p, format!("#!/bin/sh\necho '{text}' >&2\nexit {code}\n")).unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    p
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1715_the_environment_variable_wins_over_path() {
    let dir = scratch("env");
    let named = file(&dir, "chosen-qpdf");
    let on_path = dir.join("bin");
    fs::create_dir_all(&on_path).unwrap();
    file(&on_path, "qpdf");
    let found = qpdf::locate(Some(named.clone().into()), Some(on_path.into()));
    assert_eq!(found, Some(named));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1715_path_is_searched_in_order_when_the_variable_is_unset() {
    let dir = scratch("path");
    let (a, b) = (dir.join("a"), dir.join("b"));
    fs::create_dir_all(&a).unwrap();
    fs::create_dir_all(&b).unwrap();
    let want = file(&b, "qpdf");
    let path = std::env::join_paths([&a, &b]).unwrap();
    assert_eq!(qpdf::locate(None, Some(path)), Some(want));
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1715_nothing_is_found_when_neither_names_a_qpdf() {
    let dir = scratch("none");
    assert_eq!(qpdf::locate(None, Some(dir.clone().into())), None);
    assert_eq!(qpdf::locate(None, None), None);
    let gone = dir.join("no-such-qpdf");
    assert_eq!(
        qpdf::locate(Some(gone.into()), Some(dir.clone().into())),
        None,
        "a variable that names nothing is not quietly replaced by a different qpdf"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1715_a_missing_qpdf_skips_instead_of_failing() {
    let dir = scratch("skip");
    let pdf = file(&dir, "plot.pdf");
    assert_eq!(qpdf::check_with(None, &pdf), Ok(false));
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1715_a_clean_check_is_reported_as_checked() {
    let dir = scratch("clean");
    let pdf = file(&dir, "plot.pdf");
    let q = fake_qpdf(&dir, 0, "No syntax or stream encoding errors found");
    assert_eq!(qpdf::check_with(Some(q), &pdf), Ok(true));
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1715_a_failed_check_carries_qpdfs_own_words() {
    let dir = scratch("fail");
    let pdf = file(&dir, "plot.pdf");
    let q = fake_qpdf(&dir, 2, "bad xref entry");
    let err = qpdf::check_with(Some(q), &pdf).unwrap_err();
    assert!(err.contains("bad xref entry"), "{err}");
    assert!(err.contains("plot.pdf"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[cfg(feature = "pvec-convert")]
fn streamed(dir: &Path, sheets: u32) -> PathBuf {
    use libviprs::cad::CadSource;
    use libviprs::pvec::{SourceUnit, ViewProfile};
    use libviprs::pvec_convert::PdfPlot;
    let dest = dir.join("plot.pdf");
    let bytes = support::drawing(sheets, 40);
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("S001"))
        .layouts((1..=sheets).map(|n| format!("S{n:03}")))
        .convert_with::<support::Synthetic>()
        .unwrap();
    dest
}

#[cfg(feature = "pvec-convert")]
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1715_a_streamed_plot_passes_qpdf_check() {
    let dir = scratch("plot");
    let dest = streamed(&dir, 5);
    qpdf::check_pdf(&dest).unwrap();
    let _ = fs::remove_dir_all(dir);
}

/// With a real qpdf on hand, a streamed file that has lost its tail must be refused, so the check
/// is known to be able to fail. Without one the helper skips and there is nothing to assert.
#[cfg(feature = "pvec-convert")]
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1715_a_streamed_plot_with_its_tail_cut_off_fails_qpdf_check() {
    let dir = scratch("cut");
    let dest = streamed(&dir, 3);
    let raw = fs::read(&dest).unwrap();
    fs::write(&dest, &raw[..raw.len() - 400]).unwrap();
    if qpdf::check_pdf(&dest) == Ok(false) {
        return;
    }
    assert!(qpdf::check_pdf(&dest).is_err());
    let _ = fs::remove_dir_all(dir);
}
