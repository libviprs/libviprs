//! Issue #1076 (N4.5): the xref and external-reference policy.
//!
//! One fixture, three policies, three distinct outcomes. Path refusals never touch the filesystem.

use std::fs;
use std::path::Path;

use libviprs::pvec::limits::{SourceGuard, SourceLimits};
use libviprs::pvec::xref::{
    PathRefusal, XrefError, XrefOutcome, XrefPolicy, XrefRef, XrefSession, confine,
};
use libviprs::pvec::{ArrayGrid, SourceTransform};

fn xref(name: &str, path: &str) -> XrefRef {
    XrefRef {
        name: name.to_string(),
        path: path.to_string(),
        transform: SourceTransform::new([1.0, 0.0, 0.0, 1.0, 10.0, 20.0]).unwrap(),
        layer: "A-XREF".to_string(),
        grid: ArrayGrid::SINGLE,
    }
}

fn guard() -> SourceGuard {
    SourceGuard::new(SourceLimits::default())
}

/// A root holding host.dwg and wing.dwg.
fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("host.dwg"), b"x").unwrap();
    fs::write(dir.path().join("wing.dwg"), b"x").unwrap();
    dir
}

#[test]
fn default_policy_is_external_dependency() {
    assert_eq!(XrefPolicy::default(), XrefPolicy::ExternalDependency);
}

#[test]
#[cfg_attr(miri, ignore)]
fn external_dependency_records_name_path_transform_layer() {
    let dir = fixture();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::ExternalDependency);
    let out = s.admit(&mut guard(), &xref("WING", "wing.dwg")).unwrap();
    assert!(matches!(out, XrefOutcome::Omitted(0)), "{out:?}");
    let r = s.into_report();
    assert_eq!(r.unresolved.len(), 1, "an omitted xref must leave a row");
    let row = &r.unresolved[0];
    assert_eq!(row.name, "WING");
    assert_eq!(row.path, "wing.dwg");
    assert_eq!(
        row.resolved_path.as_deref(),
        Some(dir.path().join("wing.dwg").as_path())
    );
    assert_eq!(row.layer, "A-XREF");
    assert_eq!(row.transform.coefficients()[4], 10.0);
    assert!(r.resolved.is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn missing_xref_file_still_leaves_a_row_under_external_dependency() {
    let dir = fixture();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::ExternalDependency);
    s.admit(&mut guard(), &xref("GONE", "gone.dwg")).unwrap();
    let r = s.into_report();
    assert!(!r.unresolved.is_empty());
    assert_eq!(r.unresolved[0].name, "GONE");
}

#[test]
#[cfg_attr(miri, ignore)]
fn reject_fails_naming_xref_and_resolved_path() {
    let dir = fixture();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Reject);
    let e = s
        .admit(&mut guard(), &xref("WING", "wing.dwg"))
        .unwrap_err();
    match &e {
        XrefError::Rejected { name, path } => {
            assert_eq!(name, "WING");
            assert_eq!(path, &dir.path().join("wing.dwg"));
        }
        other => panic!("{other:?}"),
    }
    let msg = e.to_string();
    assert!(msg.contains("WING") && msg.contains("wing.dwg"), "{msg}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn resolve_clears_the_file_and_reports_depth() {
    let dir = fixture();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    let out = s.admit(&mut guard(), &xref("WING", "wing.dwg")).unwrap();
    match out {
        XrefOutcome::Resolve(r) => {
            assert_eq!(r.name, "WING");
            assert_eq!(r.path, dir.path().join("wing.dwg"));
            assert_eq!(r.depth, 1);
        }
        other => panic!("{other:?}"),
    }
    let r = s.into_report();
    assert!(r.unresolved.is_empty());
    assert_eq!(r.resolved, vec!["WING".to_string()]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn resolve_of_a_missing_file_is_a_typed_error_not_an_omission() {
    let dir = fixture();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    let e = s
        .admit(&mut guard(), &xref("GONE", "gone.dwg"))
        .unwrap_err();
    assert!(matches!(e, XrefError::Missing { .. }), "{e:?}");
}

#[test]
fn hostile_paths_are_refused_with_typed_errors_in_every_resolving_mode() {
    // The root does not exist: a refusal that opened anything would fail differently.
    let root = Path::new("/nonexistent-root-1076");
    let cases = [
        ("../../etc/passwd", PathRefusal::EscapesRoot),
        ("sub/../../x.dwg", PathRefusal::EscapesRoot),
        ("..\\..\\x.dwg", PathRefusal::EscapesRoot),
        ("/etc/passwd", PathRefusal::Absolute),
        ("\\\\server\\share\\x.dwg", PathRefusal::Unc),
        ("//server/share/x.dwg", PathRefusal::Unc),
        ("C:\\x.dwg", PathRefusal::DriveLetter),
        ("", PathRefusal::Malformed),
    ];
    for (path, want) in cases {
        assert_eq!(confine(root, path), Err(want), "{path:?}");
        for policy in [XrefPolicy::Resolve, XrefPolicy::Reject] {
            let mut s = XrefSession::new(root, "host.dwg", policy);
            let e = s.admit(&mut guard(), &xref("X", path)).unwrap_err();
            assert!(
                matches!(&e, XrefError::PathRefused { reason, .. } if *reason == want),
                "{path:?} {policy:?}: {e:?}"
            );
        }
    }
}

#[test]
fn confine_accepts_inner_dotdot_and_subdirectories() {
    let root = Path::new("/r");
    assert_eq!(confine(root, "a/b.dwg").unwrap(), Path::new("/r/a/b.dwg"));
    assert_eq!(confine(root, "a/../b.dwg").unwrap(), Path::new("/r/b.dwg"));
    assert_eq!(confine(root, "a\\b.dwg").unwrap(), Path::new("/r/a/b.dwg"));
}

#[test]
fn hostile_path_under_external_dependency_is_recorded_with_its_refusal_and_opens_nothing() {
    let root = Path::new("/nonexistent-root-1076");
    let mut s = XrefSession::new(root, "host.dwg", XrefPolicy::ExternalDependency);
    s.admit(&mut guard(), &xref("EVIL", "../../etc/passwd"))
        .unwrap();
    let r = s.into_report();
    assert_eq!(r.unresolved.len(), 1);
    assert_eq!(r.unresolved[0].path_refusal, Some(PathRefusal::EscapesRoot));
    assert!(r.unresolved[0].resolved_path.is_none());
}

#[test]
#[cfg_attr(miri, ignore)]
fn two_file_cycle_terminates_naming_both_paths() {
    let dir = fixture();
    fs::write(dir.path().join("b.dwg"), b"x").unwrap();
    let mut g = guard();
    // host.dwg -> b.dwg -> host.dwg
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    assert!(matches!(
        s.admit(&mut g, &xref("B", "b.dwg")).unwrap(),
        XrefOutcome::Resolve(_)
    ));
    let e = s.admit(&mut g, &xref("HOST", "host.dwg")).unwrap_err();
    match &e {
        XrefError::Cycle { chain } => assert_eq!(chain.len(), 3, "{chain:?}"),
        other => panic!("{other:?}"),
    }
    let msg = e.to_string();
    assert!(msg.contains("host.dwg") && msg.contains("b.dwg"), "{msg}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn depth_past_the_limit_names_the_limit_and_the_observed_depth() {
    let dir = tempfile::tempdir().unwrap();
    for n in ["host", "l1", "l2", "l3"] {
        fs::write(dir.path().join(format!("{n}.dwg")), b"x").unwrap();
    }
    let mut limits = SourceLimits::default();
    limits.max_block_depth = 2;
    let mut g = SourceGuard::new(limits);
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    s.admit(&mut g, &xref("L1", "l1.dwg")).unwrap();
    s.admit(&mut g, &xref("L2", "l2.dwg")).unwrap();
    let e = s.admit(&mut g, &xref("L3", "l3.dwg")).unwrap_err();
    let msg = e.to_string();
    assert!(matches!(e, XrefError::Limit(_)), "{msg}");
    assert!(
        msg.contains("max_block_depth") && msg.contains("observed 3") && msg.contains("ceiling 2"),
        "{msg}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn release_lets_a_sibling_reuse_a_depth_slot_and_a_path() {
    let dir = fixture();
    let mut g = guard();
    let mut s = XrefSession::new(dir.path(), "host.dwg", XrefPolicy::Resolve);
    s.admit(&mut g, &xref("W1", "wing.dwg")).unwrap();
    s.release(&mut g);
    // Not a cycle: the first attachment has been left.
    match s.admit(&mut g, &xref("W2", "wing.dwg")).unwrap() {
        XrefOutcome::Resolve(r) => assert_eq!(r.depth, 1),
        other => panic!("{other:?}"),
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_ok_outcome_is_accounted_for_in_the_report() {
    // No Ok may drop an xref: each admit is either a Resolve (listed in `resolved`)
    // or an Omitted(i) that points at a real row.
    let dir = fixture();
    for policy in [XrefPolicy::Resolve, XrefPolicy::ExternalDependency] {
        let mut s = XrefSession::new(dir.path(), "host.dwg", policy);
        let out = s.admit(&mut guard(), &xref("WING", "wing.dwg")).unwrap();
        let r = s.report();
        match out {
            XrefOutcome::Omitted(i) => assert_eq!(r.unresolved[i].name, "WING"),
            XrefOutcome::Resolve(_) => assert!(r.resolved.iter().any(|n| n == "WING")),
            other => panic!("{other:?}"),
        }
    }
}
