//! Issue #1314 (N6.8): atomic output. The artifact and its report are published together or not at
//! all, a partial file is never mistaken for an artifact, and a failure at any step leaves the
//! destination as it was. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;

use libviprs::CancelToken;
use libviprs::pvec::publish::{
    OpenError, PublishError, Stage, open_published, publish, report_path,
};
use libviprs::pvec::{FidelityClass, Normalized2DGeometry};
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecWriter};

const KILL_ENV: &str = "LIBVIPRS_1314_KILL_AT";
const DEST_ENV: &str = "LIBVIPRS_1314_DEST";

/// The stages the publish reports before each step, in order.
const BEFORE_COMMIT: [Stage; 5] = [
    Stage::ArtifactWriting,
    Stage::ArtifactSyncing,
    Stage::ReportWriting,
    Stage::ReportSyncing,
    Stage::Committing,
];

fn artifact(extra: f64) -> Vec<u8> {
    let g = Normalized2DGeometry::polyline(
        &[[0.0, 0.0], [4.0 + extra, 3.0]],
        FidelityClass::SupportedExact,
    )
    .unwrap();
    let mut w = PvecWriter::new(Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [10.0, 10.0],
        },
        capabilities: Vec::new(),
        producer_id: "t".into(),
        producer_version: "1".into(),
    })
    .unwrap();
    let curve: &Curve = g.geometry().unwrap();
    w.push(EntityId::new(0).unwrap(), curve).unwrap();
    w.finish().unwrap()
}

fn no_stage(_: Stage) {}

fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

fn both_or_neither(dest: &Path) -> bool {
    dest.exists() == report_path(dest).exists()
}

#[test]
fn issue_1314_a_publish_leaves_exactly_the_artifact_and_its_report() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    let bytes = artifact(0.0);
    publish(
        &dest,
        &bytes,
        b"{\"v\":1}",
        &CancelToken::new(),
        &mut no_stage,
    )
    .unwrap();
    assert_eq!(fs::read(&dest).unwrap(), bytes);
    assert_eq!(fs::read(report_path(&dest)).unwrap(), b"{\"v\":1}");
    assert_eq!(entries(dir.path()).len(), 2, "{:?}", entries(dir.path()));
    assert!(open_published(&dest).is_ok());
}

#[test]
fn issue_1314_the_stages_arrive_in_order_and_commit_comes_last() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    let mut seen = Vec::new();
    publish(&dest, &artifact(0.0), b"r", &CancelToken::new(), &mut |s| {
        seen.push(s)
    })
    .unwrap();
    let mut want = BEFORE_COMMIT.to_vec();
    want.push(Stage::Committed);
    assert_eq!(seen, want);
}

#[test]
fn issue_1314_the_in_progress_file_has_the_partial_suffix() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    let mut names = Vec::new();
    publish(&dest, &artifact(0.0), b"r", &CancelToken::new(), &mut |s| {
        if s == Stage::ArtifactSyncing {
            names = entries(dir.path());
        }
    })
    .unwrap();
    assert_eq!(names, ["plan.pvec.pvec.partial"]);
}

#[test]
fn issue_1314_the_reader_path_refuses_a_partial_by_name_before_any_parse() {
    let dir = tempfile::tempdir().unwrap();
    // A file that does not exist: the refusal has to be the name, not a failed open.
    let missing = dir.path().join("plan.pvec.partial");
    assert!(matches!(
        open_published(&missing),
        Err(OpenError::Partial(_))
    ));
    // A complete, valid artifact under the partial name is still refused.
    let valid = dir.path().join("valid.pvec.partial");
    fs::write(&valid, artifact(0.0)).unwrap();
    assert!(matches!(open_published(&valid), Err(OpenError::Partial(_))));
}

#[test]
fn issue_1314_a_truncated_artifact_is_rejected_by_pvecs_reader() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    publish(
        &dest,
        &artifact(0.0),
        b"r",
        &CancelToken::new(),
        &mut no_stage,
    )
    .unwrap();
    let full = fs::read(&dest).unwrap();
    fs::write(&dest, &full[..full.len() * 9 / 10]).unwrap();
    assert!(matches!(open_published(&dest), Err(OpenError::Pvec(_))));
}

#[test]
fn issue_1314_bytes_that_are_not_a_complete_artifact_are_not_publishable() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    let full = artifact(0.0);
    let r = publish(
        &dest,
        &full[..full.len() * 9 / 10],
        b"r",
        &CancelToken::new(),
        &mut no_stage,
    );
    assert!(matches!(r, Err(PublishError::NotAnArtifact(_))), "{r:?}");
    assert!(entries(dir.path()).is_empty());
}

#[test]
fn issue_1314_a_cancel_at_each_step_leaves_the_previous_pair_byte_identical() {
    for stage in BEFORE_COMMIT {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("plan.pvec");
        let (old, old_report) = (artifact(0.0), b"old".to_vec());
        publish(&dest, &old, &old_report, &CancelToken::new(), &mut no_stage).unwrap();

        let token = CancelToken::new();
        let t = token.clone();
        let r = publish(&dest, &artifact(1.0), b"new", &token, &mut |s| {
            if s == stage {
                t.cancel();
            }
        });
        assert!(
            matches!(r, Err(PublishError::Cancelled)),
            "{stage:?}: {r:?}"
        );
        assert_eq!(fs::read(&dest).unwrap(), old, "{stage:?}");
        assert_eq!(
            fs::read(report_path(&dest)).unwrap(),
            old_report,
            "{stage:?}"
        );
        assert_eq!(
            entries(dir.path()).len(),
            2,
            "{stage:?}: {:?}",
            entries(dir.path())
        );
    }
}

#[test]
fn issue_1314_a_cancel_with_no_previous_artifact_leaves_nothing() {
    for stage in BEFORE_COMMIT {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("plan.pvec");
        let token = CancelToken::new();
        let t = token.clone();
        let r = publish(&dest, &artifact(0.0), b"new", &token, &mut |s| {
            if s == stage {
                t.cancel();
            }
        });
        assert!(matches!(r, Err(PublishError::Cancelled)), "{stage:?}");
        assert!(
            entries(dir.path()).is_empty(),
            "{stage:?}: {:?}",
            entries(dir.path())
        );
    }
}

#[test]
fn issue_1314_a_panic_at_each_step_removes_the_partials_and_keeps_the_previous_pair() {
    for stage in BEFORE_COMMIT {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("plan.pvec");
        let old = artifact(0.0);
        publish(&dest, &old, b"old", &CancelToken::new(), &mut no_stage).unwrap();
        let r = catch_unwind(AssertUnwindSafe(|| {
            let _ = publish(
                &dest,
                &artifact(1.0),
                b"new",
                &CancelToken::new(),
                &mut |s| {
                    assert!(s != stage, "boom at {s:?}");
                },
            );
        }));
        assert!(r.is_err(), "{stage:?} did not panic");
        assert_eq!(fs::read(&dest).unwrap(), old, "{stage:?}");
        assert_eq!(fs::read(report_path(&dest)).unwrap(), b"old", "{stage:?}");
        assert_eq!(
            entries(dir.path()).len(),
            2,
            "{stage:?}: {:?}",
            entries(dir.path())
        );
    }
}

#[test]
fn issue_1314_an_io_failure_leaves_nothing_and_names_its_source() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("no-such-dir").join("plan.pvec");
    let r = publish(
        &dest,
        &artifact(0.0),
        b"r",
        &CancelToken::new(),
        &mut no_stage,
    );
    let err = r.expect_err("the directory is missing");
    assert!(matches!(err, PublishError::Io(_)), "{err:?}");
    assert!(std::error::Error::source(&err).is_some());
    assert!(entries(dir.path()).is_empty());
}

#[test]
fn issue_1314_the_pair_is_both_present_or_both_absent_on_every_path() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("plan.pvec");
    assert!(both_or_neither(&dest));
    let t = CancelToken::new();
    t.cancel();
    let _ = publish(&dest, &artifact(0.0), b"r", &t, &mut no_stage);
    assert!(both_or_neither(&dest));
    let _ = publish(&dest, b"garbage", b"r", &CancelToken::new(), &mut no_stage);
    assert!(both_or_neither(&dest));
    publish(
        &dest,
        &artifact(0.0),
        b"r",
        &CancelToken::new(),
        &mut no_stage,
    )
    .unwrap();
    assert!(both_or_neither(&dest) && dest.exists());
}

/// The child half of the kill test: publishes and aborts the process (no unwinding, no `Drop`) at
/// the stage named in the environment. Does nothing when run on its own.
#[test]
fn issue_1314_kill_child() {
    let (Ok(at), Ok(dest)) = (std::env::var(KILL_ENV), std::env::var(DEST_ENV)) else {
        return;
    };
    let _ = publish(
        Path::new(&dest),
        &artifact(1.0),
        b"new",
        &CancelToken::new(),
        &mut |s| {
            if format!("{s:?}") == at {
                std::process::abort();
            }
        },
    );
}

fn kill_at(stage: &str, dest: &Path) {
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "issue_1314_kill_child", "--nocapture"])
        .env(KILL_ENV, stage)
        .env(DEST_ENV, dest)
        .status()
        .unwrap();
    assert!(!status.success(), "the child was meant to die at {stage}");
}

#[test]
fn issue_1314_a_kill_before_the_commit_leaves_no_file_at_the_destination() {
    for stage in BEFORE_COMMIT {
        let dir = tempfile::tempdir().unwrap();
        let dest: PathBuf = dir.path().join("plan.pvec");
        kill_at(&format!("{stage:?}"), &dest);
        assert!(!dest.exists(), "{stage:?}: the artifact path exists");
        assert!(
            !report_path(&dest).exists(),
            "{stage:?}: the report path exists"
        );
        for left in entries(dir.path()) {
            assert!(
                left.ends_with(".pvec.partial") || left.ends_with(".report.json.partial"),
                "{stage:?}: {left}"
            );
        }
    }
}

#[test]
fn issue_1314_a_rerun_after_a_kill_writes_the_bytes_a_clean_run_writes() {
    let clean_dir = tempfile::tempdir().unwrap();
    let clean = clean_dir.path().join("plan.pvec");
    publish(
        &clean,
        &artifact(1.0),
        b"new",
        &CancelToken::new(),
        &mut no_stage,
    )
    .unwrap();

    for stage in BEFORE_COMMIT {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("plan.pvec");
        kill_at(&format!("{stage:?}"), &dest);
        publish(
            &dest,
            &artifact(1.0),
            b"new",
            &CancelToken::new(),
            &mut no_stage,
        )
        .unwrap();
        assert_eq!(
            fs::read(&dest).unwrap(),
            fs::read(&clean).unwrap(),
            "{stage:?}"
        );
        assert_eq!(
            entries(dir.path()),
            entries(clean_dir.path()),
            "{stage:?}: a stale partial survived"
        );
    }
}
