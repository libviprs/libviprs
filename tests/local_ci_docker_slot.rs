//! `tools/local-ci.py` takes a machine-wide Docker slot before it talks to the
//! daemon (libviprs/libviprs-tests#248).
//!
//! Ten lanes gating at once on one laptop filled Docker Desktop's whole disk,
//! and from then on every hook that touched Docker died with `No space left on
//! device`. This tool is what the pre-commit hook here runs, and it started an
//! image build and a container per job with nothing limiting how many ran at
//! once. It takes a slot from the same pool libviprs-tests' run-tests.sh uses,
//! by the same rules: `slotN/pid` directories under `LIBVIPRS_DOCKER_SLOT_DIR`,
//! `LIBVIPRS_DOCKER_MAX_PARALLEL` of them, a dead holder's slot taken over, and
//! `LIBVIPRS_DOCKER_SLOT` naming a slot the caller already holds.
//!
//! These drive the real `main` with a stub `yaml` (so they need no PyYAML) and
//! a stub `docker` first on PATH whose `info` fails, so a run stops right after
//! the point the slot is taken. The stub logs every call, which is how a row
//! tells "waited for a slot" from "went straight to the daemon".

use std::path::PathBuf;
use std::process::{Command, Output};

fn tool() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/local-ci.py")
}

/// A temp dir holding a stub `docker` and a slot pool.
struct SlotSandbox {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl SlotSandbox {
    fn new() -> SlotSandbox {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical temp path");
        std::fs::create_dir_all(root.join("bin")).expect("bin dir");
        std::fs::create_dir_all(root.join("slots")).expect("slot dir");
        let docker = root.join("bin/docker");
        std::fs::write(
            &docker,
            "#!/bin/sh\n\
             echo \"$1 slot=${LIBVIPRS_DOCKER_SLOT:-}\" >> \"$STUB_DOCKER_LOG\"\n\
             exit 1\n",
        )
        .expect("write the stub docker");
        let mut perms = std::fs::metadata(&docker).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&docker, perms).expect("chmod");
        SlotSandbox { _dir: dir, root }
    }

    fn slots(&self) -> PathBuf {
        self.root.join("slots")
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.root.join("docker.log")).unwrap_or_default()
    }

    fn slot_names(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(self.slots())
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }

    /// `local-ci.py <args>` over a one-job stub plan, against this sandbox.
    fn command(&self, args: &[&str]) -> Command {
        let program = format!(
            r#"
import sys, types, runpy
PLAN = {{'jobs': {{'probe': {{'name': 'Stub Probe',
    'steps': [{{'uses': 'dtolnay/rust-toolchain@1.2.3'}}, {{'run': 'true'}}]}}}}}}
stub = types.ModuleType('yaml')
stub.safe_load = lambda handle: PLAN
sys.modules['yaml'] = stub
sys.argv = ['local-ci.py'] + sys.argv[1:]
runpy.run_path(r'''{tool}''', run_name='__main__')
"#,
            tool = tool().to_str().expect("utf8 path")
        );
        let path = format!(
            "{}:{}",
            self.root.join("bin").display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let mut cmd = Command::new("python3");
        cmd.arg("-c")
            .arg(program)
            .args(args)
            .env("PATH", path)
            .env("STUB_DOCKER_LOG", self.root.join("docker.log"))
            .env("LIBVIPRS_DOCKER_SLOT_DIR", self.slots())
            .env("LIBVIPRS_DOCKER_SLOT_POLL", "0.1")
            .env_remove("LIBVIPRS_DOCKER_SLOT")
            .env_remove("LIBVIPRS_DOCKER_MAX_PARALLEL")
            .env_remove("DOCKER_DEFAULT_PLATFORM")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        cmd
    }
}

fn finish_within(mut child: std::process::Child, secs: u64, what: &str) -> Output {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    loop {
        if child.try_wait().expect("poll").is_some() {
            return child.wait_with_output().expect("output");
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let out = child.wait_with_output().expect("output");
            panic!(
                "{what} did not finish within {secs}s\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// With every slot held by a live process, the run waits before it asks the
/// daemon for anything, and goes ahead (taking the slot over) once the holder
/// is gone.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3 and a holder process (#714)
fn a_run_waits_for_a_free_docker_slot() {
    let sb = SlotSandbox::new();
    let mut holder = Command::new("sleep")
        .arg("60")
        .spawn()
        .expect("spawn a slot holder");
    let slot = sb.slots().join("slot1");
    std::fs::create_dir_all(&slot).expect("create slot1");
    std::fs::write(slot.join("pid"), format!("{}\n", holder.id())).expect("write the holder pid");

    let child = sb
        .command(&["--fast"])
        .env("LIBVIPRS_DOCKER_MAX_PARALLEL", "1")
        .spawn()
        .expect("spawn local-ci.py");
    std::thread::sleep(std::time::Duration::from_secs(3));
    let early = sb.log();
    let _ = holder.kill();
    let _ = holder.wait();
    let out = finish_within(
        child,
        60,
        "local-ci.py behind a slot whose holder went away",
    );

    assert!(
        !early.lines().any(|l| l.starts_with("info")),
        "with LIBVIPRS_DOCKER_MAX_PARALLEL=1 and the only slot held by a live \
         process, local-ci.py went to the daemon anyway:\n{early}\n\nNothing \
         capped how many Docker gates ran at once, which is how ten lanes filled \
         the Docker disk (libviprs/libviprs-tests#248).\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        sb.log().lines().any(|l| l.starts_with("info")),
        "once the holder was gone the run should have taken the slot over and \
         gone on to the daemon.\ndocker calls: {}\nstdout: {}\nstderr: {}",
        sb.log(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        sb.slot_names().is_empty(),
        "the run is over, so its slot should be released: {:?}",
        sb.slot_names()
    );
}

/// The slot comes from the shared pool, and what the run starts is told which
/// one it holds, so run-tests.sh or another hook further down does not queue
/// for a second slot.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3 (#714)
fn the_docker_slot_comes_from_the_shared_pool_and_is_handed_down() {
    let sb = SlotSandbox::new();
    let out = finish_within(
        sb.command(&["--fast"]).spawn().expect("spawn local-ci.py"),
        60,
        "local-ci.py with a free slot",
    );
    let info = sb
        .log()
        .lines()
        .find(|l| l.starts_with("info"))
        .map(str::to_owned)
        .unwrap_or_default();
    let want = format!("slot={}", sb.slots().join("slot1").display());
    assert!(
        info.ends_with(&want),
        "`docker info` ran as {info:?}, expected it under {want:?}.\n\nThe slot has \
         to come from LIBVIPRS_DOCKER_SLOT_DIR, the pool every Docker-starting \
         hook shares, and the run has to export LIBVIPRS_DOCKER_SLOT to what it \
         starts.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        sb.slot_names().is_empty(),
        "the run is over, so its slot should be released: {:?}",
        sb.slot_names()
    );
}

/// A caller that already holds a slot says so in `LIBVIPRS_DOCKER_SLOT`. The
/// run is already counted, and with one slot, queueing for another would wait
/// on its own caller forever.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3 (#714)
fn a_run_under_a_caller_holding_a_slot_does_not_queue_for_another() {
    let sb = SlotSandbox::new();
    let slot = sb.slots().join("slot1");
    std::fs::create_dir_all(&slot).expect("create slot1");
    std::fs::write(slot.join("pid"), format!("{}\n", std::process::id())).expect("write pid");

    let out = finish_within(
        sb.command(&["--fast"])
            .env("LIBVIPRS_DOCKER_MAX_PARALLEL", "1")
            .env("LIBVIPRS_DOCKER_SLOT", &slot)
            .spawn()
            .expect("spawn local-ci.py"),
        60,
        "local-ci.py under a caller holding the only slot",
    );
    assert!(
        sb.log().lines().any(|l| l.starts_with("info")),
        "the run never reached the daemon.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(slot.join("pid"))
            .unwrap_or_default()
            .trim(),
        std::process::id().to_string(),
        "the run released a slot that belongs to its caller"
    );
}

/// `--list` and `--print-docker-argv` start nothing, so they take no slot and
/// never wait for one: they are what somebody runs to see what a gate would do.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3 (#714)
fn listing_and_printing_take_no_docker_slot() {
    let sb = SlotSandbox::new();
    let slot = sb.slots().join("slot1");
    std::fs::create_dir_all(&slot).expect("create slot1");
    std::fs::write(slot.join("pid"), format!("{}\n", std::process::id())).expect("write pid");
    for args in [&["--list"][..], &["--print-docker-argv"][..]] {
        let out = finish_within(
            sb.command(args)
                .env("LIBVIPRS_DOCKER_MAX_PARALLEL", "1")
                .spawn()
                .expect("spawn local-ci.py"),
            30,
            "a listing run with every slot taken",
        );
        assert!(
            out.status.success(),
            "{args:?} failed\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert_eq!(sb.slot_names(), vec!["slot1".to_owned()]);
}
