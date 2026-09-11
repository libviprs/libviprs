//! Holds `tools/local-ci.py` to the promises it has to keep before several
//! worktrees can run the container gate at the same time (issue #994).
//!
//! # The platform promise
//!
//! `--native` is meant to run the gate on the host architecture. It used to do
//! that by *omitting* `--platform` and letting Docker pick, and omitting the
//! flag does not mean "the host", it means "whatever `DOCKER_DEFAULT_PLATFORM`
//! says". That variable is `linux/amd64` in the shell this repository is
//! developed in, so `--native` asked the daemon for an amd64 variant of a local
//! arm64 image and got:
//!
//! ```text
//! Unable to find image 'libviprs-ci:native' locally
//! docker: Error response from daemon: pull access denied for libviprs-ci,
//! repository does not exist or may require 'docker login': denied: ...
//! ```
//!
//! The image was sitting right there. Nothing in that message says "platform",
//! it says authentication and a missing repository, so the obvious reading is a
//! login problem or a typo in the tag. The fix is for the tool to *state* the
//! platform it wants on both the build and the run rather than inheriting one,
//! and these tests are what stop the flag quietly going away again.
//!
//! The poison value matters. This file used to set `DOCKER_DEFAULT_PLATFORM` to
//! `linux/amd64` and assert the answer equalled the host, which on an amd64 host
//! is `linux/amd64 == linux/amd64` and passes with the original bug fully
//! restored. CI is `ubuntu-latest`, x86_64, so that was every run CI ever made.
//! `linux/386` is a platform no host here can be, so the row discriminates
//! wherever it runs.
//!
//! # Who gets asked what the host architecture is
//!
//! `platform.machine()` reports the architecture of the *interpreter*. Measured
//! on this host: `uname -m` is `arm64`, `arch -x86_64 /usr/bin/uname -m` is
//! `x86_64`, and an x86_64 python3 is exactly what a shell configured to prefer
//! amd64 tends to run. So sourcing the answer from the interpreter makes
//! `--native` pin `linux/amd64` and call it the host, which is #994 again with a
//! different origin. The tool asks the daemon now and falls back to
//! `platform.machine()` only when there is nothing to ask, and it says which one
//! answered in the `arch-source` field.
//!
//! That makes the expectation here awkward in a useful way. Deriving it from
//! `uname -m` would be the tool agreeing with a copy of its own logic, so the
//! rows below take it from the daemon where there is one, and
//! [`the_daemon_beats_the_interpreter_when_they_disagree`] is what covers the
//! preference itself: on this host both sources say arm64, so nothing that only
//! compares them can fail.
//!
//! # The volume promise
//!
//! The cargo volume used to be a module constant, so every worktree on the
//! machine shared one `CARGO_TARGET_DIR`. Concurrent lanes do not corrupt each
//! other, cargo's lock sees to that, but each one invalidates the other's build
//! cache because the sources under the mount changed underneath it. With five
//! lanes that is most of the gate spent recompiling. `--volume` and
//! `LIBVIPRS_CI_VOLUME` give a lane its own, and the default has to stay
//! exactly what it was so existing use is unchanged.
//!
//! # Why these shell out instead of reading the source
//!
//! A grep over the Python would pass on a `--platform` that is built and then
//! dropped, or on a flag that is parsed and never used. Asking the tool what it
//! would actually run tests the thing the daemon sees. `--print-docker-argv`
//! exists for that and deliberately parses no YAML, so it works in the CI image,
//! which carries python3 without PyYAML and without a docker CLI at all.
//!
//! What it prints is the command that runs, not a restatement of it. `main`
//! used to slice a prefix apart with `run_prefix.index("-v")` and reassemble it
//! thirty lines later, so the printed `run` line carried no source mount, no
//! working directory and no image, and `--worktree --print-docker-argv` printed
//! exactly what `--print-docker-argv` printed while the real `--worktree` run
//! carried a different mount. A reviewer changed that reassembly to
//! `run_prefix[:3] + mounts`, dropping `--platform` from the real run, and every
//! test here stayed green. [`the_run_command_is_the_whole_command_not_a_prefix`]
//! and [`the_worktree_flag_changes_the_command_that_runs`] are the rows that
//! close it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Output};

fn tool() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/local-ci.py")
}

/// One `--print-docker-argv` run, parsed into its labelled lines.
///
/// The format is tab separated because docker arguments never contain a tab,
/// which keeps this free of a quoting round trip that could disagree with the
/// tool's own idea of where one argument ends and the next begins.
struct Argv {
    fields: HashMap<String, Vec<String>>,
}

impl Argv {
    fn get(&self, key: &str) -> &[String] {
        self.fields.get(key).unwrap_or_else(|| {
            panic!(
                "--print-docker-argv printed no `{key}` line; got {:?}",
                self.fields.keys().collect::<Vec<_>>()
            )
        })
    }

    fn one(&self, key: &str) -> &str {
        let v = self.get(key);
        assert_eq!(v.len(), 1, "`{key}` should be a single value, got {v:?}");
        &v[0]
    }

    /// The value Docker would receive for `--platform`, or `None` if the flag
    /// is absent. Absent is the bug this file exists for, so it is a distinct
    /// answer rather than an empty string.
    fn platform_arg(&self, key: &str) -> Option<&str> {
        let argv = self.get(key);
        let i = argv.iter().position(|a| a == "--platform")?;
        Some(
            argv.get(i + 1)
                .unwrap_or_else(|| panic!("`{key}` ends with a bare --platform: {argv:?}"))
                .as_str(),
        )
    }

    /// What `run` mounts at `dest`, if anything.
    fn mount_at(&self, dest: &str) -> Option<&str> {
        self.get("run").windows(2).find_map(|w| {
            let spec = w[1].as_str();
            (w[0] == "-v" && spec.split(':').nth(1) == Some(dest)).then_some(spec)
        })
    }
}

/// One raw invocation, with the ambient environment cleared to a known state.
///
/// `DOCKER_DEFAULT_PLATFORM` and `LIBVIPRS_CI_VOLUME` are removed rather than
/// inherited, so a row asserting what happens *with* one of them set is not
/// silently also true without it.
fn run_raw(args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new("python3");
    cmd.arg(tool()).args(args).arg("--print-docker-argv");
    cmd.env_remove("DOCKER_DEFAULT_PLATFORM");
    cmd.env_remove("LIBVIPRS_CI_VOLUME");
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output()
        .expect("python3 is required to run the container gate's own tests")
}

fn run(args: &[&str], env: &[(&str, &str)]) -> Argv {
    let out = run_raw(args, env);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "local-ci.py {args:?} --print-docker-argv failed: {}\n{stdout}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut fields: HashMap<String, Vec<String>> = HashMap::new();
    for line in stdout.lines() {
        let mut parts = line.split('\t');
        if let Some(key) = parts.next() {
            let value: Vec<String> = parts.map(str::to_string).collect();
            // Refusing a duplicate rather than keeping the last is the other
            // half of the volume-name validation. A name carrying a tab and a
            // newline used to inject a second `platform` line into this output,
            // and an `insert` that overwrites silently prefers the injected one.
            assert!(
                fields.insert(key.to_string(), value).is_none(),
                "--print-docker-argv printed two `{key}` lines, so something \
                 injected a field into the tab-separated output:\n{stdout}"
            );
        }
    }
    // The positive control for every assertion below: a run that printed
    // nothing at all would otherwise fail with "no `build` line", which reads
    // like a missing feature rather than a tool that did not run.
    assert!(
        !fields.is_empty(),
        "--print-docker-argv printed nothing for {args:?}"
    );
    Argv { fields }
}

/// `linux/<arch>` for an architecture spelling, matching the tool's own table.
fn docker_platform(arch: &str) -> String {
    let mapped = match arch.trim() {
        "x86_64" | "amd64" => "amd64",
        "aarch64" | "arm64" => "arm64",
        other => panic!("this test has no mapping for architecture {other}"),
    };
    format!("linux/{mapped}")
}

/// What the Docker daemon says the host is, or `None` when there is none to ask.
///
/// This is the authority the tool is supposed to use, and asking it here is the
/// point: `uname -m` is the same source as `platform.machine()` with the same
/// defect, so an expectation derived from it is the tool agreeing with a copy of
/// its own logic.
fn daemon_platform() -> Option<String> {
    let out = Command::new("docker")
        .args(["version", "--format", "{{.Server.Arch}}"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let arch = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!arch.is_empty()).then(|| docker_platform(&arch))
}

/// What `platform.machine()` maps to, which is the tool's *fallback* source.
fn interpreter_platform() -> String {
    let out = Command::new("python3")
        .args(["-c", "import platform; print(platform.machine())"])
        .output()
        .expect("python3");
    docker_platform(&String::from_utf8_lossy(&out.stdout))
}

/// The platform `--native` must produce here, and the source the tool must name.
///
/// Two arms, and they are not equally strong. Where a daemon answers, the
/// expectation comes from it and the row is a real cross-check. Where none does
/// (the CI image, which has no docker CLI), the tool falls back to
/// `platform.machine()` and so does this, which proves nothing about the
/// preference by itself. Pinning the source is what keeps that honest: the weak
/// arm cannot be taken silently on a host that has a daemon, and
/// [`the_daemon_beats_the_interpreter_when_they_disagree`] covers the preference
/// on every host.
fn expected_native() -> (String, &'static str) {
    match daemon_platform() {
        Some(p) => (p, "daemon"),
        None => (interpreter_platform(), "platform.machine"),
    }
}

#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn native_states_the_host_platform_on_both_docker_calls() {
    let a = run(&["--native"], &[]);
    let (host, source) = expected_native();
    assert_eq!(a.one("platform"), host);
    assert_eq!(
        a.one("arch-source"),
        source,
        "the tool must say where it got the host architecture, and a daemon \
         that answers must be the one that did"
    );
    assert_eq!(
        a.platform_arg("build"),
        Some(host.as_str()),
        "the --native build must name its platform, not inherit one"
    );
    assert_eq!(
        a.platform_arg("run"),
        Some(host.as_str()),
        "the --native run must name its platform, not inherit one"
    );
    // Nothing read the image before, so tagging every build `libviprs-ci:local`
    // left the whole suite green while --native mislabelled an arm64 build.
    assert_eq!(a.one("image"), "libviprs-ci:native");
    assert!(
        a.get("build")
            .windows(2)
            .any(|w| w[0] == "-t" && w[1] == "libviprs-ci:native"),
        "the build must tag the native image: {:?}",
        a.get("build")
    );
    assert_eq!(
        a.get("run").last().map(String::as_str),
        Some("libviprs-ci:native"),
        "the run must end with the image it is going to run: {:?}",
        a.get("run")
    );
}

#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn the_emulated_default_still_pins_amd64() {
    let a = run(&[], &[]);
    assert_eq!(a.one("platform"), "linux/amd64");
    assert_eq!(
        a.one("arch-source"),
        "pinned",
        "the default is a decision, not a lookup, and it should say so"
    );
    assert_eq!(a.platform_arg("build"), Some("linux/amd64"));
    assert_eq!(a.platform_arg("run"), Some("linux/amd64"));
    assert_eq!(a.one("image"), "libviprs-ci:local");
    assert_eq!(
        a.get("run").last().map(String::as_str),
        Some("libviprs-ci:local")
    );
}

/// The actual reported bug. `--native` has to mean the host even when the
/// ambient variable says otherwise.
///
/// The value is `linux/386` on purpose. With `linux/amd64` this assertion reads
/// `linux/amd64 == linux/amd64` on an amd64 host and passes with the bug fully
/// restored, which is every run CI makes, because `ubuntu-latest` is x86_64. A
/// reviewer proved it by forcing the host arch to `x86_64` and diffing clean
/// against bugged: byte-identical. No host here can be `linux/386`, so this row
/// discriminates wherever it runs.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn native_ignores_docker_default_platform() {
    let a = run(&["--native"], &[("DOCKER_DEFAULT_PLATFORM", "linux/386")]);
    let (host, _) = expected_native();
    assert_eq!(
        a.one("platform"),
        host,
        "DOCKER_DEFAULT_PLATFORM must not decide what --native means"
    );
    for key in ["build", "run"] {
        assert_eq!(a.platform_arg(key), Some(host.as_str()));
        assert!(
            !a.get(key).iter().any(|arg| arg == "linux/386"),
            "the ambient platform reached the {key} command: {:?}",
            a.get(key)
        );
    }
}

/// The daemon is the authority, and this is the only row that can prove it.
///
/// On this host `docker version --format {{.Server.Arch}}` and
/// `platform.machine()` both say arm64, so every row that merely compares the
/// tool's answer against the host passes whichever source it used. This one puts
/// a fake `docker` first on `PATH` that reports the *opposite* architecture to
/// the interpreter's, so the two answers cannot coincide, and then swaps in one
/// that cannot answer at all, to check the fallback is real rather than a crash.
#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn the_daemon_beats_the_interpreter_when_they_disagree() {
    use std::os::unix::fs::PermissionsExt;

    let interpreter = interpreter_platform();
    let opposite = if interpreter == "linux/arm64" {
        "amd64"
    } else {
        "arm64"
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let answering = dir.path().join("answering");
    let silent = dir.path().join("silent");
    std::fs::create_dir(&answering).expect("mkdir");
    std::fs::create_dir(&silent).expect("mkdir");
    // `docker version` answers and nothing else does, so a tool reaching for a
    // daemon call this row does not know about fails loudly rather than quietly
    // picking up a stub answer.
    let fakes = [
        (
            answering.join("docker"),
            format!(
                "#!/bin/sh\nif [ \"$1\" = version ]; then echo {opposite}; exit 0; fi\nexit 1\n"
            ),
        ),
        (silent.join("docker"), "#!/bin/sh\nexit 1\n".to_string()),
    ];
    for (path, body) in &fakes {
        std::fs::write(path, body).expect("write the fake docker");
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }
    let with = |dir: &std::path::Path| {
        format!(
            "{}:{}",
            dir.display(),
            std::env::var("PATH").unwrap_or_default()
        )
    };

    let a = run(&["--native"], &[("PATH", &with(&answering))]);
    assert_eq!(
        a.one("platform"),
        format!("linux/{opposite}"),
        "the daemon said {opposite} and the interpreter said {interpreter}; the \
         daemon is the thing that runs the container, so it wins"
    );
    assert_eq!(a.one("arch-source"), "daemon");
    assert_eq!(
        a.platform_arg("run"),
        Some(format!("linux/{opposite}").as_str())
    );

    // And the fallback has to be a real fallback. `--print-docker-argv` is
    // supposed to work with no Docker at all, which is its state inside the CI
    // image, so a daemon that cannot answer must not take the tool down.
    let b = run(&["--native"], &[("PATH", &with(&silent))]);
    assert_eq!(b.one("arch-source"), "platform.machine");
    assert_eq!(b.one("platform"), interpreter);
}

/// What `--print-docker-argv` prints is the whole `docker run`, not its head.
///
/// The printer used to receive a prefix and print that, so the mounts, the
/// working directory and the image never appeared, and the real command was
/// reassembled from the prefix by index elsewhere. Slicing at `[:3]` instead
/// drops `--platform` from the real run and restored #994 with every test green.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn the_run_command_is_the_whole_command_not_a_prefix() {
    let a = run(&["--native"], &[]);
    let argv = a.get("run");
    let (host, _) = expected_native();

    assert_eq!(
        &argv[..3],
        &["docker".to_string(), "run".to_string(), "--rm".to_string()],
        "the run command must start as a docker run: {argv:?}"
    );
    assert_eq!(a.platform_arg("run"), Some(host.as_str()));
    assert_eq!(
        a.mount_at("/cargo"),
        Some("libviprs-ci-cargo:/cargo"),
        "the cargo volume must be in the command that runs: {argv:?}"
    );
    assert!(
        argv.windows(2).any(|w| w[0] == "-w" && w[1] == "/src"),
        "the run must name the working directory every step resolves against: {argv:?}"
    );
    assert_eq!(argv.last().map(String::as_str), Some("libviprs-ci:native"));
    assert!(
        a.mount_at("/gitsrc/libviprs").is_some(),
        "the default mode clones from a mounted git directory, and that mount is \
         part of the command: {argv:?}"
    );
}

/// `--worktree` is a different command, and it used to print as the same one.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn the_worktree_flag_changes_the_command_that_runs() {
    let git = run(&["--native"], &[]);
    let wt = run(&["--native", "--worktree"], &[]);

    assert_eq!(git.one("mode"), "git");
    assert_eq!(wt.one("mode"), "worktree");
    assert_ne!(
        git.get("run"),
        wt.get("run"),
        "the two modes mount different things, so the commands cannot be equal"
    );

    let manifest = env!("CARGO_MANIFEST_DIR");
    assert_eq!(
        wt.mount_at("/src/libviprs"),
        Some(format!("{manifest}:/src/libviprs").as_str()),
        "--worktree bind-mounts this tree: {:?}",
        wt.get("run")
    );
    assert!(
        wt.mount_at("/gitsrc/libviprs").is_none(),
        "--worktree does not clone from git, so it must not mount the git dir: {:?}",
        wt.get("run")
    );
    assert!(
        git.mount_at("/src/libviprs").is_none(),
        "the default mode checks the tree out inside the container, so it must \
         not bind-mount it: {:?}",
        git.get("run")
    );
}

#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn the_default_cargo_volume_is_unchanged() {
    let a = run(&[], &[]);
    assert_eq!(a.one("volume"), "libviprs-ci-cargo");
    assert_eq!(
        a.mount_at("/cargo"),
        Some("libviprs-ci-cargo:/cargo"),
        "the default volume must still be mounted at /cargo: {:?}",
        a.get("run")
    );
}

#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn a_lane_can_name_its_own_volume() {
    let flag = run(&["--volume", "libviprs-ci-f11"], &[]);
    assert_eq!(flag.one("volume"), "libviprs-ci-f11");
    assert_eq!(
        flag.mount_at("/cargo"),
        Some("libviprs-ci-f11:/cargo"),
        "the named volume must reach the mount: {:?}",
        flag.get("run")
    );

    let env = run(&[], &[("LIBVIPRS_CI_VOLUME", "libviprs-ci-from-env")]);
    assert_eq!(env.one("volume"), "libviprs-ci-from-env");
    assert_eq!(env.mount_at("/cargo"), Some("libviprs-ci-from-env:/cargo"));

    let both = run(
        &["--volume", "libviprs-ci-from-flag"],
        &[("LIBVIPRS_CI_VOLUME", "libviprs-ci-from-env")],
    );
    assert_eq!(
        both.one("volume"),
        "libviprs-ci-from-flag",
        "an explicit flag beats the environment"
    );
}

/// Two lanes picking different volumes must not end up building into the same
/// storage, which is the whole point of the flag.
///
/// This used to assert nothing about a target dir at all: it checked that each
/// run mentioned its own volume and that the two commands differed, which its
/// name did not describe. The claim it should be making is that the container
/// path is the same in both, `/cargo/target-<arch>`, and that the thing mounted
/// underneath it is different, so the same path is different storage. The other
/// half of the promise is per-architecture, since an emulated amd64 run and a
/// native one must not leave each other stale binaries either.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn different_volumes_do_not_share_a_target_dir() {
    let a = run(&["--volume", "libviprs-ci-a"], &[]);
    let b = run(&["--volume", "libviprs-ci-b"], &[]);

    assert_eq!(
        a.get("cargo-target"),
        b.get("cargo-target"),
        "both lanes build at the same path inside the container"
    );
    assert_eq!(
        a.get("cargo-target"),
        ["/cargo/target-amd64", "/cargo/target-amd64-tests"],
        "and that path is under the mount point the volume supplies"
    );
    assert_eq!(a.mount_at("/cargo"), Some("libviprs-ci-a:/cargo"));
    assert_eq!(b.mount_at("/cargo"), Some("libviprs-ci-b:/cargo"));
    assert_ne!(
        a.mount_at("/cargo"),
        b.mount_at("/cargo"),
        "so the shared path has to be backed by different volumes, or the two \
         lanes are still invalidating one another"
    );
    assert_ne!(a.get("run"), b.get("run"));

    // The other axis. Sharing one target dir between an emulated amd64 run and
    // a native one leaves the other architecture's binaries in place and cargo
    // re-runs a stale one, which is what kept --native failing under Rosetta.
    let native = run(&["--native", "--volume", "libviprs-ci-a"], &[]);
    assert_eq!(
        native.get("cargo-target"),
        ["/cargo/target-native", "/cargo/target-native-tests"]
    );
    assert_ne!(native.get("cargo-target"), a.get("cargo-target"));
}

/// An empty name is not a name, and a name Docker cannot take is not one either.
///
/// `os.environ.get(KEY, DEFAULT)` falls back on an absent key and not on an
/// empty one, so `export LIBVIPRS_CI_VOLUME=` and `--volume ''` both got through
/// `docker volume inspect ''` and `docker volume create ''` and died inside
/// `docker run` on `-v :/cargo`, with a message naming neither the flag nor the
/// variable, after paying for the image build.
///
/// The injection row is the same fix seen from the other side. The print format
/// is tab separated, so a volume name carrying a tab and a newline used to add
/// whole fields to it, and the parser's `insert` kept the injected `platform`
/// line rather than the real one.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn an_empty_volume_name_falls_back_and_an_impossible_one_is_refused() {
    for (label, args, env) in [
        ("--volume ''", vec!["--volume", ""], vec![]),
        (
            "LIBVIPRS_CI_VOLUME=",
            vec![],
            vec![("LIBVIPRS_CI_VOLUME", "")],
        ),
    ] {
        let a = run(&args, &env);
        assert_eq!(
            a.one("volume"),
            "libviprs-ci-cargo",
            "{label} must fall back to the shared default"
        );
        assert_eq!(a.mount_at("/cargo"), Some("libviprs-ci-cargo:/cargo"));
    }

    // Not `-leading`: argparse takes that for a flag and refuses it first, so
    // the row would pass without `resolve_volume` existing at all. `_leading`
    // parses fine and is exactly what Docker will not take.
    for bad in ["x\nplatform\tlinux/386", "../etc", "has space", "_leading"] {
        let out = run_raw(&["--volume", bad], &[]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "{bad:?} was accepted as a volume name; stdout:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
        assert!(
            stderr.contains("--volume") && stderr.contains("LIBVIPRS_CI_VOLUME"),
            "the refusal must name the flag and the variable it came from: {stderr}"
        );
    }
}

/// `--print-docker-argv` has to work in the CI image, which carries python3 and
/// no PyYAML. If the YAML import is at module scope this refuses to run at all,
/// and the whole file above it becomes untestable in the place it matters most.
///
/// The blocker needs a control in the same process, and it did not have one.
/// A reviewer replaced the `raise ImportError` with a pass-through so it blocked
/// nothing at all, and this stayed green on a host that has PyYAML, which is
/// every host except the one it is written for. So the same process now also
/// runs `--list`, which must die saying PyYAML is required. If it lists jobs
/// instead, the blocker is not blocking and the row above it means nothing.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn printing_the_argv_does_not_need_pyyaml() {
    let program = format!(
        r#"
import sys, runpy, builtins
TOOL = r'''{tool}'''
real = builtins.__import__
blocked = []
def block(name, *a, **k):
    if name == 'yaml':
        blocked.append(name)
        raise ImportError('PyYAML blocked by the test')
    return real(name, *a, **k)
builtins.__import__ = block

def attempt(argv):
    sys.argv = argv
    try:
        runpy.run_path(TOOL, run_name='__main__')
    except SystemExit as e:
        return e.code
    return None

print('ARGV_EXIT', attempt(['local-ci.py', '--native', '--print-docker-argv']))
print('LIST_EXIT', attempt(['local-ci.py', '--list']))
print('BLOCKED', len(blocked))
"#,
        tool = tool().to_str().expect("utf8 path")
    );
    let out = Command::new("python3")
        .arg("-c")
        .arg(program)
        .env_remove("DOCKER_DEFAULT_PLATFORM")
        .output()
        .expect("python3");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        stdout.lines().any(|l| l.starts_with("platform\t")),
        "--print-docker-argv must work without PyYAML.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("ARGV_EXIT 0"),
        "--print-docker-argv must exit cleanly without PyYAML.\nstdout: {stdout}"
    );
    // The control. Without this the blocker can be inert and nothing says so.
    assert!(
        stdout.contains("LIST_EXIT PyYAML is required"),
        "under the same blocker, --list must refuse for want of PyYAML. If it \
         did not, the blocker blocked nothing and the row above proves nothing.\
         \nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout
            .lines()
            .any(|l| l.starts_with("BLOCKED ") && l != "BLOCKED 0"),
        "the blocker never saw an import of `yaml`: {stdout}"
    );
}

/// The workflow path has to *parse a plan*, not merely get past an import.
///
/// This used to accept either outcome: a listing, or a refusal naming PyYAML.
/// In the gate it only ever took the second arm, because the CI image has no
/// PyYAML, and that arm is satisfied by the tool being arbitrarily broken
/// downstream of the import guard. A reviewer put
/// `raise RuntimeError("build_plan is completely broken")` immediately after the
/// guard and it stayed green.
///
/// So this drives the success branch on every host instead, by putting a stub
/// `yaml` module in `sys.modules` whose `safe_load` returns a plan written here.
/// Everything after the import then has to work: the job filter, the toolchain
/// read off `dtolnay/rust-toolchain@`, the workflow-env and job-env merge, the
/// `if:` hold, the drop of a job with no `run` steps, and the printing itself.
/// The refusal arm keeps its own coverage in
/// [`printing_the_argv_does_not_need_pyyaml`], where it is a control rather than
/// an escape hatch.
#[test]
#[cfg_attr(miri, ignore)] // spawns python3, which Miri supports on no target (#714)
fn the_workflow_path_parses_a_plan_rather_than_only_importing_yaml() {
    let program = format!(
        r#"
import sys, types, runpy
TOOL = r'''{tool}'''
PLAN = {{
    'env': {{'CARGO_TERM_COLOR': 'always'}},
    'jobs': {{
        'probe': {{
            'name': 'Stub Probe',
            'env': {{'RUSTFLAGS': '-Dwarnings'}},
            'steps': [
                {{'uses': 'dtolnay/rust-toolchain@1.2.3'}},
                {{'name': 'a step', 'run': 'cargo probe --all'}},
            ],
        }},
        'gated': {{
            'name': 'Stub Gated',
            'if': "github.ref == 'refs/heads/main'",
            'steps': [{{'run': 'cargo gated'}}],
        }},
        'nothing': {{
            'name': 'Stub Without Run Steps',
            'steps': [{{'uses': 'actions/checkout@v4'}}],
        }},
    }},
}}
stub = types.ModuleType('yaml')
stub.safe_load = lambda handle: PLAN
sys.modules['yaml'] = stub
sys.argv = ['local-ci.py', '--list']
try:
    runpy.run_path(TOOL, run_name='__main__')
except SystemExit as e:
    print('LIST_EXIT', e.code)
"#,
        tool = tool().to_str().expect("utf8 path")
    );
    let out = Command::new("python3")
        .arg("-c")
        .arg(program)
        .env_remove("DOCKER_DEFAULT_PLATFORM")
        .output()
        .expect("python3");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        !stderr.contains("NameError") && !stdout.contains("NameError"),
        "--list raised a NameError, which means an import went missing:\n{stderr}"
    );
    assert!(
        stdout.contains("LIST_EXIT 0"),
        "--list must succeed over a plan it can parse.\nstdout: {stdout}\nstderr: {stderr}"
    );
    for want in [
        // The job, and the toolchain read off the `uses:` line rather than
        // defaulted.
        "[Stub Probe]  toolchain=1.2.3",
        // Workflow env and job env, merged.
        "'CARGO_TERM_COLOR': 'always'",
        "'RUSTFLAGS': '-Dwarnings'",
        // The step itself, which is the thing the whole tool exists to run.
        "$ cargo probe --all",
        // A job-level `if:` is held rather than run or silently dropped.
        "HELD by `if: github.ref == 'refs/heads/main'`",
    ] {
        assert!(
            stdout.contains(want),
            "--list did not report {want:?}, so build_plan did not do its job.\
             \nstdout: {stdout}\nstderr: {stderr}"
        );
    }
    assert!(
        !stdout.contains("Stub Without Run Steps"),
        "a job with no `run` steps has nothing to run here and must be dropped:\n{stdout}"
    );
}
