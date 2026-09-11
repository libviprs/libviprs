//! Holds `tools/local-ci.py` to two promises it has to keep before several
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
//! exists for that and deliberately does no YAML parsing, so it works in the CI
//! image, which carries python3 without PyYAML.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

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
        self.fields
            .get(key)
            .unwrap_or_else(|| panic!("--print-docker-argv printed no `{key}` line; got {:?}", self.fields.keys().collect::<Vec<_>>()))
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
}

fn run(args: &[&str], env: &[(&str, &str)]) -> Argv {
    let mut cmd = Command::new("python3");
    cmd.arg(tool()).args(args).arg("--print-docker-argv");
    // Start from a known state rather than inheriting the developer's shell,
    // so a test asserting what happens WITH `DOCKER_DEFAULT_PLATFORM` set is
    // not silently also true without it.
    cmd.env_remove("DOCKER_DEFAULT_PLATFORM");
    cmd.env_remove("LIBVIPRS_CI_VOLUME");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd
        .output()
        .expect("python3 is required to run the container gate's own tests");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "local-ci.py {args:?} --print-docker-argv failed: {}\n{stdout}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut fields = HashMap::new();
    for line in stdout.lines() {
        let mut parts = line.split('\t');
        if let Some(key) = parts.next() {
            fields.insert(key.to_string(), parts.map(str::to_string).collect());
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

/// The host's platform string, derived the same way the tool has to derive it.
fn host_platform() -> String {
    let arch = String::from_utf8(
        Command::new("uname")
            .arg("-m")
            .output()
            .expect("uname")
            .stdout,
    )
    .expect("utf8");
    let arch = arch.trim();
    let docker_arch = match arch {
        "x86_64" | "amd64" => "amd64",
        "aarch64" | "arm64" => "arm64",
        other => panic!("this test has no mapping for host arch {other}"),
    };
    format!("linux/{docker_arch}")
}

#[test]
fn native_states_the_host_platform_on_both_docker_calls() {
    let a = run(&["--native"], &[]);
    let host = host_platform();
    assert_eq!(a.one("platform"), host);
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
}

#[test]
fn the_emulated_default_still_pins_amd64() {
    let a = run(&[], &[]);
    assert_eq!(a.one("platform"), "linux/amd64");
    assert_eq!(a.platform_arg("build"), Some("linux/amd64"));
    assert_eq!(a.platform_arg("run"), Some("linux/amd64"));
}

/// The actual reported bug. `--native` has to mean the host even when the
/// ambient variable says otherwise, and this is the one case the old code got
/// wrong, so it is also the negative control for the test above it: that one
/// passes without the variable set, and this one proves the variable is not
/// what was carrying it.
#[test]
fn native_ignores_docker_default_platform() {
    let a = run(&["--native"], &[("DOCKER_DEFAULT_PLATFORM", "linux/amd64")]);
    let host = host_platform();
    assert_eq!(
        a.one("platform"),
        host,
        "DOCKER_DEFAULT_PLATFORM must not decide what --native means"
    );
    assert_eq!(a.platform_arg("build"), Some(host.as_str()));
    assert_eq!(a.platform_arg("run"), Some(host.as_str()));
}

#[test]
fn the_default_cargo_volume_is_unchanged() {
    let a = run(&[], &[]);
    assert_eq!(a.one("volume"), "libviprs-ci-cargo");
    assert!(
        a.get("run").windows(2).any(|w| w[0] == "-v" && w[1] == "libviprs-ci-cargo:/cargo"),
        "the default volume must still be mounted at /cargo: {:?}",
        a.get("run")
    );
}

#[test]
fn a_lane_can_name_its_own_volume() {
    let flag = run(&["--volume", "libviprs-ci-f11"], &[]);
    assert_eq!(flag.one("volume"), "libviprs-ci-f11");
    assert!(
        flag.get("run").windows(2).any(|w| w[0] == "-v" && w[1] == "libviprs-ci-f11:/cargo"),
        "the named volume must reach the mount: {:?}",
        flag.get("run")
    );

    let env = run(&[], &[("LIBVIPRS_CI_VOLUME", "libviprs-ci-from-env")]);
    assert_eq!(env.one("volume"), "libviprs-ci-from-env");

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

/// Two lanes picking different volumes have to end up with different commands,
/// which is the whole point of the flag. Asserting only that the flag is echoed
/// back would pass on a tool that parses `--volume`, prints it, and then mounts
/// the default anyway, so this reads the run command itself and requires the two
/// to differ.
#[test]
fn different_volumes_do_not_share_a_target_dir() {
    let a = run(&["--volume", "libviprs-ci-a"], &[]);
    let b = run(&["--volume", "libviprs-ci-b"], &[]);
    assert_ne!(
        a.get("run").iter().position(|x| x == "libviprs-ci-a:/cargo"),
        None
    );
    assert_ne!(
        b.get("run").iter().position(|x| x == "libviprs-ci-b:/cargo"),
        None
    );
    assert_ne!(a.get("run"), b.get("run"));
}

/// The other half of moving the yaml import, and the reason this test exists at
/// all: I moved it out of module scope and forgot to put it back inside
/// `build_plan`, so every real run died with `NameError: name 'yaml' is not
/// defined` while all seven tests above stayed green, because not one of them
/// asks the tool to parse a workflow.
///
/// It cannot simply assert that `--list` works, because the CI image this runs
/// in has no PyYAML and a refusal there is correct. What it can assert is that
/// the two legitimate outcomes are the only outcomes: the listing works, or the
/// tool says PyYAML is missing. A `NameError` is neither, and it is what the
/// mistake produces.
#[test]
fn the_workflow_path_either_works_or_says_pyyaml_is_missing() {
    let out = Command::new("python3")
        .arg(tool())
        .arg("--list")
        .env_remove("DOCKER_DEFAULT_PLATFORM")
        .output()
        .expect("python3 is required to run the container gate's own tests");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("NameError") && !stdout.contains("NameError"),
        "--list raised a NameError, which means an import went missing:\n{stderr}"
    );
    if out.status.success() {
        // The positive control. A listing that succeeded but printed nothing
        // would satisfy the NameError check above without proving the workflow
        // was ever parsed.
        assert!(
            stdout.contains("toolchain="),
            "--list succeeded without listing a job: {stdout}"
        );
    } else {
        assert!(
            stderr.contains("PyYAML is required"),
            "--list failed for a reason other than a missing PyYAML:\n{stderr}"
        );
    }
}

/// `--print-docker-argv` has to work in the CI image, which carries python3 and
/// no PyYAML. If the YAML import is at module scope this refuses to run at all,
/// and the whole file above it becomes untestable in the place it matters most.
#[test]
fn printing_the_argv_does_not_need_pyyaml() {
    let out = Command::new("python3")
        .arg("-c")
        .arg(
            "import sys, runpy, builtins\n\
             real = builtins.__import__\n\
             def blocked(name, *a, **k):\n\
             \x20   if name == 'yaml': raise ImportError('PyYAML blocked by the test')\n\
             \x20   return real(name, *a, **k)\n\
             builtins.__import__ = blocked\n\
             sys.argv = ['local-ci.py', '--native', '--print-docker-argv']\n\
             runpy.run_path(sys.argv[0] if False else r'''TOOL''', run_name='__main__')\n"
                .replace("TOOL", tool().to_str().expect("utf8 path"))
                .as_str(),
        )
        .env_remove("DOCKER_DEFAULT_PLATFORM")
        .output()
        .expect("python3");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.lines().any(|l| l.starts_with("platform\t")),
        "--print-docker-argv must work without PyYAML.\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
