//! The native DWG cells run on a schedule, as a local-gate entry (issue #1295).
//!
//! Hosted Actions are off in every repo, so the "hosted lane" the issue asked
//! for cannot exist. What I do instead is keep the job in `.github/workflows/`
//! (the job list `tools/local-ci.py` reads), give it one script that both the
//! runner and the scheduler call, and put the scheduling on MARS. This file
//! holds those pieces together so the lane cannot quietly stop being one:
//!
//! * the workflow names no command of its own, it calls `tools/native-dwg-cell.sh`;
//! * every test file that insists on the native archive is run by that script;
//! * the archive is pinned by URL and SHA-256 in a reviewable file;
//! * `make native-dwg` is the entry point, and `make ci` stays off the network;
//! * the MARS scheduler and its cron entry exist and only drive docker.

fn workflow() -> String {
    read(".github/workflows/native-dwg.yml")
}
fn cell() -> String {
    read("tools/native-dwg-cell.sh")
}
fn scheduler() -> String {
    read("tools/scheduled-native-dwg.sh")
}
fn cron() -> String {
    read("tools/native-dwg.cron")
}
fn makefile() -> String {
    read("Makefile")
}
fn contributing() -> String {
    read("CONTRIBUTING.md")
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn pin(key: &str) -> String {
    read("tools/acadsharp-native.pin")
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
        .unwrap_or_else(|| panic!("tools/acadsharp-native.pin has no `{key}=` line"))
        .trim()
        .to_owned()
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn the_workflow_is_scheduled_and_calls_the_one_script() {
    assert!(
        workflow().contains("schedule:"),
        "the lane has to carry a schedule"
    );
    assert!(
        workflow().contains("cron:"),
        "the schedule needs a cron line"
    );
    assert!(workflow().contains("workflow_dispatch:"));
    assert!(
        workflow().contains("name: Native DWG cells"),
        "one job, named Native DWG cells"
    );
    let wf = workflow();
    let runs: Vec<&str> = wf
        .lines()
        .filter_map(|l| {
            l.trim_start()
                .strip_prefix("- run:")
                .or_else(|| l.trim_start().strip_prefix("run:"))
        })
        .map(str::trim)
        .collect();
    assert_eq!(
        runs,
        ["tools/native-dwg-cell.sh"],
        "the workflow must not spell out commands, the script is the single copy"
    );
    for l in wf.lines() {
        assert!(
            !(l.starts_with("    if:") || l.starts_with("    needs:")),
            "a job-level condition is HELD by the runner and would not run: {l}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn the_pin_names_the_release_and_a_sha256() {
    assert!(
        pin("url")
            .starts_with("https://github.com/libviprs/libviprs-dep/releases/download/acadsharp-")
    );
    assert!(pin("url").ends_with("/acadsharp-linux-x64.tgz"));
    let sha = pin("sha256");
    assert_eq!(sha.len(), 64);
    assert!(sha.bytes().all(|b| b.is_ascii_hexdigit()));
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn the_cell_script_verifies_the_archive_and_requires_it() {
    let cell = cell();
    for needle in [
        "acadsharp-native.pin",
        "sha256sum",
        "ACADSHARP_NATIVE_DIR",
        "LIBVIPRS_REQUIRE_ACADSHARP_NATIVE=1",
        "--features dwg-acadsharp",
    ] {
        assert!(
            cell.contains(needle),
            "tools/native-dwg-cell.sh must contain `{needle}`"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // lists a directory, which Miri's isolation refuses
fn every_test_that_requires_the_native_archive_is_run_by_the_cell() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut needing = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "rs")
            && std::fs::read_to_string(&p)
                .unwrap()
                .contains("LIBVIPRS_REQUIRE_ACADSHARP_NATIVE")
        {
            needing.push(p.file_stem().unwrap().to_string_lossy().into_owned());
        }
    }
    let cell = cell();
    assert!(
        needing.len() >= 6,
        "found {needing:?}, the scan is reading nothing"
    );
    for t in needing {
        assert!(
            cell.contains(&format!("--test {t}")),
            "{t} insists on the native archive and the scheduled cell does not run it"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn make_native_dwg_is_the_entry_and_make_ci_stays_off_the_network() {
    let mk = makefile();
    let target: Vec<&str> = mk
        .lines()
        .skip_while(|l| !l.starts_with("native-dwg:"))
        .skip(1)
        .take_while(|l| l.starts_with('\t'))
        .collect();
    assert!(
        target
            .iter()
            .any(|l| l.contains("tools/local-ci.py --workflow native-dwg.yml")),
        "`make native-dwg` has to hand the workflow to the runner: {target:?}"
    );
    let ci: Vec<&str> = mk
        .lines()
        .skip_while(|l| !l.starts_with("ci:"))
        .skip(1)
        .take_while(|l| l.starts_with('\t'))
        .collect();
    assert!(
        !ci.iter().any(|l| l.contains("native-dwg")),
        "the scheduled lane downloads an archive, so `make ci` must not run it"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn the_mars_scheduler_only_drives_docker_and_has_a_cron_entry() {
    assert!(scheduler().contains("docker run --rm"));
    assert!(scheduler().contains("tools/native-dwg-cell.sh"));
    let sched = scheduler();
    for l in sched.lines().map(str::trim_start) {
        assert!(
            !(l.starts_with("cargo ") || l.starts_with("rustup ") || l.starts_with("curl ")),
            "the scheduler runs on a shared host and may only call docker: {l}"
        );
    }
    let cron = cron();
    let entry = cron
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .expect("tools/native-dwg.cron needs an entry");
    assert!(
        entry.split_whitespace().count() >= 6,
        "five time fields and a command: {entry}"
    );
    assert!(entry.contains("tools/scheduled-native-dwg.sh"));
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn contributing_documents_the_lane_and_the_install_step() {
    let doc = contributing();
    for needle in [
        "make native-dwg",
        "tools/scheduled-native-dwg.sh",
        "tools/native-dwg.cron",
    ] {
        assert!(
            doc.contains(needle),
            "CONTRIBUTING.md must mention `{needle}`"
        );
    }
}
