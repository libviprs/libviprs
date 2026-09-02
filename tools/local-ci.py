#!/usr/bin/env python3
"""Run the CI job list on this machine, in Docker.

The point is that this cannot drift from CI. It carries no copy of the
commands: it reads `.github/workflows/ci.yml` and runs what is actually in
there, with the toolchain each job asks for and the env each job sets. Add a
step to ci.yml and it runs here next time, with no second place to update.

    tools/local-ci.py                    # every job
    tools/local-ci.py --list             # show what would run, run nothing
    tools/local-ci.py --fast             # skip Test and Integration
    tools/local-ci.py Check Docs         # only jobs matching a filter
    tools/local-ci.py --workflow merge-gate.yml   # Miri, Loom, pdfium audit
    tools/local-ci.py --native           # host arch instead of x86_64
    tools/local-ci.py --worktree         # bind-mount the tree (fast, NOT the gate)

Two things cannot run verbatim and are adapted, out loud:

  * `actions/checkout` and the integration job's `git clone` of libviprs-tests.
    Both are provisioned from git instead, per the section below.
  * Nothing else. Any other `${{ }}` expression is a hard error rather than a
    guess, because a silently mis-substituted step is worse than a missing one.

Where the tree in the container comes from, and why it is not a bind mount
------------------------------------------------------------------------

This used to hand the container the working tree with `-v {REPO}:/src/libviprs`
and that is the one thing about the mirror that was never honest. A Docker
Desktop bind mount off an APFS host is case-insensitive, and it carries
untracked files, so the container saw a tree no runner could ever see:

    $ printf 'lower\\n' > castest/probe_case.txt
    $ docker run --rm -v "$PWD/castest:/m" alpine:3 sh -c 'cat /m/PROBE_CASE.txt'
    lower

That is not a curiosity. It is how `main` stayed red for about 55 hours in
#977 and #979: a capture script derived two fixture names that differed only in
case, APFS merged them, one was never committed, and this mirror resolved the
missing uppercase name to the surviving lowercase file and printed PASS on
every run. `ubuntu-latest` could not. `tests/fixture_paths_are_committed.rs`
now guards that one shape from the index, which is the right layer for it, but
it guards one shape and the mirror was blind to the whole class.

So by default nothing is bind-mounted. The repository's git directory goes in
read-only, and the container makes its own checkout of it:

    git clone --shared --no-checkout /gitsrc/libviprs /src/libviprs
    git -C /src/libviprs checkout --detach <rev>

`--shared` means no objects are copied, so this costs about a second even
though the object store is gigabytes. What comes out is a tree on a
case-sensitive filesystem, built from git's byte-exact names, with no untracked
file in it and with the real history and tags, which several tests in this
repository read.

`<rev>` is `git stash create` when the working tree is dirty and `HEAD` when it
is clean. That is deliberate and it is the interesting half: `git stash create`
builds a commit out of the working tree's *tracked* content without touching
the working tree, the index or the stash ref. So local edits are still what
gets checked, which is what anyone running a gate actually wants, and untracked
files still are not, which is what CI would see. Anything excluded gets listed
before the run rather than silently dropped.

The cost is that a dirty tree gets a fresh commit timestamp, so every file in
the checkout has a new mtime and cargo rebuilds this crate (not its
dependencies, which live on the `/cargo` volume and are untouched). On a clean
tree the timestamps come from the commit, so they are identical run to run and
cargo stays warm. `--worktree` puts the old bind mount back for the times when
that rebuild is not worth it, and says loudly that it is not the gate.

On architecture: the image is x86_64, the same as GitHub's ubuntu-latest, so
on an Apple Silicon host Docker emulates it. That is slower than running
native arm64 and it is the right trade, because the differences that matter
are architecture-sensitive. The worked example in this repo is `f32::mul_add`,
which lowers to a libm `fmaf` call on baseline x86-64 and to a single `fmadd`
on aarch64 (issue #581); a native arm64 gate would not see the x86 cost at
all.

The one thing emulation cannot do is the fallible-allocation tests, which
deliberately ask for an allocation large enough that Rosetta cannot reserve
the address space, and the process SIGTRAPs rather than the allocation
failing cleanly. That is a Rosetta limit, not a bug in the code: those
tests pass natively. Use `--native` for them, which trades the x86 fidelity
for a host-architecture run.

A job that does not run is not a job that passed
------------------------------------------------

A skipped job makes the whole run exit non-zero. It used to exit 0 with a note,
and a note is not a gate: the one job that crosses repos reported SKIP and the
run still said "All jobs passed" to anybody who had not cloned the sibling.
`--allow-skips` is there for when a subset is genuinely what you asked for.
"""
import argparse, collections, os, shlex, subprocess, sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip3 install pyyaml")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
TESTS_DIR = os.path.join(WORKSPACE, "libviprs-tests")
IMAGE_AMD64 = "libviprs-ci:local"
IMAGE_NATIVE = "libviprs-ci:native"
VOLUME = "libviprs-ci-cargo"
SLOW = ("test", "integration")

# Where each repository's git directory is mounted, and where its checkout is
# made. The checkout paths are the ones every step's `working-directory`
# resolves against, so they match what ci.yml's `actions/checkout` produces.
CHECKOUT = {"libviprs": "/src/libviprs", "libviprs-tests": "/src/libviprs-tests"}
GITSRC = {"libviprs": "/gitsrc/libviprs", "libviprs-tests": "/gitsrc/libviprs-tests"}


def git(repo, *args, check=True):
    """One git command against `repo`, with its stderr surfaced on failure."""
    out = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True
    )
    if check and out.returncode != 0:
        sys.exit(
            f"git {' '.join(args)} failed in {repo}, so there is nothing to "
            f"provision from:\n{out.stderr.strip()}"
        )
    return out.stdout.strip()


def git_common_dir(repo):
    """The real git directory for `repo`, resolved for linked worktrees.

    A linked worktree's `.git` is a file rather than a directory, and its
    objects live in the main checkout's store, so mounting `repo/.git` would
    mount a one line pointer at a host path the container cannot follow.
    `--git-common-dir` is the thing that is always a real directory holding the
    objects and the refs.
    """
    d = git(repo, "rev-parse", "--git-common-dir")
    return d if os.path.isabs(d) else os.path.abspath(os.path.join(repo, d))


def source_rev(repo):
    """The commit whose tree a push from `repo` right now would carry.

    `git stash create` writes a commit for the working tree's tracked content
    and prints its sha, without touching the working tree, the index or the
    stash ref. It prints nothing when there is nothing to stash, and then HEAD
    already is that commit.

    It can also refuse, in a repository mid-merge or mid-rebase for instance,
    and that must not take the whole gate down: HEAD is still a tree CI could
    have, so fall back to it and say which one is being used. Silently falling
    back would be the bad half of this, because the difference is exactly the
    uncommitted work somebody is asking about.
    """
    head = git(repo, "rev-parse", "HEAD")
    rev = git(repo, "stash", "create", check=False)
    if rev:
        return rev
    if git(repo, "status", "--porcelain", "--untracked-files=no", check=False):
        print(
            f"note: {repo} has uncommitted changes and `git stash create` gave "
            "nothing back, so this runs HEAD instead of your working tree.",
            file=sys.stderr,
        )
    return head


def untracked(repo):
    """Files git does not track, which is exactly what CI would not see."""
    return [
        line
        for line in git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
        if line
    ]


def build_plan(workflow, fast, filters):
    d = yaml.safe_load(open(workflow))
    wf_env = d.get("env") or {}
    plan = []
    for jid, j in (d.get("jobs") or {}).items():
        name = j.get("name", jid)
        if filters and not any(f.lower() in name.lower() for f in filters):
            continue
        if fast and any(s in name.lower() for s in SLOW):
            continue
        toolchain = "stable"
        for s in j.get("steps") or []:
            u = s.get("uses", "")
            if u.startswith("dtolnay/rust-toolchain@"):
                toolchain = u.split("@", 1)[1]
        env = dict(wf_env)
        env.update(j.get("env") or {})
        steps = [s for s in (j.get("steps") or []) if s.get("run")]
        if steps:
            plan.append(
                {
                    "name": name,
                    "toolchain": toolchain,
                    "env": env,
                    "steps": steps,
                    # A job-level `if:` is a condition this tool does not
                    # evaluate, and guessing at it is the thing the `${{ }}`
                    # rule above exists to forbid. It is carried so the summary
                    # can name it instead of running a job GitHub would have
                    # held back (merge-gate.yml's Miri is the live case: it is
                    # held at the release boundary and does not finish here).
                    "if": j.get("if"),
                }
            )
    return plan


def provision(needs_tests, revs):
    """The shell that puts the checkouts in place before any step runs.

    Cloning rather than copying is what makes this cheap: `--shared` leaves the
    objects in the mounted store and writes an `alternates` line pointing at
    it, so a gigabyte-scale history costs no copy at all.
    """
    lines = [
        # The mounted git directories are owned by the host user, not root, and
        # git refuses to read a repository it thinks somebody else owns.
        "git config --global --add safe.directory '*'",
    ]
    for repo in ["libviprs"] + (["libviprs-tests"] if needs_tests else []):
        dest, src = CHECKOUT[repo], GITSRC[repo]
        lines += [
            f"git clone --quiet --shared --no-checkout {src} {dest}",
            f"git -C {dest} checkout --quiet --detach {revs[repo]}",
        ]
    return lines


def container_script(job, tag, mode, revs, tests_mounted):
    """Turn one job into a shell script to run inside the container."""
    out = ["set -eo pipefail"]
    if mode == "git":
        out += provision(tests_mounted, revs)
    for k, v in job["env"].items():
        out.append(f"export {k}={shlex.quote(str(v))}")
    out.append(f"export RUSTUP_TOOLCHAIN={shlex.quote(job['toolchain'])}")
    # Per-platform target dirs on the cargo volume. Sharing the host's target/
    # between an emulated amd64 run and a native one leaves the other
    # architecture's binaries in place, and `cargo test` happily re-runs a
    # stale one: --native kept failing under Rosetta because it was executing
    # the x86_64 test binary the previous run had built. It also keeps this
    # tool from clobbering the target/ you use by hand.
    out.append(f"export CARGO_TARGET_DIR=/cargo/target-{tag}")
    out.append(f"export CARGO_TARGET_DIR_TESTS=/cargo/target-{tag}-tests")
    for s in job["steps"]:
        run = s["run"]
        label = s.get("name") or run.splitlines()[0][:60]
        if "${{" in run:
            if "libviprs-tests.git" in run:
                if not tests_mounted:
                    # Exit 99, not 0. Exiting 0 here made the runner print
                    # "PASS Integration Tests" for a job that compiled nothing,
                    # and counted it toward "All jobs passed". A lane worktree
                    # has no sibling checkout, so that was the DEFAULT state
                    # rather than an edge case: the one job that crosses repos
                    # silently reported success for everybody who had not
                    # cloned the other repo.
                    out.append('echo "SKIP: no libviprs-tests sibling to mount"; exit 99')
                else:
                    out.append(
                        'echo ">> adapted: using the libviprs-tests sibling checkout, not git clone"'
                    )
                continue
            out.append(
                f'echo "REFUSING to guess at the expression in step: {label}"; exit 90'
            )
            continue
        for k, v in (s.get("env") or {}).items():
            out.append(f"export {k}={shlex.quote(str(v))}")
        cwd = s.get("working-directory") or "libviprs"
        if not cwd.startswith("/"):
            cwd = "/src/" + cwd if cwd.startswith("libviprs") else "/src/libviprs/" + cwd
        out.append(f"cd {shlex.quote(cwd)}")
        if cwd.startswith("/src/libviprs-tests"):
            out.append('export CARGO_TARGET_DIR="$CARGO_TARGET_DIR_TESTS"')
        # Quote the echoed copy properly. Inlining a TRUNCATED command into a
        # double-quoted echo breaks the moment a step is a multi-line shell
        # script with parens in it, which the MSRV guard is: the cut landed
        # mid-token and bash died on "syntax error near unexpected token `('".
        out.append("echo " + shlex.quote("  $ " + " ".join(run.split())[:150]))
        out.append(run)
    return "\n".join(out)


def report_source(repo, label, rev):
    """Say what is going into the container, and what is being left out."""
    head = git(repo, "rev-parse", "HEAD")
    state = "HEAD" if rev == head else f"working tree over {head[:12]}"
    print(f"==> {label}: {rev[:12]} ({state}), tracked files only")
    others = untracked(repo)
    if others:
        shown = others[:10]
        print(f"    {len(others)} untracked file(s) excluded, as on a runner:")
        for f in shown:
            print(f"      {f}")
        if len(others) > len(shown):
            print(f"      ... and {len(others) - len(shown)} more")


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("filters", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--workflow", default="ci.yml")
    p.add_argument("--native", action="store_true",
                   help="run on the host architecture instead of x86_64")
    p.add_argument("--worktree", action="store_true",
                   help="bind-mount the working tree instead of checking it "
                        "out from git (fast, case-insensitive, NOT the gate)")
    p.add_argument("--allow-skips", action="store_true",
                   help="let a skipped job leave the run green")
    p.add_argument("-h", "--help", action="store_true")
    a = p.parse_args()
    if a.help:
        print(__doc__)
        return 0

    workflow = os.path.join(REPO, ".github/workflows", a.workflow)
    if not os.path.isfile(workflow):
        return f"no such workflow: {workflow}"

    plan = build_plan(workflow, a.fast, a.filters)
    if not plan:
        return "no jobs matched"

    if a.list:
        for j in plan:
            held = f'  HELD by `if: {j["if"]}`' if j["if"] else ""
            print(f'[{j["name"]}]  toolchain={j["toolchain"]}  env={j["env"]}{held}')
            for s in j["steps"]:
                cwd = f'({s["working-directory"]}) ' if s.get("working-directory") else ""
                print("   $", cwd + " ".join(s["run"].split())[:150])
        return 0

    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return "Docker is not running."

    mode = "worktree" if a.worktree else "git"
    tests_mounted = os.path.isdir(TESTS_DIR)

    image = IMAGE_NATIVE if a.native else IMAGE_AMD64
    build = ["docker", "build", "-q", "-f", f"{REPO}/tools/Dockerfile.ci", "-t", image]
    if not a.native:
        build += ["--platform", "linux/amd64"]
    build.append(f"{REPO}/tools")
    print(f"==> building {image}"
          + ("" if a.native else " (x86_64, matching ubuntu-latest)")
          + " (cached after the first run)")
    subprocess.run(build, check=True, stdout=subprocess.DEVNULL)
    if subprocess.run(["docker", "volume", "inspect", VOLUME], capture_output=True).returncode != 0:
        subprocess.run(["docker", "volume", "create", VOLUME], check=True, stdout=subprocess.DEVNULL)

    mounts = ["-v", f"{VOLUME}:/cargo"]
    revs = {}
    if mode == "git":
        revs["libviprs"] = source_rev(REPO)
        report_source(REPO, "libviprs", revs["libviprs"])
        mounts += ["-v", f"{git_common_dir(REPO)}:{GITSRC['libviprs']}:ro"]
        if tests_mounted:
            revs["libviprs-tests"] = source_rev(TESTS_DIR)
            report_source(TESTS_DIR, "libviprs-tests", revs["libviprs-tests"])
            mounts += ["-v", f"{git_common_dir(TESTS_DIR)}:{GITSRC['libviprs-tests']}:ro"]
    else:
        print("!! --worktree bind-mounts this tree into the container. A Docker")
        print("!! Desktop bind mount off an APFS host is CASE-INSENSITIVE and it")
        print("!! carries untracked files, so this mode cannot see the two bug")
        print("!! classes the default mode exists for (#977, #979). Use it to")
        print("!! iterate, not to decide whether something is ready to push.")
        mounts += ["-v", f"{REPO}:{CHECKOUT['libviprs']}"]
        if tests_mounted:
            mounts += ["-v", f"{TESTS_DIR}:{CHECKOUT['libviprs-tests']}"]
    if not tests_mounted:
        print(f"note: {TESTS_DIR} not found, the integration job will skip", file=sys.stderr)

    failed = []
    skipped = []
    held = []
    for j in plan:
        if j["if"]:
            # Not a guess in either direction. Running it would run a job
            # GitHub holds back, and calling it green would be the "skipped
            # reads as passing" trap the workflow's own comment warns about.
            print(f"\n{'=' * 64}\n  {j['name']}   HELD\n{'=' * 64}")
            print(f"  This job carries `if: {j['if']}`, which this tool does not")
            print("  evaluate. It has NOT run. The Makefile has a host-native")
            print("  target for it where one exists (`make miri`).")
            held.append(j["name"])
            continue
        print(f"\n{'=' * 64}\n  {j['name']}   (toolchain {j['toolchain']})\n{'=' * 64}")
        # Stream the output AND keep it, so a failure can never be reported
        # without the reason. Relying on inherited stdout alone lost the
        # "cargo-fmt is not installed" line the first time this ran, which
        # made a failing job indistinguishable from a mysterious one.
        run_cmd = ["docker", "run", "--rm"]
        if not a.native:
            run_cmd += ["--platform", "linux/amd64"]
        proc = subprocess.Popen(
            run_cmd + mounts + ["-w", "/src", image,
             "bash", "-c", container_script(
                 j, "native" if a.native else "amd64", mode, revs, tests_mounted)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        tail = collections.deque(maxlen=40)
        for line in proc.stdout:
            sys.stdout.write(line)
            tail.append(line)
        rc = proc.wait()
        if rc == 99:
            print(f"  SKIP  {j['name']}")
            skipped.append(j["name"])
            continue
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {j['name']}")
        if rc != 0:
            failed.append(j["name"])
            print(f"  --- last lines of {j['name']} ---")
            for line in tail:
                print("  | " + line.rstrip())
            if any("rosetta error" in ln for ln in tail):
                print("")
                print("  This is Rosetta, not your code. Emulating x86_64 on Apple")
                print("  Silicon cannot reserve the address space the fallible-alloc")
                print("  tests deliberately ask for, so the process SIGTRAPs instead")
                print("  of the allocation failing cleanly. Re-run that job with:")
                print("      tools/local-ci.py --native " + j["name"].split()[0])

    print()
    if held:
        print("HELD (did NOT run, carry a job-level `if:`): " + ", ".join(held))
    if skipped:
        print("SKIPPED (did NOT run): " + ", ".join(skipped))
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    ran = len(plan) - len(skipped) - len(held)
    if skipped and not a.allow_skips:
        print(f"{ran} of {len(plan)} jobs passed. {len(skipped)} did not run, so this")
        print("is not a green gate. Pass --allow-skips if a subset is what you meant.")
        return 1
    if skipped or held:
        print(f"{ran} of {len(plan)} jobs passed; the rest did NOT run.")
        return 0
    if mode == "worktree":
        print("All jobs passed, over a bind-mounted tree. That is not the gate:")
        print("re-run without --worktree before you push.")
        return 0
    print("All jobs passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
