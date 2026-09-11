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
    tools/local-ci.py --workflow merge-gate.yml   # Loom and the pdfium audit
    tools/local-ci.py --native           # host arch instead of x86_64
    tools/local-ci.py --volume lane-f11  # a cargo volume of your own
    tools/local-ci.py --worktree         # bind-mount the tree (fast, NOT the gate)
    tools/local-ci.py --print-docker-argv   # the docker commands, run nothing

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

Why a clone and not `git archive`
--------------------------------

`git archive <rev> | tar -x` is the obvious way to get a case-exact,
untracked-free tree, and it is much cheaper: its entries all carry the commit's
timestamp, so extracting the same commit twice gives byte-identical mtimes and
cargo stays warm. Measured back to back on an unchanged tree with the
dependencies warm, `cargo test --no-run` costs 0.58s over the old bind mount,
1.1s over an archive extraction and 26s over the clone below.

It is the wrong trade twice over.

It ships no `.git`, and four test files here shell out to git:
`tests/case_only_path_collisions.rs`, `tests/fixture_paths_are_committed.rs`
and `tests/oracle_capture_pins.rs` read `git ls-files`, and
`tests/changelog_release_claims.rs` reads `git tag` and `git grep <tag>`.
Synthesising a repository around the extraction with `git init && git add -A`
gets the first three back and not the fourth: a fresh repository has no tags,
and that guard opens by asserting `v0.1.1`, `v0.2.0` and `v0.4.0` are in the
listing precisely so that a clone without them fails instead of finding no
counterexamples. So the archive route buys case-exactness and pays for it with
a gate that is red for a reason that has nothing to do with the change in front
of it, which is the other way to make a check worth ignoring.

And the stable mtimes are a hazard rather than a feature. Cargo decides a crate
is fresh when no source file is *newer* than the artifact, so a tree whose
timestamps go backwards, which is every `git bisect` step and every "check the
commit before this one", reads as fresh and hands you the previous commit's
binaries. `git checkout` stamps the files with the time it ran, so that cannot
happen: the 26 seconds is the price of never silently testing the wrong build,
and dependencies stay warm through it either way because the target directory
lives on the `/cargo` volume rather than in the tree. Cold, that same build is
3 minutes.

`--worktree` puts the old bind mount back for the times when the rebuild is not
worth it, and says loudly that it is not the gate.

On architecture: what `--native` asks the host, and who answers
---------------------------------------------------------------

`platform.machine()` reports the architecture of the *interpreter*, not of the
machine. On this host `uname -m` is `arm64` and `arch -x86_64 /usr/bin/uname -m`
is `x86_64`, and an x86_64 python3 is exactly what a shell configured to prefer
amd64 tends to end up running. So sourcing the host architecture from the
interpreter makes `--native` pin `linux/amd64` and call it the host, which is
#994 again with a different origin.

The daemon is the thing that actually runs the container, so it is asked first:
`docker version --format '{{.Server.Arch}}'`, which answers `arm64` here.
`platform.machine()` is the fallback for when there is no daemon to ask, which
is the ordinary case inside the CI image, where `--print-docker-argv` still has
to work and has no docker CLI at all. Whichever answered is printed, on the
build line and as the `arch-source` field of `--print-docker-argv`, because a
fallback nobody can see is a fallback nobody checks.

`DOCKER_BUILDKIT=0` and the image that is not the platform you asked for
-----------------------------------------------------------------------

The legacy builder neither refuses `--platform` nor honours it. Measured here
on Docker 29.7.2, arm64 host, `DOCKER_DEFAULT_PLATFORM` unset:
`DOCKER_BUILDKIT=0 docker build --platform linux/arm64` and the same command
with `linux/amd64` produced the *identical* image id, and
`docker image inspect --format '{{.Architecture}}'` calls it `amd64` both
times, while BuildKit built a genuinely arm64 image from the same Dockerfile.
So with BuildKit off, `--native` would build an amd64 image, tag it
`libviprs-ci:native`, and then run it under `--platform linux/arm64`. That is
quieter than the bug this flag was fixed for, so the build is followed by a
check that the image really is the architecture that was asked for.

On emulation: the image is x86_64, the same as GitHub's ubuntu-latest, so
on an Apple Silicon host Docker emulates it. Both the build and the run name
that platform explicitly rather than leaving the flag off, because leaving it
off does not mean "the host", it means `DOCKER_DEFAULT_PLATFORM`, and that is
`linux/amd64` in this repository's usual shell (#994). That is slower than running
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

One cargo volume per lane
-------------------------

`--volume NAME` (env `LIBVIPRS_CI_VOLUME`) picks the volume holding `CARGO_HOME`
and the target directories, and it defaults to the shared `libviprs-ci-cargo` so
ordinary use is unchanged. Several worktrees running this at once do not corrupt
each other, cargo's lock sees to that, but they do invalidate each other's build
cache, because each run checks a different tree out into the same target
directory. Give each lane its own name and they stay warm.

What a lane's volume costs, and how to get it back
--------------------------------------------------

Each volume is a full copy of `CARGO_HOME` and of every target directory the
job list materialises, which is about two dozen artifact sets: Check & Lint
compiles ten feature permutations, Test nine more, and MSRV another seven under
a second toolchain, each with its own metadata hash rather than replacing the
last. Five lanes is five of those on the Docker VM's disk. That disk filling up
is not hypothetical, it is why this tool has a dedicated handler for the
message; I cleared 19 GB off it on the day I wrote this paragraph.

So look at what a lane is holding, and throw it away when the lane is done:

    docker run --rm -v libviprs-ci-f11:/cargo alpine:3 du -sh /cargo/*
    docker volume rm libviprs-ci-f11

`docker volume rm` refuses while a container still has the volume open, which
is the behaviour you want: a running gate cannot have its build cache pulled
out from under it.

An empty name is not a name. `LIBVIPRS_CI_VOLUME=` and `--volume ''` used to
reach `docker run -v :/cargo`, after paying for the image build, and die there
with a message naming neither the flag nor the variable. Both fall back to the
shared default now, and anything Docker would not accept as a volume name is
refused up front.


A job that does not run is not a job that passed
------------------------------------------------

A skipped job makes the whole run exit non-zero. It used to exit 0 with a note,
and a note is not a gate: the one job that crosses repos reported SKIP and the
run still said "All jobs passed" to anybody who had not cloned the sibling.
`--allow-skips` is there for when a subset is genuinely what you asked for.

A job carrying a job-level `if:` is reported HELD and not run, for the same
reason the `${{ }}` rule above refuses to guess. Today that is only
`merge-gate.yml`'s Miri, which is held at the release boundary; `make miri`
runs it here on a pinned nightly, and `tests/local_gate_is_the_job_list.rs`
fails if a job grows an `if:` with nothing covering it.
"""
import argparse, collections, os, platform, re, shlex, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE = os.path.abspath(os.path.join(REPO, ".."))
TESTS_DIR = os.path.join(WORKSPACE, "libviprs-tests")
IMAGE_AMD64 = "libviprs-ci:local"
IMAGE_NATIVE = "libviprs-ci:native"
VOLUME = "libviprs-ci-cargo"
SLOW = ("test", "integration")
# Where every step runs from, and the one `docker run` working directory. Both
# checkouts sit under it, so a step's `working-directory` resolves the same way
# `actions/checkout` makes it resolve on a runner.
WORKDIR = "/src"
# Docker's own rule for a volume name: an alphanumeric first character, then
# alphanumerics, underscore, dot or dash. Nothing else, which is what keeps a
# tab or a newline out of the tab-separated `--print-docker-argv` output.
VOLUME_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
# Every architecture spelling either the daemon or the interpreter produces,
# mapped to the half of a Docker platform string that follows `linux/`.
DOCKER_ARCH = {"x86_64": "amd64", "amd64": "amd64",
               "aarch64": "arm64", "arm64": "arm64"}

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


def is_git_repo(path):
    """Whether `path` is something the git-provisioned mode can read.

    A sibling checkout that is not a git repository cannot be provisioned from
    git, and quietly bind-mounting that one while checking the other one out
    would be the worst of both: a run that looks faithful and is not.
    """
    return (
        subprocess.run(
            ["git", "-C", path, "rev-parse", "--git-common-dir"],
            capture_output=True,
        ).returncode
        == 0
    )


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


def daemon_arch():
    """What the Docker daemon says it runs on, or `None` if it cannot say.

    `docker version --format {{.Server.Arch}}` is a GOARCH, so it comes back
    `arm64` or `amd64` and needs no translation, but it is put through
    [`DOCKER_ARCH`] anyway so that a daemon answering `aarch64` (which
    `docker info --format {{.Architecture}}` does) is handled the same way.

    Every failure is the same answer here, on purpose: no docker CLI on PATH,
    no daemon listening, a daemon too slow to answer. The caller has a real
    fallback and `--print-docker-argv` has to work in the CI image, which has
    no docker at all.
    """
    try:
        out = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Arch}}"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def host_platform():
    """`(platform string or None, which source answered, the raw architecture)`.

    `--native` used to mean "leave `--platform` off and let Docker decide", and
    what Docker decides is `DOCKER_DEFAULT_PLATFORM` when that is set. It is
    `linux/amd64` in the shell this repository is developed in, so `--native`
    asked the daemon for an amd64 variant of a local arm64 image and reported:

        Unable to find image 'libviprs-ci:native' locally
        docker: Error response from daemon: pull access denied for libviprs-ci,
        repository does not exist or may require 'docker login': denied: ...

    The image was right there. That message is about authentication and a
    missing repository and says nothing about a platform, so it reads like a
    login problem or a typo in the tag rather than the one thing it is. Naming
    the platform on both the build and the run costs one argument and makes the
    flag mean what it says (#994). `tests/local_ci_invocation.rs` holds it there.

    The architecture itself comes from the daemon rather than from
    `platform.machine()`, which reports the *interpreter's* architecture: see
    the module docstring for the measurement. `platform.machine()` is still the
    fallback, because there is nothing else to ask when there is no daemon, and
    which one answered is reported rather than assumed.

    An architecture neither of them can be mapped to gives `None` rather than a
    `sys.exit`. This is called from `--list` and `--print-docker-argv`, which
    print and quit and touch no daemon, and a refusal that reaches those is a
    refusal on a path that has nothing to refuse. `main` turns the `None` into
    a failure at the point a run would actually need the platform.
    """
    arch, source = daemon_arch(), "daemon"
    if arch is None:
        arch, source = platform.machine(), "platform.machine"
    mapped = DOCKER_ARCH.get(arch)
    return (f"linux/{mapped}" if mapped else None), source, arch


def resolve_volume(requested):
    """The cargo volume name, with an empty request falling back to the default.

    `os.environ.get("LIBVIPRS_CI_VOLUME", VOLUME)` falls back when the key is
    absent and not when it is empty, so `export LIBVIPRS_CI_VOLUME=` and
    `--volume ''` both got through. `docker volume inspect ''` fails, and the
    `docker volume create ''` that follows it fails too and is not checked, so
    the run died inside `docker run` on `-v :/cargo` with a message naming
    neither the flag nor the variable, after paying for the image build.

    Refusing a name Docker would not take closes a second hole with the same
    edit. The `--print-docker-argv` format is tab separated, so a name carrying
    a tab and a newline injects whole extra fields into it:
    `--volume "$(printf 'x\nplatform\tlinux/amd64')"` put a second `platform`
    line into the output, and the test harness's `fields.insert` kept the
    injected one. The parser refuses a duplicate key now as well, because two
    checks on that are cheap and it is the sort of thing that comes back.
    """
    name = requested or VOLUME
    if not VOLUME_NAME.fullmatch(name):
        sys.exit(
            f"{name!r} is not a Docker volume name, so --volume (env "
            "LIBVIPRS_CI_VOLUME) cannot be used as given. Docker takes an "
            "alphanumeric first character followed by alphanumerics, "
            "underscore, dot or dash. Leave it empty for the shared default "
            f"{VOLUME!r}."
        )
    return name


def target_dirs(tag):
    """Where cargo builds inside the container, for this architecture.

    One function rather than two f-strings, because `container_script` exports
    these and `--print-docker-argv` reports them, and a reported value that is
    a restatement of the real one is a value that drifts.
    """
    return f"/cargo/target-{tag}", f"/cargo/target-{tag}-tests"


def build_argv(plat, image):
    """The `docker build` command, naming its platform rather than inheriting."""
    return ["docker", "build", "-q", "--platform", plat,
            "-f", f"{REPO}/tools/Dockerfile.ci", "-t", image, f"{REPO}/tools"]


def source_mounts(mode, tests_mounted):
    """The bind mounts that put the source trees where the steps expect them.

    In `git` mode that is each repository's git directory, read-only, which the
    container clones from. In `worktree` mode it is the working trees
    themselves. The two are genuinely different commands, and they used to
    print identically because the printer restated the first few arguments
    instead of composing the real thing.
    """
    if mode == "worktree":
        mounts = [f"{REPO}:{CHECKOUT['libviprs']}"]
        if tests_mounted:
            mounts.append(f"{TESTS_DIR}:{CHECKOUT['libviprs-tests']}")
        return mounts
    mounts = [f"{git_common_dir(REPO)}:{GITSRC['libviprs']}:ro"]
    if tests_mounted:
        mounts.append(f"{git_common_dir(TESTS_DIR)}:{GITSRC['libviprs-tests']}:ro")
    return mounts


def sibling_available(mode):
    """Whether the libviprs-tests checkout can be provisioned in `mode`."""
    if not os.path.isdir(TESTS_DIR):
        return False
    return mode == "worktree" or is_git_repo(TESTS_DIR)


def run_argv(plat, image, volume, mounts):
    """The whole `docker run` command bar the script, and the only one there is.

    This used to be a prefix that `main` took apart and put back together:
    `mounts = run_prefix[run_prefix.index("-v"):]` and then
    `run_prefix[:run_prefix.index("-v")] + mounts`. Two things were wrong with
    that beyond the obvious fragility. `--print-docker-argv` printed the prefix
    and called it the command, so it never showed the source mounts, the
    working directory or the image, and `--worktree --print-docker-argv`
    printed exactly what `--print-docker-argv` did while the real `--worktree`
    run carried a different mount. And a reviewer changed the reassembly to
    `run_prefix[:3] + mounts`, which drops `--platform` from the real run and
    restores #994 verbatim, with every test still green.

    So there is one function, the job loop calls it, the printer calls it, and
    the printer is right by construction rather than by inspection.
    """
    argv = ["docker", "run", "--rm", "--platform", plat, "-v", f"{volume}:/cargo"]
    for mount in mounts:
        argv += ["-v", mount]
    return argv + ["-w", WORKDIR, image]


def build_plan(workflow, fast, filters):
    # Imported here rather than at module scope so `--print-docker-argv` works
    # in the CI image, which carries python3 and no PyYAML. Nothing before this
    # point needs to parse a workflow. `tests/local_ci_invocation.rs` checks
    # both halves: that the argv path runs without PyYAML, and that this path
    # either works or says PyYAML is missing, rather than raising NameError.
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: pip3 install pyyaml")
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


def uses_the_sibling(job):
    """Whether this job touches the libviprs-tests checkout at all.

    Only the integration job does, so the other four have no reason to pay for
    a second clone.
    """
    return any(
        (s.get("working-directory") or "").startswith("libviprs-tests")
        or "libviprs-tests" in s["run"]
        for s in job["steps"]
    )


def container_script(job, tag, mode, revs, tests_mounted):
    """Turn one job into a shell script to run inside the container."""
    out = ["set -eo pipefail"]
    if mode == "git":
        out += provision(tests_mounted and uses_the_sibling(job), revs)
    for k, v in job["env"].items():
        out.append(f"export {k}={shlex.quote(str(v))}")
    out.append(f"export RUSTUP_TOOLCHAIN={shlex.quote(job['toolchain'])}")
    # Per-platform target dirs on the cargo volume. Sharing the host's target/
    # between an emulated amd64 run and a native one leaves the other
    # architecture's binaries in place, and `cargo test` happily re-runs a
    # stale one: --native kept failing under Rosetta because it was executing
    # the x86_64 test binary the previous run had built. It also keeps this
    # tool from clobbering the target/ you use by hand.
    main_target, tests_target = target_dirs(tag)
    out.append(f"export CARGO_TARGET_DIR={main_target}")
    out.append(f"export CARGO_TARGET_DIR_TESTS={tests_target}")
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


def report_source(repo, label, rev, is_sibling=False):
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
    if not is_sibling:
        return
    # The sibling is the one repository whose revision the hosted job picks for
    # itself: ci.yml clones the matching branch of libviprs-tests, or its main,
    # from origin. This checkout is whatever it happens to be, and a stale one
    # fails the integration job on a signature that moved months ago, which
    # reads exactly like the change in front of you breaking the API. Mine was
    # twelve commits behind and cost me a confused half hour.
    if not git(repo, "rev-parse", "--verify", "--quiet", "origin/main", check=False):
        return
    behind = git(repo, "rev-list", "--count", "HEAD..origin/main", check=False)
    if behind and behind != "0":
        print(f"    NOTE: {behind} commit(s) behind its own origin/main, which is what")
        print("    the hosted integration job clones. The count is as of your last")
        print("    fetch, so it is a floor rather than the answer.")


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
    p.add_argument("--volume", default=os.environ.get("LIBVIPRS_CI_VOLUME", VOLUME),
                   help="cargo volume to use, so parallel worktrees do not "
                        "invalidate each other's build cache "
                        "(env LIBVIPRS_CI_VOLUME)")
    p.add_argument("--print-docker-argv", action="store_true",
                   help="print the docker commands this run would use, and exit")
    p.add_argument("-h", "--help", action="store_true")
    a = p.parse_args()
    if a.help:
        print(__doc__)
        return 0

    volume = resolve_volume(a.volume)
    if a.native:
        plat, arch_source, raw_arch = host_platform()
    else:
        # Not a lookup at all, so there is no source to report beyond the
        # decision itself: ubuntu-latest is x86_64 and the emulated run exists
        # to match it.
        plat, arch_source, raw_arch = "linux/amd64", "pinned", None
    image = IMAGE_NATIVE if a.native else IMAGE_AMD64
    tag = "native" if a.native else "amd64"
    mode = "worktree" if a.worktree else "git"
    tests_mounted = sibling_available(mode)
    # An architecture nothing maps still has to print. `shown` is what goes in
    # the argv, and `plat is None` is what stops a run below.
    shown = plat or f"UNMAPPED:{raw_arch}"
    build = build_argv(shown, image)

    def the_run_command():
        """The one composition of the run command, called twice, written once."""
        return run_argv(shown, image, volume, source_mounts(mode, tests_mounted))

    if a.print_docker_argv:
        # Tab separated because a docker argument never contains a tab, which
        # keeps the tests free of a quoting round trip that could disagree with
        # this tool about where one argument ends and the next begins. A volume
        # name carrying a tab would break that, which is half of why
        # `resolve_volume` refuses one.
        main_target, tests_target = target_dirs(tag)
        for key, value in (("platform", [shown]),
                           ("arch-source", [arch_source]),
                           ("image", [image]),
                           ("volume", [volume]),
                           ("mode", [mode]),
                           ("cargo-target", [main_target, tests_target]),
                           ("build", build),
                           ("run", the_run_command())):
            print("\t".join([key] + value))
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

    if plat is None:
        # Deliberately here and not in `host_platform`, which `--list` and
        # `--print-docker-argv` both reach without touching a daemon. A refusal
        # belongs at the point something actually needs the platform.
        return (
            f"no Docker platform mapping for host architecture {raw_arch!r} "
            f"(reported by {arch_source}). --list and --print-docker-argv still "
            "work; a run cannot."
        )

    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return "Docker is not running."

    if os.path.isdir(TESTS_DIR) and mode == "git" and not is_git_repo(TESTS_DIR):
        print(
            f"note: {TESTS_DIR} is not a git repository, so it cannot be "
            "provisioned the way this mode provisions everything else, so "
            "the integration job reports SKIP and the run fails unless you "
            "pass --allow-skips.",
            file=sys.stderr,
        )

    print(f"==> building {image} ({plat}"
          + (f", host architecture per {arch_source}" if a.native
             else ", matching ubuntu-latest")
          + ") (cached after the first run)")
    subprocess.run(build, check=True, stdout=subprocess.DEVNULL)
    built = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Architecture}}"],
        capture_output=True, text=True,
    )
    got = DOCKER_ARCH.get(built.stdout.strip(), built.stdout.strip())
    if built.returncode == 0 and got and got != plat.split("/", 1)[1]:
        # The classic builder takes `--platform` and ignores it: measured on
        # Docker 29.7.2, `DOCKER_BUILDKIT=0 docker build --platform linux/arm64`
        # and the same command asking for amd64 produced one identical amd64
        # image on an arm64 host. Running that under `--platform linux/arm64`
        # is the #994 failure again, one layer down and quieter.
        return (f"{image} came out {got}, not {plat.split('/', 1)[1]}, so the "
                "build ignored --platform. The legacy builder does that "
                "silently: unset DOCKER_BUILDKIT (or set it to 1) and run this "
                "again.")
    if subprocess.run(["docker", "volume", "inspect", volume], capture_output=True).returncode != 0:
        subprocess.run(["docker", "volume", "create", volume], check=True, stdout=subprocess.DEVNULL)
    if volume != VOLUME:
        print(f"==> cargo volume {volume}, not the shared {VOLUME}")

    revs = {}
    if mode == "git":
        revs["libviprs"] = source_rev(REPO)
        report_source(REPO, "libviprs", revs["libviprs"])
        if tests_mounted:
            revs["libviprs-tests"] = source_rev(TESTS_DIR)
            report_source(
                TESTS_DIR, "libviprs-tests", revs["libviprs-tests"], is_sibling=True
            )
    else:
        print("!! --worktree bind-mounts this tree into the container. A Docker")
        print("!! Desktop bind mount off an APFS host is CASE-INSENSITIVE and it")
        print("!! carries untracked files, so this mode cannot see the two bug")
        print("!! classes the default mode exists for (#977, #979). Use it to")
        print("!! iterate, not to decide whether something is ready to push.")
    run = the_run_command()
    if not os.path.isdir(TESTS_DIR):
        print(
            f"note: {TESTS_DIR} not found, so the integration job cannot run. "
            "It reports SKIP, and a skip fails the run unless you pass "
            "--allow-skips.",
            file=sys.stderr,
        )

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
        proc = subprocess.Popen(
            run + ["bash", "-c",
                   container_script(j, tag, mode, revs, tests_mounted)],
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
            if any("No space left on device" in ln for ln in tail):
                # Worth naming, because the message arrives as a compiler or a
                # git error and reads like the change is broken. It is not:
                # running the whole job list materialises about two dozen
                # distinct artifact sets on the /cargo volume, because Check &
                # Lint compiles ten feature permutations, Test nine more and
                # MSRV another seven under a second toolchain, and each one
                # gets its own metadata hash rather than replacing the last.
                print("")
                print("  The Docker VM's disk is full, not your code. The whole job")
                print("  list materialises about two dozen artifact sets on the cargo")
                print("  volume, one per feature permutation per toolchain. See what")
                print("  is on there with:")
                print(f"      docker run --rm -v {volume}:/cargo alpine:3 du -sh /cargo/*")
                print("  A lane's volume is disposable once the lane is done:")
                print(f"      docker volume rm {volume}")
                print("  Docker Desktop's disk size is under Settings, Resources.")
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
