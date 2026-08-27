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

Two things cannot run verbatim and are adapted, out loud:

  * `actions/checkout` and the integration job's `git clone` of libviprs-tests.
    The sibling checkout is bind-mounted instead, so what runs is the tree you
    have rather than whatever is on its remote.
  * Nothing else. Any other `${{ }}` expression is a hard error rather than a
    guess, because a silently mis-substituted step is worse than a missing one.

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
"""
import argparse, collections, json, os, shlex, subprocess, sys

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
            plan.append({"name": name, "toolchain": toolchain, "env": env, "steps": steps})
    return plan


def container_script(job, tag):
    """Turn one job into a shell script to run inside the container."""
    out = ["set -eo pipefail"]
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
                if not os.path.isdir(TESTS_DIR):
                    # Exit 99, not 0. Exiting 0 here made the runner print
                    # "PASS Integration Tests" for a job that compiled nothing,
                    # and counted it toward "All jobs passed". A lane worktree
                    # has no sibling checkout, so that was the DEFAULT state
                    # rather than an edge case: the one job that crosses repos
                    # silently reported success for everybody who had not
                    # cloned the other repo.
                    out.append('echo "SKIP: no libviprs-tests sibling to mount"; exit 99')
                else:
                    out.append('echo ">> adapted: using the bind-mounted libviprs-tests sibling, not git clone"')
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


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("filters", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--workflow", default="ci.yml")
    p.add_argument("--native", action="store_true",
                   help="run on the host architecture instead of x86_64")
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
            print(f'[{j["name"]}]  toolchain={j["toolchain"]}  env={j["env"]}')
            for s in j["steps"]:
                cwd = f'({s["working-directory"]}) ' if s.get("working-directory") else ""
                print("   $", cwd + " ".join(s["run"].split())[:150])
        return 0

    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return "Docker is not running."

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

    mounts = ["-v", f"{REPO}:/src/libviprs", "-v", f"{VOLUME}:/cargo"]
    if os.path.isdir(TESTS_DIR):
        mounts += ["-v", f"{TESTS_DIR}:/src/libviprs-tests"]
    else:
        print(f"note: {TESTS_DIR} not found, the integration job will skip", file=sys.stderr)

    failed = []
    skipped = []
    for j in plan:
        print(f"\n{'=' * 64}\n  {j['name']}   (toolchain {j['toolchain']})\n{'=' * 64}")
        # Stream the output AND keep it, so a failure can never be reported
        # without the reason. Relying on inherited stdout alone lost the
        # "cargo-fmt is not installed" line the first time this ran, which
        # made a failing job indistinguishable from a mysterious one.
        run_cmd = ["docker", "run", "--rm"]
        if not a.native:
            run_cmd += ["--platform", "linux/amd64"]
        proc = subprocess.Popen(
            run_cmd + mounts + ["-w", "/src/libviprs", image,
             "bash", "-c", container_script(j, "native" if a.native else "amd64")],
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
    if skipped:
        print("SKIPPED: " + ", ".join(skipped))
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    ran = len(plan) - len(skipped)
    if skipped:
        print(f"{ran} of {len(plan)} jobs passed; {len(skipped)} skipped and did NOT run.")
        return 0
    print("All jobs passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
