#!/usr/bin/env python3
"""Vendors the pinned slice of libviprs/pvec-gold-fixtures that the PVec conformance suites read.

    tools/pvec-gold-vendor.py <gold-checkout> <git-ref> [dest]

Reads the r2-open manifest (and r3-open's, when the ref has one) at <git-ref> with `git show` / `git archive` (no working tree needed),
copies for every fixture the files the suites use (source.dwg, expected.math.json, the first
oracle's signature and the verdict), checks each against the sha256 the upstream manifest records,
and writes `index.json` (the trimmed manifest) `PIN.json` (corpus revision, gold commit and the
sha256 of the upstream manifest) and `summary.json` (fixture ids per evidence class and scope, the
readable face of the pin) next to them. Bumping the pin is re-running this and committing the diff.
When an `index.json` is already there, it also prints the per-class change summary against it, so a
bump is reviewed as "which fixtures moved from which class to which" and not as an opaque hash
change.

It only reads files and writes under <dest>, so it needs no container.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Real-world drawings with no STEP evidence and no truth: over 400 KiB each, and no suite reads them.
SKIP = {"general-sample-ac1032-001", "arch-aecobjects-001"}
GENERATED = ("primary.signature.json", "verdict.json")
ARTIFACTS = ("source.dwg", "expected.math.json")


def git(gold, *args):
    return subprocess.check_output(["git", "-C", gold, *args])


def main():
    gold, ref = sys.argv[1], sys.argv[2]
    dest = Path(sys.argv[3] if len(sys.argv) > 3 else "tests/fixtures/pvec-gold")
    commit = git(gold, "rev-parse", ref).decode().strip()
    raw = git(gold, "show", f"{commit}:corpus/r2-open/manifest.json")
    manifest = json.loads(raw)
    out = {"corpus_revision": manifest["corpus_revision"], "oracle_scope": manifest["oracle_scope"],
           "fixtures": []}
    dest.mkdir(parents=True, exist_ok=True)
    previous = None
    if (dest / "index.json").exists():
        previous = {f["id"]: (f["evidence_class"], f["evidence_scope"])
                    for f in json.loads((dest / "index.json").read_text())["fixtures"]}
    for fx in manifest["fixtures"]:
        if fx["id"] in SKIP:
            continue
        files = {}
        sources = [(n, fx["artifacts"][n]["path"], fx["artifacts"][n]["sha256"])
                   for n in ARTIFACTS if n in fx["artifacts"]]
        sources += [(n, "corpus/r2-open/" + fx["generated"][n]["path"], fx["generated"][n]["sha256"])
                    for n in GENERATED if n in fx["generated"]]
        for name, path, want in sources:
            data = git(gold, "show", f"{commit}:{path}")
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                sys.exit(f"{fx['id']}/{name}: sha256 {got} does not match the manifest's {want}")
            d = dest / "fixtures" / fx["id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes(data)
            files[name] = {"sha256": got, "bytes": len(data)}
        out["fixtures"].append({
            "id": fx["id"], "group": fx["group"], "evidence_class": fx["evidence_class"],
            "evidence_scope": fx["evidence_scope"], "status": fx["status"],
            "expects_rejection": fx["expects_rejection"],
            "step_verdict": (fx.get("step_oracle") or {}).get("verdict"),
            "files": files})
    # r3-open (pvec-gold-fixtures#96) includes r2-open unchanged and adds fixtures of its own. Its entries carry
    # no STEP evidence and no `generated` block, so they are read from their own artifacts. A fixture with no
    # source.dwg (the large generated drawing) has nothing to vendor and stays out of the index.
    r3raw = None
    try:
        r3raw = git(gold, "show", f"{commit}:corpus/r3-open/manifest.json")
    except subprocess.CalledProcessError:
        pass
    if r3raw is not None:
        r3 = json.loads(r3raw)
        for fx in r3["fixtures"]:
            if "source.dwg" not in fx["artifacts"]:
                continue
            files = {}
            d = dest / "fixtures" / fx["id"]
            d.mkdir(parents=True, exist_ok=True)
            for name in ARTIFACTS:
                a = fx["artifacts"][name]
                data = git(gold, "show", f"{commit}:{a['path']}")
                got = hashlib.sha256(data).hexdigest()
                if got != a["sha256"]:
                    sys.exit(f"{fx['id']}/{name}: sha256 {got} does not match the manifest's {a['sha256']}")
                (d / name).write_bytes(data)
                files[name] = {"sha256": got, "bytes": len(data)}
            out["fixtures"].append({
                "id": fx["id"], "group": fx["group"], "evidence_class": fx["evidence_class"],
                "evidence_scope": fx["evidence_scope"], "status": fx["status"],
                "expects_rejection": False, "step_verdict": None, "files": files})
        out["corpus_revision"] = r3["corpus_revision"]
    (dest / "index.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    summary = {"corpus_revision": out["corpus_revision"], "by_class": {}}
    for f in out["fixtures"]:
        key = f"{f['evidence_class']} / {f['evidence_scope']}"
        summary["by_class"].setdefault(key, []).append(f["id"])
    summary["by_class"] = {k: sorted(v) for k, v in sorted(summary["by_class"].items())}
    (dest / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if previous is not None:
        now = {f["id"]: (f["evidence_class"], f["evidence_scope"]) for f in out["fixtures"]}
        for i in sorted(set(previous) | set(now)):
            if previous.get(i) != now.get(i):
                print(f"{i}: {previous.get(i)} -> {now.get(i)}")
    pin = {"corpus_revision": out["corpus_revision"], "gold_commit": commit,
           "upstream_manifest_sha256": hashlib.sha256(raw).hexdigest(),
           "skipped": sorted(SKIP), "vendored_by": "tools/pvec-gold-vendor.py"}
    if r3raw is not None:
        pin["r3_manifest_sha256"] = hashlib.sha256(r3raw).hexdigest()
    (dest / "PIN.json").write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
