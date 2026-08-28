#!/usr/bin/env python3
"""The oracle pin every capture.py under oracle-captures/ checks (issue #650).

`ORACLE_PIN.json` next to this file names the libvips build these captures
are measured against. This module is the code that enforces it, factored out
so an area adopts it with three lines rather than a copy of the same forty.

The first line of this docstring was aspirational for as long as it stood:
only the two convolution areas ever imported this, and the other twelve ran
against whatever vips was on the machine (issue #796). All fourteen carry it
now, and `every_capture_script_checks_the_oracle_pin` in
`tests/oracle_capture_pins.rs` fails if a script drops it or a new area
arrives without it, so the sentence has a check behind it instead of a
convention.

Why it exists: a `brew upgrade` nobody ran deliberately replaced vips 8.18.4
with 8.18.6 mid-session and deleted the old keg, so the reference
implementation for twenty-odd issues of work moved without anyone choosing to
move it. Every area recorded the version in its own meta key, nothing ever
compared those strings to the binary on the machine, and the only reason it
surfaced at all is that a lane happened to run the same command twice.

Use it like this, from a capture.py one level down:

    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(ROOT, os.pardir)))
    import oracle_pin

    VIPS_VERSION, PIN = oracle_pin.check(AREA, VIPS)

`check` exits non-zero before anything is written when the binary disagrees
with the pin, unless the capture was run with `--repin`. That flag is half of
a two-key gesture rather than an escape hatch: the other half is editing
ORACLE_PIN.json, and tests/oracle_capture_pins.rs stays red until you do.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_PATH = os.path.join(HERE, "ORACLE_PIN.json")


def load():
    """The pin file, parsed."""
    with open(PIN_PATH) as f:
        return json.load(f)


def installed_vips_version(vips):
    """`vips --version` from the binary a capture is about to use.

    Not piped: reading `$?` at the end of a pipeline gives the exit status of
    the last stage rather than of vips, which is how this epic once convinced
    itself that `vips <op> --help` was not an existence test.
    """
    res = subprocess.run([vips, "--version"], capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit("cannot run %s: %s" % (vips, res.stderr.strip()))
    return res.stdout.strip()


def check(area, vips, argv=None):
    """Refuse to capture `area` against a libvips it is not pinned to.

    Returns (version_string, pin_document). Raises SystemExit on a mismatch,
    before the caller has written anything, so a wrong-oracle run leaves no
    half-updated capture behind.
    """
    argv = sys.argv[1:] if argv is None else argv
    pin = load()
    want = pin["pinned_vips_version"]
    if area not in pin["areas"]:
        raise SystemExit(
            "%s does not declare area %r. Add it, with the version this "
            "capture records and a state of on_pin or pre_pin."
            % (PIN_PATH, area)
        )
    got = installed_vips_version(vips)
    if got != want:
        first = (
            "oracle pin mismatch: %s reports %s, ORACLE_PIN.json pins %s."
            % (vips, got, want)
        )
        if "--repin" not in argv:
            raise SystemExit(
                first + "\nNothing has been written. Either install the "
                "pinned build, or re-run with --repin AND update "
                "ORACLE_PIN.json in the same commit. Issue #650."
            )
        print("WARNING (--repin): " + first)

    # A pre_pin area is one whose committed capture predates the pin. Running
    # it here moves it, which is fine and is the whole migration path, but the
    # pin file has to say so afterwards or the Rust guard stays red.
    declared = pin["areas"][area]["vips_version"]
    if declared != want:
        print(
            "NOTE: area %r is recorded at %s and this run captures %s. Set "
            "its vips_version to %s and its state to on_pin in "
            "ORACLE_PIN.json in the same commit."
            % (area, declared, got, got)
        )
    return got, pin


def homebrew_kegs(binary):
    """Every Homebrew keg `binary` reaches transitively, with its version.

    Issue #650 asked for this to be automatic. `vips --vips-config` names
    libopenjp2, libheif, matio and the rest with no version for any of them,
    and the codec version is what a future disagreement over a captured
    number turns on: four lanes had already reinvented `otool -L` by hand
    before this existed. The stack moves independently of vips, and did, so
    recording only the vips version is not enough provenance.

    Returns None off macOS rather than pretending to know.
    """
    if sys.platform != "darwin":
        return None

    def linked(path):
        res = subprocess.run(["otool", "-L", path], capture_output=True, text=True)
        if res.returncode != 0:
            return []
        return [ln.split(" (")[0].strip() for ln in res.stdout.splitlines()[1:]]

    found, seen, stack = {}, set(), [binary]
    while stack:
        real = os.path.realpath(stack.pop())
        if real in seen or not os.path.exists(real):
            continue
        seen.add(real)
        parts = real.split("/Cellar/")
        if len(parts) == 2:
            keg = parts[1].split("/")
            if len(keg) >= 2:
                found[keg[0]] = keg[1]
        stack.extend(linked(real))
    return dict(sorted(found.items()))
