# Contributing to libviprs

Most of what is written down here is about dependencies. That is the decision
this project keeps having to make, it is the one it kept getting wrong, and it
is the one that has been used to accept and reject other people's work. The
rest of the workflow lives in the README and the `Makefile`; the last section
points at it.

## The dependency rule

**Nothing libviprs builds may need a library that is not already in the
dependency tree.**

Three clauses:

1. **No dependency may go looking for a library on the build machine.** No
   `pkg-config`, `vcpkg`, `system-deps` or `cmake` probe may run in any build
   script that this crate causes to run.
2. **No dependency may link a third-party library that somebody has to install
   first.** Linking what the platform itself always ships is fine and
   unavoidable: libc, libm, `CoreFoundation` on macOS, the Windows import
   libraries.
3. **Compiling *vendored* C or assembly through a `build.rs` is allowed**, and
   the instances are listed below. The crate already does this on every single
   build, so a rule that forbade it would be a rule the crate itself breaks.

`tests/dependency_policy.rs` runs on every `cargo test` and checks clauses 1
and 3, exactly. Per target and per feature set, the set of library-discovery
crates and the set of crates that compile vendored native code each have to
equal a list written into the file, so it goes red in both directions: a new
one turning up, and a listed one quietly leaving.

Clause 2 it does not check, and nothing could. Nothing in a manifest separates
a crate that needs a library installed on the machine from one that does not.
`pdfium-render` is the proof and it is already in this tree: no C, no `links`
key, no `-sys` suffix, and useless without a `libpdfium` on the machine. So
clause 2 gets applied by hand, against the checklist in "Adding a dependency"
below.

The test does carry two tripwires aimed at clause 2. It pins the `links` keys
in every graph to an allowlist, and it pins the set of crates whose job is to
open a shared library at runtime (`libloading`, `dlopen`, `dlopen2`) the same
way. Both are closed lists of names, so they catch the shapes we have already
been bitten by rather than the property itself. Read a green suite as "nothing
recognisable turned up", not as "clause 2 holds".

That distinction is not hypothetical. Until the runtime-loader tripwire went
in, adding `libloading` to `[dependencies]` unconditionally, a crate whose only
purpose is to open a library that is not in the tree, passed every cell.

Two features are named carve-outs, `packfile` and `pdfium`. Both are off by
default and both are written up below. The list is closed: a new feature does
not get to add a third without an explicit decision recorded in this file and
in the test.

That rule still rejects everything this project has turned down. OpenSlide
(#503), ImageMagick (#509) and `fitsio` (via `fitsio-sys`, see the note at the
top of `src/fits.rs`) all fail clause 1 or clause 2, which is the reason they
were actually rejected. The objection was never "it has a `build.rs`".

### What the rule is not

I used to enforce this as "zero `-sys` crates and zero C compiles anywhere in
the tree" (#606). That was never true of the tree, not on any commit, and every
obvious tightening of it is false too. Here is each tempting discriminator with
the counterexample that kills it, all of them in the graph today:

| Tempting rule | Counterexample, in this tree, right now |
|---|---|
| "no `build.rs`" | 21 crates in the default host graph have one, `serde`, `libc` and `proc-macro2` among them |
| "nothing compiles C" | `blake3` compiles `c/blake3_neon.c` into `libblake3_neon.a` on every aarch64 build, and assembles `.S` files through `cc` on x86_64 |
| "no `links =` key" | `rayon-core` declares `links = "rayon-core"` in every default build on every target. It is not a C library at all: `links` is doing its *other* job there, letting cargo enforce a single version of a crate across the graph |
| "no `-sys` suffix" | `libbz2-rs-sys` is pure Rust, 0 `.c` files and no `build.rs`. So are `core-foundation-sys`, `linux-raw-sys`, `js-sys` and `web-sys` |
| "no `pkg-config` in the graph" | `--features packfile` puts `pkg-config` in the build graph today, dormant but present |
| "no `cc` in the graph" | `cc` is a build-dependency in every graph, on every target, including a default build with no features enabled |

None of those six properties separates a crate that vendors its own C from a
crate that needs something installed on the machine, which is the distinction
that actually matters.

### What is in the tree, measured

Everything below comes from
`cargo tree -p libviprs -e normal,build --target <triple>` plus
`cargo metadata`, on cargo 1.98, and the test re-derives it rather than trusting
this table. It does that on four fixed triples and on the host, which it reads
out of `rustc -vV` the same way the snippet below does, so a cell labelled
"host" means the machine running it rather than the machine I happened to write
it on.

Two ways to get this wrong, both of which cost me time. The first is skipping
`--target`: `cargo metadata` without `--filter-platform` reports every target's
dependencies at once, so an audit run on a mac will flag `wasm-bindgen-shared`
and the windows crates. The second is trusting `--filter-platform` to be enough.
It filters by target cfg and not by feature, so `cargo metadata`'s resolve graph
still lists optional dependencies that nothing turned on. Measured here, the mac
filter takes the resolve from 193 packages to 163 and every one of the 30 it
drops is target-gated (`wasm-bindgen` and its tail, the `windows-*` family,
`linux-raw-sys`, `js-sys`). An unenabled optional survives it untouched.
`cargo tree` is feature-aware and has neither problem, which is why the test
drives it and reads `cargo metadata` only for manifest facts.

Whatever you use to check that, do not expect to reproduce a specific crate
name: `Cargo.lock` is not committed, so two checkouts a day apart resolve
differently. On mine, `defmt` is the example, an unenabled optional of `chrono`,
`jiff` and `tinyvec` that declares `links = "defmt"` and that survives the mac
filter while `cargo tree` prints nothing for it. On a lockfile a fortnight older
(`jiff` 0.2.23 rather than 0.2.35) it is not in the graph at all. The two cases
are easy to tell apart, and worth telling apart: `cargo tree -i <crate>` warns
"nothing to print" when the crate is in the lockfile but no enabled feature
reaches it, and errors "did not match any packages" when it is not in the
lockfile at all. Reach for the general form instead of a name:

```sh
target="$(rustc -vV | sed -n 's/^host: //p')"
comm -13 \
  <(cargo tree -p libviprs -e normal,build --target "$target" --prefix none --format '{p}' \
      | awk '{print $1}' | sort -u) \
  <(cargo metadata --format-version 1 --filter-platform "$target" \
      | python3 -c 'import json,sys; [print(n["id"].split("#")[-1].split("@")[0]) for n in json.load(sys.stdin)["resolve"]["nodes"]]' \
      | sort -u)
```

Everything that prints is in the filtered resolve and is not in this build.
Expect dev-dependencies and the `fuzz` member's tree in there too, since the
metadata side covers the whole workspace while the tree side is one package and
`normal,build` edges. `defmt` sitting in that output on a mac is the point.

**Vendored native code.** Three crates ship compilable C or assembly *and* a
build script that can compile it:

| crate | native sources | reached by | what it emits |
|---|---|---|---|
| `blake3` 1.8.7 | 11 `.c`, 1 `.cpp`, 12 `.S`/`.asm` | **default**, a direct dependency | `cargo:rustc-link-lib=static=blake3_neon` and a link search into `OUT_DIR` |
| `zstd-sys` 2.0.16 | 42 `.c`, 1 `.S` | `packfile` only | `cargo:rustc-link-lib=static=zstd` and a link search into `OUT_DIR` |
| `rav1d` 1.1.0 | 48 `.asm`, 43 `.S`, 0 `.c` | `avif` only | **nothing**, because libviprs takes it with `default-features = false` |

`rav1d` is in that table for what it *ships*, not for what it does. The scan
is static (a crate ships native source and has a build script able to compile
it) and `rav1d` carries the whole dav1d assembly whether or not it is asked to
build any of it. libviprs asks it not to: the entire `mod asm` in its
`build.rs` is `#[cfg(feature = "asm")]`, and the `avif` feature takes the crate
with `default-features = false`, so no assembler runs and no object is
produced. Measured, with a positive control: a debug build emits zero `.o` and
zero `.a` under `target/debug/build/rav1d-*`, while `blake3` in the same tree
emits `blake3_neon.o` and `libblake3_neon.a`.

That matters for clause 1 as well as clause 3. With `asm` on, `rav1d` reaches
for `nasm` on x86_64, and `nasm` is an assembler somebody has to install; with
it off, `nasm-rs` is a dormant build-dependency and nothing looks for anything.
That is the same shape as `pkg-config` under `packfile`.

Both link directives are `static=` and both search paths point inside the
target directory, which is the whole point: the library is built here, from
source that came down with the crate, and nothing on the machine is consulted.
`libbz2-rs-sys`, despite the name, has neither a `build.rs` nor a single `.c`
file and belongs in no table.

That scan runs on every cell, not just the host ones, even though it works by
reading crate source directories off the disk. It can, because the lookup table
it reads them through comes from a `cargo metadata --all-features` with no
`--filter-platform`, so cargo has unpacked every package any cell can name,
`windows-sys` and `wasm-bindgen-shared` included.

On aarch64 the blake3 C compile is unconditional, so a default build genuinely
needs a working C compiler there. On x86_64 blake3 probes for one and falls
back to Rust intrinsics if it finds none, so the compiler is preferred rather
than required. Either way, `cc` runs.

**`links` keys in the resolved graph**, per target and per feature set:

| target | default | `--features packfile` | `--all-features` |
|---|---|---|---|
| `aarch64-apple-darwin` | `rayon-core` | `rayon-core`, `zstd-sys` | `rayon-core`, `zstd-sys` |
| `x86_64-unknown-linux-gnu` | `rayon-core` | `rayon-core`, `zstd-sys` | `rayon-core`, `zstd-sys` |
| `x86_64-pc-windows-msvc` | `rayon-core` | `rayon-core`, `zstd-sys` | `rayon-core`, `zstd-sys` |
| `wasm32-unknown-unknown` | `rayon-core` | `rayon-core`, `wasm-bindgen-shared`, `zstd-sys` | `rayon-core`, `wasm-bindgen-shared`, `zstd-sys` |

`wasm-bindgen-shared` is real on wasm32 rather than a metadata artifact, and
note where it sits in that table: not in the default wasm32 build, which is the
easy thing to miss if you only check default and `--all-features`.
`packfile` is what puts it there: `zip` turns on `getrandom`'s `wasm_js`
backend and pulls `time`'s wasm clock through `js-sys`, and both land on
`wasm-bindgen`. `--features pdfium` reaches it a second way, directly. It
declares `links = "wasm_bindgen"` for the same version-unification reason
`rayon-core` does, and it compiles nothing.

**Library-discovery crates.** `pkg-config`, `system-deps`, `vcpkg`, `cmake` and
`bindgen`: none of them is in any default graph on any target. `pkg-config`
appears under `packfile` and nowhere else. `cc` is deliberately not on that
list, because `cc` compiles vendored sources rather than discovering installed
ones, and the whole argument above is that those two are different things.

Those five names are hard-coded in the test rather than derived from anything,
so `autotools`, `metadeps` and every other build-time prober nobody wrote down
goes straight past it. The runtime-loader list next to it is three names and
has the same hole, which is the reason clause 2 stays a by-hand check.

**Runtime library loaders.** `libloading` is in the graph under `pdfium` and
`pdfium-static` and nowhere else, on every target except wasm32, where
`pdfium-render` gates it on `cfg(not(target_arch = "wasm32"))` and reaches
PDFium through `js-sys` instead. It is the only mechanical trace either pdfium
feature leaves anywhere.

**Dev-dependencies** get the same rule applied one notch looser, since they
never reach a consumer. The only thing they add today is `generator` 0.8.9,
which `loom` pulls and which assembles 10 `.s` stack-switching files through
`cc`. Vendored, so clause 3 covers it.

### `packfile` is a carve-out, on purpose

`--features packfile` pulls `zip` -> `zstd` -> `zstd-safe` -> `zstd-sys`, and
`zstd-sys` is the one crate here that looks like the thing the rule forbids. It
declares `links = "zstd"`, it vendors 42 `.c` files, and it drags `pkg-config`
into the build graph as an unconditional build-dependency of its own.

It stays, and it stays as a deliberate exception rather than an accident:

- It is **off by default**, so nobody who does not write packfiles pays for it.
- It **vendors**. As resolved here `zstd-sys` gets only its `std` feature, so
  `main()` takes the `compile_zstd()` branch and builds `zstd/lib` from the
  crate's own source. The 26 `.o` files land in the target directory.
- The `pkg-config` path exists but is **dormant**. It fires only if the crate's
  own `pkg-config` feature is on, which nothing here enables, or if
  `ZSTD_SYS_USE_PKG_CONFIG` is set in the environment. That environment
  variable is worth knowing about: it is a build-time escape hatch that no
  feature resolution will warn you about, and setting it turns a libviprs build
  into one that links a system zstd. Do not set it in CI.

The honest summary is that `packfile` costs a vendored C compile and carries a
dormant discovery path. That is a smaller thing than an installed library, and
it is the price of a zip container that reads what everyone else writes.

### `pdfium` is the real external library, and it is opt-in

This is the one place the crate does depend on something that is not in the
tree, so it needs saying plainly rather than being left for someone to find:

- `--features pdfium` links no library at build time. `pdfium-render` binds the
  PDFium API dynamically through `libloading`, so the dependency lands at
  **runtime**: the process needs a `libpdfium` shared library to load. The
  README's "PDFium setup" section is the install instructions for it.
- `--features pdfium-static` turns on `pdfium-render/static`, whose `build.rs`
  emits `cargo:rustc-link-lib=static=pdfium` and a link search read out of
  `PDFIUM_STATIC_LIB_PATH`. That one *is* a build-time link against a library
  from outside the tree.

`pdfium-render` itself ships no C and declares no `links` key, so no mechanical
check keyed on those would ever have noticed either of them. What the test
pins instead is that `pdfium-render` is absent from every default graph, which
is the property that actually matters: the external dependency is opt-in and
stays opt-in.

Its `bindings` feature would pull `bindgen` and run libclang over the PDFium
headers at build time. Nothing here enables it, and nothing should.

### Adding a dependency

Before you add one, check it against the rule. Start with the mechanical half,
which covers every target and every feature set in one go:

```sh
cargo test --test dependency_policy
```

Read the diff it prints rather than the assertion text: each failure shows the
set it found against the set the file expects, so the new crate is whatever is
on one side and not the other. Checking by hand instead means a single
`cargo tree` on one target with no `--features`, which is blind to exactly the
kind of dependency both carve-outs here are, so do not.

It costs 1.3 to 1.8 s warm on this machine: one `cargo metadata --all-features`
at 0.09 s, then one `cargo tree` per cell at 0.05 to 0.09 s, twenty-one of
them, plus a walk of each resolved crate's source directory looking for files a
C toolchain could compile. The number that actually hurts is a cold
multi-target resolve, where cargo has to fetch every manifest in four target
graphs before it can answer anything at all. Same reason `cargo test --offline`
fails here with a cargo error rather than a policy message on a machine that
never fetched the windows- and wasm-only manifests. Nothing is wrong with the
tree when that happens.

Then do the half no test can do, which is reading what the new crate's build
script emits:

```sh
target_dir="${CARGO_TARGET_DIR:-target}"
cargo build
grep -h 'rustc-link-lib\|rustc-link-search' "$target_dir"/debug/build/*/output || true
```

Note the `|| true`: grep exits non-zero when it matches nothing, which is the
good case here, and without it the line aborts any `set -e` script you paste it
into. Note the `$target_dir` too, since this project runs its local CI with
`CARGO_TARGET_DIR` pointed elsewhere and a bare `target/` would quietly grep an
empty or stale directory.

Every `rustc-link-lib` should be `static=` and every `rustc-link-search` should
point inside the target directory. If one points at `/usr/lib`,
`/opt/homebrew/lib` or anywhere else on the machine, the crate fails clause 2.

That leaves the part neither of those sees: a crate that needs an installed
library and says nothing about it at build time, the way `pdfium-render` does.
Read what it links or loads, and if it needs something that is not in the tree,
it is a carve-out and it needs a decision written down here, not a pin.

Then update `tests/dependency_policy.rs`, which will already be red, and write
the reasoning into the `[workspace.dependencies]` comment next to the new pin.
Every entry in that table carries the argument for why it is there and why its
feature list is what it is; that is the house style and it is not optional.

## Allocation instruments: one shape, two questions

There is now a counting `#[global_allocator]` in the core crate, in
`tests/convolution_image_sized_allocations.rs`, and #696 is planning another
one to prove that every image-sized allocation on a path went through the
fallible reservation helper. Two instruments answering roughly the same
question is how a third gets invented, so the call is made here rather than
per-lane: **there is one instrument shape, and #696 extends the existing one
rather than building a second.**

Concretely, the shape is the one that file already has, and it is deliberately
small: a `CountingAlloc` wrapping `System`, a thread-local threshold that starts
and ends at `usize::MAX` so an unarmed thread is never charged, a `measure`
window that restores every counter on the way out including on unwind, and a
process-global counter armed only inside the window so a path that fans out onto
a worker thread cannot go on measuring as if it had not. When a second binary
needs it, lift it into a shared test-support module and keep one copy. Do not
write a second counting allocator with different accounting.

One thing #696's issue text assumes and should not: it says a counting allocator
"has to live in its own test binary because it is process-global, which is why
the harness repo has it and the core crate does not". The first half is true and
the second does not follow. `#[global_allocator]` in an integration test is
scoped to that one test binary and reaches no other test, which is why this one
lives here instead of in `libviprs-tests`.

The two questions the one instrument answers are different, and which assertion
you write depends on which you are asking:

- **A budget.** "This named operation costs exactly N image-sized allocations
  and M bytes a pixel." Pin the exact measured values, never a padded ceiling,
  and cross-check every row at two image sizes and two carriers so a constant
  fitted to one image cannot pass as a rate. That is what the convolution
  budget file does.
- **A funnel.** "Every image-sized allocation on this path went through the
  fallible helper." Compare the same counter against the helper's
  `cfg(test)` hook consumptions. That is #696's, and the counter it needs is the
  one that already exists.

Either way the bar is the one the rest of this repo runs on: a guard that stays
green under a mutation of the thing it claims to guard is worthless, so break it
and watch it go red before you believe it. That applies to the instrument as
much as to the subject. Two of the counting allocator's four arms were provably
unguarded when it first landed, because nothing on the sharpen or canny path
allocates through `realloc` or `alloc_zeroed`, and the positive control now
exercises all four on purpose.

**The cost this puts on anything touching `src/convolution.rs`**, which is
worth knowing before you start rather than when the suite goes red: the budgets
there are pinned at exact values, **sixteen rows of two numbers**, each
cross-checked at two image sizes. The file covers `conv`, `sobel`, `gaussblur`,
`compass`, `sharpen` and `canny`, and the first four share one traversal, so a
change to what `Scan` holds live reddens ten rows at once and a change to the
sharpen or canny path reddens three. Whichever it is, the rows it moves need
re-measuring with the same evidence that set them, and the file's own doc table
is where the before-and-after goes. That is intended, not a bug in the guard: it
is the price of a budget with no slack in it, and a budget with slack in it
would not have caught either of the mutations it exists for, nor shown that
#575's row window took `conv` from 27 bytes a pixel to 3.

The file was called `sharpen_canny_image_sized_allocations.rs` until #575 put
`conv`, `sobel`, `gaussblur` and `compass` rows in it. Same instrument, same
accounting, one more set of rows, which is the rule above being followed rather
than an exception to it.

## Before you push

`make ci` runs the lot: `fmt`, `clippy`, `test`, `doc`, plus `miri` and `loom`,
which are the merge gate rather than the per-push CI. Any of those runs on its
own, and the README has the longer version.
