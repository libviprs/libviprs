# Vector providers: the ACadSharp dependency, and building without it

The vector path (DWG to a streamable PMTiles archive) has one seam where a drawing
format enters: the `CadDecoder` trait. Everything after it, the tiler, the MVT
encoder, the PMTiles writer and the whole of `libviprs::vector::convert`, works on
`cad::Primitive` values and never sees a parser. A provider is whatever sits on the
other side of that trait.

The provider libviprs ships is ACadSharp, reached through the `acadsharp-rs` crate.
It is behind the `dwg-acadsharp` Cargo feature, and it is the only thing in the
library that needs a native binary.

## What the ACadSharp provider depends on

```text
libviprs  --(feature dwg-acadsharp)-->  acadsharp-rs  -->  NativeAOT archive
(tiler, MVT, PMTiles)                   (safe decoder)     (ACadSharp, from libviprs-dep)
```

* `acadsharp-rs` is a git dependency pinned by revision in `Cargo.toml`, because it
  is not on crates.io yet. It is only fetched and compiled when the feature is on.
* The decoder itself is ACadSharp compiled with NativeAOT by `libviprs-dep`, shipped
  as a release archive (`acadsharp-linux-x64.tgz`, and a macOS arm64 one). Nothing
  needs .NET at run time: `acadsharp-rs`'s build script links the archive.
* That build script finds the archive through `ACADSHARP_NATIVE_DIR` (a directory
  holding `lib/` and `metadata/LINKINFO.json`) or the cache under
  `$CARGO_HOME/acadsharp-native/`. It downloads nothing. With no archive the crate
  still builds, and the decoder reports that it is unlinked when you call
  `AcadSharpDecoder::new()`. That is a `CadError::Provider`, not a missing type.
* The reader is not a sandbox. A drawing you did not make should be opened with
  `ProviderLimits::hardened()` (the `--hardened` flag on the CLI) and run in a
  process with its own memory and wall-clock ceiling. See
  `docs/vector-performance-report.md` for what that was measured to hold.

Supported DWG versions are AC1014 through AC1032 (R14 to 2018).

## Building without it

The feature is off by default. A plain `cargo build` has no DWG reader and no native
dependency, and everything that does not read a DWG still works:

```text
cargo build                              # no provider
cargo build --features dwg-acadsharp     # with the ACadSharp provider
```

Without the feature:

* `vector::convert::default_decoder` returns `ConvertError::ProviderNotBuilt`, whose
  message names the `dwg-acadsharp` feature to rebuild with.
* `inspect` and `check` (archive info and validation) are unaffected, because they
  read a PMTiles file and not a drawing.
* `convert` and `report` take any `CadDrawing`, so a decoder you wrote yourself, or
  the provider of some future native-Rust reader, converts exactly the same way.

In the command line, `viprs vector info` and `viprs vector validate` work in every
build. `views`, `build` and `report` open a DWG, so a build without the feature
exits 1 with the same message instead of a generic "unsupported format".

## Writing another provider

Implement `CadDecoder` (probe and open) and `CadDrawing` (views and decode). The
contract, including what a decoder owes the `DecodeReport`, is in the `libviprs::cad`
module documentation, and that page's example is a complete decoder built from a
loop. A provider crate that depends on libviprs and hands its own decoder to
`vector::convert::convert` needs nothing else from this crate.
