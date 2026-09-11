# PMTiles reference vectors, and where every number in them came from

Four JSON files, copied byte for byte out of the EPIC F oracle capture. Nothing here was produced
by libviprs code and nothing here was produced by reading the v3 specification and doing the
arithmetic by hand. I copied them into the repository rather than referring to the capture
directory because only the repository is mounted into the gate container, so a path outside it
resolves to nothing at test time.

The archives these describe live one directory up, in `tests/fixtures/pmtiles/`, with their own
provenance note.

## The pin

| | |
|---|---|
| Repository | `protomaps/go-pmtiles` |
| Release tag | `v1.31.2` |
| Source commit | `a3e4951ea6a0477b784c27c1dcbfd9c130878c5a` |
| `pmtiles version` says | `pmtiles 1.31.2, commit a3e4951ea6a0477b784c27c1dcbfd9c130878c5a, built at 2026-07-22T18:59:03Z` |
| Linux arm64 release tarball sha256 | `f8bd47e7ea866863489cad588fbaf2f31f42e5821f7a03f009b3769f05801cb1` |
| Linux amd64 release tarball sha256 | `3ed7dbf4ec2e6dfe5e25b6f70d1ffc932729f93c86db353bf514dd71010a312f` |
| Measured on | 2026-09-11, in a `debian:bookworm-slim` container on linux/arm64 |

Each JSON carries the same pin in its own `produced_by` block, so a file that gets separated from
this note still says which binary wrote it.

## The files, and their sha256 as committed

| File | Bytes | sha256 |
|---|---|---|
| `tileid.json` | 55278 | `a486b48b09ab1b9d8f20208b992fc47b89ba67c235506f5f47cccd808e82b265` |
| `header.json` | 8658 | `99258d11ea1fa9cd99c8b28a74ea1bf217e0dea87b4ee00776a6b0c1ea36f1c3` |
| `directory.json` | 22915 | `9f01472702fd4e93c3025bd9897cc336429a1bc05385f35ae555f238490e4d47` |
| `directory-leaves.json` | 471350 | `acc033de338def6a600a806a03cf803cacfe87470a3a8a059f0bace3b9320d58` |

`tests/common/pmtiles_oracle.rs` pins those four digests and refuses to hand a file to a test if
its bytes have changed, so an edit to a vector is a red suite rather than a quietly moved target.
Every test that consumes one also asserts the number of rows it parsed against the `counts` block
inside the file, because a parse that silently yields nothing passes every assertion in the loop
that follows it.

## How the numbers were obtained

The CLI cannot print a tile id, a directory entry or the header's offsets, so the library was
called directly. The capture cloned `protomaps/go-pmtiles`, checked out `v1.31.2`, confirmed
`git rev-parse HEAD`, dropped a dumper into the checkout as `oracledump/main.go` and built it
**inside that module**, so the import resolves to the pinned source rather than to whatever the
module proxy would hand a fresh `go get`. The Go toolchain was `go1.27.1 linux/arm64`.

```
env -u DOCKER_DEFAULT_PLATFORM docker run --rm --platform linux/arm64 \
    -v <scratch>/src:/src -v <oracle>/tools:/tools:ro -v <scratch>/work:/work \
    -v <scratch>/gocache:/gocache -v <scratch>/gomodcache:/gomodcache -v <scratch>/gotmp:/gotmp \
    -e GOCACHE=/gocache -e GOMODCACHE=/gomodcache -e GOTMPDIR=/gotmp -e TMPDIR=/gotmp \
    -w /src/repo golang:1.27-bookworm sh -c '
      git clone https://github.com/protomaps/go-pmtiles.git /src/repo && cd /src/repo
      git checkout v1.31.2
      mkdir -p oracledump && cp /tools/oracle_dump.go oracledump/main.go
      go build -o /work/oracle_dump ./oracledump
      /work/oracle_dump tileids
      /work/oracle_dump archive /work/raster-z0z2.pmtiles'
```

| File | Which go-pmtiles functions produced it |
|---|---|
| `tileid.json` | `pmtiles.ZxyToID`, `pmtiles.IDToZxy`, `pmtiles.ParentID` |
| `header.json` | `pmtiles.DeserializeHeader`, plus the CLI's `show`, `show --header-json` and `verify` |
| `directory.json` | `pmtiles.DeserializeEntries` and `pmtiles.SerializeEntries`, plus raw bytes read straight out of the archive |
| `directory-leaves.json` | the same, for the leaf-bearing golden |

## Three rows that are evidence, not targets

`tileid.json` has an `out_of_range_observations` section and it is **not a set of expectations**.
`ZxyToID` masks an out-of-range `x` or `y` into a different, valid tile rather than refusing it, so
`(z=2, x=4, y=0)` comes back as tile id 5, which is really `(2, 0, 0)`. It saturates above zoom 31
too: z=32, z=33 and z=63 all return the same id, and `IDToZxy` then disagrees with `ZxyToID` about
what that id means. libviprs refuses all of those, per the crate's `try_*` convention, and
`tests/pmtiles_proptest.rs` asserts the refusal rather than the value. The file says the same thing
in its own text.

The section that **is** a target is `convention_discriminators`, twelve rows at zoom 9 through 15.
They matter because 150 of the 162 pairs in the file are structural coordinates that four different
candidate mappings all agree on, plain Z-order included, so passing those proves much less than it
looks. The discriminators are off every quadrant boundary, have `x` and `y` of differing parity,
and include swapped pairs (`(13, 5107, 2884)` is 79053962 while `(13, 2884, 5107)` is 53847284), so
neither a symmetric mapping nor a Z-order one survives them.
