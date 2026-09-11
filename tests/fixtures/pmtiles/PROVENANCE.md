# PMTiles golden archives

Three PMTiles v3 archives. Every byte of every one of them was written by the real
`go-pmtiles`, not by libviprs and not by anyone reading the spec and doing the arithmetic by
hand. They are the independent oracle for EPIC F (#986): a writer tested against our own reader
round-trips perfectly through a shared misreading of the format, so the only thing that can
catch that class of bug is bytes from somebody else's implementation.

They live in the repository rather than beside the epic notes because the lanes that need them
(#988 the reader, #989 the writer, #991 the fuzz and interop suite) run their tests inside a
container that only ever sees the repository.

## The tool

| | |
|---|---|
| Repository | `protomaps/go-pmtiles` |
| Release tag | `v1.31.2`, published 2026-07-22T19:02:26Z |
| Source commit | `a3e4951ea6a0477b784c27c1dcbfd9c130878c5a` |
| `pmtiles version` says | `pmtiles 1.31.2, commit a3e4951ea6a0477b784c27c1dcbfd9c130878c5a, built at 2026-07-22T18:59:03Z` |
| Release tarball, linux/arm64 | sha256 `f8bd47e7ea866863489cad588fbaf2f31f42e5821f7a03f009b3769f05801cb1` |
| Release tarball, linux/amd64 | sha256 `3ed7dbf4ec2e6dfe5e25b6f70d1ffc932729f93c86db353bf514dd71010a312f` |
| `pmtiles` binary, linux/arm64 | sha256 `8cd0affde1ba5380b7cea6de0f94c674f88e4f586c77ae5820ea9652862691f4` |
| `pmtiles` binary, linux/amd64 | sha256 `a7e9ae10184d109c83f456ccdf6df4f3e2a64ba6cf69d9ed0f9f1840305055c1` |

The release carries no `checksums.txt`, so those sha256 values are measurements of the
downloaded assets, cross-checked against the sizes the GitHub API reports for the same assets.

## The files

| Archive | Bytes | sha256 |
|---|---|---|
| `raster-z0z2.pmtiles` | 1878 | `e2ed5e64f3c29efa3ec3b679ec5f1b06569c1b234c6eea762fb9f02fc23e9c12` |
| `dupes-z0z3.pmtiles` | 5007 | `bfc9db4c6ce6a04194e02b3d4815814adb05209f1aaba8591e4e1332f6e56a27` |
| `leaves-z0z7.pmtiles` | 869 | `fe5c9636be61abc60046d7f13837f8a3efb20ce3c38303644dac0cbec8248b8d` |

All three pass `pmtiles verify` with exit code 0.

## The exact commands that produced them

`go-pmtiles` cannot synthesise an archive from nothing. Of its ten subcommands `convert` is the
only one that builds an archive from raw tiles, and the only input it accepts is an MBTiles
database. `merge`, `extract`, `cluster` and `edit` all need an archive to exist already. So each
golden goes through an MBTiles, which means the tile *payloads* are ours and the *archive
layout* is entirely go-pmtiles'. That is the honest split and it is worth stating plainly.

The oracle image is built from `.epicF/oracle/Dockerfile.oracle`, which downloads the release and
verifies the sha256 with `sha256sum -c` so a mismatch fails the build rather than warning:

```
env -u DOCKER_DEFAULT_PLATFORM docker build --platform linux/arm64 \
    -f .epicF/oracle/Dockerfile.oracle -t libviprs-pmtiles-oracle:1.31.2-arm64 .epicF/oracle
```

`DOCKER_DEFAULT_PLATFORM` is `linux/amd64` in this workspace's shell and it silently overrides
`--platform` on the pull, so the `env -u` prefix is not optional.

Then, per archive (`.epicF/oracle/tools/make_mbtiles.py` writes the MBTiles with Python's
`sqlite3` and a hand-rolled PNG encoder, so it needs no image library):

```
env -u DOCKER_DEFAULT_PLATFORM docker run --rm --platform linux/arm64 \
    -v .epicF/oracle/tools:/tools:ro -v <workdir>:/work \
    libviprs-pmtiles-oracle:1.31.2-arm64 \
    python3 /tools/make_mbtiles.py /work

env -u DOCKER_DEFAULT_PLATFORM docker run --rm --platform linux/arm64 -v <workdir>:/work \
    libviprs-pmtiles-oracle:1.31.2-arm64 \
    pmtiles convert /work/raster-z0z2.mbtiles /work/raster-z0z2.pmtiles
```

and the same two lines for `dupes-z0z3` and `leaves-z0z7`. No other flags: no
`--no-deduplication`, no `--tmpdir`. MBTiles rows are TMS so the generator stores
`tile_row = 2**z - 1 - y`, and `convert` flips it back. The archives themselves are ZXY with no
Y inversion anywhere, which matches `Layout::Xyz`.

The MBTiles inputs are deterministic and are not committed. Regenerate them and check against:

| Input | Bytes | sha256 |
|---|---|---|
| `raster-z0z2.mbtiles` | 16384 | `04a4747abab9c2fff525572191f384b280e09992cda6fa3247ae98efbdbbca62` |
| `dupes-z0z3.mbtiles` | 24576 | `bc47336a4e4aad28134167bd0aa83b959fc50d9f87d7c7b76ee23d6b0055baf0` |
| `leaves-z0z7.mbtiles` | 2252800 | `ce993245f4aa39ea7cebc3212507f35f34a589dc4d241921e2bfbe66a903bb04` |

The whole route was run three times, twice on linux/arm64 in separate clean directories and once
on linux/amd64, and all three produced byte-identical archives. That was measured, not assumed.

## What each one is for

### raster-z0z2.pmtiles

The plain one, and the first thing a reader should be tested against. Zooms 0 to 2, 21 tiles,
every tile a distinct 8x8 solid-colour PNG so nothing can deduplicate, 74 bytes per tile.
go-pmtiles reported 21 addressed tiles, 21 entries after RLE, 21 tile contents and 35 bytes of
directory. Clustered, internal compression gzip, tile compression none, tile type png, one root
directory and no leaves.

### dupes-z0z3.pmtiles

The duplicate-handling one. Zooms 0 to 3, 85 addressed tiles over 63 distinct payloads, laid out
so the directory has to show both duplicate shapes at once:

* **runs.** All four z1 tiles are the same green PNG and all sixteen z2 tiles are the same red
  PNG. Those are consecutive in tile-id space, so go-pmtiles collapsed each group into one entry,
  with `run_length` 4 and 16.
* **repeated offsets that are not adjacent.** The red PNG is also the single z0 tile, so the
  entries for tile id 0 and tile id 5 both point at offset 0 while sitting 5 apart in id. Four
  scattered z3 tiles are the same blue PNG with distinct tiles between them, so no run can form:
  the entries for tile ids 21, 49, 63 and 76 all point at offset 148, each with `run_length` 1.

A reader that handles only the first shape returns the wrong bytes for the second and looks
correct on most archives. go-pmtiles reported 85 addressed, 67 entries after RLE, 63 contents and
59 bytes of directory, and the run lengths sum to 85, which is the header's addressed tile count.

### leaves-z0z7.pmtiles

**The only PMTiles fixture with real leaf directories that I know of**, the spec repository's own
fixtures included. Nothing upstream exercises the leaf entry offset base, and a writer and a
reader that make the same wrong choice about it round-trip perfectly and both look correct, so
this file is the only thing that can catch it.

Zooms 0 to 7, 21845 addressed tiles alternating between two 72-byte PNGs by `(x + y) % 2`.
Alternating means almost nothing collapses into a run, which is how an 869-byte archive ends up
carrying 21844 entries over 2 tile payloads. go-pmtiles pushed them into 6 leaf directories (4096
entries each for the first five, 1364 for the last) behind a root of 6 pointer entries, with 42
bytes of root and 391 bytes of leaves.

The thing to get right: a root entry with `run_length == 0` is a leaf pointer and its `offset` is
relative to `header.leaf_directory_offset`, not to the start of the file. Here
`leaf_directory_offset` is 334 and the six pointers carry relative offsets 0, 71, 136, 202, 268
and 333, landing at absolute 334, 405, 470, 536, 602 and 667. Separately, a *tile* entry found
inside a leaf is still relative to `tile_data_offset`, because the spec keys the base off the
entry kind and not off which directory the entry was found in. Every tile entry in every leaf
here carries offset 0 or 72, which lands at 725 or 797, and `tile_data_length` is 144, exactly
the two 72-byte payloads. Under either tempting wrong base those offsets land inside the leaf
directory region instead.

Two more numbers that line up and are worth asserting: the six leaf lengths sum to 391, which is
`header.leaf_directory_length`, and `leaf_directory_offset + leaf_directory_length` is 725, which
is `header.tile_data_offset`.

## One warning about the reference

`pmtiles verify` returning 0 is not evidence that an archive is safe to parse. It bounds-checks
each section's *length* against the file size and never checks `offset + length`, so a root
offset of 999999 in this 1878-byte file walks straight through it, and `DeserializeEntries`
discards the gzip error (`reader, _ = gzip.NewReader(data)`), so the next read dereferences a nil
pointer and the process dies with a SIGSEGV. Our parsers check `offset.checked_add(length)
<= archive_size` on every section and return a typed error where the reference crashes. Matching
go-pmtiles there would be matching a crash.
