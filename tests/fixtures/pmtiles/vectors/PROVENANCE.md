# go-pmtiles reference vectors

Five JSON files, every number in them produced by the real `protomaps/go-pmtiles` rather than by
libviprs code and rather than by reading the spec and doing the arithmetic by hand. They are the
independent half of the PMTiles work: our writer checked against our reader round-trips perfectly
even when both share a misreading of the format, so the reader is pinned against these and not
against the writer.

The `.pmtiles` archives the vectors describe live one directory up, next to their own provenance.

## The pin

| | |
|---|---|
| Repository | `protomaps/go-pmtiles` |
| Release tag | `v1.31.2` |
| Published | 2026-07-22T19:02:26Z |
| Source commit | `a3e4951ea6a0477b784c27c1dcbfd9c130878c5a` |
| `pmtiles version` says | `pmtiles 1.31.2, commit a3e4951ea6a0477b784c27c1dcbfd9c130878c5a, built at 2026-07-22T18:59:03Z` |
| `go-pmtiles_1.31.2_Linux_arm64.tar.gz` sha256 | `f8bd47e7ea866863489cad588fbaf2f31f42e5821f7a03f009b3769f05801cb1` |
| `go-pmtiles_1.31.2_Linux_x86_64.tar.gz` sha256 | `3ed7dbf4ec2e6dfe5e25b6f70d1ffc932729f93c86db353bf514dd71010a312f` |
| Measured | 2026-09-11, in a pinned container, on linux/arm64 |

Every file carries the same pin again in its own `produced_by` block, so a vector separated from
this page still says where it came from.

## What each file holds

| File | What is in it |
|---|---|
| `header.json` | the 127-byte header of each golden as hex, plus the field-by-field decoding `pmtiles.DeserializeHeader` gives for those same bytes, plus the verbatim stdout of `pmtiles show` |
| `directory.json` | the decoded entries and the raw compressed bytes of the root directory of the two leafless goldens |
| `directory-leaves.json` | the same for `leaves-z0z7.pmtiles`, including all 21844 entries across its 6 leaf directories, and the relative and absolute offset of every leaf |
| `tiles.json` | payload length and sha256 for a sample of coordinates in each golden, plus absent probes and out-of-range probes |
| `tileid.json` | 162 distinct `(z,x,y)` to tile id pairs, level bases, and the 12 rows that tell the candidate Hilbert conventions apart |

## Two sections that are evidence rather than targets

Both say so in their own text, and both are read that way by
`tests/pmtiles_reader.rs`:

* `tileid.json`'s `out_of_range_observations` and `tiles.json`'s `out_of_range`. `ZxyToID` masks an
  out-of-range x or y into a different, valid tile instead of refusing, so `pmtiles tile
  raster-z0z2.pmtiles 2 4 0` exits 0 and writes the payload of z2 `(0,0)`. Our reader refuses those
  coordinates with a typed error. Matching the reference here would be matching a bug.
* Anything implying `pmtiles verify` is a safety model. It bounds-checks `length > fileSize` and
  never `offset + length`, so a root offset of 999999 in an 1878-byte file walks straight through
  it and then nil-dereferences inside `compress/gzip`. A green from `verify` is worth almost
  nothing, a red from it is worth a lot.

## How they are consumed

`tests/pmtiles_reader.rs` loads each file by repo-relative path at run time and checks its sha256
against a constant in the test before parsing it, then checks the number of rows it parsed against
a pinned count. A transcribed literal is indistinguishable from an invented one, and a parse that
silently yields an empty set passes every assertion made over it, so both halves are there on
purpose.

The sha256 of each file as committed:

| File | sha256 |
|---|---|
| `directory-leaves.json` | `acc033de338def6a600a806a03cf803cacfe87470a3a8a059f0bace3b9320d58` |
| `directory.json` | `9f01472702fd4e93c3025bd9897cc336429a1bc05385f35ae555f238490e4d47` |
| `header.json` | `99258d11ea1fa9cd99c8b28a74ea1bf217e0dea87b4ee00776a6b0c1ea36f1c3` |
| `tileid.json` | `a486b48b09ab1b9d8f20208b992fc47b89ba67c235506f5f47cccd808e82b265` |
| `tiles.json` | `efaebeee9399d9e1e6e0395059caf38c659f134353442bfe29fd0065e5ae6581` |
