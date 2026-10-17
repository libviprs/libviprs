# CAD fixtures

`sample_AC1018.dwg` is ACadSharp's own `samples/sample_AC1018.dwg`, taken from
https://github.com/DomCR/ACadSharp at v3.7.1 (commit
`d7dc111023477d8a9fffc2153139459c95b4f345`), MIT licensed, licence in
`LICENSE-ACadSharp.txt`.

- bytes: 1097214
- sha256: `7d26f908516ec5fb89cc8054af33aff233b68a7b63208bbc5b7f9f4d9bff9866`

It is the same file as `acadsharp-rs-tests/fixtures/dwg/sample_AC1018.dwg`.

## `g13/`

Every `g11_*.dwg` and `g13_*.dwg` from `libviprs-dep`, `acadsharp/tests/fixtures/`, at commit
`0d475bf` (its `PROVENANCE.md` there says how each was written: ACadSharp's own `DwgWriter` at
3.7.1, content ours, one file per entity kind the flattener handles or refuses, each with a
top-level half and a block half so a defect that drops nested instances cannot pass). `SHA256SUMS`
lists them. They are the small inputs the decoder, tiler and fidelity tests are written against,
where `sample_AC1018.dwg` is the one large real drawing.
