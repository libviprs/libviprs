# JBIG2 fixtures for issue #1365

Every `.jbig2` stream and `.png` reference here is copied byte for byte from
`hayro-jbig2/tests/` of https://github.com/LaurenzV/hayro at commit
`dee147646768e41f62ff4c327c649f63f57fbee5` (`streams/<name>.jbig2`, `references/<name>.png`).

Licence: Apache-2.0 OR MIT (the crate's own, both texts ship with the crate). Their README says
the streams come from a JBIG2 encoder written for that suite, with the same feature coverage as
the SerenityOS test files, so no third party image data is in them.

They are whole files (T.88 Annex D.1, sequential, with a file header). The test turns each into
what a PDF holds (Annex D.3, embedded: no file header, the page 0 segments split off as
`/JBIG2Globals`), so the originals stay untouched and nothing derived is committed.

| stream | what it covers | reference |
|---|---|---|
| `generic_template0` | generic region, arithmetic | `truchet` |
| `generic_mmr` | generic region, MMR | `truchet` |
| `symbol` | symbol dictionary plus text region, arithmetic | `honeycomb` |
| `symbol_global` | the same with the dictionary in page 0 (globals) | `honeycomb` |
| `huffman_standard_tables` | symbol dictionary plus text region, Huffman | `honeycomb` |
| `symbol_refine_text_refine` | refinement inside a symbol dictionary and a text region | `honeycomb` |
| `refine_template0` | generic refinement region | `truchet` |
| `halftone` | pattern dictionary plus halftone region | `maze` |
| `halftone_global` | the same with the pattern dictionary in page 0 | `maze` |

The T.88 Annex H stream was not used: its bytes are not freely redistributable.
