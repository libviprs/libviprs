# v1 text capability policy

Status: decided by N5.5 (#1084), part of the integration epic #1039. Feeds N5.6, N6.6, N6.7 and N8.8.
The code is `src/pvec/text_policy.rs`, and the numbers below come from
`examples/text_policy_measure.rs`.

## Decision

Decision: omit text and report the counts.

v1 does not carry text into PVec at all. No string, position, matrix, font name, size or glyph outline
reaches a geometry block. What v1 does is count: every PDF text-showing operator (`Tj`, `TJ`, `'`,
`"`) and every DWG `TEXT` and `MTEXT` entity is counted in the conversion report, with the operator
or class kept apart, so an omitted label is a number somebody can see and not a silent hole.

The other two options are not rejected for ever. They are rejected for v1, for the reasons in the
measurements.

## Why

The three candidates, priced on the same pages:

* **Omit and count.** 0 bytes in the artifact, no dependency, no new claim.
* **Preserve a text component** (string, position, matrix, font name, size, no outlines). About 7.5%
  of the size of the path data on the same pages.
* **Convert glyphs to outlines.** About 1.8 times the size of the path data, and it needs shaping and
  outline code for every font a drawing can name.

That last cost is the deciding one. The crate ships exactly one face, Bitstream Vera Sans
(`fonts/Vera.ttf`, `VERA_TTF` in `src/create.rs`, there for `Raster::text`). A drawing's text is set in
fonts the drawing names, and 24 of the 26 PDFs below have text in embedded subset fonts (or in fonts that
are not embedded at all), so converting to outlines means reading embedded font programs (TrueType,
CFF, Type 1, CID-keyed) and, for DWG, SHX shape files and the AutoCAD font substitution rules. None of
that exists here, and `CONTRIBUTING.md`'s dependency rule makes each piece a reviewed choice. Preserving
without outlines is cheap, but a consumer that has the string and no font cannot draw it, so it buys
search and selection and nothing visual, and that is a component (see below), not geometry.

So v1 omits, and the report keeps the omission honest.

## Measurements

### What I could and could not measure

The issue asked for the numbers on named `O2.1` and `O2.2` fixtures, which did not exist when this was
written (`libviprs/pvec-gold-fixtures` #11 and #12 were open). They have landed since, and the four
text-bearing DWGs are re-taken under "Re-measured on the corpus fixtures" below. The PDF side and the rest of
the DWG side still use what was on hand, labelled:

* **PDF.** The 26 licensed candidates under `pdf-candidates/` that are under 3 MB (of 35, each with its
  licence and sha256 in the `CANDIDATES.md` beside it), page 1 of each. They are real-world drawings,
  maps and forms, which is what `O2.2` is meant to curate, but they are not `O2.2`.
* **DWG.** The committed fixtures under `tests/fixtures/cad/` and `tests/fixtures/pvec/provider/`,
  read through the acadrust provider. They are writer-made probes, not known-truth drawings.

The PDF numbers are not re-taken: the O2 corpus holds no PDF, so the 26 candidates stay the PDF record.

Native x86_64 on MARS, `cargo run --release --features dwg-acadrust --example text_policy_measure`.

### Size model

A model, not an artifact. There is no PVec artifact writer for text (that is P3.7 and N6.6), so
sizes are priced from the page:

* omit: 0 bytes
* preserve: string bytes, plus 48 bytes of text matrix (six `f64`), 8 bytes of size and the font name
  length, per operator
* outline: one string byte is one glyph (too high for CID fonts), at the Vera average of 33.06
  cubic control points a glyph (quadratics promoted to cubics), 16 bytes a point, plus a 48 byte placement
* path bytes: the page's own paths at 16 bytes a point, so the deltas have a denominator

The preserve and outline columns read the page's own content stream only. Text inside a Form XObject is
counted by the adapter (the `Tj` and `TJ` columns) but not priced, so for a page whose text lives in
forms the byte columns are a lower bound. `usace_norfolk_hoskins_creek_dredging_plans.pdf` is the
extreme case: 7904 operators, all in forms, priced at 0.

### PDF, page 1

| fixture | paths | path bytes | `Tj` | `TJ` | operators | preserve bytes | outline bytes |
|---|---|---|---|---|---|---|---|
| `USTopo_quickstart.pdf` | 16 | 1024 | 366 | 165 | 531 | 34304 | 1716638 |
| `arasikere-gp-2015-A0.pdf` | 1953 | 1438352 | 1 | 434 | 435 | 30712 | 2912226 |
| `bbmp-ward1-2020-A2.pdf` | 3733 | 807872 | 1647 | 182 | 1829 | 110063 | 2297121 |
| `bbmp-ward1-yelahanka.pdf` | 2760 | 2411680 | 5484 | 0 | 5484 | 359818 | 3164384 |
| `census-harrisonburg.pdf` | 4752 | 1924608 | 6242 | 9 | 6251 | 373908 | 6549191 |
| `census_metro_micro_02_Alaska_2020.pdf` | 1519 | 3382560 | 15 | 37 | 52 | 3968 | 507201 |
| `census_metro_micro_06_California_2020.pdf` | 191 | 687072 | 38 | 124 | 162 | 11312 | 964779 |
| `census_metro_micro_12_Florida_2020.pdf` | 191 | 285152 | 78 | 116 | 194 | 13328 | 1035753 |
| `census_metro_micro_56_Wyoming_2020.pdf` | 86 | 90384 | 13 | 44 | 57 | 4057 | 380257 |
| `cms-higgs-tth-A4.pdf` | 30 | 10592 | 1 | 52 | 53 | 4316 | 671652 |
| `composter-blueprints-A4.pdf` | 4 | 256 | 0 | 5 | 5 | 402 | 64626 |
| `emb-e-v1-A4-landscape.pdf` | 124 | 11296 | 300 | 20 | 320 | 22873 | 2298852 |
| `f1040.pdf` | 502 | 22592 | 1146 | 0 | 1146 | 73833 | 2927228 |
| `f8949.pdf` | 222 | 7360 | 999 | 0 | 999 | 63478 | 2041501 |
| `iso5457-templates-A0-A4.pdf` | 82 | 2688 | 80 | 0 | 80 | 4990 | 63472 |
| `rabakavi-banahatti-covid-map-A3.pdf` | 121 | 517728 | 628 | 0 | 628 | 41022 | 362369 |
| `rilievo-palazzo-giulini-A1.pdf` | 18731 | 6421872 | 0 | 0 | 0 | 0 | 0 |
| `sa-native-title-map-A1.pdf` | 4217 | 2895232 | 6048 | 0 | 6048 | 399500 | 3489824 |
| `settlement-village-plan-A4.pdf` | 0 | 0 | 1 | 0 | 1 | 117 | 33467 |
| `us1325904-patent-drawing-A4.pdf` | 5 | 320 | 0 | 0 | 0 | 0 | 0 |
| `usace_hnc_sac_6-10_large_2022.pdf` | 117 | 25248 | 49 | 36 | 85 | 5711 | 401606 |
| `usace_hnc_std_drawings_fire_station_set1_2021.pdf` | 5479 | 236480 | 43 | 34 | 77 | 5087 | 313899 |
| `usace_hnc_yc_11-18_large_2019.pdf` | 117 | 25248 | 257 | 16 | 273 | 17537 | 825140 |
| `usace_norfolk_hoskins_creek_dredging_plans.pdf` | 1428 | 702384 | 7747 | 157 | 7904 | 0 | 0 |
| `usace_sam_tda_adjacency_diagrams_2026.pdf` | 15250 | 510048 | 14 | 312 | 326 | 24291 | 2917996 |
| `usace_swf_gpw_zone2a_large_arch.pdf` | 112 | 11136 | 284 | 140 | 424 | 29091 | 2351361 |
| total (26) | 61742 | 22429184 | | | 33364 | 1633718 | 38290543 |

`'` and `"` are 0 on every file. 24 of the 26 files show text, and the two that do not
(`rilievo-palazzo-giulini-A1.pdf`, `us1325904-patent-drawing-A4.pdf`) are a vector survey and a scan.

Over the 25 priced files (everything except the Hoskins Creek set, whose text is all in forms) the
path bytes are 21,726,800, the preserve column is 1,633,718 (7.5%) and the outline column is
38,290,543 (176%). Omit is 0. The font buckets over all 26 pages: 32,921 operators in a
subset-embedded font, 442 in a font that is not embedded, 1 in a fully embedded font, and none in a CID
font without a `/ToUnicode` (that case has a test fixture and no real file in this set).

### DWG

| file | `TEXT` | `MTEXT` |
|---|---|---|
| `g13_text.dwg` | 1 | 1 |
| `g13_long_text.dwg` | 0 | 1 |
| `g11_codepage.dwg` | 2 | 0 |
| `g11_shapes.dwg` | 1 | 0 |
| `sample_AC1018.dwg` | 29 | 5 |

The other 46 readable fixtures have no text. `g13_ac1009.dwg` is refused by the provider (old version stamp), so
it has no count at all, which is reported as the refusal and not as zero.

#### Re-measured on the corpus fixtures (#1381)

The four fixtures #1313 names, from `libviprs/pvec-gold-fixtures` at `1df9d1b`
(`fixtures/negative/<id>/source.dwg`), read through the acadrust provider with
`cargo run --release --example text_policy_measure --features "dwg-acadrust pvec-normalized" -- <files>`.
`VIPRS_INSTALLED_SHX` names the SHX fonts that count as installed; the bucket columns are
`TEXT_SHX_MISSING` with none installed and `TEXT_FONT_EMBEDDED` (read: available) with `txt.shx`.

| fixture | sha256 | entities | `TEXT` | `MTEXT` | style : font | none installed | `txt.shx` installed |
|---|---|---|---|---|---|---|---|
| `text-001` | `e233d9cd...` | 1 | 1 | 0 | `Standard` : `txt` | `TEXT_SHX_MISSING` 1 | `TEXT_FONT_EMBEDDED` 1 |
| `mtext-001` | `c09f6c0f...` | 1 | 0 | 1 | `Standard` : `txt` | `TEXT_SHX_MISSING` 1 | `TEXT_FONT_EMBEDDED` 1 |
| `table-001` | `aed7b952...` | 1 | 0 | 0 | none | none | none |
| `dimension-001` | `48bfc6d5...` | 1 | 0 | 0 | none | none | none |

Two things this shows that the stand-ins could not. First, the real text fixtures use the default
`Standard` style with the SHX font `txt` (no extension), and `shx_outcome` already compares without the
extension, so `txt.shx` installed is enough. Second, `table-001` and `dimension-001` hold one entity each
and the provider reports it as `Unmapped` (an `ACAD_TABLE` and a `DIMENSION`), so the `TEXT` and `MTEXT`
counters say 0 while the drawing does contain text inside them. Under v1 that text is omitted as part of an
unmapped entity and counted as one `UnmappedSourceClass`, not as text. If a later version keeps table or
dimension text, the counters have to learn those two classes first. None of the four names an SHX font that
is missing from a stock install, so the missing-font outcome is still exercised by a drawing written for
the purpose (`tests/pvec_1381_text_style.rs`, style `GONE` naming `nope.shx`).

The DWG size delta is not priced. The provider does not carry a string, height or style on
`RawGeometry::Unmapped`, so there is nothing to price honestly, and the size model above would be
invented. Under the outline option it would be the same per-glyph arithmetic as the PDF side.

## Missing-font behaviour

Under v1 every case below is omitted, because all text is. A case decides only the **report bucket**,
and says what a later version that kept the text could and could not recover. Each has a stable code in
`TextFontOutcome::code` and a test in `tests/pvec_text_policy_1084.rs`.

| case | bucket | what is recoverable later | fixture |
|---|---|---|---|
| a font that is not embedded | `TEXT_FONT_NOT_EMBEDDED` | the string, not the shapes | `n5_5_non_embedded_font`, and the Helvetica variant with no descriptor |
| a subset-embedded font (`ABCDEF+Name`) | `TEXT_FONT_SUBSET` | the string, and only the glyphs the drawing used, so no substitution | `n5_5_subset_embedded_font` |
| a CID (Type0) font with no `/ToUnicode` | `TEXT_CID_NO_TOUNICODE` | neither: glyph ids with no mapping to characters | `n5_5_cid_font_without_tounicode` |
| a DWG SHX font that is not present | `TEXT_SHX_MISSING` | the string, not the shapes | `n5_5_a_missing_shx_font_has_its_own_bucket` |

Two more buckets exist so the counts add up. `TEXT_FONT_EMBEDDED` is a full embedded program (or, for
DWG, an SHX font that is present). `TEXT_FONT_UNRESOLVED` is a text operator with no resolvable font
(no `Tf`, or a name missing from `/Font`). A CID font with a `/ToUnicode` is bucketed by its embedding,
like any other font. The font is graphics state, so `q` and `Q` restore it.

The PDF cases run on in-memory PDFs, so each is deterministic. The SHX case used to run only on the pure
function `shx_outcome` and hand-built entities. Since #1381 the provider boundary carries the text style
(`DwgProvider::text_styles`, `Facet::TextStyle`, `TextStyleRef`), `TextReport::bucket_dwg_entities` reads
the font from it, and `tests/pvec_1381_text_style.rs` writes a DWG whose style names `nope.shx`, reads it
back through the acadrust provider and lands in `TEXT_SHX_MISSING` with no hand-built entity. An entity
on a style the table does not have, or a style that names no font, is `TEXT_FONT_UNRESOLVED`: nothing
is claimed about a font nobody named.

## What the report says

A page's `PdfVectorPage::text` is a `TextReport`:

* `pdf.tj`, `pdf.tj_array`, `pdf.quote`, `pdf.double_quote`, forms expanded in place, so a form
  invoked twice counts twice
* `dwg.text` and `dwg.mtext`, from `TextReport::count_dwg` over a provider's entities
* `by_outcome`, the omitted items by bucket
* `omitted()`, the total

An omitted text entity is therefore visible as a count even though nothing of it reaches the output. A
page with no text reports zero, and zero is a statement, not an absence.

## Where text lives if it is preserved later

In a component stream (P3.7, emitted by N6.6), never in a canonical geometry block. A text item carries
the string, the placement matrix, the font name and size, plus the bucket from this document, and
it is addressed by the same provenance as the path it sits near. The test
`n5_5_text_never_becomes_geometry` reads the page back and asserts that the entities with text are
exactly the entities without it. Nothing in this document changes that: today there is no text
component, so there is nothing to read.

## Dependencies

No font or shaping dependency is added, because the decision is not outlines. `ab_glyph` is already in
the tree for `Raster::text` and the example uses it to measure Vera's outlines, but nothing under
`src/pvec/` uses it. `cargo test --test dependency_policy` is unaffected.

## Not done

* The measurements on the rest of the `O2.x` corpus (only the four text-bearing negatives named in #1313
  are re-taken above).
* Bidirectional and vertical writing, font substitution, and text in the browser add-on (out of scope
  by the issue).
* Pricing the DWG options.
