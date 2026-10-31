//! The rules a producer follows to put a raster into a PVec (issue #1333, RL2.2).
//!
//! ADR 0018 (libviprs/pvec) puts a list of guarantees on the producer, and they are spread over
//! the pyramid, the tile table, the placement, the PDF bake and the PMTiles metadata. This is the
//! list in one place: [`RASTER_PRODUCER_RULES`] is the data, [`render_rules_table`] is the table
//! `docs/pvec-raster.md` carries between its markers, and a test fails when the two differ. The
//! code that keeps each rule is in [`super::pyramid`], [`super::tiles`], [`super::placement`] and
//! `pvec::pdf_image`.

/// One producer rule: where it comes from, so a reader can find the sentence that asks for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProducerRule {
    /// A short stable name.
    pub id: &'static str,
    /// What the producer does.
    pub rule: &'static str,
    /// The ADR 0018 item or `RASTER-NN` / `LIMIT-NN` / `NP-NN` code it comes from.
    pub source: &'static str,
}

/// Every rule, in the order the page lists them.
pub const RASTER_PRODUCER_RULES: &[ProducerRule] = &[
    ProducerRule {
        id: "centring-off",
        rule: "Centring is off: the tile grid is anchored at pixel (0, 0) with no offset.",
        source: "ADR 0018 Producers; RASTER-01",
    },
    ProducerRule {
        id: "overlap-0",
        rule: "Tile overlap is 0.",
        source: "ADR 0018 Producers; RASTER-01",
    },
    ProducerRule {
        id: "google-tiles-past-the-grid-dropped",
        rule: "The plan is the Google layout, whose grid is a power of two on a side, so the tiles beyond `cols` by `rows` of a level are padding and are dropped (counted in the report).",
        source: "ADR 0018 Producers; RASTER-02",
    },
    ProducerRule {
        id: "skip-blanks-not-stored",
        rule: "`skip_blanks` leaves a tile out. PVec stores that as a zero-length index entry, a not-stored tile that draws nothing.",
        source: "ADR 0018 Producers; RASTER-02",
    },
    ProducerRule {
        id: "placeholder-refused",
        rule: "`BlankTileStrategy::Placeholder` writes a marker that is not an image, so it is refused.",
        source: "ADR 0018 Producers; RASTER-03",
    },
    ProducerRule {
        id: "raw-refused",
        rule: "`TileFormat::Raw` carries no dimensions and no pixel format, so it is refused.",
        source: "ADR 0018 Producers; RASTER-04",
    },
    ProducerRule {
        id: "bit-depth-16-converted",
        rule: "A 16-bit source is converted to 8 bits before tiling (`v * 255 / 65535`, rounded) and the report says `converted-bit-depth`.",
        source: "ADR 0018 pixel_format; RASTER-03",
    },
    ProducerRule {
        id: "palette-converted",
        rule: "A palette source is expanded to gray or RGB before tiling. A palette PNG is not a legal tile.",
        source: "ADR 0018 pixel_format; RASTER-03",
    },
    ProducerRule {
        id: "legal-codec-and-format-pairs",
        rule: "Only the rows of `pvec-raster-tiles.md` are written: PNG in gray (1 or 8 bit), RGB and RGBA, JPEG in gray and RGB, lossless WebP in gray, RGB and RGBA. JPEG never carries alpha.",
        source: "ADR 0018 pixel_format; RASTER-04",
    },
    ProducerRule {
        id: "tile-edge",
        rule: "The tile edge is 256, 512 or 1024.",
        source: "ADR 0018 tile_edge; RASTER-01",
    },
    ProducerRule {
        id: "limit-15",
        rule: "A stored tile is at most `2 * 4 * edge^2 + 65,536` bytes, checked at encode time, before anything reaches PVec's writer.",
        source: "LIMIT-15 (docs/SECURITY_LIMITS.md in libviprs/pvec)",
    },
    ProducerRule {
        id: "no-interlace",
        rule: "PNG tiles are not interlaced (no Adam7), and the pyramid never writes a palette PNG.",
        source: "ADR 0018 Producers; RASTER-03",
    },
    ProducerRule {
        id: "raster-id-draw-order",
        rule: "A `RasterId` is dense and assigned in source draw order (for a PDF, content-stream order).",
        source: "ADR 0018 Producers (M16)",
    },
    ProducerRule {
        id: "negative-zero-normalised",
        rule: "`-0.0` in a placement is normalised to `+0.0`.",
        source: "ADR 0018 placement; RASTER-05; NP-03",
    },
    ProducerRule {
        id: "attribution-plain-text",
        rule: "Attribution is plain text of at most 255 bytes, with no control characters, line separators or bidirectional overrides.",
        source: "ADR 0018 Licensing; RASTER-07",
    },
    ProducerRule {
        id: "source-digest-sha-256",
        rule: "`source_digest` is the SHA-256 of the samples as the producer received them (32 bytes).",
        source: "ADR 0018 Licensing; RASTER-07",
    },
    ProducerRule {
        id: "placement-np-19",
        rule: "The placement passes NP-19: every coefficient finite, the four linear entries at most 2^20, the map invertible. libviprs checks it for a PDF placement before the write, and PVec's writer checks every placement.",
        source: "ADR 0018 placement; RASTER-05; NP-19",
    },
    ProducerRule {
        id: "geo-transform-permutation",
        rule: "libviprs's `GeoTransform` `[a b c d e f]` becomes PVec's `Transform2` order `[a b d e c f]`.",
        source: "ADR 0018 Georeference",
    },
    ProducerRule {
        id: "pixel-origin-corner",
        rule: "The pixel origin is the top-left corner of pixel (0, 0). A world file or a `PixelIsPoint` transform is shifted by half a pixel first.",
        source: "ADR 0018 Georeference",
    },
    ProducerRule {
        id: "composition-precondition",
        rule: "The placement is composed only when the raster's CRS is the calibration's native CRS and the calibration is affine, with metres per unit folded in. Anything else is a typed `PlacementError`, and nothing is resampled.",
        source: "ADR 0018 Georeference",
    },
    ProducerRule {
        id: "geo-transform-in-pmtiles-metadata",
        rule: "The `GeoTransform` is stored in the PMTiles metadata under `vnd.libviprs.geo_transform` (option D'). No CRS and no geographic transform is written into a `.pvec`.",
        source: "ADR 0018 Georeference (option D')",
    },
    ProducerRule {
        id: "clip-baked-into-alpha",
        rule: "A clip that is not a pixel-aligned rectangle is baked into the alpha channel and `visible_box` is its bounding box. An aligned rectangle sets `visible_box` only.",
        source: "ADR 0018 Producers (M16)",
    },
    ProducerRule {
        id: "clipped-image-lossless-tiles",
        rule: "An image with a baked alpha channel is written as PNG or WebP, never JPEG.",
        source: "ADR 0018 Producers; RASTER-04",
    },
    ProducerRule {
        id: "pdf-colour-and-mask-bake",
        rule: "PDF images are baked before tiling: `/SMask` into alpha, `/ImageMask` into RGBA in the fill colour, `/Indexed`, `/ICCBased`, `/CalGray`, `/CalRGB` and `/Lab` into sRGB (approximated, reported), `/Decode` applied, the unit square through the CTM. `JPXDecode`, `JBIG2Decode`, `CCITTFaxDecode`, `/Separation`, `/DeviceN`, a graphics-state soft mask and a singular CTM are refused by name.",
        source: "ADR 0018 Producers (M17)",
    },
    ProducerRule {
        id: "band-from-paint-order",
        rule: "An image painted before every painted path is band 0 (underlay), after every one band 1 (overlay), and between paths it is band 0 with `interleaved-draw-order` reported.",
        source: "ADR 0018 band (M16)",
    },
    ProducerRule {
        id: "dwg-image-is-a-frame",
        rule: "A DWG `IMAGE` is reported with its frame and no placement is written, because no reader here opens a path a drawing names.",
        source: "ADR 0018 Producers",
    },
    ProducerRule {
        id: "nothing-dropped-silently",
        rule: "The image constructs of a page equal the rasters written plus the refusals, and each refusal is a report line with its reason.",
        source: "ADR 0018 Producers; EPIC R-L",
    },
];

/// The rules as a Markdown table, which is the block `docs/pvec-raster.md` carries.
#[must_use]
pub fn render_rules_table() -> String {
    let mut out = String::from("| rule | what the producer does | comes from |\n|---|---|---|\n");
    for r in RASTER_PRODUCER_RULES {
        out.push_str(&format!("| `{}` | {} | {} |\n", r.id, r.rule, r.source));
    }
    out
}
