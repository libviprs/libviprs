//! The JSON metadata section, and the libviprs namespace inside it.
//!
//! Every PMTiles archive carries a JSON **object** at
//! `header.metadata_offset`, compressed with the header's internal
//! compression. The spec makes the object mandatory and its contents almost
//! entirely optional: the minimum conformant value is `{}`, and the reference
//! archives in the spec repository really do store exactly that.
//!
//! Six keys are defined (`name`, `description`, `attribution`, `type`,
//! `version`, `encoding`), all optional, and one is conditionally required:
//! an archive whose tile type is MVT `MUST` carry `vector_layers`. Beyond
//! that, an implementation may add whatever it likes, and there is no
//! namespace convention, no reserved prefix and no forbidden key.
//!
//! # Unknown keys survive
//!
//! This is the reason [`Metadata`] keeps an `extra` map rather than deriving a
//! plain struct and letting serde drop what it does not recognise. The format
//! has no other extension point, so the metadata object is where every tool
//! puts what it cares about, and a read-modify-write cycle that silently drops
//! another tool's keys destroys their data. Anything this struct does not name
//! goes into `extra` and comes back out again in the same shape.
//!
//! # The `vnd.libviprs` object
//!
//! libviprs writes one extra key, `vnd.libviprs`, holding what it would take
//! to reproduce the pyramid: the source dimensions and pixel format, the tile
//! size and overlap, the layout, the tile encoding, the coordinate convention,
//! the version of libviprs that wrote it, and the engine knobs it ran with.
//! The `vnd.` prefix is the usual convention for a vendor extension and keeps
//! the key visibly ours, so a reader that does not know libviprs can skip it
//! without wondering whether it was supposed to mean something.
//!
//! # The georeference (`geo_transform`)
//!
//! When the run is given the source raster's pixel to geographic transform
//! ([`PmTilesSinkBuilder::geo_transform`](crate::sink_pmtiles::PmTilesSinkBuilder::geo_transform)),
//! the object carries it under `geo_transform`:
//!
//! ```text
//! "geo_transform": { "coefficients": [a, b, c, d, e, f], "pixel_origin": "corner" }
//!   geo_x = a*px + b*py + c        geo_y = d*px + e*py + f
//! ```
//!
//! Six JSON numbers in [`GeoTransform`] order, read back bit for bit. `pixel_origin` is `corner`
//! (GDAL's `PixelIsArea`, `(0, 0)` is the top-left corner of the first pixel) or `centre` (a
//! world file or `PixelIsPoint`, half a pixel in); an origin this build does not know is a parse
//! error, never a guess. No CRS is stored with it, and the header bounds still default to the
//! whole world: the transform describes the **source raster**, and a consumer that needs the
//! archive's own placement composes it with the drawing's calibration
//! ([`crate::pvec::raster_placement`]). An archive written without one has no key and reads as
//! before ([`Reader::geo_transform`](crate::pmtiles::Reader::geo_transform) answers `None`).
//!
//! Two of the other fields are types this crate already has:
//! [`SourceMetadata`] and
//! [`GenerationSettings`], the same ones
//! `manifest.json` uses. Reusing them is deliberate: the manifest and this
//! object answer the same question, "what produced this pyramid", and two
//! schemas that answer one question drift apart. The cost is that a change to
//! either type changes this wire format too, which is what
//! [`LIBVIPRS_META_VERSION`] is for, and what the pinned-JSON test in this
//! module turns into a failing build rather than a surprise.

use serde::{Deserialize, Serialize};

use crate::geo::{GeoTransform, PixelOrigin};
use crate::manifest::{GenerationSettings, SourceMetadata};
use crate::pmtiles::PmTilesError;

/// Schema version of the `vnd.libviprs` object.
///
/// Bumped when a field is removed or changes meaning. A field being *added* is
/// not a bump: unknown keys are ignored on the way in, so an older reader
/// still gets everything it knew about.
pub const LIBVIPRS_META_VERSION: u32 = 1;

/// The key the libviprs object lives under in the metadata JSON.
pub const LIBVIPRS_METADATA_KEY: &str = "vnd.libviprs";

/// The coordinate convention libviprs writes, and the only one PMTiles has.
///
/// Recorded explicitly rather than left implicit because the one thing a
/// consumer most needs to know about a tile archive is whether row 0 is at the
/// top (slippy / ZXY) or the bottom (TMS), and a reader that assumes wrong
/// gets a vertically mirrored map that still renders.
pub const COORDINATE_CONVENTION: &str = "zxy";

/// The decoded metadata object.
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::Metadata;
///
/// // The minimum conformant metadata is an empty object, and it parses.
/// let meta = Metadata::try_from_json(b"{}").unwrap();
/// assert!(meta.name.is_none());
///
/// // A key this struct does not name survives the round trip.
/// let meta = Metadata::try_from_json(br#"{"tilejson":"3.0.0"}"#).unwrap();
/// let out = String::from_utf8(meta.to_json().unwrap()).unwrap();
/// assert!(out.contains("tilejson"));
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Metadata {
    /// A name describing the tileset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// A text description of the tileset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// An attribution to show a user. A consumer may treat it as HTML or as
    /// literal text, so a writer should not assume either.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<String>,
    /// `overlay` or `baselayer`. Kept as a string rather than an enum: the
    /// spec names two values and a third would otherwise turn somebody else's
    /// archive into a parse error over a field nothing reads.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub tileset_type: Option<String>,
    /// A semver string for the tileset. A string, not a number, and it is the
    /// tileset's version rather than the format's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Extra encoding applied to the tile data. The spec defines `terrarium`,
    /// which means the raster is a terrain model with elevation in metres
    /// equal to `(red * 256 + green + blue / 256) - 32768`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    /// The libviprs namespace.
    #[serde(
        rename = "vnd.libviprs",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub vnd_libviprs: Option<LibviprsMetadata>,
    /// Every other key, preserved exactly. `vector_layers` for a vector
    /// archive arrives here, as does anything a foreign tool wrote.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// What is wrong with the `vector_layers` entry of an MVT archive's metadata,
/// or `None` when it is fine.
///
/// The spec makes `vector_layers` a MUST for a vector archive, and it is the
/// first thing MapLibre tooling and `pmtiles show` read to learn what layers
/// exist. It lands in [`Metadata::extra`] because the schema has no typed
/// field for it, which is also why nothing in the tree used to enforce it.
///
/// A valid entry is a non-empty array of objects, each with a unique non-empty
/// string `id`, a `fields` object whose values are strings, and, when present,
/// integer `minzoom` and `maxzoom` in `0..=30` with `minzoom <= maxzoom`.
#[must_use]
pub fn vector_layers_problem(extra: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    use serde_json::Value;
    let Some(layers) = extra.get("vector_layers") else {
        return Some("the metadata has no vector_layers".into());
    };
    let Some(layers) = layers.as_array() else {
        return Some("vector_layers is not an array".into());
    };
    if layers.is_empty() {
        return Some("vector_layers is empty".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    for (i, layer) in layers.iter().enumerate() {
        let Some(layer) = layer.as_object() else {
            return Some(format!("vector_layers[{i}] is not an object"));
        };
        let Some(id) = layer
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        else {
            return Some(format!("vector_layers[{i}] has no string id"));
        };
        if !seen.insert(id) {
            return Some(format!("vector_layers repeats the id {id:?}"));
        }
        match layer.get("fields") {
            Some(Value::Object(fields)) => {
                if let Some((k, _)) = fields.iter().find(|(_, v)| !v.is_string()) {
                    return Some(format!("vector_layers[{i}].fields.{k} is not a string"));
                }
            }
            _ => return Some(format!("vector_layers[{i}] has no fields object")),
        }
        let zoom =
            |key: &str| -> Result<Option<u64>, String> {
                match layer.get(key) {
                    None => Ok(None),
                    Some(v) => v.as_u64().filter(|z| *z <= 30).map(Some).ok_or_else(|| {
                        format!("vector_layers[{i}].{key} is not an integer in 0..=30")
                    }),
                }
            };
        match (zoom("minzoom"), zoom("maxzoom")) {
            (Err(e), _) | (_, Err(e)) => return Some(e),
            (Ok(Some(lo)), Ok(Some(hi))) if lo > hi => {
                return Some(format!(
                    "vector_layers[{i}] has minzoom {lo} above maxzoom {hi}"
                ));
            }
            _ => {}
        }
    }
    None
}

impl Metadata {
    /// Parse metadata from the decompressed bytes of the metadata section.
    ///
    /// Refuses anything that is not a JSON object, which the spec makes a
    /// `MUST`: an array or a bare scalar there is a malformed archive, not a
    /// variation to tolerate.
    pub fn try_from_json(bytes: &[u8]) -> Result<Self, PmTilesError> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Serialise back to JSON bytes, ready to be compressed and stored.
    pub fn to_json(&self) -> Result<Vec<u8>, PmTilesError> {
        Ok(serde_json::to_vec(self)?)
    }
}

/// The pixel to geographic transform of the source raster, as the archive stores it
/// (issue #1328, option D' of ADR 0018 in libviprs/pvec).
///
/// `coefficients` are `[a, b, c, d, e, f]` in the order [`GeoTransform`] documents:
/// `geo_x = a*px + b*py + c`, `geo_y = d*px + e*py + f`. They are JSON numbers and a reader
/// gets the same bits back (a non-finite coefficient cannot be written, the sink refuses it).
/// `pixel_origin` names where `(0, 0)` sits, so a consumer never has to guess a half pixel. The
/// transform describes the **source raster** at level `max_zoom`; no CRS is stored with it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GeoTransformMetadata {
    /// `[a, b, c, d, e, f]`.
    pub coefficients: [f64; 6],
    /// Where the pixel origin is.
    pub pixel_origin: PixelOrigin,
}

impl GeoTransformMetadata {
    /// The record for `transform` under `pixel_origin`.
    #[must_use]
    pub fn new(transform: GeoTransform, pixel_origin: PixelOrigin) -> Self {
        let GeoTransform { a, b, c, d, e, f } = transform;
        Self {
            coefficients: [a, b, c, d, e, f],
            pixel_origin,
        }
    }

    /// The stored coefficients as a [`GeoTransform`], bit for bit.
    #[must_use]
    pub fn transform(&self) -> GeoTransform {
        let [a, b, c, d, e, f] = self.coefficients;
        GeoTransform::new(a, b, c, d, e, f)
    }
}

/// What libviprs records about the pyramid it produced.
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::{LibviprsMetadata, Metadata};
///
/// // `Metadata` is `#[non_exhaustive]`, so a caller outside the crate starts
/// // from `default()` and fills in what it wants rather than writing a
/// // struct literal.
/// let mut meta = Metadata::default();
/// meta.name = Some("drawing".to_string());
/// meta.vnd_libviprs = Some(LibviprsMetadata::default());
///
/// let json = String::from_utf8(meta.to_json().unwrap()).unwrap();
/// assert!(json.contains("\"vnd.libviprs\""));
/// assert!(json.contains("\"coordinate_convention\":\"zxy\""));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LibviprsMetadata {
    /// Schema version of this object. See [`LIBVIPRS_META_VERSION`].
    pub libviprs_meta_version: u32,
    /// The version of libviprs that wrote the archive.
    pub libviprs_version: String,
    /// `zxy`. See [`COORDINATE_CONVENTION`] for why it is written down.
    pub coordinate_convention: String,
    /// The source raster's dimensions and pixel format. `None` when the
    /// archive was assembled from tiles rather than generated from a source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceMetadata>,
    /// The engine knobs the run used: tile size, overlap, layout, tile
    /// encoding, concurrency, background colour and blank-tile strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationSettings>,
    /// The source raster's pixel to geographic transform, when the run was given one. See
    /// [`GeoTransformMetadata`]. Absent in an archive written without one, which reads as
    /// before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geo_transform: Option<GeoTransformMetadata>,
    /// Anything a later libviprs wrote that this build does not know. Kept for
    /// the same reason [`Metadata::extra`] is.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for LibviprsMetadata {
    /// The namespace stamped with this build's version and schema version, and
    /// nothing else filled in.
    ///
    /// Hand-written rather than derived because two of the three required
    /// fields have a right answer that is not the zero value: the schema
    /// version is [`LIBVIPRS_META_VERSION`] and the convention is `zxy`, and a
    /// derived `Default` would write `0` and `""` into an archive.
    fn default() -> Self {
        Self {
            libviprs_meta_version: LIBVIPRS_META_VERSION,
            libviprs_version: env!("CARGO_PKG_VERSION").to_string(),
            coordinate_convention: COORDINATE_CONVENTION.to_string(),
            source: None,
            generation: None,
            geo_transform: None,
            extra: serde_json::Map::new(),
        }
    }
}

impl LibviprsMetadata {
    /// The namespace for a pyramid this build generated from a source raster.
    pub fn new(source: SourceMetadata, generation: GenerationSettings) -> Self {
        Self {
            source: Some(source),
            generation: Some(generation),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::BlankTileStrategy;
    use crate::manifest::{GenerationSettings, SourceMetadata};
    use crate::pixel::PixelFormat;
    use crate::planner::Layout;
    use crate::sink::TileFormat;

    /// The metadata of `raster-z0z2.pmtiles`, a golden written by go-pmtiles
    /// v1.31.2, exactly as it is stored (gzip removed).
    ///
    /// It is here because it is a **foreign** metadata object: every key in it
    /// is one this struct does not name, so it is the case that proves the
    /// `extra` map is doing its job rather than serde quietly dropping what it
    /// does not recognise.
    const ORACLE_METADATA: &str = concat!(
        r#"{"description":"libviprs EPIC F oracle golden, distinct 8x8 PNG per tile","#,
        r#""format":"png","maxzoom":"2","minzoom":"0","#,
        r#""name":"libviprs-oracle-raster","type":"baselayer","version":"1"}"#,
    );

    /// Every field filled with a distinct, non-default value.
    ///
    /// A sparse fixture is how a whole-struct test quietly stops covering each
    /// new field: a zero in the fixture and a zero in the output agree whether
    /// or not the field was carried. Nothing here is zero and nothing repeats.
    fn saturated() -> Metadata {
        Metadata {
            name: Some("drawing".to_string()),
            description: Some("a floor plan".to_string()),
            attribution: Some("Acme Surveying".to_string()),
            tileset_type: Some("overlay".to_string()),
            version: Some("2.3.4".to_string()),
            encoding: Some("terrarium".to_string()),
            vnd_libviprs: Some(LibviprsMetadata::new(
                SourceMetadata {
                    width: 4001,
                    height: 2999,
                    pixel_format: PixelFormat::Rgb8,
                    bytes_hash: Some("abc123".to_string()),
                },
                GenerationSettings {
                    tile_size: 512,
                    overlap: 1,
                    layout: Layout::Xyz,
                    format: TileFormat::Jpeg { quality: 83 },
                    concurrency: 7,
                    background_rgb: [1, 2, 3],
                    blank_strategy: BlankTileStrategy::PlaceholderWithTolerance {
                        max_channel_delta: 4,
                    },
                    centre: true,
                    skip_blanks: true,
                },
            )),
            ..Metadata::default()
        }
    }

    #[test]
    fn the_wire_shape_is_pinned_field_by_field() {
        // This is the drift guard. The libviprs namespace reuses
        // `SourceMetadata` and `GenerationSettings` from the manifest, so a
        // change to either of those types changes this archive format too, and
        // this test is what turns that into a failing build rather than a
        // surprise in somebody's reader. If it goes red for a deliberate
        // change, bump `LIBVIPRS_META_VERSION` with it.
        let json = String::from_utf8(saturated().to_json().unwrap()).unwrap();
        let want = concat!(
            r#"{"name":"drawing","description":"a floor plan","#,
            r#""attribution":"Acme Surveying","type":"overlay","version":"2.3.4","#,
            r#""encoding":"terrarium","vnd.libviprs":{"libviprs_meta_version":1,"#,
            r#""libviprs_version":""#,
            env!("CARGO_PKG_VERSION"),
            r#"","coordinate_convention":"zxy","#,
            r#""source":{"width":4001,"height":2999,"pixel_format":"rgb8","bytes_hash":"abc123"},"#,
            r#""generation":{"tile_size":512,"overlap":1,"layout":"xyz","#,
            r#""format":{"kind":"jpeg","quality":83},"concurrency":7,"#,
            r#""background_rgb":[1,2,3],"#,
            r#""blank_strategy":{"kind":"placeholder_with_tolerance","tolerance":4},"#,
            r#""centre":true,"skip_blanks":true}}}"#,
        );
        assert_eq!(json, want);

        // And it parses back to what it came from.
        assert_eq!(
            Metadata::try_from_json(json.as_bytes()).unwrap(),
            saturated()
        );
    }

    #[test]
    fn a_foreign_archive_s_keys_survive_a_read_and_a_write() {
        // The format has no other extension point, so the metadata object is
        // where every tool puts what it cares about. A read-modify-write that
        // dropped what it did not recognise would destroy their data.
        let meta =
            Metadata::try_from_json(ORACLE_METADATA.as_bytes()).expect("a real archive's metadata");
        assert_eq!(meta.name.as_deref(), Some("libviprs-oracle-raster"));
        assert_eq!(meta.tileset_type.as_deref(), Some("baselayer"));
        assert_eq!(meta.version.as_deref(), Some("1"));
        assert!(
            meta.vnd_libviprs.is_none(),
            "go-pmtiles wrote no libviprs namespace"
        );

        // `format`, `minzoom` and `maxzoom` are not keys the v3 spec defines
        // and not keys this struct names, so they can only have survived
        // through `extra`.
        assert_eq!(
            meta.extra.len(),
            3,
            "extra holds {:?}",
            meta.extra.keys().collect::<Vec<_>>()
        );
        assert_eq!(meta.extra["format"], serde_json::json!("png"));
        assert_eq!(meta.extra["minzoom"], serde_json::json!("0"));
        assert_eq!(meta.extra["maxzoom"], serde_json::json!("2"));

        let round_tripped = Metadata::try_from_json(&meta.to_json().unwrap()).unwrap();
        assert_eq!(round_tripped, meta);
    }

    #[test]
    fn the_minimum_conformant_metadata_is_an_empty_object() {
        // The spec's own reference archives really do store exactly this, so a
        // parser that required any key would refuse them.
        let meta = Metadata::try_from_json(b"{}").unwrap();
        assert_eq!(meta, Metadata::default());
        assert_eq!(meta.to_json().unwrap(), b"{}".to_vec());
    }

    #[test]
    fn the_namespaced_key_keeps_its_dotted_spelling() {
        // The struct field is `vnd_libviprs` and the key is `vnd.libviprs`.
        // Serde would happily write the field name, which is a key nothing
        // else in the world looks for.
        let json = String::from_utf8(saturated().to_json().unwrap()).unwrap();
        assert!(json.contains(r#""vnd.libviprs":"#), "got {json}");
        assert!(
            !json.contains("vnd_libviprs"),
            "the field name leaked: {json}"
        );
        assert_eq!(LIBVIPRS_METADATA_KEY, "vnd.libviprs");

        // And a document using the dotted key parses into the typed field
        // rather than falling into `extra`.
        let parsed = Metadata::try_from_json(json.as_bytes()).unwrap();
        assert!(parsed.vnd_libviprs.is_some());
        assert!(!parsed.extra.contains_key("vnd.libviprs"));
    }

    #[test]
    fn the_namespace_stamps_the_version_and_the_convention_without_being_asked() {
        let vnd = LibviprsMetadata::default();
        assert_eq!(vnd.libviprs_meta_version, LIBVIPRS_META_VERSION);
        assert_eq!(vnd.coordinate_convention, "zxy");
        assert_eq!(vnd.libviprs_version, env!("CARGO_PKG_VERSION"));
        // A derived `Default` would have written 0 and "" into an archive,
        // which is why this one is hand-written.
        assert_ne!(vnd.libviprs_meta_version, 0);
        assert!(!vnd.coordinate_convention.is_empty());
        assert!(!vnd.libviprs_version.is_empty());
    }

    #[test]
    fn something_that_is_not_a_json_object_is_refused() {
        // The spec makes the object a MUST, so an array or a scalar there is a
        // malformed archive rather than a variation to tolerate.
        let bad_inputs: [&[u8]; 6] = [b"[]", b"3", b"\"text\"", b"null", b"not json at all", b""];
        for bad in bad_inputs {
            assert!(
                matches!(Metadata::try_from_json(bad), Err(PmTilesError::Metadata(_))),
                "{:?} must be refused",
                String::from_utf8_lossy(bad)
            );
        }
        // The positive control: the smallest thing that is an object parses.
        assert!(Metadata::try_from_json(b"{}").is_ok());
    }
    /// The same fixture with a WebP encoding, pinned separately (issue #1123).
    ///
    /// This is an addition rather than an edit, and the reason is the trap
    /// #1119 walked into: `saturated()` above encodes `Jpeg { quality: 83 }`,
    /// and adding a variant to `TileFormat` does not change how `Jpeg`
    /// serialises. So the pin everybody believes is watching the wire format
    /// stayed green through the whole of this change and would have stayed
    /// green if `Webp` had serialised as `{"kind":"Webp"}`, or as
    /// `{"kind":"webp","quality":0}`, or had not serialised at all.
    ///
    /// Only the `format` key differs from the pin above, so the two together
    /// say the variant changed that one field and nothing else.
    fn saturated_webp() -> Metadata {
        let mut meta = saturated();
        let vnd = meta
            .vnd_libviprs
            .as_mut()
            .expect("the saturated fixture carries the namespace");
        let generation = vnd
            .generation
            .as_mut()
            .expect("the saturated fixture carries generation settings");
        generation.format = TileFormat::Webp;
        meta
    }

    #[test]
    fn the_webp_wire_shape_is_pinned_too() {
        let json = String::from_utf8(saturated_webp().to_json().unwrap()).unwrap();
        assert!(
            json.contains(r#""format":{"kind":"webp"}"#),
            "webp must be a bare tag with no quality beside it, got: {json}"
        );
        // Said negatively as well, because `contains` on the line above would
        // also pass for `{"kind":"webp","quality":0}` if serde ever emitted
        // one: the substring is a prefix of it.
        assert!(
            !json.contains(r#""kind":"webp","quality""#),
            "TileFormat::Webp carries no quality field; the encoder is \
             lossless and has no knob for one to reach"
        );

        assert_eq!(
            Metadata::try_from_json(json.as_bytes()).unwrap(),
            saturated_webp(),
            "and it parses back to what it came from"
        );

        // The two pins differ in exactly one key. Anything else moving means
        // the variant changed more than the encoding.
        let jpeg = String::from_utf8(saturated().to_json().unwrap()).unwrap();
        assert_eq!(
            jpeg.replace(r#""format":{"kind":"jpeg","quality":83}"#, "FORMAT"),
            json.replace(r#""format":{"kind":"webp"}"#, "FORMAT"),
        );
    }
}
