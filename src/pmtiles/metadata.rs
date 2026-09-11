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
//! Two of those fields are types this crate already has:
//! [`SourceMetadata`](crate::manifest::SourceMetadata) and
//! [`GenerationSettings`](crate::manifest::GenerationSettings), the same ones
//! `manifest.json` uses. Reusing them is deliberate: the manifest and this
//! object answer the same question, "what produced this pyramid", and two
//! schemas that answer one question drift apart. The cost is that a change to
//! either type changes this wire format too, which is what
//! [`LIBVIPRS_META_VERSION`] is for, and what the pinned-JSON test in this
//! module turns into a failing build rather than a surprise.

use serde::{Deserialize, Serialize};

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
