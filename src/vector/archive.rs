//! The vector PMTiles writer and the Range-readable [`VectorArchive`].

use std::collections::BTreeMap;
use std::io::{Seek, Write};
use std::path::Path;

use serde_json::{Map, Value as Json, json};
use thiserror::Error;

use crate::pmtiles::metadata::vector_layers_problem;
use crate::pmtiles::range::{FileRangeReader, RangeReader};
use crate::pmtiles::reader::Reader;
use crate::pmtiles::writer::{Finish, Writer, WriterOptions, content_hash};
use crate::pmtiles::{Compression, Metadata, PmTilesError, TileType};
use crate::vector::mvt::{self, DecodeLimits, MvtError, Tile};

/// How a property's values are typed in `vector_layers`, in the vocabulary
/// TileJSON uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// `"String"`.
    String,
    /// `"Number"`: any integer or float.
    Number,
    /// `"Boolean"`.
    Boolean,
}

impl FieldType {
    /// The TileJSON spelling: `"String"`, `"Number"` or `"Boolean"`.
    ///
    /// This is what `vector_layers` carries and what the JSON for an archive
    /// description prints. It is written out here and does not follow the
    /// variant's name, so renaming a variant cannot change a file format.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::String => "String",
            Self::Number => "Number",
            Self::Boolean => "Boolean",
        }
    }

    fn of(value: &mvt::Value) -> Self {
        match value {
            mvt::Value::String(_) => Self::String,
            mvt::Value::Bool(_) => Self::Boolean,
            _ => Self::Number,
        }
    }
}

/// One entry of the archive's `vector_layers`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSchema {
    /// The layer id, which is the MVT layer name.
    pub id: String,
    /// A sentence for a human.
    pub description: Option<String>,
    /// The properties features in this layer may carry.
    pub fields: BTreeMap<String, FieldType>,
    /// The lowest zoom the layer appears at.
    pub min_zoom: Option<u8>,
    /// The highest zoom the layer appears at.
    pub max_zoom: Option<u8>,
}

impl LayerSchema {
    /// A layer with no declared fields.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: None,
            fields: BTreeMap::new(),
            min_zoom: None,
            max_zoom: None,
        }
    }

    /// Declares a property.
    #[must_use]
    pub fn with_field(mut self, name: impl Into<String>, ty: FieldType) -> Self {
        self.fields.insert(name.into(), ty);
        self
    }

    fn to_json(&self) -> Json {
        let fields: Map<String, Json> = self
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), json!(v.name())))
            .collect();
        let mut out = Map::new();
        out.insert("id".into(), json!(self.id));
        if let Some(d) = &self.description {
            out.insert("description".into(), json!(d));
        }
        out.insert("fields".into(), Json::Object(fields));
        if let Some(z) = self.min_zoom {
            out.insert("minzoom".into(), json!(z));
        }
        if let Some(z) = self.max_zoom {
            out.insert("maxzoom".into(), json!(z));
        }
        Json::Object(out)
    }

    fn from_json(value: &Json) -> Option<Self> {
        let obj = value.as_object()?;
        let fields = obj
            .get("fields")?
            .as_object()?
            .iter()
            .map(|(k, v)| {
                let ty = match v.as_str()? {
                    "Number" => FieldType::Number,
                    "Boolean" => FieldType::Boolean,
                    // TileJSON lets a producer put any description here. A
                    // value that is not one of the two non-string names is
                    // read as a string field.
                    _ => FieldType::String,
                };
                Some((k.clone(), ty))
            })
            .collect::<Option<BTreeMap<_, _>>>()?;
        let zoom = |key| {
            obj.get(key)
                .and_then(Json::as_u64)
                .and_then(|z| u8::try_from(z).ok())
        };
        Some(Self {
            id: obj.get("id")?.as_str()?.to_owned(),
            description: obj
                .get("description")
                .and_then(Json::as_str)
                .map(str::to_owned),
            fields,
            min_zoom: zoom("minzoom"),
            max_zoom: zoom("maxzoom"),
        })
    }
}

/// Why a vector archive could not be written or read.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VectorError {
    /// The archive container refused something.
    #[error(transparent)]
    PmTiles(#[from] PmTilesError),
    /// The tile is not valid MVT.
    #[error(transparent)]
    Mvt(#[from] MvtError),
    /// No layers were declared, or a declared layer is not valid.
    #[error("the declared layers are not a valid vector_layers: {0}")]
    Schema(String),
    /// A tile layer that the archive did not declare.
    #[error("layer {layer:?} is not declared in vector_layers")]
    UndeclaredLayer {
        /// The layer.
        layer: String,
    },
    /// A property the layer did not declare, or declared with another type.
    #[error("layer {layer:?} property {field:?} is not declared as {found:?}")]
    UndeclaredField {
        /// The layer.
        layer: String,
        /// The property.
        field: String,
        /// The type the value has.
        found: FieldType,
    },
    /// The archive is not a vector archive this reader can use.
    #[error("not a readable vector archive: {0}")]
    NotVector(String),
}

/// Everything about a vector archive that is not a tile.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorArchiveOptions {
    layers: Vec<LayerSchema>,
    name: Option<String>,
    description: Option<String>,
    attribution: Option<String>,
    bounds_degrees: Option<[f64; 4]>,
    center_degrees: Option<(f64, f64)>,
    center_zoom: Option<u8>,
    extra: Map<String, Json>,
}

impl VectorArchiveOptions {
    /// Options for an archive whose tiles use exactly `layers`.
    #[must_use]
    pub fn new(layers: Vec<LayerSchema>) -> Self {
        Self {
            layers,
            name: None,
            description: None,
            attribution: None,
            bounds_degrees: None,
            center_degrees: None,
            center_zoom: None,
            extra: Map::new(),
        }
    }

    /// Sets the archive name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the archive description.
    #[must_use]
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }

    /// Sets the attribution string.
    #[must_use]
    pub fn with_attribution(mut self, a: impl Into<String>) -> Self {
        self.attribution = Some(a.into());
        self
    }

    /// West, south, east, north in degrees.
    #[must_use]
    pub fn with_bounds_degrees(mut self, bounds: [f64; 4]) -> Self {
        self.bounds_degrees = Some(bounds);
        self
    }

    /// The centre a viewer opens at.
    #[must_use]
    pub fn with_center_degrees(mut self, lon: f64, lat: f64) -> Self {
        self.center_degrees = Some((lon, lat));
        self
    }

    /// The zoom a viewer opens at.
    #[must_use]
    pub fn with_center_zoom(mut self, zoom: u8) -> Self {
        self.center_zoom = Some(zoom);
        self
    }

    /// Adds a top-level metadata entry beside `vector_layers`, for tooling
    /// that has to learn something about the archive the format has no field
    /// for. `vector_layers` itself cannot be overridden this way.
    #[must_use]
    pub fn with_metadata_entry(mut self, key: impl Into<String>, value: Json) -> Self {
        let key = key.into();
        if key != "vector_layers" {
            self.extra.insert(key, value);
        }
        self
    }

    fn schema_json(&self) -> Result<Vec<Json>, VectorError> {
        let layers: Vec<Json> = self.layers.iter().map(LayerSchema::to_json).collect();
        let mut extra = Map::new();
        extra.insert("vector_layers".into(), Json::Array(layers.clone()));
        match vector_layers_problem(&extra) {
            None => Ok(layers),
            Some(p) => Err(VectorError::Schema(p)),
        }
    }

    fn writer_options(&self) -> Result<WriterOptions, VectorError> {
        let layers = self.schema_json()?;
        let mut metadata = Metadata::default();
        metadata.name.clone_from(&self.name);
        metadata.description.clone_from(&self.description);
        metadata.attribution.clone_from(&self.attribution);
        metadata.extra.clone_from(&self.extra);
        metadata
            .extra
            .insert("vector_layers".into(), Json::Array(layers));
        let mut options = WriterOptions::default()
            .with_tile_type(TileType::Mvt)
            // The writer stores bytes as given and only labels them, so this
            // label is a promise that `add_tile` below keeps.
            .with_tile_compression(Compression::Gzip)
            .with_internal_compression(Compression::Gzip)
            .with_metadata(metadata);
        if let Some(b) = self.bounds_degrees {
            options = options.with_bounds_degrees(b);
        }
        if let Some((lon, lat)) = self.center_degrees {
            options = options.with_center_degrees(lon, lat);
        }
        if let Some(z) = self.center_zoom {
            options = options.with_center_zoom(Some(z));
        }
        Ok(options)
    }
}

pub(crate) fn gzip(bytes: &[u8]) -> Result<Vec<u8>, PmTilesError> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(bytes)?;
    Ok(enc.finish()?)
}

/// Writes MVT tiles into a PMTiles v3 archive.
///
/// Each tile is serialised, gzipped, and stored; the archive is labelled MVT
/// with gzip tiles and gzip directories, and carries `vector_layers` built
/// from the declared schema. A tile that uses a layer or a property the schema
/// did not declare is refused, so the metadata cannot drift from the tiles it
/// describes.
pub struct VectorArchiveWriter<W: Write + Seek> {
    inner: Writer<W>,
    schema: BTreeMap<String, LayerSchema>,
}

impl VectorArchiveWriter<std::fs::File> {
    /// Creates an archive at `path`, staged beside it and renamed into place
    /// by [`finish`](Self::finish).
    ///
    /// # Errors
    ///
    /// [`VectorError::Schema`] for an invalid layer declaration, or the
    /// container's own error.
    pub fn create(
        path: impl AsRef<Path>,
        options: &VectorArchiveOptions,
    ) -> Result<Self, VectorError> {
        let schema = schema_map(options)?;
        let inner = Writer::create(path, options.writer_options()?)?;
        Ok(Self { inner, schema })
    }
}

fn schema_map(
    options: &VectorArchiveOptions,
) -> Result<BTreeMap<String, LayerSchema>, VectorError> {
    options.schema_json()?;
    Ok(options
        .layers
        .iter()
        .map(|l| (l.id.clone(), l.clone()))
        .collect())
}

impl<W: Write + Seek> VectorArchiveWriter<W> {
    /// Writes into a sink the caller owns, staging under `scratch_dir`.
    ///
    /// # Errors
    ///
    /// As [`create`](VectorArchiveWriter::create).
    pub fn try_new(
        out: W,
        scratch_dir: impl AsRef<Path>,
        options: &VectorArchiveOptions,
    ) -> Result<Self, VectorError> {
        let schema = schema_map(options)?;
        let inner = Writer::try_new(out, scratch_dir, options.writer_options()?)?;
        Ok(Self { inner, schema })
    }

    fn check_against_schema(&self, tile: &Tile) -> Result<(), VectorError> {
        for layer in &tile.layers {
            let Some(declared) = self.schema.get(&layer.name) else {
                return Err(VectorError::UndeclaredLayer {
                    layer: layer.name.clone(),
                });
            };
            for feature in &layer.features {
                for (key, value) in &feature.properties {
                    let ty = FieldType::of(value);
                    if declared.fields.get(key) != Some(&ty) {
                        return Err(VectorError::UndeclaredField {
                            layer: layer.name.clone(),
                            field: key.clone(),
                            found: ty,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Adds one tile. Returns `false`, storing nothing, for a tile with no
    /// features: an empty tile is valid MVT and a client treats a missing tile
    /// the same way.
    ///
    /// # Errors
    ///
    /// [`VectorError::UndeclaredLayer`] or [`VectorError::UndeclaredField`] for
    /// a tile outside the schema, [`VectorError::Mvt`] for a tile the format
    /// cannot hold, and the container's own errors.
    pub fn add_tile(&mut self, z: u8, x: u32, y: u32, tile: &Tile) -> Result<bool, VectorError> {
        Ok(self.add_tile_sized(z, x, y, tile)?.is_some())
    }

    /// As [`add_tile`](Self::add_tile), returning the number of bytes the tile
    /// takes in the archive (gzipped), or `None` for an empty tile that was not
    /// stored. The tiler uses it to hold tiles to a size budget.
    ///
    /// # Errors
    ///
    /// As [`add_tile`](Self::add_tile).
    pub fn add_tile_sized(
        &mut self,
        z: u8,
        x: u32,
        y: u32,
        tile: &Tile,
    ) -> Result<Option<usize>, VectorError> {
        if tile.is_empty() {
            return Ok(None);
        }
        self.check_against_schema(tile)?;
        let stored = gzip(&tile.encode()?)?;
        let hash = content_hash(&stored);
        self.inner.add_tile(z, x, y, &stored, hash)?;
        Ok(Some(stored.len()))
    }

    /// Adds a tile that is already serialised MVT, after parsing it to make
    /// sure it is one and that it stays inside the schema.
    ///
    /// # Errors
    ///
    /// As [`add_tile`](Self::add_tile), and [`VectorError::Mvt`] when the bytes
    /// are not a valid tile.
    pub fn add_encoded_tile(
        &mut self,
        z: u8,
        x: u32,
        y: u32,
        mvt_bytes: &[u8],
    ) -> Result<bool, VectorError> {
        let tile = Tile::decode(mvt_bytes)?;
        if tile.is_empty() {
            return Ok(false);
        }
        self.check_against_schema(&tile)?;
        let stored = gzip(mvt_bytes)?;
        let hash = content_hash(&stored);
        self.inner.add_tile(z, x, y, &stored, hash)?;
        Ok(true)
    }

    /// How many tiles have been added.
    #[must_use]
    pub fn tile_count(&self) -> u64 {
        self.inner.tile_count()
    }

    /// Writes the directories and metadata and publishes the archive.
    ///
    /// # Errors
    ///
    /// The container's own errors.
    pub fn finish(self) -> Result<Finish, VectorError> {
        Ok(self.inner.finish()?)
    }
}

/// A vector PMTiles archive opened over any byte source.
///
/// Opening checks what makes it a vector archive: MVT tiles, gzip or no tile
/// compression, and a valid `vector_layers`. Reading a tile through
/// [`tile_bytes`](Self::tile_bytes) or [`tile`](Self::tile) inflates under a
/// ceiling, so a hostile archive cannot turn a small tile into a large buffer.
pub struct VectorArchive<R: RangeReader> {
    reader: Reader<R>,
    layers: Vec<LayerSchema>,
    limits: DecodeLimits,
}

impl VectorArchive<FileRangeReader> {
    /// Opens a local file.
    ///
    /// # Errors
    ///
    /// [`VectorError::NotVector`] when it is not a vector archive, or the
    /// container's own errors.
    pub fn try_open(path: impl AsRef<Path>) -> Result<Self, VectorError> {
        Self::try_new(FileRangeReader::try_open(path)?)
    }
}

impl<R: RangeReader> VectorArchive<R> {
    /// Opens an archive over a range reader: a file, an object store, or
    /// anything else that can answer a byte range.
    ///
    /// # Errors
    ///
    /// As [`try_open`](VectorArchive::try_open).
    pub fn try_new(source: R) -> Result<Self, VectorError> {
        let reader = Reader::try_new(source)?;
        let header = reader.header();
        if header.tile_type != TileType::Mvt {
            return Err(VectorError::NotVector(format!(
                "the tile type is {:?}, not MVT",
                header.tile_type
            )));
        }
        if !matches!(
            header.tile_compression,
            Compression::Gzip | Compression::None
        ) {
            return Err(VectorError::NotVector(format!(
                "tiles are compressed with {}, and only gzip is read",
                header.tile_compression
            )));
        }
        let metadata = reader.metadata()?;
        if let Some(problem) = vector_layers_problem(&metadata.extra) {
            return Err(VectorError::NotVector(problem));
        }
        let layers = metadata
            .extra
            .get("vector_layers")
            .and_then(Json::as_array)
            .map(|a| {
                a.iter()
                    .map(LayerSchema::from_json)
                    .collect::<Option<Vec<_>>>()
            })
            .and_then(|l| l)
            .ok_or_else(|| VectorError::NotVector("vector_layers is not readable".into()))?;
        Ok(Self {
            reader,
            layers,
            limits: DecodeLimits::default(),
        })
    }

    /// Replaces the ceilings tiles are inflated and parsed under.
    #[must_use]
    pub fn with_limits(mut self, limits: DecodeLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The layers the archive declares.
    #[must_use]
    pub fn layers(&self) -> &[LayerSchema] {
        &self.layers
    }

    /// The container underneath, for header and directory questions.
    #[must_use]
    pub fn pmtiles(&self) -> &Reader<R> {
        &self.reader
    }

    /// The tile exactly as stored: gzipped MVT. Two readers over the same
    /// archive return the same bytes however they fetched them.
    ///
    /// # Errors
    ///
    /// The container's own errors.
    pub fn stored_tile(&self, z: u8, x: u32, y: u32) -> Result<Option<Vec<u8>>, VectorError> {
        Ok(self.reader.get_tile(z, x, y)?)
    }

    /// The tile as raw MVT protobuf, inflated under the byte ceiling.
    ///
    /// # Errors
    ///
    /// The container's errors, including a decompression limit.
    pub fn tile_bytes(&self, z: u8, x: u32, y: u32) -> Result<Option<Vec<u8>>, VectorError> {
        let Some(stored) = self.stored_tile(z, x, y)? else {
            return Ok(None);
        };
        Ok(Some(
            self.reader
                .header()
                .tile_compression
                .decompress(&stored, self.limits.max_bytes)?,
        ))
    }

    /// The tile parsed.
    ///
    /// # Errors
    ///
    /// [`VectorError::Mvt`] when the stored bytes are not a valid tile.
    pub fn tile(&self, z: u8, x: u32, y: u32) -> Result<Option<Tile>, VectorError> {
        let Some(bytes) = self.tile_bytes(z, x, y)? else {
            return Ok(None);
        };
        Ok(Some(Tile::decode_with(&bytes, &self.limits)?))
    }
}
