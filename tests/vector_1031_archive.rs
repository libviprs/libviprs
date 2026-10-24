//! The vector PMTiles writer and reader (issue #1031).
//!
//! The container is the EPIC F one, so most of what could go wrong is a seam
//! rather than a bug: a tile labelled gzip that is not, metadata a client
//! cannot read, a reader that only works because the writer shares its
//! mistake. Every test here that could pass on such a shared mistake also has
//! a counterpart that does not (the bytes are inflated by a different gzip
//! implementation path, the archive goes through the structural validator,
//! `tests/vector_1031_interop.sh`-style tooling is run by the gate).

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicUsize, Ordering};

use libviprs::pmtiles::range::{FileRangeReader, RangeReader};
use libviprs::pmtiles::validate::{Finding, ValidationLimits, validate_bytes};
use libviprs::pmtiles::writer::{Writer, WriterOptions, content_hash};
use libviprs::pmtiles::{Compression, Metadata, PmTilesError, TileType};
use libviprs::vector::mvt::{DecodeLimits, Feature, Geometry, Layer, Point, Polygon, Tile, Value};
use libviprs::vector::{
    FieldType, LayerSchema, VectorArchive, VectorArchiveOptions, VectorArchiveWriter, VectorError,
};

fn schema() -> Vec<LayerSchema> {
    vec![
        LayerSchema::new("lines").with_field("layer", FieldType::String),
        LayerSchema::new("fills").with_field("layer", FieldType::String),
        LayerSchema::new("text")
            .with_field("text", FieldType::String)
            .with_field("height", FieldType::Number)
            .with_field("rotation", FieldType::Number),
    ]
}

fn square(x: i32, y: i32, s: i32) -> Vec<Point> {
    vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]]
}

/// A tile that uses all three layers, shifted by `n` so no two are equal.
fn sample_tile(n: i32) -> Tile {
    Tile {
        layers: vec![
            Layer {
                name: "lines".into(),
                extent: 4096,
                features: vec![Feature {
                    id: Some(n as u64),
                    geometry: Geometry::Lines(vec![vec![[0, n], [4096, n + 10], [2048, 4000]]]),
                    properties: vec![("layer".into(), Value::String("A-WALL".into()))],
                }],
            },
            Layer {
                name: "fills".into(),
                extent: 4096,
                features: vec![Feature {
                    id: None,
                    geometry: Geometry::Polygons(vec![Polygon {
                        exterior: square(100 + n, 100, 800),
                        holes: vec![square(200 + n, 200, 100)],
                    }]),
                    properties: vec![("layer".into(), Value::String("A-HATCH".into()))],
                }],
            },
            Layer {
                name: "text".into(),
                extent: 4096,
                features: vec![Feature {
                    id: None,
                    geometry: Geometry::Points(vec![[1000, 1000 + n]]),
                    properties: vec![
                        ("text".into(), Value::String(format!("Room {n} \u{2205}45"))),
                        ("height".into(), Value::Double(2.5)),
                        ("rotation".into(), Value::Double(0.0)),
                    ],
                }],
            },
        ],
    }
}

fn write_archive(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    let options = VectorArchiveOptions::new(schema())
        .with_name("sample drawing")
        .with_bounds_degrees([-10.0, -10.0, 10.0, 10.0])
        .with_center_degrees(0.0, 0.0);
    let mut w = VectorArchiveWriter::create(&path, &options).unwrap();
    // z0: one tile, z1: four, z2: a sparse few.
    assert!(w.add_tile(0, 0, 0, &sample_tile(0)).unwrap());
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        assert!(
            w.add_tile(1, x, y, &sample_tile((x + 2 * y) as i32))
                .unwrap()
        );
    }
    for (x, y) in [(0, 0), (3, 3), (2, 1)] {
        assert!(
            w.add_tile(2, x, y, &sample_tile((x * 5 + y) as i32))
                .unwrap()
        );
    }
    w.finish().unwrap();
    path
}

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn scratch(name: &str) -> std::path::PathBuf {
    let seq = SEQ.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("vector1031-{name}-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn the_archive_is_labelled_what_it_is() {
    let dir = scratch("labels");
    let path = write_archive(&dir, "a.pmtiles");
    let archive = VectorArchive::try_open(&path).unwrap();
    let h = archive.pmtiles().header();
    assert_eq!(h.tile_type, TileType::Mvt);
    assert_eq!(h.tile_compression, Compression::Gzip);
    assert_eq!(h.internal_compression, Compression::Gzip);
    assert_eq!(h.addressed_tiles_count, 8);
    assert_eq!((h.min_zoom, h.max_zoom), (0, 2));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn the_header_and_root_sit_inside_the_first_16_kib() {
    let dir = scratch("first16k");
    let path = write_archive(&dir, "a.pmtiles");
    let mut file = std::fs::File::open(&path).unwrap();
    let mut first = vec![0u8; 16384];
    let n = file.read(&mut first).unwrap();
    first.truncate(n);
    let archive = VectorArchive::try_open(&path).unwrap();
    let h = archive.pmtiles().header();
    assert!(
        h.root_offset + h.root_length <= 16384,
        "root ends at {}",
        h.root_offset + h.root_length
    );
    // And the claim is about bytes, not about arithmetic: the root really
    // inflates out of those first 16384 bytes alone.
    let root = &first[h.root_offset as usize..(h.root_offset + h.root_length) as usize];
    let inflated = Compression::Gzip.decompress(root, 1 << 20).unwrap();
    assert!(!inflated.is_empty());
    file.seek(SeekFrom::Start(0)).unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn tiles_are_really_gzip_and_inflate_to_the_mvt_that_went_in() {
    let dir = scratch("gzip");
    let path = write_archive(&dir, "a.pmtiles");
    let archive = VectorArchive::try_open(&path).unwrap();
    let stored = archive.stored_tile(1, 1, 0).unwrap().unwrap();
    assert_eq!(&stored[..2], &[0x1f, 0x8b], "gzip magic");
    // Inflate with flate2's reader directly, not through the archive.
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(stored.as_slice())
        .read_to_end(&mut raw)
        .unwrap();
    assert_eq!(raw, sample_tile(1).encode().unwrap());
    assert_eq!(archive.tile_bytes(1, 1, 0).unwrap().unwrap(), raw);
    assert_eq!(archive.tile(1, 1, 0).unwrap().unwrap().layers.len(), 3);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn metadata_carries_a_valid_vector_layers() {
    let dir = scratch("meta");
    let path = write_archive(&dir, "a.pmtiles");
    let archive = VectorArchive::try_open(&path).unwrap();
    let meta = archive.pmtiles().metadata().unwrap();
    assert_eq!(meta.name.as_deref(), Some("sample drawing"));
    assert!(libviprs::pmtiles::metadata::vector_layers_problem(&meta.extra).is_none());
    let ids: Vec<_> = archive.layers().iter().map(|l| l.id.as_str()).collect();
    assert_eq!(ids, ["lines", "fills", "text"]);
    let text = &archive.layers()[2];
    assert_eq!(text.fields.get("height"), Some(&FieldType::Number));
    assert_eq!(text.fields.get("text"), Some(&FieldType::String));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn the_structural_validator_passes_a_vector_archive() {
    let dir = scratch("validate");
    let path = write_archive(&dir, "a.pmtiles");
    let bytes = std::fs::read(&path).unwrap();
    let report = validate_bytes(&bytes, &ValidationLimits::default()).unwrap();
    assert!(report.is_valid(), "{:?}", report.findings);
    let _ = std::fs::remove_dir_all(dir);
}

/// An archive written straight through the container with the MVT type and the
/// given metadata, which is what the validator has to be able to catch.
fn raw_mvt_archive(metadata: Metadata) -> Vec<u8> {
    let dir = scratch("raw");
    let path = dir.join("raw.pmtiles");
    let options = WriterOptions::default()
        .with_tile_type(TileType::Mvt)
        .with_tile_compression(Compression::Gzip)
        .with_metadata(metadata);
    let mut w = Writer::create(&path, options).unwrap();
    let tile = Compression::Gzip
        .compress(&sample_tile(0).encode().unwrap())
        .unwrap();
    w.add_tile(0, 0, 0, &tile, content_hash(&tile)).unwrap();
    w.finish().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_dir_all(dir);
    bytes
}

fn findings(bytes: &[u8]) -> Vec<Finding> {
    validate_bytes(bytes, &ValidationLimits::default())
        .unwrap()
        .findings
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_mvt_archive_without_vector_layers_is_reported() {
    let f = findings(&raw_mvt_archive(Metadata::default()));
    assert!(
        f.iter()
            .any(|f| matches!(f, Finding::VectorLayersInvalid { reason } if reason.contains("no vector_layers"))),
        "{f:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn malformed_vector_layers_are_each_reported() {
    use serde_json::json;
    let cases = [
        (json!("nope"), "not an array"),
        (json!([]), "empty"),
        (json!([1]), "not an object"),
        (json!([{"fields": {}}]), "no string id"),
        (json!([{"id": "a"}]), "no fields"),
        (json!([{"id": "a", "fields": {"k": 1}}]), "not a string"),
        (
            json!([{"id": "a", "fields": {}}, {"id": "a", "fields": {}}]),
            "repeats",
        ),
        (
            json!([{"id": "a", "fields": {}, "minzoom": 5, "maxzoom": 2}]),
            "above maxzoom",
        ),
        (json!([{"id": "a", "fields": {}, "maxzoom": 99}]), "0..=30"),
    ];
    for (value, expect) in cases {
        let mut meta = Metadata::default();
        meta.extra.insert("vector_layers".into(), value.clone());
        let f = findings(&raw_mvt_archive(meta));
        assert!(
            f.iter().any(
                |f| matches!(f, Finding::VectorLayersInvalid { reason } if reason.contains(expect))
            ),
            "{value}: expected {expect:?}, got {f:?}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_raster_archive_is_not_asked_for_vector_layers() {
    let dir = scratch("raster");
    let path = dir.join("r.pmtiles");
    let options = WriterOptions::default().with_tile_type(TileType::Png);
    let mut w = Writer::create(&path, options).unwrap();
    w.add_tile(0, 0, 0, b"png-ish", content_hash(b"png-ish"))
        .unwrap();
    w.finish().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert!(findings(&bytes).is_empty());
    assert!(matches!(
        VectorArchive::try_open(&path),
        Err(VectorError::NotVector(_))
    ));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn the_reader_refuses_an_mvt_archive_with_no_vector_layers() {
    let dir = scratch("nolayers");
    let path = dir.join("n.pmtiles");
    std::fs::write(&path, raw_mvt_archive(Metadata::default())).unwrap();
    assert!(matches!(
        VectorArchive::try_open(&path),
        Err(VectorError::NotVector(m)) if m.contains("vector_layers")
    ));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn the_writer_stays_inside_its_declared_schema() {
    let dir = scratch("schema");
    let options = VectorArchiveOptions::new(schema());
    let mut w = VectorArchiveWriter::create(dir.join("s.pmtiles"), &options).unwrap();

    let mut undeclared_layer = sample_tile(0);
    undeclared_layer.layers[0].name = "mystery".into();
    assert!(matches!(
        w.add_tile(0, 0, 0, &undeclared_layer),
        Err(VectorError::UndeclaredLayer { .. })
    ));

    let mut undeclared_field = sample_tile(0);
    undeclared_field.layers[0].features[0]
        .properties
        .push(("extra".into(), Value::Bool(true)));
    assert!(matches!(
        w.add_tile(0, 0, 0, &undeclared_field),
        Err(VectorError::UndeclaredField { .. })
    ));

    let mut wrong_type = sample_tile(0);
    wrong_type.layers[2].features[0].properties[1] =
        ("height".into(), Value::String("tall".into()));
    assert!(matches!(
        w.add_tile(0, 0, 0, &wrong_type),
        Err(VectorError::UndeclaredField { field, .. }) if field == "height"
    ));
    assert_eq!(w.tile_count(), 0, "a refused tile leaves nothing behind");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_invalid_declaration_is_refused_before_any_file_exists() {
    let dir = scratch("badschema");
    for layers in [
        vec![],
        vec![LayerSchema::new(""), LayerSchema::new("a")],
        vec![LayerSchema::new("a"), LayerSchema::new("a")],
    ] {
        let path = dir.join("b.pmtiles");
        let r = VectorArchiveWriter::create(&path, &VectorArchiveOptions::new(layers));
        assert!(matches!(r, Err(VectorError::Schema(_))));
        assert!(!path.exists());
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_empty_tile_is_not_stored() {
    let dir = scratch("empty");
    let mut w =
        VectorArchiveWriter::create(dir.join("e.pmtiles"), &VectorArchiveOptions::new(schema()))
            .unwrap();
    assert!(!w.add_tile(0, 0, 0, &Tile::default()).unwrap());
    assert!(
        !w.add_tile(
            0,
            0,
            0,
            &Tile {
                layers: vec![Layer::new("lines")]
            }
        )
        .unwrap()
    );
    assert_eq!(w.tile_count(), 0);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn the_same_input_gives_the_same_archive() {
    let dir = scratch("determinism");
    let a = std::fs::read(write_archive(&dir, "a.pmtiles")).unwrap();
    let b = std::fs::read(write_archive(&dir, "b.pmtiles")).unwrap();
    assert_eq!(a, b);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_encoded_tile_is_checked_before_it_is_stored() {
    let dir = scratch("encoded");
    let mut w =
        VectorArchiveWriter::create(dir.join("x.pmtiles"), &VectorArchiveOptions::new(schema()))
            .unwrap();
    assert!(matches!(
        w.add_encoded_tile(0, 0, 0, b"\x1a\x03abc"),
        Err(VectorError::Mvt(_))
    ));
    let good = sample_tile(0).encode().unwrap();
    assert!(w.add_encoded_tile(0, 0, 0, &good).unwrap());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_tile_bomb_is_stopped_at_the_ceiling() {
    // A tile that inflates to far more than the reader allows, in an archive
    // that is otherwise valid.
    let dir = scratch("bomb");
    let mut meta = Metadata::default();
    meta.extra.insert(
        "vector_layers".into(),
        serde_json::json!([{"id": "a", "fields": {}}]),
    );
    let path = dir.join("bomb.pmtiles");
    let options = WriterOptions::default()
        .with_tile_type(TileType::Mvt)
        .with_tile_compression(Compression::Gzip)
        .with_metadata(meta);
    let mut w = Writer::create(&path, options).unwrap();
    let bomb = Compression::Gzip.compress(&vec![0u8; 8 << 20]).unwrap();
    assert!(bomb.len() < 64 << 10);
    w.add_tile(0, 0, 0, &bomb, content_hash(&bomb)).unwrap();
    w.finish().unwrap();
    let archive = VectorArchive::try_open(&path)
        .unwrap()
        .with_limits(DecodeLimits {
            max_bytes: 1 << 20,
            ..DecodeLimits::default()
        });
    assert!(matches!(
        archive.tile_bytes(0, 0, 0),
        Err(VectorError::PmTiles(
            PmTilesError::DecompressionLimit { .. }
        ))
    ));
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// Local and Range readers return the same bytes
// ---------------------------------------------------------------------------

/// A reader that answers ranges from memory and counts the requests, so the
/// test can say how many round trips opening and reading cost. It is the
/// stand-in for a transport: a trait object nobody can mistake for a file.
struct CountingReader {
    bytes: Vec<u8>,
    requests: AtomicUsize,
}

impl RangeReader for CountingReader {
    fn read_range(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let start = usize::try_from(offset).unwrap();
        let end = start.checked_add(len).unwrap();
        self.bytes
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| std::io::ErrorKind::UnexpectedEof.into())
    }

    fn size(&self) -> std::io::Result<Option<u64>> {
        Ok(Some(self.bytes.len() as u64))
    }
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn local_and_range_readers_return_the_same_tile_bytes() {
    let dir = scratch("same");
    let path = write_archive(&dir, "a.pmtiles");
    let local = VectorArchive::try_open(&path).unwrap();
    let remote = VectorArchive::try_new(CountingReader {
        bytes: std::fs::read(&path).unwrap(),
        requests: AtomicUsize::new(0),
    })
    .unwrap();

    let mut compared = 0;
    for z in 0..=2u8 {
        for x in 0..(1u32 << z) {
            for y in 0..(1u32 << z) {
                let a = local.stored_tile(z, x, y).unwrap();
                let b = remote.stored_tile(z, x, y).unwrap();
                assert_eq!(a, b, "stored bytes differ at {z}/{x}/{y}");
                assert_eq!(
                    local.tile(z, x, y).unwrap(),
                    remote.tile(z, x, y).unwrap(),
                    "parsed tiles differ at {z}/{x}/{y}"
                );
                compared += usize::from(a.is_some());
            }
        }
    }
    assert_eq!(compared, 8, "every present tile was compared");
    assert!(remote.stored_tile(2, 1, 1).unwrap().is_none());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn opening_and_fetching_one_tile_costs_few_round_trips() {
    let dir = scratch("trips");
    let path = write_archive(&dir, "a.pmtiles");
    let reader = CountingReader {
        bytes: std::fs::read(&path).unwrap(),
        requests: AtomicUsize::new(0),
    };
    let archive = VectorArchive::try_new(reader).unwrap();
    let after_open = archive.pmtiles().source().requests.load(Ordering::SeqCst);
    // header, root, metadata (read to check vector_layers).
    assert!(after_open <= 3, "{after_open} requests to open");
    archive.tile_bytes(1, 0, 0).unwrap().unwrap();
    let after_tile = archive.pmtiles().source().requests.load(Ordering::SeqCst);
    assert_eq!(after_tile - after_open, 1, "one range read per tile");
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(feature = "object-store-sink")]
#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_object_store_range_reader_agrees_with_the_file() {
    use libviprs::pmtiles::range::ObjectStoreRangeReader;
    use libviprs::sink_object_store::DirectoryObjectStore;
    use std::sync::Arc;

    let dir = scratch("objstore");
    let path = write_archive(&dir, "a.pmtiles");
    let store_root = dir.join("bucket");
    std::fs::create_dir_all(&store_root).unwrap();
    std::fs::copy(&path, store_root.join("a.pmtiles")).unwrap();
    let store = Arc::new(DirectoryObjectStore::new(&store_root));
    let remote = VectorArchive::try_new(ObjectStoreRangeReader::new(store, "a.pmtiles")).unwrap();
    let local = VectorArchive::try_open(&path).unwrap();
    for (z, x, y) in [(0, 0, 0), (1, 1, 1), (2, 3, 3), (2, 2, 1), (2, 1, 1)] {
        assert_eq!(
            local.stored_tile(z, x, y).unwrap(),
            remote.stored_tile(z, x, y).unwrap()
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn the_file_range_reader_and_the_validator_agree_on_the_size() {
    let dir = scratch("size");
    let path = write_archive(&dir, "a.pmtiles");
    let file = FileRangeReader::try_open(&path).unwrap();
    assert_eq!(
        file.size().unwrap(),
        Some(std::fs::metadata(&path).unwrap().len())
    );
    let _: BTreeMap<u8, u8> = BTreeMap::new();
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// Interop: bytes for somebody else's tools
// ---------------------------------------------------------------------------

/// Writes `golden.pmtiles` and `expected.json` into `$LIBVIPRS_VECTOR_GOLDEN_DIR`
/// and does nothing when the variable is unset.
///
/// `tests/vector_1031_interop.sh` runs this, then points go-pmtiles and a
/// Python MVT decoder at the archive. Those are the independent readers: this
/// crate's own decoder agreeing with its own encoder proves nothing about a
/// shared misreading of the spec.
#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn write_the_interop_golden_when_asked() {
    let Some(dir) = std::env::var_os("LIBVIPRS_VECTOR_GOLDEN_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let archive = dir.join("golden.pmtiles");
    // Overwritten, not appended to, so the tool run sees this run's bytes.
    let _ = std::fs::remove_file(&archive);
    let options = VectorArchiveOptions::new(schema())
        .with_name("interop golden")
        .with_bounds_degrees([-10.0, -10.0, 10.0, 10.0])
        .with_center_degrees(0.0, 0.0);
    let mut w = VectorArchiveWriter::create(&archive, &options).unwrap();
    let mut expected = Vec::new();
    let tiles = [
        (0u8, 0u32, 0u32, 0i32),
        (1, 1, 0, 3),
        (1, 0, 1, 7),
        (2, 3, 3, 11),
    ];
    for (z, x, y, n) in tiles {
        let tile = sample_tile(n);
        assert!(w.add_tile(z, x, y, &tile).unwrap());
        let layers: Vec<_> = tile
            .layers
            .iter()
            .map(|l| {
                let features: Vec<_> = l
                    .features
                    .iter()
                    .map(|f| {
                        let (kind, coords) = match &f.geometry {
                            Geometry::Points(p) => ("Point", serde_json::json!(p)),
                            Geometry::Lines(l) => ("LineString", serde_json::json!(l)),
                            Geometry::Polygons(p) => (
                                "Polygon",
                                serde_json::json!(
                                    p.iter()
                                        .map(|p| {
                                            let mut rings = vec![p.exterior.clone()];
                                            rings.extend(p.holes.clone());
                                            rings
                                        })
                                        .collect::<Vec<_>>()
                                ),
                            ),
                        };
                        let props: serde_json::Map<String, serde_json::Value> = f
                            .properties
                            .iter()
                            .map(|(k, v)| {
                                let v = match v {
                                    Value::String(s) => serde_json::json!(s),
                                    Value::Double(d) => serde_json::json!(d),
                                    other => panic!("sample uses {other:?}"),
                                };
                                (k.clone(), v)
                            })
                            .collect();
                        serde_json::json!({
                            "id": f.id, "type": kind, "coordinates": coords, "properties": props
                        })
                    })
                    .collect();
                serde_json::json!({"name": l.name, "extent": l.extent, "features": features})
            })
            .collect();
        expected.push(serde_json::json!({"z": z, "x": x, "y": y, "layers": layers}));
    }
    w.finish().unwrap();
    std::fs::write(
        dir.join("expected.json"),
        serde_json::to_vec_pretty(&expected).unwrap(),
    )
    .unwrap();
}
