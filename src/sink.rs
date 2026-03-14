use std::path::{Path, PathBuf};

use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::Raster;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SinkError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image encode error: {0}")]
    Encode(String),
    #[error("sink error: {0}")]
    Other(String),
}

/// A produced tile, ready for output.
#[derive(Debug)]
pub struct Tile {
    pub coord: TileCoord,
    pub raster: Raster,
}

/// Trait for receiving tiles produced by the engine.
///
/// Implementations handle where tiles go — filesystem, object store, memory, etc.
/// The engine calls `write_tile` from worker threads, so implementations must be
/// `Send + Sync`.
pub trait TileSink: Send + Sync {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError>;
    fn finish(&self) -> Result<(), SinkError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MemorySink
// ---------------------------------------------------------------------------

/// In-memory sink that collects all tiles. Useful for testing.
#[derive(Debug)]
pub struct MemorySink {
    tiles: std::sync::Mutex<Vec<CollectedTile>>,
}

#[derive(Debug, Clone)]
pub struct CollectedTile {
    pub coord: TileCoord,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl MemorySink {
    pub fn new() -> Self {
        Self {
            tiles: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn tiles(&self) -> Vec<CollectedTile> {
        self.tiles.lock().unwrap().clone()
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.lock().unwrap().len()
    }
}

impl Default for MemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl TileSink for MemorySink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        self.tiles.lock().unwrap().push(CollectedTile {
            coord: tile.coord,
            width: tile.raster.width(),
            height: tile.raster.height(),
            data: tile.raster.data().to_vec(),
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SlowSink (testing)
// ---------------------------------------------------------------------------

/// A sink that artificially delays writes. For testing backpressure.
#[derive(Debug)]
pub struct SlowSink {
    inner: MemorySink,
    delay: std::time::Duration,
}

impl SlowSink {
    pub fn new(delay: std::time::Duration) -> Self {
        Self {
            inner: MemorySink::new(),
            delay,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.inner.tile_count()
    }

    pub fn tiles(&self) -> Vec<CollectedTile> {
        self.inner.tiles()
    }
}

impl TileSink for SlowSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        std::thread::sleep(self.delay);
        self.inner.write_tile(tile)
    }
}

// ---------------------------------------------------------------------------
// FsSink — filesystem tile output
// ---------------------------------------------------------------------------

/// Tile image encoding format for filesystem output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileFormat {
    Png,
    Jpeg {
        quality: u8,
    },
    /// Raw pixel bytes (no encoding). Fastest, useful for pipelines that
    /// encode later or for testing.
    Raw,
}

impl TileFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg { .. } => "jpeg",
            Self::Raw => "raw",
        }
    }
}

/// Writes tiles to the local filesystem.
///
/// Directory structure follows the plan's layout:
/// - DeepZoom: `{base}/{level}/{col}_{row}.{ext}` + `{base}.dzi`
/// - XYZ: `{base}/{z}/{x}/{y}.{ext}`
#[derive(Debug)]
pub struct FsSink {
    base_dir: PathBuf,
    plan: PyramidPlan,
    format: TileFormat,
}

impl FsSink {
    pub fn new(base_dir: impl Into<PathBuf>, plan: PyramidPlan, format: TileFormat) -> Self {
        Self {
            base_dir: base_dir.into(),
            plan,
            format,
        }
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    fn tile_path(&self, coord: TileCoord) -> Option<PathBuf> {
        let rel = self.plan.tile_path(coord, self.format.extension())?;
        Some(self.base_dir.join(rel))
    }

    fn encode_tile(&self, raster: &Raster) -> Result<Vec<u8>, SinkError> {
        match self.format {
            TileFormat::Raw => Ok(raster.data().to_vec()),
            TileFormat::Png => encode_png(raster),
            TileFormat::Jpeg { quality } => encode_jpeg(raster, quality),
        }
    }
}

impl TileSink for FsSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        let path = self
            .tile_path(tile.coord)
            .ok_or_else(|| SinkError::Other(format!("invalid coord {:?}", tile.coord)))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let encoded = self.encode_tile(&tile.raster)?;
        std::fs::write(&path, &encoded)?;
        Ok(())
    }

    fn finish(&self) -> Result<(), SinkError> {
        // Write DZI manifest if applicable
        if let Some(manifest) = self.plan.dzi_manifest(self.format.extension()) {
            let dzi_path = self.base_dir.with_extension("dzi");
            std::fs::write(&dzi_path, manifest)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

fn color_type_for_format(fmt: crate::pixel::PixelFormat) -> Result<image::ColorType, SinkError> {
    use crate::pixel::PixelFormat;
    match fmt {
        PixelFormat::Gray8 => Ok(image::ColorType::L8),
        PixelFormat::Gray16 => Ok(image::ColorType::L16),
        PixelFormat::Rgb8 => Ok(image::ColorType::Rgb8),
        PixelFormat::Rgba8 => Ok(image::ColorType::Rgba8),
        PixelFormat::Rgb16 => Ok(image::ColorType::Rgb16),
        PixelFormat::Rgba16 => Ok(image::ColorType::Rgba16),
    }
}

pub(crate) fn encode_png(raster: &Raster) -> Result<Vec<u8>, SinkError> {
    let mut buf = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(std::io::Cursor::new(&mut buf));
    let ct = color_type_for_format(raster.format())?;
    image::ImageEncoder::write_image(
        encoder,
        raster.data(),
        raster.width(),
        raster.height(),
        ct.into(),
    )
    .map_err(|e| SinkError::Encode(e.to_string()))?;
    Ok(buf)
}

fn encode_jpeg(raster: &Raster, quality: u8) -> Result<Vec<u8>, SinkError> {
    let mut buf = Vec::new();
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buf), quality);
    let ct = color_type_for_format(raster.format())?;
    image::ImageEncoder::write_image(
        encoder,
        raster.data(),
        raster.width(),
        raster.height(),
        ct.into(),
    )
    .map_err(|e| SinkError::Encode(e.to_string()))?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::planner::{Layout, PyramidPlanner};

    fn make_tile(level: u32, col: u32, row: u32) -> Tile {
        Tile {
            coord: TileCoord::new(level, col, row),
            raster: Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap(),
        }
    }

    // -- MemorySink tests --

    #[test]
    fn memory_sink_collects_tiles() {
        let sink = MemorySink::new();
        sink.write_tile(&make_tile(0, 0, 0)).unwrap();
        sink.write_tile(&make_tile(1, 0, 0)).unwrap();
        sink.write_tile(&make_tile(1, 1, 0)).unwrap();
        assert_eq!(sink.tile_count(), 3);
    }

    #[test]
    fn memory_sink_preserves_coords() {
        let sink = MemorySink::new();
        sink.write_tile(&make_tile(3, 2, 5)).unwrap();
        let tiles = sink.tiles();
        assert_eq!(tiles[0].coord, TileCoord::new(3, 2, 5));
    }

    #[test]
    fn memory_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemorySink>();
    }

    // -- FsSink tests --

    #[test]
    fn fs_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FsSink>();
    }

    #[test]
    fn fs_sink_writes_tile_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        let sink = FsSink::new(
            dir.path().join("output_files"),
            plan.clone(),
            TileFormat::Raw,
        );

        let tile = Tile {
            coord: TileCoord::new(top.level, 0, 0),
            raster: Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap(),
        };
        sink.write_tile(&tile).unwrap();

        let expected_path = dir
            .path()
            .join(format!("output_files/{}/0_0.raw", top.level));
        assert!(
            expected_path.exists(),
            "Tile file not found at {expected_path:?}"
        );

        let contents = std::fs::read(&expected_path).unwrap();
        assert_eq!(contents.len(), 8 * 8 * 3);
    }

    #[test]
    fn fs_sink_creates_directory_structure() {
        let dir = tempfile::tempdir().unwrap();
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        let sink = FsSink::new(dir.path().join("tiles"), plan.clone(), TileFormat::Raw);

        // Write tiles at multiple positions
        for col in 0..top.cols {
            for row in 0..top.rows {
                let rect = plan.tile_rect(TileCoord::new(top.level, col, row)).unwrap();
                let tile = Tile {
                    coord: TileCoord::new(top.level, col, row),
                    raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                };
                sink.write_tile(&tile).unwrap();
            }
        }

        // Verify level directory exists
        assert!(dir.path().join(format!("tiles/{}", top.level)).is_dir());
    }

    #[test]
    fn fs_sink_writes_dzi_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let planner = PyramidPlanner::new(1024, 768, 256, 1, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let sink = FsSink::new(dir.path().join("output_files"), plan, TileFormat::Png);
        sink.finish().unwrap();

        let dzi_path = dir.path().join("output_files.dzi");
        assert!(dzi_path.exists(), "DZI manifest not found");

        let manifest = std::fs::read_to_string(&dzi_path).unwrap();
        assert!(manifest.contains("Format=\"png\""));
        assert!(manifest.contains("TileSize=\"256\""));
        assert!(manifest.contains("Overlap=\"1\""));
        assert!(manifest.contains("Width=\"1024\""));
        assert!(manifest.contains("Height=\"768\""));
    }

    #[test]
    fn fs_sink_no_dzi_for_xyz() {
        let dir = tempfile::tempdir().unwrap();
        let planner = PyramidPlanner::new(256, 256, 256, 0, Layout::Xyz).unwrap();
        let plan = planner.plan();

        let sink = FsSink::new(dir.path().join("tiles"), plan, TileFormat::Raw);
        sink.finish().unwrap();

        let dzi_path = dir.path().join("tiles.dzi");
        assert!(
            !dzi_path.exists(),
            "DZI should not be written for XYZ layout"
        );
    }

    #[test]
    fn fs_sink_xyz_path_structure() {
        let dir = tempfile::tempdir().unwrap();
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::Xyz).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        let sink = FsSink::new(dir.path().join("tiles"), plan.clone(), TileFormat::Raw);

        let rect = plan.tile_rect(TileCoord::new(top.level, 1, 0)).unwrap();
        let tile = Tile {
            coord: TileCoord::new(top.level, 1, 0),
            raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
        };
        sink.write_tile(&tile).unwrap();

        // XYZ layout: {z}/{x}/{y}.ext
        let expected = dir.path().join(format!("tiles/{}/1/0.raw", top.level));
        assert!(expected.exists(), "XYZ tile not found at {expected:?}");
    }

    #[test]
    fn fs_sink_encodes_png() {
        let dir = tempfile::tempdir().unwrap();
        let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top_level = plan.levels.last().unwrap().level;

        let sink = FsSink::new(dir.path().join("out"), plan, TileFormat::Png);

        let tile = Tile {
            coord: TileCoord::new(top_level, 0, 0),
            raster: Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap(),
        };
        sink.write_tile(&tile).unwrap();

        let path = dir.path().join(format!("out/{top_level}/0_0.png"));
        let bytes = std::fs::read(&path).unwrap();
        // PNG magic bytes
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);
    }

    #[test]
    fn fs_sink_encodes_jpeg() {
        let dir = tempfile::tempdir().unwrap();
        let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top_level = plan.levels.last().unwrap().level;

        let sink = FsSink::new(
            dir.path().join("out"),
            plan,
            TileFormat::Jpeg { quality: 85 },
        );

        let tile = Tile {
            coord: TileCoord::new(top_level, 0, 0),
            raster: Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap(),
        };
        sink.write_tile(&tile).unwrap();

        let path = dir.path().join(format!("out/{top_level}/0_0.jpeg"));
        let bytes = std::fs::read(&path).unwrap();
        // JPEG SOI marker
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn fs_sink_deterministic_paths() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        let sink1 = FsSink::new(dir1.path().join("out"), plan.clone(), TileFormat::Raw);
        let sink2 = FsSink::new(dir2.path().join("out"), plan.clone(), TileFormat::Raw);

        // Same tile to both sinks
        let top = plan.levels.last().unwrap();
        let data = vec![42u8; 256 * 256 * 3];
        let raster = Raster::new(256, 256, PixelFormat::Rgb8, data).unwrap();
        let tile = Tile {
            coord: TileCoord::new(top.level, 0, 0),
            raster,
        };

        sink1.write_tile(&tile).unwrap();
        sink2.write_tile(&tile).unwrap();

        let bytes1 = std::fs::read(dir1.path().join(format!("out/{}/0_0.raw", top.level))).unwrap();
        let bytes2 = std::fs::read(dir2.path().join(format!("out/{}/0_0.raw", top.level))).unwrap();
        assert_eq!(bytes1, bytes2);
    }

    // -- Encoding edge cases --

    #[test]
    fn encode_png_gray8() {
        let raster = Raster::zeroed(4, 4, PixelFormat::Gray8).unwrap();
        let bytes = encode_png(&raster).unwrap();
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);
    }

    #[test]
    fn encode_png_rgba8() {
        let raster = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        let bytes = encode_png(&raster).unwrap();
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);
    }

    #[test]
    fn encode_jpeg_rgb8() {
        let raster = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        let bytes = encode_jpeg(&raster, 90).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    }
}
