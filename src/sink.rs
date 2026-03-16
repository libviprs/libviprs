use std::path::{Path, PathBuf};

use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::Raster;
use thiserror::Error;

/// Errors that can occur when writing tiles to a sink.
///
/// Covers I/O failures (e.g. filesystem permission errors), image encoding
/// failures (e.g. unsupported pixel format for JPEG), and general sink errors
/// for invalid coordinates or configuration. Every [`TileSink`] method returns
/// `Result<(), SinkError>`.
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for error handling patterns.
#[derive(Debug, Error)]
pub enum SinkError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image encode error: {0}")]
    Encode(String),
    #[error("sink error: {0}")]
    Other(String),
}

/// Single-byte marker written in place of blank tiles when using
/// `BlankTileStrategy::Placeholder`. Consumers can detect placeholder
/// files by checking `file.len() == 1 && file[0] == BLANK_TILE_MARKER`.
///
/// # Examples
///
/// See [blank_tile_strategy tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/blank_tile_strategy.rs)
/// for placeholder detection patterns.
pub const BLANK_TILE_MARKER: u8 = 0x00;

/// A produced tile, ready for output.
///
/// Represents a single tile in the pyramid after rasterisation. The engine
/// creates a `Tile` for every grid cell in the plan, attaches the rendered
/// [`Raster`], and passes it to a [`TileSink`]. The `blank` flag allows sinks
/// to write a lightweight placeholder instead of encoding a full image when
/// all pixels are identical.
///
/// # Examples
///
/// See [blank_tile_strategy tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/blank_tile_strategy.rs)
/// for usage with blank detection.
#[derive(Debug)]
pub struct Tile {
    pub coord: TileCoord,
    pub raster: Raster,
    /// When `true`, this tile is blank (all pixels identical) and was marked
    /// for placeholder output by `BlankTileStrategy::Placeholder`.
    ///
    /// See [blank_tile_strategy tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/blank_tile_strategy.rs).
    pub blank: bool,
}

/// Trait for receiving tiles produced by the engine.
///
/// Implementations handle where tiles go — filesystem, object store, memory, etc.
/// The engine calls `write_tile` from worker threads, so implementations must be
/// `Send + Sync`.
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for filesystem sink integration tests and
/// [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// for how the CLI wires up a sink.
pub trait TileSink: Send + Sync {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError>;
    fn finish(&self) -> Result<(), SinkError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MemorySink
// ---------------------------------------------------------------------------

/// In-memory sink that collects all tiles into a `Vec<CollectedTile>`.
///
/// Primarily intended for testing: it lets you assert on the exact tiles the
/// engine produced without touching the filesystem. Thread-safe via an internal
/// `Mutex`, so it satisfies `Send + Sync`.
///
/// # Examples
///
/// See [observability tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
/// for end-to-end usage with the engine.
#[derive(Debug)]
pub struct MemorySink {
    tiles: std::sync::Mutex<Vec<CollectedTile>>,
}

/// A snapshot of a tile captured by [`MemorySink`].
///
/// Stores the tile's coordinate, dimensions, and raw pixel bytes so that tests
/// can inspect tile output without needing to decode an image format. Created
/// automatically when [`MemorySink::write_tile`] is called.
///
/// # Examples
///
/// See [observability tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/observability.rs)
/// for assertions on collected tiles.
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

/// A sink that artificially delays every `write_tile` call by a fixed duration.
///
/// Wraps a [`MemorySink`] so tiles are still collected for inspection. Exists
/// to test backpressure and concurrency behaviour in the engine: by making the
/// sink slow, you can verify that the engine correctly limits in-flight work.
///
/// # Examples
///
/// See [stress tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/stress.rs)
/// for backpressure scenarios.
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
///
/// Controls how [`FsSink`] encodes pixel data before writing to disk. Also
/// determines the file extension via [`TileFormat::extension`].
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for format selection and
/// [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// for how the CLI maps user flags to a `TileFormat`.
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
///
/// Intermediate directories are created automatically. Call [`TileSink::finish`]
/// after all tiles have been written to emit format-specific metadata (e.g. the
/// DZI manifest for DeepZoom layouts).
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for integration tests and
/// [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// for how the `pyramid` command constructs an `FsSink`.
#[derive(Debug)]
pub struct FsSink {
    base_dir: PathBuf,
    plan: PyramidPlan,
    format: TileFormat,
}

impl FsSink {
    /// Creates a new filesystem sink rooted at `base_dir` with the given
    /// pyramid plan and tile encoding format.
    pub fn new(base_dir: impl Into<PathBuf>, plan: PyramidPlan, format: TileFormat) -> Self {
        Self {
            base_dir: base_dir.into(),
            plan,
            format,
        }
    }

    /// Returns the root output directory for this sink.
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

        if tile.blank {
            std::fs::write(&path, [BLANK_TILE_MARKER])?;
        } else {
            let encoded = self.encode_tile(&tile.raster)?;
            std::fs::write(&path, &encoded)?;
        }
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

/// Encodes a [`Raster`] as a PNG image and returns the raw PNG bytes.
///
/// Supports all pixel formats defined in [`crate::pixel::PixelFormat`]. This is
/// exposed publicly so callers that bypass [`FsSink`] (e.g. custom sinks or
/// one-off exports) can still produce PNG output.
///
/// # Errors
///
/// Returns [`SinkError::Encode`] if the underlying image encoder fails.
///
/// # Examples
///
/// See [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
/// for encoding in the context of tile output.
pub fn encode_png(raster: &Raster) -> Result<Vec<u8>, SinkError> {
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
            blank: false,
        }
    }

    // -- MemorySink tests --

    /**
     * Tests that MemorySink accumulates every tile written to it.
     * Works by writing three tiles and checking tile_count() matches.
     * Input: 3 write_tile calls -> Output: tile_count() == 3.
     */
    #[test]
    fn memory_sink_collects_tiles() {
        let sink = MemorySink::new();
        sink.write_tile(&make_tile(0, 0, 0)).unwrap();
        sink.write_tile(&make_tile(1, 0, 0)).unwrap();
        sink.write_tile(&make_tile(1, 1, 0)).unwrap();
        assert_eq!(sink.tile_count(), 3);
    }

    /**
     * Tests that MemorySink faithfully preserves tile coordinates.
     * Works by writing a tile with specific coords and reading them back via tiles().
     * Input: tile at (3, 2, 5) -> Output: tiles()[0].coord == TileCoord(3, 2, 5).
     */
    #[test]
    fn memory_sink_preserves_coords() {
        let sink = MemorySink::new();
        sink.write_tile(&make_tile(3, 2, 5)).unwrap();
        let tiles = sink.tiles();
        assert_eq!(tiles[0].coord, TileCoord::new(3, 2, 5));
    }

    /**
     * Tests that MemorySink satisfies the Send + Sync bounds required by TileSink.
     * Works by using a compile-time assertion function that only accepts Send + Sync types.
     * If MemorySink is not Send + Sync, the test fails to compile.
     */
    #[test]
    fn memory_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemorySink>();
    }

    // -- FsSink tests --

    /**
     * Tests that FsSink satisfies the Send + Sync bounds required by TileSink.
     * Works by using a compile-time assertion function that only accepts Send + Sync types.
     * If FsSink is not Send + Sync, the test fails to compile.
     */
    #[test]
    fn fs_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FsSink>();
    }

    /**
     * Tests that FsSink writes raw tile data to the correct filesystem path.
     * Works by creating a DeepZoom sink, writing one tile, and verifying the file
     * exists at the expected path with the correct byte length (8*8*3 for Rgb8).
     * Input: 8x8 Rgb8 tile -> Output: file at {level}/0_0.raw with 192 bytes.
     *
     * Split for Miri: filesystem operations (mkdir, write) are blocked under
     * Miri's isolation mode. The first half tests path generation and buffer
     * sizing in memory (runs everywhere). The #[cfg(not(miri))] block adds
     * the actual filesystem round-trip (skipped under Miri).
     */
    #[test]
    fn fs_sink_writes_tile_to_disk() {
        let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        // Miri-safe: verify path generation and raw tile size
        let rel = plan
            .tile_path(TileCoord::new(top.level, 0, 0), "raw")
            .unwrap();
        assert!(rel.ends_with("0_0.raw"), "unexpected path: {rel}");
        let raster = Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap();
        assert_eq!(raster.data().len(), 8 * 8 * 3);

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink = FsSink::new(
                dir.path().join("output_files"),
                plan.clone(),
                TileFormat::Raw,
            );
            let tile = Tile {
                coord: TileCoord::new(top.level, 0, 0),
                raster,
                blank: false,
            };
            sink.write_tile(&tile).unwrap();

            let expected_path = dir.path().join("output_files").join(&rel);
            assert!(
                expected_path.exists(),
                "Tile file not found at {expected_path:?}"
            );
            let contents = std::fs::read(&expected_path).unwrap();
            assert_eq!(contents.len(), 8 * 8 * 3);
        }
    }

    /**
     * Tests that FsSink automatically creates intermediate directories.
     * Works by writing all tiles for a 512x512 image and verifying the
     * level directory was created under the base path.
     * Input: multi-tile 512x512 pyramid -> Output: tiles/{level}/ directory exists.
     *
     * Split for Miri: mkdir is blocked under Miri's isolation mode. The first
     * half verifies that tile_path produces a valid path for every coordinate
     * in the grid (runs everywhere). The #[cfg(not(miri))] block tests the
     * actual directory creation on disk (skipped under Miri).
     */
    #[test]
    fn fs_sink_creates_directory_structure() {
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        // Miri-safe: verify path generation works for all tile coords
        for col in 0..top.cols {
            for row in 0..top.rows {
                let path = plan.tile_path(TileCoord::new(top.level, col, row), "raw");
                assert!(path.is_some(), "tile_path returned None for ({col}, {row})");
            }
        }

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink = FsSink::new(dir.path().join("tiles"), plan.clone(), TileFormat::Raw);

            for col in 0..top.cols {
                for row in 0..top.rows {
                    let rect = plan.tile_rect(TileCoord::new(top.level, col, row)).unwrap();
                    let tile = Tile {
                        coord: TileCoord::new(top.level, col, row),
                        raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                        blank: false,
                    };
                    sink.write_tile(&tile).unwrap();
                }
            }

            assert!(dir.path().join(format!("tiles/{}", top.level)).is_dir());
        }
    }

    /**
     * Tests that finish() writes a valid DZI manifest for DeepZoom layouts.
     * Works by calling finish() and verifying the .dzi file contains the
     * expected XML attributes for format, tile size, overlap, and dimensions.
     * Input: 1024x768 image, tile 256, overlap 1 -> Output: .dzi with matching attributes.
     *
     * Split for Miri: file writes are blocked under Miri's isolation mode.
     * The first half calls dzi_manifest() directly and validates the XML
     * string in memory (runs everywhere). The #[cfg(not(miri))] block
     * verifies the manifest is written to disk correctly (skipped under Miri).
     */
    #[test]
    fn fs_sink_writes_dzi_manifest() {
        let planner = PyramidPlanner::new(1024, 768, 256, 1, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        // Miri-safe: verify manifest content in memory
        let manifest = plan
            .dzi_manifest("png")
            .expect("DeepZoom should produce a DZI manifest");
        assert!(manifest.contains("Format=\"png\""));
        assert!(manifest.contains("TileSize=\"256\""));
        assert!(manifest.contains("Overlap=\"1\""));
        assert!(manifest.contains("Width=\"1024\""));
        assert!(manifest.contains("Height=\"768\""));

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink = FsSink::new(dir.path().join("output_files"), plan, TileFormat::Png);
            sink.finish().unwrap();

            let dzi_path = dir.path().join("output_files.dzi");
            assert!(dzi_path.exists(), "DZI manifest not found");

            let on_disk = std::fs::read_to_string(&dzi_path).unwrap();
            assert_eq!(on_disk, manifest);
        }
    }

    /**
     * Tests that finish() does not produce a .dzi file for XYZ layouts.
     * Works by creating an XYZ sink, calling finish(), and asserting no .dzi exists.
     * Input: XYZ layout sink -> Output: no .dzi file on disk.
     *
     * Split for Miri: file writes are blocked under Miri's isolation mode.
     * The first half checks that dzi_manifest() returns None for XYZ layouts
     * (runs everywhere). The #[cfg(not(miri))] block confirms no .dzi file
     * appears on disk after finish() (skipped under Miri).
     */
    #[test]
    fn fs_sink_no_dzi_for_xyz() {
        let planner = PyramidPlanner::new(256, 256, 256, 0, Layout::Xyz).unwrap();
        let plan = planner.plan();

        // Miri-safe: XYZ layout should not produce a manifest
        assert!(
            plan.dzi_manifest("raw").is_none(),
            "DZI should not exist for XYZ layout"
        );

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink = FsSink::new(dir.path().join("tiles"), plan, TileFormat::Raw);
            sink.finish().unwrap();

            let dzi_path = dir.path().join("tiles.dzi");
            assert!(
                !dzi_path.exists(),
                "DZI should not be written for XYZ layout"
            );
        }
    }

    /**
     * Tests that FsSink uses the {z}/{x}/{y}.ext path convention for XYZ layouts.
     * Works by writing a tile at col=1, row=0 and checking the file lands at
     * tiles/{level}/1/0.raw instead of the DeepZoom col_row naming.
     * Input: tile (level, 1, 0) with XYZ layout -> Output: file at {z}/1/0.raw.
     *
     * Split for Miri: mkdir/write are blocked under Miri's isolation mode.
     * The first half verifies tile_path produces the correct XYZ-style
     * relative path in memory (runs everywhere). The #[cfg(not(miri))] block
     * writes the tile to disk and checks the file exists at that path
     * (skipped under Miri).
     */
    #[test]
    fn fs_sink_xyz_path_structure() {
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::Xyz).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        // Miri-safe: verify XYZ path convention
        let rel = plan
            .tile_path(TileCoord::new(top.level, 1, 0), "raw")
            .unwrap();
        let expected_suffix = format!("{}/1/0.raw", top.level);
        assert!(
            rel.ends_with(&expected_suffix),
            "expected XYZ path ending with {expected_suffix}, got {rel}"
        );

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let sink = FsSink::new(dir.path().join("tiles"), plan.clone(), TileFormat::Raw);

            let rect = plan.tile_rect(TileCoord::new(top.level, 1, 0)).unwrap();
            let tile = Tile {
                coord: TileCoord::new(top.level, 1, 0),
                raster: Raster::zeroed(rect.width, rect.height, PixelFormat::Rgb8).unwrap(),
                blank: false,
            };
            sink.write_tile(&tile).unwrap();

            let expected = dir.path().join("tiles").join(&rel);
            assert!(expected.exists(), "XYZ tile not found at {expected:?}");
        }
    }

    /**
     * Tests that FsSink correctly encodes tiles as PNG when configured.
     * Works by writing a tile with TileFormat::Png and verifying the output
     * file starts with the PNG magic bytes (0x89, 'P', 'N', 'G').
     * Input: 8x8 Rgb8 raster -> Output: file with PNG header bytes.
     *
     * Split for Miri: file writes are blocked under Miri's isolation mode.
     * The first half calls encode_png directly and checks the PNG magic
     * bytes in the returned buffer (runs everywhere). The #[cfg(not(miri))]
     * block writes via FsSink and reads the file back from disk to verify
     * the same magic bytes (skipped under Miri).
     */
    #[test]
    fn fs_sink_encodes_png() {
        let raster = Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap();

        // Miri-safe: verify PNG encoding produces valid magic bytes in memory
        let bytes = encode_png(&raster).unwrap();
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);

        #[cfg(not(miri))]
        {
            let dir = tempfile::tempdir().unwrap();
            let planner = PyramidPlanner::new(8, 8, 256, 0, Layout::DeepZoom).unwrap();
            let plan = planner.plan();
            let top_level = plan.levels.last().unwrap().level;

            let sink = FsSink::new(dir.path().join("out"), plan, TileFormat::Png);
            let tile = Tile {
                coord: TileCoord::new(top_level, 0, 0),
                raster,
                blank: false,
            };
            sink.write_tile(&tile).unwrap();

            let path = dir.path().join(format!("out/{top_level}/0_0.png"));
            let on_disk = std::fs::read(&path).unwrap();
            assert_eq!(&on_disk[..4], &[0x89, b'P', b'N', b'G']);
        }
    }

    /**
     * Tests that FsSink correctly encodes tiles as JPEG when configured.
     * Works by writing a tile with TileFormat::Jpeg and verifying the output
     * file starts with the JPEG SOI marker (0xFF, 0xD8).
     * Input: 8x8 Rgb8 raster, quality 85 -> Output: file with JPEG SOI marker.
     *
     * Split for Miri: file writes are blocked under Miri's isolation mode.
     * The first half calls encode_jpeg directly and checks the JPEG SOI
     * marker in the returned buffer (runs everywhere). The #[cfg(not(miri))]
     * block writes via FsSink and reads the file back from disk to verify
     * the same marker (skipped under Miri).
     */
    #[test]
    fn fs_sink_encodes_jpeg() {
        let raster = Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap();

        // Miri-safe: verify JPEG encoding produces valid SOI marker in memory
        let bytes = encode_jpeg(&raster, 85).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);

        #[cfg(not(miri))]
        {
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
                raster,
                blank: false,
            };
            sink.write_tile(&tile).unwrap();

            let path = dir.path().join(format!("out/{top_level}/0_0.jpeg"));
            let on_disk = std::fs::read(&path).unwrap();
            assert_eq!(&on_disk[..2], &[0xFF, 0xD8]);
        }
    }

    /**
     * Tests that two FsSink instances produce identical output for the same input.
     * Works by writing the same tile to two separate temp directories and comparing
     * the raw file contents byte-for-byte.
     * Input: same 256x256 tile to two sinks -> Output: identical file bytes.
     *
     * Split for Miri: tempdir/write are blocked under Miri's isolation mode.
     * The first half encodes the same raster twice via encode_png and asserts
     * byte-for-byte equality in memory (runs everywhere). The #[cfg(not(miri))]
     * block writes via two FsSink instances and compares the files on disk
     * (skipped under Miri).
     */
    #[test]
    fn fs_sink_deterministic_paths() {
        let data = vec![42u8; 256 * 256 * 3];
        let raster = Raster::new(256, 256, PixelFormat::Rgb8, data).unwrap();

        // Miri-safe: encoding the same raster twice should produce identical bytes
        let enc1 = encode_png(&raster).unwrap();
        let enc2 = encode_png(&raster).unwrap();
        assert_eq!(enc1, enc2);

        #[cfg(not(miri))]
        {
            let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
            let plan = planner.plan();
            let top = plan.levels.last().unwrap();

            let dir1 = tempfile::tempdir().unwrap();
            let dir2 = tempfile::tempdir().unwrap();
            let sink1 = FsSink::new(dir1.path().join("out"), plan.clone(), TileFormat::Raw);
            let sink2 = FsSink::new(dir2.path().join("out"), plan.clone(), TileFormat::Raw);

            let tile = Tile {
                coord: TileCoord::new(top.level, 0, 0),
                raster,
                blank: false,
            };

            sink1.write_tile(&tile).unwrap();
            sink2.write_tile(&tile).unwrap();

            let bytes1 =
                std::fs::read(dir1.path().join(format!("out/{}/0_0.raw", top.level))).unwrap();
            let bytes2 =
                std::fs::read(dir2.path().join(format!("out/{}/0_0.raw", top.level))).unwrap();
            assert_eq!(bytes1, bytes2);
        }
    }

    // -- Encoding edge cases --

    /**
     * Tests that encode_png handles the Gray8 pixel format correctly.
     * Works by encoding a 4x4 Gray8 raster and verifying the PNG magic bytes.
     * Input: 4x4 Gray8 raster -> Output: valid PNG (starts with 0x89 PNG).
     */
    #[test]
    fn encode_png_gray8() {
        let raster = Raster::zeroed(4, 4, PixelFormat::Gray8).unwrap();
        let bytes = encode_png(&raster).unwrap();
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);
    }

    /**
     * Tests that encode_png handles the Rgba8 pixel format correctly.
     * Works by encoding a 4x4 Rgba8 raster and verifying the PNG magic bytes.
     * Input: 4x4 Rgba8 raster -> Output: valid PNG (starts with 0x89 PNG).
     */
    #[test]
    fn encode_png_rgba8() {
        let raster = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        let bytes = encode_png(&raster).unwrap();
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);
    }

    /**
     * Tests that encode_jpeg handles Rgb8 pixel format correctly.
     * Works by encoding a 4x4 Rgb8 raster at quality 90 and checking
     * that the output starts with the JPEG SOI marker (0xFF, 0xD8).
     * Input: 4x4 Rgb8 raster, quality 90 -> Output: valid JPEG header.
     */
    #[test]
    fn encode_jpeg_rgb8() {
        let raster = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        let bytes = encode_jpeg(&raster, 90).unwrap();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    }
}
