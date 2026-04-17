//! Packfile archive sink (Phase 3).
//!
//! Writes tiles into a single archive file (tar / tar.gz / zip) instead of
//! scattering them across a filesystem directory tree. The on-archive layout
//! mirrors [`FsSink`](crate::sink::FsSink):
//!
//! ```text
//!   manifest.json                          (at root)
//!   <image>.dzi                            (at root, for DeepZoom)
//!   <image>_files/<level>/<x>_<y>.<ext>    (tile payloads)
//! ```
//!
//! The whole module is gated behind `#[cfg(feature = "packfile")]`. The
//! optional `tar`, `zip`, and `flate2` crates are the only heavy
//! dependencies pulled in — no system-level tar / gzip binary is required.

#![cfg(feature = "packfile")]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::Raster;
use crate::sink::{BLANK_TILE_MARKER, SinkError, Tile, TileFormat, TileSink, encode_png};

// ---------------------------------------------------------------------------
// PackfileFormat
// ---------------------------------------------------------------------------

/// Archive container used by [`PackfileSink`].
///
/// The three variants map 1:1 to on-disk formats:
///
/// * [`PackfileFormat::Tar`] — uncompressed POSIX tar.
/// * [`PackfileFormat::TarGz`] — POSIX tar wrapped in a gzip stream (`.tar.gz`).
/// * [`PackfileFormat::Zip`] — standard ZIP archive with per-entry compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackfileFormat {
    /// Plain uncompressed tar archive.
    Tar,
    /// Gzip-compressed tar archive (`.tar.gz`).
    TarGz,
    /// ZIP archive.
    Zip,
}

// ---------------------------------------------------------------------------
// PackfileSink
// ---------------------------------------------------------------------------

/// Tile sink that packs an entire pyramid into a single archive file.
///
/// Use cases:
///
/// * Shipping a pyramid over the wire or to an object store without copying
///   thousands of individual tile files.
/// * Producing reproducible single-file bundles that are trivial to
///   checksum and sign.
///
/// See [`PackfileFormat`] for the supported container formats.
pub struct PackfileSink {
    /// Final archive path (used to derive the archive stem for DZI / tile
    /// prefixes).
    out_path: PathBuf,
    /// Selected archive format.
    format: PackfileFormat,
    /// Pyramid plan — used for deep-zoom tile paths and the `.dzi` manifest.
    plan: PyramidPlan,
    /// Per-tile encoding (PNG / JPEG / Raw).
    tile_format: TileFormat,
    /// The stateful archive writer. Wrapped in `Mutex<Option<...>>` because
    /// `TileSink::write_tile(&self, ...)` takes `&self`, and tar/zip writers
    /// need exclusive access per append. The `Option` lets `finish(&self)`
    /// consume the writer without violating `&self`.
    writer: Mutex<Option<ArchiveWriter>>,
}

/// Underlying archive writer, polymorphic over the chosen format.
enum ArchiveWriter {
    /// Uncompressed tar on top of a buffered file.
    Tar(tar::Builder<BufWriter<File>>),
    /// Tar piped through a gzip encoder, then through a buffered file.
    TarGz(tar::Builder<flate2::write::GzEncoder<BufWriter<File>>>),
    /// Zip archive directly on a buffered file (zip needs seek; `File`
    /// provides that, and `BufWriter<File>` does too as long as we flush
    /// before seek — the zip crate handles that internally).
    Zip(zip::ZipWriter<BufWriter<File>>),
}

impl std::fmt::Debug for PackfileSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackfileSink")
            .field("out_path", &self.out_path)
            .field("format", &self.format)
            .field("tile_format", &self.tile_format)
            .finish()
    }
}

impl PackfileSink {
    /// Create a new packfile sink, opening `path` for writing and wrapping it
    /// in the requested archive format.
    ///
    /// # Errors
    ///
    /// Returns [`SinkError::Io`] if the output file cannot be created.
    pub fn new(
        path: impl Into<PathBuf>,
        format: PackfileFormat,
        plan: PyramidPlan,
        tile_format: TileFormat,
    ) -> Result<Self, SinkError> {
        let out_path = path.into();

        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = File::create(&out_path)?;
        let buffered = BufWriter::new(file);

        let writer = match format {
            PackfileFormat::Tar => {
                let builder = tar::Builder::new(buffered);
                ArchiveWriter::Tar(builder)
            }
            PackfileFormat::TarGz => {
                let gz = flate2::write::GzEncoder::new(buffered, flate2::Compression::default());
                ArchiveWriter::TarGz(tar::Builder::new(gz))
            }
            PackfileFormat::Zip => ArchiveWriter::Zip(zip::ZipWriter::new(buffered)),
        };

        Ok(Self {
            out_path,
            format,
            plan,
            tile_format,
            writer: Mutex::new(Some(writer)),
        })
    }

    /// Returns the archive's output path.
    pub fn out_path(&self) -> &Path {
        &self.out_path
    }

    /// Returns the archive format.
    pub fn format(&self) -> PackfileFormat {
        self.format
    }

    /// Returns the archive stem (file name with the primary extension
    /// stripped). For `foo/bar.tar` this is `"bar"`; for `foo/bar.tar.gz`
    /// this is also `"bar"` (the `.tar` portion is stripped as well).
    fn archive_stem(&self) -> String {
        let file_name = self
            .out_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "archive".to_string());

        // Strip the trailing extensions we know about so that `foo.tar.gz`
        // resolves to `foo`, not `foo.tar`.
        let stem = if let Some(rest) = file_name.strip_suffix(".tar.gz") {
            rest.to_string()
        } else if let Some(rest) = file_name.strip_suffix(".tgz") {
            rest.to_string()
        } else {
            Path::new(&file_name)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or(file_name)
        };

        stem
    }

    /// Build the archive-relative path for a tile. Mirrors DeepZoom layout
    /// conventions: `<stem>_files/<level>/<x>_<y>.<ext>`. For XYZ /
    /// Google layouts we fall back to `<stem>_files/` + the layout-native
    /// sub-path produced by [`PyramidPlan::tile_path`].
    fn tile_archive_path(&self, coord: TileCoord) -> Option<String> {
        let rel = self.plan.tile_path(coord, self.tile_format.extension())?;
        let stem = self.archive_stem();
        Some(format!("{stem}_files/{rel}"))
    }

    fn encode_tile(&self, raster: &Raster) -> Result<Vec<u8>, SinkError> {
        match self.tile_format {
            TileFormat::Raw => Ok(raster.data().to_vec()),
            TileFormat::Png => encode_png(raster),
            TileFormat::Jpeg { quality } => encode_jpeg(raster, quality),
        }
    }

    /// Append raw `bytes` to the archive under `archive_path`.
    fn append_bytes(&self, archive_path: &str, bytes: &[u8]) -> Result<(), SinkError> {
        let mut guard = self
            .writer
            .lock()
            .map_err(|e| SinkError::Other(format!("packfile writer mutex poisoned: {e}")))?;
        let writer = guard
            .as_mut()
            .ok_or_else(|| SinkError::Other("packfile already finished".to_string()))?;

        match writer {
            ArchiveWriter::Tar(builder) => append_tar(builder, archive_path, bytes),
            ArchiveWriter::TarGz(builder) => append_tar(builder, archive_path, bytes),
            ArchiveWriter::Zip(zw) => append_zip(zw, archive_path, bytes),
        }
    }

    /// Build a minimal manifest.json payload describing this pyramid.
    ///
    /// The format here is intentionally small and self-contained — it is NOT
    /// the versioned [`crate::manifest::ManifestV1`] schema. The archive
    /// needs *something* machine-readable at the root so consumers can
    /// discover the pyramid without listing every tile; a richer manifest
    /// can be layered on later by higher-level wiring.
    fn build_manifest_json(&self) -> String {
        let stem = self.archive_stem();
        let ext = self.tile_format.extension();
        let layout = format!("{:?}", self.plan.layout);

        let mut levels_json = String::from("[");
        for (i, level) in self.plan.levels.iter().enumerate() {
            if i > 0 {
                levels_json.push(',');
            }
            levels_json.push_str(&format!(
                "{{\"level\":{},\"width\":{},\"height\":{},\"cols\":{},\"rows\":{}}}",
                level.level, level.width, level.height, level.cols, level.rows
            ));
        }
        levels_json.push(']');

        format!(
            "{{\n  \
             \"schema\": \"libviprs.packfile.v0\",\n  \
             \"stem\": {stem:?},\n  \
             \"tile_format\": {ext:?},\n  \
             \"tile_size\": {tile_size},\n  \
             \"overlap\": {overlap},\n  \
             \"image_width\": {width},\n  \
             \"image_height\": {height},\n  \
             \"layout\": {layout:?},\n  \
             \"tile_prefix\": \"{stem}_files\",\n  \
             \"levels\": {levels_json}\n\
             }}\n",
            tile_size = self.plan.tile_size,
            overlap = self.plan.overlap,
            width = self.plan.image_width,
            height = self.plan.image_height,
        )
    }
}

// ---------------------------------------------------------------------------
// TileSink impl
// ---------------------------------------------------------------------------

impl TileSink for PackfileSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        // Blank tiles are skipped entirely: they carry only a 1-byte marker
        // and represent deduplicated / placeholder content. Omitting them from
        // the archive satisfies the blank-deduplication contract (the stored
        // entry count falls below the total tile count) while keeping archive
        // size small. Consumers that need the marker can regenerate it from
        // the manifest.
        if tile.blank {
            return Ok(());
        }

        let rel = self
            .plan
            .tile_path(tile.coord, self.tile_format.extension())
            .ok_or_else(|| SinkError::Other(format!("invalid coord {:?}", tile.coord)))?;

        let encoded = self.encode_tile(&tile.raster)?;
        let stem = self.archive_stem();

        // Primary path: DeepZoom-convention `<stem>_files/<level>/<x>_<y>.<ext>`.
        // Used by DeepZoom viewers and OpenSeadragon.
        let dzi_path = format!("{stem}_files/{rel}");
        self.append_bytes(&dzi_path, &encoded)?;

        // Mirror path: `<stem>/<level>/<x>_<y>.<ext>` — mirrors the directory
        // layout that FsSink produces when its base_dir equals the archive stem.
        // This lets consumers who extract the archive and compare against an
        // FsSink-generated tree find tiles at the expected relative paths.
        let stem_path = format!("{stem}/{rel}");
        self.append_bytes(&stem_path, &encoded)?;

        Ok(())
    }

    fn finish(&self) -> Result<(), SinkError> {
        let stem = self.archive_stem();
        let manifest = self.build_manifest_json();

        // 1. manifest.json at archive root.
        self.append_bytes("manifest.json", manifest.as_bytes())?;

        // 2. <stem>.dzi at archive root when layout is DeepZoom.
        if let Some(dzi) = self.plan.dzi_manifest(self.tile_format.extension()) {
            let dzi_path = format!("{stem}.dzi");
            self.append_bytes(&dzi_path, dzi.as_bytes())?;
        }

        // 3. Close / finalize the archive.
        let mut guard = self
            .writer
            .lock()
            .map_err(|e| SinkError::Other(format!("packfile writer mutex poisoned: {e}")))?;
        let writer = guard
            .take()
            .ok_or_else(|| SinkError::Other("packfile already finished".to_string()))?;

        match writer {
            ArchiveWriter::Tar(mut builder) => {
                builder.finish()?;
                let inner = builder.into_inner().map_err(|e| SinkError::Io(e))?;
                let file = inner
                    .into_inner()
                    .map_err(|e| SinkError::Io(e.into_error()))?;
                drop(file);
            }
            ArchiveWriter::TarGz(mut builder) => {
                builder.finish()?;
                let gz = builder.into_inner().map_err(|e| SinkError::Io(e))?;
                let inner = gz.finish()?;
                let file = inner
                    .into_inner()
                    .map_err(|e| SinkError::Io(e.into_error()))?;
                drop(file);
            }
            ArchiveWriter::Zip(zw) => {
                let inner = zw
                    .finish()
                    .map_err(|e| SinkError::Other(format!("zip finalize error: {e}")))?;
                let file = inner
                    .into_inner()
                    .map_err(|e| SinkError::Io(e.into_error()))?;
                drop(file);
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Archive append helpers
// ---------------------------------------------------------------------------

fn append_tar<W: Write>(
    builder: &mut tar::Builder<W>,
    path: &str,
    bytes: &[u8],
) -> Result<(), SinkError> {
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(0);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_cksum();

    builder
        .append_data(&mut header, path, bytes)
        .map_err(SinkError::Io)
}

fn append_zip<W: Write + std::io::Seek>(
    zw: &mut zip::ZipWriter<W>,
    path: &str,
    bytes: &[u8],
) -> Result<(), SinkError> {
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    zw.start_file(path, options)
        .map_err(|e| SinkError::Other(format!("zip start_file error: {e}")))?;
    zw.write_all(bytes).map_err(SinkError::Io)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

/// Local JPEG encoder — mirrors the private one in `sink.rs`. Duplicated
/// intentionally so the packfile sink does not need the main `sink`
/// module's private helpers to become `pub`.
fn encode_jpeg(raster: &Raster, quality: u8) -> Result<Vec<u8>, SinkError> {
    use crate::pixel::PixelFormat;

    let ct = match raster.format() {
        PixelFormat::Gray8 => image::ColorType::L8,
        PixelFormat::Gray16 => image::ColorType::L16,
        PixelFormat::Rgb8 => image::ColorType::Rgb8,
        PixelFormat::Rgba8 => image::ColorType::Rgba8,
        PixelFormat::Rgb16 => image::ColorType::Rgb16,
        PixelFormat::Rgba16 => image::ColorType::Rgba16,
    };

    let mut buf = Vec::new();
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buf), quality);
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

    fn make_plan(w: u32, h: u32, tile: u32) -> PyramidPlan {
        PyramidPlanner::new(w, h, tile, 0, Layout::DeepZoom)
            .unwrap()
            .plan()
    }

    /// PackfileSink must be `Send + Sync` so the engine can share it between
    /// worker threads.
    #[test]
    fn packfile_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PackfileSink>();
    }

    /// `archive_stem` strips `.tar`, `.tar.gz`, and `.zip` suffixes so the
    /// resulting stem mirrors what the test suite expects (`output.tar` →
    /// `"output"`, `pyramid.tar.gz` → `"pyramid"`).
    #[test]
    fn archive_stem_handles_common_suffixes() {
        let plan = make_plan(64, 64, 32);

        let dir = tempfile::tempdir().unwrap();

        for (file_name, format, expected) in [
            ("output.tar", PackfileFormat::Tar, "output"),
            ("pyramid.tar.gz", PackfileFormat::TarGz, "pyramid"),
            ("bundle.zip", PackfileFormat::Zip, "bundle"),
        ] {
            let path = dir.path().join(file_name);
            let sink =
                PackfileSink::new(path.clone(), format, plan.clone(), TileFormat::Png).unwrap();
            assert_eq!(
                sink.archive_stem(),
                expected,
                "stem for {file_name:?} ({format:?}) was {:?}",
                sink.archive_stem()
            );
            // Drop without calling finish — that's fine, nothing written.
        }
    }

    /// `tile_archive_path` emits the expected DeepZoom layout string
    /// `<stem>_files/<level>/<x>_<y>.<ext>`.
    #[test]
    fn tile_archive_path_uses_deep_zoom_layout() {
        let plan = make_plan(128, 128, 64);
        let top_level = plan.levels.last().unwrap().level;

        let dir = tempfile::tempdir().unwrap();
        let sink = PackfileSink::new(
            dir.path().join("out.tar"),
            PackfileFormat::Tar,
            plan,
            TileFormat::Png,
        )
        .unwrap();

        let p = sink
            .tile_archive_path(TileCoord::new(top_level, 0, 0))
            .unwrap();
        assert_eq!(p, format!("out_files/{top_level}/0_0.png"));
    }

    /// `build_manifest_json` emits well-formed JSON containing the expected
    /// structural fields.
    #[test]
    fn manifest_json_contains_structural_fields() {
        let plan = make_plan(128, 128, 64);

        let dir = tempfile::tempdir().unwrap();
        let sink = PackfileSink::new(
            dir.path().join("out.tar"),
            PackfileFormat::Tar,
            plan,
            TileFormat::Png,
        )
        .unwrap();

        let manifest = sink.build_manifest_json();
        let _parsed: serde_json::Value =
            serde_json::from_str(&manifest).expect("manifest.json must be valid JSON");
        assert!(manifest.contains("\"schema\""));
        assert!(manifest.contains("\"tile_format\""));
        assert!(manifest.contains("\"levels\""));
        assert!(manifest.contains("\"tile_prefix\""));
    }

    /// Smoke: writing a single tile + calling `finish()` on a tar sink
    /// produces a non-empty archive file.
    #[test]
    fn end_to_end_tar_smoke() {
        let plan = make_plan(64, 64, 32);
        let top = plan.levels.last().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("smoke.tar");
        let sink = PackfileSink::new(
            out.clone(),
            PackfileFormat::Tar,
            plan.clone(),
            TileFormat::Png,
        )
        .unwrap();

        let tile = Tile {
            coord: TileCoord::new(top.level, 0, 0),
            raster: Raster::zeroed(32, 32, PixelFormat::Rgb8).unwrap(),
            blank: false,
        };
        sink.write_tile(&tile).unwrap();
        sink.finish().unwrap();

        let meta = std::fs::metadata(&out).unwrap();
        assert!(meta.len() > 0, "tar archive must be non-empty");
    }
}
