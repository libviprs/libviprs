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

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::planner::{PyramidPlan, TileCoord};
use crate::raster::Raster;
use crate::sink::{SinkError, Tile, TileFormat, TileSink, color_type_for_format, encode_png};

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
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-sink)
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
///
/// On the CLI this sink is selected by passing a `packfile://…` URI to `--sink`.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-sink)
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
    /// Boxed to keep the enum variants close in size (the gzip encoder is
    /// large compared to the other writers).
    TarGz(Box<tar::Builder<flate2::write::GzEncoder<BufWriter<File>>>>),
    /// Zip archive directly on a buffered file (zip needs seek; `File`
    /// provides that, and `BufWriter<File>` does too as long as we flush
    /// before seek — the zip crate handles that internally). Boxed to
    /// avoid a large size disparity between enum variants.
    Zip(Box<zip::ZipWriter<BufWriter<File>>>),
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
    /// Start a fluent builder rooted at `path`.
    ///
    /// Call [`PackfileSinkBuilder::plan`] (required), then any combination of
    /// [`PackfileSinkBuilder::format`] (default: [`PackfileFormat::Tar`])
    /// and [`PackfileSinkBuilder::tile_format`] (default:
    /// [`TileFormat::Png`]), then [`PackfileSinkBuilder::build`]:
    ///
    /// ```
    /// use libviprs::planner::{Layout, PyramidPlanner};
    /// use libviprs::sink::TileFormat;
    /// use libviprs::sink_packfile::{PackfileFormat, PackfileSink};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let plan = PyramidPlanner::new(1024, 768, 256, 0, Layout::DeepZoom)?.plan();
    /// let dir = tempfile::tempdir()?;
    ///
    /// let sink = PackfileSink::builder(dir.path().join("out.zip"))
    ///     .plan(plan)
    ///     .format(PackfileFormat::Zip)
    ///     .tile_format(TileFormat::Jpeg { quality: 85 })
    ///     .build()?;
    /// # let _ = sink;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder(path: impl Into<PathBuf>) -> PackfileSinkBuilder {
        PackfileSinkBuilder {
            out_path: path.into(),
            format: PackfileFormat::Tar,
            tile_format: TileFormat::Png,
            plan: None,
        }
    }

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

        if let Some(parent) = out_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
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
                ArchiveWriter::TarGz(Box::new(tar::Builder::new(gz)))
            }
            PackfileFormat::Zip => ArchiveWriter::Zip(Box::new(zip::ZipWriter::new(buffered))),
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
        if let Some(rest) = file_name.strip_suffix(".tar.gz") {
            rest.to_string()
        } else if let Some(rest) = file_name.strip_suffix(".tgz") {
            rest.to_string()
        } else {
            Path::new(&file_name)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or(file_name)
        }
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
        // Fragile-write-path branch of the crate poison policy (crate::poison):
        // the archive writer appends sequentially, so a holder that panicked
        // mid-entry leaves a half-written tar/zip that every later append would
        // corrupt. We therefore do NOT recover the guard; we surface the poison
        // as a typed error so the run aborts cleanly instead of building on an
        // unusable writer.
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
// PackfileSinkBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for [`PackfileSink`].
///
/// Produced by [`PackfileSink::builder`]. The `plan` field is required —
/// calling [`PackfileSinkBuilder::build`] without one returns
/// [`SinkError::MissingField`]. The archive and tile formats default to
/// [`PackfileFormat::Tar`] and [`TileFormat::Png`] respectively so the
/// minimum-viable call is:
///
/// ```
/// use libviprs::planner::{Layout, PyramidPlanner};
/// use libviprs::sink_packfile::PackfileSink;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let plan = PyramidPlanner::new(1024, 768, 256, 0, Layout::DeepZoom)?.plan();
/// let dir = tempfile::tempdir()?;
///
/// let sink = PackfileSink::builder(dir.path().join("out.tar")).plan(plan).build()?;
/// # let _ = sink;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PackfileSinkBuilder {
    out_path: PathBuf,
    format: PackfileFormat,
    tile_format: TileFormat,
    plan: Option<PyramidPlan>,
}

impl PackfileSinkBuilder {
    /// Set the archive container format. Defaults to [`PackfileFormat::Tar`].
    pub fn format(mut self, format: PackfileFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the per-tile encoding format. Defaults to [`TileFormat::Png`].
    pub fn tile_format(mut self, tile_format: TileFormat) -> Self {
        self.tile_format = tile_format;
        self
    }

    /// Attach the pyramid plan. Required — the archive needs the plan to
    /// compute tile paths and emit companion manifests.
    pub fn plan(mut self, plan: PyramidPlan) -> Self {
        self.plan = Some(plan);
        self
    }

    /// Finalise the configuration and open the archive for writing.
    ///
    /// # Errors
    ///
    /// * [`SinkError::MissingField`] if [`PackfileSinkBuilder::plan`] was
    ///   never called.
    /// * [`SinkError::Io`] if the output file cannot be created.
    pub fn build(self) -> Result<PackfileSink, SinkError> {
        let plan = self
            .plan
            .ok_or(SinkError::MissingField("PackfileSinkBuilder::plan"))?;
        PackfileSink::new(self.out_path, self.format, plan, self.tile_format)
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

        let dzi_path = self
            .tile_archive_path(tile.coord)
            .ok_or_else(|| SinkError::Other(format!("invalid coord {:?}", tile.coord)))?;

        let encoded = self.encode_tile(&tile.raster)?;

        // Primary path: DeepZoom-convention `<stem>_files/<level>/<x>_<y>.<ext>`.
        // Used by DeepZoom viewers and OpenSeadragon.
        self.append_bytes(&dzi_path, &encoded)?;

        // Mirror path: `<stem>/<level>/<x>_<y>.<ext>` — mirrors the directory
        // layout that FsSink produces when its base_dir equals the archive stem.
        // This lets consumers who extract the archive and compare against an
        // FsSink-generated tree find tiles at the expected relative paths.
        let rel = self
            .plan
            .tile_path(tile.coord, self.tile_format.extension())
            .expect("tile_archive_path succeeded above");
        let stem_path = format!("{}/{}", self.archive_stem(), rel);
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
                let inner = builder.into_inner().map_err(SinkError::Io)?;
                let file = inner
                    .into_inner()
                    .map_err(|e| SinkError::Io(e.into_error()))?;
                drop(file);
            }
            ArchiveWriter::TarGz(mut builder) => {
                builder.finish()?;
                let gz = builder.into_inner().map_err(SinkError::Io)?;
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
// ZipSink
// ---------------------------------------------------------------------------

/// Tile sink that packs an entire pyramid into a single ZIP archive.
///
/// `ZipSink` is a thin newtype over [`PackfileSink`] with the container
/// format fixed to [`PackfileFormat::Zip`]. It gives callers who only ever
/// want ZIP output a dedicated type to name, instead of threading a
/// [`PackfileFormat`] argument through their code. Every [`TileSink`] method
/// delegates straight to the inner [`PackfileSink`] (through the
/// [`TileSink::inner_sink`] decorator hook for the engine-bookkeeping
/// methods, and directly for [`TileSink::write_tile`] /
/// [`TileSink::finish`]); ZipSink adds no archive-writing logic of its own.
#[derive(Debug)]
pub struct ZipSink {
    inner: PackfileSink,
}

impl ZipSink {
    /// Create a ZIP tile sink writing to `path`, using `plan` for tile paths
    /// and `format` for the per-tile encoding. The container format is always
    /// [`PackfileFormat::Zip`].
    ///
    /// This is the fallible constructor and the recommended entry point when a
    /// caller wants to handle archive-creation failure explicitly. It delegates
    /// to [`PackfileSink::new`]`(path, PackfileFormat::Zip, plan, format)`,
    /// which opens `path` for writing up front.
    ///
    /// # Errors
    ///
    /// Returns [`SinkError::Io`] if the output file at `path` cannot be created
    /// (for example a parent directory that cannot be created, or a permissions
    /// error).
    pub fn try_new(
        path: impl Into<PathBuf>,
        plan: PyramidPlan,
        format: TileFormat,
    ) -> Result<Self, SinkError> {
        let inner = PackfileSink::new(path, PackfileFormat::Zip, plan, format)?;
        Ok(Self { inner })
    }

    /// Create a ZIP tile sink writing to `path`, using `plan` for tile paths
    /// and `format` for the per-tile encoding. The container format is always
    /// [`PackfileFormat::Zip`]. This is a convenience wrapper over
    /// [`ZipSink::try_new`] for call sites that treat archive creation as
    /// infallible.
    ///
    /// # Panics
    ///
    /// Unlike the filesystem sinks (whose `new` constructors do no eager I/O
    /// and defer every write to their `finish` path), `ZipSink::new` performs
    /// eager I/O: it creates the archive file at `path` when the sink is
    /// constructed. It therefore panics if that file cannot be created (for
    /// example a parent directory that cannot be created, or a permissions
    /// error). Callers who need to handle that failure should use
    /// [`ZipSink::try_new`], which returns a [`Result`] instead.
    pub fn new(path: impl Into<PathBuf>, plan: PyramidPlan, format: TileFormat) -> Self {
        Self::try_new(path, plan, format)
            .expect("ZipSink::new: failed to create the ZIP archive output file")
    }

    /// Returns the archive's output path.
    pub fn out_path(&self) -> &Path {
        self.inner.out_path()
    }
}

impl TileSink for ZipSink {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        self.inner.write_tile(tile)
    }

    fn finish(&self) -> Result<(), SinkError> {
        self.inner.finish()
    }

    fn inner_sink(&self) -> Option<&dyn TileSink> {
        Some(&self.inner)
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
    // Was its own fourth copy of `sink.rs`'s mapping, kept separate so this
    // module would not need `sink`'s private helpers to become `pub`. That
    // rationale went away once #969 made the mapping `crate::pixel::image_
    // color_type` and #940's review made the wrapper around it
    // `pub(crate)`: this now calls the same one `sink.rs` and
    // `sink_object_store.rs` do, closing the gap the review found (a live,
    // untested fourth copy of exactly the mapping #969 set out to
    // consolidate).
    let ct = color_type_for_format(raster.format())?;

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
    .map_err(|e| SinkError::EncodeMsg(format!("png: {e}")))?;
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
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
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
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
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
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
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
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
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

    /// TDD for the ZipSink lane (libviprs-tests#87): driving a `ZipSink`
    /// across a small pyramid produces a `.zip` that exists, is non-empty,
    /// and, when reopened with the `zip` crate, contains the expected tile
    /// entries plus the root manifest.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn zip_sink_writes_pyramid_archive() {
        let plan = make_plan(64, 64, 32);

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("pyramid.zip");

        let sink = ZipSink::new(out.clone(), plan.clone(), TileFormat::Png);

        // Drive every tile of the small pyramid through the sink, counting the
        // total we feed in so we can pin the stored `_files/` entry count.
        let mut driven_tiles = 0usize;
        for level in &plan.levels {
            for y in 0..level.rows {
                for x in 0..level.cols {
                    let tile = Tile {
                        coord: TileCoord::new(level.level, x, y),
                        raster: Raster::zeroed(32, 32, PixelFormat::Rgb8).unwrap(),
                        blank: false,
                    };
                    sink.write_tile(&tile).unwrap();
                    driven_tiles += 1;
                }
            }
        }
        assert!(driven_tiles > 0, "test must drive at least one tile");
        sink.finish().unwrap();

        // 1. The archive file exists and is non-empty.
        let meta = std::fs::metadata(&out).unwrap();
        assert!(meta.len() > 0, "zip archive must be non-empty");

        // 2. Reopen with the zip crate and enumerate entry names.
        let file = File::open(&out).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert!(!archive.is_empty(), "zip archive must contain entries");

        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();

        // 3. It carries DeepZoom tile entries under `<stem>_files/…` and the
        //    root manifest.
        assert!(
            names
                .iter()
                .any(|n| n.contains("_files/") && n.ends_with(".png")),
            "zip must contain DeepZoom tile entries, got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "manifest.json"),
            "zip must contain the root manifest.json, got: {names:?}"
        );

        // 4. Exactly one DeepZoom `_files/<level>/<x>_<y>.png` entry per driven
        //    tile: no tile is dropped and none is duplicated on that path. (The
        //    mirror `<stem>/…` entries live outside `_files/` and are excluded.)
        let deep_zoom_tiles = names
            .iter()
            .filter(|n| n.contains("_files/") && n.ends_with(".png"))
            .count();
        assert_eq!(
            deep_zoom_tiles, driven_tiles,
            "DeepZoom `_files/*.png` entry count must equal the driven tile total, got: {names:?}"
        );

        // 5. `finish()` writes the DeepZoom `<stem>.dzi` manifest at the root.
        //    The archive stem for `pyramid.zip` is `pyramid`, so pin the
        //    contract by requiring the `pyramid.dzi` entry.
        assert!(
            names.iter().any(|n| n == "pyramid.dzi"),
            "zip must contain the DeepZoom manifest `pyramid.dzi`, got: {names:?}"
        );
    }

    /// `encode_jpeg` had no direct test of the uint/float refusal before
    /// issue #940's batch-1 review: it carried its own fourth copy of the
    /// mapping `sink.rs` and `sink_object_store.rs` already had direct tests
    /// for (`the_tile_sinks_refuse_the_uint_carrier_by_name`,
    /// `the_object_store_sink_refuses_the_uint_carrier_by_name`), so a
    /// mutation landing only in this module's copy had nothing to catch it.
    /// Mirrors those two now that this module calls the same
    /// [`crate::sink::color_type_for_format`] they do.
    #[test]
    fn the_packfile_sink_refuses_the_uint_carrier_by_name() {
        let n = |v: u16| core::num::NonZeroU16::new(v).unwrap();
        let u = PixelFormat::Uint32(n(1));
        let msg = color_type_for_format(u)
            .expect_err("a uint raster is not an image tile")
            .to_string();
        assert!(
            msg.contains("32-bit unsigned") && msg.contains("Uint32"),
            "the refusal does not name the carrier: {msg}"
        );
        let f = PixelFormat::FloatF32(n(1));
        let fmsg = color_type_for_format(f)
            .expect_err("a float raster is not an image tile")
            .to_string();
        assert!(fmsg.contains("float"), "{fmsg}");
        assert_ne!(msg, fmsg);
    }
}
