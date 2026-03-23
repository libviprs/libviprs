use thiserror::Error;

/// Errors that can occur when constructing a [`PyramidPlanner`].
///
/// These represent invalid configurations that would lead to undefined behaviour
/// during pyramid generation -- for example, zero-sized images or an overlap that
/// equals or exceeds the tile size. Catching them at planner-construction time
/// keeps the rest of the pipeline free of defensive checks.
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
#[derive(Debug, Error)]
pub enum PlannerError {
    #[error("zero dimension: {width}x{height}")]
    ZeroDimension { width: u32, height: u32 },
    #[error("tile size must be > 0, got {0}")]
    ZeroTileSize(u32),
    #[error("overlap {overlap} must be less than tile size {tile_size}")]
    OverlapTooLarge { overlap: u32, tile_size: u32 },
    #[error("dimensions too large: {width}x{height} would overflow")]
    DimensionOverflow { width: u32, height: u32 },
}

/// Output tile layout format.
///
/// Determines the directory structure and naming convention used when writing
/// pyramid tiles to disk. Different viewers expect different conventions, so
/// this enum lets callers pick the one that matches their target.
///
/// # Variants
///
/// * `DeepZoom` -- `{level}/{col}_{row}.{ext}` with a companion `.dzi` XML
///   manifest. Compatible with OpenSeadragon, Leaflet, and similar viewers.
/// * `Xyz` -- `{z}/{x}/{y}.{ext}`, the standard slippy-map / tile-server
///   convention used by web mapping libraries.
///
/// # Example usage
///
/// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Deep Zoom Image -- `{level}/{col}_{row}.{ext}`, plus `.dzi` manifest.
    DeepZoom,
    /// XYZ / slippy map -- `{z}/{x}/{y}.{ext}`.
    Xyz,
    /// Google Maps -- power-of-2 tile grids, `{z}/{x}/{y}.{ext}`.
    /// z=0 is the most zoomed-out level (single tile). No manifest.
    Google,
}

/// Configuration builder for pyramid tile generation.
///
/// Holds the source image dimensions, tile size, overlap, and target layout.
/// Call [`PyramidPlanner::plan`] to compute a complete [`PyramidPlan`] that
/// describes every level and tile coordinate in the pyramid.
///
/// Constructing a `PyramidPlanner` validates all parameters up-front, so
/// downstream code can assume the configuration is sane.
///
/// # Example usage
///
/// * [CLI plan command](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
#[derive(Debug, Clone)]
pub struct PyramidPlanner {
    image_width: u32,
    image_height: u32,
    tile_size: u32,
    overlap: u32,
    layout: Layout,
    centre: bool,
}

/// A complete, immutable pyramid plan ready for execution.
///
/// Contains all the information needed to slice an image into pyramid tiles:
/// the original image dimensions, tile parameters, the chosen layout, and a
/// [`Vec`] of [`LevelPlan`]s ordered from the smallest (most zoomed-out) level
/// to the largest (full-resolution) level.
///
/// Obtain a `PyramidPlan` by calling [`PyramidPlanner::plan`]. Tile writers
/// iterate over [`PyramidPlan::tile_coords`] and use [`PyramidPlan::tile_rect`]
/// and [`PyramidPlan::tile_path`] to decide what to read and where to write.
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
/// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyramidPlan {
    pub image_width: u32,
    pub image_height: u32,
    pub tile_size: u32,
    pub overlap: u32,
    pub layout: Layout,
    pub levels: Vec<LevelPlan>,
    /// Padded canvas width at full resolution.
    pub canvas_width: u32,
    /// Padded canvas height at full resolution.
    pub canvas_height: u32,
    /// Whether the image is centred within the tile grid.
    pub centre: bool,
    /// Horizontal centre offset at full resolution (pixels).
    pub centre_offset_x: u32,
    /// Vertical centre offset at full resolution (pixels).
    pub centre_offset_y: u32,
}

/// Plan for a single pyramid level.
///
/// Each level represents one resolution step in the multi-resolution pyramid.
/// Level 0 is the smallest (typically 1x1), and the highest-indexed level is
/// the full-resolution image. The `cols` and `rows` fields describe the tile
/// grid at this resolution, computed via ceiling division of the level
/// dimensions by the tile size.
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelPlan {
    /// Level index (0 = smallest / most zoomed out).
    pub level: u32,
    /// Pixel width at this level.
    pub width: u32,
    /// Pixel height at this level.
    pub height: u32,
    /// Number of tile columns.
    pub cols: u32,
    /// Number of tile rows.
    pub rows: u32,
}

/// Coordinates identifying a single tile in the pyramid.
///
/// A lightweight, `Copy` address for one tile, consisting of the pyramid
/// level index and the column/row position within that level's tile grid.
/// Used as a key when querying [`PyramidPlan::tile_rect`] and
/// [`PyramidPlan::tile_path`].
///
/// # Example usage
///
/// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub level: u32,
    pub col: u32,
    pub row: u32,
}

/// The pixel rectangle a tile covers within its level's image space.
///
/// Describes the origin `(x, y)` and size `(width, height)` of the region
/// that should be read from the level image to produce this tile. Edge tiles
/// are clipped to the image boundary, so their dimensions may be smaller than
/// `tile_size + 2 * overlap`. Overlap pixels are included in the rectangle
/// so that tile viewers can blend edges seamlessly.
///
/// # Example usage
///
/// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PyramidPlanner {
    /// Create a new `PyramidPlanner` after validating all parameters.
    ///
    /// Returns [`PlannerError`] if any dimension is zero, tile size is zero,
    /// or overlap is not strictly less than tile size.
    pub fn new(
        image_width: u32,
        image_height: u32,
        tile_size: u32,
        overlap: u32,
        layout: Layout,
    ) -> Result<Self, PlannerError> {
        if image_width == 0 || image_height == 0 {
            return Err(PlannerError::ZeroDimension {
                width: image_width,
                height: image_height,
            });
        }
        if tile_size == 0 {
            return Err(PlannerError::ZeroTileSize(tile_size));
        }
        if overlap >= tile_size {
            return Err(PlannerError::OverlapTooLarge { overlap, tile_size });
        }
        Ok(Self {
            image_width,
            image_height,
            tile_size,
            overlap,
            layout,
            centre: false,
        })
    }

    /// Enable or disable centring the image within the tile grid.
    pub fn with_centre(mut self, centre: bool) -> Self {
        self.centre = centre;
        self
    }

    /// Compute the full pyramid plan.
    ///
    /// Builds every [`LevelPlan`] from level 0 (1x1) up to the full-resolution
    /// level, computing tile grid dimensions at each step. The returned
    /// [`PyramidPlan`] is a pure data structure with no side-effects; it can
    /// be queried, serialised, or handed to a tile-writing executor.
    ///
    /// # Example usage
    ///
    /// * [CLI plan command](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
    /// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
    pub fn plan(&self) -> PyramidPlan {
        if self.layout == Layout::Google {
            return self.plan_google();
        }

        let levels = self.compute_levels();

        let (canvas_width, canvas_height, offset_x, offset_y) = if self.centre {
            let top = levels.last().unwrap();
            let grid_w = top.cols * self.tile_size;
            let grid_h = top.rows * self.tile_size;
            let ox = (grid_w - self.image_width) / 2;
            let oy = (grid_h - self.image_height) / 2;
            (grid_w, grid_h, ox, oy)
        } else {
            (self.image_width, self.image_height, 0, 0)
        };

        PyramidPlan {
            image_width: self.image_width,
            image_height: self.image_height,
            tile_size: self.tile_size,
            overlap: self.overlap,
            layout: self.layout,
            levels,
            canvas_width,
            canvas_height,
            centre: self.centre,
            centre_offset_x: offset_x,
            centre_offset_y: offset_y,
        }
    }

    fn plan_google(&self) -> PyramidPlan {
        let ts = self.tile_size;
        let cols_needed = ceil_div(self.image_width, ts);
        let rows_needed = ceil_div(self.image_height, ts);
        let max_grid = cols_needed.max(rows_needed);

        // n_levels = ceil(log2(max_grid)) + 1, minimum 1
        let n_levels = if max_grid <= 1 {
            1
        } else {
            (32 - (max_grid - 1).leading_zeros()) + 1
        };

        // Canvas is ts * 2^(n_levels-1) — square
        let canvas = ts * (1u32 << (n_levels - 1));

        let (offset_x, offset_y) = if self.centre {
            (
                (canvas - self.image_width) / 2,
                (canvas - self.image_height) / 2,
            )
        } else {
            (0, 0)
        };

        // Build levels: z=0 is 1x1 grid, z=n_levels-1 is full resolution grid
        let mut levels = Vec::with_capacity(n_levels as usize);
        let mut w = self.image_width;
        let mut h = self.image_height;

        // Collect image dimensions at each level (top-down)
        let mut img_dims = vec![(w, h)];
        for _ in 1..n_levels {
            w = ceil_div(w, 2);
            h = ceil_div(h, 2);
            img_dims.push((w, h));
        }
        img_dims.reverse(); // level 0 = smallest

        for z in 0..n_levels {
            let grid = 1u32 << z; // 2^z
            let (iw, ih) = img_dims[z as usize];
            levels.push(LevelPlan {
                level: z,
                width: iw,
                height: ih,
                cols: grid,
                rows: grid,
            });
        }

        PyramidPlan {
            image_width: self.image_width,
            image_height: self.image_height,
            tile_size: ts,
            overlap: self.overlap,
            layout: Layout::Google,
            levels,
            canvas_width: canvas,
            canvas_height: canvas,
            centre: self.centre,
            centre_offset_x: offset_x,
            centre_offset_y: offset_y,
        }
    }

    fn compute_levels(&self) -> Vec<LevelPlan> {
        let mut levels = Vec::new();
        let mut w = self.image_width;
        let mut h = self.image_height;

        // Build from full resolution down to 1x1-ish
        let mut dims = vec![(w, h)];
        while w > 1 || h > 1 {
            w = ceil_div(w, 2);
            h = ceil_div(h, 2);
            dims.push((w, h));
        }

        // Reverse so level 0 = smallest
        dims.reverse();

        for (level, &(w, h)) in dims.iter().enumerate() {
            let (cols, rows) = self.tile_grid(w, h);
            levels.push(LevelPlan {
                level: level as u32,
                width: w,
                height: h,
                cols,
                rows,
            });
        }

        levels
    }

    /// Compute the number of tile columns and rows for an image of the given dimensions.
    fn tile_grid(&self, width: u32, height: u32) -> (u32, u32) {
        if width == 0 || height == 0 {
            return (0, 0);
        }
        let cols = ceil_div(width, self.tile_size);
        let rows = ceil_div(height, self.tile_size);
        (cols, rows)
    }
}

impl PyramidPlan {
    /// Total number of tiles across all levels.
    ///
    /// Useful for progress reporting and pre-allocating storage. The count is
    /// the sum of `cols * rows` for every [`LevelPlan`] in the pyramid.
    pub fn total_tile_count(&self) -> u64 {
        self.levels
            .iter()
            .map(|l| l.cols as u64 * l.rows as u64)
            .sum()
    }

    /// Number of levels in the pyramid.
    ///
    /// Equal to `floor(log2(max(width, height))) + 1`, ranging from a single
    /// 1x1 level for a 1x1 image up to ~17 levels for a 50 000-pixel image.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Iterate over all tile coordinates in the pyramid.
    ///
    /// Yields [`TileCoord`]s in level-ascending, row-major order. This is the
    /// primary iteration method used by tile-writing executors to walk every
    /// tile that needs to be produced.
    ///
    /// # Example usage
    ///
    /// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
    pub fn tile_coords(&self) -> impl Iterator<Item = TileCoord> + '_ {
        self.levels.iter().flat_map(|level| {
            (0..level.rows).flat_map(move |row| {
                (0..level.cols).map(move |col| TileCoord {
                    level: level.level,
                    col,
                    row,
                })
            })
        })
    }

    /// Get the pixel rectangle for a tile, accounting for overlap.
    ///
    /// The rectangle describes where in the level's image space this tile
    /// reads from. Edge tiles are clipped to the image boundary. Interior
    /// tiles extend by `overlap` pixels on every side so that viewers can
    /// blend adjacent tiles without visible seams.
    ///
    /// Returns `None` if the coordinate is out of bounds (invalid level,
    /// column, or row).
    ///
    /// # Example usage
    ///
    /// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
    pub fn tile_rect(&self, coord: TileCoord) -> Option<TileRect> {
        let level = self.levels.get(coord.level as usize)?;
        if coord.col >= level.cols || coord.row >= level.rows {
            return None;
        }

        if self.layout == Layout::Google {
            // Google layout: tiles cover canvas space, always full tile_size
            let ts = self.tile_size;
            return Some(TileRect {
                x: coord.col * ts,
                y: coord.row * ts,
                width: ts,
                height: ts,
            });
        }

        let x_start = if coord.col == 0 {
            0
        } else {
            coord.col * self.tile_size - self.overlap
        };
        let y_start = if coord.row == 0 {
            0
        } else {
            coord.row * self.tile_size - self.overlap
        };

        let x_end_unclipped = if coord.col == 0 {
            self.tile_size + self.overlap
        } else {
            (coord.col + 1) * self.tile_size + self.overlap
        };
        let y_end_unclipped = if coord.row == 0 {
            self.tile_size + self.overlap
        } else {
            (coord.row + 1) * self.tile_size + self.overlap
        };

        let x_end = x_end_unclipped.min(level.width);
        let y_end = y_end_unclipped.min(level.height);

        Some(TileRect {
            x: x_start,
            y: y_start,
            width: x_end - x_start,
            height: y_end - y_start,
        })
    }

    /// Generate the output path for a tile given the layout.
    ///
    /// Formats the path according to the [`Layout`] selected at planning time:
    /// * [`Layout::DeepZoom`] -- `{level}/{col}_{row}.{ext}`
    /// * [`Layout::Xyz`] -- `{level}/{col}/{row}.{ext}`
    ///
    /// Returns `None` if the coordinate is out of bounds.
    ///
    /// # Example usage
    ///
    /// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
    pub fn tile_path(&self, coord: TileCoord, extension: &str) -> Option<String> {
        let level = self.levels.get(coord.level as usize)?;
        if coord.col >= level.cols || coord.row >= level.rows {
            return None;
        }
        match self.layout {
            Layout::DeepZoom => Some(format!(
                "{}/{}_{}.{}",
                coord.level, coord.col, coord.row, extension
            )),
            Layout::Xyz => Some(format!(
                "{}/{}/{}.{}",
                coord.level, coord.col, coord.row, extension
            )),
            // Google Maps convention: {z}/{y}/{x}.{ext} (row before col)
            Layout::Google => Some(format!(
                "{}/{}/{}.{}",
                coord.level, coord.row, coord.col, extension
            )),
        }
    }

    /// Generate a Deep Zoom `.dzi` manifest (XML).
    ///
    /// Produces the companion XML descriptor that Deep Zoom viewers need in
    /// order to discover tile parameters. Returns `None` if the plan's layout
    /// is not [`Layout::DeepZoom`], since other layouts have no manifest.
    ///
    /// # Example usage
    ///
    /// * [pyramid_fs_sink tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pyramid_fs_sink.rs)
    pub fn dzi_manifest(&self, format: &str) -> Option<String> {
        if self.layout != Layout::DeepZoom {
            return None;
        }
        Some(format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <Image xmlns=\"http://schemas.microsoft.com/deepzoom/2008\"\n\
             \x20 Format=\"{format}\"\n\
             \x20 Overlap=\"{overlap}\"\n\
             \x20 TileSize=\"{tile_size}\">\n\
             \x20 <Size Width=\"{width}\" Height=\"{height}\"/>\n\
             </Image>",
            overlap = self.overlap,
            tile_size = self.tile_size,
            width = self.image_width,
            height = self.image_height,
        ))
    }

    /// Canvas dimensions at the given level for Google layout.
    /// For non-Google layouts, returns the level's image dimensions.
    pub fn canvas_size_at_level(&self, level: u32) -> (u32, u32) {
        if self.layout == Layout::Google {
            let ts = self.tile_size;
            let grid = 1u32 << level;
            (ts * grid, ts * grid)
        } else {
            match self.levels.get(level as usize) {
                Some(lp) => (lp.width, lp.height),
                None => (0, 0),
            }
        }
    }

    /// Centre offset scaled to the given level.
    pub fn centre_offset_at_level(&self, level: u32) -> (u32, u32) {
        if !self.centre || (self.centre_offset_x == 0 && self.centre_offset_y == 0) {
            return (0, 0);
        }
        let top_level = self.levels.len() as u32 - 1;
        let shift = top_level - level;
        (self.centre_offset_x >> shift, self.centre_offset_y >> shift)
    }

    /// Estimate peak memory for the monolithic engine with a given pixel format.
    ///
    /// The monolithic engine holds the full canvas raster while simultaneously
    /// building the first downscaled level (half width × half height). The peak
    /// occurs when both coexist momentarily:
    ///
    /// `peak = canvas_bytes + canvas_bytes / 4`
    ///
    /// This estimate is conservative — it ignores smaller intermediate buffers
    /// that are freed quickly. Used by [`generate_pyramid_auto`](crate::streaming::generate_pyramid_auto)
    /// to decide whether the monolithic path fits within the memory budget.
    pub fn estimate_peak_memory_for_format(&self, format: crate::pixel::PixelFormat) -> u64 {
        let bpp = format.bytes_per_pixel() as u64;
        let canvas_bytes = self.canvas_width as u64 * self.canvas_height as u64 * bpp;
        // Peak = canvas + first downscaled level (1/4 of canvas)
        canvas_bytes + canvas_bytes / 4
    }

    /// Estimate peak memory for the streaming engine at a given strip height.
    ///
    /// The streaming engine holds multiple live buffers simultaneously:
    ///
    /// - At the top level: the current strip (`canvas_w × strip_h × bpp`)
    ///   plus the strip-pairing accumulator (same size, worst case).
    /// - At each subsequent level: the downscaled strip (`w/2 × h/2`) plus
    ///   its own pairing accumulator.
    ///
    /// This function walks the geometric series of halving dimensions until
    /// the level collapses to a single pixel, summing `2 × strip_bytes` per
    /// level (strip + accumulator) plus the final small-level buffer. A 10%
    /// safety margin is added for bookkeeping overhead.
    ///
    /// Used by [`compute_strip_height`](crate::streaming::compute_strip_height)
    /// to find the tallest strip that fits within the caller's budget.
    pub fn estimate_streaming_peak_memory(
        &self,
        format: crate::pixel::PixelFormat,
        strip_height: u32,
    ) -> u64 {
        let bpp = format.bytes_per_pixel() as u64;
        let cw = self.canvas_width as u64;
        let mut total: u64 = 0;
        let mut w = cw;
        let mut h = strip_height as u64;

        // Walk levels: each level holds the current strip plus its pairing
        // accumulator (waiting for the partner half-strip)
        loop {
            let strip_bytes = w * h * bpp;
            total += strip_bytes * 2;
            // Downscale dimensions for next level (matching downscale_half's
            // div_ceil behavior for odd sizes)
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            if h <= 1 || w <= 1 {
                // Final small level — only the strip itself, no accumulator
                total += w * h * bpp;
                break;
            }
        }
        // 10% safety margin for Vec overhead, alignment, and temporary buffers
        total + total / 10
    }
}

impl LevelPlan {
    /// Total tiles at this level (`cols * rows`).
    ///
    /// Convenience method for progress tracking or capacity pre-allocation
    /// when processing a single level at a time.
    pub fn tile_count(&self) -> u64 {
        self.cols as u64 * self.rows as u64
    }
}

impl TileCoord {
    /// Create a new `TileCoord` from explicit level, column, and row indices.
    pub fn new(level: u32, col: u32, row: u32) -> Self {
        Self { level, col, row }
    }
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /**
     * Tests that images with zero width or height are rejected.
     * Works by attempting to create a planner with width=0 and height=0 separately.
     * Input: 0x100 or 100x0 image → Output: Err(ZeroDimension).
     */
    #[test]
    fn zero_dimension_rejected() {
        assert!(PyramidPlanner::new(0, 100, 256, 0, Layout::DeepZoom).is_err());
        assert!(PyramidPlanner::new(100, 0, 256, 0, Layout::DeepZoom).is_err());
    }

    /**
     * Tests that a tile size of zero is rejected.
     * Works by passing tile_size=0, which would cause division by zero in grid calculations.
     * Input: 100x100 image, tile_size=0 → Output: Err(ZeroTileSize).
     */
    #[test]
    fn zero_tile_size_rejected() {
        assert!(PyramidPlanner::new(100, 100, 0, 0, Layout::DeepZoom).is_err());
    }

    /**
     * Tests that overlap must be strictly less than tile_size.
     * Works by checking overlap == tile_size (invalid), overlap > tile_size (invalid),
     * and overlap == tile_size - 1 (valid boundary case).
     * Input: overlap=256 with tile_size=256 → Err; overlap=255 → Ok.
     */
    #[test]
    fn overlap_must_be_less_than_tile_size() {
        assert!(PyramidPlanner::new(100, 100, 256, 256, Layout::DeepZoom).is_err());
        assert!(PyramidPlanner::new(100, 100, 256, 300, Layout::DeepZoom).is_err());
        assert!(PyramidPlanner::new(100, 100, 256, 255, Layout::DeepZoom).is_ok());
    }

    /**
     * Tests the degenerate case of a 1x1 pixel image.
     * Works by verifying the pyramid has exactly one level with one 1x1 tile.
     * Input: 1x1 image → Output: 1 level, 1 tile, dimensions 1x1.
     */
    #[test]
    fn single_pixel_image() {
        let planner = PyramidPlanner::new(1, 1, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        assert_eq!(plan.level_count(), 1);
        assert_eq!(plan.levels[0].width, 1);
        assert_eq!(plan.levels[0].height, 1);
        assert_eq!(plan.levels[0].cols, 1);
        assert_eq!(plan.levels[0].rows, 1);
        assert_eq!(plan.total_tile_count(), 1);
    }

    /**
     * Tests that an image smaller than tile_size produces a 1x1 tile grid at full resolution.
     * Works by creating a 100x80 image with tile_size=256 and checking the top level grid.
     * Input: 100x80, tile=256 → Output: top level has cols=1, rows=1.
     */
    #[test]
    fn image_smaller_than_tile() {
        let planner = PyramidPlanner::new(100, 80, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        // Full resolution level should have 1x1 tile grid
        let top = plan.levels.last().unwrap();
        assert_eq!(top.width, 100);
        assert_eq!(top.height, 80);
        assert_eq!(top.cols, 1);
        assert_eq!(top.rows, 1);
    }

    /**
     * Tests that an image exactly divisible by tile_size produces the correct grid.
     * Works by using a 512x512 image with tile_size=256, expecting a clean 2x2 grid.
     * Input: 512x512, tile=256 → Output: top level has cols=2, rows=2.
     */
    #[test]
    fn image_exactly_n_tiles() {
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert_eq!(top.width, 512);
        assert_eq!(top.height, 512);
        assert_eq!(top.cols, 2);
        assert_eq!(top.rows, 2);
    }

    /**
     * Tests that non-divisible dimensions use ceiling division for tile counts.
     * Works by using a 500x300 image where ceil(500/256)=2 and ceil(300/256)=2.
     * Input: 500x300, tile=256 → Output: cols=2, rows=2.
     */
    #[test]
    fn image_not_multiple_of_tile() {
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert_eq!(top.width, 500);
        assert_eq!(top.height, 300);
        assert_eq!(top.cols, 2); // ceil(500/256)
        assert_eq!(top.rows, 2); // ceil(300/256)
    }

    /**
     * Tests that non-square (wide) images compute asymmetric tile grids correctly.
     * Works by using a 1000x200 image where width needs 4 columns but height needs only 1 row.
     * Input: 1000x200, tile=256 → Output: cols=4, rows=1.
     */
    #[test]
    fn non_square_image() {
        let planner = PyramidPlanner::new(1000, 200, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert_eq!(top.width, 1000);
        assert_eq!(top.height, 200);
        assert_eq!(top.cols, 4);
        assert_eq!(top.rows, 1);
    }

    /**
     * Tests that each pyramid level's dimensions are ceil(prev/2) of the level above.
     * Works by iterating all adjacent level pairs and verifying the halving relationship,
     * plus checking that the bottom level is 1x1 and the top is the original size.
     * Input: 1024x768 → Output: top=1024x768, bottom=1x1, each level halves.
     */
    #[test]
    fn level_dimensions_halve_correctly() {
        let planner = PyramidPlanner::new(1024, 768, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();

        // Top level = full resolution
        let top = plan.levels.last().unwrap();
        assert_eq!(top.width, 1024);
        assert_eq!(top.height, 768);

        // Each level below should be ceil(prev/2)
        for i in (1..plan.levels.len()).rev() {
            let upper = &plan.levels[i];
            let lower = &plan.levels[i - 1];
            assert_eq!(lower.width, ceil_div(upper.width, 2));
            assert_eq!(lower.height, ceil_div(upper.height, 2));
        }

        // Bottom level
        assert_eq!(plan.levels[0].width, 1);
        assert_eq!(plan.levels[0].height, 1);
    }

    /**
     * Tests that level indices are assigned sequentially starting from 0.
     * Works by iterating all levels and comparing each level's index to its position.
     * Input: 2048x2048 pyramid → Output: levels[i].level == i for all i.
     */
    #[test]
    fn level_indices_are_sequential() {
        let planner = PyramidPlanner::new(2048, 2048, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        for (i, level) in plan.levels.iter().enumerate() {
            assert_eq!(level.level, i as u32);
        }
    }

    /**
     * Tests that total_tile_count() equals the manual sum of cols*rows across all levels.
     * Works by computing the sum independently and comparing to the method's result.
     * Input: 512x512, tile=256 → Output: both sums match.
     */
    #[test]
    fn total_tile_count_sums_all_levels() {
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let manual_count: u64 = plan
            .levels
            .iter()
            .map(|l| l.cols as u64 * l.rows as u64)
            .sum();
        assert_eq!(plan.total_tile_count(), manual_count);
    }

    /**
     * Tests that tile_coords() iterator yields exactly total_tile_count() items.
     * Works by counting the iterator output and comparing to the total_tile_count() method.
     * Input: 800x600, tile=256 → Output: iterator count == total_tile_count().
     */
    #[test]
    fn tile_coords_count_matches_total() {
        let planner = PyramidPlanner::new(800, 600, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let coord_count = plan.tile_coords().count() as u64;
        assert_eq!(coord_count, plan.total_tile_count());
    }

    /**
     * Tests tile_rect without overlap returns correct pixel rectangles.
     * Works by checking the origin tile (0,0), a middle tile (1,0), and the
     * bottom-right edge tile which should be clipped to the image boundary.
     * Input: 600x400, tile=256, tile(2,1) → Output: rect clipped to (512,256)-(600,400).
     */
    #[test]
    fn tile_rect_no_overlap() {
        let planner = PyramidPlanner::new(600, 400, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        // Tile (0,0) — top-left
        let r = plan.tile_rect(TileCoord::new(top.level, 0, 0)).unwrap();
        assert_eq!(r.x, 0);
        assert_eq!(r.y, 0);
        assert_eq!(r.width, 256);
        assert_eq!(r.height, 256);

        // Tile (1,0) — middle column
        let r = plan.tile_rect(TileCoord::new(top.level, 1, 0)).unwrap();
        assert_eq!(r.x, 256);
        assert_eq!(r.y, 0);
        assert_eq!(r.width, 256);
        assert_eq!(r.height, 256);

        // Tile (2,1) — bottom-right, clipped to image boundary
        // cols=ceil(600/256)=3, rows=ceil(400/256)=2
        let r = plan.tile_rect(TileCoord::new(top.level, 2, 1)).unwrap();
        assert_eq!(r.x, 512);
        assert_eq!(r.y, 256);
        assert_eq!(r.width, 600 - 512); // 88
        assert_eq!(r.height, 400 - 256); // 144
    }

    /**
     * Tests tile_rect with overlap=1 adds extra pixels to tile boundaries.
     * Works by verifying tile (0,0) gets overlap on right/bottom only, and
     * tile (1,0) gets overlap on both left and right sides.
     * Input: 600x400, overlap=1, tile(0,0) → width=257; tile(1,0) → x=255.
     */
    #[test]
    fn tile_rect_with_overlap() {
        let planner = PyramidPlanner::new(600, 400, 256, 1, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        // Tile (0,0) — no left/top overlap, right/bottom overlap = 1
        let r = plan.tile_rect(TileCoord::new(top.level, 0, 0)).unwrap();
        assert_eq!(r.x, 0);
        assert_eq!(r.y, 0);
        assert_eq!(r.width, 256 + 1); // tile_size + overlap
        assert_eq!(r.height, 256 + 1);

        // Tile (1,0) — has left overlap
        let r = plan.tile_rect(TileCoord::new(top.level, 1, 0)).unwrap();
        assert_eq!(r.x, 256 - 1); // tile_size - overlap
        assert_eq!(r.width, 258); // 256 + 2 (tile_size + 2*overlap)
    }

    /**
     * Tests that tile_rect returns None for out-of-bounds coordinates.
     * Works by requesting tiles beyond the grid dimensions and an invalid level index.
     * Input: 256x256 (1x1 grid), tile(1,0) or level=999 → Output: None.
     */
    #[test]
    fn tile_rect_out_of_bounds_returns_none() {
        let planner = PyramidPlanner::new(256, 256, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert!(plan.tile_rect(TileCoord::new(top.level, 1, 0)).is_none());
        assert!(plan.tile_rect(TileCoord::new(top.level, 0, 1)).is_none());
        assert!(plan.tile_rect(TileCoord::new(999, 0, 0)).is_none());
    }

    /**
     * Tests that tiles with no overlap cover every pixel exactly once at the top level.
     * Works by building a coverage map over all pixels and incrementing for each tile rect,
     * then asserting every pixel has count == 1 (no gaps, no overlaps).
     * Input: 500x300, tile=256, overlap=0 → Output: all pixels covered exactly once.
     */
    #[test]
    fn tile_rects_cover_full_image_no_overlap() {
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();

        // Verify every pixel is covered by exactly one tile
        let mut coverage = vec![0u32; top.width as usize * top.height as usize];

        for row in 0..top.rows {
            for col in 0..top.cols {
                let r = plan.tile_rect(TileCoord::new(top.level, col, row)).unwrap();
                for y in r.y..r.y + r.height {
                    for x in r.x..r.x + r.width {
                        coverage[y as usize * top.width as usize + x as usize] += 1;
                    }
                }
            }
        }

        for (i, &count) in coverage.iter().enumerate() {
            assert_eq!(
                count,
                1,
                "Pixel ({}, {}) covered {} times",
                i % top.width as usize,
                i / top.width as usize,
                count
            );
        }
    }

    /**
     * Tests Deep Zoom tile path format: "{level}/{col}_{row}.{ext}".
     * Works by generating a path for the top-level origin tile and comparing to expected format.
     * Input: tile(top_level, 0, 0), ext="jpeg" → Output: "{level}/0_0.jpeg".
     */
    #[test]
    fn deep_zoom_tile_path() {
        let planner = PyramidPlanner::new(256, 256, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        let path = plan
            .tile_path(TileCoord::new(top.level, 0, 0), "jpeg")
            .unwrap();
        assert_eq!(path, format!("{}/0_0.jpeg", top.level));
    }

    /**
     * Tests XYZ/slippy-map tile path format: "{level}/{col}/{row}.{ext}".
     * Works by generating a path for a specific tile coordinate and verifying the format.
     * Input: tile(top_level, 3, 5), ext="png" → Output: "{level}/3/5.png".
     */
    #[test]
    fn xyz_tile_path() {
        let planner = PyramidPlanner::new(2048, 2048, 256, 0, Layout::Xyz).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        let path = plan
            .tile_path(TileCoord::new(top.level, 3, 5), "png")
            .unwrap();
        assert_eq!(path, format!("{}/3/5.png", top.level));
    }

    /**
     * Tests that tile_path returns None for out-of-bounds level indices.
     * Works by requesting a path for a non-existent level (999).
     * Input: level=999, col=0, row=0 → Output: None.
     */
    #[test]
    fn tile_path_out_of_bounds_returns_none() {
        let planner = PyramidPlanner::new(256, 256, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        assert!(plan.tile_path(TileCoord::new(999, 0, 0), "png").is_none());
    }

    /**
     * Tests that dzi_manifest() produces valid XML with the correct attributes.
     * Works by checking that the manifest string contains the expected Format,
     * Overlap, TileSize, Width, and Height attribute values.
     * Input: 1024x768, tile=256, overlap=1, format="jpeg" → Output: XML with matching attrs.
     */
    #[test]
    fn dzi_manifest_structure() {
        let planner = PyramidPlanner::new(1024, 768, 256, 1, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        let manifest = plan.dzi_manifest("jpeg").unwrap();

        assert!(manifest.contains("Format=\"jpeg\""));
        assert!(manifest.contains("Overlap=\"1\""));
        assert!(manifest.contains("TileSize=\"256\""));
        assert!(manifest.contains("Width=\"1024\""));
        assert!(manifest.contains("Height=\"768\""));
    }

    /**
     * Tests that dzi_manifest() returns None for non-DeepZoom layouts.
     * Works by creating an Xyz layout plan and verifying the manifest is unavailable.
     * Input: Layout::Xyz → Output: None.
     */
    #[test]
    fn dzi_manifest_returns_none_for_xyz() {
        let planner = PyramidPlanner::new(256, 256, 256, 0, Layout::Xyz).unwrap();
        let plan = planner.plan();
        assert!(plan.dzi_manifest("png").is_none());
    }

    /**
     * Tests that calling plan() twice on the same planner yields identical results.
     * Works by comparing two plans via PartialEq, ensuring no hidden randomness.
     * Input: same planner called twice → Output: plan_a == plan_b.
     */
    #[test]
    fn plan_is_deterministic() {
        let planner = PyramidPlanner::new(4000, 3000, 256, 1, Layout::DeepZoom).unwrap();
        let plan_a = planner.plan();
        let plan_b = planner.plan();
        assert_eq!(plan_a, plan_b);
    }

    /**
     * Tests that very large images (50000x50000) do not panic or overflow.
     * Works by verifying the level count is in the expected range (~16-17 levels)
     * and the top level preserves the original dimensions.
     * Input: 50000x50000 → Output: 16-18 levels, top level = 50000x50000.
     */
    #[test]
    fn large_image_level_count() {
        // 50000x50000 — should not panic or overflow
        let planner = PyramidPlanner::new(50_000, 50_000, 256, 0, Layout::DeepZoom).unwrap();
        let plan = planner.plan();
        // log2(50000) ≈ 15.6, so ~16 levels + level 0
        assert!(plan.level_count() >= 16);
        assert!(plan.level_count() <= 18);
        let top = plan.levels.last().unwrap();
        assert_eq!(top.width, 50_000);
        assert_eq!(top.height, 50_000);
    }

    /**
     * Tests that varying tile sizes all produce the correct grid dimensions.
     * Works by iterating over tile sizes [64, 128, 256, 512, 1024] and checking
     * that cols and rows equal ceil(image_dim / tile_size) for each.
     * Input: 2048x1536 with various tile sizes → Output: cols/rows match ceil division.
     */
    #[test]
    fn different_tile_sizes() {
        for tile_size in [64, 128, 256, 512, 1024] {
            let planner = PyramidPlanner::new(2048, 1536, tile_size, 0, Layout::DeepZoom).unwrap();
            let plan = planner.plan();
            let top = plan.levels.last().unwrap();
            assert_eq!(top.cols, ceil_div(2048, tile_size));
            assert_eq!(top.rows, ceil_div(1536, tile_size));
        }
    }

    // -- Google layout tests --

    #[test]
    fn google_layout_single_tile() {
        // Image smaller than tile_size → 1 level, 1×1 grid
        let planner = PyramidPlanner::new(100, 80, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        assert_eq!(plan.level_count(), 1);
        assert_eq!(plan.levels[0].cols, 1);
        assert_eq!(plan.levels[0].rows, 1);
        assert_eq!(plan.canvas_width, 256);
        assert_eq!(plan.canvas_height, 256);
    }

    #[test]
    fn google_layout_level_count() {
        // 3300x5024 at ts=256: max(ceil(3300/256), ceil(5024/256)) = max(13,20) = 20
        // n_levels = ceil(log2(20)) + 1 = 5 + 1 = 6
        let planner = PyramidPlanner::new(3300, 5024, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        assert_eq!(plan.level_count(), 6);
        // Canvas = 256 * 2^5 = 8192
        assert_eq!(plan.canvas_width, 8192);
        assert_eq!(plan.canvas_height, 8192);
    }

    #[test]
    fn google_layout_power_of_2_grids() {
        let planner = PyramidPlanner::new(1000, 800, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        for (i, level) in plan.levels.iter().enumerate() {
            let expected_grid = 1u32 << i;
            assert_eq!(level.cols, expected_grid, "Level {} cols", i);
            assert_eq!(level.rows, expected_grid, "Level {} rows", i);
        }
    }

    #[test]
    fn google_layout_path_format() {
        // Google layout uses {z}/{y}/{x}.ext (row before col, matching vips convention)
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        // TileCoord(level=1, col=1, row=0) → path "1/0/1.png" (row=0, col=1)
        let path = plan.tile_path(TileCoord::new(1, 1, 0), "png").unwrap();
        assert_eq!(path, "1/0/1.png");
    }

    #[test]
    fn google_layout_no_dzi() {
        let planner = PyramidPlanner::new(512, 512, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        assert!(plan.dzi_manifest("png").is_none());
    }

    #[test]
    fn google_layout_tile_rect_full_size() {
        // All Google tiles should be full tile_size × tile_size
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        for coord in plan.tile_coords() {
            let rect = plan.tile_rect(coord).unwrap();
            assert_eq!(rect.width, 256, "Tile {:?} width", coord);
            assert_eq!(rect.height, 256, "Tile {:?} height", coord);
        }
    }

    #[test]
    fn google_centre_offsets() {
        // 500x300 at ts=256: max(2,2)=2, n_levels=2, canvas=512
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::Google)
            .unwrap()
            .with_centre(true);
        let plan = planner.plan();
        assert_eq!(plan.canvas_width, 512);
        assert_eq!(plan.canvas_height, 512);
        assert_eq!(plan.centre_offset_x, (512 - 500) / 2); // 6
        assert_eq!(plan.centre_offset_y, (512 - 300) / 2); // 106
    }

    #[test]
    fn google_no_centre_offsets_zero() {
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        assert_eq!(plan.centre_offset_x, 0);
        assert_eq!(plan.centre_offset_y, 0);
        assert!(!plan.centre);
    }

    #[test]
    fn google_layout_image_dims_halve() {
        // Level width/height should represent the image, not canvas
        let planner = PyramidPlanner::new(1000, 800, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        let top = plan.levels.last().unwrap();
        assert_eq!(top.width, 1000);
        assert_eq!(top.height, 800);

        // Each lower level halves
        for i in (1..plan.levels.len()).rev() {
            let upper = &plan.levels[i];
            let lower = &plan.levels[i - 1];
            assert_eq!(lower.width, ceil_div(upper.width, 2));
            assert_eq!(lower.height, ceil_div(upper.height, 2));
        }
    }

    #[test]
    fn google_total_tile_count() {
        // 2 levels: 1×1 + 2×2 = 5
        let planner = PyramidPlanner::new(300, 300, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        assert_eq!(plan.level_count(), 2);
        assert_eq!(plan.total_tile_count(), 1 + 4);
    }

    #[test]
    fn centre_with_deep_zoom() {
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::DeepZoom)
            .unwrap()
            .with_centre(true);
        let plan = planner.plan();
        // Grid for 500x300 at ts=256: 2x2, canvas = 512x512
        let top = plan.levels.last().unwrap();
        assert_eq!(top.cols, 2);
        assert_eq!(top.rows, 2);
        assert_eq!(plan.canvas_width, 512);
        assert_eq!(plan.canvas_height, 512);
        assert_eq!(plan.centre_offset_x, 6);
        assert_eq!(plan.centre_offset_y, 106);
    }

    #[test]
    fn canvas_size_at_level_google() {
        let planner = PyramidPlanner::new(500, 300, 256, 0, Layout::Google).unwrap();
        let plan = planner.plan();
        assert_eq!(plan.canvas_size_at_level(0), (256, 256));
        assert_eq!(plan.canvas_size_at_level(1), (512, 512));
    }

    #[test]
    fn centre_offset_at_level_scales() {
        // 3 levels for Google: 500x300 at ts=128
        // max(ceil(500/128), ceil(300/128)) = max(4,3) = 4
        // n_levels = ceil(log2(4)) + 1 = 2 + 1 = 3
        // canvas = 128 * 4 = 512
        let planner = PyramidPlanner::new(500, 300, 128, 0, Layout::Google)
            .unwrap()
            .with_centre(true);
        let plan = planner.plan();
        assert_eq!(plan.level_count(), 3);
        assert_eq!(plan.canvas_width, 512);
        // offset_x = (512-500)/2 = 6, offset_y = (512-300)/2 = 106
        let (ox_top, oy_top) = plan.centre_offset_at_level(2);
        assert_eq!(ox_top, 6);
        assert_eq!(oy_top, 106);
        // Level 1: offset halved
        let (ox1, oy1) = plan.centre_offset_at_level(1);
        assert_eq!(ox1, 3);
        assert_eq!(oy1, 53);
        // Level 0: offset quartered
        let (ox0, oy0) = plan.centre_offset_at_level(0);
        assert_eq!(ox0, 1);
        assert_eq!(oy0, 26);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            .. ProptestConfig::default()
        })]
        // Property test: tiles with no overlap cover every pixel at every level exactly once.
        // Works by checking that the bounding box of all tile rects spans (0,0) to (width,height)
        // and the total tile area equals the level's pixel area (no gaps or overlaps).
        // Input: random w in 1..2048, h in 1..2048, tile_size in 1..512.
        #[test]
        fn tile_coverage_no_overlap(
            w in 1u32..2048,
            h in 1u32..2048,
            tile_size in 1u32..512,
        ) {
            let planner = PyramidPlanner::new(w, h, tile_size, 0, Layout::DeepZoom).unwrap();
            let plan = planner.plan();

            for level in &plan.levels {
                // Every pixel at this level must be covered exactly once
                let mut min_x_covered = level.width;
                let mut min_y_covered = level.height;
                let mut max_x_covered = 0u32;
                let mut max_y_covered = 0u32;
                let mut total_area = 0u64;

                for row in 0..level.rows {
                    for col in 0..level.cols {
                        let r = plan.tile_rect(TileCoord::new(level.level, col, row)).unwrap();
                        min_x_covered = min_x_covered.min(r.x);
                        min_y_covered = min_y_covered.min(r.y);
                        max_x_covered = max_x_covered.max(r.x + r.width);
                        max_y_covered = max_y_covered.max(r.y + r.height);
                        total_area += r.width as u64 * r.height as u64;
                    }
                }

                // Coverage starts at (0,0)
                prop_assert_eq!(min_x_covered, 0, "Level {} x gap at start", level.level);
                prop_assert_eq!(min_y_covered, 0, "Level {} y gap at start", level.level);
                // Coverage reaches image boundary
                prop_assert_eq!(max_x_covered, level.width, "Level {} x short", level.level);
                prop_assert_eq!(max_y_covered, level.height, "Level {} y short", level.level);
                // No overlap means total area == image area
                prop_assert_eq!(
                    total_area,
                    level.width as u64 * level.height as u64,
                    "Level {} area mismatch (overlap/gap)",
                    level.level,
                );
            }
        }

        // Property test: each level's dimensions are exactly ceil(upper/2) of the level above,
        // the bottom level is always 1x1, and the top level matches the original dimensions.
        // Input: random w in 2..10000, h in 2..10000.
        #[test]
        fn level_halving_invariant(w in 2u32..10000, h in 2u32..10000) {
            let planner = PyramidPlanner::new(w, h, 256, 0, Layout::DeepZoom).unwrap();
            let plan = planner.plan();

            for i in 1..plan.levels.len() {
                let upper = &plan.levels[i];
                let lower = &plan.levels[i - 1];
                prop_assert_eq!(lower.width, ceil_div(upper.width, 2));
                prop_assert_eq!(lower.height, ceil_div(upper.height, 2));
            }

            // Bottom level is 1x1
            prop_assert_eq!(plan.levels[0].width, 1);
            prop_assert_eq!(plan.levels[0].height, 1);
            // Top level is original
            let top = plan.levels.last().unwrap();
            prop_assert_eq!(top.width, w);
            prop_assert_eq!(top.height, h);
        }

        // Property test: calling plan() twice always produces identical results.
        // Works by comparing two plans via PartialEq for random dimensions and tile sizes.
        // Input: random w in 1..5000, h in 1..5000, tile_size in 1..512.
        #[test]
        fn plan_determinism(
            w in 1u32..5000,
            h in 1u32..5000,
            tile_size in 1u32..512,
        ) {
            let planner = PyramidPlanner::new(w, h, tile_size, 0, Layout::DeepZoom).unwrap();
            let a = planner.plan();
            let b = planner.plan();
            prop_assert_eq!(a, b);
        }
    }
}
