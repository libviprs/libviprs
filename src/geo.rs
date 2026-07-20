//! Geo-coordinate mapping for tile pyramids.
//!
//! Maps pixel coordinates ↔ geographic coordinates using an affine transform.
//! This supports the common case of geo-referenced blueprints and plans where
//! the relationship between pixel space and geographic space is a linear
//! (affine) mapping — rotation, scale, skew, and translation.
//!
//! For most construction/AEC use cases, an affine transform is sufficient
//! because the area covered by a single plan sheet is small enough that
//! Earth curvature effects are negligible.

/// A 2D point in pixel space (column, row from top-left origin).
///
/// Represents a position within a raster image where `x` is the column offset
/// (increasing rightward) and `y` is the row offset (increasing downward).
/// Coordinates are `f64` to allow sub-pixel precision during interpolation
/// and affine transform calculations.
///
/// This type is the pixel-space counterpart of [`GeoCoord`] and is used
/// throughout the geo-referencing pipeline to express positions before they
/// are projected into geographic space via a [`GeoTransform`].
///
/// # Example usage
///
/// - [pdf_to_georeferenced_pyramid_memory](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
///   constructs `PixelCoord` values when verifying geo bounds for tile centers.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-geo-origin)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelCoord {
    pub x: f64,
    pub y: f64,
}

/// A 2D point in geographic space (longitude, latitude or easting, northing).
///
/// Represents a position in a geographic or projected coordinate system.
/// The interpretation of `x` and `y` depends on the coordinate reference
/// system in use -- for WGS-84 they are longitude and latitude; for a local
/// site coordinate system they may be easting and northing in metres.
///
/// `GeoCoord` is intentionally unit-agnostic so that the same affine
/// transform machinery works for both real-world map projections and
/// arbitrary plan-sheet coordinate systems common in AEC workflows.
///
/// # Example usage
///
/// - [pdf_to_georeferenced_pyramid_memory](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
///   creates `GeoCoord` origins to geo-reference a PDF-extracted raster.
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   parses `--geo-origin` flag values into `GeoCoord`.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-geo-origin)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCoord {
    pub x: f64,
    pub y: f64,
}

/// An affine transform mapping pixel coordinates to geographic coordinates.
///
/// The transform is:
/// ```text
/// geo_x = a * pixel_x + b * pixel_y + c
/// geo_y = d * pixel_x + e * pixel_y + f
/// ```
///
/// This is equivalent to a GDAL-style GeoTransform but stored row-major:
/// `[a, b, c, d, e, f]`.
///
/// # Example usage
///
/// - [pdf_to_georeferenced_pyramid_memory](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
///   attaches a `GeoTransform` to a PDF-extracted raster and verifies tile
///   centers fall within the expected geographic bounds.
/// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///   builds a `GeoTransform` from the `--geo-origin` and `--geo-scale` flags.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-geo-origin)
/// (and [`--geo-scale`](https://libviprs.org/cli/#flag-geo-scale))
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoTransform {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

/// Relative tolerance used for the affine singularity / collinearity tests.
///
/// The determinant of a 2x2 linear part (and the twice-signed-area determinant
/// of a triangle of ground control points) scales with the *square* of the
/// coefficient / coordinate magnitudes. An absolute threshold therefore
/// falsely rejects legitimately small-scale transforms and falsely accepts
/// near-degenerate large-coordinate ones. We instead compare `|det|` against
/// this factor times the squared matrix / edge norm — a scale-invariant
/// relative conditioning check. The value sits a few orders of magnitude above
/// `f64::EPSILON` (~2.2e-16) to absorb accumulated rounding while still
/// accepting well-conditioned transforms.
const DET_REL_EPS: f64 = 1e-12;

impl GeoTransform {
    /// Reciprocal determinant of the 2x2 linear part, or `None` if the
    /// transform is non-invertible.
    ///
    /// Rejects the transform when any linear coefficient is non-finite (so the
    /// singularity test never operates on `NaN`/`inf`, where comparisons
    /// silently misbehave and would otherwise yield a garbage inverse), and
    /// when `|det|` is small *relative to the squared coefficient norm* rather
    /// than against an absolute epsilon.
    fn linear_inv_det(&self) -> Option<f64> {
        if !(self.a.is_finite() && self.b.is_finite() && self.d.is_finite() && self.e.is_finite()) {
            return None;
        }
        let det = self.a * self.e - self.b * self.d;
        debug_assert!(
            det.is_finite(),
            "affine determinant must be finite for finite coefficients"
        );
        // Norm-relative singularity test: |det| scales as ||M||^2.
        let norm_sq = self.a * self.a + self.b * self.b + self.d * self.d + self.e * self.e;
        if !det.is_finite() || det.abs() <= DET_REL_EPS * norm_sq {
            return None;
        }
        Some(1.0 / det)
    }

    /// Create a transform from the 6 affine coefficients.
    ///
    /// The coefficients `(a, b, c, d, e, f)` correspond to the row-major
    /// representation of the 2x3 affine matrix. Use this constructor when
    /// you already have raw coefficients (e.g. read from a world file or
    /// GDAL dataset). For the common no-rotation case, prefer
    /// [`from_origin_and_scale`](Self::from_origin_and_scale).
    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self { a, b, c, d, e, f }
    }

    /// Create a simple scale + translate transform (no rotation or skew).
    ///
    /// This is the most common constructor for AEC plan sheets where the
    /// image is axis-aligned and the relationship between pixels and
    /// geographic units is a uniform scale plus an offset.
    ///
    /// - `origin`: geographic coordinate of the top-left pixel
    /// - `pixel_size_x`: geographic units per pixel in X (typically positive)
    /// - `pixel_size_y`: geographic units per pixel in Y (typically negative
    ///   for top-down rasters)
    ///
    /// # Example usage
    ///
    /// - [pdf_to_georeferenced_pyramid_memory](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
    ///   uses this method to attach a geo-reference to a rasterized PDF page.
    /// - [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
    ///   calls this via the `--geo-origin` / `--geo-scale` flags.
    pub fn from_origin_and_scale(origin: GeoCoord, pixel_size_x: f64, pixel_size_y: f64) -> Self {
        Self {
            a: pixel_size_x,
            b: 0.0,
            c: origin.x,
            d: 0.0,
            e: pixel_size_y,
            f: origin.y,
        }
    }

    /// Create a transform from ground control points (GCPs).
    ///
    /// Solves the 6 affine parameters exactly from 3 non-collinear GCPs by
    /// inverting the 3x3 system of equations. Returns `None` if the pixel
    /// positions are collinear or coincident — measured *relative* to the
    /// spacing of the points, so both tightly-clustered and far-from-origin
    /// point sets are judged correctly — or if any input coordinate is
    /// non-finite.
    ///
    /// For over-determined systems (more than 3 GCPs), use
    /// `from_gcps_least_squares`.
    ///
    /// # Example usage
    ///
    /// - [pdf_to_georeferenced_pyramid_memory](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
    ///   demonstrates the equivalent `from_origin_and_scale` path; GCPs are
    ///   exercised in the unit tests of this module.
    pub fn from_gcps_exact(gcps: &[(PixelCoord, GeoCoord); 3]) -> Option<Self> {
        let [(p0, g0), (p1, g1), (p2, g2)] = gcps;

        // Solve: [a b c] * [[px0 px1 px2], [py0 py1 py2], [1 1 1]] = [gx0 gx1 gx2]
        // Same for d, e, f with gy values.
        //
        // The system determinant equals the cross product of the triangle's
        // edge vectors (twice its signed area). Test collinearity relative to
        // the edge lengths, so tightly-spaced points or clusters far from the
        // origin are judged by their actual geometry rather than an absolute
        // epsilon. A non-finite `det` (from non-finite input coordinates) is
        // also rejected here.
        let e1x = p1.x - p0.x;
        let e1y = p1.y - p0.y;
        let e2x = p2.x - p0.x;
        let e2y = p2.y - p0.y;
        let det = e1x * e2y - e1y * e2x;
        let norm_sq = (e1x * e1x + e1y * e1y).max(e2x * e2x + e2y * e2y);
        if !det.is_finite() || det.abs() <= DET_REL_EPS * norm_sq {
            return None; // Points are collinear, coincident, or non-finite
        }

        let inv_det = 1.0 / det;

        // Solve for a, b, c (geo_x coefficients)
        let a = (g0.x * (p1.y - p2.y) - g1.x * (p0.y - p2.y) + g2.x * (p0.y - p1.y)) * inv_det;
        let b = (p0.x * (g1.x - g2.x) - p1.x * (g0.x - g2.x) + p2.x * (g0.x - g1.x)) * inv_det;
        let c = (p0.x * (p1.y * g2.x - p2.y * g1.x) - p1.x * (p0.y * g2.x - p2.y * g0.x)
            + p2.x * (p0.y * g1.x - p1.y * g0.x))
            * inv_det;

        // Solve for d, e, f (geo_y coefficients)
        let d = (g0.y * (p1.y - p2.y) - g1.y * (p0.y - p2.y) + g2.y * (p0.y - p1.y)) * inv_det;
        let e = (p0.x * (g1.y - g2.y) - p1.x * (g0.y - g2.y) + p2.x * (g0.y - g1.y)) * inv_det;
        let f = (p0.x * (p1.y * g2.y - p2.y * g1.y) - p1.x * (p0.y * g2.y - p2.y * g0.y)
            + p2.x * (p0.y * g1.y - p1.y * g0.y))
            * inv_det;

        // Reject a solution poisoned by non-finite geo coordinates or overflow.
        if !(a.is_finite()
            && b.is_finite()
            && c.is_finite()
            && d.is_finite()
            && e.is_finite()
            && f.is_finite())
        {
            return None;
        }

        Some(Self { a, b, c, d, e, f })
    }

    /// Transform a pixel coordinate to a geographic coordinate.
    ///
    /// Applies the forward affine transform to map a position in raster
    /// pixel space to the corresponding position in geographic space.
    /// This is the primary direction used when labelling tile centres with
    /// their real-world coordinates.
    ///
    /// # Example usage
    ///
    /// - [pdf_to_georeferenced_pyramid_memory](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
    ///   calls `pixel_to_geo` to verify that the image centre falls inside
    ///   the expected geographic bounds.
    pub fn pixel_to_geo(&self, pixel: PixelCoord) -> GeoCoord {
        GeoCoord {
            x: self.a * pixel.x + self.b * pixel.y + self.c,
            y: self.d * pixel.x + self.e * pixel.y + self.f,
        }
    }

    /// Transform a geographic coordinate to a pixel coordinate.
    ///
    /// Computes the inverse of the forward affine to map a geographic
    /// position back to raster pixel space. This is useful for hit-testing
    /// (e.g. "which pixel does this lat/lon correspond to?").
    ///
    /// Returns `None` if the transform is singular (degenerate) — its 2x2
    /// coefficient matrix has a determinant that is near zero *relative to the
    /// matrix norm*, or a non-finite coefficient — and cannot be inverted.
    ///
    /// # Output range
    ///
    /// The returned pixel coordinates are **unclamped**: they may be negative
    /// or exceed the raster's width/height, and are `f64` (sub-pixel). Callers
    /// must range-check and round them before using the result as an array or
    /// tile index.
    pub fn geo_to_pixel(&self, geo: GeoCoord) -> Option<PixelCoord> {
        let inv_det = self.linear_inv_det()?;
        let dx = geo.x - self.c;
        let dy = geo.y - self.f;
        Some(PixelCoord {
            x: (self.e * dx - self.b * dy) * inv_det,
            y: (-self.d * dx + self.a * dy) * inv_det,
        })
    }

    /// Compute the inverse transform (geo -> pixel as a `GeoTransform`).
    ///
    /// Returns a new `GeoTransform` whose forward direction maps geographic
    /// coordinates to pixel coordinates. This is the full-matrix inverse,
    /// unlike [`geo_to_pixel`](Self::geo_to_pixel) which returns a single
    /// point; the inverse transform object can be reused for many lookups.
    ///
    /// Returns `None` if the transform is singular — near-zero determinant
    /// *relative to the matrix norm*, or a non-finite coefficient (including a
    /// non-finite translation `c`/`f` that would poison the inverse offsets).
    pub fn inverse(&self) -> Option<Self> {
        let inv_det = self.linear_inv_det()?;
        let a_inv = self.e * inv_det;
        let b_inv = -self.b * inv_det;
        let d_inv = -self.d * inv_det;
        let e_inv = self.a * inv_det;
        let c_inv = -(a_inv * self.c + b_inv * self.f);
        let f_inv = -(d_inv * self.c + e_inv * self.f);

        // `linear_inv_det` guards a/b/d/e; c/f (and overflow) are checked here.
        if !(a_inv.is_finite()
            && b_inv.is_finite()
            && c_inv.is_finite()
            && d_inv.is_finite()
            && e_inv.is_finite()
            && f_inv.is_finite())
        {
            return None;
        }

        Some(Self {
            a: a_inv,
            b: b_inv,
            c: c_inv,
            d: d_inv,
            e: e_inv,
            f: f_inv,
        })
    }

    /// Compute the geographic bounding box for an image of given pixel dimensions.
    ///
    /// Projects all four corner pixels through the forward transform and
    /// returns the axis-aligned [`GeoBounds`] that encloses them. For
    /// rotated or skewed transforms the bounding box will be larger than
    /// the actual image footprint.
    ///
    /// # Example usage
    ///
    /// - [pdf_to_georeferenced_pyramid_memory](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
    ///   uses `image_bounds` to verify the geographic extent after
    ///   geo-referencing a rasterized PDF page.
    pub fn image_bounds(&self, width: u32, height: u32) -> GeoBounds {
        let corners = [
            self.pixel_to_geo(PixelCoord { x: 0.0, y: 0.0 }),
            self.pixel_to_geo(PixelCoord {
                x: width as f64,
                y: 0.0,
            }),
            self.pixel_to_geo(PixelCoord {
                x: width as f64,
                y: height as f64,
            }),
            self.pixel_to_geo(PixelCoord {
                x: 0.0,
                y: height as f64,
            }),
        ];

        let min_x = corners.iter().map(|c| c.x).fold(f64::INFINITY, f64::min);
        let max_x = corners
            .iter()
            .map(|c| c.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = corners.iter().map(|c| c.y).fold(f64::INFINITY, f64::min);
        let max_y = corners
            .iter()
            .map(|c| c.y)
            .fold(f64::NEG_INFINITY, f64::max);

        GeoBounds {
            min: GeoCoord { x: min_x, y: min_y },
            max: GeoCoord { x: max_x, y: max_y },
        }
    }

    /// Compute the geographic coordinate for the center of a tile.
    ///
    /// Given a tile position `(tile_x, tile_y)` within a grid of tiles each
    /// `tile_size` pixels wide and tall, returns the geographic coordinate
    /// at the centre of that tile. This is used to annotate tiles with their
    /// real-world location in metadata or debug output.
    ///
    /// # Example usage
    ///
    /// - [pdf_to_georeferenced_pyramid_memory](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
    ///   iterates over tiles and calls `tile_center` to verify each centre
    ///   falls within the expected geographic bounds.
    pub fn tile_center(&self, tile_x: u32, tile_y: u32, tile_size: u32) -> GeoCoord {
        let px = (tile_x as f64 + 0.5) * tile_size as f64;
        let py = (tile_y as f64 + 0.5) * tile_size as f64;
        self.pixel_to_geo(PixelCoord { x: px, y: py })
    }
}

/// Axis-aligned geographic bounding box.
///
/// Stores the minimum and maximum corners of a rectangle in geographic
/// coordinate space. Produced by [`GeoTransform::image_bounds`] and used
/// to test whether a given [`GeoCoord`] falls within the footprint of a
/// geo-referenced image.
///
/// # Example usage
///
/// - [pdf_to_georeferenced_pyramid_memory](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
///   obtains a `GeoBounds` via `image_bounds` and asserts that interior
///   tile centres are contained within it.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-geo-origin)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoBounds {
    pub min: GeoCoord,
    pub max: GeoCoord,
}

impl GeoBounds {
    /// Return the width of the bounding box (max.x - min.x) in geographic units.
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    /// Return the height of the bounding box (max.y - min.y) in geographic units.
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }

    /// Return the centre point of the bounding box.
    pub fn center(&self) -> GeoCoord {
        GeoCoord {
            x: (self.min.x + self.max.x) / 2.0,
            y: (self.min.y + self.max.y) / 2.0,
        }
    }

    /// Test whether `coord` lies inside (or on the boundary of) this box.
    pub fn contains(&self, coord: GeoCoord) -> bool {
        coord.x >= self.min.x
            && coord.x <= self.max.x
            && coord.y >= self.min.y
            && coord.y <= self.max.y
    }
}

impl PixelCoord {
    /// Create a new pixel coordinate from `(x, y)` column/row values.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl GeoCoord {
    /// Create a new geographic coordinate from `(x, y)` values.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn assert_geo_eq(a: GeoCoord, b: GeoCoord) {
        assert!(
            approx_eq(a.x, b.x) && approx_eq(a.y, b.y),
            "GeoCoord mismatch: ({}, {}) vs ({}, {})",
            a.x,
            a.y,
            b.x,
            b.y
        );
    }

    fn assert_pixel_eq(a: PixelCoord, b: PixelCoord) {
        assert!(
            approx_eq(a.x, b.x) && approx_eq(a.y, b.y),
            "PixelCoord mismatch: ({}, {}) vs ({}, {})",
            a.x,
            a.y,
            b.x,
            b.y
        );
    }

    // -- Scale + translate (no rotation) --

    /**
     * Tests that a simple scale+translate transform maps pixels to geo coords.
     * Works by creating a transform with known origin (-122, 37) and 0.1 deg/px,
     * then verifying pixel (0,0) maps to the origin and pixel (10,5) maps to
     * the expected offset (-121.0, 36.5) based on the scale factors.
     */
    #[test]
    fn simple_scale_translate() {
        // 1 pixel = 0.1 degrees, origin at (-122.0, 37.0), Y goes down
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(-122.0, 37.0), 0.1, -0.1);

        let geo = t.pixel_to_geo(PixelCoord::new(0.0, 0.0));
        assert_geo_eq(geo, GeoCoord::new(-122.0, 37.0));

        let geo = t.pixel_to_geo(PixelCoord::new(10.0, 5.0));
        assert_geo_eq(geo, GeoCoord::new(-121.0, 36.5));
    }

    /**
     * Tests that pixel_to_geo followed by geo_to_pixel recovers the original pixel.
     * Works because the transform is invertible (non-zero scale), so composing
     * forward and inverse must yield identity within floating-point tolerance.
     * Input: pixel (500, 300) -> geo -> Output: pixel (500, 300).
     */
    #[test]
    fn round_trip_pixel_geo_pixel() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(-122.4, 37.8), 0.001, -0.001);

        let original = PixelCoord::new(500.0, 300.0);
        let geo = t.pixel_to_geo(original);
        let back = t.geo_to_pixel(geo).unwrap();
        assert_pixel_eq(back, original);
    }

    /**
     * Tests that geo_to_pixel followed by pixel_to_geo recovers the original geo coord.
     * Works because the forward transform is the exact inverse of geo_to_pixel,
     * so the round trip must be lossless within floating-point tolerance.
     * Input: geo (2.5, -1.5) -> pixel -> Output: geo (2.5, -1.5).
     */
    #[test]
    fn round_trip_geo_pixel_geo() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(0.0, 0.0), 0.05, -0.05);

        let original = GeoCoord::new(2.5, -1.5);
        let pixel = t.geo_to_pixel(original).unwrap();
        let back = t.pixel_to_geo(pixel);
        assert_geo_eq(back, original);
    }

    // -- GCPs --

    /**
     * Tests that GCPs forming an identity mapping produce an identity transform.
     * Works by supplying three GCPs where pixel coords equal geo coords, so the
     * solved affine must be the identity matrix. Any arbitrary point like (5, 3)
     * should map to itself.
     */
    #[test]
    fn from_gcps_identity() {
        // Pixel coords == geo coords
        let gcps = [
            (PixelCoord::new(0.0, 0.0), GeoCoord::new(0.0, 0.0)),
            (PixelCoord::new(1.0, 0.0), GeoCoord::new(1.0, 0.0)),
            (PixelCoord::new(0.0, 1.0), GeoCoord::new(0.0, 1.0)),
        ];
        let t = GeoTransform::from_gcps_exact(&gcps).unwrap();

        assert_geo_eq(
            t.pixel_to_geo(PixelCoord::new(5.0, 3.0)),
            GeoCoord::new(5.0, 3.0),
        );
    }

    /**
     * Tests GCP-derived transform with a 2x scale and translation offset.
     * Works by providing three GCPs that encode scale=2 and offset=(10, 20),
     * then checking that the midpoint pixel (50, 50) maps to the expected
     * geo coord (110, 120) = (50*2+10, 50*2+20).
     */
    #[test]
    fn from_gcps_with_scale_and_offset() {
        // 2x scale + offset
        let gcps = [
            (PixelCoord::new(0.0, 0.0), GeoCoord::new(10.0, 20.0)),
            (PixelCoord::new(100.0, 0.0), GeoCoord::new(210.0, 20.0)),
            (PixelCoord::new(0.0, 100.0), GeoCoord::new(10.0, 220.0)),
        ];
        let t = GeoTransform::from_gcps_exact(&gcps).unwrap();

        assert_geo_eq(
            t.pixel_to_geo(PixelCoord::new(50.0, 50.0)),
            GeoCoord::new(110.0, 120.0),
        );
    }

    /**
     * Tests that a GCP-derived transform correctly maps all input GCPs and
     * supports round-trip conversion for arbitrary interior points.
     * Works by first verifying each GCP maps exactly, then converting an
     * interior pixel (500, 400) to geo and back, expecting identity.
     */
    #[test]
    fn from_gcps_round_trip() {
        let gcps = [
            (PixelCoord::new(0.0, 0.0), GeoCoord::new(-122.4, 37.8)),
            (PixelCoord::new(1000.0, 0.0), GeoCoord::new(-122.3, 37.8)),
            (PixelCoord::new(0.0, 800.0), GeoCoord::new(-122.4, 37.72)),
        ];
        let t = GeoTransform::from_gcps_exact(&gcps).unwrap();

        // Verify GCPs map correctly
        for (pixel, geo) in &gcps {
            assert_geo_eq(t.pixel_to_geo(*pixel), *geo);
        }

        // Round-trip
        let p = PixelCoord::new(500.0, 400.0);
        let g = t.pixel_to_geo(p);
        let back = t.geo_to_pixel(g).unwrap();
        assert_pixel_eq(back, p);
    }

    /**
     * Tests that collinear GCPs are rejected (returns None).
     * Works because three points on the line y=x produce a zero determinant
     * in the system of equations, making the affine unsolvable.
     * Input: three points along y=x -> Output: None.
     */
    #[test]
    fn from_gcps_collinear_returns_none() {
        let gcps = [
            (PixelCoord::new(0.0, 0.0), GeoCoord::new(0.0, 0.0)),
            (PixelCoord::new(1.0, 1.0), GeoCoord::new(1.0, 1.0)),
            (PixelCoord::new(2.0, 2.0), GeoCoord::new(2.0, 2.0)),
        ];
        assert!(GeoTransform::from_gcps_exact(&gcps).is_none());
    }

    // -- Inverse --

    /**
     * Tests that a non-degenerate transform has a valid inverse and that
     * geo_to_pixel correctly reverses pixel_to_geo.
     * Works by converting pixel (100, 200) to geo, then back via geo_to_pixel,
     * and asserting the result matches the original pixel coordinates.
     */
    #[test]
    fn inverse_exists() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(10.0, 20.0), 0.5, -0.5);
        let inv = t.inverse().unwrap();
        let p = PixelCoord::new(100.0, 200.0);
        let g = t.pixel_to_geo(p);
        // inv maps geo→pixel: treating geo coords as "pixel" input to the inverse
        let _back = inv.pixel_to_geo(PixelCoord::new(g.x, g.y));
        let back2 = t.geo_to_pixel(g).unwrap();
        assert_pixel_eq(back2, p);
    }

    /**
     * Tests that a singular (degenerate) transform returns None for inverse
     * and geo_to_pixel. Works because a=0,b=0 makes the determinant (a*e-b*d)
     * equal to zero, so the transform is not invertible.
     * Input: degenerate transform with a=0, b=0 -> Output: None for both.
     */
    #[test]
    fn singular_transform_no_inverse() {
        // Degenerate: all geo X = 0 regardless of pixel
        let t = GeoTransform::new(0.0, 0.0, 5.0, 0.0, 1.0, 0.0);
        assert!(t.inverse().is_none());
        assert!(t.geo_to_pixel(GeoCoord::new(5.0, 3.0)).is_none());
    }

    /**
     * Regression: a legitimately small-scale transform must be invertible.
     * Works because the 2x2 determinant of a 1e-7 deg/px transform is ~1e-14,
     * which an absolute `det.abs() < 1e-12` threshold falsely rejects even
     * though the transform is perfectly well-conditioned. The norm-relative
     * test compares |det| against ~1e-12 * norm^2 (~1e-26), so it is accepted
     * and the pixel<->geo round trip is lossless.
     */
    #[test]
    fn small_scale_transform_is_invertible() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(-122.0, 37.0), 1e-7, -1e-7);

        // Both inverse paths must succeed on this well-conditioned transform.
        assert!(t.inverse().is_some());

        let original = PixelCoord::new(4000.0, 2500.0);
        let geo = t.pixel_to_geo(original);
        let back = t
            .geo_to_pixel(geo)
            .expect("small-scale transform must be invertible");
        // Absolute tolerance loosened to match the 1e-7 coordinate scale.
        assert!(
            (back.x - original.x).abs() < 1e-3,
            "x: {} vs {}",
            back.x,
            original.x
        );
        assert!(
            (back.y - original.y).abs() < 1e-3,
            "y: {} vs {}",
            back.y,
            original.y
        );
    }

    /**
     * Regression: a near-degenerate large-coordinate transform must be rejected.
     * Works because rows [1e8, 1e8] and [1e8, 1e8+1e-4] are nearly parallel:
     * the determinant is ~1e4, which an absolute 1e-12 threshold happily
     * accepts (yielding a garbage, wildly ill-conditioned inverse). Relative
     * to the matrix norm^2 (~4e16) the threshold is ~4e4, so |det|=1e4 is
     * correctly judged singular.
     */
    #[test]
    fn near_singular_large_coordinate_rejected() {
        let t = GeoTransform::new(1e8, 1e8, 0.0, 1e8, 1e8 + 1e-4, 0.0);
        assert!(
            t.inverse().is_none(),
            "near-singular transform must be rejected"
        );
        assert!(
            t.geo_to_pixel(GeoCoord::new(1.0, 2.0)).is_none(),
            "near-singular transform must be rejected"
        );
    }

    /**
     * Regression: non-finite coefficients must be rejected, not propagated.
     * Works because a NaN/inf coefficient makes `det` non-finite, and the old
     * `det.abs() < 1e-12` comparison is false for NaN, so the code fell through
     * and returned a `Some` full of NaN pixel coordinates a caller could use as
     * an index. The finiteness guard now returns `None` for every such input.
     */
    #[test]
    fn non_finite_transform_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let t = GeoTransform::new(bad, 0.0, 0.0, 0.0, 1.0, 0.0);
            assert!(t.inverse().is_none(), "inverse must reject {bad}");
            assert!(
                t.geo_to_pixel(GeoCoord::new(1.0, 1.0)).is_none(),
                "geo_to_pixel must reject {bad}"
            );
            // A non-finite translation must also be rejected by inverse().
            let t2 = GeoTransform::new(1.0, 0.0, bad, 0.0, 1.0, 0.0);
            assert!(
                t2.inverse().is_none(),
                "inverse must reject non-finite translation {bad}"
            );
        }
    }

    /**
     * Regression: GCP collinearity is judged relative to point spacing.
     * Three genuinely collinear points offset far from the origin must still be
     * rejected (their edge-relative determinant is zero), while a tiny but
     * non-degenerate triangle (~1e-4 across) must still solve — its raw
     * determinant is ~1e-8, which the norm-relative test correctly accepts
     * because it is large relative to the ~1e-8 squared edge norm.
     */
    #[test]
    fn from_gcps_collinear_far_from_origin_rejected() {
        // Collinear along y = x, offset far from the origin.
        let base = 1.0e6;
        let gcps = [
            (PixelCoord::new(base, base), GeoCoord::new(0.0, 0.0)),
            (
                PixelCoord::new(base + 1.0, base + 1.0),
                GeoCoord::new(1.0, 1.0),
            ),
            (
                PixelCoord::new(base + 2.0, base + 2.0),
                GeoCoord::new(2.0, 2.0),
            ),
        ];
        assert!(
            GeoTransform::from_gcps_exact(&gcps).is_none(),
            "collinear points far from origin must be rejected"
        );

        // A small but genuinely non-degenerate triangle must still solve.
        let gcps_ok = [
            (PixelCoord::new(0.0, 0.0), GeoCoord::new(0.0, 0.0)),
            (PixelCoord::new(1e-4, 0.0), GeoCoord::new(1e-4, 0.0)),
            (PixelCoord::new(0.0, 1e-4), GeoCoord::new(0.0, 1e-4)),
        ];
        assert!(
            GeoTransform::from_gcps_exact(&gcps_ok).is_some(),
            "small non-degenerate triangle must be solvable"
        );
    }

    // -- Image bounds --

    /**
     * Tests that image_bounds computes the correct geographic bounding box.
     * Works by using a 1000x800 image with 0.001 deg/px scale and origin at
     * (-122, 37), so the bbox should span [-122, -121] in X and [36.2, 37] in Y
     * (Y decreases because pixel_size_y is negative).
     */
    #[test]
    fn image_bounds_simple() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(-122.0, 37.0), 0.001, -0.001);
        let bounds = t.image_bounds(1000, 800);
        assert!(approx_eq(bounds.min.x, -122.0));
        assert!(approx_eq(bounds.max.x, -121.0));
        assert!(approx_eq(bounds.min.y, 36.2));
        assert!(approx_eq(bounds.max.y, 37.0));
    }

    /**
     * Tests that GeoBounds::contains correctly classifies interior and exterior
     * points. Works by computing bounds for a 100x100 image, then checking that
     * the center point (0.5, 9.5) is inside and (-1.0, 9.5) is outside.
     */
    #[test]
    fn image_bounds_contains() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(0.0, 10.0), 0.01, -0.01);
        let bounds = t.image_bounds(100, 100);
        assert!(bounds.contains(GeoCoord::new(0.5, 9.5)));
        assert!(!bounds.contains(GeoCoord::new(-1.0, 9.5)));
    }

    /**
     * Tests that GeoBounds::center returns the midpoint of the bounding box.
     * Works by constructing a box from (0,0) to (10,20) and verifying the
     * center is at (5, 10), the arithmetic mean of the min/max coords.
     */
    #[test]
    fn bounds_center() {
        let bounds = GeoBounds {
            min: GeoCoord::new(0.0, 0.0),
            max: GeoCoord::new(10.0, 20.0),
        };
        let center = bounds.center();
        assert!(approx_eq(center.x, 5.0));
        assert!(approx_eq(center.y, 10.0));
    }

    /**
     * Tests that GeoBounds::width and height return correct dimensions.
     * Works by constructing a box from (-5,-3) to (5,7) and verifying
     * width = 10.0 and height = 10.0 via simple subtraction of extremes.
     */
    #[test]
    fn bounds_dimensions() {
        let bounds = GeoBounds {
            min: GeoCoord::new(-5.0, -3.0),
            max: GeoCoord::new(5.0, 7.0),
        };
        assert!(approx_eq(bounds.width(), 10.0));
        assert!(approx_eq(bounds.height(), 10.0));
    }

    // -- Tile center --

    /**
     * Tests that tile_center computes the geo coord at the center of a tile.
     * Works by using tile (0,0) with tile_size=256, so the center pixel is
     * (128, 128). With 1:1 scale and origin (0, 100), the geo center should
     * be (128.0, -28.0) i.e. 100 - 128.
     */
    #[test]
    fn tile_center_calculation() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(0.0, 100.0), 1.0, -1.0);
        // Tile (0,0) at tile_size=256: center pixel = (128, 128)
        let center = t.tile_center(0, 0, 256);
        assert!(approx_eq(center.x, 128.0));
        assert!(approx_eq(center.y, 100.0 - 128.0));
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
        // Property test: for any valid origin, scale, and pixel coordinate,
        // pixel_to_geo followed by geo_to_pixel must recover the original pixel.
        // Works because the scale factors are constrained to be non-zero (and thus
        // invertible), guaranteeing the round trip is lossless within tolerance.
        // Input ranges: origin in [-180,180]x[-90,90], scale in [0.0001,1.0],
        // pixel in [0, 10000]x[0, 10000].
        #[test]
        fn round_trip_pixel_geo(
            ox in -180.0f64..180.0,
            oy in -90.0f64..90.0,
            sx in 0.0001f64..1.0,
            sy in -1.0f64..-0.0001,
            px in 0.0f64..10000.0,
            py in 0.0f64..10000.0,
        ) {
            let t = GeoTransform::from_origin_and_scale(
                GeoCoord::new(ox, oy), sx, sy,
            );
            let pixel = PixelCoord::new(px, py);
            let geo = t.pixel_to_geo(pixel);
            let back = t.geo_to_pixel(geo).unwrap();
            prop_assert!((back.x - pixel.x).abs() < 1e-6, "X: {} vs {}", back.x, pixel.x);
            prop_assert!((back.y - pixel.y).abs() < 1e-6, "Y: {} vs {}", back.y, pixel.y);
        }
    }
}
