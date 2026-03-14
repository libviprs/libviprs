/// Geo-coordinate mapping for tile pyramids.
///
/// Maps pixel coordinates ↔ geographic coordinates using an affine transform.
/// This supports the common case of geo-referenced blueprints and plans where
/// the relationship between pixel space and geographic space is a linear
/// (affine) mapping — rotation, scale, skew, and translation.
///
/// For most construction/AEC use cases, an affine transform is sufficient
/// because the area covered by a single plan sheet is small enough that
/// Earth curvature effects are negligible.

/// A 2D point in pixel space (column, row from top-left origin).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelCoord {
    pub x: f64,
    pub y: f64,
}

/// A 2D point in geographic space (longitude, latitude or easting, northing).
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoTransform {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl GeoTransform {
    /// Create a transform from the 6 affine coefficients.
    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self { a, b, c, d, e, f }
    }

    /// Create a simple scale + translate transform (no rotation or skew).
    ///
    /// - `origin`: geographic coordinate of the top-left pixel
    /// - `pixel_size_x`: geographic units per pixel in X (typically positive)
    /// - `pixel_size_y`: geographic units per pixel in Y (typically negative
    ///   for top-down rasters)
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
    /// Requires exactly 3 non-collinear GCPs to solve the 6 affine parameters.
    /// For over-determined systems (more than 3 GCPs), use `from_gcps_least_squares`.
    pub fn from_gcps_exact(gcps: &[(PixelCoord, GeoCoord); 3]) -> Option<Self> {
        let [(p0, g0), (p1, g1), (p2, g2)] = gcps;

        // Solve: [a b c] * [[px0 px1 px2], [py0 py1 py2], [1 1 1]] = [gx0 gx1 gx2]
        // Same for d, e, f with gy values.
        let det = p0.x * (p1.y - p2.y) - p1.x * (p0.y - p2.y) + p2.x * (p0.y - p1.y);

        if det.abs() < 1e-12 {
            return None; // Points are collinear
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

        Some(Self { a, b, c, d, e, f })
    }

    /// Transform a pixel coordinate to a geographic coordinate.
    pub fn pixel_to_geo(&self, pixel: PixelCoord) -> GeoCoord {
        GeoCoord {
            x: self.a * pixel.x + self.b * pixel.y + self.c,
            y: self.d * pixel.x + self.e * pixel.y + self.f,
        }
    }

    /// Transform a geographic coordinate to a pixel coordinate.
    ///
    /// Returns `None` if the transform is singular (degenerate).
    pub fn geo_to_pixel(&self, geo: GeoCoord) -> Option<PixelCoord> {
        let det = self.a * self.e - self.b * self.d;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let dx = geo.x - self.c;
        let dy = geo.y - self.f;
        Some(PixelCoord {
            x: (self.e * dx - self.b * dy) * inv_det,
            y: (-self.d * dx + self.a * dy) * inv_det,
        })
    }

    /// Compute the inverse transform (geo → pixel as a GeoTransform).
    ///
    /// Returns `None` if the transform is singular.
    pub fn inverse(&self) -> Option<Self> {
        let det = self.a * self.e - self.b * self.d;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let a_inv = self.e * inv_det;
        let b_inv = -self.b * inv_det;
        let d_inv = -self.d * inv_det;
        let e_inv = self.a * inv_det;
        let c_inv = -(a_inv * self.c + b_inv * self.f);
        let f_inv = -(d_inv * self.c + e_inv * self.f);

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
    pub fn tile_center(&self, tile_x: u32, tile_y: u32, tile_size: u32) -> GeoCoord {
        let px = (tile_x as f64 + 0.5) * tile_size as f64;
        let py = (tile_y as f64 + 0.5) * tile_size as f64;
        self.pixel_to_geo(PixelCoord { x: px, y: py })
    }
}

/// Axis-aligned geographic bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoBounds {
    pub min: GeoCoord,
    pub max: GeoCoord,
}

impl GeoBounds {
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }

    pub fn center(&self) -> GeoCoord {
        GeoCoord {
            x: (self.min.x + self.max.x) / 2.0,
            y: (self.min.y + self.max.y) / 2.0,
        }
    }

    pub fn contains(&self, coord: GeoCoord) -> bool {
        coord.x >= self.min.x
            && coord.x <= self.max.x
            && coord.y >= self.min.y
            && coord.y <= self.max.y
    }
}

impl PixelCoord {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl GeoCoord {
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

    #[test]
    fn simple_scale_translate() {
        // 1 pixel = 0.1 degrees, origin at (-122.0, 37.0), Y goes down
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(-122.0, 37.0), 0.1, -0.1);

        let geo = t.pixel_to_geo(PixelCoord::new(0.0, 0.0));
        assert_geo_eq(geo, GeoCoord::new(-122.0, 37.0));

        let geo = t.pixel_to_geo(PixelCoord::new(10.0, 5.0));
        assert_geo_eq(geo, GeoCoord::new(-121.0, 36.5));
    }

    #[test]
    fn round_trip_pixel_geo_pixel() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(-122.4, 37.8), 0.001, -0.001);

        let original = PixelCoord::new(500.0, 300.0);
        let geo = t.pixel_to_geo(original);
        let back = t.geo_to_pixel(geo).unwrap();
        assert_pixel_eq(back, original);
    }

    #[test]
    fn round_trip_geo_pixel_geo() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(0.0, 0.0), 0.05, -0.05);

        let original = GeoCoord::new(2.5, -1.5);
        let pixel = t.geo_to_pixel(original).unwrap();
        let back = t.pixel_to_geo(pixel);
        assert_geo_eq(back, original);
    }

    // -- GCPs --

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

    #[test]
    fn singular_transform_no_inverse() {
        // Degenerate: all geo X = 0 regardless of pixel
        let t = GeoTransform::new(0.0, 0.0, 5.0, 0.0, 1.0, 0.0);
        assert!(t.inverse().is_none());
        assert!(t.geo_to_pixel(GeoCoord::new(5.0, 3.0)).is_none());
    }

    // -- Image bounds --

    #[test]
    fn image_bounds_simple() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(-122.0, 37.0), 0.001, -0.001);
        let bounds = t.image_bounds(1000, 800);
        assert!(approx_eq(bounds.min.x, -122.0));
        assert!(approx_eq(bounds.max.x, -121.0));
        assert!(approx_eq(bounds.min.y, 36.2));
        assert!(approx_eq(bounds.max.y, 37.0));
    }

    #[test]
    fn image_bounds_contains() {
        let t = GeoTransform::from_origin_and_scale(GeoCoord::new(0.0, 10.0), 0.01, -0.01);
        let bounds = t.image_bounds(100, 100);
        assert!(bounds.contains(GeoCoord::new(0.5, 9.5)));
        assert!(!bounds.contains(GeoCoord::new(-1.0, 9.5)));
    }

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
