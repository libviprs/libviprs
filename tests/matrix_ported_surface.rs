//! Pins the matrix/LUT call surface required by the libviprs-tests
//! ported suite (libviprs-tests issue #55, `tests/ported_create.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves: `invertlut` (`test_invertlut`) and
//! `matrixinvert` (`test_matrixinvert`), each ported test body
//! reproduced literally. Behaviour depth is covered by the unit tests
//! in `src/matrix.rs`. The `from_matrix` setup both bodies share, and
//! the `buildlut` companion, are pinned in
//! `tests/create_ported_surface.rs` with the rest of the create batch.

use libviprs::{MatrixError, Raster};

/// The ported `test_invertlut` body.
#[test]
fn ported_invertlut() {
    let lut = Raster::from_matrix(&[
        vec![0.1, 0.2, 0.3, 0.1],
        vec![0.2, 0.4, 0.4, 0.2],
        vec![0.7, 0.5, 0.6, 0.3],
    ]);
    let im = lut.invertlut();

    assert_eq!(im.width(), 256);
    assert_eq!(im.height(), 1);
    assert_eq!(im.format().channels(), 3);

    let p = im.getpoint(0, 0);
    for &v in &p {
        assert!(v.abs() < 0.001);
    }
    let p = im.getpoint(255, 0);
    for &v in &p {
        assert!((v - 1.0).abs() < 0.001);
    }

    let p = im.getpoint((0.2 * 255.0) as u32, 0);
    assert!((p[0] - 0.1).abs() < 0.1);
}

/// The ported `test_matrixinvert` body.
#[test]
fn ported_matrixinvert() {
    let mat = Raster::from_matrix(&[
        vec![4.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 2.0, 0.0],
        vec![0.0, 1.0, 2.0, 0.0],
        vec![1.0, 0.0, 0.0, 1.0],
    ]);
    let inv = mat.matrixinvert();

    assert_eq!(inv.width(), 4);
    assert_eq!(inv.height(), 4);

    let p = inv.getpoint(0, 0);
    assert!((p[0] - 0.25).abs() < 0.001);
    let p = inv.getpoint(3, 3);
    assert!((p[0] - 1.0).abs() < 0.001);
}

/// The `try_` twins and the sized `invertlut` variant compile from the
/// external position, and singular input is a typed error rather than
/// a panic.
#[test]
fn matrix_family_surface() {
    let mat = Raster::from_matrix(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
    let r: Result<Raster, MatrixError> = mat.try_matrixinvert();
    assert!(matches!(r, Err(MatrixError::Singular)));

    let lut = Raster::from_matrix(&[vec![0.2, 0.5], vec![0.7, 0.9]]);
    let _: Result<Raster, MatrixError> = lut.try_invertlut();
    let _: Result<Raster, MatrixError> = lut.try_invertlut_size(512);
    let sized: Raster = lut.invertlut_size(512);
    assert_eq!(sized.width(), 512);
}

/// `matrixmultiply` and its `try_` twin compile from the external
/// position, produce the measured vips product, and report incompatible
/// sizes as a typed error rather than a panic.
#[test]
fn matrixmultiply_surface() {
    let left = Raster::from_matrix(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let right = Raster::from_matrix(&[vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);

    let product: Raster = left.matrixmultiply(&right);
    assert_eq!(product.width(), 2);
    assert_eq!(product.height(), 2);
    let p = product.getpoint(0, 0);
    assert!((p[0] - 58.0).abs() < 0.001, "(0,0): {}", p[0]);
    let p = product.getpoint(1, 1);
    assert!((p[0] - 154.0).abs() < 0.001, "(1,1): {}", p[0]);

    let ok: Result<Raster, MatrixError> = left.try_matrixmultiply(&right);
    assert!(ok.is_ok());

    let bad: Result<Raster, MatrixError> = left.try_matrixmultiply(&left);
    assert!(matches!(bad, Err(MatrixError::ShapeMismatch { .. })));
}
