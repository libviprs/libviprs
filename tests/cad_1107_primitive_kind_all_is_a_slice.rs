//! Issue #1107: `PrimitiveKind::ALL` must not republish the variant count in
//! its type.
//!
//! `PrimitiveKind` is `#[non_exhaustive]` so a ninth shape is not a breaking
//! change. `ALL` as `[Self; 8]` would turn the ninth shape into one anyway,
//! for anyone who annotated it or passed it to a `[T; 8]` parameter. Integration
//! tests compile as an external crate, so the annotation below is exactly what
//! a consumer writes, and it stops compiling the day `ALL` is an array again.

use libviprs::cad::PrimitiveKind;

#[test]
fn all_is_a_static_slice_and_not_an_array() {
    let all: &'static [PrimitiveKind] = PrimitiveKind::ALL;
    assert_eq!(all.len(), 8);
}

#[test]
fn iterating_all_yields_references() {
    // `[T; N]::into_iter` yields `T` and `&[T]::into_iter` yields `&T`, which
    // is why moving to a slice after a consumer exists is a break.
    let mut names = Vec::new();
    for kind in PrimitiveKind::ALL {
        let kind: &PrimitiveKind = kind;
        names.push(kind.name());
    }
    assert_eq!(
        names,
        [
            "line", "polyline", "arc", "circle", "ellipse", "spline", "polygon", "text"
        ]
    );
}
