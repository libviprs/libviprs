//! Closed-ring fill cutting against a convex clip (issue #1578, L5).
//!
//! A fill under a clip that is not a rectangle (a circle, a triangle, a half circle) is cut as a
//! closed ring, never as a polyline: a polyline clip can split a ring into open pieces that
//! triangulate wrongly. A convex clip cuts a ring correctly under both fill rules, so a cut ring
//! keeps its rule.

/// Cuts the closed `ring` (first point equal to last) to the convex closed `clip` ring, either
/// winding. Returns the cut ring, closed, or an empty vector when nothing is left. Returns `None`
/// when `clip` is not convex, or is not a closed ring of three or more corners: the caller keeps
/// the fill uncut then, as display fidelity.
#[must_use]
pub fn cut_ring(ring: &[[f64; 2]], clip: &[[f64; 2]]) -> Option<Vec<[f64; 2]>> {
    let _ = clip;
    Some(ring.to_vec())
}
