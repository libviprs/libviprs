//! Issue #1578, L5: a fill is cut to a convex clip as a closed ring.

use libviprs::pvec::pdf_ring_cut::cut_ring;

const SQUARE: [[f64; 2]; 5] = [
    [0.0, 0.0],
    [10.0, 0.0],
    [10.0, 10.0],
    [0.0, 10.0],
    [0.0, 0.0],
];

fn area(ring: &[[f64; 2]]) -> f64 {
    ring.windows(2)
        .map(|w| w[0][0] * w[1][1] - w[1][0] * w[0][1])
        .sum::<f64>()
        .abs()
        / 2.0
}

fn triangle() -> Vec<[f64; 2]> {
    vec![[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [0.0, 0.0]]
}

#[test]
fn a_square_cut_by_a_triangle_keeps_the_shared_corner() {
    // The triangle x + y <= 20 contains the square's corner (10,10) exactly on its edge.
    let cut = cut_ring(&SQUARE, &triangle()).expect("convex clip");
    assert_eq!(cut.first(), cut.last(), "closed");
    assert!(
        (area(&cut) - 100.0).abs() < 1e-9,
        "all of the square is inside, got {cut:?}"
    );
}

#[test]
fn a_ring_half_outside_is_cut_to_the_clip_edge() {
    let clip = vec![
        [5.0, -5.0],
        [15.0, -5.0],
        [15.0, 15.0],
        [5.0, 15.0],
        [5.0, -5.0],
    ];
    let cut = cut_ring(&SQUARE, &clip).expect("convex clip");
    assert!(
        (area(&cut) - 50.0).abs() < 1e-9,
        "half the square, got {cut:?}"
    );
    assert!(cut.iter().all(|p| p[0] >= 5.0 - 1e-9));
}

#[test]
fn a_diagonal_clip_cuts_a_corner_off() {
    let clip = vec![[-5.0, -5.0], [10.0, -5.0], [-5.0, 10.0], [-5.0, -5.0]];
    let cut = cut_ring(&SQUARE, &clip).expect("convex clip");
    // x + y <= 5 inside the square: a triangle of legs 5.
    assert!((area(&cut) - 12.5).abs() < 1e-9, "got {cut:?}");
    assert_eq!(cut.first(), cut.last());
}

#[test]
fn a_ring_wholly_outside_is_empty() {
    let clip = vec![[20.0, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 20.0]];
    assert_eq!(cut_ring(&SQUARE, &clip), Some(Vec::new()));
}

#[test]
fn a_ring_wholly_inside_is_unchanged_in_area() {
    let clip = vec![
        [-50.0, -50.0],
        [50.0, -50.0],
        [50.0, 50.0],
        [-50.0, 50.0],
        [-50.0, -50.0],
    ];
    let cut = cut_ring(&SQUARE, &clip).expect("convex clip");
    assert!((area(&cut) - 100.0).abs() < 1e-9);
}

#[test]
fn the_clip_winding_does_not_matter() {
    let mut clockwise = triangle();
    clockwise.reverse();
    let a = cut_ring(&SQUARE, &triangle()).unwrap();
    let b = cut_ring(&SQUARE, &clockwise).unwrap();
    assert!((area(&a) - area(&b)).abs() < 1e-9);
    let half = vec![
        [5.0, -5.0],
        [15.0, -5.0],
        [15.0, 15.0],
        [5.0, 15.0],
        [5.0, -5.0],
    ];
    let mut half_reversed = half.clone();
    half_reversed.reverse();
    let c = cut_ring(&SQUARE, &half_reversed).unwrap();
    assert!((area(&c) - 50.0).abs() < 1e-9, "got {c:?}");
}

#[test]
fn a_circle_clip_flattened_to_a_ring_cuts_a_square_to_a_disc() {
    // Radius 5 circle at (5,5) flattened to 64 corners, inscribed in the square.
    let mut circle: Vec<[f64; 2]> = (0..64)
        .map(|i| {
            let a = i as f64 / 64.0 * std::f64::consts::TAU;
            [5.0 + 5.0 * a.cos(), 5.0 + 5.0 * a.sin()]
        })
        .collect();
    circle.push(circle[0]);
    let cut = cut_ring(&SQUARE, &circle).expect("a circle is convex");
    assert!(
        (area(&cut) - area(&circle)).abs() < 1e-9,
        "the disc, got area {}",
        area(&cut)
    );
}

#[test]
fn a_clip_that_is_not_convex_is_refused() {
    // An L shape.
    let l = vec![
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 5.0],
        [5.0, 5.0],
        [5.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ];
    assert_eq!(cut_ring(&SQUARE, &l), None);
}

#[test]
fn a_degenerate_clip_is_refused() {
    assert_eq!(
        cut_ring(&SQUARE, &[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        None
    );
    assert_eq!(cut_ring(&SQUARE, &[]), None);
}

#[test]
fn a_ring_crossing_a_clip_corner_keeps_one_ring_not_two_pieces() {
    // A thin diagonal ring across the triangle's hypotenuse stays one closed ring.
    let thin = vec![[8.0, 8.0], [14.0, 8.0], [14.0, 9.0], [8.0, 9.0], [8.0, 8.0]];
    let cut = cut_ring(&thin, &triangle()).unwrap();
    assert_eq!(cut.first(), cut.last());
    // x + y <= 20 within y in [8,9], x in [8,14]: x <= 12 at y=8, x <= 11 at y=9.
    assert!((area(&cut) - 3.5).abs() < 1e-9, "got {cut:?}");
    assert!(cut.iter().all(|p| p[0] + p[1] <= 20.0 + 1e-9));
}
