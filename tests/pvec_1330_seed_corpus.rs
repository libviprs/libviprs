//! Issue #1330: the seed corpus for libviprs/pvec#219's fuzz targets, one 256 px tile of each
//! variant libviprs writes. The files under `tests/fixtures/pvec-raster-seeds/` must equal what
//! the generator below produces today, and `SHA256SUMS` must match them.
//!
//! Regenerate with `PVEC_SEEDS_BLESS=1 cargo test --test pvec_1330_seed_corpus`.

#[path = "common/tile_headers.rs"]
mod tile_headers;

use std::path::PathBuf;

use libviprs::{PixelFormat, Raster};
use sha2::{Digest, Sha256};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec-raster-seeds")
}

/// A drawing-like tile: a grid, a diagonal and a flat patch, so it compresses and the codecs have
/// structure to chew on, and it is the same bytes on every host.
fn tile(format: PixelFormat) -> Raster {
    let bpp = format.bytes_per_pixel();
    let mut data = Vec::with_capacity(256 * 256 * bpp);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let ink = x % 32 == 0 || y % 32 == 0 || x == y || (x > 192 && y > 192);
            for c in 0..bpp {
                data.push(match (ink, c) {
                    (true, 3) => 255,
                    (true, _) => 16,
                    (false, 3) => 128,
                    (false, _) => 240u32.saturating_sub((x + y) / 8 + c as u32) as u8,
                });
            }
        }
    }
    Raster::new(256, 256, format, data).unwrap()
}

/// Every seed: file name and bytes. The names say codec, channels and depth.
fn seeds() -> Vec<(&'static str, Vec<u8>)> {
    let webp = |r: &Raster| {
        r.encode_webp(libviprs::webp::SaveOptions::default())
            .unwrap()
    };
    vec![
        (
            "png-gray1.png",
            tile(PixelFormat::Gray8).encode_png_bilevel(128).unwrap(),
        ),
        (
            "png-gray8.png",
            tile(PixelFormat::Gray8).encode_png(6).unwrap(),
        ),
        (
            "png-rgb8.png",
            tile(PixelFormat::Rgb8).encode_png(6).unwrap(),
        ),
        (
            "png-rgba8.png",
            tile(PixelFormat::Rgba8).encode_png(6).unwrap(),
        ),
        (
            "jpeg-gray8-baseline.jpg",
            tile(PixelFormat::Gray8).encode_jpeg(85).unwrap(),
        ),
        (
            "jpeg-rgb8-baseline.jpg",
            tile(PixelFormat::Rgb8).encode_jpeg(85).unwrap(),
        ),
        ("webp-vp8l-rgb8.webp", webp(&tile(PixelFormat::Rgb8))),
        ("webp-vp8l-rgba8.webp", webp(&tile(PixelFormat::Rgba8))),
    ]
}

fn sums(seeds: &[(&str, Vec<u8>)]) -> String {
    let mut out = String::new();
    for (name, bytes) in seeds {
        out.push_str(&format!("{}  {name}\n", hex(&Sha256::digest(bytes))));
    }
    out
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem
fn bless_the_seeds_when_asked() {
    if std::env::var_os("PVEC_SEEDS_BLESS").is_none() {
        return;
    }
    std::fs::create_dir_all(dir()).unwrap();
    let seeds = seeds();
    for (name, bytes) in &seeds {
        std::fs::write(dir().join(name), bytes).unwrap();
    }
    std::fs::write(dir().join("SHA256SUMS"), sums(&seeds)).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem
fn the_committed_seeds_equal_what_the_generator_produces_today() {
    if std::env::var_os("PVEC_SEEDS_BLESS").is_some() {
        return;
    }
    for (name, bytes) in seeds() {
        let committed = std::fs::read(dir().join(name))
            .unwrap_or_else(|e| panic!("{name} is not committed ({e}); bless the corpus"));
        assert_eq!(
            Sha256::digest(&committed),
            Sha256::digest(&bytes),
            "{name} drifted from the generator"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem
fn sha256sums_matches_the_files_and_lists_nothing_else() {
    if std::env::var_os("PVEC_SEEDS_BLESS").is_some() {
        return;
    }
    let listed = std::fs::read_to_string(dir().join("SHA256SUMS")).expect("SHA256SUMS");
    let mut on_disk = String::new();
    let mut names: Vec<_> = seeds().into_iter().map(|(n, _)| n).collect();
    for name in &names {
        let bytes = std::fs::read(dir().join(name)).unwrap();
        on_disk.push_str(&format!("{}  {name}\n", hex(&Sha256::digest(bytes))));
    }
    assert_eq!(listed, on_disk);
    names.extend(["SHA256SUMS", "PROVENANCE.md"]);
    let mut present: Vec<_> = std::fs::read_dir(dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    present.sort();
    names.sort_unstable();
    assert_eq!(present, names, "no stray file in the corpus");
}

#[test]
fn every_seed_is_256_square_and_ends_where_its_stream_ends() {
    for (name, bytes) in seeds() {
        if name.ends_with(".png") {
            let h = tile_headers::png(&bytes);
            assert_eq!((h.width, h.height, h.interlace), (256, 256, 0), "{name}");
            assert!(h.ends_at_iend, "{name}");
        } else if name.ends_with(".jpg") {
            let h = tile_headers::jpeg(&bytes);
            assert_eq!((h.width, h.height), (256, 256), "{name}");
            assert!(h.ends_in_eoi, "{name}");
        } else {
            let h = tile_headers::webp(&bytes);
            assert_eq!((h.width, h.height), (256, 256), "{name}");
            assert!(h.riff_size_matches, "{name}");
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem
fn provenance_names_the_generator_and_every_seed() {
    if std::env::var_os("PVEC_SEEDS_BLESS").is_some() {
        return;
    }
    let text = std::fs::read_to_string(dir().join("PROVENANCE.md")).expect("PROVENANCE.md");
    assert!(
        text.contains("tests/pvec_1330_seed_corpus.rs"),
        "names the generator"
    );
    assert!(text.contains("PVEC_SEEDS_BLESS"), "says how to regenerate");
    for (name, _) in seeds() {
        assert!(text.contains(name), "{name} is described");
    }
}
