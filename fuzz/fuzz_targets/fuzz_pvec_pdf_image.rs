#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use libviprs::pvec::pdf_image::{PdfImageOptions, bake_page_images};
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

/// Fuzz the PDF image bake (#1362, RL2.1 follow-up 1/6).
///
/// The input is not a PDF: it describes one. A single-image page is synthesised with `lopdf`
/// from the fields below, so the fuzzer spends its time on the filter seam (the chain, the
/// `/DecodeParms` shapes, the dimensions against the payload, the colour space against the
/// bit depth) and not on rediscovering the PDF syntax that `fuzz_pvec_pdf_vector` already
/// covers. The payload is the stream's bytes, either as they are or wrapped in Flate, so the
/// codecs see both a stream that inflates and one that does not.
///
/// `max_pixels` is held small enough that a refusal is cheap. Whatever `bake_page_images`
/// returns must be a value: a panic, an abort or an unbounded allocation is the finding.
#[derive(Arbitrary, Debug)]
struct Case {
    /// Index into the filter names, taken modulo the list.
    filter: u8,
    flate: bool,
    /// Wrap the payload in Flate before it goes into the stream.
    deflate: bool,
    width: u16,
    height: u16,
    bpc: u8,
    space: u8,
    parms: u8,
    k: i8,
    columns: u16,
    /// Send `/Columns` equal to `/Width`, so a fax stream reaches the codec instead of the
    /// `/Columns` check (#1363). Without this nearly every case would stop at the check.
    columns_match: bool,
    /// The other fax parameters: `/Rows`, `/EndOfBlock`, `/EndOfLine`, `/EncodedByteAlign`,
    /// `/BlackIs1`, each present as true, present as false, or absent (#1363).
    fax_flags: [u8; 5],
    predictor: u8,
    decode_flip: bool,
    image_mask: bool,
    smask: bool,
    /// For `JPXDecode` with the `jp2k` feature: start from a valid JP2 file (so the fuzzer gets
    /// past the container parser) and let `payload` mutate it, as (offset, byte) pairs (#1364).
    #[cfg_attr(not(feature = "jp2k"), allow(dead_code))]
    mutate_valid_jp2: bool,
    payload: Vec<u8>,
}

const FILTERS: [&str; 7] = [
    "",
    "DCTDecode",
    "CCITTFaxDecode",
    "JBIG2Decode",
    "JPXDecode",
    "LZWDecode",
    "ASCII85Decode",
];

fn name(n: &str) -> Object {
    Object::Name(n.as_bytes().to_vec())
}

/// A small lossless JP2 file, encoded once.
#[cfg(feature = "jp2k")]
fn valid_jp2() -> &'static Vec<u8> {
    static JP2: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    JP2.get_or_init(|| {
        let px: Vec<u8> = (0..8 * 8 * 3).map(|i| (i * 5) as u8).collect();
        libviprs::raster::Raster::new(8, 8, libviprs::pixel::PixelFormat::Rgb8, px)
            .ok()
            .and_then(|r| r.encode_jp2k(Default::default()).ok())
            .unwrap_or_default()
    })
}

fn build(case: &Case) -> Vec<u8> {
    use std::io::Write;
    let width = i64::from(case.width % 2048) + 1;
    let height = i64::from(case.height % 2048) + 1;
    let mut payload = case.payload.clone();
    #[cfg(feature = "jp2k")]
    if case.mutate_valid_jp2 && FILTERS[usize::from(case.filter) % FILTERS.len()] == "JPXDecode" {
        let mut valid = valid_jp2().clone();
        for pair in case.payload.chunks_exact(3) {
            let at = usize::from(u16::from_le_bytes([pair[0], pair[1]])) % valid.len();
            valid[at] = pair[2];
        }
        payload = valid;
    }
    if case.deflate {
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        let _ = z.write_all(&payload);
        payload = z.finish().unwrap_or_default();
    }
    let mut chain = Vec::new();
    if case.flate || case.deflate {
        chain.push(name("FlateDecode"));
    }
    let codec = FILTERS[usize::from(case.filter) % FILTERS.len()];
    if !codec.is_empty() {
        chain.push(name(codec));
    }
    let mut d = dictionary! {
        "Type" => "XObject", "Subtype" => "Image", "Width" => width, "Height" => height,
        "BitsPerComponent" => i64::from(case.bpc % 20),
    };
    d.set(
        "ColorSpace",
        match case.space % 5 {
            0 => name("DeviceGray"),
            1 => name("DeviceRGB"),
            2 => name("DeviceCMYK"),
            3 => Object::Array(vec![
                name("Indexed"),
                name("DeviceRGB"),
                1.into(),
                Object::string_literal(vec![0, 0, 0, 255, 255, 255]),
            ]),
            _ => name("Pattern"),
        },
    );
    match chain.len() {
        0 => {}
        1 => d.set("Filter", chain[0].clone()),
        _ => d.set("Filter", Object::Array(chain.clone())),
    }
    let mut parms = Dictionary::new();
    parms.set("K", i64::from(case.k));
    parms.set(
        "Columns",
        if case.columns_match {
            width
        } else {
            i64::from(case.columns)
        },
    );
    for (key, flag) in ["EndOfBlock", "EndOfLine", "EncodedByteAlign", "BlackIs1"]
        .into_iter()
        .zip(&case.fax_flags[1..])
    {
        match flag % 3 {
            0 => {}
            1 => parms.set(key, true),
            _ => parms.set(key, false),
        }
    }
    match case.fax_flags[0] % 3 {
        0 => {}
        1 => parms.set("Rows", height),
        _ => parms.set("Rows", height + 1),
    }
    parms.set("Predictor", i64::from(case.predictor % 17));
    parms.set("Colors", 1);
    parms.set("BitsPerComponent", 8);
    match case.parms % 4 {
        0 => {}
        1 => d.set("DecodeParms", Object::Dictionary(parms)),
        // An array matched to the filters by position, with a null where a stage has none.
        2 => d.set(
            "DecodeParms",
            Object::Array(
                std::iter::repeat_n(Object::Null, chain.len().saturating_sub(1))
                    .chain([Object::Dictionary(parms)])
                    .collect(),
            ),
        ),
        _ => d.set(
            "DecodeParms",
            Object::Array(vec![Object::Dictionary(parms), Object::Null, Object::Null]),
        ),
    }
    if case.decode_flip {
        d.set("Decode", vec![1.into(), 0.into()]);
    }
    if case.image_mask {
        d.set("ImageMask", true);
    }
    let mut doc = Document::with_version("1.7");
    if case.smask {
        let mask = doc.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image", "Width" => 2, "Height" => 2,
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8,
            },
            vec![0, 85, 170, 255],
        ));
        d.set("SMask", mask);
    }
    let pages = doc.new_object_id();
    let image = doc.add_object(Stream::new(d, payload));
    let content = doc.add_object(Stream::new(
        Dictionary::new(),
        b"q 100 0 0 100 0 0 cm /Im1 Do Q".to_vec(),
    ));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Contents" => content,
        "Resources" => dictionary! { "XObject" => dictionary! { "Im1" => image } },
    });
    doc.objects.insert(
        pages,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    let _ = doc.save_to(&mut out);
    out
}

fuzz_target!(|case: Case| {
    let bytes = build(&case);
    let limits = PdfVectorLimits {
        max_image_bytes: 1 << 20,
        max_image_stream_bytes: 1 << 20,
        max_content_bytes: 1 << 20,
        ..PdfVectorLimits::default()
    };
    let Ok(page) = extract_page_paths_from_bytes(&bytes, 1, &limits) else {
        return;
    };
    let mut options = PdfImageOptions::default();
    options.max_pixels = 1 << 20;
    let bake = bake_page_images(&page, &options);
    // Every image is accounted for, whatever became of it.
    assert_eq!(
        bake.report.images,
        bake.report.written + bake.report.refused_total()
    );
    assert_eq!(
        bake.report.written_by_filter.values().sum::<u64>(),
        bake.report.written
    );
});
