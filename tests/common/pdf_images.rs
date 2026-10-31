//! PDF fixtures with one image construct each, shared by the issue #1332 tests. Built in memory
//! with `lopdf`, so every expected pixel in a test is computed by hand from what is written here.

#![allow(dead_code)]

use libviprs::pvec::pdf_vector::{PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes};
use lopdf::{Dictionary, Document, Object, Stream, StringFormat, dictionary};

/// A page resource entry: an image or form under `/XObject`, or a state under `/ExtGState`.
pub struct Resources {
    pub xobjects: Vec<(&'static str, Object)>,
    pub gstates: Vec<(&'static str, Object)>,
}

impl Resources {
    pub fn images(xobjects: Vec<(&'static str, Object)>) -> Self {
        Self {
            xobjects,
            gstates: Vec::new(),
        }
    }
}

/// A one page PDF, 612 by 792 points, whose content stream is `content`.
pub fn pdf(doc: Document, resources: Resources, content: &[u8]) -> Vec<u8> {
    let mut doc = doc;
    let pages_id = doc.new_object_id();
    let content = doc.add_object(Stream::new(Dictionary::new(), content.to_vec()));
    let mut xo = Dictionary::new();
    for (name, obj) in resources.xobjects {
        xo.set(name, obj);
    }
    let mut gs = Dictionary::new();
    for (name, obj) in resources.gstates {
        gs.set(name, obj);
    }
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Contents" => content,
        "Resources" => dictionary! { "XObject" => xo, "ExtGState" => gs },
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

pub fn doc() -> Document {
    Document::with_version("1.7")
}

/// An image XObject. `extra` is merged over `/Type /XObject /Subtype /Image /Width /Height`.
pub fn image(w: i64, h: i64, extra: Dictionary, data: Vec<u8>) -> Stream {
    let mut d =
        dictionary! { "Type" => "XObject", "Subtype" => "Image", "Width" => w, "Height" => h };
    for (k, v) in extra.iter() {
        d.set(k.clone(), v.clone());
    }
    Stream::new(d, data)
}

pub fn gray8(w: i64, h: i64, data: &[u8]) -> Stream {
    image(
        w,
        h,
        dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8 },
        data.to_vec(),
    )
}

pub fn rgb8(w: i64, h: i64, data: &[u8]) -> Stream {
    image(
        w,
        h,
        dictionary! { "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8 },
        data.to_vec(),
    )
}

pub fn name(n: &str) -> Object {
    Object::Name(n.as_bytes().to_vec())
}

pub fn literal(bytes: &[u8]) -> Object {
    Object::String(bytes.to_vec(), StringFormat::Literal)
}

pub fn page(bytes: &[u8]) -> PdfVectorPage {
    extract_page_paths_from_bytes(bytes, 1, &PdfVectorLimits::default()).unwrap()
}

/// One image, `Im1`, painted by `content`.
pub fn one(stream: Stream, content: &str) -> Vec<u8> {
    pdf(
        doc(),
        Resources::images(vec![("Im1", Object::Stream(stream))]),
        content.as_bytes(),
    )
}
