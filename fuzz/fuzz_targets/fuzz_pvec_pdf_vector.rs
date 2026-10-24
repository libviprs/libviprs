#![no_main]

use libfuzzer_sys::fuzz_target;
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};

/// Fuzz the PDF vector path (N5.8, #1087).
///
/// Feeds arbitrary bytes as a PDF to the content-stream reader with limits small
/// enough that a refusal is cheap, so a run that finds an amplification shows up as an
/// out-of-memory or a timeout rather than a slow pass. Whatever comes back must be a
/// value: a panic, an abort or an unbounded allocation is the finding.
fuzz_target!(|data: &[u8]| {
    let limits = PdfVectorLimits {
        max_form_depth: 8,
        max_operations: 50_000,
        max_graphics_stack: 64,
        max_entities: 10_000,
        max_content_bytes: 1 << 20,
        max_segments_per_path: 10_000,
    };
    for page in 1..=2 {
        let _ = extract_page_paths_from_bytes(data, page, &limits);
    }
});
