#![no_main]

use libfuzzer_sys::fuzz_target;

/// Fuzz the PDF parsing and image extraction path.
///
/// Feeds arbitrary bytes as a "PDF" to lopdf's parser, then attempts
/// to extract page info and images. Catches panics, OOM, and infinite
/// loops in the PDF parsing code.
fuzz_target!(|data: &[u8]| {
    // Write the fuzz input to a temp file since lopdf expects a path.
    let dir = std::env::temp_dir().join("libviprs_fuzz");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("fuzz.pdf");
    if std::fs::write(&path, data).is_err() {
        return;
    }

    // Try pdf_info — exercises lopdf parsing
    let _ = libviprs::pdf_info(&path);

    // Try extract_page_image — exercises stream decompression and image decode
    let _ = libviprs::extract_page_image(&path, 1);

    let _ = std::fs::remove_file(&path);
});
