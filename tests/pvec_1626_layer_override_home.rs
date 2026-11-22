//! Issue #1626, panel call C1: `LayerOverride` is the user's visibility override and lives in
//! `pvec::view_profile` next to `LayerVisibility`, which `pvec` can reach without `pvec_convert`.
//! `pvec_convert` re-exports the same type for the manifest and the choices.

#![cfg(feature = "pvec-convert")]

#[test]
fn layer_override_lives_in_pvec_and_pvec_convert_reexports_it() {
    let from_pvec: libviprs::pvec::LayerOverride = libviprs::pvec::LayerOverride::Hidden;
    let from_convert: libviprs::pvec_convert::LayerOverride = from_pvec;
    assert_eq!(from_convert, libviprs::pvec::LayerOverride::Hidden);
    assert_eq!(
        serde_json::to_string(&from_convert).expect("serialises"),
        "\"hidden\""
    );
}
