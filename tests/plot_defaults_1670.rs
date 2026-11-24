//! Issue #1670: the user's default plot pair is stored as copies in a per-user config directory. The
//! library only touches the directory when asked: `store`, `load` and `clear` take it as an argument,
//! `config_dir` works out the usual place from the environment it is given, and `PdfPlot` reads
//! none of it. A stored file is validated going in and every time it is read. Needs
//! `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::write::ZlibEncoder;
use libviprs::cad::plot_defaults::{PlotDefaultFile, clear, config_dir, load, store};
use libviprs::cad::plot_files::{
    MAX_PLOT_FILE_BYTES, MAX_PLOT_STYLE_INFLATED_BYTES, default_pc3, default_plot_style,
    load_plot_style,
};

use PlotDefaultFile::{Ctb, Pc3};

fn ctb() -> Vec<u8> {
    default_plot_style().to_bytes().unwrap()
}

fn pc3() -> Vec<u8> {
    default_pc3().to_bytes()
}

/// A scratch directory that does not exist yet.
fn fresh(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1670s-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

/// A CTB whose header announces `announced` bytes of text and whose payload is `text`.
fn ctb_of(announced: u32, text: &[u8]) -> Vec<u8> {
    let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
    z.write_all(text).unwrap();
    let payload = z.finish().unwrap();
    let mut file = b"PIAFILEVERSION_2.0,CTBVER1,compress\r\npmzlibcodec".to_vec();
    file.extend_from_slice(&0u32.to_le_bytes());
    file.extend_from_slice(&announced.to_le_bytes());
    file.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    file.extend_from_slice(&payload);
    file
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    v.sort();
    v
}

/// The code a call was refused with, or `done` when it was not refused.
fn code<T>(r: Result<T, libviprs::cad::plot_defaults::PlotDefaultsError>) -> &'static str {
    r.err().map_or("done", |e| e.code())
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_stored_pair_comes_back_as_the_bytes_that_went_in() {
    let dir = fresh("roundtrip");
    store(&dir, Ctb, &ctb()).unwrap();
    store(&dir, Pc3, &pc3()).unwrap();
    assert_eq!(load(&dir, Ctb).unwrap(), Some(ctb()));
    assert_eq!(load(&dir, Pc3).unwrap(), Some(pc3()));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_two_files_are_the_state_and_nothing_else_is_written() {
    let dir = fresh("layout");
    store(&dir, Ctb, &ctb()).unwrap();
    store(&dir, Pc3, &pc3()).unwrap();
    assert_eq!(names(&dir), ["plot"]);
    assert_eq!(names(&dir.join("plot")), ["default.ctb", "default.pc3"]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_pair_dropped_in_by_hand_loads() {
    let dir = fresh("byhand");
    fs::create_dir_all(dir.join("plot")).unwrap();
    fs::write(dir.join("plot/default.ctb"), ctb()).unwrap();
    assert_eq!(load(&dir, Ctb).unwrap(), Some(ctb()));
    assert_eq!(load(&dir, Pc3).unwrap(), None);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_loading_from_a_directory_that_is_not_there_is_none() {
    let dir = fresh("absent");
    assert_eq!(load(&dir, Ctb).unwrap(), None);
    assert_eq!(load(&dir, Pc3).unwrap(), None);
    assert!(!dir.exists(), "a load made the directory");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_clear_removes_one_file_and_says_whether_there_was_one() {
    let dir = fresh("clear");
    store(&dir, Ctb, &ctb()).unwrap();
    store(&dir, Pc3, &pc3()).unwrap();
    assert!(clear(&dir, Ctb).unwrap());
    assert!(!clear(&dir, Ctb).unwrap());
    assert_eq!(load(&dir, Ctb).unwrap(), None);
    assert_eq!(load(&dir, Pc3).unwrap(), Some(pc3()));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_second_store_replaces_the_first_and_leaves_no_scratch_file() {
    let dir = fresh("replace");
    let other = {
        let mut t = default_plot_style();
        t.aci_entries[7].lineweight = 12;
        t.to_bytes().unwrap()
    };
    store(&dir, Ctb, &ctb()).unwrap();
    store(&dir, Ctb, &other).unwrap();
    assert_eq!(load(&dir, Ctb).unwrap(), Some(other));
    assert_eq!(names(&dir.join("plot")), ["default.ctb"]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_file_over_the_cap_is_refused_and_nothing_is_written() {
    let dir = fresh("cap");
    let mut big = ctb();
    big.resize(MAX_PLOT_FILE_BYTES + 1, 0);
    assert_eq!(code(store(&dir, Ctb, &big)), "plot-file-too-large");
    let mut big = pc3();
    big.resize(MAX_PLOT_FILE_BYTES + 1, b' ');
    assert_eq!(code(store(&dir, Pc3, &big)), "plot-file-too-large");
    assert!(!dir.exists(), "a refused store made the directory");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_ctb_that_inflates_past_the_bound_is_refused() {
    let dir = fresh("bomb");
    let bomb = ctb_of(
        100,
        &vec![b' '; MAX_PLOT_STYLE_INFLATED_BYTES + 1024 * 1024],
    );
    assert_eq!(
        code(store(&dir, Ctb, &bomb)),
        "plot-file-inflated-too-large"
    );
    assert!(!dir.exists());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_stb_is_refused_because_it_applies_nothing() {
    let dir = fresh("stb");
    let text = default_plot_style()
        .to_text()
        .replace("aci_table_available=TRUE", "aci_table_available=FALSE");
    let stb = ctb_of(text.len() as u32, text.as_bytes());
    assert!(
        load_plot_style("x.ctb", &stb).is_ok(),
        "the loader reads it, so the refusal is for what it is"
    );
    assert_eq!(code(store(&dir, Ctb, &stb)), "plot-default-stb");
    assert!(!dir.exists());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_pc3_with_no_readable_device_is_refused() {
    let dir = fresh("nodevice");
    let mut classic = b"PIAFILEVERSION_2.0,PC3VER1,compress\r\npmzlibcodec".to_vec();
    classic.extend_from_slice(&[1, 2, 3, 4]);
    assert_eq!(code(store(&dir, Pc3, &classic)), "plot-default-no-device");
    let empty = b"PIAFILEVERSION_3.0,json\n{ \"data\" : { } }\n";
    assert_eq!(code(store(&dir, Pc3, empty)), "plot-default-no-device");
    assert!(!dir.exists());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_file_of_the_wrong_kind_is_refused() {
    let dir = fresh("kind");
    assert_eq!(code(store(&dir, Pc3, &ctb())), "plot-file-unrecognised");
    assert_eq!(code(store(&dir, Ctb, &pc3())), "plot-file-unrecognised");
    assert!(!dir.exists());
}

#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_symlinked_target_is_neither_written_nor_read_through() {
    let dir = fresh("link");
    store(&dir, Ctb, &ctb()).unwrap();
    let elsewhere = dir.join("elsewhere.ctb");
    fs::write(&elsewhere, b"keep").unwrap();
    let target = dir.join("plot/default.ctb");
    fs::remove_file(&target).unwrap();
    std::os::unix::fs::symlink(&elsewhere, &target).unwrap();
    assert_eq!(code(store(&dir, Ctb, &ctb())), "plot-default-unsafe");
    assert_eq!(code(load(&dir, Ctb)), "plot-default-unsafe");
    assert_eq!(fs::read(&elsewhere).unwrap(), b"keep");
}

#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_symlinked_plot_directory_is_not_followed() {
    let dir = fresh("linkdir");
    let elsewhere = dir.join("elsewhere");
    fs::create_dir_all(&elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, dir.join("plot")).unwrap();
    assert_eq!(code(store(&dir, Ctb, &ctb())), "plot-default-unsafe");
    assert_eq!(code(load(&dir, Ctb)), "plot-default-unsafe");
    assert!(
        names(&elsewhere).is_empty(),
        "a file was written through the link"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_directory_where_a_default_belongs_is_refused() {
    let dir = fresh("isdir");
    fs::create_dir_all(dir.join("plot/default.ctb")).unwrap();
    assert_eq!(code(load(&dir, Ctb)), "plot-default-unsafe");
    assert_eq!(code(store(&dir, Ctb, &ctb())), "plot-default-unsafe");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_stored_default_that_no_longer_loads_is_an_error_not_a_fall_back() {
    let dir = fresh("corrupt");
    store(&dir, Ctb, &ctb()).unwrap();
    store(&dir, Pc3, &pc3()).unwrap();
    fs::write(dir.join("plot/default.ctb"), b"garbage").unwrap();
    let mut cut = pc3();
    cut.truncate(cut.len() / 2);
    fs::write(dir.join("plot/default.pc3"), cut).unwrap();
    assert_eq!(code(load(&dir, Ctb)), "plot-default-unreadable");
    assert_eq!(code(load(&dir, Pc3)), "plot-default-unreadable");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_a_stored_file_over_the_cap_is_unreadable_without_being_read() {
    let dir = fresh("ondisk");
    fs::create_dir_all(dir.join("plot")).unwrap();
    let f = fs::File::create(dir.join("plot/default.ctb")).unwrap();
    f.set_len(MAX_PLOT_FILE_BYTES as u64 + 1).unwrap();
    assert_eq!(code(load(&dir, Ctb)), "plot-default-unreadable");
}

#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_directory_is_private_and_so_are_the_files() {
    use std::os::unix::fs::PermissionsExt;
    let dir = fresh("modes");
    store(&dir, Ctb, &ctb()).unwrap();
    let mode = |p: PathBuf| fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(dir.join("plot")), 0o700);
    assert_eq!(mode(dir.join("plot/default.ctb")), 0o600);
}

// ---- where the directory is ---------------------------------------------------------------

fn dir_for(os: &str, set: &[(&str, &str)]) -> Option<PathBuf> {
    config_dir(os, |k| {
        set.iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| (*v).to_owned())
    })
}

#[test]
fn issue_1670_the_config_dir_variable_names_the_directory_itself() {
    for os in ["linux", "macos", "windows"] {
        let got = dir_for(
            os,
            &[("LIBVIPRS_CONFIG_DIR", "/srv/lv"), ("HOME", "/home/u")],
        );
        assert_eq!(got, Some(PathBuf::from("/srv/lv")), "{os}");
    }
}

#[test]
fn issue_1670_a_relative_value_is_ignored() {
    let got = dir_for(
        "linux",
        &[
            ("LIBVIPRS_CONFIG_DIR", "lv"),
            ("XDG_CONFIG_HOME", "cfg"),
            ("HOME", "/home/u"),
        ],
    );
    assert_eq!(got, Some(PathBuf::from("/home/u/.config/libviprs")));
}

#[test]
fn issue_1670_linux_uses_xdg_then_home() {
    let xdg = dir_for(
        "linux",
        &[("XDG_CONFIG_HOME", "/x/cfg"), ("HOME", "/home/u")],
    );
    assert_eq!(xdg, Some(PathBuf::from("/x/cfg/libviprs")));
    let home = dir_for("linux", &[("HOME", "/home/u")]);
    assert_eq!(home, Some(PathBuf::from("/home/u/.config/libviprs")));
}

#[test]
fn issue_1670_macos_uses_application_support_under_home() {
    let got = dir_for(
        "macos",
        &[("HOME", "/Users/u"), ("XDG_CONFIG_HOME", "/x/cfg")],
    );
    assert_eq!(
        got,
        Some(PathBuf::from(
            "/Users/u/Library/Application Support/libviprs"
        ))
    );
}

#[test]
fn issue_1670_windows_uses_appdata() {
    let got = dir_for("windows", &[("APPDATA", "/c/Roaming"), ("HOME", "/home/u")]);
    assert_eq!(got, Some(PathBuf::from("/c/Roaming/libviprs")));
}

#[test]
fn issue_1670_with_nothing_to_go_on_there_is_no_directory() {
    for os in ["linux", "macos", "windows"] {
        assert_eq!(dir_for(os, &[]), None, "{os}");
    }
}

// ---- the plot never reads it ------------------------------------------------------------

#[test]
fn issue_1670_the_plot_reads_neither_the_environment_nor_the_store() {
    for (name, src) in [
        (
            "pdf_plot.rs",
            include_str!("../src/pvec_convert/pdf_plot.rs"),
        ),
        (
            "pmtiles_plot.rs",
            include_str!("../src/pvec_convert/pmtiles_plot.rs"),
        ),
    ] {
        for needle in ["plot_defaults", "env::var", "env::vars", "config_dir"] {
            assert!(!src.contains(needle), "{name} mentions {needle}");
        }
    }
}
