//! Build script for the `tzrust` package.
//!
//! 1. rpath: every binary looks for libtorch first in `lib/` next to the executable (the
//!    layout `bundle.sh` produces), then in the libtorch directory it was built against,
//!    so binaries run without `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`.
//! 2. Default network: `TZ_DEFAULT_NET_FILE` names the net that `ucimain` compiles into
//!    itself. Set `TZ_DEFAULT_NET` at build time to choose it (absolute, or relative to the
//!    workspace root); it defaults to `tz_36.pt`. If the file is missing the build still
//!    succeeds (with a warning) and `ucimain` then needs a net path on its command line.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

const DEFAULT_NET: &str = "tz_36.pt";

fn main() {
    set_rpath();
    select_default_net();
}

fn set_rpath() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let local = match target_os.as_str() {
        "macos" => "@executable_path/lib",
        "linux" => "$ORIGIN/lib",
        _ => return, // Windows looks next to the executable already
    };
    println!("cargo:rustc-link-arg=-Wl,-rpath,{local}");
    // set by torch-sys (links = "tch") on Linux and macOS
    if let Ok(libtorch) = env::var("DEP_TCH_LIBTORCH_LIB") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libtorch}");
    }
}

fn select_default_net() {
    println!("cargo:rerun-if-env-changed=TZ_DEFAULT_NET");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("..").join("..");
    let requested = env::var("TZ_DEFAULT_NET").unwrap_or_else(|_| DEFAULT_NET.to_string());
    let net = root.join(&requested); // join keeps an absolute `requested` as is

    let file = if net.is_file() {
        println!("cargo:rerun-if-changed={}", net.display());
        fs::canonicalize(&net).unwrap()
    } else {
        // re-run when the file appears (its directory's mtime changes)
        let dir = net.parent().unwrap_or(Path::new("."));
        println!("cargo:rerun-if-changed={}", dir.display());
        println!(
            "cargo:warning=default net {} not found; ucimain will be built without an \
             embedded net and will need one on its command line",
            net.display()
        );
        let empty = PathBuf::from(env::var("OUT_DIR").unwrap()).join("no_default_net.pt");
        fs::write(&empty, b"").unwrap();
        empty
    };
    println!("cargo:rustc-env=TZ_DEFAULT_NET_FILE={}", file.display());
    println!("cargo:rustc-env=TZ_DEFAULT_NET_NAME={requested}");
}
