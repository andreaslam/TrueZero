//! Packs the payload the launcher embeds: a zstd-compressed tar holding the binary named by
//! `TZ_BUNDLE_EXE` and every shared library in `TZ_BUNDLE_LIBS` (the libtorch lib dir).
//!
//! Layout inside the archive: `<name>` and `lib/<libs>` (Unix; matches the binary's
//! `@executable_path/lib` / `$ORIGIN/lib` rpath), or `<name>.exe` with the DLLs beside it
//! (Windows).

use sha2::{Digest, Sha256};
use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Read},
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-env-changed=TZ_BUNDLE_EXE");
    println!("cargo:rerun-if-env-changed=TZ_BUNDLE_LIBS");
    println!("cargo:rerun-if-env-changed=TZ_BUNDLE_LEVEL");
    let exe = PathBuf::from(env::var("TZ_BUNDLE_EXE").expect(
        "TZ_BUNDLE_EXE is not set; build this crate with src/rust/bundle.sh, not directly",
    ));
    let libs = PathBuf::from(env::var("TZ_BUNDLE_LIBS").expect("TZ_BUNDLE_LIBS is not set"));
    let level: i32 = env::var("TZ_BUNDLE_LEVEL")
        .ok()
        .and_then(|l| l.parse().ok())
        .unwrap_or(12);
    println!("cargo:rerun-if-changed={}", exe.display());
    println!("cargo:rerun-if-changed={}", libs.display());

    let windows = env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";
    let name = exe.file_name().unwrap().to_string_lossy().into_owned();
    let lib_prefix = if windows { "" } else { "lib/" };

    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("payload.tar.zst");
    {
        let mut encoder = zstd::Encoder::new(BufWriter::new(File::create(&out).unwrap()), level)
            .unwrap();
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get() as u32);
        encoder.multithread(threads).unwrap();
        encoder.long_distance_matching(true).unwrap();
        let mut tar = tar::Builder::new(encoder);
        tar.follow_symlinks(true); // store library symlinks as plain files
        tar.append_path_with_name(&exe, &name).unwrap();
        let mut entries: Vec<_> = fs::read_dir(&libs)
            .unwrap_or_else(|e| panic!("cannot read TZ_BUNDLE_LIBS {}: {e}", libs.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_file() && is_needed_library(p))
            .collect();
        entries.sort();
        assert!(!entries.is_empty(), "no libraries found in {}", libs.display());
        for lib in entries {
            let file_name = lib.file_name().unwrap().to_string_lossy().into_owned();
            tar.append_path_with_name(&lib, format!("{lib_prefix}{file_name}"))
                .unwrap();
        }
        tar.into_inner().unwrap().finish().unwrap();
    }

    let mut hasher = Sha256::new();
    let mut payload = File::open(&out).unwrap();
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = payload.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hash: String = hasher.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    println!("cargo:rustc-env=TZ_BUNDLE_PAYLOAD={}", out.display());
    println!("cargo:rustc-env=TZ_BUNDLE_HASH={hash}");
    println!("cargo:rustc-env=TZ_BUNDLE_NAME={name}");
}

/// Shared libraries the binary can load. The Python bindings (`libtorch_python`, `libshm`
/// used by torch.multiprocessing) are only needed inside a Python process.
fn is_needed_library(path: &Path) -> bool {
    let name = path.file_name().unwrap().to_string_lossy();
    let is_lib = name.ends_with(".dylib") || name.ends_with(".dll") || name.contains(".so");
    let python_only = name.contains("torch_python") || name.starts_with("libshm");
    is_lib && !python_only
}
