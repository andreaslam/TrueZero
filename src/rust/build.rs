//! Embeds the git commit the binaries are built from, and whether tracked files had
//! uncommitted changes, so a run can be pinned to one commit (see `runconfig.rs`).
//! The commit is taken at build time on purpose: it is the code the binary runs, which
//! can differ from the working tree when it starts.

use std::{path::Path, process::Command};

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn main() {
    // Builds without a .git directory (the Docker image) pass the commit in instead.
    println!("cargo:rerun-if-env-changed=TZ_BUILD_GIT_SHA");
    let injected = std::env::var("TZ_BUILD_GIT_SHA")
        .ok()
        .filter(|s| !s.is_empty() && s != "unknown");
    let sha = injected
        .clone()
        .or_else(|| git(&["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unknown".into());
    // Untracked files (run.toml, nets/, ...) do not make the tree dirty.
    let dirty = match (&injected, git(&["status", "--porcelain", "--untracked-files=no"])) {
        (Some(_), _) => false,
        (None, Some(s)) => !s.is_empty(),
        (None, None) => true,
    };
    println!("cargo:rustc-env=TZ_GIT_SHA={sha}");
    println!("cargo:rustc-env=TZ_GIT_DIRTY={}", dirty as u8);

    // Re-run when the commit or the index changes, or when any source file is edited, so
    // the dirty flag is not stale. Python edits made after the build are not seen here;
    // the trainer checks the working tree itself. Directories are scanned recursively by cargo; `target`
    // is deliberately not listed.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let git_dir = manifest.join("../../.git");
    for p in ["HEAD", "index"] {
        println!("cargo:rerun-if-changed={}", git_dir.join(p).display());
    }
    if let Some(head_ref) = std::fs::read_to_string(git_dir.join("HEAD"))
        .ok()
        .and_then(|h| h.strip_prefix("ref: ").map(|r| r.trim().to_owned()))
    {
        println!(
            "cargo:rerun-if-changed={}",
            git_dir.join(head_ref).display()
        );
    }
    println!("cargo:rerun-if-changed={}", manifest.join("src").display());
    for entry in std::fs::read_dir(manifest).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "rs" || e == "toml") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
