//! Single-file launcher for a tzrust binary (built by `src/rust/bundle.sh`).
//!
//! The real binary and its libtorch libraries are compiled into this executable as a
//! compressed archive. On first run they are unpacked into a per-version cache directory;
//! after that the launcher just starts the cached binary with the same arguments and stdio.
//!
//! Cache location: `$TZ_BUNDLE_CACHE` if set, else `~/Library/Caches/truezero` (macOS),
//! `$XDG_CACHE_HOME/truezero` or `~/.cache/truezero` (Linux), `%LOCALAPPDATA%\truezero`
//! (Windows). Each build gets its own `<name>-<hash>` subdirectory, so old versions can be
//! deleted at any time.

use std::{
    env, fs,
    io::{self, Cursor},
    path::{Path, PathBuf},
    process::{self, Command},
};

static PAYLOAD: &[u8] = include_bytes!(env!("TZ_BUNDLE_PAYLOAD"));
const HASH: &str = env!("TZ_BUNDLE_HASH");
const NAME: &str = env!("TZ_BUNDLE_NAME");
const READY_MARKER: &str = ".complete";

fn main() {
    let dir = match prepare() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("{NAME}: failed to unpack bundled files: {e}");
            process::exit(1);
        }
    };
    let mut cmd = Command::new(dir.join(NAME));
    cmd.args(env::args_os().skip(1));
    exec(cmd);
}

/// Returns the directory holding the unpacked binary, unpacking it if needed.
fn prepare() -> io::Result<PathBuf> {
    let root = cache_root()?;
    let dir = root.join(format!("{NAME}-{HASH}"));
    if dir.join(READY_MARKER).is_file() {
        return Ok(dir);
    }
    fs::create_dir_all(&root)?;
    // unpack into a private directory, then rename it into place, so a concurrent first
    // run or an interrupted unpack never leaves a half-written directory behind
    let tmp = root.join(format!(".{NAME}-{HASH}.tmp{}", process::id()));
    let _ = fs::remove_dir_all(&tmp);
    let result = unpack(&tmp).and_then(|()| {
        if dir.exists() {
            fs::remove_dir_all(&dir)?; // incomplete earlier attempt
        }
        fs::rename(&tmp, &dir)
    });
    if let Err(e) = result {
        let _ = fs::remove_dir_all(&tmp);
        // another process may have finished first
        if dir.join(READY_MARKER).is_file() {
            return Ok(dir);
        }
        return Err(e);
    }
    Ok(dir)
}

fn unpack(into: &Path) -> io::Result<()> {
    let decoder = zstd::Decoder::new(Cursor::new(PAYLOAD))?;
    let mut archive = tar::Archive::new(decoder);
    archive.set_preserve_permissions(true);
    archive.unpack(into)?;
    fs::write(into.join(READY_MARKER), HASH)
}

fn cache_root() -> io::Result<PathBuf> {
    if let Some(dir) = env::var_os("TZ_BUNDLE_CACHE") {
        return Ok(PathBuf::from(dir));
    }
    let base = if cfg!(target_os = "macos") {
        env::var_os("HOME").map(|h| PathBuf::from(h).join("Library").join("Caches"))
    } else if cfg!(windows) {
        env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    };
    let base = base.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "no cache directory (set HOME or TZ_BUNDLE_CACHE)",
        )
    })?;
    Ok(base.join("truezero"))
}

#[cfg(unix)]
fn exec(mut cmd: Command) -> ! {
    use std::os::unix::process::CommandExt;
    // replaces this process, so signals and the exit status belong to the real binary
    let e = cmd.exec();
    eprintln!("{NAME}: failed to start bundled binary: {e}");
    process::exit(1);
}

#[cfg(not(unix))]
fn exec(mut cmd: Command) -> ! {
    match cmd.status() {
        Ok(status) => process::exit(status.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("{NAME}: failed to start bundled binary: {e}");
            process::exit(1);
        }
    }
}
