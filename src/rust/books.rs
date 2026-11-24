//! Opening books for the gate and the ladder, fetched from
//! <https://github.com/official-stockfish/books> when missing.
//!
//! The file named in truezero.toml (e.g. `books/8moves_v3.epd`) is looked for in this order:
//! 1. already there: nothing to do;
//! 2. at the old location, `hidden/<name>` in the repository root: copied;
//! 3. `<name>.zip` in official-stockfish/books: downloaded and unzipped;
//! 4. `<stem>.pgn.zip` (8moves_v3 is only published as a PGN): downloaded, and the final
//!    position of every game is written as one full FEN per line, which is what
//!    `trainenginetest` reads.
//!
//! Everything is done here in Rust so that a fresh clone needs nothing but `cargo`.

use std::{
    fs,
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

use cozy_chess::{util::parse_san_move, Board};

pub const BOOKS_URL: &str = "https://github.com/official-stockfish/books/raw/master";

/// Make sure the book at `dest` exists. `repo_root` is where the old `hidden/` folder
/// lives. Returns a short description of what was done, for the log.
pub fn ensure_book(dest: &Path, repo_root: &Path) -> Result<String, String> {
    ensure_book_with(dest, repo_root, download)
}

/// [`ensure_book`] with the download swapped out (for tests). `fetch(url)` returns
/// `Ok(None)` when the server says the file does not exist.
pub fn ensure_book_with(
    dest: &Path,
    repo_root: &Path,
    fetch: impl Fn(&str) -> Result<Option<Vec<u8>>, String>,
) -> Result<String, String> {
    if dest.is_file() {
        return Ok("present".into());
    }
    let name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("{} is not a file name", dest.display()))?
        .to_owned();
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }

    let legacy = repo_root.join("hidden").join(&name);
    if legacy.is_file() {
        let tmp = tmp_path(dest);
        fs::copy(&legacy, &tmp).map_err(|e| format!("cannot copy {}: {e}", legacy.display()))?;
        replace(&tmp, dest)?;
        return Ok(format!("copied from {}", legacy.display()));
    }

    let url = format!("{BOOKS_URL}/{name}.zip");
    if let Some(zip) = fetch(&url)? {
        let bytes = unzip_member(&zip, &name)?;
        write_atomic(dest, &bytes)?;
        return Ok(format!("downloaded {url}"));
    }

    // Not published under that name. An .epd may still be published as a PGN.
    let stem = Path::new(&name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&name)
        .to_owned();
    let pgn_name = format!("{stem}.pgn");
    let pgn_url = format!("{BOOKS_URL}/{pgn_name}.zip");
    if name.ends_with(".epd") {
        if let Some(zip) = fetch(&pgn_url)? {
            let pgn = unzip_member(&zip, &pgn_name)?;
            let fens = pgn_to_fens(&String::from_utf8_lossy(&pgn))?;
            if fens.is_empty() {
                return Err(format!("{pgn_url} has no games"));
            }
            let mut text = fens.join("\n");
            text.push('\n');
            write_atomic(dest, text.as_bytes())?;
            return Ok(format!("downloaded {pgn_url} and converted {} games to FENs", fens.len()));
        }
    }
    Err(format!("neither {url} nor {pgn_url} exists"))
}

/// GET `url`. `Ok(None)` on 404. Honours HTTPS_PROXY / HTTP_PROXY.
fn download(url: &str) -> Result<Option<Vec<u8>>, String> {
    let agent = ureq::AgentBuilder::new()
        .try_proxy_from_env(true)
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(300))
        .build();
    match agent.get(url).call() {
        Ok(resp) => {
            let mut bytes = Vec::new();
            resp.into_reader()
                .read_to_end(&mut bytes)
                .map_err(|e| format!("downloading {url}: {e}"))?;
            Ok(Some(bytes))
        }
        Err(ureq::Error::Status(404, _)) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

/// The contents of the archive member whose file name is `name` (in any folder).
pub fn unzip_member(zip_bytes: &[u8], name: &str) -> Result<Vec<u8>, String> {
    let mut archive =
        zip::ZipArchive::new(Cursor::new(zip_bytes)).map_err(|e| format!("bad zip archive: {e}"))?;
    let mut members = Vec::new();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| format!("bad zip archive: {e}"))?;
        let member = file.name().to_owned();
        if Path::new(&member).file_name().and_then(|n| n.to_str()) == Some(name) {
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)
                .map_err(|e| format!("cannot unzip {member}: {e}"))?;
            return Ok(bytes);
        }
        members.push(member);
    }
    Err(format!("{name} is not in the archive (it has {members:?})"))
}

/// The final position of every game in a PGN, as full FENs. Tags, comments, NAGs, move
/// numbers, variations and results are skipped; every move must be legal SAN.
pub fn pgn_to_fens(pgn: &str) -> Result<Vec<String>, String> {
    let mut fens = Vec::new();
    let mut board: Option<Board> = None;
    let mut game = 0usize;
    for line in pgn.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            // A tag starts a new game (if the previous one had no result token).
            if let Some(b) = board.take() {
                fens.push(b.to_string());
            }
            continue;
        }
        // Strip comments and variations (8moves_v3 has none; be tolerant anyway).
        let mut text = String::with_capacity(line.len());
        let mut depth = 0i32;
        for c in line.chars() {
            match c {
                '{' | '(' => depth += 1,
                '}' | ')' => depth -= 1,
                _ if depth == 0 => text.push(c),
                _ => {}
            }
        }
        if line.starts_with(';') {
            continue;
        }
        for token in text.split_whitespace() {
            if matches!(token, "1-0" | "0-1" | "1/2-1/2" | "*") {
                if let Some(b) = board.take() {
                    fens.push(b.to_string());
                }
                continue;
            }
            if token.starts_with('$') {
                continue;
            }
            // "12." / "12..." / "12.e4"
            let san = token.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
            let san = san.trim_end_matches(['!', '?']);
            if san.is_empty() {
                continue;
            }
            let san = san.replace('0', "O");
            let b = board.get_or_insert_with(|| {
                game += 1;
                Board::default()
            });
            let mv = parse_san_move(b, &san)
                .map_err(|_| format!("game {game}: cannot parse move {token} in {b}"))?;
            b.try_play(mv)
                .map_err(|_| format!("game {game}: illegal move {token} in {b}"))?;
        }
    }
    if let Some(b) = board {
        fens.push(b.to_string());
    }
    Ok(fens)
}

/// `path` with `.` and `..` resolved lexically (no file system access, so a network drive
/// keeps its drive letter). For readable messages.
pub fn normalise(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn tmp_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    dest.with_file_name(name)
}

fn replace(tmp: &Path, dest: &Path) -> Result<(), String> {
    fs::rename(tmp, dest).map_err(|e| format!("cannot write {}: {e}", dest.display()))
}

fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = tmp_path(dest);
    fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    replace(&tmp, dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, io::Write};

    fn zip_of(name: &str, contents: &str) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            w.start_file(name, zip::write::FileOptions::default()).unwrap();
            w.write_all(contents.as_bytes()).unwrap();
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tz_books_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn pgn_final_positions_match_python_chess() {
        // Same game and expected FEN as tests/test_run_config.py (python-chess).
        let pgn = "[Event \"?\"]\n[FEN \"x\"]\n\n1. e4 e5 2. Nf3 Nc6 *\n\n\
                   [Event \"?\"]\n\n1. d4 {book} d5 2. c4 $1 dxc4 3. O-O-O?? 1-0\n";
        // Second game has an illegal castle: must be an error, not a silent skip.
        assert!(pgn_to_fens(pgn).is_err());

        let pgn = "[Event \"?\"]\n\n1. e4 e5 2. Nf3 Nc6 *\n\n[Event \"?\"]\n\n\
                   1. d4 {book} d5 2. c4+ $1 dxc4!? 1/2-1/2\n";
        let fens = pgn_to_fens(pgn).unwrap();
        assert_eq!(
            fens,
            vec![
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
            ]
        );
        // trainenginetest reads them back with Board::from_fen.
        for f in &fens {
            Board::from_fen(f, false).unwrap();
        }
    }

    #[test]
    fn epd_falls_back_to_the_published_pgn() {
        let dir = temp_dir("pgn");
        let dest = dir.join("books").join("8moves_v3.epd");
        let asked = RefCell::new(Vec::new());
        let how = ensure_book_with(&dest, &dir, |url| {
            asked.borrow_mut().push(url.to_owned());
            if url.ends_with("8moves_v3.pgn.zip") {
                Ok(Some(zip_of("8moves_v3.pgn", "[E \"?\"]\n\n1. e4 e5 2. Nf3 Nc6 *\n")))
            } else {
                Ok(None)
            }
        })
        .unwrap();
        assert!(how.contains("converted 1 games"), "{how}");
        assert_eq!(
            *asked.borrow(),
            vec![
                format!("{BOOKS_URL}/8moves_v3.epd.zip"),
                format!("{BOOKS_URL}/8moves_v3.pgn.zip")
            ]
        );
        assert_eq!(
            fs::read_to_string(&dest).unwrap(),
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\n"
        );
        // Present now: no download.
        let again = ensure_book_with(&dest, &dir, |_| panic!("should not download")).unwrap();
        assert_eq!(again, "present");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_published_epd_is_unzipped_and_a_legacy_copy_wins() {
        let dir = temp_dir("epd");
        let dest = dir.join("books").join("UHO.epd");
        ensure_book_with(&dest, &dir, |url| {
            assert!(url.ends_with("/UHO.epd.zip"));
            Ok(Some(zip_of("sub/UHO.epd", "8/8/8/8/8/8/8/K1k5 w - - 0 1\n")))
        })
        .unwrap();
        assert_eq!(fs::read_to_string(&dest).unwrap(), "8/8/8/8/8/8/8/K1k5 w - - 0 1\n");

        let dest = dir.join("books").join("old.epd");
        fs::create_dir_all(dir.join("hidden")).unwrap();
        fs::write(dir.join("hidden").join("old.epd"), "legacy\n").unwrap();
        let how = ensure_book_with(&dest, &dir, |_| panic!("should not download")).unwrap();
        assert!(how.starts_with("copied from"), "{how}");
        assert_eq!(fs::read_to_string(&dest).unwrap(), "legacy\n");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_everywhere_is_an_error() {
        let dir = temp_dir("none");
        let err = ensure_book_with(&dir.join("nope.epd"), &dir, |_| Ok(None)).unwrap_err();
        assert!(err.contains("nope.pgn.zip"), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn paths_are_normalised_lexically() {
        let p = Path::new("U:/TrueZero/src/rust/../../books/8moves_v3.epd");
        assert_eq!(normalise(p), PathBuf::from("U:/TrueZero/books/8moves_v3.epd"));
    }
}
