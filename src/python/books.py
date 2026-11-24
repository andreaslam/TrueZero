"""Fetch an opening book from https://github.com/official-stockfish/books if it is missing.

    uv run python src/python/books.py [DEST]

DEST defaults to truezero.toml [gate] openings (books/8moves_v3.epd). The book is
downloaded as <name>.zip and unzipped to DEST. If the repository only has the PGN version
(8moves_v3 is published as 8moves_v3.pgn.zip), that is downloaded instead and each game's
final position is written as one full FEN per line, which is what trainenginetest reads
(the same as the repository's pgn2epd.py, plus move counters). The launcher runs this
before the gate starts.
"""

import io
import os
import urllib.error
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BOOKS_URL = "https://github.com/official-stockfish/books/raw/master/{name}.zip"
REPO_ROOT = Path(__file__).resolve().parents[2]


def extract_book(zip_bytes, name, dest):
    """Write the member `name` of the zip archive to `dest` (atomically)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        member = next((m for m in z.namelist() if Path(m).name == name), None)
        if member is None:
            raise ValueError(f"{name} is not in the archive (has {z.namelist()})")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        with z.open(member) as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out)
        os.replace(tmp, dest)


def pgn_to_fens(pgn_text):
    """The final position of every game in a PGN, as full FENs (needs python-chess, a
    TrueZero dependency)."""
    import chess.pgn

    fens = []
    stream = io.StringIO(pgn_text)
    while (game := chess.pgn.read_game(stream)) is not None:
        fens.append(game.end().board().fen())
    return fens


def extract_pgn_book(zip_bytes, pgn_name, dest):
    """Unzip `pgn_name` and write its games' final positions to `dest` (atomically)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        member = next((m for m in z.namelist() if Path(m).name == pgn_name), None)
        if member is None:
            raise ValueError(f"{pgn_name} is not in the archive (has {z.namelist()})")
        text = z.read(member).decode("utf-8", errors="replace")
    fens = pgn_to_fens(text)
    if not fens:
        raise ValueError(f"no games in {pgn_name}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text("\n".join(fens) + "\n")
    os.replace(tmp, dest)
    return len(fens)


def ensure_book(dest, fetch=None):
    """Make sure `dest` exists. Copies a book left at the old location (hidden/ in the
    repository root) if there is one, otherwise downloads it. Returns how it got there."""
    dest = Path(dest)
    if dest.is_file():
        return "present"
    legacy = REPO_ROOT / "hidden" / dest.name
    if legacy.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy, dest)
        return f"copied from {legacy}"
    if fetch is None:

        def fetch(u):
            with urllib.request.urlopen(u, timeout=120) as r:
                return r.read()

    url = BOOKS_URL.format(name=dest.name)
    try:
        extract_book(fetch(url), dest.name, dest)
        return f"downloaded from {url}"
    except urllib.error.HTTPError as e:
        if e.code != 404 or dest.suffix != ".epd":
            raise
    pgn_name = dest.stem + ".pgn"
    url = BOOKS_URL.format(name=pgn_name)
    n = extract_pgn_book(fetch(url), pgn_name, dest)
    return f"converted {n} games from {url}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        dest = Path(argv[0])
    else:
        import runconfig

        cfg = runconfig.load_config().config
        dest = REPO_ROOT / cfg["gate"]["openings"]
    try:
        how = ensure_book(dest)
    except Exception as e:  # report and fail; the launcher explains what to do
        print(f"[books] could not get {dest}: {e!r}", file=sys.stderr)
        return 1
    print(f"[books] {dest}: {how}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
