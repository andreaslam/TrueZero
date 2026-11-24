use std::{env, path::Path, process};
use tzrust::{data_path_str, uci::run_uci};

/// Usage: `ucimain [NET]`
///
/// `NET` is resolved with `data_path()`, so a relative path such as
/// `nets/tz_42.pt` is taken relative to the data root (the current run's folder,
/// `runs/<run>/`, or `TZ_DATA_ROOT`), and an absolute path is used as given.
/// Without an argument, falls back to `TZ_MODEL_PATH`, then `nets/tz_0.pt`.
fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let model_path = match env::args().nth(1) {
        Some(arg) => data_path_str(&arg),
        None => env::var("TZ_MODEL_PATH").unwrap_or_else(|_| data_path_str("nets/tz_0.pt")),
    };
    if !Path::new(&model_path).is_file() {
        eprintln!("ucimain: network file not found: {}", model_path);
        eprintln!("usage: ucimain [NET]  (NET is relative to the data root, e.g. nets/tz_42.pt)");
        process::exit(1);
    }
    run_uci(&model_path);
}
