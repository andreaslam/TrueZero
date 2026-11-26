use std::{env, path::Path, process};
use tzrust::{data_path_str, mcts_trainer::NetSource, uci::run_uci};

/// Network compiled into the binary; chosen at build time with `TZ_DEFAULT_NET`
/// (see build.rs). Empty when that file was missing at build time.
static DEFAULT_NET: &[u8] = include_bytes!(env!("TZ_DEFAULT_NET_FILE"));

/// Usage: `ucimain [NET]`
///
/// Without an argument, uses the network embedded at build time, so the binary needs no
/// files beside it. `NET` overrides it: a path that exists as given (absolute, or relative
/// to the current directory) is used directly; otherwise it is resolved with `data_path()`,
/// so `nets/tz_42.pt` also works relative to the workspace root.
fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let source = match env::args().nth(1) {
        Some(arg) => {
            // a path that exists as given (e.g. relative to the current directory) wins;
            // otherwise resolve it against the workspace data root
            let model_path = if Path::new(&arg).is_file() {
                arg
            } else {
                data_path_str(&arg)
            };
            if !Path::new(&model_path).is_file() {
                eprintln!("ucimain: network file not found: {}", model_path);
                eprintln!(
                    "usage: ucimain [NET]  (NET is relative to the data root, e.g. nets/tz_42.pt)"
                );
                process::exit(1);
            }
            NetSource::Path(model_path)
        }
        None if !DEFAULT_NET.is_empty() => NetSource::Embedded(DEFAULT_NET),
        None => {
            eprintln!(
                "ucimain: built without an embedded net ({} was missing at build time)",
                env!("TZ_DEFAULT_NET_NAME")
            );
            eprintln!("usage: ucimain NET  (NET is relative to the data root, e.g. nets/tz_42.pt)");
            process::exit(1);
        }
    };
    run_uci(source);
}
