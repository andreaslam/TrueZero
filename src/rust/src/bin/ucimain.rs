use std::env;
use tzrust::{data_path_str, uci::run_uci};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let model_path = env::var("TZ_MODEL_PATH")
        .unwrap_or_else(|_| data_path_str("nets/tz_0.pt"));
    run_uci(&model_path);
}
