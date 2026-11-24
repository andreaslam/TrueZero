//! Settings and runs. The Python side (`src/python/runconfig.py`) implements the same rules
//! on the same files; keep the two in step (tests on both sides check the shared cases).
//!
//! - `truezero.example.toml` (repository root, committed): every setting and its default.
//! - `truezero.toml` (repository root, not committed): the user's settings. Only the keys
//!   that differ from the example are needed; it is created from the example on first use.
//!   An unknown key or a value of the wrong type is an error.
//! - `<runs_dir>/current_run`: the id of the current run, written only by the server.
//! - `<runs_dir>/<run id>/`: one folder per run (`nets/`, `games/`, `hidden/`, ...), with
//!   `run.json` (id, creation time, git commit, config hash) and `config.toml` (a copy of
//!   truezero.toml when the run was created).
//! - `<run>/hidden/run_log.jsonl`: one line per process start; a start with a different
//!   truezero.toml saves a copy to `<run>/hidden/config_history/`. Nothing is refused.
//! - `<run>/hidden/provenance/datagen.jsonl`: self-play provenance (see `main.rs`).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::{
    mcts_trainer::{EvalMode, TypeRequest},
    settings::{
        CPUCTSettings, FPUSettings, MovesLeftSettings, PSTSettings, PlayoutCapSettings,
        SearchSettings,
    },
};

/// The git commit this binary was built from (`build.rs`).
pub const GIT_SHA: &str = env!("TZ_GIT_SHA");
/// Whether tracked files had uncommitted changes when this binary was built.
pub const GIT_DIRTY: bool = matches!(env!("TZ_GIT_DIRTY").as_bytes(), b"1");

/// Exit code a client uses to ask the launcher to start it again (a new run began, or the
/// server went away). EX_TEMPFAIL from sysexits.h.
pub const RESTART_EXIT_CODE: i32 = 75;
/// Printed by a program just before it exits for a restart, so the launcher restarts it
/// even if the exit itself goes wrong (client.py prints it too).
pub const RESTART_SENTINEL: &str = "exiting so the launcher restarts";

pub const EXAMPLE_FILE: &str = "truezero.example.toml";
pub const CONFIG_FILE: &str = "truezero.toml";
pub const CURRENT_RUN_FILE: &str = "current_run";

/// Environment variables that used to configure TrueZero. They have no effect any more; a
/// start that finds one warns, so a value left in a PowerShell session is noticed.
pub const RETIRED_ENV_VARS: [(&str, &str); 8] = [
    ("TZ_NUM_GENERATORS", "[selfplay] num_generators"),
    ("TZ_VIRTUAL_LEAVES", "[selfplay] virtual_leaves"),
    ("TZ_PCR_FULL_PROB", "[selfplay] pcr_full_prob"),
    ("TZ_PCR_FAST_VISITS", "[selfplay] pcr_fast_visits"),
    ("TZ_PCR_EARLY_STOP", "[selfplay] pcr_early_stop"),
    ("TZ_NUM_EXECUTORS", "[selfplay] num_executors"),
    ("TZ_BATCH_TIMEOUT_US", "[selfplay] batch_timeout_us"),
    ("TZ_SERVER_PORT", "[server] port"),
];

/// The repository root (two levels above this crate).
/// `..` is resolved lexically, not canonicalised: on Windows canonicalising turns a mapped
/// drive (U:) into its UNC path, which some tools mishandle.
pub fn repo_root() -> PathBuf {
    crate::books::normalise(&Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(".."))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RunSection {
    pub name: String,
    pub seed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NetSection {
    pub arch: String,
    pub blocks: u32,
    pub channels: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerSection {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelfplaySection {
    pub num_generators: usize,
    pub virtual_leaves: usize,
    pub num_executors: usize,
    pub batch_size: usize,
    pub batch_timeout_us: u64,
    pub max_nodes: u32,
    pub root_fpu: f32,
    pub children_fpu: f32,
    pub root_c_puct: f32,
    pub children_c_puct: f32,
    pub root_pst: f32,
    pub children_pst: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,
    pub moves_left_weight: f32,
    pub moves_left_clip: f32,
    pub moves_left_sharpness: f32,
    pub pcr_full_prob: f32,
    pub pcr_fast_visits: u32,
    pub pcr_early_stop: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GateSection {
    pub openings: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LauncherSection {
    pub python: Vec<String>,
    pub venv: String,
    pub restart_on_crash: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PathsSection {
    pub runs_dir: String,
}

/// The effective settings. `[trainer]`, `[wandb]` and `[ladder]` belong to the Python
/// trainer, which checks them; here they only have to be tables.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub run: RunSection,
    pub net: NetSection,
    pub server: ServerSection,
    pub selfplay: SelfplaySection,
    pub gate: GateSection,
    pub trainer: toml::Table,
    pub wandb: toml::Table,
    pub ladder: toml::Table,
    pub launcher: LauncherSection,
    pub paths: PathsSection,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn type_name(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::String(_) => "a string",
        toml::Value::Integer(_) => "an integer",
        toml::Value::Float(_) => "a number",
        toml::Value::Boolean(_) => "true or false",
        toml::Value::Datetime(_) => "a date",
        toml::Value::Array(_) => "a list",
        toml::Value::Table(_) => "a table",
    }
}

/// The example's tables with the user's values laid over them. Every user key must exist
/// in the example with a value of the same type (an integer is accepted for a number).
pub fn merge(example: &toml::Table, user: &toml::Table) -> Result<toml::Table, String> {
    let mut merged = example.clone();
    let mut problems = Vec::new();
    for (table, values) in user {
        let (Some(toml::Value::Table(base)), toml::Value::Table(values)) =
            (merged.get_mut(table), values)
        else {
            problems.push(format!("unknown table [{table}]"));
            continue;
        };
        for (key, value) in values {
            let Some(default) = base.get(key) else {
                problems.push(format!("unknown key [{table}] {key}"));
                continue;
            };
            let value = match (default, value) {
                (toml::Value::Float(_), toml::Value::Integer(i)) => toml::Value::Float(*i as f64),
                _ => value.clone(),
            };
            if std::mem::discriminant(default) != std::mem::discriminant(&value) {
                problems.push(format!(
                    "[{table}] {key} must be {}, got {}",
                    type_name(default),
                    type_name(&value)
                ));
                continue;
            }
            base.insert(key.clone(), value);
        }
    }
    if problems.is_empty() {
        Ok(merged)
    } else {
        Err(problems.join("; "))
    }
}

/// Parses the example and the user's file and returns the checked effective settings.
pub fn config_from_texts(example: &str, user: &str) -> Result<Config, String> {
    let example: toml::Table =
        toml::from_str(example).map_err(|e| format!("{EXAMPLE_FILE}: {e}"))?;
    let user: toml::Table = toml::from_str(user).map_err(|e| format!("{CONFIG_FILE}: {e}"))?;
    let merged = merge(&example, &user).map_err(|e| format!("{CONFIG_FILE}: {e}"))?;
    let cfg: Config = toml::Value::Table(merged)
        .try_into()
        .map_err(|e| format!("{CONFIG_FILE}: {e}"))?;
    cfg.validate().map_err(|e| format!("{CONFIG_FILE}: {e}"))?;
    Ok(cfg)
}

/// The settings, with the SHA-256 of truezero.toml's bytes (what provenance records).
#[derive(Clone, Debug)]
pub struct LoadedConfig {
    pub config: Config,
    pub user_bytes: Vec<u8>,
    pub config_sha256: String,
}

/// Creates `truezero.toml` from the example if it does not exist yet.
pub fn ensure_user_config(root: &Path) -> Result<(), String> {
    let path = root.join(CONFIG_FILE);
    if path.exists() {
        return Ok(());
    }
    let example = fs::read(root.join(EXAMPLE_FILE))
        .map_err(|e| format!("cannot read {}: {e}", root.join(EXAMPLE_FILE).display()))?;
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut f) => {
            f.write_all(&example)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            println!(
                "[config] Created {} from {EXAMPLE_FILE}; edit it to change any setting.",
                path.display()
            );
            Ok(())
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// Loads `truezero.toml` (creating it if needed) over `truezero.example.toml` in `root`.
pub fn load_config_in(root: &Path) -> Result<LoadedConfig, String> {
    ensure_user_config(root)?;
    let example = fs::read_to_string(root.join(EXAMPLE_FILE))
        .map_err(|e| format!("cannot read {}: {e}", root.join(EXAMPLE_FILE).display()))?;
    let path = root.join(CONFIG_FILE);
    let user_bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let user = std::str::from_utf8(&user_bytes).map_err(|e| format!("{CONFIG_FILE}: {e}"))?;
    let config = config_from_texts(&example, user)?;
    Ok(LoadedConfig {
        config_sha256: sha256_hex(&user_bytes),
        config,
        user_bytes,
    })
}

/// Loads the settings, or exits with the reason.
pub fn load_config() -> LoadedConfig {
    load_config_in(&repo_root()).unwrap_or_else(|e| {
        eprintln!("[config] {e}");
        std::process::exit(2)
    })
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.run.name.trim().is_empty() || self.run.name.contains(['/', '\\', ':']) {
            return Err("[run] name must be non-empty and contain no / \\ or :".into());
        }
        if !matches!(self.net.arch.as_str(), "TrueNet" | "TrueNetXS") {
            return Err("[net] arch must be \"TrueNet\" or \"TrueNetXS\"".into());
        }
        if self.net.blocks == 0 || self.net.channels == 0 {
            return Err("[net] blocks and channels must be > 0".into());
        }
        let s = &self.selfplay;
        if s.num_generators == 0
            || s.virtual_leaves == 0
            || s.max_nodes == 0
            || s.num_executors == 0
            || s.batch_size == 0
        {
            return Err("[selfplay] num_generators, virtual_leaves, num_executors, batch_size \
                        and max_nodes must be > 0"
                .into());
        }
        if !(0.0..=1.0).contains(&s.pcr_full_prob) {
            return Err("[selfplay] pcr_full_prob must be in [0, 1]".into());
        }
        if s.pcr_full_prob < 1.0 && s.pcr_fast_visits < 2 {
            return Err("[selfplay] pcr_fast_visits must be at least 2".into());
        }
        if self.launcher.python.is_empty() {
            return Err("[launcher] python must name a program".into());
        }
        Ok(())
    }

    /// The self-play search settings.
    pub fn search_settings(&self) -> SearchSettings {
        let s = &self.selfplay;
        SearchSettings {
            fpu: FPUSettings {
                root_fpu: s.root_fpu,
                children_fpu: s.children_fpu,
            },
            wdl: EvalMode::Wdl,
            moves_left: Some(MovesLeftSettings {
                moves_left_weight: s.moves_left_weight,
                moves_left_clip: s.moves_left_clip,
                moves_left_sharpness: s.moves_left_sharpness,
            }),
            c_puct: CPUCTSettings {
                root_c_puct: s.root_c_puct,
                children_c_puct: s.children_c_puct,
            },
            max_nodes: Some(s.max_nodes as u128),
            alpha: s.dirichlet_alpha,
            eps: s.dirichlet_eps,
            search_type: TypeRequest::TrainerSearch(None),
            pst: PSTSettings {
                root_pst: s.root_pst,
                children_pst: s.children_pst,
            },
            batch_size: 1,
        }
    }

    /// Playout cap randomisation, `None` when every move is a full search.
    pub fn playout_cap(&self) -> Option<PlayoutCapSettings> {
        let s = &self.selfplay;
        (s.pcr_full_prob < 1.0)
            .then(|| PlayoutCapSettings::new(s.pcr_full_prob, s.pcr_fast_visits, s.pcr_early_stop))
    }

    /// Where clients connect: `[server] host`, or `$TZ_SERVER_HOST` (for Docker, where the
    /// server is another container), and `[server] port`.
    pub fn server_addr(&self) -> String {
        let host = std::env::var("TZ_SERVER_HOST").unwrap_or_else(|_| self.server.host.clone());
        format!("{host}:{}", self.server.port)
    }

    /// The folder holding the runs.
    pub fn runs_dir(&self, root: &Path) -> PathBuf {
        root.join(&self.paths.runs_dir)
    }

    /// A path from the config that is relative to the repository root.
    pub fn repo_path(&self, root: &Path, rel: &str) -> PathBuf {
        root.join(rel)
    }
}

// ---------------------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------------------

/// `YYYYMMDD-HHMM` in UTC for a Unix time.
pub fn utc_stamp(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let secs = unix_secs % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}{m:02}{d:02}-{:02}{:02}",
        secs / 3600,
        (secs % 3600) / 60
    )
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// The current run's id, if the server has created one.
pub fn current_run_id(runs_dir: &Path) -> Option<String> {
    fs::read_to_string(runs_dir.join(CURRENT_RUN_FILE))
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RunRecord {
    pub run_id: String,
    pub name: String,
    pub created_at: f64,
    pub git_sha: String,
    pub git_dirty: bool,
    pub config_sha256: String,
}

/// Creates a new run folder `<name>-<YYYYMMDD-HHMM>` (with `-2`, `-3`, ... if that exists),
/// records the commit and settings in it, and makes it the current run. Earlier runs are
/// left exactly as they are.
pub fn create_run(runs_dir: &Path, loaded: &LoadedConfig, unix_secs: u64) -> io::Result<String> {
    fs::create_dir_all(runs_dir)?;
    let base = format!("{}-{}", loaded.config.run.name, utc_stamp(unix_secs));
    let mut run_id = base.clone();
    let mut n = 1;
    let dir = loop {
        let dir = runs_dir.join(&run_id);
        match fs::create_dir(&dir) {
            Ok(()) => break dir,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                n += 1;
                run_id = format!("{base}-{n}");
            }
            Err(e) => return Err(e),
        }
    };
    for sub in ["nets", "games", "hidden"] {
        fs::create_dir_all(dir.join(sub))?;
    }
    fs::write(dir.join("config.toml"), &loaded.user_bytes)?;
    let record = RunRecord {
        run_id: run_id.clone(),
        name: loaded.config.run.name.clone(),
        created_at: now_secs(),
        git_sha: GIT_SHA.to_owned(),
        git_dirty: GIT_DIRTY,
        config_sha256: loaded.config_sha256.clone(),
    };
    fs::write(
        dir.join("run.json"),
        serde_json::to_string_pretty(&record).expect("serialise run record"),
    )?;
    // Replace current_run in one step, so a reader never sees half a name.
    let tmp = runs_dir.join(format!("{CURRENT_RUN_FILE}.tmp"));
    fs::write(&tmp, format!("{run_id}\n"))?;
    fs::rename(&tmp, runs_dir.join(CURRENT_RUN_FILE))?;
    Ok(run_id)
}

/// The folder the programs read and write (`nets/`, `games/`, `hidden/`, ...):
/// `$TZ_DATA_ROOT` if set (pins it, e.g. in Docker), else the current run's folder, else the
/// repository root (no run yet; for tools such as `ucimain`).
pub fn resolve_data_root() -> PathBuf {
    if let Some(root) = std::env::var_os("TZ_DATA_ROOT") {
        return PathBuf::from(root);
    }
    let root = repo_root();
    let runs_dir = match load_config_in(&root) {
        Ok(l) => l.config.runs_dir(&root),
        Err(_) => root.join("runs"),
    };
    match current_run_id(&runs_dir) {
        Some(id) => runs_dir.join(id),
        None => root,
    }
}

static DATA_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// The data root for this process, fixed at first use.
pub fn data_root() -> &'static Path {
    DATA_ROOT.get_or_init(resolve_data_root)
}

/// A warning naming every retired environment variable that is set, or `None`.
pub fn retired_env_var_warning() -> Option<String> {
    let set: Vec<String> = RETIRED_ENV_VARS
        .iter()
        .filter(|(name, _)| std::env::var_os(name).is_some())
        .map(|(name, key)| format!("{name} (now truezero.toml {key})"))
        .collect();
    (!set.is_empty()).then(|| {
        format!(
            "IGNORING environment variables that no longer configure TrueZero: {}. \
             truezero.toml decides; remove them (PowerShell: Remove-Item Env:NAME) to \
             silence this.",
            set.join(", ")
        )
    })
}

/// Who is starting, with what code and settings.
#[derive(Clone, Debug)]
pub struct StartIdentity {
    pub run_id: String,
    pub run_name: String,
    pub config_sha256: String,
    pub git_sha: String,
    pub git_dirty: bool,
    pub process: String,
}

/// Appends this start to `hidden/run_log.jsonl` and, if truezero.toml differs from the one
/// the run was created with, keeps a copy in `hidden/config_history/`. Returns the
/// warnings to print. Never refuses a start.
pub fn record_start(run_dir: &Path, id: &StartIdentity, user_bytes: &[u8]) -> Vec<String> {
    let hidden = run_dir.join("hidden");
    let mut notes = Vec::new();
    if id.git_dirty {
        notes.push("tracked files have uncommitted changes (logged as git_dirty)".to_owned());
    }
    let record: Option<RunRecord> = fs::read(run_dir.join("run.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    if let Some(r) = &record {
        if r.git_sha != id.git_sha {
            notes.push(format!(
                "the run was created at commit {} and this program is at {} (logged)",
                r.git_sha, id.git_sha
            ));
        }
        if r.config_sha256 != id.config_sha256 {
            let copy = hidden
                .join("config_history")
                .join(format!("{}.toml", &id.config_sha256[..12]));
            if !copy.exists() {
                let _ = fs::create_dir_all(copy.parent().unwrap());
                if let Err(e) = fs::write(&copy, user_bytes) {
                    eprintln!("[run][Warning] could not save {}: {e}", copy.display());
                }
            }
            notes.push(format!(
                "truezero.toml changed since the run started (copy kept in {})",
                copy.display()
            ));
        }
    }
    append_jsonl(
        &hidden.join("run_log.jsonl"),
        &serde_json::json!({
            "time": now_secs(),
            "process": id.process,
            "pid": std::process::id(),
            "run_id": id.run_id,
            "git_sha": id.git_sha,
            "git_dirty": id.git_dirty,
            "config_sha256": id.config_sha256,
        }),
    );
    notes
}

/// Appends one JSON line; an I/O error is reported, never fatal (a network drive hiccup
/// must not stop self-play).
pub fn append_jsonl(path: &Path, value: &serde_json::Value) {
    let result = (|| {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut f = OpenOptions::new().create(true).append(true).open(path)?;
        f.write_all(format!("{value}\n").as_bytes())
    })();
    if let Err(e) = result {
        eprintln!("[run][Warning] could not append to {}: {e}", path.display());
    }
}

/// Start-up for a client (self-play, gate): loads the settings, waits for the server to
/// have created a run, fixes this process's data root to it and records the start.
pub fn start_client(process: &str) -> (LoadedConfig, StartIdentity) {
    if let Some(w) = retired_env_var_warning() {
        eprintln!("[run][Warning] {w}");
    }
    let loaded = load_config();
    let root = repo_root();
    let run_dir = if let Some(pinned) = std::env::var_os("TZ_DATA_ROOT") {
        PathBuf::from(pinned)
    } else {
        let runs_dir = loaded.config.runs_dir(&root);
        let mut waited = false;
        loop {
            if let Some(id) = current_run_id(&runs_dir) {
                break runs_dir.join(id);
            }
            if !waited {
                println!(
                    "[run] No run yet in {}; waiting for the server to create one.",
                    runs_dir.display()
                );
                waited = true;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    };
    let fixed = DATA_ROOT.get_or_init(|| run_dir.clone());
    assert_eq!(fixed, &run_dir, "data root used before start_client");
    let run_id = run_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let id = StartIdentity {
        run_id,
        run_name: loaded.config.run.name.clone(),
        config_sha256: loaded.config_sha256.clone(),
        git_sha: GIT_SHA.to_owned(),
        git_dirty: GIT_DIRTY,
        process: process.to_owned(),
    };
    let notes = record_start(&run_dir, &id, &loaded.user_bytes);
    println!(
        "[run] run {} in {}, settings sha256 {}, commit {}{}",
        id.run_id,
        run_dir.display(),
        &id.config_sha256[..12],
        id.git_sha,
        if id.git_dirty { " (dirty)" } else { "" }
    );
    for n in &notes {
        println!("[run][Warning] {n}");
    }
    (loaded, id)
}

/// Called when the server says the run changed (or it went away): exit so the launcher
/// starts this program again on the current run.
pub fn exit_for_restart(why: &str) -> ! {
    println!("[run] {why}; {RESTART_SENTINEL} this program.");
    let _ = io::stdout().flush();
    std::process::exit(RESTART_EXIT_CODE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_text() -> String {
        fs::read_to_string(repo_root().join(EXAMPLE_FILE)).expect("example must exist")
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tz-runs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The example's self-play settings are the ones self-play used before this file.
    #[test]
    fn example_matches_previous_selfplay_settings() {
        let cfg = config_from_texts(&example_text(), "").unwrap();
        let expected = SearchSettings {
            fpu: FPUSettings {
                root_fpu: 1.0,
                children_fpu: 0.5,
            },
            wdl: EvalMode::Wdl,
            moves_left: Some(MovesLeftSettings {
                moves_left_weight: 0.05,
                moves_left_clip: 20.0,
                moves_left_sharpness: 0.5,
            }),
            c_puct: CPUCTSettings {
                root_c_puct: 2.0,
                children_c_puct: 2.0,
            },
            max_nodes: Some(1600),
            alpha: 0.03,
            eps: 0.25,
            search_type: TypeRequest::TrainerSearch(None),
            pst: PSTSettings {
                root_pst: 1.5,
                children_pst: 1.5,
            },
            batch_size: 1,
        };
        assert_eq!(cfg.search_settings(), expected);
        assert_eq!(
            cfg.playout_cap(),
            Some(PlayoutCapSettings::new(0.25, 200, true))
        );
        assert_eq!(cfg.server_addr(), "127.0.0.1:38475");
        assert_eq!(cfg.selfplay.num_executors, 2);
        assert_eq!(cfg.selfplay.batch_size, 2048);
        assert_eq!(cfg.selfplay.batch_timeout_us, 1);
    }

    /// Shared with tests/test_run_config.py: the same user files must give the same
    /// verdicts and values on both sides.
    #[test]
    fn overlay_cases_shared_with_python() {
        let ex = example_text();
        let path = repo_root().join("tests/fixtures/config_overlay_cases.json");
        let cases: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        for case in cases["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let result = config_from_texts(&ex, case["user"].as_str().unwrap());
            assert_eq!(result.is_ok(), case["ok"].as_bool().unwrap(), "{name}: {result:?}");
            let Ok(cfg) = result else { continue };
            let effective = serde_json::to_value(&cfg).unwrap();
            for e in case["expect"].as_array().into_iter().flatten() {
                let (t, k) = (e[0].as_str().unwrap(), e[1].as_str().unwrap());
                let got = &effective[t][k];
                let want = &e[2];
                let same = match (got.as_f64(), want.as_f64()) {
                    (Some(g), Some(w)) => (g - w).abs() < 1e-9,
                    _ => got == want,
                };
                assert!(same, "{name}: [{t}] {k} = {got}, want {want}");
            }
        }
        let off = config_from_texts(&ex, "[selfplay]\npcr_full_prob = 1.0\n").unwrap();
        assert_eq!(off.playout_cap(), None);
    }

    /// The wire form tests/test_run_config.py feeds to client.py's run_change.
    #[test]
    fn run_messages_on_the_wire() {
        use crate::message_types::{MessageServer, MessageType};
        let info = MessageServer {
            purpose: MessageType::RunInfo("r-20261124-1503".into()),
        };
        assert_eq!(
            serde_json::to_string(&info).unwrap(),
            r#"{"purpose":{"RunInfo":"r-20261124-1503"}}"#
        );
        let reset = MessageServer {
            purpose: MessageType::ResetRun("r2".into()),
        };
        assert_eq!(
            serde_json::to_string(&reset).unwrap(),
            r#"{"purpose":{"ResetRun":"r2"}}"#
        );
    }

    #[test]
    fn utc_stamps() {
        assert_eq!(utc_stamp(0), "19700101-0000");
        // 2026-11-24 15:03:59 UTC
        assert_eq!(utc_stamp(1_795_532_639), "20261124-1503");
        // 2000-02-29 23:59 UTC (leap day)
        assert_eq!(utc_stamp(951_868_799), "20000229-2359");
    }

    #[test]
    fn creating_runs_never_touches_earlier_ones() {
        let root = temp_dir("create");
        fs::write(root.join(EXAMPLE_FILE), example_text()).unwrap();
        let loaded = load_config_in(&root).unwrap();
        assert!(root.join(CONFIG_FILE).exists(), "truezero.toml is created");
        let runs = loaded.config.runs_dir(&root);
        assert_eq!(current_run_id(&runs), None);

        let a = create_run(&runs, &loaded, 1_795_532_639).unwrap();
        assert_eq!(a, "truenet-128x10-20261124-1503");
        fs::write(runs.join(&a).join("nets").join("tz_0.pt"), b"x").unwrap();
        let b = create_run(&runs, &loaded, 1_795_532_639).unwrap();
        assert_eq!(b, "truenet-128x10-20261124-1503-2");
        assert_eq!(current_run_id(&runs).as_deref(), Some(b.as_str()));
        assert!(runs.join(&a).join("nets").join("tz_0.pt").exists());
        let rec: RunRecord =
            serde_json::from_slice(&fs::read(runs.join(&b).join("run.json")).unwrap()).unwrap();
        assert_eq!(rec.config_sha256, loaded.config_sha256);
        assert_eq!(fs::read(runs.join(&b).join("config.toml")).unwrap(), loaded.user_bytes);
    }

    #[test]
    fn a_changed_config_is_logged_not_refused() {
        let root = temp_dir("changed");
        fs::write(root.join(EXAMPLE_FILE), example_text()).unwrap();
        let loaded = load_config_in(&root).unwrap();
        let runs = loaded.config.runs_dir(&root);
        let run = runs.join(create_run(&runs, &loaded, 0).unwrap());
        let mut id = StartIdentity {
            run_id: "r".into(),
            run_name: "n".into(),
            config_sha256: loaded.config_sha256.clone(),
            git_sha: GIT_SHA.into(),
            git_dirty: false,
            process: "test".into(),
        };
        assert!(record_start(&run, &id, &loaded.user_bytes).is_empty());
        id.config_sha256 = "b".repeat(64);
        let notes = record_start(&run, &id, b"[run]\nseed = 1\n");
        assert_eq!(notes.len(), 1);
        assert!(run.join("hidden/config_history/bbbbbbbbbbbb.toml").exists());
        let log = fs::read_to_string(run.join("hidden/run_log.jsonl")).unwrap();
        assert_eq!(log.lines().count(), 2);
    }
}
