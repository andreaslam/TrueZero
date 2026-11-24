//! Run identity: the frozen `run.toml`, the run lock that pins a run to one git commit,
//! and the per-run start log. The Python trainer (`src/python/runconfig.py`) implements
//! the same lock rules on the same files; keep the two in step.
//!
//! - `<data root>/run.toml`: the run's hyperparameters (see `run.example.toml`).
//! - `<data root>/hidden/run_lock.json`: written by whichever process starts first; records
//!   the run_id, the SHA-256 of run.toml's bytes and the git commit.
//! - `<data root>/hidden/run_log.jsonl`: one line per process start.
//! - `<data root>/hidden/provenance/datagen.jsonl`: self-play provenance (see `main.rs`).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    data_path,
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

/// Environment variables that used to set run hyperparameters. They have no effect any more;
/// a start that finds one warns, so a value left in a PowerShell session is noticed but can
/// neither change a run nor block a start.
/// (The trainer refuses its own, TZ_LR_SCHEDULE and TZ_LR_WARMUP_SAMPLES.)
pub const RETIRED_ENV_VARS: [(&str, &str); 5] = [
    ("TZ_NUM_GENERATORS", "[selfplay] num_generators"),
    ("TZ_VIRTUAL_LEAVES", "[selfplay] virtual_leaves"),
    ("TZ_PCR_FULL_PROB", "[selfplay] pcr_full_prob"),
    ("TZ_PCR_FAST_VISITS", "[selfplay] pcr_fast_visits"),
    ("TZ_PCR_EARLY_STOP", "[selfplay] pcr_early_stop"),
];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RunSection {
    pub run_id: String,
    pub parent_run_id: String,
    pub fork_sample: u64,
    pub seed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelfplaySection {
    pub num_generators: usize,
    pub virtual_leaves: usize,
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

/// The whole file. `[trainer]` belongs to the Python trainer, which validates it; here it
/// only has to be a table.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub run: RunSection,
    pub selfplay: SelfplaySection,
    pub trainer: toml::Table,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn run_config_path() -> PathBuf {
    data_path("run.toml")
}

/// Parses and checks run.toml. Returns the config and the SHA-256 of the file's bytes.
pub fn load_run_config(path: &Path) -> Result<(RunConfig, String), String> {
    let bytes = fs::read(path).map_err(|e| {
        format!(
            "cannot read {}: {e}. Copy run.example.toml from the repository root there \
             and edit it before the first start.",
            path.display()
        )
    })?;
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let cfg: RunConfig = toml::from_str(text).map_err(|e| format!("{}: {e}", path.display()))?;
    cfg.validate()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((cfg, sha256_hex(&bytes)))
}

impl RunConfig {
    pub fn validate(&self) -> Result<(), String> {
        let r = &self.run;
        if r.run_id.trim().is_empty() {
            return Err("[run] run_id must not be empty".into());
        }
        if r.parent_run_id.is_empty() != (r.fork_sample == 0) {
            return Err("[run] parent_run_id and fork_sample must be set together".into());
        }
        let s = &self.selfplay;
        if s.num_generators == 0 || s.virtual_leaves == 0 || s.max_nodes == 0 {
            return Err(
                "[selfplay] num_generators, virtual_leaves and max_nodes must be > 0".into(),
            );
        }
        if !(0.0..=1.0).contains(&s.pcr_full_prob) {
            return Err("[selfplay] pcr_full_prob must be in [0, 1]".into());
        }
        if s.pcr_full_prob < 1.0 && s.pcr_fast_visits < 2 {
            return Err("[selfplay] pcr_fast_visits must be at least 2".into());
        }
        Ok(())
    }

    /// The self-play search settings (what `generator_main` used to hard-code).
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
}

/// A warning naming every retired run-hyperparameter environment variable that is set, or
/// `None`. They are ignored: run.toml decides.
pub fn retired_env_var_warning() -> Option<String> {
    let set: Vec<String> = RETIRED_ENV_VARS
        .iter()
        .filter(|(name, _)| std::env::var_os(name).is_some())
        .map(|(name, key)| format!("{name} (now run.toml {key})"))
        .collect();
    (!set.is_empty()).then(|| {
        format!(
            "IGNORING environment variables that no longer configure a run: {}. run.toml \
             decides; remove them (PowerShell: Remove-Item Env:NAME) to silence this.",
            set.join(", ")
        )
    })
}

/// Who is starting, with what code and config.
#[derive(Clone, Debug)]
pub struct StartIdentity {
    pub run: RunSection,
    pub config_sha256: String,
    pub git_sha: String,
    pub git_dirty: bool,
    pub process: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Overrides {
    pub allow_config_change: bool,
    pub allow_code_change: bool,
}

impl Overrides {
    /// `--allow-config-change` and `--allow-code-change` from the command line.
    pub fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        Self {
            allow_config_change: args.iter().any(|a| a == "--allow-config-change"),
            allow_code_change: args.iter().any(|a| a == "--allow-code-change"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RunLock {
    pub run_id: String,
    pub parent_run_id: String,
    pub fork_sample: u64,
    pub seed: u64,
    pub git_sha: String,
    pub config_sha256: String,
    pub created_at: f64,
    pub created_by: String,
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Checks `id` against the run lock in `hidden_dir`, creating the lock on the first start,
/// and appends this start to `run_log.jsonl`. Returns the warnings and overrides used.
///
/// Rules (same as the Python trainer):
/// - a different run_id in the same data root is always refused;
/// - a different commit, or an unknown commit, needs `allow_code_change`;
/// - uncommitted changes to tracked files only warn (and are logged as `git_dirty`);
/// - a different run.toml needs `allow_config_change`.
pub fn check_run_lock(
    hidden_dir: &Path,
    id: &StartIdentity,
    overrides: Overrides,
) -> Result<Vec<String>, String> {
    fs::create_dir_all(hidden_dir).map_err(|e| format!("{}: {e}", hidden_dir.display()))?;
    let lock_path = hidden_dir.join("run_lock.json");
    let dirty_warning = id.git_dirty.then(|| {
        "warning: tracked files have uncommitted changes (logged as git_dirty)".to_owned()
    });
    let mut used: Vec<String> = dirty_warning.iter().cloned().collect();

    let mut code_problems = Vec::new();
    if id.git_sha == "unknown" {
        code_problems.push("the git commit is unknown".to_owned());
    }

    let lock = match fs::read(&lock_path) {
        Ok(bytes) => serde_json::from_slice::<RunLock>(&bytes)
            .map_err(|e| format!("{}: {e}", lock_path.display()))?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let lock = RunLock {
                run_id: id.run.run_id.clone(),
                parent_run_id: id.run.parent_run_id.clone(),
                fork_sample: id.run.fork_sample,
                seed: id.run.seed,
                git_sha: id.git_sha.clone(),
                config_sha256: id.config_sha256.clone(),
                created_at: now_secs(),
                created_by: id.process.clone(),
            };
            // Checked before the lock exists, so an unknown commit cannot be pinned.
            if !code_problems.is_empty() {
                if !overrides.allow_code_change {
                    return Err(format!(
                        "refusing to start run {}: {}. Build from a git checkout (or set \
                         TZ_BUILD_GIT_SHA), or pass --allow-code-change.",
                        id.run.run_id,
                        code_problems.join("; ")
                    ));
                }
                used.push(format!("allow-code-change: {}", code_problems.join("; ")));
            }
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(mut f) => {
                    let text = serde_json::to_string_pretty(&lock).expect("serialise lock");
                    f.write_all(text.as_bytes())
                        .and_then(|_| f.sync_all())
                        .map_err(|e| format!("{}: {e}", lock_path.display()))?;
                    println!(
                        "[run] Created run lock {} for run {} at commit {}",
                        lock_path.display(),
                        lock.run_id,
                        lock.git_sha
                    );
                    append_run_log(hidden_dir, id, &used);
                    return Ok(used);
                }
                // Another process created it first: check against theirs.
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    used = dirty_warning.iter().cloned().collect();
                    let bytes = fs::read(&lock_path)
                        .map_err(|e| format!("{}: {e}", lock_path.display()))?;
                    serde_json::from_slice::<RunLock>(&bytes)
                        .map_err(|e| format!("{}: {e}", lock_path.display()))?
                }
                Err(e) => return Err(format!("{}: {e}", lock_path.display())),
            }
        }
        Err(e) => return Err(format!("{}: {e}", lock_path.display())),
    };

    if lock.run_id != id.run.run_id {
        return Err(format!(
            "run.toml says run_id {:?} but this data root belongs to run {:?} ({}). \
             A new run or a branch needs its own data root.",
            id.run.run_id,
            lock.run_id,
            lock_path.display()
        ));
    }
    if lock.git_sha != id.git_sha {
        code_problems.insert(
            0,
            format!(
                "built from commit {} but the run is pinned to {}",
                id.git_sha, lock.git_sha
            ),
        );
    }
    if !code_problems.is_empty() {
        if !overrides.allow_code_change {
            return Err(format!(
                "refusing to start run {}: {}. Check out the pinned commit, or pass \
                 --allow-code-change.",
                id.run.run_id,
                code_problems.join("; ")
            ));
        }
        used.push(format!("allow-code-change: {}", code_problems.join("; ")));
    }
    if lock.config_sha256 != id.config_sha256 {
        let problem = format!(
            "run.toml (sha256 {}) differs from the locked one (sha256 {})",
            &id.config_sha256[..12],
            &lock.config_sha256[..12.min(lock.config_sha256.len())]
        );
        if !overrides.allow_config_change {
            return Err(format!(
                "refusing to start run {}: {problem}. Restore it, start a new run in its own \
                 data root, or pass --allow-config-change.",
                id.run.run_id
            ));
        }
        used.push(format!("allow-config-change: {problem}"));
    }
    append_run_log(hidden_dir, id, &used);
    Ok(used)
}

fn append_run_log(hidden_dir: &Path, id: &StartIdentity, overrides_used: &[String]) {
    let entry = serde_json::json!({
        "time": now_secs(),
        "process": id.process,
        "pid": std::process::id(),
        "run_id": id.run.run_id,
        "git_sha": id.git_sha,
        "git_dirty": id.git_dirty,
        "config_sha256": id.config_sha256,
        "overrides_used": overrides_used,
    });
    append_jsonl(&hidden_dir.join("run_log.jsonl"), &entry);
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

/// Loads run.toml, refuses retired env vars, checks the run lock, and prints the effective
/// config. Exits the process with a message on any refusal.
pub fn start_run(process: &str) -> (RunConfig, StartIdentity) {
    let fail = |msg: String| -> ! {
        eprintln!("[run] {msg}");
        std::process::exit(2)
    };
    if let Some(w) = retired_env_var_warning() {
        eprintln!("[run][Warning] {w}");
    }
    let path = run_config_path();
    let (cfg, config_sha256) = load_run_config(&path).unwrap_or_else(|e| fail(e));
    let id = StartIdentity {
        run: cfg.run.clone(),
        config_sha256,
        git_sha: GIT_SHA.to_owned(),
        git_dirty: GIT_DIRTY,
        process: process.to_owned(),
    };
    let used = check_run_lock(&data_path("hidden"), &id, Overrides::from_args())
        .unwrap_or_else(|e| fail(e));
    println!(
        "[run] run {} (parent {:?}, fork_sample {}), config {} sha256 {}, commit {}{}",
        id.run.run_id,
        id.run.parent_run_id,
        id.run.fork_sample,
        path.display(),
        id.config_sha256,
        id.git_sha,
        if id.git_dirty { " (dirty)" } else { "" }
    );
    for o in &used {
        println!("[run][Warning] {o}");
    }
    println!("[run] effective [selfplay]: {:?}", cfg.selfplay);
    (cfg, id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> (RunConfig, String) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../run.example.toml");
        load_run_config(&path).expect("run.example.toml must load")
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tz-runlock-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn identity(cfg: &RunConfig, sha: &str, config: &str, dirty: bool) -> StartIdentity {
        StartIdentity {
            run: cfg.run.clone(),
            config_sha256: config.repeat(64 / config.len()),
            git_sha: sha.to_owned(),
            git_dirty: dirty,
            process: "test".to_owned(),
        }
    }

    const NONE: Overrides = Overrides {
        allow_config_change: false,
        allow_code_change: false,
    };

    /// The example file reproduces the self-play settings `main.rs` hard-coded before
    /// run.toml existed, so adopting it changes nothing about the current run.
    #[test]
    fn example_matches_previous_hard_coded_settings() {
        let (cfg, _) = example();
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
        assert_eq!(cfg.selfplay.num_generators, 4096);
        assert_eq!(cfg.selfplay.virtual_leaves, 1);
    }

    #[test]
    fn missing_and_unknown_keys_are_errors() {
        let text = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../run.example.toml"),
        )
        .unwrap();
        let missing = text.replace("max_nodes = 1600\n", "");
        assert!(toml::from_str::<RunConfig>(&missing).is_err());
        let typo = text.replace("max_nodes = 1600", "max_node = 1600");
        assert!(toml::from_str::<RunConfig>(&typo).is_err());
        let off = text.replace("pcr_full_prob = 0.25", "pcr_full_prob = 1.0");
        let cfg: RunConfig = toml::from_str(&off).unwrap();
        assert_eq!(cfg.playout_cap(), None);
    }

    #[test]
    fn first_start_creates_the_lock_and_identical_restart_passes() {
        let (cfg, _) = example();
        let dir = temp_dir("first");
        let id = identity(&cfg, "abc", "a", false);
        assert_eq!(check_run_lock(&dir, &id, NONE), Ok(vec![]));
        let lock: RunLock =
            serde_json::from_slice(&fs::read(dir.join("run_lock.json")).unwrap()).unwrap();
        assert_eq!(lock.git_sha, "abc");
        assert_eq!(lock.run_id, cfg.run.run_id);
        assert_eq!(check_run_lock(&dir, &id, NONE), Ok(vec![]));
        let log = fs::read_to_string(dir.join("run_log.jsonl")).unwrap();
        assert_eq!(log.lines().count(), 2);
    }

    #[test]
    fn dirty_tree_only_warns() {
        let (cfg, _) = example();
        let dir = temp_dir("dirty");
        let dirty = identity(&cfg, "abc", "a", true);
        let notes = check_run_lock(&dir, &dirty, NONE).unwrap();
        assert_eq!(notes.len(), 1);
        assert!(notes[0].starts_with("warning:"));
        assert!(dir.join("run_lock.json").exists());
        // Still pinned: the same dirty tree at another commit is refused.
        assert!(check_run_lock(&dir, &identity(&cfg, "def", "a", true), NONE).is_err());
        let unknown = identity(&cfg, "unknown", "a", false);
        assert!(check_run_lock(&temp_dir("unknown"), &unknown, NONE).is_err());
    }

    #[test]
    fn other_commit_or_config_is_refused_unless_overridden() {
        let (cfg, _) = example();
        let dir = temp_dir("mismatch");
        check_run_lock(&dir, &identity(&cfg, "abc", "a", false), NONE).unwrap();

        let other_commit = identity(&cfg, "def", "a", false);
        assert!(check_run_lock(&dir, &other_commit, NONE).is_err());
        let code = Overrides {
            allow_code_change: true,
            ..NONE
        };
        assert_eq!(check_run_lock(&dir, &other_commit, code).unwrap().len(), 1);

        let other_config = identity(&cfg, "abc", "b", false);
        assert!(check_run_lock(&dir, &other_config, NONE).is_err());
        assert!(check_run_lock(&dir, &other_config, code).is_err());
        let config = Overrides {
            allow_config_change: true,
            ..NONE
        };
        assert_eq!(
            check_run_lock(&dir, &other_config, config).unwrap().len(),
            1
        );
    }

    #[test]
    fn another_run_id_is_always_refused() {
        let (cfg, _) = example();
        let dir = temp_dir("runid");
        check_run_lock(&dir, &identity(&cfg, "abc", "a", false), NONE).unwrap();
        let mut other = identity(&cfg, "abc", "a", false);
        other.run.run_id = "someone-else".into();
        let both = Overrides {
            allow_config_change: true,
            allow_code_change: true,
        };
        assert!(check_run_lock(&dir, &other, both).is_err());
    }
}
