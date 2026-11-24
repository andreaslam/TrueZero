//! One command for a whole training run.
//!
//!     cargo run --release --bin truezero              # resume the current run
//!     cargo run --release --bin truezero -- reset     # start a fresh run
//!
//! Builds the other binaries, fetches the gate's opening book if it is missing, then starts
//! the server, the trainer (client.py), self-play (main) and the gate (trainenginetest).
//! Their output is shown here with a `[name]` prefix and appended to
//! `<runs_dir>/logs/<name>.log`. A program that exits asking to be restarted (a new run
//! began: `server reset` from another terminal, or the server went away) is started again
//! at once; one that crashes is started again after a pause, unless it keeps crashing.
//! Ctrl+C stops everything. All settings come from truezero.toml.

use std::{
    collections::VecDeque,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use tzrust::runconfig::{
    load_config, repo_root, retired_env_var_warning, Config, RESTART_EXIT_CODE,
    RESTART_SENTINEL,
};

/// A program crashing this many times within `CRASH_WINDOW` stops the launcher.
const MAX_CRASHES: usize = 5;
const CRASH_WINDOW: Duration = Duration::from_secs(600);
const CRASH_PAUSE: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    Server,
    Trainer,
    Datagen,
    Gate,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Role::Server => "server",
            Role::Trainer => "trainer",
            Role::Datagen => "datagen",
            Role::Gate => "gate",
        }
    }
}

/// A running program. `restart_requested` is set when it prints `RESTART_SENTINEL`: a
/// program that exits after that is restarting for a new run, whatever its exit status
/// (libtorch can crash while a process exits with its threads still running).
struct Running {
    child: Child,
    restart_requested: Arc<AtomicBool>,
    readers: Vec<JoinHandle<()>>,
}

struct Launcher {
    cfg: Config,
    root: PathBuf,
    bin_dir: PathBuf,
    logs_dir: PathBuf,
    console: Arc<Mutex<()>>,
}

fn exe(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

impl Launcher {
    fn python_command(&self, script_args: &[&str]) -> Command {
        let py = &self.cfg.launcher.python;
        let mut cmd = Command::new(&py[0]);
        cmd.args(&py[1..])
            .args(script_args)
            .current_dir(self.root.join("src").join("python"))
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8");
        if !self.cfg.launcher.venv.is_empty() {
            cmd.env("UV_PROJECT_ENVIRONMENT", &self.cfg.launcher.venv);
        }
        cmd
    }

    fn command(&self, role: Role, reset: bool) -> Command {
        match role {
            Role::Server => {
                let mut c = Command::new(exe(&self.bin_dir, "server"));
                if reset {
                    c.arg("reset");
                }
                c
            }
            Role::Trainer => self.python_command(&["client.py"]),
            Role::Datagen => Command::new(exe(&self.bin_dir, "main")),
            Role::Gate => Command::new(exe(&self.bin_dir, "trainenginetest")),
        }
    }

    /// Starts `role`, copying its output to the console and its log file.
    fn spawn(&self, role: Role, reset: bool) -> Running {
        let mut cmd = self.command(role, reset);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().unwrap_or_else(|e| {
            eprintln!("[truezero] could not start {} ({cmd:?}): {e}", role.name());
            std::process::exit(1)
        });
        let log_path = self.logs_dir.join(format!("{}.log", role.name()));
        let log = Arc::new(Mutex::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .map_err(|e| eprintln!("[truezero] cannot open {}: {e}", log_path.display()))
                .ok(),
        ));
        let out: Box<dyn Read + Send> = Box::new(child.stdout.take().unwrap());
        let err: Box<dyn Read + Send> = Box::new(child.stderr.take().unwrap());
        let restart_requested = Arc::new(AtomicBool::new(false));
        let readers = [out, err]
            .into_iter()
            .map(|stream| {
                let (log, console) = (log.clone(), self.console.clone());
                let flag = restart_requested.clone();
                thread::spawn(move || copy_lines(role.name(), stream, &log, &console, &flag))
            })
            .collect();
        self.say(&format!("started {} (pid {})", role.name(), child.id()));
        Running {
            child,
            restart_requested,
            readers,
        }
    }

    fn say(&self, msg: &str) {
        let _g = self.console.lock().unwrap();
        println!("[truezero] {msg}");
    }
}

fn copy_lines(
    name: &str,
    stream: Box<dyn Read + Send>,
    log: &Mutex<Option<fs::File>>,
    console: &Mutex<()>,
    restart_requested: &AtomicBool,
) {
    let mut reader = BufReader::new(stream);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&buf);
        let line = line.trim_end_matches(['\r', '\n']);
        if line.contains(RESTART_SENTINEL) {
            restart_requested.store(true, Ordering::SeqCst);
        }
        if let Some(f) = log.lock().unwrap().as_mut() {
            // A network-drive hiccup loses a log line, never the program.
            let _ = writeln!(f, "{line}");
        }
        let _g = console.lock().unwrap();
        println!("[{name}] {line}");
    }
}

fn port_open(port: u16) -> bool {
    TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_millis(300)).is_ok()
}

fn build(root: &Path) {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    println!("[truezero] building server, main, trainenginetest and ladder ...");
    let mut cmd = Command::new(&cargo);
    // `cargo run` gave this process variables describing the truezero crate; passed on,
    // they would make cargo think dependencies changed (ring tracks CARGO_MANIFEST_DIR)
    // and rebuild libtorch's bindings on every launch.
    for (key, _) in std::env::vars_os() {
        let k = key.to_string_lossy();
        if ["CARGO_MANIFEST_", "CARGO_PKG_", "CARGO_CRATE_", "CARGO_BIN_", "CARGO_PRIMARY_", "TZ_GIT_"]
            .iter()
            .any(|p| k.starts_with(p))
            || k == "OUT_DIR"
        {
            cmd.env_remove(&key);
        }
    }
    let status = cmd
        .args(["build", "--release"])
        .args(["--bin", "server", "--bin", "main", "--bin", "trainenginetest"])
        .args(["--bin", "ladder"])
        .current_dir(root.join("src").join("rust"))
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("[truezero] the build failed ({s}); fix it and run again.");
            std::process::exit(1)
        }
        Err(e) => eprintln!(
            "[truezero][Warning] could not run {cargo} ({e}); using the binaries as they are."
        ),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reset = args.iter().any(|a| a == "reset");
    if let Some(a) = args.iter().find(|a| *a != "reset") {
        eprintln!("usage: truezero [reset]   (unknown argument {a:?})");
        std::process::exit(2);
    }
    if let Some(w) = retired_env_var_warning() {
        eprintln!("[truezero][Warning] {w}");
    }
    let cfg = load_config().config;
    let root = repo_root();
    let runs_dir = cfg.runs_dir(&root);
    let logs_dir = runs_dir.join("logs");
    fs::create_dir_all(&logs_dir).unwrap_or_else(|e| {
        eprintln!("[truezero] cannot create {}: {e}", logs_dir.display());
        std::process::exit(1)
    });
    let bin_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| root.join("src/rust/target/release"));

    if port_open(cfg.server.port) {
        eprintln!(
            "[truezero] Something is already listening on port {} (another launcher or \
             server?). Close it first. To reset a running setup instead, use \
             `cargo run --release --bin server -- reset`.",
            cfg.server.port
        );
        std::process::exit(1);
    }

    build(&root);
    let launcher = Launcher {
        cfg,
        root,
        bin_dir,
        logs_dir,
        console: Arc::new(Mutex::new(())),
    };

    // The gate's opening book (a fresh clone has none).
    let book = launcher.root.join(&launcher.cfg.gate.openings);
    if !book.is_file() {
        let dest = book.to_string_lossy().into_owned();
        let status = launcher
            .python_command(&["books.py", &dest])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !status || !book.is_file() {
            eprintln!(
                "[truezero] Could not get the opening book {dest}. Download 8moves_v3.epd.zip \
                 from https://github.com/official-stockfish/books, unzip it there and run again."
            );
            std::process::exit(1);
        }
    }

    // The server first: it creates the run the others join.
    let mut server = launcher.spawn(Role::Server, reset);
    let deadline = Instant::now() + Duration::from_secs(120);
    while !port_open(launcher.cfg.server.port) {
        if let Ok(Some(status)) = server.child.try_wait() {
            eprintln!("[truezero] the server exited ({status}) before listening; see its output above.");
            std::process::exit(1);
        }
        if Instant::now() > deadline {
            eprintln!("[truezero] the server did not start listening within 2 minutes.");
            let _ = server.child.kill();
            std::process::exit(1);
        }
        thread::sleep(Duration::from_millis(200));
    }

    let mut children: Vec<(Role, Running, VecDeque<Instant>)> =
        vec![(Role::Server, server, VecDeque::new())];
    for role in [Role::Trainer, Role::Datagen, Role::Gate] {
        children.push((role, launcher.spawn(role, false), VecDeque::new()));
    }

    loop {
        thread::sleep(Duration::from_millis(500));
        for i in 0..children.len() {
            let (role, running, crashes) = &mut children[i];
            let role = *role;
            let status = match running.child.try_wait() {
                Ok(Some(status)) => status,
                Ok(None) => continue,
                Err(e) => {
                    launcher.say(&format!("cannot check {}: {e}", role.name()));
                    continue;
                }
            };
            // Its last lines may still be in the pipe: read them before deciding (briefly:
            // a leftover grandchild could hold the pipe open).
            let until = Instant::now() + Duration::from_secs(2);
            while running.readers.iter().any(|r| !r.is_finished()) && Instant::now() < until {
                thread::sleep(Duration::from_millis(20));
            }
            running.readers.clear();
            if status.code() == Some(RESTART_EXIT_CODE)
                || running.restart_requested.load(Ordering::SeqCst)
            {
                launcher.say(&format!("{} asked to restart; starting it again", role.name()));
            } else {
                let now = Instant::now();
                crashes.push_back(now);
                while crashes.front().is_some_and(|t| now - *t > CRASH_WINDOW) {
                    crashes.pop_front();
                }
                let give_up = !launcher.cfg.launcher.restart_on_crash
                    || crashes.len() >= MAX_CRASHES;
                if give_up {
                    launcher.say(&format!(
                        "{} exited ({status}){}; stopping everything. Its log: {}",
                        role.name(),
                        if crashes.len() >= MAX_CRASHES {
                            format!(" for the {MAX_CRASHES}th time in 10 minutes")
                        } else {
                            String::new()
                        },
                        launcher.logs_dir.join(format!("{}.log", role.name())).display()
                    ));
                    for (_, c, _) in children.iter_mut() {
                        let _ = c.child.kill();
                    }
                    std::process::exit(1);
                }
                launcher.say(&format!(
                    "{} exited ({status}); restarting in {}s",
                    role.name(),
                    CRASH_PAUSE.as_secs()
                ));
                thread::sleep(CRASH_PAUSE);
            }
            children[i].1 = launcher.spawn(role, false);
        }
    }
}
