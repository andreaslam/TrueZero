//! Lichess bot.
//!
//! Gets its networks from the training server (as a `RustDataGen`), evaluates them with the
//! shared batched executors and plays on Lichess: it accepts incoming challenges and
//! challenges other online bots.
//!
//! Recording games for training is off by default. Set `TZ_LICHESS_COLLECT_DATA=1` to send
//! finished games to the server like self-play games.

use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    net::TcpStream,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use flume::{Receiver, Sender};
use tzrust::{
    data_path, data_path_str,
    executor::{executor_main, Packet},
    fileformat::BinaryOutput,
    message_types::{DataFileType, Entity, MessageServer, MessageType, Statistics},
    selfplay::CollectorMessage,
};

#[path = "lichess/game.rs"]
mod game;
#[path = "lichess/runtime.rs"]
mod runtime;

const GAMES_PER_FILE: usize = 100;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct Config {
    pub token: String,
    pub server_addr: String,
    pub num_executors: usize,
    pub batch_size: usize,
    pub queue_capacity: usize,
    pub max_concurrent_games: usize,
    /// Record finished games and send them to the server for training.
    pub collect_data: bool,
}

impl Config {
    fn from_env() -> Self {
        let num_executors = env_usize("TZ_NUM_EXECUTORS", 2);
        let batch_size = env_usize("TZ_BATCH_SIZE", 1024);
        Self {
            token: env::var("LICHESS_API_KEY").expect("LICHESS_API_KEY must be set"),
            server_addr: format!(
                "{}:{}",
                env::var("TZ_SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
                env::var("TZ_SERVER_PORT").unwrap_or_else(|_| "38475".into()),
            ),
            num_executors,
            batch_size,
            queue_capacity: env_usize("TZ_TENSOR_QUEUE_CAPACITY", batch_size * num_executors),
            max_concurrent_games: env_usize("TZ_MAX_CONCURRENT_GAMES", 4),
            collect_data: env_bool("TZ_LICHESS_COLLECT_DATA"),
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
        .max(1)
}

fn env_bool(name: &str) -> bool {
    env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn main() {
    dotenv::dotenv().ok();
    let config = Config::from_env();
    println!(
        "[Lichess] {} executor(s), batch size {}, up to {} concurrent games, \
         training data collection {}",
        config.num_executors,
        config.batch_size,
        config.max_concurrent_games,
        if config.collect_data {
            "ON"
        } else {
            "off (set TZ_LICHESS_COLLECT_DATA=1 to enable)"
        }
    );

    let stream = loop {
        match TcpStream::connect(&config.server_addr) {
            Ok(stream) => break stream,
            Err(_) => std::thread::sleep(Duration::from_secs(1)),
        }
    };
    send_message(&mut &stream, MessageType::Initialise(Entity::RustDataGen))
        .expect("failed to register with the server");

    let (net_senders, net_receivers): (Vec<_>, Vec<_>) = (0..config.num_executors)
        .map(|_| flume::bounded::<String>(1))
        .unzip();
    let (executor_ready_send, executor_ready_recv) = flume::unbounded::<bool>();
    let (net_loaded_send, net_loaded_recv) = flume::bounded::<()>(1);
    let (tensor_send, tensor_recv) = flume::bounded::<Packet>(config.queue_capacity);
    let (collector_send, collector_recv) =
        flume::bounded::<CollectorMessage>(config.max_concurrent_games);
    let (id_send, id_recv) = flume::bounded::<usize>(1);

    std::thread::scope(|scope| {
        let command_stream = stream.try_clone().expect("clone server stream");
        let num_executors = config.num_executors;
        scope.spawn(move || {
            commander_main(
                command_stream,
                net_senders,
                executor_ready_recv,
                num_executors,
                net_loaded_send,
                id_send,
            )
        });

        for (executor_id, net_receiver) in net_receivers.into_iter().enumerate() {
            let tensor_recv = tensor_recv.clone();
            let collector_send = collector_send.clone();
            let executor_ready_send = executor_ready_send.clone();
            let batch_size = config.batch_size;
            std::thread::Builder::new()
                .name(format!("executor-{executor_id}"))
                .spawn_scoped(scope, move || {
                    executor_main(
                        net_receiver,
                        tensor_recv,
                        batch_size,
                        Some(collector_send),
                        Some(executor_ready_send),
                        executor_id,
                    )
                })
                .expect("failed to spawn executor thread");
        }

        let collect_stream = stream.try_clone().expect("clone server stream");
        scope.spawn(move || collector_main(collector_recv, collect_stream, id_recv));

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to create the Tokio runtime");
        let result = runtime.block_on(async {
            println!("[Lichess] Waiting for the server to provide the first network...");
            net_loaded_recv
                .recv_async()
                .await
                .map_err(|_| "the server connection closed before a network was loaded")?;
            println!("[Lichess] Network loaded.");
            let collector = config.collect_data.then_some(collector_send);
            runtime::run(&config, tensor_send, collector).await
        });
        if let Err(error) = result {
            eprintln!("[Lichess] Fatal error: {error}");
        }
        // the executors and the commander block forever, so leave instead of joining them
        std::process::exit(1);
    });
}

fn send_message(stream: &mut impl Write, purpose: MessageType) -> std::io::Result<()> {
    let wire = serde_json::to_string(&MessageServer { purpose }).expect("serialisation failed");
    writeln!(stream, "{wire}")?;
    stream.flush()
}

/// Receives networks from the server and loads them into every executor.
///
/// Each network is written to a temporary file that is deleted once all executors loaded it.
/// `net_loaded` fires once, after the first network is loaded.
fn commander_main(
    mut stream: TcpStream,
    net_senders: Vec<Sender<String>>,
    executor_ready: Receiver<bool>,
    num_executors: usize,
    net_loaded: Sender<()>,
    id_send: Sender<usize>,
) {
    let _ = fs::create_dir_all(data_path("nets"));
    let mut reader = BufReader::new(stream.try_clone().expect("clone server stream"));
    if let Err(error) = send_message(&mut stream, MessageType::RequestingNet()) {
        eprintln!("[Lichess] Could not request a network: {error}");
        return;
    }
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => {
                eprintln!("[Lichess] Server connection closed.");
                return;
            }
            Ok(_) => {}
        }
        let Ok(message) = serde_json::from_str::<MessageServer>(&line) else {
            continue;
        };
        match message.purpose {
            MessageType::IdentityConfirmation((Entity::RustDataGen, id)) => {
                let _ = id_send.try_send(id);
            }
            MessageType::NewNetworkData(data) => {
                let path = data_path_str(&format!(
                    "nets/lichess_net_{}_{}.pt",
                    std::process::id(),
                    unix_nanos()
                ));
                if let Err(error) = fs::write(&path, &data) {
                    eprintln!("[Lichess] Could not save the received network: {error}");
                    continue;
                }
                for sender in &net_senders {
                    sender
                        .send(path.clone())
                        .expect("executor network channel disconnected");
                }
                for _ in 0..num_executors {
                    executor_ready
                        .recv()
                        .expect("executor stopped before loading the network");
                }
                let _ = fs::remove_file(&path);
                println!("[Lichess] Loaded a new network.");
                let _ = net_loaded.try_send(());
            }
            _ => {}
        }
    }
}

/// Forwards executor statistics to the server and, when data collection is on, writes finished
/// games to files that are sent to the server every `GAMES_PER_FILE` games.
fn collector_main(
    receiver: Receiver<CollectorMessage>,
    mut stream: TcpStream,
    id_recv: Receiver<usize>,
) {
    let mut id = None;
    let mut output: Option<(String, BinaryOutput)> = None;
    let mut evals = 0usize;
    let mut evals_since = Instant::now();

    for message in receiver.iter() {
        match message {
            CollectorMessage::FinishedGame(simulation) => {
                let id = *id.get_or_insert_with(|| id_recv.recv().unwrap_or(0));
                let (path, file) = output.get_or_insert_with(|| {
                    let _ = fs::create_dir_all(data_path("games"));
                    let path = data_path_str(&format!("games/lichess_gen_{id}_{}", unix_nanos()));
                    let file = BinaryOutput::new(&path, "chess").expect("create game file");
                    (path, file)
                });
                if let Err(error) = file.append(&simulation) {
                    eprintln!("[Lichess] Could not write a game: {error}");
                    continue;
                }
                println!(
                    "[Lichess] Recorded game {}/{GAMES_PER_FILE}.",
                    file.game_count()
                );
                if file.game_count() >= GAMES_PER_FILE {
                    let path = path.clone();
                    let (_, mut file) = output.take().expect("output is open");
                    match send_game_file(&mut stream, &path, &mut file) {
                        Ok(()) => println!("[Lichess] Sent {GAMES_PER_FILE} games to the server."),
                        Err(error) => eprintln!("[Lichess] Could not send games: {error}"),
                    }
                    for suffix in [".bin", ".off", ".json", ".json.tmp"] {
                        let _ = fs::remove_file(format!("{path}{suffix}"));
                    }
                }
            }
            CollectorMessage::ExecutorStatistics(batch) => {
                evals += batch;
                let elapsed = evals_since.elapsed();
                if elapsed >= Duration::from_secs(1) {
                    let per_second = (evals as f64 / elapsed.as_secs_f64()) as usize;
                    let statistics =
                        MessageType::StatisticsSend(Statistics::EvalsPerSecond(per_second));
                    if let Err(error) = send_message(&mut stream, statistics) {
                        eprintln!("[Lichess] Could not send statistics: {error}");
                    }
                    evals = 0;
                    evals_since = Instant::now();
                }
            }
            _ => {}
        }
    }
}

fn send_game_file(
    stream: &mut TcpStream,
    path: &str,
    file: &mut BinaryOutput,
) -> std::io::Result<()> {
    file.finish()?;
    let data = vec![
        DataFileType::BinFile(fs::read(format!("{path}.bin"))?),
        DataFileType::OffFile(fs::read(format!("{path}.off"))?),
        DataFileType::MetaDataFile(fs::read(format!("{path}.json"))?),
    ];
    send_message(stream, MessageType::JobSendData(data))
}

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}
