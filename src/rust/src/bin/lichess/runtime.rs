//! Lichess account, event stream and matchmaking.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use flume::Sender;
use futures_util::StreamExt;
use litchee::{
    api::gameplay::{
        board::LichessIncomingEvent,
        challenges::{LichessChallenge, LichessChallengeColor, LichessChallengeStatus},
    },
    model::{LichessColor, LichessVariantKey},
    LichessClient,
};
use rand::seq::SliceRandom;
use tokio::time::sleep;
use tzrust::{executor::Packet, selfplay::CollectorMessage};

use super::{game, BoxError, Config};

const EVENT_STREAM_RECONNECT_DELAY: Duration = Duration::from_secs(2);

/// How often to look for online bots to challenge while no game is being played.
const MATCHMAKING_INTERVAL: Duration = Duration::from_secs(60);
/// Do not challenge the same bot again within this time.
const CHALLENGE_COOLDOWN: Duration = Duration::from_secs(30 * 60);
const CHALLENGE_SEND_DELAY: Duration = Duration::from_secs(2);
const CHALLENGE_RATED: bool = true;

/// Clock (initial seconds, increment seconds) for each outgoing challenge, picked at random.
const TIME_CONTROLS: [(u32, u32); 4] = [(1_800, 0), (600, 5), (180, 2), (60, 0)];

/// State shared by the event loop, matchmaking and every game.
pub struct Bot {
    pub client: LichessClient,
    pub username: String,
    pub tensor_send: Sender<Packet>,
    /// `Some` only when finished games are recorded for training.
    pub collector: Option<Sender<CollectorMessage>>,
    max_games: usize,
    /// Ids of the games that currently have a worker.
    games: Mutex<HashSet<String>>,
}

impl Bot {
    pub fn active_games(&self) -> usize {
        self.games.lock().unwrap().len()
    }
}

/// Removes a game from `Bot::games` when its worker ends, however it ends.
struct GameSlot {
    bot: Arc<Bot>,
    id: String,
}

impl Drop for GameSlot {
    fn drop(&mut self) {
        self.bot.games.lock().unwrap().remove(&self.id);
    }
}

pub async fn run(
    config: &Config,
    tensor_send: Sender<Packet>,
    collector: Option<Sender<CollectorMessage>>,
) -> Result<(), BoxError> {
    let client = LichessClient::builder().token(&config.token).build()?;
    let profile = client.account().profile().await?;
    println!("[Lichess] Logged in as {}.", profile.user.username);

    let bot = Arc::new(Bot {
        client,
        username: profile.user.username,
        tensor_send,
        collector,
        max_games: config.max_concurrent_games,
        games: Mutex::new(HashSet::new()),
    });

    let matchmaking = tokio::spawn(matchmaking_loop(Arc::clone(&bot)));
    tokio::select! {
        result = event_loop(Arc::clone(&bot)) => result,
        result = matchmaking => Err(format!("matchmaking stopped: {result:?}").into()),
    }
}

async fn event_loop(bot: Arc<Bot>) -> Result<(), BoxError> {
    loop {
        match bot.client.bot().stream_events().await {
            Ok(mut events) => {
                println!("[Lichess] Event stream connected.");
                while let Some(event) = events.next().await {
                    match event {
                        Ok(LichessIncomingEvent::Challenge { challenge }) => {
                            handle_challenge(&bot, &challenge).await
                        }
                        Ok(LichessIncomingEvent::GameStart { game }) => {
                            match (game.game_id.or(game.id), game.color) {
                                (Some(id), Some(color)) => start_game(&bot, id, color).await,
                                _ => eprintln!("[Lichess] gameStart without id or colour."),
                            }
                        }
                        Ok(_) => {}
                        Err(error) => {
                            eprintln!("[Lichess] Event stream error: {error}");
                            break;
                        }
                    }
                }
                eprintln!("[Lichess] Event stream ended; reconnecting.");
            }
            Err(error) => eprintln!("[Lichess] Could not open the event stream: {error}"),
        }
        sleep(EVENT_STREAM_RECONNECT_DELAY).await;
    }
}

async fn handle_challenge(bot: &Bot, challenge: &LichessChallenge) {
    // the event stream also reports our own outgoing challenges
    let incoming = challenge.direction.as_deref() != Some("out")
        && challenge
            .challenger
            .as_ref()
            .is_some_and(|user| !user.name.eq_ignore_ascii_case(&bot.username));
    if !incoming || !matches!(challenge.status, LichessChallengeStatus::Created) {
        return;
    }
    let challenger = challenge
        .challenger
        .as_ref()
        .map_or("<unknown>", |user| user.name.as_str());

    let variant = challenge.variant.as_ref().and_then(|variant| variant.key);
    let decline_reason = if !matches!(
        variant,
        None | Some(LichessVariantKey::Standard) | Some(LichessVariantKey::FromPosition)
    ) {
        Some("standard")
    } else if bot.active_games() >= bot.max_games {
        Some("later")
    } else {
        None
    };

    let challenges = bot.client.challenges();
    let result = match decline_reason {
        Some(reason) => {
            println!(
                "[Lichess] Declining challenge {} from {challenger} ({reason}).",
                challenge.id
            );
            challenges.decline(&challenge.id, Some(reason)).await
        }
        None => {
            println!(
                "[Lichess] Accepting challenge {} from {challenger}.",
                challenge.id
            );
            challenges.accept(&challenge.id, None).await
        }
    };
    if let Err(error) = result {
        eprintln!(
            "[Lichess] Could not answer challenge {}: {error}",
            challenge.id
        );
    }
}

async fn start_game(bot: &Arc<Bot>, id: String, color: LichessColor) {
    {
        let mut games = bot.games.lock().unwrap();
        // Lichess repeats gameStart for ongoing games when the event stream reconnects
        if games.contains(&id) {
            return;
        }
        if games.len() >= bot.max_games {
            drop(games);
            eprintln!("[Lichess] Too many concurrent games; aborting game {id}.");
            if let Err(error) = bot.client.bot().abort(&id).await {
                eprintln!("[Lichess] Could not abort game {id}: {error}");
            }
            return;
        }
        games.insert(id.clone());
    }
    let slot = GameSlot {
        bot: Arc::clone(bot),
        id,
    };
    tokio::spawn(async move {
        println!("[Lichess] Game {} started as {color:?}.", slot.id);
        if let Err(error) = game::run_game(&slot.bot, &slot.id, color).await {
            eprintln!("[Lichess] Game {} error: {error}", slot.id);
        }
        println!("[Lichess] Game {} ended.", slot.id);
    });
}

/// Challenges online bots while no game is being played.
async fn matchmaking_loop(bot: Arc<Bot>) {
    let mut last_challenged: HashMap<String, Instant> = HashMap::new();
    let mut interval = tokio::time::interval(MATCHMAKING_INTERVAL);
    loop {
        interval.tick().await;
        if bot.active_games() > 0 {
            continue;
        }
        last_challenged.retain(|_, at| at.elapsed() < CHALLENGE_COOLDOWN);
        if let Err(error) = challenge_online_bots(&bot, &mut last_challenged).await {
            eprintln!("[Lichess] Matchmaking error: {error}");
        }
    }
}

async fn challenge_online_bots(
    bot: &Bot,
    last_challenged: &mut HashMap<String, Instant>,
) -> Result<(), BoxError> {
    let mut candidates = Vec::new();
    let mut online = bot.client.bot().online(None).await?;
    while let Some(user) = online.next().await {
        match user {
            Ok(user)
                if !user.username.eq_ignore_ascii_case(&bot.username)
                    && !last_challenged.contains_key(&user.username) =>
            {
                candidates.push(user.username)
            }
            Ok(_) => {}
            Err(error) => eprintln!("[Lichess] Error reading the online bot list: {error}"),
        }
    }
    candidates.shuffle(&mut rand::thread_rng());
    candidates.truncate(bot.max_games);

    for (index, username) in candidates.into_iter().enumerate() {
        if index > 0 {
            sleep(CHALLENGE_SEND_DELAY).await;
        }
        let (initial, increment) = *TIME_CONTROLS
            .choose(&mut rand::thread_rng())
            .expect("time controls are not empty");
        println!(
            "[Lichess] Challenging {username} ({}+{increment}).",
            initial / 60
        );
        last_challenged.insert(username.clone(), Instant::now());
        let result = bot
            .client
            .challenges()
            .challenge(&username)
            .color(LichessChallengeColor::Random)
            .rated(CHALLENGE_RATED)
            .clock(initial, increment)
            .send()
            .await;
        if let Err(error) = result {
            eprintln!("[Lichess] Could not challenge {username}: {error}");
        }
    }
    Ok(())
}
