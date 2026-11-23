use crate::mcts_trainer::{EvalMode, TypeRequest};

#[derive(Clone, Debug, PartialEq, Copy)]

pub struct SearchSettings {
    pub fpu: FPUSettings,
    pub wdl: EvalMode, // if WDL is None then automatically use value
    pub moves_left: Option<MovesLeftSettings>,
    pub c_puct: CPUCTSettings,
    pub max_nodes: Option<u128>,
    pub alpha: f32,
    pub eps: f32,
    pub search_type: TypeRequest,
    pub pst: PSTSettings,
    pub batch_size: usize,
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct PSTSettings {
    pub root_pst: f32,
    pub children_pst: f32,
}
impl PSTSettings {
    /// returns the default value of FPUSettings
    pub fn default(self) -> PSTSettings {
        PSTSettings {
            root_pst: 1.75,
            children_pst: 1.5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct FPUSettings {
    pub root_fpu: f32,
    pub children_fpu: f32,
}
impl FPUSettings {
    /// returns the default value of FPUSettings
    pub fn default(self) -> FPUSettings {
        FPUSettings {
            root_fpu: 0.1,
            children_fpu: 0.2,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Copy)]
pub struct CPUCTSettings {
    pub root_c_puct: f32,
    pub children_c_puct: f32,
}
impl CPUCTSettings {
    /// returns the default value of CPUCTSettings
    pub fn default(self) -> CPUCTSettings {
        CPUCTSettings {
            root_c_puct: 1.0,
            children_c_puct: 2.0,
        }
    }
}
/// Playout cap randomisation (KataGo, Wu 2019, section 3.1).
///
/// Each self-play move is a full search with probability `full_search_prob`, using the
/// normal `SearchSettings::max_nodes`. Every other move is a fast search capped at
/// `fast_visits`, with no Dirichlet noise, and is recorded with `is_full_search = false`.
/// Fast searches finish games sooner (more value targets per unit of compute) while the
/// full searches still give strong policy targets.
#[derive(Clone, Debug, PartialEq, Copy)]
pub struct PlayoutCapSettings {
    pub full_search_prob: f32,
    pub fast_visits: u32,
    /// Stop a fast search once its most-visited root move cannot be overtaken by the visits
    /// left. Only applied when the move is played greedily, so it never changes the move.
    pub fast_early_stop: bool,
}

impl PlayoutCapSettings {
    pub fn new(full_search_prob: f32, fast_visits: u32, fast_early_stop: bool) -> Self {
        assert!(
            (0.0..=1.0).contains(&full_search_prob),
            "full_search_prob must be in [0, 1], got {full_search_prob}"
        );
        assert!(
            fast_visits >= 2,
            "fast_visits must be at least 2, got {fast_visits}"
        );
        Self {
            full_search_prob,
            fast_visits,
            fast_early_stop,
        }
    }

    /// The search settings for one move: unchanged for a full search; for a fast search the
    /// visit cap is `fast_visits` (never above the full cap) and root noise is off.
    pub fn move_settings(&self, settings: &SearchSettings, full: bool) -> SearchSettings {
        if full {
            return *settings;
        }
        let full_cap = settings
            .max_nodes
            .expect("playout cap randomisation needs a node limit");
        SearchSettings {
            max_nodes: Some(full_cap.min(self.fast_visits as u128)),
            eps: 0.0,
            ..*settings
        }
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct MovesLeftSettings {
    pub moves_left_weight: f32,
    pub moves_left_clip: f32,
    pub moves_left_sharpness: f32,
}

impl MovesLeftSettings {
    pub fn new(moves_left_weight: f32, moves_left_clip: f32, moves_left_sharpness: f32) -> Self {
        Self {
            moves_left_weight,
            moves_left_clip,
            moves_left_sharpness,
        }
    }
}
