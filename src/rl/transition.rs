use super::reward::Reward;
use serde::Serialize;
#[derive(Debug, Serialize)]
pub struct TransitionInfo {
    pub env_id: usize,
    pub episode_id: u64,
    pub episode_seed: u64,
    pub decision_index: u64,
    pub tick_start: u64,
    pub tick_end: u64,
    pub actual_dt: f64,
    pub requested_action: [f64; 5],
    pub applied_action: [f64; 5],
    pub progress_before: f64,
    pub progress_after: f64,
    pub gate_cursor: usize,
    pub contact: bool,
    pub events: Vec<String>,
    pub reason: Option<String>,
    pub valid_completion: bool,
    pub valid_observation: bool,
    pub position: [f64; 2],
    pub velocity: [f64; 2],
    /// Raw IEEE bits survive JSON even if a diagnostic is NaN/Inf.
    pub raw_state_bits: Vec<u64>,
    pub reward: Reward,
    pub cumulative_return: f64,
    pub task_kind: String,
}
#[derive(Debug, Serialize)]
pub struct StepBatch {
    pub schema: &'static str,
    pub observations: Vec<Vec<f32>>,
    pub next_observations: Vec<Vec<f32>>,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub info: Vec<TransitionInfo>,
}
