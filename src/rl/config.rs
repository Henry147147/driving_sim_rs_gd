use super::rules::Route;
use crate::math::profile::MathProfile;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvConfig {
    pub schema: String,
    pub num_envs: usize,
    pub action_repeat_ticks: u32,
    pub deadline_physics_ticks: u64,
    pub target_gates: usize,
    pub task_kind: String,
    pub reward_profile: String,
    pub gamma: f64,
    pub run_seed: u64,
    pub math_profile: MathProfile,
    pub route: Route,
}
impl EnvConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != "altd_rl_env_v1"
            || self.num_envs == 0
            || self.action_repeat_ticks == 0
            || self.deadline_physics_ticks == 0
        {
            return Err("invalid environment schema/count/tick configuration".into());
        }
        if self.gamma != 1. || !["dense_progress_time_v1", "strict_time_v1"].contains(&self.reward_profile.as_str()) {
            return Err("reward profile requires gamma=1".into());
        }
        if !["full_lap", "segment"].contains(&self.task_kind.as_str())
            || self.target_gates == 0
            || self.target_gates > self.route.cells.len()
        {
            return Err("invalid task/target".into());
        }
        if self.task_kind == "full_lap" && (!self.route.closed || self.target_gates != self.route.cells.len()) {
            return Err("full_lap requires all gates of a closed route".into());
        }
        self.route.validate()
    }
}
/// SplitMix64: stable independent seed, no scheduling-dependent global RNG.
pub fn episode_seed(run: u64, env: usize, episode: u64) -> u64 {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    mix(run ^ mix(env as u64) ^ mix(episode.wrapping_add(1)))
}
