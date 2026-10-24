//! Independent, externally controlled time-trial environments. Physics is unchanged.
pub mod action;
pub mod config;
pub mod env;
pub mod observation;
pub mod reward;
pub mod rules;
pub mod transition;
pub use config::EnvConfig;
pub use env::RlVecEnv;

#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;

pub mod legacy;
