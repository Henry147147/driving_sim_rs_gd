//! Training agents own their controls, statistics, and inference scratch.

use super::TrainingStats;
use crate::nn::network::{ForwardScratch, Network};
use crate::physics::car::{Car, Controls, SensorScratch, DT};
use crate::track::world::World;
use crate::training::evolution::AgentResult;

/// Per-agent scratch buffers, kept with the agent so threads never share them.
#[derive(Default, Clone)]
pub(super) struct AgentScratch {
    pub(super) sensors: SensorScratch,
    pub(super) inputs: Vec<f64>,
    pub(super) forward: ForwardScratch,
}

#[derive(Clone)]
pub struct TrainingAgent {
    pub network: Network,
    pub car: Car,
    pub stats: TrainingStats,
    pub controls: Controls,
    pub pending_contact: bool,
    /// Tick at which the car became inactive in the current window.
    pub deactivated_at: Option<u64>,
    pub(super) scratch: AgentScratch,
}

impl TrainingAgent {
    /// `Schedule::update_stats`: statistics on ticks where
    /// `tick % 6 == stats_phase`, then the idle elimination.
    pub fn update_stats(&mut self, world: &World, tick: u64, stats_phase: u64, eliminate_when_idle: bool) {
        let was_active = self.car.active;
        if tick % 6 == stats_phase {
            let first_stats_tick = if stats_phase == 0 { 6 } else { stats_phase };
            self.stats.update(
                world,
                &self.car,
                &self.controls,
                tick - first_stats_tick,
                self.pending_contact,
            );
            self.pending_contact = false;
            if self.car.active && eliminate_when_idle && self.stats.idle_ticks > 80 {
                self.car.deactivate();
            }
        }
        if was_active && !self.car.active {
            self.deactivated_at = Some(tick);
        }
    }
    /// One physics tick: `Car::step` with the agent's controls when `drive`,
    /// otherwise `Car::passive_step`. Records contacts and deactivation, and
    /// the boost of driving steps (`TrainingStats::account_boost`).
    pub(super) fn step(&mut self, world: &World, eliminate_on_wall: bool, drive: bool, tick: u64) {
        let was_active = self.car.active;
        self.pending_contact |= if drive {
            self.stats.account_boost(world, &self.car, &self.controls);
            self.car.step(world, &self.controls, DT, eliminate_on_wall)
        } else {
            self.car.passive_step(world, DT, eliminate_on_wall)
        };
        if was_active && !self.car.active {
            self.deactivated_at = Some(tick);
        }
    }
    pub fn result(&self) -> AgentResult<'_> {
        AgentResult {
            network: &self.network,
            metrics: self.stats.metrics(),
            update_count: self.stats.update_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec2::V2;

    #[test]
    fn driving_steps_account_boost_on_every_tick() {
        let world =
            World::from_scene(&serde_json::from_str(include_str!("../../assets/scenes/rally_template.json")).unwrap());
        let mut agent = TrainingAgent {
            network: Network::from_vector(&[1, 1], vec![0.0; 2]),
            car: Car::new(&world, V2::ZERO, 0.0),
            stats: TrainingStats::new(0.0),
            controls: Controls {
                acceleration: 1.0,
                boost: 1.0,
                ..Controls::default()
            },
            pending_contact: false,
            deactivated_at: None,
            scratch: AgentScratch::default(),
        };
        // Passive steps neither drive nor boost.
        agent.step(&world, false, false, 0);
        assert_eq!(
            (agent.stats.total_boost_useful, agent.stats.total_boost_wasted),
            (0.0, 0.0)
        );
        let mut tick = 0;
        while agent.car.boost_energy > 0.0 {
            tick += 1;
            agent.step(&world, false, true, tick);
        }
        // Full throttle from a standstill: all of the tank adds thrust. The
        // float drains are exact in f64, so they sum to the whole tank.
        assert_eq!(
            (agent.stats.total_boost_useful, agent.stats.total_boost_wasted),
            (1.0, 0.0)
        );
        for _ in 0..10 {
            tick += 1;
            agent.step(&world, false, true, tick);
        }
        // Ten forgone recharges on the empty tank, then none without boost.
        let forgone = (0..10).fold(0.0, |total, _| total + 0.002);
        assert_eq!(agent.stats.total_boost_wasted, forgone);
        agent.controls.boost = 0.0;
        agent.step(&world, false, true, tick + 1);
        assert_eq!(agent.stats.total_boost_wasted, forgone);
    }
}
