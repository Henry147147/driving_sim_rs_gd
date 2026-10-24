use super::{
    action,
    config::{episode_seed, EnvConfig},
    observation, reward,
    rules::{Point, Tracker},
    transition::{StepBatch, TransitionInfo},
};
use crate::{
    math::{godot_math, vec2::V2},
    physics::car::{Car, SensorScratch, DT},
    track::world::World,
    training::SensorLayout,
};
use rayon::prelude::*;
pub(crate) struct Slot {
    pub car: Car,
    pub scratch: SensorScratch,
    pub tracker: Tracker,
    pub previous: [f64; 5],
    pub elapsed: u64,
    pub episode: u64,
    pub seed: u64,
    pub decision: u64,
    pub done: bool,
    pub total: f64,
}
pub struct RlVecEnv {
    pub world: World,
    pub layout: SensorLayout,
    pub config: EnvConfig,
    pub(crate) slots: Vec<Slot>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) gpu: Option<super::gpu::Backend>,
}
pub(crate) fn point(v: V2) -> Point {
    [v.x, v.y]
}
pub(crate) fn wheels(world: &World, car: &Car) -> Vec<Point> {
    world
        .vehicle
        .wheels
        .iter()
        .map(|w| point(godot_math::transform_point(car.position, car.body_basis, w.position)))
        .collect()
}
impl RlVecEnv {
    pub fn new(world: World, layout: SensorLayout, config: EnvConfig) -> Result<Self, String> {
        config.validate()?;
        if world.math != config.math_profile {
            return Err("world/config math profile mismatch".into());
        }
        let mut e = Self {
            world,
            layout,
            config,
            slots: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            gpu: None,
        };
        for i in 0..e.config.num_envs {
            e.slots.push(e.fresh(i, 0));
        }
        for s in &e.slots {
            let w = wheels(&e.world, &s.car);
            if w.iter().any(|&p| !e.config.route.legal_sweep(0, p, p)) {
                return Err("spawn footprint outside corridor".into());
            }
        }
        Ok(e)
    }
    fn fresh(&self, id: usize, episode: u64) -> Slot {
        let r = &self.config.route;
        Slot {
            car: Car::new(&self.world, V2::new(r.spawn[0], r.spawn[1]), r.rotation),
            scratch: SensorScratch::default(),
            tracker: Tracker::default(),
            previous: [0.; 5],
            elapsed: 0,
            episode,
            seed: episode_seed(self.config.run_seed, id, episode),
            decision: 0,
            done: false,
            total: 0.,
        }
    }
    pub fn reset(&mut self, ids: &[usize]) -> Result<Vec<Vec<f32>>, String> {
        self.validate_ids(ids)?;
        for &id in ids {
            self.slots[id] = self.fresh(id, self.slots[id].episode + 1);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(gpu) = &self.gpu {
            for &id in ids {
                gpu.reset(id, &self.slots[id].car, &self.world)?;
            }
        }
        self.observe(ids)
    }

    /// Explicit reset options: spawn cross-section must stay at route station zero.
    pub fn reset_with(
        &mut self,
        ids: &[usize],
        seeds: Option<&[u64]>,
        spawns: Option<&[[f64; 3]]>,
    ) -> Result<Vec<Vec<f32>>, String> {
        self.validate_ids(ids)?;
        if seeds.is_some_and(|x| x.len() != ids.len()) || spawns.is_some_and(|x| x.len() != ids.len()) {
            return Err("reset options must match env ids".into());
        }
        let mut replacements = Vec::new();
        for (j, &id) in ids.iter().enumerate() {
            let mut slot = self.fresh(id, self.slots[id].episode + 1);
            if let Some(seeds) = seeds {
                slot.seed = seeds[j];
            }
            if let Some(spawns) = spawns {
                let [x, y, rot] = spawns[j];
                if ![x, y, rot].iter().all(|x| x.is_finite()) {
                    return Err("nonfinite spawn".into());
                }
                let c = &self.config.route.cells[0];
                let dx = c.end[0] - c.start[0];
                let dy = c.end[1] - c.start[1];
                if ((x - c.start[0]) * dx + (y - c.start[1]) * dy).abs() > 1e-4 {
                    return Err("spawn must be on the initial cross-section".into());
                }
                slot.car = Car::new(&self.world, V2::new(x, y), rot);
                if wheels(&self.world, &slot.car)
                    .iter()
                    .any(|&p| !self.config.route.legal_sweep(0, p, p))
                {
                    return Err("spawn footprint outside corridor".into());
                }
            }
            replacements.push((id, slot));
        }
        for (id, slot) in replacements {
            self.slots[id] = slot;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(gpu) = &self.gpu {
            for &id in ids {
                gpu.reset(id, &self.slots[id].car, &self.world)?;
            }
        }
        self.observe(ids)
    }
    fn validate_ids(&self, ids: &[usize]) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        if ids.iter().any(|&id| id >= self.slots.len() || !seen.insert(id)) {
            return Err("invalid or duplicate env id".into());
        }
        Ok(())
    }
    pub fn observe(&mut self, ids: &[usize]) -> Result<Vec<Vec<f32>>, String> {
        self.validate_ids(ids)?;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(gpu) = &self.gpu {
            let raw = gpu.sim.sensors(crate::gpu::simulation::ReadMask::ALL);
            let k = self.layout.names.len();
            return ids
                .iter()
                .map(|&id| {
                    let (v, ok) = observation::finish(
                        &self.layout,
                        &self.config,
                        &self.slots[id],
                        raw[id * k..(id + 1) * k].to_vec(),
                    );
                    if ok {
                        Ok(v)
                    } else {
                        Err("nonfinite GPU observation".into())
                    }
                })
                .collect();
        }
        ids.iter()
            .map(|&id| {
                let (v, valid) = observation::read(&self.world, &self.layout, &self.config, &mut self.slots[id]);
                if valid {
                    Ok(v)
                } else {
                    Err("nonfinite observation".into())
                }
            })
            .collect()
    }
    pub fn schema(&self) -> serde_json::Value {
        observation::schema(&self.layout)
    }
    pub fn step(&mut self, actions: &[[f64; 5]]) -> Result<StepBatch, String> {
        if actions.len() != self.slots.len() {
            return Err("action shape must be [N,5]".into());
        }
        if self.slots.iter().any(|s| s.done) {
            return Err("selectively reset every completed environment before step".into());
        }
        let controls: Vec<_> = actions.iter().map(action::validate).collect::<Result<_, _>>()?;
        #[cfg(not(target_arch = "wasm32"))]
        if self.gpu.is_some() {
            return self.step_gpu(actions);
        }
        let world = &self.world;
        let layout = &self.layout;
        let cfg = &self.config;
        let rows: Vec<_> = self
            .slots
            .par_iter_mut()
            .enumerate()
            .map(|(id, s)| {
                let (input, input_valid) = observation::read(world, layout, cfg, s);
                let start = s.elapsed;
                let before = s.tracker.progress;
                let mut contact = false;
                let mut events = Vec::new();
                let mut success = false;
                if !input_valid {
                    events.push("numerical_failure".to_string());
                }
                for _ in 0..cfg.action_repeat_ticks {
                    if !events.is_empty() {
                        break;
                    }
                    let old = point(s.car.position);
                    let old_wheels = wheels(world, &s.car);
                    contact |= s.car.step(world, &controls[id], DT, false);
                    s.elapsed += 1;
                    let next = point(s.car.position);
                    let new_wheels = wheels(world, &s.car);
                    if next
                        .iter()
                        .chain(
                            [
                                s.car.velocity.x,
                                s.car.velocity.y,
                                s.car.rotation,
                                s.car.angular_velocity,
                                s.car.boost_energy,
                            ]
                            .iter(),
                        )
                        .any(|x| !x.is_finite())
                    {
                        events.push("numerical_failure".into());
                    } else if let Err(reason) =
                        s.tracker
                            .advance(&cfg.route, cfg.target_gates, old, next, &old_wheels, &new_wheels)
                    {
                        events.push(reason.into());
                    }
                    if contact {
                        events.push("wall_contact".into());
                    }
                    success = s.tracker.cursor >= cfg.target_gates;
                    if success {
                        events.push("valid_completion".into());
                    }
                    if s.elapsed >= cfg.deadline_physics_ticks {
                        events.push("deadline".into());
                    }
                    if !events.is_empty() {
                        break;
                    }
                }
                s.previous = actions[id];
                let (mut next, valid) = observation::read(world, layout, cfg, s);
                if !valid && !events.iter().any(|x| x == "numerical_failure") {
                    events.insert(0, "numerical_failure".into());
                }
                let failure = events
                    .iter()
                    .any(|x| !["valid_completion", "deadline"].contains(&x.as_str()))
                    || (!success && events.iter().any(|x| x == "deadline"));
                success &= !failure;
                s.done = failure || success;
                if !valid {
                    next.fill(0.);
                }
                let r = reward::compute(
                    before,
                    s.tracker.progress,
                    s.elapsed - start,
                    s.elapsed,
                    cfg.deadline_physics_ticks,
                    failure,
                    cfg.reward_profile == "strict_time_v1",
                );
                s.total += r.total;
                let reason = if failure {
                    events
                        .iter()
                        .find(|x| x.as_str() != "valid_completion" && x.as_str() != "deadline")
                        .cloned()
                        .or(Some("deadline".into()))
                } else if success {
                    Some("valid_completion".into())
                } else {
                    None
                };
                let info = TransitionInfo {
                    env_id: id,
                    episode_id: s.episode,
                    episode_seed: s.seed,
                    decision_index: s.decision,
                    tick_start: start,
                    tick_end: s.elapsed,
                    actual_dt: (s.elapsed - start) as f64 * DT,
                    requested_action: actions[id],
                    applied_action: actions[id],
                    progress_before: before,
                    progress_after: s.tracker.progress,
                    gate_cursor: s.tracker.cursor,
                    contact,
                    events,
                    reason,
                    valid_completion: success,
                    valid_observation: valid,
                    position: point(s.car.position),
                    velocity: point(s.car.velocity),
                    raw_state_bits: [
                        s.car.position.x,
                        s.car.position.y,
                        s.car.velocity.x,
                        s.car.velocity.y,
                        s.car.rotation,
                        s.car.angular_velocity,
                        s.car.boost_energy,
                    ]
                    .iter()
                    .map(|x| x.to_bits())
                    .collect(),
                    reward: r,
                    cumulative_return: s.total,
                    task_kind: cfg.task_kind.clone(),
                };
                s.decision += 1;
                (input, next, r.total, s.done, info)
            })
            .collect();
        let mut b = StepBatch {
            schema: "altd_rl_transition_v1",
            observations: Vec::new(),
            next_observations: Vec::new(),
            rewards: Vec::new(),
            terminated: Vec::new(),
            truncated: vec![false; rows.len()],
            info: Vec::new(),
        };
        for (o, n, r, t, i) in rows {
            b.observations.push(o);
            b.next_observations.push(n);
            b.rewards.push(r);
            b.terminated.push(t);
            b.info.push(i);
        }
        Ok(b)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn enable_gpu(&mut self) -> Result<(), String> {
        if self.gpu.is_some() {
            return Err("GPU already enabled".into());
        }
        self.gpu = Some(super::gpu::Backend::new(
            &self.world,
            &self.layout,
            &self.slots.iter().map(|s| &s.car).collect::<Vec<_>>(),
        )?);
        Ok(())
    }
    /// Read-only access for numerical differential tests and diagnostics.
    pub fn car(&self, id: usize) -> &Car {
        &self.slots[id].car
    }
}

#[cfg(test)]
mod numerical_tests {
    use super::*;
    #[test]
    fn numerical_failure_has_finite_storage_and_raw_diagnostics() {
        let scene = serde_json::from_str(include_str!("../../configs/rl/tracks/rally_ring.scene.json")).unwrap();
        let layout = SensorLayout::from_ordered(
            &serde_json::from_str(include_str!("../../configs/rl/rally.sensors.json")).unwrap(),
        )
        .unwrap();
        let config = EnvConfig {
            schema: "altd_rl_env_v1".into(),
            num_envs: 1,
            action_repeat_ticks: 4,
            deadline_physics_ticks: 100,
            target_gates: 64,
            task_kind: "full_lap".into(),
            reward_profile: "dense_progress_time_v1".into(),
            gamma: 1.,
            run_seed: 1729,
            math_profile: Default::default(),
            route: serde_json::from_str(include_str!("../../configs/rl/tracks/rally_ring.rules.json")).unwrap(),
        };
        let mut env = RlVecEnv::new(World::from_scene(&scene), layout, config).unwrap();
        env.slots[0].car.velocity.x = f64::NAN;
        let b = env.step(&[[0.; 5]]).unwrap();
        assert!(b.terminated[0]);
        assert!(!b.info[0].valid_observation);
        assert_eq!(b.info[0].reason.as_deref(), Some("numerical_failure"));
        assert!(b.next_observations[0].iter().all(|x| x.is_finite()));
        assert_eq!(b.rewards[0], -2.);
        assert!(f64::from_bits(b.info[0].raw_state_bits[2]).is_nan());
        assert!(serde_json::to_string(&b).is_ok());
    }
}
