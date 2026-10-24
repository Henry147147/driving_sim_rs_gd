use crate::{
    gpu::{
        hip::GpuWorld,
        simulation::{sensor_desc, vehicle_desc, GpuAgent, GpuSim, RlReport},
    },
    math::{godot_math::F2, vec2::V2},
    physics::car::Car,
    track::world::World,
    training::{session::hip, SensorLayout},
};
pub(crate) struct Backend {
    pub sim: GpuSim<'static>,
    pub world: GpuWorld<'static>,
    _claim: hip::Claim,
}
impl Backend {
    pub fn new(world: &World, layout: &SensorLayout, cars: &[&Car]) -> Result<Self, String> {
        if world.track.native_broadphase {
            return Err("GPU RL does not support native broadphase".into());
        }
        if world.track.curve.is_none()
            && layout.sensors.iter().any(|s| {
                matches!(
                    s,
                    crate::physics::car::Sensor::CorrectDirection | crate::physics::car::Sensor::TrackCurvature { .. }
                )
            })
        {
            return Err("GPU path sensors require an explicit baked Curve2D; use rally_ring_gpu.scene.json".into());
        }
        let claim = hip::Claim::take().ok_or("another session is using the GPU")?;
        let (gpu, _) = hip::device()?;
        let gw = GpuWorld::try_from_prepared(gpu, crate::gpu::hip::PreparedGpuWorld::new(world)?)?;
        let mut sim = GpuSim::try_new(
            &gw,
            &vehicle_desc(&world.vehicle)?,
            &layout.sensors.iter().map(sensor_desc).collect::<Vec<_>>(),
            cars.len(),
        )?;
        let gc: Vec<_> = cars
            .iter()
            .map(|c| c.gpu_export(&world.vehicle, &gw.arrays.surfaces))
            .collect::<Result<_, _>>()?;
        sim.try_upload(&gc, Some(&vec![GpuAgent::default(); cars.len()]))?;
        Ok(Self {
            sim,
            world: gw,
            _claim: claim,
        })
    }
    pub fn reset(&self, id: usize, car: &Car, world: &World) -> Result<(), String> {
        self.sim
            .rl_reset(id, &car.gpu_export(&world.vehicle, &self.world.arrays.surfaces)?)
    }
}
/// Copies only externally visible facts. Hidden contact history remains authoritative on device.
pub(crate) fn apply(car: &mut Car, r: &RlReport) {
    let p = r.pose;
    car.position = V2::new(p[0] as f64, p[1] as f64);
    car.velocity = V2::new(p[2] as f64, p[3] as f64);
    car.acceleration = V2::new(p[4] as f64, p[5] as f64);
    car.body_basis = (F2 { x: p[6], y: p[7] }, F2 { x: p[8], y: p[9] });
    car.rotation = p[10] as f64;
    car.transform_angle = p[11] as f64;
    car.angular_velocity = p[12] as f64;
    car.boost_energy = p[13] as f64;
    for (j, w) in car.wheels.iter_mut().enumerate() {
        w.angle_deg = p[14 + j] as f64;
    }
    car.collision_count = r.flags[0] as u64;
    car.tick = r.tick;
}

impl super::env::RlVecEnv {
    pub(crate) fn step_gpu(&mut self, actions: &[[f64; 5]]) -> Result<super::transition::StepBatch, String> {
        use super::{
            action,
            env::{point, wheels},
            reward,
            transition::{StepBatch, TransitionInfo},
        };
        use crate::physics::car::DT;
        let n = self.config.num_envs;
        let ids: Vec<_> = (0..n).collect();
        let inputs = self.observe(&ids)?;
        let starts: Vec<_> = self.slots.iter().map(|s| s.elapsed).collect();
        let before: Vec<_> = self.slots.iter().map(|s| s.tracker.progress).collect();
        let mut events: Vec<Vec<String>> = vec![Vec::new(); n];
        let mut contacts = vec![false; n];
        let mut successes = vec![false; n];
        let gpu = self.gpu.as_ref().unwrap();
        gpu.sim
            .rl_controls(&actions.iter().copied().map(action::pack_gpu).collect::<Vec<_>>())?;
        for _ in 0..self.config.action_repeat_ticks {
            let live: Vec<u8> = events.iter().map(|x| x.is_empty() as u8).collect();
            if live.iter().all(|x| *x == 0) {
                break;
            }
            let old: Vec<_> = self
                .slots
                .iter()
                .map(|s| (point(s.car.position), wheels(&self.world, &s.car)))
                .collect();
            let reports = gpu.sim.rl_tick(&live)?;
            let cfg = &self.config;
            let world = &self.world;
            use rayon::prelude::*;
            self.slots
                .par_iter_mut()
                .with_min_len(256)
                .zip(events.par_iter_mut())
                .zip(contacts.par_iter_mut())
                .zip(successes.par_iter_mut())
                .enumerate()
                .for_each(|(i, (((s, events_i), contact_i), success_i))| {
                    if live[i] == 0 {
                        return;
                    }
                    let r = &reports[i];
                    apply(&mut s.car, r);
                    s.elapsed += 1;
                    let next = point(s.car.position);
                    let new_wheels = wheels(world, &s.car);
                    if r.flags[2] != 0 || r.pose.iter().any(|x| !x.is_finite()) {
                        events_i.push("numerical_failure".into());
                    } else if let Err(reason) =
                        s.tracker
                            .advance(&cfg.route, cfg.target_gates, old[i].0, next, &old[i].1, &new_wheels)
                    {
                        events_i.push(reason.into());
                    }
                    *contact_i |= r.flags[3] != 0;
                    if *contact_i {
                        events_i.push("wall_contact".into());
                    }
                    *success_i = s.tracker.cursor >= cfg.target_gates;
                    if *success_i {
                        events_i.push("valid_completion".into());
                    }
                    if s.elapsed >= cfg.deadline_physics_ticks {
                        events_i.push("deadline".into());
                    }
                });
        }
        for (s, a) in self.slots.iter_mut().zip(actions) {
            s.previous = *a;
        }
        // ALL explicitly reads every row, including terminal states. No masked stale rows are used.
        let raw = gpu.sim.sensors(crate::gpu::simulation::ReadMask::ALL);
        let k = self.layout.names.len();
        let mut b = StepBatch {
            schema: "altd_rl_transition_v1",
            observations: inputs,
            next_observations: Vec::new(),
            rewards: Vec::new(),
            terminated: Vec::new(),
            truncated: vec![false; n],
            info: Vec::new(),
        };
        for i in 0..n {
            let s = &mut self.slots[i];
            let (next, valid) =
                super::observation::finish(&self.layout, &self.config, s, raw[i * k..(i + 1) * k].to_vec());
            if !valid {
                events[i].insert(0, "numerical_failure".into());
            }
            let failure = events[i]
                .iter()
                .any(|x| !["valid_completion", "deadline"].contains(&x.as_str()))
                || (!successes[i] && events[i].iter().any(|x| x == "deadline"));
            let success = successes[i] && !failure;
            s.done = failure || success;
            let reason = if failure {
                events[i]
                    .iter()
                    .find(|x| x.as_str() != "valid_completion" && x.as_str() != "deadline")
                    .cloned()
                    .or(Some("deadline".into()))
            } else if success {
                Some("valid_completion".into())
            } else {
                None
            };
            let r = reward::compute(
                before[i],
                s.tracker.progress,
                s.elapsed - starts[i],
                s.elapsed,
                self.config.deadline_physics_ticks,
                failure,
                self.config.reward_profile == "strict_time_v1",
            );
            s.total += r.total;
            b.info.push(TransitionInfo {
                env_id: i,
                episode_id: s.episode,
                episode_seed: s.seed,
                decision_index: s.decision,
                tick_start: starts[i],
                tick_end: s.elapsed,
                actual_dt: (s.elapsed - starts[i]) as f64 * DT,
                requested_action: actions[i],
                applied_action: actions[i],
                progress_before: before[i],
                progress_after: s.tracker.progress,
                gate_cursor: s.tracker.cursor,
                contact: contacts[i],
                events: events[i].clone(),
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
                task_kind: self.config.task_kind.clone(),
            });
            s.decision += 1;
            b.next_observations.push(next);
            b.rewards.push(r.total);
            b.terminated.push(s.done);
        }
        Ok(b)
    }
}
