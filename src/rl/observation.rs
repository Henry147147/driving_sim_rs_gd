use super::{action, config::EnvConfig, env::Slot};
use crate::{track::world::World, training::SensorLayout};
use serde_json::{json, Value};
pub fn schema(layout: &SensorLayout) -> Value {
    let mut names = layout.names.clone();
    names.extend(action::NAMES.iter().map(|x| format!("previous_{x}")));
    names.extend(
        [
            "remaining_time",
            "validated_progress",
            "gate_forward",
            "gate_side",
            "lateral_offset",
            "remaining_distance",
            "task_kind",
        ]
        .iter()
        .map(|x| x.to_string()),
    );
    json!({"schema":"native_rl_v1","names":names,"sensor_descriptors":layout.sensors.iter().map(|s|format!("{s:?}")).collect::<Vec<_>>(),"dtype":"float32","sensor_transform":"existing SensorLayout definitions, no extra clipping","extra_scales":[1,1,1,1,1,1,1,1000,1000,"local_half_width",1000,1],"extra_clip":"[-10,10] except time/progress [0,1]","missing_data":"terminate numerical_failure; finite zero placeholder; no bootstrap"})
}
pub(crate) fn read(world: &World, layout: &SensorLayout, cfg: &EnvConfig, s: &mut Slot) -> (Vec<f32>, bool) {
    if [
        s.car.position.x,
        s.car.position.y,
        s.car.velocity.x,
        s.car.velocity.y,
        s.car.angular_velocity,
        s.car.rotation,
        s.car.boost_energy,
    ]
    .iter()
    .any(|x| !x.is_finite())
    {
        return (vec![0.; layout.names.len() + 12], false);
    }
    let mut v = Vec::new();
    layout.read_into(world, &s.car, &mut s.scratch, &mut v);
    finish(layout, cfg, s, v)
}
pub(crate) fn finish(layout: &SensorLayout, cfg: &EnvConfig, s: &Slot, mut v: Vec<f64>) -> (Vec<f32>, bool) {
    v.extend(s.previous);
    let c = &cfg.route.cells[s.tracker.cursor.min(cfg.route.cells.len() - 1)];
    let dx = c.end[0] - s.car.position.x;
    let dy = c.end[1] - s.car.position.y;
    let right = s.car.body_basis.0;
    let forward = s.car.body_basis.1;
    let active = &cfg.route.cells[s.tracker.cell];
    let d = [active.end[0] - active.start[0], active.end[1] - active.start[1]];
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let w =
        ((active.gate[1][0] - active.gate[0][0]).powi(2) + (active.gate[1][1] - active.gate[0][1]).powi(2)).sqrt() / 2.;
    let lateral = (d[0] * (s.car.position.y - active.start[1]) - d[1] * (s.car.position.x - active.start[0])) / l / w;
    v.extend([
        (cfg.deadline_physics_ticks.saturating_sub(s.elapsed) as f64) / (cfg.deadline_physics_ticks as f64),
        s.tracker.progress,
        -(dx * forward.x as f64 + dy * forward.y as f64) / 1000.,
        (dx * right.x as f64 + dy * right.y as f64) / 1000.,
        lateral,
        (cfg.route.length(cfg.target_gates) - s.tracker.signed_distance).max(0.) / 1000.,
        if cfg.task_kind == "full_lap" { 1. } else { 0. },
    ]);
    for x in v.iter_mut().skip(layout.names.len()) {
        *x = x.clamp(-10., 10.);
    }
    let valid = v.iter().all(|x| (*x as f32).is_finite());
    (
        v.into_iter().map(|x| if valid { x as f32 } else { 0. }).collect(),
        valid,
    )
}
