use altd_sim::{
    math::vec2::V2,
    physics::car::{Car, Controls, DT},
    track::world::World,
};
use serde_json::{json, Value};

/// Replays a game population capture with every car stepped on its own.
pub(crate) fn verify(scene: &Value, states: &[Value], controls: &[Value]) -> Value {
    let mut world = World::from_scene(scene);
    let count = states.last().unwrap()["tick"].as_u64().unwrap() as usize + 1;
    let mut frames = vec![Vec::new(); count];
    let mut outputs = vec![Vec::new(); count];
    for state in states {
        frames[state["tick"].as_u64().unwrap() as usize].push(state);
    }
    for control in controls {
        outputs[control["tick"].as_u64().unwrap() as usize].push(control);
    }
    let mut cars: Vec<Car> = frames[0].iter().map(|s| vehicle::restore(&world, s)).collect();
    for _ in 0..8 {
        for car in &mut cars {
            car.passive_step(&world, DT, false);
        }
    }
    let mut errors = Vec::new();
    let mut exact = 0;
    let reset_tick = scene["reset_tick"].as_u64().unwrap_or(90) as usize;
    let empty = Vec::new();
    let redraws = scene["redraw_events"].as_array().unwrap_or(&empty);
    for tick in 0..frames.len() - 1 {
        if let Some(event) = redraws.iter().find(|e| e["tick"].as_u64() == Some(tick as u64)) {
            // A new track invalidates every car's pairs and warm contacts.
            world = World::from_scene(&event["scene"]);
            let position = V2::from(altd_sim::track::curve::json_vector(&event["spawn"]));
            let rotation = event["rotation"].as_f64().unwrap();
            for car in &mut cars {
                car.clear_track_contacts();
                car.queue_reset(position, rotation);
            }
        }
        if redraws.is_empty() && tick == reset_tick {
            for car in &mut cars {
                let position = scene
                    .get("reset_position")
                    .map(|v| V2::from(altd_sim::track::curve::json_vector(v)))
                    .unwrap_or(V2::new(3840.0, 1664.0));
                let rotation = scene["reset_rotation"]
                    .as_f64()
                    .unwrap_or(std::f32::consts::FRAC_PI_2 as f64);
                car.queue_reset(position, rotation);
            }
        }
        let controls: Vec<_> = outputs[tick]
            .iter()
            .map(|c| {
                let c = &c["output"];
                Controls {
                    acceleration: c[0].as_f64().unwrap(),
                    steering: c[1].as_f64().unwrap(),
                    brake: c[2].as_f64().unwrap(),
                    handbrake: c[3].as_f64().unwrap(),
                    boost: c[4].as_f64().unwrap(),
                }
            })
            .collect();
        let waiting = if redraws.is_empty() {
            (reset_tick..reset_tick + 8).contains(&tick)
        } else {
            redraws.iter().any(|e| {
                let t = e["tick"].as_u64().unwrap() as usize;
                (t..t + 8).contains(&tick)
            })
        };
        for (i, car) in cars.iter_mut().enumerate() {
            if waiting {
                car.passive_step(&world, DT, false);
            } else {
                car.step(&world, &controls[i], DT, false);
            }
        }
        assert_eq!(cars.len(), frames[tick + 1].len());
        for (i, (car, s)) in cars.iter().zip(&frames[tick + 1]).enumerate() {
            let bad = vehicle::differences(car, s);
            if bad.as_object().unwrap().is_empty() {
                exact += 1;
            } else {
                errors.push(json!({"tick":tick+1,"car":i,"differences":bad}));
            }
        }
    }
    json!({"transitions":states.len()-frames[0].len(),"exact":exact,"errors":errors})
}
#[path = "vehicle.rs"]
mod vehicle;
