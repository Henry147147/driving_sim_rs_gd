//! Game population captures replayed with every car stepped on its own.
//! The game keeps a population in one physics space, but these captures
//! match independent cars exactly.
use altd_sim::{
    math::godot_math::F2,
    physics::car::{Car, Controls, DT},
    track::curve::json_vector,
    track::world::World,
};
use serde_json::Value;

#[path = "support/oracle.rs"]
mod oracle;

fn check_game_population(fixture: &str, expected: u64) {
    let data: Value = serde_json::from_str(fixture).unwrap();
    let result = game_population::verify(
        &oracle::scene(&data),
        data["states"].as_array().unwrap(),
        data["controls"].as_array().unwrap(),
    );
    assert_eq!(result["exact"], expected, "{result}");
    assert!(result["errors"].as_array().unwrap().is_empty(), "{result}");
}

#[test]
fn coalesced_track_initialization_matches_native_contacts() {
    check_game_population(
        include_str!("fixtures/native_tilemap_generated_coalesced600.json"),
        1797,
    );
}

#[test]
fn tilemap_redraw_replaces_the_track_of_every_car() {
    check_game_population(include_str!("fixtures/native_tilemap_generated_redraw600.json"), 1797);
}

fn bits(v: F2) -> [u32; 2] {
    [v.x.to_bits(), v.y.to_bits()]
}
fn check_car_state(car: &Car, state: &Value, id: usize, tick: &Value) {
    for (key, actual, expected) in [
        ("position", F2::from(car.position), json_vector(&state["position"])),
        ("velocity", F2::from(car.velocity), json_vector(&state["velocity"])),
        ("basis_x", car.body_basis.0, json_vector(&state["basis"][0])),
        ("basis_y", car.body_basis.1, json_vector(&state["basis"][1])),
    ] {
        assert_eq!(bits(actual), bits(expected), "{key}, car {id}, tick {}", tick);
    }
    assert_eq!(
        (car.angular_velocity as f32).to_bits(),
        (state["angular"].as_f64().unwrap() as f32).to_bits(),
        "angular, car {id}, tick {}",
        tick
    );
    let contacts = state["contacts"].as_array().unwrap();
    assert_eq!(car.wall_contacts.len(), contacts.len());
    for (a, e) in car.wall_contacts.iter().zip(contacts) {
        assert_eq!(bits(F2::from(a.normal)), bits(json_vector(&e["normal"])));
        assert_eq!(bits(F2::from(a.point)), bits(json_vector(&e["point"])));
    }
}

#[test]
fn population_crosses_native_dynamic_leaf_capacity() {
    let data: Value = serde_json::from_str(include_str!("fixtures/native_population_large80.json")).unwrap();
    let world = World::from_scene(&data["scene"]);
    let frames = data["frames"].as_array().unwrap();
    let first = frames[0]["states"].as_array().unwrap();
    let mut cars: Vec<_> = first
        .iter()
        .map(|s| Car::new(&world, json_vector(&s["position"]).into(), 0.0))
        .collect();
    let step = |cars: &mut [Car]| {
        for car in cars {
            car.step(&world, &Controls::default(), DT, false);
        }
    };
    for _ in 0..8 {
        step(&mut cars);
    }
    for (car, state) in cars.iter_mut().zip(first) {
        car.velocity = json_vector(&state["velocity"]).into();
    }
    for frame in frames.iter().skip(1) {
        step(&mut cars);
        let expected = frame["states"].as_array().unwrap();
        assert_eq!(cars.len(), expected.len());
        for (id, (car, state)) in cars.iter().zip(expected).enumerate() {
            check_car_state(car, state, id, &frame["tick"]);
        }
    }
}

#[path = "support/population.rs"]
mod game_population;
#[test]
fn packed_game_population_preserves_reset_callbacks_and_contacts() {
    check_game_population(include_str!("fixtures/native_game_population180.json"), 537);
}
#[test]
fn generated_tilemap_population_matches_native_drives_and_resets() {
    for (fixture, expected) in [
        (include_str!("fixtures/native_tilemap_generated_slot05_drive.json"), 537),
        (include_str!("fixtures/native_tilemap_generated_slot06_drive.json"), 537),
        (
            include_str!("fixtures/native_tilemap_generated_slot05_contact600.json"),
            1797,
        ),
        (
            include_str!("fixtures/native_tilemap_generated_slot06_contact600.json"),
            1797,
        ),
    ] {
        check_game_population(fixture, expected);
    }
}
#[test]
fn original_vehicle_types_preserve_native_tilemap_contact_bits() {
    for (fixture, expected) in [
        (include_str!("fixtures/native_tilemap_generated_truck600.json"), 1797),
        (
            include_str!("fixtures/native_tilemap_generated_snowmobile600.json"),
            1797,
        ),
    ] {
        check_game_population(fixture, expected);
    }
}
