use altd_sim::{
    gpu::simulation::{vehicle_desc, SurfaceTable},
    math::vec2::V2,
    physics::car::Car,
    track::world::World,
};

#[test]
fn snowmobile_can_use_the_gpu_vehicle_layout() {
    let scene: serde_json::Value =
        serde_json::from_str(include_str!("../assets/scenes/snowmobile_template.json")).unwrap();
    assert_eq!(scene["vehicle"]["wheels"].as_array().unwrap().len(), 5);
    let world = World::from_scene(&scene);
    vehicle_desc(&world.vehicle).expect("snowmobile must not fall back to CPU because of its wheel count");
}

#[test]
fn snowmobile_gpu_state_roundtrip_preserves_all_five_wheels() {
    let scene = serde_json::from_str(include_str!("../assets/scenes/snowmobile_template.json")).unwrap();
    let world = World::from_scene(&scene);
    let surfaces = SurfaceTable::new(&world.vehicle);
    let mut car = Car::new(&world, V2::ZERO, 0.0);
    for (i, wheel) in car.wheels.iter_mut().enumerate() {
        wheel.angle_deg = i as f64 * 0.5;
        wheel.previous_position = (i % 2 == 0).then_some(V2::new(i as f64, -(i as f64)));
    }
    let state = car.gpu_export(&world.vehicle, &surfaces).unwrap();
    let mut restored = Car::new(&world, V2::new(1.0, 2.0), 0.0);
    restored.gpu_import(&state, &surfaces).unwrap();
    assert_eq!(restored.wheels.len(), 5);
    for (original, restored) in car.wheels.iter().zip(&restored.wheels) {
        assert_eq!(original.angle_deg.to_bits(), restored.angle_deg.to_bits());
        assert_eq!(original.previous_position, restored.previous_position);
        assert_eq!(original.previous_surface, restored.previous_surface);
    }
}

#[test]
fn vehicles_above_five_wheels_still_report_the_gpu_limit() {
    let mut scene: serde_json::Value =
        serde_json::from_str(include_str!("../assets/scenes/snowmobile_template.json")).unwrap();
    let wheels = scene["vehicle"]["wheels"].as_array_mut().unwrap();
    wheels.push(wheels[4].clone());
    let world = World::from_scene(&scene);
    assert!(vehicle_desc(&world.vehicle)
        .unwrap_err()
        .contains("6 wheels exceed MAX_WHEELS"));
    let car = Car::new(&world, V2::ZERO, 0.0);
    assert!(car
        .gpu_export(&world.vehicle, &SurfaceTable::new(&world.vehicle))
        .unwrap_err()
        .contains("6 wheels exceed MAX_WHEELS"));
}
