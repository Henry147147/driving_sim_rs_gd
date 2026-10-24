use altd_sim::{
    physics::car::{Car, DT},
    rl::{
        action, reward,
        rules::{Route, Tracker},
        EnvConfig, RlVecEnv,
    },
    track::world::World,
    training::SensorLayout,
};
fn env(n: usize, repeat: u32, deadline: u64) -> RlVecEnv {
    let route: Route = serde_json::from_str(include_str!("../configs/rl/tracks/rally_ring.rules.json")).unwrap();
    let scene = serde_json::from_str(include_str!("../configs/rl/tracks/rally_ring.scene.json")).unwrap();
    let sensors = serde_json::from_str(include_str!("../configs/rl/rally.sensors.json")).unwrap();
    let config = EnvConfig {
        schema: "altd_rl_env_v1".into(),
        num_envs: n,
        action_repeat_ticks: repeat,
        deadline_physics_ticks: deadline,
        target_gates: 64,
        task_kind: "full_lap".into(),
        reward_profile: "dense_progress_time_v1".into(),
        gamma: 1.,
        run_seed: 1729,
        math_profile: Default::default(),
        route,
    };
    RlVecEnv::new(
        World::from_scene(&scene),
        SensorLayout::from_ordered(&sensors).unwrap(),
        config,
    )
    .unwrap()
}
#[test]
fn external_actions_match_direct_physics_and_batch_size() {
    for repeat in [1, 4] {
        let mut one = env(1, repeat, 7200);
        let mut many = env(8, repeat, 7200);
        let mut direct = Car::new(&one.world, one.car(0).position, one.car(0).transform_angle);
        let a = [0.1, 0.7, 0., 0., 0.];
        let controls = action::validate(&a).unwrap();
        for _ in 0..12 {
            for _ in 0..repeat {
                direct.step(&one.world, &controls, DT, false);
            }
            let b = one.step(&[a]).unwrap();
            let m = many.step(&[a; 8]).unwrap();
            assert_eq!(one.car(0).position, direct.position);
            assert_eq!(one.car(0).velocity, direct.velocity);
            assert_eq!(one.car(0).body_basis.0.x.to_bits(), direct.body_basis.0.x.to_bits());
            assert_eq!(one.car(0).body_basis.1.y.to_bits(), direct.body_basis.1.y.to_bits());
            assert_eq!(one.car(0).boost_energy, direct.boost_energy);
            assert_eq!(b.next_observations, m.next_observations[..1]);
            assert_eq!(b.info[0].tick_end - b.info[0].tick_start, repeat as u64);
        }
    }
}
#[test]
fn terminal_duration_and_explicit_reset() {
    for ticks in [1, 2, 3] {
        let mut e = env(2, 4, ticks);
        let b = e.step(&[[0.; 5]; 2]).unwrap();
        assert!(b.terminated.iter().all(|x| *x));
        assert!(b.truncated.iter().all(|x| !*x));
        assert_eq!(b.info[0].actual_dt, ticks as f64 / 60.);
        assert!((b.info[0].cumulative_return + 2.).abs() < 1e-12);
        assert!(e.step(&[[0.; 5]; 2]).is_err());
        let other = e.car(1).position;
        let reset = e.reset(&[0]).unwrap();
        assert_eq!(e.car(1).position, other);
        assert_eq!(e.car(1).tick, ticks);
        let mut fresh = env(1, 4, ticks);
        assert_eq!(reset, fresh.observe(&[0]).unwrap());
        assert!(e.reset(&[0, 0]).is_err());
    }
}
#[test]
fn action_basis_and_invalid_inputs() {
    for j in 0..5 {
        let mut a = [0.; 5];
        a[j] = 1.;
        let c = action::validate(&a).unwrap();
        assert_eq!(c.get(action::NAMES[j]), 1.);
        assert_eq!(action::unpack_gpu(action::pack_gpu(a)), a);
    }
    let mut e = env(1, 4, 60);
    assert!(e.step(&[[f64::NAN; 5]]).is_err());
    assert!(e.step(&[[0., 0., 0., -0.1, 0.]]).is_err());
    assert_eq!(e.car(0).tick, 0);
    assert_eq!(e.observe(&[0]).unwrap()[0].len(), 32);
}
#[test]
fn reward_episode_identities() {
    for strict in [false, true] {
        for progress in [0., 0.1, 0.75, 1.] {
            for time in [2, 600, 7200] {
                let dense = reward::compute(0., progress, time, time, 7200, false, strict).total;
                let terminal = reward::compute(progress, progress, 0, time, 7200, true, strict).total;
                assert!((dense + terminal - if strict { -2. } else { progress - 2. }).abs() < 1e-12);
            }
        }
    }
    for ticks in [3600, 5400, 7200] {
        let r = reward::compute(0., 1., ticks, ticks, 7200, false, false);
        assert!((r.total - (1. - ticks as f64 / 7200.)).abs() < 1e-12);
    }
    assert_eq!(reward::compute(0.2, 0.2, 4, 100, 7200, false, false).failure, 0.);
}
#[test]
fn route_reverse_seam_oscillation_and_shortcuts() {
    let e = env(1, 1, 7200);
    let r = &e.config.route;
    let mut t = Tracker::default();
    let a = r.spawn;
    let b = [a[0], a[1] + 20.];
    t.advance(r, 64, a, b, &[a], &[b]).unwrap();
    assert!(t.progress > 0.);
    t.advance(r, 64, b, a, &[b], &[a]).unwrap();
    assert!(t.progress.abs() < 1e-12);
    assert_eq!(t.cursor, 0);
    let backwards = [a[0], a[1] - 20.];
    t.advance(r, 64, a, backwards, &[a], &[backwards]).unwrap();
    assert_eq!(t.progress, 0.);
    assert_eq!(t.cursor, 0);
    assert!(t
        .advance(r, 64, backwards, [0., 0.], &[backwards], &[[0., 0.]])
        .is_err());
    let jump = r.cells[20].start;
    assert!(t.advance(r, 64, a, jump, &[a], &[jump]).is_err());
    // Traverse every real cell; immediate start crossing never counts as a lap.
    let mut t = Tracker::default();
    let mut a = r.spawn;
    for i in 0..64 {
        let c = &r.cells[i];
        let d = [c.end[0] - c.start[0], c.end[1] - c.start[1]];
        let b = [c.end[0] + d[0] * 0.001, c.end[1] + d[1] * 0.001];
        let done = t.advance(r, 64, a, b, &[a], &[b]).unwrap();
        assert_eq!(done, i == 63);
        a = b;
    }
    assert_eq!(t.progress, 1.);
}

#[test]
#[ignore = "requires built ALTD_GPU_LIB and a supported device"]
fn gpu_external_actions_reset_and_terminal_parity() {
    fn gpu_scene_env() -> RlVecEnv {
        let base = env(3, 4, 7200);
        let scene = serde_json::from_str(include_str!("../configs/rl/tracks/rally_ring_gpu.scene.json")).unwrap();
        RlVecEnv::new(World::from_scene(&scene), base.layout, base.config).unwrap()
    }
    let mut cpu = gpu_scene_env();
    let mut gpu = gpu_scene_env();
    gpu.enable_gpu().unwrap();
    for tick in 0..100 {
        let actions = [[0.05, 0.6, 0., 0., 0.], [0., 0.9, 0., 0., 0.], [0.1, 0.4, 0., 0., 0.]];
        let a = cpu.step(&actions).unwrap();
        let b = gpu.step(&actions).unwrap();
        for i in 0..3 {
            assert_eq!(cpu.car(i).position, gpu.car(i).position, "tick {tick} car {i}");
            assert_eq!(cpu.car(i).velocity, gpu.car(i).velocity);
            assert_eq!(a.terminated[i], b.terminated[i]);
            assert_eq!(a.rewards[i], b.rewards[i]);
            assert_eq!(a.next_observations[i], b.next_observations[i]);
        }
        let ids: Vec<_> = a
            .terminated
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| t.then_some(i))
            .collect();
        if !ids.is_empty() {
            assert_eq!(cpu.reset(&ids).unwrap(), gpu.reset(&ids).unwrap());
        }
        if tick == 20 {
            assert_eq!(cpu.reset(&[1]).unwrap(), gpu.reset(&[1]).unwrap());
        }
    }
}

#[test]
fn gate_consumed_once_and_reverse_across_seam_returns_zero() {
    let e = env(1, 1, 7200);
    let r = &e.config.route;
    let c = &r.cells[0];
    let d = [c.end[0] - c.start[0], c.end[1] - c.start[1]];
    let b = [c.end[0] + d[0] * 0.01, c.end[1] + d[1] * 0.01];
    let mut t = Tracker::default();
    t.advance(r, 64, r.spawn, b, &[r.spawn], &[b]).unwrap();
    assert_eq!(t.cursor, 1);
    let a = c.start;
    t.advance(r, 64, b, a, &[b], &[a]).unwrap();
    t.advance(r, 64, a, b, &[a], &[b]).unwrap();
    assert_eq!(t.cursor, 1);
    let mut t = Tracker::default();
    let back = [r.spawn[0], r.spawn[1] - 20.];
    t.advance(r, 64, r.spawn, back, &[r.spawn], &[back]).unwrap();
    let front = [r.spawn[0], r.spawn[1] + 1.];
    t.advance(r, 64, back, front, &[back], &[front]).unwrap();
    assert_eq!(t.cursor, 0);
    assert!(t.progress < 0.001);
}

#[test]
fn valid_finish_at_deadline_precedes_timeout() {
    let mut e = env(1, 4, 62);
    e.config.target_gates = 1;
    e.config.task_kind = "segment".into();
    loop {
        let b = e.step(&[[0., 1., 0., 0., 0.]]).unwrap();
        if b.terminated[0] {
            assert!(b.info[0].valid_completion);
            assert_eq!(b.info[0].tick_end, 62);
            assert!(b.info[0].events.iter().any(|x| x == "deadline"));
            assert!((b.info[0].cumulative_return).abs() < 1e-12);
            break;
        }
    }
}
#[test]
fn reset_override_validation_is_atomic() {
    let mut e = env(2, 1, 100);
    e.step(&[[0., 1., 0., 0., 0.]; 2]).unwrap();
    let state = e.car(0).position;
    assert!(e
        .reset_with(&[0, 1], None, Some(&[[1000., 0., std::f64::consts::PI], [0., 0., 0.]]))
        .is_err());
    assert_eq!(state, e.car(0).position);
    e.reset_with(&[0], Some(&[42]), None).unwrap();
    let b = e.step(&[[0.; 5]; 2]).unwrap();
    assert_eq!(b.info[0].episode_seed, 42);
    assert_ne!(b.info[1].episode_seed, 42);
}
