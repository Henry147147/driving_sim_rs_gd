//! Native environment-only benchmark; policy inference and Python bridge excluded.
use altd_sim::{
    rl::{EnvConfig, RlVecEnv},
    track::world::World,
    training::SensorLayout,
};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let gpu = args.get(1).is_some_and(|s| s == "gpu");
    let mut reports = Vec::new();
    for n in [64, 256, 1024, 4096] {
        let scene = serde_json::from_str(include_str!("../configs/rl/tracks/rally_ring_gpu.scene.json")).unwrap();
        let route = serde_json::from_str(include_str!("../configs/rl/tracks/rally_ring.rules.json")).unwrap();
        let layout = SensorLayout::from_ordered(
            &serde_json::from_str(include_str!("../configs/rl/rally.sensors.json")).unwrap(),
        )
        .unwrap();
        let cfg = EnvConfig {
            schema: "altd_rl_env_v1".into(),
            num_envs: n,
            action_repeat_ticks: 4,
            deadline_physics_ticks: 7200,
            target_gates: 64,
            task_kind: "full_lap".into(),
            reward_profile: "dense_progress_time_v1".into(),
            gamma: 1.,
            run_seed: 1729,
            math_profile: Default::default(),
            route,
        };
        let mut env = RlVecEnv::new(World::from_scene(&scene), layout, cfg).unwrap();
        if gpu {
            env.enable_gpu().unwrap();
        }
        let actions = vec![[0.05, 0.7, 0., 0., 0.]; n];
        let start = std::time::Instant::now();
        let mut ticks = 0;
        let mut resets = 0;
        for _ in 0..64 {
            let b = env.step(&actions).unwrap();
            ticks += b.info.iter().map(|i| i.tick_end - i.tick_start).sum::<u64>();
            let ids: Vec<_> = b
                .terminated
                .iter()
                .enumerate()
                .filter_map(|(i, &t)| t.then_some(i))
                .collect();
            if !ids.is_empty() {
                resets += ids.len();
                env.reset(&ids).unwrap();
            }
        }
        let seconds = start.elapsed().as_secs_f64();
        reports.push(serde_json::json!({"n":n,"steps":64,"physics_ticks":ticks,"resets":resets,"seconds":seconds,"transitions_per_second":n as f64*64./seconds}));
    }
    println!(
        "{}",
        serde_json::json!({"backend":if gpu {"gpu"}else{"cpu"},"scope":"environment including rules and selective resets; excludes initialization, policy and Python","reports":reports})
    );
}
