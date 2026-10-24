# Native racing reinforcement learning

This adds an independent time-trial execution mode. GA/ARS, their scheduler, checkpoints, server, WASM exports, and vehicle arithmetic remain intact. The checkout started at `ab3ec6799f55aaca5fb286e4fef60276b15f333a`. The six implementation milestones have executable implementations and smoke evidence; measured results and limitations are in [rl-results.md](rl-results.md). Passing a smoke run is not a general racing-performance claim.

## Install and run

Run commands from the repository root. Rust 1.93+ is required by the simulator; development validation used 1.95.0. Python dependencies are optional, outside the ordinary Rust crate. The separately locked binding crate pins PyO3 0.27.2. The tested Windows runtime is CPython 3.14.3, PyTorch 2.13.0+cpu, NumPy 2.5.2, PyYAML 6.0.3, maturin 1.12.6.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate; Linux/macOS: source .venv/bin/activate
python -m pip install maturin==1.12.6
maturin develop --release
python -m altd_rl.train --config configs/rl/segment_ppo.yaml --seed 1729
python -m altd_rl.train --resume runs/segment_ppo/checkpoint.pt --updates 10
python -m altd_rl.evaluate --checkpoint runs/segment_ppo/best.pt --episodes 8
python -m altd_rl.trace --checkpoint runs/segment_ppo/best.pt --episodes 3 --out runs/segment_trace
python -m altd_rl.inspect runs/segment_trace
```

`maturin build --release --out target/wheels`, followed by `pip install` of that wheel, is an alternative. `--updates` counts additional completed PPO updates on resume. `--seconds` stops between updates. Ctrl+C preserves the last completed checkpoint; an unfinished collection/update is discarded. Training defaults are starting values, not optimal settings.

For full laps:

```sh
python -m altd_rl.train --config configs/rl/rally_ppo.yaml --seed 1729
python -m altd_rl.evaluate --checkpoint runs/rally_ppo/best.pt --suite configs/rl/held_out_suite.json
python -m altd_rl.evaluate --checkpoint runs/rally_ppo/actor.pt --config configs/rl/rally_ppo.yaml --episodes 8
```

The shipped suite is a fixed-start regression suite on one analytic track. Its name is retained from the requested interface; it does **not** establish held-out track generalization. Seeds identify independent training runs and episode streams. The physics and supplied spawn are deterministic, so repeated frozen-policy episodes at that spawn reproduce one case. Reports explicitly leave the confidence interval unavailable. Add independently verified scenes/routes and distinct starts before making generalization claims.

## Milestone map

1. `src/rl/{action,observation,env,transition}.rs`: external actions, fresh resets, raw facts and observations. `python -m altd_rl.trace --config configs/rl/segment_ppo.yaml --episodes 1 --out runs/scripted` records a reproducible scripted controller.
2. `rules.rs`, `reward.rs`, the ring sidecar, `tests/rl.rs`: signed progress, ordered gates, corridor sweeps, clean-lap outcomes and return-identity tests.
3. Rayon car-major CPU collection, byte-array PyO3 bridge, selective resets, Python facade and two benchmark commands.
4. Shared squashed-Gaussian actor/value networks, complete rollouts, GAE, clipped PPO, KL stopping, atomic checkpoints, actor artifacts and deterministic evaluation. Actual segment and full-lap experiments are recorded.
5. Compact GPU actions/pose facts, independent external-action kernels, resident hidden physics, selective resets, exact CPU/GPU differential test and measured environment throughput. Framework inference holds one actor for the batch; the old 16-unit GPU network kernel is never used for PPO.
6. The pinned **official** FlashSAC agent, shared environment/evaluator, transition and wall-budget harnesses, three training seeds, recorded successes and failures. This is a documented racing adaptation, not a reproduction of the paper's robot benchmarks.

## Environment and timing contracts

`RlVecEnv::step` accepts `[N,5]` actions in **Steering, Acceleration, Handbrake, Brake, Boost** order. Bounds are `[-1,-1,0,0,0]` to `[1,1,1,1,1]`. Invalid counts, out-of-range values, NaN and infinity fail before mutation. The adapter does not permit negative brake. Negative acceleration means reverse drive. GPU packing explicitly changes order to acceleration/steering/brake/handbrake/boost.

At 60 Hz, each control is held for up to four unchanged physics ticks. Rules run after every tick. A terminal at tick 1, 2 or 3 shortens the transition; actual duration is the executed ticks divided by 60. Deadline ticks are positive integers. A valid finish at the exact deadline succeeds; numerical/rule/contact failure takes priority over a simultaneous finish. All observed events are retained. Buffer boundaries are ordinary continuing transitions, not truncations or failures. The initial environment emits no artificial truncations.

The returned observations are after the action and **before** reset. Every completed car must be explicitly reset before another whole-vector step. No passive post-terminal transitions are emitted. Cars share immutable geometry and cannot sense or collide with each other. Reset constructs a fresh `Car` and fresh scratch/tracker/control state, rather than relying on deferred legacy resets.

Python calls accept whole batches. Observations cross the binding as contiguous little-endian float32 bytes. Actions cross as contiguous float64 bytes and are mapped to named Rust controls. Metadata/rewards/flags use one batch JSON envelope. The bridge is synchronous and holds the GIL; Rust physics uses Rayon internally. Serialization and per-transition diagnostics remain optimization opportunities.

`reset(ids, seeds=None, spawn_specs=None)` accepts matching seed lists and optional `[x,y,rotation]` spawn rows. Overrides are checked before resetting any row. Spawns must remain on the first route station cross-section with a legal wheel footprint. Ordinary resets use documented SplitMix64 composition of `(run_seed, env_id, episode_id)`; there is no mutable global environment RNG. No unrequested randomized vehicle dynamics are added.

## Authoritative geometry

`configs/rl/tracks/rally_ring.rules.json` explicitly defines a 64-cell polygon ribbon around a radius-1000 centerline, with half-width 130 game units. It uses the repository's Rally vehicle template and native wall solver. Physical wall quads and road cells derive from the same analytic cross-sections. This is a new synthetic track, not a captured campaign track. The source generator is `tools/rl_track.py`; `--check` verifies checked-in assets.

Each convex CCW cell has centerline endpoints and a directed exit gate. Gates must be complete shared cell edges, with the exit station at their midpoint. Unsupported/nonfinite/disconnected geometry fails validation. Missing road metadata is never interpreted as default asphalt legality.

The footprint rule checks the car center and all wheel contact locations, including their straight swept chords between successive physics ticks. Union-of-convex-cell interval clipping covers the entire chord, including gaps; endpoint-only corridor tests are insufficient. The cell cursor can move backward and unwrap the seam; the consumed-gate cursor only advances in order. Progress may decrease and is clamped only after signed-distance calculation. Returning from behind the spawn cannot claim a lap. Previously consumed gates yield no second credit. Only reachable adjacent cells are projected, preventing hairpin jumps. A movement spanning unsupported nonadjacent cells is conservatively invalid. The v1 wheel-chord rule and tick-quantized event times are explicit discrete simulation rules; it does not claim continuous rotating-body sweep interpolation.

The GPU scene additionally contains the explicit BSP polygons and Curve2D required by the legacy GPU sensor implementation. Its sensor semantics/provenance differ from the simpler CPU scene, so it has a separate configuration. GPU capability checks reject unsupported native broadphase or missing Curve2D path-sensor data.

## Reward and observations

Both profiles require gamma=1. With progress `p`, actual tick duration `dt`, deadline `H`, and post-transition elapsed time `t`, components are:

- progress: `p_next-p_before`;
- time: `-dt/H`;
- true failure: `-((H-t)/H+1)`;
- strict profile only, on failure: additional `-p_next`.

Complete returns are `1-T/H` for a legal finish; `p_fail-2` for dense failure; exactly `-2` for strict failure. No reward clipping, raw-speed bonus, smoothing cost, idle elimination or discount change is hidden in this contract. Every transition records components and separately labeled cumulative return.

`NativeEnv.schema()` is machine-readable: ordered sensor descriptors, names, dtype, extra scales, clipping and invalid-data behavior. K native sensor outputs are followed by 12 values: five previous applied controls, remaining-time fraction, validated progress, next gate forward/side displacement divided by 1000, signed lateral offset divided by local half-width, remaining distance divided by 1000, and task kind (full lap 1 / segment 0). Supplied layouts have K=20 and D=32. Sensor definitions are unchanged; extra values are clipped to [-10,10], with progress/time already bounded to [0,1]. Initial controls are zero.

No running observation normalization is used: the immutable feature scales plus identity policy transform are identical during collection, updates, evaluation and actor inference. Scenario deadline is fixed. Nonfinite state/observations terminate with finite zero storage placeholders and no bootstrap; raw IEEE state bits preserve diagnostics in JSON. Schema, track, scene, vehicle, reward, native-binary and action hashes accompany artifacts/traces.

## Learning and artifacts

The policy uses tanh hidden layers, a linear latent mean, learned state-independent log standard deviation, then `mid+scale*tanh(u)`. Rollouts retain latent `u` and stable Jacobian-adjusted log probabilities. Applied action is not treated as an unsquashed Gaussian sample. Initial unsigned-control mean biases are -3 (near zero braking/handbrake/boost); acceleration latent bias is .55. Entropy is estimated from the transformed distribution.

GAE bootstraps continuing states and truncations from the true final observation. True terminals have zero next value. The recurrence stops at either terminal or truncation, preventing leakage across resets. Advantages are normalized once across the complete rollout; PPO shuffles fresh minibatches each epoch, clips policy ratios, trains value MSE, clips gradient norm and stops on excess approximate KL. Stored policy observations and transforms never change during that update.

`checkpoint.pt` is `altd_ppo_checkpoint_v1`: actor/value/log-std, optimizer moments, RNG states, counters, fixed transforms, config, schemas/provenance and dependency versions. Save is atomic after a completed update. Resume restores learning state, **resets environments and starts a new rollout**; it is not continuation of hidden contact state or a partial buffer. Provenance mismatches in core scene/schema/rules/reward/math settings are rejected.

`actor.pt` is `altd_native_actor_v1`; it contains the actor and feature/action transforms without critic weights. `best.pt` is selected at configured validation intervals: first meet the reliability threshold, then minimize median legal time. Below threshold, higher completion rate ranks first. The validation selection score is checkpointed. Validation defaults use the CPU evaluator and fixed starts. Final test scenarios should be reserved separately.

Legacy export is rejected by the native actor: extra observations and affine unsigned controls cannot be represented in the old JSON. `baseline.py` instead reads an existing GA/ARS JSON and executes the real native legacy network on its original named sensors. It then applies the disclosed RL action bounds/control cadence and the new validator. That is not the old fitness contract.

## Traces and inspection

Training always writes `episodes.jsonl`, `updates.jsonl`, resolved config and provenance. `logging.trace_mode=sampled_chunks` enables compressed NumPy NPZ chunks plus JSON sidecars. Optional `trace_env_ids` and `trace_chunk_size` control sampling and flush size (default environment 0, 256 rows). Training uses RAM; there is no per-physics-tick disk write. Flushes happen synchronously outside physics at chunk boundaries.

Chunks retain raw/policy observations, latents, requested/applied actions, old log probabilities, old and next values, episode-start/valid-transition masks, policy version, rewards and final next observations. Sidecars contain run provenance, per-environment episode/decision identity, tick times, signed progress facts, gate/contact outcomes, raw state bits and episode totals. `altd_rl.inspect` verifies tensor finiteness, chunk shapes and reward-component sums. GA/ARS traces lack behavior likelihoods and are not accepted as PPO rollout buffers.

## GPU bridge and ownership

Build using the repository's `gpu/build-cuda.sh` / `.ps1` or HIP scripts, set `ALTD_GPU_LIB` to the built library, then select `configs/rl/rally_gpu.yaml`. Only one native GPU session may hold the shared device claim per process, including legacy sessions. CPU inference works with CPU-only PyTorch; GPU framework inference additionally requires a compatible CUDA PyTorch installation and `--device cuda`.

The RL ABI adds three symbols: `altd_gpu_rl_controls`, `altd_gpu_rl_tick`, `altd_gpu_rl_reset`. Controls upload only N×5 float64 values. Each physics tick uploads an N-byte live mask and downloads N×104-byte pose/velocity/basis/wheel-angle/contact/error/tick reports. Hidden contacts and full car/agent state stay resident. Completed cars are skipped by the independent kernel; legacy inference, stats and its 24-tick graph do not execute. Selective reset copies a freshly constructed complete car and zero agent only for requested rows. Sensor reads use `ReadMask::ALL`; every returned observation row is fresh.

Buffers belong to the native session. Calls use ordered default-stream kernels and synchronous transfers; no tensor view outlives an allocation. Strides are five float64 controls/car, 104-byte repr(C) report/car, sensor-count float64 values/car. The caller owns downloaded arrays. There is no DLPack/zero-copy claim. The actor is held once by PyTorch and evaluated on the whole batch, avoiding both duplicated per-car weights and old hidden-layer limits. Further resident rollout/host-rule-kernel work is deferred behind measured throughput; the current GPU bridge is slower than parallel CPU at large N in the recorded native benchmark.

The Windows CUDA 12.8 compiler crashed on a standalone trivial kernel on this host. GPU validation therefore used WSL Ubuntu 26.04, a project-local CUDA 13.2 redistribution and an RTX 5070 Ti. No machine-wide compiler or driver was replaced. Standard supported build scripts remain the installation path. Results identify the actual platform.

To reproduce the local Linux/WSL compiler setup without installing a driver:

```sh
python3 tools/rl_cuda_redist.py --out target/cuda-13.2
mkdir -p target/gpu
target/cuda-13.2/bin/nvcc -x cu -O3 -std=c++20 -fmad=false -allow-unsupported-compiler -arch=sm_120 -shared -Xcompiler -fPIC -Ltarget/cuda-13.2/lib -o target/gpu/libaltd_gpu.so gpu/sim/altd_gpu.hip
export ALTD_GPU_LIB="$PWD/target/gpu/libaltd_gpu.so"
cargo test --test rl gpu_external_actions_reset_and_terminal_parity -- --ignored
```

The helper verifies archive hashes against NVIDIA's pinned 13.2.0 manifest, extracts only compiler/runtime components into the selected local directory, and keeps the component licenses. The unsupported-compiler flag is disclosed because Ubuntu 26.04 ships gcc 15; exact differential checks are required before performance claims.

## Research comparisons and baselines

Install `pip install '.[research]'` when using the optional comparison. `compare.py` clones only if needed and insists on an unmodified official FlashSAC checkout at `87edc9061150ae9e962dd84e6544e27a1554b3ab`. `configs/rl/flashsac.json` records the pin and resolved starting configuration. The adaptation preserves the official categorical double critic, unit-normalized network blocks, target updates, temperature, replay and persistent action noise. It uses gamma=1 for the native finite-horizon objective, an affine mapping from research actions to RL bounds, n-step=1, and CPU eager execution; it is not a claim to reproduce published throughput. The upstream recommended Python/Torch environment differs from this tested CPU adaptation and is recorded in the report.

```sh
python -m altd_rl.compare --config configs/rl/segment_ppo.yaml --steps 256 --seed 1729 --out runs/flashsac
python -m altd_rl.experiments --updates 4 --flashsac --seeds 1729 1730 1731 --out runs/equal_transitions
python -m altd_rl.experiments --updates 1000 --flashsac --wall-seconds 2 --out runs/equal_wall
```

The harness uses `--wall-seconds`; the learner's direct flag is `--seconds`. Set a large update ceiling so the wall budget controls termination. The measured budget excludes setup and final evaluation; individual updates may overrun the deadline. Reports include actual counts. `UTD=gradient_updates/(vector_steps*N)`; sampled replay items per transition additionally multiply by minibatch size. A vector step is never labeled UTD=1.

Legacy baseline commands use actual repository CLI options:

```sh
cargo run --release -- --math-profile proton train-scratch --scene configs/rl/tracks/rally_ring.scene.json --spawn-trace configs/rl/tracks/rally_ring.spawn.json --settings configs/rl/legacy_ars.json --shape 20,16,5 --population 32 --generations 3 --ticks 600 --seed 1729 --out-dir runs/legacy_ars
# Change settings to configs/rl/legacy_ga.json for GA.
python -m altd_rl.baseline --network PATH_TO_SAVED_NETWORK.json --out runs/baseline_eval.json
```

Three-generation baseline smokes produced no saved completed-lap candidate; that is recorded, not replaced with a fabricated time. Their old generation budgets and sensor/action schedules differ from PPO. A fair superiority study needs larger matched sample and wall budgets, independent scenario suites and revalidated legacy drivers.

## Checks and profiling

```sh
cargo test --test rl
python -m pytest tests/rl -q
cargo test
cargo test --release
cargo check --no-default-features
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
cargo test --test rl gpu_external_actions_reset_and_terminal_parity -- --ignored
python -m altd_rl.benchmark --steps 32
cargo run --release --example rl_benchmark -- cpu
cargo run --release --example rl_benchmark -- gpu
```

The ignored GPU test requires a working built library/device and compares positions, velocities, all observations, rewards, flags, selective resets and independent terminal lengths exactly. Timing benchmarks have different scopes: Python measures inference, combined environment/bridge, and reset separately; the Rust example excludes policy/Python and includes rules/reset. Neither is a lap-time measurement. Broad native regression results are retained in the local run logs; no capture expectations or physics formulas were changed.

References: [official PyO3 0.27.2 guide](https://pyo3.rs/v0.27.2/), [pinned FlashSAC source](https://github.com/Holiday-Robot/FlashSAC/tree/87edc9061150ae9e962dd84e6544e27a1554b3ab), [NVIDIA CUDA 13.2 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.0.json). The plan's paper list motivates experiments; the measurements here are the simulator's own results.


## Integrated dashboard, epoch examples and Windows GPU

See [dashboard setup and controls](../dashboard/README.md). The GUI lives in this repository and exposes configuration sliders, all CLI tools, progress graphs and deterministic epoch/generation replays. `training.preview_every` defaults to 0 for CLI runs; a positive interval records each completed PPO epoch into `examples/uNNNNNN_eNN.json`. Examples now run from spawn to completion, failure or deadline; the former `preview_ticks` limit is ignored. `train-scratch --preview-every 1` records a full cloned car-0 replay for each generation, using the original schedule. Examples do not enter rollouts or mutate the population/RNG. They cost extra runtime and do not establish driving quality.

Windows NVIDIA setup is reproducible with `tools/setup_rl_gpu.ps1 -AllowUnsupportedCompiler`. It builds a repository-local CUDA 13.2 simulation DLL and installs CUDA-enabled PyTorch in `.venv-gpu`. Validated locally on an RTX 5070 Ti: exact RL CPU/GPU parity; three GPU simulation/CUDA learner updates (49,152 transitions); two further updates with all four epoch examples and CPU-reference validation enabled. Update metrics record `learner_device` and `environment_backend`. The earlier Linux-only redistribution helper now also supports Windows archives. Compiler artifacts and environments remain ignored by Git.

PPO ring tracks come from `tools/rl_track.py` (64 ordered cells, radius 1000, half-width 130). Native random training reuses the seeded game-tile generator and its existing JSON settings; these are separate scenario pipelines. Full-lap PPO requires every ring gate; native legacy lap metrics use the original game rules.


## Random game-track PPO curriculum and editable inputs

Use `configs/rl/random_ppo.yaml` or `configs/rl/random_gpu.yaml`. `scenario.track_mode: random` and `training.track_every: 1` request a new game track at each rollout/update boundary. `learner.epochs` still denotes optimizer passes over the same collected batch. Actor weights and optimizer state persist across tracks; in-flight trajectories end at the explicit track-change boundary, with GAE already bootstrapped from their final observations. Resume reproduces the next track index and verifies the generator template/settings contract.

The Python binding calls the existing `training_scene_at` generator. Physics, wall and surface geometry come from its original TileSet exports. The RL rule adapter uses the baked curve to construct a conservative 64-unit half-width centerline corridor with approximately 80-unit stations. Invalid/nonconvex corridor candidates are rejected deterministically, up to 32 attempts per track index. Exported `.track.json` files record the exact chosen native generation and rejection reasons; `.scene.json` and `.rules.json` retain the complete geometry and gates. No fixed-ring substitution occurs. Checkpoint validation uses fixed generated track 0 across updates; validation on one track is not a generalization result.

The dashboard Configuration tab exposes rollout and curriculum settings, hidden layer count/widths and sensor selection/order/parameters. Ordered sensor layout copies are used for new PPO and native GA/ARS runs; native input/output sizes are derived automatically. PPO appends its 12 required history/task inputs. Checkpoint resume keeps its saved input/architecture contract. A CUDA RNG restore correction converts saved RNG tensors back to CPU byte tensors even when loading weights onto CUDA.

Full replays include spawn and the final terminal position. Update metrics separate track setup, preview and learner time; measured transition throughput includes track setup and full-preview overhead. Starting a new dashboard job invalidates pending responses and clears all data in the progress view.
