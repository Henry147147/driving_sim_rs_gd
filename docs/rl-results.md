# RL implementation measurements

Measured on 2026-10-24. These are reproducible smoke experiments, not an optimizer ranking or a generalization study. Raw summaries are in [rl-results.json](rl-results.json); commands and contracts are in [rl.md](rl.md). All run artifacts remain under `runs/` (Git-ignored).

Hardware: Ryzen 9 9950X (16 cores / 32 threads), RTX 5070 Ti (16 GiB), driver 616.92. Python training used Windows 11, CPython 3.14.3, Torch 2.13.0+cpu and NumPy 2.5.2. GPU differential tests/native benchmarks used WSL Ubuntu 26.04, CUDA 13.2.51, Rust 1.95.0 and gcc 15.2.0.

## Full-lap PPO

Each run collected 100 updates of 64 environments × 256 decisions. Select a checkpoint every 10 updates: completion rate ≥90%, then lowest median valid time. The final evaluation repeated that selected deterministic driver eight times at the fixed spawn.

| Training seed | Transitions | Collection + learner seconds | Valid attempts | Median valid lap seconds |
|---|---:|---:|---:|---:|
| 1729 | 1,638,400 | 60.78 | 8/8 | 13.6333 |
| 1730 | 1,638,400 | 62.83 | 8/8 | 13.3833 |
| 1731 | 1,638,400 | 65.76 | 8/8 | 12.5000 |

The freshly initialized deterministic policy completed 0/4 full laps for each of the three seeds under the same validator; all selected trained drivers completed 8/8.

The track is an explicit 64-cell synthetic polygon ring using the repository Rally vehicle and native physics, not a game campaign track. The reported 1/60-second resolution is tick quantization. Repeating the same deterministic start verifies reproducibility; it does not provide independent uncertainty samples. Variation across three training seeds is the available optimization evidence. An earlier pilot completed a lap in 11.65 seconds; it used an earlier metadata/build snapshot and is not substituted for these reproducible selected-checkpoint results.

## Segment: matched transition budget

| Algorithm | Seed | Transitions | Valid attempts | Median valid segment seconds |
|---|---:|---:|---:|---:|
| PPO | 1729 | 8,192 | 4/4 | 1.3667 |
| official FlashSAC | 1729 | 8,192 | 4/4 | 1.0667 |
| PPO | 1730 | 8,192 | 4/4 | 1.3333 |
| official FlashSAC | 1730 | 8,192 | 0/4 | unavailable |
| PPO | 1731 | 8,192 | 4/4 | 1.3500 |
| official FlashSAC | 1731 | 8,192 | 0/4 | unavailable |

PPO had four updates per run. The official FlashSAC checkout was pinned to `87edc9061150ae9e962dd84e6544e27a1554b3ab`; with 32 cars and one gradient step per vector step after warmup, 241 updates / 8,192 new transitions gives gradient-step UTD 0.02942 and 7.53125 replay items sampled per new transition (minibatch 256). This is not UTD=1. Raw reports retain every failed run and failure reason. The initial PPO deterministic driver already completed this short segment: learning evidence here is the before/after legal time, rather than discovery of a previously unreachable target.

## Segment: two-second compute cap

| Algorithm | Seed | Actual transitions | Collection + updates seconds | Valid attempts | Median seconds |
|---|---:|---:|---:|---:|---:|
| PPO | 1729 | 32,768 | 1.899 | 4/4 | 1.1333 |
| official FlashSAC | 1729 | 1,728 | 2.051 | 4/4 | 1.8000 |
| PPO | 1730 | 32,768 | 1.940 | 4/4 | 1.1500 |
| official FlashSAC | 1730 | 1,728 | 2.014 | 0/4 | unavailable |
| PPO | 1731 | 32,768 | 1.966 | 4/4 | 1.1667 |
| official FlashSAC | 1731 | 1,728 | 2.043 | 0/4 | unavailable |

The cap is checked between completed updates/vector steps, so one indivisible operation may overrun it. Setup, checkpoint serialization and final evaluation have separate overheads; the JSON retains PPO process wall time too. These very short budgets and differing initial action biases are insufficient to name a winner. Full-lap FlashSAC tuning and held-out track evaluation remain future experiments supported by the implementation.

## Throughput and the GPU path

Exact CPU/GPU differential checks passed for external actions, all observations, positions/velocities, rewards, terminal flags and selective resets. The test ran 100 four-tick decisions across three independently controlled cars, resetting completed rows and selectively resetting a continuing row. The checked GPU scene has explicit BSP/Curve2D metadata, required by the GPU sensor implementation. The original CPU scene and GPU sensor scene have distinct hashes.

| N | Native CPU transitions/s | Native GPU transitions/s |
|---:|---:|---:|
| 64 | 50,451 | 116,226 |
| 256 | 314,693 | 183,849 |
| 1,024 | 762,486 | 252,471 |
| 4,096 | 995,408 | 398,022 |

Scope: one 64-step scripted native run per size, includes rule checks and selective resets, excludes initialization, policy inference and the Python bridge. These short measurements have startup/scheduling variance. GPU hidden physics stays resident; compact facts cross the boundary every physics tick for authoritative host rules. Larger populations favor the parallel CPU in this implementation. No GPU speedup is claimed. Host rule checking is parallelized; moving rules/rollouts onto the device or zero-copy interop would require a further independently validated implementation.

The separate Windows Python benchmark measured the complete byte/JSON bridge plus inference/reset at approximately 64 cars: 46,637 transitions/s, 256 cars: 66,114 transitions/s, 1,024 cars: 74,673 transitions/s, 4,096 cars: 62,112 transitions/s. Its combined environment/bridge timings must not be equated with the native-only benchmark.

## Other checks

- Scripted segment trace completed its target; selected PPO full-lap trace recorded three valid laps, 615 transition rows, and zero reward-component summation error.
- Stop/resume restored the learning update count and optimizer/RNG state and continued a fresh rollout; an actor-only artifact also evaluated successfully.
- Native reward tests establish dense failure `p-2`, strict failure `-2`, valid return `1-T/H`, reversal cancellation and shorter terminal holds.
- GA and customized ARS each ran three generations of 32 cars on the ring through the unchanged CLI. Neither smoke produced a completed-lap candidate. They use the old fitness/schedule, so this is compatibility evidence rather than a fair optimizer comparison.
- Debug/release native regressions, focused RL checks, Python learning/artifact checks, no-default-feature build, WASM compile and Clippy checks are recorded in the final validation evidence. Existing physics and capture expectations were not modified.

## Limits

- One analytic polygon ring and one fixed deterministic spawn; repeated evaluation episodes are not independent track-generalization cases.
- Native benchmarks are one short 64-vector-step run per size, excluding initialization, framework inference and Python; no repeated-run uncertainty.
- Two-second compute caps can be exceeded by one indivisible update; compare actual measured counts/time.
- FlashSAC is the pinned official agent with gamma=1, n-step=1, CPU eager mode and affine control mapping; not a robot-paper reproduction.
- Legacy smokes use their existing fitness and schedules; not matched fairness comparisons.
- Windows CUDA12.8 host compiler crashes even for a trivial kernel; validated GPU library is Linux/WSL.
- Resident rollout/zero-copy optimization is not claimed.

Final native suite totals: debug-tests: 259 passed, 0 failed, 8 ignored, release-tests: 259 passed, 0 failed, 8 ignored. Seven Python tests and the explicitly invoked exact GPU parity check passed. The numerical-failure storage unit test was also exercised after its addition.

Lint scope: Rust 1.95 Clippy passed for the library, bins, tests and new benchmark example. The broader all-targets command additionally reports an existing `unnecessary_sort_by` lint at `examples/gpu_check/evolution.rs:227`; that unrelated example was left unchanged.
