# altd-sim

[![CI](https://github.com/Zorro909/driving_sim_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Zorro909/driving_sim_rs/actions/workflows/ci.yml)

`altd-sim` is a Rust vehicle simulator and neural-network trainer for AI Learns To Drive. It runs headless on the CPU, through a runtime-loaded AMD HIP library, or in WebAssembly. It reproduces the game's floating-point arithmetic, contact ordering, sensors and evolution behavior against captured regression fixtures. Exactness claims apply to the tested inputs; [fidelity](docs/fidelity.md) records the limits.

The repository includes vehicle and sensor templates, tile resources, fixed generated tracks and self-contained tests. Campaign tracks are not distributed. Training supports the existing genetic algorithm (GA) and customized Augmented Random Search (ARS), plus a shared-policy PPO learner and an optional pinned FlashSAC research adapter. See [reinforcement learning](#reinforcement-learning) for the Python setup and commands.

## Local training dashboard

The repository includes a hostable web GUI for PPO, FlashSAC, GA/ARS and all CLI tools, with configuration sliders/toggles, live graphs, logs, checkpoints and per-epoch/generation driving examples.

```powershell
$env:RUSTUP_TOOLCHAIN="1.95"
python -m pip install -e ".[dashboard]"
cargo +1.95 build --release --bin altd-sim
.\StartDashboard.cmd
```

Open [Racing Lab](http://127.0.0.1:8765). On Linux/macOS run `python dashboard/app.py`. For NVIDIA GPU setup, track generation details and dashboard controls, see [dashboard instructions](dashboard/README.md). The dashboard runs the simulator and trainers from this repository.

## Getting started

This guide sets up `altd-sim serve`, the local simulator for [Drive Lab](https://drivinglab.jectrum.de). Drive Lab keeps your tracks, runs and downloads in the browser; the server drives the cars on your CPU threads or an AMD GPU. That removes the browser's limits on cars and network size and its eight-thread cap.

### 1. Download

Download the archive for your system from [Releases](https://github.com/Zorro909/driving_sim_rs/releases). Tagged versions are stable; the `nightly` pre-release is rebuilt from every push to `main` and keeps the same file names. `SHA256SUMS` lists the archive checksums.

| System | Archive | Training on |
| --- | --- | --- |
| Linux x86_64 | `altd-sim-<version>-x86_64-unknown-linux-gnu-cpu.tar.gz` | CPU |
| | `altd-sim-<version>-x86_64-unknown-linux-gnu-amd.tar.gz` | CPU, or AMD GPU through HIP |
| | `altd-sim-<version>-x86_64-unknown-linux-gnu-nvidia.tar.gz` | CPU, or NVIDIA GPU through CUDA |
| Windows x86_64 | `altd-sim-<version>-x86_64-pc-windows-msvc-cpu.zip` | CPU |
| | `altd-sim-<version>-x86_64-pc-windows-msvc-amd.zip` | CPU, or AMD GPU through HIP |
| | `altd-sim-<version>-x86_64-pc-windows-msvc-nvidia.zip` | CPU, or NVIDIA GPU through CUDA |
| macOS, Apple silicon | `altd-sim-<version>-aarch64-apple-darwin-cpu.tar.gz` | CPU |

The `amd` and `nvidia` archives add that vendor's GPU library beside the same `altd-sim` binary. Unpack one and open a terminal in the unpacked folder:

```sh
tar -xzf altd-sim-nightly-x86_64-unknown-linux-gnu-cpu.tar.gz
cd altd-sim-nightly-x86_64-unknown-linux-gnu-cpu
./altd-sim --version
```

On Windows, extract the `.zip`, then open PowerShell in the folder and use `.\altd-sim.exe` wherever this guide says `./altd-sim`. macOS binaries are unsigned, so first remove the download quarantine with `xattr -d com.apple.quarantine altd-sim`. Apple silicon uses the portable trigonometry described in [fidelity](docs/fidelity.md).

### 2. Start the server

```sh
./altd-sim serve
```

```text
altd-sim serve: listening on ws://127.0.0.1:47800/v1
  allowed origins: https://drivinglab.jectrum.de
  CPU threads: 16
  HIP: available (AMD Radeon RX 7900 XTX)
```

Keep the terminal open while you train; <kbd>Ctrl</kbd>+<kbd>C</kbd> stops the server. It listens only on `127.0.0.1`, so the browser must run on the same computer.

### 3. Connect Drive Lab

1. In Drive Lab, open **New run**. Under **Compute**, enable **Use a local altd-sim**. The port field defaults to 47800.
2. If the browser asks to allow access to devices on your local network, allow it.
3. The status shows the connection, the CPU thread count and the GPU, if HIP is available. Choose **Native CPU** or **Native GPU (HIP)** and start the run.

Each browser connection gets its own session. Only one session at a time can use the GPU; another one asking for HIP trains on CPU and says why. If the server stops during training, the run stops and keeps its last saved generation. Start the server again and resume the run.

### Options

| Option | Use it to |
| --- | --- |
| `--threads N` | Limit the CPU threads, for example to keep the computer responsive. The default is every logical CPU the process may use. |
| `--math-profile PROFILE` | Reproduce `proton`, `win10-fma3` or `win11-fma3` arithmetic. The default follows the server's operating system; clients can select a profile for each session. |
| `--port N` | Listen on another port when 47800 is taken. Set the same port in Drive Lab. |
| `--allow-origin ORIGIN` | Allow another Drive Lab address, such as a development copy: `--allow-origin http://127.0.0.1:5173`. It replaces the default; repeat the option to allow several. |

```sh
./altd-sim serve --threads 8 --port 47801
```

All builds include all three math profiles. Windows 11 24H2 and newer default to `win11-fma3`; earlier Windows releases use `win10-fma3`; Linux and macOS use `proton`. To train on Linux for a game running on current Windows, for example, pass `--math-profile win11-fma3`. Checkpoints retain their profile. [Math profiles](math/README.md) describes the kernels and captured reference data.

### Training on a GPU

The `amd` and `nvidia` archives contain the GPU library next to `altd-sim`: `libaltd_gpu.so` on Linux, `altd_gpu.dll` on Windows. AMD libraries need a ROCm 7.x runtime that provides `libamdhip64.so.7` on Linux, or an AMD driver or HIP SDK 7.1 that provides `amdhip64_7.dll` on Windows, and one of the [covered GPU architectures](gpu/README.md). NVIDIA libraries need only an NVIDIA driver that supports CUDA 13 (580 or newer) and a consumer GPU from RTX 20 (Turing) to RTX 50 (Blackwell) or newer; datacenter GPUs such as A100 and H100 need a library built for their target, see the [GPU guide](gpu/README.md#nvidia-cuda). `./altd-sim gpu-info` checks that the library loads and prints `"layout": "ok"`; with a GPU present it also runs one kernel and prints the device, and fails if the library has no code for that GPU; the server's `HIP:` line shows the device or why HIP is unavailable.

A HIP session drives its first 12 ticks on both the GPU and the CPU and compares them. If they differ, or the network or track needs something HIP does not support, the session trains on CPU and Drive Lab shows the reason. Native CPU and HIP produce the same results. Browser training can differ from both in the last digits of a few values ([issue #3](https://github.com/Zorro909/driving_sim_rs/issues/3)), so a run moved between the browser and the server can develop differently from then on.

NVIDIA GPUs use the same backend, so HIP in Drive Lab and the server output also covers CUDA. To build a library yourself, see the [GPU guide](gpu/README.md).

### Troubleshooting

- **`cannot listen on 127.0.0.1:47800: Address already in use`**: another server is already running, or another program uses the port. Stop it or choose another `--port`.
- **Drive Lab cannot connect**: check that the server is running and that both use the same port. A Drive Lab address other than `https://drivinglab.jectrum.de` must be allowed with `--allow-origin`, including its scheme and port. If you denied local network access, allow it again in the browser's site settings.
- **`HIP: unavailable`**: the line gives the reason. `./altd-sim gpu-info` reports library problems, and [GPU setup](gpu/README.md) covers ROCm and other GPU architectures.

[Server documentation](docs/server.md) describes the protocol and its security checks for other clients.

### Training without Drive Lab

The CLI can also train on its own. This example trains Rally networks on generated tracks and writes them to `runs/example`:

```sh
./altd-sim train-scratch --scene assets/scenes/rally_template.json --track-mode random \
  --random-track-settings assets/random_track_settings.json \
  --population 256 --generations 3 --ticks 600 --out-dir runs/example
```

Add `--gpu` to train on an AMD GPU. `./altd-sim --help` lists every command, and the [CLI reference](docs/cli.md) describes their inputs and outputs.

## Build from source

Rust 1.94 or newer is the tested native toolchain. Build from the repository root. `--offline` works when dependencies are already cached; omit it on the first build if needed.

```sh
cargo build --release --offline
python3 tools/benchmark.py --population 1024 --ticks 600 --repeats 5 \
  --report target/benchmark.json

target/release/altd-sim --threads 8 train-scratch \
  --scene assets/scenes/rally_template.json --track-mode random \
  --random-track-settings assets/random_track_settings.json \
  --population 256 --generations 3 --ticks 600 --seed 1729 \
  --eliminate-on-wall --idle-eliminate --save-final-candidate \
  --out-dir runs/example
```

The training command starts Xavier networks and generates a reproducible track each generation. Increase the population, generations and tick limit for longer runs. Supply the same scene and track settings when resuming, with `--resume` and a higher `--generations` target.

`.cargo/config.toml` uses `target-cpu=native` for x86 local builds, including the published CPU measurements. For a portable x86_64 binary, override it explicitly:

```sh
RUSTFLAGS='-C target-cpu=x86-64' cargo build --release --offline
```

AVX2 network and geometry paths use runtime detection and retain scalar fallbacks. Exact native trigonometry uses x87 on x86; other targets use the portable implementation described in the fidelity notes.

## Other backends

AMD GPU builds need ROCm and `hipcc`. The default architecture is `gfx1100`, for the RX 7900 XTX.

```sh
gpu/build.sh
# Add --gpu to the train-scratch command above.
```

For other AMD GPUs, set `GPU_ARCH` to one or more architectures when building. Rust builds do not require ROCm. See [GPU setup](gpu/README.md) for validation and profiling.

The default build also includes a local WebSocket server for native CPU or HIP sessions:

```sh
target/release/altd-sim --threads 8 serve
```

[Getting started](#getting-started) explains its options and how to connect Drive Lab; [server documentation](docs/server.md) covers the protocol, security checks and checkpoint resume.

WebAssembly builds need the wasm32 target and the matching `wasm-bindgen` CLI:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version "$(wasm/bindgen-version.sh)" --locked
wasm/build.sh
```

The JavaScript library includes CPU simulation, a threaded CPU package and optional WebGPU backends. See [WASM setup](wasm/README.md) for initialization, browser requirements and tests. Expanded native/WASM comparisons have known last-bit differences; serial and threaded WASM are compared separately.

## CLI and output

`target/release/altd-sim --help` and `target/release/altd-sim COMMAND --help` show arguments. `bench` measures a fixed population; `train` evolves an exported network; `train-scratch` starts Xavier networks and supports checkpoints, generated-track batches and stop conditions; `evaluate` runs a frozen candidate against a hash-validated suite; `serve` exposes sessions over a loopback WebSocket. The `compare-*` commands compare user-supplied captures with the simulator. Scenes and recorded traces are explicit inputs. [CLI reference](docs/cli.md) covers defaults, resume behavior and report formats.

A `train-scratch` run writes:

| File | Contents |
| --- | --- |
| `run.json`, `log.jsonl`, `progress.json` | Effective options, completed generations and current progress |
| `checkpoint.json`, `checkpoint_gNNNNN.bin` | Generation-boundary population and RNG state |
| `best.json`, `best_laps/` | Networks that set a new best lap |
| `candidate.json` | Final fitness leader when `--save-final-candidate` is enabled |
| `tracks/` | Generated track geometry for replay |

## Code and tests

`src/math/` contains runtime-compatible arithmetic and vectors. `src/physics/` implements vehicles and contacts, while `src/track/` owns scene geometry, spatial queries and generated tracks. `src/nn/` provides scalar and AVX2 inference. `src/training/` owns populations, evolution, RNGs, evaluation and embedding sessions. `src/gpu/hip.rs` loads HIP; `src/gpu/simulation.rs` shares device layouts with `src/wasm/`, which supplies JavaScript and WebGPU bindings. The CLI entrypoint is `src/bin/altd-sim/main.rs`, beside its argument, comparison, benchmark and checkpoint modules. Rust imports follow these groups, such as `altd_sim::physics::car::Car` and `altd_sim::training::session::Session`. See [design notes](docs/design.md).

```sh
cargo test --release --offline
cargo test --offline
cargo build --offline --target wasm32-unknown-unknown --no-default-features --features wasm
```

Tests retain self-contained native captures and label generated golden data as current-code regressions. Browser integration checks live in `wasm/test/`; HIP equivalence checks use `gpu_check`. The exhaustive trigonometry scan and hardware-dependent tests are opt-in. See [performance](docs/performance.md), [GPU internals](docs/gpu.md) and [WASM fidelity](docs/wasm.md).

GitHub Actions checks formatting, native and WASM Clippy, native release and debug tests, Windows and macOS Clippy and release tests, the HIP library build and layout, rustdoc, Rust 1.93 compatibility, and WASM packages with CPU browser comparisons. The release workflow builds the AMD and NVIDIA libraries for Linux and Windows, smoke-tests the CPU, AMD and NVIDIA archives, then publishes tags and nightly builds. Its native jobs override `target-cpu=native` with `x86-64` and include the flags in their cache keys, so cached builds can move between runner CPUs. Serial/threaded browser tests retain the documented expanded native differences. HIP equivalence and hardware WebGPU checks run separately because the hosted runners have no GPU.

The project uses the [MIT license](LICENSE). Godot-derived code retains the [Godot notice](GODOT_LICENSE), and numerical modules retain their source notices.

## Reinforcement learning

The optional Python package trains a shared driving policy from many independent cars in the Rust simulator. It includes batched CPU/GPU environments, explicit track-validity rules, per-step rewards, deterministic evaluation, checkpoints and compressed transition traces. These learners run from Python; `altd-sim serve` and the existing Drive Lab workflow continue to use the population-training interface.

| Algorithm | Implementation | How to run |
| --- | --- | --- |
| GA | Existing genetic algorithm; evolves a population of networks. | `altd-sim train-scratch` with GA settings. |
| Customized ARS | Existing Augmented Random Search with mirrored perturbations, elites and adaptive search settings. | `altd-sim train-scratch` with ARS settings. |
| **PPO + GAE** | New shared actor/critic, bounded stochastic actions, clipped PPO updates and generalized advantage estimation. Supports training, stop/resume, actor export and deterministic evaluation. | `python -m altd_rl.train` |
| **FlashSAC** | New adapter to the pinned official research implementation, using the same Rust environment and evaluator. Optional research dependencies are required. | `python -m altd_rl.compare` |

The shipped tasks are a short segment and a full lap on a verified synthetic Rally ring. [Measured results](docs/rl-results.md) include three-seed PPO and FlashSAC experiments, unsuccessful runs and CPU/GPU throughput. They do not establish performance on campaign tracks or superiority over GA/ARS.

### Install the Python learner

Use the source checkout and run all commands from its root. Install Python 3.11+ and Rust 1.95.0 (the tested toolchain); FlashSAC also needs Git. The Python package installs its pinned NumPy, PyTorch and PyYAML dependencies and builds the native Rust binding. Ordinary Rust builds do not require Python.

On Windows, in PowerShell:

```powershell
cd C:\Users\henry\Desktop\training\driving_sim_rs_gd
rustup toolchain install 1.95.0 --profile minimal
$env:RUSTUP_TOOLCHAIN = "1.95.0"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
```

The explicit virtual-environment executable avoids changing PowerShell's script execution policy. On Linux/macOS, create the same environment with `python3 -m venv .venv` and use `.venv/bin/python` instead. Set `RUSTUP_TOOLCHAIN=1.95.0` for the build, or select another supported Rust toolchain. Re-run the package installation after changing native code.

### Train PPO

Start with the short segment on CPU:

```powershell
.\.venv\Scripts\python.exe -m altd_rl.train --config configs/rl/segment_ppo.yaml --seed 1729 --updates 20
```

Then train the full-lap task:

```powershell
.\.venv\Scripts\python.exe -m altd_rl.train --config configs/rl/rally_ppo.yaml --seed 1729 --updates 100
```

The segment configuration uses 32 cars and writes to `runs/segment_ppo`; the full-lap configuration uses 64 cars and writes to `runs/rally_ppo`. Add `--out runs/my_experiment` for a separate run. Each completed update prints learning metrics and saves `checkpoint.pt` and the actor-only `actor.pt`. Validation runs every ten updates and saves its selected checkpoint as `best.pt`; a run shorter than ten updates may not have that file. Episode summaries and learning metrics are in `episodes.jsonl` and `updates.jsonl`.

Press Ctrl+C to stop. Resume from the most recent completed update:

```powershell
.\.venv\Scripts\python.exe -m altd_rl.train --resume runs/rally_ppo/checkpoint.pt --updates 100
```

`--updates 100` means **100 additional updates** when resuming. Learning and optimizer state are restored; environments reset and collection starts a new rollout. If the original run used `--out`, pass that same output directory on resume too. Use `--seconds 60` to stop between updates after a compute-time budget; an update already in progress can overrun it.

### Evaluate and inspect a PPO driver

Evaluate the selected full-lap checkpoint, then record and inspect its transitions:

```powershell
.\.venv\Scripts\python.exe -m altd_rl.evaluate --checkpoint runs/rally_ppo/best.pt --episodes 8
.\.venv\Scripts\python.exe -m altd_rl.trace --checkpoint runs/rally_ppo/best.pt --episodes 3 --out runs/rally_ppo/trace
.\.venv\Scripts\python.exe -m altd_rl.inspect runs/rally_ppo/trace
```

Use `checkpoint.pt` if `best.pt` does not exist yet. Evaluation reports valid completions, legal lap times and failures separately. These commands are headless: evaluation saves JSON, and traces save compressed NPZ arrays with JSON metadata. Repeated deterministic attempts at the shipped fixed start are reproducibility checks, not independent generalization trials.

### Run FlashSAC and compare algorithms

Install the optional research dependencies, then run the pinned official agent on the segment:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[research]"
.\.venv\Scripts\python.exe -m altd_rl.compare --config configs/rl/segment_ppo.yaml --steps 256 --seed 1729 --out runs/flashsac
```

The adapter fetches the official FlashSAC repository if needed and requires its unmodified pinned revision. It writes a research-agent checkpoint, episode summaries and `report.json`, including deterministic evaluation and the actual update-to-data ratios. FlashSAC checkpoints use the upstream format; PPO's `--resume` and `--checkpoint` commands do not load them. The adapter's tested CPU configuration and its differences from the paper's benchmark setup are documented in the [RL guide](docs/rl.md#research-comparisons-and-baselines).

For a three-seed comparison with equal transition budgets:

```powershell
.\.venv\Scripts\python.exe -m altd_rl.experiments --config configs/rl/segment_ppo.yaml --updates 4 --flashsac --seeds 1729 1730 1731 --out runs/equal_transitions
```

For equal compute-time caps, use a high update ceiling and add `--wall-seconds 2`:

```powershell
.\.venv\Scripts\python.exe -m altd_rl.experiments --config configs/rl/segment_ppo.yaml --updates 1000 --flashsac --wall-seconds 2 --seeds 1729 1730 1731 --out runs/equal_wall
```

### CPU and GPU selection

CPU is the default and the recommended starting point. `configs/rl/rally_gpu.yaml` selects the simulator's external-action GPU backend and requires a freshly built GPU library with the RL entry points. GPU parity was validated under WSL on an RTX 5070 Ti; the installed Windows CUDA 12.8 compiler failed even on a standalone kernel. See the [GPU bridge setup](docs/rl.md#gpu-bridge-and-ownership) for the tested WSL build and capability limits.

Policy inference and simulator execution are separate choices: `--device cuda` selects PyTorch inference/learning on CUDA and additionally requires a CUDA-enabled PyTorch installation. It does not switch the simulator backend. The current large-batch benchmarks favor parallel CPU simulation over the GPU bridge, so a GPU is not required to start training.

See the [complete RL guide](docs/rl.md) for schemas, reward formulas, reset semantics, artifact formats, legacy-network evaluation and correctness checks.
