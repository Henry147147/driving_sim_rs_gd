# Racing Lab dashboard

This hostable dashboard is part of the simulator repository. It executes this checkout's Rust CLI and Python trainers. No separate frontend repository or Node build is required.

## Install and run

From the repository root, with Python 3.11+ and Rust 1.95 installed:

```powershell
$env:RUSTUP_TOOLCHAIN="1.95"
python -m pip install -e ".[dashboard,test]"
cargo +1.95 build --release --bin altd-sim
.\StartDashboard.cmd
```

After updating native binding sources, rerun the install command before restarting the dashboard.

Open http://127.0.0.1:8765. Linux/macOS: `python dashboard/app.py`. Use `--python /path/to/python` to select the regular job interpreter. Jobs run from the repository root. Config copies, job logs and uploads go into `dashboard/.dashboard`; training artifacts go into `runs`. These directories are ignored by Git.

## GPU on Windows / NVIDIA

Requires an NVIDIA driver and Visual Studio C++ Build Tools. Run:

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_rl_gpu.ps1 -AllowUnsupportedCompiler
```

This installs the RL package and CUDA 13.0 PyTorch 2.13 in `.venv-gpu`, downloads SHA256-verified CUDA 13.2 compiler components into `target`, builds the simulation DLL, verifies a CUDA backward pass and runs the exact CPU/GPU RL parity test. It leaves global CPU PyTorch installed. The compiler override supports newer Visual Studio versions; parity must pass before using the result. `-SkipInstall` reuses the environment, and `-Arch sm_120` targets RTX 50-series hardware explicitly.

In the dashboard choose `configs/rl/random_gpu.yaml`, policy device **CUDA**, then Start. CUDA jobs automatically use `.venv-gpu`; the local simulation library is discovered automatically. Check GPU reports the selected interpreter and performs a real CUDA backward pass. Simulation backend and policy device are independent: CPU simulation with CUDA policy learning is also supported. FlashSAC remains the existing CPU eager research adapter.

## Controls and results

- Open **Configure · sensors & network** (also available from the sidebar). It opens by default and exposes the main training sliders directly. All configuration fields opens the detailed copy editor. The controls include sliders for cars, action repeat, rollout, epochs, minibatch, learning rate, clipping, validation and replay cadence; selects for simulation, math, task, reward and traces. Raw configuration retains access to every field. Saving creates a copy.
- PPO records a deterministic example after each completed epoch by default in the simple form. Uncheck the option to disable it. CLI users set `training.preview_every` (0 disables). Replays always run to a terminal event; the old `preview_ticks` cap is ignored. Resume preserves the checkpoint configuration.
- Network shape controls edit hidden layer widths and layer count. Sensor rows can be enabled, reordered and added, with ray angles/lengths and other sensor parameters. PPO includes 12 required state/history inputs beyond the selected sensors, with five actor outputs and a one-output value head. Native GA/ARS input and output layer sizes are derived from the selected layout and five driving controls. Copies of the sensor layout and native model/network metadata are saved with the launch configuration. Resume retains the saved architecture and observation contract.
- Native train-scratch has population, generation, seed, duration, GPU and replay controls. Edit random-track settings for tile lengths, doubles and surfaces. The complete CLI argument editor supports GA/ARS settings, track suites, resume, stopping rules and every other native option.
- Replay selector shows update/epoch or generation/car. Latest follows new examples; turn it off to examine older ones. Play and the slider replay a full attempt from tick 0 to completion, failure or deadline, including the final state. A short attempt can be a real early failure; the terminal reason and time explain why it ended. PPO examples use CPU reference physics with the current actor; native examples clone car 0 at generation start. Examples do not alter training data or RNG, and are illustrative rather than best-driver evaluations. KL early stopping can produce fewer epochs than configured. Previews add runtime overhead.
- Starting a new run clears all progress graphs, replay, metrics, results and artifacts before launch. Responses from earlier views cannot restore stale data. A fresh run requesting an occupied output directory gets a new directory with a timestamp suffix; old runs are preserved. Resume progress shows metrics and examples written by the new job, while Saved runs retains full history.
- Graphs show returns, losses, KL, population scores and throughput as available. Console shows live output. Saved runs reopen from disk; checkpoints can resume PPO or record a full trace. Native legacy lap metrics retain their original rules.
- All Python RL tools and native subcommands are exposed. CLI option helpers expose additional flags as toggles, numeric sliders, selects and text fields from the command help. They work together with the simple training, sensor and shape controls. The complete JSON argument array remains available for manual overrides, executed without a shell. Stop requests cooperative cancellation; Force stop terminates the process tree and discards work after the last saved checkpoint.

## How tracks are generated

The default PPO configuration is now `configs/rl/random_ppo.yaml` (or `random_gpu.yaml`). It uses the existing game-tile generator and exports the original wall, collision and surface geometry. `scenario.track_mode: random` enables it; `training.track_every: 1` selects a new track before every rollout/update. PPO optimizer epochs reuse the collected on-policy batch. Set `learner.epochs: 1` if you want one optimization epoch per new track. Changing the track inside PPO optimizer epochs would not collect new data. The actor and optimizer continue across tracks; environments reset explicitly between them.

Generated scenes, native track exports and gate rules are saved under `runs/<run>/tracks/`. Selection is deterministic from the seed and track index, including resume. The RL adapter samples the game's baked centerline at approximately 80-unit stations and creates a conservative corridor with half-width 64. These corridor edges are RL training boundaries, not the game's road edges or legacy lap rules. Candidates with nonconvex route geometry are rejected deterministically (at most 32 attempts), with the chosen native generation and rejected reasons recorded in the track export. No ring fallback is used. Full laps require all generated gates. Fixed track 0 is used for validation across updates, so checkpoint scores remain comparable; this does not replace a held-out generalization study.

PPO's optional fixed teaching scenario is an analytic closed ring: `tools/rl_track.py` creates 64 polygon road cells at radius 1000 with half-width 130, ordered gates and separate CPU/GPU scene exports. `--check` verifies the generated assets. A segment targets a subset of gates; a full lap requires all 64. Changing target gates does not generate a new track.

Native random training uses the existing seeded game-tile generator and `assets/random_track_settings.json`. It assembles closed routes subject to even tile lengths, surface choices and doubles. Training's track buffer prepares the seeded sequence; saved generated tracks and run metadata make it reproducible. Multi-track generations remain available in advanced CLI options. Game campaign tracks are not included. The new PPO adapter supplies explicit RL road/gate metadata for generated tracks. This curriculum currently applies to PPO; FlashSAC keeps its pinned comparison scenario.

## Hosting and checks

Default binding is loopback. For another interface pass `--host 0.0.0.0 --allow-host YOUR_HOST --token YOUR_TOKEN`; use HTTPS via a reverse proxy outside a trusted local machine. This dashboard can execute training and load trusted PyTorch checkpoints; only grant access to trusted users. Host/origin checks, a CSRF token and optional bearer authentication protect the API. Closing the browser leaves jobs running; stopping the server stops its managed jobs.

```powershell
$env:PYTHONPATH=(Resolve-Path python).Path
python -m pytest dashboard/tests tests/rl -q
npm.cmd install --prefix dashboard/.dashboard/browser --no-audit --no-fund playwright@1.56.0
cd dashboard
node tests/browser.cjs
node tests/browser-features.cjs
node tests/browser-curriculum.cjs
```

Browser checks use installed Chrome. GPU feature checks require the setup above. No generated runs, wheels, CUDA downloads or screenshots belong in the commit.
