# Verification

Validated in the integrated repository on Windows, Python 3.14, installed Chrome and an NVIDIA GeForce RTX 5070 Ti.

- 16 Python checks passed: eight dashboard API/process tests and eight RL learning tests. Covers real subprocesses, cancellation and checkpoint preservation, nested child cancellation, path/host/origin/CSRF controls, example selection, config copies, PPO checkpointing and preview invariance for policy/RNG/environment.
- 125 native library tests passed (three hardware-specific tests ignored); eight RL integration tests passed (GPU test separately invoked).
- Windows CUDA 13.2 simulation build, CUDA PyTorch 2.13 backward pass and exact CPU/GPU external-action/reset/terminal parity passed using the repository-local setup helper.
- Three full GPU/CUDA learner updates processed 49,152 transitions. Two further updates passed with four epoch previews and CPU-reference validation enabled.
- Base browser checks passed: real PPO training, checkpoint evaluation, trace recording/playback, native help, mobile layout and no JavaScript runtime errors.
- GPU feature browser checks passed: slider edits and config copies, CUDA learner plus GPU simulation via the dashboard, epoch selection/playback, native GPU generation examples on generated tracks, random-track editing, mobile layout and no JavaScript runtime errors.
- Rust formatting, JavaScript syntax, generated RL ring asset checks and the root dashboard launcher passed.

Reproduction commands are in README.md. Local logs are under `target/dashboard-*`; generated training runs are in `runs/`; screenshots and job metadata are in `dashboard/.dashboard/`. These outputs, CUDA components, virtual environments and binaries are ignored by Git. GPU tests verify execution/parity; short training examples do not establish general driving quality.


## Game-track curriculum and configuration follow-up

- 21 checks passed with CUDA PyTorch: ten dashboard API/process tests and eleven learning checks, including deterministic generated tracks, different successive geometry, custom sensor input dimensions and CUDA checkpoint RNG restore.
- Native library tests: 125 passed, three ignored. RL integration tests: eight passed, one GPU-only test separately covered by local CUDA execution.
- Exact generated-track CPU/CUDA comparison passed for observations, raw states, rewards, resets and terminals over 20 action steps.
- GPU PPO completed track changes with four full terminal epoch replays; checkpoint resume advanced to the next generated track. Fixed generated track 0 validation and per-track trace output passed.
- Browser curriculum check passed: visible training sliders, custom sensor/shape controls used by PPO and native ARS, changed game tracks, complete spawn-to-terminal replays, CLI helper integration, mobile layout and clearing progress while a deliberately late previous-run replay response arrives.
- Base browser and GPU feature workflows also passed after this update. New output-directory isolation and summary offsets preserve historical runs while preventing their data appearing as fresh-job metrics.

The separate verification server uses port 8766 and isolated job metadata. The user's existing server and active training job on port 8765 are left running.
