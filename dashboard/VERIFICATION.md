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
