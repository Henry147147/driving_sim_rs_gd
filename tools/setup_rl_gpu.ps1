# Repository-local NVIDIA setup. Requires Python, Rust 1.95 and VS C++ Build Tools.
param([switch]$SkipInstall,[switch]$AllowUnsupportedCompiler,[string]$Arch="native")
$ErrorActionPreference="Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path .venv-gpu/Scripts/python.exe)) {
    python -m venv --system-site-packages .venv-gpu
    if ($LASTEXITCODE -ne 0) { throw "Python venv creation failed" }
}
$python=(Resolve-Path .venv-gpu/Scripts/python.exe).Path
if (-not $SkipInstall) {
    $env:RUSTUP_TOOLCHAIN="1.95"
    & $python -m pip install -e ".[dashboard]"
    if ($LASTEXITCODE -ne 0) { throw "RL/dashboard installation failed" }
    & $python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0 --force-reinstall --no-deps
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
}
$cuda=Join-Path (Get-Location) "target/cuda-13.2-win"
if (-not (Test-Path "$cuda/bin/nvcc.exe")) {
    & $python tools/rl_cuda_redist.py --out $cuda
    if ($LASTEXITCODE -ne 0) { throw "CUDA compiler download failed" }
}
# nvcc rejects the doubled separators present in some inherited Windows TEMP paths.
$oldTemp=$env:TEMP; $oldTmp=$env:TMP
$taskTemp=Join-Path (Get-Location) "target/cuda-tmp"
New-Item -ItemType Directory -Force $taskTemp | Out-Null
try {
    $env:TEMP=$taskTemp.Replace('\','/'); $env:TMP=$env:TEMP
    & ./gpu/build-cuda.ps1 -Nvcc "$cuda/bin/nvcc.exe" -Arch $Arch -AllowUnsupportedCompiler:$AllowUnsupportedCompiler
    if ($LASTEXITCODE -ne 0) { throw "CUDA library build failed" }
} finally { $env:TEMP=$oldTemp; $env:TMP=$oldTmp }
$env:ALTD_GPU_LIB=(Resolve-Path target/gpu/altd_gpu.dll).Path
$env:PYTHONPATH=(Join-Path (Get-Location) "python")
& $python -c "import torch; assert torch.cuda.is_available(); x=torch.ones(4,device='cuda',requires_grad=True); x.sum().backward(); print(torch.__version__,torch.cuda.get_device_name(0),x.grad)"
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch verification failed" }
cargo +1.95 test --test rl gpu_external_actions_reset_and_terminal_parity -- --ignored
if ($LASTEXITCODE -ne 0) { throw "GPU simulation parity check failed" }
Write-Host "GPU setup verified. Launch StartDashboard.cmd, choose rally_gpu.yaml and CUDA."
