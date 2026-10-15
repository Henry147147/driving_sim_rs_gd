# Build altd_gpu.dll (the GPU simulator parts) for AMD GPUs with the HIP SDK for Windows into $OUT.
# GPU_ARCH lists one or more offload targets separated by commas or spaces.
# -ffp-contract=off: hipcc otherwise fuses a*b+c into FMA, which rounds once.
# HIP_PATH is the HIP SDK root, which its installer sets.
param(
    [string]$Out = $(if ($env:OUT) { $env:OUT } else { Join-Path $PSScriptRoot "..\target\gpu" }),
    [string]$Arch = $(if ($env:GPU_ARCH) { $env:GPU_ARCH } else { "gfx1100" })
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path

# The py launcher first: python3 on Windows is often the Microsoft Store placeholder.
$tables = Join-Path $PSScriptRoot "gen_tables.py"
if (Get-Command py -ErrorAction SilentlyContinue) { py -3 $tables --check } else { python $tables --check }
if ($LASTEXITCODE -ne 0) { throw "gen_tables.py --check failed" }

$targets = @($Arch -split '[,\s]+' | Where-Object { $_ })
if (-not $targets) { throw "GPU_ARCH lists no targets" }
if (-not $env:HIP_PATH) { throw "HIP_PATH is not set; install the HIP SDK for Windows" }
$hipcc = Join-Path $env:HIP_PATH "bin\hipcc.exe"

$src = Join-Path $PSScriptRoot "sim\altd_gpu.hip"
$dll = Join-Path $Out "altd_gpu.dll"
& $hipcc -O3 -ffp-contract=off @($targets | ForEach-Object { "--offload-arch=$_" }) -shared -std=c++20 `
    -fhip-fp32-correctly-rounded-divide-sqrt -o $dll $src -L (Join-Path $env:HIP_PATH "lib") -lamdhip64
if ($LASTEXITCODE -ne 0) { throw "hipcc failed" }
Write-Host "built $dll for $($targets -join ' ')"
