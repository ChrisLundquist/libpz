# Run cargo with the Visual Studio MSVC environment loaded (link.exe, LIB, etc.).
# Use from repo root: .\scripts\cargo-msvc.ps1 build
# Or from libpz-rs: ..\scripts\cargo-msvc.ps1 build
#
# OpenCL: if Khronos SDK is at D:\opencl\OpenCL-SDK-v2025.07.23-Win-x64, opencl-sys will find OpenCL.lib.
# You can override by setting OPENCL_PATH (or OPENCL_ROOT) before running this script.

$openclSdk = "D:\opencl\OpenCL-SDK-v2025.07.23-Win-x64"
if ((Test-Path $openclSdk) -and (-not $env:OPENCL_PATH) -and (-not $env:OPENCL_ROOT)) {
    $env:OPENCL_PATH = $openclSdk
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere not found. Install Build Tools for Visual Studio with C++ workload."
    exit 1
}

$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    Write-Error "Visual Studio (or Build Tools) with C++ tools not found."
    exit 1
}

$devCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $devCmd)) {
    Write-Error "VsDevCmd.bat not found at $devCmd"
    exit 1
}

$libpzRs = Join-Path $PSScriptRoot "..\libpz-rs"
$cargoArgs = $args -join " "
# Set OPENCL_PATH so opencl-sys build script can emit link search (optional; libpz-rs/.cargo/config.toml also adds the path)
$openclEnvBefore = if (Test-Path $openclSdk) { "set OPENCL_PATH=$openclSdk && " } else { "" }
# Build the batch command as one string; pass to cmd without Invoke-Expression so "(x86)" in VS path isn't re-parsed by PowerShell
$run = "${openclEnvBefore}`"$devCmd`" -arch=amd64 && cd /d `"$libpzRs`" && cargo $cargoArgs"
& cmd /c $run
exit $LASTEXITCODE
