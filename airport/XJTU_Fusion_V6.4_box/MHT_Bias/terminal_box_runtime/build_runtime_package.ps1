$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $root
$packageDir = Join-Path $root "package"

if (Test-Path $packageDir) {
    Remove-Item -Recurse -Force $packageDir
}
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

$copyDirs = @(
    "Src\\core",
    "MHT",
    "common",
    "Classify",
    "Sensor_Config",
    "CV\\code_image\\ultralytics",
    "CV\\train10\\weights",
    "Src\\tools\\network"
)

foreach ($rel in $copyDirs) {
    $src = Join-Path $projectRoot $rel
    $dst = Join-Path $packageDir $rel
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Copy-Item -Recurse -Force $src $dst
}

$copyFiles = @(
    "Src\\terminal_box_main.py",
    "Src\\main2.py",
    "CV\\code_image\\rtsp_detect_show.py",
    "terminal_box_runtime\\run_terminal_box.sh",
    "terminal_box_runtime\\run_cv_detect.sh",
    "terminal_box_runtime\\README.md"
)

foreach ($rel in $copyFiles) {
    $src = Join-Path $projectRoot $rel
    if ($rel -like "terminal_box_runtime\\*") {
        $dst = Join-Path $packageDir ([System.IO.Path]::GetFileName($rel))
    } else {
        $dst = Join-Path $packageDir $rel
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    }
    Copy-Item -Force $src $dst
}

Write-Host "[runtime] package created at $packageDir"
