param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [bool]$SelfContained = $true
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$launcherProject = Join-Path $projectRoot "launcher\derpy-turtle-kokoro-trainer.csproj"
$outputDir = Join-Path $projectRoot "dist\launcher"
$exeOut = Join-Path $projectRoot "derpy-turtle-kokoro-trainer.exe"

if (-not (Test-Path $launcherProject)) {
    throw "Launcher project not found: $launcherProject"
}

$selfContainedValue = if ($SelfContained) { "true" } else { "false" }

$publishArgs = @(
    "publish"
    $launcherProject
    "-c"
    $Configuration
    "-r"
    "win-x64"
    "--self-contained"
    $selfContainedValue
    "-p:PublishSingleFile=true"
    "-p:IncludeNativeLibrariesForSelfExtract=true"
    "-p:DebugType=None"
    "-p:DebugSymbols=false"
    "-o"
    $outputDir
)

Write-Host "Restoring launcher for win-x64..."
& dotnet restore $launcherProject -r win-x64
if ($LASTEXITCODE -ne 0) {
    throw "dotnet restore failed with exit code $LASTEXITCODE"
}

Write-Host "Publishing launcher..."
& dotnet @publishArgs
if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE"
}

$publishedExe = Join-Path $outputDir "derpy-turtle-kokoro-trainer.exe"
if (-not (Test-Path $publishedExe)) {
    throw "Expected output executable not found: $publishedExe"
}

Copy-Item -Path $publishedExe -Destination $exeOut -Force
Write-Host "Launcher ready: $exeOut"
Write-Host "Run it to bootstrap .venv, install deps, and open the GUI."

