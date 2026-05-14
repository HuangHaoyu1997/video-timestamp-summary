param(
    [switch]$Overwrite,
    [switch]$DryRun,
    [int]$Limit = 0,
    [string]$ModelPath = (Join-Path $env:USERPROFILE ".cache\whisper\large-v3-turbo.pt")
)

$ErrorActionPreference = "Stop"

$pythonArgs = @(
    ".\asr_batch.py",
    "--input", "video",
    "--output", "asr",
    "--model-path", $ModelPath
)

if ($Overwrite) {
    $pythonArgs += "--overwrite"
}

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

if ($Limit -gt 0) {
    $pythonArgs += @("--limit", "$Limit")
}

python @pythonArgs
