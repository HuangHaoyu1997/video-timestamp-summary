param(
  [string]$VideoDir = "video",
  [string]$FrameDir = "frame",
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$root = Get-Location
$videoRoot = Join-Path $root $VideoDir
$frameRoot = Join-Path $root $FrameDir
$manifestPath = Join-Path $root "frame_manifest.csv"
$logPath = Join-Path $root "frame_extraction.log"
$statusPath = Join-Path $root "frame_extraction_status.json"

$videoExts = @(".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".mpeg", ".mpg")

function Write-Status {
  param(
    [string]$Status,
    [int]$Completed,
    [int]$Total,
    [string]$CurrentVideo = "",
    [string]$Message = ""
  )

  [ordered]@{
    status = $Status
    completed = $Completed
    total = $Total
    currentVideo = $CurrentVideo
    message = $Message
    updatedAt = (Get-Date).ToString("s")
  } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Format-Timecode {
  param([double]$Seconds)

  $ts = [TimeSpan]::FromSeconds([Math]::Max(0, $Seconds))
  if ($ts.TotalHours -ge 1) {
    return $ts.ToString("hh\:mm\:ss")
  }
  return $ts.ToString("mm\:ss")
}

function Get-FrameInterval {
  param([double]$DurationSec)

  if ($DurationSec -le 300) {
    return 1
  }
  if ($DurationSec -le 480) {
    return 2
  }
  return 4
}

function Get-VideoDuration {
  param([string]$Path)

  $durationText = & ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 -- $Path 2>$null
  $duration = 0.0
  [void][double]::TryParse(
    ($durationText | Select-Object -First 1),
    [Globalization.NumberStyles]::Float,
    [Globalization.CultureInfo]::InvariantCulture,
    [ref]$duration
  )
  return $duration
}

function Assert-ChildPath {
  param(
    [string]$Parent,
    [string]$Child
  )

  $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
  $childFull = [IO.Path]::GetFullPath($Child)
  $prefix = $parentFull + [IO.Path]::DirectorySeparatorChar
  if (-not $childFull.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to modify path outside frame root: $childFull"
  }
}

if (-not (Test-Path -LiteralPath $videoRoot)) {
  throw "Video directory not found: $videoRoot"
}

New-Item -ItemType Directory -Force -Path $frameRoot | Out-Null
"[$((Get-Date).ToString("s"))] start frame extraction" | Set-Content -LiteralPath $logPath -Encoding UTF8

$videos = @(
  Get-ChildItem -LiteralPath $videoRoot -Recurse -File |
    Where-Object { $videoExts -contains $_.Extension.ToLowerInvariant() } |
    Sort-Object FullName
)

$results = New-Object System.Collections.Generic.List[object]
$completed = 0
Write-Status -Status "running" -Completed 0 -Total $videos.Count -Message "starting"

foreach ($video in $videos) {
  $completed += 1
  Write-Status -Status "running" -Completed ($completed - 1) -Total $videos.Count -CurrentVideo $video.Name

  $durationSec = Get-VideoDuration -Path $video.FullName
  $intervalSec = Get-FrameInterval -DurationSec $durationSec
  $videoName = [IO.Path]::GetFileNameWithoutExtension($video.Name)
  $outputDir = Join-Path $frameRoot $videoName
  $patternName = "frame_%06d_every{0:D3}s.jpg" -f $intervalSec
  $status = "ok"
  $errorMessage = ""

  "[$((Get-Date).ToString("s"))] $completed/$($videos.Count) interval=${intervalSec}s video=$($video.Name)" |
    Add-Content -LiteralPath $logPath -Encoding UTF8

  $existingFrames = @(
    Get-ChildItem -LiteralPath $outputDir -File -Filter ("frame_*_every{0:D3}s.jpg" -f $intervalSec) -ErrorAction SilentlyContinue
  )

  if ((-not $Force) -and $existingFrames.Count -gt 0) {
    "[$((Get-Date).ToString("s"))] skip existing frames video=$($video.Name) frames=$($existingFrames.Count)" |
      Add-Content -LiteralPath $logPath -Encoding UTF8
  } else {
    $tempDir = Join-Path $frameRoot (".tmp_{0}_{1}" -f $videoName, [Guid]::NewGuid().ToString("N"))
    Assert-ChildPath -Parent $frameRoot -Child $tempDir
    Assert-ChildPath -Parent $frameRoot -Child $outputDir
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    $outputPattern = Join-Path $tempDir $patternName

    $ffmpegArgs = @(
      "-hide_banner",
      "-loglevel", "error",
      "-y",
      "-i", $video.FullName,
      "-vf", "fps=1/$intervalSec",
      "-q:v", "3",
      "-start_number", "1",
      $outputPattern
    )

    $ffmpegOutput = & ffmpeg @ffmpegArgs 2>&1
    $tempFrames = @(
      Get-ChildItem -LiteralPath $tempDir -File -Filter ("frame_*_every{0:D3}s.jpg" -f $intervalSec) -ErrorAction SilentlyContinue
    )

    if ($LASTEXITCODE -ne 0 -or $tempFrames.Count -eq 0) {
      $status = "failed"
      $errorMessage = ($ffmpegOutput | Out-String).Trim()
      if (-not $errorMessage) {
        $errorMessage = "ffmpeg produced no frames"
      }
      "[$((Get-Date).ToString("s"))] failed video=$($video.Name) error=$errorMessage" |
        Add-Content -LiteralPath $logPath -Encoding UTF8
      Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    } else {
      if (Test-Path -LiteralPath $outputDir) {
        Remove-Item -LiteralPath $outputDir -Recurse -Force
      }
      Move-Item -LiteralPath $tempDir -Destination $outputDir
    }
  }

  $actualFrames = @(
    Get-ChildItem -LiteralPath $outputDir -File -Filter ("frame_*_every{0:D3}s.jpg" -f $intervalSec) -ErrorAction SilentlyContinue
  ).Count

  $results.Add([pscustomobject]@{
    video_name = $video.Name
    frame_dir = Resolve-Path -LiteralPath $outputDir
    duration_sec = [Math]::Round($durationSec, 3)
    duration_timecode = Format-Timecode -Seconds $durationSec
    interval_sec = $intervalSec
    frame_name_pattern = $patternName
    frame_count = $actualFrames
    first_frame_sec = 0
    last_frame_sec = if ($actualFrames -gt 0) { ($actualFrames - 1) * $intervalSec } else { "" }
    status = $status
    error = $errorMessage
  }) | Out-Null

  $results | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8
  Write-Status -Status "running" -Completed $completed -Total $videos.Count -CurrentVideo $video.Name
}

$failed = @($results | Where-Object { $_.status -ne "ok" }).Count
$message = if ($failed -eq 0) { "done" } else { "$failed video(s) failed" }
Write-Status -Status "done" -Completed $completed -Total $videos.Count -Message $message
"[$((Get-Date).ToString("s"))] done completed=$completed failed=$failed" |
  Add-Content -LiteralPath $logPath -Encoding UTF8
