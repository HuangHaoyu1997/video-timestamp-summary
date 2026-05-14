param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Url,

    [Parameter(Position = 1)]
    [string] $OutputDir = (Join-Path $env:USERPROFILE 'Downloads\douyin'),

    [int] $WaitSeconds = 12,

    [switch] $KeepTab
)

$ErrorActionPreference = 'Stop'

function Assert-Command {
    param([string] $Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Get-WebAccessSkillDir {
    $skillsRoot = Join-Path $env:USERPROFILE '.codex\skills'
    $match = Get-ChildItem -LiteralPath $skillsRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'Web Access*' } |
        Select-Object -First 1

    if (-not $match) {
        throw "Cannot find Web Access skill under $skillsRoot"
    }

    return $match.FullName
}

function ConvertTo-SafeFileName {
    param(
        [string] $Name,
        [int] $MaxLength = 80
    )

    if ([string]::IsNullOrWhiteSpace($Name)) {
        $Name = 'douyin-video'
    }

    $safe = $Name -replace '\s+', ' '
    foreach ($char in [IO.Path]::GetInvalidFileNameChars()) {
        $safe = $safe.Replace($char, '_')
    }

    $safe = $safe.Trim(" .`t`r`n")
    if ([string]::IsNullOrWhiteSpace($safe)) {
        $safe = 'douyin-video'
    }

    if ($safe.Length -gt $MaxLength) {
        $safe = $safe.Substring(0, $MaxLength).Trim(" .`t`r`n")
    }

    return $safe
}

function Invoke-CdpEval {
    param(
        [string] $TargetId,
        [string] $Script
    )

    $response = Invoke-RestMethod -Method Post -Uri "http://localhost:3456/eval?target=$TargetId" -ContentType 'text/plain; charset=utf-8' -Body $Script
    if ($null -eq $response.value) {
        $details = $response | ConvertTo-Json -Depth 8 -Compress
        throw "CDP eval returned no value: $details"
    }

    return $response.value
}

function Wait-CdpNavigation {
    param(
        [string] $TargetId,
        [int] $TimeoutSeconds = 15
    )

    $lastUrl = ''
    $stableCount = 0
    for ($i = 0; $i -lt $TimeoutSeconds * 2; $i++) {
        try {
            $info = Invoke-RestMethod -Uri "http://localhost:3456/info?target=$TargetId"
            $urlNow = [string] $info.url
            if ($urlNow -and $urlNow -ne 'about:blank') {
                if ($urlNow -eq $lastUrl) {
                    $stableCount++
                } else {
                    $stableCount = 0
                    $lastUrl = $urlNow
                }

                if ($stableCount -ge 2 -and $info.ready -eq 'complete') {
                    return
                }
            }
        } catch {
            Start-Sleep -Milliseconds 500
            continue
        }

        Start-Sleep -Milliseconds 500
    }
}

Assert-Command -Name 'node'
Assert-Command -Name 'yt-dlp'

$skillDir = Get-WebAccessSkillDir
$checkDeps = Join-Path $skillDir 'scripts\check-deps.mjs'
if (-not (Test-Path -LiteralPath $checkDeps)) {
    throw "Cannot find check-deps.mjs at $checkDeps"
}

& node $checkDeps | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Web Access dependency check failed"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$targetId = $null
$createdTab = $false

try {
    $encodedUrl = [Uri]::EscapeDataString($Url)
    $tab = Invoke-RestMethod -Uri "http://localhost:3456/new?url=$encodedUrl"
    $targetId = $tab.targetId
    if ([string]::IsNullOrWhiteSpace($targetId)) {
        throw "Could not create Chrome background tab"
    }
    $createdTab = $true
    Wait-CdpNavigation -TargetId $targetId

    $extractScript = @"
(async () => {
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

  function pickTitle() {
    const text = document.body?.innerText || '';
    const lines = text
      .split('\n')
      .map(line => line.trim())
      .filter(Boolean)
      .filter(line => !/^(开启读屏标签|读屏标签已关闭|精选|推荐|搜索|关注|朋友|我的|直播|放映厅|短剧|小游戏|下载抖音|全部评论|留下你的精彩评论吧)$/.test(line))
      .filter(line => !/^\d+$/.test(line))
      .filter(line => !/^发布时间[:：]/.test(line));

    const hashtagLine = lines.find(line => line.includes('#') && line.length >= 6);
    const aiTextIndex = lines.indexOf('内容由AI生成');
    const afterAiText = aiTextIndex >= 0 ? lines[aiTextIndex + 1] : '';
    return hashtagLine || afterAiText || document.title || 'douyin-video';
  }

  function pickAwemeId(urls) {
    const all = [location.href, ...urls].join('\n');
    return all.match(/video\/(\d{10,})/)?.[1] ||
      all.match(/[?&]__vid=(\d{10,})/)?.[1] ||
      all.match(/[?&]aweme_id=(\d{10,})/)?.[1] ||
      String(Date.now());
  }

  for (let attempt = 0; attempt < $WaitSeconds; attempt++) {
    const videos = [...document.querySelectorAll('video')];
    for (const video of videos) {
      try {
        video.muted = true;
        const playPromise = video.play?.();
        if (playPromise?.catch) playPromise.catch(() => {});
      } catch (_) {}
    }

    const urls = videos.flatMap(video => [
      video.currentSrc,
      video.src,
      ...[...video.querySelectorAll('source')].map(source => source.src)
    ]).filter(Boolean);

    const candidates = urls.filter(url => /douyinvod\.com|\/aweme\/v1\/play|mime_type=video/i.test(url));
    const videoUrl =
      candidates.find(url => /[?&]cs=0(?:&|$)/.test(url)) ||
      candidates.find(url => /mime_type=video_mp4/i.test(url) && !/[?&]cs=2(?:&|$)/.test(url)) ||
      candidates[0];
    if (videoUrl) {
      return JSON.stringify({
        ok: true,
        videoUrl,
        webpageUrl: location.href,
        title: pickTitle(),
        awemeId: pickAwemeId(urls)
      });
    }

    await sleep(1000);
  }

  return JSON.stringify({
    ok: false,
    webpageUrl: location.href,
    title: document.title || '',
    text: (document.body?.innerText || '').slice(0, 800)
  });
})()
"@

    $data = Invoke-CdpEval -TargetId $targetId -Script $extractScript | ConvertFrom-Json
    if (-not $data.ok -or [string]::IsNullOrWhiteSpace($data.videoUrl)) {
        $hint = if ($data.text) { $data.text } else { $data.webpageUrl }
        throw "No Douyin video stream found. Page hint: $hint"
    }

    $titleForFile = (($data.title -split '#')[0]).Trim()
    if ([string]::IsNullOrWhiteSpace($titleForFile)) {
        $titleForFile = $data.title
    }
    $safeTitle = ConvertTo-SafeFileName -Name $titleForFile
    $safeId = ConvertTo-SafeFileName -Name $data.awemeId -MaxLength 32
    $outputPath = Join-Path $OutputDir "$safeTitle-$safeId.mp4"

    $userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
    $ytDlpArgs = @(
        '--force-overwrites',
        '--no-warnings',
        '--add-header', 'Referer:https://www.douyin.com/',
        '--add-header', "User-Agent:$userAgent",
        '-o', $outputPath,
        $data.videoUrl
    )

    Write-Host "Douyin page: $($data.webpageUrl)"
    Write-Host "Title: $($data.title)"
    Write-Host "SafeTitle: $safeTitle"
    Write-Host "Output: $outputPath"
    & yt-dlp @ytDlpArgs
    if ($LASTEXITCODE -ne 0) {
        throw "yt-dlp download failed with exit code $LASTEXITCODE"
    }

    if (Get-Command ffprobe -ErrorAction SilentlyContinue) {
        & ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,duration -show_entries format=duration,size -of default=noprint_wrappers=1 $outputPath | Out-Host
    }

    Write-Host "Done: $outputPath"
}
finally {
    if ($createdTab -and $targetId -and -not $KeepTab) {
        try {
            Invoke-RestMethod -Uri "http://localhost:3456/close?target=$targetId" | Out-Null
        } catch {
            Write-Warning "Could not close background tab: $($_.Exception.Message)"
        }
    }
}
