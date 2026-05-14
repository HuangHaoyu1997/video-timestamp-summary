# Setup And Inputs

Use this reference when preparing a target project, copying bundled resources, configuring model/API values, parsing link files, or running the pipeline from local videos.

## Copy Bundled Files

Copy the bundled scripts into the target project root, or use existing project copies if they are already present:

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" }
$skillDir = Join-Path $codexHome "skills\video-timestamp-summary"
$scriptDir = Join-Path $skillDir "scripts"
Copy-Item -Path (Join-Path $scriptDir "*") -Destination "." -Force
$referenceDir = Join-Path $skillDir "references"
if (Test-Path -LiteralPath $referenceDir) {
  Copy-Item -Path $referenceDir -Destination "." -Recurse -Force
}
```

## Runtime Dependencies

Install or verify:

- Python packages: `openpyxl`, `openai-whisper`, `torch`.
- CLI tools: `yt-dlp`, `ffmpeg`, `ffprobe`, PowerShell.
- Douyin downloads: local Web Access skill dependencies, because `Save-DouyinVideo.ps1` uses its browser/CDP helper.
- ASR model: pass a valid Whisper checkpoint with `--model-path`, or use `--model-name` to let Whisper download/cache an official model. For installation, weight download, CUDA notes, and script usage, read `references/whisper-setup.md`.

## Model Configuration

Set model config once per shell session:

```powershell
$env:OPENAI_MODEL = "gpt-5.4-mini"
$env:OPENAI_BASE_URL = "https://your-openai-compatible-base-url"
$env:OPENAI_API_KEY = "sk-..."
```

Equivalent flags are available on `prompt_profile_batch.py`, `caption_batch.py`, `plan_batch.py`, `summary_batch.py`, and `timestamp_batch.py`:

```powershell
--model gpt-5.4-mini --base-url "https://..." --api-key "sk-..."
```

## Link Files

Use `download_videos.py --input <file>` for link files. Supported formats:

- `txt` / `md`: every line is scanned for URLs; remaining text on the line becomes the source query/context.
- `csv`: every cell is scanned for URLs; query defaults to a recognized query column or the second column.
- `jsonl`: each JSON object is scanned recursively for URL strings; query defaults to keys such as `url`, `title`.
- `xlsx` / `xlsm`: every sheet is scanned; query defaults to a recognized query header or the second column.

Useful commands:

```powershell
python download_videos.py --input video_links.txt
python download_videos.py --input links.csv --query-field query --url-field url
python download_videos.py --input part1.jsonl --input part2.xlsx
```

If `--input` is omitted, the script auto-detects common names such as `video_links.txt`, `video_links.csv`, and `video_links.jsonl`.

Expected download outputs:

```text
video/
video_urls.csv
video_url_occurrences.csv
video_title_url_mapping.csv
video_download_status.jsonl
download.log
```

`video_title_url_mapping.csv` preserves `source_files`, source rows, URL, platform, downloaded title, and file path. Later stages use this mapping for metadata and ordering.

## Local Videos

If videos already exist locally, place them under `video/` and provide or create `video_title_url_mapping.csv` when source URL/title provenance matters.

ASR and later stages can run from local videos once `frame_manifest.csv` exists. In the final stage, `build_final_jsonl.py` can fill a missing source URL from a local `file://` URL with `--missing-url-policy file-url` (the default) or write an empty string with `--missing-url-policy empty`.
