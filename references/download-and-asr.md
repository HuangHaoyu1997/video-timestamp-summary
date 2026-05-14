# Download And ASR

Use this reference for the deterministic media-prep stages: downloading/registering videos, transcribing audio, and extracting frames.

## Download Or Register Videos

Run from the project root:

```powershell
python download_videos.py --input video_links.txt
```

The download stage writes:

```text
video/
video_urls.csv
video_url_occurrences.csv
video_title_url_mapping.csv
video_download_status.jsonl
download.log
```

Rerunning `download_videos.py` skips successful URLs already present in `video_title_url_mapping.csv`.

## ASR

Generate ASR with sentence-level timestamps:

```powershell
python asr_batch.py --input video --output asr --model-name large-v3-turbo --model-path "$env:USERPROFILE\.cache\whisper\large-v3-turbo.pt"
```

Use `--dry-run`, `--limit N`, and `--overwrite` for planning, sampling, and forced reruns.

Downstream stages expect:

```text
asr/transcript/{视频名}.srt
```

If Whisper, PyTorch, ffmpeg, or model weights are not ready, read `references/whisper-setup.md` before running ASR.

## Frame Extraction

Extract frames after ASR/download:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\extract_frames.ps1
```

Frame interval rules:

```text
duration <= 300s: 1 frame/sec
duration <= 480s: 1 frame/2 sec
duration > 480s: 1 frame/4 sec
```

Expected outputs:

```text
frame/{视频名}/frame_000001_every001s.jpg
frame_manifest.csv
frame_extraction.log
frame_extraction_status.json
```

Frame timestamp is inferred from the filename:

```text
timestamp_sec = (frame_index - 1) * interval_sec
```

The extractor skips existing matching frame directories by default. Use `-Force` to regenerate frames. Successful extraction writes to a temporary directory first and replaces the old directory only after `ffmpeg` succeeds.
