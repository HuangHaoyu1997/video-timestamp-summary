# Video Timestamp Summary

Batch workflow for turning video links or local videos into timestamped content-summary JSONL.

This repository is packaged as a Codex skill. It includes scripts for downloading videos, running ASR, extracting frames, captioning visual evidence, generating summaries, inferring segment timestamps, and building a final `final_video_segments.jsonl` file.

## Output Shape

Each final JSONL row has this shape:

```json
{"time_stamp":[[0,0],[1,15]],"segment":[{"title":"","seg_abs":["全文摘要"]},{"title":"分段标题","seg_abs":["分段摘要"]}],"url":"原视频链接","tt":"视频标题","dctt":"全文摘要"}
```

## Requirements

- Python 3.12+
- PowerShell
- `ffmpeg` and `ffprobe`
- `yt-dlp`
- Python packages from `requirements.txt`
- PyTorch and Whisper for ASR
- OpenAI-compatible chat and vision model access for model-calling stages

Install the lightweight Python dependencies:

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Install PyTorch for your machine separately using the official PyTorch selector. For Whisper setup details, see `references/whisper-setup.md`.

## Usage

Create or choose a working directory for one video batch, then copy the bundled scripts into that directory:

```powershell
New-Item -ItemType Directory -Force -Path .\work | Out-Null
Copy-Item -Path .\scripts\* -Destination .\work -Force
Copy-Item -Path .\references -Destination .\work -Recurse -Force
Set-Location .\work
```

Configure model access:

```powershell
$env:OPENAI_MODEL = "gpt-5.4-mini"
$env:OPENAI_BASE_URL = "https://your-openai-compatible-base-url"
$env:OPENAI_API_KEY = "sk-..."
```

Run the pipeline:

```powershell
python download_videos.py --input video_links.txt
python asr_batch.py --input video --output asr --model-name large-v3-turbo --model-path "$env:USERPROFILE\.cache\whisper\large-v3-turbo.pt"
powershell -NoProfile -ExecutionPolicy Bypass -File .\extract_frames.ps1
python prompt_profile_batch.py
python caption_batch.py
python plan_batch.py
python summary_batch.py
python timestamp_batch.py
python build_final_jsonl.py --strict
```

Use `--dry-run`, `--limit`, `--only`, and stage-specific force flags while testing or repairing a partial batch.

## Documentation

- `SKILL.md`: Codex skill entry point and core workflow.
- `references/setup-and-inputs.md`: project setup, inputs, local videos, and model configuration.
- `references/download-and-asr.md`: download, ASR, and frame extraction stages.
- `references/model-pipeline.md`: prompt profiles, captions, plans, summaries, and timestamps.
- `references/output-contract.md`: summary, timestamp, and final JSONL contracts.
- `references/rerun-debugging.md`: resume and repair strategy.
- `references/whisper-setup.md`: Whisper, PyTorch, ffmpeg, and model weight setup.

## Notes

Generated media, transcripts, captions, logs, and final JSONL files are intentionally ignored by git. Keep secrets in environment variables or local `.env` files, not in tracked files.
