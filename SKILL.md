---
name: video-timestamp-summary
description: Batch workflow for turning video links into timestamped content-summary JSONL. Use when Codex needs to process video URLs supplied through txt, csv, jsonl, xlsx, pasted lists, or existing local videos; download social videos such as Douyin/Bilibili; run ASR; extract frames; generate reusable topic prompt profiles; caption frames with a VLM; generate long-video summary plans; generate per-video summary JSON; infer segment-level timestamps; or build final_video_segments.jsonl with the bundled scripts.
---

# Video Timestamp Summary

## Overview

Use this skill to turn video links or local videos into `final_video_segments.jsonl`, where each row contains:

```json
{"time_stamp":[[0,0],[1,15]],"segment":[{"title":"","seg_abs":["全文摘要"]},{"title":"分段标题","seg_abs":["分段摘要"]}],"url":"原视频链接","tt":"视频标题","dctt":"全文摘要"}
```

This is a staged batch pipeline. Run commands from the project root, use the bundled scripts under `scripts/`, and load reference files only when their details are needed.

## Reference Guide

- Read `references/setup-and-inputs.md` when preparing a project, copying scripts, configuring model/API values, parsing link files, or handling local-only videos.
- Read `references/download-and-asr.md` when downloading/registering videos, running ASR, extracting frames, or diagnosing ffmpeg/Whisper issues. For Whisper installation, model weights, CUDA notes, and ASR examples, also read `references/whisper-setup.md`.
- Read `references/model-pipeline.md` when running prompt profile generation, frame captioning, long-video planning, summary generation, or timestamp inference.
- Read `references/output-contract.md` before changing JSON fields, validating summaries/timestamps, or building `final_video_segments.jsonl`.
- Read `references/rerun-debugging.md` when repairing partial runs, using `--force`/`--only`/`--limit`, or deciding which stage to rerun.

## Core Workflow

1. Prepare the target project and inputs. See `references/setup-and-inputs.md`.

```powershell
python download_videos.py --input video_links.txt
```

2. Generate ASR transcripts.

```powershell
python asr_batch.py --input video --output asr --model-name large-v3-turbo --model-path "$env:USERPROFILE\.cache\whisper\large-v3-turbo.pt"
```

Downstream stages expect `asr/transcript/{视频名}.srt`.

3. Extract frames.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\extract_frames.ps1
```

This produces `frame/{视频名}/...`, `frame_manifest.csv`, and extraction status/log files.

4. Generate reusable topic prompt profiles.

```powershell
python prompt_profile_batch.py --dry-run
python prompt_profile_batch.py
```

Profiles are batch/topic-level guidance, not facts. Do not create a bespoke prompt for every video.

5. Caption frame groups with a VLM.

```powershell
python caption_batch.py --dry-run
python caption_batch.py
```

If `prompt_profiles/` exists, use the assigned profile unless debugging with `--ignore-profile`.

6. Generate long-video summary plans.

```powershell
python plan_batch.py --dry-run
python plan_batch.py
```

By default, this only processes videos longer than 15 minutes. Plans guide summary generation; they are not final output.

7. Generate per-video summary JSON.

```powershell
python summary_batch.py --dry-run
python summary_batch.py
```

The script merges ASR and captions into one time-ordered timeline before prompting. If plan/profile files exist, use them unless debugging.

8. Infer segment-level timestamps.

```powershell
python timestamp_batch.py --dry-run
python timestamp_batch.py
```

The timestamp array must align exactly with the summary `segment` array.

9. Build the final JSONL.

```powershell
python build_final_jsonl.py --dry-run --strict
python build_final_jsonl.py --strict
```

For local-only videos, use `--missing-url-policy file-url` or `--missing-url-policy empty` as appropriate.

## Invariants

- Generate facts only from video metadata, ASR, captions, frames, plans, summaries, and timestamp evidence available in the pipeline. Do not invent content.
- Keep video stems aligned across `video/`, `asr/transcript/`, `frame/`, `caption/captions/`, `plan/`, `abstract/summaries/`, and `timestamp/time_stamps/`.
- Preserve `segment[0]` as the full-video summary even though `dctt` duplicates it in the final JSONL.
- Preserve segment/timestamp alignment: `time_stamp[0]` is `[0, 0]`, and `len(time_stamp) == len(segment)`.
- Prefer stage-local resume and narrow repair before rerunning the whole batch.

## Final Output

`final_video_segments.jsonl` records must preserve these fields exactly:

- `time_stamp`: segment-aligned timestamp intervals.
- `segment`: complete segment array, including full-video summary at index 0.
- `url`: source video URL; local-only videos may use a local `file://` URL or `""` depending on `build_final_jsonl.py --missing-url-policy`.
- `tt`: video title.
- `dctt`: `segment[0].seg_abs[0]`.

This pipeline produces single-video records. Query-level aggregation or downstream knowledge-base ingestion belongs to a later step.
