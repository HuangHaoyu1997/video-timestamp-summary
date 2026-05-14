# Model Pipeline

Use this reference for model-calling stages after ASR and frame extraction: prompt profiles, captions, long-video plans, summaries, and segment timestamps.

## Prompt Profiles

Generate reusable topic prompt profiles:

```powershell
python prompt_profile_batch.py --dry-run
python prompt_profile_batch.py
```

This stage happens after ASR and frame extraction, before caption and abstract. It samples the batch's videos, combines ASR excerpts with representative frames, and asks the model to create a small set of reusable topic profiles.

Expected outputs:

```text
prompt_profiles/profiles.json
prompt_profiles/video_assignments.jsonl
prompt_profiles/raw/profile_generation.jsonl
prompt_profiles/profile_status.jsonl
prompt_profiles/profile_errors.log
```

Profile rules:

- Generate a small number of reusable profiles for the batch, normally 1-6.
- A profile captures topic type and prompt guidance, for example caption visual priorities and abstract summarization priorities.
- `video_assignments.jsonl` maps each video stem to one profile; this assignment is allowed per video, but profile generation itself is batch/topic level.
- Profiles are guidance, not facts. Caption, plan, abstract, and timestamp facts must still come from frames, ASR, captions, and timeline evidence.
- Use `--sample-size`, `--frames-per-video`, and `--max-profiles` to control cost and granularity.
- Use `--force` to regenerate profiles when the batch domain changes.

## Captions

Caption frame groups with a VLM:

```powershell
python caption_batch.py --dry-run
python caption_batch.py --limit 2
python caption_batch.py
python caption_batch.py --ignore-profile
```

Expected outputs:

```text
caption/captions/{视频名}.txt
caption/raw/{视频名}.jsonl
caption/caption_status.jsonl
caption/caption_errors.log
```

Caption requirements:

- Process frames in video time order.
- If `prompt_profiles/` exists, use the assigned profile's `caption_prompt` to adapt the visual description priorities; use `--ignore-profile` only for debugging.
- Send 5-10 images per request depending on duration: about 5 for <=1 min, 6 for 1-5 min, 8 for 5-8 min, 10 for >8 min.
- Write each caption line with a leading time range, for example `[00:00:01-00:00:05] 画面显示...`.
- Default behavior skips non-empty finished caption files and reuses successful raw frame groups; use `--force` to rerun.

## Long-Video Plans

Generate long-video summary plans:

```powershell
python plan_batch.py --dry-run
python plan_batch.py --limit 3
python plan_batch.py
```

This stage runs after ASR and caption, before abstract generation. By default it only processes videos with `duration_sec > 900` (longer than 15 minutes). Shorter videos skip planning and go directly to summary generation.

The plan is not the final summary and does not enter `final_video_segments.jsonl`. It is a guidance document for the summary model: content structure, knowledge to preserve, material to ignore, suggested continuous knowledge-unit segmentation, and low-confidence caveats.

Expected outputs:

```text
plan/{视频名}.txt
plan/raw/{视频名}.jsonl
plan/plan_status.jsonl
plan/plan_errors.log
```

Plan requirements:

- Generate only from ASR, captions, and video metadata; do not invent facts.
- Keep `plan/{视频名}.txt` stem aligned with the video stem so `summary_batch.py` can find it.
- Cover `内容主线`, `重点保留`, `规避和忽略`, `分段安排`, `风险和低置信度`, and `给摘要模型的约束`.
- Treat the plan as a strategy for RAG-friendly summarization, not as a replacement for the final JSON summary.
- Default behavior skips existing valid plan files; use `--force` to rerun.
- Use `--min-duration-sec N` to change the long-video threshold when needed.

## Summary JSON

Generate per-video summary JSON:

```powershell
python summary_batch.py --dry-run
python summary_batch.py --limit 3
python summary_batch.py
python summary_batch.py --ignore-plan
python summary_batch.py --ignore-profile
```

The script merges `asr/transcript/{视频名}.srt` and `caption/captions/{视频名}.txt` into one time-ordered timeline before prompting the LLM. Preserve this merge rule when debugging manually; do not feed ASR and captions as two unrelated blocks.

If `plan/{视频名}.txt` exists, `summary_batch.py` reads it as important guidance for content organization and filtering. Use `--ignore-plan` only for debugging or when plan quality is known to be harmful. Use `--plan-dir <dir>` when plans are stored outside the default `plan/` directory.

If `prompt_profiles/` exists, `summary_batch.py` also reads the assigned profile's `summary_prompt` to adapt the summary priorities to the batch topic. Use `--ignore-profile` only for debugging or when the generated profile is unsuitable. Use `--profile-dir <dir>` when profiles are stored outside the default `prompt_profiles/` directory.

Expected outputs:

```text
abstract/summaries/{视频名}.json
abstract/raw/{视频名}.jsonl
abstract/abstract_status.jsonl
abstract/abstract_errors.log
```

Summary JSON rules:

- Top-level fields must be exactly `video` and `segment`.
- `video` contains `video_id`, `title`, `url`, `duration_sec`, and `platform`.
- `segment[0]` is the full-video summary; its `title` is `""` and `seg_abs` contains exactly one string.
- `segment[1:]` are continuous knowledge units in video order, not per-shot or per-click fragments.
- Merge adjacent material when it belongs to one function explanation, operation flow, product dimension, or evaluation conclusion.
- Keep short videos compact: usually one detail segment, at most two unless the content clearly changes topic.
- Use `--max-detail-segments N` only when the default duration-based limit needs adjustment.

Default detail segment caps:

```text
duration <= 90s: 1
duration <= 180s: 2
duration <= 300s: 3
duration <= 480s: 4
duration > 480s: 6
```

## Timestamps

Infer segment-level timestamps:

```powershell
python timestamp_batch.py --dry-run
python timestamp_batch.py --limit 3
python timestamp_batch.py
```

Expected outputs:

```text
timestamp/time_stamps/{视频名}.json
timestamp/raw/{视频名}.jsonl
timestamp/timestamp_status.jsonl
timestamp/timestamp_errors.log
```

Timestamp JSON contract:

```json
{"time_stamp":[[0,0],[1,15],[16,38]]}
```

- `time_stamp[0]` corresponds to `segment[0]` and must be `[0, 0]`.
- `len(time_stamp)` must equal `len(segment)`.
- Each later item is `[start_sec, end_sec]` as integers.
- Intervals must be non-duplicated, monotonic, and not exceed video duration.
- A segment timestamp covers the whole segment, not a single `seg_abs` item.
