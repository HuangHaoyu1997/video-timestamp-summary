# Rerun And Debugging

Use this reference when repairing partial runs, narrowing a failing stage, or deciding whether to rerun upstream/downstream outputs.

## Resume Strategy

Prefer stage-local resume behavior:

- Download: rerun `python download_videos.py --input ...`; successful URLs in `video_title_url_mapping.csv` are skipped.
- ASR: rerun without `--overwrite` to skip completed transcripts; add `--overwrite` for forced regeneration.
- Prompt profiles: rerun without `--force` to reuse existing profiles; add `--force` when the batch topic mix changes or profile quality is poor. If profiles change after caption or summary already ran, rerun affected downstream stages with `--force`.
- Caption: rerun without `--force` to skip caption txt only when it covers every expected frame group; invalid or partial caption files are regenerated while successful raw frame groups are reused.
- Plan: rerun without `--force` to skip valid plan txt; add `--only "视频名片段"` for targeted repair, or `--min-duration-sec 0` to force planning for shorter videos during testing.
- Summary: rerun without `--force` to skip valid summary JSON; add `--only "视频名片段"` for targeted repair. If a plan changes, rerun the corresponding summary with `--force`.
- Timestamp: rerun without `--force` to skip valid timestamp JSON; add `--only "视频名片段"` for targeted repair.
- Final JSONL: rerun `build_final_jsonl.py --strict` after upstream repair; this stage is deterministic and overwrites atomically. For local-only videos, use `--missing-url-policy file-url` or `--missing-url-policy empty`.

## Failure Handling

When a stage fails, inspect its status file and error log first:

```text
video_download_status.jsonl
download.log
frame_extraction_status.json
frame_extraction.log
prompt_profiles/profile_status.jsonl
prompt_profiles/profile_errors.log
caption/caption_status.jsonl
caption/caption_errors.log
plan/plan_status.jsonl
plan/plan_errors.log
abstract/abstract_status.jsonl
abstract/abstract_errors.log
timestamp/timestamp_status.jsonl
timestamp/timestamp_errors.log
```

Patch data or rerun a narrow `--only`/`--limit` subset before launching the whole batch again.

## Downstream Repair Rules

- If videos are redownloaded or renamed, verify `video_title_url_mapping.csv`, then rerun ASR and frame extraction for affected videos.
- If ASR changes, rerun prompt profile generation when batch samples are affected, then rerun captions only if the profile guidance changes. Always rerun plans, summaries, timestamps, and final JSONL for affected videos.
- If frames change, rerun prompt profiles if frame samples are affected, then rerun captions, plans, summaries, timestamps, and final JSONL.
- If prompt profiles change, rerun captions and summaries that used the old profile, then rerun plans only when caption output changes or profile guidance materially affects planning.
- If captions change, rerun plans, summaries, timestamps, and final JSONL.
- If plans change, rerun summaries, timestamps, and final JSONL.
- If summaries change, rerun timestamps and final JSONL.
- If timestamps change, rerun final JSONL.
