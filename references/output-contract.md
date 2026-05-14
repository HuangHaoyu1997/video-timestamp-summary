# Output Contract

Use this reference before changing JSON fields, validating summaries or timestamps, or building `final_video_segments.jsonl`.

## Per-Video Summary JSON

`abstract/summaries/{视频名}.json` must use exactly these top-level fields:

```json
{"video":{},"segment":[]}
```

`video` contains:

- `video_id`
- `title`
- `url`
- `duration_sec`
- `platform`

`segment` rules:

- `segment[0]` is the full-video summary.
- `segment[0].title` is `""`.
- `segment[0].seg_abs` contains exactly one string.
- `segment[1:]` are continuous knowledge units in video order, not per-shot or per-click fragments.
- Adjacent material should be merged when it belongs to one function explanation, operation flow, product dimension, or evaluation conclusion.
- Short videos should stay compact: usually one detail segment, at most two unless the content clearly changes topic.

## Timestamp JSON

`timestamp/time_stamps/{视频名}.json` must contain:

```json
{"time_stamp":[[0,0],[1,15],[16,38]]}
```

Rules:

- `time_stamp[0]` corresponds to `segment[0]` and must be `[0, 0]`.
- `len(time_stamp)` must equal `len(segment)`.
- Each later item is `[start_sec, end_sec]` as integers.
- Intervals must be non-duplicated, monotonic, and not exceed video duration.
- A segment timestamp covers the whole segment, not a single `seg_abs` item.

## Final JSONL

Build the final JSONL:

```powershell
python build_final_jsonl.py --dry-run --strict
python build_final_jsonl.py --strict
python build_final_jsonl.py --strict --missing-url-policy file-url --video-dir video
python build_final_jsonl.py --strict --missing-url-policy empty
```

Expected output:

```text
final_video_segments.jsonl
```

Each row must preserve these fields exactly:

- `time_stamp`: segment-aligned timestamp intervals.
- `segment`: complete segment array, including the full-video summary at index 0.
- `url`: source video URL; for local-only videos this may be a local `file://` URL or `""` according to `build_final_jsonl.py --missing-url-policy`.
- `tt`: video title.
- `dctt`: `segment[0].seg_abs[0]`.

Do not remove `segment[0]` even though `dctt` duplicates the full summary; `time_stamp[0]` still aligns with it.

Final URL policy:

- Existing `abstract.video.url` always wins.
- `--missing-url-policy file-url` fills missing URLs from `video_title_url_mapping.csv` `file_path`/`filename` or files under `--video-dir`.
- `--missing-url-policy empty` allows local-only records to keep `url` as `""`.
- `--missing-url-policy error` preserves the old strict non-empty URL behavior.

This pipeline produces single-video records. Query-level aggregation or downstream knowledge-base ingestion belongs to a later step.
