from __future__ import annotations

import argparse
import http.client
import json
import math
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import summary_batch as summary


DEFAULT_ABSTRACT_DIR = Path("abstract") / "summaries"
DEFAULT_OUTPUT = Path("timestamp")


class TimestampValidationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch infer segment-level timestamps from abstract, ASR, and caption timelines."
    )
    parser.add_argument("--manifest", default=str(summary.DEFAULT_MANIFEST), help="Frame manifest CSV path.")
    parser.add_argument("--mapping", default=str(summary.DEFAULT_MAPPING), help="Video title/url mapping CSV path.")
    parser.add_argument("--abstract-dir", default=str(DEFAULT_ABSTRACT_DIR), help="Abstract summary JSON directory.")
    parser.add_argument("--asr-dir", default=str(summary.DEFAULT_ASR_DIR), help="ASR transcript directory.")
    parser.add_argument("--caption-dir", default=str(summary.DEFAULT_CAPTION_DIR), help="Caption txt directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory. Defaults to ./timestamp.")
    parser.add_argument("--model", default=None, help=f"Model name. Defaults to {summary.DEFAULT_MODEL}.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Falls back to OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos after filtering.")
    parser.add_argument("--only", action="append", default=[], help="Only process videos whose name contains this text. Repeatable.")
    parser.add_argument("--start-after", default=None, help="Skip videos until this text is found in video name.")
    parser.add_argument("--force", action="store_true", help="Re-run even when final timestamp JSON already exists and is valid.")
    parser.add_argument("--workers", type=int, default=5, help="Number of videos to process concurrently.")
    parser.add_argument("--rpm", type=int, default=60, help="Max API requests per rolling minute. Use 0 to disable.")
    parser.add_argument("--tpm", type=int, default=900000, help="Estimated max tokens per rolling minute. Use 0 to disable.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout per request in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for API, JSON parse, and validation failures.")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Initial retry sleep in seconds.")
    parser.add_argument("--max-retry-sleep", type=float, default=60.0, help="Maximum sleep between retries in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens per timestamp inference.")
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming responses. Streaming is enabled by default for this text-only endpoint.",
    )
    parser.set_defaults(stream=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse inputs and write no model outputs. Useful for checking abstract/timeline availability.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(summary.read_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def output_paths(output_dir: Path, job: summary.VideoJob) -> dict[str, Path]:
    return {
        "timestamp": output_dir / "time_stamps" / f"{job.stem}.json",
        "raw": output_dir / "raw" / f"{job.stem}.jsonl",
    }


def abstract_path(abstract_dir: Path, job: summary.VideoJob) -> Path:
    return abstract_dir / f"{job.stem}.json"


def load_abstract(abstract_dir: Path, job: summary.VideoJob) -> dict[str, Any]:
    path = abstract_path(abstract_dir, job)
    if not path.exists():
        raise FileNotFoundError(f"Abstract JSON not found: {path}")
    data = read_json(path)
    summary.validate_summary(data)
    return data


def max_end_sec_for_duration(duration_sec: float) -> int:
    return max(0, int(math.floor(duration_sec)))


def validate_timestamp_payload(payload: dict[str, Any], abstract_data: dict[str, Any], duration_sec: float) -> None:
    errors: list[str] = []
    if set(payload) != {"time_stamp"}:
        extra = sorted(set(payload) - {"time_stamp"})
        missing = sorted({"time_stamp"} - set(payload))
        if missing:
            errors.append(f"顶层缺少字段: {', '.join(missing)}。")
        if extra:
            errors.append(f"顶层包含额外字段: {', '.join(extra)}。")

    stamps = payload.get("time_stamp")
    segments = abstract_data.get("segment")
    if not isinstance(stamps, list):
        errors.append("time_stamp 必须是数组。")
        raise TimestampValidationError(" ".join(errors))
    if not isinstance(segments, list):
        errors.append("abstract.segment 必须是数组。")
        raise TimestampValidationError(" ".join(errors))

    if len(stamps) != len(segments):
        errors.append(f"len(time_stamp) 必须等于 len(segment)：当前 {len(stamps)} != {len(segments)}。")

    parsed: list[tuple[int, int]] = []
    for idx, item in enumerate(stamps):
        if not isinstance(item, list) or len(item) != 2:
            errors.append(f"time_stamp[{idx}] 必须是长度为 2 的数组。")
            continue
        start, end = item
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"time_stamp[{idx}] 的 start/end 必须是 int。")
            continue
        parsed.append((start, end))

    if parsed:
        if parsed[0] != (0, 0):
            errors.append("time_stamp[0] 必须严格等于 [0, 0]。")
        seen: set[tuple[int, int]] = set()
        previous_end: int | None = None
        max_end = max_end_sec_for_duration(duration_sec)
        for idx, (start, end) in enumerate(parsed):
            if (start, end) in seen:
                errors.append(f"time_stamp[{idx}] 与之前区间重复: [{start}, {end}]。")
            seen.add((start, end))

            if idx > 0:
                if start < 0:
                    errors.append(f"time_stamp[{idx}].start 不能小于 0。")
                if end < start:
                    errors.append(f"time_stamp[{idx}] 必须满足 start <= end。")
                if previous_end is not None and start < previous_end:
                    errors.append(
                        f"time_stamp[{idx}] 不单调：start={start} 小于前一个 end={previous_end}。"
                    )
                if end > max_end:
                    errors.append(
                        f"time_stamp[{idx}].end={end} 超过视频时长允许的最大整数秒 {max_end}。"
                    )
            previous_end = end

    if errors:
        raise TimestampValidationError(" ".join(errors))


def normalize_timestamp_payload(payload: dict[str, Any], abstract_data: dict[str, Any], duration_sec: float) -> dict[str, Any]:
    validate_timestamp_payload(payload, abstract_data, duration_sec)
    return {"time_stamp": payload["time_stamp"]}


def existing_timestamp_is_valid(path: Path, abstract_data: dict[str, Any], duration_sec: float) -> tuple[bool, str | None]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, "timestamp missing"
    try:
        payload = read_json(path)
        validate_timestamp_payload(payload, abstract_data, duration_sec)
        return True, None
    except Exception as exc:
        return False, str(exc)


def render_segments(abstract_data: dict[str, Any]) -> str:
    lines = ["[SEGMENTS]"]
    for idx, segment in enumerate(abstract_data["segment"]):
        title = segment.get("title", "")
        lines.append(f"segment[{idx}] title: {title!r}")
        for abs_idx, item in enumerate(segment.get("seg_abs", []), start=1):
            lines.append(f"segment[{idx}].seg_abs[{abs_idx}]: {item}")
        lines.append("")
    return "\n".join(lines).strip()


def build_messages(
    abstract_data: dict[str, Any],
    timeline_text: str,
    duration_sec: float,
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    segment_count = len(abstract_data["segment"])
    max_end = max_end_sec_for_duration(duration_sec)
    validation_hint = ""
    if validation_error:
        validation_hint = (
            "\n\n上一次输出没有通过脚本校验，错误如下：\n"
            f"{validation_error}\n"
            "请重新输出严格合法 JSON，并修复以上问题。"
        )

    system_prompt = (
        "你是视频摘要与时间线对齐专家。"
        "你会根据单视频 abstract、ASR 和画面 caption 的统一时间线，"
        "为每个 segment 反推出覆盖的视频时间范围。"
    )
    user_prompt = (
        "请根据下面的 abstract segments 和 timeline，输出 segment 级时间戳 JSON。\n\n"
        "硬性要求：\n"
        "1. 只输出严格 JSON，不要 Markdown 代码块，不要注释，不要解释性文字。\n"
        "2. 顶层只能包含 time_stamp 一个字段。\n"
        f"3. time_stamp 必须包含 {segment_count} 个区间，数量必须等于 segment 数量。\n"
        "4. time_stamp[0] 对应全文摘要 segment[0]，必须严格为 [0, 0]。\n"
        "5. time_stamp[i] 对应 segment[i]，不是 seg_abs 级时间戳。\n"
        "6. 每个区间格式是 [start_sec, end_sec]，单位为秒，start_sec 和 end_sec 必须是 int。\n"
        "7. 你需要结合 segment 标题、seg_abs 内容、ASR 文本和 CAPTION 画面描述，判断该 segment 覆盖的连续视频时间范围。\n"
        "8. 区间不能重复，必须按时间单调递增；后一个区间的 start_sec 必须大于等于前一个区间的 end_sec。\n"
        f"9. 最后一个 end_sec 不能超过视频时长；本视频最大允许 end_sec 为 {max_end}。\n"
        "10. 对于一个 segment 内多个 seg_abs，要给出覆盖整个 segment 内容的起止范围，而不是只覆盖其中某一句。\n"
        "11. start_sec 必须取该 segment 内容在 timeline 中首次出现的时间；如果 segment 讲的是完整操作流程，入口、打开页面、准备动作也属于该 segment，不能只从核心点击或结果展示开始。\n"
        "12. end_sec 必须覆盖该 segment 最后一个相关动作、结果展示或注意事项。\n"
        "13. 若边界不确定，优先选择能完整覆盖该 segment 内容的最小连续区间，不要跨到后续无关主题。\n"
        "14. 如果除全文摘要外只有一个 segment，且它概括整条短视频的主要流程，通常应覆盖从第一个相关 ASR/CAPTION 到最后一个相关 ASR/CAPTION。\n\n"
        "输出 JSON 示例：\n"
        "{\n"
        "  \"time_stamp\": [[0, 0], [0, 18], [18, 43]]\n"
        "}\n\n"
        f"[VIDEO_DURATION_SEC]\n{duration_sec}\n\n"
        f"{render_segments(abstract_data)}\n\n"
        f"{timeline_text}"
        f"{validation_hint}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def estimate_tokens(timeline_text: str, abstract_data: dict[str, Any], max_tokens: int) -> int:
    return max(1, (len(timeline_text) + len(render_segments(abstract_data))) // 2 + max_tokens + 800)


def request_timestamp(
    api: summary.ApiConfig,
    abstract_data: dict[str, Any],
    timeline_text: str,
    duration_sec: float,
    args: argparse.Namespace,
    raw_path: Path,
    rate_limiter: summary.SlidingWindowRateLimiter | None,
) -> dict[str, Any]:
    url = summary.chat_completions_url(api.base_url)
    headers = {
        "Authorization": f"Bearer {api.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "User-Agent": summary.DEFAULT_USER_AGENT,
    }
    video = abstract_data.get("video", {})
    title = video.get("title", "")
    video_id = video.get("video_id", "")
    validation_error: str | None = None
    estimated_tokens = estimate_tokens(timeline_text, abstract_data, args.max_tokens)

    for attempt in range(1, args.retries + 2):
        payload = {
            "model": api.model,
            "messages": build_messages(abstract_data, timeline_text, duration_sec, validation_error),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": bool(args.stream),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if rate_limiter is not None:
            rate_limiter.acquire(estimated_tokens)
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                response_text = resp.read().decode("utf-8")
            response_data = summary.parse_response_data(response_text)
            content = summary.parse_chat_content(response_data)
            parsed = summary.parse_strict_json(content)
            normalized = normalize_timestamp_payload(parsed, abstract_data, duration_sec)
            summary.append_jsonl(
                raw_path,
                {
                    "time": summary.now_iso(),
                    "status": "ok",
                    "attempt": attempt,
                    "video_id": video_id,
                    "title": title,
                    "content": content,
                    "response": response_data,
                },
            )
            return normalized
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            summary.append_jsonl(
                raw_path,
                {
                    "time": summary.now_iso(),
                    "status": "error",
                    "attempt": attempt,
                    "video_id": video_id,
                    "title": title,
                    "error_type": "http",
                    "error": f"HTTP {exc.code}: {response_body[:1000]}",
                },
            )
            if attempt > args.retries or not summary.should_retry_http(exc.code):
                raise RuntimeError(f"HTTP {exc.code}: {response_body[:1000]}") from exc
            time.sleep(summary.retry_sleep_seconds(args, attempt, exc.headers))
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            ConnectionResetError,
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            summary.append_jsonl(
                raw_path,
                {
                    "time": summary.now_iso(),
                    "status": "error",
                    "attempt": attempt,
                    "video_id": video_id,
                    "title": title,
                    "error_type": "transport_or_response",
                    "error": str(exc),
                },
            )
            if attempt > args.retries:
                raise RuntimeError(str(exc)) from exc
            time.sleep(summary.retry_sleep_seconds(args, attempt))
        except (summary.ValidationError, TimestampValidationError) as exc:
            validation_error = str(exc)
            summary.append_jsonl(
                raw_path,
                {
                    "time": summary.now_iso(),
                    "status": "error",
                    "attempt": attempt,
                    "video_id": video_id,
                    "title": title,
                    "error_type": "validation",
                    "error": validation_error,
                },
            )
            if attempt > args.retries:
                raise RuntimeError(validation_error) from exc
            time.sleep(summary.retry_sleep_seconds(args, attempt))

    raise RuntimeError("unreachable retry loop exit")


def build_timeline_for_job(job: summary.VideoJob, asr_dir: Path, caption_dir: Path) -> tuple[str, int, int, int]:
    events, asr_count, caption_count = summary.build_timeline_for_job(job, asr_dir, caption_dir)
    return summary.render_timeline(events), asr_count, caption_count, len(events)


def process_video(
    job: summary.VideoJob,
    api: summary.ApiConfig,
    abstract_dir: Path,
    asr_dir: Path,
    caption_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    status_lock: threading.Lock,
    error_lock: threading.Lock,
    rate_limiter: summary.SlidingWindowRateLimiter | None,
) -> dict[str, Any]:
    paths = output_paths(output_dir, job)
    status_path = output_dir / "timestamp_status.jsonl"
    error_path = output_dir / "timestamp_errors.log"

    try:
        abstract_data = load_abstract(abstract_dir, job)
        duration_sec = float(abstract_data.get("video", {}).get("duration_sec") or job.duration_sec)

        if args.force:
            for path in paths.values():
                if path.exists():
                    path.unlink()
        else:
            valid, invalid_reason = existing_timestamp_is_valid(paths["timestamp"], abstract_data, duration_sec)
            if valid:
                record = {
                    "time": summary.now_iso(),
                    "status": "skipped",
                    "reason": "timestamp exists",
                    "video_name": job.video_name,
                    "timestamp_path": str(paths["timestamp"]),
                    "raw_path": str(paths["raw"]),
                }
                summary.append_jsonl(status_path, record, status_lock)
                return record
            if paths["timestamp"].exists():
                summary.append_jsonl(
                    status_path,
                    {
                        "time": summary.now_iso(),
                        "status": "rerun",
                        "reason": f"existing timestamp invalid: {invalid_reason}",
                        "video_name": job.video_name,
                        "timestamp_path": str(paths["timestamp"]),
                    },
                    status_lock,
                )

        timeline_text, asr_count, caption_count, timeline_count = build_timeline_for_job(job, asr_dir, caption_dir)
        if args.dry_run:
            record = {
                "time": summary.now_iso(),
                "status": "dry_run",
                "video_name": job.video_name,
                "duration_sec": duration_sec,
                "segments": len(abstract_data["segment"]),
                "asr_events": asr_count,
                "caption_events": caption_count,
                "timeline_events": timeline_count,
                "timeline_chars": len(timeline_text),
                "timestamp_path": str(paths["timestamp"]),
                "raw_path": str(paths["raw"]),
            }
            summary.append_jsonl(status_path, record, status_lock)
            return record

        payload = request_timestamp(api, abstract_data, timeline_text, duration_sec, args, paths["raw"], rate_limiter)
        write_json_atomic(paths["timestamp"], payload)
        record = {
            "time": summary.now_iso(),
            "status": "ok",
            "video_name": job.video_name,
            "duration_sec": duration_sec,
            "segments": len(abstract_data["segment"]),
            "time_stamp_count": len(payload["time_stamp"]),
            "asr_events": asr_count,
            "caption_events": caption_count,
            "timeline_events": timeline_count,
            "timestamp_path": str(paths["timestamp"]),
            "raw_path": str(paths["raw"]),
        }
        summary.append_jsonl(status_path, record, status_lock)
        return record
    except Exception as exc:
        tb = traceback.format_exc()
        summary.append_error(error_path, f"[{summary.now_iso()}] {job.video_name}\n{tb}", error_lock)
        record = {
            "time": summary.now_iso(),
            "status": "error",
            "video_name": job.video_name,
            "error": str(exc),
            "timestamp_path": str(paths["timestamp"]),
            "raw_path": str(paths["raw"]),
        }
        summary.append_jsonl(status_path, record, status_lock)
        return record


def print_plan(jobs: list[summary.VideoJob], abstract_dir: Path, asr_dir: Path, caption_dir: Path) -> None:
    total_segments = 0
    total_asr = 0
    total_caption = 0
    total_timeline = 0
    total_chars = 0
    missing = 0
    for job in jobs:
        try:
            abstract_data = load_abstract(abstract_dir, job)
            timeline_text, asr_count, caption_count, timeline_count = build_timeline_for_job(job, asr_dir, caption_dir)
        except Exception:
            missing += 1
            continue
        total_segments += len(abstract_data["segment"])
        total_asr += asr_count
        total_caption += caption_count
        total_timeline += timeline_count
        total_chars += len(timeline_text)

    print(f"Videos: {len(jobs)}")
    print(f"Segments needing timestamps: {total_segments}")
    print(f"ASR events: {total_asr}")
    print(f"Caption events: {total_caption}")
    print(f"Timeline events: {total_timeline}")
    print(f"Timeline chars: {total_chars}")
    if missing:
        print(f"Videos with missing/unparseable inputs: {missing}")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    abstract_dir = Path(args.abstract_dir).resolve()
    asr_dir = Path(args.asr_dir).resolve()
    caption_dir = Path(args.caption_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "time_stamps").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    api = summary.parse_api_config(
        args.model,
        args.base_url,
        args.api_key,
        require_credentials=not args.dry_run,
    )
    jobs = summary.load_manifest(manifest_path)
    jobs = summary.filter_jobs(jobs, args.only, args.start_after, args.limit)

    print(f"Model: {api.model}")
    print(f"Base URL: {api.base_url or '(not required for dry run)'}")
    print(f"Output: {output_dir}")
    print(f"Rate limits: rpm={args.rpm}, estimated_tpm={args.tpm}")
    print_plan(jobs, abstract_dir, asr_dir, caption_dir)

    if args.dry_run:
        print("Dry run: no API requests will be sent.")
    if not jobs:
        print("No videos to process.")
        return 0

    status_lock = threading.Lock()
    error_lock = threading.Lock()
    rate_limiter = None if args.dry_run else summary.SlidingWindowRateLimiter(args.rpm, args.tpm)
    workers = max(1, args.workers)
    ok = skipped = dry_run = failed = rerun = 0

    if workers == 1:
        for idx, job in enumerate(jobs, start=1):
            print(f"[{idx}/{len(jobs)}] {job.video_name}")
            record = process_video(
                job,
                api,
                abstract_dir,
                asr_dir,
                caption_dir,
                output_dir,
                args,
                status_lock,
                error_lock,
                rate_limiter,
            )
            status = record["status"]
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            elif status == "dry_run":
                dry_run += 1
            elif status == "rerun":
                rerun += 1
            else:
                failed += 1
            print(f"  {status}: {record.get('reason') or record.get('error') or record.get('time_stamp_count', '')}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(
                    process_video,
                    job,
                    api,
                    abstract_dir,
                    asr_dir,
                    caption_dir,
                    output_dir,
                    args,
                    status_lock,
                    error_lock,
                    rate_limiter,
                ): job
                for job in jobs
            }
            completed = 0
            for future in as_completed(future_to_job):
                completed += 1
                job = future_to_job[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {"status": "error", "error": str(exc), "video_name": job.video_name}
                status = record["status"]
                if status == "ok":
                    ok += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "dry_run":
                    dry_run += 1
                elif status == "rerun":
                    rerun += 1
                else:
                    failed += 1
                print(f"[{completed}/{len(jobs)}] {status}: {job.video_name}")

    print(f"Done. ok={ok}, skipped={skipped}, dry_run={dry_run}, rerun={rerun}, failed={failed}")
    print(f"Status: {output_dir / 'timestamp_status.jsonl'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
