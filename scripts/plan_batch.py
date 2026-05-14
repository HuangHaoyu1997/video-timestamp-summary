from __future__ import annotations

import argparse
import http.client
import json
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import summary_batch as summary


DEFAULT_OUTPUT = Path("plan")
DEFAULT_MIN_DURATION_SEC = 900.0


class PlanValidationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch generate long-video summary planning txt files from ASR and frame captions."
    )
    parser.add_argument("--manifest", default=str(summary.DEFAULT_MANIFEST), help="Frame manifest CSV path.")
    parser.add_argument("--mapping", default=str(summary.DEFAULT_MAPPING), help="Video title/url mapping CSV path.")
    parser.add_argument("--asr-dir", default=str(summary.DEFAULT_ASR_DIR), help="ASR transcript directory.")
    parser.add_argument("--caption-dir", default=str(summary.DEFAULT_CAPTION_DIR), help="Caption txt directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory. Defaults to ./plan.")
    parser.add_argument("--model", default=None, help=f"Model name. Defaults to {summary.DEFAULT_MODEL}.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Falls back to OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY.")
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=DEFAULT_MIN_DURATION_SEC,
        help="Only plan videos longer than this many seconds. Defaults to 900 seconds.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos after filtering.")
    parser.add_argument("--only", action="append", default=[], help="Only process videos whose name contains this text. Repeatable.")
    parser.add_argument("--start-after", default=None, help="Skip videos until this text is found in video name.")
    parser.add_argument("--force", action="store_true", help="Re-run even when final plan txt already exists and is valid.")
    parser.add_argument("--workers", type=int, default=5, help="Number of videos to process concurrently.")
    parser.add_argument("--rpm", type=int, default=60, help="Max API requests per rolling minute. Use 0 to disable.")
    parser.add_argument("--tpm", type=int, default=900000, help="Estimated max tokens per rolling minute. Use 0 to disable.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout per request in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for API and validation failures.")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Initial retry sleep in seconds.")
    parser.add_argument("--max-retry-sleep", type=float, default=60.0, help="Maximum sleep between retries in seconds.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature.")
    parser.add_argument("--max-tokens", type=int, default=6144, help="Max output tokens per video plan.")
    parser.add_argument("--min-plan-chars", type=int, default=120, help="Minimum non-whitespace characters for a valid plan.")
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming responses. Streaming is enabled by default for this text-only endpoint.",
    )
    parser.set_defaults(stream=True)
    parser.add_argument(
        "--max-timeline-chars",
        type=int,
        default=0,
        help="Fail fast when rendered timeline exceeds this many characters. Use 0 to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse inputs and write no model outputs. Useful for checking long-video availability.",
    )
    return parser.parse_args()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp_path.replace(path)


def output_paths(output_dir: Path, job: summary.VideoJob) -> dict[str, Path]:
    return {
        "plan": output_dir / f"{job.stem}.txt",
        "raw": output_dir / "raw" / f"{job.stem}.jsonl",
    }


def validate_plan_text(text: str, min_chars: int) -> str:
    stripped = text.strip()
    if not stripped:
        raise PlanValidationError("规划文本为空。")
    if stripped.startswith("```") or stripped.endswith("```"):
        raise PlanValidationError("规划文本不能包含 Markdown 代码块。")

    compact = "".join(stripped.split())
    if len(compact) < min_chars:
        raise PlanValidationError(f"规划文本过短：当前 {len(compact)} 字符，至少需要 {min_chars} 字符。")

    required_hits = sum(
        1
        for phrase in ("内容主线", "重点保留", "规避", "分段", "低置信度", "摘要模型")
        if phrase in stripped
    )
    if required_hits < 3:
        raise PlanValidationError("规划文本缺少必要板块，请至少覆盖内容主线、重点保留、规避内容、分段安排等信息。")
    return stripped


def existing_plan_is_valid(path: Path, min_chars: int) -> tuple[bool, str | None]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, "plan missing"
    try:
        validate_plan_text(summary.read_text(path), min_chars)
        return True, None
    except Exception as exc:
        return False, str(exc)


def build_timeline_for_job(job: summary.VideoJob, asr_dir: Path, caption_dir: Path) -> tuple[str, int, int, int]:
    events, asr_count, caption_count = summary.build_timeline_for_job(job, asr_dir, caption_dir)
    return summary.render_timeline(events), asr_count, caption_count, len(events)


def build_messages(
    meta: summary.VideoMeta,
    timeline_text: str,
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    expected_video_json = json.dumps(meta.to_json(), ensure_ascii=False, indent=2)
    validation_hint = ""
    if validation_error:
        validation_hint = (
            "\n\n上一次输出没有通过脚本校验，错误如下：\n"
            f"{validation_error}\n"
            "请重新输出规划文本，并修复以上问题。"
        )

    system_prompt = (
        "你是面向视频知识抽取和 RAG 入库的视频摘要规划专家。"
        "你会阅读按时间顺序交错排列的 ASR 和画面 caption，"
        "为后续摘要模型制定长视频内容组织、忽略项和分段策略。"
    )
    user_prompt = (
        "请根据下面的视频元信息和时间线，生成一份长视频摘要规划文本。\n\n"
        "重要说明：\n"
        "1. 这是给后续摘要模型使用的规划，不是最终摘要，不要输出 JSON。\n"
        "2. 不要使用 Markdown 代码块，不要写客套话，不要复述本指令。\n"
        "3. 规划必须基于 ASR 和 CAPTION 中能互相支撑的信息；没有证据的信息不要编造。\n"
        "4. ASR 可能有错字；可结合画面 caption 和上下文纠正明显错听，但低置信度内容要标明谨慎处理或忽略。\n"
        "5. 分段建议必须服务于 RAG 召回：按连续知识单元、主题章节、操作流程、概念关系、产品维度或评测结论组织，不要按镜头、点击或逐秒碎片化。\n"
        "6. 明确指出应忽略的寒暄、点赞关注、重复口播、广告引流、无信息量铺垫、闲聊、画面噪声和无关片段。\n\n"
        "请严格使用以下标题组织输出：\n"
        "# 摘要规划\n"
        "## 内容主线\n"
        "说明最终摘要应该围绕哪些主题或章节展开，哪些内容是视频主干。\n"
        "## 重点保留\n"
        "列出应写入摘要的有效知识，包括实体名称、关键概念、步骤路径、按钮/入口、操作结果、型号/版本、适用场景、限制条件或对比结论。\n"
        "## 规避和忽略\n"
        "列出后续摘要应忽略或弱化的内容，尤其是无知识价值和不适合入库的信息。\n"
        "## 分段安排\n"
        "给出建议的连续知识单元分段；能判断时间范围时写出大致时间范围，不能判断时只写主题顺序；说明哪些相邻内容必须合并。\n"
        "## 风险和低置信度\n"
        "列出 ASR/CAPTION 不一致、识别不清、疑似错听或不应强行下结论的信息。\n"
        "## 给摘要模型的约束\n"
        "用简短条目写清后续摘要生成时必须遵守的组织和取舍规则。\n\n"
        "[VIDEO]\n"
        f"{expected_video_json}\n\n"
        f"{timeline_text}"
        f"{validation_hint}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def request_plan(
    api: summary.ApiConfig,
    meta: summary.VideoMeta,
    timeline_text: str,
    args: argparse.Namespace,
    raw_path: Path,
    rate_limiter: summary.SlidingWindowRateLimiter | None,
) -> str:
    url = summary.chat_completions_url(api.base_url)
    headers = {
        "Authorization": f"Bearer {api.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "User-Agent": summary.DEFAULT_USER_AGENT,
    }

    validation_error: str | None = None
    estimated_tokens = summary.estimate_tokens(timeline_text, args.max_tokens)
    for attempt in range(1, args.retries + 2):
        payload = {
            "model": api.model,
            "messages": build_messages(meta, timeline_text, validation_error),
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
            plan_text = validate_plan_text(content, args.min_plan_chars)
            summary.append_jsonl(
                raw_path,
                {
                    "time": summary.now_iso(),
                    "status": "ok",
                    "attempt": attempt,
                    "video_id": meta.video_id,
                    "title": meta.title,
                    "content": content,
                    "response": response_data,
                },
            )
            return plan_text
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            summary.append_jsonl(
                raw_path,
                {
                    "time": summary.now_iso(),
                    "status": "error",
                    "attempt": attempt,
                    "video_id": meta.video_id,
                    "title": meta.title,
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
            ValueError,
        ) as exc:
            summary.append_jsonl(
                raw_path,
                {
                    "time": summary.now_iso(),
                    "status": "error",
                    "attempt": attempt,
                    "video_id": meta.video_id,
                    "title": meta.title,
                    "error_type": "transport_or_response",
                    "error": str(exc),
                },
            )
            if attempt > args.retries:
                raise RuntimeError(str(exc)) from exc
            time.sleep(summary.retry_sleep_seconds(args, attempt))
        except PlanValidationError as exc:
            validation_error = str(exc)
            summary.append_jsonl(
                raw_path,
                {
                    "time": summary.now_iso(),
                    "status": "error",
                    "attempt": attempt,
                    "video_id": meta.video_id,
                    "title": meta.title,
                    "error_type": "validation",
                    "error": validation_error,
                },
            )
            if attempt > args.retries:
                raise RuntimeError(validation_error) from exc
            time.sleep(summary.retry_sleep_seconds(args, attempt))

    raise RuntimeError("unreachable retry loop exit")


def process_video(
    job: summary.VideoJob,
    api: summary.ApiConfig,
    mapping: dict[str, dict[str, str]],
    asr_dir: Path,
    caption_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    status_lock: threading.Lock,
    error_lock: threading.Lock,
    rate_limiter: summary.SlidingWindowRateLimiter | None,
) -> dict[str, Any]:
    paths = output_paths(output_dir, job)
    status_path = output_dir / "plan_status.jsonl"
    error_path = output_dir / "plan_errors.log"
    meta = summary.build_video_meta(job, mapping)

    if args.force:
        for path in paths.values():
            if path.exists():
                path.unlink()
    else:
        valid, invalid_reason = existing_plan_is_valid(paths["plan"], args.min_plan_chars)
        if valid:
            record = {
                "time": summary.now_iso(),
                "status": "skipped",
                "reason": "plan exists",
                "video_name": job.video_name,
                "plan_path": str(paths["plan"]),
                "raw_path": str(paths["raw"]),
            }
            summary.append_jsonl(status_path, record, status_lock)
            return record
        if paths["plan"].exists():
            summary.append_jsonl(
                status_path,
                {
                    "time": summary.now_iso(),
                    "status": "rerun",
                    "reason": f"existing plan invalid: {invalid_reason}",
                    "video_name": job.video_name,
                    "plan_path": str(paths["plan"]),
                },
                status_lock,
            )

    try:
        timeline_text, asr_count, caption_count, timeline_count = build_timeline_for_job(job, asr_dir, caption_dir)
        if args.max_timeline_chars and len(timeline_text) > args.max_timeline_chars:
            raise RuntimeError(
                f"Timeline too long: {len(timeline_text)} chars > {args.max_timeline_chars}. "
                "Increase --max-timeline-chars or add timeline chunking before planning."
            )

        if args.dry_run:
            record = {
                "time": summary.now_iso(),
                "status": "dry_run",
                "video_name": job.video_name,
                "video_id": meta.video_id,
                "duration_sec": meta.duration_sec,
                "asr_events": asr_count,
                "caption_events": caption_count,
                "timeline_events": timeline_count,
                "timeline_chars": len(timeline_text),
                "plan_path": str(paths["plan"]),
                "raw_path": str(paths["raw"]),
            }
            summary.append_jsonl(status_path, record, status_lock)
            return record

        plan_text = request_plan(api, meta, timeline_text, args, paths["raw"], rate_limiter)
        write_text_atomic(paths["plan"], plan_text)
        record = {
            "time": summary.now_iso(),
            "status": "ok",
            "video_name": job.video_name,
            "video_id": meta.video_id,
            "duration_sec": meta.duration_sec,
            "asr_events": asr_count,
            "caption_events": caption_count,
            "timeline_events": timeline_count,
            "timeline_chars": len(timeline_text),
            "plan_chars": len(plan_text),
            "plan_path": str(paths["plan"]),
            "raw_path": str(paths["raw"]),
        }
        summary.append_jsonl(status_path, record, status_lock)
        return record
    except Exception as exc:
        tb = traceback.format_exc()
        summary.append_error(
            error_path,
            f"[{summary.now_iso()}] {job.video_name}\n{tb}",
            error_lock,
        )
        record = {
            "time": summary.now_iso(),
            "status": "error",
            "video_name": job.video_name,
            "error": str(exc),
            "plan_path": str(paths["plan"]),
            "raw_path": str(paths["raw"]),
        }
        summary.append_jsonl(status_path, record, status_lock)
        return record


def print_plan_stats(jobs: list[summary.VideoJob], asr_dir: Path, caption_dir: Path, output_dir: Path, min_plan_chars: int) -> None:
    total_asr = 0
    total_caption = 0
    total_timeline = 0
    total_chars = 0
    missing = 0
    existing = 0

    for job in jobs:
        try:
            timeline_text, asr_count, caption_count, timeline_count = build_timeline_for_job(job, asr_dir, caption_dir)
        except Exception:
            missing += 1
            continue
        total_asr += asr_count
        total_caption += caption_count
        total_timeline += timeline_count
        total_chars += len(timeline_text)
        valid, _ = existing_plan_is_valid(output_paths(output_dir, job)["plan"], min_plan_chars)
        if valid:
            existing += 1

    print(f"Long videos: {len(jobs)}")
    print(f"Existing valid plans: {existing}")
    print(f"ASR events: {total_asr}")
    print(f"Caption events: {total_caption}")
    print(f"Timeline events: {total_timeline}")
    print(f"Timeline chars: {total_chars}")
    if missing:
        print(f"Videos with missing/unparseable inputs: {missing}")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    mapping_path = Path(args.mapping).resolve()
    asr_dir = Path(args.asr_dir).resolve()
    caption_dir = Path(args.caption_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    api = summary.parse_api_config(
        args.model,
        args.base_url,
        args.api_key,
        require_credentials=not args.dry_run,
    )
    mapping = summary.load_video_mapping(mapping_path)
    all_jobs = summary.load_manifest(manifest_path)
    filtered_jobs = summary.filter_jobs(all_jobs, args.only, args.start_after, None)
    long_jobs = [job for job in filtered_jobs if job.duration_sec > args.min_duration_sec]
    short_count = len(filtered_jobs) - len(long_jobs)
    jobs = long_jobs[: max(0, args.limit)] if args.limit is not None else long_jobs

    print(f"Model: {api.model}")
    print(f"Base URL: {api.base_url or '(not required for dry run)'}")
    print(f"Output: {output_dir}")
    print(f"Long-video threshold: duration_sec > {args.min_duration_sec:g}")
    print(f"Filtered out by duration: {short_count}")
    if args.limit is not None:
        print(f"Limit after duration filter: {args.limit}")
    print(f"Rate limits: rpm={args.rpm}, estimated_tpm={args.tpm}")
    print_plan_stats(jobs, asr_dir, caption_dir, output_dir, args.min_plan_chars)

    if args.dry_run:
        print("Dry run: no API requests will be sent.")

    if not jobs:
        print("No long videos to process.")
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
                mapping,
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
            print(f"  {status}: {record.get('reason') or record.get('error') or record.get('plan_chars', '')}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(
                    process_video,
                    job,
                    api,
                    mapping,
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
    print(f"Status: {output_dir / 'plan_status.jsonl'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
