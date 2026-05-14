from __future__ import annotations

import argparse
import json
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import caption_batch as caption
import summary_batch as summary


DEFAULT_OUTPUT = Path("prompt_profiles")
DEFAULT_FRAME_DIR = Path("frame")


class ProfileValidationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reusable topic prompt profiles for a batch of videos."
    )
    parser.add_argument("--manifest", default=str(summary.DEFAULT_MANIFEST), help="Frame manifest CSV path.")
    parser.add_argument("--mapping", default=str(summary.DEFAULT_MAPPING), help="Video title/url mapping CSV path.")
    parser.add_argument("--asr-dir", default=str(summary.DEFAULT_ASR_DIR), help="ASR transcript directory.")
    parser.add_argument("--frame-dir", default=str(DEFAULT_FRAME_DIR), help="Extracted frame root directory.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory. Defaults to ./prompt_profiles.")
    parser.add_argument("--model", default=None, help=f"Model name. Defaults to {summary.DEFAULT_MODEL}.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Falls back to OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY.")
    parser.add_argument("--limit", type=int, default=None, help="Consider at most N videos after filtering.")
    parser.add_argument("--only", action="append", default=[], help="Only consider videos whose name contains this text. Repeatable.")
    parser.add_argument("--start-after", default=None, help="Skip videos until this text is found in video name.")
    parser.add_argument("--sample-size", type=int, default=24, help="Max videos used to generate batch profiles.")
    parser.add_argument("--frames-per-video", type=int, default=3, help="Representative frames per sampled video.")
    parser.add_argument("--max-asr-chars", type=int, default=900, help="Max ASR evidence chars per sampled video.")
    parser.add_argument("--max-profiles", type=int, default=6, help="Maximum topic profiles to request.")
    parser.add_argument("--force", action="store_true", help="Regenerate even when profiles.json already exists.")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future assignment work; generation is one batch call.")
    parser.add_argument("--rpm", type=int, default=30, help="Max API requests per rolling minute. Use 0 to disable.")
    parser.add_argument("--tpm", type=int, default=600000, help="Estimated max tokens per rolling minute. Use 0 to disable.")
    parser.add_argument("--timeout", type=float, default=240.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for API and validation failures.")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Initial retry sleep in seconds.")
    parser.add_argument("--max-retry-sleep", type=float, default=60.0, help="Maximum sleep between retries in seconds.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens for profile generation.")
    parser.add_argument(
        "--image-detail",
        choices=("none", "auto", "low", "high"),
        default="low",
        help="Optional image detail field for compatible APIs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected samples and write no outputs.")
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


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def normalize_profile_id(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "general_video"


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "profiles": output_dir / "profiles.json",
        "assignments": output_dir / "video_assignments.jsonl",
        "status": output_dir / "profile_status.jsonl",
        "error": output_dir / "profile_errors.log",
        "raw": output_dir / "raw" / "profile_generation.jsonl",
    }


def filter_jobs(jobs: list[summary.VideoJob], only: list[str], start_after: str | None, limit: int | None) -> list[summary.VideoJob]:
    filtered = summary.filter_jobs(jobs, only, start_after, None)
    if limit is not None:
        filtered = filtered[: max(0, limit)]
    return filtered


def select_sample_jobs(jobs: list[summary.VideoJob], sample_size: int) -> list[summary.VideoJob]:
    if sample_size <= 0 or len(jobs) <= sample_size:
        return jobs
    selected: list[summary.VideoJob] = []
    seen: set[str] = set()
    sorted_by_duration = sorted(jobs, key=lambda item: item.duration_sec, reverse=True)
    for job in sorted_by_duration[: max(1, sample_size // 3)]:
        selected.append(job)
        seen.add(job.stem)

    remaining_slots = sample_size - len(selected)
    if remaining_slots <= 0:
        return selected[:sample_size]
    step = max(1, len(jobs) // remaining_slots)
    for idx in range(0, len(jobs), step):
        job = jobs[idx]
        if job.stem not in seen:
            selected.append(job)
            seen.add(job.stem)
        if len(selected) >= sample_size:
            break
    return selected


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 3]
    mid_start = max(0, len(text) // 2 - max_chars // 6)
    mid = text[mid_start : mid_start + max_chars // 3]
    tail = text[-max_chars // 3 :]
    return f"{head} ... {mid} ... {tail}"


def asr_evidence(job: summary.VideoJob, asr_dir: Path, max_chars: int) -> str:
    events = summary.parse_srt(asr_dir / f"{job.stem}.srt")
    lines = [f"[{summary.format_timestamp(int(event.start_sec))}-{summary.format_timestamp(int(event.end_sec))}] {event.text}" for event in events]
    return compact_text("\n".join(lines), max_chars)


def representative_frames(job: summary.VideoJob, frame_root: Path, count: int) -> list[caption.FrameItem]:
    frame_dir = frame_root / job.stem
    frames = caption.scan_frames(frame_dir)
    if count <= 0 or len(frames) <= count:
        return frames
    indexes = sorted({round(i * (len(frames) - 1) / max(1, count - 1)) for i in range(count)})
    return [frames[idx] for idx in indexes]


def build_messages(
    sample_jobs: list[summary.VideoJob],
    mapping: dict[str, summary.VideoMeta],
    asr_dir: Path,
    frame_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "你正在为一批视频生成可复用的 prompt profiles。"
                "目标不是为每个视频单独写 prompt，而是识别少量 topic / 内容类型，"
                "给后续 caption 和 abstract 阶段复用。\n\n"
                "请根据每个样本视频的标题、时长、ASR 摘要片段和代表性帧，输出严格 JSON。\n"
                f"最多生成 {args.max_profiles} 个 profiles；如果视频都属于同类，可以只生成 1 个。\n"
                "profile 应可复用到同一批中多个视频，避免过细、过拟合到单条视频。\n\n"
                "每个 profile 需要包含：\n"
                "- profile_id: 英文小写、数字、下划线，稳定可读。\n"
                "- name: 中文短名。\n"
                "- topic: 内容类型说明。\n"
                "- keywords: 用于后续自动分配的关键词数组。\n"
                "- caption_prompt: 给画面 caption 阶段的领域提示，说明视觉上重点看什么、忽略什么。\n"
                "- summary_prompt: 给 abstract 阶段的领域提示，说明摘要重点保留什么、如何分段、忽略什么。\n"
                "- assignment_rule: 哪类视频适合使用该 profile。\n\n"
                "输出格式：\n"
                "{\n"
                "  \"default_profile_id\": \"...\",\n"
                "  \"profiles\": [\n"
                "    {\"profile_id\":\"...\",\"name\":\"...\",\"topic\":\"...\",\"keywords\":[\"...\"],\"caption_prompt\":\"...\",\"summary_prompt\":\"...\",\"assignment_rule\":\"...\"}\n"
                "  ],\n"
                "  \"sample_assignments\": [\n"
                "    {\"video_stem\":\"...\",\"profile_id\":\"...\",\"confidence\":0.8,\"reason\":\"...\"}\n"
                "  ]\n"
                "}\n\n"
                "只输出 JSON，不要 Markdown 代码块，不要解释。"
            ),
        }
    ]

    for idx, job in enumerate(sample_jobs, start=1):
        meta = summary.build_video_meta(job, mapping)
        content.append(
            {
                "type": "text",
                "text": (
                    f"\n[SAMPLE_VIDEO {idx}]\n"
                    f"video_stem: {job.stem}\n"
                    f"title: {meta.title}\n"
                    f"duration_sec: {meta.duration_sec}\n"
                    f"platform: {meta.platform}\n"
                    f"ASR_EXCERPT:\n{asr_evidence(job, asr_dir, args.max_asr_chars)}\n"
                    "REPRESENTATIVE_FRAMES:\n"
                ),
            }
        )
        for ordinal, frame in enumerate(representative_frames(job, frame_root, args.frames_per_video), start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"样本视频 {idx} 的第 {ordinal} 张代表帧，时间约 {caption.format_timestamp(frame.timestamp_sec)}：",
                }
            )
            image_url: dict[str, Any] = {"url": caption.image_data_url(frame.path)}
            if args.image_detail != "none":
                image_url["detail"] = args.image_detail
            content.append({"type": "image_url", "image_url": image_url})

    return [
        {"role": "system", "content": "你是视频批处理工作流的 topic 与提示词 profile 设计专家。"},
        {"role": "user", "content": content},
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ProfileValidationError("Profile response root must be object.")
    return data


def validate_profiles(data: dict[str, Any]) -> dict[str, Any]:
    profiles = data.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ProfileValidationError("profiles must be a non-empty array.")

    normalized_profiles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            raise ProfileValidationError(f"profiles[{idx}] must be object.")
        profile_id = normalize_profile_id(str(profile.get("profile_id") or profile.get("name") or f"profile_{idx+1}"))
        if profile_id in seen:
            raise ProfileValidationError(f"Duplicate profile_id: {profile_id}")
        seen.add(profile_id)
        name = str(profile.get("name") or profile_id).strip()
        topic = str(profile.get("topic") or "").strip()
        caption_prompt = str(profile.get("caption_prompt") or "").strip()
        summary_prompt = str(profile.get("summary_prompt") or "").strip()
        assignment_rule = str(profile.get("assignment_rule") or "").strip()
        keywords = profile.get("keywords")
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(item).strip() for item in keywords if str(item).strip()]
        if not caption_prompt or not summary_prompt:
            raise ProfileValidationError(f"profile {profile_id} missing caption_prompt or summary_prompt.")
        normalized_profiles.append(
            {
                "profile_id": profile_id,
                "name": name,
                "topic": topic,
                "keywords": keywords,
                "caption_prompt": caption_prompt,
                "summary_prompt": summary_prompt,
                "assignment_rule": assignment_rule,
            }
        )

    default_profile_id = normalize_profile_id(str(data.get("default_profile_id") or normalized_profiles[0]["profile_id"]))
    if default_profile_id not in {profile["profile_id"] for profile in normalized_profiles}:
        default_profile_id = normalized_profiles[0]["profile_id"]

    sample_assignments = data.get("sample_assignments")
    if not isinstance(sample_assignments, list):
        sample_assignments = []

    return {
        "version": 1,
        "created_at": summary.now_iso(),
        "default_profile_id": default_profile_id,
        "profiles": normalized_profiles,
        "sample_assignments": sample_assignments,
    }


def request_profiles(
    api: summary.ApiConfig,
    sample_jobs: list[summary.VideoJob],
    mapping: dict[str, summary.VideoMeta],
    asr_dir: Path,
    frame_root: Path,
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
    estimated_tokens = max(1, len(sample_jobs) * (args.max_asr_chars // 2 + args.frames_per_video * 900) + args.max_tokens)

    for attempt in range(1, args.retries + 2):
        payload = {
            "model": api.model,
            "messages": build_messages(sample_jobs, mapping, asr_dir, frame_root, args),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if rate_limiter is not None:
            rate_limiter.acquire(estimated_tokens)
        req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                response_text = resp.read().decode("utf-8")
            response_data = json.loads(response_text)
            content = summary.parse_chat_content(response_data)
            parsed = validate_profiles(parse_json_object(content))
            summary.append_jsonl(raw_path, {"time": summary.now_iso(), "status": "ok", "attempt": attempt, "content": content, "response": response_data})
            return parsed
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            summary.append_jsonl(raw_path, {"time": summary.now_iso(), "status": "error", "attempt": attempt, "error": f"HTTP {exc.code}: {response_body[:1000]}"})
            if attempt > args.retries or not summary.should_retry_http(exc.code):
                raise RuntimeError(f"HTTP {exc.code}: {response_body[:1000]}") from exc
            time.sleep(summary.retry_sleep_seconds(args, attempt, exc.headers))
        except Exception as exc:
            summary.append_jsonl(raw_path, {"time": summary.now_iso(), "status": "error", "attempt": attempt, "error": str(exc)})
            if attempt > args.retries:
                raise
            time.sleep(summary.retry_sleep_seconds(args, attempt))
    raise RuntimeError("Profile generation failed.")


def token_set(text: str) -> set[str]:
    lowered = (text or "").lower()
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", lowered)
    return set(parts)


def assign_profile(job: summary.VideoJob, meta: summary.VideoMeta, profiles_data: dict[str, Any], sample_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if job.stem in sample_map:
        profile_id = normalize_profile_id(str(sample_map[job.stem].get("profile_id") or ""))
        if any(profile["profile_id"] == profile_id for profile in profiles_data["profiles"]):
            return {
                "video_name": job.video_name,
                "video_stem": job.stem,
                "profile_id": profile_id,
                "confidence": sample_map[job.stem].get("confidence", 0.8),
                "reason": sample_map[job.stem].get("reason", "sample assignment"),
                "assignment_source": "model_sample",
            }

    text = f"{meta.title} {job.video_name}"
    text_tokens = token_set(text)
    best_profile = None
    best_score = 0
    for profile in profiles_data["profiles"]:
        keywords = set(str(item).lower() for item in profile.get("keywords", []))
        profile_tokens = token_set(" ".join([profile.get("name", ""), profile.get("topic", ""), profile.get("assignment_rule", ""), " ".join(profile.get("keywords", []))]))
        score = len(text_tokens & (keywords | profile_tokens))
        if score > best_score:
            best_score = score
            best_profile = profile
    if best_profile is None:
        profile_id = profiles_data["default_profile_id"]
        confidence = 0.35
        reason = "fallback default profile"
    else:
        profile_id = best_profile["profile_id"]
        confidence = min(0.75, 0.45 + best_score * 0.08)
        reason = f"keyword/rule overlap score={best_score}"
    return {
        "video_name": job.video_name,
        "video_stem": job.stem,
        "profile_id": profile_id,
        "confidence": round(float(confidence), 3),
        "reason": reason,
        "assignment_source": "rule",
    }


def build_assignments(jobs: list[summary.VideoJob], mapping: dict[str, summary.VideoMeta], profiles_data: dict[str, Any]) -> list[dict[str, Any]]:
    sample_map = {
        str(item.get("video_stem")): item
        for item in profiles_data.get("sample_assignments", [])
        if isinstance(item, dict) and item.get("video_stem")
    }
    records = []
    for job in jobs:
        meta = summary.build_video_meta(job, mapping)
        records.append(assign_profile(job, meta, profiles_data, sample_map))
    return records


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    mapping_path = Path(args.mapping).resolve()
    asr_dir = Path(args.asr_dir).resolve()
    frame_root = Path(args.frame_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)

    if paths["profiles"].exists() and paths["assignments"].exists() and not args.force and not args.dry_run:
        print(f"Profiles already exist: {paths['profiles']}")
        print("Use --force to regenerate.")
        return 0

    api = summary.parse_api_config(
        args.model,
        args.base_url,
        args.api_key,
        require_credentials=not args.dry_run,
    )
    mapping = summary.load_video_mapping(mapping_path)
    jobs = filter_jobs(summary.load_manifest(manifest_path), args.only, args.start_after, args.limit)
    sample_jobs = select_sample_jobs(jobs, args.sample_size)

    print(f"Model: {api.model}")
    print(f"Base URL: {api.base_url or '(not required for dry run)'}")
    print(f"Videos considered: {len(jobs)}")
    print(f"Sample videos for profile generation: {len(sample_jobs)}")
    print(f"Output: {output_dir}")

    if args.dry_run:
        for job in sample_jobs:
            print(f"- {job.video_name} ({job.duration_sec:.1f}s)")
        return 0

    status_lock = threading.Lock()
    rate_limiter = summary.SlidingWindowRateLimiter(args.rpm, args.tpm)
    try:
        profiles_data = request_profiles(api, sample_jobs, mapping, asr_dir, frame_root, args, paths["raw"], rate_limiter)
        assignments = build_assignments(jobs, mapping, profiles_data)
        write_json_atomic(paths["profiles"], profiles_data)
        write_jsonl_atomic(paths["assignments"], assignments)
        summary.append_jsonl(
            paths["status"],
            {
                "time": summary.now_iso(),
                "status": "ok",
                "profiles": len(profiles_data["profiles"]),
                "assignments": len(assignments),
                "profiles_path": str(paths["profiles"]),
                "assignments_path": str(paths["assignments"]),
            },
            status_lock,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        summary.append_error(paths["error"], f"[{summary.now_iso()}]\n{tb}")
        summary.append_jsonl(paths["status"], {"time": summary.now_iso(), "status": "error", "error": str(exc)}, status_lock)
        raise

    print(f"Profiles: {paths['profiles']}")
    print(f"Assignments: {paths['assignments']}")
    print(f"Status: {paths['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
