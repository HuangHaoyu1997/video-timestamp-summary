from __future__ import annotations

import argparse
import base64
import csv
import http.client
import json
import mimetypes
import os
import random
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("frame_manifest.csv")
DEFAULT_PROFILE_DIR = Path("prompt_profiles")
DEFAULT_OUTPUT = Path("caption")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

FRAME_NAME_RE = re.compile(r"^frame_(\d{6})_every(\d{3})s\.jpe?g$", re.IGNORECASE)
LEADING_TIME_RANGE_RE = re.compile(r"^\s*\[[^\]]{3,40}\]\s*")


@dataclass(frozen=True)
class ApiConfig:
    model: str
    base_url: str
    api_key: str


class SlidingWindowRateLimiter:
    def __init__(self, rpm: int | None, tpm: int | None, window_sec: float = 60.0) -> None:
        self.rpm = max(0, int(rpm or 0))
        self.tpm = max(0, int(tpm or 0))
        self.window_sec = window_sec
        self.events: deque[tuple[float, int]] = deque()
        self.lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.rpm or self.tpm)

    def acquire(self, estimated_tokens: int) -> None:
        if not self.enabled:
            return

        token_cost = max(1, int(estimated_tokens))
        if self.tpm:
            token_cost = min(token_cost, self.tpm)

        while True:
            wait_sec = 0.0
            with self.lock:
                now = time.monotonic()
                self._prune(now)
                used_tokens = sum(tokens for _, tokens in self.events)
                rpm_ok = not self.rpm or len(self.events) < self.rpm
                tpm_ok = not self.tpm or used_tokens + token_cost <= self.tpm

                if rpm_ok and tpm_ok:
                    self.events.append((now, token_cost))
                    return

                waits: list[float] = []
                if self.events:
                    if self.rpm and len(self.events) >= self.rpm:
                        waits.append(self.window_sec - (now - self.events[0][0]) + 0.05)
                    if self.tpm and used_tokens + token_cost > self.tpm:
                        remaining_tokens = used_tokens
                        for timestamp, tokens in self.events:
                            remaining_tokens -= tokens
                            if remaining_tokens + token_cost <= self.tpm:
                                waits.append(self.window_sec - (now - timestamp) + 0.05)
                                break
                        else:
                            waits.append(self.window_sec - (now - self.events[0][0]) + 0.05)
                wait_sec = max(0.1, min(waits) if waits else 0.5)
            time.sleep(wait_sec)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self.events and self.events[0][0] <= cutoff:
            self.events.popleft()


@dataclass(frozen=True)
class FrameItem:
    path: Path
    index: int
    interval_sec: int
    timestamp_sec: int


@dataclass(frozen=True)
class FrameGroup:
    group_index: int
    frames: list[FrameItem]
    start_sec: int
    end_sec: int


@dataclass(frozen=True)
class VideoJob:
    order: int
    video_name: str
    frame_dir: Path
    duration_sec: float
    interval_sec: int
    frame_count: int

    @property
    def stem(self) -> str:
        return Path(self.video_name).stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch caption extracted video frames with an OpenAI-compatible VLM."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Frame manifest CSV path.")
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR), help="Optional prompt profile directory.")
    parser.add_argument("--ignore-profile", action="store_true", help="Ignore prompt profiles even when available.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory. Defaults to ./caption.")
    parser.add_argument("--model", default=None, help=f"Model name. Defaults to {DEFAULT_MODEL}.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Falls back to OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos after filtering.")
    parser.add_argument("--only", action="append", default=[], help="Only process videos whose name contains this text. Repeatable.")
    parser.add_argument("--start-after", default=None, help="Skip videos until this text is found in video name.")
    parser.add_argument("--force", action="store_true", help="Re-run even when final caption txt already exists.")
    parser.add_argument("--workers", type=int, default=1, help="Number of videos to process concurrently.")
    parser.add_argument("--rpm", type=int, default=60, help="Max API requests per rolling minute. Use 0 to disable.")
    parser.add_argument("--tpm", type=int, default=300000, help="Estimated max tokens per rolling minute. Use 0 to disable.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per request in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per frame group.")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Initial retry sleep in seconds.")
    parser.add_argument("--max-retry-sleep", type=float, default=60.0, help="Maximum sleep between retries in seconds.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature.")
    parser.add_argument("--max-tokens", type=int, default=700, help="Max output tokens per frame group.")
    parser.add_argument(
        "--image-detail",
        choices=("none", "auto", "low", "high"),
        default="none",
        help="Optional image detail field for compatible APIs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan groups and write no model outputs. Useful for checking filtering and request counts.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def parse_api_config(
    model_override: str | None,
    base_url_override: str | None,
    api_key_override: str | None,
    require_credentials: bool = True,
) -> ApiConfig:
    model = (
        model_override
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_MODEL
    )
    base_url = (
        base_url_override
        or os.environ.get("OPENAI_BASE_URL")
    )
    api_key = (
        api_key_override
        or os.environ.get("OPENAI_API_KEY")
    )

    missing = []
    if not base_url:
        missing.append("base URL (--base-url or OPENAI_BASE_URL)")
    if not api_key:
        missing.append("API key (--api-key or OPENAI_API_KEY)")
    if missing and require_credentials:
        raise ValueError("Missing API config: " + ", ".join(missing))

    return ApiConfig(model=model, base_url=(base_url or "").rstrip("/"), api_key=api_key or "")


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    if lock is None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def append_error(path: Path, text: str, lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lock is None:
        with path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
        return
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")


def load_manifest(path: Path) -> list[VideoJob]:
    if not path.exists():
        raise FileNotFoundError(f"Frame manifest not found: {path}")

    jobs: list[VideoJob] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for order, row in enumerate(reader, start=1):
            if row.get("status") != "ok":
                continue
            video_name = (row.get("video_name") or "").strip()
            frame_dir = Path((row.get("frame_dir") or "").strip())
            if not video_name or not str(frame_dir):
                continue
            jobs.append(
                VideoJob(
                    order=order,
                    video_name=video_name,
                    frame_dir=frame_dir,
                    duration_sec=float(row.get("duration_sec") or 0),
                    interval_sec=int(float(row.get("interval_sec") or 0)),
                    frame_count=int(float(row.get("frame_count") or 0)),
                )
            )
    return jobs


def filter_jobs(jobs: list[VideoJob], only: list[str], start_after: str | None, limit: int | None) -> list[VideoJob]:
    filtered = jobs
    if start_after:
        lowered = start_after.lower()
        start_index = None
        for idx, job in enumerate(filtered):
            if lowered in job.video_name.lower() or lowered in job.stem.lower():
                start_index = idx + 1
                break
        if start_index is not None:
            filtered = filtered[start_index:]
    for needle in only:
        lowered = needle.lower()
        filtered = [job for job in filtered if lowered in job.video_name.lower() or lowered in job.stem.lower()]
    if limit is not None:
        filtered = filtered[: max(0, limit)]
    return filtered


def group_size_for_duration(duration_sec: float) -> int:
    if duration_sec <= 60:
        return 5
    if duration_sec <= 300:
        return 6
    if duration_sec <= 480:
        return 8
    return 10


def format_timestamp(total_sec: int | float) -> str:
    sec = int(round(total_sec))
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    seconds = sec % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def scan_frames(frame_dir: Path) -> list[FrameItem]:
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    frames: list[FrameItem] = []
    for path in frame_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_NAME_RE.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        interval_sec = int(match.group(2))
        frames.append(
            FrameItem(
                path=path,
                index=index,
                interval_sec=interval_sec,
                timestamp_sec=(index - 1) * interval_sec,
            )
        )
    frames.sort(key=lambda item: item.index)
    return frames


def build_groups(frames: list[FrameItem], duration_sec: float) -> list[FrameGroup]:
    size = group_size_for_duration(duration_sec)
    groups: list[FrameGroup] = []
    for offset in range(0, len(frames), size):
        chunk = frames[offset : offset + size]
        groups.append(
            FrameGroup(
                group_index=len(groups) + 1,
                frames=chunk,
                start_sec=chunk[0].timestamp_sec,
                end_sec=chunk[-1].timestamp_sec,
            )
        )
    return groups


def estimate_request_tokens(group: FrameGroup, max_tokens: int) -> int:
    prompt_tokens_by_image_count = {
        1: 1800,
        2: 2400,
        3: 3000,
        4: 3600,
        5: 4100,
        6: 6400,
        7: 8500,
        8: 10500,
        9: 13200,
        10: 16200,
    }
    image_count = len(group.frames)
    prompt_tokens = prompt_tokens_by_image_count.get(image_count, 1800 * image_count)
    return prompt_tokens + max(0, int(max_tokens)) + 300


def mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "image/jpeg"


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type(path)};base64,{encoded}"


def load_prompt_profiles(profile_dir: Path, ignore_profile: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    if ignore_profile:
        return {}, {}
    profiles_path = profile_dir / "profiles.json"
    assignments_path = profile_dir / "video_assignments.jsonl"
    if not profiles_path.exists() or not assignments_path.exists():
        return {}, {}
    try:
        data = json.loads(read_text(profiles_path))
        profiles = {
            str(profile.get("profile_id")): profile
            for profile in data.get("profiles", [])
            if isinstance(profile, dict) and profile.get("profile_id")
        }
        assignments: dict[str, str] = {}
        with assignments_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                stem = record.get("video_stem")
                profile_id = record.get("profile_id")
                if stem and profile_id:
                    assignments[str(stem)] = str(profile_id)
        return profiles, assignments
    except Exception as exc:
        raise RuntimeError(f"Failed to load prompt profiles from {profile_dir}: {exc}") from exc


def profile_for_job(job: VideoJob, profiles: dict[str, Any], assignments: dict[str, str]) -> dict[str, Any] | None:
    profile_id = assignments.get(job.stem)
    if not profile_id:
        return None
    return profiles.get(profile_id)


def profile_block(profile: dict[str, Any] | None, prompt_key: str) -> str:
    if not profile:
        return ""
    lines = [
        "[PROMPT_PROFILE]",
        f"profile_id: {profile.get('profile_id', '')}",
        f"name: {profile.get('name', '')}",
        f"topic: {profile.get('topic', '')}",
        str(profile.get(prompt_key) or "").strip(),
    ]
    return "\n".join(line for line in lines if str(line).strip())


def build_messages(video_name: str, group: FrameGroup, image_detail: str, profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    system_prompt = (
        "你是面向视频知识抽取的视觉字幕助手。"
        "你的任务是根据连续视频帧生成准确、可检索的中文画面 caption。"
    )
    profile_guidance = profile_block(profile, "caption_prompt")
    start = format_timestamp(group.start_sec)
    end = format_timestamp(group.end_sec)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"以下图片来自同一个视频，按时间顺序排列。\n"
                f"视频名：{video_name}\n"
                f"本组覆盖时间：{start}-{end}\n\n"
                f"{profile_guidance}\n\n"
                "请只根据画面写一段中文 caption，用于后续视频内容摘要。要求：\n"
                "1. 优先描述画面中可检索、可复用的实体、页面、操作、场景、文字、参数、结果或结论。\n"
                "2. 描述用户正在进行的可见变化，例如进入页面、点击入口、切换状态、展示效果、对比对象或关键画面转场。\n"
                "3. 根据 [PROMPT_PROFILE] 调整领域重点；没有 profile 时保持通用客观描述。\n"
                "4. 看不清的文字写“文字不清”或“疑似”，不要编造画面外信息。\n"
                "5. 输出一段 80-180 字中文自然语言；不要分点；不要重复输出时间区间。"
            ),
        }
    ]
    for ordinal, frame in enumerate(group.frames, start=1):
        content.append(
            {
                "type": "text",
                "text": f"第 {ordinal} 张，视频时间约 {format_timestamp(frame.timestamp_sec)}：",
            }
        )
        image_url: dict[str, Any] = {"url": image_data_url(frame.path)}
        if image_detail != "none":
            image_url["detail"] = image_detail
        content.append({"type": "image_url", "image_url": image_url})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_chat_content(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat response shape: {json.dumps(data, ensure_ascii=False)[:1000]}") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def normalize_caption(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    stripped = LEADING_TIME_RANGE_RE.sub("", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped


def should_retry_http(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or status_code >= 500


def retry_after_seconds(headers: Any) -> float | None:
    value = headers.get("Retry-After") if headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def retry_sleep_seconds(args: argparse.Namespace, attempt: int, headers: Any = None) -> float:
    retry_after = retry_after_seconds(headers)
    if retry_after is not None:
        return min(args.max_retry_sleep, retry_after)
    base = args.retry_sleep * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, min(1.0, base * 0.15))
    return min(args.max_retry_sleep, base + jitter)


def request_caption(
    api: ApiConfig,
    video_name: str,
    group: FrameGroup,
    args: argparse.Namespace,
    rate_limiter: SlidingWindowRateLimiter | None,
    profile: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    url = chat_completions_url(api.base_url)
    payload = {
        "model": api.model,
        "messages": build_messages(video_name, group, args.image_detail, profile),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "User-Agent": DEFAULT_USER_AGENT,
    }

    attempt = 0
    estimated_tokens = estimate_request_tokens(group, args.max_tokens)
    while True:
        attempt += 1
        if rate_limiter is not None:
            rate_limiter.acquire(estimated_tokens)
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                response_text = resp.read().decode("utf-8")
            response_data = json.loads(response_text)
            caption = normalize_caption(parse_chat_content(response_data))
            if not caption:
                raise ValueError("Model returned empty caption")
            return caption, response_data
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            if attempt > args.retries or not should_retry_http(exc.code):
                raise RuntimeError(f"HTTP {exc.code}: {response_body[:1000]}") from exc
            sleep_sec = retry_sleep_seconds(args, attempt, exc.headers)
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
            if attempt > args.retries:
                raise RuntimeError(str(exc)) from exc
            sleep_sec = retry_sleep_seconds(args, attempt)
        time.sleep(sleep_sec)


def load_existing_raw(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok" and "caption" in record:
                records[int(record["group_index"])] = record
    return records


def write_caption_file(path: Path, groups: list[FrameGroup], captions: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for group in groups:
        caption = captions[group.group_index]
        lines.append(f"[{format_timestamp(group.start_sec)}-{format_timestamp(group.end_sec)}] {caption}")
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def existing_caption_is_valid(path: Path, groups: list[FrameGroup]) -> tuple[bool, str | None]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, "caption missing"

    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) != len(groups):
        return False, f"caption line count {len(lines)} != expected group count {len(groups)}"

    for idx, (line, group) in enumerate(zip(lines, groups, strict=True), start=1):
        prefix = f"[{format_timestamp(group.start_sec)}-{format_timestamp(group.end_sec)}] "
        if not line.startswith(prefix):
            return False, f"line {idx} time range mismatch"
        if not line[len(prefix) :].strip():
            return False, f"line {idx} caption is empty"

    return True, None


def output_paths(output_dir: Path, job: VideoJob) -> dict[str, Path]:
    return {
        "caption": output_dir / "captions" / f"{job.stem}.txt",
        "raw": output_dir / "raw" / f"{job.stem}.jsonl",
    }


def process_video(
    job: VideoJob,
    api: ApiConfig,
    output_dir: Path,
    args: argparse.Namespace,
    status_lock: threading.Lock,
    error_lock: threading.Lock,
    rate_limiter: SlidingWindowRateLimiter | None,
    profiles: dict[str, Any] | None = None,
    assignments: dict[str, str] | None = None,
) -> dict[str, Any]:
    paths = output_paths(output_dir, job)
    status_path = output_dir / "caption_status.jsonl"
    error_path = output_dir / "caption_errors.log"

    try:
        profile = profile_for_job(job, profiles or {}, assignments or {})
        frames = scan_frames(job.frame_dir)
        if not frames:
            raise RuntimeError(f"No frame files matched expected naming rule in {job.frame_dir}")
        groups = build_groups(frames, job.duration_sec)

        if args.force:
            for path in paths.values():
                if path.exists():
                    path.unlink()
        else:
            valid, invalid_reason = existing_caption_is_valid(paths["caption"], groups)
            if valid:
                record = {
                    "time": now_iso(),
                    "status": "skipped",
                    "reason": "caption exists",
                    "video_name": job.video_name,
                    "caption_path": str(paths["caption"]),
                    "raw_path": str(paths["raw"]),
                }
                append_jsonl(status_path, record, status_lock)
                return record
            if paths["caption"].exists():
                append_jsonl(
                    status_path,
                    {
                        "time": now_iso(),
                        "status": "rerun",
                        "reason": f"existing caption invalid: {invalid_reason}",
                        "video_name": job.video_name,
                        "caption_path": str(paths["caption"]),
                    },
                    status_lock,
                )

        if args.dry_run:
            record = {
                "time": now_iso(),
                "status": "dry_run",
                "video_name": job.video_name,
                "frame_count": len(frames),
                "group_size": group_size_for_duration(job.duration_sec),
                "group_count": len(groups),
                "profile_id": profile.get("profile_id") if profile else "",
                "caption_path": str(paths["caption"]),
                "raw_path": str(paths["raw"]),
            }
            append_jsonl(status_path, record, status_lock)
            return record

        existing = load_existing_raw(paths["raw"])
        captions: dict[int, str] = {
            group_index: normalize_caption(record["caption"])
            for group_index, record in existing.items()
            if normalize_caption(record["caption"])
        }

        for group in groups:
            if group.group_index in captions:
                continue

            caption, response_data = request_caption(api, job.video_name, group, args, rate_limiter, profile)
            raw_record = {
                "time": now_iso(),
                "status": "ok",
                "video_name": job.video_name,
                "profile_id": profile.get("profile_id") if profile else "",
                "group_index": group.group_index,
                "group_count": len(groups),
                "start_sec": group.start_sec,
                "end_sec": group.end_sec,
                "time_range": f"{format_timestamp(group.start_sec)}-{format_timestamp(group.end_sec)}",
                "frames": [frame.path.name for frame in group.frames],
                "caption": caption,
                "response": response_data,
            }
            append_jsonl(paths["raw"], raw_record)
            captions[group.group_index] = caption

        missing = [group.group_index for group in groups if group.group_index not in captions]
        if missing:
            raise RuntimeError(f"Missing captions for groups: {missing}")

        write_caption_file(paths["caption"], groups, captions)
        record = {
            "time": now_iso(),
            "status": "ok",
            "video_name": job.video_name,
            "frame_count": len(frames),
            "group_size": group_size_for_duration(job.duration_sec),
            "group_count": len(groups),
            "profile_id": profile.get("profile_id") if profile else "",
            "caption_path": str(paths["caption"]),
            "raw_path": str(paths["raw"]),
        }
        append_jsonl(status_path, record, status_lock)
        return record
    except Exception as exc:
        tb = traceback.format_exc()
        append_error(
            error_path,
            f"[{now_iso()}] {job.video_name}\n{tb}",
            error_lock,
        )
        record = {
            "time": now_iso(),
            "status": "error",
            "video_name": job.video_name,
            "error": str(exc),
            "caption_path": str(paths["caption"]),
            "raw_path": str(paths["raw"]),
        }
        append_jsonl(status_path, record, status_lock)
        return record


def print_plan(jobs: list[VideoJob]) -> None:
    total_frames = 0
    total_groups = 0
    group_size_counts: dict[int, int] = {}
    for job in jobs:
        frames = scan_frames(job.frame_dir)
        groups = build_groups(frames, job.duration_sec)
        total_frames += len(frames)
        total_groups += len(groups)
        group_size = group_size_for_duration(job.duration_sec)
        group_size_counts[group_size] = group_size_counts.get(group_size, 0) + 1

    print(f"Videos: {len(jobs)}")
    print(f"Frames: {total_frames}")
    print(f"Estimated requests: {total_groups}")
    print(f"Group size distribution: {group_size_counts}")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    profile_dir = Path(args.profile_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "captions").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    api = parse_api_config(
        args.model,
        args.base_url,
        args.api_key,
        require_credentials=not args.dry_run,
    )
    jobs = load_manifest(manifest_path)
    jobs = filter_jobs(jobs, args.only, args.start_after, args.limit)
    profiles, assignments = load_prompt_profiles(profile_dir, args.ignore_profile)

    print(f"Model: {api.model}")
    print(f"Base URL: {api.base_url or '(not required for dry run)'}")
    print(f"Output: {output_dir}")
    print(f"Prompt profiles: {len(profiles)} ({'ignored' if args.ignore_profile else profile_dir})")
    print(f"Rate limits: rpm={args.rpm}, estimated_tpm={args.tpm}")
    print_plan(jobs)

    if args.dry_run:
        print("Dry run: no API requests will be sent.")

    status_lock = threading.Lock()
    error_lock = threading.Lock()
    rate_limiter = None if args.dry_run else SlidingWindowRateLimiter(args.rpm, args.tpm)
    workers = max(1, args.workers)
    ok = skipped = dry_run = failed = 0

    if not jobs:
        print("No videos to process.")
        return 0

    if workers == 1:
        for idx, job in enumerate(jobs, start=1):
            print(f"[{idx}/{len(jobs)}] {job.video_name}")
            record = process_video(job, api, output_dir, args, status_lock, error_lock, rate_limiter, profiles, assignments)
            if record["status"] == "ok":
                ok += 1
            elif record["status"] == "skipped":
                skipped += 1
            elif record["status"] == "dry_run":
                dry_run += 1
            else:
                failed += 1
            print(f"  {record['status']}: {record.get('reason') or record.get('error') or record.get('group_count', '')}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(process_video, job, api, output_dir, args, status_lock, error_lock, rate_limiter, profiles, assignments): job
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
                if record["status"] == "ok":
                    ok += 1
                elif record["status"] == "skipped":
                    skipped += 1
                elif record["status"] == "dry_run":
                    dry_run += 1
                else:
                    failed += 1
                print(f"[{completed}/{len(jobs)}] {record['status']}: {job.video_name}")

    print(f"Done. ok={ok}, skipped={skipped}, dry_run={dry_run}, failed={failed}")
    print(f"Status: {output_dir / 'caption_status.jsonl'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

