from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import random
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("frame_manifest.csv")
DEFAULT_MAPPING = Path("video_title_url_mapping.csv")
DEFAULT_ASR_DIR = Path("asr") / "transcript"
DEFAULT_CAPTION_DIR = Path("caption") / "captions"
DEFAULT_PLAN_DIR = Path("plan")
DEFAULT_PROFILE_DIR = Path("prompt_profiles")
DEFAULT_OUTPUT = Path("abstract")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

SRT_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
CAPTION_RANGE_RE = re.compile(
    r"^\s*\[(?P<start>[0-9:：,.，]+)\s*[-–—~至]\s*(?P<end>[0-9:：,.，]+)\]\s*(?P<text>.*)$"
)


@dataclass(frozen=True)
class ApiConfig:
    model: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class VideoJob:
    order: int
    video_name: str
    duration_sec: float

    @property
    def stem(self) -> str:
        return Path(self.video_name).stem


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    title: str
    url: str
    duration_sec: float
    platform: str

    def to_json(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "url": self.url,
            "duration_sec": round(self.duration_sec, 3),
            "platform": self.platform,
        }


@dataclass(frozen=True)
class TimelineEvent:
    start_sec: float
    end_sec: float
    kind: str
    text: str


class ValidationError(Exception):
    pass


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch generate per-video abstract JSON from ASR and frame captions."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Frame manifest CSV path.")
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="Video title/url mapping CSV path.")
    parser.add_argument("--asr-dir", default=str(DEFAULT_ASR_DIR), help="ASR transcript directory.")
    parser.add_argument("--caption-dir", default=str(DEFAULT_CAPTION_DIR), help="Caption txt directory.")
    parser.add_argument("--plan-dir", default=str(DEFAULT_PLAN_DIR), help="Optional summary plan txt directory.")
    parser.add_argument("--ignore-plan", action="store_true", help="Ignore plan/{video}.txt even when it exists.")
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR), help="Optional prompt profile directory.")
    parser.add_argument("--ignore-profile", action="store_true", help="Ignore prompt profiles even when available.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory. Defaults to ./abstract.")
    parser.add_argument("--model", default=None, help=f"Model name. Defaults to {DEFAULT_MODEL}.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Falls back to OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key. Falls back to OPENAI_API_KEY.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos after filtering.")
    parser.add_argument("--only", action="append", default=[], help="Only process videos whose name contains this text. Repeatable.")
    parser.add_argument("--start-after", default=None, help="Skip videos until this text is found in video name.")
    parser.add_argument("--force", action="store_true", help="Re-run even when final summary JSON already exists and is valid.")
    parser.add_argument("--workers", type=int, default=5, help="Number of videos to process concurrently.")
    parser.add_argument("--rpm", type=int, default=60, help="Max API requests per rolling minute. Use 0 to disable.")
    parser.add_argument("--tpm", type=int, default=900000, help="Estimated max tokens per rolling minute. Use 0 to disable.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout per request in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for API, JSON parse, and validation failures.")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Initial retry sleep in seconds.")
    parser.add_argument("--max-retry-sleep", type=float, default=60.0, help="Maximum sleep between retries in seconds.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens per video summary.")
    parser.add_argument(
        "--max-detail-segments",
        type=int,
        default=None,
        help=(
            "Override the auto maximum for segment[1:]. "
            "Use 0 to disable segment-count validation."
        ),
    )
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
        help="Parse inputs and write no model outputs. Useful for checking timeline merge and filtering.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


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
            if not video_name:
                continue
            jobs.append(
                VideoJob(
                    order=order,
                    video_name=video_name,
                    duration_sec=float(row.get("duration_sec") or 0),
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


def load_video_mapping(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    mapping: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            mapping[filename] = {
                "title": (row.get("title") or Path(filename).stem).strip(),
                "url": (row.get("url") or "").strip(),
                "platform": (row.get("platform") or "unknown").strip() or "unknown",
            }
    return mapping


def make_video_id(video_name: str, url: str) -> str:
    seed = url.strip() or video_name.strip()
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"vid_{digest}"


def build_video_meta(job: VideoJob, mapping: dict[str, dict[str, str]]) -> VideoMeta:
    mapped = mapping.get(job.video_name, {})
    url = mapped.get("url", "")
    return VideoMeta(
        video_id=make_video_id(job.video_name, url),
        title=mapped.get("title") or job.stem,
        url=url,
        duration_sec=job.duration_sec,
        platform=mapped.get("platform") or "unknown",
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_timecode(value: str) -> float:
    cleaned = value.strip().replace("：", ":").replace("，", ".").replace(",", ".")
    if not cleaned:
        raise ValueError("empty timecode")

    parts = cleaned.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 1:
            return float(parts[0])
    except ValueError as exc:
        raise ValueError(f"invalid timecode: {value}") from exc
    raise ValueError(f"invalid timecode: {value}")


def format_timestamp(total_sec: float) -> str:
    sec = int(round(total_sec))
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    seconds = sec % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_srt(path: Path) -> list[TimelineEvent]:
    if not path.exists():
        raise FileNotFoundError(f"ASR SRT not found: {path}")

    events: list[TimelineEvent] = []
    blocks = re.split(r"\n\s*\n", read_text(path).replace("\r\n", "\n").replace("\r", "\n"))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        range_index = None
        range_match = None
        for idx, line in enumerate(lines):
            range_match = SRT_RANGE_RE.search(line)
            if range_match:
                range_index = idx
                break
        if range_index is None or range_match is None:
            continue

        text = normalize_text(" ".join(lines[range_index + 1 :]))
        if not text:
            continue
        events.append(
            TimelineEvent(
                start_sec=parse_timecode(range_match.group("start")),
                end_sec=parse_timecode(range_match.group("end")),
                kind="ASR",
                text=text,
            )
        )
    return events


def parse_caption(path: Path) -> list[TimelineEvent]:
    if not path.exists():
        raise FileNotFoundError(f"Caption txt not found: {path}")

    events: list[TimelineEvent] = []
    current: TimelineEvent | None = None
    for raw_line in read_text(path).replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = CAPTION_RANGE_RE.match(line)
        if match:
            current = TimelineEvent(
                start_sec=parse_timecode(match.group("start")),
                end_sec=parse_timecode(match.group("end")),
                kind="CAPTION",
                text=normalize_text(match.group("text")),
            )
            if current.text:
                events.append(current)
            continue

        if events:
            previous = events[-1]
            events[-1] = TimelineEvent(
                start_sec=previous.start_sec,
                end_sec=previous.end_sec,
                kind=previous.kind,
                text=normalize_text(previous.text + " " + line),
            )
    return events


def merge_timeline(asr_events: list[TimelineEvent], caption_events: list[TimelineEvent]) -> list[TimelineEvent]:
    events = [*asr_events, *caption_events]
    events.sort(key=lambda x: (x.start_sec, x.end_sec, 0 if x.kind == "ASR" else 1))
    return events


def render_timeline(events: list[TimelineEvent]) -> str:
    lines = ["[TIMELINE]"]
    for event in events:
        lines.append(
            f"[{format_timestamp(event.start_sec)}-{format_timestamp(event.end_sec)}] "
            f"{event.kind}: {event.text}"
        )
    return "\n".join(lines)


def estimate_tokens(text: str, max_tokens: int) -> int:
    # Chinese text is roughly 1-2 chars per token. This deliberately overestimates for rate limiting.
    return max(1, len(text) // 2 + max_tokens + 800)


def segment_count_guidance(duration_sec: float) -> str:
    if duration_sec <= 90:
        return "这条视频很短，segment[1:] 最多只能有 1 个连续知识单元；把入口、设置、演示效果和注意事项都放进同一个 segment 的 seg_abs。"
    if duration_sec <= 180:
        return "这条视频较短，segment[1:] 最多只能有 2 个连续知识单元；只有出现两个明显不同的问题或功能时才拆成 2 个。"
    if duration_sec <= 300:
        return "这条视频为中短视频，segment[1:] 通常控制在 1-3 个连续知识单元。"
    if duration_sec <= 480:
        return "这条视频较长，segment[1:] 通常控制在 2-4 个连续知识单元。"
    return "这条视频很长或可能包含合集内容，segment[1:] 通常控制在 3-6 个连续知识单元；只有明显多个互不相关主题时才超过 6 个。"


def auto_max_detail_segments(duration_sec: float) -> int:
    if duration_sec <= 90:
        return 1
    if duration_sec <= 180:
        return 2
    if duration_sec <= 300:
        return 3
    if duration_sec <= 480:
        return 4
    return 6


def effective_max_detail_segments(meta: VideoMeta, args: argparse.Namespace) -> int | None:
    if args.max_detail_segments is None:
        return auto_max_detail_segments(meta.duration_sec)
    if args.max_detail_segments <= 0:
        return None
    return args.max_detail_segments


def build_messages(
    meta: VideoMeta,
    timeline_text: str,
    plan_text: str | None = None,
    profile: dict[str, Any] | None = None,
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    expected_video_json = json.dumps(meta.to_json(), ensure_ascii=False, indent=2)
    plan_block = ""
    if plan_text and plan_text.strip():
        plan_block = (
            "\n\n[SUMMARY_PLAN]\n"
            f"{plan_text.strip()}"
        )
    profile_block = ""
    if profile:
        profile_block = (
            "\n\n[PROMPT_PROFILE]\n"
            f"profile_id: {profile.get('profile_id', '')}\n"
            f"name: {profile.get('name', '')}\n"
            f"topic: {profile.get('topic', '')}\n"
            f"{str(profile.get('summary_prompt') or '').strip()}"
        )
    validation_hint = ""
    if validation_error:
        validation_hint = (
            "\n\n上一次输出没有通过脚本校验，错误如下：\n"
            f"{validation_error}\n"
            "请重新输出严格合法 JSON，并修复以上问题。"
        )

    system_prompt = (
        "你是面向视频知识抽取和 RAG 入库的视频摘要专家。"
        "你会阅读按时间顺序交错排列的 ASR 和画面 caption，生成单视频摘要 JSON。"
        "必须严格遵守给定 JSON schema。"
    )
    user_prompt = (
        "请根据下面的视频元信息和时间线，生成一个单视频摘要 JSON。\n\n"
        "硬性要求：\n"
        "1. 只输出严格 JSON，不要 Markdown 代码块，不要注释，不要解释性文字。\n"
        "2. 顶层只能包含 video 和 segment 两个字段。\n"
        "3. video 字段必须原样复制下面给定的视频元信息，不要改写 title/url/platform/duration_sec/video_id。\n"
        "4. segment[0] 固定为全文摘要：title 必须是空字符串，seg_abs 必须且只能有一个字符串。\n"
        "5. segment[1:] 是给召回和 RAG 使用的连续知识单元，不是逐秒时间片，也不是每个点击/页面/镜头一个段落。\n"
        f"6. {segment_count_guidance(meta.duration_sec)}\n"
        "7. 每个分段必须严格按照视频时间顺序排列，并对应视频中连续出现的一段内容。\n"
        "8. 如果相邻内容属于同一个概念讲解、操作流程、主题展开或评测结论，必须合并到同一个 segment；例如“进入入口、切换开关、展示效果、补充注意事项”应作为同一段内的多个 seg_abs 要点，而不是拆成多个 segment。\n"
        "9. 只有当视频进入新的主题、功能、对象维度、问题回答或明显新的章节时，才新增 segment。\n"
        "10. 禁止为了显得细致而过度切分；segment 要少而完整，保证每段可作为一个连贯 RAG chunk 被检索和阅读。\n"
        "11. segment[1:] 每段 title 必须是非空中文标题，seg_abs 是该连续知识单元内的多个摘要要点。\n"
        "12. 摘要要保留具体实体、术语、型号/版本、功能名、步骤路径、按钮/入口、状态变化、操作结果或结论；没有证据的信息不要编造。\n"
        "13. ASR 可能有错字；可以结合 caption 和上下文纠正明显错听，但不要创造时间线中不存在的结论。\n\n"
        "14. 如果提供了 [PROMPT_PROFILE]，必须按其中的 topic 取舍规则调整摘要重点；但 profile 不是事实来源，所有事实仍以 [TIMELINE] 为准。\n"
        "15. 如果提供了 [SUMMARY_PLAN]，必须优先遵循其中的内容组织、规避内容和分段建议；但规划不是事实来源，所有事实仍以 [TIMELINE] 为准。\n"
        "16. 如果 [SUMMARY_PLAN] 提醒某些片段低置信度、无知识价值或应忽略，摘要中不要强行写入这些内容。\n\n"
        "输出 JSON 结构示例：\n"
        "{\n"
        "  \"video\": {\"video_id\": \"...\", \"title\": \"...\", \"url\": \"...\", \"duration_sec\": 123, \"platform\": \"...\"},\n"
        "  \"segment\": [\n"
        "    {\"title\": \"\", \"seg_abs\": [\"整条视频的一段全文摘要。\"]},\n"
        "    {\"title\": \"第一段标题\", \"seg_abs\": [\"第一段第一个要点。\", \"第一段第二个要点。\"]}\n"
        "  ]\n"
        "}\n\n"
        "[VIDEO]\n"
        f"{expected_video_json}\n\n"
        f"{profile_block}\n\n"
        f"{plan_block}\n\n"
        f"{timeline_text}"
        f"{validation_hint}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


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


def preview_text(text: str, limit: int = 1000) -> str:
    return text[:limit].replace("\r", "\\r").replace("\n", "\\n")


def parse_response_data(response_text: str) -> dict[str, Any]:
    stripped = response_text.strip()
    if not stripped:
        raise ValueError("Empty response body from API")

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data

    # Some OpenAI-compatible gateways return Server-Sent Events even when stream=false.
    if stripped.startswith("data:"):
        merged_content: list[str] = []
        last_json: dict[str, Any] | None = None
        for line in stripped.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict):
                last_json = chunk
                try:
                    delta = chunk["choices"][0].get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        merged_content.append(content)
                except (KeyError, IndexError, TypeError, AttributeError):
                    pass
        if merged_content:
            return {"choices": [{"message": {"content": "".join(merged_content)}}]}
        if last_json is not None:
            return last_json

    raise ValueError(f"API response is not JSON. Preview: {preview_text(response_text)}")


def parse_strict_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```") or stripped.endswith("```"):
        raise ValidationError("模型输出包含 Markdown 代码块，必须只输出原始 JSON。")
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"JSON 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("顶层 JSON 必须是 object。")
    return data


def validate_video_object(video: Any) -> list[str]:
    errors: list[str] = []
    required = {"video_id", "title", "url", "duration_sec", "platform"}
    if not isinstance(video, dict):
        return ["video 必须是 object。"]
    keys = set(video)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        errors.append(f"video 缺少字段: {', '.join(missing)}。")
    if extra:
        errors.append(f"video 包含额外字段: {', '.join(extra)}。")
    for key in ("video_id", "title", "url", "platform"):
        if key in video and not isinstance(video[key], str):
            errors.append(f"video.{key} 必须是字符串。")
    if "duration_sec" in video and not isinstance(video["duration_sec"], (int, float)):
        errors.append("video.duration_sec 必须是数字。")
    return errors


def validate_segments(segment: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(segment, list):
        return ["segment 必须是数组。"]
    if not segment:
        return ["segment 必须是非空数组。"]

    for idx, item in enumerate(segment):
        prefix = f"segment[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} 必须是 object。")
            continue
        keys = set(item)
        required = {"title", "seg_abs"}
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        if missing:
            errors.append(f"{prefix} 缺少字段: {', '.join(missing)}。")
        if extra:
            errors.append(f"{prefix} 包含额外字段: {', '.join(extra)}。")
        title = item.get("title")
        seg_abs = item.get("seg_abs")
        if not isinstance(title, str):
            errors.append(f"{prefix}.title 必须是字符串。")
        elif idx == 0 and title != "":
            errors.append("segment[0].title 必须等于空字符串。")
        elif idx > 0 and not title.strip():
            errors.append(f"{prefix}.title 必须是非空字符串。")

        if not isinstance(seg_abs, list):
            errors.append(f"{prefix}.seg_abs 必须是数组。")
            continue
        if idx == 0 and len(seg_abs) != 1:
            errors.append("segment[0].seg_abs 必须且只能包含一个 item。")
        if not seg_abs:
            errors.append(f"{prefix}.seg_abs 必须是非空数组。")
        for abs_idx, text in enumerate(seg_abs):
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{prefix}.seg_abs[{abs_idx}] 必须是非空字符串。")
    return errors


def validate_summary(data: dict[str, Any]) -> None:
    errors: list[str] = []
    required = {"video", "segment"}
    keys = set(data)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        errors.append(f"顶层缺少字段: {', '.join(missing)}。")
    if extra:
        errors.append(f"顶层包含额外字段: {', '.join(extra)}。")
    if "video" in data:
        errors.extend(validate_video_object(data["video"]))
    if "segment" in data:
        errors.extend(validate_segments(data["segment"]))
    if errors:
        raise ValidationError(" ".join(errors))


def validate_detail_segment_count(data: dict[str, Any], meta: VideoMeta, args: argparse.Namespace) -> None:
    max_detail_segments = effective_max_detail_segments(meta, args)
    if max_detail_segments is None:
        return
    detail_count = len(data.get("segment", [])) - 1
    if detail_count > max_detail_segments:
        raise ValidationError(
            f"segment[1:] 数量过多：当前 {detail_count} 个，按视频时长最多允许 {max_detail_segments} 个。"
            "请合并相邻的同一操作流程、同一功能讲解或同一评测结论，保证每段是连贯 RAG chunk。"
        )


def normalize_summary(data: dict[str, Any], meta: VideoMeta) -> dict[str, Any]:
    # Metadata is deterministic local context; keep it exact even if the model changes punctuation or number shape.
    normalized = {
        "video": meta.to_json(),
        "segment": data["segment"],
    }
    validate_summary(normalized)
    return normalized


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


def request_summary(
    api: ApiConfig,
    meta: VideoMeta,
    timeline_text: str,
    plan_text: str | None,
    profile: dict[str, Any] | None,
    args: argparse.Namespace,
    raw_path: Path,
    rate_limiter: SlidingWindowRateLimiter | None,
) -> dict[str, Any]:
    url = chat_completions_url(api.base_url)
    headers = {
        "Authorization": f"Bearer {api.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "User-Agent": DEFAULT_USER_AGENT,
    }

    validation_error: str | None = None
    token_source = timeline_text
    if plan_text:
        token_source += "\n" + plan_text
    if profile:
        token_source += "\n" + str(profile.get("summary_prompt") or "")
    estimated_tokens = estimate_tokens(token_source, args.max_tokens)
    for attempt in range(1, args.retries + 2):
        payload = {
            "model": api.model,
            "messages": build_messages(meta, timeline_text, plan_text, profile, validation_error),
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
            response_data = parse_response_data(response_text)
            content = parse_chat_content(response_data)
            parsed = parse_strict_json(content)
            validate_summary(parsed)
            validate_detail_segment_count(parsed, meta, args)
            normalized = normalize_summary(parsed, meta)
            append_jsonl(
                raw_path,
                {
                    "time": now_iso(),
                    "status": "ok",
                    "attempt": attempt,
                    "video_id": meta.video_id,
                    "title": meta.title,
                    "plan_chars": len(plan_text or ""),
                    "content": content,
                    "response": response_data,
                },
            )
            return normalized
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            append_jsonl(
                raw_path,
                {
                    "time": now_iso(),
                    "status": "error",
                    "attempt": attempt,
                    "video_id": meta.video_id,
                    "title": meta.title,
                    "error_type": "http",
                    "error": f"HTTP {exc.code}: {response_body[:1000]}",
                },
            )
            if attempt > args.retries or not should_retry_http(exc.code):
                raise RuntimeError(f"HTTP {exc.code}: {response_body[:1000]}") from exc
            time.sleep(retry_sleep_seconds(args, attempt, exc.headers))
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            ConnectionResetError,
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
            ValueError,
        ) as exc:
            append_jsonl(
                raw_path,
                {
                    "time": now_iso(),
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
            time.sleep(retry_sleep_seconds(args, attempt))
        except ValidationError as exc:
            validation_error = str(exc)
            append_jsonl(
                raw_path,
                {
                    "time": now_iso(),
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
            time.sleep(retry_sleep_seconds(args, attempt))

    raise RuntimeError("unreachable retry loop exit")


def output_paths(output_dir: Path, job: VideoJob) -> dict[str, Path]:
    return {
        "summary": output_dir / "summaries" / f"{job.stem}.json",
        "raw": output_dir / "raw" / f"{job.stem}.jsonl",
    }


def plan_path_for_job(plan_dir: Path, job: VideoJob) -> Path:
    return plan_dir / f"{job.stem}.txt"


def load_plan_for_job(plan_dir: Path, job: VideoJob, ignore_plan: bool = False) -> tuple[str | None, Path]:
    path = plan_path_for_job(plan_dir, job)
    if ignore_plan or not path.exists() or path.stat().st_size <= 0:
        return None, path
    return read_text(path).strip(), path


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


def existing_summary_is_valid(path: Path) -> tuple[bool, str | None]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, "summary missing"
    try:
        data = json.loads(read_text(path))
        validate_summary(data)
        return True, None
    except Exception as exc:
        return False, str(exc)


def build_timeline_for_job(job: VideoJob, asr_dir: Path, caption_dir: Path) -> tuple[list[TimelineEvent], int, int]:
    asr_events = parse_srt(asr_dir / f"{job.stem}.srt")
    caption_events = parse_caption(caption_dir / f"{job.stem}.txt")
    if not asr_events:
        raise RuntimeError(f"No ASR events parsed for {job.video_name}")
    if not caption_events:
        raise RuntimeError(f"No caption events parsed for {job.video_name}")
    return merge_timeline(asr_events, caption_events), len(asr_events), len(caption_events)


def process_video(
    job: VideoJob,
    api: ApiConfig,
    mapping: dict[str, dict[str, str]],
    asr_dir: Path,
    caption_dir: Path,
    plan_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    status_lock: threading.Lock,
    error_lock: threading.Lock,
    rate_limiter: SlidingWindowRateLimiter | None,
    profiles: dict[str, Any] | None = None,
    assignments: dict[str, str] | None = None,
) -> dict[str, Any]:
    paths = output_paths(output_dir, job)
    status_path = output_dir / "abstract_status.jsonl"
    error_path = output_dir / "abstract_errors.log"
    meta = build_video_meta(job, mapping)

    if args.force:
        for path in paths.values():
            if path.exists():
                path.unlink()
    else:
        valid, invalid_reason = existing_summary_is_valid(paths["summary"])
        if valid:
            record = {
                "time": now_iso(),
                "status": "skipped",
                "reason": "summary exists",
                "video_name": job.video_name,
                "summary_path": str(paths["summary"]),
                "raw_path": str(paths["raw"]),
            }
            append_jsonl(status_path, record, status_lock)
            return record
        if paths["summary"].exists():
            append_jsonl(
                status_path,
                {
                    "time": now_iso(),
                    "status": "rerun",
                    "reason": f"existing summary invalid: {invalid_reason}",
                    "video_name": job.video_name,
                    "summary_path": str(paths["summary"]),
                },
                status_lock,
            )

    try:
        timeline, asr_count, caption_count = build_timeline_for_job(job, asr_dir, caption_dir)
        timeline_text = render_timeline(timeline)
        plan_text, plan_path = load_plan_for_job(plan_dir, job, args.ignore_plan)
        profile = profile_for_job(job, profiles or {}, assignments or {})
        if args.max_timeline_chars and len(timeline_text) > args.max_timeline_chars:
            raise RuntimeError(
                f"Timeline too long: {len(timeline_text)} chars > {args.max_timeline_chars}. "
                "Increase --max-timeline-chars or run a two-stage summarizer."
            )

        if args.dry_run:
            record = {
                "time": now_iso(),
                "status": "dry_run",
                "video_name": job.video_name,
                "video_id": meta.video_id,
                "duration_sec": meta.duration_sec,
                "asr_events": asr_count,
                "caption_events": caption_count,
                "timeline_events": len(timeline),
                "timeline_chars": len(timeline_text),
                "plan_used": bool(plan_text),
                "plan_chars": len(plan_text or ""),
                "plan_path": str(plan_path),
                "profile_id": profile.get("profile_id") if profile else "",
                "summary_path": str(paths["summary"]),
                "raw_path": str(paths["raw"]),
            }
            append_jsonl(status_path, record, status_lock)
            return record

        summary = request_summary(api, meta, timeline_text, plan_text, profile, args, paths["raw"], rate_limiter)
        write_json_atomic(paths["summary"], summary)
        record = {
            "time": now_iso(),
            "status": "ok",
            "video_name": job.video_name,
            "video_id": meta.video_id,
            "duration_sec": meta.duration_sec,
            "asr_events": asr_count,
            "caption_events": caption_count,
            "timeline_events": len(timeline),
            "timeline_chars": len(timeline_text),
            "plan_used": bool(plan_text),
            "plan_chars": len(plan_text or ""),
            "plan_path": str(plan_path),
            "profile_id": profile.get("profile_id") if profile else "",
            "segments": len(summary["segment"]),
            "summary_path": str(paths["summary"]),
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
            "summary_path": str(paths["summary"]),
            "raw_path": str(paths["raw"]),
        }
        append_jsonl(status_path, record, status_lock)
        return record


def print_plan(jobs: list[VideoJob], asr_dir: Path, caption_dir: Path, plan_dir: Path) -> None:
    total_asr = 0
    total_caption = 0
    total_timeline = 0
    total_chars = 0
    missing = 0
    plans_available = 0
    for job in jobs:
        try:
            timeline, asr_count, caption_count = build_timeline_for_job(job, asr_dir, caption_dir)
            timeline_text = render_timeline(timeline)
        except Exception:
            missing += 1
            continue
        total_asr += asr_count
        total_caption += caption_count
        total_timeline += len(timeline)
        total_chars += len(timeline_text)
        plan_text, _ = load_plan_for_job(plan_dir, job)
        if plan_text:
            plans_available += 1

    print(f"Videos: {len(jobs)}")
    print(f"Plans available: {plans_available}")
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
    plan_dir = Path(args.plan_dir).resolve()
    profile_dir = Path(args.profile_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    api = parse_api_config(
        args.model,
        args.base_url,
        args.api_key,
        require_credentials=not args.dry_run,
    )
    mapping = load_video_mapping(mapping_path)
    jobs = load_manifest(manifest_path)
    jobs = filter_jobs(jobs, args.only, args.start_after, args.limit)
    profiles, assignments = load_prompt_profiles(profile_dir, args.ignore_profile)

    print(f"Model: {api.model}")
    print(f"Base URL: {api.base_url or '(not required for dry run)'}")
    print(f"Output: {output_dir}")
    print(f"Plan dir: {plan_dir} ({'ignored' if args.ignore_plan else 'enabled'})")
    print(f"Prompt profiles: {len(profiles)} ({'ignored' if args.ignore_profile else profile_dir})")
    print(f"Rate limits: rpm={args.rpm}, estimated_tpm={args.tpm}")
    print_plan(jobs, asr_dir, caption_dir, plan_dir)

    if args.dry_run:
        print("Dry run: no API requests will be sent.")

    if not jobs:
        print("No videos to process.")
        return 0

    status_lock = threading.Lock()
    error_lock = threading.Lock()
    rate_limiter = None if args.dry_run else SlidingWindowRateLimiter(args.rpm, args.tpm)
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
                plan_dir,
                output_dir,
                args,
                status_lock,
                error_lock,
                rate_limiter,
                profiles,
                assignments,
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
            print(f"  {status}: {record.get('reason') or record.get('error') or record.get('segments', '')}")
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
                    plan_dir,
                    output_dir,
                    args,
                    status_lock,
                    error_lock,
                    rate_limiter,
                    profiles,
                    assignments,
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
    print(f"Status: {output_dir / 'abstract_status.jsonl'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
