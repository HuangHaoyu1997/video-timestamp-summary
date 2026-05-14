from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_PATH = Path.home() / ".cache" / "whisper" / "large-v3-turbo.pt"
DEFAULT_INITIAL_PROMPT = "以下是普通话中文视频的转写，请使用简体中文和自然标点。"

HARD_END_PUNCT = set("。！？!?")
SOFT_BREAK_PUNCT = set("，,；;、：:")
CLOSING_PUNCT = set("\"'”’」』）】》〉")
LEFT_PUNCT = set("（【「『《〈")
RIGHT_PUNCT = set("，。！？、；：,.!?;:%）】」』》〉")


@dataclass
class TimedText:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch transcribe Chinese videos with Whisper and export sentence-level timestamps."
        )
    )
    parser.add_argument("--input", default="video", help="Input video directory.")
    parser.add_argument("--output", default="asr", help="Output directory.")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the local Whisper .pt checkpoint.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Optional official Whisper model name. If omitted and the model path stem is "
            "official, the script loads by that name from the checkpoint cache."
        ),
    )
    parser.add_argument("--pattern", default="*.mp4", help="Input file glob pattern.")
    parser.add_argument("--recursive", action="store_true", help="Search input recursively.")
    parser.add_argument("--language", default="zh", help="ASR language code.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or a torch device.")
    parser.add_argument(
        "--fp16",
        choices=("auto", "true", "false"),
        default="auto",
        help="Use fp16 decoding. Auto enables it on CUDA only.",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=1.5,
        help="Merge timestamp items shorter than this duration.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=15.0,
        help="Split timestamp items longer than this duration.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=6,
        help="Merge timestamp items with fewer non-space characters than this.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=80,
        help="Split timestamp items with more non-space characters than this.",
    )
    parser.add_argument(
        "--srt-line-chars",
        type=int,
        default=36,
        help="Approximate max characters per SRT subtitle line.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_INITIAL_PROMPT,
        help="Prompt passed to Whisper to encourage Chinese punctuation.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Enable Whisper conditioning on previous text. Disabled by default to reduce repetition.",
    )
    parser.add_argument(
        "--no-internal-word-timestamps",
        dest="internal_word_timestamps",
        action="store_false",
        help=(
            "Do not ask Whisper for word timestamps internally. Output remains sentence-level either way."
        ),
    )
    parser.set_defaults(internal_word_timestamps=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run files even when JSON/SRT/TXT outputs already exist.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos.")
    parser.add_argument("--dry-run", action="store_true", help="List work without loading ASR model.")
    return parser.parse_args()


def import_runtime() -> tuple[Any, Any]:
    try:
        import torch
        import whisper
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install openai-whisper and torch before running this script."
        ) from exc
    return torch, whisper


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_fp16(device: str, requested: str) -> bool:
    if requested == "auto":
        return device.startswith("cuda")
    return requested == "true"


def iter_input_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    files = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted((p for p in files if p.is_file()), key=lambda p: str(p).casefold())


def output_paths(video_path: Path, input_dir: Path, output_dir: Path) -> dict[str, Path]:
    try:
        rel = video_path.relative_to(input_dir)
    except ValueError:
        rel = Path(video_path.name)
    base = output_dir / "transcript" / rel.parent / video_path.stem
    return {
        "json": base.parent / f"{base.name}.json",
        "srt": base.parent / f"{base.name}.srt",
        "txt": base.parent / f"{base.name}.txt",
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    ensure_parent(path)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8", newline="\n")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？、；：,.!?;:%）】」』》〉])", r"\1", text)
    text = re.sub(r"([（【「『《〈])\s+", r"\1", text)
    return text.strip()


def char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def ends_hard(text: str) -> bool:
    stripped = clean_text(text)
    while stripped and stripped[-1] in CLOSING_PUNCT:
        stripped = stripped[:-1]
    return bool(stripped and stripped[-1] in HARD_END_PUNCT)


def needs_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left[-1].isspace() or right[0].isspace():
        return False
    if left[-1] in LEFT_PUNCT or right[0] in RIGHT_PUNCT:
        return False
    return left[-1].isascii() and right[0].isascii() and left[-1].isalnum() and right[0].isalnum()


def join_text(left: str, right: str) -> str:
    left = clean_text(left)
    right = clean_text(right)
    if not left:
        return right
    if not right:
        return left
    sep = " " if needs_space(left, right) else ""
    return clean_text(f"{left}{sep}{right}")


def split_on_hard_punct(text: str) -> list[str]:
    text = text or ""
    parts: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] in HARD_END_PUNCT:
            j = i + 1
            while j < len(text) and (text[j] in HARD_END_PUNCT or text[j] in CLOSING_PUNCT):
                j += 1
            part = clean_text(text[start:j])
            if part:
                parts.append(part)
            start = j
            i = j
            continue
        i += 1
    tail = clean_text(text[start:])
    if tail:
        parts.append(tail)
    return parts


def split_on_soft_punct(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] in SOFT_BREAK_PUNCT:
            j = i + 1
            while j < len(text) and text[j] in CLOSING_PUNCT:
                j += 1
            part = clean_text(text[start:j])
            if part:
                parts.append(part)
            start = j
            i = j
            continue
        i += 1
    tail = clean_text(text[start:])
    if tail:
        parts.append(tail)
    return parts


def allocate_time(parts: list[str], start: float, end: float) -> list[TimedText]:
    cleaned = [clean_text(part) for part in parts if clean_text(part)]
    if not cleaned:
        return []
    duration = max(0.0, end - start)
    weights = [max(1, char_count(part)) for part in cleaned]
    total = sum(weights)
    cursor = start
    items: list[TimedText] = []
    consumed = 0
    for idx, (part, weight) in enumerate(zip(cleaned, weights)):
        consumed += weight
        part_end = end if idx == len(cleaned) - 1 else start + duration * consumed / total
        items.append(TimedText(start=round(cursor, 3), end=round(part_end, 3), text=part))
        cursor = part_end
    return items


def segment_to_units(segment: dict[str, Any], use_word_timestamps: bool) -> list[TimedText]:
    seg_start = float(segment.get("start") or 0.0)
    seg_end = float(segment.get("end") or seg_start)
    words = segment.get("words") if use_word_timestamps else None
    units: list[TimedText] = []

    if words:
        for word in words:
            text = clean_text(str(word.get("word") or ""))
            if not text:
                continue
            start = float(word.get("start") if word.get("start") is not None else seg_start)
            end = float(word.get("end") if word.get("end") is not None else start)
            if end < start:
                end = start
            parts = split_on_hard_punct(text)
            if len(parts) > 1:
                units.extend(allocate_time(parts, start, end))
            else:
                units.append(TimedText(start=start, end=end, text=text))
        if units:
            return units

    text = clean_text(str(segment.get("text") or ""))
    return allocate_time(split_on_hard_punct(text), seg_start, seg_end)


def build_sentence_candidates(
    raw_segments: Iterable[dict[str, Any]], use_word_timestamps: bool
) -> list[TimedText]:
    candidates: list[TimedText] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None

    def flush() -> None:
        nonlocal current_text, current_start, current_end
        text = clean_text(current_text)
        if text and current_start is not None and current_end is not None:
            candidates.append(TimedText(current_start, current_end, text))
        current_text = ""
        current_start = None
        current_end = None

    for segment in raw_segments:
        for unit in segment_to_units(segment, use_word_timestamps):
            unit.text = clean_text(unit.text)
            if not unit.text:
                continue
            if current_start is None:
                current_start = unit.start
            current_text = join_text(current_text, unit.text)
            current_end = unit.end
            if ends_hard(unit.text):
                flush()
    flush()
    return candidates


def merge_two(left: TimedText, right: TimedText) -> TimedText:
    return TimedText(
        start=min(left.start, right.start),
        end=max(left.end, right.end),
        text=join_text(left.text, right.text),
    )


def split_text_evenly(text: str, chunks: int) -> list[str]:
    text = clean_text(text)
    if chunks <= 1 or char_count(text) <= 1:
        return [text]

    target = max(1, math.ceil(len(text) / chunks))
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + target)
        if end < len(text):
            window = text[start:end]
            soft_positions = [window.rfind(p) for p in SOFT_BREAK_PUNCT]
            best = max(soft_positions)
            if best >= max(4, target // 2):
                end = start + best + 1
        parts.append(clean_text(text[start:end]))
        start = end
    return [part for part in parts if part]


def split_oversized_plain(item: TimedText, max_seconds: float, max_chars: int) -> list[TimedText]:
    count = max(1, char_count(item.text))
    by_chars = math.ceil(count / max(1, max_chars))
    by_time = math.ceil(item.duration / max(0.1, max_seconds))
    chunks = max(1, by_chars, by_time)
    return allocate_time(split_text_evenly(item.text, chunks), item.start, item.end)


def split_long_item(item: TimedText, max_seconds: float, max_chars: int) -> list[TimedText]:
    if item.duration <= max_seconds and char_count(item.text) <= max_chars:
        return [item]

    soft_parts = split_on_soft_punct(item.text)
    if len(soft_parts) <= 1:
        return split_oversized_plain(item, max_seconds, max_chars)

    timed_parts = allocate_time(soft_parts, item.start, item.end)
    grouped: list[TimedText] = []
    current: TimedText | None = None

    for part in timed_parts:
        if current is None:
            current = part
            continue
        combined = merge_two(current, part)
        if combined.duration > max_seconds or char_count(combined.text) > max_chars:
            grouped.extend(split_oversized_plain(current, max_seconds, max_chars))
            current = part
        else:
            current = combined
    if current is not None:
        grouped.extend(split_oversized_plain(current, max_seconds, max_chars))
    return grouped


def is_short(item: TimedText, min_seconds: float, min_chars: int) -> bool:
    return item.duration < min_seconds or char_count(item.text) < min_chars


def merge_short_items(
    items: list[TimedText], min_seconds: float, min_chars: int, max_seconds: float
) -> list[TimedText]:
    if not items:
        return []

    result: list[TimedText] = []
    i = 0
    while i < len(items):
        item = items[i]
        if is_short(item, min_seconds, min_chars) and i + 1 < len(items):
            candidate = merge_two(item, items[i + 1])
            if candidate.duration <= max_seconds * 1.25:
                item = candidate
                i += 1

        if result and is_short(item, min_seconds, min_chars):
            candidate = merge_two(result[-1], item)
            if candidate.duration <= max_seconds * 1.25:
                result[-1] = candidate
            else:
                result.append(item)
        else:
            result.append(item)
        i += 1
    return result


def refine_timestamps(
    candidates: list[TimedText],
    min_seconds: float,
    max_seconds: float,
    min_chars: int,
    max_chars: int,
) -> list[TimedText]:
    items: list[TimedText] = []
    for candidate in candidates:
        items.extend(split_long_item(candidate, max_seconds, max_chars))

    for _ in range(2):
        items = merge_short_items(items, min_seconds, min_chars, max_seconds)

    final: list[TimedText] = []
    for item in items:
        final.extend(split_long_item(item, max_seconds, max_chars))

    return [
        TimedText(round(item.start, 3), round(item.end, 3), clean_text(item.text))
        for item in final
        if clean_text(item.text)
    ]


def srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def clock_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def wrap_subtitle(text: str, width: int) -> str:
    text = clean_text(text)
    if width <= 0 or len(text) <= width:
        return text
    lines: list[str] = []
    remaining = text
    while len(remaining) > width:
        window = remaining[:width]
        break_points = [window.rfind(p) for p in "，,；;、：: "]
        cut = max(break_points)
        if cut < max(6, width // 2):
            cut = width
        else:
            cut += 1
        lines.append(clean_text(remaining[:cut]))
        remaining = clean_text(remaining[cut:])
    if remaining:
        lines.append(remaining)
    return "\n".join(lines)


def render_srt(items: list[TimedText], line_chars: int) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(items, start=1):
        blocks.append(
            f"{idx}\n{srt_time(item.start)} --> {srt_time(item.end)}\n"
            f"{wrap_subtitle(item.text, line_chars)}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_txt(items: list[TimedText]) -> str:
    return "\n".join(
        f"[{clock_time(item.start)} --> {clock_time(item.end)}] {item.text}"
        for item in items
    ) + ("\n" if items else "")


def render_json(
    video_path: Path,
    model_path: Path,
    model_label: str,
    raw_result: dict[str, Any],
    items: list[TimedText],
    args: argparse.Namespace,
) -> str:
    payload = {
        "source": str(video_path),
        "model": model_label,
        "model_path": str(model_path),
        "language": args.language,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "timestamp_level": "sentence",
            "internal_word_timestamps": args.internal_word_timestamps,
            "min_seconds": args.min_seconds,
            "max_seconds": args.max_seconds,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "condition_on_previous_text": args.condition_on_previous_text,
        },
        "raw_segment_count": len(raw_result.get("segments") or []),
        "segments": [
            {
                "id": idx,
                "start": item.start,
                "end": item.end,
                "duration": round(item.duration, 3),
                "text": item.text,
            }
            for idx, item in enumerate(items, start=1)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_model(torch: Any, whisper: Any, args: argparse.Namespace) -> tuple[Any, str]:
    model_path = Path(args.model_path).expanduser()
    device = resolve_device(torch, args.device)
    available = set(whisper.available_models())
    model_name = args.model_name
    if model_name is None and model_path.stem in available:
        model_name = model_path.stem

    if model_name:
        print(f"Loading Whisper model '{model_name}' from cache: {model_path.parent}")
        model = whisper.load_model(model_name, device=device, download_root=str(model_path.parent))
        return model, model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading Whisper checkpoint: {model_path}")
    model = whisper.load_model(str(model_path), device=device)
    return model, str(model_path)


def transcribe_with_retry(model: Any, video_path: Path, args: argparse.Namespace, fp16: bool) -> dict[str, Any]:
    common_kwargs = {
        "language": args.language,
        "task": "transcribe",
        "fp16": fp16,
        "verbose": False,
        "temperature": 0.0,
        "condition_on_previous_text": args.condition_on_previous_text,
        "initial_prompt": args.initial_prompt or None,
    }
    if args.internal_word_timestamps:
        try:
            return model.transcribe(str(video_path), word_timestamps=True, **common_kwargs)
        except Exception as exc:
            print(
                f"Word timestamp pass failed for {video_path.name}; retrying with segment timestamps only. "
                f"Reason: {exc}",
                file=sys.stderr,
            )
    return model.transcribe(str(video_path), word_timestamps=False, **common_kwargs)


def process_one(
    model: Any,
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    model_label: str,
    args: argparse.Namespace,
    fp16: bool,
) -> dict[str, Any]:
    paths = output_paths(video_path, input_dir, output_dir)
    if not args.overwrite and all(path.exists() for path in paths.values()):
        return {
            "file": str(video_path),
            "status": "skipped",
            "reason": "outputs exist",
            "outputs": {key: str(value) for key, value in paths.items()},
        }

    raw_result = transcribe_with_retry(model, video_path, args, fp16)
    candidates = build_sentence_candidates(
        raw_result.get("segments") or [],
        use_word_timestamps=args.internal_word_timestamps,
    )
    items = refine_timestamps(
        candidates,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    write_text_atomic(
        paths["json"],
        render_json(video_path, model_path, model_label, raw_result, items, args),
    )
    write_text_atomic(paths["srt"], render_srt(items, args.srt_line_chars))
    write_text_atomic(paths["txt"], render_txt(items))

    return {
        "file": str(video_path),
        "status": "ok",
        "segments": len(items),
        "outputs": {key: str(value) for key, value in paths.items()},
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    model_path = Path(args.model_path).expanduser()
    status_path = output_dir / "asr_status.jsonl"
    error_path = output_dir / "asr_errors.log"

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    videos = iter_input_files(input_dir, args.pattern, args.recursive)
    if args.limit is not None:
        videos = videos[: args.limit]

    print(f"Found {len(videos)} video(s) under {input_dir}")
    if args.dry_run:
        for video in videos:
            paths = output_paths(video, input_dir, output_dir)
            print(f"- {video} -> {paths['json']}")
        return 0

    torch, whisper = import_runtime()
    device = resolve_device(torch, args.device)
    fp16 = resolve_fp16(device, args.fp16)
    print(f"Device: {device}; fp16: {fp16}; sentence timestamp range: {args.min_seconds}-{args.max_seconds}s")

    model, model_label = load_model(torch, whisper, args)

    ok = 0
    skipped = 0
    failed = 0
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path.name}")
        try:
            record = process_one(
                model,
                video_path,
                input_dir,
                output_dir,
                model_path,
                model_label,
                args,
                fp16,
            )
            append_jsonl(status_path, record)
            if record["status"] == "ok":
                ok += 1
                print(f"  ok: {record['segments']} sentence-level segment(s)")
            else:
                skipped += 1
                print(f"  skipped: {record.get('reason', '')}")
        except Exception as exc:
            failed += 1
            record = {"file": str(video_path), "status": "error", "error": str(exc)}
            append_jsonl(status_path, record)
            ensure_parent(error_path)
            with error_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {video_path}\n")
                f.write(traceback.format_exc())
            print(f"  error: {exc}", file=sys.stderr)

    print(f"Done. ok={ok}, skipped={skipped}, failed={failed}")
    print(f"Status: {status_path}")
    if failed:
        print(f"Errors: {error_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
