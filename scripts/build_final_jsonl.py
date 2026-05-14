from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ABSTRACT_DIR = Path("abstract") / "summaries"
DEFAULT_TIMESTAMP_DIR = Path("timestamp") / "time_stamps"
DEFAULT_MAPPING = Path("video_title_url_mapping.csv")
DEFAULT_OUTPUT = Path("final_video_segments.jsonl")
DEFAULT_VIDEO_DIR = Path("video")
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".flv",
    ".wmv",
    ".mpeg",
    ".mpg",
}


@dataclass(frozen=True)
class BuildIssue:
    stem: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge abstract summaries and segment timestamps into final JSONL."
    )
    parser.add_argument(
        "--abstract-dir",
        default=str(DEFAULT_ABSTRACT_DIR),
        help="Directory containing abstract/summaries/{video}.json files.",
    )
    parser.add_argument(
        "--timestamp-dir",
        default=str(DEFAULT_TIMESTAMP_DIR),
        help="Directory containing timestamp/time_stamps/{video}.json files.",
    )
    parser.add_argument(
        "--mapping",
        default=str(DEFAULT_MAPPING),
        help="Optional CSV used to preserve video order. Uses filename/file_path/title columns.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSONL path. Existing file is overwritten atomically.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only include videos whose stem contains this text. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Write at most N valid records.")
    parser.add_argument(
        "--missing-url-policy",
        choices=("file-url", "empty", "error"),
        default="file-url",
        help=(
            "How to fill final url when abstract.video.url is empty. "
            "file-url resolves a local file from --video-dir or mapping file_path; "
            "empty writes an empty string; error fails the record."
        ),
    )
    parser.add_argument(
        "--video-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Local video directory used by --missing-url-policy file-url.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report counts without writing the JSONL file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error when any matching file is missing or invalid.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        help="Maximum number of issues to print.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def json_files_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
        if path.stem in files:
            raise ValueError(f"Duplicate JSON stem in {directory}: {path.stem}")
        files[path.stem] = path
    return files


def stem_from_mapping_row(row: dict[str, str]) -> str | None:
    filename = (row.get("filename") or "").strip()
    if filename:
        return Path(filename).stem

    file_path = (row.get("file_path") or "").strip()
    if file_path:
        return Path(file_path).stem

    title = (row.get("title") or "").strip()
    if title:
        return Path(title).stem

    return None


def load_order(mapping_path: Path, available_stems: set[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = stem_from_mapping_row(row)
                if stem and stem in available_stems and stem not in seen:
                    ordered.append(stem)
                    seen.add(stem)

    for stem in sorted(available_stems):
        if stem not in seen:
            ordered.append(stem)
            seen.add(stem)

    return ordered


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def existing_path_from_value(value: str, base_dirs: list[Path]) -> Path | None:
    raw = (value or "").strip()
    if not raw:
        return None

    path = Path(raw)
    candidates = [path] if path.is_absolute() else [path, *(base / raw for base in base_dirs)]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def load_local_video_index(mapping_path: Path, video_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    mapping_base = mapping_path.parent if mapping_path.parent != Path("") else Path(".")

    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, base_dirs in (
                    ("file_path", [mapping_base, video_dir]),
                    ("filename", [video_dir, mapping_base]),
                ):
                    path = existing_path_from_value(row.get(key, ""), base_dirs)
                    if path is not None and path.suffix.lower() in VIDEO_EXTENSIONS:
                        index.setdefault(path.stem, path)

    if video_dir.exists():
        for path in sorted(video_dir.rglob("*"), key=lambda item: str(item).casefold()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                index.setdefault(path.stem, path.resolve())

    return index


def resolve_video_url(
    stem: str,
    value: Any,
    missing_url_policy: str,
    local_video_index: dict[str, Path],
) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()

    if missing_url_policy == "empty":
        return ""

    if missing_url_policy == "file-url":
        path = local_video_index.get(stem)
        if path is not None:
            return path.resolve().as_uri()
        raise ValueError(
            "video.url is empty and no local video file was found for "
            f"{stem!r}; pass --missing-url-policy empty to allow a blank url"
        )

    raise ValueError(
        "video.url must be a non-empty string; pass --missing-url-policy "
        "file-url or empty to use local-only videos"
    )


def validate_segment(segment: Any) -> list[dict[str, Any]]:
    if not isinstance(segment, list) or not segment:
        raise ValueError("segment must be a non-empty array")

    for idx, item in enumerate(segment):
        if not isinstance(item, dict):
            raise ValueError(f"segment[{idx}] must be an object")
        title = item.get("title")
        seg_abs = item.get("seg_abs")
        if not isinstance(title, str):
            raise ValueError(f"segment[{idx}].title must be a string")
        if idx == 0 and title != "":
            raise ValueError("segment[0].title must be an empty string")
        if idx > 0 and not title.strip():
            raise ValueError(f"segment[{idx}].title must be non-empty")
        if not isinstance(seg_abs, list) or not seg_abs:
            raise ValueError(f"segment[{idx}].seg_abs must be a non-empty array")
        for abs_idx, value in enumerate(seg_abs):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"segment[{idx}].seg_abs[{abs_idx}] must be a non-empty string")

    if len(segment[0].get("seg_abs", [])) != 1:
        raise ValueError("segment[0].seg_abs must contain exactly one item")

    return segment


def validate_time_stamp(time_stamp: Any, duration_sec: int | float | None = None) -> list[list[int]]:
    if not isinstance(time_stamp, list):
        raise ValueError("time_stamp must be an array")

    max_end: int | None = None
    if duration_sec is not None:
        if isinstance(duration_sec, bool) or not isinstance(duration_sec, (int, float)) or duration_sec < 0:
            raise ValueError("abstract.video.duration_sec must be a non-negative number")
        max_end = int(duration_sec)

    seen: set[tuple[int, int]] = set()
    previous_end: int | None = None
    for idx, item in enumerate(time_stamp):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"time_stamp[{idx}] must be a two-item array")
        start, end = item
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError(f"time_stamp[{idx}] must contain integer start/end values")
        if idx == 0 and item != [0, 0]:
            raise ValueError("time_stamp[0] must be [0, 0]")
        if idx > 0 and (start < 0 or end < start):
            raise ValueError(f"time_stamp[{idx}] must satisfy 0 <= start <= end")
        pair = (start, end)
        if pair in seen:
            raise ValueError(f"time_stamp[{idx}] duplicates an earlier interval: {item}")
        seen.add(pair)
        if idx > 0 and previous_end is not None and start < previous_end:
            raise ValueError(f"time_stamp[{idx}] must be monotonic: start {start} < previous end {previous_end}")
        if idx > 0 and max_end is not None and end > max_end:
            raise ValueError(f"time_stamp[{idx}].end exceeds video duration: {end} > {max_end}")
        previous_end = end

    return time_stamp


def build_record(
    stem: str,
    abstract_path: Path,
    timestamp_path: Path,
    missing_url_policy: str,
    local_video_index: dict[str, Path],
) -> dict[str, Any]:
    abstract_data = read_json(abstract_path)
    timestamp_data = read_json(timestamp_path)

    video = abstract_data.get("video")
    if not isinstance(video, dict):
        raise ValueError("abstract.video must be an object")
    duration_sec = video.get("duration_sec")
    if isinstance(duration_sec, bool) or not isinstance(duration_sec, (int, float)) or duration_sec < 0:
        raise ValueError("abstract.video.duration_sec must be a non-negative number")
    if set(timestamp_data) != {"time_stamp"}:
        raise ValueError("timestamp JSON must contain only the time_stamp field")

    segment = validate_segment(abstract_data.get("segment"))
    time_stamp = validate_time_stamp(timestamp_data.get("time_stamp"), duration_sec)

    if len(time_stamp) != len(segment):
        raise ValueError(
            f"len(time_stamp) must equal len(segment): {len(time_stamp)} != {len(segment)}"
        )

    return {
        "time_stamp": time_stamp,
        "segment": segment,
        "url": resolve_video_url(stem, video.get("url"), missing_url_policy, local_video_index),
        "tt": require_string(video.get("title"), "video.title"),
        "dctt": segment[0]["seg_abs"][0],
    }


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def print_issues(label: str, issues: list[BuildIssue], max_issues: int) -> None:
    if not issues:
        return
    print(f"{label}: {len(issues)}")
    for issue in issues[: max(0, max_issues)]:
        print(f"  - {issue.stem}: {issue.reason}")
    if len(issues) > max_issues:
        print(f"  ... {len(issues) - max_issues} more")


def main() -> int:
    args = parse_args()
    abstract_dir = Path(args.abstract_dir)
    timestamp_dir = Path(args.timestamp_dir)
    mapping_path = Path(args.mapping)
    output_path = Path(args.output)
    video_dir = Path(args.video_dir)

    abstract_files = json_files_by_stem(abstract_dir)
    timestamp_files = json_files_by_stem(timestamp_dir)

    abstract_stems = set(abstract_files)
    timestamp_stems = set(timestamp_files)

    if args.only:
        filters = [item.lower() for item in args.only if item]

        def matches_filter(stem: str) -> bool:
            lowered = stem.lower()
            return all(item in lowered for item in filters)

        abstract_stems = {stem for stem in abstract_stems if matches_filter(stem)}
        timestamp_stems = {stem for stem in timestamp_stems if matches_filter(stem)}

    common_stems = abstract_stems & timestamp_stems

    missing: list[BuildIssue] = []
    for stem in sorted(abstract_stems - timestamp_stems):
        missing.append(BuildIssue(stem, "timestamp JSON missing"))
    for stem in sorted(timestamp_stems - abstract_stems):
        missing.append(BuildIssue(stem, "abstract JSON missing"))

    order = load_order(mapping_path, common_stems)
    local_video_index = load_local_video_index(mapping_path, video_dir)
    records: list[dict[str, Any]] = []
    invalid: list[BuildIssue] = []

    for stem in order:
        if args.limit is not None and len(records) >= args.limit:
            break
        try:
            records.append(
                build_record(
                    stem,
                    abstract_files[stem],
                    timestamp_files[stem],
                    args.missing_url_policy,
                    local_video_index,
                )
            )
        except Exception as exc:
            invalid.append(BuildIssue(stem, str(exc)))

    print(f"Abstract JSON files: {len(abstract_files)}")
    print(f"Timestamp JSON files: {len(timestamp_files)}")
    print(f"Matched stems: {len(common_stems)}")
    print(f"Missing URL policy: {args.missing_url_policy}")
    print(f"Indexed local videos: {len(local_video_index)}")
    print(f"Valid records: {len(records)}")
    print_issues("Missing pairs", missing, args.max_issues)
    print_issues("Invalid records", invalid, args.max_issues)

    if args.strict and (missing or invalid):
        print("Strict mode failed; output was not written.")
        return 1

    if args.dry_run:
        print("Dry run only; output was not written.")
        return 0

    write_jsonl_atomic(output_path, records)
    print(f"Wrote: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
