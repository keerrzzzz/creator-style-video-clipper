#!/usr/bin/env python3
"""YAMNet-assisted song review workflow for 足球需要100个特兰."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from football_song_detector import detect_singing_ranges
from transcript_rough_cut import Segment, get_transcript, format_time


def song_candidates(segments: list[Segment], ranges: list[tuple[float, float]], evidence_path: Path) -> list[dict]:
    candidates = []
    for index, (start, end) in enumerate(ranges, 1):
        transcript = [item.text for item in segments if item.end >= start and item.start <= end]
        candidates.append({
            "id": f"{index:02d}",
            "kind": "song",
            "start": format_time(start),
            "end": format_time(end),
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "hook": transcript[0][:42] if transcript else "待确认歌曲",
            "context": " ".join(transcript)[:800],
            "yamnet_evidence": str(evidence_path),
            "verification": "pending",
            "reason": "YAMNet 检出连续演唱区间；已按短暂停顿、调音、点评或重唱合并，需人工确认实际首尾。",
        })
    return candidates


def load_approved(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips", [])
    return [clip for clip in clips if clip.get("verification") == "audio+transcript"]


def main() -> int:
    parser = argparse.ArgumentParser(description="特兰私有：YAMNet 歌曲候选审核工作流。")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path, help="已下载的本地视频")
    source_group.add_argument("--input-url", help="用户授权下载的视频链接")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transcript-srt", type=Path)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--merge-gap-seconds", type=float, default=90.0)
    parser.add_argument("--approved-song-plan", type=Path, help="人工核验后将 verification 设为 audio+transcript 的歌曲计划")
    parser.add_argument("--export", action="store_true", help="仅导出已核验歌曲")
    parser.add_argument("--reencode", action="store_true")
    args = parser.parse_args()
    if args.input and not args.input.is_file():
        parser.error(f"input not found: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input:
        source = args.input.resolve()
    else:
        from local.downloader import download_video
        source = Path(download_video(args.input_url, fmt="720", out_dir=str(args.output / "source"))).resolve()
    segments = get_transcript(source, args.transcript_srt, "zh")
    evidence_path = args.output / "yamnet-singing-ranges.json"
    ranges = detect_singing_ranges(str(source), args.output / ".yamnet-cache", args.threshold, args.merge_gap_seconds)
    evidence_path.write_text(json.dumps({"source": str(source), "threshold": args.threshold, "merge_gap_seconds": args.merge_gap_seconds, "ranges": [{"start": round(start, 3), "end": round(end, 3)} for start, end in ranges]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    candidate_path = args.output / "song-candidate-plan.json"
    candidate_path.write_text(json.dumps({"source": str(source), "clips": song_candidates(segments, ranges, evidence_path)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    exports: list[str] = []
    if args.export:
        if not args.approved_song_plan:
            parser.error("--export requires --approved-song-plan with audio+transcript verification")
        approved = load_approved(args.approved_song_plan)
        if not approved:
            parser.error("approved plan has no clips verified as audio+transcript")
        export_plan = args.output / "song-cut-plan.json"
        export_plan.write_text(json.dumps({"source": str(source), "clips": approved}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        from rough_cut import main as rough_cut_main
        old_argv = sys.argv
        try:
            sys.argv = ["rough_cut.py", "--input", str(source), "--plan", str(export_plan), "--output", str(args.output / "song-exports")] + (["--reencode"] if args.reencode else [])
            rough_cut_main()
        finally:
            sys.argv = old_argv
        exports = [str(item) for item in sorted((args.output / "song-exports").glob("*.mp4"))]
    print(json.dumps({"evidence": str(evidence_path), "candidates": str(candidate_path), "exports": exports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
