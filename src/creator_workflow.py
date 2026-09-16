#!/usr/bin/env python3
"""One-command creator workflow: profile, hourly ranking, and rough cuts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from analyze_creator_profile import build_creator_profile
from transcript_rough_cut import (
    Segment,
    analyze_hourly,
    choose_candidates,
    compile_signals,
    find_event_reaction_candidates,
    find_song_candidates,
    get_transcript,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent


def _write_srt(path: Path, segments: list[Segment]) -> None:
    def stamp(value: float) -> str:
        milliseconds = round(value * 1000)
        seconds, ms = divmod(milliseconds, 1000)
        minutes, second = divmod(seconds, 60)
        hour, minute = divmod(minutes, 60)
        return f"{hour:02d}:{minute:02d}:{second:02d},{ms:03d}"

    blocks = [f"{i}\n{stamp(item.start)} --> {stamp(item.end)}\n{item.text}" for i, item in enumerate(segments, 1)]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _load_feedback(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid clip feedback JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("clip feedback must be a JSON object")
    return payload


def _apply_feedback(candidates: list[dict], feedback: dict) -> list[dict]:
    """Apply user-maintained keyword preferences without hiding the reason."""
    preferences = feedback.get("preferences", {})
    if not isinstance(preferences, dict):
        return candidates
    boost = [str(item).strip() for item in preferences.get("boost_keywords", []) if str(item).strip()]
    suppress = [str(item).strip() for item in preferences.get("suppress_keywords", []) if str(item).strip()]
    ranked: list[dict] = []
    for candidate in candidates:
        context = str(candidate.get("context", ""))
        if any(keyword in context for keyword in suppress):
            continue
        matches = [keyword for keyword in boost if keyword in context]
        item = dict(candidate)
        if matches:
            item["score"] = int(item.get("score", 0)) + (2 * len(matches))
            item["reason"] = f"{item.get('reason', '')}、人工反馈偏好（{'、'.join(matches)}）".strip("、")
        ranked.append(item)
    return ranked


def _empty_learned_preferences(creator: str) -> dict:
    return {"version": 1, "creator": creator, "positive_terms": {}, "negative_terms": {}, "semantic_samples": [], "boundary_adjustments": {"start_seconds": 0.0, "end_seconds": 0.0, "samples": 0}, "processed_feedback": [], "history": []}


def _load_learned_preferences(path: Path | None, creator: str) -> dict:
    if not path:
        return _empty_learned_preferences(creator)
    learned = _load_feedback(path)
    if not learned:
        return _empty_learned_preferences(creator)
    if learned.get("creator") not in (None, creator):
        raise ValueError(f"Learned preferences belong to {learned.get('creator')}, not {creator}")
    learned.setdefault("version", 1)
    learned["creator"] = creator
    learned.setdefault("positive_terms", {})
    learned.setdefault("negative_terms", {})
    learned.setdefault("semantic_samples", [])
    learned.setdefault("boundary_adjustments", {"start_seconds": 0.0, "end_seconds": 0.0, "samples": 0})
    learned.setdefault("processed_feedback", [])
    learned.setdefault("history", [])
    return learned


def _time_to_seconds(value: object) -> float | None:
    try:
        hour, minute, second = (int(part) for part in str(value).split(":"))
        return hour * 3600 + minute * 60 + second
    except (TypeError, ValueError):
        return None


def _creator_learning_path(store: Path, creator: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", creator).strip("_") or "creator"
    identity = hashlib.sha256(creator.encode("utf-8")).hexdigest()[:10]
    return store / f"{safe_name}-{identity}.json"


def _feedback_terms(entry: dict) -> list[str]:
    """Extract editable keywords plus short recurring phrases from a review."""
    explicit = [str(item).strip() for item in entry.get("keywords", []) if str(item).strip()]
    context = str(entry.get("context", ""))
    phrases = [part.strip() for part in re.split(r"[，。！？!?；;、\n]+", context) if 2 <= len(part.strip()) <= 20]
    return list(dict.fromkeys(explicit + phrases))


def _update_learned_preferences(learned: dict, feedback: dict, embed=None) -> dict:
    """Turn reviewed keep/reject decisions into transparent term weights."""
    reviews = feedback.get("feedback", [])
    if not isinstance(reviews, list):
        return learned
    digest = hashlib.sha256(json.dumps(reviews, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    processed = learned["processed_feedback"]
    if digest in processed:
        return learned
    for entry in reviews:
        if not isinstance(entry, dict):
            continue
        decision = entry.get("decision")
        if decision not in ("keep", "reject", "adjust"):
            continue
        if decision in ("keep", "reject"):
            bucket = learned["positive_terms"] if decision == "keep" else learned["negative_terms"]
            for term in _feedback_terms(entry):
                bucket[term] = int(bucket.get(term, 0)) + 1
            context = str(entry.get("context", "")).strip()
            if embed and context:
                learned["semantic_samples"].append({"label": decision, "text": context, "embedding": embed(context)})
        if decision == "adjust":
            values = [_time_to_seconds(entry.get(name)) for name in ("original_start", "original_end", "suggested_start", "suggested_end")]
            if all(value is not None for value in values):
                original_start, original_end, suggested_start, suggested_end = values
                boundary = learned["boundary_adjustments"]
                boundary["start_seconds"] += suggested_start - original_start
                boundary["end_seconds"] += suggested_end - original_end
                boundary["samples"] += 1
        learned["history"].append({"decision": decision, "terms": _feedback_terms(entry), "clip_id": entry.get("clip_id")})
    processed.append(digest)
    learned["processed_feedback"] = processed[-100:]
    learned["history"] = learned["history"][-100:]
    learned["semantic_samples"] = learned["semantic_samples"][-200:]
    return learned


def _apply_learned_preferences(candidates: list[dict], learned: dict) -> list[dict]:
    """Apply accumulated keep/reject evidence without silently filtering clips."""
    positive = learned.get("positive_terms", {})
    negative = learned.get("negative_terms", {})
    if not isinstance(positive, dict) or not isinstance(negative, dict):
        return candidates
    ranked: list[dict] = []
    for candidate in candidates:
        context = str(candidate.get("context", ""))
        kept = [term for term in positive if term and term in context]
        rejected = [term for term in negative if term and term in context]
        item = dict(candidate)
        if kept:
            item["score"] = int(item.get("score", 0)) + sum(min(int(positive[term]), 3) for term in kept)
            item["reason"] = f"{item.get('reason', '')}、已学习保留偏好（{'、'.join(kept)}）".strip("、")
        if rejected:
            item["score"] = int(item.get("score", 0)) - sum(min(int(negative[term]), 3) for term in rejected)
            item["reason"] = f"{item.get('reason', '')}、已学习拒绝偏好（{'、'.join(rejected)}）".strip("、")
        ranked.append(item)
    return ranked


def _apply_learned_boundaries(candidates: list[dict], learned: dict) -> list[dict]:
    boundary = learned.get("boundary_adjustments", {})
    samples = int(boundary.get("samples", 0)) if isinstance(boundary, dict) else 0
    if not samples:
        return candidates
    start_shift = max(-15.0, min(15.0, float(boundary["start_seconds"]) / samples))
    end_shift = max(-15.0, min(15.0, float(boundary["end_seconds"]) / samples))
    adjusted = []
    for candidate in candidates:
        item = dict(candidate)
        start, end = max(0.0, float(item["start_seconds"]) + start_shift), float(item["end_seconds"]) + end_shift
        if end - start >= 15:
            item["start_seconds"], item["end_seconds"] = start, end
            item["reason"] = f"{item.get('reason', '')}、已学习边界偏移（{start_shift:+.0f}s/{end_shift:+.0f}s）".strip("、")
        adjusted.append(item)
    return adjusted


def _apply_semantic_similarity(candidates: list[dict], learned: dict, embed=None) -> list[dict]:
    """Score candidates against reviewed keep/reject examples by meaning."""
    samples = learned.get("semantic_samples", [])
    if not embed or not isinstance(samples, list) or len(samples) < 3:
        return candidates
    from local.embedder import cosine_similarity

    ranked = []
    for candidate in candidates:
        vector = embed(str(candidate.get("context", "")))
        keep_scores = [cosine_similarity(vector, sample["embedding"]) for sample in samples if sample.get("label") == "keep"]
        reject_scores = [cosine_similarity(vector, sample["embedding"]) for sample in samples if sample.get("label") == "reject"]
        item = dict(candidate)
        if keep_scores and max(keep_scores) >= 0.55:
            similarity = max(keep_scores)
            item["score"] = int(item.get("score", 0)) + min(3, max(1, round((similarity - 0.5) * 6)))
            item["reason"] = f"{item.get('reason', '')}、语义接近已保留片段（{similarity:.2f}）".strip("、")
        if reject_scores and max(reject_scores) >= 0.55:
            similarity = max(reject_scores)
            item["score"] = int(item.get("score", 0)) - min(3, max(1, round((similarity - 0.5) * 6)))
            item["reason"] = f"{item.get('reason', '')}、语义接近已拒绝片段（{similarity:.2f}）".strip("、")
        ranked.append(item)
    return ranked


def _write_feedback_template(path: Path, creator: str, clips: list[dict], existing: dict) -> None:
    if existing:
        return
    template = {
        "creator": creator,
        "instructions": "填写 feedback 中每段的 decision/reason；把会出现在未来字幕中的词填入 preferences，以影响下次选段。",
        "preferences": {"boost_keywords": [], "suppress_keywords": []},
        "feedback": [
            {
                "clip_id": clip.get("id"),
                "decision": "undecided",
                "reason": "",
                "keywords": [],
                "context": clip.get("context", ""),
                "original_start": clip.get("start"),
                "original_end": clip.get("end"),
                "suggested_start": clip.get("start"),
                "suggested_end": clip.get("end"),
            }
            for clip in clips
        ],
    }
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_workflow(
    target_url: str,
    creator: str,
    output: Path,
    history_urls: list[str] | None = None,
    history_srt: list[Path] | None = None,
    target_srt: Path | None = None,
    creator_profile: Path | None = None,
    clip_feedback: Path | None = None,
    learned_preferences: Path | None = None,
    learning_store: Path | None = None,
    note: str = "",
    language: str = "zh",
    max_clips: int = 5,
    window_seconds: int = 3600,
    candidates_per_window: int = 5,
    download_height: str = "720",
    export: bool = False,
    reencode: bool = False,
) -> dict:
    """Run the reproducible workflow; transcript paths make it testable offline."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from local.downloader import download_video

    output.mkdir(parents=True, exist_ok=True)
    if creator_profile:
        if not creator_profile.is_file():
            raise ValueError(f"Creator profile not found: {creator_profile}")
        profile_path = creator_profile
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    else:
        history_dir = output / "history-transcripts"
        history_dir.mkdir(exist_ok=True)
        profile_inputs = list(history_srt or [])
        for index, url in enumerate(history_urls or [], 1):
            media = str(Path(download_video(url, fmt=download_height, out_dir=str(output / "history-source"))).resolve())
            segments = get_transcript(Path(media), None, language)
            transcript_path = history_dir / f"history-{index:02d}.srt"
            _write_srt(transcript_path, segments)
            profile_inputs.append(transcript_path)
        if not profile_inputs:
            raise ValueError("Provide --creator-profile, --history-url, or --history-transcript-srt.")
        profile = build_creator_profile(creator, profile_inputs, note)
        profile_path = output / "creator-profile.json"
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    source = str(Path(download_video(target_url, fmt=download_height, out_dir=str(output / "source"))).resolve())
    segments = get_transcript(Path(source), target_srt, language)
    if not segments:
        raise ValueError("Target transcript contains no speech segments.")
    (output / "transcript.json").write_text(
        json.dumps({"duration": segments[-1].end, "segments": [item.__dict__ for item in segments]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    feedback_path = clip_feedback or (output / "clip-feedback.json")
    feedback = _load_feedback(feedback_path)
    shared_learning_path = _creator_learning_path(learning_store, profile.get("creator", creator)) if learning_store else None
    learned = _load_learned_preferences(learned_preferences or shared_learning_path, profile.get("creator", creator))
    embedding_cache = (learning_store or output) / "embedding-model"

    def embed(text: str) -> list[float]:
        from local.embedder import embed_texts
        return embed_texts([text], embedding_cache)[0]

    patterns, profile = compile_signals(profile_path)
    # Complete topic blocks are preferred over an arbitrary 45–120 second
    # padded window.  Short but self-contained replies remain eligible; a
    # still-active topic over three minutes is left for manual review instead
    # of being cut in the middle.
    hourly, candidates = analyze_hourly(segments, window_seconds, candidates_per_window, 15, 180, patterns)
    candidates.extend(find_song_candidates(segments, profile))
    candidates.extend(find_event_reaction_candidates(segments, profile))
    candidates = _apply_feedback(candidates, feedback)
    candidates = _apply_learned_preferences(candidates, learned)
    candidates = _apply_learned_boundaries(candidates, learned)
    candidates = _apply_semantic_similarity(candidates, learned, embed)
    hourly_path = output / "hourly-analysis.json"
    hourly_path.write_text(json.dumps({"window_seconds": window_seconds, "windows": hourly}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    clips = choose_candidates(candidates, max_clips)
    _write_feedback_template(feedback_path, profile.get("creator", creator), clips, feedback)
    learned = _update_learned_preferences(learned, feedback, embed)
    learned_path = output / "learned-preferences.json"
    learned_path.write_text(json.dumps(learned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if shared_learning_path:
        shared_learning_path.parent.mkdir(parents=True, exist_ok=True)
        shared_learning_path.write_text(json.dumps(learned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plan = {"title": Path(source).stem, "source": source, "creator": profile.get("creator", creator), "creator_profile": str(profile_path), "clips": clips}
    plan_path = output / "candidate-cut-plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    exports: list[str] = []
    if export and clips:
        from rough_cut import main as rough_cut_main
        old_argv = sys.argv
        try:
            sys.argv = ["rough_cut.py", "--input", source, "--plan", str(plan_path), "--output", str(output / "exports")] + (["--reencode"] if reencode else [])
            rough_cut_main()
        finally:
            sys.argv = old_argv
        exports = [str(path) for path in sorted((output / "exports").glob("*.mp4"))]
    return {"creator_profile": str(profile_path), "hourly_analysis": str(hourly_path), "plan": str(plan_path), "clip_feedback": str(feedback_path), "learned_preferences": str(learned_path), "learning_store": str(shared_learning_path) if shared_learning_path else None, "clips": clips, "exports": exports}


def main() -> int:
    parser = argparse.ArgumentParser(description="多平台长视频一键：爆款画像 → 每小时筛选 → 粗剪。")
    parser.add_argument("target_url", help="yt-dlp 支持的公开视频链接或本地文件")
    parser.add_argument("--creator", required=True, help="博主/账号名称")
    parser.add_argument("--history-url", action="append", default=[], help="已确认的历史爆款视频链接；可重复传入")
    parser.add_argument("--history-transcript-srt", type=Path, action="append", default=[], help="历史爆款 SRT；适合离线复跑")
    parser.add_argument("--creator-profile", type=Path, help="复用已保存的创作者画像 JSON，跳过历史视频分析")
    parser.add_argument("--clip-feedback", type=Path, help="读取既有 clip-feedback.json；首次运行会在输出目录创建模板")
    parser.add_argument("--learned-preferences", type=Path, help="读取同一创作者既有的 learned-preferences.json")
    parser.add_argument("--learning-store", type=Path, default=Path("creator-learning"), help="跨任务自动保存每位创作者学习档案的目录")
    parser.add_argument("--target-transcript-srt", type=Path, help="复用目标 SRT，跳过目标转录")
    parser.add_argument("--output", type=Path, required=True, help="本次任务输出目录")
    parser.add_argument("--note", default="", help="账号定位/人工观察")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--analysis-window-seconds", type=int, default=3600)
    parser.add_argument("--candidates-per-window", type=int, default=5)
    parser.add_argument("--format", default="720")
    parser.add_argument("--export", action="store_true", help="按选段计划导出 MP4 粗剪")
    parser.add_argument("--reencode", action="store_true", help="粗剪采用更准确但更慢的重编码")
    args = parser.parse_args()
    try:
        result = run_workflow(args.target_url, args.creator, args.output, args.history_url, args.history_transcript_srt, args.target_transcript_srt, args.creator_profile, args.clip_feedback, args.learned_preferences, args.learning_store, args.note, args.language, args.max_clips, args.analysis_window_seconds, args.candidates_per_window, args.format, args.export, args.reencode)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
