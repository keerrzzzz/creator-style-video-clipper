import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPTS))
from creator_workflow import _apply_learned_boundaries, _apply_learned_preferences, _apply_semantic_similarity, _empty_learned_preferences, _update_learned_preferences, run_workflow
from transcript_rough_cut import Segment, compile_signals, make_candidates, score_text, topic_bounds
from local.downloader import _resolve_local_path


def write_srt(path: Path, rows: list[tuple[int, int, str]]) -> None:
    blocks = []
    for index, (start, end, text) in enumerate(rows, 1):
        blocks.append(f"{index}\n00:{start // 60:02d}:{start % 60:02d},000 --> 00:{end // 60:02d}:{end % 60:02d},000\n{text}")
    path.write_text("\n\n".join(blocks), encoding="utf-8")


class CreatorWorkflowTests(unittest.TestCase):
    def test_topic_bounds_keeps_the_whole_speech_block(self):
        segments = [
            Segment(10, 14, "先介绍话题背景"),
            Segment(14, 19, "再给出明确观点"),
            Segment(20, 26, "最后把这个话题说完"),
            Segment(31, 36, "这是另一个话题"),
        ]
        self.assertEqual(topic_bounds(segments, 1, 180), (10, 26))

    def test_topic_bounds_skips_an_overlong_unfinished_topic(self):
        segments = [Segment(0, 100, "话题开始"), Segment(100, 200, "仍在同一话题")]
        self.assertIsNone(topic_bounds(segments, 0, 180))

    def test_hourly_candidate_uses_full_transcript_for_topic_end(self):
        full = [
            Segment(3590, 3598, "我先说一下这件事"),
            Segment(3598, 3605, "这次结果确实反转"),
            Segment(3605, 3620, "所以这个话题说完"),
            Segment(3630, 3640, "下一个话题"),
        ]
        patterns, _ = compile_signals(None)
        candidates = make_candidates(full[:2], 15, 180, patterns, boundary_segments=full)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["end_seconds"], 3620)

    def test_builds_profile_hourly_analysis_and_cut_plan_offline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history, target = root / "viral.srt", root / "target.srt"
            write_srt(history, [(0, 10, "这个结果让观众都笑了"), (12, 22, "我觉得这个反转太精彩")])
            write_srt(target, [(30, 45, "我觉得这次结果真的反转"), (3660, 3675, "观众笑了，结果居然反转")])
            with patch("local.downloader.download_video", return_value=str(root / "target.mp4")):
                result = run_workflow("https://video.example/target", "测试博主", root / "result", history_srt=[history], target_srt=target, max_clips=2)
            profile = json.loads(Path(result["creator_profile"]).read_text(encoding="utf-8"))
            hourly = json.loads(Path(result["hourly_analysis"]).read_text(encoding="utf-8"))
            plan = json.loads(Path(result["plan"]).read_text(encoding="utf-8"))
            self.assertEqual(profile["creator"], "测试博主")
            self.assertTrue(profile["observed_style"]["frequent_terms"])
            self.assertEqual(len(hourly["windows"]), 2)
            self.assertTrue(plan["clips"])
            self.assertIn("reason", plan["clips"][0])
            feedback = json.loads(Path(result["clip_feedback"]).read_text(encoding="utf-8"))
            self.assertEqual(feedback["feedback"][0]["decision"], "undecided")

    def test_creator_profile_selects_a_distinctive_creator_expression(self):
        """A creator-specific signal must make its complete topic selectable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile_path, target = root / "creator-profile.json", root / "target.srt"
            profile_path.write_text(
                json.dumps(
                    {
                        "creator": "示例创作者",
                        "signals": {
                            "topic": ["独家拆解"],
                            "stance": ["招牌判断"],
                            "payoff": ["最后揭晓"],
                            "adaptation": [],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            write_srt(
                target,
                [
                    (10, 16, "今天做一次独家拆解"),
                    (16, 23, "这是我的招牌判断"),
                    (23, 31, "最后揭晓真正结论"),
                    (40, 48, "这里开始另一个普通话题"),
                ],
            )
            with patch("local.downloader.download_video", return_value=str(root / "target.mp4")):
                result = run_workflow(
                    "https://example.com/video",
                    "示例创作者",
                    root / "result",
                    creator_profile=profile_path,
                    target_srt=target,
                    max_clips=1,
                )
            self.assertEqual(len(result["clips"]), 1)
            self.assertEqual(result["clips"][0]["start"], "00:00:10")
            self.assertIn("创作者观点", result["clips"][0]["reason"])
            self.assertIn("反转或反应", result["clips"][0]["reason"])

    def test_profile_does_not_inherit_another_creators_catchphrases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "creator-profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "creator": "美食创作者",
                        "signals": {
                            "topic": ["红烧肉"],
                            "stance": ["我的做法"],
                            "payoff": [],
                            "adaptation": [],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            patterns, _ = compile_signals(profile_path)
            self.assertIsNone(patterns["adaptation"].search("陛下你也不想想"))
            self.assertEqual(score_text("红烧肉是我的做法", patterns)[0], 5)

    def test_feedback_keywords_boost_matching_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history, target, feedback_path = root / "viral.srt", root / "target.srt", root / "clip-feedback.json"
            write_srt(history, [(0, 10, "这个反转让观众都笑了")])
            write_srt(target, [(30, 45, "我觉得这次结果真的反转")])
            feedback_path.write_text(json.dumps({"preferences": {"boost_keywords": ["反转"], "suppress_keywords": []}}, ensure_ascii=False), encoding="utf-8")
            with patch("local.downloader.download_video", return_value=str(root / "target.mp4")):
                result = run_workflow("https://example.com/video", "测试博主", root / "result", history_srt=[history], target_srt=target, clip_feedback=feedback_path)
            self.assertIn("人工反馈偏好（反转）", result["clips"][0]["reason"])

    def test_keep_and_reject_feedback_builds_reusable_learned_preferences(self):
        learned = _empty_learned_preferences("测试创作者")
        feedback = {
            "feedback": [
                {"decision": "keep", "keywords": ["深度拆解"], "context": "这次深度拆解很完整"},
                {"decision": "reject", "keywords": ["日常播报"], "context": "普通的日常播报"},
            ]
        }
        learned = _update_learned_preferences(learned, feedback)
        self.assertGreater(learned["positive_terms"]["深度拆解"], 0)
        self.assertGreater(learned["negative_terms"]["日常播报"], 0)
        candidates = _apply_learned_preferences(
            [
                {"score": 5, "reason": "完整话题段", "context": "这是一次深度拆解"},
                {"score": 5, "reason": "完整话题段", "context": "这是日常播报"},
            ],
            learned,
        )
        self.assertGreater(candidates[0]["score"], 5)
        self.assertLess(candidates[1]["score"], 5)
        self.assertIn("已学习保留偏好", candidates[0]["reason"])
        self.assertIn("已学习拒绝偏好", candidates[1]["reason"])

    def test_adjust_feedback_learns_boundary_offset(self):
        learned = _update_learned_preferences(
            _empty_learned_preferences("测试创作者"),
            {"feedback": [{"decision": "adjust", "original_start": "00:01:00", "original_end": "00:02:00", "suggested_start": "00:00:55", "suggested_end": "00:02:08"}]},
        )
        candidate = _apply_learned_boundaries([{"start_seconds": 100, "end_seconds": 160, "reason": "完整话题段"}], learned)[0]
        self.assertEqual((candidate["start_seconds"], candidate["end_seconds"]), (95, 168))
        self.assertIn("已学习边界偏移", candidate["reason"])

    def test_semantic_similarity_uses_reviewed_examples(self):
        learned = _empty_learned_preferences("测试创作者")
        learned["semantic_samples"] = [
            {"label": "keep", "embedding": [1.0, 0.0]},
            {"label": "keep", "embedding": [0.9, 0.1]},
            {"label": "reject", "embedding": [0.0, 1.0]},
        ]
        vectors = {"像保留": [1.0, 0.0], "像拒绝": [0.0, 1.0]}
        candidates = _apply_semantic_similarity(
            [{"score": 5, "reason": "完整话题段", "context": "像保留"}, {"score": 5, "reason": "完整话题段", "context": "像拒绝"}],
            learned,
            lambda text: vectors[text],
        )
        self.assertGreater(candidates[0]["score"], 5)
        self.assertLess(candidates[1]["score"], 5)
        self.assertIn("语义接近已保留片段", candidates[0]["reason"])

    def test_local_downloader_accepts_local_files_and_defers_remote_urls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media = root / "sample.mp4"
            media.touch()
            self.assertEqual(_resolve_local_path(str(media)), str(media.resolve()))
            self.assertIsNone(_resolve_local_path("https://video.example/recording"))
