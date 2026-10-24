import importlib.util
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("telan_song_workflow", SRC / "telan_song_workflow.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from football_song_detector import merge_singing_frames


class TelanSongWorkflowTests(unittest.TestCase):
    def test_song_candidates_keep_audio_ranges_and_pending_verification(self):
        segments = [module.Segment(10, 15, "歌词第一句"), module.Segment(15, 20, "歌词第二句")]
        result = module.song_candidates(segments, [(10, 40), (200, 220)], Path("evidence.json"))
        self.assertEqual([item["verification"] for item in result], ["pending", "pending"])
        self.assertEqual(result[0]["start"], "00:00:10")
        self.assertEqual(result[1]["end"], "00:03:40")

    def test_short_restarts_merge_into_one_song_candidate(self):
        ranges = merge_singing_frames([(10, 30), (75, 110), (205, 230)], gap_seconds=90)
        self.assertEqual(ranges, [(10, 110), (205, 230)])
