"""Private audio-first singing detector for 足球需要100个特兰 only."""
from __future__ import annotations
import argparse
import csv, json, os, subprocess, tempfile, wave
from pathlib import Path
import numpy as np

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

def merge_singing_frames(frames: list[tuple[float, float]], gap_seconds: float = 35.0) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in frames:
        if merged and start - merged[-1][1] <= gap_seconds: merged[-1][1] = end
        else: merged.append([start, end])
    return [(start, end) for start, end in merged]

def detect_singing_ranges(media_path: str, cache_dir: Path, threshold: float = 0.12, merge_gap_seconds: float = 90.0) -> list[tuple[float, float]]:
    import tensorflow as tf
    import tensorflow_hub as hub
    from urllib.request import urlretrieve
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels_path = cache_dir / "yamnet_class_map.csv"
    if not labels_path.exists(): urlretrieve(CLASS_MAP_URL, labels_path)
    with labels_path.open(encoding="utf-8") as handle:
        labels = [row[2] for row in csv.reader(handle) if len(row) >= 3 and row[0] != "index"]
    os.environ.setdefault("TFHUB_CACHE_DIR", str(cache_dir / "yamnet"))
    model = hub.load(YAMNET_URL)
    with tempfile.TemporaryDirectory(prefix="song-detector-") as temp:
        wav_path = Path(temp) / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-i", media_path, "-vn", "-ac", "1", "-ar", "16000", str(wav_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with wave.open(str(wav_path), "rb") as source:
            waveform = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    scores, _, _ = model(tf.convert_to_tensor(waveform, dtype=tf.float32))
    values = scores.numpy(); singing = labels.index("Singing"); music = labels.index("Music")
    active = np.where((values[:, singing] >= threshold) & (values[:, music] >= threshold / 2))[0]
    return merge_singing_frames([(float(i) * .48, float(i) * .48 + .96) for i in active], merge_gap_seconds)

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect likely singing ranges with YAMNet.")
    parser.add_argument("media_path")
    parser.add_argument("--output", required=True, help="Write detector evidence as JSON.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--merge-gap-seconds", type=float, default=90.0)
    args = parser.parse_args()
    ranges = detect_singing_ranges(args.media_path, Path(args.cache_dir), args.threshold, args.merge_gap_seconds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source": str(Path(args.media_path).resolve()),
        "threshold": args.threshold,
        "merge_gap_seconds": args.merge_gap_seconds,
        "ranges": [{"start": round(start, 3), "end": round(end, 3)} for start, end in ranges],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
