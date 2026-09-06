import cuda_bootstrap
from pathlib import Path
import json
import csv
import sys

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "audio"
OUT_DIR = ROOT / "transcripts"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# İlk test için sadece s01e01
audio_files = [AUDIO_DIR / "s01e02.wav"]

MODEL_SIZE = "medium"

def load_model():
    try:
        print("Trying CUDA...")
        return WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception as e:
        print("CUDA failed, using CPU int8.")
        print("Reason:", e)
        return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

model = load_model()

for audio_path in audio_files:
    if not audio_path.exists():
        print("Missing:", audio_path)
        sys.exit(1)

    episode = audio_path.stem
    print("Transcribing:", episode)

    segments, info = model.transcribe(
        str(audio_path),
        language="tr",
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 120,
        },
    )

    rows = []
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        if not text:
            continue

        row = {
            "id": i,
            "start": float(seg.start),
            "end": float(seg.end),
            "duration": float(seg.end - seg.start),
            "text": text,
        }
        rows.append(row)
        print(f"{i:04d} {seg.start:8.2f} -> {seg.end:8.2f} | {text}")

    json_path = OUT_DIR / f"{episode}.json"
    tsv_path = OUT_DIR / f"{episode}.tsv"

    json_path.write_text(
        json.dumps(
            {
                "episode": episode,
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "start", "end", "duration", "text"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", json_path)
    print("Saved:", tsv_path)
    print("Segments:", len(rows))

