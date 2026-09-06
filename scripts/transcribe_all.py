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

MODEL_SIZE = "medium"

args = sys.argv[1:]

if args:
    episodes = args
else:
    episodes = sorted(p.stem for p in AUDIO_DIR.glob("s*.wav"))

print("Episodes:", episodes)

model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

for episode in episodes:
    audio_path = AUDIO_DIR / f"{episode}.wav"

    if not audio_path.exists():
        print("Missing:", audio_path)
        continue

    json_path = OUT_DIR / f"{episode}.json"
    tsv_path = OUT_DIR / f"{episode}.tsv"

    if json_path.exists() and tsv_path.exists():
        print("Skip existing:", episode)
        continue

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

    print("Saved:", episode, "segments:", len(rows))
