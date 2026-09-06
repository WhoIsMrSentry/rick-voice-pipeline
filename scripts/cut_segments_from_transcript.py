from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIR = ROOT / "audio"
TRANSCRIPT_DIR = ROOT / "transcripts"
SEGMENTS_DIR = ROOT / "segments_all"

episode = sys.argv[1] if len(sys.argv) > 1 else "s01e01"

audio_path = AUDIO_DIR / f"{episode}.wav"
json_path = TRANSCRIPT_DIR / f"{episode}.json"

out_root = SEGMENTS_DIR / episode
wav_dir = out_root / "wavs"
wav_dir.mkdir(parents=True, exist_ok=True)

if not audio_path.exists():
    raise FileNotFoundError(audio_path)

if not json_path.exists():
    raise FileNotFoundError(json_path)

data = json.loads(json_path.read_text(encoding="utf-8"))
segments = data.get("segments", [])

rows_tsv = ["transcript_id\tstart\tend\tduration\twav\ttext"]
rows_pipe = []

kept = 0
skipped = 0

for seg in segments:
    transcript_id = int(seg["id"])
    start = float(seg["start"])
    end = float(seg["end"])
    text = str(seg["text"]).strip()
    duration = end - start

    if not text:
        skipped += 1
        continue

    if duration < 0.75 or duration > 12.0:
        skipped += 1
        continue

    kept += 1

    stem = f"{episode}_{transcript_id:06d}"
    wav_path = wav_dir / f"{stem}.wav"

    pad = 0.08
    ss = max(0.0, start - pad)
    dur = duration + (pad * 2)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{ss:.3f}",
        "-i",
        str(audio_path),
        "-t",
        f"{dur:.3f}",
        "-ac",
        "1",
        "-ar",
        "22050",
        "-af",
        "highpass=f=80,lowpass=f=7600,loudnorm=I=-18:TP=-2:LRA=10",
        str(wav_path),
    ]

    subprocess.run(cmd, check=True)

    rows_tsv.append(
        f"{transcript_id}\t{start:.3f}\t{end:.3f}\t{duration:.3f}\t{wav_path.name}\t{text}"
    )

    rows_pipe.append(f"{stem}|{text}")

(out_root / "segments.tsv").write_text(
    "\n".join(rows_tsv) + "\n",
    encoding="utf-8",
)

(out_root / "metadata_pipe.csv").write_text(
    "\n".join(rows_pipe) + "\n",
    encoding="utf-8",
)

print("Episode:", episode)
print("Input segments:", len(segments))
print("Kept:", kept)
print("Skipped:", skipped)
print("WAV dir:", wav_dir)
print("TSV:", out_root / "segments.tsv")
print("Pipe metadata:", out_root / "metadata_pipe.csv")
