from pathlib import Path
import csv
import re
import shutil
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]

episode = sys.argv[1] if len(sys.argv) > 1 else "s01e01"

REVIEW_DIR = ROOT / "rick_candidates" / episode / "review"
SEG_TSV = ROOT / "segments_all" / episode / "segments.tsv"

CLEAN_DIR = ROOT / "rick_clean" / episode
CLEAN_WAVS = CLEAN_DIR / "wavs"

if not REVIEW_DIR.exists():
    raise FileNotFoundError(REVIEW_DIR)

if CLEAN_DIR.exists():
    shutil.rmtree(CLEAN_DIR)

CLEAN_WAVS.mkdir(parents=True, exist_ok=True)

texts = {}

with SEG_TSV.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        stem = Path(row["wav"]).stem
        texts[stem] = row["text"]

meta_lines = []
report_lines = ["stem\tseconds\ttext"]

total_seconds = 0.0
kept = 0

for p in sorted(REVIEW_DIR.glob("*.wav")):
    m = re.search(r"(s\d+e\d+_\d{6})", p.stem)
    if not m:
        continue

    stem = m.group(1)
    text = texts.get(stem, "").strip()

    if not text:
        continue

    dst = CLEAN_WAVS / f"{stem}.wav"
    shutil.copy2(p, dst)

    with wave.open(str(dst), "rb") as w:
        seconds = w.getnframes() / float(w.getframerate())

    total_seconds += seconds
    kept += 1

    meta_lines.append(f"{stem}|{text}")
    report_lines.append(f"{stem}\t{seconds:.3f}\t{text}")

(CLEAN_DIR / "metadata.csv").write_text(
    "\n".join(meta_lines) + "\n",
    encoding="utf-8",
)

(CLEAN_DIR / "clean_report.tsv").write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("Clean dir:", CLEAN_DIR)
print("Kept:", kept)
print("Total seconds:", round(total_seconds, 2))
print("Total minutes:", round(total_seconds / 60, 2))
print("Metadata:", CLEAN_DIR / "metadata.csv")
print("Wavs:", CLEAN_WAVS)
