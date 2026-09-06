from pathlib import Path
import csv
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]

episode = sys.argv[1] if len(sys.argv) > 1 else "s01e01"

BASE = ROOT / "rick_candidates" / episode
REPORT = BASE / f"{episode}_rick_similarity.tsv"
SRC_WAVS = BASE / "wavs"
REVIEW_DIR = BASE / "review"

if REVIEW_DIR.exists():
    shutil.rmtree(REVIEW_DIR)

REVIEW_DIR.mkdir(parents=True, exist_ok=True)

rows_out = ["rank\tsimilarity\tstem\treview_wav\ttext"]

with REPORT.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        if row["selected"] != "1":
            continue

        rank = int(row["rank"])
        stem = row["stem"]
        sim = float(row["similarity"])
        text = row["text"]

        src = SRC_WAVS / f"{stem}.wav"
        if not src.exists():
            continue

        review_name = f"{rank:03d}_{sim:.3f}_{stem}.wav"
        dst = REVIEW_DIR / review_name

        shutil.copy2(src, dst)
        rows_out.append(f"{rank}\t{sim:.6f}\t{stem}\t{review_name}\t{text}")

(REVIEW_DIR / "_review_index.tsv").write_text(
    "\n".join(rows_out) + "\n",
    encoding="utf-8",
)

print("Review dir:", REVIEW_DIR)
print("Files:", len(list(REVIEW_DIR.glob('*.wav'))))
print("Index:", REVIEW_DIR / "_review_index.tsv")
